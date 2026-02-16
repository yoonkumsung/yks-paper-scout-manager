"""DB Cache Management for GitHub Actions.

Provides functions for managing DB cache in CI/CD:
- Restore from actions/cache or GitHub Release assets
- Save to actions/cache with schema-based keys
- Upload DB backups to GitHub Releases
- Cleanup old Release assets
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Schema version constant (update when schema changes)
SCHEMA_VERSION = "v1"

# Default cache key format: db-{branch}-{schema_hash}
DEFAULT_BRANCH = "main"


def get_schema_hash(db_path: str | None = None) -> str:
    """Get schema version hash.

    Args:
        db_path: Path to SQLite DB (optional, for future use)

    Returns:
        Schema hash string (8 characters)
    """
    # For now, use a constant schema version
    # In the future, could read from DB or schema file
    return hashlib.sha256(SCHEMA_VERSION.encode()).hexdigest()[:8]


def get_cache_key(branch: str, schema_hash: str) -> str:
    """Build cache key string.

    Args:
        branch: Git branch name
        schema_hash: Schema version hash

    Returns:
        Cache key in format: db-{branch}-{schema_hash}
    """
    return f"db-{branch}-{schema_hash}"


def restore_from_release(repo: str, db_path: str) -> bool:
    """Download latest DB from GitHub Release assets.

    Args:
        repo: Repository in format "owner/repo"
        db_path: Target path for downloaded DB

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Attempting to restore DB from Release assets: {repo}")

    try:
        # Check if gh CLI is available
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("gh CLI not available")
            return False

        # List releases and find latest DB asset
        result = subprocess.run(
            ["gh", "release", "list", "-R", repo, "--limit", "10"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to list releases: {result.stderr}")
            return False

        releases = result.stdout.strip().split("\n")
        if not releases or not releases[0]:
            logger.warning("No releases found")
            return False

        # Try each release to find DB asset
        for release_line in releases:
            parts = release_line.split("\t")
            if not parts:
                continue
            tag = parts[0]

            # List assets for this release
            result = subprocess.run(
                ["gh", "release", "view", tag, "-R", repo, "--json", "assets"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                continue

            release_data = json.loads(result.stdout)
            assets = release_data.get("assets", [])

            # Find DB asset (paper-scout-db-*.sqlite)
            db_assets = [
                a for a in assets if a["name"].startswith("paper-scout-db-")
                and a["name"].endswith(".sqlite")
            ]
            if not db_assets:
                continue

            # Download the first matching asset
            asset_name = db_assets[0]["name"]
            logger.info(f"Found DB asset: {asset_name} in release {tag}")

            # Ensure target directory exists
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

            # Download asset
            result = subprocess.run(
                [
                    "gh", "release", "download", tag,
                    "-R", repo,
                    "-p", asset_name,
                    "-O", db_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error(f"Failed to download asset: {result.stderr}")
                return False

            logger.info(f"Successfully restored DB from {tag}/{asset_name}")
            return True

        logger.warning("No DB assets found in releases")
        return False

    except subprocess.TimeoutExpired:
        logger.error("gh command timed out")
        return False
    except Exception as e:
        logger.error(f"Error restoring from release: {e}")
        return False


def upload_to_release(repo: str, db_path: str, tag: str) -> bool:
    """Upload DB as GitHub Release asset.

    Args:
        repo: Repository in format "owner/repo"
        db_path: Path to DB file to upload
        tag: Release tag name

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(db_path):
        logger.error(f"DB file not found: {db_path}")
        return False

    try:
        # Generate asset name with date
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        asset_name = f"paper-scout-db-{date_str}.sqlite"

        logger.info(f"Uploading DB to release {tag} as {asset_name}")

        # Upload to release
        result = subprocess.run(
            [
                "gh", "release", "upload", tag,
                db_path,
                "--repo", repo,
                "--clobber",  # Replace if exists
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"Failed to upload asset: {result.stderr}")
            return False

        logger.info(f"Successfully uploaded DB to {tag}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("gh upload command timed out")
        return False
    except Exception as e:
        logger.error(f"Error uploading to release: {e}")
        return False


def cleanup_old_releases(repo: str, keep_weeks: int = 4) -> int:
    """Delete Release assets older than keep_weeks.

    Args:
        repo: Repository in format "owner/repo"
        keep_weeks: Number of weeks to keep (default 4)

    Returns:
        Number of assets deleted
    """
    logger.info(f"Cleaning up Release assets older than {keep_weeks} weeks")
    deleted_count = 0
    cutoff_date = datetime.now(timezone.utc) - timedelta(weeks=keep_weeks)

    try:
        # List all releases
        result = subprocess.run(
            ["gh", "release", "list", "-R", repo, "--limit", "100"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"Failed to list releases: {result.stderr}")
            return deleted_count

        releases = result.stdout.strip().split("\n")

        for release_line in releases:
            parts = release_line.split("\t")
            if not parts:
                continue
            tag = parts[0]

            # Get release details with assets
            result = subprocess.run(
                ["gh", "release", "view", tag, "-R", repo, "--json", "assets,publishedAt"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                continue

            release_data = json.loads(result.stdout)
            published_at = release_data.get("publishedAt", "")
            assets = release_data.get("assets", [])

            # Parse publish date
            try:
                publish_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Invalid publish date for {tag}: {published_at}")
                continue

            # Check if release is old enough to clean up
            if publish_date >= cutoff_date:
                continue

            # Delete DB assets from old release
            for asset in assets:
                asset_name = asset.get("name", "")
                if asset_name.startswith("paper-scout-db-") and asset_name.endswith(".sqlite"):
                    logger.info(f"Deleting old asset: {tag}/{asset_name}")
                    result = subprocess.run(
                        [
                            "gh", "release", "delete-asset", tag,
                            asset_name,
                            "-R", repo,
                            "--yes",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        deleted_count += 1
                    else:
                        logger.warning(f"Failed to delete {asset_name}: {result.stderr}")

        logger.info(f"Cleanup complete: {deleted_count} assets deleted")
        return deleted_count

    except subprocess.TimeoutExpired:
        logger.error("gh command timed out during cleanup")
        return deleted_count
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return deleted_count


def ensure_db(db_path: str, repo: str | None = None) -> str:
    """Ensure DB exists, creating or restoring if needed.

    Recovery chain:
    1. If DB exists, return path
    2. Try restore from GitHub Release
    3. Create empty DB

    Args:
        db_path: Path to DB file
        repo: Repository in format "owner/repo" (optional)

    Returns:
        Path to DB file
    """
    if os.path.exists(db_path):
        logger.info(f"DB already exists: {db_path}")
        return db_path

    logger.info(f"DB not found at {db_path}")

    # Try restore from Release if repo provided
    if repo:
        logger.info("Attempting restore from GitHub Release")
        if restore_from_release(repo, db_path):
            return db_path
        logger.info("No Release asset found, will create empty DB")

    # Create empty DB
    logger.info("Creating empty DB")
    from core.storage.db_manager import DBManager
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    db = DBManager(db_path)
    db.close()
    logger.info(f"Created empty DB at {db_path}")

    return db_path


def main() -> None:
    """CLI entry point for cache management."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="DB Cache Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore DB from Release")
    restore_parser.add_argument("--repo", required=True, help="Repository (owner/repo)")
    restore_parser.add_argument("--db-path", default="data/paper_scout.db", help="DB path")

    # upload command
    upload_parser = subparsers.add_parser("upload", help="Upload DB to Release")
    upload_parser.add_argument("--repo", required=True, help="Repository (owner/repo)")
    upload_parser.add_argument("--db-path", default="data/paper_scout.db", help="DB path")
    upload_parser.add_argument("--tag", required=True, help="Release tag")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old Release assets")
    cleanup_parser.add_argument("--repo", required=True, help="Repository (owner/repo)")
    cleanup_parser.add_argument("--keep-weeks", type=int, default=4, help="Weeks to keep")

    # ensure command
    ensure_parser = subparsers.add_parser("ensure", help="Ensure DB exists")
    ensure_parser.add_argument("--repo", help="Repository (owner/repo)")
    ensure_parser.add_argument("--db-path", default="data/paper_scout.db", help="DB path")

    # cache-key command
    key_parser = subparsers.add_parser("cache-key", help="Generate cache key")
    key_parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch")
    key_parser.add_argument("--db-path", help="DB path (optional)")

    args = parser.parse_args()

    if args.command == "restore":
        success = restore_from_release(args.repo, args.db_path)
        sys.exit(0 if success else 1)

    elif args.command == "upload":
        success = upload_to_release(args.repo, args.db_path, args.tag)
        sys.exit(0 if success else 1)

    elif args.command == "cleanup":
        deleted = cleanup_old_releases(args.repo, args.keep_weeks)
        logger.info(f"Deleted {deleted} assets")
        sys.exit(0)

    elif args.command == "ensure":
        ensure_db(args.db_path, args.repo)
        sys.exit(0)

    elif args.command == "cache-key":
        schema_hash = get_schema_hash(args.db_path)
        cache_key = get_cache_key(args.branch, schema_hash)
        print(cache_key)
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
