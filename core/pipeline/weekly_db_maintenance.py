"""Weekly Database Maintenance Orchestrator.

Performs comprehensive weekly maintenance operations:
1. Purge old evaluations (90 days)
2. Purge old query_stats (90 days)
3. Purge old runs (90 days)
4. Purge orphan remind_tracking entries
5. Purge old papers (365 days)
6. VACUUM database to reclaim space
7. Upload database as GitHub Release asset
8. Delete Release assets older than retention period
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.storage.db_manager import DBManager

logger = logging.getLogger(__name__)


def run_weekly_maintenance(
    db_path: str = "data/paper_scout.db",
    eval_days: int = 90,
    papers_days: int = 365,
    release_tag: str = "db-backup",
    asset_retention_days: int = 28,
) -> dict[str, Any]:
    """Run all weekly DB maintenance operations.

    Args:
        db_path: Path to the SQLite database file.
        eval_days: Days to retain evaluations (default: 90).
        papers_days: Days to retain papers (default: 365).
        release_tag: GitHub Release tag for asset upload (default: "db-backup").
        asset_retention_days: Days to retain Release assets (default: 28).

    Returns:
        Summary dict with purge counts and status:
        {
            "purged_evaluations": int,
            "purged_query_stats": int,
            "purged_runs": int,
            "purged_remind_tracking": int,
            "purged_papers": int,
            "vacuum_done": bool,
            "release_asset_uploaded": bool,
            "old_assets_deleted": int,
        }
    """
    summary: dict[str, Any] = {
        "purged_evaluations": 0,
        "purged_query_stats": 0,
        "purged_runs": 0,
        "purged_remind_tracking": 0,
        "purged_papers": 0,
        "vacuum_done": False,
        "release_asset_uploaded": False,
        "old_assets_deleted": 0,
    }

    logger.info("Starting weekly database maintenance")

    # Open database connection
    db = DBManager(db_path)

    try:
        # Phase 1: Purge operations (order matters - foreign key constraints)
        logger.info("Phase 1: Purging old evaluations (>%d days)", eval_days)
        try:
            summary["purged_evaluations"] = db.purge_old_evaluations(eval_days)
            logger.info("Purged %d evaluations", summary["purged_evaluations"])
        except (sqlite3.Error, AttributeError) as e:
            logger.warning("Failed to purge evaluations: %s", e)

        logger.info("Phase 2: Purging old query_stats (>%d days)", eval_days)
        try:
            summary["purged_query_stats"] = db.purge_old_query_stats(eval_days)
            logger.info("Purged %d query_stats", summary["purged_query_stats"])
        except (sqlite3.Error, AttributeError) as e:
            logger.warning("Failed to purge query_stats: %s", e)

        logger.info("Phase 3: Purging old runs (>%d days)", eval_days)
        try:
            summary["purged_runs"] = db.purge_old_runs(eval_days)
            logger.info("Purged %d runs", summary["purged_runs"])
        except (sqlite3.Error, AttributeError) as e:
            logger.warning("Failed to purge runs: %s", e)

        logger.info("Phase 4: Purging orphan remind_tracking entries")
        try:
            summary["purged_remind_tracking"] = db.purge_orphan_remind_tracking()
            logger.info("Purged %d remind_tracking entries", summary["purged_remind_tracking"])
        except (sqlite3.Error, AttributeError) as e:
            logger.warning("Failed to purge remind_tracking: %s", e)

        logger.info("Phase 5: Purging old papers (>%d days)", papers_days)
        try:
            summary["purged_papers"] = db.purge_old_papers(papers_days)
            logger.info("Purged %d papers", summary["purged_papers"])
        except (sqlite3.Error, AttributeError) as e:
            logger.warning("Failed to purge papers: %s", e)

        # Phase 2: VACUUM
        logger.info("Phase 6: Running VACUUM to reclaim space")
        try:
            db.vacuum()
            summary["vacuum_done"] = True
            logger.info("VACUUM completed successfully")
        except sqlite3.Error as e:
            logger.warning("Failed to run VACUUM: %s", e)

    finally:
        db.close()

    # Phase 3: Release asset backup
    logger.info("Phase 7: Uploading database as Release asset")
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    try:
        uploaded = _upload_release_asset(db_path, release_tag, date_str)
        summary["release_asset_uploaded"] = uploaded
        if uploaded:
            logger.info("Successfully uploaded Release asset")
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Failed to upload Release asset: %s", e)

    # Phase 4: Cleanup old Release assets
    logger.info("Phase 8: Cleaning up old Release assets (>%d days)", asset_retention_days)
    try:
        deleted_count = _cleanup_old_assets(release_tag, asset_retention_days)
        summary["old_assets_deleted"] = deleted_count
        logger.info("Deleted %d old Release assets", deleted_count)
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Failed to cleanup old Release assets: %s", e)

    logger.info("Weekly maintenance completed: %s", summary)
    return summary


def _upload_release_asset(db_path: str, release_tag: str, date_str: str) -> bool:
    """Upload DB file as GitHub Release asset.

    Args:
        db_path: Path to the database file to upload.
        release_tag: GitHub Release tag to attach asset to.
        date_str: Date string for asset filename (YYYYMMDD format).

    Returns:
        True if upload succeeded, False otherwise.
    """
    # Check if gh CLI is available
    if not shutil.which("gh"):
        logger.warning("gh CLI not found, skipping Release asset upload")
        return False

    # Check if GITHUB_TOKEN is set
    if not os.environ.get("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN not set, skipping Release asset upload")
        return False

    # Check if database file exists
    db_file = Path(db_path)
    if not db_file.exists():
        logger.warning("Database file not found: %s", db_path)
        return False

    # Asset filename format: paper-scout-db-{YYYYMMDD}.sqlite
    asset_name = f"paper-scout-db-{date_str}.sqlite"

    try:
        # Upload asset using gh CLI
        result = subprocess.run(
            [
                "gh",
                "release",
                "upload",
                release_tag,
                db_path,
                "--clobber",  # Overwrite if exists
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            logger.warning(
                "gh release upload failed (exit %d): %s",
                result.returncode,
                result.stderr,
            )
            return False

        logger.info("Uploaded Release asset: %s", asset_name)
        return True

    except subprocess.TimeoutExpired:
        logger.warning("gh release upload timed out after 120 seconds")
        return False
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("gh release upload failed: %s", e)
        return False


def _cleanup_old_assets(release_tag: str, retention_days: int) -> int:
    """Delete Release assets older than retention_days.

    Args:
        release_tag: GitHub Release tag to clean up.
        retention_days: Delete assets older than this many days.

    Returns:
        Number of assets deleted.
    """
    # Check if gh CLI is available
    if not shutil.which("gh"):
        logger.warning("gh CLI not found, skipping asset cleanup")
        return 0

    # Check if GITHUB_TOKEN is set
    if not os.environ.get("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN not set, skipping asset cleanup")
        return 0

    try:
        # List assets for the release
        result = subprocess.run(
            [
                "gh",
                "release",
                "view",
                release_tag,
                "--json",
                "assets",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(
                "gh release view failed (exit %d): %s",
                result.returncode,
                result.stderr,
            )
            return 0

        # Parse JSON output
        data = json.loads(result.stdout)
        assets = data.get("assets", [])

        # Calculate cutoff date
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        # Filter old assets
        deleted_count = 0
        for asset in assets:
            asset_name = asset.get("name", "")
            updated_at_str = asset.get("updatedAt", "")

            # Skip if no name or timestamp
            if not asset_name or not updated_at_str:
                continue

            # Parse timestamp
            try:
                # GitHub API returns ISO 8601 format
                updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning("Failed to parse timestamp for asset: %s", asset_name)
                continue

            # Delete if older than cutoff
            if updated_at < cutoff:
                logger.info("Deleting old Release asset: %s (updated: %s)", asset_name, updated_at_str)
                delete_result = subprocess.run(
                    [
                        "gh",
                        "release",
                        "delete-asset",
                        release_tag,
                        asset_name,
                        "--yes",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )

                if delete_result.returncode == 0:
                    deleted_count += 1
                else:
                    logger.warning(
                        "Failed to delete asset %s: %s",
                        asset_name,
                        delete_result.stderr,
                    )

        return deleted_count

    except subprocess.TimeoutExpired:
        logger.warning("gh release view timed out after 30 seconds")
        return 0
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as e:
        logger.warning("Asset cleanup failed: %s", e)
        return 0
