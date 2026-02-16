"""Prune old report directories from gh-pages deployment.

This script deletes report directories older than the retention period
while preserving root-level index.html and latest.html files.

Usage:
    python scripts/prune_gh_pages.py --publish-dir tmp/reports --retention-days 90
    python scripts/prune_gh_pages.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DirectoryInfo(NamedTuple):
    """Information about a dated directory."""

    path: Path
    date: datetime.date


class PruneSummary(NamedTuple):
    """Summary of pruning operation."""

    deleted_count: int
    kept_count: int
    deleted_dirs: list[str]
    kept_dirs: list[str]


# Date pattern for YYYY-MM-DD directories
DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def find_dated_directories(base_dir: str) -> list[DirectoryInfo]:
    """Find all YYYY-MM-DD directories under reports/.

    Args:
        base_dir: Base directory containing reports/ folder

    Returns:
        List of DirectoryInfo tuples with path and parsed date
    """
    base_path = Path(base_dir)
    reports_dir = base_path / "reports"

    if not reports_dir.exists():
        logger.debug(f"Reports directory not found: {reports_dir}")
        return []

    if not reports_dir.is_dir():
        logger.warning(f"Reports path exists but is not a directory: {reports_dir}")
        return []

    dated_dirs: list[DirectoryInfo] = []

    for entry in reports_dir.iterdir():
        if not entry.is_dir():
            continue

        match = DATE_PATTERN.match(entry.name)
        if not match:
            logger.debug(f"Skipping non-date directory: {entry.name}")
            continue

        try:
            year, month, day = map(int, match.groups())
            date_obj = datetime.date(year, month, day)
            dated_dirs.append(DirectoryInfo(path=entry, date=date_obj))
        except ValueError as e:
            logger.warning(f"Invalid date in directory name {entry.name}: {e}")
            continue

    logger.info(f"Found {len(dated_dirs)} dated directories")
    return dated_dirs


def prune_old_directories(
    base_dir: str,
    retention_days: int,
    dry_run: bool = False,
) -> PruneSummary:
    """Delete report directories older than retention period.

    Args:
        base_dir: Base directory containing reports/ folder
        retention_days: Number of days to keep (older directories are deleted)
        dry_run: If True, print what would be deleted without actually deleting

    Returns:
        PruneSummary with deletion statistics
    """
    if retention_days < 0:
        raise ValueError(f"retention_days must be non-negative, got {retention_days}")

    base_path = Path(base_dir)

    # Verify base directory exists
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Find all dated directories
    dated_dirs = find_dated_directories(base_dir)

    if not dated_dirs:
        logger.info("No dated directories found")
        return PruneSummary(
            deleted_count=0,
            kept_count=0,
            deleted_dirs=[],
            kept_dirs=[],
        )

    # Calculate cutoff date
    cutoff_date = datetime.date.today() - datetime.timedelta(days=retention_days)
    logger.info(f"Cutoff date: {cutoff_date} (retention: {retention_days} days)")

    deleted_dirs: list[str] = []
    kept_dirs: list[str] = []

    for dir_info in dated_dirs:
        if dir_info.date < cutoff_date:
            # Directory is too old, delete it
            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {dir_info.path.name}")
            else:
                try:
                    shutil.rmtree(dir_info.path)
                    logger.info(f"Deleted: {dir_info.path.name}")
                except OSError as e:
                    logger.error(f"Failed to delete {dir_info.path.name}: {e}")
                    continue

            deleted_dirs.append(dir_info.path.name)
        else:
            # Directory is recent, keep it
            logger.debug(f"Keeping: {dir_info.path.name}")
            kept_dirs.append(dir_info.path.name)

    # Verify root files are preserved
    for root_file in ["index.html", "latest.html"]:
        root_path = base_path / root_file
        if root_path.exists():
            logger.debug(f"Root file preserved: {root_file}")

    return PruneSummary(
        deleted_count=len(deleted_dirs),
        kept_count=len(kept_dirs),
        deleted_dirs=deleted_dirs,
        kept_dirs=kept_dirs,
    )


def main() -> int:
    """CLI entry point for pruning gh-pages directories.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Prune old report directories from gh-pages deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prune directories older than 90 days
  python scripts/prune_gh_pages.py --publish-dir tmp/reports --retention-days 90

  # Dry run to preview what would be deleted
  python scripts/prune_gh_pages.py --publish-dir tmp/reports --dry-run

  # Use default settings (tmp/reports, 90 days retention)
  python scripts/prune_gh_pages.py
        """,
    )

    parser.add_argument(
        "--publish-dir",
        type=str,
        default="tmp/reports",
        help="Directory to prune (default: tmp/reports)",
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        default=90,
        help="Number of days to keep (default: 90)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Starting gh-pages pruning script")
    logger.info(f"Publish directory: {args.publish_dir}")
    logger.info(f"Retention period: {args.retention_days} days")
    logger.info(f"Dry run mode: {args.dry_run}")
    logger.info("=" * 60)

    try:
        summary = prune_old_directories(
            base_dir=args.publish_dir,
            retention_days=args.retention_days,
            dry_run=args.dry_run,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Pruning Summary:")
        logger.info(f"  Deleted: {summary.deleted_count} directories")
        logger.info(f"  Kept: {summary.kept_count} directories")

        if summary.deleted_dirs:
            logger.info("  Deleted directories:")
            for dir_name in sorted(summary.deleted_dirs):
                logger.info(f"    - {dir_name}")

        if args.dry_run:
            logger.info("[DRY RUN] No actual changes were made")

        logger.info("=" * 60)
        logger.info("Pruning completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Pruning failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
