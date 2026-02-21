"""Prune old report directories from gh-pages deployment.

This script deletes report directories older than the retention period
while preserving root-level index.html and weekly report folders.

Supports three directory formats:
- New daily: YYMMDD_daily_report/
- New weekly: YYMMWNN_weekly_report/ (permanent by default)
- Legacy: YYYY-MM-DD/ (pruned by daily retention)

Usage:
    python scripts/prune_gh_pages.py --publish-dir tmp/reports --daily-retention-days 180
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
    dir_type: str  # "daily", "weekly", "legacy"


class PruneSummary(NamedTuple):
    """Summary of pruning operation."""

    deleted_count: int
    kept_count: int
    deleted_dirs: list[str]
    kept_dirs: list[str]


# Directory patterns
DAILY_DIR_PATTERN = re.compile(r"^(\d{6})_daily_report$")
WEEKLY_DIR_PATTERN = re.compile(r"^(\d{4})W(\d{2})_weekly_report$")
LEGACY_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Legacy root-level file patterns (always cleaned up)
LEGACY_DAILY_FILE = re.compile(
    r"^(\d{6})_daily_paper_report(?:_readonly)?\.html$"
)
LEGACY_WEEKLY_FILE = re.compile(
    r"^(\d{8})_weekly_(?:summary|paper_report)\.(?:html|md)$"
)


def find_report_directories(base_dir: str) -> list[DirectoryInfo]:
    """Find all report directories (daily, weekly, legacy).

    Scans the base directory directly (not a reports/ subfolder).

    Args:
        base_dir: Directory containing report folders.

    Returns:
        List of DirectoryInfo tuples with path, parsed date, and type.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.debug("Directory not found: %s", base_path)
        return []

    if not base_path.is_dir():
        logger.warning("Path exists but is not a directory: %s", base_path)
        return []

    dirs: list[DirectoryInfo] = []

    for entry in base_path.iterdir():
        if not entry.is_dir():
            continue

        # New daily folders: YYMMDD_daily_report
        m = DAILY_DIR_PATTERN.match(entry.name)
        if m:
            try:
                date_obj = datetime.datetime.strptime(
                    m.group(1), "%y%m%d"
                ).date()
                dirs.append(DirectoryInfo(path=entry, date=date_obj, dir_type="daily"))
            except ValueError:
                logger.warning("Invalid date in daily dir: %s", entry.name)
            continue

        # Weekly folders: YYMMWNN_weekly_report
        m = WEEKLY_DIR_PATTERN.match(entry.name)
        if m:
            try:
                yymm = m.group(1)
                year = 2000 + int(yymm[:2])
                month = int(yymm[2:4])
                date_obj = datetime.date(year, month, 1)
                dirs.append(DirectoryInfo(path=entry, date=date_obj, dir_type="weekly"))
            except ValueError:
                logger.warning("Invalid date in weekly dir: %s", entry.name)
            continue

        # Legacy: YYYY-MM-DD
        m = LEGACY_DATE_PATTERN.match(entry.name)
        if m:
            try:
                year, month, day = map(int, m.groups())
                date_obj = datetime.date(year, month, day)
                dirs.append(DirectoryInfo(path=entry, date=date_obj, dir_type="legacy"))
            except ValueError:
                logger.warning("Invalid date in legacy dir: %s", entry.name)
            continue

    logger.info(
        "Found %d report directories (daily=%d, weekly=%d, legacy=%d)",
        len(dirs),
        sum(1 for d in dirs if d.dir_type == "daily"),
        sum(1 for d in dirs if d.dir_type == "weekly"),
        sum(1 for d in dirs if d.dir_type == "legacy"),
    )
    return dirs


def _cleanup_legacy_files(base_dir: str, dry_run: bool = False) -> list[str]:
    """Remove legacy root-level report files.

    Returns list of deleted filenames.
    """
    base_path = Path(base_dir)
    deleted: list[str] = []

    if not base_path.is_dir():
        return deleted

    for entry in base_path.iterdir():
        if not entry.is_file():
            continue
        if entry.name == "index.html":
            continue

        if LEGACY_DAILY_FILE.match(entry.name) or LEGACY_WEEKLY_FILE.match(entry.name):
            if dry_run:
                logger.info("[DRY RUN] Would delete legacy file: %s", entry.name)
            else:
                entry.unlink()
                logger.info("Deleted legacy file: %s", entry.name)
            deleted.append(entry.name)

    return deleted


def prune_old_directories(
    base_dir: str,
    daily_retention_days: int = 180,
    weekly_retention_days: int = 0,
    dry_run: bool = False,
) -> PruneSummary:
    """Delete report directories older than retention period.

    Args:
        base_dir: Directory containing report folders.
        daily_retention_days: Days to keep daily and legacy folders.
        weekly_retention_days: Days to keep weekly folders. 0 = permanent.
        dry_run: If True, print what would be deleted without deleting.

    Returns:
        PruneSummary with deletion statistics.
    """
    if daily_retention_days < 0:
        raise ValueError(
            f"daily_retention_days must be non-negative, got {daily_retention_days}"
        )

    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Find all report directories
    report_dirs = find_report_directories(base_dir)

    # Also clean up legacy root-level files
    legacy_files = _cleanup_legacy_files(base_dir, dry_run=dry_run)

    if not report_dirs and not legacy_files:
        logger.info("No report directories or legacy files found")
        return PruneSummary(
            deleted_count=0,
            kept_count=0,
            deleted_dirs=[],
            kept_dirs=[],
        )

    daily_cutoff = datetime.date.today() - datetime.timedelta(days=daily_retention_days)
    logger.info(
        "Daily cutoff: %s (retention: %d days)", daily_cutoff, daily_retention_days
    )

    deleted_dirs: list[str] = list(legacy_files)
    kept_dirs: list[str] = []

    for dir_info in report_dirs:
        if dir_info.dir_type == "weekly":
            # Weekly folders: permanent if weekly_retention_days == 0
            if weekly_retention_days == 0:
                kept_dirs.append(dir_info.path.name)
                continue
            weekly_cutoff = datetime.date.today() - datetime.timedelta(
                days=weekly_retention_days
            )
            if dir_info.date < weekly_cutoff:
                if dry_run:
                    logger.info("[DRY RUN] Would delete: %s", dir_info.path.name)
                else:
                    try:
                        shutil.rmtree(dir_info.path)
                        logger.info("Deleted weekly: %s", dir_info.path.name)
                    except OSError as e:
                        logger.error("Failed to delete %s: %s", dir_info.path.name, e)
                        continue
                deleted_dirs.append(dir_info.path.name)
            else:
                kept_dirs.append(dir_info.path.name)

        elif dir_info.dir_type in ("daily", "legacy"):
            if dir_info.date < daily_cutoff:
                if dry_run:
                    logger.info("[DRY RUN] Would delete: %s", dir_info.path.name)
                else:
                    try:
                        shutil.rmtree(dir_info.path)
                        logger.info("Deleted: %s", dir_info.path.name)
                    except OSError as e:
                        logger.error("Failed to delete %s: %s", dir_info.path.name, e)
                        continue
                deleted_dirs.append(dir_info.path.name)
            else:
                kept_dirs.append(dir_info.path.name)

    # Verify root files are preserved
    root_index = base_path / "index.html"
    if root_index.exists():
        logger.debug("Root file preserved: index.html")

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
  # Prune daily directories older than 180 days
  python scripts/prune_gh_pages.py --publish-dir tmp/reports --daily-retention-days 180

  # Dry run to preview what would be deleted
  python scripts/prune_gh_pages.py --publish-dir tmp/reports --dry-run

  # Use default settings (tmp/reports, 180 days daily, permanent weekly)
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
        "--daily-retention-days",
        type=int,
        default=180,
        help="Days to keep daily report folders (default: 180)",
    )

    parser.add_argument(
        "--weekly-retention-days",
        type=int,
        default=0,
        help="Days to keep weekly report folders, 0=permanent (default: 0)",
    )

    # Backward compat alias
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Deprecated: use --daily-retention-days instead",
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

    # Backward compat: --retention-days fallback
    daily_retention = args.daily_retention_days
    if args.retention_days is not None:
        logger.warning("--retention-days is deprecated, use --daily-retention-days")
        if daily_retention == 180:
            daily_retention = args.retention_days

    logger.info("=" * 60)
    logger.info("Starting gh-pages pruning script")
    logger.info("Publish directory: %s", args.publish_dir)
    logger.info("Daily retention: %d days", daily_retention)
    logger.info("Weekly retention: %s", "permanent" if args.weekly_retention_days == 0 else f"{args.weekly_retention_days} days")
    logger.info("Dry run mode: %s", args.dry_run)
    logger.info("=" * 60)

    try:
        summary = prune_old_directories(
            base_dir=args.publish_dir,
            daily_retention_days=daily_retention,
            weekly_retention_days=args.weekly_retention_days,
            dry_run=args.dry_run,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Pruning Summary:")
        logger.info("  Deleted: %d items", summary.deleted_count)
        logger.info("  Kept: %d items", summary.kept_count)

        if summary.deleted_dirs:
            logger.info("  Deleted items:")
            for dir_name in sorted(summary.deleted_dirs):
                logger.info("    - %s", dir_name)

        if args.dry_run:
            logger.info("[DRY RUN] No actual changes were made")

        logger.info("=" * 60)
        logger.info("Pruning completed successfully")

        return 0

    except Exception as e:
        logger.error("Pruning failed: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
