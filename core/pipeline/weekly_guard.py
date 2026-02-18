"""Weekly Task Guard for Paper Scout.

Implements UTC Sunday detection and weekly task duplication prevention.
Ensures weekly tasks (full crawling, RSS feed scraping) run at most once per ISO week.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_weekly_due(flag_path: str = "data/weekly_done.flag") -> bool:
    """Check if today is UTC Sunday AND weekly tasks haven't been done this week.

    Args:
        flag_path: Path to the weekly completion flag file.

    Returns:
        True if today is Sunday AND (flag is missing OR flag is from a previous week).
        False otherwise.
    """
    now = datetime.now(timezone.utc)

    # Check if today is Sunday (weekday 6 in isocalendar)
    iso_year, iso_week, iso_weekday = now.isocalendar()
    if iso_weekday != 7:  # Sunday is 7 in isocalendar
        return False

    # Get current week identifier
    current_week_id = f"{iso_year}-W{iso_week:02d}"

    # Check flag status
    flag_week_id = read_weekly_flag(flag_path)

    # Weekly tasks are due if no flag exists or flag is from a different week
    return flag_week_id != current_week_id


def mark_weekly_done(flag_path: str = "data/weekly_done.flag") -> None:
    """Create or update the weekly completion flag file with current ISO week.

    Args:
        flag_path: Path to the weekly completion flag file.
    """
    now = datetime.now(timezone.utc)
    iso_year, iso_week, _ = now.isocalendar()
    week_id = f"{iso_year}-W{iso_week:02d}"

    # Create parent directories if needed
    path = Path(flag_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write week identifier to file
    path.write_text(week_id, encoding="utf-8")


def read_weekly_flag(flag_path: str = "data/weekly_done.flag") -> Optional[str]:
    """Read the current weekly completion flag value.

    Args:
        flag_path: Path to the weekly completion flag file.

    Returns:
        The ISO week identifier (e.g., "2026-W07") if file exists, None otherwise.
    """
    path = Path(flag_path)

    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
        return content if content else None
    except (IOError, OSError):
        logger.warning("Failed to read weekly flag file: %s", flag_path, exc_info=True)
        return None
