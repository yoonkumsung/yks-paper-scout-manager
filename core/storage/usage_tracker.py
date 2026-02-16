"""High-level usage tracking with rolling cleanup.

Wraps the same file-based persistence as RateLimiter and adds:
- Topics completed/skipped tracking with richer metadata
  (output counts for completed, reasons for skipped)
- 30-day rolling cleanup of old usage files
- Summary API for the orchestrator

DevSpec Section 16-2.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _today_key() -> str:
    """Return today's date as YYYYMMDD string."""
    return date.today().strftime("%Y%m%d")


def _today_iso() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().isoformat()


class UsageTracker:
    """High-level usage tracking with rolling cleanup.

    Wraps RateLimiter's daily persistence and adds:
    - Topics completed/skipped tracking with output counts and reasons
    - 30-day rolling cleanup of old usage files
    - Summary API for the orchestrator

    Note: The orchestrator manages the RateLimiter separately for RPM
    control.  UsageTracker only manages the file-based tracking with
    a richer format.
    """

    def __init__(self, usage_dir: str = "data/usage") -> None:
        self._usage_dir = Path(usage_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_today_usage(self) -> Dict[str, Any]:
        """Load today's usage file or return default structure."""
        file_path = self._today_file_path()
        try:
            if file_path.exists():
                raw = json.loads(file_path.read_text(encoding="utf-8"))
                return self._normalise(raw)
        except (json.JSONDecodeError, OSError, TypeError):
            logger.warning(
                "Corrupted or unreadable usage file: %s. Starting fresh.",
                file_path,
            )
        return self._default_usage()

    def save_usage(self, usage: Dict[str, Any]) -> None:
        """Save today's usage data to disk."""
        self._usage_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._today_file_path()
        try:
            file_path.write_text(
                json.dumps(usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save usage file: %s", file_path)

    def record_topic_completed(
        self, topic_slug: str, total_output: int
    ) -> None:
        """Record a topic as completed with paper count."""
        usage = self.get_today_usage()
        # Avoid duplicate entries for the same slug.
        existing_slugs = {
            entry["slug"] if isinstance(entry, dict) else entry
            for entry in usage["topics_completed"]
        }
        if topic_slug not in existing_slugs:
            usage["topics_completed"].append(
                {"slug": topic_slug, "total_output": total_output}
            )
            self.save_usage(usage)

    def record_topic_skipped(self, topic_slug: str, reason: str) -> None:
        """Record a topic as skipped with reason."""
        usage = self.get_today_usage()
        existing_slugs = {
            entry["slug"] if isinstance(entry, dict) else entry
            for entry in usage["topics_skipped"]
        }
        if topic_slug not in existing_slugs:
            usage["topics_skipped"].append(
                {"slug": topic_slug, "reason": reason}
            )
            self.save_usage(usage)

    def increment_api_calls(self, count: int = 1) -> None:
        """Increment today's API call count."""
        usage = self.get_today_usage()
        usage["api_calls"] += count
        self.save_usage(usage)

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of today's usage."""
        usage = self.get_today_usage()
        completed = usage["topics_completed"]
        skipped = usage["topics_skipped"]

        total_output = 0
        for entry in completed:
            if isinstance(entry, dict):
                total_output += entry.get("total_output", 0)

        return {
            "date": usage["date"],
            "api_calls": usage["api_calls"],
            "topics_completed_count": len(completed),
            "topics_skipped_count": len(skipped),
            "total_papers_output": total_output,
        }

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """Delete usage files older than *max_age_days*.

        Returns the number of files deleted.
        """
        if not self._usage_dir.exists():
            return 0

        cutoff = date.today() - timedelta(days=max_age_days)
        deleted = 0

        for path in self._usage_dir.glob("*.json"):
            file_date = self._parse_file_date(path)
            if file_date is not None and file_date < cutoff:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    logger.warning("Failed to delete old file: %s", path)

        return deleted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _today_file_path(self) -> Path:
        return self._usage_dir / f"{_today_key()}.json"

    @staticmethod
    def _default_usage() -> Dict[str, Any]:
        """Return a fresh usage dict for today."""
        return {
            "date": _today_iso(),
            "api_calls": 0,
            "topics_completed": [],
            "topics_skipped": [],
        }

    @staticmethod
    def _normalise(data: Any) -> Dict[str, Any]:
        """Normalise loaded data into the expected schema.

        Handles both the simple RateLimiter format (list of strings)
        and the richer UsageTracker format (list of dicts).
        """
        if not isinstance(data, dict):
            raise TypeError("usage data is not a dict")

        return {
            "date": str(data.get("date", _today_iso())),
            "api_calls": int(data.get("api_calls", 0)),
            "topics_completed": list(data.get("topics_completed", [])),
            "topics_skipped": list(data.get("topics_skipped", [])),
        }

    @staticmethod
    def _parse_file_date(path: Path) -> date | None:
        """Try to parse YYYYMMDD from a filename like ``20260210.json``."""
        stem = path.stem
        if len(stem) != 8 or not stem.isdigit():
            return None
        try:
            return date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))
        except ValueError:
            return None
