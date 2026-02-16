"""Rate limiter with RPM sliding window and daily usage persistence.

Provides dynamic delay calculation based on detected RPM, sliding window
enforcement, and daily API call tracking with file-based persistence.

DevSpec Sections: 16-2 (Daily Usage Persistence), 16-3 (Dynamic Delay).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RPM = 10
_DEFAULT_DAILY_LIMIT = 200
_DELAY_BUFFER = 0.5
_WINDOW_SECONDS = 60.0


def _today_key() -> str:
    """Return today's date as YYYYMMDD string."""
    return date.today().strftime("%Y%m%d")


def _today_iso() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().isoformat()


class RateLimiter:
    """Rate limiter with RPM sliding window and daily usage persistence.

    Args:
        detected_rpm: RPM from preflight detection (fallback: 10).
        daily_limit: Daily API call limit (fallback: 200).
        usage_dir: Directory for daily usage files.
    """

    def __init__(
        self,
        detected_rpm: int = _DEFAULT_RPM,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
        usage_dir: str = "data/usage",
    ) -> None:
        self._detected_rpm = detected_rpm if detected_rpm > 0 else _DEFAULT_RPM
        self._daily_limit = daily_limit if daily_limit > 0 else _DEFAULT_DAILY_LIMIT
        self._usage_dir = Path(usage_dir)
        self._call_timestamps: deque[float] = deque()
        self._usage = self.load_usage()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def delay(self) -> float:
        """Dynamic delay: 60 / detected_rpm + 0.5."""
        return _WINDOW_SECONDS / self._detected_rpm + _DELAY_BUFFER

    @property
    def daily_calls_remaining(self) -> int:
        """Remaining API calls for today."""
        return max(0, self._daily_limit - self._usage["api_calls"])

    @property
    def is_daily_limit_reached(self) -> bool:
        """True if daily limit reached."""
        return self._usage["api_calls"] >= self._daily_limit

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def wait(self) -> None:
        """Sleep for the dynamic delay, enforcing RPM sliding window.

        Ensures:
        1. Dynamic delay between consecutive calls.
        2. RPM sliding window -- if needed, wait longer.
        """
        now = time.time()

        # Purge timestamps older than 60 seconds.
        self._purge_old_timestamps(now)

        # If we have hit the RPM limit within the window, wait until
        # the oldest call falls outside the window.
        if len(self._call_timestamps) >= self._detected_rpm:
            oldest = self._call_timestamps[0]
            extra_wait = _WINDOW_SECONDS - (now - oldest)
            if extra_wait > 0:
                time.sleep(extra_wait)
                now = time.time()
                self._purge_old_timestamps(now)

        # Always enforce the minimum inter-call delay.
        time.sleep(self.delay)

    def record_call(self) -> None:
        """Increment today's API call count and save to file."""
        now = time.time()
        self._call_timestamps.append(now)
        self._usage["api_calls"] += 1
        self.save_usage()

    def record_topic_completed(self, topic_slug: str) -> None:
        """Add topic to completed list (if not already present)."""
        if topic_slug not in self._usage["topics_completed"]:
            self._usage["topics_completed"].append(topic_slug)
            self.save_usage()

    def record_topic_skipped(self, topic_slug: str) -> None:
        """Add topic to skipped list (if not already present)."""
        if topic_slug not in self._usage["topics_skipped"]:
            self._usage["topics_skipped"].append(topic_slug)
            self.save_usage()

    def should_skip_topic(self, topic_slug: str) -> bool:
        """True if daily limit reached or topic already completed."""
        if self.is_daily_limit_reached:
            return True
        if topic_slug in self._usage["topics_completed"]:
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_usage(self) -> None:
        """Persist current usage to data/usage/YYYYMMDD.json."""
        self._usage_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._usage_file_path()
        try:
            file_path.write_text(
                json.dumps(self._usage, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save usage file: %s", file_path)

    def load_usage(self) -> dict[str, Any]:
        """Load today's usage file.  Return default dict if not found."""
        file_path = self._usage_file_path()
        try:
            if file_path.exists():
                data = json.loads(file_path.read_text(encoding="utf-8"))
                return self._validated_usage(data)
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            logger.warning(
                "Corrupted or unreadable usage file: %s. Starting from 0.",
                file_path,
            )
        return self._default_usage()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _usage_file_path(self) -> Path:
        """Return the path for today's usage JSON file."""
        return self._usage_dir / f"{_today_key()}.json"

    def _purge_old_timestamps(self, now: float) -> None:
        """Remove timestamps older than 60 seconds."""
        cutoff = now - _WINDOW_SECONDS
        while self._call_timestamps and self._call_timestamps[0] < cutoff:
            self._call_timestamps.popleft()

    @staticmethod
    def _default_usage() -> dict[str, Any]:
        """Return a fresh usage dict for today."""
        return {
            "date": _today_iso(),
            "api_calls": 0,
            "topics_completed": [],
            "topics_skipped": [],
        }

    @staticmethod
    def _validated_usage(data: Any) -> dict[str, Any]:
        """Validate and normalise a loaded usage dict.

        Returns a valid usage dict or raises to trigger fallback.
        """
        if not isinstance(data, dict):
            raise TypeError("usage data is not a dict")

        return {
            "date": str(data.get("date", _today_iso())),
            "api_calls": int(data["api_calls"]),
            "topics_completed": list(data.get("topics_completed", [])),
            "topics_skipped": list(data.get("topics_skipped", [])),
        }
