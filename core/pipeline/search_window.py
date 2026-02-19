"""Search window computation for Paper Scout pipeline.

Implements incremental windowing with 3-level fallback chain
and manual date override support.

DevSpec Section 9-2 (Search Window).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

BUFFER_MINUTES = 30
FALLBACK_HOURS = 72


class SearchWindowComputer:
    """Compute search windows for paper collection.

    Implements incremental windowing with 3-level fallback chain
    and manual date override support.

    Fallback priority for window_start (auto mode):
      1. RunMeta(DB): latest completed run for this topic_slug
      2. data/last_success.json: topic-specific last_success_window_end_utc
      3. 72-hour fallback: window_end - 72h

    window_end is always today UTC 00:00.
    Both sides get +-30min buffer added.
    """

    def __init__(
        self,
        db_manager: object = None,  # Optional DBManager for RunMeta lookup
        last_success_path: str = "data/last_success.json",
    ) -> None:
        self._db = db_manager
        self._last_success_path = last_success_path

    def compute(
        self,
        topic_slug: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        now: Optional[datetime] = None,  # For testing
    ) -> Tuple[datetime, datetime]:
        """Compute (window_start, window_end) with +-30min buffer.

        Args:
            topic_slug: Topic identifier for per-topic window lookup.
            date_from: Manual start override (UTC).
            date_to: Manual end override (UTC).
            now: Override "now" for testing (UTC).

        Returns:
            Tuple of (window_start_utc, window_end_utc) with buffer applied.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if date_from is not None and date_to is not None:
            # Manual mode: parse strings to datetime if needed
            if isinstance(date_from, str):
                date_from = datetime.fromisoformat(date_from)
            if isinstance(date_to, str):
                date_to = datetime.fromisoformat(date_to)
            # Ensure timezone-aware
            if date_from.tzinfo is None:
                date_from = date_from.replace(tzinfo=timezone.utc)
            if date_to.tzinfo is None:
                date_to = date_to.replace(tzinfo=timezone.utc)
            # Manual mode: snap to exact day boundaries (no buffer)
            date_from = date_from.replace(hour=0, minute=0, second=0, microsecond=0)
            date_to = date_to.replace(hour=23, minute=59, second=59, microsecond=0)
            return (date_from, date_to)

        # Auto mode: compute window_end from UTC 00:00
        window_end = self._compute_window_end(now)

        # Compute window_start from fallback chain
        window_start = self._compute_window_start(topic_slug, window_end)

        return self._apply_buffer(window_start, window_end)

    def _compute_window_end(self, now: datetime) -> datetime:
        """Today UTC 00:00."""
        utc_now = now.astimezone(timezone.utc)
        return utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _compute_window_start(
        self, topic_slug: str, window_end: datetime
    ) -> datetime:
        """3-level fallback chain for window_start."""
        # Level 1: DB RunMeta (with sanity check against window_end)
        if self._db is not None:
            run = self._db.get_latest_completed_run(topic_slug)
            if run is not None and run.window_end_utc is not None:
                ws = run.window_end_utc
                # Ensure timezone-aware for safe comparison
                if ws.tzinfo is None:
                    ws = ws.replace(tzinfo=timezone.utc)
                # Sanity check: DB date should not be more than 90 days
                # before window_end.  Stale test runs can leave old dates
                # (e.g. 2025 instead of 2026) that corrupt the search window.
                if (window_end - ws).days > 90:
                    logger.warning(
                        "DB window_start for %s is stale (%s, %d days old). "
                        "Falling through to last_success.json.",
                        topic_slug,
                        ws,
                        (window_end - ws).days,
                    )
                else:
                    logger.info(
                        "Window start from DB for %s: %s",
                        topic_slug,
                        ws,
                    )
                    return ws

        # Level 2: last_success.json
        last_success = self._load_last_success()
        if topic_slug in last_success:
            ts = last_success[topic_slug].get("last_success_window_end_utc")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid timestamp in last_success.json for %s: %s",
                        topic_slug,
                        ts,
                    )
                else:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    logger.info(
                        "Window start from last_success.json for %s: %s",
                        topic_slug,
                        dt,
                    )
                    return dt

        # Level 3: 72h fallback
        fallback = window_end - timedelta(hours=FALLBACK_HOURS)
        logger.info(
            "Window start from 72h fallback for %s: %s",
            topic_slug,
            fallback,
        )
        return fallback

    def _apply_buffer(
        self, start: datetime, end: datetime
    ) -> Tuple[datetime, datetime]:
        """Apply +-30min buffer."""
        buffer = timedelta(minutes=BUFFER_MINUTES)
        return (start - buffer, end + buffer)

    def _load_last_success(self) -> dict:
        """Load data/last_success.json or return empty dict."""
        try:
            if os.path.exists(self._last_success_path):
                with open(self._last_success_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load %s", self._last_success_path)
        return {}

    def update_last_success(
        self,
        topic_slug: str,
        window_end_utc: datetime,
    ) -> None:
        """Update last_success.json with max(existing, window_end_utc).

        Never overwrites with an older value (backfill protection).
        """
        data = self._load_last_success()

        existing_ts = None
        if topic_slug in data:
            existing_ts_str = data[topic_slug].get("last_success_window_end_utc")
            if existing_ts_str:
                try:
                    existing_ts = datetime.fromisoformat(existing_ts_str)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid existing timestamp for %s: %s",
                        topic_slug,
                        existing_ts_str,
                    )

        new_ts = window_end_utc
        # Ensure both are timezone-aware for safe comparison
        if new_ts.tzinfo is None:
            new_ts = new_ts.replace(tzinfo=timezone.utc)
        if existing_ts is not None:
            if existing_ts.tzinfo is None:
                existing_ts = existing_ts.replace(tzinfo=timezone.utc)
            if existing_ts > new_ts:
                new_ts = existing_ts  # Keep existing if newer

        data[topic_slug] = {"last_success_window_end_utc": new_ts.isoformat()}

        import tempfile

        dir_path = os.path.dirname(self._last_success_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._last_success_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
