"""Search window computation for Paper Scout pipeline.

Implements incremental windowing with 2-level fallback chain
and manual date override support.

DevSpec Section 9-2 (Search Window).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

BUFFER_MINUTES = 30
FALLBACK_HOURS = 72


class SearchWindowComputer:
    """Compute search windows for paper collection.

    Implements incremental windowing with 2-level fallback chain
    and manual date override support.

    Fallback priority for window_start (auto mode):
      1. RunMeta(DB): latest completed run for this topic_slug
      2. 72-hour fallback: window_end - 72h

    window_end is always today UTC 00:00.
    Both sides get +-30min buffer added.
    """

    def __init__(
        self,
        db_manager: object = None,  # Optional DBManager for RunMeta lookup
        last_success_path: str = "data/last_success.json",
    ) -> None:
        self._db = db_manager

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

        # Guard: if window_start >= window_end (e.g. DB returned a time
        # after today UTC 00:00), clamp window_start to window_end - 24h
        # so we always search at least a 24-hour range.
        if window_start >= window_end:
            logger.warning(
                "Window inversion detected for %s: start=%s >= end=%s. "
                "Clamping start to end - 24h.",
                topic_slug,
                window_start,
                window_end,
            )
            window_start = window_end - timedelta(hours=24)

        return self._apply_buffer(window_start, window_end)

    def _compute_window_end(self, now: datetime) -> datetime:
        """Today UTC 00:00."""
        utc_now = now.astimezone(timezone.utc)
        return utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _compute_window_start(
        self, topic_slug: str, window_end: datetime
    ) -> datetime:
        """2-level fallback chain for window_start."""
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
                        "Falling through to 72h fallback.",
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

        # Level 2: 72h fallback
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
