"""2-tier dedup system for Paper Scout (Section 4-5).

Tier 1 (in-run): In-memory set checked per paper_key, always ON.
Tier 2 (cross-run): DB-backed check per (paper_key, topic_slug),
mode-dependent (skip_recent or none).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from core.models import Paper
from core.storage.db_manager import DBManager


class DedupManager:
    """2-tier dedup system for Paper Scout."""

    def __init__(
        self,
        db_manager: DBManager,
        seen_items_path: str = "data/seen_items.jsonl",
        rolling_days: int = 30,
        dedup_mode: str = "skip_recent",  # "skip_recent" or "none"
    ) -> None:
        """Initialize dedup manager.

        Args:
            db_manager: Database manager for papers table lookup.
            seen_items_path: Unused, kept for backward compatibility.
            rolling_days: Days to keep in rolling window (default 30).
            dedup_mode: Dedup mode - "skip_recent" or "none".
        """
        self._db = db_manager
        self._rolling_days = rolling_days
        self._dedup_mode = dedup_mode

        # Tier 1: in-run set (always ON)
        self._in_run_set: set[str] = set()

        # Tier 2: cross-run memory set (loaded from DB)
        self._seen_set: set[tuple[str, str]] = set()  # (paper_key, topic_slug)

        # Load seen keys from DB if mode is skip_recent
        if self._dedup_mode == "skip_recent":
            self._load_seen_items()

    def _load_seen_items(self) -> None:
        """Load seen (paper_key, topic_slug) pairs from DB."""
        try:
            self._seen_set = self._db.get_seen_paper_keys(self._rolling_days)
        except Exception:
            logger.warning("Failed to load seen paper keys from DB", exc_info=True)

    def reset_in_run(self) -> None:
        """Reset the in-run set for a new topic/run."""
        self._in_run_set.clear()

    def is_duplicate(self, paper_key: str, topic_slug: str) -> bool:
        """Check if paper should be skipped.

        Tier 1: Always check in_run set.
        Tier 2: Check seen_items if mode is skip_recent.

        If not duplicate, adds to in_run set.

        Returns:
            True if paper should be skipped (duplicate).
        """
        # Tier 1: in-run dedup (always ON)
        if paper_key in self._in_run_set:
            return True

        # Tier 2: cross-run dedup (mode-dependent)
        if self._dedup_mode == "skip_recent":
            if (paper_key, topic_slug) in self._seen_set:
                return True

        # Not a duplicate - add to in-run set
        self._in_run_set.add(paper_key)
        return False

    def get_existing_paper(self, paper_key: str) -> Paper | None:
        """Check papers table for metadata reuse.

        Returns existing Paper record if found, None if new paper.
        """
        return self._db.get_paper(paper_key)

    def mark_seen(self, paper_key: str, topic_slug: str) -> None:
        """Mark a paper as seen for this run.

        Only records if dedup_mode is skip_recent.
        """
        if self._dedup_mode != "skip_recent":
            return

        # Keep in-memory set in sync so subsequent is_duplicate checks work
        self._seen_set.add((paper_key, topic_slug))

    def save_seen_items(self) -> None:
        """No-op. Cross-run dedup data is now persisted via DB evaluations."""
        pass

    @property
    def in_run_count(self) -> int:
        """Number of papers in the in-run set."""
        return len(self._in_run_set)

    @property
    def new_items_count(self) -> int:
        """Number of new items added during this run."""
        return 0
