"""2-tier dedup system for Paper Scout (Section 4-5).

Tier 1 (in-run): In-memory set checked per paper_key, always ON.
Tier 2 (cross-run): seen_items.jsonl checked per (paper_key, topic_slug),
mode-dependent (skip_recent or none).
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

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
            seen_items_path: Path to seen_items.jsonl file.
            rolling_days: Days to keep in rolling window (default 30).
            dedup_mode: Dedup mode - "skip_recent" or "none".
        """
        self._db = db_manager
        self._seen_items_path = Path(seen_items_path)
        self._rolling_days = rolling_days
        self._dedup_mode = dedup_mode

        # Tier 1: in-run set (always ON)
        self._in_run_set: set[str] = set()

        # Tier 2: cross-run memory set (loaded from seen_items.jsonl)
        self._seen_items: list[dict] = []  # Full entries for save-back
        self._seen_set: set[tuple[str, str]] = set()  # (paper_key, topic_slug)

        # New items added during this run (for appending to seen_items)
        self._new_items: list[dict] = []

        # Load seen_items if mode is skip_recent
        if self._dedup_mode == "skip_recent":
            self._load_seen_items()

    def _load_seen_items(self) -> None:
        """Load seen_items.jsonl with 30-day rolling cleanup."""
        if not self._seen_items_path.exists():
            return

        cutoff = date.today() - timedelta(days=self._rolling_days)
        raw_lines = self._seen_items_path.read_text(encoding="utf-8").splitlines()

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            # Apply rolling window filter
            entry_date = date.fromisoformat(entry.get("date", "1970-01-01"))
            if entry_date < cutoff:
                continue

            paper_key = entry.get("paper_key")
            topic_slug = entry.get("topic_slug")
            if not paper_key or not topic_slug:
                continue
            self._seen_items.append(entry)
            self._seen_set.add((paper_key, topic_slug))

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
        """Mark a paper as seen for this run (for later save).

        Only records if dedup_mode is skip_recent.
        """
        if self._dedup_mode != "skip_recent":
            return

        entry = {
            "paper_key": paper_key,
            "topic_slug": topic_slug,
            "date": date.today().isoformat(),
        }
        self._new_items.append(entry)

    def save_seen_items(self) -> None:
        """Save seen_items.jsonl with new items appended.

        Only saves if dedup_mode is skip_recent.
        Applies 30-day rolling cleanup before save.
        """
        if self._dedup_mode != "skip_recent":
            return

        # Re-filter existing items through rolling window (in case
        # time has passed since load)
        cutoff = date.today() - timedelta(days=self._rolling_days)

        filtered_existing: list[dict] = []
        for entry in self._seen_items:
            entry_date = date.fromisoformat(entry.get("date", "1970-01-01"))
            if entry_date >= cutoff:
                filtered_existing.append(entry)

        # Filter new items through rolling window as well
        filtered_new: list[dict] = []
        for entry in self._new_items:
            entry_date = date.fromisoformat(entry.get("date", "1970-01-01"))
            if entry_date >= cutoff:
                filtered_new.append(entry)

        all_items = filtered_existing + filtered_new

        # Create parent directory if needed
        self._seen_items_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        lines = [json.dumps(item, ensure_ascii=False) for item in all_items]
        content = "\n".join(lines) + "\n" if lines else ""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._seen_items_path.parent),
                suffix=".tmp",
            )
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                Path(tmp_path).replace(self._seen_items_path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except OSError as oe:
            # Fallback: direct write if atomic write fails (not atomic)
            logger.warning(
                "Atomic write failed for %s, falling back to direct write: %s",
                self._seen_items_path, oe,
            )
            self._seen_items_path.write_text(content, encoding="utf-8")

    @property
    def in_run_count(self) -> int:
        """Number of papers in the in-run set."""
        return len(self._in_run_set)

    @property
    def new_items_count(self) -> int:
        """Number of new items added during this run."""
        return len(self._new_items)
