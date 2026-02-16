"""Tests for core.storage.dedup (P1 priority).

Covers the 2-tier dedup system: in-run dedup (Tier 1),
cross-run dedup via seen_items.jsonl (Tier 2), rolling cleanup,
metadata reuse from papers table, and mode-dependent behavior.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from core.models import Paper
from core.storage.db_manager import DBManager
from core.storage.dedup import DedupManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """Provide a fresh DBManager instance with temporary database."""
    db_path = str(tmp_path / "test.db")
    manager = DBManager(db_path=db_path)
    yield manager
    manager.close()


@pytest.fixture()
def seen_items_path(tmp_path):
    """Return a temporary path for seen_items.jsonl."""
    return str(tmp_path / "data" / "seen_items.jsonl")


def _make_paper(
    paper_key: str = "arxiv:2401.00001",
    run_id: int = 1,
) -> Paper:
    """Create a sample Paper dataclass."""
    return Paper(
        source="arxiv",
        native_id="2401.00001",
        paper_key=paper_key,
        url="https://arxiv.org/abs/2401.00001",
        title="Test Paper",
        abstract="An abstract about testing.",
        authors=["Alice", "Bob"],
        categories=["cs.AI"],
        published_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        first_seen_run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _write_seen_items(path: str, items: list[dict]) -> None:
    """Write a list of dicts as JSONL to the given path."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item) for item in items]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Tier 1 - in_run always ON
# ---------------------------------------------------------------------------


class TestTier1InRun:
    """Tier 1 in-run dedup must always operate regardless of mode."""

    def test_first_paper_not_duplicate(self, db, seen_items_path):
        """First occurrence of a paper_key should not be duplicate."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is False

    def test_same_paper_key_is_duplicate(self, db, seen_items_path):
        """Second occurrence of the same paper_key is a duplicate."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.is_duplicate("arxiv:2401.00001", "ai-sports")
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is True

    def test_in_run_works_in_none_mode(self, db, seen_items_path):
        """In-run dedup still works when dedup_mode is 'none'."""
        dedup = DedupManager(
            db, seen_items_path=seen_items_path, dedup_mode="none"
        )
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is False
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is True

    def test_in_run_count(self, db, seen_items_path):
        """in_run_count reflects the number of papers checked."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.is_duplicate("arxiv:p1", "topic-a")
        dedup.is_duplicate("arxiv:p2", "topic-a")
        assert dedup.in_run_count == 2


# ---------------------------------------------------------------------------
# 2. Tier 2 - cross_run with skip_recent
# ---------------------------------------------------------------------------


class TestTier2SkipRecent:
    """Tier 2 cross-run dedup with skip_recent mode."""

    def test_paper_in_seen_items_is_duplicate(self, db, seen_items_path):
        """Paper present in seen_items should be flagged as duplicate."""
        _write_seen_items(seen_items_path, [
            {
                "paper_key": "arxiv:2401.00001",
                "topic_slug": "ai-sports",
                "date": date.today().isoformat(),
            },
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is True

    def test_paper_not_in_seen_items_is_not_duplicate(
        self, db, seen_items_path
    ):
        """Paper not in seen_items should not be duplicate."""
        _write_seen_items(seen_items_path, [
            {
                "paper_key": "arxiv:other",
                "topic_slug": "ai-sports",
                "date": date.today().isoformat(),
            },
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:2401.99999", "ai-sports") is False

    def test_same_paper_different_topic_not_duplicate(
        self, db, seen_items_path
    ):
        """Same paper in a different topic is NOT deduped (independent)."""
        _write_seen_items(seen_items_path, [
            {
                "paper_key": "arxiv:2401.00001",
                "topic_slug": "ai-sports",
                "date": date.today().isoformat(),
            },
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        # Same paper, different topic -> should NOT be duplicate
        assert dedup.is_duplicate("arxiv:2401.00001", "nlp-general") is False


# ---------------------------------------------------------------------------
# 3. Tier 2 - cross_run with none mode
# ---------------------------------------------------------------------------


class TestTier2NoneMode:
    """Tier 2 cross-run dedup with none mode (disabled)."""

    def test_seen_items_not_loaded(self, db, seen_items_path):
        """In none mode, seen_items should NOT be loaded."""
        _write_seen_items(seen_items_path, [
            {
                "paper_key": "arxiv:2401.00001",
                "topic_slug": "ai-sports",
                "date": date.today().isoformat(),
            },
        ])

        dedup = DedupManager(
            db, seen_items_path=seen_items_path, dedup_mode="none"
        )
        # Paper is in seen_items but none mode does not read it
        assert dedup.is_duplicate("arxiv:2401.00001", "ai-sports") is False

    def test_save_seen_items_does_nothing(self, db, seen_items_path):
        """In none mode, save_seen_items should not create a file."""
        dedup = DedupManager(
            db, seen_items_path=seen_items_path, dedup_mode="none"
        )
        dedup.mark_seen("arxiv:2401.00001", "ai-sports")
        dedup.save_seen_items()

        from pathlib import Path

        assert not Path(seen_items_path).exists()


# ---------------------------------------------------------------------------
# 4. seen_items.jsonl file operations
# ---------------------------------------------------------------------------


class TestSeenItemsFile:
    """Tests for seen_items.jsonl loading, saving, and rolling cleanup."""

    def test_load_from_file(self, db, seen_items_path):
        """Entries should be parsed correctly from JSONL file."""
        today = date.today().isoformat()
        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:p1", "topic_slug": "topic-a", "date": today},
            {"paper_key": "arxiv:p2", "topic_slug": "topic-b", "date": today},
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:p1", "topic-a") is True
        assert dedup.is_duplicate("arxiv:p2", "topic-b") is True

    def test_rolling_cleanup_on_load(self, db, seen_items_path):
        """Entries older than rolling_days should be discarded on load."""
        old_date = (date.today() - timedelta(days=35)).isoformat()
        recent_date = (date.today() - timedelta(days=5)).isoformat()

        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:old", "topic_slug": "t", "date": old_date},
            {
                "paper_key": "arxiv:recent",
                "topic_slug": "t",
                "date": recent_date,
            },
        ])

        dedup = DedupManager(
            db, seen_items_path=seen_items_path, rolling_days=30
        )
        # Old entry discarded
        assert dedup.is_duplicate("arxiv:old", "t") is False
        # Recent entry kept
        assert dedup.is_duplicate("arxiv:recent", "t") is True

    def test_save_appends_new_items(self, db, seen_items_path):
        """Save should combine existing + new items."""
        today = date.today().isoformat()
        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:p1", "topic_slug": "t", "date": today},
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.mark_seen("arxiv:p2", "t")
        dedup.save_seen_items()

        # Read back the file
        from pathlib import Path

        lines = Path(seen_items_path).read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        keys = {e["paper_key"] for e in entries}
        assert "arxiv:p1" in keys
        assert "arxiv:p2" in keys

    def test_missing_file_starts_empty(self, db, seen_items_path):
        """If seen_items.jsonl does not exist, start with empty list."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:p1", "t") is False
        assert dedup.in_run_count == 1

    def test_empty_file_starts_empty(self, db, seen_items_path):
        """If seen_items.jsonl is empty, start with empty list."""
        from pathlib import Path

        p = Path(seen_items_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:p1", "t") is False

    def test_malformed_lines_skipped(self, db, seen_items_path):
        """Malformed JSON lines should be skipped without error."""
        from pathlib import Path

        p = Path(seen_items_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        content = (
            "not valid json\n"
            + json.dumps(
                {"paper_key": "arxiv:ok", "topic_slug": "t", "date": today}
            )
            + "\n"
        )
        p.write_text(content, encoding="utf-8")

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        assert dedup.is_duplicate("arxiv:ok", "t") is True


# ---------------------------------------------------------------------------
# 5. Papers table metadata reuse
# ---------------------------------------------------------------------------


class TestMetadataReuse:
    """Tests for get_existing_paper (papers table lookup)."""

    def test_paper_exists_in_db(self, db, seen_items_path):
        """get_existing_paper returns Paper when found in DB."""
        paper = _make_paper("arxiv:2401.00001")
        db.insert_paper(paper)

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        result = dedup.get_existing_paper("arxiv:2401.00001")
        assert result is not None
        assert result.paper_key == "arxiv:2401.00001"
        assert result.title == "Test Paper"

    def test_paper_not_in_db(self, db, seen_items_path):
        """get_existing_paper returns None when paper is not in DB."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        result = dedup.get_existing_paper("arxiv:nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# 6. mark_seen
# ---------------------------------------------------------------------------


class TestMarkSeen:
    """Tests for mark_seen recording behavior."""

    def test_mark_seen_skip_recent_records(self, db, seen_items_path):
        """In skip_recent mode, mark_seen should record the item."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.mark_seen("arxiv:p1", "topic-a")
        assert dedup.new_items_count == 1

    def test_mark_seen_none_mode_does_not_record(self, db, seen_items_path):
        """In none mode, mark_seen should NOT record the item."""
        dedup = DedupManager(
            db, seen_items_path=seen_items_path, dedup_mode="none"
        )
        dedup.mark_seen("arxiv:p1", "topic-a")
        assert dedup.new_items_count == 0


# ---------------------------------------------------------------------------
# 7. save_seen_items
# ---------------------------------------------------------------------------


class TestSaveSeenItems:
    """Tests for save_seen_items file output."""

    def test_combines_existing_and_new(self, db, seen_items_path):
        """Save should combine filtered existing + new items."""
        today = date.today().isoformat()
        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:old", "topic_slug": "t", "date": today},
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.mark_seen("arxiv:new", "t")
        dedup.save_seen_items()

        from pathlib import Path

        lines = Path(seen_items_path).read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        assert len(entries) == 2
        keys = {e["paper_key"] for e in entries}
        assert keys == {"arxiv:old", "arxiv:new"}

    def test_creates_parent_directory(self, db, tmp_path):
        """Save should create parent directory if it does not exist."""
        nested_path = str(tmp_path / "deep" / "nested" / "seen_items.jsonl")
        dedup = DedupManager(db, seen_items_path=nested_path)
        dedup.mark_seen("arxiv:p1", "t")
        dedup.save_seen_items()

        from pathlib import Path

        assert Path(nested_path).exists()

    def test_rolling_cleanup_on_save(self, db, seen_items_path):
        """Save should apply rolling cleanup, removing old entries."""
        old_date = (date.today() - timedelta(days=35)).isoformat()
        recent_date = (date.today() - timedelta(days=5)).isoformat()

        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:old", "topic_slug": "t", "date": old_date},
            {
                "paper_key": "arxiv:recent",
                "topic_slug": "t",
                "date": recent_date,
            },
        ])

        dedup = DedupManager(
            db, seen_items_path=seen_items_path, rolling_days=30
        )
        dedup.mark_seen("arxiv:new", "t")
        dedup.save_seen_items()

        from pathlib import Path

        lines = Path(seen_items_path).read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        keys = {e["paper_key"] for e in entries}
        # Old entry should be gone (already filtered on load)
        assert "arxiv:old" not in keys
        assert "arxiv:recent" in keys
        assert "arxiv:new" in keys


# ---------------------------------------------------------------------------
# 8. reset_in_run
# ---------------------------------------------------------------------------


class TestResetInRun:
    """Tests for reset_in_run behavior."""

    def test_clears_in_run_set(self, db, seen_items_path):
        """reset_in_run should clear the in-run set."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.is_duplicate("arxiv:p1", "t")
        assert dedup.in_run_count == 1

        dedup.reset_in_run()
        assert dedup.in_run_count == 0

    def test_does_not_affect_cross_run(self, db, seen_items_path):
        """reset_in_run should not affect cross-run seen data."""
        today = date.today().isoformat()
        _write_seen_items(seen_items_path, [
            {"paper_key": "arxiv:p1", "topic_slug": "t", "date": today},
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)
        dedup.reset_in_run()

        # Cross-run entry should still cause duplicate
        assert dedup.is_duplicate("arxiv:p1", "t") is True


# ---------------------------------------------------------------------------
# 9. Multi-topic allowance
# ---------------------------------------------------------------------------


class TestMultiTopicAllowance:
    """Same paper evaluated in different topics should both proceed."""

    def test_paper_in_topic_a_then_topic_b(self, db, seen_items_path):
        """Paper evaluated in topic A, then topic B: both proceed."""
        dedup = DedupManager(db, seen_items_path=seen_items_path)

        # First check in topic A
        assert dedup.is_duplicate("arxiv:p1", "topic-a") is False
        dedup.mark_seen("arxiv:p1", "topic-a")

        # Reset in-run for new topic
        dedup.reset_in_run()

        # Second check in topic B (same paper, different topic)
        assert dedup.is_duplicate("arxiv:p1", "topic-b") is False

    def test_cross_run_multi_topic(self, db, seen_items_path):
        """Cross-run: same paper in seen_items for topic A, new in topic B."""
        today = date.today().isoformat()
        _write_seen_items(seen_items_path, [
            {
                "paper_key": "arxiv:p1",
                "topic_slug": "topic-a",
                "date": today,
            },
        ])

        dedup = DedupManager(db, seen_items_path=seen_items_path)

        # Duplicate in topic A
        assert dedup.is_duplicate("arxiv:p1", "topic-a") is True
        # Not duplicate in topic B
        assert dedup.is_duplicate("arxiv:p1", "topic-b") is False
