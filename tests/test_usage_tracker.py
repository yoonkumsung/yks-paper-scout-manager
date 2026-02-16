"""Tests for core.storage.usage_tracker module.

Covers daily usage loading/saving, topic tracking with rich metadata,
rolling cleanup, corrupted file handling, and summary API.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

from core.storage.usage_tracker import UsageTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def usage_dir(tmp_path: Path) -> Path:
    """Return a temporary usage directory."""
    d = tmp_path / "usage"
    d.mkdir()
    return d


@pytest.fixture()
def tracker(usage_dir: Path) -> UsageTracker:
    """Return a UsageTracker pointing at a temporary directory."""
    return UsageTracker(usage_dir=str(usage_dir))


def _today_key() -> str:
    return date.today().strftime("%Y%m%d")


def _today_iso() -> str:
    return date.today().isoformat()


def _write_usage(usage_dir: Path, data: Dict[str, Any]) -> None:
    path = usage_dir / f"{_today_key()}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_usage(usage_dir: Path) -> Dict[str, Any]:
    path = usage_dir / f"{_today_key()}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Load today's usage -- existing file
# ---------------------------------------------------------------------------


class TestLoadExistingUsage:
    """Existing usage file is loaded correctly."""

    def test_loads_existing_file(self, usage_dir: Path) -> None:
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 42,
            "topics_completed": [
                {"slug": "ai-sports", "total_output": 10},
            ],
            "topics_skipped": [
                {"slug": "video-analytics", "reason": "daily_limit_reached"},
            ],
        })

        tracker = UsageTracker(usage_dir=str(usage_dir))
        usage = tracker.get_today_usage()

        assert usage["api_calls"] == 42
        assert len(usage["topics_completed"]) == 1
        assert usage["topics_completed"][0]["slug"] == "ai-sports"
        assert len(usage["topics_skipped"]) == 1


# ---------------------------------------------------------------------------
# 2. Load missing file -- returns default
# ---------------------------------------------------------------------------


class TestLoadMissingFile:
    """Missing file returns fresh default structure."""

    def test_returns_default_when_no_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_usage"
        empty.mkdir()

        tracker = UsageTracker(usage_dir=str(empty))
        usage = tracker.get_today_usage()

        assert usage["date"] == _today_iso()
        assert usage["api_calls"] == 0
        assert usage["topics_completed"] == []
        assert usage["topics_skipped"] == []


# ---------------------------------------------------------------------------
# 3. Save and reload -- roundtrip persistence
# ---------------------------------------------------------------------------


class TestSaveAndReload:
    """Roundtrip persistence works correctly."""

    def test_roundtrip(self, usage_dir: Path) -> None:
        tracker = UsageTracker(usage_dir=str(usage_dir))
        tracker.record_topic_completed("topic-a", total_output=25)
        tracker.record_topic_skipped("topic-b", reason="rate_limited")
        tracker.increment_api_calls(5)

        # Reload via new instance
        tracker2 = UsageTracker(usage_dir=str(usage_dir))
        usage = tracker2.get_today_usage()

        assert usage["api_calls"] == 5
        assert len(usage["topics_completed"]) == 1
        assert usage["topics_completed"][0]["slug"] == "topic-a"
        assert usage["topics_completed"][0]["total_output"] == 25
        assert len(usage["topics_skipped"]) == 1
        assert usage["topics_skipped"][0]["slug"] == "topic-b"
        assert usage["topics_skipped"][0]["reason"] == "rate_limited"


# ---------------------------------------------------------------------------
# 4. Record topic completed -- added with count
# ---------------------------------------------------------------------------


class TestRecordTopicCompleted:
    """Topic completion is recorded with output count."""

    def test_adds_completed_entry(
        self, tracker: UsageTracker, usage_dir: Path
    ) -> None:
        tracker.record_topic_completed("ai-sports-device", total_output=42)

        data = _read_usage(usage_dir)
        assert len(data["topics_completed"]) == 1
        entry = data["topics_completed"][0]
        assert entry["slug"] == "ai-sports-device"
        assert entry["total_output"] == 42

    def test_no_duplicate_completed(self, tracker: UsageTracker) -> None:
        tracker.record_topic_completed("topic-x", total_output=10)
        tracker.record_topic_completed("topic-x", total_output=15)

        usage = tracker.get_today_usage()
        assert len(usage["topics_completed"]) == 1


# ---------------------------------------------------------------------------
# 5. Record topic skipped -- added with reason
# ---------------------------------------------------------------------------


class TestRecordTopicSkipped:
    """Topic skip is recorded with reason."""

    def test_adds_skipped_entry(
        self, tracker: UsageTracker, usage_dir: Path
    ) -> None:
        tracker.record_topic_skipped("video-analytics", reason="daily_limit_reached")

        data = _read_usage(usage_dir)
        assert len(data["topics_skipped"]) == 1
        entry = data["topics_skipped"][0]
        assert entry["slug"] == "video-analytics"
        assert entry["reason"] == "daily_limit_reached"

    def test_no_duplicate_skipped(self, tracker: UsageTracker) -> None:
        tracker.record_topic_skipped("topic-y", reason="limit")
        tracker.record_topic_skipped("topic-y", reason="error")

        usage = tracker.get_today_usage()
        assert len(usage["topics_skipped"]) == 1


# ---------------------------------------------------------------------------
# 6. Increment API calls
# ---------------------------------------------------------------------------


class TestIncrementApiCalls:
    """API call counter increments correctly."""

    def test_increments_by_one(self, tracker: UsageTracker) -> None:
        tracker.increment_api_calls()
        assert tracker.get_today_usage()["api_calls"] == 1

    def test_increments_by_count(self, tracker: UsageTracker) -> None:
        tracker.increment_api_calls(5)
        tracker.increment_api_calls(3)
        assert tracker.get_today_usage()["api_calls"] == 8


# ---------------------------------------------------------------------------
# 7. Daily summary
# ---------------------------------------------------------------------------


class TestDailySummary:
    """Summary returns correct aggregate totals."""

    def test_summary_with_data(self, tracker: UsageTracker) -> None:
        tracker.increment_api_calls(10)
        tracker.record_topic_completed("t1", total_output=20)
        tracker.record_topic_completed("t2", total_output=15)
        tracker.record_topic_skipped("t3", reason="limit")

        summary = tracker.get_daily_summary()

        assert summary["date"] == _today_iso()
        assert summary["api_calls"] == 10
        assert summary["topics_completed_count"] == 2
        assert summary["topics_skipped_count"] == 1
        assert summary["total_papers_output"] == 35

    def test_summary_empty(self, tracker: UsageTracker) -> None:
        summary = tracker.get_daily_summary()

        assert summary["api_calls"] == 0
        assert summary["topics_completed_count"] == 0
        assert summary["topics_skipped_count"] == 0
        assert summary["total_papers_output"] == 0


# ---------------------------------------------------------------------------
# 8. Rolling cleanup -- old files deleted
# ---------------------------------------------------------------------------


class TestRollingCleanup:
    """Files older than 30 days are deleted."""

    def test_deletes_old_files(self, usage_dir: Path) -> None:
        # Create a file 31 days ago
        old_date = date.today() - timedelta(days=31)
        old_name = old_date.strftime("%Y%m%d") + ".json"
        (usage_dir / old_name).write_text(
            json.dumps({"date": old_date.isoformat(), "api_calls": 5}),
            encoding="utf-8",
        )

        # Create a file 40 days ago
        older_date = date.today() - timedelta(days=40)
        older_name = older_date.strftime("%Y%m%d") + ".json"
        (usage_dir / older_name).write_text(
            json.dumps({"date": older_date.isoformat(), "api_calls": 3}),
            encoding="utf-8",
        )

        tracker = UsageTracker(usage_dir=str(usage_dir))
        deleted = tracker.cleanup_old_files(max_age_days=30)

        assert deleted == 2
        assert not (usage_dir / old_name).exists()
        assert not (usage_dir / older_name).exists()


# ---------------------------------------------------------------------------
# 9. Rolling cleanup -- recent files preserved
# ---------------------------------------------------------------------------


class TestCleanupPreservesRecent:
    """Files within 30 days are preserved."""

    def test_preserves_recent_files(self, usage_dir: Path) -> None:
        # Create a file 10 days ago
        recent_date = date.today() - timedelta(days=10)
        recent_name = recent_date.strftime("%Y%m%d") + ".json"
        (usage_dir / recent_name).write_text(
            json.dumps({"date": recent_date.isoformat(), "api_calls": 7}),
            encoding="utf-8",
        )

        # Create today's file
        today_name = _today_key() + ".json"
        (usage_dir / today_name).write_text(
            json.dumps({"date": _today_iso(), "api_calls": 2}),
            encoding="utf-8",
        )

        tracker = UsageTracker(usage_dir=str(usage_dir))
        deleted = tracker.cleanup_old_files(max_age_days=30)

        assert deleted == 0
        assert (usage_dir / recent_name).exists()
        assert (usage_dir / today_name).exists()


# ---------------------------------------------------------------------------
# 10. Corrupted file handling -- graceful degradation
# ---------------------------------------------------------------------------


class TestCorruptedFileHandling:
    """Corrupted or invalid files are handled gracefully."""

    def test_corrupted_json(self, usage_dir: Path) -> None:
        path = usage_dir / f"{_today_key()}.json"
        path.write_text("{not valid json!!!", encoding="utf-8")

        tracker = UsageTracker(usage_dir=str(usage_dir))
        usage = tracker.get_today_usage()

        assert usage["api_calls"] == 0
        assert usage["topics_completed"] == []

    def test_non_dict_json(self, usage_dir: Path) -> None:
        path = usage_dir / f"{_today_key()}.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        tracker = UsageTracker(usage_dir=str(usage_dir))
        usage = tracker.get_today_usage()

        assert usage["api_calls"] == 0


# ---------------------------------------------------------------------------
# 11. Empty directory -- no errors
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    """Operations on an empty or non-existent directory work safely."""

    def test_cleanup_empty_dir(self, usage_dir: Path) -> None:
        tracker = UsageTracker(usage_dir=str(usage_dir))
        deleted = tracker.cleanup_old_files()
        assert deleted == 0

    def test_cleanup_nonexistent_dir(self, tmp_path: Path) -> None:
        tracker = UsageTracker(usage_dir=str(tmp_path / "does_not_exist"))
        deleted = tracker.cleanup_old_files()
        assert deleted == 0


# ---------------------------------------------------------------------------
# 12. Multiple topics -- tracked independently
# ---------------------------------------------------------------------------


class TestMultipleTopics:
    """Multiple completions and skips are tracked independently."""

    def test_multiple_completions(self, tracker: UsageTracker) -> None:
        tracker.record_topic_completed("topic-a", total_output=10)
        tracker.record_topic_completed("topic-b", total_output=20)
        tracker.record_topic_completed("topic-c", total_output=30)

        usage = tracker.get_today_usage()
        assert len(usage["topics_completed"]) == 3

        slugs = {e["slug"] for e in usage["topics_completed"]}
        assert slugs == {"topic-a", "topic-b", "topic-c"}

        outputs = {e["slug"]: e["total_output"] for e in usage["topics_completed"]}
        assert outputs["topic-a"] == 10
        assert outputs["topic-b"] == 20
        assert outputs["topic-c"] == 30

    def test_multiple_skips(self, tracker: UsageTracker) -> None:
        tracker.record_topic_skipped("s1", reason="limit_reached")
        tracker.record_topic_skipped("s2", reason="error")

        usage = tracker.get_today_usage()
        assert len(usage["topics_skipped"]) == 2

        reasons = {e["slug"]: e["reason"] for e in usage["topics_skipped"]}
        assert reasons["s1"] == "limit_reached"
        assert reasons["s2"] == "error"


# ---------------------------------------------------------------------------
# 13. Compatibility with RateLimiter simple format
# ---------------------------------------------------------------------------


class TestRateLimiterCompatibility:
    """UsageTracker handles the simple RateLimiter format gracefully."""

    def test_loads_simple_format(self, usage_dir: Path) -> None:
        """RateLimiter stores topics_completed as plain strings."""
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 10,
            "topics_completed": ["topic-a", "topic-b"],
            "topics_skipped": ["topic-c"],
        })

        tracker = UsageTracker(usage_dir=str(usage_dir))
        usage = tracker.get_today_usage()

        assert usage["api_calls"] == 10
        # Simple strings are loaded as-is
        assert "topic-a" in usage["topics_completed"]
        assert "topic-b" in usage["topics_completed"]

    def test_no_duplicate_with_simple_format(self, usage_dir: Path) -> None:
        """Recording a topic already present as string does not duplicate."""
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 5,
            "topics_completed": ["existing-topic"],
            "topics_skipped": ["skipped-one"],
        })

        tracker = UsageTracker(usage_dir=str(usage_dir))
        tracker.record_topic_completed("existing-topic", total_output=10)

        usage = tracker.get_today_usage()
        # Should NOT add a duplicate entry
        assert len(usage["topics_completed"]) == 1


# ---------------------------------------------------------------------------
# 14. Cleanup with non-JSON files in directory
# ---------------------------------------------------------------------------


class TestCleanupNonJsonFiles:
    """Cleanup ignores non-JSON and oddly named files."""

    def test_ignores_non_date_filenames(self, usage_dir: Path) -> None:
        (usage_dir / "notes.json").write_text("{}", encoding="utf-8")
        (usage_dir / "readme.txt").write_text("hello", encoding="utf-8")

        tracker = UsageTracker(usage_dir=str(usage_dir))
        deleted = tracker.cleanup_old_files(max_age_days=30)

        assert deleted == 0
        assert (usage_dir / "notes.json").exists()
        assert (usage_dir / "readme.txt").exists()
