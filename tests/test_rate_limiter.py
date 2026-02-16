"""Tests for core.llm.rate_limiter module.

Covers dynamic delay calculation, daily usage persistence, restart recovery,
RPM sliding window enforcement, and topic tracking per DevSpec Section 17-1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.llm.rate_limiter import RateLimiter, _DEFAULT_RPM, _DELAY_BUFFER


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
def limiter(usage_dir: Path) -> RateLimiter:
    """Return a default RateLimiter using tmp_path."""
    return RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))


def _today_key() -> str:
    """Consistent helper matching the module's internal function."""
    from datetime import date

    return date.today().strftime("%Y%m%d")


def _today_iso() -> str:
    from datetime import date

    return date.today().isoformat()


def _write_usage(usage_dir: Path, data: dict[str, Any]) -> None:
    """Write a usage JSON file for today."""
    path = usage_dir / f"{_today_key()}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_usage(usage_dir: Path) -> dict[str, Any]:
    """Read today's usage JSON file."""
    path = usage_dir / f"{_today_key()}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Dynamic delay calculation
# ---------------------------------------------------------------------------


class TestDynamicDelay:
    """Test delay property for various RPM values."""

    def test_delay_rpm_10(self) -> None:
        """Default RPM=10 yields 6.5s delay."""
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(6.5)

    def test_delay_rpm_20(self) -> None:
        rl = RateLimiter(detected_rpm=20, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(3.5)

    def test_delay_rpm_60(self) -> None:
        rl = RateLimiter(detected_rpm=60, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(1.5)

    def test_delay_rpm_1(self) -> None:
        rl = RateLimiter(detected_rpm=1, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(60.5)

    def test_delay_rpm_30(self) -> None:
        rl = RateLimiter(detected_rpm=30, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(2.5)

    def test_delay_formula_is_60_div_rpm_plus_buffer(self) -> None:
        """Verify formula: delay = 60 / detected_rpm + 0.5."""
        for rpm in [5, 10, 15, 20, 50, 100]:
            rl = RateLimiter(detected_rpm=rpm, daily_limit=200, usage_dir="/tmp/unused")
            expected = 60.0 / rpm + _DELAY_BUFFER
            assert rl.delay == pytest.approx(expected), f"Failed for RPM={rpm}"

    def test_delay_fallback_on_zero_rpm(self) -> None:
        """RPM=0 falls back to default RPM=10."""
        rl = RateLimiter(detected_rpm=0, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(6.5)

    def test_delay_fallback_on_negative_rpm(self) -> None:
        """Negative RPM falls back to default RPM=10."""
        rl = RateLimiter(detected_rpm=-5, daily_limit=200, usage_dir="/tmp/unused")
        assert rl.delay == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# 2. Daily usage persistence - save and load
# ---------------------------------------------------------------------------


class TestUsagePersistence:
    """Test saving and loading daily usage files."""

    def test_save_creates_file(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()

        file_path = usage_dir / f"{_today_key()}.json"
        assert file_path.exists()

    def test_save_file_format(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()

        data = _read_usage(usage_dir)
        assert "date" in data
        assert "api_calls" in data
        assert "topics_completed" in data
        assert "topics_skipped" in data

    def test_load_existing_usage(self, usage_dir: Path) -> None:
        """Load today's file on process start."""
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 50,
            "topics_completed": ["topic-a"],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.daily_calls_remaining == 150
        assert rl.should_skip_topic("topic-a") is True

    def test_load_accumulates_on_multiple_runs(self, usage_dir: Path) -> None:
        """Multiple manual runs on same day accumulate usage."""
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 50,
            "topics_completed": ["topic-a"],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()
        rl.record_call()
        rl.record_call()

        data = _read_usage(usage_dir)
        assert data["api_calls"] == 53


# ---------------------------------------------------------------------------
# 3. Cumulative record_call tracking
# ---------------------------------------------------------------------------


class TestRecordCall:
    """Test that record_call increments count."""

    def test_single_call(self, limiter: RateLimiter) -> None:
        limiter.record_call()
        assert limiter.daily_calls_remaining == 199

    def test_multiple_calls(self, limiter: RateLimiter) -> None:
        for _ in range(5):
            limiter.record_call()
        assert limiter.daily_calls_remaining == 195

    def test_call_count_persisted(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        for _ in range(10):
            rl.record_call()

        data = _read_usage(usage_dir)
        assert data["api_calls"] == 10


# ---------------------------------------------------------------------------
# 4. Daily limit enforcement
# ---------------------------------------------------------------------------


class TestDailyLimit:
    """Test is_daily_limit_reached and daily_calls_remaining."""

    def test_not_reached_initially(self, limiter: RateLimiter) -> None:
        assert limiter.is_daily_limit_reached is False

    def test_reached_at_limit(self, usage_dir: Path) -> None:
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 200,
            "topics_completed": [],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.is_daily_limit_reached is True
        assert rl.daily_calls_remaining == 0

    def test_reached_above_limit(self, usage_dir: Path) -> None:
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 250,
            "topics_completed": [],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.is_daily_limit_reached is True
        assert rl.daily_calls_remaining == 0

    def test_daily_calls_remaining_accurate(self, usage_dir: Path) -> None:
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 123,
            "topics_completed": [],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.daily_calls_remaining == 77

    def test_daily_limit_fallback_on_negative(self) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=-1, usage_dir="/tmp/unused")
        assert rl.daily_calls_remaining == 200


# ---------------------------------------------------------------------------
# 5. Topic completed / skipped tracking
# ---------------------------------------------------------------------------


class TestTopicTracking:
    """Test topic completed and skipped recording."""

    def test_record_topic_completed(self, limiter: RateLimiter, usage_dir: Path) -> None:
        limiter.record_topic_completed("ai-sports-device")

        data = _read_usage(usage_dir)
        assert "ai-sports-device" in data["topics_completed"]

    def test_record_topic_skipped(self, limiter: RateLimiter, usage_dir: Path) -> None:
        limiter.record_topic_skipped("slow-topic")

        data = _read_usage(usage_dir)
        assert "slow-topic" in data["topics_skipped"]

    def test_no_duplicate_completed(self, limiter: RateLimiter) -> None:
        limiter.record_topic_completed("topic-x")
        limiter.record_topic_completed("topic-x")

        data = _read_usage(limiter._usage_dir)
        assert data["topics_completed"].count("topic-x") == 1

    def test_no_duplicate_skipped(self, limiter: RateLimiter) -> None:
        limiter.record_topic_skipped("topic-y")
        limiter.record_topic_skipped("topic-y")

        data = _read_usage(limiter._usage_dir)
        assert data["topics_skipped"].count("topic-y") == 1


# ---------------------------------------------------------------------------
# 6. should_skip_topic
# ---------------------------------------------------------------------------


class TestShouldSkipTopic:
    """Test should_skip_topic logic."""

    def test_returns_false_for_new_topic(self, limiter: RateLimiter) -> None:
        assert limiter.should_skip_topic("new-topic") is False

    def test_returns_true_when_limit_reached(self, usage_dir: Path) -> None:
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 200,
            "topics_completed": [],
            "topics_skipped": [],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.should_skip_topic("any-topic") is True

    def test_returns_true_for_completed_topic(self, limiter: RateLimiter) -> None:
        limiter.record_topic_completed("done-topic")
        assert limiter.should_skip_topic("done-topic") is True

    def test_returns_false_for_skipped_topic(self, limiter: RateLimiter) -> None:
        """Skipped topics are NOT blocked -- only completed ones are."""
        limiter.record_topic_skipped("skipped-topic")
        assert limiter.should_skip_topic("skipped-topic") is False


# ---------------------------------------------------------------------------
# 7. Missing / corrupted usage file
# ---------------------------------------------------------------------------


class TestUsageFileEdgeCases:
    """Test graceful handling of missing or corrupted files."""

    def test_missing_file_starts_from_zero(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_usage"
        empty_dir.mkdir()

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(empty_dir))
        assert rl.daily_calls_remaining == 200
        assert rl.is_daily_limit_reached is False

    def test_corrupted_json_starts_from_zero(self, usage_dir: Path) -> None:
        file_path = usage_dir / f"{_today_key()}.json"
        file_path.write_text("{invalid json content!!!", encoding="utf-8")

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.daily_calls_remaining == 200

    def test_non_dict_json_starts_from_zero(self, usage_dir: Path) -> None:
        file_path = usage_dir / f"{_today_key()}.json"
        file_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.daily_calls_remaining == 200

    def test_missing_api_calls_key_starts_from_zero(self, usage_dir: Path) -> None:
        file_path = usage_dir / f"{_today_key()}.json"
        file_path.write_text(json.dumps({"date": "2026-02-16"}), encoding="utf-8")

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl.daily_calls_remaining == 200


# ---------------------------------------------------------------------------
# 8. Usage directory creation
# ---------------------------------------------------------------------------


class TestUsageDirectoryCreation:
    """Test that usage directory is created if it does not exist."""

    def test_creates_nested_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "usage"
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(nested))
        rl.record_call()

        assert nested.exists()
        assert (nested / f"{_today_key()}.json").exists()


# ---------------------------------------------------------------------------
# 9. RPM sliding window enforcement
# ---------------------------------------------------------------------------


class TestRpmSlidingWindow:
    """Test RPM sliding window in wait()."""

    @patch("core.llm.rate_limiter.time.sleep")
    @patch("core.llm.rate_limiter.time.time")
    def test_wait_calls_sleep_with_delay(
        self, mock_time: Any, mock_sleep: Any, usage_dir: Path
    ) -> None:
        """wait() should call time.sleep with the computed delay."""
        mock_time.return_value = 1000.0

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.wait()

        # Should sleep for the dynamic delay (6.5s for RPM=10)
        mock_sleep.assert_called_with(pytest.approx(6.5))

    @patch("core.llm.rate_limiter.time.sleep")
    @patch("core.llm.rate_limiter.time.time")
    def test_wait_enforces_rpm_window(
        self, mock_time: Any, mock_sleep: Any, usage_dir: Path
    ) -> None:
        """When RPM calls fill the window, wait should add extra sleep."""
        rl = RateLimiter(detected_rpm=3, daily_limit=200, usage_dir=str(usage_dir))

        # Simulate 3 recent calls within the last 60 seconds
        base_time = 1000.0
        rl._call_timestamps.append(base_time - 10.0)  # 10s ago
        rl._call_timestamps.append(base_time - 5.0)  # 5s ago
        rl._call_timestamps.append(base_time - 2.0)  # 2s ago

        # time.time() returns base_time initially, then updated after extra sleep
        mock_time.side_effect = [base_time, base_time + 50.0]
        rl.wait()

        # First call should be the extra wait: 60 - (1000 - 990) = 50s
        # Second call should be the normal delay: 60/3 + 0.5 = 20.5s
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == pytest.approx(50.0)  # extra wait
        assert calls[1][0][0] == pytest.approx(20.5)  # normal delay

    @patch("core.llm.rate_limiter.time.sleep")
    @patch("core.llm.rate_limiter.time.time")
    def test_wait_no_extra_sleep_under_rpm(
        self, mock_time: Any, mock_sleep: Any, usage_dir: Path
    ) -> None:
        """When under RPM limit, only the dynamic delay is applied."""
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))

        # Only 2 calls in the window (under RPM=10)
        base_time = 1000.0
        rl._call_timestamps.append(base_time - 30.0)
        rl._call_timestamps.append(base_time - 15.0)

        mock_time.return_value = base_time
        rl.wait()

        # Only one sleep call for the dynamic delay
        mock_sleep.assert_called_once_with(pytest.approx(6.5))

    @patch("core.llm.rate_limiter.time.sleep")
    @patch("core.llm.rate_limiter.time.time")
    def test_old_timestamps_purged(
        self, mock_time: Any, mock_sleep: Any, usage_dir: Path
    ) -> None:
        """Timestamps older than 60 seconds are removed."""
        rl = RateLimiter(detected_rpm=5, daily_limit=200, usage_dir=str(usage_dir))

        base_time = 1000.0
        # Add old timestamps (> 60s ago)
        rl._call_timestamps.append(base_time - 120.0)
        rl._call_timestamps.append(base_time - 90.0)
        # Add recent timestamps
        rl._call_timestamps.append(base_time - 10.0)

        mock_time.return_value = base_time
        rl.wait()

        # Old timestamps should be purged; 1 recent remains (< RPM=5)
        assert len(rl._call_timestamps) == 1


# ---------------------------------------------------------------------------
# 10. wait() sleep values for various RPMs
# ---------------------------------------------------------------------------


class TestWaitSleepValues:
    """Test that wait() calls time.sleep with correct delay values."""

    @pytest.mark.parametrize(
        "rpm,expected_delay",
        [
            (10, 6.5),
            (20, 3.5),
            (60, 1.5),
            (1, 60.5),
            (30, 2.5),
        ],
        ids=["rpm-10", "rpm-20", "rpm-60", "rpm-1", "rpm-30"],
    )
    @patch("core.llm.rate_limiter.time.sleep")
    @patch("core.llm.rate_limiter.time.time")
    def test_wait_delay_matches_rpm(
        self,
        mock_time: Any,
        mock_sleep: Any,
        rpm: int,
        expected_delay: float,
        usage_dir: Path,
    ) -> None:
        mock_time.return_value = 1000.0
        rl = RateLimiter(detected_rpm=rpm, daily_limit=200, usage_dir=str(usage_dir))
        rl.wait()

        mock_sleep.assert_called_with(pytest.approx(expected_delay))


# ---------------------------------------------------------------------------
# 11. Multiple runs same day accumulate
# ---------------------------------------------------------------------------


class TestMultipleRunsSameDay:
    """Test that multiple process runs on the same day accumulate usage."""

    def test_second_run_accumulates(self, usage_dir: Path) -> None:
        # First run
        rl1 = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        for _ in range(5):
            rl1.record_call()
        rl1.record_topic_completed("topic-a")

        # Second run (new instance, same day)
        rl2 = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        for _ in range(3):
            rl2.record_call()
        rl2.record_topic_completed("topic-b")

        assert rl2.daily_calls_remaining == 192
        data = _read_usage(usage_dir)
        assert data["api_calls"] == 8
        assert "topic-a" in data["topics_completed"]
        assert "topic-b" in data["topics_completed"]

    def test_third_run_continues(self, usage_dir: Path) -> None:
        # Pre-existing usage
        _write_usage(usage_dir, {
            "date": _today_iso(),
            "api_calls": 100,
            "topics_completed": ["t1", "t2"],
            "topics_skipped": ["t3"],
        })

        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()
        rl.record_topic_completed("t4")

        data = _read_usage(usage_dir)
        assert data["api_calls"] == 101
        assert set(data["topics_completed"]) == {"t1", "t2", "t4"}
        assert data["topics_skipped"] == ["t3"]


# ---------------------------------------------------------------------------
# 12. record_call appends to sliding window
# ---------------------------------------------------------------------------


class TestRecordCallTimestamps:
    """Test that record_call adds timestamps to the sliding window."""

    def test_timestamps_appended(self, limiter: RateLimiter) -> None:
        limiter.record_call()
        limiter.record_call()
        limiter.record_call()

        assert len(limiter._call_timestamps) == 3

    def test_timestamps_are_monotonic(self, limiter: RateLimiter) -> None:
        limiter.record_call()
        limiter.record_call()

        stamps = list(limiter._call_timestamps)
        assert stamps[0] <= stamps[1]


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Various edge case tests."""

    def test_initial_state_zero_calls(self, limiter: RateLimiter) -> None:
        assert limiter.daily_calls_remaining == 200
        assert limiter.is_daily_limit_reached is False

    def test_save_and_reload_roundtrip(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()
        rl.record_call()
        rl.record_topic_completed("topic-1")
        rl.record_topic_skipped("topic-2")

        # Reload from file
        rl2 = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert rl2.daily_calls_remaining == 198
        assert rl2.should_skip_topic("topic-1") is True
        assert rl2.should_skip_topic("topic-2") is False
        assert rl2.should_skip_topic("topic-3") is False

    def test_usage_file_date_field(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        rl.record_call()

        data = _read_usage(usage_dir)
        assert data["date"] == _today_iso()

    def test_empty_deque_on_init(self, usage_dir: Path) -> None:
        rl = RateLimiter(detected_rpm=10, daily_limit=200, usage_dir=str(usage_dir))
        assert len(rl._call_timestamps) == 0
