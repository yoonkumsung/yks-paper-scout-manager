"""Tests for SearchWindowComputer (TASK-035).

Covers:
  - Auto mode with DB hit, last_success.json hit, 72h fallback
  - Manual mode with date_from/date_to
  - Buffer application (+-30min)
  - Window end is KST 11:00 -> UTC conversion
  - last_success update / max protection / missing file / corrupted JSON
  - DB manager None / topic isolation / timezone handling
  - Fallback priority verification
  - KST date boundary around midnight
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from core.pipeline.search_window import (
    BUFFER_MINUTES,
    FALLBACK_HOURS,
    KST_OFFSET,
    SearchWindowComputer,
)

# ── helpers ──────────────────────────────────────────────────────────────


@dataclass
class _FakeRunMeta:
    """Minimal RunMeta stub with window_end_utc."""

    window_end_utc: datetime


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Shorthand UTC datetime constructor."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


BUFFER = timedelta(minutes=BUFFER_MINUTES)


# ── 1. Auto mode: DB hit ────────────────────────────────────────────────


class TestAutoModeDBHit:
    """When DB has a completed run for the topic, use its window_end_utc."""

    def test_uses_db_window_end_as_start(self, tmp_path):
        db_end = _utc(2026, 2, 8, 2, 0)
        db = MagicMock()
        db.get_latest_completed_run.return_value = _FakeRunMeta(
            window_end_utc=db_end
        )

        swc = SearchWindowComputer(
            db_manager=db,
            last_success_path=str(tmp_path / "ls.json"),
        )
        now = _utc(2026, 2, 10, 3, 0)  # KST 12:00
        start, end = swc.compute("ai-sports-device", now=now)

        expected_start = db_end - BUFFER
        assert start == expected_start
        db.get_latest_completed_run.assert_called_once_with("ai-sports-device")


# ── 2. Auto mode: last_success.json hit ─────────────────────────────────


class TestAutoModeLastSuccessHit:
    """DB miss falls through to last_success.json."""

    def test_uses_json_when_db_misses(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        ls_data = {
            "ai-sports-device": {
                "last_success_window_end_utc": "2026-02-07T02:00:00+00:00",
            }
        }
        ls_path.write_text(json.dumps(ls_data), encoding="utf-8")

        db = MagicMock()
        db.get_latest_completed_run.return_value = None

        swc = SearchWindowComputer(
            db_manager=db, last_success_path=str(ls_path)
        )
        now = _utc(2026, 2, 10, 3, 0)
        start, _end = swc.compute("ai-sports-device", now=now)

        expected_start = _utc(2026, 2, 7, 2, 0) - BUFFER
        assert start == expected_start


# ── 3. Auto mode: 72h fallback ──────────────────────────────────────────


class TestAutoMode72hFallback:
    """Both DB and JSON miss -> 72h fallback."""

    def test_falls_back_to_72h(self, tmp_path):
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "nonexistent.json"),
        )
        now = _utc(2026, 2, 10, 3, 0)  # KST 12:00 Feb 10
        start, end = swc.compute("new-topic", now=now)

        # window_end = KST 11:00 Feb 10 = UTC 02:00 Feb 10
        window_end = _utc(2026, 2, 10, 2, 0)
        expected_start = (window_end - timedelta(hours=FALLBACK_HOURS)) - BUFFER
        assert start == expected_start


# ── 4. Manual mode ──────────────────────────────────────────────────────


class TestManualMode:
    """User provides both date_from and date_to."""

    def test_uses_provided_dates_with_buffer(self, tmp_path):
        date_from = _utc(2026, 1, 1, 0, 0)
        date_to = _utc(2026, 1, 5, 0, 0)

        swc = SearchWindowComputer(
            last_success_path=str(tmp_path / "ls.json")
        )
        start, end = swc.compute(
            "any-topic", date_from=date_from, date_to=date_to
        )

        assert start == date_from - BUFFER
        assert end == date_to + BUFFER

    def test_manual_mode_ignores_db_and_json(self, tmp_path):
        """DB and JSON are not consulted in manual mode."""
        db = MagicMock()
        ls_path = tmp_path / "ls.json"
        ls_path.write_text(
            json.dumps({"t": {"last_success_window_end_utc": "2026-01-01T00:00:00+00:00"}}),
            encoding="utf-8",
        )

        swc = SearchWindowComputer(db_manager=db, last_success_path=str(ls_path))
        date_from = _utc(2026, 2, 1, 0, 0)
        date_to = _utc(2026, 2, 2, 0, 0)
        swc.compute("t", date_from=date_from, date_to=date_to)

        db.get_latest_completed_run.assert_not_called()


# ── 5. Buffer application ───────────────────────────────────────────────


class TestBufferApplication:
    """+-30min buffer on both start and end."""

    def test_buffer_is_30min(self, tmp_path):
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        now = _utc(2026, 2, 10, 3, 0)
        start, end = swc.compute("t", now=now)

        # window_end = UTC 02:00, buffered end = UTC 02:30
        assert end == _utc(2026, 2, 10, 2, 30)

        # window_start = window_end - 72h = Feb 7 02:00, buffered = Feb 7 01:30
        assert start == _utc(2026, 2, 7, 1, 30)


# ── 6. Window end is KST 11:00 ──────────────────────────────────────────


class TestWindowEndKST1100:
    """window_end = today KST 11:00 = UTC 02:00."""

    def test_utc_conversion(self, tmp_path):
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        # now is UTC 05:00 Feb 10 = KST 14:00 Feb 10
        now = _utc(2026, 2, 10, 5, 0)
        _start, end = swc.compute("t", now=now)

        # KST 11:00 Feb 10 = UTC 02:00 Feb 10, plus 30min buffer
        assert end == _utc(2026, 2, 10, 2, 30)


# ── 7. Last success update ──────────────────────────────────────────────


class TestLastSuccessUpdate:
    """update_last_success writes new value."""

    def test_writes_new_value(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        swc = SearchWindowComputer(last_success_path=str(ls_path))

        ts = _utc(2026, 2, 10, 2, 0)
        swc.update_last_success("ai-sports-device", ts)

        data = json.loads(ls_path.read_text(encoding="utf-8"))
        assert "ai-sports-device" in data
        assert data["ai-sports-device"]["last_success_window_end_utc"] == ts.isoformat()

    def test_preserves_other_topics(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        existing = {
            "prompt-engineering": {
                "last_success_window_end_utc": "2026-02-09T02:00:00+00:00"
            }
        }
        ls_path.write_text(json.dumps(existing), encoding="utf-8")

        swc = SearchWindowComputer(last_success_path=str(ls_path))
        swc.update_last_success("ai-sports-device", _utc(2026, 2, 10, 2, 0))

        data = json.loads(ls_path.read_text(encoding="utf-8"))
        assert "prompt-engineering" in data
        assert "ai-sports-device" in data


# ── 8. Last success max protection ──────────────────────────────────────


class TestLastSuccessMaxProtection:
    """Older value does not overwrite newer."""

    def test_keeps_newer_existing_value(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        newer_ts = "2026-02-15T02:00:00+00:00"
        existing = {
            "ai-sports-device": {"last_success_window_end_utc": newer_ts}
        }
        ls_path.write_text(json.dumps(existing), encoding="utf-8")

        swc = SearchWindowComputer(last_success_path=str(ls_path))
        # Try to write an older timestamp
        older = _utc(2026, 2, 10, 2, 0)
        swc.update_last_success("ai-sports-device", older)

        data = json.loads(ls_path.read_text(encoding="utf-8"))
        # Should keep the newer value
        assert data["ai-sports-device"]["last_success_window_end_utc"] == newer_ts


# ── 9. Last success file missing ────────────────────────────────────────


class TestLastSuccessFileMissing:
    """Missing file returns empty dict, no error."""

    def test_missing_file_returns_empty(self, tmp_path):
        swc = SearchWindowComputer(
            last_success_path=str(tmp_path / "nonexistent.json")
        )
        # Should not raise -- falls through to 72h fallback
        start, end = swc.compute("t", now=_utc(2026, 2, 10, 3, 0))
        assert start is not None
        assert end is not None


# ── 10. Last success corrupted JSON ─────────────────────────────────────


class TestLastSuccessCorruptedJSON:
    """Corrupted JSON logged as warning, returns empty dict."""

    def test_corrupted_json_handled_gracefully(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        ls_path.write_text("{invalid json", encoding="utf-8")

        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(ls_path),
        )
        # Should not raise
        start, end = swc.compute("t", now=_utc(2026, 2, 10, 3, 0))
        assert start is not None


# ── 11. DB manager None ─────────────────────────────────────────────────


class TestDBManagerNone:
    """Skips DB lookup gracefully when db_manager is None."""

    def test_skips_db_when_none(self, tmp_path):
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        # Should not raise
        start, end = swc.compute("t", now=_utc(2026, 2, 10, 3, 0))
        assert start is not None


# ── 12. Topic isolation ─────────────────────────────────────────────────


class TestTopicIsolation:
    """Different topics get different windows."""

    def test_different_topics_independent(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        ls_data = {
            "topic-a": {
                "last_success_window_end_utc": "2026-02-08T02:00:00+00:00",
            },
            "topic-b": {
                "last_success_window_end_utc": "2026-02-05T02:00:00+00:00",
            },
        }
        ls_path.write_text(json.dumps(ls_data), encoding="utf-8")

        swc = SearchWindowComputer(
            db_manager=None, last_success_path=str(ls_path)
        )
        now = _utc(2026, 2, 10, 3, 0)

        start_a, _ = swc.compute("topic-a", now=now)
        start_b, _ = swc.compute("topic-b", now=now)

        # topic-a has a later last_success -> later start
        assert start_a > start_b


# ── 13. Timezone handling ────────────────────────────────────────────────


class TestTimezoneHandling:
    """All datetimes are UTC-aware."""

    def test_all_datetimes_utc_aware(self, tmp_path):
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        now = _utc(2026, 2, 10, 3, 0)
        start, end = swc.compute("t", now=now)

        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start.utcoffset() == timedelta(0)
        assert end.utcoffset() == timedelta(0)


# ── 14. Fallback priority: DB > JSON > 72h ──────────────────────────────


class TestFallbackPriority:
    """Verify DB > JSON > 72h ordering."""

    def test_db_takes_priority_over_json(self, tmp_path):
        db_end = _utc(2026, 2, 9, 2, 0)
        db = MagicMock()
        db.get_latest_completed_run.return_value = _FakeRunMeta(
            window_end_utc=db_end
        )

        ls_path = tmp_path / "ls.json"
        ls_data = {
            "t": {
                "last_success_window_end_utc": "2026-02-07T02:00:00+00:00",
            }
        }
        ls_path.write_text(json.dumps(ls_data), encoding="utf-8")

        swc = SearchWindowComputer(db_manager=db, last_success_path=str(ls_path))
        now = _utc(2026, 2, 10, 3, 0)
        start, _ = swc.compute("t", now=now)

        # DB value (Feb 9) should be used, not JSON (Feb 7) or 72h fallback
        assert start == db_end - BUFFER

    def test_json_takes_priority_over_72h(self, tmp_path):
        db = MagicMock()
        db.get_latest_completed_run.return_value = None  # DB miss

        json_end = _utc(2026, 2, 8, 2, 0)
        ls_path = tmp_path / "ls.json"
        ls_data = {
            "t": {
                "last_success_window_end_utc": json_end.isoformat(),
            }
        }
        ls_path.write_text(json.dumps(ls_data), encoding="utf-8")

        swc = SearchWindowComputer(db_manager=db, last_success_path=str(ls_path))
        now = _utc(2026, 2, 10, 3, 0)
        start, _ = swc.compute("t", now=now)

        # JSON value should be used, not 72h fallback
        expected_from_json = json_end - BUFFER
        expected_from_72h = (_utc(2026, 2, 10, 2, 0) - timedelta(hours=72)) - BUFFER
        assert start == expected_from_json
        assert start != expected_from_72h


# ── 15. KST date boundary ───────────────────────────────────────────────


class TestKSTDateBoundary:
    """Around midnight KST, window_end should use KST date correctly."""

    def test_just_before_midnight_kst(self, tmp_path):
        """UTC 14:59 = KST 23:59 -> KST date is same day."""
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        # UTC 14:59 Feb 9 = KST 23:59 Feb 9
        now = _utc(2026, 2, 9, 14, 59)
        _start, end = swc.compute("t", now=now)

        # KST 11:00 Feb 9 = UTC 02:00 Feb 9, plus buffer
        assert end == _utc(2026, 2, 9, 2, 30)

    def test_just_after_midnight_kst(self, tmp_path):
        """UTC 15:01 = KST 00:01 next day -> KST date is next day."""
        swc = SearchWindowComputer(
            db_manager=None,
            last_success_path=str(tmp_path / "ls.json"),
        )
        # UTC 15:01 Feb 9 = KST 00:01 Feb 10
        now = _utc(2026, 2, 9, 15, 1)
        _start, end = swc.compute("t", now=now)

        # KST 11:00 Feb 10 = UTC 02:00 Feb 10, plus buffer
        assert end == _utc(2026, 2, 10, 2, 30)


# ── 16. Update creates directory if missing ──────────────────────────────


class TestUpdateCreatesDirectory:
    """update_last_success creates parent directory if it does not exist."""

    def test_creates_parent_dir(self, tmp_path):
        ls_path = tmp_path / "nested" / "dir" / "ls.json"
        swc = SearchWindowComputer(last_success_path=str(ls_path))

        swc.update_last_success("t", _utc(2026, 2, 10, 2, 0))

        assert ls_path.exists()
        data = json.loads(ls_path.read_text(encoding="utf-8"))
        assert "t" in data


# ── 17. Update overwrites with newer value ───────────────────────────────


class TestUpdateOverwritesWithNewer:
    """When new value is newer than existing, it should overwrite."""

    def test_overwrites_older_with_newer(self, tmp_path):
        ls_path = tmp_path / "ls.json"
        older_ts = "2026-02-05T02:00:00+00:00"
        existing = {
            "t": {"last_success_window_end_utc": older_ts}
        }
        ls_path.write_text(json.dumps(existing), encoding="utf-8")

        swc = SearchWindowComputer(last_success_path=str(ls_path))
        newer = _utc(2026, 2, 10, 2, 0)
        swc.update_last_success("t", newer)

        data = json.loads(ls_path.read_text(encoding="utf-8"))
        assert data["t"]["last_success_window_end_utc"] == newer.isoformat()
