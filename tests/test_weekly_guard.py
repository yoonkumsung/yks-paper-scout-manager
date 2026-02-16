"""Tests for Weekly Task Guard."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from core.pipeline.weekly_guard import is_weekly_due, mark_weekly_done, read_weekly_flag


class TestWeeklyGuard:
    """Test suite for weekly task guard functionality."""

    def test_is_weekly_due_returns_true_on_sunday_when_no_flag_exists(self):
        """Test that is_weekly_due returns True on Sunday when no flag exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Mock datetime to return a Sunday
            sunday = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2026-02-15 is Sunday
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = sunday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                result = is_weekly_due(flag_path)

            assert result is True

    def test_is_weekly_due_returns_false_on_non_sunday(self):
        """Test that is_weekly_due returns False on non-Sunday."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Mock datetime to return a Monday
            monday = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)  # 2026-02-16 is Monday
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = monday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                result = is_weekly_due(flag_path)

            assert result is False

    def test_is_weekly_due_returns_false_on_sunday_when_flag_matches_current_week(self):
        """Test that is_weekly_due returns False on Sunday when flag matches current week."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Mock datetime to return a Sunday
            sunday = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2026-02-15 is Sunday
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = sunday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                # Create flag file with current week
                Path(flag_path).parent.mkdir(parents=True, exist_ok=True)
                Path(flag_path).write_text("2026-W07", encoding="utf-8")

                result = is_weekly_due(flag_path)

            assert result is False

    def test_is_weekly_due_returns_true_on_sunday_when_flag_is_from_previous_week(self):
        """Test that is_weekly_due returns True on Sunday when flag is from previous week."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Mock datetime to return a Sunday
            sunday = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2026-02-15 is Sunday (W07)
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = sunday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                # Create flag file with previous week
                Path(flag_path).parent.mkdir(parents=True, exist_ok=True)
                Path(flag_path).write_text("2026-W06", encoding="utf-8")

                result = is_weekly_due(flag_path)

            assert result is True

    def test_mark_weekly_done_creates_flag_file_with_correct_format(self):
        """Test that mark_weekly_done creates flag file with correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Mock datetime to return a specific date
            test_date = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2026-W07
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = test_date
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                mark_weekly_done(flag_path)

            # Verify file exists and content is correct
            assert Path(flag_path).exists()
            content = Path(flag_path).read_text(encoding="utf-8").strip()
            assert content == "2026-W07"

    def test_mark_weekly_done_creates_parent_directories(self):
        """Test that mark_weekly_done creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/subdir/nested/weekly_done.flag"

            # Mock datetime
            test_date = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = test_date
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                mark_weekly_done(flag_path)

            # Verify parent directories were created
            assert Path(flag_path).parent.exists()
            assert Path(flag_path).exists()

    def test_read_weekly_flag_returns_none_when_file_does_not_exist(self):
        """Test that read_weekly_flag returns None when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            result = read_weekly_flag(flag_path)

            assert result is None

    def test_read_weekly_flag_returns_correct_value_when_file_exists(self):
        """Test that read_weekly_flag returns correct value when file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Create flag file
            Path(flag_path).parent.mkdir(parents=True, exist_ok=True)
            Path(flag_path).write_text("2026-W07", encoding="utf-8")

            result = read_weekly_flag(flag_path)

            assert result == "2026-W07"

    def test_flag_format_is_yyyy_wnn(self):
        """Test that flag format is YYYY-WNN (e.g., 2026-W07)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Test week 7
            test_date = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)  # 2026-W07
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = test_date
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                mark_weekly_done(flag_path)
                content = Path(flag_path).read_text(encoding="utf-8").strip()

            assert content == "2026-W07"
            # Verify format: YYYY-WNN
            assert len(content) == 8
            assert content[4] == "-"
            assert content[5] == "W"
            assert content[6:8].isdigit()

    def test_edge_case_year_boundary_week_52_to_week_1(self):
        """Test edge case: year boundary (week 52/53 to week 1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Test week 52 of 2025
            week_52 = datetime(2025, 12, 28, 12, 0, 0, tzinfo=timezone.utc)  # Sunday of 2025-W52
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = week_52
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                mark_weekly_done(flag_path)
                content_52 = Path(flag_path).read_text(encoding="utf-8").strip()

            assert content_52 == "2025-W52"

            # Test week 1 of 2026
            week_1 = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)  # Sunday of 2026-W01
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = week_1
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                # Flag should be outdated, so is_weekly_due should return True
                result = is_weekly_due(flag_path)

            assert result is True

    def test_read_weekly_flag_handles_empty_file(self):
        """Test that read_weekly_flag handles empty file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Create empty file
            Path(flag_path).parent.mkdir(parents=True, exist_ok=True)
            Path(flag_path).write_text("", encoding="utf-8")

            result = read_weekly_flag(flag_path)

            assert result is None

    def test_integration_full_weekly_cycle(self):
        """Integration test: full weekly cycle from detection to completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flag_path = f"{tmpdir}/weekly_done.flag"

            # Sunday morning: tasks are due
            sunday = datetime(2026, 2, 15, 8, 0, 0, tzinfo=timezone.utc)
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = sunday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                assert is_weekly_due(flag_path) is True

                # Mark tasks as done
                mark_weekly_done(flag_path)

                # Check again: should not be due anymore
                assert is_weekly_due(flag_path) is False

            # Later on the same Sunday: still not due
            sunday_afternoon = datetime(2026, 2, 15, 18, 0, 0, tzinfo=timezone.utc)
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = sunday_afternoon
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                assert is_weekly_due(flag_path) is False

            # Monday: not Sunday, so not due
            monday = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = monday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                assert is_weekly_due(flag_path) is False

            # Next Sunday: new week, tasks are due again
            next_sunday = datetime(2026, 2, 22, 8, 0, 0, tzinfo=timezone.utc)  # 2026-W08
            with patch("core.pipeline.weekly_guard.datetime") as mock_dt:
                mock_dt.now.return_value = next_sunday
                mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                assert is_weekly_due(flag_path) is True
