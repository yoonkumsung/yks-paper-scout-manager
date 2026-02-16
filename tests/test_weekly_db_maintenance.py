"""Tests for weekly DB maintenance orchestrator."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from core.pipeline.weekly_db_maintenance import (
    _cleanup_old_assets,
    _upload_release_asset,
    run_weekly_maintenance,
)
from core.storage.db_manager import DBManager


@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DBManager(str(db_path))
        db.close()
        yield str(db_path)


def test_run_weekly_maintenance_full_cycle(temp_db):
    """Test that all purge steps execute in order."""
    with mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        mock_upload.return_value = True
        mock_cleanup.return_value = 2

        summary = run_weekly_maintenance(
            db_path=temp_db,
            eval_days=90,
            papers_days=365,
        )

        # Verify all operations completed
        assert "purged_evaluations" in summary
        assert "purged_query_stats" in summary
        assert "purged_runs" in summary
        assert "purged_remind_tracking" in summary
        assert "purged_papers" in summary
        assert "vacuum_done" in summary
        assert "release_asset_uploaded" in summary
        assert "old_assets_deleted" in summary

        # Verify upload and cleanup were called
        mock_upload.assert_called_once()
        mock_cleanup.assert_called_once()


def test_purge_counts_reported(temp_db):
    """Test that summary dict has correct counts."""
    with mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        mock_upload.return_value = False
        mock_cleanup.return_value = 0

        summary = run_weekly_maintenance(db_path=temp_db)

        # All counts should be integers
        assert isinstance(summary["purged_evaluations"], int)
        assert isinstance(summary["purged_query_stats"], int)
        assert isinstance(summary["purged_runs"], int)
        assert isinstance(summary["purged_remind_tracking"], int)
        assert isinstance(summary["purged_papers"], int)
        assert isinstance(summary["old_assets_deleted"], int)


def test_vacuum_called(temp_db):
    """Test that VACUUM runs after purges."""
    with mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        mock_upload.return_value = False
        mock_cleanup.return_value = 0

        summary = run_weekly_maintenance(db_path=temp_db)

        # VACUUM should complete
        assert summary["vacuum_done"] is True


def test_release_upload_success(temp_db):
    """Test Release asset upload with mocked subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        test_db.write_text("fake db content")

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with mock.patch("shutil.which", return_value="/usr/bin/gh"), \
             mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}), \
             mock.patch("subprocess.run", return_value=mock_result):

            result = _upload_release_asset(
                str(test_db),
                "db-backup",
                "20260217",
            )

            assert result is True


def test_release_upload_failure_continues(temp_db):
    """Test that upload error doesn't stop pipeline."""
    mock_result = mock.Mock()
    mock_result.returncode = 1
    mock_result.stderr = "upload failed"

    with mock.patch("shutil.which", return_value="/usr/bin/gh"), \
         mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}), \
         mock.patch("subprocess.run", return_value=mock_result), \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        mock_cleanup.return_value = 0

        summary = run_weekly_maintenance(db_path=temp_db)

        # Pipeline should complete despite upload failure
        assert "purged_evaluations" in summary
        assert summary["release_asset_uploaded"] is False


def test_gh_not_available():
    """Test that missing gh CLI is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        test_db.write_text("fake db content")

        with mock.patch("shutil.which", return_value=None):
            result = _upload_release_asset(
                str(test_db),
                "db-backup",
                "20260217",
            )

            assert result is False


def test_no_github_token():
    """Test that missing GITHUB_TOKEN is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        test_db.write_text("fake db content")

        with mock.patch("shutil.which", return_value="/usr/bin/gh"), \
             mock.patch.dict(os.environ, {}, clear=True):

            result = _upload_release_asset(
                str(test_db),
                "db-backup",
                "20260217",
            )

            assert result is False


def test_cleanup_old_assets():
    """Test that correct assets are identified and deleted."""
    # Mock release view output
    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=35)).isoformat().replace("+00:00", "Z")
    recent_date = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")

    mock_view_result = mock.Mock()
    mock_view_result.returncode = 0
    mock_view_result.stdout = json.dumps({
        "assets": [
            {"name": "paper-scout-db-20260101.sqlite", "updatedAt": old_date},
            {"name": "paper-scout-db-20260210.sqlite", "updatedAt": recent_date},
        ]
    })

    mock_delete_result = mock.Mock()
    mock_delete_result.returncode = 0

    with mock.patch("shutil.which", return_value="/usr/bin/gh"), \
         mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}), \
         mock.patch("subprocess.run") as mock_run:

        # First call returns view result, second call returns delete result
        mock_run.side_effect = [mock_view_result, mock_delete_result]

        deleted_count = _cleanup_old_assets("db-backup", 28)

        # Should delete 1 old asset (35 days old)
        assert deleted_count == 1

        # Verify delete was called
        assert mock_run.call_count == 2
        delete_call = mock_run.call_args_list[1]
        assert "delete-asset" in delete_call[0][0]
        assert "paper-scout-db-20260101.sqlite" in delete_call[0][0]


def test_cleanup_no_old_assets():
    """Test cleanup when no assets are old enough to delete."""
    now = datetime.now(timezone.utc)
    recent_date = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")

    mock_view_result = mock.Mock()
    mock_view_result.returncode = 0
    mock_view_result.stdout = json.dumps({
        "assets": [
            {"name": "paper-scout-db-20260210.sqlite", "updatedAt": recent_date},
        ]
    })

    with mock.patch("shutil.which", return_value="/usr/bin/gh"), \
         mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"}), \
         mock.patch("subprocess.run", return_value=mock_view_result):

        deleted_count = _cleanup_old_assets("db-backup", 28)

        # Should delete nothing
        assert deleted_count == 0


def test_exception_in_purge_continues(temp_db):
    """Test that one purge failure doesn't stop others."""
    with mock.patch("core.storage.db_manager.DBManager.purge_old_evaluations") as mock_purge, \
         mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        # First purge fails
        mock_purge.side_effect = RuntimeError("DB locked")
        mock_upload.return_value = False
        mock_cleanup.return_value = 0

        summary = run_weekly_maintenance(db_path=temp_db)

        # Pipeline should complete
        assert "purged_evaluations" in summary
        assert "purged_query_stats" in summary
        assert "vacuum_done" in summary


def test_custom_retention_days(temp_db):
    """Test that non-default parameters work."""
    with mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
         mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

        mock_upload.return_value = True
        mock_cleanup.return_value = 1

        summary = run_weekly_maintenance(
            db_path=temp_db,
            eval_days=60,
            papers_days=180,
            release_tag="custom-tag",
            asset_retention_days=14,
        )

        # Verify cleanup was called with custom retention
        mock_cleanup.assert_called_once_with("custom-tag", 14)

        # Pipeline should complete
        assert summary["vacuum_done"] is True


def test_integration_with_real_db():
    """Test with in-memory DB and test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_integration.db"
        db = DBManager(str(db_path))

        # Create test data (no old runs - nothing to purge)
        from core.models import RunMeta
        run_meta = RunMeta(
            topic_slug="test-topic",
            window_start_utc=datetime.now(timezone.utc) - timedelta(days=1),
            window_end_utc=datetime.now(timezone.utc),
            display_date_kst="2026-02-17",
            embedding_mode="voyage",
            scoring_weights={},
            detected_rpm=None,
            detected_daily_limit=None,
            response_format_supported=True,
            prompt_versions={},
            topic_override_fields={},
            total_collected=0,
            total_filtered=0,
            total_scored=0,
            total_discarded=0,
            total_output=0,
            threshold_used=60,
            threshold_lowered=False,
            status="completed",
            errors=None,
        )
        db.create_run(run_meta)
        db.close()

        with mock.patch("core.pipeline.weekly_db_maintenance._upload_release_asset") as mock_upload, \
             mock.patch("core.pipeline.weekly_db_maintenance._cleanup_old_assets") as mock_cleanup:

            mock_upload.return_value = False
            mock_cleanup.return_value = 0

            summary = run_weekly_maintenance(db_path=str(db_path))

            # Should complete without errors
            assert summary["purged_evaluations"] == 0
            assert summary["purged_query_stats"] == 0
            assert summary["purged_runs"] == 0
            assert summary["vacuum_done"] is True
