"""Tests for scripts/cache_manager.py."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from scripts.cache_manager import (
    SCHEMA_VERSION,
    cleanup_old_releases,
    ensure_db,
    get_cache_key,
    get_schema_hash,
    restore_from_release,
    upload_to_release,
)


class TestGetSchemaHash:
    """Test get_schema_hash function."""

    def test_returns_consistent_hash(self):
        """Should return consistent hash for same schema version."""
        hash1 = get_schema_hash()
        hash2 = get_schema_hash()
        assert hash1 == hash2

    def test_returns_8_character_hash(self):
        """Should return 8-character hash."""
        schema_hash = get_schema_hash()
        assert len(schema_hash) == 8

    def test_accepts_optional_db_path(self):
        """Should accept db_path parameter (for future use)."""
        schema_hash = get_schema_hash("/tmp/test.db")
        assert len(schema_hash) == 8


class TestGetCacheKey:
    """Test get_cache_key function."""

    def test_correct_format(self):
        """Should return cache key in correct format."""
        cache_key = get_cache_key("main", "abc12345")
        assert cache_key == "db-main-abc12345"

    def test_different_branches(self):
        """Should generate different keys for different branches."""
        key1 = get_cache_key("main", "abc12345")
        key2 = get_cache_key("develop", "abc12345")
        assert key1 != key2
        assert key1 == "db-main-abc12345"
        assert key2 == "db-develop-abc12345"

    def test_different_schemas(self):
        """Should generate different keys for different schemas."""
        key1 = get_cache_key("main", "abc12345")
        key2 = get_cache_key("main", "def67890")
        assert key1 != key2


class TestRestoreFromRelease:
    """Test restore_from_release function."""

    @patch("subprocess.run")
    def test_gh_not_available(self, mock_run):
        """Should return False when gh CLI not available."""
        mock_run.return_value = MagicMock(returncode=1, stderr="command not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            result = restore_from_release("owner/repo", db_path)

        assert result is False
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_no_releases_found(self, mock_run):
        """Should return False when no releases exist."""
        # gh --version succeeds
        mock_run.return_value = MagicMock(returncode=0, stdout="gh version 2.0.0")

        # gh release list returns empty
        def side_effect(*args, **kwargs):
            if "release" in args[0] and "list" in args[0]:
                return MagicMock(returncode=0, stdout="")
            return MagicMock(returncode=0, stdout="gh version 2.0.0")

        mock_run.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            result = restore_from_release("owner/repo", db_path)

        assert result is False

    @patch("subprocess.run")
    def test_successful_restore(self, mock_run):
        """Should successfully download DB from release."""
        release_data = {
            "assets": [
                {"name": "paper-scout-db-20260101.sqlite"},
            ]
        }

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "release" in cmd and "list" in cmd:
                return MagicMock(returncode=0, stdout="v1.0.0\tLatest\n")
            elif "release" in cmd and "view" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps(release_data))
            elif "release" in cmd and "download" in cmd:
                # Create a dummy DB file
                db_path = kwargs.get("timeout")  # Hacky way to get db_path
                # Actually, we need to look at -O argument
                output_idx = cmd.index("-O") + 1
                db_path = cmd[output_idx]
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                Path(db_path).write_text("dummy db")
                return MagicMock(returncode=0, stdout="")
            else:
                return MagicMock(returncode=0, stdout="gh version 2.0.0")

        mock_run.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "data", "test.db")
            result = restore_from_release("owner/repo", db_path)

        assert result is True

    @patch("subprocess.run")
    def test_no_db_assets_in_release(self, mock_run):
        """Should return False when release has no DB assets."""
        release_data = {
            "assets": [
                {"name": "some-other-file.txt"},
            ]
        }

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "release" in cmd and "list" in cmd:
                return MagicMock(returncode=0, stdout="v1.0.0\tLatest\n")
            elif "release" in cmd and "view" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps(release_data))
            else:
                return MagicMock(returncode=0, stdout="gh version 2.0.0")

        mock_run.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            result = restore_from_release("owner/repo", db_path)

        assert result is False


class TestUploadToRelease:
    """Test upload_to_release function."""

    def test_db_file_not_found(self):
        """Should return False when DB file doesn't exist."""
        result = upload_to_release("owner/repo", "/nonexistent/test.db", "v1.0.0")
        assert result is False

    @patch("subprocess.run")
    def test_successful_upload(self, mock_run):
        """Should successfully upload DB to release."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        with tempfile.NamedTemporaryFile(suffix=".db") as tmpfile:
            tmpfile.write(b"dummy db content")
            tmpfile.flush()

            result = upload_to_release("owner/repo", tmpfile.name, "v1.0.0")

        assert result is True
        mock_run.assert_called_once()
        # Verify gh release upload was called
        call_args = mock_run.call_args[0][0]
        assert "gh" in call_args
        assert "release" in call_args
        assert "upload" in call_args

    @patch("subprocess.run")
    def test_upload_failure(self, mock_run):
        """Should return False when upload fails."""
        mock_run.return_value = MagicMock(returncode=1, stderr="upload failed")

        with tempfile.NamedTemporaryFile(suffix=".db") as tmpfile:
            tmpfile.write(b"dummy db content")
            tmpfile.flush()

            result = upload_to_release("owner/repo", tmpfile.name, "v1.0.0")

        assert result is False


class TestCleanupOldReleases:
    """Test cleanup_old_releases function."""

    @patch("subprocess.run")
    def test_no_releases(self, mock_run):
        """Should return 0 when no releases exist."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        deleted = cleanup_old_releases("owner/repo", keep_weeks=4)

        assert deleted == 0

    @patch("subprocess.run")
    def test_delete_old_assets(self, mock_run):
        """Should delete assets older than keep_weeks."""
        old_date = (datetime.now(timezone.utc) - timedelta(weeks=5)).isoformat()
        recent_date = (datetime.now(timezone.utc) - timedelta(weeks=1)).isoformat()

        old_release_data = {
            "publishedAt": old_date,
            "assets": [{"name": "paper-scout-db-20250101.sqlite"}]
        }
        recent_release_data = {
            "publishedAt": recent_date,
            "assets": [{"name": "paper-scout-db-20260101.sqlite"}]
        }

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "release" in cmd and "list" in cmd:
                return MagicMock(returncode=0, stdout="v1.0.0\tOld\nv2.0.0\tRecent\n")
            elif "release" in cmd and "view" in cmd:
                if "v1.0.0" in cmd:
                    return MagicMock(returncode=0, stdout=json.dumps(old_release_data))
                else:
                    return MagicMock(returncode=0, stdout=json.dumps(recent_release_data))
            elif "delete-asset" in cmd:
                return MagicMock(returncode=0, stdout="")
            else:
                return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect

        deleted = cleanup_old_releases("owner/repo", keep_weeks=4)

        # Should delete 1 old asset, keep 1 recent
        assert deleted == 1

    @patch("subprocess.run")
    def test_keep_recent_assets(self, mock_run):
        """Should not delete recent assets."""
        recent_date = (datetime.now(timezone.utc) - timedelta(weeks=1)).isoformat()

        recent_release_data = {
            "publishedAt": recent_date,
            "assets": [{"name": "paper-scout-db-20260101.sqlite"}]
        }

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "release" in cmd and "list" in cmd:
                return MagicMock(returncode=0, stdout="v1.0.0\tRecent\n")
            elif "release" in cmd and "view" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps(recent_release_data))
            else:
                return MagicMock(returncode=0, stdout="")

        mock_run.side_effect = side_effect

        deleted = cleanup_old_releases("owner/repo", keep_weeks=4)

        assert deleted == 0


class TestEnsureDB:
    """Test ensure_db function."""

    def test_db_already_exists(self):
        """Should return path when DB already exists."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpfile:
            tmpfile.write(b"existing db")
            db_path = tmpfile.name

        try:
            result = ensure_db(db_path)
            assert result == db_path
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    @patch("scripts.cache_manager.restore_from_release")
    @patch("core.storage.db_manager.DBManager")
    def test_restore_from_release_success(self, mock_dbmanager, mock_restore):
        """Should restore from release when available."""
        mock_restore.return_value = True
        mock_db_instance = MagicMock()
        mock_dbmanager.return_value = mock_db_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            result = ensure_db(db_path, repo="owner/repo")

        assert result == db_path
        mock_restore.assert_called_once_with("owner/repo", db_path)
        # DBManager should not be called since restore succeeded
        mock_dbmanager.assert_not_called()

    @patch("scripts.cache_manager.restore_from_release")
    @patch("core.storage.db_manager.DBManager")
    def test_create_empty_db_when_no_release(self, mock_dbmanager, mock_restore):
        """Should create empty DB when no release available."""
        mock_restore.return_value = False
        mock_db_instance = MagicMock()
        mock_dbmanager.return_value = mock_db_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "data", "test.db")
            result = ensure_db(db_path, repo="owner/repo")

        assert result == db_path
        mock_restore.assert_called_once_with("owner/repo", db_path)
        mock_dbmanager.assert_called_once_with(db_path)
        mock_db_instance.close.assert_called_once()

    @patch("core.storage.db_manager.DBManager")
    def test_create_empty_db_without_repo(self, mock_dbmanager):
        """Should create empty DB when no repo specified."""
        mock_db_instance = MagicMock()
        mock_dbmanager.return_value = mock_db_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            result = ensure_db(db_path)

        assert result == db_path
        mock_dbmanager.assert_called_once_with(db_path)
        mock_db_instance.close.assert_called_once()


class TestIntegration:
    """Integration tests for cache manager workflow."""

    def test_cache_key_generation_workflow(self):
        """Should generate consistent cache keys for workflow."""
        schema_hash = get_schema_hash()

        # Same branch, same schema -> same key
        key1 = get_cache_key("main", schema_hash)
        key2 = get_cache_key("main", schema_hash)
        assert key1 == key2

        # Different branch -> different key
        key3 = get_cache_key("develop", schema_hash)
        assert key1 != key3

    @patch("core.storage.db_manager.DBManager")
    def test_full_ensure_workflow_no_repo(self, mock_dbmanager):
        """Should handle full ensure workflow without repo."""
        mock_db_instance = MagicMock()
        mock_dbmanager.return_value = mock_db_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "data", "paper_scout.db")

            # First call: create DB
            result1 = ensure_db(db_path)
            assert result1 == db_path

            # Second call: DB exists, no creation
            Path(db_path).touch()  # Simulate DB creation
            result2 = ensure_db(db_path)
            assert result2 == db_path
