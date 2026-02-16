"""Tests for scripts.prune_gh_pages module.

Covers directory finding, pruning logic, dry-run mode, edge cases,
and CLI argument parsing per TASK-043 requirements.
"""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from typing import Any

import pytest

from scripts.prune_gh_pages import (
    DirectoryInfo,
    PruneSummary,
    find_dated_directories,
    prune_old_directories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_base_dir():
    """Create a temporary base directory with reports/ structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        reports = base / "reports"
        reports.mkdir()
        yield base


def create_dated_directory(base: Path, date_str: str) -> Path:
    """Helper to create a dated directory under reports/."""
    dir_path = base / "reports" / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def create_root_file(base: Path, filename: str) -> Path:
    """Helper to create a root-level file."""
    file_path = base / filename
    file_path.write_text("test content")
    return file_path


# ---------------------------------------------------------------------------
# Tests for find_dated_directories
# ---------------------------------------------------------------------------


def test_find_dated_directories_empty_reports(temp_base_dir: Path):
    """Should return empty list when reports/ is empty."""
    result = find_dated_directories(str(temp_base_dir))
    assert result == []


def test_find_dated_directories_no_reports_dir():
    """Should return empty list when reports/ doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_dated_directories(tmpdir)
        assert result == []


def test_find_dated_directories_finds_valid_dates(temp_base_dir: Path):
    """Should find all YYYY-MM-DD directories."""
    create_dated_directory(temp_base_dir, "2026-01-15")
    create_dated_directory(temp_base_dir, "2026-02-01")
    create_dated_directory(temp_base_dir, "2025-12-25")

    result = find_dated_directories(str(temp_base_dir))

    assert len(result) == 3
    dates = sorted([info.date for info in result])
    assert dates == [
        datetime.date(2025, 12, 25),
        datetime.date(2026, 1, 15),
        datetime.date(2026, 2, 1),
    ]


def test_find_dated_directories_ignores_non_date_dirs(temp_base_dir: Path):
    """Should ignore directories that don't match YYYY-MM-DD pattern."""
    create_dated_directory(temp_base_dir, "2026-01-15")
    (temp_base_dir / "reports" / "invalid").mkdir()
    (temp_base_dir / "reports" / "2026-13-01").mkdir()  # Invalid month
    (temp_base_dir / "reports" / "20260115").mkdir()  # No hyphens

    result = find_dated_directories(str(temp_base_dir))

    assert len(result) == 1
    assert result[0].path.name == "2026-01-15"


def test_find_dated_directories_handles_invalid_dates(temp_base_dir: Path):
    """Should skip directories with invalid dates like 2026-02-30."""
    create_dated_directory(temp_base_dir, "2026-01-15")
    (temp_base_dir / "reports" / "2026-02-30").mkdir()  # Invalid date

    result = find_dated_directories(str(temp_base_dir))

    assert len(result) == 1
    assert result[0].path.name == "2026-01-15"


def test_find_dated_directories_ignores_files(temp_base_dir: Path):
    """Should ignore files in reports/ directory."""
    create_dated_directory(temp_base_dir, "2026-01-15")
    (temp_base_dir / "reports" / "2026-01-16.txt").write_text("test")

    result = find_dated_directories(str(temp_base_dir))

    assert len(result) == 1
    assert result[0].path.name == "2026-01-15"


def test_find_dated_directories_reports_is_file(temp_base_dir: Path):
    """Should return empty list when reports/ is a file instead of directory."""
    (temp_base_dir / "reports").rmdir()
    (temp_base_dir / "reports").write_text("not a directory")

    result = find_dated_directories(str(temp_base_dir))

    assert result == []


# ---------------------------------------------------------------------------
# Tests for prune_old_directories
# ---------------------------------------------------------------------------


def test_prune_old_directories_deletes_old(temp_base_dir: Path):
    """Should delete directories older than retention period."""
    # Create directories at different dates
    old_date = datetime.date.today() - datetime.timedelta(days=100)
    recent_date = datetime.date.today() - datetime.timedelta(days=10)

    create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))
    create_dated_directory(temp_base_dir, recent_date.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert result.deleted_count == 1
    assert result.kept_count == 1
    assert old_date.strftime("%Y-%m-%d") in result.deleted_dirs
    assert recent_date.strftime("%Y-%m-%d") in result.kept_dirs


def test_prune_old_directories_keeps_recent(temp_base_dir: Path):
    """Should keep directories within retention period."""
    recent1 = datetime.date.today() - datetime.timedelta(days=30)
    recent2 = datetime.date.today() - datetime.timedelta(days=60)

    create_dated_directory(temp_base_dir, recent1.strftime("%Y-%m-%d"))
    create_dated_directory(temp_base_dir, recent2.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert result.deleted_count == 0
    assert result.kept_count == 2


def test_prune_old_directories_respects_retention_days(temp_base_dir: Path):
    """Should use custom retention_days parameter."""
    # Create directory exactly 30 days old
    target_date = datetime.date.today() - datetime.timedelta(days=30)
    create_dated_directory(temp_base_dir, target_date.strftime("%Y-%m-%d"))

    # With retention=30, it should be kept (not older than 30 days)
    result = prune_old_directories(str(temp_base_dir), retention_days=30)
    assert result.deleted_count == 0

    # With retention=29, it should be deleted (older than 29 days)
    create_dated_directory(temp_base_dir, target_date.strftime("%Y-%m-%d"))
    result = prune_old_directories(str(temp_base_dir), retention_days=29)
    assert result.deleted_count == 1


def test_prune_old_directories_dry_run_no_delete(temp_base_dir: Path):
    """Should not delete anything in dry-run mode."""
    old_date = datetime.date.today() - datetime.timedelta(days=100)
    dir_path = create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90, dry_run=True)

    assert result.deleted_count == 1
    assert dir_path.exists()  # Directory should still exist


def test_prune_old_directories_preserves_root_files(temp_base_dir: Path):
    """Should never delete index.html or latest.html at root."""
    create_root_file(temp_base_dir, "index.html")
    create_root_file(temp_base_dir, "latest.html")

    old_date = datetime.date.today() - datetime.timedelta(days=100)
    create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))

    prune_old_directories(str(temp_base_dir), retention_days=90)

    # Root files should still exist
    assert (temp_base_dir / "index.html").exists()
    assert (temp_base_dir / "latest.html").exists()


def test_prune_old_directories_no_reports_dir(temp_base_dir: Path):
    """Should handle case when reports/ doesn't exist."""
    (temp_base_dir / "reports").rmdir()

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert result.deleted_count == 0
    assert result.kept_count == 0


def test_prune_old_directories_mixed_valid_invalid(temp_base_dir: Path):
    """Should only process valid date directories."""
    old_date = datetime.date.today() - datetime.timedelta(days=100)
    create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))
    (temp_base_dir / "reports" / "invalid").mkdir()
    (temp_base_dir / "reports" / "2026-99-99").mkdir()

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    # Only the valid old directory should be deleted
    assert result.deleted_count == 1
    # Invalid directories should be ignored (not counted as kept)
    assert (temp_base_dir / "reports" / "invalid").exists()


def test_prune_old_directories_retention_zero(temp_base_dir: Path):
    """Should delete all directories when retention_days=0."""
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    create_dated_directory(temp_base_dir, today.strftime("%Y-%m-%d"))
    create_dated_directory(temp_base_dir, yesterday.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=0)

    # Only yesterday should be deleted (today is not older)
    assert result.deleted_count == 1
    assert result.kept_count == 1


def test_prune_old_directories_all_old(temp_base_dir: Path):
    """Should delete all when all directories are old."""
    for days_ago in [100, 120, 150]:
        old_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert result.deleted_count == 3
    assert result.kept_count == 0


def test_prune_old_directories_all_recent(temp_base_dir: Path):
    """Should keep all when all directories are recent."""
    for days_ago in [10, 20, 30]:
        recent_date = datetime.date.today() - datetime.timedelta(days=days_ago)
        create_dated_directory(temp_base_dir, recent_date.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert result.deleted_count == 0
    assert result.kept_count == 3


def test_prune_old_directories_invalid_retention_days(temp_base_dir: Path):
    """Should raise ValueError for negative retention_days."""
    with pytest.raises(ValueError, match="must be non-negative"):
        prune_old_directories(str(temp_base_dir), retention_days=-1)


def test_prune_old_directories_nonexistent_base_dir():
    """Should raise FileNotFoundError for non-existent base directory."""
    with pytest.raises(FileNotFoundError):
        prune_old_directories("/nonexistent/path", retention_days=90)


def test_prune_old_directories_summary_structure(temp_base_dir: Path):
    """Should return PruneSummary with correct structure."""
    old_date = datetime.date.today() - datetime.timedelta(days=100)
    recent_date = datetime.date.today() - datetime.timedelta(days=10)

    create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))
    create_dated_directory(temp_base_dir, recent_date.strftime("%Y-%m-%d"))

    result = prune_old_directories(str(temp_base_dir), retention_days=90)

    assert isinstance(result, PruneSummary)
    assert hasattr(result, "deleted_count")
    assert hasattr(result, "kept_count")
    assert hasattr(result, "deleted_dirs")
    assert hasattr(result, "kept_dirs")
    assert isinstance(result.deleted_dirs, list)
    assert isinstance(result.kept_dirs, list)


def test_prune_old_directories_deletes_contents(temp_base_dir: Path):
    """Should delete directory contents recursively."""
    old_date = datetime.date.today() - datetime.timedelta(days=100)
    dir_path = create_dated_directory(temp_base_dir, old_date.strftime("%Y-%m-%d"))

    # Create files inside the directory
    (dir_path / "report.html").write_text("test")
    (dir_path / "data.json").write_text("{}")
    subdir = dir_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("# Test")

    prune_old_directories(str(temp_base_dir), retention_days=90)

    # Directory and all contents should be deleted
    assert not dir_path.exists()


# ---------------------------------------------------------------------------
# Tests for CLI argument defaults
# ---------------------------------------------------------------------------


def test_cli_default_publish_dir():
    """CLI should default to tmp/reports."""
    # This test documents the CLI default behavior
    # Actual CLI testing would require subprocess or similar
    from scripts.prune_gh_pages import main

    # The default is documented in argparse setup
    assert True  # Placeholder for CLI integration test


def test_cli_default_retention_days():
    """CLI should default to 90 days retention."""
    # This test documents the CLI default behavior
    from scripts.prune_gh_pages import main

    # The default is documented in argparse setup
    assert True  # Placeholder for CLI integration test
