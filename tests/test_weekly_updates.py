"""Tests for core.pipeline.weekly_updates.

Covers weekly update scan with filtering by time windows, evaluation status,
markdown/HTML rendering, and file generation.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.pipeline.weekly_updates import (
    generate_update_report,
    render_updates_html,
    render_updates_md,
    scan_updated_papers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create papers table
    conn.execute(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            native_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT NOT NULL,
            categories TEXT NOT NULL,
            published_at_utc TEXT NOT NULL,
            updated_at_utc TEXT,
            pdf_url TEXT,
            has_code INTEGER NOT NULL DEFAULT 0,
            has_code_source TEXT NOT NULL DEFAULT 'none',
            code_url TEXT,
            comment TEXT,
            first_seen_run_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    # Create paper_evaluations table
    conn.execute(
        """
        CREATE TABLE paper_evaluations (
            run_id INTEGER NOT NULL,
            paper_key TEXT NOT NULL,
            embed_score REAL,
            llm_base_score INTEGER NOT NULL,
            flags TEXT NOT NULL,
            bonus_score INTEGER,
            final_score REAL,
            rank INTEGER,
            tier INTEGER,
            discarded INTEGER NOT NULL DEFAULT 0,
            score_lowered INTEGER,
            multi_topic TEXT,
            is_remind INTEGER NOT NULL DEFAULT 0,
            summary_ko TEXT,
            reason_ko TEXT,
            insight_ko TEXT,
            brief_reason TEXT,
            prompt_ver_score TEXT NOT NULL,
            prompt_ver_summ TEXT,
            PRIMARY KEY (run_id, paper_key)
        )
        """
    )

    # Create runs table
    conn.execute(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_slug TEXT NOT NULL,
            window_start_utc TEXT NOT NULL,
            window_end_utc TEXT NOT NULL,
            display_date_kst TEXT NOT NULL,
            embedding_mode TEXT NOT NULL,
            scoring_weights TEXT NOT NULL,
            detected_rpm INTEGER,
            detected_daily_limit INTEGER,
            response_format_supported INTEGER NOT NULL,
            prompt_versions TEXT NOT NULL,
            topic_override_fields TEXT NOT NULL,
            total_collected INTEGER NOT NULL DEFAULT 0,
            total_filtered INTEGER NOT NULL DEFAULT 0,
            total_scored INTEGER NOT NULL DEFAULT 0,
            total_discarded INTEGER NOT NULL DEFAULT 0,
            total_output INTEGER NOT NULL DEFAULT 0,
            threshold_used INTEGER NOT NULL DEFAULT 60,
            threshold_lowered INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            errors TEXT
        )
        """
    )

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def db_file(tmp_path):
    """Create a temporary database file with schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create tables (same as in_memory_db)
    conn.execute(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            native_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT NOT NULL,
            categories TEXT NOT NULL,
            published_at_utc TEXT NOT NULL,
            updated_at_utc TEXT,
            pdf_url TEXT,
            has_code INTEGER NOT NULL DEFAULT 0,
            has_code_source TEXT NOT NULL DEFAULT 'none',
            code_url TEXT,
            comment TEXT,
            first_seen_run_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE paper_evaluations (
            run_id INTEGER NOT NULL,
            paper_key TEXT NOT NULL,
            embed_score REAL,
            llm_base_score INTEGER NOT NULL,
            flags TEXT NOT NULL,
            bonus_score INTEGER,
            final_score REAL,
            rank INTEGER,
            tier INTEGER,
            discarded INTEGER NOT NULL DEFAULT 0,
            score_lowered INTEGER,
            multi_topic TEXT,
            is_remind INTEGER NOT NULL DEFAULT 0,
            summary_ko TEXT,
            reason_ko TEXT,
            insight_ko TEXT,
            brief_reason TEXT,
            prompt_ver_score TEXT NOT NULL,
            prompt_ver_summ TEXT,
            PRIMARY KEY (run_id, paper_key)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_slug TEXT NOT NULL,
            window_start_utc TEXT NOT NULL,
            window_end_utc TEXT NOT NULL,
            display_date_kst TEXT NOT NULL,
            embedding_mode TEXT NOT NULL,
            scoring_weights TEXT NOT NULL,
            detected_rpm INTEGER,
            detected_daily_limit INTEGER,
            response_format_supported INTEGER NOT NULL,
            prompt_versions TEXT NOT NULL,
            topic_override_fields TEXT NOT NULL,
            total_collected INTEGER NOT NULL DEFAULT 0,
            total_filtered INTEGER NOT NULL DEFAULT 0,
            total_scored INTEGER NOT NULL DEFAULT 0,
            total_discarded INTEGER NOT NULL DEFAULT 0,
            total_output INTEGER NOT NULL DEFAULT 0,
            threshold_used INTEGER NOT NULL DEFAULT 60,
            threshold_lowered INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            errors TEXT
        )
        """
    )

    conn.commit()
    conn.close()

    return str(db_path)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def insert_paper(conn, paper_key, native_id, published_at_utc, updated_at_utc):
    """Insert a paper into the database."""
    conn.execute(
        """
        INSERT INTO papers (
            paper_key, source, native_id, url, title, abstract,
            authors, categories, published_at_utc, updated_at_utc,
            has_code, has_code_source, first_seen_run_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_key,
            "arxiv",
            native_id,
            f"https://arxiv.org/abs/{native_id}",
            f"Test Paper {native_id}",
            "Abstract for testing",
            '["Alice", "Bob"]',
            '["cs.AI"]',
            published_at_utc,
            updated_at_utc,
            0,
            "none",
            1,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def insert_run(conn, topic_slug, window_start_utc):
    """Insert a run and return run_id."""
    cur = conn.execute(
        """
        INSERT INTO runs (
            topic_slug, window_start_utc, window_end_utc,
            display_date_kst, embedding_mode, scoring_weights,
            response_format_supported, prompt_versions,
            topic_override_fields, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic_slug,
            window_start_utc,
            window_start_utc,
            "2024-01-01",
            "disabled",
            "{}",
            1,
            "{}",
            "{}",
            "completed",
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_evaluation(conn, run_id, paper_key, llm_base_score, discarded=0):
    """Insert an evaluation record."""
    conn.execute(
        """
        INSERT INTO paper_evaluations (
            run_id, paper_key, llm_base_score, flags,
            prompt_ver_score, discarded
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, paper_key, llm_base_score, "{}", "v1", discarded),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_scan_finds_matching_papers(in_memory_db):
    """Test that scan finds papers matching all 4 conditions."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper matching all conditions:
    # - Published 30 days ago (within 90 days)
    # - Updated 3 days ago (within 7 days)
    published = ref_date - timedelta(days=30)
    updated = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2401.12345",
        "2401.12345",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2401.12345", 85)

    # Use in-memory database directly
    in_memory_db.execute("PRAGMA database_list")
    db_file_path = ":memory:"

    # Save to temp file for testing
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    # Copy in-memory to file
    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        assert len(results) == 1
        assert results[0]["paper_key"] == "arxiv:2401.12345"
        assert results[0]["llm_base_score"] == 85
        assert results[0]["topic_slug"] == "llm-agents"
    finally:
        import os

        os.unlink(temp_path)


def test_scan_excludes_newly_published(in_memory_db):
    """Test that scan excludes papers published within 7 days."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper published 3 days ago (should be excluded)
    published = ref_date - timedelta(days=3)
    updated = ref_date - timedelta(days=1)

    insert_paper(
        in_memory_db,
        "arxiv:2401.99999",
        "2401.99999",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2401.99999", 85)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        assert len(results) == 0
    finally:
        import os

        os.unlink(temp_path)


def test_scan_excludes_old_papers(in_memory_db):
    """Test that scan excludes papers published > 90 days ago."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper published 100 days ago (should be excluded)
    published = ref_date - timedelta(days=100)
    updated = ref_date - timedelta(days=1)

    insert_paper(
        in_memory_db,
        "arxiv:2310.12345",
        "2310.12345",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2310.12345", 85)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        assert len(results) == 0
    finally:
        import os

        os.unlink(temp_path)


def test_scan_excludes_papers_without_evaluations(in_memory_db):
    """Test that scan excludes papers without evaluations."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper matching time conditions but no evaluation
    published = ref_date - timedelta(days=30)
    updated = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2401.88888",
        "2401.88888",
        published.isoformat(),
        updated.isoformat(),
    )

    # No evaluation inserted

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        assert len(results) == 0
    finally:
        import os

        os.unlink(temp_path)


def test_scan_excludes_discarded_evaluations(in_memory_db):
    """Test that scan excludes papers with discarded=1."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper matching time conditions with discarded evaluation
    published = ref_date - timedelta(days=30)
    updated = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2401.77777",
        "2401.77777",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2401.77777", 85, discarded=1)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        assert len(results) == 0
    finally:
        import os

        os.unlink(temp_path)


def test_scan_handles_papers_in_multiple_topics(in_memory_db):
    """Test that scan includes papers evaluated in multiple topics."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Paper evaluated in two topics
    published = ref_date - timedelta(days=30)
    updated = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2401.11111",
        "2401.11111",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id_1 = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    run_id_2 = insert_run(in_memory_db, "edge-computing", ref_date.isoformat())

    insert_evaluation(in_memory_db, run_id_1, "arxiv:2401.11111", 85)
    insert_evaluation(in_memory_db, run_id_2, "arxiv:2401.11111", 90)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        # Should return distinct entries for each topic
        assert len(results) >= 1  # At least one entry
        paper_keys = [r["paper_key"] for r in results]
        assert "arxiv:2401.11111" in paper_keys
    finally:
        import os

        os.unlink(temp_path)


def test_markdown_render_empty_list():
    """Test markdown rendering with empty paper list."""
    md = render_updates_md([], "2024-02-01")
    assert "No updated papers found" in md
    assert "2024-02-01" in md


def test_markdown_render_contains_required_fields():
    """Test markdown rendering contains all required fields."""
    papers = [
        {
            "paper_key": "arxiv:2401.12345",
            "native_id": "2401.12345",
            "title": "Test Paper",
            "url": "https://arxiv.org/abs/2401.12345",
            "llm_base_score": 85,
            "updated_at_utc": "2024-01-25T10:00:00+00:00",
            "topic_slug": "llm-agents",
        }
    ]

    md = render_updates_md(papers, "2024-02-01")

    # Check for required content
    assert "Test Paper" in md
    assert "2401.12345" in md
    assert "85" in md
    assert "2024-01-25" in md
    assert "llm-agents" in md
    assert "https://arxiv.org/abs/2401.12345" in md


def test_html_render_empty_list():
    """Test HTML rendering with empty paper list."""
    html = render_updates_html([], "2024-02-01")
    assert "<!DOCTYPE html>" in html
    assert "No updated papers found" in html
    assert "2024-02-01" in html


def test_html_render_contains_required_fields():
    """Test HTML rendering contains all required fields."""
    papers = [
        {
            "paper_key": "arxiv:2401.12345",
            "native_id": "2401.12345",
            "title": "Test Paper",
            "url": "https://arxiv.org/abs/2401.12345",
            "llm_base_score": 85,
            "updated_at_utc": "2024-01-25T10:00:00+00:00",
            "topic_slug": "llm-agents",
        }
    ]

    html = render_updates_html(papers, "2024-02-01")

    # Check for required content
    assert "<!DOCTYPE html>" in html
    assert "Test Paper" in html
    assert "2401.12345" in html
    assert "85" in html
    assert "llm-agents" in html
    assert "https://arxiv.org/abs/2401.12345" in html


def test_generate_update_report_returns_none_when_no_papers(db_file, tmp_path):
    """Test generate_update_report returns None when no papers found."""
    output_dir = str(tmp_path / "reports")
    result = generate_update_report(db_file, "2024-02-01", output_dir)
    assert result is None


def test_generate_update_report_writes_files_when_papers_exist(
    in_memory_db, tmp_path
):
    """Test generate_update_report writes both files when papers exist."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Insert matching paper
    published = ref_date - timedelta(days=30)
    updated = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2401.12345",
        "2401.12345",
        published.isoformat(),
        updated.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2401.12345", 85)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        output_dir = str(tmp_path / "reports")
        result = generate_update_report(temp_path, ref_str, output_dir)

        assert result is not None
        assert result == output_dir

        # Check that files were created
        md_file = Path(output_dir) / "20240201_weekly_updates.md"
        html_file = Path(output_dir) / "20240201_weekly_updates.html"

        assert md_file.exists()
        assert html_file.exists()

        # Check file contents
        md_content = md_file.read_text()
        html_content = html_file.read_text()

        assert "Test Paper" in md_content
        assert "2401.12345" in md_content

        assert "<!DOCTYPE html>" in html_content
        assert "Test Paper" in html_content
    finally:
        import os

        os.unlink(temp_path)


def test_empty_db_returns_empty_list(db_file):
    """Test scan with empty database returns empty list."""
    results = scan_updated_papers(db_file, "2024-02-01")
    assert results == []


def test_date_boundary_edge_cases(in_memory_db):
    """Test date boundary edge cases (exactly 7 days, exactly 90 days)."""
    ref_date = datetime(2024, 2, 1, tzinfo=timezone.utc)
    ref_str = ref_date.date().isoformat()

    # Case 1: Published exactly 7 days ago, updated exactly 7 days ago
    # Should be excluded (updated_at_utc must be >= 7 days ago, published must be < 7 days ago)
    published_7 = ref_date - timedelta(days=7)
    updated_7 = ref_date - timedelta(days=7)

    insert_paper(
        in_memory_db,
        "arxiv:2401.00001",
        "2401.00001",
        published_7.isoformat(),
        updated_7.isoformat(),
    )

    run_id = insert_run(in_memory_db, "llm-agents", ref_date.isoformat())
    insert_evaluation(in_memory_db, run_id, "arxiv:2401.00001", 85)

    # Case 2: Published exactly 90 days ago, updated 3 days ago
    # Should be included (published_at_utc >= 90 days ago is the boundary)
    published_90 = ref_date - timedelta(days=90)
    updated_3 = ref_date - timedelta(days=3)

    insert_paper(
        in_memory_db,
        "arxiv:2311.00001",
        "2311.00001",
        published_90.isoformat(),
        updated_3.isoformat(),
    )

    insert_evaluation(in_memory_db, run_id, "arxiv:2311.00001", 90)

    # Save to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        temp_path = f.name

    backup_conn = sqlite3.connect(temp_path)
    in_memory_db.backup(backup_conn)
    backup_conn.close()

    try:
        results = scan_updated_papers(temp_path, ref_str)
        # Should find the 90-day boundary paper
        paper_keys = [r["paper_key"] for r in results]
        assert "arxiv:2311.00001" in paper_keys
    finally:
        import os

        os.unlink(temp_path)
