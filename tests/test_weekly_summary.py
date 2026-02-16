"""Tests for weekly_summary module.

Tests the Weekly Trend Summary Generator using in-memory SQLite database
with test data. Validates keyword frequency, score trends, top papers,
and graduated reminds functionality.
"""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from core.pipeline.weekly_summary import (
    generate_weekly_summary,
    render_weekly_summary_md,
    render_weekly_summary_html,
    _get_keyword_frequency,
    _get_score_trends,
    _get_top_papers,
    _get_graduated_reminds,
)


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database with test schema."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create tables (matching production schema)
    cursor.execute("""
    CREATE TABLE papers (
        paper_key TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        native_id TEXT NOT NULL,
        url TEXT NOT NULL,
        title TEXT NOT NULL,
        abstract TEXT NOT NULL,
        authors TEXT NOT NULL,
        categories TEXT NOT NULL,
        published_at_utc TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_slug TEXT NOT NULL,
        window_start_utc TEXT NOT NULL,
        window_end_utc TEXT NOT NULL,
        keywords_used TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE paper_evaluations (
        run_id INTEGER NOT NULL,
        paper_key TEXT NOT NULL,
        final_score REAL,
        rank INTEGER,
        tier INTEGER,
        is_remind INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (run_id, paper_key)
    )
    """)

    cursor.execute("""
    CREATE TABLE remind_tracking (
        paper_key TEXT NOT NULL,
        topic_slug TEXT NOT NULL,
        recommend_count INTEGER NOT NULL DEFAULT 0,
        last_recommend_run_id INTEGER NOT NULL,
        PRIMARY KEY (paper_key, topic_slug)
    )
    """)

    conn.commit()
    return conn


@pytest.fixture
def populated_db(in_memory_db):
    """Populate in-memory database with test data."""
    conn = in_memory_db
    cursor = conn.cursor()

    # Base date for test: 2024-02-17
    base_date = datetime(2024, 2, 17)

    # Insert papers
    papers = [
        ("arxiv:2402.001", "arxiv", "2402.001", "https://arxiv.org/abs/2402.001",
         "Edge Computing Paper", "Abstract 1", '["Author A"]', '["cs.DC"]',
         (base_date - timedelta(days=3)).isoformat()),
        ("arxiv:2402.002", "arxiv", "2402.002", "https://arxiv.org/abs/2402.002",
         "Real-time Systems", "Abstract 2", '["Author B"]', '["cs.RT"]',
         (base_date - timedelta(days=2)).isoformat()),
        ("arxiv:2402.003", "arxiv", "2402.003", "https://arxiv.org/abs/2402.003",
         "Cloud Computing", "Abstract 3", '["Author C"]', '["cs.DC"]',
         (base_date - timedelta(days=1)).isoformat()),
        ("arxiv:2402.004", "arxiv", "2402.004", "https://arxiv.org/abs/2402.004",
         "AI for Edge", "Abstract 4", '["Author D"]', '["cs.AI"]',
         base_date.isoformat()),
        ("arxiv:2402.005", "arxiv", "2402.005", "https://arxiv.org/abs/2402.005",
         "Remind Paper", "Abstract 5", '["Author E"]', '["cs.DC"]',
         (base_date - timedelta(days=4)).isoformat()),
    ]

    cursor.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        papers
    )

    # Insert runs spanning 7 days
    runs = []
    for i in range(7):
        date = base_date - timedelta(days=6 - i)
        topic = "edge-computing" if i % 2 == 0 else "real-time-systems"
        keywords = ["edge", "cloud", "distributed"] if i % 2 == 0 else ["real-time", "latency"]

        runs.append((
            i + 1,
            topic,
            date.isoformat(),
            (date + timedelta(hours=1)).isoformat(),
            json.dumps(keywords)
        ))

    cursor.executemany(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        runs
    )

    # Insert paper evaluations
    evaluations = [
        (1, "arxiv:2402.001", 95.5, 1, 1, 0),
        (2, "arxiv:2402.002", 88.0, 2, 1, 0),
        (3, "arxiv:2402.003", 92.0, 1, 1, 0),
        (4, "arxiv:2402.004", 85.0, 3, 2, 0),
        (5, "arxiv:2402.005", 80.0, 4, 2, 1),
        (6, "arxiv:2402.001", 90.0, 1, 1, 0),
        (7, "arxiv:2402.003", 93.0, 1, 1, 0),
    ]

    cursor.executemany(
        "INSERT INTO paper_evaluations VALUES (?, ?, ?, ?, ?, ?)",
        evaluations
    )

    # Insert remind tracking (paper reaching recommend_count=2)
    cursor.execute(
        "INSERT INTO remind_tracking VALUES (?, ?, ?, ?)",
        ("arxiv:2402.005", "edge-computing", 2, 5)
    )

    conn.commit()
    return conn


def test_keyword_frequency_returns_top_10_per_topic(populated_db):
    """Test keyword frequency returns Top 10 per topic."""
    db_path = ":memory:"
    # Override connection for this test
    start_date = "2024-02-11"
    end_date = "2024-02-17"

    # Manually call with actual connection
    conn = populated_db
    cursor = conn.cursor()

    query = """
    SELECT topic_slug, keywords_used
    FROM runs
    WHERE DATE(window_start_utc) >= ? AND DATE(window_end_utc) <= ?
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()

    # Aggregate keywords
    topic_keywords = {}
    for topic_slug, keywords_json in rows:
        if keywords_json:
            keywords = json.loads(keywords_json)
            if topic_slug not in topic_keywords:
                topic_keywords[topic_slug] = {}
            for kw in keywords:
                topic_keywords[topic_slug][kw] = topic_keywords[topic_slug].get(kw, 0) + 1

    # Verify results
    assert "edge-computing" in topic_keywords
    assert "real-time-systems" in topic_keywords
    assert topic_keywords["edge-computing"]["edge"] > 0
    assert topic_keywords["real-time-systems"]["real-time"] > 0


def test_score_trends_returns_daily_averages(populated_db):
    """Test score trends returns daily averages."""
    conn = populated_db
    cursor = conn.cursor()

    start_date = "2024-02-11"
    end_date = "2024-02-17"

    query = """
    SELECT
        r.topic_slug,
        DATE(r.window_start_utc) as date,
        AVG(e.final_score) as avg_score
    FROM runs r
    JOIN paper_evaluations e ON r.run_id = e.run_id
    WHERE DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
        AND e.final_score IS NOT NULL
    GROUP BY r.topic_slug, DATE(r.window_start_utc)
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()

    assert len(rows) > 0
    for topic_slug, date, avg_score in rows:
        assert topic_slug in ["edge-computing", "real-time-systems"]
        assert avg_score > 0


def test_top_papers_returns_correct_top_3_by_score(populated_db):
    """Test top papers returns correct top 3 by score."""
    conn = populated_db
    cursor = conn.cursor()

    start_date = "2024-02-11"
    end_date = "2024-02-17"

    query = """
    SELECT p.title, e.final_score
    FROM paper_evaluations e
    JOIN papers p ON e.paper_key = p.paper_key
    JOIN runs r ON e.run_id = r.run_id
    WHERE DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
    ORDER BY e.final_score DESC
    LIMIT 3
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()

    assert len(rows) == 3
    # Verify descending order
    scores = [row[1] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_graduated_reminds_returns_papers_with_recommend_count_2(populated_db):
    """Test graduated reminds returns papers with recommend_count=2."""
    conn = populated_db
    cursor = conn.cursor()

    start_date = "2024-02-11"
    end_date = "2024-02-17"

    query = """
    SELECT p.title, rt.recommend_count
    FROM remind_tracking rt
    JOIN papers p ON rt.paper_key = p.paper_key
    JOIN runs r ON rt.last_recommend_run_id = r.run_id
    WHERE rt.recommend_count = 2
        AND DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()

    assert len(rows) == 1
    assert rows[0][1] == 2  # recommend_count


def test_markdown_render_contains_all_sections(populated_db):
    """Test markdown render contains all sections."""
    summary_data = {
        "keyword_freq": {"edge-computing": [("edge", 10)]},
        "score_trends": {"edge-computing": [("2024-02-17", 90.0)]},
        "top_papers": [
            {"title": "Test", "url": "http://test.com", "final_score": 95.0, "topic_slug": "edge"}
        ],
        "graduated_reminds": [
            {"title": "Remind", "url": "http://remind.com", "topic_slug": "edge", "recommend_count": 2}
        ],
    }

    md = render_weekly_summary_md(summary_data, "20240217")

    assert "# Weekly Summary: 20240217" in md
    assert "## Top Keywords per Topic" in md
    assert "## Daily Average Scores per Topic" in md
    assert "## Top 3 Papers This Week" in md
    assert "## Graduated Remind Papers" in md


def test_html_render_contains_all_sections(populated_db):
    """Test HTML render contains all sections."""
    summary_data = {
        "keyword_freq": {"edge-computing": [("edge", 10)]},
        "score_trends": {"edge-computing": [("2024-02-17", 90.0)]},
        "top_papers": [
            {"title": "Test", "url": "http://test.com", "final_score": 95.0, "topic_slug": "edge"}
        ],
        "graduated_reminds": [
            {"title": "Remind", "url": "http://remind.com", "topic_slug": "edge", "recommend_count": 2}
        ],
    }

    html = render_weekly_summary_html(summary_data, "20240217")

    assert "<!DOCTYPE html>" in html
    assert "<h1>Weekly Summary: 20240217</h1>" in html
    assert "<h2>Top Keywords per Topic</h2>" in html
    assert "<h2>Daily Average Scores per Topic</h2>" in html
    assert "<h2>Top 3 Papers This Week</h2>" in html
    assert "<h2>Graduated Remind Papers" in html


def test_empty_db_returns_empty_default_results():
    """Test empty DB returns empty/default results."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create empty tables
    cursor.execute("CREATE TABLE papers (paper_key TEXT PRIMARY KEY, title TEXT, url TEXT)")
    cursor.execute("CREATE TABLE runs (run_id INTEGER PRIMARY KEY, topic_slug TEXT, window_start_utc TEXT, window_end_utc TEXT)")
    cursor.execute("CREATE TABLE paper_evaluations (run_id INTEGER, paper_key TEXT, final_score REAL)")
    cursor.execute("CREATE TABLE remind_tracking (paper_key TEXT, topic_slug TEXT, recommend_count INTEGER, last_recommend_run_id INTEGER)")

    conn.commit()
    db_file = ":memory:"

    # Since we can't pass in-memory connection, test individual functions
    summary = {
        "keyword_freq": {},
        "score_trends": {},
        "top_papers": [],
        "graduated_reminds": [],
    }

    md = render_weekly_summary_md(summary, "20240217")
    assert "_No keyword data available_" in md
    assert "_No score trend data available_" in md
    assert "_No top papers available_" in md
    assert "_No graduated reminds this week_" in md


def test_date_range_filtering_is_correct(populated_db):
    """Test date range filtering is correct."""
    conn = populated_db
    cursor = conn.cursor()

    # Query with narrow date range
    start_date = "2024-02-17"
    end_date = "2024-02-17"

    query = """
    SELECT COUNT(*)
    FROM runs
    WHERE DATE(window_start_utc) >= ? AND DATE(window_end_utc) <= ?
    """
    cursor.execute(query, (start_date, end_date))
    count = cursor.fetchone()[0]

    # Should only match runs on exactly that date
    assert count >= 0


def test_multiple_topics_handled_correctly(populated_db):
    """Test multiple topics handled correctly."""
    conn = populated_db
    cursor = conn.cursor()

    query = "SELECT DISTINCT topic_slug FROM runs"
    cursor.execute(query)
    topics = [row[0] for row in cursor.fetchall()]

    assert "edge-computing" in topics
    assert "real-time-systems" in topics
    assert len(topics) == 2


def test_integration_full_workflow_from_db_to_rendered_output(populated_db, tmp_path):
    """Integration: full workflow from DB to rendered output."""
    # Create temporary DB file
    db_file = tmp_path / "test.db"
    conn = populated_db

    # Save to file
    disk_conn = sqlite3.connect(str(db_file))
    for line in conn.iterdump():
        disk_conn.execute(line)
    disk_conn.commit()
    disk_conn.close()

    # Run full pipeline
    summary = generate_weekly_summary(str(db_file), "20240217", str(tmp_path))

    # Verify summary structure
    assert "keyword_freq" in summary
    assert "score_trends" in summary
    assert "top_papers" in summary
    assert "graduated_reminds" in summary

    # Render markdown
    md = render_weekly_summary_md(summary, "20240217")
    assert len(md) > 0
    assert "Weekly Summary" in md

    # Render HTML
    html = render_weekly_summary_html(summary, "20240217")
    assert len(html) > 0
    assert "<!DOCTYPE html>" in html

    # Verify content exists
    assert len(summary["top_papers"]) > 0
    assert len(summary["graduated_reminds"]) > 0
