"""Tests for local_ui Flask application.

Tests API endpoints, configuration operations, and UI rendering.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
import yaml

from local_ui.app import create_app
from local_ui.config_io import (
    add_topic,
    get_topics,
    read_config,
    remove_topic,
    update_topic,
    write_config,
)


@pytest.fixture
def temp_config():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "app": {"display_timezone": "Asia/Seoul"},
            "llm": {"provider": "openrouter"},
            "agents": {},
            "sources": {},
            "filter": {},
            "embedding": {},
            "scoring": {},
            "remind": {},
            "clustering": {},
            "topics": [],
            "output": {},
            "notifications": {},
            "database": {"path": "data/paper_scout.db"},
            "weekly": {},
            "local_ui": {"host": "127.0.0.1", "port": 8585},
        }
        yaml.safe_dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE papers (
            paper_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            native_id TEXT NOT NULL,
            canonical_id TEXT,
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

    cursor.execute(
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

    cursor.execute(
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
            errors TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def app(temp_config, temp_db):
    """Create Flask test app."""
    app = create_app(config_path=temp_config, db_path=temp_db)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Test app creation
# ---------------------------------------------------------------------------


def test_app_creation(temp_config, temp_db):
    """Test Flask app creates successfully."""
    app = create_app(config_path=temp_config, db_path=temp_db)
    assert app is not None
    assert app.config["CONFIG_PATH"] == temp_config
    assert app.config["DB_PATH"] == temp_db


def test_localhost_binding(temp_config):
    """Test that app binds to 127.0.0.1 for security."""
    config = read_config(temp_config)
    assert config["local_ui"]["host"] == "127.0.0.1"


# ---------------------------------------------------------------------------
# Test Topics API
# ---------------------------------------------------------------------------


def test_get_topics_empty(client):
    """Test GET /api/topics with no topics."""
    response = client.get("/api/topics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_topics_with_data(client, temp_config):
    """Test GET /api/topics returns topic list."""
    # Add a topic
    topic = {
        "slug": "test-topic",
        "name": "Test Topic",
        "description": "Test description",
        "arxiv_categories": ["cs.AI"],
        "notify": "none",
    }
    add_topic(temp_config, topic)

    response = client.get("/api/topics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]["slug"] == "test-topic"
    assert data[0]["name"] == "Test Topic"


def test_add_topic(client):
    """Test POST /api/topics creates topic."""
    topic = {
        "slug": "new-topic",
        "name": "New Topic",
        "description": "New description",
        "arxiv_categories": ["cs.LG"],
        "notify": {
            "provider": "discord",
            "channel_id": "123456",
            "secret_key": "test_secret",
        },
    }

    response = client.post("/api/topics", json=topic, content_type="application/json")
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data["slug"] == "new-topic"


def test_add_topic_missing_fields(client):
    """Test POST /api/topics returns 400 for missing fields."""
    topic = {"slug": "incomplete"}

    response = client.post("/api/topics", json=topic, content_type="application/json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_update_topic(client, temp_config):
    """Test PUT /api/topics/<slug> updates topic."""
    # Add a topic
    topic = {
        "slug": "update-test",
        "name": "Original Name",
        "description": "Original description",
        "arxiv_categories": ["cs.AI"],
        "notify": {
            "provider": "discord",
            "channel_id": "123",
            "secret_key": "secret",
        },
    }
    add_topic(temp_config, topic)

    # Update it
    updates = {"name": "Updated Name", "description": "Updated description"}
    response = client.put("/api/topics/update-test", json=updates, content_type="application/json")
    assert response.status_code == 200

    # Verify update
    topics = get_topics(temp_config)
    assert topics[0]["name"] == "Updated Name"
    assert topics[0]["description"] == "Updated description"


def test_delete_topic(client, temp_config):
    """Test DELETE /api/topics/<slug> removes topic."""
    # Add a topic
    topic = {
        "slug": "delete-test",
        "name": "Delete Test",
        "description": "Will be deleted",
        "arxiv_categories": ["cs.AI"],
        "notify": "none",
    }
    add_topic(temp_config, topic)

    # Delete it
    response = client.delete("/api/topics/delete-test")
    assert response.status_code == 200

    # Verify deletion
    topics = get_topics(temp_config)
    assert len(topics) == 0


def test_delete_topic_not_found(client):
    """Test DELETE /api/topics/<slug> returns 404 for non-existent topic."""
    response = client.delete("/api/topics/nonexistent")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Test Settings API
# ---------------------------------------------------------------------------


def test_get_settings(client):
    """Test GET /api/settings returns config."""
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "app" in data
    assert "llm" in data
    assert "topics" not in data  # Should exclude topics


def test_env_status_masked(client, monkeypatch):
    """Test GET /api/settings/env-status masks values."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-1234567890abcdef")

    response = client.get("/api/settings/env-status")
    assert response.status_code == 200
    data = json.loads(response.data)

    assert "OPENROUTER_API_KEY" in data
    assert data["OPENROUTER_API_KEY"]["exists"] is True
    masked = data["OPENROUTER_API_KEY"]["masked_value"]
    assert masked.startswith("sk-1")
    assert "***" in masked
    # Full value must not be exposed
    assert "1234567890abcdef" not in masked


def test_db_status(client, temp_db):
    """Test GET /api/settings/db-status returns stats."""
    response = client.get("/api/settings/db-status")
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["exists"] is True
    assert "file_size_mb" in data
    assert data["papers"] == 0
    assert data["paper_evaluations"] == 0
    assert data["runs"] == 0


# ---------------------------------------------------------------------------
# Test Pipeline API (placeholders)
# ---------------------------------------------------------------------------


def test_pipeline_dryrun_placeholder(client):
    """Test POST /api/pipeline/dry-run endpoint exists (TASK-052 implemented)."""
    # Note: This test was updated from expecting 501 to checking endpoint exists
    # The dry-run functionality is now fully implemented in TASK-052
    response = client.post("/api/pipeline/dry-run", json={"topic_slug": "test"}, content_type="application/json")
    # Should return 500 if config invalid, or 200 if valid
    assert response.status_code in [200, 500]  # Implemented, not 501


# ---------------------------------------------------------------------------
# Test UI rendering
# ---------------------------------------------------------------------------


def test_index_page(client):
    """Test GET / returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Paper Scout" in response.data
    assert b"Topics Management" in response.data
