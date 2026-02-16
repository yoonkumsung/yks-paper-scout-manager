"""Tests for Settings UI API endpoints.

Tests enhanced settings API with config editor, env-status validation, and comprehensive DB status.
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


@pytest.fixture
def temp_config():
    """Create temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "topics": [
                {
                    "slug": "test-topic",
                    "name": "Test Topic",
                    "description": "Test description",
                }
            ],
            "llm": {"provider": "openrouter", "model": "claude-3"},
            "scoring": {"min_score": 7},
            "thresholds": {"code_threshold": 0.5},
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "database": {"path": "data/papers.db"},
        }
        yaml.safe_dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY)")
    cursor.execute("CREATE TABLE paper_evaluations (id INTEGER PRIMARY KEY)")
    cursor.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY)")
    cursor.execute("CREATE TABLE query_stats (id INTEGER PRIMARY KEY)")
    cursor.execute("CREATE TABLE remind_tracking (id INTEGER PRIMARY KEY)")

    # Insert test data
    cursor.execute("INSERT INTO papers (id) VALUES (1)")
    cursor.execute("INSERT INTO papers (id) VALUES (2)")
    cursor.execute("INSERT INTO paper_evaluations (id) VALUES (1)")
    cursor.execute("INSERT INTO runs (id) VALUES (1)")
    cursor.execute("INSERT INTO query_stats (id) VALUES (1)")
    cursor.execute("INSERT INTO query_stats (id) VALUES (2)")
    cursor.execute("INSERT INTO query_stats (id) VALUES (3)")

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client(temp_config, temp_db):
    """Create Flask test client."""
    app = create_app()
    app.config["CONFIG_PATH"] = temp_config
    app.config["DB_PATH"] = temp_db
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_get_settings_structure(client):
    """Test GET /api/settings returns proper structure."""
    response = client.get("/api/settings")
    assert response.status_code == 200

    data = response.json
    assert "llm" in data
    assert "scoring" in data
    assert "thresholds" in data
    assert "embedding" in data
    assert "database" in data


def test_get_settings_excludes_topics(client):
    """Test GET /api/settings excludes topics section."""
    response = client.get("/api/settings")
    assert response.status_code == 200

    data = response.json
    assert "topics" not in data


def test_put_settings_partial_update(client):
    """Test PUT /api/settings supports partial updates."""
    update = {"llm": {"provider": "openrouter", "model": "gpt-4"}}

    response = client.put(
        "/api/settings",
        data=json.dumps(update),
        content_type="application/json",
    )
    assert response.status_code == 200

    # Verify update
    response = client.get("/api/settings")
    data = response.json
    assert data["llm"]["model"] == "gpt-4"
    assert "scoring" in data  # Other sections preserved


def test_put_settings_preserves_topics(client, temp_config):
    """Test PUT /api/settings preserves topics section."""
    update = {"llm": {"provider": "anthropic", "model": "claude-3"}}

    response = client.put(
        "/api/settings",
        data=json.dumps(update),
        content_type="application/json",
    )
    assert response.status_code == 200

    # Read config directly to verify topics preserved
    with open(temp_config, "r") as f:
        config = yaml.safe_load(f)

    assert "topics" in config
    assert len(config["topics"]) == 1
    assert config["topics"][0]["slug"] == "test-topic"


def test_put_settings_rejects_topics(client):
    """Test PUT /api/settings rejects updates to topics."""
    update = {"topics": [{"slug": "malicious", "name": "Malicious"}]}

    response = client.put(
        "/api/settings",
        data=json.dumps(update),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Cannot update topics" in response.json["error"]


def test_put_settings_validates_type(client):
    """Test PUT /api/settings validates section type."""
    update = {"llm": "invalid_string"}  # Should be dict

    response = client.put(
        "/api/settings",
        data=json.dumps(update),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "must be an object" in response.json["error"]


def test_env_status_masking(client):
    """Test GET /api/settings/env-status masks API keys correctly."""
    # Set test environment variable
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test1234567890"

    response = client.get("/api/settings/env-status")
    assert response.status_code == 200

    data = response.json
    assert data["OPENROUTER_API_KEY"]["exists"] is True
    assert data["OPENROUTER_API_KEY"]["masked_value"] == "sk-o***"
    assert "test1234567890" not in data["OPENROUTER_API_KEY"]["masked_value"]

    # Cleanup
    del os.environ["OPENROUTER_API_KEY"]


def test_env_status_missing_key(client):
    """Test GET /api/settings/env-status handles missing keys."""
    # Ensure key doesn't exist
    if "DISCORD_WEBHOOK_URL" in os.environ:
        del os.environ["DISCORD_WEBHOOK_URL"]

    response = client.get("/api/settings/env-status")
    assert response.status_code == 200

    data = response.json
    assert data["DISCORD_WEBHOOK_URL"]["exists"] is False
    assert data["DISCORD_WEBHOOK_URL"]["masked_value"] is None
    assert data["DISCORD_WEBHOOK_URL"]["valid"] is False


def test_env_status_validity_checks(client):
    """Test GET /api/settings/env-status validates key formats."""
    # Valid key
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-valid123"
    response = client.get("/api/settings/env-status")
    data = response.json
    assert data["OPENROUTER_API_KEY"]["valid"] is True

    # Invalid key (missing sk-or- prefix)
    os.environ["OPENROUTER_API_KEY"] = "invalid-key-123"
    response = client.get("/api/settings/env-status")
    data = response.json
    assert data["OPENROUTER_API_KEY"]["valid"] is False

    # Cleanup
    del os.environ["OPENROUTER_API_KEY"]


def test_db_status_with_data(client, temp_db):
    """Test GET /api/settings/db-status returns all table counts."""
    response = client.get("/api/settings/db-status")
    assert response.status_code == 200

    data = response.json
    assert data["exists"] is True
    assert data["papers"] == 2
    assert data["paper_evaluations"] == 1
    assert data["runs"] == 1
    assert data["query_stats"] == 3
    assert data["remind_tracking"] == 0
    assert "file_size_mb" in data
    assert "db_path" in data


def test_db_status_graceful_no_db(client):
    """Test GET /api/settings/db-status handles missing database gracefully."""
    # Override with non-existent path
    client.application.config["DB_PATH"] = "/tmp/nonexistent.db"

    response = client.get("/api/settings/db-status")
    assert response.status_code == 200

    data = response.json
    assert data["exists"] is False
    assert data["file_size_mb"] == 0
    assert data["papers"] == 0
    assert data["paper_evaluations"] == 0
