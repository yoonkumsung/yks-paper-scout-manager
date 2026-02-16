"""Tests for TASK-051: Topic Management UI - API enhancements and validation."""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from local_ui.app import create_app


@pytest.fixture
def temp_config():
    """Create temporary config file with test topics."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "topics": [
                {
                    "slug": "test-topic",
                    "name": "Test Topic",
                    "description": "Test description",
                    "arxiv_categories": ["cs.AI"],
                    "notify": {
                        "provider": "discord",
                        "channel_id": "123",
                        "secret_key": "secret",
                    },
                },
                {
                    "slug": "optional-fields-topic",
                    "name": "Optional Fields Topic",
                    "description": "Has optional fields",
                    "arxiv_categories": ["cs.LG"],
                    "must_concepts_en": ["neural", "network"],
                    "should_concepts_en": ["transformer"],
                    "must_not_en": ["survey"],
                    "notify": {
                        "provider": "telegram",
                        "channel_id": "456",
                        "secret_key": "secret2",
                    },
                },
            ]
        }
        yaml.dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def temp_db():
    """Create temporary database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Create database schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY,
            topic_slug TEXT,
            display_date_kst TEXT,
            total_collected INTEGER,
            total_output INTEGER
        )
    """
    )

    # Insert test data
    cursor.execute(
        """
        INSERT INTO runs (run_id, topic_slug, display_date_kst, total_collected, total_output)
        VALUES (1, 'test-topic', '2024-01-15', 100, 50)
    """
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with cache files."""
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, "keyword_cache")
    os.makedirs(cache_dir)

    # Create test cache file
    cache_file = os.path.join(cache_dir, "test-topic_keywords.json")
    with open(cache_file, "w") as f:
        json.dump({"keywords": ["test"]}, f)

    # Set modification time to recent (within expiry)
    os.utime(cache_file, None)

    yield temp_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def app(temp_config, temp_db, temp_data_dir):
    """Create Flask test app with temporary files."""
    app = create_app(config_path=temp_config, db_path=temp_db, data_path=temp_data_dir)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


def test_get_topics_with_cache_status(client):
    """Test GET /api/topics includes cache_status field."""
    response = client.get("/api/topics")
    assert response.status_code == 200

    topics = response.get_json()
    assert len(topics) > 0

    # Check first topic has cache_status
    topic = topics[0]
    assert "cache_status" in topic
    assert "exists" in topic["cache_status"]
    assert "expires_in_days" in topic["cache_status"]


def test_get_topics_with_run_stats(client):
    """Test GET /api/topics includes last_run_stats field."""
    response = client.get("/api/topics")
    assert response.status_code == 200

    topics = response.get_json()
    assert len(topics) > 0

    # Find test-topic which has run stats
    test_topic = next((t for t in topics if t["slug"] == "test-topic"), None)
    assert test_topic is not None

    # Check run stats
    assert "last_run_stats" in test_topic
    stats = test_topic["last_run_stats"]
    assert "last_run_date" in stats
    assert "total_collected" in stats
    assert "total_output" in stats

    # Verify values from database
    assert stats["last_run_date"] == "2024-01-15"
    assert stats["total_collected"] == 100
    assert stats["total_output"] == 50


def test_add_topic_validation_slug_format(client):
    """Test POST /api/topics rejects invalid slug format."""
    # Test uppercase
    response = client.post(
        "/api/topics",
        json={
            "slug": "Invalid-Slug",
            "name": "Test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400
    error = response.get_json()
    assert "slug" in error["error"].lower()

    # Test spaces
    response = client.post(
        "/api/topics",
        json={
            "slug": "invalid slug",
            "name": "Test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400

    # Test special characters
    response = client.post(
        "/api/topics",
        json={
            "slug": "invalid_slug!",
            "name": "Test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400


def test_add_topic_validation_duplicate_slug(client):
    """Test POST /api/topics rejects duplicate slug."""
    response = client.post(
        "/api/topics",
        json={
            "slug": "test-topic",  # Already exists
            "name": "Duplicate",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400
    error = response.get_json()
    assert "already exists" in error["error"].lower()


def test_add_topic_validation_missing_fields(client):
    """Test POST /api/topics validates required fields."""
    # Missing slug
    response = client.post(
        "/api/topics",
        json={
            "name": "Test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400

    # Missing name
    response = client.post(
        "/api/topics",
        json={
            "slug": "test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400

    # Missing description
    response = client.post(
        "/api/topics",
        json={
            "slug": "test",
            "name": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400

    # Empty arxiv_categories
    response = client.post(
        "/api/topics",
        json={
            "slug": "test",
            "name": "Test",
            "description": "Test",
            "arxiv_categories": [],
            "notify": {"provider": "discord", "channel_id": "123", "secret_key": "key"},
        },
    )
    assert response.status_code == 400

    # Missing notify fields
    response = client.post(
        "/api/topics",
        json={
            "slug": "test",
            "name": "Test",
            "description": "Test",
            "arxiv_categories": ["cs.AI"],
            "notify": {"provider": "discord"},  # Missing channel_id and secret_key
        },
    )
    assert response.status_code == 400


def test_add_topic_success(client):
    """Test POST /api/topics creates valid topic."""
    response = client.post(
        "/api/topics",
        json={
            "slug": "new-topic",
            "name": "New Topic",
            "description": "New description",
            "arxiv_categories": ["cs.AI", "cs.LG"],
            "notify": {"provider": "telegram", "channel_id": "999", "secret_key": "newsecret"},
        },
    )
    assert response.status_code == 201

    # Verify topic was created
    response = client.get("/api/topics")
    topics = response.get_json()
    new_topic = next((t for t in topics if t["slug"] == "new-topic"), None)
    assert new_topic is not None
    assert new_topic["name"] == "New Topic"


def test_update_topic_description_change_notice(client):
    """Test PUT /api/topics/<slug> returns cache_invalidated flag on description change."""
    response = client.put(
        "/api/topics/test-topic",
        json={
            "description": "Updated description",  # Different from original
        },
    )
    assert response.status_code == 200

    result = response.get_json()
    assert "cache_invalidated" in result
    assert result["cache_invalidated"] is True


def test_update_topic_no_description_change(client):
    """Test PUT /api/topics/<slug> no cache_invalidated flag when description unchanged."""
    # Get current description
    response = client.get("/api/topics")
    topics = response.get_json()
    test_topic = next((t for t in topics if t["slug"] == "test-topic"), None)
    original_description = test_topic["description"]

    # Update with same description
    response = client.put(
        "/api/topics/test-topic",
        json={
            "name": "Updated Name",  # Change name but not description
            "description": original_description,
        },
    )
    assert response.status_code == 200

    result = response.get_json()
    assert "cache_invalidated" not in result


def test_delete_topic_success(client):
    """Test DELETE /api/topics/<slug> removes topic."""
    response = client.delete("/api/topics/test-topic")
    assert response.status_code == 200

    # Verify topic was removed
    response = client.get("/api/topics")
    topics = response.get_json()
    assert not any(t["slug"] == "test-topic" for t in topics)


def test_optional_fields_in_response(client):
    """Test optional fields shown only when set."""
    response = client.get("/api/topics")
    topics = response.get_json()

    # Topic without optional fields should not have them
    test_topic = next((t for t in topics if t["slug"] == "test-topic"), None)
    assert test_topic is not None
    assert "must_concepts_en" not in test_topic
    assert "should_concepts_en" not in test_topic
    assert "must_not_en" not in test_topic

    # Topic with optional fields should have them (but not empty ones)
    optional_topic = next((t for t in topics if t["slug"] == "optional-fields-topic"), None)
    assert optional_topic is not None
    assert "must_concepts_en" in optional_topic
    assert "should_concepts_en" in optional_topic
    assert "must_not_en" in optional_topic
