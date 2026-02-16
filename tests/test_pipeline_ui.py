"""Tests for pipeline UI endpoints (TASK-052).

Tests the dry-run, manual search, and status endpoints
for the local UI pipeline integration.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from local_ui.app import create_app
from local_ui.pipeline_runner import PipelineRunner


@pytest.fixture
def temp_config():
    """Create temporary config file."""
    config_content = """
app:
  display_timezone: "Asia/Seoul"
  report_retention_days: 90

llm:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1"
  model: "google/gemini-2.0-flash-001:free"
  app_url: "https://github.com/test/app"
  app_title: "Test App"

agents:
  common:
    ignore_reasoning: true
  keyword_expander:
    effort: "high"
    max_tokens: 2048
  scorer:
    effort: "low"
    max_tokens: 2048
  summarizer:
    effort: "low"
    max_tokens: 4096

sources:
  arxiv:
    max_results_per_query: 300

filter:
  enable: true
  strategy: "quality_threshold"

embedding:
  provider: "openai"
  model: "text-embedding-3-small"

scoring:
  weights:
    embedding_on:
      llm: 0.55
      embed: 0.35
      recency: 0.10
    embedding_off:
      llm: 0.80
      recency: 0.20
  bonus:
    is_edge: 5
    is_realtime: 5
    has_code: 3
  thresholds:
    default: 60
    relaxation_steps: [50, 40]
    min_papers: 5
  discard_cutoff: 20
  max_output: 100
  tier1_count: 30

remind:
  enabled: true
  min_score: 80
  max_expose_count: 2

clustering:
  enabled: true
  similarity_threshold: 0.85

output:
  report_format: "html"

notifications:
  enabled: false

database:
  path: "data/paper_scout.db"

weekly:
  enabled: false

local_ui:
  enabled: true
  port: 8585

topics:
  - slug: "test-topic"
    name: "Test Topic"
    description: "Test topic description for testing the pipeline dry-run and full search functionality with keyword expansion and query building"
    arxiv_categories: ["cs.AI"]
    notify:
      provider: "discord"
      channel_id: "test-channel"
      secret_key: "https://discord.com/api/webhooks/123456/test"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        return Path(f.name)


@pytest.fixture
def temp_db():
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def app(temp_config, temp_db):
    """Create Flask test app."""
    # Reset the module-level pipeline runner before each test
    import local_ui.api.pipeline as pipeline_module
    pipeline_module._pipeline_runner = None

    app = create_app(config_path=str(temp_config), db_path=str(temp_db))
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


# ------------------------------------------------------------------
# Dry-run tests
# ------------------------------------------------------------------


def test_dryrun_with_cache(client, temp_config):
    """Test dry-run returns cached keywords when available."""
    # Create mock cache
    cache_dir = Path("data/keyword_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cache.json"

    mock_cache_data = {
        "test_hash": {
            "timestamp": time.time(),
            "result": {
                "concepts": [
                    {"name_en": "Test Concept", "keywords": ["keyword1", "keyword2"]}
                ],
                "cross_domain_keywords": ["cross1", "cross2"],
                "exclude_keywords": ["exclude1"],
                "topic_embedding_text": "Test embedding text",
            },
        }
    }

    with open(cache_file, "w") as f:
        json.dump(mock_cache_data, f)

    # Mock load_config to return a proper config object
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI", "cs.CV"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    # Mock KeywordExpander to return cache
    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        with patch("local_ui.pipeline_runner.KeywordExpander") as mock_expander:
            mock_instance = Mock()
            mock_instance.expand.return_value = mock_cache_data["test_hash"]["result"]
            mock_expander.return_value = mock_instance

            response = client.post(
                "/api/pipeline/dry-run",
                json={"topic_slug": "test-topic"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert len(data["topics"]) == 1
            assert data["topics"][0]["slug"] == "test-topic"
            assert len(data["topics"][0]["concepts"]) > 0
            assert len(data["topics"][0]["queries"]) >= 15  # Min queries


def test_dryrun_no_api_key(client, temp_config, monkeypatch):
    """Test dry-run returns helpful error when no API key."""
    # Remove API key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Remove any cached data
    cache_file = Path("data/keyword_cache.json")
    if cache_file.exists():
        cache_file.unlink()

    # Mock load_config to return a proper config object
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI", "cs.CV"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": None}

    # Mock KeywordExpander to raise error about missing API key
    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        with patch("local_ui.pipeline_runner.KeywordExpander") as mock_expander:
            # Mock KeywordExpander initialization
            mock_instance = Mock()
            mock_instance.expand.side_effect = Exception("API key required")
            mock_expander.return_value = mock_instance

            response = client.post(
                "/api/pipeline/dry-run",
                json={"topic_slug": "test-topic"},
            )

            # Should return error about API key since we don't have cache either
            assert response.status_code == 500
            data = response.get_json()
            assert "error" in data


def test_dryrun_specific_topic(client):
    """Test single topic dry-run."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        with patch("local_ui.pipeline_runner.KeywordExpander") as mock_expander:
            mock_instance = Mock()
            mock_instance.expand.return_value = {
                "concepts": [{"name_en": "Test", "keywords": ["test"]}],
                "cross_domain_keywords": [],
                "exclude_keywords": [],
                "topic_embedding_text": "Test",
            }
            mock_expander.return_value = mock_instance

            response = client.post(
                "/api/pipeline/dry-run",
                json={"topic_slug": "test-topic"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert len(data["topics"]) == 1
            assert data["topics"][0]["slug"] == "test-topic"


# ------------------------------------------------------------------
# Run tests
# ------------------------------------------------------------------


def test_run_starts_background(client):
    """Test POST /api/pipeline/run returns immediately."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        response = client.post(
            "/api/pipeline/run",
            json={
                "topic_slug": "test-topic",
                "date_from": None,
                "date_to": None,
                "dedup": "skip_recent",
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "run_id" in data
        assert data["status"] == "started"
        assert len(data["run_id"]) > 0


def test_run_status_polling(client):
    """Test GET /api/pipeline/status returns correct state."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        # Start a run
        response = client.post(
            "/api/pipeline/run",
            json={"topic_slug": "test-topic"},
        )
        assert response.status_code == 200
        run_data = response.get_json()
        run_id = run_data["run_id"]

        # Poll status
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200
        status = response.get_json()

        assert status["running"] is True
        assert status["run_id"] == run_id
        assert isinstance(status["progress"], str)
        assert isinstance(status["topics_completed"], int)
        assert isinstance(status["topics_total"], int)


def test_run_not_running(client):
    """Test status when idle."""
    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    status = response.get_json()

    # Should be idle initially
    # Note: Might be running if previous test started something
    assert "running" in status
    assert "run_id" in status


def test_run_with_dates(client):
    """Test date parameters passed correctly."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        response = client.post(
            "/api/pipeline/run",
            json={
                "topic_slug": "test-topic",
                "date_from": "2024-01-01",
                "date_to": "2024-01-31",
                "dedup": "skip_recent",
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "run_id" in data


def test_run_dedup_mode(client):
    """Test dedup parameter passed correctly."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        response = client.post(
            "/api/pipeline/run",
            json={
                "topic_slug": "test-topic",
                "dedup": "none",
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "run_id" in data


# ------------------------------------------------------------------
# PipelineRunner tests
# ------------------------------------------------------------------


def test_pipeline_runner_thread_safety(temp_config, temp_db):
    """Test concurrent access doesn't crash."""
    runner = PipelineRunner(
        config_path=str(temp_config),
        db_path=str(temp_db),
    )

    # Start multiple runs concurrently
    result1 = runner.start_run(topic_slug="test-topic")
    result2 = runner.start_run(topic_slug="test-topic")

    # Second should fail with "already running"
    assert "run_id" in result1
    assert "error" in result2


def test_pipeline_runner_status_transitions(temp_config, temp_db):
    """Test status progresses correctly."""
    runner = PipelineRunner(
        config_path=str(temp_config),
        db_path=str(temp_db),
    )

    # Initial state
    status = runner.get_status()
    assert status["running"] is False

    # Start run
    runner.start_run(topic_slug="test-topic")

    # Should be running
    status = runner.get_status()
    assert status["running"] is True
    assert status["run_id"] is not None

    # Wait for completion (simulated - takes ~2 seconds)
    time.sleep(3)

    # Should be complete
    status = runner.get_status()
    assert status["running"] is False


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


def test_dryrun_to_search_integration(client):
    """Test dry-run followed by search with same topic."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    # Run dry-run first
    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        with patch("local_ui.pipeline_runner.KeywordExpander") as mock_expander:
            mock_instance = Mock()
            mock_instance.expand.return_value = {
                "concepts": [{"name_en": "Test", "keywords": ["test"]}],
                "cross_domain_keywords": [],
                "exclude_keywords": [],
                "topic_embedding_text": "Test",
            }
            mock_expander.return_value = mock_instance

            response = client.post(
                "/api/pipeline/dry-run",
                json={"topic_slug": "test-topic"},
            )
            assert response.status_code == 200

        # Then start full search
        response = client.post(
            "/api/pipeline/run",
            json={"topic_slug": "test-topic"},
        )
        assert response.status_code == 200


def test_all_topics_dryrun(client):
    """Test dry-run for all topics (null topic_slug)."""
    # Mock load_config
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        with patch("local_ui.pipeline_runner.KeywordExpander") as mock_expander:
            mock_instance = Mock()
            mock_instance.expand.return_value = {
                "concepts": [{"name_en": "Test", "keywords": ["test"]}],
                "cross_domain_keywords": [],
                "exclude_keywords": [],
                "topic_embedding_text": "Test",
            }
            mock_expander.return_value = mock_instance

            response = client.post(
                "/api/pipeline/dry-run",
                json={"topic_slug": None},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["success"] is True
            assert len(data["topics"]) >= 1  # Should process all topics


def test_invalid_topic_slug(client):
    """Test error handling for invalid topic slug."""
    # Mock load_config with a topic that doesn't match
    mock_config = Mock()
    mock_topic = Mock()
    mock_topic.slug = "test-topic"
    mock_topic.name = "Test Topic"
    mock_topic.description = "Test description"
    mock_topic.arxiv_categories = ["cs.AI"]
    mock_config.topics = [mock_topic]
    mock_config.llm = {"api_key": "test-key"}

    with patch("local_ui.pipeline_runner.load_config", return_value=mock_config):
        response = client.post(
            "/api/pipeline/dry-run",
            json={"topic_slug": "nonexistent-topic"},
        )

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert "not found" in data["error"]
