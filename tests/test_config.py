"""Tests for core.config module.

Covers config loading, schema validation, topic validation,
and error cases per SPEC requirements.
"""

from __future__ import annotations

import os
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

from core.config import (
    AppConfig,
    ConfigError,
    load_config,
    validate_config,
    validate_topic,
)
from core.models import NotifyConfig, TopicSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_topic(**overrides: Any) -> dict:
    """Return a minimal valid topic dict, with optional overrides."""
    base = {
        "slug": "test-topic",
        "name": "Test Topic",
        "description": "A" * 150,  # within 100-300 range
        "arxiv_categories": ["cs.AI"],
        "notify": {
            "provider": "discord",
            "channel_id": "123456",
            "secret_key": "secret",
        },
    }
    base.update(overrides)
    return base


def _make_raw_config(**overrides: Any) -> dict:
    """Return a minimal valid raw config dict, with optional overrides."""
    base: dict[str, Any] = {
        "app": {"display_timezone": "UTC"},
        "llm": {"model": "test-model"},
        "agents": {},
        "sources": {},
        "filter": {},
        "embedding": {},
        "scoring": {
            "weights": {
                "embedding_on": {"llm": 0.55, "embed": 0.35, "recency": 0.10},
                "embedding_off": {"llm": 0.80, "recency": 0.20},
            },
            "discard_cutoff": 20,
            "max_output": 100,
        },
        "remind": {},
        "clustering": {},
        "topics": [_make_topic()],
        "output": {},
        "notifications": {},
        "database": {"path": "data/test.db"},
        "weekly": {},
        "local_ui": {},
    }
    base.update(overrides)
    return base


@pytest.fixture()
def valid_config_file(tmp_path: Path) -> Path:
    """Write a valid config.yaml to a temp directory and return its path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(_make_raw_config(), default_flow_style=False),
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# 1. Valid config loads successfully
# ---------------------------------------------------------------------------


class TestValidConfig:
    """Test that a well-formed config loads without errors."""

    def test_valid_config_loads(self, valid_config_file: Path) -> None:
        result = load_config(str(valid_config_file))

        assert isinstance(result, AppConfig)
        assert len(result.topics) == 1
        assert result.topics[0].slug == "test-topic"
        assert result.topics[0].name == "Test Topic"
        assert isinstance(result.topics[0].notify, NotifyConfig)
        assert result.topics[0].notify.provider == "discord"

    def test_valid_config_returns_correct_sections(
        self, valid_config_file: Path
    ) -> None:
        result = load_config(str(valid_config_file))

        assert isinstance(result.app, dict)
        assert isinstance(result.llm, dict)
        assert result.llm["model"] == "test-model"
        assert isinstance(result.database, dict)
        assert result.database["path"] == "data/test.db"

    def test_env_var_config_path(
        self, valid_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PAPER_SCOUT_CONFIG", str(valid_config_file))
        result = load_config()

        assert isinstance(result, AppConfig)
        assert len(result.topics) == 1


# ---------------------------------------------------------------------------
# 2. Missing required section raises ConfigError
# ---------------------------------------------------------------------------


class TestMissingSections:
    """Test that missing top-level sections are caught."""

    @pytest.mark.parametrize(
        "missing_section",
        [
            "app",
            "llm",
            "agents",
            "sources",
            "filter",
            "embedding",
            "scoring",
            "remind",
            "clustering",
            "topics",
            "output",
            "notifications",
            "database",
            "weekly",
            "local_ui",
        ],
    )
    def test_missing_section_raises(self, missing_section: str) -> None:
        raw = _make_raw_config()
        del raw[missing_section]

        with pytest.raises(ConfigError, match=f"Missing required config section: '{missing_section}'"):
            validate_config(raw)


# ---------------------------------------------------------------------------
# 3. Empty topics array raises ConfigError
# ---------------------------------------------------------------------------


class TestEmptyTopics:
    """Test that an empty topics list is rejected."""

    def test_empty_topics_raises(self) -> None:
        raw = _make_raw_config(topics=[])

        with pytest.raises(ConfigError, match="topics: must be a non-empty list"):
            validate_config(raw)


# ---------------------------------------------------------------------------
# 4. Topic missing required field (slug) raises ConfigError
# ---------------------------------------------------------------------------


class TestTopicMissingSlug:
    """Test that a topic without slug is rejected."""

    def test_missing_slug_raises(self) -> None:
        topic = _make_topic()
        del topic["slug"]

        with pytest.raises(ConfigError, match="missing required field 'slug'"):
            validate_topic(topic, 0)


# ---------------------------------------------------------------------------
# 5. Topic missing required field (arxiv_categories) raises ConfigError
# ---------------------------------------------------------------------------


class TestTopicMissingArxivCategories:
    """Test that a topic without arxiv_categories is rejected."""

    def test_missing_arxiv_categories_raises(self) -> None:
        topic = _make_topic()
        del topic["arxiv_categories"]

        with pytest.raises(ConfigError, match="missing required field 'arxiv_categories'"):
            validate_topic(topic, 0)

    def test_empty_arxiv_categories_raises(self) -> None:
        topic = _make_topic(arxiv_categories=[])

        with pytest.raises(ConfigError, match="must be a non-empty list"):
            validate_topic(topic, 0)


# ---------------------------------------------------------------------------
# 6. Duplicate slugs raise ConfigError
# ---------------------------------------------------------------------------


class TestDuplicateSlugs:
    """Test that duplicate topic slugs are detected."""

    def test_duplicate_slugs_raises(self) -> None:
        topic_a = _make_topic(slug="my-topic", name="Topic A")
        topic_b = _make_topic(slug="my-topic", name="Topic B")
        raw = _make_raw_config(topics=[topic_a, topic_b])

        with pytest.raises(ConfigError, match="duplicate slug 'my-topic'"):
            validate_config(raw)


# ---------------------------------------------------------------------------
# 7. Invalid slug format raises ConfigError
# ---------------------------------------------------------------------------


class TestInvalidSlugFormat:
    """Test that slug format is enforced (ASCII + hyphens only)."""

    @pytest.mark.parametrize(
        "bad_slug",
        [
            "has spaces",
            "HAS_UPPER",
            "special!chars",
            "trailing-",
            "-leading",
            "under_score",
            "",
        ],
    )
    def test_invalid_slug_format_raises(self, bad_slug: str) -> None:
        topic = _make_topic(slug=bad_slug)

        with pytest.raises(ConfigError, match="must be lowercase ASCII and hyphens only"):
            validate_topic(topic, 0)

    @pytest.mark.parametrize(
        "good_slug",
        [
            "valid",
            "valid-topic",
            "multi-word-slug",
            "a1-b2-c3",
        ],
    )
    def test_valid_slug_format_accepted(self, good_slug: str) -> None:
        topic = _make_topic(slug=good_slug)
        result = validate_topic(topic, 0)

        assert result.slug == good_slug


# ---------------------------------------------------------------------------
# 8. Invalid notify provider raises ConfigError
# ---------------------------------------------------------------------------


class TestInvalidNotifyProvider:
    """Test that only discord/telegram are accepted as providers."""

    def test_invalid_provider_raises(self) -> None:
        topic = _make_topic(
            notify={
                "provider": "slack",
                "channel_id": "123",
                "secret_key": "secret",
            }
        )

        with pytest.raises(ConfigError, match="must be one of"):
            validate_topic(topic, 0)

    def test_missing_notify_provider_raises(self) -> None:
        topic = _make_topic(
            notify={"channel_id": "123", "secret_key": "secret"}
        )

        with pytest.raises(ConfigError, match="missing required field 'provider'"):
            validate_topic(topic, 0)


# ---------------------------------------------------------------------------
# 9. Optional fields with wrong type raise ConfigError
# ---------------------------------------------------------------------------


class TestOptionalFieldsWrongType:
    """Test that optional fields with incorrect types are rejected."""

    def test_must_concepts_en_wrong_type_raises(self) -> None:
        topic = _make_topic(must_concepts_en="not a list")

        with pytest.raises(ConfigError, match="must_concepts_en: must be a list"):
            validate_topic(topic, 0)

    def test_should_concepts_en_wrong_type_raises(self) -> None:
        topic = _make_topic(should_concepts_en=42)

        with pytest.raises(ConfigError, match="should_concepts_en: must be a list"):
            validate_topic(topic, 0)

    def test_must_not_en_wrong_type_raises(self) -> None:
        topic = _make_topic(must_not_en={"key": "val"})

        with pytest.raises(ConfigError, match="must_not_en: must be a list"):
            validate_topic(topic, 0)

    def test_must_concepts_en_non_string_element_raises(self) -> None:
        topic = _make_topic(must_concepts_en=["valid", 123])

        with pytest.raises(ConfigError, match="must_concepts_en\\[1\\]: must be a string"):
            validate_topic(topic, 0)


# ---------------------------------------------------------------------------
# 10. Valid optional fields parse correctly
# ---------------------------------------------------------------------------


class TestValidOptionalFields:
    """Test that valid optional fields are correctly parsed."""

    def test_optional_fields_parse(self) -> None:
        topic = _make_topic(
            must_concepts_en=["edge computing", "IoT"],
            should_concepts_en=["latency", "5G"],
            must_not_en=["survey", "review"],
        )
        result = validate_topic(topic, 0)

        assert result.must_concepts_en == ["edge computing", "IoT"]
        assert result.should_concepts_en == ["latency", "5G"]
        assert result.must_not_en == ["survey", "review"]


# ---------------------------------------------------------------------------
# 11. Missing optional fields default to None
# ---------------------------------------------------------------------------


class TestMissingOptionalFields:
    """Test that absent optional fields default to None."""

    def test_optional_fields_default_none(self) -> None:
        topic = _make_topic()
        result = validate_topic(topic, 0)

        assert result.must_concepts_en is None
        assert result.should_concepts_en is None
        assert result.must_not_en is None


# ---------------------------------------------------------------------------
# 12. Config file not found raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestConfigFileNotFound:
    """Test that a missing config file raises FileNotFoundError."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_missing_default_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PAPER_SCOUT_CONFIG", raising=False)

        with pytest.raises(FileNotFoundError):
            load_config()


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestScoringValidation:
    """Test scoring section validation."""

    def test_discard_cutoff_out_of_range_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["discard_cutoff"] = 150

        with pytest.raises(ConfigError, match="discard_cutoff: must be 0-100"):
            validate_config(raw)

    def test_discard_cutoff_negative_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["discard_cutoff"] = -5

        with pytest.raises(ConfigError, match="discard_cutoff: must be 0-100"):
            validate_config(raw)

    def test_max_output_zero_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["max_output"] = 0

        with pytest.raises(ConfigError, match="max_output: must be a positive integer"):
            validate_config(raw)

    def test_max_output_negative_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["max_output"] = -10

        with pytest.raises(ConfigError, match="max_output: must be a positive integer"):
            validate_config(raw)

    def test_missing_embedding_on_keys_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["weights"]["embedding_on"] = {"llm": 0.5}

        with pytest.raises(ConfigError, match="embedding_on: missing keys"):
            validate_config(raw)

    def test_missing_embedding_off_keys_raises(self) -> None:
        raw = _make_raw_config()
        raw["scoring"]["weights"]["embedding_off"] = {"llm": 0.8}

        with pytest.raises(ConfigError, match="embedding_off: missing keys"):
            validate_config(raw)


class TestLlmValidation:
    """Test llm section validation."""

    def test_empty_model_raises(self) -> None:
        raw = _make_raw_config()
        raw["llm"]["model"] = ""

        with pytest.raises(ConfigError, match="llm.model: must be a non-empty string"):
            validate_config(raw)

    def test_missing_model_raises(self) -> None:
        raw = _make_raw_config()
        del raw["llm"]["model"]

        with pytest.raises(ConfigError, match="llm.model: must be a non-empty string"):
            validate_config(raw)


class TestDatabaseValidation:
    """Test database section validation."""

    def test_non_string_path_raises(self) -> None:
        raw = _make_raw_config()
        raw["database"]["path"] = 12345

        with pytest.raises(ConfigError, match="database.path: must be a string"):
            validate_config(raw)


class TestDescriptionWarning:
    """Test that description length warning is emitted."""

    def test_short_description_warns(self) -> None:
        topic = _make_topic(description="Too short")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_topic(topic, 0)

            assert len(w) == 1
            assert "recommended length is 100-300 chars" in str(w[0].message)

    def test_long_description_warns(self) -> None:
        topic = _make_topic(description="A" * 500)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_topic(topic, 0)

            assert len(w) == 1
            assert "recommended length is 100-300 chars" in str(w[0].message)

    def test_good_description_no_warning(self) -> None:
        topic = _make_topic(description="A" * 200)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_topic(topic, 0)

            desc_warnings = [
                x for x in w if "recommended length" in str(x.message)
            ]
            assert len(desc_warnings) == 0


class TestMultipleTopics:
    """Test that multiple valid topics parse correctly."""

    def test_multiple_topics_load(self) -> None:
        topics = [
            _make_topic(slug="topic-a", name="Topic A"),
            _make_topic(slug="topic-b", name="Topic B"),
            _make_topic(slug="topic-c", name="Topic C"),
        ]
        raw = _make_raw_config(topics=topics)
        result = validate_config(raw)

        assert len(result.topics) == 3
        slugs = [t.slug for t in result.topics]
        assert slugs == ["topic-a", "topic-b", "topic-c"]


class TestNotifyFieldTypes:
    """Test that notify sub-fields must be strings."""

    def test_notify_missing_channel_id_raises(self) -> None:
        topic = _make_topic(
            notify={"provider": "discord", "secret_key": "secret"}
        )

        with pytest.raises(ConfigError, match="missing required field 'channel_id'"):
            validate_topic(topic, 0)

    def test_notify_missing_secret_key_raises(self) -> None:
        topic = _make_topic(
            notify={"provider": "discord", "channel_id": "123"}
        )

        with pytest.raises(ConfigError, match="missing required field 'secret_key'"):
            validate_topic(topic, 0)


class TestTelegramProvider:
    """Test that telegram is a valid provider."""

    def test_telegram_accepted(self) -> None:
        topic = _make_topic(
            notify={
                "provider": "telegram",
                "channel_id": "chat-id",
                "secret_key": "bot-token",
            }
        )
        result = validate_topic(topic, 0)

        assert result.notify.provider == "telegram"
