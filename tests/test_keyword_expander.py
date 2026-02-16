"""Comprehensive tests for KeywordExpander (Agent 1).

Covers:
  - expand() happy path and cache interactions
  - Cache hit/miss/expiry behavior
  - Cache key sensitivity to field changes
  - build_messages() content for various TopicSpec configurations
  - _merge_user_constraints() for must/should/must_not fields
  - Fallback on 2x parse failure
  - _validate_output() structural checks
  - agent_name, prompt_version properties
  - Cache file creation and persistence
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.keyword_expander import KeywordExpander
from core.config import AppConfig
from core.models import NotifyConfig, TopicSpec


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(agents: dict | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    if agents is None:
        agents = {
            "common": {"ignore_reasoning": True},
            "keyword_expander": {
                "effort": "high",
                "max_tokens": 2048,
                "temperature": 0.3,
                "prompt_version": "agent1-v3",
                "cache_ttl_days": 30,
            },
        }

    return AppConfig(
        app={},
        llm={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "test/model:free",
            "app_url": "https://github.com/test",
            "app_title": "Test",
            "retry": {"max_retries": 1, "backoff_base": 0, "jitter": False},
        },
        agents=agents,
        sources={},
        filter={},
        embedding={},
        scoring={
            "weights": {
                "embedding_on": {"llm": 0.5, "embed": 0.3, "recency": 0.2},
                "embedding_off": {"llm": 0.8, "recency": 0.2},
            },
            "discard_cutoff": 20,
            "max_output": 100,
        },
        remind={},
        clustering={},
        topics=[],
        output={},
        notifications={},
        database={"path": ":memory:"},
        weekly={},
        local_ui={},
    )


def _make_topic(
    *,
    description: str = "Automatic cinematography for sports broadcasting using deep learning.",
    arxiv_categories: list[str] | None = None,
    must_concepts_en: list[str] | None = None,
    should_concepts_en: list[str] | None = None,
    must_not_en: list[str] | None = None,
) -> TopicSpec:
    """Create a TopicSpec for testing."""
    return TopicSpec(
        slug="auto-camera",
        name="Auto Camera",
        description=description,
        arxiv_categories=arxiv_categories or ["cs.CV", "cs.AI"],
        notify=NotifyConfig(provider="discord", channel_id="123", secret_key="secret"),
        must_concepts_en=must_concepts_en,
        should_concepts_en=should_concepts_en,
        must_not_en=must_not_en,
    )


def _valid_llm_result() -> dict:
    """Return a valid LLM output dict."""
    return {
        "concepts": [
            {
                "name_ko": "automatic cinematography",
                "name_en": "automatic cinematography",
                "keywords": ["camera selection", "view planning"],
            }
        ],
        "cross_domain_keywords": ["neural radiance field"],
        "exclude_keywords": ["medical imaging"],
        "topic_embedding_text": (
            "Automatic cinematography and camera selection for sports broadcasting "
            "using deep learning techniques."
        ),
    }


def _make_expander(
    tmp_path, config: AppConfig | None = None
) -> KeywordExpander:
    """Instantiate a KeywordExpander with mocked client and tmp cache file."""
    cfg = config or _make_config()
    client = MagicMock()
    expander = KeywordExpander(cfg, client)
    # Point cache to tmp_path
    expander.CACHE_FILE = str(tmp_path / "keyword_cache.json")
    return expander


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestExpandHappyPath:
    """Test expand() returns valid result on successful LLM call."""

    def test_expand_returns_valid_result(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        result = _valid_llm_result()

        with patch.object(expander, "call_llm", return_value=result):
            output = expander.expand(topic)

        assert "concepts" in output
        assert "cross_domain_keywords" in output
        assert "exclude_keywords" in output
        assert "topic_embedding_text" in output
        assert isinstance(output["concepts"], list)


class TestCacheHit:
    """Test cache hit returns cached result (no LLM call)."""

    def test_cache_hit_no_llm_call(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        cached_result = _valid_llm_result()

        # Pre-populate cache
        cache_key = expander._compute_cache_key(topic)
        cache_data = {
            cache_key: {
                "result": cached_result,
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "prompt_version": "agent1-v3",
            }
        }
        with open(expander.CACHE_FILE, "w") as fh:
            json.dump(cache_data, fh)

        with patch.object(expander, "call_llm") as mock_llm:
            output = expander.expand(topic)

        mock_llm.assert_not_called()
        assert output == cached_result


class TestCacheMiss:
    """Test cache miss triggers LLM call."""

    def test_cache_miss_calls_llm(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        result = _valid_llm_result()

        with patch.object(expander, "call_llm", return_value=result):
            output = expander.expand(topic)

        assert output["concepts"][0]["name_en"] == "automatic cinematography"


class TestCacheExpired:
    """Test expired cache triggers new LLM call."""

    def test_expired_cache_triggers_llm_call(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        old_result = _valid_llm_result()
        old_result["topic_embedding_text"] = "old text"
        new_result = _valid_llm_result()
        new_result["topic_embedding_text"] = "new text"

        # Pre-populate with expired cache (31 days ago)
        cache_key = expander._compute_cache_key(topic)
        expired_time = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        cache_data = {
            cache_key: {
                "result": old_result,
                "cached_at": expired_time,
                "prompt_version": "agent1-v3",
            }
        }
        with open(expander.CACHE_FILE, "w") as fh:
            json.dump(cache_data, fh)

        with patch.object(expander, "call_llm", return_value=new_result):
            output = expander.expand(topic)

        assert output["topic_embedding_text"] == "new text"


class TestCacheKeySensitivity:
    """Test cache key changes when input fields change."""

    def test_cache_key_changes_when_description_changes(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic1 = _make_topic(description="Description A")
        topic2 = _make_topic(description="Description B")

        key1 = expander._compute_cache_key(topic1)
        key2 = expander._compute_cache_key(topic2)

        assert key1 != key2

    def test_cache_key_changes_when_must_concepts_changes(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic1 = _make_topic(must_concepts_en=["concept A"])
        topic2 = _make_topic(must_concepts_en=["concept B"])

        key1 = expander._compute_cache_key(topic1)
        key2 = expander._compute_cache_key(topic2)

        assert key1 != key2

    def test_cache_key_changes_when_should_concepts_changes(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic1 = _make_topic(should_concepts_en=["kw1"])
        topic2 = _make_topic(should_concepts_en=["kw2"])

        key1 = expander._compute_cache_key(topic1)
        key2 = expander._compute_cache_key(topic2)

        assert key1 != key2

    def test_cache_key_changes_when_must_not_changes(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic1 = _make_topic(must_not_en=["exclude1"])
        topic2 = _make_topic(must_not_en=["exclude2"])

        key1 = expander._compute_cache_key(topic1)
        key2 = expander._compute_cache_key(topic2)

        assert key1 != key2


class TestCacheSaved:
    """Test cache is saved after successful LLM call."""

    def test_cache_saved_after_successful_call(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        result = _valid_llm_result()

        with patch.object(expander, "call_llm", return_value=result):
            expander.expand(topic)

        assert os.path.isfile(expander.CACHE_FILE)
        with open(expander.CACHE_FILE, "r") as fh:
            saved = json.load(fh)

        cache_key = expander._compute_cache_key(topic)
        assert cache_key in saved
        assert saved[cache_key]["prompt_version"] == "agent1-v3"
        assert "cached_at" in saved[cache_key]


class TestBuildMessages:
    """Test build_messages() includes the correct fields."""

    def test_includes_description(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(description="My special research topic")
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "My special research topic" in user_msg

    def test_includes_arxiv_categories(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(arxiv_categories=["cs.CV", "cs.AI"])
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "cs.CV" in user_msg
        assert "cs.AI" in user_msg

    def test_includes_must_concepts_en(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(must_concepts_en=["camera selection", "view planning"])
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "camera selection" in user_msg
        assert "MUST be included" in user_msg

    def test_includes_should_concepts_en(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(should_concepts_en=["neural radiance field"])
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "neural radiance field" in user_msg
        assert "cross_domain_keywords" in user_msg

    def test_includes_must_not_en(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(must_not_en=["medical imaging"])
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "medical imaging" in user_msg
        assert "exclude_keywords" in user_msg

    def test_omits_optional_fields_when_none(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()  # all optional fields are None
        messages = expander.build_messages(topic=topic)

        user_msg = messages[1]["content"]
        assert "MUST be included" not in user_msg
        assert "Merge these into cross_domain_keywords" not in user_msg
        assert "Merge these into exclude_keywords" not in user_msg

    def test_system_prompt_present(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        messages = expander.build_messages(topic=topic)

        assert messages[0]["role"] == "system"
        assert "expert research librarian" in messages[0]["content"]


class TestMergeUserConstraints:
    """Test _merge_user_constraints() merging logic."""

    def test_adds_must_concepts_to_concepts(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        topic = _make_topic(must_concepts_en=["novel concept X"])

        merged = expander._merge_user_constraints(result, topic)

        concept_names = [c["name_en"] for c in merged["concepts"]]
        assert "novel concept X" in concept_names

    def test_does_not_duplicate_existing_must_concepts(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        topic = _make_topic(must_concepts_en=["automatic cinematography"])

        merged = expander._merge_user_constraints(result, topic)

        concept_names = [c["name_en"] for c in merged["concepts"]]
        assert concept_names.count("automatic cinematography") == 1

    def test_adds_should_concepts_to_cross_domain(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        topic = _make_topic(should_concepts_en=["attention mechanism"])

        merged = expander._merge_user_constraints(result, topic)

        assert "attention mechanism" in merged["cross_domain_keywords"]

    def test_adds_must_not_to_exclude_keywords(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        topic = _make_topic(must_not_en=["satellite"])

        merged = expander._merge_user_constraints(result, topic)

        assert "satellite" in merged["exclude_keywords"]

    def test_handles_none_optional_fields(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        topic = _make_topic()  # all optional fields None

        merged = expander._merge_user_constraints(result, topic)

        # Should not change the result
        assert merged == result


class TestFallback:
    """Test fallback on 2x parse failure."""

    def test_fallback_on_2x_parse_failure(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(arxiv_categories=["cs.CV", "cs.AI"])

        with patch.object(expander, "call_llm", return_value=None):
            output = expander.expand(topic)

        assert len(output["concepts"]) == 2
        assert output["concepts"][0]["name_en"] == "cs.CV"
        assert output["concepts"][1]["name_en"] == "cs.AI"
        assert output["cross_domain_keywords"] == []
        assert output["topic_embedding_text"] == "cs.CV cs.AI"

    def test_fallback_includes_must_not_in_exclude(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(must_not_en=["medical imaging", "satellite"])

        with patch.object(expander, "call_llm", return_value=None):
            output = expander.expand(topic)

        assert "medical imaging" in output["exclude_keywords"]
        assert "satellite" in output["exclude_keywords"]

    def test_fallback_with_invalid_output_structure(self, tmp_path):
        """LLM returns a dict missing required keys -- triggers fallback."""
        expander = _make_expander(tmp_path)
        topic = _make_topic(arxiv_categories=["cs.LG"])

        bad_result = {"concepts": "not a list"}
        with patch.object(expander, "call_llm", return_value=bad_result):
            output = expander.expand(topic)

        # Should fallback
        assert output["concepts"][0]["name_en"] == "cs.LG"


class TestValidateOutput:
    """Test _validate_output() checks required keys and structure."""

    def test_valid_output(self, tmp_path):
        expander = _make_expander(tmp_path)
        assert expander._validate_output(_valid_llm_result()) is True

    def test_missing_key(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        del result["concepts"]
        assert expander._validate_output(result) is False

    def test_concepts_not_list(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        result["concepts"] = "not a list"
        assert expander._validate_output(result) is False

    def test_concept_missing_name_en(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        result["concepts"] = [{"name_ko": "test", "keywords": ["k"]}]
        assert expander._validate_output(result) is False

    def test_concept_missing_keywords(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        result["concepts"] = [{"name_ko": "test", "name_en": "test"}]
        assert expander._validate_output(result) is False

    def test_non_dict_input(self, tmp_path):
        expander = _make_expander(tmp_path)
        assert expander._validate_output([1, 2, 3]) is False

    def test_cross_domain_not_list(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        result["cross_domain_keywords"] = "not a list"
        assert expander._validate_output(result) is False

    def test_topic_embedding_text_not_string(self, tmp_path):
        expander = _make_expander(tmp_path)
        result = _valid_llm_result()
        result["topic_embedding_text"] = 123
        assert expander._validate_output(result) is False


class TestAgentProperties:
    """Test agent_name and prompt_version."""

    def test_agent_name(self, tmp_path):
        expander = _make_expander(tmp_path)
        assert expander.agent_name == "keyword_expander"

    def test_agent_config_key(self, tmp_path):
        expander = _make_expander(tmp_path)
        assert expander.agent_config_key == "keyword_expander"

    def test_prompt_version_from_config(self, tmp_path):
        expander = _make_expander(tmp_path)
        assert expander.prompt_version == "agent1-v3"


class TestCacheFileLocation:
    """Test cache file is created in correct location."""

    def test_cache_file_created(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        result = _valid_llm_result()

        with patch.object(expander, "call_llm", return_value=result):
            expander.expand(topic)

        assert os.path.isfile(expander.CACHE_FILE)

    def test_cache_file_in_subdirectory(self, tmp_path):
        expander = _make_expander(tmp_path)
        expander.CACHE_FILE = str(tmp_path / "sub" / "dir" / "cache.json")
        topic = _make_topic()
        result = _valid_llm_result()

        with patch.object(expander, "call_llm", return_value=result):
            expander.expand(topic)

        assert os.path.isfile(expander.CACHE_FILE)


class TestCacheCorruption:
    """Test behavior when cache file is corrupt."""

    def test_corrupt_cache_returns_empty(self, tmp_path):
        expander = _make_expander(tmp_path)

        # Write invalid JSON
        with open(expander.CACHE_FILE, "w") as fh:
            fh.write("not json{{{")

        cache = expander._load_cache()
        assert cache == {}

    def test_cache_non_dict_returns_empty(self, tmp_path):
        expander = _make_expander(tmp_path)

        with open(expander.CACHE_FILE, "w") as fh:
            json.dump([1, 2, 3], fh)

        cache = expander._load_cache()
        assert cache == {}


class TestCacheValidity:
    """Test _is_cache_valid() with various timestamps."""

    def test_valid_entry(self, tmp_path):
        expander = _make_expander(tmp_path)
        entry = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "result": {},
        }
        assert expander._is_cache_valid(entry) is True

    def test_expired_entry(self, tmp_path):
        expander = _make_expander(tmp_path)
        old_time = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        entry = {"cached_at": old_time, "result": {}}
        assert expander._is_cache_valid(entry) is False

    def test_missing_cached_at(self, tmp_path):
        expander = _make_expander(tmp_path)
        entry = {"result": {}}
        assert expander._is_cache_valid(entry) is False

    def test_invalid_timestamp(self, tmp_path):
        expander = _make_expander(tmp_path)
        entry = {"cached_at": "not-a-date", "result": {}}
        assert expander._is_cache_valid(entry) is False

    def test_naive_timestamp_treated_as_utc(self, tmp_path):
        """Naive datetime (no timezone) should be treated as UTC."""
        expander = _make_expander(tmp_path)
        # Recent naive timestamp
        recent = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        entry = {"cached_at": recent, "result": {}}
        assert expander._is_cache_valid(entry) is True


class TestExpandWithFirstFailureThenSuccess:
    """Test that expand retries on first parse failure."""

    def test_first_failure_second_success(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()
        good = _valid_llm_result()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # first call fails
            return good  # second call succeeds

        with patch.object(expander, "call_llm", side_effect=side_effect):
            output = expander.expand(topic)

        assert call_count == 2
        assert output["concepts"][0]["name_en"] == "automatic cinematography"


class TestFallbackNotCached:
    """Test that fallback results are not saved to cache."""

    def test_fallback_not_cached(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()

        with patch.object(expander, "call_llm", return_value=None):
            expander.expand(topic)

        # Cache file should not exist (or be empty)
        if os.path.isfile(expander.CACHE_FILE):
            with open(expander.CACHE_FILE, "r") as fh:
                data = json.load(fh)
            assert len(data) == 0
        # else: file doesn't exist, which is also correct


class TestCacheKeyDeterministic:
    """Test that cache key is deterministic for same input."""

    def test_same_input_same_key(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic(
            description="Same description",
            must_concepts_en=["a", "b"],
        )

        key1 = expander._compute_cache_key(topic)
        key2 = expander._compute_cache_key(topic)

        assert key1 == key2

    def test_key_is_sha256_hex(self, tmp_path):
        expander = _make_expander(tmp_path)
        topic = _make_topic()

        key = expander._compute_cache_key(topic)

        # SHA-256 hex digest is 64 characters
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)
