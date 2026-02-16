"""Comprehensive tests for BaseAgent.

Covers:
  - call_llm JSON parsing (dict, list, failure)
  - call_llm_raw with think block removal
  - response_format_supported flag flow
  - agent_config and prompt_version properties
  - Argument passthrough to OpenRouterClient and JsonParser
  - Common config access
  - Error propagation
  - Statelessness across multiple calls
  - Abstract method enforcement
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm.openrouter_client import OpenRouterClient, OpenRouterError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(agents: dict | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    if agents is None:
        agents = {
            "common": {"always_strip_think": True, "response_field": "content"},
            "test_agent": {
                "effort": "high",
                "max_tokens": 2048,
                "temperature": 0.3,
                "prompt_version": "test-v1",
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


class _TestableAgent(BaseAgent):
    """Concrete subclass of BaseAgent for testing.

    Prefixed with underscore to avoid pytest collection warning
    (pytest ignores classes with __init__ constructors but still warns).
    """

    @property
    def agent_name(self) -> str:
        return "test_agent"

    @property
    def agent_config_key(self) -> str:
        return "test_agent"

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"role": "user", "content": "test"}]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a mock OpenRouterClient."""
    return MagicMock(spec=OpenRouterClient)


@pytest.fixture()
def config() -> AppConfig:
    """Return a minimal AppConfig."""
    return _make_config()


@pytest.fixture()
def agent(config: AppConfig, mock_client: MagicMock) -> _TestableAgent:
    """Return a _TestableAgent with mocked client."""
    return _TestableAgent(config, mock_client, response_format_supported=False)


@pytest.fixture()
def agent_rfs(config: AppConfig, mock_client: MagicMock) -> _TestableAgent:
    """Return a _TestableAgent with response_format_supported=True."""
    return _TestableAgent(config, mock_client, response_format_supported=True)


# ==================================================================
# call_llm tests
# ==================================================================


class TestCallLlm:
    """Tests for BaseAgent.call_llm()."""

    def test_returns_parsed_dict(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm returns parsed dict on valid JSON object response."""
        mock_client.call.return_value = '{"keywords": ["ml", "nlp"]}'
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result == {"keywords": ["ml", "nlp"]}

    def test_returns_parsed_list(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm returns parsed list on valid JSON array response."""
        mock_client.call.return_value = '[{"id": 1}, {"id": 2}]'
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result == [{"id": 1}, {"id": 2}]

    def test_returns_none_on_unparseable(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm returns None when response cannot be parsed."""
        mock_client.call.return_value = "This is not JSON at all"
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result is None

    def test_returns_none_on_empty_response(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm returns None on empty string response."""
        mock_client.call.return_value = ""
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result is None

    def test_passes_response_format_to_client_false(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes response_format_supported=False to client."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        agent.call_llm(messages)

        mock_client.call.assert_called_once()
        _, kwargs = mock_client.call.call_args
        assert kwargs["response_format_supported"] is False

    def test_passes_response_format_to_client_true(
        self, agent_rfs: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes response_format_supported=True to client."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        agent_rfs.call_llm(messages)

        mock_client.call.assert_called_once()
        _, kwargs = mock_client.call.call_args
        assert kwargs["response_format_supported"] is True

    def test_passes_agent_config_to_client(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes agent_config to client."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        agent.call_llm(messages)

        _, kwargs = mock_client.call.call_args
        assert kwargs["agent_config"]["effort"] == "high"
        assert kwargs["agent_config"]["max_tokens"] == 2048
        assert kwargs["agent_config"]["temperature"] == 0.3

    def test_passes_extra_body_to_client(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes extra_body to client."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]
        extra = {"transforms": ["middle-out"]}

        agent.call_llm(messages, extra_body=extra)

        _, kwargs = mock_client.call.call_args
        assert kwargs["extra_body"] == {"transforms": ["middle-out"]}

    def test_extra_body_default_none(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes extra_body=None by default."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        agent.call_llm(messages)

        _, kwargs = mock_client.call.call_args
        assert kwargs["extra_body"] is None

    def test_passes_messages_to_client(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes messages as the first positional argument."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hello"},
        ]

        agent.call_llm(messages)

        args, _ = mock_client.call.call_args
        assert args[0] == messages

    def test_handles_think_blocks_in_json(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm strips think blocks before parsing JSON."""
        mock_client.call.return_value = (
            '<think>reasoning about stuff</think>{"result": 42}'
        )
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result == {"result": 42}

    def test_handles_unterminated_think_block(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm handles unterminated think blocks before JSON."""
        mock_client.call.return_value = (
            '<think>some reasoning that never closes {"result": 99}'
        )
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm(messages)

        assert result == {"result": 99}

    def test_batch_index_default_zero(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm defaults batch_index to 0."""
        mock_client.call.return_value = "not json"
        messages = [{"role": "user", "content": "test"}]

        # Verify by checking the debug dump filename would use index 0
        with patch.object(agent._parser, "parse", wraps=agent._parser.parse) as spy:
            agent.call_llm(messages)
            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs["batch_index"] == 0

    def test_batch_index_passthrough(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes batch_index to parser."""
        mock_client.call.return_value = "not json"
        messages = [{"role": "user", "content": "test"}]

        with patch.object(agent._parser, "parse", wraps=agent._parser.parse) as spy:
            agent.call_llm(messages, batch_index=7)
            _, kwargs = spy.call_args
            assert kwargs["batch_index"] == 7

    def test_agent_name_passed_to_parser(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm passes agent_name to parser for debug dumps."""
        mock_client.call.return_value = "not json"
        messages = [{"role": "user", "content": "test"}]

        with patch.object(agent._parser, "parse", wraps=agent._parser.parse) as spy:
            agent.call_llm(messages)
            _, kwargs = spy.call_args
            assert kwargs["agent_name"] == "test_agent"


# ==================================================================
# call_llm_raw tests
# ==================================================================


class TestCallLlmRaw:
    """Tests for BaseAgent.call_llm_raw()."""

    def test_returns_cleaned_text(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw returns text with think blocks removed."""
        mock_client.call.return_value = (
            "<think>internal reasoning</think>This is the answer."
        )
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm_raw(messages)

        assert result == "This is the answer."

    def test_handles_empty_response(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw handles empty string response."""
        mock_client.call.return_value = ""
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm_raw(messages)

        assert result == ""

    def test_handles_only_think_block(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw returns empty when response is only think block."""
        mock_client.call.return_value = "<think>just thinking</think>"
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm_raw(messages)

        assert result == ""

    def test_no_think_blocks_passthrough(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw passes through text without think blocks unchanged."""
        mock_client.call.return_value = "plain text response"
        messages = [{"role": "user", "content": "test"}]

        result = agent.call_llm_raw(messages)

        assert result == "plain text response"

    def test_passes_extra_body(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw passes extra_body to client."""
        mock_client.call.return_value = "response"
        messages = [{"role": "user", "content": "test"}]
        extra = {"custom_param": True}

        agent.call_llm_raw(messages, extra_body=extra)

        _, kwargs = mock_client.call.call_args
        assert kwargs["extra_body"] == {"custom_param": True}

    def test_passes_agent_config_to_client(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw passes agent_config to client."""
        mock_client.call.return_value = "response"
        messages = [{"role": "user", "content": "test"}]

        agent.call_llm_raw(messages)

        _, kwargs = mock_client.call.call_args
        assert kwargs["agent_config"]["effort"] == "high"

    def test_passes_response_format_to_client(
        self, agent_rfs: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm_raw passes response_format_supported to client."""
        mock_client.call.return_value = "response"
        messages = [{"role": "user", "content": "test"}]

        agent_rfs.call_llm_raw(messages)

        _, kwargs = mock_client.call.call_args
        assert kwargs["response_format_supported"] is True


# ==================================================================
# Property tests
# ==================================================================


class TestAgentProperties:
    """Tests for BaseAgent properties."""

    def test_agent_config_returns_correct_config(
        self, agent: _TestableAgent
    ) -> None:
        """agent_config returns the correct section from AppConfig."""
        cfg = agent.agent_config

        assert cfg["effort"] == "high"
        assert cfg["max_tokens"] == 2048
        assert cfg["temperature"] == 0.3
        assert cfg["prompt_version"] == "test-v1"

    def test_agent_config_returns_empty_dict_when_missing(
        self, mock_client: MagicMock
    ) -> None:
        """agent_config returns {} when agent key is not in config."""
        config = _make_config(agents={"common": {}})
        agent = _TestableAgent(config, mock_client)

        assert agent.agent_config == {}

    def test_prompt_version_returns_value(self, agent: _TestableAgent) -> None:
        """prompt_version returns the configured value."""
        assert agent.prompt_version == "test-v1"

    def test_prompt_version_returns_unknown_when_missing(
        self, mock_client: MagicMock
    ) -> None:
        """prompt_version returns 'unknown' when not configured."""
        config = _make_config(agents={"common": {}, "test_agent": {}})
        agent = _TestableAgent(config, mock_client)

        assert agent.prompt_version == "unknown"

    def test_agent_name(self, agent: _TestableAgent) -> None:
        """agent_name returns the correct value."""
        assert agent.agent_name == "test_agent"

    def test_agent_config_key(self, agent: _TestableAgent) -> None:
        """agent_config_key returns the correct value."""
        assert agent.agent_config_key == "test_agent"

    def test_common_config_accessible(self, agent: _TestableAgent) -> None:
        """Common config is accessible via _common_config."""
        assert agent._common_config["always_strip_think"] is True
        assert agent._common_config["response_field"] == "content"

    def test_common_config_empty_when_missing(
        self, mock_client: MagicMock
    ) -> None:
        """Common config defaults to empty dict when not present."""
        config = _make_config(agents={"test_agent": {"effort": "low"}})
        agent = _TestableAgent(config, mock_client)

        assert agent._common_config == {}


# ==================================================================
# Error propagation and statelessness
# ==================================================================


class TestErrorAndState:
    """Tests for error propagation and statelessness."""

    def test_client_exception_propagates(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """Exception from client.call() propagates to caller."""
        mock_client.call.side_effect = OpenRouterError("API failure", status_code=500)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(OpenRouterError, match="API failure"):
            agent.call_llm(messages)

    def test_client_exception_propagates_raw(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """Exception from client.call() propagates through call_llm_raw."""
        mock_client.call.side_effect = OpenRouterError("timeout", status_code=429)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(OpenRouterError, match="timeout"):
            agent.call_llm_raw(messages)

    def test_multiple_calls_stateless(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """Multiple calls work correctly (agent is stateless)."""
        mock_client.call.side_effect = [
            '{"call": 1}',
            '{"call": 2}',
            '{"call": 3}',
        ]
        messages = [{"role": "user", "content": "test"}]

        assert agent.call_llm(messages) == {"call": 1}
        assert agent.call_llm(messages) == {"call": 2}
        assert agent.call_llm(messages) == {"call": 3}

    def test_call_llm_then_raw_independent(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """call_llm and call_llm_raw work independently."""
        mock_client.call.side_effect = [
            '{"data": true}',
            "<think>think</think>raw text",
        ]
        messages = [{"role": "user", "content": "test"}]

        json_result = agent.call_llm(messages)
        raw_result = agent.call_llm_raw(messages)

        assert json_result == {"data": True}
        assert raw_result == "raw text"


# ==================================================================
# Abstract method enforcement
# ==================================================================


class TestAbstractEnforcement:
    """Tests that BaseAgent cannot be used directly."""

    def test_agent_name_not_implemented(self, mock_client: MagicMock) -> None:
        """BaseAgent.agent_name raises NotImplementedError."""
        config = _make_config()

        class BareAgent(BaseAgent):
            @property
            def agent_config_key(self) -> str:
                return "test"

            def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
                return []

        agent = BareAgent(config, mock_client)
        with pytest.raises(NotImplementedError):
            _ = agent.agent_name

    def test_agent_config_key_not_implemented(
        self, mock_client: MagicMock
    ) -> None:
        """BaseAgent.agent_config_key raises NotImplementedError."""
        config = _make_config()

        class BareAgent(BaseAgent):
            @property
            def agent_name(self) -> str:
                return "test"

            def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
                return []

        agent = BareAgent(config, mock_client)
        with pytest.raises(NotImplementedError):
            _ = agent.agent_config_key

    def test_build_messages_not_implemented(
        self, mock_client: MagicMock
    ) -> None:
        """BaseAgent.build_messages raises NotImplementedError."""
        config = _make_config()

        class BareAgent(BaseAgent):
            @property
            def agent_name(self) -> str:
                return "test"

            @property
            def agent_config_key(self) -> str:
                return "test"

        agent = BareAgent(config, mock_client)
        with pytest.raises(NotImplementedError):
            agent.build_messages()


# ==================================================================
# Integration: response_format_supported flow
# ==================================================================


class TestResponseFormatFlow:
    """Tests verifying response_format_supported flows correctly."""

    def test_rfs_true_to_both_client_and_parser(
        self, agent_rfs: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """response_format_supported=True flows to client and parser."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        with patch.object(
            agent_rfs._parser, "parse", wraps=agent_rfs._parser.parse
        ) as spy:
            result = agent_rfs.call_llm(messages)

            # Verify client received rfs=True
            _, client_kwargs = mock_client.call.call_args
            assert client_kwargs["response_format_supported"] is True

            # Verify parser received rfs=True
            _, parser_kwargs = spy.call_args
            assert parser_kwargs["response_format_supported"] is True

        assert result == {"ok": True}

    def test_rfs_false_to_both_client_and_parser(
        self, agent: _TestableAgent, mock_client: MagicMock
    ) -> None:
        """response_format_supported=False flows to client and parser."""
        mock_client.call.return_value = '{"ok": true}'
        messages = [{"role": "user", "content": "test"}]

        with patch.object(
            agent._parser, "parse", wraps=agent._parser.parse
        ) as spy:
            agent.call_llm(messages)

            _, client_kwargs = mock_client.call.call_args
            assert client_kwargs["response_format_supported"] is False

            _, parser_kwargs = spy.call_args
            assert parser_kwargs["response_format_supported"] is False
