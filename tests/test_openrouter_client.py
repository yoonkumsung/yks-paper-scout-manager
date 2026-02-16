"""Comprehensive tests for OpenRouterClient.

Covers:
  - Successful API calls and content extraction
  - Header configuration (HTTP-Referer, X-Title)
  - response_format passthrough
  - extra_body merging
  - Agent config (effort, max_tokens, temperature)
  - Retry on 429 (rate-limit) and 5xx (server errors)
  - OpenRouterError after retries exhausted
  - API key validation (check_api_key)
  - Response-format probing (probe_response_format)
  - Edge cases (empty content, connection errors)
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.config import AppConfig
from core.llm.openrouter_client import OpenRouterClient, OpenRouterError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(**overrides: Any) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    llm = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "test/model:free",
        "app_url": "https://github.com/user/paper-scout",
        "app_title": "Paper Scout",
        "retry": {
            "max_retries": 3,
            "backoff_base": 0,  # zero for fast tests
            "jitter": False,
        },
    }
    llm.update(overrides)

    # Dummy values for all required AppConfig fields
    return AppConfig(
        app={},
        llm=llm,
        agents={"common": {"strip_think_blocks": True}},
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


def _mock_response(content: str = '{"ok": true}') -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


SAMPLE_MESSAGES = [{"role": "user", "content": "Hello"}]
SAMPLE_AGENT_CFG: dict[str, Any] = {
    "max_tokens": 2048,
    "temperature": 0.3,
}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure OPENROUTER_API_KEY is always set in test env."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")


@pytest.fixture()
def client() -> OpenRouterClient:
    """Return an OpenRouterClient with fast-retry config."""
    return OpenRouterClient(_make_config())


# ------------------------------------------------------------------
# 1. Successful call
# ------------------------------------------------------------------


class TestSuccessfulCall:
    def test_returns_content_string(self, client: OpenRouterClient) -> None:
        mock_resp = _mock_response('{"score": 42}')
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ):
            result = client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        assert result == '{"score": 42}'

    def test_returns_empty_string_when_content_is_none(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response(content="")
        mock_resp.choices[0].message.content = None
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ):
            result = client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        assert result == ""


# ------------------------------------------------------------------
# 2. Headers
# ------------------------------------------------------------------


class TestHeaders:
    def test_default_headers_contain_referer_and_title(self) -> None:
        cfg = _make_config()
        with patch("core.llm.openrouter_client.OpenAI") as MockOpenAI:
            OpenRouterClient(cfg)
            call_kwargs = MockOpenAI.call_args
            headers = call_kwargs.kwargs.get(
                "default_headers"
            ) or call_kwargs[1].get("default_headers")

        assert headers["HTTP-Referer"] == cfg.llm["app_url"]
        assert headers["X-Title"] == cfg.llm["app_title"]

    def test_custom_app_url_and_title(self) -> None:
        cfg = _make_config(
            app_url="https://custom.example.com",
            app_title="Custom App",
        )
        with patch("core.llm.openrouter_client.OpenAI") as MockOpenAI:
            OpenRouterClient(cfg)
            call_kwargs = MockOpenAI.call_args
            headers = call_kwargs.kwargs.get(
                "default_headers"
            ) or call_kwargs[1].get("default_headers")

        assert headers["HTTP-Referer"] == "https://custom.example.com"
        assert headers["X-Title"] == "Custom App"


# ------------------------------------------------------------------
# 3. response_format
# ------------------------------------------------------------------


class TestResponseFormat:
    def test_adds_response_format_when_supported(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config=SAMPLE_AGENT_CFG,
                response_format_supported=True,
            )
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_no_response_format_when_not_supported(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config=SAMPLE_AGENT_CFG,
                response_format_supported=False,
            )
        call_kwargs = mock_create.call_args.kwargs
        assert "response_format" not in call_kwargs


# ------------------------------------------------------------------
# 4. extra_body merging
# ------------------------------------------------------------------


class TestExtraBody:
    def test_extra_body_passed_through(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        extra = {"reasoning": {"max_tokens": 4096}}
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config=SAMPLE_AGENT_CFG,
                extra_body=extra,
            )
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["extra_body"]["reasoning"]["max_tokens"] == 4096

    def test_extra_body_none_produces_no_extra_body_key(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        agent_cfg = {"max_tokens": 512, "temperature": 0.5}
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config=agent_cfg,
                extra_body=None,
            )
        call_kwargs = mock_create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    def test_extra_body_merged_with_effort(
        self, client: OpenRouterClient
    ) -> None:
        """When both extra_body and agent effort exist, they merge."""
        mock_resp = _mock_response()
        extra = {"transforms": ["middle-out"]}
        agent_cfg = {
            "max_tokens": 512,
            "temperature": 0.3,
            "effort": "high",
        }
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config=agent_cfg,
                extra_body=extra,
            )
        call_kwargs = mock_create.call_args.kwargs
        eb = call_kwargs["extra_body"]
        assert eb["transforms"] == ["middle-out"]
        assert eb["reasoning"]["effort"] == "high"

    def test_original_extra_body_dict_not_mutated(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        extra = {"key": "value"}
        original_copy = dict(extra)
        agent_cfg = {
            "max_tokens": 512,
            "temperature": 0.3,
            "effort": "low",
        }
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ):
            client.call(
                SAMPLE_MESSAGES,
                agent_config=agent_cfg,
                extra_body=extra,
            )
        assert extra == original_copy


# ------------------------------------------------------------------
# 5. Retry on 429
# ------------------------------------------------------------------


class TestRetry429:
    def test_retries_on_rate_limit_then_succeeds(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        rate_error = _openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_resp = _mock_response('{"retried": true}')

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[rate_error, mock_resp],
        ):
            result = client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        assert result == '{"retried": true}'

    def test_raises_after_all_429_retries_exhausted(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        rate_error = _openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[rate_error, rate_error, rate_error],
        ):
            with pytest.raises(OpenRouterError, match="retries exhausted"):
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )


# ------------------------------------------------------------------
# 6. Retry on 5xx
# ------------------------------------------------------------------


class TestRetry5xx:
    def test_retries_on_500_then_succeeds(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 500
        resp_mock.headers = {}
        server_error = _openai.APIStatusError(
            message="internal error",
            response=resp_mock,
            body=None,
        )
        mock_resp = _mock_response('{"ok": true}')

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[server_error, mock_resp],
        ):
            result = client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        assert result == '{"ok": true}'

    def test_retries_on_502(self, client: OpenRouterClient) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 502
        resp_mock.headers = {}
        server_error = _openai.APIStatusError(
            message="bad gateway",
            response=resp_mock,
            body=None,
        )
        mock_resp = _mock_response()

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[server_error, mock_resp],
        ):
            result = client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        assert result == '{"ok": true}'

    def test_does_not_retry_on_400(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 400
        resp_mock.headers = {}
        bad_request = _openai.APIStatusError(
            message="bad request",
            response=resp_mock,
            body=None,
        )

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=bad_request,
        ):
            with pytest.raises(OpenRouterError) as exc_info:
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )
        assert exc_info.value.status_code == 400

    def test_does_not_retry_on_401(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 401
        resp_mock.headers = {}
        unauth = _openai.APIStatusError(
            message="unauthorized",
            response=resp_mock,
            body=None,
        )

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=unauth,
        ):
            with pytest.raises(OpenRouterError) as exc_info:
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )
        assert exc_info.value.status_code == 401


# ------------------------------------------------------------------
# 7. OpenRouterError after max retries
# ------------------------------------------------------------------


class TestOpenRouterError:
    def test_error_preserves_status_code(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 503
        resp_mock.headers = {}
        error = _openai.APIStatusError(
            message="service unavailable",
            response=resp_mock,
            body=None,
        )

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[error, error, error],
        ):
            with pytest.raises(OpenRouterError) as exc_info:
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )
        assert exc_info.value.status_code == 503

    def test_error_message_contains_retry_info(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        rate_error = _openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[rate_error] * 3,
        ):
            with pytest.raises(OpenRouterError, match="3 retries"):
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )

    def test_connection_error_retries_then_raises(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        conn_error = _openai.APIConnectionError(request=MagicMock())

        with patch.object(
            client._client.chat.completions,
            "create",
            side_effect=[conn_error] * 3,
        ):
            with pytest.raises(OpenRouterError, match="retries exhausted"):
                client.call(
                    SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
                )


# ------------------------------------------------------------------
# 8. check_api_key
# ------------------------------------------------------------------


class TestCheckApiKey:
    def test_returns_key_info_dict(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"label": "test-key", "limit": 100}
        }
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "core.llm.openrouter_client._requests_lib.get",
            return_value=mock_resp,
        ) as mock_get:
            result = client.check_api_key()

        assert result == {"data": {"label": "test-key", "limit": 100}}
        mock_get.assert_called_once()

        # Verify URL ends with /v1/key
        call_args = mock_get.call_args
        url = call_args[0][0]
        assert url.endswith("/v1/key")

    def test_raises_on_network_error(
        self, client: OpenRouterClient
    ) -> None:
        with patch(
            "core.llm.openrouter_client._requests_lib.get",
            side_effect=ConnectionError("connection failed"),
        ):
            with pytest.raises(OpenRouterError, match="key validation failed"):
                client.check_api_key()

    def test_check_api_key_sends_auth_header(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "core.llm.openrouter_client._requests_lib.get",
            return_value=mock_resp,
        ) as mock_get:
            client.check_api_key()

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get(
            "headers"
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")


# ------------------------------------------------------------------
# 9. probe_response_format
# ------------------------------------------------------------------


class TestProbeResponseFormat:
    def test_returns_true_when_supported(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response('{"ok": true}')
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ):
            result = client.probe_response_format("test/model:free")
        assert result is True

    def test_returns_false_when_not_supported(
        self, client: OpenRouterClient
    ) -> None:
        import openai as _openai

        resp_mock = MagicMock()
        resp_mock.status_code = 400
        resp_mock.headers = {}
        error = _openai.APIStatusError(
            message="response_format not supported",
            response=resp_mock,
            body=None,
        )
        with patch.object(
            client._client.chat.completions, "create", side_effect=error
        ):
            result = client.probe_response_format("unsupported/model")
        assert result is False

    def test_probe_uses_json_object_response_format(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.probe_response_format("test/model")
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["model"] == "test/model"
        assert call_kwargs["max_tokens"] == 32


# ------------------------------------------------------------------
# 10. Agent config passthrough
# ------------------------------------------------------------------


class TestAgentConfig:
    def test_max_tokens_passed(self, client: OpenRouterClient) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config={"max_tokens": 4096, "temperature": 0.5},
            )
        kw = mock_create.call_args.kwargs
        assert kw["max_tokens"] == 4096

    def test_temperature_passed(self, client: OpenRouterClient) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config={"max_tokens": 1024, "temperature": 0.7},
            )
        kw = mock_create.call_args.kwargs
        assert kw["temperature"] == 0.7

    def test_effort_maps_to_reasoning_extra_body(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config={
                    "max_tokens": 1024,
                    "temperature": 0.2,
                    "effort": "high",
                },
            )
        kw = mock_create.call_args.kwargs
        assert kw["extra_body"]["reasoning"]["effort"] == "high"

    def test_effort_low(self, client: OpenRouterClient) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES,
                agent_config={
                    "max_tokens": 1024,
                    "temperature": 0.2,
                    "effort": "low",
                },
            )
        kw = mock_create.call_args.kwargs
        assert kw["extra_body"]["reasoning"]["effort"] == "low"

    def test_minimal_agent_config_no_optional_keys(
        self, client: OpenRouterClient
    ) -> None:
        """Agent config with no max_tokens/temperature/effort."""
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(SAMPLE_MESSAGES, agent_config={})
        kw = mock_create.call_args.kwargs
        assert "max_tokens" not in kw
        assert "temperature" not in kw
        assert "extra_body" not in kw


# ------------------------------------------------------------------
# 11. Model and messages passthrough
# ------------------------------------------------------------------


class TestModelAndMessages:
    def test_model_from_config(self, client: OpenRouterClient) -> None:
        mock_resp = _mock_response()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(
                SAMPLE_MESSAGES, agent_config=SAMPLE_AGENT_CFG
            )
        kw = mock_create.call_args.kwargs
        assert kw["model"] == "test/model:free"

    def test_messages_passed_unmodified(
        self, client: OpenRouterClient
    ) -> None:
        mock_resp = _mock_response()
        msgs = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            client.call(msgs, agent_config=SAMPLE_AGENT_CFG)
        kw = mock_create.call_args.kwargs
        assert kw["messages"] == msgs


# ------------------------------------------------------------------
# 12. Retry delay calculation
# ------------------------------------------------------------------


class TestRetryDelay:
    def test_delay_no_jitter(self) -> None:
        cfg = _make_config()
        cfg.llm["retry"]["backoff_base"] = 2
        cfg.llm["retry"]["jitter"] = False
        c = OpenRouterClient(cfg)

        assert c._retry_delay(1) == 2.0   # 2 * 1^2
        assert c._retry_delay(2) == 8.0   # 2 * 2^2
        assert c._retry_delay(3) == 18.0  # 2 * 3^2

    def test_delay_with_jitter_is_greater_or_equal(self) -> None:
        cfg = _make_config()
        cfg.llm["retry"]["backoff_base"] = 2
        cfg.llm["retry"]["jitter"] = True
        c = OpenRouterClient(cfg)

        delay = c._retry_delay(1)
        # base delay is 2, jitter adds [0, 1)
        assert 2.0 <= delay < 3.0


# ------------------------------------------------------------------
# 13. API key resolution
# ------------------------------------------------------------------


class TestApiKeyResolution:
    def test_reads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "my-secret-key")
        key = OpenRouterClient._resolve_api_key()
        assert key == "my-secret-key"

    def test_returns_empty_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        key = OpenRouterClient._resolve_api_key()
        assert key == ""


# ------------------------------------------------------------------
# 14. Config defaults
# ------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_base_url(self) -> None:
        cfg = _make_config()
        del cfg.llm["base_url"]
        c = OpenRouterClient(cfg)
        assert c._base_url == "https://openrouter.ai/api/v1"

    def test_default_retry_values(self) -> None:
        cfg = _make_config()
        del cfg.llm["retry"]
        c = OpenRouterClient(cfg)
        assert c._max_retries == 3
        assert c._backoff_base == 2.0
        assert c._jitter is True
