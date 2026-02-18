"""OpenRouter API client using the OpenAI Python SDK.

A thin wrapper that configures the OpenAI client for OpenRouter,
adds app-identification headers, and provides exponential-backoff
retry logic for transient failures (429 / 5xx).

Think-block removal is NOT performed here -- that responsibility
belongs to the base_agent layer.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

import httpx
import openai
import requests as _requests_lib
from openai import OpenAI

from core.config import AppConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OpenRouterError(Exception):
    """Raised when an OpenRouter API call fails after all retries."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Synchronous OpenRouter client built on the OpenAI SDK.

    Provides:
    * App-identification headers (``HTTP-Referer``, ``X-Title``).
    * Retry with exponential back-off for 429 and 5xx errors.
    * ``response_format`` pass-through when the model supports it.
    * ``extra_body`` pass-through for OpenRouter-specific params.
    """

    def __init__(self, config: AppConfig) -> None:
        llm = config.llm

        self._model: str = llm["model"]
        self._fallback_models: list[str] = llm.get("fallback_models", [])
        self._base_url: str = llm.get("base_url", "https://openrouter.ai/api/v1")
        self._app_url: str = llm.get("app_url", "")
        self._app_title: str = llm.get("app_title", "")

        retry_cfg = llm.get("retry", {})
        self._max_retries: int = retry_cfg.get("max_retries", 3)
        self._backoff_base: float = float(retry_cfg.get("backoff_base", 2))
        self._jitter: bool = retry_cfg.get("jitter", True)

        # Progressive timeout: [600s, 600s] for primary, 900s for fallback
        # Uses httpx.Timeout with explicit read timeout to prevent
        # slow-streaming responses from resetting the timer on each byte.
        timeout_cfg = llm.get("timeout", {})
        if isinstance(timeout_cfg, (int, float)):
            t1 = float(timeout_cfg)
            read_timeouts = [t1, t1 * 1.25, t1 * 2.5]
        elif isinstance(timeout_cfg, dict):
            read_timeouts = [
                float(timeout_cfg.get("attempt_1", 600)),
                float(timeout_cfg.get("attempt_2", 600)),
                float(timeout_cfg.get("fallback", 900)),
            ]
        else:
            read_timeouts = [600.0, 600.0, 900.0]

        self._timeouts: list[httpx.Timeout] = [
            httpx.Timeout(connect=10.0, read=rt, write=30.0, pool=10.0)
            for rt in read_timeouts
        ]

        # Create client without global timeout -- per-request timeout
        # is set via the ``timeout=`` kwarg in each create() call,
        # avoiding SDK-level timeout conflicts.
        self._client = OpenAI(
            api_key=self._resolve_api_key(),
            base_url=self._base_url,
            default_headers={
                "HTTP-Referer": self._app_url,
                "X-Title": self._app_title,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        messages: list[dict[str, str]],
        *,
        agent_config: dict[str, Any],
        response_format_supported: bool = False,
        extra_body: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        """Call the OpenRouter chat-completion endpoint with model fallback.

        Tries the primary model (or agent-specific override) first.
        If all retries fail, falls back to the next model in the chain.

        Args:
            messages: OpenAI-format message list.
            agent_config: Agent-specific settings (``effort``,
                ``max_tokens``, ``temperature``, optional ``model``).
            response_format_supported: When ``True`` the request
                includes ``response_format: {"type": "json_object"}``.
            extra_body: Extra body params forwarded to OpenRouter
                (e.g. reasoning configuration).
            model_override: When set, use this model as primary,
                bypassing agent_config and default model selection.

        Returns:
            The raw ``content`` string from the first choice.

        Raises:
            OpenRouterError: After all models and retry attempts are exhausted.
        """
        # Build model chain: model_override > agent-specific > primary > fallbacks
        primary = model_override or agent_config.get("model", self._model)
        models_to_try = [primary] + [
            m for m in self._fallback_models if m != primary
        ]

        last_error: Exception | None = None

        for model_idx, model in enumerate(models_to_try):
            kwargs = self._build_kwargs(
                messages,
                agent_config=agent_config,
                response_format_supported=response_format_supported,
                extra_body=extra_body,
                model_override=model,
            )

            model_label = f"[{model_idx + 1}/{len(models_to_try)}] {model}"
            is_fallback = model_idx > 0
            timeout_consecutive = 0  # Track consecutive timeout failures

            for attempt in range(1, self._max_retries + 1):
                # Progressive timeout: primary model uses timeouts[0], [1];
                # fallback model uses timeouts[2]
                if is_fallback:
                    req_timeout = self._timeouts[2] if len(self._timeouts) > 2 else self._timeouts[-1]
                else:
                    idx = min(attempt - 1, len(self._timeouts) - 1)
                    req_timeout = self._timeouts[idx]

                try:
                    response = self._client.chat.completions.create(
                        **kwargs, timeout=req_timeout
                    )
                    content = response.choices[0].message.content
                    # Detect truncated JSON responses
                    stripped = (content or "").strip()
                    if stripped and response_format_supported:
                        # When JSON mode is on, response must end with } or ]
                        if not (
                            stripped.endswith("}")
                            or stripped.endswith("]")
                        ):
                            logger.warning(
                                "%s: response appears truncated "
                                "(doesn't end with } or ]), retrying.",
                                model_label,
                            )
                            last_error = ValueError(
                                f"Truncated response: ...{stripped[-50:]}"
                            )
                            delay = self._retry_delay(attempt)
                            time.sleep(delay)
                            continue
                    if is_fallback:
                        logger.info(
                            "Fallback model %s succeeded", model_label
                        )
                    return content or ""
                except openai.RateLimitError as exc:
                    last_error = exc
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "%s: 429 (attempt %d/%d). Retrying in %.1fs ...",
                        model_label, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                except openai.APIStatusError as exc:
                    last_error = exc
                    status = getattr(exc, "status_code", None)
                    if status is not None and 500 <= status < 600:
                        delay = self._retry_delay(attempt)
                        logger.warning(
                            "%s: %d (attempt %d/%d). Retrying in %.1fs ...",
                            model_label, status, attempt,
                            self._max_retries, delay,
                        )
                        time.sleep(delay)
                    else:
                        # Non-retryable status → skip to next model
                        logger.warning(
                            "%s: non-retryable error %d, trying next model ...",
                            model_label, status or 0,
                        )
                        break
                except openai.APITimeoutError as exc:
                    last_error = exc
                    timeout_consecutive += 1
                    logger.warning(
                        "%s: timeout after %.0fs (attempt %d/%d, "
                        "consecutive timeouts: %d). ",
                        model_label, req_timeout.read, attempt,
                        self._max_retries, timeout_consecutive,
                    )
                    # After 2 consecutive timeouts on primary, skip to fallback
                    if not is_fallback and timeout_consecutive >= 2 and model_idx < len(models_to_try) - 1:
                        logger.warning(
                            "%s: 2 consecutive timeouts, switching to fallback model ...",
                            model_label,
                        )
                        break
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "Retrying in %.1fs ...", delay
                    )
                    time.sleep(delay)
                except openai.APIConnectionError as exc:
                    last_error = exc
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "%s: connection error (attempt %d/%d). "
                        "Retrying in %.1fs ...",
                        model_label, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    last_error = exc
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "%s: response parse error (attempt %d/%d): %s. "
                        "Retrying in %.1fs ...",
                        model_label, attempt, self._max_retries, exc, delay,
                    )
                    time.sleep(delay)

            # All retries for this model exhausted
            if model_idx < len(models_to_try) - 1:
                logger.warning(
                    "%s: retries exhausted (or timeout skip). "
                    "Falling back to next model ...",
                    model_label,
                )

        raise OpenRouterError(
            f"All {len(models_to_try)} models exhausted "
            f"(each tried {self._max_retries}x): {last_error}",
            status_code=getattr(last_error, "status_code", None),
        )

    def check_api_key(self) -> dict[str, Any]:
        """Validate the API key via ``GET /api/v1/key``.

        Retries on 5xx and connection errors (up to ``max_retries``
        attempts).  Fails immediately on 401/403 (invalid key).

        Returns:
            Parsed JSON response from the key-info endpoint.

        Raises:
            OpenRouterError: If the request fails after all retries.
        """
        url = self._base_url.rstrip("/").replace("/v1", "") + "/v1/key"
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "HTTP-Referer": self._app_url,
            "X-Title": self._app_title,
        }

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = _requests_lib.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                return resp.json()  # type: ignore[no-any-return]
            except _requests_lib.exceptions.HTTPError as exc:
                last_error = exc
                status = getattr(exc.response, "status_code", None)
                if status is not None and 500 <= status < 600:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "check_api_key: %d (attempt %d/%d). "
                        "Retrying in %.1fs ...",
                        status, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise OpenRouterError(
                        f"API key validation failed: {exc}",
                        status_code=status,
                    ) from exc
            except (
                _requests_lib.exceptions.ConnectionError,
                _requests_lib.exceptions.Timeout,
            ) as exc:
                last_error = exc
                delay = self._retry_delay(attempt)
                logger.warning(
                    "check_api_key: connection error (attempt %d/%d). "
                    "Retrying in %.1fs ...",
                    attempt, self._max_retries, delay,
                )
                time.sleep(delay)
            except Exception as exc:
                raise OpenRouterError(
                    f"API key validation failed: {exc}"
                ) from exc

        raise OpenRouterError(
            f"API key validation failed after {self._max_retries} retries: {last_error}"
        )

    def probe_response_format(self, model: str) -> bool:
        """Test whether *model* supports ``response_format``.

        Makes a minimal chat-completion call with
        ``response_format: {"type": "json_object"}`` and returns
        ``True`` if the call succeeds, ``False`` otherwise.
        """
        try:
            self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"ok": true}',
                    }
                ],
                max_tokens=32,
                response_format={"type": "json_object"},
            )
            return True
        except Exception:
            logger.debug(
                "Model %s does not support response_format", model
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key() -> str:
        """Read the OpenRouter API key from the environment."""
        key = os.environ.get("OPENROUTER_API_KEY", "")
        return key

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        agent_config: dict[str, Any],
        response_format_supported: bool,
        extra_body: dict[str, Any] | None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Build the keyword arguments for ``chat.completions.create``."""
        kwargs: dict[str, Any] = {
            "model": model_override or self._model,
            "messages": messages,
        }

        if "max_tokens" in agent_config:
            kwargs["max_tokens"] = agent_config["max_tokens"]

        if "temperature" in agent_config:
            kwargs["temperature"] = agent_config["temperature"]

        if response_format_supported:
            kwargs["response_format"] = {"type": "json_object"}

        # Merge extra_body (OpenRouter-specific params)
        merged_extra: dict[str, Any] = dict(extra_body) if extra_body else {}

        # Map agent effort -> reasoning.effort in extra_body
        if "effort" in agent_config:
            reasoning = merged_extra.setdefault("reasoning", {})
            reasoning["effort"] = agent_config["effort"]

        if merged_extra:
            kwargs["extra_body"] = merged_extra

        return kwargs

    def _retry_delay(self, attempt: int) -> float:
        """Compute delay for retry *attempt* (1-based).

        Formula: ``backoff_base * (attempt ** 2) + jitter``
        where jitter is a random float in ``[0, 1)`` when enabled.
        """
        delay = self._backoff_base * (attempt ** 2)
        if self._jitter:
            delay += random.random()
        return delay
