"""Base agent class for all LLM agents in Paper Scout.

Provides the common LLM-calling and JSON-parsing pipeline used by
keyword_expander, scorer, and summarizer agents.  Subclasses must
override ``agent_name``, ``agent_config_key``, and ``build_messages()``.

Think-block removal is handled in two places:
  1. ``JsonParser.parse()`` always strips think blocks (Layer 1).
  2. ``call_llm_raw()`` strips think blocks explicitly for non-JSON use.

The ``response_format_supported`` flag flows through to both
``OpenRouterClient.call()`` (API parameter) and ``JsonParser.parse()``
(parsing mode selection).
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import AppConfig
from core.llm.openrouter_client import OpenRouterClient
from core.parsing.json_parser import JsonParser

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all LLM agents (keyword_expander, scorer, summarizer).

    Subclasses must override:
        - agent_name: str property
        - agent_config_key: str property (key in config.agents, e.g. "keyword_expander")
        - build_messages(self, **kwargs) -> list[dict]: Build messages for the LLM call
    """

    def __init__(
        self,
        config: AppConfig,
        client: OpenRouterClient,
        response_format_supported: bool = False,
    ) -> None:
        """Initialize the base agent.

        Args:
            config: Application config.
            client: OpenRouter client instance (shared across agents).
            response_format_supported: Whether the model supports
                ``response_format`` (determined during preflight).
        """
        self._config = config
        self._client = client
        self._response_format_supported = response_format_supported
        self._parser = JsonParser()
        self._common_config = config.agents.get("common", {})

    # ------------------------------------------------------------------
    # Abstract properties (must be overridden by subclasses)
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Unique agent identifier for logging and debug dumps."""
        raise NotImplementedError

    @property
    def agent_config_key(self) -> str:
        """Key in config.agents for this agent's settings."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete properties
    # ------------------------------------------------------------------

    @property
    def agent_config(self) -> dict:
        """This agent's config section from AppConfig."""
        return self._config.agents.get(self.agent_config_key, {})

    @property
    def prompt_version(self) -> str:
        """Return the prompt_version from agent config."""
        return self.agent_config.get("prompt_version", "unknown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
        batch_index: int = 0,
    ) -> dict | list | None:
        """Call the LLM and parse the response as JSON.

        Flow:
            1. Call ``OpenRouterClient.call()`` with messages, agent_config,
               and response_format_supported.
            2. Parse response via ``JsonParser.parse()`` with
               response_format_supported, agent_name, and batch_index.
            3. Return parsed JSON or ``None``.

        Args:
            messages: OpenAI-format messages.
            extra_body: Additional params for OpenRouter.
            batch_index: For debug dump filenames.

        Returns:
            Parsed JSON dict/list, or ``None`` on parse failure.
        """
        raw = self._client.call(
            messages,
            agent_config=self.agent_config,
            response_format_supported=self._response_format_supported,
            extra_body=extra_body,
        )

        return self._parser.parse(
            raw,
            response_format_supported=self._response_format_supported,
            agent_name=self.agent_name,
            batch_index=batch_index,
        )

    def call_llm_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Call the LLM and return raw content (with think blocks removed).

        Used when raw text is needed instead of parsed JSON.

        Flow:
            1. Call ``OpenRouterClient.call()``.
            2. Strip think blocks via ``JsonParser.remove_think_blocks()``.
            3. Return cleaned text.

        Args:
            messages: OpenAI-format messages.
            extra_body: Additional params for OpenRouter.

        Returns:
            Cleaned text with think blocks removed.
        """
        raw = self._client.call(
            messages,
            agent_config=self.agent_config,
            response_format_supported=self._response_format_supported,
            extra_body=extra_body,
        )

        return self._parser.remove_think_blocks(raw)

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Build the message list for the LLM call.

        Must be overridden by subclasses.

        Args:
            **kwargs: Subclass-specific arguments.

        Returns:
            OpenAI-format message list.
        """
        raise NotImplementedError
