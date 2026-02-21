"""LLM Analyst for weekly intelligence report.

Wraps OpenRouterClient with rate limiting, token splitting,
and graceful fallback for weekly report LLM calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.llm.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class WeeklyLLMAnalyst:
    """Manages LLM calls for weekly intelligence report generation.

    Provides rate-limited, fault-tolerant LLM calls with automatic
    token budget management.
    """

    def __init__(self, config: Any, rate_limiter: Any | None = None) -> None:
        """Initialize with app config and optional rate limiter.

        Args:
            config: AppConfig object with llm configuration.
            rate_limiter: Optional RateLimiter instance from preflight.
        """
        self._client = OpenRouterClient(config)
        self._rate_limiter = rate_limiter

        intel_llm = config.weekly.get("intelligence", {}).get("llm", {})
        self._delay = intel_llm.get("delay_between_calls_sec", 3)
        self._max_input_tokens = intel_llm.get("max_input_tokens_per_call", 8000)
        self._model_override = intel_llm.get("model", "") or None

    def call(self, system_prompt: str, user_prompt: str, label: str) -> str | None:
        """Make a single LLM call with rate limiting and graceful fallback.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message with the data to analyze.
            label: Human-readable label for logging.

        Returns:
            LLM response string, or None on failure.
        """
        try:
            if self._rate_limiter:
                self._rate_limiter.wait()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            agent_config = {
                "effort": "medium",
                "max_tokens": 2000,
                "temperature": 0.3,
            }

            result = self._client.call(
                messages,
                agent_config=agent_config,
                model_override=self._model_override,
            )

            if self._rate_limiter:
                self._rate_limiter.record_call()

            time.sleep(self._delay)
            return result

        except Exception as e:
            logger.warning("Weekly LLM call '%s' failed: %s", label, e)
            return None

    def split_papers_for_prompt(
        self, papers: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Split papers into full (with abstract) and title-only based on token limit.

        Papers are sorted by score descending. Higher-scored papers get
        full abstract inclusion; lower-scored ones get title-only.

        Args:
            papers: List of paper dicts with title, abstract, final_score.

        Returns:
            Tuple of (full_papers, title_only_papers).
        """
        papers_sorted = sorted(
            papers, key=lambda p: p.get("final_score", 0) or 0, reverse=True
        )
        full_papers: list[dict] = []
        title_only: list[dict] = []
        token_count = 0

        for p in papers_sorted:
            abstract = p.get("abstract", "") or ""
            est_tokens = len(abstract) // 3  # rough char-to-token estimate
            if token_count + est_tokens <= self._max_input_tokens:
                full_papers.append(p)
                token_count += est_tokens
            else:
                title_only.append(p)

        return full_papers, title_only
