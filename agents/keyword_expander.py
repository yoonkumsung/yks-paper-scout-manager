"""Agent 1: Keyword Expander for Paper Scout.

Expands Korean topic descriptions into English search concepts,
cross-domain keywords, exclude keywords, and a topic embedding text.
Results are cached to ``data/keyword_cache.json`` with a configurable
TTL (default 30 days).

Section 7-1 of the devspec covers the full specification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm.openrouter_client import OpenRouterClient
from core.models import TopicSpec

logger = logging.getLogger(__name__)

# Required top-level keys in the LLM output
_REQUIRED_KEYS = {"concepts", "cross_domain_keywords", "exclude_keywords", "topic_embedding_text"}

# System prompt for the keyword expander agent
_SYSTEM_PROMPT = (
    "You are an expert research librarian specializing in computer science "
    "and engineering papers. Analyze the project description and generate "
    "comprehensive search concepts and keywords in English.\n\n"
    "Also generate a \"topic_embedding_text\" field: a single English paragraph "
    "combining all key concepts and keywords, suitable for semantic similarity "
    "matching against paper abstracts."
)


class KeywordExpander(BaseAgent):
    """Agent 1: Expands Korean topic descriptions into English search concepts.

    Flow (``expand``):
      1. Compute cache key (SHA-256 of description + optional fields).
      2. Check cache -- return if hit and not expired.
      3. Call LLM via ``call_llm()``.
      4. Validate output structure.
      5. Merge ``must_concepts_en`` into ``concepts`` (if provided).
      6. Merge ``should_concepts_en`` into ``cross_domain_keywords`` (if provided).
      7. Merge ``must_not_en`` into ``exclude_keywords`` (if provided).
      8. Save to cache.
      9. Return result.

    On 2x consecutive parse failure the agent generates a category-based
    fallback without any LLM call.
    """

    CACHE_FILE = "data/keyword_cache.json"

    # ------------------------------------------------------------------
    # Abstract property overrides
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        return "keyword_expander"

    @property
    def agent_config_key(self) -> str:
        return "keyword_expander"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, topic: TopicSpec) -> dict:
        """Main entry point. Returns keyword expansion result.

        Args:
            topic: The topic specification to expand.

        Returns:
            dict with keys: concepts, cross_domain_keywords,
            exclude_keywords, topic_embedding_text.
        """
        cache_key = self._compute_cache_key(topic)

        # --- Cache lookup ---
        cache = self._load_cache()
        if cache_key in cache and self._is_cache_valid(cache[cache_key]):
            logger.info("keyword_expander: cache hit for topic '%s'", topic.slug)
            return cache[cache_key]["result"]

        # --- LLM call with retry (max 2 attempts) ---
        messages = self.build_messages(topic=topic)
        result: dict | None = None
        parse_failures = 0

        for attempt in range(2):
            raw = self.call_llm(messages)
            if raw is not None and self._validate_output(raw):
                result = raw
                break
            parse_failures += 1
            logger.warning(
                "keyword_expander: parse failure %d/2 for topic '%s'",
                parse_failures,
                topic.slug,
            )

        # --- Fallback on 2x failure ---
        if result is None:
            logger.warning(
                "keyword_expander: 2x parse failure, using fallback for '%s'",
                topic.slug,
            )
            result = self._generate_fallback(topic)
            return result  # fallback is not cached

        # --- Merge user constraints ---
        result = self._merge_user_constraints(result, topic)

        # --- Save to cache ---
        cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "prompt_version": self.prompt_version,
        }
        self._save_cache(cache)

        return result

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Build system + user messages for the LLM call.

        Expects ``topic`` keyword argument of type :class:`TopicSpec`.
        """
        topic: TopicSpec = kwargs["topic"]

        user_parts: list[str] = []

        user_parts.append(f"Project description:\n{topic.description}")
        user_parts.append(f"\narXiv categories: {', '.join(topic.arxiv_categories)}")

        if topic.must_concepts_en:
            user_parts.append(
                f"\nThese concepts MUST be included in concepts: "
                f"{', '.join(topic.must_concepts_en)}"
            )
        if topic.should_concepts_en:
            user_parts.append(
                f"\nMerge these into cross_domain_keywords: "
                f"{', '.join(topic.should_concepts_en)}"
            )
        if topic.must_not_en:
            user_parts.append(
                f"\nMerge these into exclude_keywords: "
                f"{', '.join(topic.must_not_en)}"
            )

        user_parts.append(
            "\nRespond with a JSON object containing:\n"
            '  "concepts": [{"name_ko": "...", "name_en": "...", "keywords": ["..."]}],\n'
            '  "cross_domain_keywords": ["..."],\n'
            '  "exclude_keywords": ["..."],\n'
            '  "topic_embedding_text": "A single English paragraph..."\n'
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _compute_cache_key(self, topic: TopicSpec) -> str:
        """Hash description + optional fields for cache lookup."""
        parts = [topic.description]
        if topic.must_concepts_en:
            parts.append("|must:" + ",".join(sorted(topic.must_concepts_en)))
        if topic.should_concepts_en:
            parts.append("|should:" + ",".join(sorted(topic.should_concepts_en)))
        if topic.must_not_en:
            parts.append("|not:" + ",".join(sorted(topic.must_not_en)))

        raw = "".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        """Load cache file. Return empty dict if missing/corrupt."""
        try:
            with open(self.CACHE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            return {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self, cache: dict) -> None:
        """Save cache to file, creating parent directories if needed."""
        cache_dir = os.path.dirname(self.CACHE_FILE)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)

    def _is_cache_valid(self, entry: dict) -> bool:
        """Check if cache entry is within TTL."""
        cached_at_str = entry.get("cached_at")
        if not cached_at_str:
            return False

        try:
            cached_at = datetime.fromisoformat(cached_at_str)
        except (ValueError, TypeError):
            return False

        # Ensure timezone-aware comparison
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)

        ttl_days = self.agent_config.get("cache_ttl_days", 30)
        now = datetime.now(timezone.utc)
        return (now - cached_at) < timedelta(days=ttl_days)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_output(self, result: dict) -> bool:
        """Validate output has required keys and correct structure.

        Returns True if valid, False otherwise.
        """
        if not isinstance(result, dict):
            return False

        if not _REQUIRED_KEYS.issubset(result.keys()):
            return False

        # concepts must be a list of dicts with name_en and keywords
        concepts = result.get("concepts")
        if not isinstance(concepts, list):
            return False
        for concept in concepts:
            if not isinstance(concept, dict):
                return False
            if "name_en" not in concept or "keywords" not in concept:
                return False

        if not isinstance(result.get("cross_domain_keywords"), list):
            return False
        if not isinstance(result.get("exclude_keywords"), list):
            return False
        if not isinstance(result.get("topic_embedding_text"), str):
            return False

        return True

    # ------------------------------------------------------------------
    # Constraint merging
    # ------------------------------------------------------------------

    def _merge_user_constraints(self, result: dict, topic: TopicSpec) -> dict:
        """Merge must_concepts_en, should_concepts_en, must_not_en into result.

        - must_concepts_en: ensure each concept appears in result["concepts"]
        - should_concepts_en: merge into result["cross_domain_keywords"]
        - must_not_en: merge into result["exclude_keywords"]
        """
        # --- must_concepts_en -> concepts ---
        if topic.must_concepts_en:
            existing_names = {
                c.get("name_en", "").lower() for c in result.get("concepts", [])
            }
            for concept_name in topic.must_concepts_en:
                if concept_name.lower() not in existing_names:
                    result["concepts"].append(
                        {
                            "name_ko": concept_name,
                            "name_en": concept_name,
                            "keywords": [concept_name.lower()],
                        }
                    )

        # --- should_concepts_en -> cross_domain_keywords ---
        if topic.should_concepts_en:
            existing_kw = {
                kw.lower() for kw in result.get("cross_domain_keywords", [])
            }
            for kw in topic.should_concepts_en:
                if kw.lower() not in existing_kw:
                    result["cross_domain_keywords"].append(kw)

        # --- must_not_en -> exclude_keywords ---
        if topic.must_not_en:
            existing_excl = {
                kw.lower() for kw in result.get("exclude_keywords", [])
            }
            for kw in topic.must_not_en:
                if kw.lower() not in existing_excl:
                    result["exclude_keywords"].append(kw)

        return result

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _generate_fallback(self, topic: TopicSpec) -> dict:
        """Generate basic keywords from arxiv_categories without LLM."""
        concepts = [
            {
                "name_ko": cat,
                "name_en": cat,
                "keywords": [cat.lower()],
            }
            for cat in topic.arxiv_categories
        ]

        return {
            "concepts": concepts,
            "cross_domain_keywords": [],
            "exclude_keywords": topic.must_not_en or [],
            "topic_embedding_text": " ".join(topic.arxiv_categories),
        }
