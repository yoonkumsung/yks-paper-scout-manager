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
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from agents.base_agent import BaseAgent
from core.config import AppConfig
from core.llm.openrouter_client import OpenRouterClient
from core.models import TopicSpec

logger = logging.getLogger(__name__)

# Required top-level keys in the LLM output
_REQUIRED_KEYS = {"concepts", "cross_domain_keywords", "exclude_keywords", "topic_embedding_text"}

# Representative English keywords for major arXiv categories (fallback)
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "cs.AI": ["artificial intelligence", "reasoning", "knowledge representation", "planning"],
    "cs.LG": ["machine learning", "deep learning", "neural network", "representation learning"],
    "cs.CL": ["natural language processing", "language model", "text generation", "sentiment analysis"],
    "cs.CV": ["computer vision", "image recognition", "object detection", "image segmentation"],
    "cs.NE": ["neural architecture", "evolutionary computation", "genetic algorithm", "neuroevolution"],
    "cs.IR": ["information retrieval", "search engine", "recommendation system", "ranking"],
    "cs.MA": ["multi-agent system", "cooperative learning", "agent communication", "swarm intelligence"],
    "cs.RO": ["robotics", "motion planning", "robot learning", "manipulation"],
    "cs.SE": ["software engineering", "code generation", "software testing", "DevOps"],
    "cs.DC": ["distributed computing", "parallel processing", "cloud computing", "MapReduce"],
    "cs.DB": ["database", "query optimization", "data management", "knowledge graph"],
    "cs.OS": ["operating system", "kernel", "scheduling", "virtualization"],
    "cs.PL": ["programming language", "compiler", "type system", "static analysis"],
    "cs.AR": ["computer architecture", "hardware accelerator", "GPU", "FPGA"],
    "cs.NI": ["computer network", "network protocol", "routing", "software-defined networking"],
    "cs.PF": ["performance evaluation", "benchmarking", "workload characterization"],
    "cs.DS": ["data structure", "algorithm", "graph algorithm", "sorting"],
    "cs.CC": ["computational complexity", "NP-hard", "approximation algorithm"],
    "cs.LO": ["logic", "formal verification", "model checking", "theorem proving"],
    "cs.FL": ["formal language", "automata theory", "regular expression", "parsing"],
    "cs.DM": ["discrete mathematics", "combinatorics", "graph theory"],
    "cs.IT": ["information theory", "coding theory", "entropy", "channel capacity"],
    "cs.NA": ["numerical analysis", "numerical methods", "finite element"],
    "cs.SC": ["symbolic computation", "computer algebra", "symbolic reasoning"],
    "cs.GT": ["game theory", "mechanism design", "auction", "Nash equilibrium"],
    "cs.CG": ["computational geometry", "convex hull", "Voronoi diagram"],
    "cs.CR": ["cryptography", "security", "privacy", "encryption"],
    "cs.HC": ["human-computer interaction", "user interface", "usability", "UX design"],
    "cs.CY": ["computers and society", "digital ethics", "AI policy", "fairness"],
    "cs.CE": ["computational engineering", "simulation", "finite element analysis"],
    "cs.GR": ["computer graphics", "rendering", "3D modeling", "ray tracing"],
    "cs.MM": ["multimedia", "video processing", "audio processing", "streaming"],
    "cs.SD": ["sound", "speech synthesis", "audio generation", "music information retrieval"],
    "cs.SI": ["social network", "information diffusion", "community detection", "influence maximization"],
    "cs.DL": ["digital library", "scholarly communication", "metadata"],
    "cs.ET": ["emerging technology", "quantum computing", "neuromorphic computing"],
    "cs.MS": ["mathematical software", "numerical library", "scientific computing"],
    "cs.GL": ["general literature", "survey", "tutorial"],
    "cs.OH": ["other", "miscellaneous"],
    "cs.SY": ["systems and control", "control theory", "feedback control"],
    "stat.ML": ["statistical learning", "Bayesian inference", "kernel methods", "ensemble methods"],
    "stat.ME": ["statistical methodology", "hypothesis testing", "regression", "causal inference"],
    "stat.TH": ["statistical theory", "asymptotic analysis", "estimation theory"],
    "stat.AP": ["applied statistics", "biostatistics", "spatial statistics"],
    "stat.CO": ["computational statistics", "Monte Carlo", "MCMC", "bootstrap"],
    "eess.SP": ["signal processing", "filter design", "spectral analysis", "compressed sensing"],
    "eess.AS": ["audio signal processing", "speech recognition", "speaker verification"],
    "eess.IV": ["image processing", "video processing", "super-resolution", "denoising"],
    "eess.SY": ["systems and control", "dynamical systems", "optimal control"],
    "math.OC": ["optimization", "convex optimization", "linear programming", "gradient descent"],
    "math.ST": ["statistics theory", "mathematical statistics", "decision theory"],
    "math.PR": ["probability", "stochastic process", "random variable", "martingale"],
    "q-bio.QM": ["quantitative biology", "systems biology", "bioinformatics", "genomics"],
    "q-bio.NC": ["neuroscience", "computational neuroscience", "brain-computer interface", "neural coding"],
    "q-fin.ST": ["statistical finance", "financial modeling", "risk analysis"],
    "q-fin.CP": ["computational finance", "algorithmic trading", "portfolio optimization"],
    "physics.data-an": ["data analysis", "statistical physics", "experimental data", "signal detection"],
}

# System prompt for the keyword expander agent
_SYSTEM_PROMPT = (
    "You are an expert research librarian specializing in computer science "
    "and engineering papers on arXiv. Analyze the project description and generate "
    "comprehensive search concepts and keywords in English.\n\n"
    "CRITICAL RULES:\n"
    "1. Use ONLY academic/scholarly terminology that actually appears in arXiv paper "
    "titles and abstracts. NEVER use commercial, marketing, or colloquial terms.\n"
    "   - WRONG: 'automatic filming', 'smart camera', 'tiktok-style'\n"
    "   - RIGHT: 'autonomous cinematography', 'active camera system', 'short-form video generation'\n"
    "2. Each keyword should be a term you would find in a real paper's abstract or title.\n"
    "3. Think about how researchers describe these concepts in their publications.\n\n"
    "4. From the provided arXiv categories, identify which ones are NOT relevant to "
    "the project and return them in \"exclude_categories\". Keep all categories that "
    "could plausibly contain relevant papers. Only exclude clearly unrelated ones.\n\n"
    "5. DEDUPLICATION: Each concept must be clearly distinct from all others. "
    "Do NOT create concepts that overlap significantly.\n"
    "   - WRONG: 'Autonomous Cinematography' AND 'Autonomous Cinematography System'\n"
    "   - WRONG: 'Video Enhancement' AND 'Video and Image Enhancement'\n"
    "   - RIGHT: Keep only ONE concept per distinct idea.\n\n"
    "6. EXCLUDE KEYWORDS must only contain terms that would bring IRRELEVANT papers. "
    "Never exclude terms that are CORE to the project description. "
    "Exclude keywords should filter out papers from unrelated domains "
    "(e.g., medical imaging when searching for sports vision).\n"
    "   - WRONG: excluding 'smart camera' when the project IS about smart cameras\n"
    "   - WRONG: excluding 'social media platform' when the project aims to BUILD one\n"
    "   - RIGHT: excluding 'medical imaging', 'satellite imagery', 'autonomous driving'\n\n"
    "7. Generate 8-12 concepts maximum. If concepts overlap, merge them.\n\n"
    "8. NARROW QUERIES GUIDANCE: Cross-domain and narrow keywords must be specific enough "
    "to return fewer than ~10,000 papers on arXiv. Never use ultra-generic terms like "
    "'computer vision', 'machine learning', 'deep learning', 'artificial intelligence' alone. "
    "Instead combine them with domain-specific qualifiers.\n"
    "   - WRONG: 'computer vision' (too broad, millions of papers)\n"
    "   - RIGHT: 'computer vision for sports analysis', 'vision-based pose estimation'\n\n"
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

    def expand(self, topic: TopicSpec, *, skip_cache: bool = False) -> dict:
        """Main entry point. Returns keyword expansion result.

        Args:
            topic: The topic specification to expand.
            skip_cache: If True, bypass cache and force LLM call.

        Returns:
            dict with keys: concepts, cross_domain_keywords,
            exclude_keywords, topic_embedding_text.
        """
        cache_key = self._compute_cache_key(topic)

        # --- Cache lookup ---
        cache = self._load_cache()
        if not skip_cache:
            if cache_key in cache and self._is_cache_valid(cache[cache_key]):
                logger.info("keyword_expander: cache hit for topic '%s'", topic.slug)
                return cache[cache_key]["result"]

        # --- Chunk categories if too many ---
        chunk_size = self.agent_config.get("chunk_size", 15)
        if not chunk_size or chunk_size <= 0:
            chunk_size = 15
        cats = topic.arxiv_categories
        if len(cats) > chunk_size:
            logger.info(
                "keyword_expander: chunking %d categories into batches of %d for '%s'",
                len(cats), chunk_size, topic.slug,
            )
            result = self._expand_chunked(topic, chunk_size)
        else:
            result = self._expand_single(topic)

        if result is None:
            logger.warning(
                "keyword_expander: all attempts failed, using fallback for '%s'",
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

    def _expand_single(self, topic: TopicSpec) -> dict | None:
        """LLM call with retry (max 2 attempts) for a single topic."""
        messages = self.build_messages(topic=topic)
        parse_failures = 0

        for attempt in range(2):
            raw = self.call_llm(messages)
            if raw is not None and self._validate_output(raw):
                return raw
            parse_failures += 1
            logger.warning(
                "keyword_expander: parse failure %d/2 for topic '%s'",
                parse_failures,
                topic.slug,
            )

        return None

    def _expand_chunked(self, topic: TopicSpec, chunk_size: int) -> dict | None:
        """Split categories into chunks, call LLM per chunk, merge results."""
        cats = topic.arxiv_categories
        merged: dict | None = None
        total_chunks = (len(cats) + chunk_size - 1) // chunk_size

        for i in range(0, len(cats), chunk_size):
            chunk_cats = cats[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            logger.info(
                "keyword_expander: chunk %d/%d (%d categories) for '%s' - LLM 호출 중...",
                chunk_num, total_chunks, len(chunk_cats), topic.slug,
            )
            # Create a temporary TopicSpec-like object with subset of categories
            chunk_topic = TopicSpec(
                slug=topic.slug,
                name=topic.name,
                description=topic.description,
                arxiv_categories=chunk_cats,
                must_concepts_en=topic.must_concepts_en if i == 0 else None,
                should_concepts_en=topic.should_concepts_en if i == 0 else None,
                must_not_en=topic.must_not_en if i == 0 else None,
            )
            chunk_result = self._expand_single(chunk_topic)
            if chunk_result is None:
                logger.warning(
                    "keyword_expander: chunk %d/%d failed for '%s', using fallback for this chunk",
                    chunk_num, total_chunks, topic.slug,
                )
                continue
            logger.info(
                "keyword_expander: chunk %d/%d complete for '%s'",
                chunk_num, total_chunks, topic.slug,
            )

            if merged is None:
                merged = chunk_result
            else:
                merged = self._merge_chunk_results(merged, chunk_result)

        return merged

    def _merge_chunk_results(self, base: dict, addition: dict) -> dict:
        """Merge two chunk results, deduplicating concepts and keywords."""
        existing_concepts = {c.get("name_en", "").lower() for c in base.get("concepts", [])}
        for concept in addition.get("concepts", []):
            if concept.get("name_en", "").lower() not in existing_concepts:
                base["concepts"].append(concept)
                existing_concepts.add(concept.get("name_en", "").lower())

        existing_cross = {kw.lower() for kw in base.get("cross_domain_keywords", [])}
        for kw in addition.get("cross_domain_keywords", []):
            if kw.lower() not in existing_cross:
                base["cross_domain_keywords"].append(kw)
                existing_cross.add(kw.lower())

        existing_excl = {kw.lower() for kw in base.get("exclude_keywords", [])}
        for kw in addition.get("exclude_keywords", []):
            if kw.lower() not in existing_excl:
                base["exclude_keywords"].append(kw)
                existing_excl.add(kw.lower())

        # Combine embedding texts
        if addition.get("topic_embedding_text"):
            base["topic_embedding_text"] = (
                base.get("topic_embedding_text", "") + " " + addition["topic_embedding_text"]
            )

        return base

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
            '  "exclude_categories": ["cat.XX", ...] (categories from the input list that are NOT relevant),\n'
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
        """Save cache to file atomically, creating parent directories if needed."""
        cache_dir = os.path.dirname(self.CACHE_FILE)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        try:
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=cache_dir or ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.CACHE_FILE)  # atomic on POSIX
            except Exception:
                os.unlink(tmp_path)  # cleanup temp file
                raise
        except OSError as exc:
            logger.warning("Failed to save keyword cache: %s", exc)

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
        """Generate meaningful keywords from arxiv_categories without LLM.

        Uses _CATEGORY_KEYWORDS dictionary to produce real English keywords
        instead of raw category codes like 'cs.AI'.
        """
        concepts = []
        all_keywords: list[str] = []

        categories = topic.arxiv_categories
        if not categories:
            logger.warning(
                "Topic '%s' has no arxiv_categories. Using default CS categories.",
                topic.slug,
            )
            categories = ["cs.AI", "cs.LG", "cs.CV", "cs.CL"]

        for cat in categories:
            kws = _CATEGORY_KEYWORDS.get(cat, [cat.lower()])
            concepts.append(
                {
                    "name_ko": cat,
                    "name_en": kws[0] if kws else cat,
                    "keywords": kws,
                }
            )
            all_keywords.extend(kws)

        # Build embedding text from all collected keywords
        unique_keywords = list(dict.fromkeys(all_keywords))  # preserve order, deduplicate
        embedding_text = " ".join(unique_keywords[:50])  # cap at 50 keywords

        return {
            "concepts": concepts,
            "cross_domain_keywords": unique_keywords[:20],
            "exclude_keywords": topic.must_not_en or [],
            "topic_embedding_text": embedding_text,
        }
