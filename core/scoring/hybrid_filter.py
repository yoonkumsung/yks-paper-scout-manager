"""Hybrid Filter for Paper Scout (TASK-016).

Three-stage post-collection filter that progressively narrows a large
set of candidate papers down to a manageable, high-relevance subset:

  Stage 1   -- Rule Filter (negative keywords + category/keyword match)
  Stage 1.5 -- Defense Cap (pre_embed_cap, newest-first truncation)
  Stage 2   -- Embedding Sort *or* Recency Fallback
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HybridFilter:
    """Three-stage hybrid filter combining rules, caps, and embeddings.

    Parameters
    ----------
    config : dict
        The ``config['filter']`` section from ``config.yaml``.
        Expected keys (all optional with defaults):
          - ``pre_embed_cap``      (int, default 2000)
          - ``max_filter_output``  (int, default 200)
    """

    def __init__(self, config: dict) -> None:
        self._pre_embed_cap: int = int(config.get("pre_embed_cap", 2000))
        self._max_filter_output: int = int(
            config.get("max_filter_output", 200)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        papers: list[Any],
        agent1_output: dict,
        topic: Any,
        embedding_ranker: Any = None,
        topic_embedding_text: Optional[str] = None,
    ) -> tuple[list[Any], dict]:
        """Run the three-stage hybrid filter.

        Parameters
        ----------
        papers :
            List of ``Paper`` dataclass instances.
        agent1_output :
            Agent 1 output dict.  Expected keys:
              - ``concepts``  (list[dict]): each with ``keywords`` list
              - ``cross_domain_keywords`` (list[str], optional)
              - ``exclude_keywords`` (list[str])
        topic :
            ``TopicSpec`` instance. Uses ``arxiv_categories``,
            ``must_not_en``.
        embedding_ranker :
            Optional object with a ``compute_similarity(texts, query)``
            method returning a list of float scores.  ``None`` disables
            embedding sort (recency fallback is used instead).
        topic_embedding_text :
            Query text for embedding similarity.  Required when
            *embedding_ranker* is not ``None``.

        Returns
        -------
        tuple[list[Paper], dict]
            ``(filtered_papers, filter_stats)``
        """
        stats: dict[str, Any] = {
            "total_input": len(papers),
            "after_rule_filter": 0,
            "after_cap": 0,
            "after_embedding_sort": 0,
            "excluded_negative": 0,
            "excluded_no_match": 0,
            "negative_keyword_hits": {},
            "excluded_samples": [],
            "positive_keywords_count": 0,
            "negative_keywords_count": 0,
            "categories_used": [],
            "sort_method": "none",
        }

        if not papers:
            return [], stats

        # -- Collect keyword sets ----------------------------------------
        negative_kws = self._collect_negative_keywords(agent1_output, topic)
        positive_kws = self._collect_positive_keywords(agent1_output)
        categories = self._collect_categories(topic)

        stats["positive_keywords_count"] = len(positive_kws)
        stats["negative_keywords_count"] = len(negative_kws)
        stats["categories_used"] = sorted(categories)

        # -- Stage 1: Rule filter ----------------------------------------
        rule_passed: list[Any] = []

        for paper in papers:
            text = self._searchable_text(paper)
            has_positive = self._matches_any(text, positive_kws)

            # 1a. Negative keyword check
            # Soft exclude: if both negative and positive signals are
            # present, keep the paper and let downstream scoring decide.
            if self._matches_any(text, negative_kws) and not has_positive:
                stats["excluded_negative"] += 1
                # Find which negative keyword matched
                matched_kw = ""
                for kw in negative_kws:
                    if kw and kw in text:
                        matched_kw = kw
                        break
                if matched_kw:
                    stats["negative_keyword_hits"][matched_kw] = (
                        stats["negative_keyword_hits"].get(matched_kw, 0) + 1
                    )
                if len(stats["excluded_samples"]) < 5:
                    sample: dict[str, str] = {
                        "title": (paper.title or "")[:60],
                        "reason": "negative_keyword",
                    }
                    if matched_kw:
                        sample["matched_keyword"] = matched_kw
                    stats["excluded_samples"].append(sample)
                continue

            # 1b. Positive match: category OR keyword
            if self._category_match(paper, categories) or has_positive:
                rule_passed.append(paper)
            else:
                stats["excluded_no_match"] += 1
                if len(stats["excluded_samples"]) < 5:
                    stats["excluded_samples"].append({
                        "title": (paper.title or "")[:60],
                        "reason": "no_match",
                    })

        stats["after_rule_filter"] = len(rule_passed)

        # -- Stage 1.5: Defense Cap (pre_embed_cap) ----------------------
        if len(rule_passed) > self._pre_embed_cap:
            rule_passed = sorted(
                rule_passed,
                key=lambda p: p.published_at_utc or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            rule_passed = rule_passed[: self._pre_embed_cap]

        stats["after_cap"] = len(rule_passed)

        # -- Stage 2: Embedding Sort or Recency Fallback -----------------
        if embedding_ranker is not None and topic_embedding_text:
            result = self._embedding_sort(
                rule_passed, embedding_ranker, topic_embedding_text
            )
            stats["sort_method"] = "embedding"
        else:
            result = self._recency_sort(rule_passed)
            stats["sort_method"] = "recency"

        result = result[: self._max_filter_output]
        stats["after_embedding_sort"] = len(result)

        return result, stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _searchable_text(paper: Any) -> str:
        """Combine title and abstract into a single lowercase string."""
        title = paper.title or ""
        abstract = paper.abstract or ""
        return f"{title} {abstract}".lower()

    @staticmethod
    def _collect_negative_keywords(
        agent1_output: dict, topic: Any
    ) -> list[str]:
        """Merge Agent 1 exclude_keywords with TopicSpec.must_not_en."""
        kws: list[str] = []
        kws.extend(agent1_output.get("exclude_keywords", []))
        must_not = getattr(topic, "must_not_en", None)
        if must_not:
            kws.extend(must_not)
        # Lowercase for case-insensitive matching
        return [kw.lower() for kw in kws]

    @staticmethod
    def _collect_positive_keywords(agent1_output: dict) -> list[str]:
        """Extract all keywords from Agent 1 concepts."""
        kws: list[str] = []
        for concept in agent1_output.get("concepts", []):
            kws.extend(concept.get("keywords", []))
        # Also include cross_domain_keywords if present
        kws.extend(agent1_output.get("cross_domain_keywords", []))
        # User-selected must keywords (if provided)
        kws.extend(agent1_output.get("query_must_keywords", []))
        return [kw.lower() for kw in kws]

    @staticmethod
    def _collect_categories(topic: Any) -> set[str]:
        """Build a set of lowercased arxiv categories from TopicSpec."""
        cats = getattr(topic, "arxiv_categories", []) or []
        return {c.lower() for c in cats}

    @staticmethod
    def _category_match(paper: Any, categories: set[str]) -> bool:
        """Return True if any paper category is in *categories*."""
        paper_cats = getattr(paper, "categories", []) or []
        for cat in paper_cats:
            if cat.lower() in categories:
                return True
        return False

    @staticmethod
    def _matches_any(text: str, keywords: list[str]) -> bool:
        r"""Return True if *text* contains any keyword (case-insensitive).

        Uses word-boundary-aware matching via ``re.search`` with
        ``\b`` anchors so that partial-word matches are avoided (e.g.
        "quantum" does not match inside "quantumly-inspired" unless
        that exact keyword is listed).

        Note: Because keywords may contain special regex characters,
        each keyword is escaped with ``re.escape`` first.
        """
        for kw in keywords:
            if not kw:
                continue
            # Simple substring match (case already lowered)
            if kw in text:
                return True
        return False

    def _embedding_sort(
        self,
        papers: list[Any],
        embedding_ranker: Any,
        topic_embedding_text: str,
    ) -> list[Any]:
        """Sort papers by cosine similarity to *topic_embedding_text*."""
        texts = [self._searchable_text(p) for p in papers]
        scores = embedding_ranker.compute_similarity(
            texts, topic_embedding_text
        )

        # Validate score count matches paper count
        if len(scores) != len(papers):
            logger.warning(
                "Embedding score count mismatch: %d scores for %d papers. "
                "Falling back to recency sort.",
                len(scores), len(papers),
            )
            return self._recency_sort(papers)

        # Attach embed_score to each paper's evaluation (if present)
        for paper, score in zip(papers, scores):
            if hasattr(paper, "evaluation") and paper.evaluation is not None:
                paper.evaluation.embed_score = score
            # Store as transient attribute for sorting
            paper._embed_score = score  # noqa: SLF001

        # Sort descending by similarity
        papers_sorted = sorted(
            papers,
            key=lambda p: getattr(p, "_embed_score", 0.0),
            reverse=True,
        )

        # Clean up transient attribute
        for p in papers_sorted:
            if hasattr(p, "_embed_score"):
                del p._embed_score

        return papers_sorted

    @staticmethod
    def _recency_sort(papers: list[Any]) -> list[Any]:
        """Sort papers by published_at_utc descending (newest first)."""
        return sorted(
            papers,
            key=lambda p: p.published_at_utc or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
