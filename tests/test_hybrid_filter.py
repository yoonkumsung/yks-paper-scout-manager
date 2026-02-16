"""Tests for core.scoring.hybrid_filter.

Covers the three-stage Hybrid Filter: rule-based filtering (negative
keyword exclusion, category/keyword matching), defense cap, embedding
sort, recency fallback, statistics tracking, and edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import MagicMock

import pytest

from core.scoring.hybrid_filter import HybridFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeTopicSpec:
    """Minimal TopicSpec stand-in for tests."""

    slug: str = "test-topic"
    name: str = "Test Topic"
    description: str = ""
    arxiv_categories: list = field(default_factory=list)
    must_not_en: Optional[list] = None


@dataclass
class _FakeEvaluation:
    """Minimal Evaluation stand-in for embedding score storage."""

    embed_score: Optional[float] = None


@dataclass
class _FakePaper:
    """Minimal Paper stand-in for tests."""

    title: str = ""
    abstract: str = ""
    categories: list = field(default_factory=list)
    published_at_utc: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    evaluation: Optional[_FakeEvaluation] = None


def _make_paper(
    title: str = "A Paper",
    abstract: str = "Some abstract text.",
    categories: Optional[list] = None,
    days_ago: int = 0,
    evaluation: Optional[_FakeEvaluation] = None,
) -> _FakePaper:
    """Create a _FakePaper with convenient defaults."""
    pub = datetime(2025, 6, 1, tzinfo=timezone.utc) - timedelta(days=days_ago)
    return _FakePaper(
        title=title,
        abstract=abstract,
        categories=categories or [],
        published_at_utc=pub,
        evaluation=evaluation,
    )


def _default_agent1(
    keywords: Optional[list] = None,
    exclude: Optional[list] = None,
) -> dict:
    """Return a minimal Agent 1 output dict."""
    return {
        "concepts": [{"keywords": keywords or []}],
        "cross_domain_keywords": [],
        "exclude_keywords": exclude or [],
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def hfilter() -> HybridFilter:
    """Provide a HybridFilter with default config."""
    return HybridFilter({"pre_embed_cap": 2000, "max_filter_output": 200})


@pytest.fixture()
def topic() -> _FakeTopicSpec:
    """Provide a default TopicSpec."""
    return _FakeTopicSpec(arxiv_categories=["cs.AI", "cs.LG"])


# ---------------------------------------------------------------------------
# 1. Negative keyword exclusion
# ---------------------------------------------------------------------------


class TestNegativeKeywordExclusion:
    """Papers matching exclude keywords are removed."""

    def test_exclude_keyword_in_title(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Paper whose title contains an exclude keyword is removed."""
        papers = [
            _make_paper(
                title="Survey of biology methods",
                categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1(exclude=["biology"])
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_negative"] == 1

    def test_exclude_keyword_in_abstract(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Paper whose abstract contains an exclude keyword is removed."""
        papers = [
            _make_paper(
                abstract="This paper discusses biology of cells.",
                categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1(exclude=["biology"])
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_negative"] == 1


# ---------------------------------------------------------------------------
# 2. Category matching
# ---------------------------------------------------------------------------


class TestCategoryMatching:
    """Papers with matching categories pass the rule filter."""

    def test_matching_category_passes(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Paper with a matching arxiv category passes."""
        papers = [_make_paper(categories=["cs.AI"])]
        agent1 = _default_agent1()
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1

    def test_non_matching_category_excluded(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Paper with no matching category or keyword is excluded."""
        papers = [_make_paper(categories=["astro-ph.SR"])]
        agent1 = _default_agent1()
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_no_match"] == 1


# ---------------------------------------------------------------------------
# 3. Keyword matching
# ---------------------------------------------------------------------------


class TestKeywordMatching:
    """Papers with matching keywords in abstract/title pass."""

    def test_keyword_in_title_passes(self, hfilter: HybridFilter):
        """Paper matching a positive keyword in title passes."""
        topic = _FakeTopicSpec(arxiv_categories=[])
        papers = [
            _make_paper(title="Deep reinforcement learning for robotics"),
        ]
        agent1 = _default_agent1(keywords=["reinforcement learning"])
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1

    def test_keyword_in_abstract_passes(self, hfilter: HybridFilter):
        """Paper matching a positive keyword in abstract passes."""
        topic = _FakeTopicSpec(arxiv_categories=[])
        papers = [
            _make_paper(abstract="We use transformer architectures."),
        ]
        agent1 = _default_agent1(keywords=["transformer"])
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. Combined negative + positive (negative takes priority)
# ---------------------------------------------------------------------------


class TestNegativePriority:
    """Negative keywords take priority over positive matches."""

    def test_negative_overrides_positive(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Paper matching both exclude and positive keyword is excluded."""
        papers = [
            _make_paper(
                title="Reinforcement learning in biology",
                categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1(
            keywords=["reinforcement learning"],
            exclude=["biology"],
        )
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_negative"] == 1


# ---------------------------------------------------------------------------
# 5. Case-insensitive matching
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    """Both negative and positive matching is case-insensitive."""

    def test_negative_case_insensitive(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Exclude keyword 'BIOLOGY' matches 'biology' in text."""
        papers = [
            _make_paper(
                abstract="This involves biology methods.",
                categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1(exclude=["BIOLOGY"])
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_negative"] == 1

    def test_positive_case_insensitive(self, hfilter: HybridFilter):
        """Positive keyword 'TRANSFORMER' matches 'Transformer' in title."""
        topic = _FakeTopicSpec(arxiv_categories=[])
        papers = [
            _make_paper(title="Transformer Models for NLP"),
        ]
        agent1 = _default_agent1(keywords=["TRANSFORMER"])
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1

    def test_category_case_insensitive(self, hfilter: HybridFilter):
        """Category matching is case-insensitive."""
        topic = _FakeTopicSpec(arxiv_categories=["CS.AI"])
        papers = [_make_paper(categories=["cs.ai"])]
        agent1 = _default_agent1()
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 6. Pre-embed cap
# ---------------------------------------------------------------------------


class TestPreEmbedCap:
    """When papers exceed cap, only newest kept."""

    def test_cap_trims_to_limit(self, topic: _FakeTopicSpec):
        """Papers exceeding pre_embed_cap are trimmed to newest."""
        hf = HybridFilter({"pre_embed_cap": 3, "max_filter_output": 200})
        papers = [
            _make_paper(title=f"Paper {i}", categories=["cs.AI"], days_ago=i)
            for i in range(5)
        ]
        agent1 = _default_agent1()
        result, stats = hf.filter(papers, agent1, topic)
        assert stats["after_rule_filter"] == 5
        assert stats["after_cap"] == 3
        # Newest papers kept (days_ago 0, 1, 2)
        for p in result:
            assert p.title in ("Paper 0", "Paper 1", "Paper 2")

    def test_under_cap_no_trim(self, hfilter: HybridFilter, topic: _FakeTopicSpec):
        """Papers under cap are not trimmed."""
        papers = [
            _make_paper(categories=["cs.AI"], days_ago=i) for i in range(3)
        ]
        agent1 = _default_agent1()
        _, stats = hfilter.filter(papers, agent1, topic)
        assert stats["after_cap"] == 3


# ---------------------------------------------------------------------------
# 7. Embedding sort
# ---------------------------------------------------------------------------


class TestEmbeddingSort:
    """When embedding_ranker provided, sort by similarity."""

    def test_embedding_sort_order(self, topic: _FakeTopicSpec):
        """Papers sorted by embedding similarity descending."""
        hf = HybridFilter({"pre_embed_cap": 2000, "max_filter_output": 200})

        papers = [
            _make_paper(title="Low relevance", categories=["cs.AI"]),
            _make_paper(title="High relevance", categories=["cs.AI"]),
            _make_paper(title="Medium relevance", categories=["cs.AI"]),
        ]

        # Mock embedding ranker: returns scores in input order
        ranker = MagicMock()
        ranker.compute_similarity.return_value = [0.2, 0.9, 0.5]

        agent1 = _default_agent1()
        result, _ = hf.filter(
            papers, agent1, topic,
            embedding_ranker=ranker,
            topic_embedding_text="AI research",
        )
        assert len(result) == 3
        assert result[0].title == "High relevance"
        assert result[1].title == "Medium relevance"
        assert result[2].title == "Low relevance"

    def test_embedding_score_stored_on_evaluation(self, topic: _FakeTopicSpec):
        """embed_score is stored on paper.evaluation when present."""
        hf = HybridFilter({"pre_embed_cap": 2000, "max_filter_output": 200})

        ev = _FakeEvaluation()
        papers = [
            _make_paper(
                title="Paper A", categories=["cs.AI"], evaluation=ev
            ),
        ]

        ranker = MagicMock()
        ranker.compute_similarity.return_value = [0.75]

        agent1 = _default_agent1()
        result, _ = hf.filter(
            papers, agent1, topic,
            embedding_ranker=ranker,
            topic_embedding_text="query",
        )
        assert result[0].evaluation.embed_score == 0.75


# ---------------------------------------------------------------------------
# 8. Recency fallback
# ---------------------------------------------------------------------------


class TestRecencyFallback:
    """When no embedding_ranker, sort by published_at_utc."""

    def test_recency_sort_order(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Papers sorted newest-first when no embedding ranker."""
        papers = [
            _make_paper(title="Oldest", categories=["cs.AI"], days_ago=10),
            _make_paper(title="Newest", categories=["cs.AI"], days_ago=0),
            _make_paper(title="Middle", categories=["cs.AI"], days_ago=5),
        ]
        agent1 = _default_agent1()
        result, _ = hfilter.filter(papers, agent1, topic)
        assert result[0].title == "Newest"
        assert result[1].title == "Middle"
        assert result[2].title == "Oldest"


# ---------------------------------------------------------------------------
# 9. Max filter output
# ---------------------------------------------------------------------------


class TestMaxFilterOutput:
    """Only top max_filter_output papers returned."""

    def test_output_truncated_to_max(self, topic: _FakeTopicSpec):
        """Output limited to max_filter_output (e.g. 5)."""
        hf = HybridFilter({"pre_embed_cap": 2000, "max_filter_output": 5})
        papers = [
            _make_paper(
                title=f"Paper {i}", categories=["cs.AI"], days_ago=i
            )
            for i in range(10)
        ]
        agent1 = _default_agent1()
        result, stats = hf.filter(papers, agent1, topic)
        assert len(result) == 5
        assert stats["after_embedding_sort"] == 5


# ---------------------------------------------------------------------------
# 10. Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Returns empty list gracefully."""

    def test_empty_papers_list(
        self, hfilter: HybridFilter, topic: _FakeTopicSpec
    ):
        """Empty input returns empty result with zero stats."""
        agent1 = _default_agent1()
        result, stats = hfilter.filter([], agent1, topic)
        assert result == []
        assert stats["total_input"] == 0
        assert stats["after_rule_filter"] == 0
        assert stats["after_cap"] == 0
        assert stats["after_embedding_sort"] == 0


# ---------------------------------------------------------------------------
# 11. No concepts / no keywords
# ---------------------------------------------------------------------------


class TestNoConceptsKeywords:
    """Papers pass by category only when no keywords provided."""

    def test_category_only_pass(self, hfilter: HybridFilter):
        """Paper with matching category passes when no keywords."""
        topic = _FakeTopicSpec(arxiv_categories=["cs.AI"])
        papers = [_make_paper(categories=["cs.AI"])]
        agent1: dict = {
            "concepts": [],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1

    def test_no_category_no_keyword_excluded(self, hfilter: HybridFilter):
        """Paper with no matching category and no keywords is excluded."""
        topic = _FakeTopicSpec(arxiv_categories=["cs.AI"])
        papers = [_make_paper(categories=["math.CO"])]
        agent1: dict = {
            "concepts": [],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_no_match"] == 1


# ---------------------------------------------------------------------------
# 12. Filter stats
# ---------------------------------------------------------------------------


class TestFilterStats:
    """Correct counts at each stage."""

    def test_stats_comprehensive(self, topic: _FakeTopicSpec):
        """Stats correctly track each filtering stage."""
        hf = HybridFilter({"pre_embed_cap": 2000, "max_filter_output": 200})

        papers = [
            # Passes: matching category
            _make_paper(title="Good paper", categories=["cs.AI"]),
            # Excluded: negative keyword
            _make_paper(
                title="Biology paper", categories=["cs.AI"],
            ),
            # Excluded: no match (wrong category, no keyword match)
            _make_paper(
                title="Astronomy paper", categories=["astro-ph.SR"],
            ),
            # Passes: keyword match
            _make_paper(
                title="Neural network paper", categories=["stat.ML"],
            ),
        ]

        agent1 = _default_agent1(
            keywords=["neural network"],
            exclude=["biology"],
        )
        result, stats = hf.filter(papers, agent1, topic)

        assert stats["total_input"] == 4
        assert stats["excluded_negative"] == 1
        assert stats["excluded_no_match"] == 1
        assert stats["after_rule_filter"] == 2
        assert stats["after_cap"] == 2
        assert stats["after_embedding_sort"] == 2
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 13. must_not_en integration
# ---------------------------------------------------------------------------


class TestMustNotEnIntegration:
    """TopicSpec.must_not_en merged with agent1 exclude_keywords."""

    def test_must_not_en_excludes_paper(self, hfilter: HybridFilter):
        """must_not_en keyword from TopicSpec excludes papers."""
        topic = _FakeTopicSpec(
            arxiv_categories=["cs.AI"],
            must_not_en=["quantum"],
        )
        papers = [
            _make_paper(
                title="Quantum computing survey", categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1()
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_negative"] == 1

    def test_must_not_en_merged_with_agent1(self, hfilter: HybridFilter):
        """Both must_not_en and agent1 exclude_keywords are applied."""
        topic = _FakeTopicSpec(
            arxiv_categories=["cs.AI"],
            must_not_en=["quantum"],
        )
        papers = [
            _make_paper(
                title="Quantum paper", categories=["cs.AI"],
            ),
            _make_paper(
                title="Biology paper", categories=["cs.AI"],
            ),
            _make_paper(
                title="Good ML paper", categories=["cs.AI"],
            ),
        ]
        agent1 = _default_agent1(exclude=["biology"])
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1
        assert result[0].title == "Good ML paper"
        assert stats["excluded_negative"] == 2

    def test_must_not_en_none_handled(self, hfilter: HybridFilter):
        """TopicSpec with must_not_en=None is handled gracefully."""
        topic = _FakeTopicSpec(
            arxiv_categories=["cs.AI"],
            must_not_en=None,
        )
        papers = [_make_paper(categories=["cs.AI"])]
        agent1 = _default_agent1()
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 14. Papers with no categories
# ---------------------------------------------------------------------------


class TestNoCategoryPapers:
    """Papers with no categories can still pass by keyword match."""

    def test_no_categories_keyword_pass(self, hfilter: HybridFilter):
        """Paper with empty categories but matching keyword passes."""
        topic = _FakeTopicSpec(arxiv_categories=["cs.AI"])
        papers = [
            _make_paper(
                title="Transformer attention mechanisms",
                categories=[],
            ),
        ]
        agent1 = _default_agent1(keywords=["transformer"])
        result, _ = hfilter.filter(papers, agent1, topic)
        assert len(result) == 1

    def test_no_categories_no_keyword_excluded(self, hfilter: HybridFilter):
        """Paper with empty categories and no keyword match is excluded."""
        topic = _FakeTopicSpec(arxiv_categories=["cs.AI"])
        papers = [
            _make_paper(
                title="A generic paper",
                categories=[],
            ),
        ]
        agent1 = _default_agent1()
        result, stats = hfilter.filter(papers, agent1, topic)
        assert len(result) == 0
        assert stats["excluded_no_match"] == 1
