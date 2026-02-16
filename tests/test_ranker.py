"""Tests for core.scoring.ranker – Ranker module (TASK-018).

Covers:
  1.  Bonus calculation (is_edge +5, is_realtime +5, has_code +3)
  2.  Bonus capping (llm_adjusted <= 100)
  3.  Recency score for all day buckets (0-7+)
  4.  Final score with embedding ON (3 weights)
  5.  Final score with embedding OFF (2 weights)
  6.  Sort order (descending by final_score)
  7.  Threshold relaxation step 1 (< 5 papers at 60 -> try 50)
  8.  Threshold relaxation step 2 (< 5 papers at 50 -> try 40)
  9.  No relaxation needed (>= 5 papers at threshold)
  10. score_lowered flag
  11. Tier 1 assignment (rank 1-30)
  12. Tier 2 assignment (rank 31-100)
  13. Max output 100
  14. Empty input
  15. All discarded
  16. Config weights used correctly
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

from core.models import Paper, EvaluationFlags
from core.scoring.ranker import Ranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_scoring_config() -> dict:
    """Return a minimal scoring config matching config.yaml structure."""
    return {
        "weights": {
            "embedding_on": {"llm": 0.55, "embed": 0.35, "recency": 0.10},
            "embedding_off": {"llm": 0.80, "recency": 0.20},
        },
        "bonus": {"is_edge": 5, "is_realtime": 5, "has_code": 3},
        "thresholds": {
            "default": 60,
            "relaxation_steps": [50, 40],
            "min_papers": 5,
        },
        "discard_cutoff": 20,
        "max_output": 100,
        "tier1_count": 30,
    }


def _make_paper(
    key: str,
    published_at_utc: datetime | None = None,
    has_code: bool = False,
) -> Paper:
    """Create a minimal Paper for testing."""
    if published_at_utc is None:
        published_at_utc = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    return Paper(
        source="arxiv",
        native_id=key,
        paper_key=f"arxiv:{key}",
        url=f"http://arxiv.org/abs/{key}",
        title=f"Paper {key}",
        abstract="Abstract text",
        authors=["Author A"],
        categories=["cs.AI"],
        published_at_utc=published_at_utc,
        has_code=has_code,
    )


def _make_eval(
    paper_key: str,
    base_score: int,
    is_edge: bool = False,
    is_realtime: bool = False,
    mentions_code: bool = False,
    embed_score: float | None = None,
    discarded: bool = False,
) -> dict:
    """Create an evaluation dict as would come from Scorer."""
    return {
        "paper_key": paper_key,
        "llm_base_score": base_score,
        "flags": EvaluationFlags(
            is_edge=is_edge,
            is_realtime=is_realtime,
            mentions_code=mentions_code,
        ),
        "embed_score": embed_score,
        "discarded": discarded,
    }


# ---------------------------------------------------------------------------
# 1. Bonus calculation
# ---------------------------------------------------------------------------

class TestBonusCalculation:
    """Verify deterministic bonus from flags."""

    def test_is_edge_adds_5(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 70, is_edge=True)]

        results = ranker.rank(evals, papers, window_end)
        assert len(results) == 1
        assert results[0]["bonus_score"] == 5

    def test_is_realtime_adds_5(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 70, is_realtime=True)]

        results = ranker.rank(evals, papers, window_end)
        assert results[0]["bonus_score"] == 5

    def test_has_code_adds_3(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end, has_code=True)}
        evals = [_make_eval("arxiv:001", 70, mentions_code=True)]

        results = ranker.rank(evals, papers, window_end)
        assert results[0]["bonus_score"] == 3

    def test_all_bonuses_combined(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end, has_code=True)}
        evals = [_make_eval("arxiv:001", 70, is_edge=True, is_realtime=True, mentions_code=True)]

        results = ranker.rank(evals, papers, window_end)
        # 5 + 5 + 3 = 13
        assert results[0]["bonus_score"] == 13


# ---------------------------------------------------------------------------
# 2. Bonus capping
# ---------------------------------------------------------------------------

class TestBonusCapping:
    """llm_adjusted = min(base + bonus, 100)."""

    def test_llm_adjusted_capped_at_100(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end, has_code=True)}
        evals = [_make_eval("arxiv:001", 95, is_edge=True, is_realtime=True, mentions_code=True)]

        results = ranker.rank(evals, papers, window_end)
        # 95 + 13 = 108 -> capped to 100
        assert results[0]["llm_adjusted"] == 100

    def test_llm_adjusted_not_capped_when_below(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 60, is_edge=True)]

        results = ranker.rank(evals, papers, window_end)
        assert results[0]["llm_adjusted"] == 65  # 60 + 5


# ---------------------------------------------------------------------------
# 3. Recency score
# ---------------------------------------------------------------------------

class TestRecencyScore:
    """Recency score mapped by days_elapsed from window_end."""

    @pytest.mark.parametrize(
        "days_elapsed, expected_recency",
        [
            (0, 100),
            (1, 90),
            (2, 80),
            (3, 70),
            (4, 60),
            (5, 50),
            (6, 40),
            (7, 30),
            (10, 30),   # 7+ all map to 30
            (30, 30),
        ],
    )
    def test_recency_by_day(self, days_elapsed, expected_recency):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        pub_date = window_end - timedelta(days=days_elapsed)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=pub_date)}
        evals = [_make_eval("arxiv:001", 80)]

        results = ranker.rank(evals, papers, window_end)
        # Embedding off: final = 0.80 * llm_adjusted + 0.20 * recency
        expected_final = 0.80 * 80 + 0.20 * expected_recency
        assert abs(results[0]["final_score"] - expected_final) < 0.01


# ---------------------------------------------------------------------------
# 4. Final score with embedding ON
# ---------------------------------------------------------------------------

class TestFinalScoreEmbeddingOn:
    """final = 0.55 * llm_adjusted + 0.35 * (embed*100) + 0.10 * recency."""

    def test_formula_with_embedding(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 80, embed_score=0.75)]

        results = ranker.rank(evals, papers, window_end, embedding_mode="en_synthetic")

        # recency = 100 (same day)
        # llm_adjusted = 80 (no bonus)
        # final = 0.55 * 80 + 0.35 * 75 + 0.10 * 100 = 44 + 26.25 + 10 = 80.25
        assert abs(results[0]["final_score"] - 80.25) < 0.01


# ---------------------------------------------------------------------------
# 5. Final score with embedding OFF
# ---------------------------------------------------------------------------

class TestFinalScoreEmbeddingOff:
    """final = 0.80 * llm_adjusted + 0.20 * recency."""

    def test_formula_without_embedding(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 80)]

        results = ranker.rank(evals, papers, window_end, embedding_mode="disabled")

        # recency = 100 (same day)
        # final = 0.80 * 80 + 0.20 * 100 = 64 + 20 = 84.0
        assert abs(results[0]["final_score"] - 84.0) < 0.01


# ---------------------------------------------------------------------------
# 6. Sort order
# ---------------------------------------------------------------------------

class TestSortOrder:
    """Results sorted descending by final_score."""

    def test_descending_order(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {
            "arxiv:001": _make_paper("001", published_at_utc=window_end),
            "arxiv:002": _make_paper("002", published_at_utc=window_end),
            "arxiv:003": _make_paper("003", published_at_utc=window_end),
        }
        evals = [
            _make_eval("arxiv:001", 50),
            _make_eval("arxiv:002", 90),
            _make_eval("arxiv:003", 70),
        ]

        results = ranker.rank(evals, papers, window_end)

        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["paper_key"] == "arxiv:002"
        assert results[1]["paper_key"] == "arxiv:003"


# ---------------------------------------------------------------------------
# 7. Threshold relaxation step 1
# ---------------------------------------------------------------------------

class TestThresholdRelaxationStep1:
    """If < min_papers at default (60), relax to 50."""

    def test_relaxation_to_50(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        # 3 papers with scores that produce final_score >= 60 (embedding off)
        # 2 papers with final_score between 50-59
        # final = 0.80 * llm_adjusted + 0.20 * 100 (same day)
        # score 70 -> final = 76, score 65 -> final = 72, score 60 -> final = 68
        # score 45 -> final = 56, score 43 -> final = 54.4
        papers = {}
        evals = []
        for i, base in enumerate([70, 65, 60, 45, 43]):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, base))

        results = ranker.rank(evals, papers, window_end)

        # With threshold 60: 3 papers qualify (final >= 60: 76, 72, 68)
        # Relaxation to 50: adds 2 more (56, 54.4) -> total 5, >= min_papers
        assert len(results) >= 5
        # Papers added via relaxation should have score_lowered=True
        lowered = [r for r in results if r.get("score_lowered")]
        assert len(lowered) >= 2


# ---------------------------------------------------------------------------
# 8. Threshold relaxation step 2
# ---------------------------------------------------------------------------

class TestThresholdRelaxationStep2:
    """If < min_papers at 50, relax to 40."""

    def test_relaxation_to_40(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        # 2 papers >= 60, 1 paper in [50, 60), 2 papers in [40, 50)
        # final = 0.80 * base + 0.20 * 100
        # base 70 -> 76, base 62 -> 69.6
        # base 40 -> 52, (in [50,60) range)
        # base 30 -> 44, base 28 -> 42.4 (in [40,50) range)
        papers = {}
        evals = []
        for i, base in enumerate([70, 62, 40, 30, 28]):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, base))

        results = ranker.rank(evals, papers, window_end)

        # threshold 60: 2 qualify (76, 69.6)
        # threshold 50: 3 qualify (adds 52) -> still < 5
        # threshold 40: 5 qualify (adds 44, 42.4) -> total 5
        assert len(results) >= 5


# ---------------------------------------------------------------------------
# 9. No relaxation needed
# ---------------------------------------------------------------------------

class TestNoRelaxation:
    """If >= min_papers at default threshold, no relaxation."""

    def test_enough_papers_at_default(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        papers = {}
        evals = []
        for i in range(10):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, 80))  # final = 84 for all

        results = ranker.rank(evals, papers, window_end)

        assert len(results) == 10
        for r in results:
            assert r.get("score_lowered") is False


# ---------------------------------------------------------------------------
# 10. score_lowered flag
# ---------------------------------------------------------------------------

class TestScoreLoweredFlag:
    """Papers included via relaxation get score_lowered=True, others False."""

    def test_flag_assignment(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        # 3 papers above 60, 2 below 60 but above 50
        papers = {}
        evals = []
        for i, base in enumerate([80, 75, 70, 45, 42]):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, base))

        results = ranker.rank(evals, papers, window_end)

        for r in results:
            final = r["final_score"]
            if final >= 60:
                assert r["score_lowered"] is False
            else:
                assert r["score_lowered"] is True


# ---------------------------------------------------------------------------
# 11. Tier 1 assignment (rank 1-30)
# ---------------------------------------------------------------------------

class TestTier1:
    """Rank 1-30 get tier=1."""

    def test_tier1_assignment(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        papers = {}
        evals = []
        for i in range(35):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, 90 - i))  # distinct scores

        results = ranker.rank(evals, papers, window_end)

        for r in results:
            if r["rank"] <= 30:
                assert r["tier"] == 1


# ---------------------------------------------------------------------------
# 12. Tier 2 assignment (rank 31-100)
# ---------------------------------------------------------------------------

class TestTier2:
    """Rank 31-100 get tier=2."""

    def test_tier2_assignment(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        papers = {}
        evals = []
        for i in range(35):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, 90 - i))

        results = ranker.rank(evals, papers, window_end)

        for r in results:
            if r["rank"] > 30:
                assert r["tier"] == 2


# ---------------------------------------------------------------------------
# 13. Max output 100
# ---------------------------------------------------------------------------

class TestMaxOutput:
    """Only top max_output (100) papers returned."""

    def test_max_output_cap(self):
        cfg = _default_scoring_config()
        cfg["max_output"] = 10  # use smaller cap for test speed
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        papers = {}
        evals = []
        for i in range(20):
            key = f"arxiv:{i:03d}"
            papers[key] = _make_paper(f"{i:03d}", published_at_utc=window_end)
            evals.append(_make_eval(key, 90))

        results = ranker.rank(evals, papers, window_end)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# 14. Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Ranker returns empty list for empty evaluations."""

    def test_empty_evaluations(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        results = ranker.rank([], {}, window_end)
        assert results == []


# ---------------------------------------------------------------------------
# 15. All discarded
# ---------------------------------------------------------------------------

class TestAllDiscarded:
    """All discarded papers -> empty result."""

    def test_all_discarded(self):
        cfg = _default_scoring_config()
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

        papers = {
            "arxiv:001": _make_paper("001", published_at_utc=window_end),
            "arxiv:002": _make_paper("002", published_at_utc=window_end),
        }
        evals = [
            _make_eval("arxiv:001", 80, discarded=True),
            _make_eval("arxiv:002", 90, discarded=True),
        ]

        results = ranker.rank(evals, papers, window_end)
        assert results == []


# ---------------------------------------------------------------------------
# 16. Config weights
# ---------------------------------------------------------------------------

class TestConfigWeights:
    """Ranker uses weights from config, not hardcoded values."""

    def test_custom_weights(self):
        cfg = _default_scoring_config()
        cfg["weights"]["embedding_off"] = {"llm": 0.50, "recency": 0.50}
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 80)]

        results = ranker.rank(evals, papers, window_end, embedding_mode="disabled")

        # recency = 100 (same day), llm_adjusted = 80
        # final = 0.50 * 80 + 0.50 * 100 = 40 + 50 = 90.0
        assert abs(results[0]["final_score"] - 90.0) < 0.01

    def test_custom_embedding_on_weights(self):
        cfg = _default_scoring_config()
        cfg["weights"]["embedding_on"] = {"llm": 0.40, "embed": 0.40, "recency": 0.20}
        ranker = Ranker(cfg)
        window_end = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        papers = {"arxiv:001": _make_paper("001", published_at_utc=window_end)}
        evals = [_make_eval("arxiv:001", 80, embed_score=0.50)]

        results = ranker.rank(evals, papers, window_end, embedding_mode="en_synthetic")

        # recency = 100, llm_adjusted = 80, embed_score*100 = 50
        # final = 0.40 * 80 + 0.40 * 50 + 0.20 * 100 = 32 + 20 + 20 = 72.0
        assert abs(results[0]["final_score"] - 72.0) < 0.01
