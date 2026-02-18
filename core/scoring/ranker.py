"""Ranker module for Paper Scout (TASK-018).

Takes scored evaluations from Scorer and produces the final ranked list
with bonus scores, recency weighting, threshold relaxation, and tier
assignment.

Compatible with Python 3.9+.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.models import EvaluationFlags, Paper


# Recency score lookup: days_elapsed -> score.
_RECENCY_TABLE = {
    0: 100,
    1: 90,
    2: 80,
    3: 70,
    4: 60,
    5: 50,
    6: 40,
}
_RECENCY_DEFAULT = 30  # 7+ days


class Ranker:
    """Rank scored papers and produce enriched evaluation dicts.

    Args:
        scoring_config: The ``config['scoring']`` section from config.yaml.
    """

    def __init__(self, scoring_config: dict) -> None:
        self._cfg = scoring_config

        # Weights
        weights = scoring_config.get("weights", {})
        self._w_emb_on = weights.get("embedding_on", {})
        self._w_emb_off = weights.get("embedding_off", {})

        # Bonus values
        bonus_cfg = scoring_config.get("bonus", {})
        self._bonus_edge: int = bonus_cfg.get("is_edge", 5)
        self._bonus_realtime: int = bonus_cfg.get("is_realtime", 5)
        self._bonus_code: int = bonus_cfg.get("has_code", 3)

        # Thresholds
        thresh_cfg = scoring_config.get("thresholds", {})
        self._default_threshold: int = thresh_cfg.get("default", 60)
        self._relaxation_steps: List[int] = list(
            thresh_cfg.get("relaxation_steps", [50, 40])
        )
        self._min_papers: int = thresh_cfg.get("min_papers", 5)

        # Limits
        self._max_output: int = scoring_config.get("max_output", 100)
        self._tier1_count: int = scoring_config.get("tier1_count", 30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        evaluations: List[dict],
        papers: Dict[str, Paper],
        window_end: datetime,
        embedding_mode: str = "disabled",
    ) -> List[dict]:
        """Rank papers and return enriched evaluation dicts.

        Args:
            evaluations: List of evaluation dicts from Scorer. Each dict
                must contain at minimum ``paper_key``, ``llm_base_score``,
                ``flags`` (an ``EvaluationFlags``), ``embed_score``
                (float or None), and ``discarded`` (bool).
            papers: Mapping of paper_key to ``Paper`` objects.
            window_end: End of the collection window (UTC-aware datetime).
            embedding_mode: ``"disabled"`` or ``"en_synthetic"``.

        Returns:
            List of enriched evaluation dicts sorted by ``final_score``
            descending. Each dict has the original fields plus:
            ``bonus_score``, ``llm_adjusted``, ``final_score``, ``rank``,
            ``tier``, and ``score_lowered``.
        """
        if not evaluations:
            return []

        # Step 0: Filter out discarded papers.
        active = [e for e in evaluations if not e.get("discarded", False)]
        if not active:
            return []

        # Step 1-3: Calculate scores for each evaluation.
        scored: List[dict] = []
        for ev in active:
            paper_key: str = ev["paper_key"]
            paper = papers.get(paper_key)
            if paper is None:
                continue

            enriched = dict(ev)

            # Step 1: Deterministic bonus.
            flags: EvaluationFlags = ev["flags"]
            bonus = self._calculate_bonus(flags, paper)
            base_score: int = ev["llm_base_score"]
            llm_adjusted = min(base_score + bonus, 100)

            enriched["bonus_score"] = bonus
            enriched["llm_adjusted"] = llm_adjusted

            # Step 2: Recency score.
            recency = self._calculate_recency(paper.published_at_utc, window_end)

            # Step 3: Final score.
            final = self._calculate_final_score(
                llm_adjusted, ev.get("embed_score"), recency, embedding_mode
            )
            enriched["final_score"] = round(final, 4)

            scored.append(enriched)

        # Step 4: Sort descending by final_score.
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        # Step 4b: Threshold relaxation.
        threshold = self._default_threshold
        selected = [s for s in scored if s["final_score"] >= threshold]

        if len(selected) < self._min_papers:
            for step in self._relaxation_steps:
                threshold = step
                selected = [s for s in scored if s["final_score"] >= threshold]
                if len(selected) >= self._min_papers:
                    break

        # If even after all relaxation steps we don't have enough,
        # just use whatever we have at the lowest threshold.
        if not selected:
            selected = scored

        # Step 4c: Cap at max_output.
        selected = selected[: self._max_output]

        # Step 5 & 6: Assign rank, tier, and score_lowered.
        for idx, item in enumerate(selected):
            rank = idx + 1
            item["rank"] = rank
            item["tier"] = 1 if rank <= self._tier1_count else 2
            item["score_lowered"] = item["final_score"] < self._default_threshold

        return selected

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_bonus(self, flags: EvaluationFlags, paper: Paper) -> int:
        """Calculate deterministic bonus from flags and paper metadata."""
        bonus = 0
        if flags.is_edge:
            bonus += self._bonus_edge
        if flags.is_realtime:
            bonus += self._bonus_realtime
        if flags.mentions_code or paper.has_code:
            bonus += self._bonus_code
        return bonus

    @staticmethod
    def _calculate_recency(
        published_at_utc: datetime, window_end: datetime
    ) -> int:
        """Calculate recency score based on days elapsed."""
        delta = window_end - published_at_utc
        total_seconds = delta.total_seconds()
        days_elapsed = int(math.floor(total_seconds / 86400))  # 24 * 60 * 60
        if days_elapsed < 0:
            days_elapsed = 0
        return _RECENCY_TABLE.get(days_elapsed, _RECENCY_DEFAULT)

    def _calculate_final_score(
        self,
        llm_adjusted: int,
        embed_score: Optional[float],
        recency: int,
        embedding_mode: str,
    ) -> float:
        """Apply weighted formula to compute final score."""
        if embedding_mode != "disabled" and embed_score is not None:
            w = self._w_emb_on
            return (
                w.get("llm", 0.5) * llm_adjusted
                + w.get("embed", 0.3) * (embed_score * 100)
                + w.get("recency", 0.2) * recency
            )
        else:
            w = self._w_emb_off
            return w.get("llm", 0.7) * llm_adjusted + w.get("recency", 0.3) * recency
