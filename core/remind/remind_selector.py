"""Remind (re-recommendation) selection module for Paper Scout.

Selects previously high-scoring papers for re-recommendation, respecting
topic isolation, recommend count limits, and current-run exclusion.
Summaries are reused from the most recent Evaluation (no Agent 3 re-call).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from core.models import RemindTracking

if TYPE_CHECKING:
    from core.storage.db_manager import DBManager


class RemindSelector:
    """Select papers for re-recommendation from previous runs."""

    def __init__(self, db: DBManager) -> None:
        self._db = db

    def select(
        self,
        topic_slug: str,
        current_run_id: int,
        min_score: float = 80.0,
        max_recommend_count: int = 2,
    ) -> list[dict]:
        """Select remind papers for a topic.

        Returns list of dicts with paper info + existing summaries.
        Each dict contains:
          - paper_key, title, url, pdf_url, has_code, code_url,
            categories, published_at_utc
          - final_score, recommend_count (incremented)
          - summary_ko, reason_ko, insight_ko (reused from last eval)
          - is_remind: True

        Args:
            topic_slug: The topic to select remind papers for.
            current_run_id: The current run ID; papers from this run
                are excluded (they are today's new papers).
            min_score: Minimum final_score threshold (default 80.0).
            max_recommend_count: Maximum times a paper can be reminded
                before graduating (default 2).

        Returns:
            A list of remind paper dicts, sorted by final_score descending.
        """
        candidates = self._get_candidates(
            topic_slug, min_score, max_recommend_count, current_run_id
        )

        remind_papers: list[dict] = []
        for paper_key, final_score in candidates:
            paper = self._db.get_paper(paper_key)
            if paper is None:
                continue

            evaluation = self._db.get_latest_evaluation(paper_key, topic_slug)
            if evaluation is None:
                continue

            # Get or initialize tracking record
            tracking = self._db.get_remind_tracking(paper_key, topic_slug)
            if tracking is None:
                tracking = RemindTracking(
                    paper_key=paper_key,
                    topic_slug=topic_slug,
                    last_recommend_run_id=evaluation.run_id,
                    recommend_count=0,
                )

            # Increment recommend_count
            tracking.recommend_count += 1
            tracking.last_recommend_run_id = current_run_id

            # Persist the updated tracking
            self._db.upsert_remind_tracking(tracking)

            remind_dict = self._build_remind_paper(
                paper, evaluation, tracking
            )
            remind_papers.append(remind_dict)

        # Sort by final_score descending
        remind_papers.sort(key=lambda d: d["final_score"], reverse=True)
        return remind_papers

    def _get_candidates(
        self,
        topic_slug: str,
        min_score: float,
        max_count: int,
        current_run_id: int,
    ) -> list[tuple[str, float]]:
        """Collect candidate paper_keys for remind selection.

        Merges two sources:
        1. Papers already in remind_tracking (via get_remind_candidates)
        2. High-score papers not yet tracked (first-time candidates)

        Filters out:
        - Papers from the current run
        - Papers with recommend_count >= max_count
        - Discarded papers (handled by DB queries)

        Returns:
            List of (paper_key, final_score) tuples, deduplicated.
        """
        seen: set[str] = set()
        result: list[tuple[str, float]] = []

        # Source 1: Already-tracked candidates
        tracked = self._db.get_remind_candidates(
            topic_slug, min_score, max_count
        )
        for row in tracked:
            pk = row["paper_key"]
            # Skip current-run papers
            if row.get("last_recommend_run_id") == current_run_id:
                continue
            if pk not in seen:
                seen.add(pk)
                result.append((pk, row["final_score"]))

        # Source 2: High-score evaluations not yet in remind_tracking
        high_score_evals = self._db.get_high_score_papers(
            topic_slug, min_score
        )
        for ev in high_score_evals:
            # Skip current run papers
            if ev.run_id == current_run_id:
                continue
            if ev.paper_key in seen:
                continue
            # Check if already tracked
            tracking = self._db.get_remind_tracking(
                ev.paper_key, topic_slug
            )
            if tracking is not None and tracking.recommend_count >= max_count:
                continue
            seen.add(ev.paper_key)
            result.append((ev.paper_key, ev.final_score))  # type: ignore[arg-type]

        return result

    @staticmethod
    def _build_remind_paper(
        paper: "Paper",  # noqa: F821
        evaluation: "Evaluation",  # noqa: F821
        tracking: RemindTracking,
    ) -> dict:
        """Build a remind paper dict from Paper + Evaluation + RemindTracking.

        The summary fields are reused from the most recent Evaluation
        so that Agent 3 is NOT re-called.
        """
        # published_at_utc may be a datetime; convert to ISO date string
        pub_date = paper.published_at_utc
        if hasattr(pub_date, "strftime"):
            pub_str = pub_date.strftime("%Y-%m-%d")
        else:
            pub_str = str(pub_date)

        return {
            "paper_key": paper.paper_key,
            "title": paper.title,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "has_code": paper.has_code,
            "code_url": paper.code_url,
            "categories": paper.categories,
            "published_at_utc": pub_str,
            "final_score": evaluation.final_score,
            "recommend_count": tracking.recommend_count,
            "summary_ko": evaluation.summary_ko,
            "reason_ko": evaluation.reason_ko,
            "insight_ko": evaluation.insight_ko,
            "is_remind": True,
        }
