"""Unified data loading context for weekly intelligence reports.

Loads all required data in a single pass: papers, evaluations, reminds,
previous snapshots, keyword matches, and product line classifications.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class WeeklyDataContext:
    """Pre-computed data context for weekly intelligence report generation.

    All DB queries execute in __init__ to minimize connection usage.
    The caller provides an open connection and placeholder style.
    """

    def __init__(
        self,
        conn: Any,
        ph: str,
        config: dict,
        reference_date: datetime,
    ) -> None:
        self.config = config
        self.reference_date = reference_date
        self.trend_weeks = config.get("trend_weeks", 4)

        # Compute ISO year/week for the reference date
        iso_cal = reference_date.isocalendar()
        self.iso_year = iso_cal[0]
        self.iso_week = iso_cal[1]

        # Compute week boundaries (Monday-based, 4 weeks back)
        self.week_boundaries = self._compute_week_boundaries(reference_date, self.trend_weeks)

        # Load all data
        self.papers_by_week: list[list[dict]] = self._load_papers(conn, ph)
        self.all_papers_deduped: list[dict] = self._deduplicate_all_papers()
        self.reminds: list[dict] = self._load_reminds(conn, ph)
        self.prev_snapshot: dict | None = self._load_prev_snapshot(conn, ph)

        # Keyword matching
        self.must_track_matches: dict[str, list[str]] = {}
        self.product_line_matches: dict[str, list[str]] = {}
        self.co_occurrence_matrix: dict[tuple[str, str], int] = {}
        self._run_keyword_matching()

    def _compute_week_boundaries(
        self, ref_date: datetime, num_weeks: int
    ) -> list[tuple[str, str]]:
        """Compute (start, end) date strings for each week going back."""
        # Find the Monday of the reference date's week
        weekday = ref_date.weekday()  # 0=Monday
        week_start = ref_date - timedelta(days=weekday)

        boundaries = []
        for _ in range(num_weeks):
            boundaries.append((
                week_start.strftime("%Y-%m-%d"),
                (week_start + timedelta(days=7)).strftime("%Y-%m-%d"),
            ))
            week_start -= timedelta(days=7)

        # Reverse so index 0 = oldest, index -1 = newest
        boundaries.reverse()
        return boundaries

    def _load_papers(self, conn: Any, ph: str) -> list[list[dict]]:
        """Load papers+evaluations for all weeks in a single query."""
        if not self.week_boundaries:
            return [[] for _ in range(self.trend_weeks)]

        oldest_start = self.week_boundaries[0][0]
        newest_end = self.week_boundaries[-1][1]

        cursor = conn.cursor()
        query = f"""
        SELECT
            p.paper_key, p.title, p.abstract, p.url, p.authors,
            p.categories, p.comment, p.pdf_url, p.has_code,
            pe.run_id, pe.final_score, pe.tier, pe.discarded,
            pe.summary_ko, pe.reason_ko, pe.insight_ko,
            pe.embed_score, pe.llm_base_score, pe.bonus_score,
            pe.is_remind, pe.multi_topic,
            r.topic_slug, r.window_start_utc
        FROM paper_evaluations pe
        JOIN papers p ON pe.paper_key = p.paper_key
        JOIN runs r ON pe.run_id = r.run_id
        WHERE r.window_start_utc >= {ph}
            AND r.window_start_utc < {ph}
            AND pe.discarded = 0
            AND r.status = 'completed'
        ORDER BY pe.final_score DESC
        """
        try:
            cursor.execute(query, (oldest_start, newest_end))
            rows = cursor.fetchall()
        except Exception:
            logger.warning("Failed to load papers for weekly intelligence", exc_info=True)
            return [[] for _ in range(self.trend_weeks)]

        # Distribute papers into weekly buckets
        papers_by_week: list[list[dict]] = [[] for _ in range(self.trend_weeks)]
        for row in rows:
            paper = self._row_to_dict(row)
            window_start = paper.get("window_start_utc", "")
            if isinstance(window_start, datetime):
                window_start = window_start.strftime("%Y-%m-%d")
            else:
                window_start = str(window_start)[:10]

            for i, (ws, we) in enumerate(self.week_boundaries):
                if ws <= window_start < we:
                    papers_by_week[i].append(paper)
                    break

        return papers_by_week

    def _row_to_dict(self, row: Any) -> dict:
        """Convert a DB row to a plain dict."""
        if isinstance(row, dict):
            return dict(row)
        # sqlite3.Row
        return {key: row[key] for key in row.keys()}

    def _deduplicate_all_papers(self) -> list[dict]:
        """Deduplicate across all weeks by paper_key, keeping highest score."""
        best: dict[str, dict] = {}
        for week_papers in self.papers_by_week:
            for paper in week_papers:
                pk = paper["paper_key"]
                score = paper.get("final_score", 0) or 0
                if pk not in best or score > (best[pk].get("final_score", 0) or 0):
                    best[pk] = paper
        return list(best.values())

    def _load_reminds(self, conn: Any, ph: str) -> list[dict]:
        """Load remind tracking entries with paper details."""
        if not self.week_boundaries:
            return []

        cursor = conn.cursor()
        query = """
        SELECT
            rt.paper_key, rt.topic_slug, rt.recommend_count,
            rt.last_recommend_run_id,
            p.title, p.url, p.authors,
            r.window_start_utc,
            pe.final_score
        FROM remind_tracking rt
        JOIN papers p ON rt.paper_key = p.paper_key
        JOIN runs r ON rt.last_recommend_run_id = r.run_id
        LEFT JOIN paper_evaluations pe
            ON rt.paper_key = pe.paper_key
            AND rt.last_recommend_run_id = pe.run_id
        ORDER BY rt.recommend_count DESC, pe.final_score DESC
        """
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        except Exception:
            logger.warning("Failed to load reminds", exc_info=True)
            return []

    def _load_prev_snapshot(self, conn: Any, ph: str) -> dict | None:
        """Load previous week's snapshot data."""
        # Calculate previous week's ISO year/week
        prev_date = self.reference_date - timedelta(days=7)
        prev_iso = prev_date.isocalendar()
        prev_year = prev_iso[0]
        prev_week = prev_iso[1]

        cursor = conn.cursor()
        query = f"""
        SELECT section, data_json
        FROM weekly_snapshots
        WHERE iso_year = {ph} AND iso_week = {ph}
        """
        try:
            cursor.execute(query, (prev_year, prev_week))
            rows = cursor.fetchall()
        except Exception:
            logger.warning("Failed to load prev snapshot (table may not exist yet)", exc_info=True)
            return None

        if not rows:
            return None

        snapshot: dict = {}
        for row in rows:
            if isinstance(row, dict):
                section = row["section"]
                data = row["data_json"]
            else:
                section = row[0]
                data = row[1]
            try:
                snapshot[section] = json.loads(data) if isinstance(data, str) else data
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse snapshot for section %s", section)

        return snapshot

    def _run_keyword_matching(self) -> None:
        """Match must_track keywords and product lines against papers."""
        kw_groups = self.config.get("must_track_keywords", {}).get("groups", {})
        product_lines = self.config.get("product_lines", {})

        # Compile all keyword patterns
        all_keywords: list[tuple[str, str, re.Pattern]] = []  # (group, kw, pattern)
        for group_name, keywords in kw_groups.items():
            for kw in keywords:
                pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                all_keywords.append((group_name, kw, pattern))

        pl_keywords: list[tuple[str, str, re.Pattern]] = []  # (line_name, kw, pattern)
        for line_name, keywords in product_lines.items():
            for kw in keywords:
                pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                pl_keywords.append((line_name, kw, pattern))

        # Match against all deduped papers
        co_occur_counter: Counter = Counter()

        for paper in self.all_papers_deduped:
            pk = paper["paper_key"]
            text = (paper.get("title", "") or "") + " " + (paper.get("abstract", "") or "")

            matched_kws: list[str] = []
            for group_name, kw, pattern in all_keywords:
                if pattern.search(text):
                    matched_kws.append(kw)

            if matched_kws:
                self.must_track_matches[pk] = matched_kws
                # Co-occurrence pairs
                for i in range(len(matched_kws)):
                    for j in range(i + 1, len(matched_kws)):
                        pair = tuple(sorted([matched_kws[i], matched_kws[j]]))
                        co_occur_counter[pair] += 1

            # Product line matching
            matched_lines: list[str] = []
            for line_name, kw, pattern in pl_keywords:
                if pattern.search(text):
                    if line_name not in matched_lines:
                        matched_lines.append(line_name)
            if matched_lines:
                self.product_line_matches[pk] = matched_lines

        self.co_occurrence_matrix = dict(co_occur_counter)
