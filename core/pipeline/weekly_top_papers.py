"""Section C: Top Papers for weekly intelligence report.

Selects top N papers by score with metadata enrichment and
graduated remind papers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_top_papers(ctx: Any) -> dict:
    """Build Section C: Top Papers + Graduated Reminds.

    Args:
        ctx: WeeklyDataContext with pre-loaded data.

    Returns:
        Dict with papers list and graduated_reminds list.
    """
    top_n = ctx.config.get("top_papers_count", 10)
    current_week = ctx.papers_by_week[-1] if ctx.papers_by_week else []

    # Deduplicate by paper_key (keep highest score)
    best: dict[str, dict] = {}
    for p in current_week:
        pk = p["paper_key"]
        score = p.get("final_score", 0) or 0
        if pk not in best or score > (best[pk].get("final_score", 0) or 0):
            best[pk] = p

    sorted_papers = sorted(
        best.values(), key=lambda x: x.get("final_score", 0) or 0, reverse=True
    )[:top_n]

    # Check prev snapshot for re-appeared papers
    prev_top_keys = set()
    if ctx.prev_snapshot and "top_papers" in ctx.prev_snapshot:
        prev_papers = ctx.prev_snapshot["top_papers"].get("papers", [])
        prev_top_keys = {p.get("paper_key", "") for p in prev_papers}

    papers_out = []
    for p in sorted_papers:
        authors_raw = p.get("authors", "[]")
        try:
            authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
        except (json.JSONDecodeError, TypeError):
            authors = []

        if isinstance(authors, list) and len(authors) > 3:
            authors_display = authors[:3] + ["et al."]
        elif isinstance(authors, list):
            authors_display = authors
        else:
            authors_display = []

        cats_raw = p.get("categories", "[]")
        try:
            categories = json.loads(cats_raw) if isinstance(cats_raw, str) else cats_raw
        except (json.JSONDecodeError, TypeError):
            categories = []

        papers_out.append({
            "paper_key": p["paper_key"],
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "final_score": p.get("final_score", 0),
            "topic_slug": p.get("topic_slug", ""),
            "authors": authors_display,
            "categories": categories if isinstance(categories, list) else [],
            "summary_ko": p.get("summary_ko"),
            "reason_ko": p.get("reason_ko"),
            "insight_ko": p.get("insight_ko"),
            "comment": p.get("comment", ""),
            "keyword_tags": ctx.must_track_matches.get(p["paper_key"], []),
            "re_appeared": p["paper_key"] in prev_top_keys,
        })

    # Graduated reminds (recommend_count = 2)
    graduated = []
    newest_start = ctx.week_boundaries[-1][0] if ctx.week_boundaries else ""
    newest_end = ctx.week_boundaries[-1][1] if ctx.week_boundaries else ""

    for r in ctx.reminds:
        if r.get("recommend_count", 0) != 2:
            continue
        grad_date = r.get("window_start_utc", "")
        if isinstance(grad_date, str):
            grad_date_str = grad_date[:10]
        else:
            grad_date_str = str(grad_date)[:10] if grad_date else ""

        # Only include if graduated this week
        if newest_start and newest_end and newest_start <= grad_date_str < newest_end:
            graduated.append({
                "paper_key": r.get("paper_key", ""),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "topic_slug": r.get("topic_slug", ""),
                "final_score": r.get("final_score"),
                "graduation_date": grad_date_str,
            })

    return {
        "papers": papers_out,
        "graduated_reminds": graduated,
    }
