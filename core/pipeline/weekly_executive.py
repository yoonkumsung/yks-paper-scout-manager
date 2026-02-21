"""Section A: Executive Summary for weekly intelligence report.

Computes key metrics, WoW changes, 4-week trends, category breakdown,
and remind status from WeeklyDataContext.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def build_executive_summary(ctx: Any) -> dict:
    """Build Section A: Executive Summary.

    Args:
        ctx: WeeklyDataContext with pre-loaded data.

    Returns:
        Dict with metrics, trends_4week, categories, remind, llm_briefing.
    """
    current_week = ctx.papers_by_week[-1] if ctx.papers_by_week else []
    prev_week = ctx.papers_by_week[-2] if len(ctx.papers_by_week) >= 2 else []

    # Deduplicate current/prev week by paper_key (keep highest score)
    current_deduped = _dedupe_papers(current_week)
    prev_deduped = _dedupe_papers(prev_week)

    # Basic metrics
    total_evaluated = len(current_deduped)
    tier1_count = sum(1 for p in current_deduped if (p.get("tier") or 0) == 1)

    # CS.CV count (papers with cs.CV category)
    cs_cv_count = _count_category(current_deduped, "cs.CV")
    prev_cs_cv = _count_category(prev_deduped, "cs.CV")

    # Keyword hits
    keyword_hits = sum(
        1 for p in current_deduped if p["paper_key"] in ctx.must_track_matches
    )
    prev_keyword_hits = sum(
        1 for p in prev_deduped if p["paper_key"] in ctx.must_track_matches
    )

    # Average score
    scores = [p["final_score"] for p in current_deduped if p.get("final_score")]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    prev_scores = [p["final_score"] for p in prev_deduped if p.get("final_score")]
    prev_avg = round(sum(prev_scores) / len(prev_scores), 2) if prev_scores else 0.0

    has_prev = ctx.prev_snapshot is not None

    # WoW calculations
    cs_cv_wow = (cs_cv_count - prev_cs_cv) if has_prev else "N/A"
    kw_hits_wow = (keyword_hits - prev_keyword_hits) if has_prev else "N/A"
    avg_wow = round(avg_score - prev_avg, 2) if has_prev else "N/A"

    # 4-week trends
    trends = _compute_4week_trends(ctx)

    # Top 5 categories
    categories = _top_categories(current_deduped, prev_deduped, has_prev, 5)

    # Remind stats
    graduated = [r for r in ctx.reminds if r.get("recommend_count", 0) == 2]
    active = [r for r in ctx.reminds if r.get("recommend_count", 0) == 1]

    return {
        "metrics": {
            "total_collected": sum(len(w) for w in ctx.papers_by_week),
            "total_evaluated": total_evaluated,
            "tier1_count": tier1_count,
            "cs_cv_count": cs_cv_count,
            "cs_cv_wow": cs_cv_wow,
            "keyword_hits": keyword_hits,
            "keyword_hits_wow": kw_hits_wow,
            "avg_score": avg_score,
            "avg_score_wow": avg_wow,
        },
        "trends_4week": trends,
        "categories": categories,
        "remind": {
            "graduated_count": len(graduated),
            "active_count": len(active),
        },
        "llm_briefing": None,
    }


def _dedupe_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate papers by paper_key, keeping highest final_score."""
    best: dict[str, dict] = {}
    for p in papers:
        pk = p["paper_key"]
        score = p.get("final_score", 0) or 0
        if pk not in best or score > (best[pk].get("final_score", 0) or 0):
            best[pk] = p
    return list(best.values())


def _count_category(papers: list[dict], category: str) -> int:
    """Count papers that have a given category."""
    count = 0
    for p in papers:
        cats_raw = p.get("categories", "[]")
        try:
            cats = json.loads(cats_raw) if isinstance(cats_raw, str) else cats_raw
        except (json.JSONDecodeError, TypeError):
            cats = []
        if isinstance(cats, list) and category in cats:
            count += 1
    return count


def _compute_4week_trends(ctx: Any) -> dict:
    """Compute 4-week trend data."""
    cs_cv_weekly = []
    avg_score_weekly = []
    total_weekly = []

    for week_papers in ctx.papers_by_week:
        deduped = _dedupe_papers(week_papers)
        total_weekly.append(len(deduped))
        cs_cv_weekly.append(_count_category(deduped, "cs.CV"))
        scores = [p["final_score"] for p in deduped if p.get("final_score")]
        avg = round(sum(scores) / len(scores), 2) if scores else 0.0
        avg_score_weekly.append(avg)

    # Trend direction based on last 2 weeks
    if len(total_weekly) >= 2:
        if total_weekly[-1] > total_weekly[-2]:
            direction = "uptrend"
        elif total_weekly[-1] < total_weekly[-2]:
            direction = "downtrend"
        else:
            direction = "stable"
    else:
        direction = "stable"

    return {
        "cs_cv": cs_cv_weekly,
        "avg_score": avg_score_weekly,
        "total_papers": total_weekly,
        "trend_direction": direction,
    }


def _top_categories(
    current: list[dict], prev: list[dict], has_prev: bool, limit: int
) -> list[dict]:
    """Get top N categories with WoW percentage change."""
    curr_counter: Counter = Counter()
    prev_counter: Counter = Counter()

    for p in current:
        cats_raw = p.get("categories", "[]")
        try:
            cats = json.loads(cats_raw) if isinstance(cats_raw, str) else cats_raw
        except (json.JSONDecodeError, TypeError):
            cats = []
        if isinstance(cats, list):
            for c in cats:
                curr_counter[c] += 1

    for p in prev:
        cats_raw = p.get("categories", "[]")
        try:
            cats = json.loads(cats_raw) if isinstance(cats_raw, str) else cats_raw
        except (json.JSONDecodeError, TypeError):
            cats = []
        if isinstance(cats, list):
            for c in cats:
                prev_counter[c] += 1

    result = []
    for name, count in curr_counter.most_common(limit):
        if has_prev and prev_counter.get(name, 0) > 0:
            wow_pct = round(
                ((count - prev_counter[name]) / prev_counter[name]) * 100, 1
            )
        else:
            wow_pct = "N/A"
        result.append({"name": name, "count": count, "wow_pct": wow_pct})

    return result


def add_llm_briefing(section_data: dict, analyst: Any) -> None:
    """Add LLM-generated executive briefing.

    Args:
        section_data: Section A data dict (modified in place).
        analyst: WeeklyLLMAnalyst instance.
    """
    if analyst is None or not section_data:
        return

    metrics = section_data.get("metrics", {})
    if not metrics:
        return

    system_prompt = (
        "You are a senior research analyst. Based on the weekly metrics data, "
        "write a concise 2-3 sentence executive briefing in Korean summarizing "
        "the key trends and notable changes this week."
    )

    user_prompt = (
        f"This week's metrics:\n"
        f"- Total papers evaluated: {metrics.get('total_evaluated', 0)}\n"
        f"- Tier 1 papers: {metrics.get('tier1_count', 0)}\n"
        f"- CS.CV papers: {metrics.get('cs_cv_count', 0)} (WoW: {metrics.get('cs_cv_wow', 'N/A')})\n"
        f"- Keyword hits: {metrics.get('keyword_hits', 0)} (WoW: {metrics.get('keyword_hits_wow', 'N/A')})\n"
        f"- Average score: {metrics.get('avg_score', 0)} (WoW: {metrics.get('avg_score_wow', 'N/A')})\n"
    )

    trends = section_data.get("trends_4week", {})
    if trends:
        user_prompt += f"\n4-week trend direction: {trends.get('trend_direction', 'stable')}\n"
        user_prompt += f"4-week paper counts: {trends.get('total_papers', [])}\n"

    result = analyst.call(system_prompt, user_prompt, "executive_briefing")
    section_data["llm_briefing"] = result
