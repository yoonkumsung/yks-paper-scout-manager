"""Section D: Product Intelligence for weekly intelligence report.

Classifies papers by product line and tracks trends.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def build_product_intel(ctx: Any) -> dict:
    """Build Section D: Product Intelligence.

    Args:
        ctx: WeeklyDataContext with pre-loaded data.

    Returns:
        Dict with product_lines list and llm_strategy placeholder.
    """
    product_lines_config = ctx.config.get("product_lines", {})
    current_week = ctx.papers_by_week[-1] if ctx.papers_by_week else []
    prev_week = ctx.papers_by_week[-2] if len(ctx.papers_by_week) >= 2 else []
    has_prev = ctx.prev_snapshot is not None

    # Build per-line data
    result_lines = []

    for line_name in product_lines_config:
        # Current week papers for this line
        current_papers = []
        for p in current_week:
            pk = p["paper_key"]
            if pk in ctx.product_line_matches and line_name in ctx.product_line_matches[pk]:
                current_papers.append(p)

        if not current_papers:
            continue

        # Previous week count
        prev_count = 0
        for p in prev_week:
            pk = p["paper_key"]
            if pk in ctx.product_line_matches and line_name in ctx.product_line_matches[pk]:
                prev_count += 1

        count_wow = (len(current_papers) - prev_count) if has_prev else "N/A"

        # 4-week trend
        trend_4w = []
        for week_papers in ctx.papers_by_week:
            week_count = 0
            for p in week_papers:
                pk = p["paper_key"]
                if pk in ctx.product_line_matches and line_name in ctx.product_line_matches[pk]:
                    week_count += 1
            trend_4w.append(week_count)

        # Average score
        scores = [p.get("final_score", 0) or 0 for p in current_papers]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        # Top paper
        sorted_papers = sorted(current_papers, key=lambda x: x.get("final_score", 0) or 0, reverse=True)
        top_paper = sorted_papers[0] if sorted_papers else None

        result_lines.append({
            "name": line_name,
            "papers_count": len(current_papers),
            "count_wow": count_wow,
            "trend_4w": trend_4w,
            "avg_score": avg_score,
            "top_paper": {
                "title": top_paper.get("title", "") if top_paper else "",
                "url": top_paper.get("url", "") if top_paper else "",
                "score": top_paper.get("final_score", 0) if top_paper else 0,
            } if top_paper else None,
            "papers": [
                {
                    "paper_key": p["paper_key"],
                    "title": p.get("title", ""),
                    "abstract": (p.get("abstract", "") or "")[:200],
                    "score": p.get("final_score", 0),
                }
                for p in sorted_papers[:5]
            ],
        })

    # Sort by papers_count descending
    result_lines.sort(key=lambda x: x["papers_count"], reverse=True)

    return {
        "product_lines": result_lines,
        "llm_strategy": None,
    }


def add_llm_strategy(
    section_data: dict,
    analyst: Any,
    top10_missing_insight: list[dict],
) -> None:
    """Add LLM strategy analysis for product lines.

    Args:
        section_data: Section D data dict (modified in place).
        analyst: WeeklyLLMAnalyst instance.
        top10_missing_insight: Top 10 papers missing insight_ko.
    """
    if analyst is None or not section_data:
        return

    product_lines = section_data.get("product_lines", [])
    if not product_lines and not top10_missing_insight:
        return

    system_prompt = (
        "You are a product strategy analyst specializing in computer vision and AI. "
        "Analyze the research papers classified by product line and provide strategic "
        "insights in Korean. For each paper missing insight_ko, generate a brief "
        "1-2 sentence strategic insight in Korean.\n\n"
        "Output format:\n"
        "STRATEGY: [2-3 sentence overall strategy assessment]\n"
        "INSIGHTS:\n"
        "- [paper_key]: [insight in Korean]\n"
    )

    # Collect papers for prompt
    all_papers = []
    for pl in product_lines[:3]:
        for p in pl.get("papers", [])[:3]:
            all_papers.append(p)

    user_prompt = "Product Lines:\n"
    for pl in product_lines:
        user_prompt += f"\n## {pl['name']} ({pl['papers_count']} papers, avg score: {pl['avg_score']})\n"
        for p in pl.get("papers", [])[:3]:
            user_prompt += f"- {p.get('title', '')} (score: {p.get('score', 0)})\n"

    if top10_missing_insight:
        user_prompt += "\n\nPapers needing insight_ko:\n"
        for p in top10_missing_insight:
            user_prompt += f"- [{p.get('paper_key', '')}] {p.get('title', '')}\n"

    try:
        result = analyst.call(system_prompt, user_prompt, "product_strategy")
        if result:
            section_data["llm_strategy"] = result
            # Parse insights for top10 papers
            _apply_generated_insights(top10_missing_insight, result)
    except Exception:
        logger.warning("LLM product strategy failed", exc_info=True)


def _apply_generated_insights(papers: list[dict], llm_result: str) -> None:
    """Parse LLM output and apply generated insight_ko to papers."""
    if not llm_result or not papers:
        return

    for paper in papers:
        pk = paper.get("paper_key", "")
        if not pk:
            continue
        # Look for pattern: - [paper_key]: [insight]
        pattern = re.compile(rf'-\s*\[?{re.escape(pk)}\]?\s*:\s*(.+)', re.IGNORECASE)
        match = pattern.search(llm_result)
        if match:
            paper["insight_ko"] = match.group(1).strip()
