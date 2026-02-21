"""Section E: Research Network for weekly intelligence report.

Identifies notable authors and detects conference-related papers.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def build_research_network(ctx: Any) -> dict:
    """Build Section E: Research Network.

    Args:
        ctx: WeeklyDataContext with pre-loaded data.

    Returns:
        Dict with notable_authors and conference_papers.
    """
    notable_authors = _find_notable_authors(ctx)
    conference_papers = _detect_conferences(ctx)

    return {
        "notable_authors": notable_authors,
        "conference_papers": conference_papers,
    }


def _find_notable_authors(ctx: Any, min_papers: int = 2, limit: int = 10) -> list[dict]:
    """Find authors with multiple high-score papers across 4 weeks.

    Only considers papers with final_score >= 60.
    """
    author_papers: dict[str, list[dict]] = defaultdict(list)

    for paper in ctx.all_papers_deduped:
        score = paper.get("final_score", 0) or 0
        if score < 60:
            continue

        authors_raw = paper.get("authors", "[]")
        try:
            authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(authors, list):
            continue

        for author in authors:
            if not isinstance(author, str):
                continue
            normalized = author.strip().lower()
            if not normalized:
                continue
            author_papers[normalized].append(paper)

    # Filter to notable (>= min_papers) and compute stats
    notable = []
    current_week_keys = set()
    if ctx.papers_by_week:
        current_week_keys = {p["paper_key"] for p in ctx.papers_by_week[-1]}

    for name, papers in author_papers.items():
        if len(papers) < min_papers:
            continue

        scores = [p.get("final_score", 0) or 0 for p in papers]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        # Papers from this week
        this_week_papers = [
            {
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "score": p.get("final_score", 0),
            }
            for p in papers
            if p["paper_key"] in current_week_keys
        ]

        # Use original casing from first occurrence
        display_name = name.title()
        for p in papers:
            authors_raw = p.get("authors", "[]")
            try:
                authors_list = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
            except (json.JSONDecodeError, TypeError):
                authors_list = []
            if isinstance(authors_list, list):
                for a in authors_list:
                    if isinstance(a, str) and a.strip().lower() == name:
                        display_name = a.strip()
                        break

        notable.append({
            "name": display_name,
            "count": len(papers),
            "avg_score": avg_score,
            "this_week": this_week_papers,
        })

    # Sort by count descending, then avg_score
    notable.sort(key=lambda x: (-x["count"], -x["avg_score"]))
    return notable[:limit]


def _detect_conferences(ctx: Any) -> list[dict]:
    """Detect conference-related papers from this week.

    Searches the 'comment' field for conference name patterns.
    """
    config_conferences = ctx.config.get("conferences", [])
    if not config_conferences:
        return []

    conf_patterns = {}
    for conf_name in config_conferences:
        if isinstance(conf_name, str):
            pattern = re.compile(
                rf'\b{re.escape(conf_name)}\b[\s\-]*(\d{{4}})?',
                re.IGNORECASE,
            )
            conf_patterns[conf_name] = pattern

    current_week = ctx.papers_by_week[-1] if ctx.papers_by_week else []
    detected = []

    for paper in current_week:
        comment = paper.get("comment", "") or ""
        title = paper.get("title", "") or ""
        search_text = comment + " " + title

        for conf_name, pattern in conf_patterns.items():
            match = pattern.search(search_text)
            if match:
                year = match.group(1) if match.lastindex and match.group(1) else ""
                detected.append({
                    "paper_key": paper.get("paper_key", ""),
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "conference": conf_name,
                    "year": year,
                    "final_score": paper.get("final_score", 0),
                })
                break  # One conference per paper

    # Sort by conference name, then score
    detected.sort(key=lambda x: (x["conference"], -(x.get("final_score", 0) or 0)))
    return detected
