"""Section B: Tech Radar for weekly intelligence report.

Keyword intelligence with TF-IDF analysis, co-occurrence tracking,
and 4-week trend sparklines.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def build_tech_radar(ctx: Any) -> dict:
    """Build Section B: Tech Radar.

    Args:
        ctx: WeeklyDataContext with pre-loaded data.

    Returns:
        Dict with keyword_groups, co_occurrence, tfidf, llm_trends.
    """
    config = ctx.config
    kw_groups_config = config.get("must_track_keywords", {}).get("groups", {})
    tfidf_top_n = config.get("tfidf_top_n", 20)

    # Build keyword groups with per-week counts
    keyword_groups = _build_keyword_groups(ctx, kw_groups_config)

    # Co-occurrence (top 5)
    co_occurrence = _build_co_occurrence(ctx, limit=5)

    # TF-IDF analysis
    current_week_papers = ctx.papers_by_week[-1] if ctx.papers_by_week else []
    tfidf_result = _compute_tfidf_section(current_week_papers, ctx.prev_snapshot, tfidf_top_n)

    return {
        "keyword_groups": keyword_groups,
        "co_occurrence": co_occurrence,
        "tfidf": tfidf_result,
        "llm_trends": None,
    }


def add_llm_trends(section_data: dict, analyst: Any) -> None:
    """Add LLM trend analysis to each keyword group.

    Args:
        section_data: Section B data dict (modified in place).
        analyst: WeeklyLLMAnalyst instance.
    """
    if analyst is None or not section_data:
        return

    keyword_groups = section_data.get("keyword_groups", {})
    system_prompt = (
        "You are a computer vision research analyst. Analyze the keyword trend data "
        "and provide a brief 2-3 sentence analysis in Korean about what the trends indicate "
        "for this technology area. Focus on emerging patterns and practical implications."
    )

    for group_name, group_data in keyword_groups.items():
        if not isinstance(group_data, dict):
            continue
        keywords = group_data.get("keywords", [])
        if not keywords:
            continue

        user_prompt = f"Keyword group: {group_name}\n\n"
        for kw_info in keywords:
            name = kw_info.get("keyword", "")
            count = kw_info.get("count", 0)
            spark = kw_info.get("sparkline_4w", [])
            user_prompt += f"- {name}: count={count}, 4-week trend={spark}\n"

        try:
            result = analyst.call(system_prompt, user_prompt, f"trend_{group_name}")
            group_data["llm_analysis"] = result
        except Exception:
            logger.warning("LLM trend analysis for %s failed", group_name)
            group_data["llm_analysis"] = None


def _build_keyword_groups(ctx: Any, kw_groups_config: dict) -> dict:
    """Build keyword group data with counts and sparklines."""
    result = {}

    for group_name, keywords in kw_groups_config.items():
        group_keywords = []

        for kw in keywords:
            pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)

            # Per-week counts for sparkline
            weekly_counts = []
            matching_papers = []

            for week_idx, week_papers in enumerate(ctx.papers_by_week):
                week_count = 0
                for p in week_papers:
                    text = (p.get("title", "") or "") + " " + (p.get("abstract", "") or "")
                    if pattern.search(text):
                        week_count += 1
                        if week_idx == len(ctx.papers_by_week) - 1:
                            matching_papers.append({
                                "title": p.get("title", ""),
                                "url": p.get("url", ""),
                                "score": p.get("final_score", 0),
                            })
                weekly_counts.append(week_count)

            current_count = weekly_counts[-1] if weekly_counts else 0
            prev_count = weekly_counts[-2] if len(weekly_counts) >= 2 else 0

            has_prev = ctx.prev_snapshot is not None
            count_wow = (current_count - prev_count) if has_prev else "N/A"

            group_keywords.append({
                "keyword": kw,
                "papers": matching_papers[:5],
                "count": current_count,
                "count_wow": count_wow,
                "sparkline_4w": weekly_counts,
            })

        result[group_name] = {
            "keywords": group_keywords,
            "llm_analysis": None,
        }

    return result


def _build_co_occurrence(ctx: Any, limit: int = 5) -> list[dict]:
    """Get top co-occurring keyword pairs."""
    sorted_pairs = sorted(
        ctx.co_occurrence_matrix.items(), key=lambda x: x[1], reverse=True
    )[:limit]

    return [
        {"pair": list(pair), "count": count}
        for pair, count in sorted_pairs
    ]


# ---------------------------------------------------------------
# TF-IDF (pure Python implementation)
# ---------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "this", "that",
    "these", "those", "it", "its", "we", "our", "us", "they", "them",
    "their", "he", "she", "his", "her", "i", "me", "my", "you", "your",
    "not", "no", "nor", "as", "if", "then", "than", "so", "up", "out",
    "about", "into", "over", "after", "before", "between", "through",
    "during", "each", "all", "both", "more", "most", "other", "some",
    "such", "only", "own", "same", "also", "very", "just", "because",
    "while", "when", "where", "which", "who", "whom", "what", "how",
    "using", "based", "via", "propose", "proposed", "method", "methods",
    "approach", "paper", "show", "results", "model", "models",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering stopwords and short tokens."""
    words = re.findall(r'[a-z][a-z0-9]+', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _compute_tfidf(papers: list[dict], top_n: int = 20) -> list[dict]:
    """Pure Python TF-IDF computation.

    Args:
        papers: List of paper dicts with title and abstract.
        top_n: Number of top keywords to return.

    Returns:
        List of {keyword, score} sorted by score descending.
    """
    if not papers:
        return []

    # Build document corpus
    docs = []
    for p in papers:
        text = (p.get("title", "") or "") + " " + (p.get("abstract", "") or "")
        tokens = _tokenize(text)
        docs.append(tokens)

    n_docs = len(docs)
    if n_docs == 0:
        return []

    # Document frequency
    df: Counter = Counter()
    for tokens in docs:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    # TF-IDF per term (aggregated across all docs)
    tfidf_scores: Counter = Counter()
    for tokens in docs:
        if not tokens:
            continue
        tf = Counter(tokens)
        doc_len = len(tokens)
        for term, count in tf.items():
            tf_val = count / doc_len
            idf_val = math.log(n_docs / (1 + df[term]))
            tfidf_scores[term] += tf_val * idf_val

    return [
        {"keyword": term, "score": round(score, 4)}
        for term, score in tfidf_scores.most_common(top_n)
    ]


def _compute_tfidf_section(
    current_papers: list[dict],
    prev_snapshot: dict | None,
    top_n: int,
) -> dict:
    """Compute TF-IDF with classification vs previous week."""
    current_tfidf = _compute_tfidf(current_papers, top_n)

    # Classify against previous snapshot
    prev_tfidf_list = []
    if prev_snapshot and "tech_radar" in prev_snapshot:
        prev_tfidf_list = prev_snapshot["tech_radar"].get("tfidf", {}).get("keywords", [])

    prev_set = {kw["keyword"] for kw in prev_tfidf_list}
    prev_scores = {kw["keyword"]: kw["score"] for kw in prev_tfidf_list}
    current_set = {kw["keyword"] for kw in current_tfidf}

    for kw in current_tfidf:
        name = kw["keyword"]
        if name not in prev_set:
            kw["classification"] = "NEW"
        elif kw["score"] > prev_scores.get(name, 0) * 1.5:
            kw["classification"] = "RISING"
        elif kw["score"] < prev_scores.get(name, 0) * 0.5:
            kw["classification"] = "FALLING"
        else:
            kw["classification"] = "STABLE"

    # DISAPPEARED keywords
    for prev_kw in prev_set - current_set:
        current_tfidf.append({
            "keyword": prev_kw,
            "score": 0,
            "classification": "DISAPPEARED",
        })

    return {"keywords": current_tfidf}
