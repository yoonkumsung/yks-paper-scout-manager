"""Markdown report generator for Paper Scout (TASK-025).

Generates Markdown report files following devspec 10-5 format.
Supports Tier 1 (rank 1-30) and Tier 2 (rank 31-100) paper formats.

Compatible with Python 3.9+.  No external dependencies (pure string formatting).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Flag display mapping
# ---------------------------------------------------------------------------

_FLAG_LABELS: Dict[str, str] = {
    "is_edge": "엣지",
    "is_realtime": "실시간",
    "has_code": "코드 공개",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_markdown(
    report_data: dict,
    output_dir: str = "tmp/reports",
) -> str:
    """Generate a Markdown report file.

    Args:
        report_data: Dict from JSON exporter containing ``meta``, ``papers``,
            ``clusters``, and ``remind_papers``.
        output_dir: Directory to write the report file to.

    Returns:
        Absolute path to the written ``.md`` file.
        File naming: ``{YYMMDD}_paper_{slug}.md`` (raw topic-loop output).
    """
    meta: dict = report_data.get("meta", {})
    papers: List[dict] = report_data.get("papers", [])
    clusters: List[dict] = report_data.get("clusters", [])
    remind_papers: List[dict] = report_data.get("remind_papers", [])

    # Build lookup maps.
    key_to_rank = _build_key_to_rank(papers)
    key_to_cluster_mates = _build_cluster_mates(clusters, key_to_rank)

    sections: List[str] = []

    # 1. Header
    sections.append(_render_header(meta))

    # 2. Keywords section
    sections.append(_render_keywords(meta))

    # 3. Papers section
    sections.append(_render_papers(papers, key_to_cluster_mates))

    # 4. Remind section
    if remind_papers:
        sections.append(_render_remind(remind_papers))

    # 5. Footer (arXiv notice)
    sections.append(_render_footer())

    content = "\n".join(sections)

    # Write to file.
    os.makedirs(output_dir, exist_ok=True)
    filename = _build_filename(meta)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(content)

    return os.path.abspath(filepath)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_header(meta: dict) -> str:
    """Render the report header with title, stats, and window info."""
    lines: List[str] = []
    lines.append("# %s" % meta.get("display_title", ""))
    lines.append("")

    # Stats summary
    stats_parts: List[str] = []
    total_collected = meta.get("total_collected", 0)
    total_filtered = meta.get("total_filtered", 0)
    total_discarded = meta.get("total_discarded", 0)
    total_scored = meta.get("total_scored", 0)
    total_output = meta.get("total_output", 0)
    threshold_used = meta.get("threshold_used", 60)

    stats_parts.append("수집 %d" % total_collected)
    stats_parts.append("필터 %d" % total_filtered)
    stats_parts.append("폐기 %d" % total_discarded)
    stats_parts.append("평가 %d" % total_scored)
    stats_parts.append("출력 %d" % total_output)
    stats_parts.append("기준 %d점" % threshold_used)

    lines.append("> " + " | ".join(stats_parts))
    lines.append("")

    # Window info
    window_start = meta.get("window_start_utc", "")
    window_end = meta.get("window_end_utc", "")
    embedding_mode = meta.get("embedding_mode", "disabled")
    run_id = meta.get("run_id", "")

    lines.append(
        "검색 윈도우: %s ~ %s | 임베딩: %s | run_id: %s"
        % (window_start, window_end, embedding_mode, run_id)
    )
    lines.append("")

    return "\n".join(lines)


def _render_keywords(meta: dict) -> str:
    """Render the keywords section."""
    keywords = meta.get("keywords_used", [])
    if not keywords:
        return ""

    lines: List[str] = []
    lines.append("---")
    lines.append("")
    lines.append("## 검색 키워드")
    lines.append("")
    lines.append(", ".join(keywords))
    lines.append("")

    return "\n".join(lines)


def _render_papers(
    papers: List[dict],
    key_to_cluster_mates: Dict[str, List[str]],
) -> str:
    """Render the papers section with Tier 1 and Tier 2 formatting."""
    if not papers:
        return ""

    lines: List[str] = []
    lines.append("---")
    lines.append("")

    for paper in papers:
        tier = paper.get("tier", 2)
        if tier == 1:
            lines.append(_render_tier1_paper(paper, key_to_cluster_mates))
        else:
            lines.append(_render_tier2_paper(paper))

    return "\n".join(lines)


def _render_tier1_paper(
    paper: dict,
    key_to_cluster_mates: Dict[str, List[str]],
) -> str:
    """Render a Tier 1 paper (rank 1-30) in full format."""
    lines: List[str] = []
    rank = paper.get("rank", 0)
    title = paper.get("title", "")
    score_lowered = paper.get("score_lowered", False)

    # Title line with optional [완화] tag
    title_line = "## %d위: %s" % (rank, title)
    if score_lowered:
        title_line += " [완화]"
    lines.append(title_line)
    lines.append("")

    # arXiv link
    url = paper.get("url", "")
    lines.append("- arXiv: %s" % url)

    # PDF link
    pdf_url = paper.get("pdf_url", "")
    if pdf_url:
        lines.append("- PDF: %s" % pdf_url)

    # Code link (only when has_code AND code_url)
    has_code = paper.get("has_code", False)
    code_url = paper.get("code_url")
    if has_code and code_url:
        lines.append("- 코드: %s" % code_url)

    # Published date
    published = paper.get("published_at_utc", "")
    lines.append("- 발행일: %s" % published)

    # Categories
    categories = paper.get("categories", [])
    if categories:
        lines.append("- 카테고리: %s" % ", ".join(categories))

    # Score line with bonus breakdown
    final_score = paper.get("final_score", 0.0)
    llm_adjusted = paper.get("llm_adjusted", 0)
    llm_base = paper.get("llm_base_score", 0)
    bonus = paper.get("bonus_score", 0)

    score_line = "- 점수: final %s (llm_adjusted:%s = base:%s + bonus:+%s)" % (
        _format_score(final_score),
        llm_adjusted,
        llm_base,
        bonus,
    )
    lines.append(score_line)

    # Flags
    flags_text = _render_flags(paper)
    if flags_text:
        lines.append("- 플래그: %s" % flags_text)

    lines.append("")

    # Summary
    summary_ko = paper.get("summary_ko", "")
    if summary_ko:
        lines.append("**개요**")
        lines.append(summary_ko)
        lines.append("")

    # Reason
    reason_ko = paper.get("reason_ko", "")
    if reason_ko:
        lines.append("**선정 근거**")
        lines.append(reason_ko)
        lines.append("")

    # Insight
    insight_ko = paper.get("insight_ko", "")
    if insight_ko:
        lines.append("**활용 인사이트**")
        lines.append(insight_ko)
        lines.append("")

    # Cluster mates
    paper_key = paper.get("paper_key", "")
    mates = key_to_cluster_mates.get(paper_key, [])
    if mates:
        lines.append("같은 클러스터: %s" % ", ".join(mates))
        lines.append("")

    return "\n".join(lines)


def _render_tier2_paper(paper: dict) -> str:
    """Render a Tier 2 paper (rank 31-100) in compact format."""
    lines: List[str] = []
    rank = paper.get("rank", 0)
    title = paper.get("title", "")
    score_lowered = paper.get("score_lowered", False)

    # Title line with optional [완화] tag
    title_line = "## %d위: %s" % (rank, title)
    if score_lowered:
        title_line += " [완화]"
    lines.append(title_line)
    lines.append("")

    # Compact info line: arXiv link | date | score
    url = paper.get("url", "")
    published = paper.get("published_at_utc", "")
    final_score = paper.get("final_score", 0.0)
    lines.append(
        "- arXiv: %s | %s | final %s" % (url, published, _format_score(final_score))
    )
    lines.append("")

    # Abstract preview (first 2 sentences)
    summary_ko = paper.get("summary_ko", "")
    if summary_ko:
        import re
        parts = re.split(r"(?<=[.!?])\s+", summary_ko.strip(), maxsplit=2)
        lines.append(" ".join(parts[:2]))
        lines.append("")

    # Reason with arrow prefix
    reason_ko = paper.get("reason_ko", "")
    if reason_ko:
        lines.append("-> %s" % reason_ko)
        lines.append("")

    return "\n".join(lines)


def _render_remind(remind_papers: List[dict]) -> str:
    """Render the '다시 보기' section for remind papers."""
    lines: List[str] = []
    lines.append("---")
    lines.append("")
    lines.append("## 다시 보기")
    lines.append("")

    for paper in remind_papers:
        rank = paper.get("rank", 0)
        title = paper.get("title", "")
        recommend_count = paper.get("recommend_count", 1)

        lines.append("### %s (%d회째 추천)" % (title, recommend_count))
        lines.append("")

        url = paper.get("url", "")
        lines.append("- arXiv: %s" % url)

        final_score = paper.get("final_score", 0.0)
        lines.append("- 점수: final %s" % _format_score(final_score))
        lines.append("")

        summary_ko = paper.get("summary_ko", "")
        if summary_ko:
            lines.append(summary_ko)
            lines.append("")

        reason_ko = paper.get("reason_ko", "")
        if reason_ko:
            lines.append("-> %s" % reason_ko)
            lines.append("")

    return "\n".join(lines)


def _render_footer() -> str:
    """Render the arXiv API usage notice footer."""
    lines: List[str] = []
    lines.append("---")
    lines.append("")
    lines.append("이 리포트는 arXiv API를 사용하여 생성되었습니다.")
    lines.append("arXiv 논문의 저작권은 각 저자에게 있습니다.")
    lines.append(
        "Thank you to arXiv for use of its open access interoperability."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _build_filename(meta: dict) -> str:
    """Build the output filename from meta.

    Format: ``{YYMMDD}_paper_{slug}.md``
    """
    date_str = meta.get("date", "")
    slug = meta.get("topic_slug", "unknown")
    # date is in "YYYY-MM-DD" format; convert to "YYMMDD"
    date_compact = date_str.replace("-", "")[2:]
    return "%s_paper_%s.md" % (date_compact, slug)


def _build_key_to_rank(papers: List[dict]) -> Dict[str, int]:
    """Build a mapping from paper_key to rank."""
    mapping: Dict[str, int] = {}
    for paper in papers:
        key = paper.get("paper_key", "")
        rank = paper.get("rank", 0)
        if key:
            mapping[key] = rank
    return mapping


def _build_cluster_mates(
    clusters: List[dict],
    key_to_rank: Dict[str, int],
) -> Dict[str, List[str]]:
    """Build a mapping from paper_key to its cluster mates as '#N위' strings.

    Only includes clusters with more than one member.
    """
    result: Dict[str, List[str]] = {}

    for cluster in clusters:
        members = cluster.get("member_keys", [])
        if len(members) <= 1:
            continue

        for key in members:
            mates: List[str] = []
            for other_key in members:
                if other_key == key:
                    continue
                other_rank = key_to_rank.get(other_key)
                if other_rank is not None:
                    mates.append("#%d위" % other_rank)
            if mates:
                result[key] = mates

    return result


def _render_flags(paper: dict) -> str:
    """Render flag labels as comma-separated Korean names."""
    flags = paper.get("flags", {})
    labels: List[str] = []

    if flags.get("is_edge", False):
        labels.append(_FLAG_LABELS["is_edge"])
    if flags.get("is_realtime", False):
        labels.append(_FLAG_LABELS["is_realtime"])
    # has_code flag comes from the paper-level field
    if paper.get("has_code", False):
        labels.append(_FLAG_LABELS["has_code"])

    return ", ".join(labels)


def _format_score(score: Any) -> str:
    """Format a score value, removing unnecessary trailing zeros.

    87.5 -> "87.5"
    70.0 -> "70.0"
    68.2 -> "68.2"
    87.5000 -> "87.5"
    """
    if isinstance(score, float):
        # Use one decimal place minimum; strip trailing zeros but keep at
        # least one decimal digit.
        formatted = "%.4f" % score
        # Remove trailing zeros but keep at least one decimal place.
        if "." in formatted:
            formatted = formatted.rstrip("0")
            if formatted.endswith("."):
                formatted += "0"
        return formatted
    return str(score)
