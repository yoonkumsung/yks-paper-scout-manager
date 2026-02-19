"""Weekly Trend Summary Generator for Paper Scout.

Pure stats aggregation module that generates weekly summary reports
without LLM calls. Outputs HTML and Markdown formats for gh-pages deployment.

Provides keyword frequency, score trends, top papers, and graduated reminds.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.storage.db_connection import get_connection


def generate_weekly_summary(
    db_path: str = "data/paper_scout.db",
    date_str: str = "",
    output_dir: str = "tmp/reports",
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> dict:
    """Generate weekly summary data from database.

    Main entry point. Queries DB for weekly data (7 days backward from date_str),
    generates summary data dict.

    Args:
        db_path: Path to SQLite database
        date_str: End date in YYYYMMDD format
        output_dir: Output directory for reports (default: "tmp/reports")
        provider: Database provider ("sqlite" or "supabase")
        connection_string: PostgreSQL connection string (when provider is "supabase")

    Returns:
        dict: Summary data containing:
            - keyword_freq: Dict[topic_slug, List[Tuple[keyword, count]]]
            - score_trends: Dict[topic_slug, List[Tuple[date, avg_score]]]
            - top_papers: List[dict] (top 3 papers by final_score)
            - graduated_reminds: List[dict] (papers reaching recommend_count=2)
    """
    # Parse date and calculate week range
    end_date = datetime.strptime(date_str, "%Y%m%d")
    start_date = end_date - timedelta(days=7)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Gather all summary components
    keyword_freq = _get_keyword_frequency(
        db_path, start_str, end_str, provider, connection_string, output_dir,
    )
    score_trends = _get_score_trends(db_path, start_str, end_str, provider, connection_string)
    top_papers = _get_top_papers(db_path, start_str, end_str, 3, provider, connection_string)
    graduated_reminds = _get_graduated_reminds(db_path, start_str, end_str, provider, connection_string)

    return {
        "keyword_freq": keyword_freq,
        "score_trends": score_trends,
        "top_papers": top_papers,
        "graduated_reminds": graduated_reminds,
    }


def _get_keyword_frequency(
    db_path: str,
    start_date: str,
    end_date: str,
    provider: str = "sqlite",
    connection_string: str | None = None,
    output_dir: str = "tmp/reports",
) -> Dict[str, List[Tuple[str, int]]]:
    """Collect keyword frequency per topic for the week from JSON reports.

    Keywords are stored in each run's JSON report file (``meta.keywords_used``),
    not in the database.  This scans report directories whose date falls
    within ``[start_date, end_date]``.

    Args:
        db_path: Unused (kept for call-site compatibility).
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        provider: Unused (kept for call-site compatibility).
        connection_string: Unused (kept for call-site compatibility).
        output_dir: Directory containing date-based report subdirectories.

    Returns:
        Dict[topic_slug, List[Tuple[keyword, count]]]:
            Top 10 keywords per topic sorted by count descending
    """
    import os
    import re

    report_root = Path(output_dir)
    if not report_root.is_dir():
        return {}

    # Convert YYYY-MM-DD boundaries to comparable strings (YYYYMMDD)
    start_cmp = start_date.replace("-", "")
    end_cmp = end_date.replace("-", "")

    topic_keywords: Dict[str, Dict[str, int]] = {}

    # Scan date directories that fall within the range
    for entry in sorted(os.listdir(report_root)):
        date_dir = report_root / entry
        if not date_dir.is_dir():
            continue
        # Directory names are YYYYMMDD
        if not re.fullmatch(r"\d{8}", entry):
            continue
        if entry < start_cmp or entry > end_cmp:
            continue

        # Find JSON report files: {YYYYMMDD}_paper_{slug}.json
        for fname in os.listdir(date_dir):
            m = re.match(r"\d{8}_paper_(.+)\.json$", fname)
            if not m:
                continue
            slug = m.group(1)
            json_path = date_dir / fname
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            keywords = report_data.get("meta", {}).get("keywords_used", [])
            if not isinstance(keywords, list):
                continue

            if slug not in topic_keywords:
                topic_keywords[slug] = {}

            for kw in keywords:
                if isinstance(kw, str):
                    topic_keywords[slug][kw] = topic_keywords[slug].get(kw, 0) + 1

    # Sort and limit to top 10 per topic
    result: Dict[str, List[Tuple[str, int]]] = {}
    for topic_slug, kw_counts in topic_keywords.items():
        sorted_kws = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        result[topic_slug] = sorted_kws

    return result


def _get_score_trends(
    db_path: str,
    start_date: str,
    end_date: str,
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Daily average final_score per topic.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        provider: Database provider
        connection_string: PostgreSQL connection string

    Returns:
        Dict[topic_slug, List[Tuple[date, avg_score]]]:
            Daily averages sorted by date
    """
    with get_connection(db_path, provider, connection_string) as (conn, ph):
        if conn is None:
            return {}
        cursor = conn.cursor()

        query = f"""
        SELECT
            r.topic_slug,
            r.window_start_utc as date,
            AVG(e.final_score) as avg_score
        FROM runs r
        JOIN paper_evaluations e ON r.run_id = e.run_id
        WHERE r.window_start_utc >= {ph} AND r.window_end_utc <= {ph}
            AND e.final_score IS NOT NULL
        GROUP BY r.topic_slug, r.window_start_utc
        ORDER BY r.topic_slug, date
        """
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()

    # Group by topic
    result: Dict[str, List[Tuple[str, float]]] = {}
    for row in rows:
        if isinstance(row, dict):
            topic_slug, date, avg_score = row["topic_slug"], row["date"], row["avg_score"]
        else:
            topic_slug, date, avg_score = row[0], row[1], row[2]
        # Extract date portion if full timestamp
        date_str = str(date)[:10] if date else ""
        if topic_slug not in result:
            result[topic_slug] = []
        result[topic_slug].append((date_str, round(float(avg_score), 2)))

    return result


def _get_top_papers(
    db_path: str,
    start_date: str,
    end_date: str,
    limit: int = 3,
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> List[dict]:
    """Top N papers across all topics by final_score.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of top papers to return (default: 3)
        provider: Database provider
        connection_string: PostgreSQL connection string

    Returns:
        List[dict]: Papers with title, url, final_score, topic_slug
    """
    with get_connection(db_path, provider, connection_string) as (conn, ph):
        if conn is None:
            return []
        cursor = conn.cursor()

        query = f"""
        SELECT
            p.title,
            p.url,
            e.final_score,
            r.topic_slug
        FROM paper_evaluations e
        JOIN papers p ON e.paper_key = p.paper_key
        JOIN runs r ON e.run_id = r.run_id
        WHERE r.window_start_utc >= {ph} AND r.window_end_utc <= {ph}
            AND e.final_score IS NOT NULL
        ORDER BY e.final_score DESC
        LIMIT {ph}
        """
        cursor.execute(query, (start_date, end_date, limit))
        rows = cursor.fetchall()

    return [
        {
            "title": row["title"] if isinstance(row, dict) else row[0],
            "url": row["url"] if isinstance(row, dict) else row[1],
            "final_score": round(float(row["final_score"] if isinstance(row, dict) else row[2]), 2),
            "topic_slug": row["topic_slug"] if isinstance(row, dict) else row[3],
        }
        for row in rows
    ]


def _get_graduated_reminds(
    db_path: str,
    start_date: str,
    end_date: str,
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> List[dict]:
    """Papers that reached recommend_count=2 this week.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        provider: Database provider
        connection_string: PostgreSQL connection string

    Returns:
        List[dict]: Papers with title, url, topic_slug, recommend_count
    """
    with get_connection(db_path, provider, connection_string) as (conn, ph):
        if conn is None:
            return []
        cursor = conn.cursor()

        # Find papers with recommend_count=2 whose last_recommend_run_id is in date range
        query = f"""
        SELECT
            p.title,
            p.url,
            rt.topic_slug,
            rt.recommend_count
        FROM remind_tracking rt
        JOIN papers p ON rt.paper_key = p.paper_key
        JOIN runs r ON rt.last_recommend_run_id = r.run_id
        WHERE rt.recommend_count = 2
            AND r.window_start_utc >= {ph} AND r.window_end_utc <= {ph}
        ORDER BY p.title
        """
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()

    return [
        {
            "title": row["title"] if isinstance(row, dict) else row[0],
            "url": row["url"] if isinstance(row, dict) else row[1],
            "topic_slug": row["topic_slug"] if isinstance(row, dict) else row[2],
            "recommend_count": row["recommend_count"] if isinstance(row, dict) else row[3],
        }
        for row in rows
    ]


def render_weekly_summary_md(summary_data: dict, date_str: str) -> str:
    """Render summary data to markdown string.

    Args:
        summary_data: Summary data from generate_weekly_summary
        date_str: Date in YYYYMMDD format

    Returns:
        str: Markdown formatted summary
    """
    lines = []
    lines.append(f"# Weekly Summary: {date_str}")
    lines.append("")

    # Keyword Frequency
    lines.append("## Top Keywords per Topic")
    lines.append("")
    keyword_freq = summary_data.get("keyword_freq", {})
    if keyword_freq:
        for topic_slug, keywords in sorted(keyword_freq.items()):
            lines.append(f"### {topic_slug}")
            lines.append("")
            for kw, count in keywords:
                lines.append(f"- **{kw}**: {count}")
            lines.append("")
    else:
        lines.append("_No keyword data available_")
        lines.append("")

    # Score Trends
    lines.append("## Daily Average Scores per Topic")
    lines.append("")
    score_trends = summary_data.get("score_trends", {})
    if score_trends:
        for topic_slug, trends in sorted(score_trends.items()):
            lines.append(f"### {topic_slug}")
            lines.append("")
            lines.append("| Date | Avg Score |")
            lines.append("|------|-----------|")
            for date, avg_score in trends:
                lines.append(f"| {date} | {avg_score} |")
            lines.append("")
    else:
        lines.append("_No score trend data available_")
        lines.append("")

    # Top Papers
    lines.append("## Top 3 Papers This Week")
    lines.append("")
    top_papers = summary_data.get("top_papers", [])
    if top_papers:
        for i, paper in enumerate(top_papers, 1):
            lines.append(f"### {i}. {paper['title']}")
            lines.append(f"- **Score**: {paper['final_score']}")
            lines.append(f"- **Topic**: {paper['topic_slug']}")
            lines.append(f"- **URL**: [{paper['url']}]({paper['url']})")
            lines.append("")
    else:
        lines.append("_No top papers available_")
        lines.append("")

    # Graduated Reminds
    lines.append("## Graduated Remind Papers (2nd Recommendation)")
    lines.append("")
    graduated = summary_data.get("graduated_reminds", [])
    if graduated:
        for paper in graduated:
            lines.append(f"- **{paper['title']}** ({paper['topic_slug']})")
            lines.append(f"  - URL: [{paper['url']}]({paper['url']})")
            lines.append("")
    else:
        lines.append("_No graduated reminds this week_")
        lines.append("")

    return "\n".join(lines)


def render_weekly_summary_html(summary_data: dict, date_str: str, chart_paths: list[str] | None = None) -> str:
    """Render summary data to HTML string (simple inline-styled HTML).

    Args:
        summary_data: Summary data from generate_weekly_summary
        date_str: Date in YYYYMMDD format
        chart_paths: Optional list of chart image file paths to embed

    Returns:
        str: HTML formatted summary
    """
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html>")
    lines.append("<head>")
    lines.append("<meta charset='UTF-8'>")
    lines.append(f"<title>Weekly Summary: {date_str}</title>")
    lines.append("<style>")
    lines.append("body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }")
    lines.append("h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }")
    lines.append("h2 { color: #4CAF50; margin-top: 30px; }")
    lines.append("h3 { color: #555; }")
    lines.append("table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
    lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    lines.append("th { background-color: #4CAF50; color: white; }")
    lines.append("ul { line-height: 1.6; }")
    lines.append("a { color: #1976D2; text-decoration: none; }")
    lines.append("a:hover { text-decoration: underline; }")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"<h1>Weekly Summary: {date_str}</h1>")

    # Keyword Frequency
    lines.append("<h2>Top Keywords per Topic</h2>")
    keyword_freq = summary_data.get("keyword_freq", {})
    if keyword_freq:
        for topic_slug, keywords in sorted(keyword_freq.items()):
            lines.append(f"<h3>{html.escape(str(topic_slug))}</h3>")
            lines.append("<ul>")
            for kw, count in keywords:
                lines.append(f"<li><strong>{html.escape(str(kw))}</strong>: {count}</li>")
            lines.append("</ul>")
    else:
        lines.append("<p><em>No keyword data available</em></p>")

    # Score Trends
    lines.append("<h2>Daily Average Scores per Topic</h2>")
    score_trends = summary_data.get("score_trends", {})
    if score_trends:
        for topic_slug, trends in sorted(score_trends.items()):
            lines.append(f"<h3>{html.escape(str(topic_slug))}</h3>")
            lines.append("<table>")
            lines.append("<tr><th>Date</th><th>Avg Score</th></tr>")
            for date, avg_score in trends:
                lines.append(f"<tr><td>{html.escape(str(date))}</td><td>{avg_score}</td></tr>")
            lines.append("</table>")
    else:
        lines.append("<p><em>No score trend data available</em></p>")

    # Charts (inline base64 images)
    if chart_paths:
        import base64
        lines.append("<h2>Score Distribution</h2>")
        for chart_path in chart_paths:
            try:
                with open(chart_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                lines.append(f"<img src='data:image/png;base64,{img_data}' style='max-width:100%;height:auto;margin:10px 0;' alt='Chart'>")
            except (OSError, IOError):
                continue

    # Top Papers
    lines.append("<h2>Top 3 Papers This Week</h2>")
    top_papers = summary_data.get("top_papers", [])
    if top_papers:
        for i, paper in enumerate(top_papers, 1):
            escaped_title = html.escape(str(paper['title']))
            escaped_topic = html.escape(str(paper['topic_slug']))
            escaped_url = html.escape(str(paper['url']))
            lines.append(f"<h3>{i}. {escaped_title}</h3>")
            lines.append("<ul>")
            lines.append(f"<li><strong>Score</strong>: {paper['final_score']}</li>")
            lines.append(f"<li><strong>Topic</strong>: {escaped_topic}</li>")
            lines.append(f"<li><strong>URL</strong>: <a href='{escaped_url}'>{escaped_url}</a></li>")
            lines.append("</ul>")
    else:
        lines.append("<p><em>No top papers available</em></p>")

    # Graduated Reminds
    lines.append("<h2>Graduated Remind Papers (2nd Recommendation)</h2>")
    graduated = summary_data.get("graduated_reminds", [])
    if graduated:
        lines.append("<ul>")
        for paper in graduated:
            escaped_title = html.escape(str(paper['title']))
            escaped_topic = html.escape(str(paper['topic_slug']))
            escaped_url = html.escape(str(paper['url']))
            lines.append(f"<li><strong>{escaped_title}</strong> ({escaped_topic})")
            lines.append(f"<br>URL: <a href='{escaped_url}'>{escaped_url}</a></li>")
        lines.append("</ul>")
    else:
        lines.append("<p><em>No graduated reminds this week</em></p>")

    lines.append("</body>")
    lines.append("</html>")

    return "\n".join(lines)
