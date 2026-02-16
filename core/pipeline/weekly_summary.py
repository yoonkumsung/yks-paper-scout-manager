"""Weekly Trend Summary Generator for Paper Scout.

Pure stats aggregation module that generates weekly summary reports
without LLM calls. Outputs HTML and Markdown formats for gh-pages deployment.

Provides keyword frequency, score trends, top papers, and graduated reminds.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def generate_weekly_summary(
    db_path: str,
    date_str: str,
    output_dir: str = "tmp/reports"
) -> dict:
    """Generate weekly summary data from database.

    Main entry point. Queries DB for weekly data (7 days backward from date_str),
    generates summary data dict.

    Args:
        db_path: Path to SQLite database
        date_str: End date in YYYYMMDD format
        output_dir: Output directory for reports (default: "tmp/reports")

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
    keyword_freq = _get_keyword_frequency(db_path, start_str, end_str)
    score_trends = _get_score_trends(db_path, start_str, end_str)
    top_papers = _get_top_papers(db_path, start_str, end_str, limit=3)
    graduated_reminds = _get_graduated_reminds(db_path, start_str, end_str)

    return {
        "keyword_freq": keyword_freq,
        "score_trends": score_trends,
        "top_papers": top_papers,
        "graduated_reminds": graduated_reminds,
    }


def _get_keyword_frequency(
    db_path: str,
    start_date: str,
    end_date: str
) -> Dict[str, List[Tuple[str, int]]]:
    """Query keyword frequency per topic for the week.

    Uses keywords_used from runs table meta (stored as JSON).

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dict[topic_slug, List[Tuple[keyword, count]]]:
            Top 10 keywords per topic sorted by count descending
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query runs within date range
    query = """
    SELECT topic_slug, keywords_used
    FROM runs
    WHERE DATE(window_start_utc) >= ? AND DATE(window_end_utc) <= ?
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()

    # Aggregate keywords by topic
    topic_keywords: Dict[str, Dict[str, int]] = {}

    for topic_slug, keywords_json in rows:
        if keywords_json is None:
            continue

        try:
            keywords = json.loads(keywords_json)
            if not isinstance(keywords, list):
                continue

            if topic_slug not in topic_keywords:
                topic_keywords[topic_slug] = {}

            for kw in keywords:
                if isinstance(kw, str):
                    topic_keywords[topic_slug][kw] = topic_keywords[topic_slug].get(kw, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue

    # Sort and limit to top 10 per topic
    result: Dict[str, List[Tuple[str, int]]] = {}
    for topic_slug, kw_counts in topic_keywords.items():
        sorted_kws = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        result[topic_slug] = sorted_kws

    return result


def _get_score_trends(
    db_path: str,
    start_date: str,
    end_date: str
) -> Dict[str, List[Tuple[str, float]]]:
    """Daily average final_score per topic.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dict[topic_slug, List[Tuple[date, avg_score]]]:
            Daily averages sorted by date
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT
        r.topic_slug,
        DATE(r.window_start_utc) as date,
        AVG(e.final_score) as avg_score
    FROM runs r
    JOIN paper_evaluations e ON r.run_id = e.run_id
    WHERE DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
        AND e.final_score IS NOT NULL
    GROUP BY r.topic_slug, DATE(r.window_start_utc)
    ORDER BY r.topic_slug, date
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()

    # Group by topic
    result: Dict[str, List[Tuple[str, float]]] = {}
    for topic_slug, date, avg_score in rows:
        if topic_slug not in result:
            result[topic_slug] = []
        result[topic_slug].append((date, round(avg_score, 2)))

    return result


def _get_top_papers(
    db_path: str,
    start_date: str,
    end_date: str,
    limit: int = 3
) -> List[dict]:
    """Top N papers across all topics by final_score.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of top papers to return (default: 3)

    Returns:
        List[dict]: Papers with title, url, final_score, topic_slug
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT
        p.title,
        p.url,
        e.final_score,
        r.topic_slug
    FROM paper_evaluations e
    JOIN papers p ON e.paper_key = p.paper_key
    JOIN runs r ON e.run_id = r.run_id
    WHERE DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
        AND e.final_score IS NOT NULL
    ORDER BY e.final_score DESC
    LIMIT ?
    """
    cursor.execute(query, (start_date, end_date, limit))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "title": title,
            "url": url,
            "final_score": round(final_score, 2),
            "topic_slug": topic_slug,
        }
        for title, url, final_score, topic_slug in rows
    ]


def _get_graduated_reminds(
    db_path: str,
    start_date: str,
    end_date: str
) -> List[dict]:
    """Papers that reached recommend_count=2 this week.

    Args:
        db_path: Path to SQLite database
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List[dict]: Papers with title, url, topic_slug, recommend_count
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find papers with recommend_count=2 whose last_recommend_run_id is in date range
    query = """
    SELECT
        p.title,
        p.url,
        rt.topic_slug,
        rt.recommend_count
    FROM remind_tracking rt
    JOIN papers p ON rt.paper_key = p.paper_key
    JOIN runs r ON rt.last_recommend_run_id = r.run_id
    WHERE rt.recommend_count = 2
        AND DATE(r.window_start_utc) >= ? AND DATE(r.window_end_utc) <= ?
    ORDER BY p.title
    """
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "title": title,
            "url": url,
            "topic_slug": topic_slug,
            "recommend_count": recommend_count,
        }
        for title, url, topic_slug, recommend_count in rows
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


def render_weekly_summary_html(summary_data: dict, date_str: str) -> str:
    """Render summary data to HTML string (simple inline-styled HTML).

    Args:
        summary_data: Summary data from generate_weekly_summary
        date_str: Date in YYYYMMDD format

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
            lines.append(f"<h3>{topic_slug}</h3>")
            lines.append("<ul>")
            for kw, count in keywords:
                lines.append(f"<li><strong>{kw}</strong>: {count}</li>")
            lines.append("</ul>")
    else:
        lines.append("<p><em>No keyword data available</em></p>")

    # Score Trends
    lines.append("<h2>Daily Average Scores per Topic</h2>")
    score_trends = summary_data.get("score_trends", {})
    if score_trends:
        for topic_slug, trends in sorted(score_trends.items()):
            lines.append(f"<h3>{topic_slug}</h3>")
            lines.append("<table>")
            lines.append("<tr><th>Date</th><th>Avg Score</th></tr>")
            for date, avg_score in trends:
                lines.append(f"<tr><td>{date}</td><td>{avg_score}</td></tr>")
            lines.append("</table>")
    else:
        lines.append("<p><em>No score trend data available</em></p>")

    # Top Papers
    lines.append("<h2>Top 3 Papers This Week</h2>")
    top_papers = summary_data.get("top_papers", [])
    if top_papers:
        for i, paper in enumerate(top_papers, 1):
            lines.append(f"<h3>{i}. {paper['title']}</h3>")
            lines.append("<ul>")
            lines.append(f"<li><strong>Score</strong>: {paper['final_score']}</li>")
            lines.append(f"<li><strong>Topic</strong>: {paper['topic_slug']}</li>")
            lines.append(f"<li><strong>URL</strong>: <a href='{paper['url']}'>{paper['url']}</a></li>")
            lines.append("</ul>")
    else:
        lines.append("<p><em>No top papers available</em></p>")

    # Graduated Reminds
    lines.append("<h2>Graduated Remind Papers (2nd Recommendation)</h2>")
    graduated = summary_data.get("graduated_reminds", [])
    if graduated:
        lines.append("<ul>")
        for paper in graduated:
            lines.append(f"<li><strong>{paper['title']}</strong> ({paper['topic_slug']})")
            lines.append(f"<br>URL: <a href='{paper['url']}'>{paper['url']}</a></li>")
        lines.append("</ul>")
    else:
        lines.append("<p><em>No graduated reminds this week</em></p>")

    lines.append("</body>")
    lines.append("</html>")

    return "\n".join(lines)
