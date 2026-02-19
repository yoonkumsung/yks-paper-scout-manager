"""Weekly Update Scan for Paper Scout.

Finds papers that were recently updated but originally published earlier.
Separate report for papers with significant updates.

Target conditions:
- updated_at_utc within last 7 days
- published_at_utc > 7 days ago (not newly published this week)
- published_at_utc within 90 days (still has Evaluation data before purge)
- Existing Evaluation in paper_evaluations table for the paper
"""

from __future__ import annotations

import html as html_mod
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from core.storage.db_connection import get_connection


def scan_updated_papers(
    db_path: str,
    reference_date: str,
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> list[dict]:
    """Scan for papers matching weekly update criteria.

    Args:
        db_path: Path to the SQLite database file.
        reference_date: Reference date in YYYY-MM-DD format (UTC).
        provider: Database provider ("sqlite" or "supabase").
        connection_string: PostgreSQL connection string (when provider is "supabase").

    Returns:
        List of dicts with: paper_key, title, url, native_id, llm_base_score,
        updated_at_utc, published_at_utc, topic_slug
    """
    # Parse reference date
    ref_dt = datetime.fromisoformat(reference_date).replace(tzinfo=timezone.utc)

    # Calculate time boundaries
    seven_days_ago = ref_dt - timedelta(days=7)
    ninety_days_ago = ref_dt - timedelta(days=90)

    with get_connection(db_path, provider, connection_string) as (conn, ph):
        if conn is None:
            return []
        cursor = conn.cursor()

        query = f"""
            SELECT DISTINCT
                p.paper_key,
                p.title,
                p.url,
                p.native_id,
                pe.llm_base_score,
                p.updated_at_utc,
                p.published_at_utc,
                r.topic_slug
            FROM papers p
            JOIN paper_evaluations pe ON p.paper_key = pe.paper_key
            JOIN runs r ON pe.run_id = r.run_id
            WHERE p.updated_at_utc >= {ph}
              AND p.published_at_utc < {ph}
              AND p.published_at_utc >= {ph}
              AND pe.discarded = 0
            ORDER BY p.updated_at_utc DESC
        """

        cursor.execute(
            query,
            (
                seven_days_ago.isoformat(),
                seven_days_ago.isoformat(),
                ninety_days_ago.isoformat(),
            ),
        )
        rows = cursor.fetchall()

    return [dict(row) if isinstance(row, dict) else dict(zip(
        ["paper_key", "title", "url", "native_id", "llm_base_score",
         "updated_at_utc", "published_at_utc", "topic_slug"], row
    )) for row in rows]


def render_updates_md(papers: list[dict], date_str: str) -> str:
    """Render update scan results to markdown.

    Args:
        papers: List of paper dicts from scan_updated_papers.
        date_str: Date string for report heading (YYYY-MM-DD).

    Returns:
        Markdown formatted report string.
    """
    if not papers:
        return f"# Weekly Updates Report - {date_str}\n\nNo updated papers found for this week.\n"

    lines = [
        f"# Weekly Updates Report - {date_str}",
        "",
        f"Found {len(papers)} paper(s) with recent updates.",
        "",
        "---",
        "",
    ]

    for paper in papers:
        # Extract arXiv ID from native_id or paper_key
        arxiv_id = paper.get("native_id", "")
        if not arxiv_id and paper.get("paper_key", "").startswith("arxiv:"):
            arxiv_id = paper["paper_key"].split(":", 1)[1]

        lines.append(f"## {paper['title']}")
        lines.append("")
        lines.append(f"**arXiv ID**: [{arxiv_id}]({paper['url']})")
        lines.append(f"**Original Score**: {paper['llm_base_score']}")
        lines.append(f"**Updated**: {paper['updated_at_utc']}")
        lines.append(f"**Topic**: {paper['topic_slug']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def render_updates_html(papers: list[dict], date_str: str) -> str:
    """Render update scan results to HTML.

    Args:
        papers: List of paper dicts from scan_updated_papers.
        date_str: Date string for report heading (YYYY-MM-DD).

    Returns:
        HTML formatted report string with inline styles.
    """
    if not papers:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Updates - {date_str}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .no-results {{ color: #718096; font-style: italic; }}
    </style>
</head>
<body>
    <h1>Weekly Updates Report - {date_str}</h1>
    <p class="no-results">No updated papers found for this week.</p>
</body>
</html>
"""

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"    <title>Weekly Updates - {date_str}</title>",
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f7fafc; }",
        "        h1 { color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }",
        "        .summary { background: white; padding: 15px; border-radius: 6px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "        .paper { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "        .paper h2 { margin-top: 0; color: #2d3748; font-size: 1.25rem; }",
        "        .paper-meta { color: #718096; font-size: 0.9rem; margin: 10px 0; }",
        "        .paper-meta strong { color: #4a5568; }",
        "        .arxiv-link { color: #3182ce; text-decoration: none; }",
        "        .arxiv-link:hover { text-decoration: underline; }",
        "        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 8px; }",
        "        .badge-score { background: #c6f6d5; color: #22543d; }",
        "        .badge-updated { background: #fed7d7; color: #742a2a; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Weekly Updates Report - {date_str}</h1>",
        '    <div class="summary">',
        f"        <p>Found <strong>{len(papers)}</strong> paper(s) with recent updates.</p>",
        "    </div>",
    ]

    for paper in papers:
        # Extract arXiv ID
        arxiv_id = paper.get("native_id", "")
        if not arxiv_id and paper.get("paper_key", "").startswith("arxiv:"):
            arxiv_id = paper["paper_key"].split(":", 1)[1]

        escaped_title = html_mod.escape(str(paper['title']))
        escaped_url = html_mod.escape(str(paper['url']))
        escaped_arxiv_id = html_mod.escape(str(arxiv_id))
        escaped_topic = html_mod.escape(str(paper['topic_slug']))
        escaped_updated = html_mod.escape(str(paper['updated_at_utc']))

        html_parts.extend(
            [
                '    <div class="paper">',
                f"        <h2>{escaped_title}</h2>",
                '        <div class="paper-meta">',
                f'            <a href="{escaped_url}" class="arxiv-link" target="_blank">arXiv:{escaped_arxiv_id}</a>',
                "        </div>",
                '        <div class="paper-meta">',
                f'            <span class="badge badge-score">Score: {paper["llm_base_score"]}</span>',
                f'            <span class="badge badge-updated">Updated: {escaped_updated}</span>',
                "        </div>",
                '        <div class="paper-meta">',
                f'            <strong>Topic:</strong> {escaped_topic}',
                "        </div>",
                "    </div>",
            ]
        )

    html_parts.extend(["</body>", "</html>"])

    return "\n".join(html_parts)


def generate_update_report(
    db_path: str,
    date_str: str,
    output_dir: str = "tmp/reports",
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> Optional[str]:
    """Generate weekly update report in both formats.

    Args:
        db_path: Path to the SQLite database file.
        date_str: Report date in YYYY-MM-DD format.
        output_dir: Directory to save report files.
        provider: Database provider ("sqlite" or "supabase").
        connection_string: PostgreSQL connection string (when provider is "supabase").

    Returns:
        Path to the output directory if papers found, None otherwise.
    """
    # Scan for updated papers
    papers = scan_updated_papers(db_path, date_str, provider, connection_string)

    # Return None if no papers found
    if not papers:
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename base
    date_clean = date_str.replace("-", "")
    filename_base = f"{date_clean}_weekly_updates"

    # Write markdown report
    md_content = render_updates_md(papers, date_str)
    md_path = output_path / f"{filename_base}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Write HTML report
    html_content = render_updates_html(papers, date_str)
    html_path = output_path / f"{filename_base}.html"
    html_path.write_text(html_content, encoding="utf-8")

    return str(output_path)
