"""Weekly Paper Report Orchestrator.

Coordinates all section builders and produces the final report
in dict, Markdown, and HTML formats.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from core.storage.db_connection import get_connection

logger = logging.getLogger(__name__)


def generate_weekly_intelligence(
    db_path: str,
    date_str: str,
    config: Any,
    provider: str = "sqlite",
    connection_string: str | None = None,
    rate_limiter: Any | None = None,
    topic_slug: str | None = None,
) -> tuple[dict, str, str]:
    """Generate weekly intelligence report.

    Args:
        db_path: Path to SQLite database.
        date_str: Reference date in YYYYMMDD format.
        config: AppConfig object.
        provider: Database provider ("sqlite" or "supabase").
        connection_string: PostgreSQL connection string.
        rate_limiter: Optional RateLimiter instance for LLM calls.
        topic_slug: If provided, generate report for this topic only.

    Returns:
        Tuple of (summary_data, md_content, html_content).
    """
    from core.pipeline.weekly_data_context import WeeklyDataContext
    from core.pipeline.weekly_executive import build_executive_summary
    from core.pipeline.weekly_top_papers import build_top_papers

    reference_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    intel_config = config.weekly.get("intelligence", {})

    with get_connection(db_path, provider, connection_string) as (conn, ph):
        if conn is None:
            logger.warning("Database unavailable for weekly intelligence")
            empty = {"sections": {}, "date_str": date_str}
            return empty, "", ""

        ctx = WeeklyDataContext(conn, ph, intel_config, reference_date, topic_slug=topic_slug)

        # Build sections (Phase 1: A and C only)
        sections: dict[str, Any] = {}

        try:
            sections["executive"] = build_executive_summary(ctx)
        except Exception:
            logger.warning("Section A (executive) build failed", exc_info=True)
            sections["executive"] = {}

        try:
            sections["top_papers"] = build_top_papers(ctx)
        except Exception:
            logger.warning("Section C (top_papers) build failed", exc_info=True)
            sections["top_papers"] = {"papers": [], "graduated_reminds": []}

        # Phase 2: Tech Radar (Section B)
        try:
            from core.pipeline.weekly_tech_radar import build_tech_radar
            sections["tech_radar"] = build_tech_radar(ctx)
        except ImportError:
            sections["tech_radar"] = {}
        except Exception:
            logger.warning("Section B (tech_radar) build failed", exc_info=True)
            sections["tech_radar"] = {}

        # Phase 3: Product Intel (Section D)
        try:
            from core.pipeline.weekly_product_intel import build_product_intel
            sections["product_intel"] = build_product_intel(ctx)
        except ImportError:
            sections["product_intel"] = {}
        except Exception:
            logger.warning("Section D (product_intel) build failed", exc_info=True)
            sections["product_intel"] = {}

        # Phase 3: Research Network (Section E)
        try:
            from core.pipeline.weekly_research_net import build_research_network
            sections["research_net"] = build_research_network(ctx)
        except ImportError:
            sections["research_net"] = {}
        except Exception:
            logger.warning("Section E (research_net) build failed", exc_info=True)
            sections["research_net"] = {}

        # Phase 4: LLM integration
        llm_cfg = intel_config.get("llm", {})
        if llm_cfg.get("enabled", True):
            try:
                from core.pipeline.weekly_llm_analyst import WeeklyLLMAnalyst
                analyst = WeeklyLLMAnalyst(config, rate_limiter=rate_limiter)

                try:
                    from core.pipeline.weekly_executive import add_llm_briefing
                    add_llm_briefing(sections.get("executive", {}), analyst)
                except (ImportError, Exception):
                    pass

                try:
                    from core.pipeline.weekly_tech_radar import add_llm_trends
                    add_llm_trends(sections.get("tech_radar", {}), analyst)
                except (ImportError, Exception):
                    pass

                try:
                    from core.pipeline.weekly_product_intel import add_llm_strategy
                    top10_missing = [
                        p for p in sections.get("top_papers", {}).get("papers", [])
                        if not p.get("insight_ko")
                    ]
                    add_llm_strategy(
                        sections.get("product_intel", {}), analyst, top10_missing
                    )
                except (ImportError, Exception):
                    pass

            except ImportError:
                logger.debug("LLM analyst not available yet")
            except Exception:
                logger.warning("LLM integration failed (non-fatal)", exc_info=True)

        # Assemble summary data
        iso_cal = reference_date.isocalendar()
        summary_data = {
            "sections": sections,
            "date_str": date_str,
            "iso_year": iso_cal[0],
            "iso_week": iso_cal[1],
            # Legacy compatibility key
            "top_papers": sections.get("top_papers", {}).get("papers", []),
        }

        # Render templates
        md_content = _render_md(summary_data)
        html_content = _render_html(summary_data, config)

        # Save snapshots
        try:
            _save_snapshots(conn, ph, sections, iso_cal[0], iso_cal[1], provider, topic_slug=topic_slug or "all")
        except Exception:
            logger.warning("Failed to save weekly snapshots", exc_info=True)

    return summary_data, md_content, html_content


def _render_md(summary_data: dict) -> str:
    """Render Markdown report using Jinja2 template."""
    template_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = template_dir / "weekly_paper_report.md.j2"

    if template_path.exists():
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("weekly_paper_report.md.j2")
        return template.render(**summary_data)

    # Fallback: simple markdown generation
    return _render_md_fallback(summary_data)


def _render_html(summary_data: dict, config: Any) -> str:
    """Render HTML report using Jinja2 template."""
    template_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = template_dir / "weekly_paper_report.html.j2"

    if template_path.exists():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("weekly_paper_report.html.j2")
        return template.render(**summary_data)

    # Fallback: simple HTML generation
    return _render_html_fallback(summary_data)


def _render_md_fallback(data: dict) -> str:
    """Simple markdown fallback without Jinja2 template."""
    lines = [f"# Weekly Paper Report: {data.get('date_str', '')}"]
    lines.append("")

    sections = data.get("sections", {})

    # Section A
    exec_data = sections.get("executive", {})
    metrics = exec_data.get("metrics", {})
    if metrics:
        lines.append("## Executive Summary")
        lines.append("")
        if exec_data.get("llm_briefing"):
            lines.append(exec_data["llm_briefing"])
            lines.append("")
        lines.append(f"- Total Evaluated: {metrics.get('total_evaluated', 0)}")
        lines.append(f"- Tier 1: {metrics.get('tier1_count', 0)}")
        lines.append(f"- Avg Score: {metrics.get('avg_score', 0)}")
        lines.append(f"- Keyword Hits: {metrics.get('keyword_hits', 0)}")
        lines.append("")

    # Section C
    top = sections.get("top_papers", {})
    papers = top.get("papers", [])
    if papers:
        lines.append("## Top Papers")
        lines.append("")
        for i, p in enumerate(papers, 1):
            tags = " ".join(f"`{t}`" for t in p.get("keyword_tags", []))
            re_mark = " (re-appeared)" if p.get("re_appeared") else ""
            lines.append(f"### {i}. {p['title']}{re_mark}")
            lines.append(f"- Score: {p.get('final_score', 0)} | Topic: {p.get('topic_slug', '')}")
            lines.append(f"- URL: [{p['url']}]({p['url']})")
            if tags:
                lines.append(f"- Tags: {tags}")
            if p.get("summary_ko"):
                lines.append(f"- {p['summary_ko']}")
            lines.append("")

    graduated = top.get("graduated_reminds", [])
    if graduated:
        lines.append("## Graduated Reminds")
        lines.append("")
        for g in graduated:
            lines.append(f"- **{g['title']}** (score: {g.get('final_score', 'N/A')}, graduated: {g.get('graduation_date', '')})")
            lines.append(f"  [{g['url']}]({g['url']})")
            lines.append("")

    return "\n".join(lines)


def _render_html_fallback(data: dict) -> str:
    """Simple HTML fallback without Jinja2 template."""
    import html as html_mod

    sections = data.get("sections", {})
    date_str = data.get("date_str", "")

    lines = [
        "<!DOCTYPE html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>Weekly Paper Report: {date_str}</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 1rem; }",
        "h1 { border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }",
        "h2 { color: #2563eb; margin-top: 2rem; }",
        ".metric { display: inline-block; margin: 0.5rem; padding: 0.5rem 1rem; background: #f3f4f6; border-radius: 8px; }",
        ".metric-value { font-size: 1.5rem; font-weight: 700; }",
        ".metric-label { font-size: 0.8rem; color: #6b7280; }",
        ".paper-card { background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }",
        ".tag { display: inline-block; padding: 0.1rem 0.4rem; background: #dbeafe; color: #1d4ed8; border-radius: 3px; font-size: 0.75rem; margin: 0.1rem; }",
        "a { color: #2563eb; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "table { border-collapse: collapse; width: 100%; margin: 1rem 0; }",
        "th, td { border: 1px solid #e5e7eb; padding: 0.5rem; text-align: left; }",
        "th { background: #2563eb; color: white; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Weekly Paper Report: {date_str}</h1>",
    ]

    # Section A: Executive
    exec_data = sections.get("executive", {})
    metrics = exec_data.get("metrics", {})
    if metrics:
        lines.append("<h2>Executive Summary</h2>")
        if exec_data.get("llm_briefing"):
            lines.append(f"<p>{html_mod.escape(exec_data['llm_briefing'])}</p>")

        lines.append("<div>")
        for label, key in [
            ("Evaluated", "total_evaluated"),
            ("Tier 1", "tier1_count"),
            ("Avg Score", "avg_score"),
            ("Keywords", "keyword_hits"),
        ]:
            val = metrics.get(key, 0)
            lines.append(f"<div class='metric'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>")
        lines.append("</div>")

        # Categories table
        categories = exec_data.get("categories", [])
        if categories:
            lines.append("<h3>Top Categories</h3>")
            lines.append("<table><tr><th>Category</th><th>Count</th><th>WoW %</th></tr>")
            for cat in categories:
                lines.append(f"<tr><td>{html_mod.escape(str(cat['name']))}</td><td>{cat['count']}</td><td>{cat['wow_pct']}</td></tr>")
            lines.append("</table>")

    # Section C: Top Papers
    top = sections.get("top_papers", {})
    papers = top.get("papers", [])
    if papers:
        lines.append("<h2>Top Papers</h2>")
        for i, p in enumerate(papers, 1):
            re_mark = " <span style='color:#f59e0b'>(re-appeared)</span>" if p.get("re_appeared") else ""
            lines.append("<div class='paper-card'>")
            lines.append(f"<strong>#{i}</strong> <a href='{html_mod.escape(p['url'])}'>{html_mod.escape(p['title'])}</a>{re_mark}")
            lines.append(f"<br>Score: {p.get('final_score', 0)} | Topic: {html_mod.escape(str(p.get('topic_slug', '')))}")
            tags = p.get("keyword_tags", [])
            if tags:
                lines.append("<br>" + " ".join(f"<span class='tag'>{html_mod.escape(t)}</span>" for t in tags))
            if p.get("summary_ko"):
                lines.append(f"<br><em>{html_mod.escape(p['summary_ko'])}</em>")
            lines.append("</div>")

    graduated = top.get("graduated_reminds", [])
    if graduated:
        lines.append("<h2>Graduated Reminds</h2>")
        lines.append("<ul>")
        for g in graduated:
            lines.append(f"<li><strong>{html_mod.escape(g['title'])}</strong> (score: {g.get('final_score', 'N/A')})")
            lines.append(f"<br><a href='{html_mod.escape(g['url'])}'>{html_mod.escape(g['url'])}</a></li>")
        lines.append("</ul>")

    lines.extend(["</body>", "</html>"])
    return "\n".join(lines)


def _migrate_weekly_snapshots(conn: Any, provider: str) -> None:
    """Migrate weekly_snapshots table to include topic_slug in PK.

    SQLite: Recreate table with new PK (iso_year, iso_week, section, topic_slug).
    Supabase: Add column + drop/recreate constraint.
    """
    cursor = conn.cursor()
    try:
        # Check if topic_slug column already exists
        if provider == "supabase":
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'weekly_snapshots' AND column_name = 'topic_slug'"
            )
            has_col = len(cursor.fetchall()) > 0
        else:
            cursor.execute("PRAGMA table_info(weekly_snapshots)")
            rows = cursor.fetchall()
            has_col = any(
                (r["name"] if isinstance(r, dict) else r[1]) == "topic_slug"
                for r in rows
            )

        if has_col:
            # Column exists - check if PK includes topic_slug
            if provider == "supabase":
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM information_schema.key_column_usage "
                    "WHERE table_name = 'weekly_snapshots' "
                    "AND constraint_name = 'weekly_snapshots_pkey'"
                )
                row = cursor.fetchone()
                pk_count = row["cnt"] if isinstance(row, dict) else row[0]
                if pk_count < 4:
                    cursor.execute(
                        "ALTER TABLE weekly_snapshots "
                        "DROP CONSTRAINT weekly_snapshots_pkey"
                    )
                    cursor.execute(
                        "ALTER TABLE weekly_snapshots "
                        "ADD PRIMARY KEY (iso_year, iso_week, section, topic_slug)"
                    )
                    conn.commit()
                    logger.info("Supabase: rebuilt weekly_snapshots PK to 4 columns")
            else:
                cursor.execute("PRAGMA table_info(weekly_snapshots)")
                cols = cursor.fetchall()
                pk_cols = [
                    (r["name"] if isinstance(r, dict) else r[1])
                    for r in cols
                    if (r["pk"] if isinstance(r, dict) else r[5]) > 0
                ]
                if len(pk_cols) < 4:
                    _recreate_sqlite_table(conn, has_topic_slug=True)
            return

        # Column doesn't exist - full migration needed
        if provider == "supabase":
            cursor.execute(
                "ALTER TABLE weekly_snapshots "
                "ADD COLUMN topic_slug TEXT NOT NULL DEFAULT 'all'"
            )
            cursor.execute(
                "ALTER TABLE weekly_snapshots "
                "DROP CONSTRAINT weekly_snapshots_pkey"
            )
            cursor.execute(
                "ALTER TABLE weekly_snapshots "
                "ADD PRIMARY KEY (iso_year, iso_week, section, topic_slug)"
            )
            conn.commit()
        else:
            # SQLite: must recreate table to change PK
            _recreate_sqlite_table(conn, has_topic_slug=False)

        logger.info("Migrated weekly_snapshots: added topic_slug to PK")
    except Exception:
        logger.debug("weekly_snapshots migration skipped", exc_info=True)


def _recreate_sqlite_table(conn: Any, has_topic_slug: bool = False) -> None:
    """Recreate weekly_snapshots with new 4-column PK (SQLite only)."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weekly_snapshots_new (
            iso_year    INTEGER NOT NULL,
            iso_week    INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            section     TEXT NOT NULL,
            topic_slug  TEXT NOT NULL DEFAULT 'all',
            data_json   TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (iso_year, iso_week, section, topic_slug)
        )
    """)
    if has_topic_slug:
        cursor.execute("""
            INSERT OR IGNORE INTO weekly_snapshots_new
                (iso_year, iso_week, snapshot_date, section, topic_slug, data_json, created_at)
            SELECT iso_year, iso_week, snapshot_date, section,
                   COALESCE(topic_slug, 'all'), data_json, created_at
            FROM weekly_snapshots
        """)
    else:
        cursor.execute("""
            INSERT OR IGNORE INTO weekly_snapshots_new
                (iso_year, iso_week, snapshot_date, section, topic_slug, data_json, created_at)
            SELECT iso_year, iso_week, snapshot_date, section,
                   'all', data_json, created_at
            FROM weekly_snapshots
        """)
    cursor.execute("DROP TABLE weekly_snapshots")
    cursor.execute("ALTER TABLE weekly_snapshots_new RENAME TO weekly_snapshots")
    conn.commit()
    logger.info("Recreated weekly_snapshots table with 4-column PK")


def _save_snapshots(
    conn: Any, ph: str, sections: dict, iso_year: int, iso_week: int,
    provider: str, topic_slug: str = "all",
) -> None:
    """Save section data as snapshots for future WoW comparison."""
    _migrate_weekly_snapshots(conn, provider)
    cursor = conn.cursor()
    snapshot_date = datetime.now(timezone.utc).isoformat()

    for section_name, section_data in sections.items():
        data_json = json.dumps(section_data, ensure_ascii=False, default=str)

        if provider == "supabase":
            query = f"""
            INSERT INTO weekly_snapshots (iso_year, iso_week, snapshot_date, section, topic_slug, data_json)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            ON CONFLICT (iso_year, iso_week, section, topic_slug)
            DO UPDATE SET data_json = EXCLUDED.data_json, snapshot_date = EXCLUDED.snapshot_date
            """
        else:
            query = f"""
            INSERT OR REPLACE INTO weekly_snapshots
                (iso_year, iso_week, snapshot_date, section, topic_slug, data_json)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            """

        cursor.execute(query, (iso_year, iso_week, snapshot_date, section_name, topic_slug, data_json))

    conn.commit()
