"""HTML report generator for Paper Scout (TASK-026).

Generates HTML report files using Jinja2 templates following devspec 10-2.
Produces:
  - ``reports/YYYY-MM-DD/{YYYYMMDD}_paper_{slug}.html`` -- per-topic report
  - ``index.html`` -- topic list with date navigation
  - ``latest.html`` -- always-overwritten bookmark URL

Security: Jinja2 autoescape=True, no ``|safe`` filter usage.

Compatible with Python 3.9+.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flag display mapping
# ---------------------------------------------------------------------------

_FLAG_LABELS: Dict[str, str] = {
    "is_edge": "엣지",
    "is_realtime": "실시간",
    "has_code": "코드",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report_html(
    report_data: dict,
    output_dir: str = "tmp/reports",
    template_dir: str = "templates",
) -> str:
    """Generate an HTML report page.

    Args:
        report_data: Dict from JSON exporter containing ``meta``, ``papers``,
            ``clusters``, and ``remind_papers``.
        output_dir: Directory to write the report file to.
        template_dir: Directory containing Jinja2 template files.

    Returns:
        Absolute path to the written ``.html`` file.
    """
    env = _create_env(template_dir)
    template = env.get_template("report.html.j2")

    meta: dict = report_data.get("meta", {})
    papers: List[dict] = report_data.get("papers", [])
    clusters: List[dict] = report_data.get("clusters", [])
    remind_papers: List[dict] = report_data.get("remind_papers", [])
    discarded_papers: List[dict] = report_data.get("discarded_papers", [])

    # Build cluster mate lookup.
    key_to_rank = _build_key_to_rank(papers)
    key_to_cluster_mates = _build_cluster_mates(clusters, key_to_rank)

    # Inject cluster_mates into each paper dict for template access.
    enriched_papers = _enrich_papers_with_cluster_mates(
        papers, key_to_cluster_mates
    )

    rendered = template.render(
        meta=meta,
        papers=enriched_papers,
        remind_papers=remind_papers,
        discarded_papers=discarded_papers,
    )

    # Write output file.
    os.makedirs(output_dir, exist_ok=True)
    filename = _build_filename(meta)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    abs_path = os.path.abspath(filepath)
    logger.info("HTML report written to %s", abs_path)
    return abs_path


def generate_index_html(
    reports: List[dict],
    output_dir: str = "tmp/reports",
    template_dir: str = "templates",
) -> str:
    """Generate index.html with topic/date navigation.

    Args:
        reports: List of dicts with keys ``topic_slug``, ``topic_name``,
            ``date``, ``filepath``.
        output_dir: Directory to write index.html into.
        template_dir: Directory containing Jinja2 template files.

    Returns:
        Absolute path to the written ``index.html``.
    """
    env = _create_env(template_dir)
    template = env.get_template("index.html.j2")

    rendered = template.render(reports=reports)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "index.html")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    abs_path = os.path.abspath(filepath)
    logger.info("Index HTML written to %s", abs_path)
    return abs_path


def generate_latest_html(
    report_data: dict,
    output_dir: str = "tmp/reports",
    template_dir: str = "templates",
) -> str:
    """Generate latest.html that always overwrites for bookmarking.

    Uses the same report template but always writes to ``latest.html``.

    Args:
        report_data: Dict from JSON exporter containing ``meta``, ``papers``,
            ``clusters``, and ``remind_papers``.
        output_dir: Directory to write latest.html into.
        template_dir: Directory containing Jinja2 template files.

    Returns:
        Absolute path to the written ``latest.html``.
    """
    env = _create_env(template_dir)
    template = env.get_template("report.html.j2")

    meta: dict = report_data.get("meta", {})
    papers: List[dict] = report_data.get("papers", [])
    clusters: List[dict] = report_data.get("clusters", [])
    remind_papers: List[dict] = report_data.get("remind_papers", [])
    discarded_papers: List[dict] = report_data.get("discarded_papers", [])

    # Build cluster mate lookup.
    key_to_rank = _build_key_to_rank(papers)
    key_to_cluster_mates = _build_cluster_mates(clusters, key_to_rank)

    enriched_papers = _enrich_papers_with_cluster_mates(
        papers, key_to_cluster_mates
    )

    rendered = template.render(
        meta=meta,
        papers=enriched_papers,
        remind_papers=remind_papers,
        discarded_papers=discarded_papers,
    )

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "latest.html")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    abs_path = os.path.abspath(filepath)
    logger.info("Latest HTML written to %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_KST = timezone(timedelta(hours=9))


def _format_window_date(iso_str: str) -> str:
    """Convert ISO 8601 datetime string to human-readable dual-timezone format.

    Example:
        '2026-01-01T00:00:00+00:00'
        -> 'UTC 2026-01-01 00:00 / KST 2026-01-01 09:00'
    """
    if not iso_str:
        return iso_str
    try:
        dt = datetime.fromisoformat(iso_str)
        utc_dt = dt.astimezone(timezone.utc)
        kst_dt = dt.astimezone(_KST)
        return "UTC %s / KST %s" % (
            utc_dt.strftime("%Y-%m-%d %H:%M"),
            kst_dt.strftime("%Y-%m-%d %H:%M"),
        )
    except (ValueError, TypeError):
        return iso_str


def _create_env(template_dir: str) -> Environment:
    """Create a Jinja2 Environment with autoescape enabled.

    Security requirement: autoescape=True prevents XSS.
    """
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["window_date"] = _format_window_date
    return env


def _build_filename(meta: dict) -> str:
    """Build the output filename from meta.

    Format: ``{YYYYMMDD}_paper_{slug}.html``
    """
    date_str = meta.get("date", "")
    slug = meta.get("topic_slug", "unknown")
    # date is in "YYYY-MM-DD" format; convert to "YYYYMMDD"
    date_compact = date_str.replace("-", "")
    return "%s_paper_%s.html" % (date_compact, slug)


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


def _enrich_papers_with_cluster_mates(
    papers: List[dict],
    key_to_cluster_mates: Dict[str, List[str]],
) -> List[dict]:
    """Add ``cluster_mates`` key to each paper dict for template rendering.

    Returns new list of dicts (does not mutate originals).
    """
    enriched: List[dict] = []
    for paper in papers:
        enriched_paper = dict(paper)
        paper_key = paper.get("paper_key", "")
        enriched_paper["cluster_mates"] = key_to_cluster_mates.get(
            paper_key, []
        )
        enriched.append(enriched_paper)
    return enriched
