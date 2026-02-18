"""JSON exporter for Paper Scout reports.

Generates report JSON files following the devspec 10-4 schema.
Produces files named ``{YYYYMMDD}_paper_{slug}.json`` containing
meta, clusters, papers, and remind_papers sections.

Section reference: TASK-024 from SPEC-PAPER-001.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _DataclassEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclass and datetime instances."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return super().default(o)


def export_json(
    topic_slug: str,
    topic_name: str,
    date_str: str,  # YYYYMMDD
    display_title: str,
    window_start_utc: str,  # ISO 8601
    window_end_utc: str,  # ISO 8601
    embedding_mode: str,
    scoring_weights: dict,
    stats: dict,  # total_collected, total_filtered, total_discarded, total_scored, total_output
    threshold_used: int,
    threshold_lowered: bool,
    run_id: int,
    keywords_used: List[str],
    papers: List[dict],  # ranked papers with evaluations/summaries
    clusters: List[dict],  # from Clusterer
    remind_papers: List[dict],  # remind tab papers
    discarded_papers: Optional[List[dict]] = None,  # discarded papers with reason
    output_dir: str = "tmp/reports",
) -> str:
    """Export report data as JSON file.

    Assembles the full report payload following the devspec 10-4 schema
    and writes it to disk as a UTF-8 encoded JSON file.

    Args:
        topic_slug: URL-safe topic identifier (e.g. ``"ai-sports-device"``).
        topic_name: Human-readable topic name.
        date_str: Report date in ``YYYYMMDD`` format.
        display_title: Localized display title for the report.
        window_start_utc: Collection window start in ISO 8601 format.
        window_end_utc: Collection window end in ISO 8601 format.
        embedding_mode: Embedding strategy used (e.g. ``"en_synthetic"``).
        scoring_weights: Dict of scoring weight factors.
        stats: Aggregated pipeline statistics dict with keys
            ``total_collected``, ``total_filtered``, ``total_discarded``,
            ``total_scored``, ``total_output``.
        threshold_used: Final score threshold used for filtering.
        threshold_lowered: Whether the threshold was dynamically lowered.
        run_id: Unique pipeline run identifier.
        keywords_used: Keywords used for paper collection.
        papers: List of ranked paper dicts with evaluations and summaries.
        clusters: List of cluster dicts from :class:`Clusterer`.
        remind_papers: List of remind-tab paper dicts.
        discarded_papers: List of discarded paper dicts with title, url, reason.
        output_dir: Directory to write the JSON file into.
            Created if it does not exist.  Defaults to ``"tmp/reports"``.

    Returns:
        Absolute path to the written JSON file.
    """
    # Format date from YYYYMMDD to YYYY-MM-DD
    formatted_date = _format_date(date_str)

    # Build the report payload
    payload: Dict[str, Any] = {
        "meta": {
            "topic_name": topic_name,
            "topic_slug": topic_slug,
            "date": formatted_date,
            "display_title": display_title,
            "window_start_utc": window_start_utc,
            "window_end_utc": window_end_utc,
            "embedding_mode": embedding_mode,
            "scoring_weights": scoring_weights,
            "total_collected": stats.get("total_collected", 0),
            "total_filtered": stats.get("total_filtered", 0),
            "total_discarded": stats.get("total_discarded", 0),
            "total_scored": stats.get("total_scored", 0),
            "total_output": stats.get("total_output", 0),
            "threshold_used": threshold_used,
            "threshold_lowered": threshold_lowered,
            "run_id": run_id,
            "keywords_used": keywords_used,
        },
        "clusters": clusters,
        "papers": papers,
        "remind_papers": remind_papers,
        "discarded_papers": discarded_papers or [],
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build filename: {YYYYMMDD}_paper_{slug}.json
    filename = f"{date_str}_paper_{topic_slug}.json"
    filepath = os.path.join(output_dir, filename)

    # Write JSON with UTF-8 encoding (ensure_ascii=False for Korean text)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_DataclassEncoder)

    abs_path = os.path.abspath(filepath)
    logger.info("JSON report written to %s", abs_path)

    return abs_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_date(date_str: str) -> str:
    """Convert ``YYYYMMDD`` to ``YYYY-MM-DD``.

    Args:
        date_str: Date string in ``YYYYMMDD`` format.

    Returns:
        Date string in ``YYYY-MM-DD`` format.

    Raises:
        ValueError: If *date_str* is not exactly 8 characters.
    """
    if len(date_str) != 8:
        raise ValueError(
            f"date_str must be 8 characters (YYYYMMDD), got {len(date_str)!r}"
        )
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
