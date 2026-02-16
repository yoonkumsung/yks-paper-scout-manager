"""Pipeline API endpoints.

Provides endpoints for running pipeline operations.

Reference: TASK-052 devspec Section 18-3.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from local_ui.pipeline_runner import PipelineRunner

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)

# Module-level pipeline runner instance
# Initialized lazily on first request
_pipeline_runner: PipelineRunner | None = None


def _get_pipeline_runner() -> PipelineRunner:
    """Get or create the pipeline runner instance."""
    global _pipeline_runner
    if _pipeline_runner is None:
        config_path = current_app.config.get("CONFIG_PATH", "config.yaml")
        db_path = current_app.config.get("DB_PATH", "data/paper_scout.db")
        _pipeline_runner = PipelineRunner(config_path=config_path, db_path=db_path)
    return _pipeline_runner


@pipeline_bp.route("/dry-run", methods=["POST"])
def dry_run():
    """Execute dry-run for a topic.

    Runs Agent 1 (keyword expansion) + QueryBuilder only.
    Does NOT run scoring/summarization - just keywords and queries.

    Request body:
        {
            "topic_slug": "string | null"  # null means all topics
        }

    Returns:
        200: Success with keywords and queries
        {
            "success": true,
            "topics": [
                {
                    "slug": "topic-slug",
                    "concepts": [...],
                    "cross_domain_keywords": [...],
                    "exclude_keywords": [...],
                    "queries": [...]
                }
            ]
        }

        500: Error
        {"error": "error message"}
    """
    try:
        data = request.get_json() or {}
        topic_slug = data.get("topic_slug")

        runner = _get_pipeline_runner()
        result = runner.start_dryrun(topic_slug=topic_slug)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify({"error": result.get("error", "Unknown error")}), 500

    except Exception as e:
        logger.error(f"Error in dry-run: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/run", methods=["POST"])
def run():
    """Execute full pipeline run in background.

    Runs complete pipeline: collect -> score -> summarize -> report.
    Returns immediately with run_id for status polling.

    Request body:
        {
            "topic_slug": "string | null",
            "date_from": "YYYY-MM-DD | null",
            "date_to": "YYYY-MM-DD | null",
            "dedup": "skip_recent" | "none"
        }

    Returns:
        200: Run started
        {"run_id": "abc123", "status": "started"}

        400: Already running
        {"error": "Pipeline already running", "run_id": "current_run_id"}

        500: Error
        {"error": "error message"}
    """
    try:
        data = request.get_json() or {}
        topic_slug = data.get("topic_slug")
        date_from = data.get("date_from")
        date_to = data.get("date_to")
        dedup = data.get("dedup", "skip_recent")

        runner = _get_pipeline_runner()
        result = runner.start_run(
            topic_slug=topic_slug,
            date_from=date_from,
            date_to=date_to,
            dedup=dedup,
        )

        if "error" in result:
            return jsonify(result), 400
        else:
            return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in pipeline run: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/status", methods=["GET"])
def status():
    """Get current pipeline status.

    Used for polling during pipeline execution.

    Returns:
        200: Current status
        {
            "running": bool,
            "run_id": "string | null",
            "progress": "string",
            "topics_completed": int,
            "topics_total": int,
            "current_topic": "string | null",
            "error": "string | null"
        }
    """
    try:
        runner = _get_pipeline_runner()
        status_data = runner.get_status()
        return jsonify(status_data), 200

    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return jsonify({"error": str(e)}), 500
