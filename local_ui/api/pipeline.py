"""Pipeline API endpoints.

Provides endpoints for running pipeline operations.

Reference: TASK-052 devspec Section 18-3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request

from local_ui.config_io import read_config
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
        skip_cache = data.get("skip_cache", False)

        runner = _get_pipeline_runner()
        result = runner.start_dryrun(topic_slug=topic_slug, skip_cache=skip_cache)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify({"error": result.get("error", "Unknown error")}), 500

    except Exception as e:
        logger.error(f"Error in dry-run: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/dry-run-stream", methods=["GET"])
def dry_run_stream():
    """SSE endpoint for dry-run with real-time progress.

    Streams progress events during keyword expansion and query building.

    Query parameters:
        topic_slug: Specific topic slug, or empty for all topics

    SSE events:
        event: log   - progress messages with step info
        event: result - final JSON result
        event: error  - error messages
    """
    topic_slug = request.args.get("topic_slug") or None
    skip_cache = request.args.get("skip_cache") == "1"

    runner = _get_pipeline_runner()

    def generate():
        try:
            for event_type, data in runner.start_dryrun_streamed(topic_slug=topic_slug, skip_cache=skip_cache):
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            logger.error(f"Error in dry-run stream: {e}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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


@pipeline_bp.route("/logs", methods=["GET"])
def logs():
    """Get pipeline execution logs with offset-based pagination.

    Query parameters:
        offset: Line offset to start reading from (default 0)

    Returns:
        200: Log lines from the current/last run
        {
            "lines": [{"ts": "...", "step": "...", "msg": "..."}, ...],
            "total": int
        }
    """
    try:
        runner = _get_pipeline_runner()
        status_data = runner.get_status()
        log_file = status_data.get("log_file")

        offset = int(request.args.get("offset", 0))

        if not log_file or not os.path.exists(log_file):
            return jsonify({"lines": [], "total": 0}), 200

        lines = []
        total = 0
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            total = len(all_lines)
            for raw_line in all_lines[offset:]:
                raw_line = raw_line.strip()
                if raw_line:
                    try:
                        lines.append(json.loads(raw_line))
                    except json.JSONDecodeError:
                        lines.append({"ts": "", "step": "raw", "msg": raw_line})
        except OSError:
            pass

        return jsonify({"lines": lines, "total": total}), 200

    except Exception as e:
        logger.error(f"Error reading pipeline logs: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/cancel", methods=["POST"])
def cancel():
    """Cancel running pipeline execution.

    Returns:
        200: Cancel result
        {"success": bool, "message": "string"}
    """
    try:
        runner = _get_pipeline_runner()
        result = runner.cancel()
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error cancelling pipeline: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Keyword cache helpers
# ------------------------------------------------------------------

_REQUIRED_RESULT_KEYS = {
    "concepts",
    "cross_domain_keywords",
    "exclude_keywords",
    "topic_embedding_text",
}


def _find_topic_in_config(topic_slug: str) -> dict | None:
    """Find a topic dict in config.yaml by slug."""
    config_path = current_app.config.get("CONFIG_PATH", "config.yaml")
    config = read_config(config_path)
    for topic in config.get("topics", []):
        if topic.get("slug") == topic_slug:
            return topic
    return None


def _compute_cache_key(topic: dict) -> str:
    """Compute SHA-256 cache key matching KeywordExpander logic."""
    parts = [topic["description"]]
    must = topic.get("must_concepts_en")
    if must:
        parts.append("|must:" + ",".join(sorted(must)))
    should = topic.get("should_concepts_en")
    if should:
        parts.append("|should:" + ",".join(sorted(should)))
    must_not = topic.get("must_not_en")
    if must_not:
        parts.append("|not:" + ",".join(sorted(must_not)))
    raw = "".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _keyword_cache_file() -> Path:
    """Resolve keyword cache path relative to config file directory."""
    config_path = Path(current_app.config.get("CONFIG_PATH", "config.yaml")).resolve()
    return config_path.parent / "data" / "keyword_cache.json"


def _legacy_keyword_cache_file() -> Path:
    """Legacy cache location resolved from process cwd."""
    return Path("data/keyword_cache.json").resolve()


def _load_keyword_cache() -> dict:
    """Load keyword cache file. Return empty dict if missing/corrupt."""
    merged: dict = {}
    primary = _keyword_cache_file()
    legacy = _legacy_keyword_cache_file()

    files = [primary]
    if legacy != primary:
        files.append(legacy)

    for idx, cache_file in enumerate(files):
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            if idx == 0:
                merged.update(data)
            else:
                # Keep primary entries authoritative.
                for k, v in data.items():
                    merged.setdefault(k, v)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue

    return merged


def _save_keyword_cache(cache: dict) -> None:
    """Save keyword cache to file."""
    cache_file = _keyword_cache_file()
    cache_dir = os.path.dirname(str(cache_file))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Keyword cache endpoints
# ------------------------------------------------------------------


@pipeline_bp.route("/keywords/<topic_slug>", methods=["GET"])
def get_keywords(topic_slug: str):
    """Get cached keywords for a topic.

    Returns the cached keyword expansion result for the given topic,
    or 404 if no cache entry exists.

    Returns:
        200: Cached result
        {
            "result": { "concepts": [...], ... },
            "cached_at": "ISO timestamp",
            "prompt_version": "..."
        }

        404: Not cached
        {"error": "No cached keywords for topic '<slug>'"}
    """
    try:
        topic = _find_topic_in_config(topic_slug)
        if topic is None:
            return jsonify({"error": f"Topic '{topic_slug}' not found in config"}), 404

        cache_key = _compute_cache_key(topic)
        cache = _load_keyword_cache()

        if cache_key not in cache:
            return jsonify({"error": f"No cached keywords for topic '{topic_slug}'"}), 404

        return jsonify(cache[cache_key]), 200

    except Exception as e:
        logger.error(f"Error getting keywords for {topic_slug}: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/keywords/<topic_slug>", methods=["PUT"])
def update_keywords(topic_slug: str):
    """Update cached keywords for a topic.

    Request body must contain the result structure:
        {
            "concepts": [...],
            "cross_domain_keywords": [...],
            "exclude_keywords": [...],
            "topic_embedding_text": "..."
        }

    Returns:
        200: Success
        {"success": true, "cache_key": "..."}

        400: Validation error
        {"error": "Missing required keys: ..."}

        404: Topic not found
        {"error": "Topic '<slug>' not found in config"}
    """
    try:
        topic = _find_topic_in_config(topic_slug)
        if topic is None:
            return jsonify({"error": f"Topic '{topic_slug}' not found in config"}), 404

        body = request.get_json()
        if not body:
            return jsonify({"error": "Request body is required"}), 400

        missing = _REQUIRED_RESULT_KEYS - set(body.keys())
        if missing:
            return jsonify({"error": f"Missing required keys: {', '.join(sorted(missing))}"}), 400

        cache_key = _compute_cache_key(topic)
        cache = _load_keyword_cache()
        previous_result = cache.get(cache_key, {}).get("result", {})
        if not isinstance(previous_result, dict):
            previous_result = {}

        # Backward compatibility:
        # If an older UI payload omits optional fields, preserve existing ones.
        body.setdefault(
            "query_must_keywords",
            previous_result.get("query_must_keywords", []),
        )
        body.setdefault(
            "exclude_mode",
            previous_result.get("exclude_mode", "soft"),
        )

        cache[cache_key] = {
            "result": body,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "prompt_version": "manual-edit",
        }
        _save_keyword_cache(cache)

        return jsonify({"success": True, "cache_key": cache_key}), 200

    except Exception as e:
        logger.error(f"Error updating keywords for {topic_slug}: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/keywords/<topic_slug>", methods=["DELETE"])
def delete_keywords(topic_slug: str):
    """Delete cached keywords for a topic.

    Returns:
        200: Success
        {"success": true}

        404: Topic or cache entry not found
        {"error": "..."}
    """
    try:
        topic = _find_topic_in_config(topic_slug)
        if topic is None:
            return jsonify({"error": f"Topic '{topic_slug}' not found in config"}), 404

        cache_key = _compute_cache_key(topic)
        cache = _load_keyword_cache()

        if cache_key not in cache:
            return jsonify({"error": f"No cached keywords for topic '{topic_slug}'"}), 404

        del cache[cache_key]
        _save_keyword_cache(cache)

        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(f"Error deleting keywords for {topic_slug}: {e}")
        return jsonify({"error": str(e)}), 500
