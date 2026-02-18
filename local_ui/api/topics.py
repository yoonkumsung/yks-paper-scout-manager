"""Topics API endpoints.

Provides CRUD operations for topic management.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from local_ui.config_io import add_topic, get_topics, remove_topic, update_topic

logger = logging.getLogger(__name__)

topics_bp = Blueprint("topics", __name__)


def _get_cache_status(data_path: str, topic_slug: str) -> dict:
    """Get cache status for a topic.

    Args:
        data_path: Path to data directory containing keyword_cache
        topic_slug: Topic slug

    Returns:
        Dictionary with exists (bool) and expires_in_days (int | None)
    """
    cache_dir = os.path.join(data_path, "keyword_cache")
    if not os.path.exists(cache_dir):
        return {"exists": False, "expires_in_days": None}

    # Look for cache files matching pattern: {slug}_*.json
    pattern = os.path.join(cache_dir, f"{topic_slug}_*.json")
    cache_files = glob.glob(pattern)

    if not cache_files:
        return {"exists": False, "expires_in_days": None}

    # Get most recent cache file
    latest_cache = max(cache_files, key=os.path.getmtime)
    cache_mtime = os.path.getmtime(latest_cache)
    cache_age_days = (datetime.now().timestamp() - cache_mtime) / (24 * 3600)

    # Cache expires after 30 days
    CACHE_EXPIRY_DAYS = 30
    remaining_days = int(CACHE_EXPIRY_DAYS - cache_age_days)

    return {"exists": True, "expires_in_days": max(0, remaining_days)}


def _get_topic_stats(db_path: str, topic_slug: str) -> dict:
    """Get statistics for a topic from database.

    Args:
        db_path: Path to SQLite database
        topic_slug: Topic slug

    Returns:
        Dictionary with last_run_date, total_collected, total_output
    """
    if not os.path.exists(db_path):
        return {"last_run_date": None, "total_collected": 0, "total_output": 0}

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get last run date and totals
        cursor.execute(
            """
            SELECT display_date_kst, total_collected, total_output
            FROM runs
            WHERE topic_slug = ?
            ORDER BY run_id DESC
            LIMIT 1
            """,
            (topic_slug,),
        )
        row = cursor.fetchone()

        if row:
            return {
                "last_run_date": row["display_date_kst"],
                "total_collected": row["total_collected"],
                "total_output": row["total_output"],
            }
        else:
            return {"last_run_date": None, "total_collected": 0, "total_output": 0}

    except sqlite3.Error as e:
        logger.error(f"Database error getting topic stats: {e}")
        return {"last_run_date": None, "total_collected": 0, "total_output": 0}
    finally:
        if conn is not None:
            conn.close()


def _validate_slug(slug: str) -> bool:
    """Validate slug format.

    Args:
        slug: Slug to validate

    Returns:
        True if slug is valid (lowercase alphanumeric + hyphens only)
    """
    return bool(re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", slug))


def _filter_optional_fields(topic: dict) -> dict:
    """Include optional fields only if they are set.

    Args:
        topic: Topic dictionary

    Returns:
        Topic dictionary with optional fields included only if set
    """
    optional_fields = ["must_concepts_en", "should_concepts_en", "must_not_en"]

    # Create a copy to avoid mutating original
    result = topic.copy()

    for field in optional_fields:
        if field in topic and (not topic[field] or topic[field] == [] or topic[field] == ""):
            # Remove empty optional fields
            del result[field]

    return result


@topics_bp.route("", methods=["GET"])
def list_topics():
    """List all topics with stats, cache status, and optional fields.

    Returns:
        JSON array of topics with statistics, cache status, and filtered optional fields
    """
    try:
        config_path = current_app.config["CONFIG_PATH"]
        db_path = current_app.config["DB_PATH"]
        data_path = current_app.config.get("DATA_PATH", "data")

        topics = get_topics(config_path)

        # Enrich with stats and cache status
        enriched_topics = []
        for topic in topics:
            slug = topic.get("slug", "")

            # Add run stats (flatten into topic for frontend access)
            stats = _get_topic_stats(db_path, slug)
            topic["last_run_stats"] = stats
            topic["last_run_date"] = stats.get("last_run_date")
            topic["total_collected"] = stats.get("total_collected", 0)
            topic["total_output"] = stats.get("total_output", 0)

            # Add cache status
            cache_status = _get_cache_status(data_path, slug)
            topic["cache_status"] = cache_status

            # Filter optional fields (only include if set)
            topic = _filter_optional_fields(topic)
            enriched_topics.append(topic)

        return jsonify(enriched_topics), 200

    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        return jsonify({"error": str(e)}), 500


@topics_bp.route("", methods=["POST"])
def create_topic():
    """Add a new topic with validation.

    Request body:
        {
            "slug": "string",
            "name": "string",
            "description": "string",
            "arxiv_categories": ["string"],
            "notify": [
                {
                    "provider": "discord" | "telegram",
                    "secret_key": "string",
                    "channel_id": "string" (optional, default ""),
                    "events": ["start", "complete"] (optional, default ["complete"])
                }
            ] | null
        }

    Also accepts legacy single-object format for backward compatibility.

    Returns:
        201 on success with created topic
        400 on validation error
        500 on server error
    """
    try:
        data = request.get_json()

        # Validate required fields
        required = ["slug", "name", "description", "arxiv_categories"]
        for field in required:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Validate slug format (lowercase, alphanumeric + hyphens only)
        slug = data["slug"]
        if not _validate_slug(slug):
            return (
                jsonify(
                    {
                        "error": "Invalid slug format. Must be lowercase alphanumeric with hyphens only (e.g., 'my-topic')"
                    }
                ),
                400,
            )

        # Validate name is non-empty
        if not data["name"].strip():
            return jsonify({"error": "name must be non-empty"}), 400

        # Validate description is non-empty
        if not data["description"].strip():
            return jsonify({"error": "description must be non-empty"}), 400

        # Validate arxiv_categories is a non-empty list
        if not isinstance(data["arxiv_categories"], list) or len(data["arxiv_categories"]) == 0:
            return jsonify({"error": "arxiv_categories must be a non-empty list"}), 400

        # Validate notify structure (optional, supports single object or list)
        notify = data.get("notify")
        if notify is not None:
            # Normalize single object to list for backward compatibility
            if isinstance(notify, dict):
                # Single object format: wrap into list with default events
                entry = dict(notify)
                entry.setdefault("events", ["complete"])
                entry.setdefault("channel_id", "")
                notify = [entry]
            elif not isinstance(notify, list):
                return jsonify({"error": "notify must be a list or an object"}), 400

            # Validate each entry in the list
            valid_providers = {"discord", "telegram"}
            valid_events = {"start", "complete"}
            validated = []
            for i, entry in enumerate(notify):
                if not isinstance(entry, dict):
                    return jsonify({"error": f"notify[{i}] must be an object"}), 400

                if "provider" not in entry or not entry["provider"]:
                    return jsonify({"error": f"notify[{i}].provider is required"}), 400
                if entry["provider"] not in valid_providers:
                    return jsonify({"error": f"notify[{i}].provider must be 'discord' or 'telegram'"}), 400

                if "secret_key" not in entry or not entry["secret_key"]:
                    return jsonify({"error": f"notify[{i}].secret_key is required"}), 400

                # channel_id is optional, default to empty string
                entry.setdefault("channel_id", "")

                # events is optional, default to ["complete"]
                events = entry.get("events", ["complete"])
                if not isinstance(events, list) or not events:
                    events = ["complete"]
                for ev in events:
                    if ev not in valid_events:
                        return jsonify({"error": f"notify[{i}].events contains invalid event '{ev}'. Must be 'start' or 'complete'"}), 400
                entry["events"] = events

                validated.append(entry)

            data["notify"] = validated if validated else None
        else:
            data["notify"] = None

        config_path = current_app.config["CONFIG_PATH"]

        # add_topic will raise ValueError if slug already exists
        add_topic(config_path, data)

        return jsonify(data), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating topic: {e}")
        return jsonify({"error": str(e)}), 500


@topics_bp.route("/<slug>", methods=["PUT"])
def update_topic_endpoint(slug: str):
    """Update a topic by slug with description change detection.

    Args:
        slug: Topic slug

    Request body:
        {
            "name": "string" (optional),
            "description": "string" (optional),
            ...
        }

    Returns:
        200 on success with cache_invalidated flag if description changed
        400 on validation error
        404 if topic not found
        500 on server error
    """
    try:
        data = request.get_json()

        config_path = current_app.config["CONFIG_PATH"]

        # Get current topic to detect description change
        topics = get_topics(config_path)
        current_topic = next((t for t in topics if t.get("slug") == slug), None)

        if not current_topic:
            return jsonify({"error": f"Topic with slug '{slug}' not found"}), 404

        # Check if description changed
        old_description = current_topic.get("description", "")
        new_description = data.get("description", old_description)
        description_changed = old_description != new_description

        # Prevent slug modification
        data.pop("slug", None)

        # Validate notify structure if provided
        notify = data.get("notify")
        if notify is not None:
            if isinstance(notify, dict):
                entry = dict(notify)
                entry.setdefault("events", ["complete"])
                entry.setdefault("channel_id", "")
                notify = [entry]
            elif not isinstance(notify, list):
                return jsonify({"error": "notify must be a list or an object"}), 400

            valid_providers = {"discord", "telegram"}
            valid_events = {"start", "complete"}
            validated = []
            for i, entry in enumerate(notify):
                if not isinstance(entry, dict):
                    return jsonify({"error": f"notify[{i}] must be an object"}), 400
                if "provider" not in entry or not entry["provider"]:
                    return jsonify({"error": f"notify[{i}].provider is required"}), 400
                if entry["provider"] not in valid_providers:
                    return jsonify({"error": f"notify[{i}].provider must be 'discord' or 'telegram'"}), 400
                if "secret_key" not in entry or not entry["secret_key"]:
                    return jsonify({"error": f"notify[{i}].secret_key is required"}), 400
                entry.setdefault("channel_id", "")
                events = entry.get("events", ["complete"])
                if not isinstance(events, list) or not events:
                    events = ["complete"]
                for ev in events:
                    if ev not in valid_events:
                        return jsonify({"error": f"notify[{i}].events contains invalid event '{ev}'"}), 400
                entry["events"] = events
                validated.append(entry)
            data["notify"] = validated if validated else None

        # Update the topic
        update_topic(config_path, slug, data)

        # Prepare response
        response = {"message": "Topic updated successfully"}

        if description_changed:
            response["cache_invalidated"] = True

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error updating topic: {e}")
        return jsonify({"error": str(e)}), 500


@topics_bp.route("/<slug>", methods=["DELETE"])
def delete_topic(slug: str):
    """Delete a topic by slug.

    Args:
        slug: Topic slug

    Returns:
        200 on success
        404 if topic not found
        500 on server error
    """
    try:
        config_path = current_app.config["CONFIG_PATH"]
        success = remove_topic(config_path, slug)

        if success:
            return jsonify({"message": "Topic deleted successfully"}), 200
        else:
            return jsonify({"error": "Topic not found"}), 404

    except Exception as e:
        logger.error(f"Error deleting topic: {e}")
        return jsonify({"error": str(e)}), 500
