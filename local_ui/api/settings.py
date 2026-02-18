"""Settings API endpoints.

Provides endpoints for managing non-topic configuration and status checks.
"""

from __future__ import annotations

import logging
import os
import sqlite3

from flask import Blueprint, current_app, jsonify, request

from local_ui.config_io import read_config, write_config

logger = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("", methods=["GET"])
def get_settings():
    """Get current settings (non-topic config).

    Returns:
        JSON with all non-topic configuration sections
    """
    try:
        config_path = current_app.config["CONFIG_PATH"]
        config = read_config(config_path)

        # Remove topics from response
        settings = {k: v for k, v in config.items() if k != "topics"}

        return jsonify(settings), 200

    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("", methods=["PUT"])
def update_settings():
    """Update settings (non-topic sections).

    Request body:
        {
            "app": {...},
            "llm": {...},
            ...
        }

    Returns:
        200 on success
        400 on validation error
        500 on server error
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate data type
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        # Prevent updating topics through this endpoint
        if "topics" in data:
            return jsonify({"error": "Cannot update topics through settings endpoint"}), 400

        config_path = current_app.config["CONFIG_PATH"]
        config = read_config(config_path)

        # Update non-topic sections
        for key, value in data.items():
            if key != "topics":
                # Validate that value is also a dict (config sections are dicts)
                if not isinstance(value, dict):
                    return jsonify({"error": f"Invalid value for section '{key}': must be an object"}), 400
                config[key] = value

        write_config(config_path, config)

        return jsonify({"message": "Settings updated successfully"}), 200

    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/env-status", methods=["GET"])
def env_status():
    """Get .env API key status (masked with validity checks).

    Returns:
        JSON with environment variable status, masking, and validity
    """
    try:
        env_keys = {
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "DISCORD_WEBHOOK_URL": os.getenv("DISCORD_WEBHOOK_URL"),
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
            "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
        }

        status = {}
        for key, value in env_keys.items():
            if value:
                # Mask the value (show first 4 characters + ***)
                if len(value) > 4:
                    masked = f"{value[:4]}***"
                else:
                    masked = "***"

                # Validate format
                is_valid = True
                if key == "OPENROUTER_API_KEY":
                    is_valid = value.startswith("sk-or-")
                elif key == "DISCORD_WEBHOOK_URL":
                    is_valid = value.startswith("https://discord.com/api/webhooks/")
                elif key == "TELEGRAM_BOT_TOKEN":
                    is_valid = ":" in value  # Basic format: digits:token
                elif key == "GITHUB_TOKEN":
                    is_valid = value.startswith("ghp_") or value.startswith("github_pat_")

                status[key] = {
                    "exists": True,
                    "masked_value": masked,
                    "valid": is_valid
                }
            else:
                status[key] = {
                    "exists": False,
                    "masked_value": None,
                    "valid": False
                }

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error getting env status: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/db-status", methods=["GET"])
def db_status():
    """Get comprehensive DB stats.

    Returns:
        JSON with database record counts, file size, and maintenance info
    """
    try:
        db_path = current_app.config["DB_PATH"]

        if not os.path.exists(db_path):
            return (
                jsonify(
                    {
                        "exists": False,
                        "db_path": db_path,
                        "file_size_mb": 0,
                        "papers": 0,
                        "paper_evaluations": 0,
                        "runs": 0,
                        "query_stats": 0,
                        "remind_tracking": 0,
                        "last_purge_date": None,
                    }
                ),
                200,
            )

        # Get file size
        file_size_bytes = os.path.getsize(db_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        # Get record counts for all tables
        conn = None
        counts = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            tables = ["papers", "paper_evaluations", "runs", "query_stats", "remind_tracking"]
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    counts[table] = 0
        finally:
            if conn is not None:
                conn.close()

        # Check for last purge date from weekly_done.flag
        last_purge_date = None
        flag_path = os.path.join(os.path.dirname(db_path), "..", "weekly_done.flag")
        if os.path.exists(flag_path):
            import time
            mtime = os.path.getmtime(flag_path)
            last_purge_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

        return (
            jsonify(
                {
                    "exists": True,
                    "db_path": db_path,
                    "file_size_mb": file_size_mb,
                    "papers": counts.get("papers", 0),
                    "paper_evaluations": counts.get("paper_evaluations", 0),
                    "runs": counts.get("runs", 0),
                    "query_stats": counts.get("query_stats", 0),
                    "remind_tracking": counts.get("remind_tracking", 0),
                    "last_purge_date": last_purge_date,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting DB status: {e}")
        return jsonify({"error": str(e)}), 500
