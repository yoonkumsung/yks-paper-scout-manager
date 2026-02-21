"""Flask blueprint for weekly intelligence report API.

Provides endpoints to retrieve, generate, and configure
weekly intelligence reports.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from flask import Blueprint, current_app, jsonify, request

weekly_intel_bp = Blueprint("weekly_intel", __name__)

logger = logging.getLogger(__name__)

# Simple lock to prevent concurrent weekly generation
_generation_lock = threading.Lock()
_generation_status: dict[str, Any] = {"running": False, "run_id": None, "error": None}


@weekly_intel_bp.route("/latest", methods=["GET"])
def latest() -> tuple[Any, int]:
    """Get the latest weekly intelligence report data.

    Returns the most recent report from weekly_snapshots table,
    assembled from all section rows.
    """
    from core.storage.db_connection import get_connection

    db_path = current_app.config["DB_PATH"]
    provider = current_app.config.get("DB_PROVIDER", "sqlite")
    conn_str = current_app.config.get("SUPABASE_DB_URL")

    try:
        with get_connection(db_path, provider, conn_str) as (conn, ph):
            if conn is None:
                return jsonify({"error": "Database unavailable"}), 503

            cursor = conn.cursor()

            # Get the latest iso_year, iso_week
            cursor.execute(
                "SELECT iso_year, iso_week FROM weekly_snapshots "
                "ORDER BY iso_year DESC, iso_week DESC LIMIT 1"
            )
            latest_row = cursor.fetchone()

            if not latest_row:
                return jsonify({"error": "No weekly reports available"}), 404

            if isinstance(latest_row, dict):
                latest_year = latest_row["iso_year"]
                latest_week = latest_row["iso_week"]
            else:
                latest_year = latest_row[0]
                latest_week = latest_row[1]

            # Get all sections for that week
            cursor.execute(
                f"SELECT section, data_json, snapshot_date FROM weekly_snapshots "
                f"WHERE iso_year = {ph} AND iso_week = {ph}",
                (latest_year, latest_week),
            )
            rows = cursor.fetchall()

    except Exception as e:
        logger.error("Failed to fetch latest weekly report: %s", e)
        return jsonify({"error": "Database query failed"}), 500

    sections = {}
    snapshot_date = ""
    for row in rows:
        if isinstance(row, dict):
            section = row["section"]
            data = row["data_json"]
            snapshot_date = row.get("snapshot_date", "")
        else:
            section = row[0]
            data = row[1]
            snapshot_date = row[2] if len(row) > 2 else ""

        try:
            sections[section] = json.loads(data) if isinstance(data, str) else data
        except (json.JSONDecodeError, TypeError):
            sections[section] = {}

    return jsonify({
        "iso_year": latest_year,
        "iso_week": latest_week,
        "snapshot_date": snapshot_date,
        "sections": sections,
    }), 200


@weekly_intel_bp.route("/run", methods=["POST"])
def run_generation() -> tuple[Any, int]:
    """Trigger weekly intelligence report generation.

    Runs in a background thread. Returns 409 if already running.
    """
    global _generation_status

    if _generation_status["running"]:
        return jsonify({
            "error": "Weekly intelligence generation already running",
            "run_id": _generation_status.get("run_id"),
        }), 409

    # Check if pipeline is running
    try:
        from local_ui.api.pipeline import _pipeline_runner
        if _pipeline_runner and _pipeline_runner.get_status().get("running"):
            return jsonify({
                "error": "Pipeline is currently running",
            }), 409
    except (ImportError, Exception):
        pass

    import uuid
    run_id = str(uuid.uuid4())[:8]

    thread = threading.Thread(
        target=_run_weekly_background,
        args=(run_id,),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started", "run_id": run_id}), 202


@weekly_intel_bp.route("/status", methods=["GET"])
def generation_status() -> tuple[Any, int]:
    """Get the current generation status."""
    return jsonify(_generation_status), 200


@weekly_intel_bp.route("/config", methods=["GET"])
def get_config() -> tuple[Any, int]:
    """Get current weekly intelligence configuration."""
    import yaml

    config_path = current_app.config.get("CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        intel_config = raw.get("weekly", {}).get("intelligence", {})
        return jsonify(intel_config), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@weekly_intel_bp.route("/config", methods=["PUT"])
def update_config() -> tuple[Any, int]:
    """Update weekly intelligence configuration."""
    import yaml

    config_path = current_app.config.get("CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        new_config = request.get_json()
        if not new_config:
            return jsonify({"error": "No configuration provided"}), 400

        if "weekly" not in raw:
            raw["weekly"] = {}
        raw["weekly"]["intelligence"] = new_config

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)

        return jsonify({"status": "updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _run_weekly_background(run_id: str) -> None:
    """Background thread for weekly intelligence generation."""
    global _generation_status

    _generation_status = {"running": True, "run_id": run_id, "error": None}

    try:
        from datetime import datetime, timezone
        from core.config import load_config
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        config = load_config()
        db_path = config.database.get("path", "data/paper_scout.db")
        provider = config.database.get("provider", "sqlite")
        conn_str = None
        if provider == "supabase":
            import os
            env_key = config.database.get("supabase", {}).get(
                "connection_string_env", "SUPABASE_DB_URL"
            )
            conn_str = os.environ.get(env_key)

        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        summary_data, md_content, html_content = generate_weekly_intelligence(
            db_path=db_path,
            date_str=today_str,
            config=config,
            provider=provider,
            connection_string=conn_str,
        )

        # Save output files
        import os
        from pathlib import Path

        report_dir = config.output.get("report_dir", "tmp/reports")
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        if md_content:
            md_path = os.path.join(report_dir, f"{today_str}_weekly_paper_report.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        if html_content:
            html_path = os.path.join(report_dir, f"{today_str}_weekly_paper_report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        _generation_status = {
            "running": False,
            "run_id": run_id,
            "error": None,
            "completed": True,
        }

    except Exception as e:
        logger.error("Weekly intelligence generation failed: %s", e, exc_info=True)
        _generation_status = {
            "running": False,
            "run_id": run_id,
            "error": str(e),
        }
