"""Settings API endpoints.

Provides endpoints for managing non-topic configuration and status checks.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from core.storage.db_connection import get_connection
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
            "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
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
                elif key == "SUPABASE_DB_URL":
                    is_valid = value.startswith("postgresql://")

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
        provider = current_app.config.get("DB_PROVIDER", "sqlite")
        conn_str = current_app.config.get("SUPABASE_DB_URL")
        # Fallback to sqlite if supabase connection string is missing
        if provider == "supabase" and not conn_str:
            provider = "sqlite"

        empty_result = {
            "exists": False,
            "db_path": db_path,
            "provider": provider,
            "file_size_mb": 0,
            "papers": 0,
            "paper_evaluations": 0,
            "runs": 0,
            "query_stats": 0,
            "remind_tracking": 0,
            "last_purge_date": None,
        }

        # Get file size (SQLite only)
        if provider == "sqlite" and not os.path.exists(db_path):
            return jsonify(empty_result), 200

        file_size_mb = 0.0
        if provider == "sqlite" and os.path.exists(db_path):
            file_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)

        # Get record counts
        counts = {}
        with get_connection(db_path, provider, conn_str) as (conn, ph):
            if conn is None:
                return jsonify(empty_result), 200
            cursor = conn.cursor()
            tables = ["papers", "paper_evaluations", "runs", "query_stats", "remind_tracking"]
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
                    row = cursor.fetchone()
                    counts[table] = row["cnt"] if isinstance(row, dict) else row[0]
                except Exception:
                    counts[table] = 0

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
                    "provider": provider,
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


@settings_bp.route("/data-status", methods=["GET"])
def data_status():
    """Get detailed status of all data files and DB tables.

    Returns:
        JSON with DB tables (name, rows, description) and cache files
        (name, path, size, description, exists).
    """
    try:
        db_path = current_app.config["DB_PATH"]
        data_path = current_app.config.get("DATA_PATH", "data")

        # --- DB tables ---
        table_info = [
            {"name": "papers", "description": "수집된 논문 메타데이터"},
            {"name": "paper_evaluations", "description": "논문 평가 및 요약 결과"},
            {"name": "runs", "description": "파이프라인 실행 기록"},
            {"name": "query_stats", "description": "arXiv 검색 쿼리 통계"},
            {"name": "remind_tracking", "description": "재추천 논문 추적"},
        ]

        provider = current_app.config.get("DB_PROVIDER", "sqlite")
        conn_str = current_app.config.get("SUPABASE_DB_URL")
        if provider == "supabase" and not conn_str:
            provider = "sqlite"

        tables = []
        db_error = None
        try:
            with get_connection(db_path, provider, conn_str) as (conn, ph):
                if conn is not None:
                    cursor = conn.cursor()
                    for t in table_info:
                        try:
                            cursor.execute(f"SELECT COUNT(*) AS cnt FROM {t['name']}")
                            row = cursor.fetchone()
                            rows = row["cnt"] if isinstance(row, dict) else row[0]
                        except Exception:
                            rows = 0
                        tables.append({
                            "name": t["name"],
                            "rows": rows,
                            "description": t["description"],
                        })
                else:
                    for t in table_info:
                        tables.append({"name": t["name"], "rows": 0, "description": t["description"]})
        except Exception as e:
            logger.warning(f"DB connection failed: {e}")
            db_error = str(e)
            for t in table_info:
                tables.append({"name": t["name"], "rows": 0, "description": t["description"]})

        # --- Cache files ---
        cache_files_info = [
            {
                "name": "keyword_cache.json",
                "path": os.path.join(data_path, "keyword_cache.json"),
                "description": "키워드 확장 캐시 (LLM 호출 절감)",
            },
            {
                "name": "last_success.json",
                "path": os.path.join(data_path, "last_success.json"),
                "description": "토픽별 마지막 성공 시간 (검색 윈도우 계산용)",
            },
            {
                "name": "model_caps.json",
                "path": os.path.join(data_path, "model_caps.json"),
                "description": "LLM 모델 기능 캐시 (API 호환성 확인용)",
            },
            {
                "name": "seen_items.jsonl",
                "path": os.path.join(data_path, "seen_items.jsonl"),
                "description": "중복 논문 필터링 기록",
            },
        ]

        # Usage directory
        usage_dir = os.path.join(data_path, "usage")
        usage_size = 0
        usage_count = 0
        if os.path.isdir(usage_dir):
            for f in os.listdir(usage_dir):
                fp = os.path.join(usage_dir, f)
                if os.path.isfile(fp):
                    usage_size += os.path.getsize(fp)
                    usage_count += 1

        cache_files = []
        for cf in cache_files_info:
            exists = os.path.exists(cf["path"])
            size_kb = round(os.path.getsize(cf["path"]) / 1024, 1) if exists else 0
            cache_files.append({
                "name": cf["name"],
                "path": cf["path"],
                "exists": exists,
                "size_kb": size_kb,
                "description": cf["description"],
            })

        # Add usage dir as a single entry
        cache_files.append({
            "name": f"usage/ ({usage_count} files)",
            "path": usage_dir,
            "exists": usage_count > 0,
            "size_kb": round(usage_size / 1024, 1),
            "description": "일별 API 사용량 통계",
        })

        # DB file info
        db_exists = os.path.exists(db_path)
        db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2) if db_exists else 0

        result = {
            "db_path": db_path,
            "db_exists": db_exists,
            "db_size_mb": db_size_mb,
            "tables": tables,
            "cache_files": cache_files,
        }
        if db_error:
            result["db_error"] = db_error
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/reset-table", methods=["POST"])
def reset_table():
    """Delete all rows from a specific DB table.

    Request body:
        {"table": "paper_evaluations"}

    Returns:
        200 on success with deleted row count.
    """
    try:
        data = request.get_json()
        if not data or "table" not in data:
            return jsonify({"error": "Missing 'table' field"}), 400

        table = data["table"]
        allowed = {"papers", "paper_evaluations", "runs", "query_stats", "remind_tracking"}
        if table not in allowed:
            return jsonify({"error": f"Invalid table: {table}"}), 400

        db_path = current_app.config["DB_PATH"]
        provider = current_app.config.get("DB_PROVIDER", "sqlite")
        conn_str = current_app.config.get("SUPABASE_DB_URL")
        if provider == "supabase" and not conn_str:
            provider = "sqlite"

        if provider == "sqlite" and not os.path.exists(db_path):
            return jsonify({"error": "Database does not exist"}), 404

        with get_connection(db_path, provider, conn_str) as (conn, ph):
            if conn is None:
                return jsonify({"error": "Database does not exist"}), 404
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
            row = cursor.fetchone()
            count = row["cnt"] if isinstance(row, dict) else row[0]
            cursor.execute(f"DELETE FROM {table}")
            conn.commit()

        logger.info("Reset table '%s': deleted %d rows", table, count)
        return jsonify({"table": table, "deleted": count}), 200

    except Exception as e:
        logger.error(f"Error resetting table: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/reset-cache", methods=["POST"])
def reset_cache():
    """Delete a specific cache file or directory.

    Request body:
        {"path": "data/keyword_cache.json"}

    Returns:
        200 on success.
    """
    try:
        data = request.get_json()
        if not data or "path" not in data:
            return jsonify({"error": "Missing 'path' field"}), 400

        target = data["path"]
        data_path = current_app.config.get("DATA_PATH", "data")

        # Security: only allow deletion within data directory
        resolved = Path(target).resolve()
        allowed_root = Path(data_path).resolve()
        if not str(resolved).startswith(str(allowed_root)):
            return jsonify({"error": "Path outside data directory"}), 403

        if resolved.is_dir():
            import shutil
            shutil.rmtree(str(resolved))
            resolved.mkdir(parents=True, exist_ok=True)
            logger.info("Reset cache directory: %s", target)
        elif resolved.is_file():
            resolved.unlink()
            logger.info("Reset cache file: %s", target)
        else:
            return jsonify({"error": "File not found"}), 404

        return jsonify({"path": target, "deleted": True}), 200

    except Exception as e:
        logger.error(f"Error resetting cache: {e}")
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/reset-db", methods=["POST"])
def reset_db():
    """Delete and recreate the entire database.

    Returns:
        200 on success.
    """
    try:
        db_path = current_app.config["DB_PATH"]
        provider = current_app.config.get("DB_PROVIDER", "sqlite")
        conn_str = current_app.config.get("SUPABASE_DB_URL")
        if provider == "supabase" and not conn_str:
            provider = "sqlite"

        if provider == "supabase":
            # Truncate all tables instead of deleting the file
            tables = ["paper_evaluations", "query_stats", "remind_tracking", "papers", "runs"]
            with get_connection(db_path, provider, conn_str) as (conn, ph):
                if conn is None:
                    return jsonify({"error": "Cannot connect to database"}), 500
                cursor = conn.cursor()
                for table in tables:
                    cursor.execute(f"DELETE FROM {table}")
                conn.commit()
            logger.info("All Supabase tables truncated")
            return jsonify({"deleted": True, "provider": "supabase"}), 200

        # SQLite: delete the file
        if os.path.exists(db_path):
            os.remove(db_path)
            for suffix in ("-wal", "-shm"):
                wal_path = db_path + suffix
                if os.path.exists(wal_path):
                    os.remove(wal_path)

        logger.info("Database deleted: %s", db_path)
        return jsonify({"deleted": True, "path": db_path}), 200

    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return jsonify({"error": str(e)}), 500
