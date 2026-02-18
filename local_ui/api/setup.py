"""Setup API endpoints for the Setup Wizard.

Provides endpoints for .env management, GitHub Secrets registration,
and unified save-and-deploy workflow.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from flask import Blueprint, current_app, jsonify, request

from local_ui.config_io import read_config, write_config
from local_ui.env_io import read_env, write_env
from local_ui.github_secrets import detect_github_repo, push_secrets

logger = logging.getLogger(__name__)

setup_bp = Blueprint("setup", __name__)


@setup_bp.route("/status", methods=["GET"])
def setup_status():
    """Get overall setup status.

    Returns:
        JSON with env status, GitHub repo info, and config readiness.
    """
    try:
        env_status = read_env()
        repo_info = detect_github_repo()

        # Check if config exists
        config_path = current_app.config["CONFIG_PATH"]
        try:
            config = read_config(config_path)
            config_ok = bool(config.get("topics"))
        except Exception:
            config_ok = False

        # Determine if setup is needed (only required keys matter)
        required_keys = ["OPENROUTER_API_KEY"]
        required_missing = any(
            not env_status[k]["exists"] for k in required_keys if k in env_status
        )
        needs_setup = required_missing or not config_ok

        return jsonify({
            "needs_setup": needs_setup,
            "env": env_status,
            "repo": repo_info,
            "config_ok": config_ok,
        }), 200

    except Exception as e:
        logger.error(f"Error getting setup status: {e}")
        return jsonify({"error": str(e)}), 500


@setup_bp.route("/env", methods=["POST"])
def save_env():
    """Save API keys to .env file.

    Request body:
        {"OPENROUTER_API_KEY": "sk-or-...", "GITHUB_TOKEN": "ghp_..."}

    Returns:
        JSON with per-key results.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        results = write_env(data)

        # Reload dotenv so the app picks up new values
        load_dotenv(override=True)

        return jsonify({
            "message": "Environment variables saved",
            "results": results,
        }), 200

    except Exception as e:
        logger.error(f"Error saving env: {e}")
        return jsonify({"error": str(e)}), 500


@setup_bp.route("/github-secrets", methods=["POST"])
def save_github_secrets():
    """Push secrets to GitHub Actions.

    Request body:
        {"OPENROUTER_API_KEY": "sk-or-...", "GITHUB_TOKEN": "ghp_..."}

    Returns:
        JSON with method used and per-key results.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        result = push_secrets(data)
        status_code = 200 if result["success"] else 207

        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Error pushing GitHub secrets: {e}")
        return jsonify({"error": str(e)}), 500


@setup_bp.route("/save-and-deploy", methods=["POST"])
def save_and_deploy():
    """Unified save: config.yaml + .env + GitHub Secrets.

    Request body:
        {
            "config": {...},        // optional: config.yaml updates
            "env": {...},           // optional: .env key-value pairs
            "push_secrets": true    // optional: push to GitHub
        }

    Returns:
        JSON with step-by-step results.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        steps = []

        # Step 1: Save config.yaml
        config_updates = data.get("config")
        if config_updates and isinstance(config_updates, dict):
            try:
                config_path = current_app.config["CONFIG_PATH"]
                config = read_config(config_path)
                for key, value in config_updates.items():
                    if key != "topics":
                        config[key] = value
                write_config(config_path, config)
                steps.append({"step": "config", "status": "ok"})
            except Exception as e:
                steps.append({"step": "config", "status": "error", "error": str(e)})
        else:
            steps.append({"step": "config", "status": "skipped"})

        # Step 2: Save .env
        env_data = data.get("env")
        if env_data and isinstance(env_data, dict):
            try:
                env_results = write_env(env_data)
                load_dotenv(override=True)
                steps.append({"step": "env", "status": "ok", "results": env_results})
            except Exception as e:
                steps.append({"step": "env", "status": "error", "error": str(e)})
        else:
            steps.append({"step": "env", "status": "skipped"})

        # Step 3: Push GitHub Secrets
        if data.get("push_secrets") and env_data:
            try:
                gh_result = push_secrets(env_data)
                gh_status = "ok" if gh_result["success"] else "partial"
                steps.append({
                    "step": "github_secrets",
                    "status": gh_status,
                    "method": gh_result.get("method"),
                    "results": gh_result.get("results"),
                    "manual_url": gh_result.get("manual_url"),
                })
            except Exception as e:
                steps.append({"step": "github_secrets", "status": "error", "error": str(e)})
        else:
            steps.append({"step": "github_secrets", "status": "skipped"})

        all_ok = all(s["status"] in ("ok", "skipped") for s in steps)
        return jsonify({
            "success": all_ok,
            "steps": steps,
        }), 200

    except Exception as e:
        logger.error(f"Error in save-and-deploy: {e}")
        return jsonify({"error": str(e)}), 500
