"""Flask application for Paper Scout local UI.

Provides web interface for managing topics, running pipeline, and viewing settings.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)


def create_app(
    config_path: str = "config.yaml",
    db_path: str = "data/paper_scout.db",
    data_path: str = "data",
) -> Flask:
    """Create and configure the Flask app.

    Args:
        config_path: Path to config.yaml
        db_path: Path to SQLite database
        data_path: Path to data directory (for cache status checks)

    Returns:
        Configured Flask application instance
    """
    from dotenv import load_dotenv
    load_dotenv()

    app = Flask(__name__)

    # Store paths in app config
    app.config["CONFIG_PATH"] = config_path
    app.config["DB_PATH"] = db_path
    app.config["DATA_PATH"] = data_path

    # Load database provider config for topic stats
    import os
    import yaml

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        db_cfg = raw_config.get("database", {})
        app.config["DB_PROVIDER"] = db_cfg.get("provider", "sqlite")
        if app.config["DB_PROVIDER"] == "supabase":
            supabase_cfg = db_cfg.get("supabase", {})
            env_var = supabase_cfg.get("connection_string_env", "SUPABASE_DB_URL")
            app.config["SUPABASE_DB_URL"] = os.environ.get(env_var, "")
    except Exception:
        app.config["DB_PROVIDER"] = "sqlite"

    # Register blueprints
    from local_ui.api.topics import topics_bp
    from local_ui.api.pipeline import pipeline_bp
    from local_ui.api.settings import settings_bp
    from local_ui.api.setup import setup_bp
    from local_ui.api.recommend import recommend_bp
    from local_ui.api.weekly_intel import weekly_intel_bp

    app.register_blueprint(topics_bp, url_prefix="/api/topics")
    app.register_blueprint(pipeline_bp, url_prefix="/api/pipeline")
    app.register_blueprint(settings_bp, url_prefix="/api/settings")
    app.register_blueprint(setup_bp, url_prefix="/api/setup")
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(weekly_intel_bp, url_prefix="/api/weekly-paper-report")

    @app.route("/")
    def index() -> str:
        """Render main UI page."""
        return render_template("index.html")

    @app.route("/reports")
    @app.route("/reports/<path:subpath>")
    def reports(subpath: str = "") -> Any:
        """Serve report files from tmp/reports directory."""
        from flask import send_from_directory, abort
        reports_dir = Path("tmp/reports").resolve()
        if not reports_dir.exists():
            abort(404)
        filename = subpath or "index.html"
        target = reports_dir / filename
        if target.exists() and target.is_file():
            return send_from_directory(str(reports_dir), filename)
        # If no specific file requested, list available reports
        if not subpath:
            html_files = sorted(reports_dir.glob("**/*.html"), reverse=True)
            if html_files:
                # Serve the most recent report
                rel = html_files[0].relative_to(reports_dir)
                return send_from_directory(str(reports_dir), str(rel))
        abort(404)

    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[dict, int]:
        """Handle 404 errors."""
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> tuple[dict, int]:
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    return app


def start_server(
    host: str = "127.0.0.1",
    port: int = 8585,
    open_browser: bool = True,
    config_path: str = "config.yaml",
    db_path: str = "data/paper_scout.db",
    data_path: str = "data",
) -> None:
    """Start the Flask development server.

    Args:
        host: Host to bind (default: 127.0.0.1 for security)
        port: Port to listen on (default: 8585)
        open_browser: Auto-open browser after 1 second
        config_path: Path to config.yaml
        db_path: Path to SQLite database
        data_path: Path to data directory (for cache status checks)
    """
    app = create_app(config_path=config_path, db_path=db_path, data_path=data_path)

    url = f"http://{host}:{port}"
    logger.info(f"Starting Paper Scout local UI at {url}")
    print(f"\n{'=' * 60}")
    print(f"Paper Scout Local UI")
    print(f"{'=' * 60}")
    print(f"Server URL: {url}")
    print(f"Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    if open_browser:
        # Open browser after 1 second delay
        def open_browser_delayed() -> None:
            import time

            time.sleep(1)
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    app.run(host=host, port=port, debug=False)
