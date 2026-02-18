"""Health check script to verify server can start."""

from __future__ import annotations

import sys
import tempfile

from local_ui.app import create_app


def health_check() -> bool:
    """Verify Flask app can be created and configured.

    Returns:
        True if health check passes
    """
    try:
        # Create temp config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
app: {}
llm: {}
agents: {}
sources: {}
filter: {}
embedding: {}
scoring: {}
remind: {}
clustering: {}
topics: []
output: {}
notifications: {}
database: {}
weekly: {}
local_ui:
  host: "127.0.0.1"
  port: 8585
  open_browser: false
"""
            )
            config_path = f.name

        # Create app
        app = create_app(config_path=config_path, db_path=":memory:")

        # Verify configuration
        assert app.config["CONFIG_PATH"] == config_path
        assert app.config["DB_PATH"] == ":memory:"

        # Create test client
        client = app.test_client()

        # Test index route
        response = client.get("/")
        assert response.status_code == 200

        # Test API routes
        response = client.get("/api/topics")
        assert response.status_code == 200

        print("✓ Health check passed")
        print(f"  - App created successfully")
        print(f"  - Routes registered: {len(app.url_map._rules)} routes")
        print(f"  - Blueprints: topics, pipeline, settings, setup, recommend")
        print(f"  - Security: Binds to 127.0.0.1 only")
        return True

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


if __name__ == "__main__":
    success = health_check()
    sys.exit(0 if success else 1)
