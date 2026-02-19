"""Database manager factory for Paper Scout.

Selects SQLite or Supabase (PostgreSQL) backend based on config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def create_db_manager(config: dict[str, Any]):
    """Create a database manager based on config provider setting.

    Args:
        config: Database config dict (the ``database`` section of config.yaml).
            Expected keys:
            - provider: "sqlite" (default) or "supabase"
            - path: SQLite file path (used when provider is "sqlite")
            - supabase.connection_string_env: env var name for PostgreSQL URL

    Returns:
        DBManager (SQLite) or SupabaseDBManager (PostgreSQL) instance.

    Raises:
        ValueError: If provider is "supabase" but connection string is not set.
    """
    provider = config.get("provider", "sqlite")

    if provider == "supabase":
        supabase_cfg = config.get("supabase", {})
        env_var = supabase_cfg.get("connection_string_env", "SUPABASE_DB_URL")
        connection_string = os.environ.get(env_var)
        if not connection_string:
            raise ValueError(
                f"Database provider is 'supabase' but environment variable "
                f"'{env_var}' is not set."
            )
        from core.storage.supabase_db_manager import SupabaseDBManager

        logger.info("Using Supabase PostgreSQL database backend")
        return SupabaseDBManager(connection_string)

    # Default: SQLite
    from core.storage.db_manager import DBManager

    db_path = config.get("path", "data/paper_scout.db")
    logger.info("Using SQLite database backend: %s", db_path)
    return DBManager(db_path)
