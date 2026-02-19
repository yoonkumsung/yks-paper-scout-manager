"""Database connection helper for provider-agnostic queries.

Provides a context manager that returns a connection and placeholder style
based on the configured database provider (sqlite or supabase/postgresql).
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator
from urllib.parse import quote


def _sanitize_dsn(dsn: str) -> str:
    """URL-encode special characters in the password portion of a DSN.

    Supabase can generate passwords containing @, %, ? which break
    standard URL parsing. This extracts the password using the known
    structure and re-encodes it.
    """
    # Match: scheme://user:password@host...
    # Use greedy match for password to capture embedded @ characters
    m = re.match(
        r"(postgresql(?:\+\w+)?://[^:]+:)"  # scheme + user + colon
        r"(.+)"                               # password (greedy)
        r"(@[^@]+\.supabase\.co(?:m)?[:/].+)$",  # @host (anchor on supabase.co or .com)
        dsn,
    )
    if not m:
        return dsn
    raw_password = m.group(2)
    encoded_password = quote(raw_password, safe="")
    return m.group(1) + encoded_password + m.group(3)


@contextmanager
def get_connection(
    db_path: str = "data/paper_scout.db",
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> Generator[tuple[Any, str], None, None]:
    """Context manager that yields (connection, placeholder_char).

    Args:
        db_path: Path to SQLite DB (used when provider is "sqlite").
        provider: "sqlite" or "supabase".
        connection_string: PostgreSQL connection string (used when provider is "supabase").

    Yields:
        (connection, placeholder): connection object and placeholder string
            ("?" for SQLite, "%s" for PostgreSQL).
            Rows are accessible by column name in both cases.

    Raises:
        ValueError: If provider is "supabase" but no connection string is available.
    """
    conn = None
    try:
        if provider == "supabase":
            import psycopg2
            import psycopg2.extras

            if not connection_string:
                raise ValueError(
                    "Database provider is 'supabase' but no connection string provided."
                )
            sanitized = _sanitize_dsn(connection_string)
            conn = psycopg2.connect(
                sanitized,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
            yield conn, "%s"
        else:
            if not os.path.exists(db_path):
                yield None, "?"
                return
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            yield conn, "?"
    finally:
        if conn is not None:
            conn.close()
