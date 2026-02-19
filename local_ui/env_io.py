""".env file I/O operations for Setup Wizard.

Provides functions to read and write .env files with masking support.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, set_key

# Keys managed by the setup wizard
ENV_KEYS = [
    "OPENROUTER_API_KEY",
    "DISCORD_WEBHOOK_URL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "GITHUB_TOKEN",
    "SUPABASE_DB_URL",
]

# Validation rules per key
_VALIDATORS = {
    "OPENROUTER_API_KEY": lambda v: v.startswith("sk-or-"),
    "DISCORD_WEBHOOK_URL": lambda v: v.startswith("https://discord.com/api/webhooks/"),
    "TELEGRAM_BOT_TOKEN": lambda v: ":" in v,
    "TELEGRAM_CHAT_ID": lambda v: bool(v.strip()),
    "GITHUB_TOKEN": lambda v: v.startswith("ghp_") or v.startswith("github_pat_"),
    "SUPABASE_DB_URL": lambda v: v.startswith("postgresql://"),
}

# Help URLs for each key (external documentation links)
HELP_URLS = {
    "OPENROUTER_API_KEY": "https://openrouter.ai/keys",
    "DISCORD_WEBHOOK_URL": "https://support.discord.com/hc/en-us/articles/228383668",
    "TELEGRAM_BOT_TOKEN": "https://core.telegram.org/bots#botfather",
    "TELEGRAM_CHAT_ID": "https://t.me/userinfobot",
    "GITHUB_TOKEN": "https://github.com/settings/tokens/new",
    "SUPABASE_DB_URL": "https://supabase.com/dashboard/project/_/settings/database",
}

ENV_PATH = Path(".env")


def _mask(value: str) -> str:
    """Mask a secret value, showing only the first 4 characters."""
    if len(value) > 4:
        return f"{value[:4]}{'*' * min(len(value) - 4, 20)}"
    return "****"


def read_env() -> dict[str, dict]:
    """Read .env file and return status for each managed key.

    Returns:
        Dict mapping key name to status dict with exists, masked_value, valid fields.
    """
    file_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

    result = {}
    for key in ENV_KEYS:
        value = file_values.get(key) or os.getenv(key)
        if value:
            validator = _VALIDATORS.get(key, lambda _: True)
            result[key] = {
                "exists": True,
                "masked_value": _mask(value),
                "valid": validator(value),
            }
        else:
            result[key] = {
                "exists": False,
                "masked_value": None,
                "valid": False,
            }
    return result


def write_env(updates: dict[str, str]) -> dict[str, str]:
    """Write key-value pairs to .env file.

    Args:
        updates: Dict of key-value pairs to set. Empty string values are skipped.

    Returns:
        Dict mapping key to "ok" or error message.
    """
    # Create .env if it doesn't exist
    if not ENV_PATH.exists():
        ENV_PATH.touch()

    results = {}
    for key, value in updates.items():
        if key not in ENV_KEYS:
            results[key] = f"Unknown key: {key}"
            continue
        if not value or not value.strip():
            results[key] = "skipped"
            continue
        try:
            success, _, _ = set_key(str(ENV_PATH), key, value.strip())
            results[key] = "ok" if success else "write failed"
        except Exception as e:
            results[key] = str(e)

    return results
