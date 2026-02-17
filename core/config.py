"""Configuration loading and validation for Paper Scout.

Loads config.yaml, validates against the schema defined in devspec
Section 20-A, and returns a structured AppConfig object.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import yaml

from core.models import NotifyConfig, TopicSpec


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Configuration validation error."""

    pass


# ---------------------------------------------------------------------------
# AppConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """Complete application configuration."""

    app: dict
    llm: dict
    agents: dict
    sources: dict
    filter: dict
    embedding: dict
    scoring: dict
    remind: dict
    clustering: dict
    topics: list[TopicSpec]
    output: dict
    notifications: dict
    database: dict
    weekly: dict
    local_ui: dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "app",
    "llm",
    "agents",
    "sources",
    "filter",
    "embedding",
    "scoring",
    "remind",
    "clustering",
    "topics",
    "output",
    "notifications",
    "database",
    "weekly",
    "local_ui",
]

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_VALID_NOTIFY_PROVIDERS = {"discord", "telegram"}


# ---------------------------------------------------------------------------
# Topic validation
# ---------------------------------------------------------------------------


def validate_topic(topic_data: dict, index: int) -> TopicSpec:
    """Validate a single topic configuration.

    Args:
        topic_data: Raw dict from YAML.
        index: Topic index for error messages.

    Returns:
        Validated TopicSpec object.

    Raises:
        ConfigError: On validation failure.
    """
    if not isinstance(topic_data, dict):
        raise ConfigError(f"topics[{index}]: must be a mapping, got {type(topic_data).__name__}")

    # --- Required fields ---
    _require_field(topic_data, "slug", index)
    _require_field(topic_data, "name", index)
    _require_field(topic_data, "description", index)
    _require_field(topic_data, "arxiv_categories", index)

    # --- slug format ---
    slug = topic_data["slug"]
    if not isinstance(slug, str) or not _SLUG_PATTERN.match(slug):
        raise ConfigError(
            f"topics[{index}].slug: must be lowercase ASCII and hyphens only "
            f"(pattern: {_SLUG_PATTERN.pattern}), got '{slug}'"
        )

    # --- description length warning (non-fatal) ---
    description = topic_data["description"]
    if not isinstance(description, str):
        raise ConfigError(f"topics[{index}].description: must be a string")
    desc_len = len(description)
    if desc_len < 100 or desc_len > 300:
        warnings.warn(
            f"topics[{index}].description: recommended length is 100-300 chars, "
            f"got {desc_len}",
            UserWarning,
            stacklevel=2,
        )

    # --- arxiv_categories ---
    cats = topic_data["arxiv_categories"]
    if not isinstance(cats, list) or len(cats) == 0:
        raise ConfigError(
            f"topics[{index}].arxiv_categories: must be a non-empty list of strings"
        )
    for i, cat in enumerate(cats):
        if not isinstance(cat, str):
            raise ConfigError(
                f"topics[{index}].arxiv_categories[{i}]: must be a string, "
                f"got {type(cat).__name__}"
            )

    # --- notify (optional) ---
    notify: NotifyConfig | None = None
    notify_data = topic_data.get("notify")
    if notify_data is not None and isinstance(notify_data, dict) and notify_data:
        provider = notify_data.get("provider", "")
        if provider and provider in _VALID_NOTIFY_PROVIDERS:
            _require_notify_field(notify_data, "channel_id", index)
            _require_notify_field(notify_data, "secret_key", index)

            channel_id = notify_data["channel_id"]
            if not isinstance(channel_id, str):
                raise ConfigError(
                    f"topics[{index}].notify.channel_id: must be a string, "
                    f"got {type(channel_id).__name__}"
                )

            secret_key = notify_data["secret_key"]
            if not isinstance(secret_key, str):
                raise ConfigError(
                    f"topics[{index}].notify.secret_key: must be a string, "
                    f"got {type(secret_key).__name__}"
                )

            notify = NotifyConfig(
                provider=provider,
                channel_id=str(channel_id),
                secret_key=str(secret_key),
            )
        elif provider and provider not in _VALID_NOTIFY_PROVIDERS:
            raise ConfigError(
                f"topics[{index}].notify.provider: must be one of "
                f"{sorted(_VALID_NOTIFY_PROVIDERS)}, got '{provider}'"
            )

    # --- Optional fields type checking ---
    must_concepts_en = _validate_optional_str_list(
        topic_data, "must_concepts_en", index
    )
    should_concepts_en = _validate_optional_str_list(
        topic_data, "should_concepts_en", index
    )
    must_not_en = _validate_optional_str_list(topic_data, "must_not_en", index)

    return TopicSpec(
        slug=slug,
        name=topic_data["name"],
        description=description,
        arxiv_categories=cats,
        notify=notify,
        must_concepts_en=must_concepts_en,
        should_concepts_en=should_concepts_en,
        must_not_en=must_not_en,
    )


# ---------------------------------------------------------------------------
# Section defaults
# ---------------------------------------------------------------------------

_SECTION_DEFAULTS: dict[str, dict] = {
    "app": {
        "display_timezone": "UTC",
        "report_retention_days": 90,
    },
    "agents": {
        "common": {"ignore_reasoning": True},
        "keyword_expander": {},
        "scorer": {},
        "summarizer": {},
    },
    "sources": {
        "primary": "arxiv",
        "arxiv": {"max_results_per_query": 100},
        "seen_items_path": "data/seen_items.jsonl",
    },
    "filter": {
        "enable": True,
        "strategy": "quality_threshold",
    },
    "embedding": {
        "provider": "",
        "model": "",
        "cache_dir": "data",
    },
    "remind": {
        "enabled": False,
        "min_score": 80.0,
        "max_recommend_count": 2,
    },
    "clustering": {
        "enabled": True,
        "threshold": 0.85,
    },
    "output": {
        "report_dir": "tmp/reports",
        "report_format": "html",
        "template_dir": "templates",
    },
    "notifications": {
        "on_zero_results": True,
        "on_error": True,
    },
    "weekly": {
        "enabled": False,
    },
    "local_ui": {
        "host": "127.0.0.1",
        "port": 8585,
    },
}


def _apply_defaults(section_name: str, user_config: dict) -> dict:
    """Merge user config over section defaults."""
    defaults = _SECTION_DEFAULTS.get(section_name, {})
    return {**defaults, **user_config}


# ---------------------------------------------------------------------------
# Full config validation
# ---------------------------------------------------------------------------


def validate_config(raw: dict) -> AppConfig:
    """Validate the complete config structure.

    Args:
        raw: Raw dict parsed from YAML.

    Returns:
        Validated AppConfig object.

    Raises:
        ConfigError: On validation failure.
    """
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a YAML mapping")

    # --- Required top-level sections ---
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ConfigError(f"Missing required config section: '{section}'")

    # --- llm.model ---
    llm = raw["llm"]
    if not isinstance(llm, dict):
        raise ConfigError("llm: must be a mapping")
    model = llm.get("model")
    if not model or not isinstance(model, str) or not model.strip():
        raise ConfigError("llm.model: must be a non-empty string")

    # --- scoring ---
    scoring = raw["scoring"]
    if not isinstance(scoring, dict):
        raise ConfigError("scoring: must be a mapping")

    weights = scoring.get("weights")
    if not isinstance(weights, dict):
        raise ConfigError("scoring.weights: must be a mapping")

    _validate_weight_keys(
        weights,
        "embedding_on",
        {"llm", "embed", "recency"},
    )
    _validate_weight_keys(
        weights,
        "embedding_off",
        {"llm", "recency"},
    )

    discard_cutoff = scoring.get("discard_cutoff")
    if discard_cutoff is None or not isinstance(discard_cutoff, (int, float)):
        raise ConfigError("scoring.discard_cutoff: must be a number")
    if not (0 <= discard_cutoff <= 100):
        raise ConfigError(
            f"scoring.discard_cutoff: must be 0-100, got {discard_cutoff}"
        )

    max_output = scoring.get("max_output")
    if max_output is None or not isinstance(max_output, int) or max_output < 1:
        raise ConfigError(
            "scoring.max_output: must be a positive integer"
        )

    # --- database.path ---
    db = raw["database"]
    if not isinstance(db, dict):
        raise ConfigError("database: must be a mapping")
    db_path = db.get("path")
    if not isinstance(db_path, str):
        raise ConfigError("database.path: must be a string")

    # --- topics ---
    topics_raw = raw["topics"]
    if not isinstance(topics_raw, list):
        raise ConfigError("topics: must be a list")
    if len(topics_raw) == 0:
        raise ConfigError("topics: must be a non-empty list")

    seen_slugs: set[str] = set()
    topics: list[TopicSpec] = []
    for idx, topic_data in enumerate(topics_raw):
        topic = validate_topic(topic_data, idx)

        if topic.slug in seen_slugs:
            raise ConfigError(
                f"topics[{idx}].slug: duplicate slug '{topic.slug}'"
            )
        seen_slugs.add(topic.slug)
        topics.append(topic)

    return AppConfig(
        app=_apply_defaults("app", raw.get("app", {})),
        llm=llm,
        agents=_apply_defaults("agents", raw.get("agents", {})),
        sources=_apply_defaults("sources", raw.get("sources", {})),
        filter=_apply_defaults("filter", raw.get("filter", {})),
        embedding=_apply_defaults("embedding", raw.get("embedding", {})),
        scoring=scoring,
        remind=_apply_defaults("remind", raw.get("remind", {})),
        clustering=_apply_defaults("clustering", raw.get("clustering", {})),
        topics=topics,
        output=_apply_defaults("output", raw.get("output", {})),
        notifications=_apply_defaults("notifications", raw.get("notifications", {})),
        database=db,
        weekly=_apply_defaults("weekly", raw.get("weekly", {})),
        local_ui=_apply_defaults("local_ui", raw.get("local_ui", {})),
    )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | None = None) -> AppConfig:
    """Load and validate config.yaml.

    Args:
        config_path: Override path. If None, uses PAPER_SCOUT_CONFIG env var
                     or defaults to 'config.yaml'.

    Returns:
        Validated AppConfig object.

    Raises:
        ConfigError: On validation failure with descriptive message.
        FileNotFoundError: If config file doesn't exist.
    """
    if config_path is None:
        config_path = os.environ.get("PAPER_SCOUT_CONFIG", "config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ConfigError("Config file is empty or contains only comments")

    return validate_config(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_field(topic_data: dict, key: str, index: int) -> None:
    """Raise ConfigError if a required topic field is missing."""
    if key not in topic_data:
        raise ConfigError(f"topics[{index}]: missing required field '{key}'")


def _require_notify_field(notify_data: dict, key: str, index: int) -> None:
    """Raise ConfigError if a required notify field is missing."""
    if key not in notify_data:
        raise ConfigError(
            f"topics[{index}].notify: missing required field '{key}'"
        )


def _validate_optional_str_list(
    topic_data: dict, key: str, index: int
) -> list[str] | None:
    """Validate an optional list[str] field, returning None if absent."""
    value = topic_data.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigError(
            f"topics[{index}].{key}: must be a list, got {type(value).__name__}"
        )
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(
                f"topics[{index}].{key}[{i}]: must be a string, "
                f"got {type(item).__name__}"
            )
    return value


def _validate_weight_keys(
    weights: dict, group_name: str, required_keys: set[str]
) -> None:
    """Validate that a weight group has the required keys."""
    group = weights.get(group_name)
    if not isinstance(group, dict):
        raise ConfigError(f"scoring.weights.{group_name}: must be a mapping")
    missing = required_keys - set(group.keys())
    if missing:
        raise ConfigError(
            f"scoring.weights.{group_name}: missing keys {sorted(missing)}"
        )
