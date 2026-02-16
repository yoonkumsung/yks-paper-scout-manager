"""Configuration file I/O operations for local UI.

Provides functions to read and write config.yaml with topic management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_config(config_path: str) -> dict:
    """Read full config.yaml.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary containing full configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_config(config_path: str, data: dict) -> None:
    """Write full config.yaml (preserving comments where possible).

    Args:
        config_path: Path to config.yaml file
        data: Dictionary containing full configuration

    Note:
        YAML comments are not preserved due to safe_load/safe_dump limitations.
        For production use, consider ruamel.yaml for comment preservation.
    """
    path = Path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_topics(config_path: str) -> list[dict]:
    """Get topics list from config.

    Args:
        config_path: Path to config.yaml file

    Returns:
        List of topic dictionaries
    """
    config = read_config(config_path)
    return config.get("topics", [])


def add_topic(config_path: str, topic: dict) -> None:
    """Add a topic to config.

    Args:
        config_path: Path to config.yaml file
        topic: Topic dictionary with required fields (slug, name, description, etc.)

    Raises:
        ValueError: If topic with same slug already exists
    """
    config = read_config(config_path)
    topics = config.get("topics", [])

    # Check for duplicate slug
    if any(t.get("slug") == topic.get("slug") for t in topics):
        raise ValueError(f"Topic with slug '{topic.get('slug')}' already exists")

    topics.append(topic)
    config["topics"] = topics
    write_config(config_path, config)


def update_topic(config_path: str, slug: str, updates: dict) -> None:
    """Update a topic in config by slug.

    Args:
        config_path: Path to config.yaml file
        slug: Topic slug to update
        updates: Dictionary of fields to update

    Raises:
        ValueError: If topic not found
    """
    config = read_config(config_path)
    topics = config.get("topics", [])

    for topic in topics:
        if topic.get("slug") == slug:
            topic.update(updates)
            config["topics"] = topics
            write_config(config_path, config)
            return

    raise ValueError(f"Topic with slug '{slug}' not found")


def remove_topic(config_path: str, slug: str) -> bool:
    """Remove a topic from config.

    Args:
        config_path: Path to config.yaml file
        slug: Topic slug to remove

    Returns:
        True if topic was found and removed, False otherwise
    """
    config = read_config(config_path)
    topics = config.get("topics", [])

    original_count = len(topics)
    topics = [t for t in topics if t.get("slug") != slug]

    if len(topics) < original_count:
        config["topics"] = topics
        write_config(config_path, config)
        return True

    return False
