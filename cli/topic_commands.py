"""Topic CRUD commands for Paper Scout CLI."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


def topic_list(config_path: str = "config.yaml") -> None:
    """List all registered topics in table format.

    Args:
        config_path: Path to config.yaml file
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file: {e}")
        return

    topics = config.get("topics", [])

    if not topics:
        print("No topics registered.")
        return

    # Table header
    print(f"{'Slug':<25} {'Name':<30} {'Categories'}")
    print("-" * 80)

    # Table rows
    for topic in topics:
        slug = topic.get("slug", "")
        name = topic.get("name", "")
        categories = ", ".join(topic.get("arxiv_categories", []))
        print(f"{slug:<25} {name:<30} {categories}")

    print(f"\nTotal: {len(topics)} topic(s)")


def topic_add(config_path: str = "config.yaml") -> None:
    """Add a new topic interactively.

    Args:
        config_path: Path to config.yaml file
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file: {e}")
        return

    topics = config.get("topics", [])

    # Interactive prompts
    print("=== Add New Topic ===")
    slug = input("Topic slug (lowercase-with-dashes): ").strip()
    if not slug:
        print("Error: Slug cannot be empty.")
        return

    # Check for duplicate slug
    if any(t.get("slug") == slug for t in topics):
        print(f"Error: Topic with slug '{slug}' already exists.")
        return

    name = input("Topic name: ").strip()
    if not name:
        print("Error: Name cannot be empty.")
        return

    description = input("Description: ").strip()

    categories_input = input("arXiv categories (comma-separated, e.g., cs.AI,cs.CV): ").strip()
    categories = [c.strip() for c in categories_input.split(",") if c.strip()]

    # Notification settings
    print("\n--- Notification Settings (optional) ---")
    provider = input("Notify provider (discord/slack/none): ").strip() or None
    channel_id = None
    secret_key = None

    if provider and provider != "none":
        channel_id = input("Channel ID: ").strip() or None
        secret_key = input("Secret key env var (e.g., DISCORD_WEBHOOK_URL): ").strip() or None

    # Build topic dict
    new_topic: dict[str, Any] = {
        "slug": slug,
        "name": name,
        "description": description,
        "arxiv_categories": categories,
    }

    if provider and provider != "none":
        new_topic["notify"] = {
            "provider": provider,
            "channel_id": channel_id,
            "secret_key": secret_key,
        }

    # Add to topics
    topics.append(new_topic)
    config["topics"] = topics

    # Write back
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"\n✓ Topic '{slug}' added successfully.")
    except Exception as e:
        print(f"Error: Failed to write configuration file: {e}")


def topic_edit(config_path: str = "config.yaml", slug: str = "") -> None:
    """Edit a topic's YAML section in $EDITOR.

    Args:
        config_path: Path to config.yaml file
        slug: Topic slug to edit
    """
    if not slug:
        print("Error: No topic slug provided.")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file: {e}")
        return

    topics = config.get("topics", [])

    # Find topic
    topic_idx = None
    for i, topic in enumerate(topics):
        if topic.get("slug") == slug:
            topic_idx = i
            break

    if topic_idx is None:
        print(f"Error: Topic '{slug}' not found.")
        return

    # Create temp file with topic YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
        yaml.dump(topics[topic_idx], tmp, default_flow_style=False, allow_unicode=True)

    # Open in $EDITOR (fallback to vim)
    editor = os.environ.get("EDITOR", "vim")
    try:
        subprocess.run([editor, tmp_path], check=True)
    except subprocess.CalledProcessError:
        print(f"Error: Failed to open editor '{editor}'.")
        Path(tmp_path).unlink(missing_ok=True)
        return
    except FileNotFoundError:
        print(f"Error: Editor '{editor}' not found.")
        Path(tmp_path).unlink(missing_ok=True)
        return

    # Read back edited content
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            edited_topic = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse edited YAML: {e}")
        Path(tmp_path).unlink(missing_ok=True)
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Update config
    topics[topic_idx] = edited_topic
    config["topics"] = topics

    # Write back
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ Topic '{slug}' updated successfully.")
    except Exception as e:
        print(f"Error: Failed to write configuration file: {e}")


def topic_remove(config_path: str = "config.yaml", slug: str = "") -> None:
    """Remove a topic from config.

    Args:
        config_path: Path to config.yaml file
        slug: Topic slug to remove
    """
    if not slug:
        print("Error: No topic slug provided.")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse configuration file: {e}")
        return

    topics = config.get("topics", [])

    # Find and remove topic
    original_count = len(topics)
    topics = [t for t in topics if t.get("slug") != slug]

    if len(topics) == original_count:
        print(f"Error: Topic '{slug}' not found.")
        return

    # Confirm removal
    confirm = input(f"Remove topic '{slug}'? (y/N): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Update config
    config["topics"] = topics

    # Write back
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✓ Topic '{slug}' removed successfully.")
    except Exception as e:
        print(f"Error: Failed to write configuration file: {e}")
