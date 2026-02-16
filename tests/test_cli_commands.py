"""Tests for CLI commands."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml


# ========== Topic List Tests ==========


def test_topic_list_shows_topics():
    """Test that topic list displays topics correctly."""
    from cli.topic_commands import topic_list

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {
            "topics": [
                {
                    "slug": "ai-sports",
                    "name": "AI + Sports",
                    "arxiv_categories": ["cs.AI", "cs.CV"],
                },
                {
                    "slug": "ml-health",
                    "name": "ML in Healthcare",
                    "arxiv_categories": ["cs.LG", "stat.ML"],
                },
            ]
        }
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        with mock.patch("builtins.print") as mock_print:
            topic_list(config_path=tmp_path)

        # Verify header and data rows were printed
        calls = [str(call) for call in mock_print.call_args_list]
        output = "\n".join(calls)

        assert "Slug" in output
        assert "Name" in output
        assert "Categories" in output
        assert "ai-sports" in output
        assert "ml-health" in output
        assert "Total: 2 topic(s)" in output
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_topic_list_empty():
    """Test topic list with no topics registered."""
    from cli.topic_commands import topic_list

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {"topics": []}
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        with mock.patch("builtins.print") as mock_print:
            topic_list(config_path=tmp_path)

        mock_print.assert_called_once_with("No topics registered.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ========== Topic Add Tests ==========


def test_topic_add_interactive():
    """Test interactive topic addition."""
    from cli.topic_commands import topic_add

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {"topics": []}
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        # Mock user inputs
        inputs = [
            "test-topic",  # slug
            "Test Topic",  # name
            "A test topic",  # description
            "cs.AI,cs.CV",  # categories
            "none",  # provider
        ]
        with mock.patch("builtins.input", side_effect=inputs):
            with mock.patch("builtins.print"):
                topic_add(config_path=tmp_path)

        # Verify topic was added
        with open(tmp_path, "r", encoding="utf-8") as f:
            updated_config = yaml.safe_load(f)

        assert len(updated_config["topics"]) == 1
        assert updated_config["topics"][0]["slug"] == "test-topic"
        assert updated_config["topics"][0]["name"] == "Test Topic"
        assert updated_config["topics"][0]["arxiv_categories"] == ["cs.AI", "cs.CV"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ========== Topic Remove Tests ==========


def test_topic_remove_existing():
    """Test removing an existing topic."""
    from cli.topic_commands import topic_remove

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {
            "topics": [
                {"slug": "topic1", "name": "Topic 1"},
                {"slug": "topic2", "name": "Topic 2"},
            ]
        }
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        # Mock confirmation
        with mock.patch("builtins.input", return_value="y"):
            with mock.patch("builtins.print"):
                topic_remove(config_path=tmp_path, slug="topic1")

        # Verify topic was removed
        with open(tmp_path, "r", encoding="utf-8") as f:
            updated_config = yaml.safe_load(f)

        assert len(updated_config["topics"]) == 1
        assert updated_config["topics"][0]["slug"] == "topic2"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_topic_remove_nonexistent():
    """Test removing a nonexistent topic."""
    from cli.topic_commands import topic_remove

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {"topics": [{"slug": "topic1", "name": "Topic 1"}]}
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        with mock.patch("builtins.print") as mock_print:
            topic_remove(config_path=tmp_path, slug="nonexistent")

        # Verify error message
        calls = [str(call) for call in mock_print.call_args_list]
        output = "\n".join(calls)
        assert "not found" in output
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ========== Subcommand Parsing Tests ==========


def test_dryrun_subcommand_parsing():
    """Test dry-run subcommand argument parsing."""
    from cli.commands import create_cli_parser

    parser = create_cli_parser()
    args = parser.parse_args(["dry-run", "--topic", "test-topic", "--date-from", "2026-02-01"])

    assert args.command == "dry-run"
    assert args.topic == "test-topic"
    assert args.date_from == "2026-02-01"
    assert args.dedup == "skip_recent"


def test_run_subcommand_parsing():
    """Test run subcommand argument parsing."""
    from cli.commands import create_cli_parser

    parser = create_cli_parser()
    args = parser.parse_args([
        "run",
        "--date-from", "2026-02-01",
        "--date-to", "2026-02-07",
        "--dedup", "none",
    ])

    assert args.command == "run"
    assert args.date_from == "2026-02-01"
    assert args.date_to == "2026-02-07"
    assert args.dedup == "none"
    assert args.mode == "full"


def test_ui_subcommand_parsing():
    """Test ui subcommand argument parsing."""
    from cli.commands import create_cli_parser

    parser = create_cli_parser()
    args = parser.parse_args(["ui", "--port", "9090", "--no-browser"])

    assert args.command == "ui"
    assert args.port == 9090
    assert args.no_browser is True


def test_backward_compat_no_subcommand():
    """Test backward compatibility when no subcommand is provided."""
    from main import create_parser

    parser = create_parser()
    args = parser.parse_args(["--mode", "dry-run", "--topic", "test"])

    assert args.mode == "dry-run"
    assert args.topic == "test"


def test_topic_edit_uses_editor():
    """Test that topic edit invokes $EDITOR."""
    from cli.topic_commands import topic_edit

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        config = {
            "topics": [
                {"slug": "test-topic", "name": "Test Topic"},
            ]
        }
        yaml.dump(config, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        with mock.patch("subprocess.run") as mock_run:
            with mock.patch("builtins.print"):
                topic_edit(config_path=tmp_path, slug="test-topic")

        # Verify editor was invoked
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        # Check that the command includes an editor (default is vim)
        assert any("vim" in str(arg) or "EDITOR" in str(arg) for arg in call_args) or call_args[0] == "vim"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
