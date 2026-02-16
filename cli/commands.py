"""Main CLI module with subcommand routing for Paper Scout."""

from __future__ import annotations

import argparse
import sys


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the full CLI argument parser with subcommands.

    Returns:
        Configured ArgumentParser with subparsers
    """
    parser = argparse.ArgumentParser(
        prog="paper-scout",
        description="Automated arXiv paper collection and analysis system",
    )

    # Global options
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (default: INFO)",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== topic subcommand ==========
    topic_parser = subparsers.add_parser("topic", help="Manage topics")
    topic_subparsers = topic_parser.add_subparsers(dest="topic_command", help="Topic operations")

    # topic list
    topic_subparsers.add_parser("list", help="List registered topics")

    # topic add
    topic_subparsers.add_parser("add", help="Add a new topic interactively")

    # topic edit
    edit_parser = topic_subparsers.add_parser("edit", help="Edit a topic in $EDITOR")
    edit_parser.add_argument("slug", type=str, help="Topic slug to edit")

    # topic remove
    remove_parser = topic_subparsers.add_parser("remove", help="Remove a topic")
    remove_parser.add_argument("slug", type=str, help="Topic slug to remove")

    # ========== dry-run subcommand ==========
    dry_run_parser = subparsers.add_parser("dry-run", help="Run in dry-run mode")
    dry_run_parser.add_argument(
        "--topic", type=str, default=None,
        help="Run only the specified topic slug",
    )
    dry_run_parser.add_argument(
        "--date-from", type=str, default=None,
        help="Start date (YYYY-MM-DD format, UTC)",
    )
    dry_run_parser.add_argument(
        "--date-to", type=str, default=None,
        help="End date (YYYY-MM-DD format, UTC)",
    )
    dry_run_parser.add_argument(
        "--dedup", choices=["skip_recent", "none"], default="skip_recent",
        help="Dedup mode (default: skip_recent)",
    )

    # ========== run subcommand ==========
    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--date-from", type=str, default=None,
        help="Start date (YYYY-MM-DD format, UTC)",
    )
    run_parser.add_argument(
        "--date-to", type=str, default=None,
        help="End date (YYYY-MM-DD format, UTC)",
    )
    run_parser.add_argument(
        "--dedup", choices=["skip_recent", "none"], default="skip_recent",
        help="Dedup mode (default: skip_recent)",
    )
    run_parser.add_argument(
        "--topic", type=str, default=None,
        help="Run only the specified topic slug",
    )
    run_parser.add_argument(
        "--mode", choices=["full", "dry-run"], default="full",
        help="Execution mode (default: full)",
    )

    # ========== ui subcommand ==========
    ui_parser = subparsers.add_parser("ui", help="Launch the local web UI")
    ui_parser.add_argument(
        "--port", type=int, default=8585,
        help="Port to listen on (default: 8585)",
    )
    ui_parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open browser automatically",
    )

    return parser


def handle_topic_command(args: argparse.Namespace) -> int:
    """Handle topic subcommand routing.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code (0 = success)
    """
    from cli.topic_commands import topic_add, topic_edit, topic_list, topic_remove

    if args.topic_command == "list":
        topic_list()
    elif args.topic_command == "add":
        topic_add()
    elif args.topic_command == "edit":
        topic_edit(slug=args.slug)
    elif args.topic_command == "remove":
        topic_remove(slug=args.slug)
    else:
        print("Error: No topic command specified. Use 'paper-scout topic --help'")
        return 1

    return 0


def handle_dry_run_command(args: argparse.Namespace) -> int:
    """Handle dry-run subcommand.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code from run_pipeline
    """
    # Import main to avoid circular dependency
    from main import run_pipeline

    # Construct args namespace compatible with run_pipeline
    pipeline_args = argparse.Namespace(
        mode="dry-run",
        date_from=args.date_from,
        date_to=args.date_to,
        dedup=args.dedup,
        topic=args.topic,
        log_level=args.log_level,
    )

    return run_pipeline(pipeline_args)


def handle_run_command(args: argparse.Namespace) -> int:
    """Handle run subcommand.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code from run_pipeline
    """
    from main import run_pipeline

    # Construct args namespace compatible with run_pipeline
    pipeline_args = argparse.Namespace(
        mode=args.mode,
        date_from=args.date_from,
        date_to=args.date_to,
        dedup=args.dedup,
        topic=args.topic,
        log_level=args.log_level,
    )

    return run_pipeline(pipeline_args)


def handle_ui_command(args: argparse.Namespace) -> int:
    """Handle ui subcommand (placeholder for TASK-050).

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code (0 = success)
    """
    print(f"Web UI starting on port {args.port}...")
    if args.no_browser:
        print("Browser auto-open disabled.")
    else:
        print("Browser will open automatically.")

    # Placeholder for Flask server (TASK-050)
    print("(Flask server implementation pending in TASK-050)")

    return 0


def main() -> int:
    """Main CLI entry point with subcommand routing.

    Returns:
        Exit code
    """
    parser = create_cli_parser()
    args = parser.parse_args()

    # If no subcommand, show help
    if args.command is None:
        parser.print_help()
        return 0

    # Route to appropriate handler
    if args.command == "topic":
        return handle_topic_command(args)
    elif args.command == "dry-run":
        return handle_dry_run_command(args)
    elif args.command == "run":
        return handle_run_command(args)
    elif args.command == "ui":
        return handle_ui_command(args)
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
