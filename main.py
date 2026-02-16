"""Paper Scout - Main entry point for the arXiv paper collection pipeline.

Orchestrates the full pipeline: Preflight -> Topic Loop -> Post-Loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="paper-scout",
        description="Automated arXiv paper collection and analysis system",
    )
    parser.add_argument(
        "--mode", choices=["full", "dry-run"], default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--date-from", type=str, default=None,
        help="Start date (YYYY-MM-DD format, UTC)",
    )
    parser.add_argument(
        "--date-to", type=str, default=None,
        help="End date (YYYY-MM-DD format, UTC)",
    )
    parser.add_argument(
        "--dedup", choices=["skip_recent", "none"], default="skip_recent",
        help="Dedup mode (default: skip_recent)",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Run only the specified topic slug",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def _parse_date(date_str: str | None) -> str | None:
    """Validate and return a date string in YYYY-MM-DD format."""
    if date_str is None:
        return None
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(
            f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD."
        )
    return date_str


def _setup_logging(level_name: str) -> None:
    """Configure the root logger."""
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _is_sunday_utc() -> bool:
    """Return True if the current UTC day is Sunday (weekday 6)."""
    return datetime.now(timezone.utc).weekday() == 6


def _to_utc_dt(date_str: str | None) -> datetime | None:
    """Convert a YYYY-MM-DD string to a UTC datetime, or None."""
    if date_str is None:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full Paper Scout pipeline.

    Returns:
        Exit code: 0 = success, 1 = partial failure, 2 = total failure.
    """
    # 1. Load config
    try:
        from core.config import load_config
        config = load_config()
    except Exception:
        logger.critical("Failed to load configuration", exc_info=True)
        return 2

    # 2. Preflight checks
    try:
        from core.pipeline.preflight import run_preflight
        date_from = _parse_date(args.date_from)
        date_to = _parse_date(args.date_to)
        preflight_result = run_preflight(
            date_from=_to_utc_dt(date_from),
            date_to=_to_utc_dt(date_to),
        )
        config = preflight_result.config
        for w in preflight_result.warnings:
            logger.warning("Preflight warning: %s", w)
    except SystemExit:
        raise
    except Exception:
        logger.critical("Preflight checks failed", exc_info=True)
        return 2

    # 3. Initialize shared resources
    try:
        from core.pipeline.search_window import SearchWindowComputer
        from core.storage.db_manager import DBManager
        from core.storage.usage_tracker import UsageTracker
        db_path = config.database.get("path", "data/paper_scout.db")
        db_manager = DBManager(db_path)
        rate_limiter = preflight_result.rate_limiter
        search_window = SearchWindowComputer(db_manager=db_manager)
        usage_tracker = UsageTracker()
    except Exception:
        logger.critical("Failed to initialize shared resources", exc_info=True)
        return 2

    # 4. Build run_options
    run_options: dict = {
        "mode": args.mode,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "dedup": args.dedup,
        "topic_slugs": [args.topic] if args.topic else None,
    }

    # 5. Topic loop
    exit_code = 0
    topic_results: dict = {}
    try:
        from core.pipeline.topic_loop import TopicLoopOrchestrator
        orchestrator = TopicLoopOrchestrator(
            config=config, db_manager=db_manager,
            rate_limiter=rate_limiter, search_window=search_window,
            usage_tracker=usage_tracker, run_options=run_options,
        )
        topic_results = orchestrator.run_all_topics()
        completed = topic_results.get("topics_completed", [])
        failed = topic_results.get("topics_failed", [])
        total = len(completed) + len(failed)
        if total == 0:
            logger.info("No topics were processed")
        elif len(failed) == 0:
            logger.info("All %d topics completed successfully", len(completed))
        elif len(completed) == 0:
            logger.error("All %d topics failed", len(failed))
            exit_code = 2
        else:
            logger.warning(
                "%d topics completed, %d topics failed",
                len(completed), len(failed),
            )
            exit_code = 1
    except Exception:
        logger.critical("Topic loop failed with exception", exc_info=True)
        return 2

    # 6. Post-loop (full mode only)
    if args.mode == "full" and topic_results:
        try:
            from core.pipeline.post_loop import PostLoopProcessor
            report_dir = config.output.get("report_dir", "tmp/reports")
            post_processor = PostLoopProcessor(
                config=config, db_manager=db_manager, report_dir=report_dir,
            )
            post_summary = post_processor.process(topic_results)
            logger.info("Post-loop summary: %s", post_summary)
        except Exception:
            logger.error("Post-loop processing failed (non-fatal)", exc_info=True)

    # 7. Weekly tasks placeholder
    if _is_sunday_utc():
        logger.info("Weekly tasks: skipped (not implemented yet)")

    # 8. Cleanup
    try:
        db_manager.close()
    except Exception:
        logger.warning("Failed to close database", exc_info=True)

    return exit_code


def main() -> int:
    """Main entry point with backward compatibility.

    If no subcommand is provided (backward compatibility mode),
    use the legacy argument parser and call run_pipeline directly.
    Otherwise, delegate to cli.commands for subcommand routing.
    """
    # Check if subcommand syntax is being used
    if len(sys.argv) > 1 and sys.argv[1] in ["topic", "dry-run", "run", "ui"]:
        # Use new CLI structure
        from cli.commands import main as cli_main
        return cli_main()

    # Backward compatibility: no subcommand, use legacy parser
    parser = create_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
