"""Paper Scout - Main entry point for the arXiv paper collection pipeline.

Orchestrates the full pipeline: Preflight -> Topic Loop -> Post-Loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

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
        config.llm["response_format_supported"] = (
            preflight_result.response_format_supported
        )
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
        from core.storage.usage_tracker import UsageTracker
        db_path = config.database.get("path", "data/paper_scout.db")
        # Reuse the DB connection from preflight (avoids connection leak)
        db_manager = preflight_result.db
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
        "dedup_mode": args.dedup,
        "topics": [args.topic] if args.topic else None,
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

    # 5-b. Error alert for failed topics
    failed = topic_results.get("topics_failed", [])
    if failed:
        try:
            from output.notifiers.error_alert import send_error_alert
            from output.notifiers.registry import NotifierRegistry
            from datetime import datetime as _dt

            display_date = _dt.now(timezone.utc).strftime("%y년 %m월 %d일")
            registry = NotifierRegistry()
            # Try to find a notifier from any configured topic
            for t_spec in config.topics:
                if hasattr(t_spec, "notify") and t_spec.notify:
                    notifiers = registry.get_notifiers_for_event(
                        t_spec.notify, "error"
                    )
                    if not notifiers:
                        notifiers = registry.get_notifiers_for_event(
                            t_spec.notify, "complete"
                        )
                    if notifiers:
                        first_fail = failed[0]
                        completed = topic_results.get("topics_completed", [])
                        send_error_alert(
                            notifier=notifiers[0],
                            display_date=display_date,
                            failed_topic=first_fail["slug"],
                            failed_stage="pipeline",
                            error_cause=first_fail.get("error", "unknown")[:200],
                            completed_topics=completed,
                        )
                        break
        except Exception:
            logger.warning("Failed to send error alert (non-fatal)", exc_info=True)

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

    # 7. Weekly tasks (skip if pipeline fully failed)
    if exit_code >= 2:
        logger.info("Skipping weekly tasks: pipeline failed (exit_code=%d)", exit_code)
    else:
        try:
            from core.pipeline.weekly_guard import is_weekly_due, mark_weekly_done
            if is_weekly_due():
                logger.info("Running weekly tasks...")
                # Close main DB connection before VACUUM (requires exclusive access)
                try:
                    db_manager.close()
                except Exception:
                    logger.warning("Failed to close DB before weekly maintenance", exc_info=True)
                from core.pipeline.weekly_db_maintenance import run_weekly_maintenance
                db_config = config.database
                maint_summary = run_weekly_maintenance(db_path=db_path, db_config=db_config)
                logger.info("Weekly maintenance: %s", maint_summary)
                today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
                _provider = db_config.get("provider", "sqlite")
                _conn_str = None
                if _provider == "supabase":
                    _env_key = db_config.get("supabase", {}).get("connection_string_env", "SUPABASE_DB_URL")
                    _conn_str = os.environ.get(_env_key)

                summary_data: dict = {}
                chart_files: list = []
                report_dir = config.output.get("report_dir", "tmp/reports")
                md_path = ""
                html_path = ""
                md_content = ""
                html_content = ""

                # Build weekly folder name using calendar year + ISO week
                ref_date_obj = datetime.now(timezone.utc).date()
                _, iso_week, _ = ref_date_obj.isocalendar()
                yy = f"{ref_date_obj.year % 100:02d}"
                mm = f"{ref_date_obj.month:02d}"
                ww = f"{iso_week:02d}"
                weekly_folder_name = f"{yy}{mm}W{ww}_weekly_report"

                # --- Weekly summary data ---
                intel_cfg = config.weekly.get("intelligence", {})
                if intel_cfg.get("enabled", False):
                    try:
                        from core.pipeline.weekly_intelligence import generate_weekly_intelligence
                        summary_data, md_content, html_content = generate_weekly_intelligence(
                            db_path=db_path, date_str=today_str, config=config,
                            provider=_provider, connection_string=_conn_str,
                            rate_limiter=rate_limiter,
                        )
                    except Exception:
                        logger.warning("Weekly intelligence generation failed (non-fatal)", exc_info=True)
                else:
                    try:
                        from core.pipeline.weekly_summary import generate_weekly_summary
                        summary_data = generate_weekly_summary(
                            db_path=db_path, date_str=today_str,
                            provider=_provider, connection_string=_conn_str,
                        )
                    except Exception:
                        logger.warning("Weekly summary generation failed (non-fatal)", exc_info=True)

                # --- Weekly charts ---
                try:
                    viz_enabled = config.weekly.get("visualization", {}).get("enabled", False)
                    if viz_enabled:
                        from core.pipeline.weekly_viz import generate_weekly_charts
                        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        # Output charts into weekly folder
                        weekly_chart_dir = os.path.join(report_dir, weekly_folder_name)
                        os.makedirs(weekly_chart_dir, exist_ok=True)
                        chart_files = generate_weekly_charts(
                            db_path=db_path, date_str=today_iso,
                            output_dir=weekly_chart_dir,
                            provider=_provider, connection_string=_conn_str,
                        )
                        if chart_files:
                            logger.info("Weekly charts generated: %s", chart_files)
                        else:
                            logger.debug("Weekly charts: no charts generated (no data or viz unavailable)")
                except Exception:
                    logger.warning("Weekly chart generation failed (non-fatal)", exc_info=True)

                # --- Step A: Render HTML and MD ---
                try:
                    from pathlib import Path
                    Path(report_dir).mkdir(parents=True, exist_ok=True)

                    if not intel_cfg.get("enabled", False):
                        from core.pipeline.weekly_summary import (
                            render_weekly_summary_html,
                            render_weekly_summary_md,
                        )
                        md_content = render_weekly_summary_md(summary_data, today_str)
                        html_content = render_weekly_summary_html(
                            summary_data, today_str, chart_paths=chart_files,
                        )

                    # Write into weekly folder
                    weekly_dir = os.path.join(report_dir, weekly_folder_name)
                    Path(weekly_dir).mkdir(parents=True, exist_ok=True)

                    md_path = os.path.join(weekly_dir, "report.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_content)

                    html_path = os.path.join(weekly_dir, "report.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logger.info("Weekly report files: %s, %s", md_path, html_path)
                except Exception:
                    logger.warning("Weekly report rendering failed (non-fatal)", exc_info=True)

                # --- Step B: Deploy to gh-pages ---
                gh_pages_cfg = config.output.get("gh_pages", {})
                if gh_pages_cfg.get("enabled"):
                    try:
                        from core.pipeline.post_loop import PostLoopProcessor
                        post_proc = PostLoopProcessor(
                            config=config, db_manager=None, report_dir=report_dir,
                        )
                        deployed = post_proc._deploy_gh_pages(report_dir, gh_pages_cfg)
                        logger.info(
                            "Weekly gh-pages deploy: %s",
                            "succeeded" if deployed else "skipped",
                        )
                    except Exception:
                        logger.warning("Weekly gh-pages deploy failed (non-fatal)", exc_info=True)

                # --- Step C: Send notification ---
                try:
                    from output.notifiers.base import NotifyPayload
                    from output.notifiers.registry import NotifierRegistry
                    from core.models import NotifyConfig

                    registry = NotifierRegistry()
                    notifier = None
                    send_modes: list = ["link", "md"]

                    # Try weekly_summary config first
                    weekly_notify_cfg = config.notifications.get("weekly_summary", {})
                    if weekly_notify_cfg.get("provider"):
                        events = weekly_notify_cfg.get("events", ["complete"])
                        send_modes = weekly_notify_cfg.get("send", ["link", "md"])
                        if "complete" in events:
                            notify_cfg = NotifyConfig(
                                provider=weekly_notify_cfg["provider"],
                                channel_id=weekly_notify_cfg.get("channel_id", ""),
                                secret_key=weekly_notify_cfg.get("secret_key", ""),
                                events=events,
                                send=send_modes,
                            )
                            try:
                                notifier = registry.get_notifier(notify_cfg)
                            except ValueError:
                                pass

                    # Fall back to first topic's notification channel
                    if notifier is None:
                        for t_spec in config.topics:
                            if hasattr(t_spec, "notify") and t_spec.notify:
                                notifiers = registry.get_notifiers_for_event(
                                    t_spec.notify, "complete",
                                )
                                if notifiers:
                                    notifier = notifiers[0]
                                    # Inherit send modes from topic config
                                    for nc in t_spec.notify:
                                        if "complete" in nc.events:
                                            send_modes = nc.send or ["link", "md"]
                                            break
                                    break

                    if notifier is not None:
                        gh_pages_base = gh_pages_cfg.get("base_url", "").rstrip("/")
                        gh_pages_url = None
                        if "link" in send_modes and gh_pages_base:
                            gh_pages_url = (
                                f"{gh_pages_base}/{weekly_folder_name}/report.html"
                            )

                        display_date = datetime.now(timezone.utc).strftime("%y년 %m월 %d일")

                        file_paths_notify: dict = {}
                        if "md" in send_modes and md_path and os.path.exists(md_path):
                            file_paths_notify["md"] = os.path.abspath(md_path)

                        allowed_fmts = [f for f in send_modes if f in ("md", "html")]

                        # Build weekly-specific custom message with date range
                        ref_date = datetime.strptime(today_str, "%Y%m%d")
                        week_start = ref_date - timedelta(days=ref_date.weekday() + 7)
                        week_end = week_start + timedelta(days=6)
                        # Use total evaluated papers count (not just top 10)
                        sections = summary_data.get("sections", {})
                        exec_metrics = sections.get("executive", {}).get("metrics", {})
                        paper_count = exec_metrics.get("total_evaluated", 0)
                        if paper_count == 0:
                            paper_count = len(summary_data.get("top_papers", []))
                        custom_msg = (
                            f"{week_start.strftime('%y%m%d')}~"
                            f"{week_end.strftime('%y%m%d')} "
                            f"Weekly Paper Report"
                            f" (논문 {paper_count}편)"
                        )
                        if gh_pages_url:
                            custom_msg += f"\n\n{gh_pages_url}"

                        payload = NotifyPayload(
                            topic_slug="weekly-summary",
                            topic_name="Weekly Paper Report",
                            display_date=display_date,
                            keywords=[],
                            total_output=paper_count or 1,
                            file_paths=file_paths_notify,
                            gh_pages_url=gh_pages_url,
                            notify_mode="file",
                            allowed_formats=allowed_fmts or ["md"],
                            event_type="complete",
                            custom_message=custom_msg,
                        )
                        success = notifier.notify(payload)
                        if success:
                            logger.info("Weekly summary notification sent")
                        else:
                            logger.warning("Weekly summary notification failed")
                    else:
                        logger.debug("No notifier configured for weekly summary")
                except Exception:
                    logger.warning(
                        "Weekly summary notification failed (non-fatal)", exc_info=True,
                    )

                mark_weekly_done()
                logger.info("Weekly tasks completed")
            else:
                logger.debug("Weekly tasks: not due yet")
        except Exception:
            logger.warning("Weekly tasks failed (non-fatal)", exc_info=True)

    # 8. Cleanup (db_manager may already be closed by weekly maintenance)
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
    from dotenv import load_dotenv
    load_dotenv()

    # Check if subcommand syntax is being used
    if len(sys.argv) > 1 and sys.argv[1] in ["topic", "dry-run", "run", "ui"]:
        from cli.commands import main as cli_main
        return cli_main()

    # Check if legacy flags are used (e.g. --mode full)
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
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

    # Default: launch UI
    from local_ui.app import start_server
    start_server()
    return 0


if __name__ == "__main__":
    sys.exit(main())
