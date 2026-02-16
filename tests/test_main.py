"""Tests for main.py entry point (TASK-039).

Covers:
- CLI argument parsing (defaults, custom values, combinations)
- Pipeline execution (full mode, dry-run, error handling)
- Exit code logic (0=success, 1=partial, 2=total failure)
- Post-loop error isolation (non-fatal)
- Weekly placeholder logging
- KeyboardInterrupt handling
- Integration-level ordering (config -> preflight -> resources -> loop)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from main import create_parser, main, run_pipeline, _parse_date, _setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults."""
    defaults = {
        "mode": "full",
        "date_from": None,
        "date_to": None,
        "dedup": "skip_recent",
        "topic": None,
        "log_level": "INFO",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_config() -> MagicMock:
    """Build a mock AppConfig."""
    cfg = MagicMock()
    cfg.database = {"path": ":memory:"}
    cfg.output = {"report_dir": "tmp/reports"}
    cfg.llm = {"model": "test-model"}
    cfg.topics = []
    return cfg


def _mock_preflight_result(config: Any = None) -> MagicMock:
    """Build a mock PreflightResult."""
    result = MagicMock()
    result.config = config or _mock_config()
    result.warnings = []
    result.rate_limiter = MagicMock()
    return result


# ===========================================================================
# CLI Argument Parsing Tests
# ===========================================================================


class TestCLIArgumentParsing:
    """Tests for create_parser() and argument parsing."""

    def test_default_args(self) -> None:
        """Default args: mode=full, dedup=skip_recent, log-level=INFO."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.mode == "full"
        assert args.dedup == "skip_recent"
        assert args.log_level == "INFO"
        assert args.date_from is None
        assert args.date_to is None
        assert args.topic is None

    def test_dry_run_mode(self) -> None:
        """--mode dry-run sets mode correctly."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "dry-run"])
        assert args.mode == "dry-run"

    def test_date_range_args(self) -> None:
        """--date-from and --date-to are parsed as strings."""
        parser = create_parser()
        args = parser.parse_args([
            "--date-from", "2026-01-01",
            "--date-to", "2026-01-15",
        ])
        assert args.date_from == "2026-01-01"
        assert args.date_to == "2026-01-15"

    def test_topic_filter(self) -> None:
        """--topic sets the topic slug filter."""
        parser = create_parser()
        args = parser.parse_args(["--topic", "llm-agents"])
        assert args.topic == "llm-agents"

    def test_dedup_none(self) -> None:
        """--dedup none disables dedup."""
        parser = create_parser()
        args = parser.parse_args(["--dedup", "none"])
        assert args.dedup == "none"

    def test_invalid_mode_rejected(self) -> None:
        """Invalid --mode value is rejected by argparse."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])

    def test_log_level_setting(self) -> None:
        """--log-level sets the logging level."""
        parser = create_parser()
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_all_args_combined(self) -> None:
        """All arguments can be specified together."""
        parser = create_parser()
        args = parser.parse_args([
            "--mode", "dry-run",
            "--date-from", "2026-02-01",
            "--date-to", "2026-02-14",
            "--dedup", "none",
            "--topic", "rag-systems",
            "--log-level", "WARNING",
        ])
        assert args.mode == "dry-run"
        assert args.date_from == "2026-02-01"
        assert args.date_to == "2026-02-14"
        assert args.dedup == "none"
        assert args.topic == "rag-systems"
        assert args.log_level == "WARNING"

    def test_invalid_dedup_rejected(self) -> None:
        """Invalid --dedup value is rejected by argparse."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dedup", "aggressive"])


# ===========================================================================
# _parse_date Tests
# ===========================================================================


class TestParseDate:
    """Tests for _parse_date helper."""

    def test_none_returns_none(self) -> None:
        assert _parse_date(None) is None

    def test_valid_date(self) -> None:
        assert _parse_date("2026-01-15") == "2026-01-15"

    def test_invalid_format_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_date("01-15-2026")

    def test_invalid_date_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_date("not-a-date")


# ===========================================================================
# _setup_logging Tests
# ===========================================================================


class TestSetupLogging:
    """Tests for _setup_logging helper."""

    def test_valid_level(self) -> None:
        """Valid level name configures logging without error."""
        _setup_logging("DEBUG")

    def test_invalid_level_falls_back_to_info(self) -> None:
        """Invalid level name falls back to INFO."""
        _setup_logging("INVALID_LEVEL")


# ===========================================================================
# Pipeline Execution Tests
# ===========================================================================


# Patch paths point to the module where the name is *looked up*, which is
# "main" even though the import happens at call-time (the name is resolved
# inside the function scope, but the actual object lives in the source
# module).  We patch the *source module* so that the function-level import
# picks up the mock.

_P_LOAD = "core.config.load_config"
_P_PREFLIGHT = "core.pipeline.preflight.run_preflight"
_P_DB = "core.storage.db_manager.DBManager"
_P_SW = "core.pipeline.search_window.SearchWindowComputer"
_P_UT = "core.storage.usage_tracker.UsageTracker"
_P_ORCH = "core.pipeline.topic_loop.TopicLoopOrchestrator"
_P_POST = "core.pipeline.post_loop.PostLoopProcessor"
_P_SUNDAY = "main._is_sunday_utc"


class TestRunPipeline:
    """Tests for run_pipeline() function."""

    @patch(_P_LOAD)
    def test_config_load_failure_returns_2(
        self, mock_load: MagicMock
    ) -> None:
        """Config load failure returns exit code 2."""
        mock_load.side_effect = FileNotFoundError("config.yaml not found")
        args = _make_args()
        result = run_pipeline(args)
        assert result == 2

    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_preflight_failure_returns_2(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        """Preflight failure returns exit code 2."""
        mock_load.return_value = _mock_config()
        mock_preflight.side_effect = RuntimeError("API key invalid")
        args = _make_args()
        result = run_pipeline(args)
        assert result == 2

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_all_topics_succeed_returns_0(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """All topics succeed -> exit code 0."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [{"slug": "t1", "total_output": 5}],
            "topics_skipped": [],
            "topics_failed": [],
        }

        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        result = run_pipeline(args)
        assert result == 0

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_partial_failure_returns_1(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Some topics fail -> exit code 1."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [{"slug": "t1", "total_output": 5}],
            "topics_skipped": [],
            "topics_failed": [{"slug": "t2", "error": "some error"}],
        }

        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        result = run_pipeline(args)
        assert result == 1

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_all_topics_fail_returns_2(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """All topics fail -> exit code 2."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [
                {"slug": "t1", "error": "err1"},
                {"slug": "t2", "error": "err2"},
            ],
        }

        args = _make_args()
        result = run_pipeline(args)
        assert result == 2

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_topic_loop_exception_returns_2(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Topic loop unhandled exception -> exit code 2."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.side_effect = RuntimeError("fatal loop error")

        args = _make_args()
        result = run_pipeline(args)
        assert result == 2

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_post_loop_error_does_not_change_exit_code(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Post-loop error is non-fatal; exit code remains 0."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [{"slug": "t1", "total_output": 3}],
            "topics_skipped": [],
            "topics_failed": [],
        }

        mock_post = mock_post_cls.return_value
        mock_post.process.side_effect = RuntimeError("notification failure")

        args = _make_args()
        result = run_pipeline(args)
        assert result == 0

    @patch(_P_SUNDAY, return_value=True)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_weekly_placeholder_logged_on_sunday(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Weekly tasks placeholder is logged on Sunday."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [{"slug": "t1", "total_output": 1}],
            "topics_skipped": [],
            "topics_failed": [],
        }
        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        with caplog.at_level(logging.INFO, logger="main"):
            run_pipeline(args)

        assert any(
            "Weekly tasks" in r.message for r in caplog.records
        )

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_dry_run_skips_post_loop(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Dry-run mode skips post-loop processing."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [{"slug": "t1", "total_output": 0}],
            "topics_skipped": [],
            "topics_failed": [],
        }

        args = _make_args(mode="dry-run")

        with patch(_P_POST) as mock_post_cls:
            result = run_pipeline(args)
            assert result == 0
            # PostLoopProcessor should NOT be called in dry-run
            mock_post_cls.assert_not_called()

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_no_topics_processed_returns_0(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """No topics processed (all skipped) -> exit code 0."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [{"slug": "t1", "reason": "already_completed"}],
            "topics_failed": [],
        }

        args = _make_args()
        result = run_pipeline(args)
        assert result == 0

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_preflight_warnings_are_logged(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Preflight warnings are logged during pipeline execution."""
        config = _mock_config()
        mock_load.return_value = config
        pf_result = _mock_preflight_result(config)
        pf_result.warnings = ["Discord webhook looks invalid"]
        mock_preflight.return_value = pf_result

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }
        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        with caplog.at_level(logging.WARNING, logger="main"):
            run_pipeline(args)

        assert any(
            "Discord webhook" in r.message for r in caplog.records
        )


# ===========================================================================
# Main Function Tests
# ===========================================================================


class TestMain:
    """Tests for main() function."""

    @patch("main.run_pipeline", return_value=0)
    @patch("main._setup_logging")
    @patch("main.create_parser")
    def test_main_returns_pipeline_result(
        self,
        mock_parser_fn: MagicMock,
        mock_logging: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """main() returns the run_pipeline exit code."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _make_args()
        mock_parser_fn.return_value = mock_parser

        result = main()
        assert result == 0
        mock_logging.assert_called_once_with("INFO")

    @patch("main.run_pipeline", side_effect=KeyboardInterrupt)
    @patch("main._setup_logging")
    @patch("main.create_parser")
    def test_keyboard_interrupt_returns_130(
        self,
        mock_parser_fn: MagicMock,
        mock_logging: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """KeyboardInterrupt returns exit code 130."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _make_args()
        mock_parser_fn.return_value = mock_parser

        result = main()
        assert result == 130

    @patch("main.run_pipeline", return_value=1)
    @patch("main._setup_logging")
    @patch("main.create_parser")
    def test_main_propagates_nonzero_exit(
        self,
        mock_parser_fn: MagicMock,
        mock_logging: MagicMock,
        mock_pipeline: MagicMock,
    ) -> None:
        """main() propagates non-zero exit code from pipeline."""
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = _make_args()
        mock_parser_fn.return_value = mock_parser

        result = main()
        assert result == 1


# ===========================================================================
# Integration-Level Tests
# ===========================================================================


class TestIntegration:
    """Integration-level tests verifying ordering and data flow."""

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_config_loaded_before_preflight(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Config is loaded before preflight is called."""
        call_order: list[str] = []

        config = _mock_config()

        def load_side_effect() -> Any:
            call_order.append("load_config")
            return config

        def preflight_side_effect(**kw: Any) -> Any:
            call_order.append("run_preflight")
            return _mock_preflight_result(config)

        mock_load.side_effect = load_side_effect
        mock_preflight.side_effect = preflight_side_effect

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }
        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        run_pipeline(args)

        assert call_order.index("load_config") < call_order.index(
            "run_preflight"
        )

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_resources_initialized_before_topic_loop(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db_cls: MagicMock,
        mock_sw_cls: MagicMock,
        mock_ut_cls: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Shared resources are initialized before TopicLoopOrchestrator."""
        call_order: list[str] = []

        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        def track_db(*a: Any, **kw: Any) -> MagicMock:
            call_order.append("DBManager")
            return MagicMock()

        mock_db_cls.side_effect = track_db

        def track_orch(*a: Any, **kw: Any) -> MagicMock:
            call_order.append("TopicLoopOrchestrator")
            m = MagicMock()
            m.run_all_topics.return_value = {
                "topics_completed": [],
                "topics_skipped": [],
                "topics_failed": [],
            }
            return m

        mock_orch_cls.side_effect = track_orch
        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args()
        run_pipeline(args)

        assert call_order.index("DBManager") < call_order.index(
            "TopicLoopOrchestrator"
        )

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_post_loop_receives_topic_results(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """PostLoopProcessor receives topic_results from topic loop."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        expected_results = {
            "topics_completed": [{"slug": "t1", "total_output": 7}],
            "topics_skipped": [],
            "topics_failed": [],
        }
        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = expected_results

        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args(mode="full")
        run_pipeline(args)

        mock_post.process.assert_called_once_with(expected_results)

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_POST)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_run_options_passed_to_orchestrator(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_post_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """run_options dict is correctly built and passed to orchestrator."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }
        mock_post = mock_post_cls.return_value
        mock_post.process.return_value = {}

        args = _make_args(
            mode="dry-run",
            date_from="2026-02-01",
            date_to="2026-02-14",
            dedup="none",
            topic="rag-systems",
        )
        run_pipeline(args)

        # Verify orchestrator was constructed with correct run_options
        orch_call_kwargs = mock_orch_cls.call_args[1]
        run_options = orch_call_kwargs["run_options"]
        assert run_options["mode"] == "dry-run"
        assert run_options["date_from"] == "2026-02-01"
        assert run_options["date_to"] == "2026-02-14"
        assert run_options["dedup"] == "none"
        assert run_options["topic_slugs"] == ["rag-systems"]

    @patch(_P_SUNDAY, return_value=False)
    @patch(_P_ORCH)
    @patch(_P_UT)
    @patch(_P_SW)
    @patch(_P_DB)
    @patch(_P_PREFLIGHT)
    @patch(_P_LOAD)
    def test_db_close_called_on_success(
        self,
        mock_load: MagicMock,
        mock_preflight: MagicMock,
        mock_db_cls: MagicMock,
        mock_sw: MagicMock,
        mock_ut: MagicMock,
        mock_orch_cls: MagicMock,
        mock_sunday: MagicMock,
    ) -> None:
        """Database close is called after pipeline completes."""
        config = _mock_config()
        mock_load.return_value = config
        mock_preflight.return_value = _mock_preflight_result(config)

        mock_db_inst = MagicMock()
        mock_db_cls.return_value = mock_db_inst

        mock_orch = mock_orch_cls.return_value
        mock_orch.run_all_topics.return_value = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }

        args = _make_args()
        run_pipeline(args)

        mock_db_inst.close.assert_called_once()
