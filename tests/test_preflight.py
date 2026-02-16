"""Tests for core.pipeline.preflight module.

Covers all 8 preflight checks: config validation, API key,
RPM/daily limits, response_format cache, notifications,
search windows, and RateLimiter initialization.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import yaml

from core.config import AppConfig, ConfigError
from core.llm.openrouter_client import OpenRouterClient, OpenRouterError
from core.llm.rate_limiter import RateLimiter
from core.models import NotifyConfig, RunMeta, TopicSpec
from core.pipeline.preflight import (
    KST,
    UTC,
    PreflightError,
    PreflightResult,
    _check_api_key,
    _check_config,
    _check_notifications,
    _check_response_format,
    _compute_topic_window,
    _compute_windows,
    _detect_limits,
    _init_rate_limiter,
    _is_cache_valid,
    _load_model_caps_cache,
    _save_model_caps_cache,
    run_preflight,
)
from core.storage.db_manager import DBManager


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

KST_TZ = ZoneInfo("Asia/Seoul")


def _make_topic(**overrides: Any) -> dict:
    """Return a minimal valid topic dict, with optional overrides."""
    base = {
        "slug": "test-topic",
        "name": "Test Topic",
        "description": "A" * 150,
        "arxiv_categories": ["cs.AI"],
        "notify": {
            "provider": "discord",
            "channel_id": "123456",
            "secret_key": "https://discord.com/api/webhooks/123/abc",
        },
    }
    base.update(overrides)
    return base


def _make_raw_config(**overrides: Any) -> dict:
    """Return a minimal valid raw config dict."""
    base: dict[str, Any] = {
        "app": {"display_timezone": "UTC"},
        "llm": {
            "model": "test-model/v1",
            "preflight": {
                "fallback_rpm": 10,
                "fallback_daily": 200,
                "model_caps_path": "data/model_caps.json",
                "model_caps_ttl_days": 7,
            },
        },
        "agents": {},
        "sources": {},
        "filter": {},
        "embedding": {},
        "scoring": {
            "weights": {
                "embedding_on": {"llm": 0.55, "embed": 0.35, "recency": 0.10},
                "embedding_off": {"llm": 0.80, "recency": 0.20},
            },
            "discard_cutoff": 20,
            "max_output": 100,
        },
        "remind": {},
        "clustering": {},
        "topics": [_make_topic()],
        "output": {},
        "notifications": {},
        "database": {"path": "data/test.db"},
        "weekly": {},
        "local_ui": {},
    }
    base.update(overrides)
    return base


def _write_config(tmp_path: Path, raw: dict | None = None) -> Path:
    """Write a config.yaml to tmp_path and return its path."""
    config_path = tmp_path / "config.yaml"
    raw = raw or _make_raw_config()
    config_path.write_text(
        yaml.dump(raw, default_flow_style=False), encoding="utf-8"
    )
    return config_path


def _make_app_config(**overrides: Any) -> AppConfig:
    """Build an AppConfig from raw config dict."""
    from core.config import validate_config

    raw = _make_raw_config(**overrides)
    return validate_config(raw)


def _make_topic_spec(
    slug: str = "test-topic",
    provider: str = "discord",
    secret_key: str = "https://discord.com/api/webhooks/123/abc",
) -> TopicSpec:
    """Create a TopicSpec with given notification settings."""
    return TopicSpec(
        slug=slug,
        name="Test",
        description="A" * 150,
        arxiv_categories=["cs.AI"],
        notify=NotifyConfig(
            provider=provider,
            channel_id="123",
            secret_key=secret_key,
        ),
    )


@pytest.fixture()
def valid_config_file(tmp_path: Path) -> Path:
    """Write a valid config.yaml and return its path."""
    return _write_config(tmp_path)


@pytest.fixture()
def app_config() -> AppConfig:
    """Return a valid AppConfig for unit tests."""
    return _make_app_config()


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a mocked OpenRouterClient."""
    client = MagicMock(spec=OpenRouterClient)
    client.check_api_key.return_value = {
        "data": {
            "label": "test-key",
            "rate_limit": {"requests": 20, "interval": "60s"},
            "limit": 500,
        }
    }
    client.probe_response_format.return_value = True
    return client


@pytest.fixture()
def mock_db(tmp_path: Path) -> DBManager:
    """Return a real DBManager with a temp database."""
    db_path = str(tmp_path / "test_preflight.db")
    return DBManager(db_path)


# ===========================================================================
# Test Check 1+2: Config validation
# ===========================================================================


class TestCheckConfig:
    """Tests for _check_config (Checks 1 and 2)."""

    def test_valid_config_loads(self, valid_config_file: Path) -> None:
        """Check 1: Valid config loads successfully."""
        config = _check_config(str(valid_config_file))
        assert isinstance(config, AppConfig)
        assert len(config.topics) == 1
        assert config.topics[0].slug == "test-topic"

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Check 1: Missing config file raises PreflightError."""
        with pytest.raises(PreflightError, match="Config validation failed"):
            _check_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_config_raises(self, tmp_path: Path) -> None:
        """Check 1: Invalid config schema raises PreflightError."""
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("just_a_string", encoding="utf-8")
        with pytest.raises(PreflightError, match="Config validation failed"):
            _check_config(str(bad_path))

    def test_invalid_topic_raises(self, tmp_path: Path) -> None:
        """Check 2: Invalid topic (missing required field) raises error."""
        raw = _make_raw_config()
        raw["topics"] = [{"slug": "bad"}]  # missing required fields
        path = _write_config(tmp_path, raw)
        with pytest.raises(PreflightError, match="Config validation failed"):
            _check_config(str(path))

    def test_empty_config_raises(self, tmp_path: Path) -> None:
        """Check 1: Empty config file raises PreflightError."""
        empty_path = tmp_path / "empty.yaml"
        empty_path.write_text("", encoding="utf-8")
        with pytest.raises(PreflightError, match="Config validation failed"):
            _check_config(str(empty_path))


# ===========================================================================
# Test Check 3: API key validation
# ===========================================================================


class TestCheckApiKey:
    """Tests for _check_api_key (Check 3)."""

    def test_valid_api_key(self, mock_client: MagicMock) -> None:
        """Check 3: Valid API key returns key info."""
        result = _check_api_key(mock_client)
        assert "data" in result
        mock_client.check_api_key.assert_called_once()

    def test_invalid_api_key_raises(self, mock_client: MagicMock) -> None:
        """Check 3: Invalid API key raises PreflightError."""
        mock_client.check_api_key.side_effect = OpenRouterError(
            "Invalid key", status_code=401
        )
        with pytest.raises(PreflightError, match="API key validation failed"):
            _check_api_key(mock_client)


# ===========================================================================
# Test Check 4: RPM / daily limit detection
# ===========================================================================


class TestDetectLimits:
    """Tests for _detect_limits (Check 4)."""

    def test_detects_from_key_info(self, app_config: AppConfig) -> None:
        """Check 4: RPM/daily detected from key info."""
        key_info = {
            "data": {
                "rate_limit": {"requests": 30, "interval": "60s"},
                "limit": 1000,
            }
        }
        warnings: list[str] = []
        rpm, daily = _detect_limits(key_info, app_config, warnings)
        assert rpm == 30
        assert daily == 1000
        assert len(warnings) == 0

    def test_fallback_on_missing_rate_limit(
        self, app_config: AppConfig
    ) -> None:
        """Check 4: Fallback RPM=10, daily=200 when key info lacks data."""
        key_info = {"data": {}}
        warnings: list[str] = []
        rpm, daily = _detect_limits(key_info, app_config, warnings)
        assert rpm == 10
        assert daily == 200
        assert len(warnings) == 1
        assert "conservative defaults" in warnings[0]

    def test_fallback_on_zero_rpm(self, app_config: AppConfig) -> None:
        """Check 4: Fallback when RPM is 0."""
        key_info = {
            "data": {
                "rate_limit": {"requests": 0},
                "limit": 500,
            }
        }
        warnings: list[str] = []
        rpm, daily = _detect_limits(key_info, app_config, warnings)
        assert rpm == 10
        assert daily == 200

    def test_fallback_on_invalid_type(self, app_config: AppConfig) -> None:
        """Check 4: Fallback when key_info has unexpected structure."""
        key_info = "not a dict"  # type: ignore[assignment]
        warnings: list[str] = []
        rpm, daily = _detect_limits(key_info, app_config, warnings)  # type: ignore[arg-type]
        assert rpm == 10
        assert daily == 200

    def test_custom_fallback_from_config(self) -> None:
        """Check 4: Custom fallback values from config.llm.preflight."""
        config = _make_app_config(
            llm={
                "model": "test",
                "preflight": {
                    "fallback_rpm": 5,
                    "fallback_daily": 100,
                },
            }
        )
        key_info = {"data": {}}
        warnings: list[str] = []
        rpm, daily = _detect_limits(key_info, config, warnings)
        assert rpm == 5
        assert daily == 100


# ===========================================================================
# Test Check 5: response_format support + cache
# ===========================================================================


class TestCheckResponseFormat:
    """Tests for _check_response_format (Check 5)."""

    def test_cache_hit_returns_cached(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        """Check 5: Cache hit returns cached value without API call."""
        caps_path = tmp_path / "model_caps.json"
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        caps_path.write_text(
            json.dumps(
                {
                    "model_name": "test-model/v1",
                    "response_format_supported": True,
                    "checked_date": today,
                }
            ),
            encoding="utf-8",
        )
        config = _make_app_config(
            llm={
                "model": "test-model/v1",
                "preflight": {
                    "model_caps_path": str(caps_path),
                    "model_caps_ttl_days": 7,
                },
            }
        )
        result = _check_response_format(mock_client, config)
        assert result is True
        mock_client.probe_response_format.assert_not_called()

    def test_cache_miss_probes(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        """Check 5: Cache miss triggers probe call."""
        caps_path = tmp_path / "model_caps.json"
        config = _make_app_config(
            llm={
                "model": "test-model/v1",
                "preflight": {
                    "model_caps_path": str(caps_path),
                    "model_caps_ttl_days": 7,
                },
            }
        )
        mock_client.probe_response_format.return_value = False
        result = _check_response_format(mock_client, config)
        assert result is False
        mock_client.probe_response_format.assert_called_once_with(
            "test-model/v1"
        )

    def test_expired_cache_probes(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        """Check 5: Expired cache (>TTL days) triggers new probe."""
        caps_path = tmp_path / "model_caps.json"
        old_date = (
            datetime.now(UTC) - timedelta(days=10)
        ).strftime("%Y-%m-%d")
        caps_path.write_text(
            json.dumps(
                {
                    "model_name": "test-model/v1",
                    "response_format_supported": False,
                    "checked_date": old_date,
                }
            ),
            encoding="utf-8",
        )
        config = _make_app_config(
            llm={
                "model": "test-model/v1",
                "preflight": {
                    "model_caps_path": str(caps_path),
                    "model_caps_ttl_days": 7,
                },
            }
        )
        mock_client.probe_response_format.return_value = True
        result = _check_response_format(mock_client, config)
        assert result is True
        mock_client.probe_response_format.assert_called_once()

    def test_cache_saved_after_probe(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        """Check 5: Cache file is saved after probe."""
        caps_path = tmp_path / "data" / "model_caps.json"
        config = _make_app_config(
            llm={
                "model": "test-model/v1",
                "preflight": {
                    "model_caps_path": str(caps_path),
                    "model_caps_ttl_days": 7,
                },
            }
        )
        mock_client.probe_response_format.return_value = True
        _check_response_format(mock_client, config)

        assert caps_path.exists()
        cached = json.loads(caps_path.read_text(encoding="utf-8"))
        assert cached["model_name"] == "test-model/v1"
        assert cached["response_format_supported"] is True
        assert "checked_date" in cached

    def test_different_model_name_triggers_probe(
        self, tmp_path: Path, mock_client: MagicMock
    ) -> None:
        """Check 5: Different model name from cache triggers probe."""
        caps_path = tmp_path / "model_caps.json"
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        caps_path.write_text(
            json.dumps(
                {
                    "model_name": "other-model",
                    "response_format_supported": True,
                    "checked_date": today,
                }
            ),
            encoding="utf-8",
        )
        config = _make_app_config(
            llm={
                "model": "test-model/v1",
                "preflight": {
                    "model_caps_path": str(caps_path),
                    "model_caps_ttl_days": 7,
                },
            }
        )
        mock_client.probe_response_format.return_value = False
        result = _check_response_format(mock_client, config)
        assert result is False
        mock_client.probe_response_format.assert_called_once()


# ===========================================================================
# Test Check 6: Notification channel validation
# ===========================================================================


class TestCheckNotifications:
    """Tests for _check_notifications (Check 6)."""

    def test_valid_discord_no_warning(self) -> None:
        """Check 6: Valid Discord webhook passes."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(
                provider="discord",
                secret_key="https://discord.com/api/webhooks/123/abc-def",
            )
        ]
        warnings = _check_notifications(config)
        assert len(warnings) == 0

    def test_invalid_discord_returns_warning(self) -> None:
        """Check 6: Invalid Discord webhook returns warning (not error)."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(
                provider="discord",
                secret_key="not-a-webhook-url",
            )
        ]
        warnings = _check_notifications(config)
        assert len(warnings) == 1
        assert "Discord webhook" in warnings[0]

    def test_valid_telegram_no_warning(self) -> None:
        """Check 6: Valid Telegram bot token passes."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(
                provider="telegram",
                secret_key="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ012345678",
            )
        ]
        warnings = _check_notifications(config)
        assert len(warnings) == 0

    def test_invalid_telegram_returns_warning(self) -> None:
        """Check 6: Invalid Telegram token returns warning."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(
                provider="telegram",
                secret_key="bad-token",
            )
        ]
        warnings = _check_notifications(config)
        assert len(warnings) == 1
        assert "Telegram bot token" in warnings[0]

    def test_multiple_topics_each_checked(self) -> None:
        """Check 6: All topics checked, multiple warnings possible."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(
                slug="topic-a", provider="discord", secret_key="bad"
            ),
            _make_topic_spec(
                slug="topic-b", provider="telegram", secret_key="bad"
            ),
        ]
        warnings = _check_notifications(config)
        assert len(warnings) == 2


# ===========================================================================
# Test Check 7: Search window computation
# ===========================================================================


class TestComputeWindows:
    """Tests for _compute_windows and _compute_topic_window (Check 7)."""

    def test_window_from_db_latest_run(
        self, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: Window start from DB latest completed run."""
        # Insert a completed run
        run_end = datetime(2026, 2, 15, 2, 0, 0, tzinfo=UTC)
        run = RunMeta(
            topic_slug="test-topic",
            window_start_utc=datetime(2026, 2, 14, 2, 0, 0, tzinfo=UTC),
            window_end_utc=run_end,
            display_date_kst="2026-02-15",
            embedding_mode="disabled",
            scoring_weights={},
            response_format_supported=True,
            prompt_versions={},
            status="completed",
        )
        mock_db.create_run(run)

        result = _compute_windows(app_config, mock_db, None, None)
        start, end = result["test-topic"]

        # window_start should be based on the DB run's window_end_utc
        # minus 30min buffer
        expected_start = run_end - timedelta(minutes=30)
        assert start == expected_start

    def test_window_from_last_success_json(
        self, tmp_path: Path, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: Window start from data/last_success.json."""
        last_success_path = tmp_path / "data" / "last_success.json"
        last_success_path.parent.mkdir(parents=True, exist_ok=True)
        ts = "2026-02-14T02:00:00+00:00"
        last_success_path.write_text(
            json.dumps({"test-topic": {"window_end_utc": ts}}),
            encoding="utf-8",
        )

        with patch(
            "core.pipeline.preflight._LAST_SUCCESS_PATH",
            str(last_success_path),
        ):
            result = _compute_windows(app_config, mock_db, None, None)
            start, _ = result["test-topic"]

        expected = datetime.fromisoformat(ts) - timedelta(minutes=30)
        assert start == expected

    def test_72h_fallback_window(
        self, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: 72h fallback when no DB run and no last_success."""
        with patch(
            "core.pipeline.preflight._load_last_success", return_value=None
        ):
            result = _compute_windows(app_config, mock_db, None, None)
            start, end = result["test-topic"]

        # window_end is today KST 11:00 -> UTC + 30min buffer
        now_kst = datetime.now(KST_TZ)
        today_11 = now_kst.replace(hour=11, minute=0, second=0, microsecond=0)
        expected_end = today_11.astimezone(UTC) + timedelta(minutes=30)
        expected_start = (
            today_11.astimezone(UTC)
            - timedelta(hours=72)
            - timedelta(minutes=30)
        )

        assert end == expected_end
        assert start == expected_start

    def test_manual_date_from_to_override(
        self, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: Manual date_from and date_to override automatic mode."""
        date_from = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        date_to = datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC)

        result = _compute_windows(
            app_config, mock_db, date_from, date_to
        )
        start, end = result["test-topic"]

        assert start == date_from - timedelta(minutes=30)
        assert end == date_to + timedelta(minutes=30)

    def test_buffer_applied(
        self, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: +-30min buffer is applied to both start and end."""
        date_from = datetime(2026, 2, 10, 12, 0, 0, tzinfo=UTC)
        date_to = datetime(2026, 2, 15, 12, 0, 0, tzinfo=UTC)

        result = _compute_windows(
            app_config, mock_db, date_from, date_to
        )
        start, end = result["test-topic"]

        assert start == date_from - timedelta(minutes=30)
        assert end == date_to + timedelta(minutes=30)
        # Verify the buffer is exactly 30 minutes
        assert (date_from - start).total_seconds() == 30 * 60
        assert (end - date_to).total_seconds() == 30 * 60

    def test_kst_11_to_utc_conversion(
        self, mock_db: DBManager, app_config: AppConfig
    ) -> None:
        """Check 7: KST 11:00 converts to UTC 02:00."""
        with patch(
            "core.pipeline.preflight._load_last_success", return_value=None
        ):
            result = _compute_windows(app_config, mock_db, None, None)
            _, end = result["test-topic"]

        # end should have 30min buffer added to UTC 02:00
        # So end.hour should be 2, end.minute should be 30 (in UTC)
        # Note: we subtract buffer to get the raw window_end
        raw_end = end - timedelta(minutes=30)
        assert raw_end.hour == 2
        assert raw_end.minute == 0

    def test_multiple_topics_each_get_window(
        self, mock_db: DBManager
    ) -> None:
        """Check 7: Each topic gets its own window."""
        config = _make_app_config()
        config.topics = [
            _make_topic_spec(slug="topic-a"),
            _make_topic_spec(slug="topic-b"),
        ]
        # Create a completed run for topic-a only
        run = RunMeta(
            topic_slug="topic-a",
            window_start_utc=datetime(2026, 2, 14, 0, 0, 0, tzinfo=UTC),
            window_end_utc=datetime(2026, 2, 15, 2, 0, 0, tzinfo=UTC),
            display_date_kst="2026-02-15",
            embedding_mode="disabled",
            scoring_weights={},
            response_format_supported=True,
            prompt_versions={},
            status="completed",
        )
        mock_db.create_run(run)

        with patch(
            "core.pipeline.preflight._load_last_success", return_value=None
        ):
            result = _compute_windows(config, mock_db, None, None)

        assert "topic-a" in result
        assert "topic-b" in result

        # topic-a should use DB run, topic-b should use 72h fallback
        start_a, _ = result["topic-a"]
        start_b, _ = result["topic-b"]

        # topic-a's start comes from DB window_end_utc - 30min
        expected_a = datetime(2026, 2, 15, 2, 0, 0, tzinfo=UTC) - timedelta(
            minutes=30
        )
        assert start_a == expected_a

        # topic-b should have a different (earlier) start from 72h fallback
        # (it should be roughly 72h before today's KST 11:00 UTC)
        assert start_b != start_a


# ===========================================================================
# Test Check 8: RateLimiter initialization
# ===========================================================================


class TestInitRateLimiter:
    """Tests for _init_rate_limiter (Check 8)."""

    def test_rate_limiter_initialized(self) -> None:
        """Check 8: RateLimiter initialized with detected values."""
        rl = _init_rate_limiter(detected_rpm=25, daily_limit=500)
        assert isinstance(rl, RateLimiter)
        # Verify the delay calculation uses detected RPM
        expected_delay = 60.0 / 25 + 0.5
        assert abs(rl.delay - expected_delay) < 0.01

    def test_rate_limiter_with_defaults(self) -> None:
        """Check 8: RateLimiter works with fallback defaults."""
        rl = _init_rate_limiter(detected_rpm=10, daily_limit=200)
        expected_delay = 60.0 / 10 + 0.5
        assert abs(rl.delay - expected_delay) < 0.01


# ===========================================================================
# Test PreflightResult
# ===========================================================================


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_result_contains_all_fields(
        self, app_config: AppConfig, mock_client: MagicMock, mock_db: DBManager
    ) -> None:
        """PreflightResult has all required fields."""
        rl = RateLimiter(detected_rpm=10, daily_limit=200)
        result = PreflightResult(
            config=app_config,
            client=mock_client,
            rate_limiter=rl,
            db=mock_db,
            response_format_supported=True,
            detected_rpm=10,
            detected_daily_limit=200,
            topic_windows={"test": (datetime.now(UTC), datetime.now(UTC))},
            warnings=["test warning"],
        )
        assert result.config is app_config
        assert result.client is mock_client
        assert result.rate_limiter is rl
        assert result.db is mock_db
        assert result.response_format_supported is True
        assert result.detected_rpm == 10
        assert result.detected_daily_limit == 200
        assert "test" in result.topic_windows
        assert len(result.warnings) == 1

    def test_result_default_warnings(
        self, app_config: AppConfig, mock_client: MagicMock, mock_db: DBManager
    ) -> None:
        """PreflightResult defaults warnings to empty list."""
        rl = RateLimiter(detected_rpm=10, daily_limit=200)
        result = PreflightResult(
            config=app_config,
            client=mock_client,
            rate_limiter=rl,
            db=mock_db,
            response_format_supported=False,
            detected_rpm=10,
            detected_daily_limit=200,
            topic_windows={},
        )
        assert result.warnings == []


# ===========================================================================
# Test cache helpers
# ===========================================================================


class TestCacheHelpers:
    """Tests for model_caps cache helper functions."""

    def test_is_cache_valid_within_ttl(self) -> None:
        """Cache entry within TTL is valid."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert _is_cache_valid({"checked_date": today}, 7) is True

    def test_is_cache_valid_expired(self) -> None:
        """Cache entry beyond TTL is invalid."""
        old = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
        assert _is_cache_valid({"checked_date": old}, 7) is False

    def test_is_cache_valid_no_date(self) -> None:
        """Cache entry with no checked_date is invalid."""
        assert _is_cache_valid({}, 7) is False

    def test_is_cache_valid_invalid_date(self) -> None:
        """Cache entry with invalid date format is invalid."""
        assert _is_cache_valid({"checked_date": "not-a-date"}, 7) is False

    def test_load_cache_nonexistent(self, tmp_path: Path) -> None:
        """Loading nonexistent cache returns None."""
        result = _load_model_caps_cache(tmp_path / "nope.json")
        assert result is None

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        """Save then load round-trips correctly."""
        path = tmp_path / "caps.json"
        _save_model_caps_cache(path, "my-model", True)
        loaded = _load_model_caps_cache(path)
        assert loaded is not None
        assert loaded["model_name"] == "my-model"
        assert loaded["response_format_supported"] is True

    def test_load_corrupted_cache(self, tmp_path: Path) -> None:
        """Loading corrupted JSON returns None."""
        path = tmp_path / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")
        result = _load_model_caps_cache(path)
        assert result is None


# ===========================================================================
# Test full run_preflight integration
# ===========================================================================


class TestRunPreflight:
    """Integration tests for run_preflight orchestrator."""

    @patch("core.pipeline.preflight.OpenRouterClient")
    @patch("core.pipeline.preflight.DBManager")
    @patch("core.pipeline.preflight._load_last_success", return_value=None)
    def test_full_preflight_succeeds(
        self,
        mock_last_success: MagicMock,
        MockDBManager: MagicMock,
        MockClient: MagicMock,
        valid_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """Full preflight succeeds with valid config and mocked externals."""
        # Configure mocked client
        mock_client_inst = MockClient.return_value
        mock_client_inst.check_api_key.return_value = {
            "data": {
                "rate_limit": {"requests": 15, "interval": "60s"},
                "limit": 300,
            }
        }
        mock_client_inst.probe_response_format.return_value = True

        # Configure mocked DB
        mock_db_inst = MockDBManager.return_value
        mock_db_inst.get_latest_completed_run.return_value = None

        result = run_preflight(config_path=str(valid_config_file))

        assert isinstance(result, PreflightResult)
        assert result.detected_rpm == 15
        assert result.detected_daily_limit == 300
        assert result.response_format_supported is True
        assert isinstance(result.rate_limiter, RateLimiter)
        assert "test-topic" in result.topic_windows

    @patch("core.pipeline.preflight.OpenRouterClient")
    @patch("core.pipeline.preflight.DBManager")
    @patch("core.pipeline.preflight._load_last_success", return_value=None)
    def test_warnings_accumulated(
        self,
        mock_last_success: MagicMock,
        MockDBManager: MagicMock,
        MockClient: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Warnings from checks 4 and 6 are accumulated in result."""
        # Config with bad notification webhook
        raw = _make_raw_config()
        raw["topics"] = [
            _make_topic(
                notify={
                    "provider": "discord",
                    "channel_id": "123",
                    "secret_key": "bad-webhook",
                }
            )
        ]
        config_path = _write_config(tmp_path, raw)

        # Client returns empty data (triggers limit detection fallback)
        mock_client_inst = MockClient.return_value
        mock_client_inst.check_api_key.return_value = {"data": {}}
        mock_client_inst.probe_response_format.return_value = False

        mock_db_inst = MockDBManager.return_value
        mock_db_inst.get_latest_completed_run.return_value = None

        result = run_preflight(config_path=str(config_path))

        # Should have warnings from: limit detection fallback + Discord format
        assert len(result.warnings) >= 2
        assert result.detected_rpm == 10
        assert result.detected_daily_limit == 200

    def test_preflight_fails_on_bad_config(self, tmp_path: Path) -> None:
        """run_preflight raises PreflightError for invalid config."""
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text("not_valid: true", encoding="utf-8")
        with pytest.raises(PreflightError):
            run_preflight(config_path=str(bad_path))
