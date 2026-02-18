"""Preflight checks for Paper Scout pipeline.

Discovers failures early (before an entire run) by validating config,
API keys, rate limits, model capabilities, notification channels,
search windows, and rate-limiter initialization.

DevSpec Section 9-1 (Preflight Checks) and 9-2 (Search Window).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import AppConfig, ConfigError, load_config
from core.llm.openrouter_client import OpenRouterClient, OpenRouterError
from core.llm.rate_limiter import RateLimiter
from core.models import TopicSpec
from core.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

UTC = timezone.utc
_BUFFER_MINUTES = 30
_FALLBACK_HOURS = 72
_DEFAULT_FALLBACK_RPM = 10
_DEFAULT_FALLBACK_DAILY = 200
_DEFAULT_MODEL_CAPS_PATH = "data/model_caps.json"
_DEFAULT_MODEL_CAPS_TTL_DAYS = 7
_LAST_SUCCESS_PATH = "data/last_success.json"

# Simple patterns for notification validation
_DISCORD_WEBHOOK_PATTERN = re.compile(
    r"^https://discord(app)?\.com/api/webhooks/\d+/.+"
)
_TELEGRAM_TOKEN_PATTERN = re.compile(r"^\d+:[A-Za-z0-9_-]{35,}$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PreflightError(Exception):
    """Raised when a critical preflight check fails (checks 1-3)."""

    pass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Aggregated output from all preflight checks."""

    config: AppConfig
    client: OpenRouterClient
    rate_limiter: RateLimiter
    db: DBManager
    response_format_supported: bool
    detected_rpm: int
    detected_daily_limit: int
    topic_windows: dict[str, tuple[datetime, datetime]]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_preflight(
    config_path: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> PreflightResult:
    """Execute all 8 preflight checks in order.

    Args:
        config_path: Override config file path.
        date_from: Manual window start (for CLI/dispatch).
        date_to: Manual window end (for CLI/dispatch).

    Returns:
        PreflightResult with all initialized components.

    Raises:
        PreflightError: On critical failures (checks 1-3).
    """
    warnings_list: list[str] = []

    # Check 1+2: Config schema + topic validation
    config = _check_config(config_path)

    # Instantiate OpenRouterClient
    client = OpenRouterClient(config)

    # Check 3: API key validity
    key_info = _check_api_key(client)

    # Check 4: RPM / daily limit detection
    detected_rpm, detected_daily_limit = _detect_limits(
        key_info, config, warnings_list
    )

    # Check 5: response_format support (with cache)
    response_format_supported = _check_response_format(client, config)

    # Check 6: Notification channel connectivity
    notif_warnings = _check_notifications(config)
    warnings_list.extend(notif_warnings)

    # Initialize DBManager for checks 7+8
    db_path = config.database.get("path", "data/paper_scout.db")
    db = DBManager(db_path)

    # Quick integrity check
    try:
        result = db._conn.execute("PRAGMA integrity_check(1)").fetchone()
        if result and result[0] != "ok":
            logger.warning("DB integrity issue detected: %s", result[0])
    except Exception as ic_exc:
        logger.warning("DB integrity check failed: %s", ic_exc)

    try:
        # Check 7: Search windows per topic
        topic_windows = _compute_windows(config, db, date_from, date_to)
    except Exception:
        db.close()
        raise

    # Check 8: RateLimiter initialization
    rate_limiter = _init_rate_limiter(detected_rpm, detected_daily_limit)

    return PreflightResult(
        config=config,
        client=client,
        rate_limiter=rate_limiter,
        db=db,
        response_format_supported=response_format_supported,
        detected_rpm=detected_rpm,
        detected_daily_limit=detected_daily_limit,
        topic_windows=topic_windows,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_config(config_path: str | None) -> AppConfig:
    """Check 1+2: Load and validate config + topics.

    Raises:
        PreflightError: If config is invalid or missing.
    """
    try:
        return load_config(config_path)
    except (ConfigError, FileNotFoundError) as exc:
        raise PreflightError(str(exc)) from exc


def _check_api_key(client: OpenRouterClient) -> dict[str, Any]:
    """Check 3: Validate API key via GET /api/v1/key.

    Raises:
        PreflightError: If the API key is invalid.
    """
    try:
        return client.check_api_key()
    except OpenRouterError as exc:
        raise PreflightError(str(exc)) from exc


def _detect_limits(
    key_info: dict[str, Any],
    config: AppConfig,
    warnings_list: list[str],
) -> tuple[int, int]:
    """Check 4: Extract RPM and daily limit from key info.

    On failure, falls back to conservative defaults from config
    (llm.preflight.fallback_rpm / fallback_daily) or hardcoded defaults.

    Returns:
        (detected_rpm, detected_daily_limit)
    """
    preflight_cfg = config.llm.get("preflight", {})
    fallback_rpm = preflight_cfg.get(
        "conservative_rpm",
        preflight_cfg.get("fallback_rpm", _DEFAULT_FALLBACK_RPM),
    )
    fallback_daily = preflight_cfg.get(
        "conservative_daily",
        preflight_cfg.get("fallback_daily", _DEFAULT_FALLBACK_DAILY),
    )

    try:
        data = key_info.get("data", key_info)

        # Extract RPM from rate_limit structure
        rate_limit = data.get("rate_limit", {})
        rpm = rate_limit.get("requests")
        if rpm is not None:
            rpm = int(rpm)
        else:
            rpm = None

        # Extract daily limit from limit field
        daily = data.get("limit")
        if daily is not None:
            daily = int(daily)
        else:
            daily = None

        if rpm is None or rpm <= 0:
            raise ValueError("RPM not found or invalid")
        if daily is None or daily <= 0:
            raise ValueError("Daily limit not found or invalid")

        return rpm, daily

    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        msg = (
            f"Could not detect rate limits from API key info: {exc}. "
            f"Using conservative defaults: RPM={fallback_rpm}, "
            f"daily={fallback_daily}"
        )
        logger.warning(msg)
        warnings_list.append(msg)
        return fallback_rpm, fallback_daily


def _check_response_format(
    client: OpenRouterClient, config: AppConfig
) -> bool:
    """Check 5: Check response_format support with model_caps.json cache.

    Cache format:
        {"model_name": "...", "response_format_supported": true/false,
         "checked_date": "YYYY-MM-DD"}

    TTL: 7 days (configurable via llm.preflight.model_caps_ttl_days).
    Cache hit skips the probe call.  Cache miss triggers a probe.

    Returns:
        True if model supports response_format, False otherwise.
    """
    preflight_cfg = config.llm.get("preflight", {})
    caps_path_str = preflight_cfg.get(
        "model_caps_path", _DEFAULT_MODEL_CAPS_PATH
    )
    ttl_days = preflight_cfg.get(
        "model_caps_ttl_days", _DEFAULT_MODEL_CAPS_TTL_DAYS
    )
    model_name = config.llm["model"]
    caps_path = Path(caps_path_str)

    # Try cache hit
    cached = _load_model_caps_cache(caps_path)
    if cached is not None:
        if (
            cached.get("model_name") == model_name
            and _is_cache_valid(cached, ttl_days)
        ):
            logger.info(
                "model_caps cache hit for %s (checked %s)",
                model_name,
                cached.get("checked_date"),
            )
            return bool(cached["response_format_supported"])

    # Cache miss or expired -- probe
    logger.info("Probing response_format support for model: %s", model_name)
    supported = client.probe_response_format(model_name)

    # Save cache
    _save_model_caps_cache(caps_path, model_name, supported)

    return supported


def _check_notifications(config: AppConfig) -> list[str]:
    """Check 6: Verify notification channels per topic.

    Performs simple format validation for now:
    - Discord: webhook URL pattern
    - Telegram: bot token pattern

    Returns list of warning strings (never raises).
    """
    warnings_list: list[str] = []

    for topic in config.topics:
        notify_items = topic.notify or []
        for notify in notify_items:
            provider = notify.provider

            if provider == "discord":
                if not _DISCORD_WEBHOOK_PATTERN.match(notify.secret_key):
                    warnings_list.append(
                        f"Topic '{topic.slug}': Discord webhook URL "
                        f"format looks invalid"
                    )
            elif provider == "telegram":
                if not _TELEGRAM_TOKEN_PATTERN.match(notify.secret_key):
                    warnings_list.append(
                        f"Topic '{topic.slug}': Telegram bot token "
                        f"format looks invalid"
                    )
            else:
                warnings_list.append(
                    f"Topic '{topic.slug}': Unknown notification "
                    f"provider '{provider}'"
                )

    return warnings_list


def _compute_windows(
    config: AppConfig,
    db: DBManager,
    date_from: datetime | None,
    date_to: datetime | None,
) -> dict[str, tuple[datetime, datetime]]:
    """Check 7: Compute search windows per topic.

    Returns:
        Dict mapping topic slug to (window_start, window_end) tuple.
    """
    result: dict[str, tuple[datetime, datetime]] = {}

    for topic in config.topics:
        start, end = _compute_topic_window(
            topic.slug, db, date_from, date_to
        )
        result[topic.slug] = (start, end)

    return result


def _compute_topic_window(
    topic_slug: str,
    db: DBManager,
    date_from: datetime | None,
    date_to: datetime | None,
) -> tuple[datetime, datetime]:
    """Compute (window_start, window_end) for a single topic.

    Manual mode (date_from/date_to provided):
        Use provided values with +-30min buffer.

    Automatic mode:
        window_end = today UTC 00:00
        window_start lookup chain:
            1. DB: get_latest_completed_run(topic_slug) -> window_end_utc
            2. File: data/last_success.json -> topic's window_end_utc
            3. Fallback: window_end - 72 hours
        Apply +-30min buffer to both sides.
    """
    buffer = timedelta(minutes=_BUFFER_MINUTES)

    if date_from is not None and date_to is not None:
        # Manual mode: apply buffer to both
        return (date_from - buffer, date_to + buffer)

    # Automatic mode: compute window_end as today UTC 00:00
    now_utc = datetime.now(UTC)
    window_end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # If manual date_to provided, use it instead
    if date_to is not None:
        window_end = date_to

    # Compute window_start via lookup chain
    if date_from is not None:
        window_start = date_from
    else:
        window_start = _lookup_window_start(topic_slug, db, window_end)

    return (window_start - buffer, window_end + buffer)


def _lookup_window_start(
    topic_slug: str,
    db: DBManager,
    window_end: datetime,
) -> datetime:
    """Look up window_start for a topic via the priority chain.

    Priority 1: RunMeta(DB) - latest completed run's window_end_utc
    Priority 2: data/last_success.json
    Priority 3: 72-hour fallback (window_end - 72h)
    """
    # Priority 1: DB
    latest_run = db.get_latest_completed_run(topic_slug)
    if latest_run is not None and latest_run.window_end_utc is not None:
        logger.info(
            "Topic '%s': window_start from DB (run_id=%s): %s",
            topic_slug,
            latest_run.run_id,
            latest_run.window_end_utc,
        )
        return latest_run.window_end_utc

    # Priority 2: last_success.json
    last_success = _load_last_success()
    if last_success is not None and topic_slug in last_success:
        topic_data = last_success[topic_slug]
        ts = topic_data.get("last_success_window_end_utc") or topic_data.get("window_end_utc")
        if ts is not None:
            try:
                dt = datetime.fromisoformat(ts)
                # Ensure timezone-aware (assume UTC if naive)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                logger.info(
                    "Topic '%s': window_start from last_success.json: %s",
                    topic_slug,
                    dt,
                )
                return dt
            except (ValueError, TypeError):
                logger.warning(
                    "Topic '%s': invalid timestamp in last_success.json: %s",
                    topic_slug,
                    ts,
                )

    # Priority 3: 72-hour fallback
    fallback = window_end - timedelta(hours=_FALLBACK_HOURS)
    logger.info(
        "Topic '%s': using 72h fallback window_start: %s",
        topic_slug,
        fallback,
    )
    return fallback


def _init_rate_limiter(
    detected_rpm: int, daily_limit: int
) -> RateLimiter:
    """Check 8: Initialize RateLimiter with today's usage."""
    return RateLimiter(
        detected_rpm=detected_rpm,
        daily_limit=daily_limit,
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_model_caps_cache(caps_path: Path) -> dict[str, Any] | None:
    """Load model_caps.json cache. Returns None on any error."""
    try:
        if caps_path.exists():
            return json.loads(caps_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read model_caps cache: %s", caps_path)
    return None


def _save_model_caps_cache(
    caps_path: Path, model_name: str, supported: bool
) -> None:
    """Save model_caps.json cache."""
    data = {
        "model_name": model_name,
        "response_format_supported": supported,
        "checked_date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    try:
        caps_path.parent.mkdir(parents=True, exist_ok=True)
        caps_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Could not write model_caps cache: %s", caps_path)


def _is_cache_valid(cached: dict[str, Any], ttl_days: int) -> bool:
    """Check if a cached entry is still within TTL."""
    checked = cached.get("checked_date")
    if checked is None:
        return False
    try:
        checked_date = datetime.strptime(checked, "%Y-%m-%d").date()
        age = (datetime.now(UTC).date() - checked_date).days
        return age < ttl_days
    except (ValueError, TypeError):
        return False


def _load_last_success() -> dict[str, Any] | None:
    """Load data/last_success.json. Returns None on any error."""
    path = Path(_LAST_SUCCESS_PATH)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read last_success.json")
    return None
