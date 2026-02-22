"""Error alert formatting and delivery for Paper Scout.

Builds and sends error alert messages when the pipeline encounters
failures.  Format follows DevSpec 11-9.

Example output::

    [Paper Scout] ai-sports-device 논문 수집 실패

    📅 26년 02월 10일 화요일
    ❌ 단계: Agent 2 점수화
    ❌ 원인: OpenRouter 429 Too Many Requests
"""

from __future__ import annotations

import html
import logging
from typing import Dict, List

from output.notifiers.base import NotifierBase, NotifyPayload

logger = logging.getLogger(__name__)


def build_error_alert(
    display_date: str,
    failed_topic: str,
    failed_stage: str,
    error_cause: str,
    completed_topics: List[Dict[str, object]],
) -> str:
    """Build error alert message per DevSpec 11-9.

    Args:
        display_date: Pre-formatted Korean date string.
        failed_topic: Slug of the topic that failed.
        failed_stage: Pipeline stage where the failure occurred.
        error_cause: Human-readable error description.
        completed_topics: List of dicts with "slug" and "total_output" keys
            for topics that completed successfully before the failure.

    Returns:
        Formatted error alert message string.
    """
    topic = html.escape(failed_topic)
    lines = [
        f"[Paper Scout] {topic} 논문 수집 실패",
        "",
        f"❌ 단계: {html.escape(failed_stage)}",
        f"❌ 원인: {html.escape(error_cause)}",
    ]

    return "\n".join(lines)


def send_error_alert(
    notifier: NotifierBase,
    display_date: str,
    failed_topic: str,
    failed_stage: str,
    error_cause: str,
    completed_topics: List[Dict[str, object]],
    on_error: bool = True,
) -> bool:
    """Send error alert if on_error is True.

    Uses the provided notifier (first topic's channel).

    Args:
        notifier: Configured notifier instance to send through.
        display_date: Pre-formatted Korean date string.
        failed_topic: Slug of the topic that failed.
        failed_stage: Pipeline stage where the failure occurred.
        error_cause: Human-readable error description.
        completed_topics: Successfully completed topic info.
        on_error: If False, skip sending and return False.

    Returns:
        True if sent successfully, False if disabled or failed.
    """
    if not on_error:
        return False

    message_text = build_error_alert(
        display_date,
        failed_topic,
        failed_stage,
        error_cause,
        completed_topics,
    )

    # Create a minimal payload for the notifier.
    payload = NotifyPayload(
        topic_slug=failed_topic,
        topic_name=failed_topic,
        display_date=display_date,
        keywords=[],
        total_output=0,
        file_paths={},
    )

    try:
        return notifier._send(message_text, {}, payload)
    except Exception:
        logger.warning(
            "Failed to send error alert for topic %s",
            failed_topic,
            exc_info=True,
        )
        return False
