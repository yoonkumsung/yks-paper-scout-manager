"""Tests for output.notifiers.error_alert.

Covers:
1. Error alert message format matches devspec exactly
2. Failed topic name included
3. Failed stage included
4. Error cause included
5. Completed topics listed with counts
6. No completed topics: "완료된 토픽: 없음"
7. Multiple completed topics: comma-separated
8. on_error=False: not sent, returns False
9. on_error=True: alert sent via notifier
10. Send failure: returns False, no exception

Section refs: DevSpec 11-9.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from output.notifiers.base import NotifierBase, NotifyPayload
from output.notifiers.error_alert import build_error_alert, send_error_alert


# ---------------------------------------------------------------
# Mock notifier for testing send_error_alert
# ---------------------------------------------------------------


class MockAlertNotifier(NotifierBase):
    """Concrete notifier that records _send calls."""

    def __init__(
        self,
        *,
        send_result: bool = True,
        send_exception: Optional[Exception] = None,
    ) -> None:
        self._send_result = send_result
        self._send_exception = send_exception
        self.send_called = False
        self.send_count = 0
        self.last_message: Optional[str] = None
        self.last_files: Optional[Dict[str, str]] = None

    def _send(
        self,
        message: str,
        file_paths: Dict[str, str],
        payload: NotifyPayload,
    ) -> bool:
        self.send_called = True
        self.send_count += 1
        self.last_message = message
        self.last_files = file_paths
        if self._send_exception:
            raise self._send_exception
        return self._send_result

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        return False


# ---------------------------------------------------------------
# Common test data
# ---------------------------------------------------------------

DISPLAY_DATE = "26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c"
FAILED_TOPIC = "ai-sports-device"
FAILED_STAGE = "Agent 2 \uc810\uc218\ud654"
ERROR_CAUSE = "OpenRouter 429 Too Many Requests"


# ===============================================================
# 1. Error alert message format matches devspec exactly
# ===============================================================


class TestErrorAlertFormat:
    def test_full_format(self) -> None:
        completed = [{"slug": "prompt-engineering", "total_output": 42}]
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, ERROR_CAUSE, completed
        )
        lines = msg.split("\n")
        assert lines[0] == (
            "[Paper Scout \uc624\ub958] "
            "26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c "
            "\uc2e4\ud589 \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4."
        )
        assert lines[1] == "- \uc2e4\ud328 \ud1a0\ud53d: ai-sports-device"
        assert lines[2] == "- \ub2e8\uacc4: Agent 2 \uc810\uc218\ud654"
        assert lines[3] == "- \uc6d0\uc778: OpenRouter 429 Too Many Requests"
        assert lines[4] == "- \uc644\ub8cc\ub41c \ud1a0\ud53d: prompt-engineering (42\ud3b8)"


# ===============================================================
# 2. Failed topic name included
# ===============================================================


class TestFailedTopicIncluded:
    def test_topic_name_in_message(self) -> None:
        msg = build_error_alert(DISPLAY_DATE, "my-topic", FAILED_STAGE, ERROR_CAUSE, [])
        assert "my-topic" in msg
        assert "\uc2e4\ud328 \ud1a0\ud53d: my-topic" in msg


# ===============================================================
# 3. Failed stage included
# ===============================================================


class TestFailedStageIncluded:
    def test_stage_in_message(self) -> None:
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, "Agent 3 \uc694\uc57d", ERROR_CAUSE, []
        )
        assert "\ub2e8\uacc4: Agent 3 \uc694\uc57d" in msg


# ===============================================================
# 4. Error cause included
# ===============================================================


class TestErrorCauseIncluded:
    def test_cause_in_message(self) -> None:
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, "Timeout after 30s", []
        )
        assert "\uc6d0\uc778: Timeout after 30s" in msg


# ===============================================================
# 5. Completed topics listed with counts
# ===============================================================


class TestCompletedTopicsWithCounts:
    def test_single_completed_topic(self) -> None:
        completed = [{"slug": "prompt-engineering", "total_output": 42}]
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, ERROR_CAUSE, completed
        )
        assert "prompt-engineering (42\ud3b8)" in msg


# ===============================================================
# 6. No completed topics: "완료된 토픽: 없음"
# ===============================================================


class TestNoCompletedTopics:
    def test_empty_completed_shows_none(self) -> None:
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, ERROR_CAUSE, []
        )
        assert "\uc644\ub8cc\ub41c \ud1a0\ud53d: \uc5c6\uc74c" in msg


# ===============================================================
# 7. Multiple completed topics: comma-separated
# ===============================================================


class TestMultipleCompletedTopics:
    def test_two_completed_topics(self) -> None:
        completed = [
            {"slug": "prompt-engineering", "total_output": 42},
            {"slug": "llm-safety", "total_output": 15},
        ]
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, ERROR_CAUSE, completed
        )
        assert (
            "prompt-engineering (42\ud3b8), llm-safety (15\ud3b8)"
            in msg
        )

    def test_three_completed_topics(self) -> None:
        completed = [
            {"slug": "topic-a", "total_output": 10},
            {"slug": "topic-b", "total_output": 20},
            {"slug": "topic-c", "total_output": 30},
        ]
        msg = build_error_alert(
            DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE, ERROR_CAUSE, completed
        )
        assert "topic-a (10\ud3b8), topic-b (20\ud3b8), topic-c (30\ud3b8)" in msg


# ===============================================================
# 8. on_error=False: not sent, returns False
# ===============================================================


class TestOnErrorFalse:
    def test_disabled_returns_false(self) -> None:
        notifier = MockAlertNotifier()
        result = send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=False,
        )
        assert result is False
        assert notifier.send_called is False

    def test_disabled_does_not_call_send(self) -> None:
        notifier = MockAlertNotifier()
        send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=False,
        )
        assert notifier.send_count == 0


# ===============================================================
# 9. on_error=True: alert sent via notifier
# ===============================================================


class TestOnErrorTrue:
    def test_enabled_sends_alert(self) -> None:
        notifier = MockAlertNotifier(send_result=True)
        result = send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE,
            [{"slug": "done-topic", "total_output": 5}],
            on_error=True,
        )
        assert result is True
        assert notifier.send_called is True
        assert notifier.send_count == 1

    def test_message_content_correct(self) -> None:
        notifier = MockAlertNotifier(send_result=True)
        send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=True,
        )
        assert "[Paper Scout \uc624\ub958]" in notifier.last_message
        assert FAILED_TOPIC in notifier.last_message

    def test_no_files_attached(self) -> None:
        notifier = MockAlertNotifier(send_result=True)
        send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=True,
        )
        assert notifier.last_files == {}


# ===============================================================
# 10. Send failure: returns False, no exception
# ===============================================================


class TestSendFailure:
    def test_send_returns_false_on_failure(self) -> None:
        notifier = MockAlertNotifier(send_result=False)
        result = send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=True,
        )
        assert result is False

    def test_send_exception_caught(self) -> None:
        notifier = MockAlertNotifier(
            send_exception=RuntimeError("network error"),
        )
        # Must not raise
        result = send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=True,
        )
        assert result is False

    def test_send_exception_does_not_propagate(self) -> None:
        notifier = MockAlertNotifier(
            send_exception=ConnectionError("timeout"),
        )
        # Should return False, never raise
        result = send_error_alert(
            notifier, DISPLAY_DATE, FAILED_TOPIC, FAILED_STAGE,
            ERROR_CAUSE, [], on_error=True,
        )
        assert result is False
