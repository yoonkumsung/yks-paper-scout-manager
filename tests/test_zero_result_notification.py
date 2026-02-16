"""Tests for zero-result notification flow.

Verifies that the full notification pipeline handles 0 papers correctly:
1. Zero-result message format correct
2. 0-result report files still attached (file_paths populated)
3. Notify still called with 0 papers
4. latest.html still in file_paths for 0-result

The base class already generates:
  "{date}, 오늘은 {topic_name} 관련 신규 논문이 없습니다."

Section refs: DevSpec 11-7.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from output.notifiers.base import NotifierBase, NotifyPayload


# ---------------------------------------------------------------
# Mock notifier for verifying full flow
# ---------------------------------------------------------------


class MockNotifier(NotifierBase):
    """Concrete notifier that records all calls for assertion."""

    def __init__(
        self,
        *,
        send_result: bool = True,
    ) -> None:
        self._send_result = send_result
        self.send_called = False
        self.send_count = 0
        self.last_message: Optional[str] = None
        self.last_files: Optional[Dict[str, str]] = None
        self.last_payload: Optional[NotifyPayload] = None

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
        self.last_payload = payload
        return self._send_result

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        return False


# ---------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------


def _make_zero_payload(
    *,
    file_paths: Optional[Dict[str, str]] = None,
    topic_name: str = "AI Sports Device",
) -> NotifyPayload:
    return NotifyPayload(
        topic_slug="ai-sports-device",
        topic_name=topic_name,
        display_date="26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c",
        keywords=["sports", "device", "wearable"],
        total_output=0,
        file_paths=file_paths or {},
        gh_pages_url=None,
    )


# ===============================================================
# 1. Zero-result message format correct
# ===============================================================


class TestZeroResultMessageFormat:
    def test_zero_result_message_exact(self) -> None:
        payload = _make_zero_payload()
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        expected = (
            "26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c, "
            "\uc624\ub298\uc740 AI Sports Device \uad00\ub828 "
            "\uc2e0\uaddc \ub17c\ubb38\uc774 \uc5c6\uc2b5\ub2c8\ub2e4."
        )
        assert msg == expected

    def test_zero_result_uses_topic_name(self) -> None:
        payload = _make_zero_payload(topic_name="Prompt Engineering")
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert "Prompt Engineering" in msg
        assert "\uad00\ub828 \uc2e0\uaddc \ub17c\ubb38\uc774 \uc5c6\uc2b5\ub2c8\ub2e4." in msg

    def test_zero_result_does_not_contain_keyword_text(self) -> None:
        payload = _make_zero_payload()
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        # Zero-result message should NOT contain keyword section
        assert "arXiv" not in msg
        assert "\ud0a4\uc6cc\ub4dc" not in msg


# ===============================================================
# 2. 0-result report files still attached
# ===============================================================


class TestZeroResultFilesAttached:
    def test_files_passed_to_send_with_zero_papers(self, tmp_path) -> None:
        """Even with 0 papers, file_paths in payload are forwarded."""
        html_file = tmp_path / "latest.html"
        html_file.write_text("<html>empty report</html>")
        md_file = tmp_path / "latest.md"
        md_file.write_text("# Empty report")

        payload = _make_zero_payload(
            file_paths={
                "html": str(html_file),
                "md": str(md_file),
            }
        )
        notifier = MockNotifier()
        notifier.notify(payload)

        assert notifier.last_files is not None
        assert "html" in notifier.last_files
        assert "md" in notifier.last_files


# ===============================================================
# 3. Notify still called with 0 papers
# ===============================================================


class TestNotifyCalledWithZero:
    def test_send_invoked_for_zero_result(self) -> None:
        """_send() is called even when total_output=0."""
        payload = _make_zero_payload()
        notifier = MockNotifier()
        result = notifier.notify(payload)
        assert result is True
        assert notifier.send_called is True
        assert notifier.send_count == 1

    def test_notify_returns_true_on_success(self) -> None:
        payload = _make_zero_payload()
        notifier = MockNotifier(send_result=True)
        assert notifier.notify(payload) is True

    def test_notify_returns_false_on_failure(self) -> None:
        payload = _make_zero_payload()
        notifier = MockNotifier(send_result=False)
        assert notifier.notify(payload) is False


# ===============================================================
# 4. latest.html still in file_paths for 0-result
# ===============================================================


class TestLatestHtmlInZeroResult:
    def test_latest_html_forwarded(self, tmp_path) -> None:
        """latest.html should be in the attachable files for zero-result."""
        html_file = tmp_path / "latest.html"
        html_file.write_text("<html>no papers today</html>")

        payload = _make_zero_payload(
            file_paths={"html": str(html_file)},
        )
        notifier = MockNotifier()
        notifier.notify(payload)

        assert notifier.last_files is not None
        assert "html" in notifier.last_files
        assert notifier.last_files["html"] == str(html_file)

    def test_zero_result_message_sent_with_html_file(self, tmp_path) -> None:
        """Verify the message is zero-result AND file is attached."""
        html_file = tmp_path / "latest.html"
        html_file.write_text("<html>no papers</html>")

        payload = _make_zero_payload(
            file_paths={"html": str(html_file)},
        )
        notifier = MockNotifier()
        notifier.notify(payload)

        # Message should be zero-result format
        assert "\uc2e0\uaddc \ub17c\ubb38\uc774 \uc5c6\uc2b5\ub2c8\ub2e4." in notifier.last_message
        # File should still be attached
        assert "html" in notifier.last_files
