"""Tests for output.notifiers.base (NotifyPayload + NotifierBase).

Covers:
- Message formatting (normal, zero-result, edge-case keyword counts)
- File-size checking (within limit, exceed, missing file)
- Retry logic (1 try + 1 retry, link-only fallback)
- Failure isolation (exceptions never propagate)
- NotifyPayload dataclass field defaults
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from output.notifiers.base import NotifierBase, NotifyPayload


# ---------------------------------------------------------------
# MockNotifier: concrete subclass for testing the ABC
# ---------------------------------------------------------------


class MockNotifier(NotifierBase):
    """Concrete notifier with configurable send behaviour."""

    def __init__(
        self,
        *,
        send_results: Optional[List[bool]] = None,
        send_exceptions: Optional[List[Optional[Exception]]] = None,
        link_only_result: bool = False,
        link_only_exception: Optional[Exception] = None,
        max_file_size: Optional[int] = None,
    ) -> None:
        # Sequence of True/False for successive _send calls.
        self._send_results = list(send_results or [])
        # Sequence of exceptions (or None) for successive _send calls.
        self._send_exceptions = list(send_exceptions or [])
        self._link_only_result = link_only_result
        self._link_only_exception = link_only_exception
        self._send_call_count = 0
        self._link_only_call_count = 0
        self.last_message: Optional[str] = None
        self.last_files: Optional[Dict[str, str]] = None
        if max_file_size is not None:
            self.MAX_FILE_SIZE = max_file_size

    def _send(
        self,
        message: str,
        file_paths: Dict[str, str],
        payload: NotifyPayload,
    ) -> bool:
        self.last_message = message
        self.last_files = file_paths
        idx = self._send_call_count
        self._send_call_count += 1
        if idx < len(self._send_exceptions) and self._send_exceptions[idx]:
            raise self._send_exceptions[idx]  # type: ignore[misc]
        if idx < len(self._send_results):
            return self._send_results[idx]
        return True

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        self._link_only_call_count += 1
        if self._link_only_exception:
            raise self._link_only_exception
        return self._link_only_result


# ---------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------


def _make_payload(
    *,
    keywords: Optional[List[str]] = None,
    total_output: int = 10,
    file_paths: Optional[Dict[str, str]] = None,
    gh_pages_url: Optional[str] = None,
) -> NotifyPayload:
    return NotifyPayload(
        topic_slug="sports-camera",
        topic_name="Sports Camera",
        display_date="26년 02월 10일 화요일",
        keywords=keywords if keywords is not None else [
            "sports camera",
            "highlight detection",
            "pose estimation",
            "action recognition",
            "video analysis",
        ],
        total_output=total_output,
        file_paths=file_paths or {},
        gh_pages_url=gh_pages_url,
    )


# ===============================================================
# 1. Normal message format
# ===============================================================


class TestNormalMessage:
    """DevSpec 11-2: Korean date + top-3 keywords + remaining count."""

    def test_normal_message_top3_plus_remaining(self) -> None:
        """5 keywords -> top 3 shown, '외 2개' suffix."""
        payload = _make_payload(
            keywords=[
                "sports camera",
                "highlight detection",
                "pose estimation",
                "action recognition",
                "video analysis",
            ],
        )
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        expected = (
            '26년 02월 10일 화요일, 오늘의 키워드인 '
            '"sports camera", "highlight detection", '
            '"pose estimation" 외 2개에 대한 '
            'arXiv 논문 정리입니다.'
        )
        assert msg == expected

    def test_message_with_15_keywords(self) -> None:
        """15 keywords -> '외 12개' suffix."""
        kws = [f"kw{i}" for i in range(15)]
        payload = _make_payload(keywords=kws)
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외 12개' in msg

    def test_message_with_many_keywords_20_plus(self) -> None:
        """20+ keywords -> correct remaining count."""
        kws = [f"keyword_{i}" for i in range(25)]
        payload = _make_payload(keywords=kws)
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외 22개' in msg
        # Top-3 present
        assert '"keyword_0"' in msg
        assert '"keyword_1"' in msg
        assert '"keyword_2"' in msg


# ===============================================================
# 2. Keywords <= 3 (no "외 N개" suffix)
# ===============================================================


class TestKeywordsThreeOrFewer:
    def test_exactly_3_keywords(self) -> None:
        payload = _make_payload(
            keywords=["alpha", "beta", "gamma"],
        )
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외' not in msg
        assert '"alpha", "beta", "gamma"' in msg
        assert msg.endswith("arXiv 논문 정리입니다.")

    def test_2_keywords(self) -> None:
        payload = _make_payload(keywords=["alpha", "beta"])
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외' not in msg
        assert '"alpha", "beta"' in msg

    def test_1_keyword(self) -> None:
        payload = _make_payload(keywords=["solo"])
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외' not in msg
        assert '"solo"' in msg
        assert msg.endswith("arXiv 논문 정리입니다.")


# ===============================================================
# 3. Keywords == 0 edge case
# ===============================================================


class TestEmptyKeywords:
    def test_empty_keywords_list_with_nonzero_output(self) -> None:
        """Empty keywords but total_output > 0 -> no keyword text."""
        payload = _make_payload(keywords=[], total_output=5)
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        # Should still produce a message (graceful degradation).
        assert "arXiv 논문 정리입니다." in msg


# ===============================================================
# 4. Zero-result message (DevSpec 11-7)
# ===============================================================


class TestZeroResultMessage:
    def test_zero_total_output(self) -> None:
        payload = _make_payload(total_output=0)
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        expected = (
            "26년 02월 10일 화요일, "
            "오늘은 Sports Camera 관련 신규 논문이 없습니다."
        )
        assert msg == expected


# ===============================================================
# 5-7. File size checks
# ===============================================================


class TestFileSizeCheck:
    def test_file_within_limit_included(self, tmp_path) -> None:
        """File smaller than MAX -> included."""
        f = tmp_path / "small.html"
        f.write_text("hello")
        notifier = MockNotifier()
        result = notifier._check_file_sizes({"html": str(f)})
        assert "html" in result

    def test_file_exceeding_limit_excluded(self, tmp_path) -> None:
        """File bigger than MAX -> excluded."""
        f = tmp_path / "big.html"
        f.write_bytes(b"x" * 100)
        notifier = MockNotifier(max_file_size=50)
        result = notifier._check_file_sizes({"html": str(f)})
        assert "html" not in result

    def test_nonexistent_file_skipped(self) -> None:
        """Missing file -> silently skipped."""
        notifier = MockNotifier()
        result = notifier._check_file_sizes({"html": "/no/such/file.html"})
        assert result == {}


# ===============================================================
# 8. Retry on failure (first fails, second succeeds)
# ===============================================================


class TestRetryLogic:
    def test_first_attempt_fails_second_succeeds(self) -> None:
        notifier = MockNotifier(send_results=[False, True])
        payload = _make_payload()
        assert notifier.notify(payload) is True
        assert notifier._send_call_count == 2

    def test_first_attempt_exception_second_succeeds(self) -> None:
        notifier = MockNotifier(
            send_exceptions=[RuntimeError("network"), None],
            send_results=[False, True],
        )
        payload = _make_payload()
        assert notifier.notify(payload) is True
        assert notifier._send_call_count == 2


# ===============================================================
# 9. Both attempts fail -> returns False
# ===============================================================


class TestBothAttemptsFail:
    def test_both_false_no_gh_url(self) -> None:
        notifier = MockNotifier(send_results=[False, False])
        payload = _make_payload(gh_pages_url=None)
        assert notifier.notify(payload) is False

    def test_both_false_warns(self, caplog) -> None:
        notifier = MockNotifier(send_results=[False, False])
        payload = _make_payload(gh_pages_url=None)
        with caplog.at_level(logging.WARNING):
            notifier.notify(payload)
        assert any(
            "All notification attempts failed" in r.message
            for r in caplog.records
        )


# ===============================================================
# 10. Link-only fallback
# ===============================================================


class TestLinkOnlyFallback:
    def test_fallback_success(self) -> None:
        notifier = MockNotifier(
            send_results=[False, False],
            link_only_result=True,
        )
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html",
        )
        assert notifier.notify(payload) is True
        assert notifier._link_only_call_count == 1

    def test_fallback_not_triggered_without_url(self) -> None:
        notifier = MockNotifier(
            send_results=[False, False],
            link_only_result=True,
        )
        payload = _make_payload(gh_pages_url=None)
        notifier.notify(payload)
        assert notifier._link_only_call_count == 0


# ===============================================================
# 11. Link-only fallback also fails
# ===============================================================


class TestLinkOnlyFallbackAlsoFails:
    def test_all_paths_exhausted(self) -> None:
        notifier = MockNotifier(
            send_results=[False, False],
            link_only_result=False,
            link_only_exception=RuntimeError("total failure"),
        )
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html",
        )
        assert notifier.notify(payload) is False


# ===============================================================
# 12. Exception isolation (DevSpec 11-8)
# ===============================================================


class TestExceptionIsolation:
    def test_send_exception_never_propagates(self) -> None:
        notifier = MockNotifier(
            send_exceptions=[
                RuntimeError("boom1"),
                RuntimeError("boom2"),
            ],
        )
        payload = _make_payload(gh_pages_url=None)
        # Must not raise.
        result = notifier.notify(payload)
        assert result is False

    def test_link_only_exception_never_propagates(self) -> None:
        notifier = MockNotifier(
            send_results=[False, False],
            link_only_exception=RuntimeError("link boom"),
        )
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html",
        )
        result = notifier.notify(payload)
        assert result is False


# ===============================================================
# 13. NotifyPayload dataclass
# ===============================================================


class TestNotifyPayload:
    def test_all_fields_serialize(self) -> None:
        p = NotifyPayload(
            topic_slug="slug",
            topic_name="Name",
            display_date="26년 01월 01일 수요일",
            keywords=["a", "b"],
            total_output=42,
            file_paths={"html": "/tmp/a.html"},
            gh_pages_url="https://example.com",
        )
        assert p.topic_slug == "slug"
        assert p.topic_name == "Name"
        assert p.display_date == "26년 01월 01일 수요일"
        assert p.keywords == ["a", "b"]
        assert p.total_output == 42
        assert p.file_paths == {"html": "/tmp/a.html"}
        assert p.gh_pages_url == "https://example.com"

    def test_defaults(self) -> None:
        p = NotifyPayload(
            topic_slug="s",
            topic_name="N",
            display_date="d",
            keywords=[],
            total_output=0,
        )
        assert p.file_paths == {}
        assert p.gh_pages_url is None


# ===============================================================
# 14. Message with 1 keyword (no "외" suffix)
# ===============================================================


class TestSingleKeyword:
    def test_single_keyword_no_suffix(self) -> None:
        payload = _make_payload(keywords=["only_one"])
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '"only_one"' in msg
        assert '외' not in msg
        assert msg.endswith("arXiv 논문 정리입니다.")


# ===============================================================
# 15. Message with many keywords (20+, correct count)
# ===============================================================


class TestManyKeywords:
    def test_20_plus_keywords_correct_count(self) -> None:
        kws = [f"k{i}" for i in range(23)]
        payload = _make_payload(keywords=kws)
        notifier = MockNotifier()
        msg = notifier.build_message(payload)
        assert '외 20개' in msg
        assert '"k0"' in msg
        assert '"k1"' in msg
        assert '"k2"' in msg
        # 4th keyword should NOT appear.
        assert '"k3"' not in msg


# ===============================================================
# Additional edge-case: notify() passes attachable files to _send
# ===============================================================


class TestNotifyPassesFiles:
    def test_attachable_files_forwarded(self, tmp_path) -> None:
        f = tmp_path / "report.html"
        f.write_text("<html></html>")
        notifier = MockNotifier(send_results=[True])
        payload = _make_payload(file_paths={"html": str(f)})
        notifier.notify(payload)
        assert notifier.last_files == {"html": str(f)}

    def test_oversized_file_not_forwarded(self, tmp_path) -> None:
        f = tmp_path / "big.html"
        f.write_bytes(b"x" * 200)
        notifier = MockNotifier(send_results=[True], max_file_size=100)
        payload = _make_payload(file_paths={"html": str(f)})
        notifier.notify(payload)
        assert notifier.last_files == {}
