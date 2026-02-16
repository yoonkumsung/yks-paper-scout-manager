"""Tests for file attachment size policy across notification providers.

Verifies that NotifierBase._check_file_sizes() correctly enforces
per-provider file size limits:
- Discord: 8 MB (8 * 1024 * 1024 bytes)
- Telegram: 50 MB (50 * 1024 * 1024 bytes)

Also verifies link-only fallback when all files exceed the limit,
mixed scenarios, and retry fallback behavior.

Section refs: DevSpec 11-1, 11-3.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pytest

from output.notifiers.base import NotifierBase, NotifyPayload


# ---------------------------------------------------------------
# Mock notifiers with provider-specific MAX_FILE_SIZE
# ---------------------------------------------------------------


class _BaseMockNotifier(NotifierBase):
    """Base mock with tracking for _send / _send_link_only calls."""

    def __init__(
        self,
        *,
        send_results: Optional[List[bool]] = None,
        link_only_result: bool = False,
    ) -> None:
        self._send_results = list(send_results or [True])
        self._link_only_result = link_only_result
        self._send_call_count = 0
        self._link_only_call_count = 0
        self.last_message: Optional[str] = None
        self.last_files: Optional[Dict[str, str]] = None

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
        if idx < len(self._send_results):
            return self._send_results[idx]
        return True

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        self._link_only_call_count += 1
        return self._link_only_result


class DiscordMockNotifier(_BaseMockNotifier):
    """Mock notifier with Discord 8 MB file size limit."""

    MAX_FILE_SIZE: int = 8 * 1024 * 1024  # 8 MB


class TelegramMockNotifier(_BaseMockNotifier):
    """Mock notifier with Telegram 50 MB file size limit."""

    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------


def _make_payload(
    *,
    file_paths: Optional[Dict[str, str]] = None,
    gh_pages_url: Optional[str] = None,
) -> NotifyPayload:
    return NotifyPayload(
        topic_slug="test-topic",
        topic_name="Test Topic",
        display_date="26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c",
        keywords=["kw1", "kw2", "kw3"],
        total_output=5,
        file_paths=file_paths or {},
        gh_pages_url=gh_pages_url,
    )


def _create_file(tmp_path, name: str, size_bytes: int) -> str:
    """Create a file of exactly *size_bytes* at tmp_path/name."""
    f = tmp_path / name
    f.write_bytes(b"\x00" * size_bytes)
    return str(f)


# ===============================================================
# 1. Files within Discord 8 MB limit pass
# ===============================================================


class TestDiscordWithinLimit:
    def test_small_file_passes(self, tmp_path) -> None:
        path = _create_file(tmp_path, "report.html", 1024)  # 1 KB
        notifier = DiscordMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" in result
        assert result["html"] == path

    def test_file_exactly_at_limit_passes(self, tmp_path) -> None:
        limit = 8 * 1024 * 1024
        path = _create_file(tmp_path, "exact.html", limit)
        notifier = DiscordMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" in result


# ===============================================================
# 2. Files exceeding 8 MB are excluded for Discord
# ===============================================================


class TestDiscordExceedLimit:
    def test_file_one_byte_over_excluded(self, tmp_path) -> None:
        limit = 8 * 1024 * 1024
        path = _create_file(tmp_path, "big.html", limit + 1)
        notifier = DiscordMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" not in result

    def test_large_file_excluded(self, tmp_path) -> None:
        path = _create_file(tmp_path, "huge.html", 10 * 1024 * 1024)  # 10 MB
        notifier = DiscordMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert result == {}


# ===============================================================
# 3. Files within Telegram 50 MB limit pass
# ===============================================================


class TestTelegramWithinLimit:
    def test_small_file_passes(self, tmp_path) -> None:
        path = _create_file(tmp_path, "report.html", 1024)
        notifier = TelegramMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" in result

    def test_file_at_9mb_passes_telegram(self, tmp_path) -> None:
        """9 MB file passes Telegram (50 MB limit) but would fail Discord."""
        path = _create_file(tmp_path, "medium.html", 9 * 1024 * 1024)
        notifier = TelegramMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" in result

    def test_file_exactly_at_limit_passes(self, tmp_path) -> None:
        limit = 50 * 1024 * 1024
        path = _create_file(tmp_path, "exact.html", limit)
        notifier = TelegramMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" in result


# ===============================================================
# 4. Files exceeding 50 MB are excluded for Telegram
# ===============================================================


class TestTelegramExceedLimit:
    def test_file_one_byte_over_excluded(self, tmp_path) -> None:
        limit = 50 * 1024 * 1024
        path = _create_file(tmp_path, "big.html", limit + 1)
        notifier = TelegramMockNotifier()
        result = notifier._check_file_sizes({"html": path})
        assert "html" not in result


# ===============================================================
# 5. Auto link-only fallback when all files exceed limit
# ===============================================================


class TestAutoLinkOnlyFallback:
    def test_all_files_over_limit_empty_attachable(self, tmp_path) -> None:
        """When all files exceed limit, attachable set is empty."""
        path1 = _create_file(tmp_path, "big1.html", 100)
        path2 = _create_file(tmp_path, "big2.md", 200)
        notifier = _BaseMockNotifier()
        notifier.MAX_FILE_SIZE = 50  # Very small limit
        result = notifier._check_file_sizes({"html": path1, "md": path2})
        assert result == {}

    def test_notify_sends_empty_files_when_all_oversized(self, tmp_path) -> None:
        """notify() still calls _send with empty file dict."""
        path = _create_file(tmp_path, "big.html", 200)
        notifier = DiscordMockNotifier(send_results=[True])
        notifier.MAX_FILE_SIZE = 50
        payload = _make_payload(file_paths={"html": str(path)})
        result = notifier.notify(payload)
        assert result is True
        assert notifier.last_files == {}


# ===============================================================
# 6. Mixed: some files pass, some don't
# ===============================================================


class TestMixedFiles:
    def test_small_passes_large_excluded(self, tmp_path) -> None:
        small = _create_file(tmp_path, "small.html", 100)
        large = _create_file(tmp_path, "large.md", 500)
        notifier = _BaseMockNotifier()
        notifier.MAX_FILE_SIZE = 200
        result = notifier._check_file_sizes({"html": small, "md": large})
        assert "html" in result
        assert "md" not in result

    def test_three_files_two_pass_one_fails(self, tmp_path) -> None:
        f1 = _create_file(tmp_path, "a.html", 50)
        f2 = _create_file(tmp_path, "b.md", 50)
        f3 = _create_file(tmp_path, "c.pdf", 300)
        notifier = _BaseMockNotifier()
        notifier.MAX_FILE_SIZE = 100
        result = notifier._check_file_sizes(
            {"html": f1, "md": f2, "pdf": f3}
        )
        assert "html" in result
        assert "md" in result
        assert "pdf" not in result


# ===============================================================
# 7. Retry triggers link-only fallback on send failure
# ===============================================================


class TestRetryLinkOnlyFallback:
    def test_send_fails_twice_triggers_link_only(self, tmp_path) -> None:
        """When _send fails twice and gh_pages_url is set,
        link-only fallback is attempted."""
        path = _create_file(tmp_path, "report.html", 100)
        notifier = DiscordMockNotifier(
            send_results=[False, False],
            link_only_result=True,
        )
        payload = _make_payload(
            file_paths={"html": str(path)},
            gh_pages_url="https://example.github.io/latest.html",
        )
        result = notifier.notify(payload)
        assert result is True
        assert notifier._link_only_call_count == 1

    def test_send_fails_twice_no_url_returns_false(self, tmp_path) -> None:
        """When _send fails twice without gh_pages_url, returns False."""
        path = _create_file(tmp_path, "report.html", 100)
        notifier = DiscordMockNotifier(send_results=[False, False])
        payload = _make_payload(
            file_paths={"html": str(path)},
            gh_pages_url=None,
        )
        result = notifier.notify(payload)
        assert result is False
        assert notifier._link_only_call_count == 0
