"""Tests for output.notifiers.discord (DiscordNotifier).

Covers:
- Successful send with file attachments (multipart structure)
- Successful send without files (text-only)
- allowed_mentions: {"parse": []} in payload_json
- flags: 4 (SUPPRESS_EMBEDS) in payload_json
- wait=true query parameter on webhook URL
- Link-only fallback with gh_pages_url
- HTTP error handling (non-2xx returns False)
- MAX_FILE_SIZE constant (8 MB)
- File attachment field names (files[0], files[1])
- Webhook URL validation (empty raises ValueError)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from output.notifiers.base import NotifyPayload
from output.notifiers.discord import DiscordNotifier


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_payload(
    *,
    file_paths: Optional[Dict[str, str]] = None,
    gh_pages_url: Optional[str] = None,
    total_output: int = 5,
) -> NotifyPayload:
    return NotifyPayload(
        topic_slug="ml-vision",
        topic_name="ML Vision",
        display_date="26\ub144 02\uc6d4 16\uc77c \uc77c\uc694\uc77c",
        keywords=["object detection", "segmentation", "tracking"],
        total_output=total_output,
        file_paths=file_paths or {},
        gh_pages_url=gh_pages_url,
    )


def _mock_response(status_code: int = 200, ok: bool = True) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.text = "OK"
    return resp


def _extract_payload_json(mock_post: MagicMock) -> Dict[str, Any]:
    """Extract and parse the payload_json from a mocked requests.post call."""
    _, kwargs = mock_post.call_args
    files_arg = kwargs.get("files", {})
    # payload_json is a tuple (None, json_string, content_type)
    payload_tuple = files_arg["payload_json"]
    return json.loads(payload_tuple[1])


# ===============================================================
# 1. Successful send with files
# ===============================================================


class TestSendWithFiles:
    @patch("output.notifiers.discord.requests.post")
    def test_files_attached_correct_multipart_structure(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Files are attached with correct multipart field names."""
        f1 = tmp_path / "report.html"
        f1.write_text("<html>report</html>")
        f2 = tmp_path / "report.md"
        f2.write_text("# Report")

        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send(
            "test message",
            {"html": str(f1), "md": str(f2)},
            _make_payload(file_paths={"html": str(f1), "md": str(f2)}),
        )

        assert result is True
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        files_arg = kwargs["files"]
        assert "payload_json" in files_arg
        assert "files[0]" in files_arg
        assert "files[1]" in files_arg


# ===============================================================
# 2. Successful send without files
# ===============================================================


class TestSendWithoutFiles:
    @patch("output.notifiers.discord.requests.post")
    def test_text_only_when_no_files(self, mock_post: MagicMock) -> None:
        """When no files, only payload_json is sent."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send(
            "test message",
            {},
            _make_payload(),
        )

        assert result is True
        _, kwargs = mock_post.call_args
        files_arg = kwargs["files"]
        assert "payload_json" in files_arg
        # No files[N] keys
        file_keys = [k for k in files_arg if k.startswith("files[")]
        assert file_keys == []


# ===============================================================
# 3. allowed_mentions: {"parse": []} in payload_json
# ===============================================================


class TestAllowedMentions:
    @patch("output.notifiers.discord.requests.post")
    def test_parse_empty_blocks_all_mentions(
        self, mock_post: MagicMock
    ) -> None:
        """payload_json includes allowed_mentions with empty parse list."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send("msg", {}, _make_payload())

        pj = _extract_payload_json(mock_post)
        assert pj["allowed_mentions"] == {"parse": []}


# ===============================================================
# 4. flags: 4 (SUPPRESS_EMBEDS) in payload_json
# ===============================================================


class TestSuppressEmbeds:
    @patch("output.notifiers.discord.requests.post")
    def test_flags_suppress_embeds(self, mock_post: MagicMock) -> None:
        """payload_json includes flags=4 for SUPPRESS_EMBEDS."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send("msg", {}, _make_payload())

        pj = _extract_payload_json(mock_post)
        assert pj["flags"] == 4


# ===============================================================
# 5. wait=true query parameter
# ===============================================================


class TestWaitParam:
    @patch("output.notifiers.discord.requests.post")
    def test_wait_true_in_url(self, mock_post: MagicMock) -> None:
        """Webhook URL includes ?wait=true query parameter."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send("msg", {}, _make_payload())

        called_url = mock_post.call_args[0][0] if mock_post.call_args[0] else mock_post.call_args[1].get("url", "")
        # positional arg
        if not called_url:
            called_url = mock_post.call_args[0][0]
        assert "?wait=true" in called_url

    @patch("output.notifiers.discord.requests.post")
    def test_wait_true_in_link_only_url(self, mock_post: MagicMock) -> None:
        """Link-only fallback also uses ?wait=true."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send_link_only(
            "msg",
            _make_payload(gh_pages_url="https://example.github.io/report"),
        )

        called_url = mock_post.call_args[0][0]
        assert "?wait=true" in called_url


# ===============================================================
# 6. Link-only fallback includes gh_pages_url
# ===============================================================


class TestLinkOnlyFallback:
    @patch("output.notifiers.discord.requests.post")
    def test_message_includes_gh_pages_url(
        self, mock_post: MagicMock
    ) -> None:
        """Link-only message appends gh_pages_url."""
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        gh_url = "https://example.github.io/report.html"
        notifier._send_link_only(
            "base message",
            _make_payload(gh_pages_url=gh_url),
        )

        pj = _extract_payload_json(mock_post)
        assert gh_url in pj["content"]
        assert "base message" in pj["content"]

    @patch("output.notifiers.discord.requests.post")
    def test_link_only_returns_true_on_success(
        self, mock_post: MagicMock
    ) -> None:
        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send_link_only(
            "msg",
            _make_payload(gh_pages_url="https://example.com"),
        )
        assert result is True


# ===============================================================
# 7. HTTP error handling (non-2xx returns False)
# ===============================================================


class TestHttpErrorHandling:
    @patch("output.notifiers.discord.requests.post")
    def test_non_2xx_returns_false(self, mock_post: MagicMock) -> None:
        """Non-2xx response from Discord returns False."""
        mock_post.return_value = _mock_response(status_code=400, ok=False)
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send("msg", {}, _make_payload())
        assert result is False

    @patch("output.notifiers.discord.requests.post")
    def test_rate_limit_429_returns_false(self, mock_post: MagicMock) -> None:
        """429 rate limit returns False."""
        mock_post.return_value = _mock_response(status_code=429, ok=False)
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send("msg", {}, _make_payload())
        assert result is False

    @patch("output.notifiers.discord.requests.post")
    def test_server_error_500_returns_false(
        self, mock_post: MagicMock
    ) -> None:
        """500 server error returns False."""
        mock_post.return_value = _mock_response(status_code=500, ok=False)
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send("msg", {}, _make_payload())
        assert result is False

    @patch("output.notifiers.discord.requests.post")
    def test_link_only_non_2xx_returns_false(
        self, mock_post: MagicMock
    ) -> None:
        """Link-only fallback also returns False on non-2xx."""
        mock_post.return_value = _mock_response(status_code=403, ok=False)
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        result = notifier._send_link_only("msg", _make_payload())
        assert result is False


# ===============================================================
# 8. MAX_FILE_SIZE constant
# ===============================================================


class TestMaxFileSize:
    def test_max_file_size_is_8mb(self) -> None:
        """MAX_FILE_SIZE should be 8 MB (8 * 1024 * 1024 bytes)."""
        assert DiscordNotifier.MAX_FILE_SIZE == 8 * 1024 * 1024

    def test_instance_inherits_max_file_size(self) -> None:
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")
        assert notifier.MAX_FILE_SIZE == 8 * 1024 * 1024


# ===============================================================
# 9. File attachment format (files[0], files[1])
# ===============================================================


class TestFileAttachmentFormat:
    @patch("output.notifiers.discord.requests.post")
    def test_file_field_names_indexed(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Files use files[0], files[1], ... field names."""
        f1 = tmp_path / "a.html"
        f1.write_text("html")
        f2 = tmp_path / "b.md"
        f2.write_text("md")
        f3 = tmp_path / "c.txt"
        f3.write_text("txt")

        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send(
            "msg",
            {"html": str(f1), "md": str(f2), "txt": str(f3)},
            _make_payload(),
        )

        _, kwargs = mock_post.call_args
        files_arg = kwargs["files"]
        assert "files[0]" in files_arg
        assert "files[1]" in files_arg
        assert "files[2]" in files_arg

    @patch("output.notifiers.discord.requests.post")
    def test_filename_is_basename(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Attached file uses basename, not full path."""
        f = tmp_path / "deep" / "nested" / "report.html"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("<html></html>")

        mock_post.return_value = _mock_response()
        notifier = DiscordNotifier("https://discord.com/api/webhooks/123/abc")

        notifier._send("msg", {"html": str(f)}, _make_payload())

        _, kwargs = mock_post.call_args
        file_tuple = kwargs["files"]["files[0]"]
        assert file_tuple[0] == "report.html"


# ===============================================================
# 10. Webhook URL validation
# ===============================================================


class TestWebhookUrlValidation:
    def test_empty_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            DiscordNotifier("")

    def test_blank_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            DiscordNotifier("   ")

    def test_valid_url_accepted(self) -> None:
        notifier = DiscordNotifier(
            "https://discord.com/api/webhooks/123/abc"
        )
        assert notifier._webhook_url == (
            "https://discord.com/api/webhooks/123/abc"
        )

    def test_url_stripped(self) -> None:
        notifier = DiscordNotifier(
            "  https://discord.com/api/webhooks/123/abc  "
        )
        assert notifier._webhook_url == (
            "https://discord.com/api/webhooks/123/abc"
        )
