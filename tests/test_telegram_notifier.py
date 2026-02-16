"""Tests for output.notifiers.telegram (TelegramNotifier).

Covers:
- sendMessage call: correct API URL, plain text, no parse_mode.
- sendDocument calls: separate call per file, correct caption.
- File naming: correct basename in multipart.
- No parse_mode: verify parse_mode is NOT in sendMessage params.
- Link-only fallback: message includes gh_pages_url.
- HTTP error on sendMessage: returns False.
- HTTP error on sendDocument: returns False.
- MAX_FILE_SIZE: set to 50MB.
- Empty files dict: only sendMessage, no sendDocument.
- Caption format: "{topic_name} - HTML/MD report".
- Bot token/chat_id validation: empty values raise ValueError.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from output.notifiers.base import NotifyPayload
from output.notifiers.telegram import TelegramNotifier


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
        display_date="26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c",
        keywords=keywords
        if keywords is not None
        else [
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


def _mock_response(ok: bool = True) -> MagicMock:
    resp = MagicMock()
    resp.ok = ok
    return resp


# ===============================================================
# 1. sendMessage call: correct API URL, plain text, no parse_mode
# ===============================================================


class TestSendMessage:
    @patch("output.notifiers.telegram.requests.post")
    def test_send_message_correct_url_and_payload(
        self, mock_post: MagicMock
    ) -> None:
        """sendMessage uses correct API URL and plain text body."""
        mock_post.return_value = _mock_response(ok=True)

        notifier = TelegramNotifier(bot_token="TOKEN123", chat_id="CHAT456")
        payload = _make_payload()
        message = notifier.build_message(payload)
        result = notifier._send(message, {}, payload)

        assert result is True
        mock_post.assert_called_once()

        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/botTOKEN123/sendMessage"
        assert kwargs["json"]["chat_id"] == "CHAT456"
        assert kwargs["json"]["text"] == message

    @patch("output.notifiers.telegram.requests.post")
    def test_no_parse_mode_in_send_message(
        self, mock_post: MagicMock
    ) -> None:
        """parse_mode must NOT be present in sendMessage params."""
        mock_post.return_value = _mock_response(ok=True)

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload()
        message = notifier.build_message(payload)
        notifier._send(message, {}, payload)

        _, kwargs = mock_post.call_args
        assert "parse_mode" not in kwargs["json"]


# ===============================================================
# 2. sendDocument calls: separate call per file, correct caption
# ===============================================================


class TestSendDocument:
    @patch("output.notifiers.telegram.requests.post")
    def test_separate_document_per_file(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Each file should trigger a separate sendDocument call."""
        mock_post.return_value = _mock_response(ok=True)

        html_file = tmp_path / "report.html"
        html_file.write_text("<html></html>")
        md_file = tmp_path / "report.md"
        md_file.write_text("# Report")

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(
            file_paths={"html": str(html_file), "md": str(md_file)}
        )
        message = notifier.build_message(payload)
        result = notifier._send(message, payload.file_paths, payload)

        assert result is True
        # 1 sendMessage + 2 sendDocument = 3 calls
        assert mock_post.call_count == 3

    @patch("output.notifiers.telegram.requests.post")
    def test_caption_format(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Caption should be '{topic_name} - {KEY} report'."""
        mock_post.return_value = _mock_response(ok=True)

        html_file = tmp_path / "report.html"
        html_file.write_text("<html></html>")

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(file_paths={"html": str(html_file)})
        message = notifier.build_message(payload)
        notifier._send(message, payload.file_paths, payload)

        # Second call is sendDocument
        doc_call = mock_post.call_args_list[1]
        assert doc_call[1]["data"]["caption"] == "Sports Camera - HTML report"


# ===============================================================
# 3. File naming: correct basename in multipart
# ===============================================================


class TestFileNaming:
    @patch("output.notifiers.telegram.requests.post")
    def test_correct_basename_in_multipart(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """File should be sent with its basename, not full path."""
        mock_post.return_value = _mock_response(ok=True)

        html_file = tmp_path / "my_report.html"
        html_file.write_text("<html></html>")

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(file_paths={"html": str(html_file)})
        message = notifier.build_message(payload)
        notifier._send(message, payload.file_paths, payload)

        doc_call = mock_post.call_args_list[1]
        files_arg = doc_call[1]["files"]
        filename = files_arg["document"][0]
        assert filename == "my_report.html"


# ===============================================================
# 4. Link-only fallback: message includes gh_pages_url
# ===============================================================


class TestLinkOnlyFallback:
    @patch("output.notifiers.telegram.requests.post")
    def test_link_only_includes_url(
        self, mock_post: MagicMock
    ) -> None:
        """Link-only message should include gh_pages_url."""
        mock_post.return_value = _mock_response(ok=True)

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html"
        )
        message = notifier.build_message(payload)
        result = notifier._send_link_only(message, payload)

        assert result is True
        _, kwargs = mock_post.call_args
        sent_text = kwargs["json"]["text"]
        assert "https://example.github.io/latest.html" in sent_text
        assert message in sent_text

    @patch("output.notifiers.telegram.requests.post")
    def test_link_only_no_parse_mode(
        self, mock_post: MagicMock
    ) -> None:
        """Link-only sendMessage should also have no parse_mode."""
        mock_post.return_value = _mock_response(ok=True)

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html"
        )
        message = notifier.build_message(payload)
        notifier._send_link_only(message, payload)

        _, kwargs = mock_post.call_args
        assert "parse_mode" not in kwargs["json"]


# ===============================================================
# 5. HTTP error on sendMessage: returns False
# ===============================================================


class TestSendMessageError:
    @patch("output.notifiers.telegram.requests.post")
    def test_send_message_failure_returns_false(
        self, mock_post: MagicMock
    ) -> None:
        """If sendMessage HTTP call fails, _send returns False."""
        mock_post.return_value = _mock_response(ok=False)

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload()
        message = notifier.build_message(payload)
        result = notifier._send(message, {}, payload)

        assert result is False


# ===============================================================
# 6. HTTP error on sendDocument: returns False
# ===============================================================


class TestSendDocumentError:
    @patch("output.notifiers.telegram.requests.post")
    def test_send_document_failure_returns_false(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """If sendDocument HTTP call fails, _send returns False."""
        html_file = tmp_path / "report.html"
        html_file.write_text("<html></html>")

        # First call (sendMessage) succeeds, second (sendDocument) fails
        mock_post.side_effect = [
            _mock_response(ok=True),
            _mock_response(ok=False),
        ]

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(file_paths={"html": str(html_file)})
        message = notifier.build_message(payload)
        result = notifier._send(message, payload.file_paths, payload)

        assert result is False


# ===============================================================
# 7. MAX_FILE_SIZE: set to 50MB
# ===============================================================


class TestMaxFileSize:
    def test_max_file_size_is_50mb(self) -> None:
        """TelegramNotifier.MAX_FILE_SIZE should be 50MB."""
        assert TelegramNotifier.MAX_FILE_SIZE == 50 * 1024 * 1024


# ===============================================================
# 8. Empty files dict: only sendMessage, no sendDocument
# ===============================================================


class TestEmptyFilesDict:
    @patch("output.notifiers.telegram.requests.post")
    def test_empty_files_only_send_message(
        self, mock_post: MagicMock
    ) -> None:
        """With no files, only sendMessage is called."""
        mock_post.return_value = _mock_response(ok=True)

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload()
        message = notifier.build_message(payload)
        result = notifier._send(message, {}, payload)

        assert result is True
        assert mock_post.call_count == 1
        args, _ = mock_post.call_args
        assert "/sendMessage" in args[0]


# ===============================================================
# 9. Bot token/chat_id validation: empty values raise ValueError
# ===============================================================


class TestValidation:
    def test_empty_bot_token_raises(self) -> None:
        """Empty bot_token should raise ValueError."""
        with pytest.raises(ValueError, match="bot_token"):
            TelegramNotifier(bot_token="", chat_id="CH")

    def test_empty_chat_id_raises(self) -> None:
        """Empty chat_id should raise ValueError."""
        with pytest.raises(ValueError, match="chat_id"):
            TelegramNotifier(bot_token="TOK", chat_id="")


# ===============================================================
# 10. Integration with base class notify() flow
# ===============================================================


class TestIntegrationWithBase:
    @patch("output.notifiers.telegram.requests.post")
    def test_notify_success_flow(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """Full notify() flow: sendMessage + sendDocument succeed."""
        mock_post.return_value = _mock_response(ok=True)

        html_file = tmp_path / "report.html"
        html_file.write_text("<html></html>")

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(file_paths={"html": str(html_file)})
        result = notifier.notify(payload)

        assert result is True

    @patch("output.notifiers.telegram.requests.post")
    def test_notify_link_only_fallback(
        self, mock_post: MagicMock
    ) -> None:
        """notify() falls back to link-only on sendMessage failure."""
        # First two _send attempts fail, link-only succeeds
        mock_post.side_effect = [
            _mock_response(ok=False),  # attempt 1 sendMessage
            _mock_response(ok=False),  # attempt 2 sendMessage
            _mock_response(ok=True),   # link-only fallback
        ]

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(
            gh_pages_url="https://example.github.io/latest.html"
        )
        result = notifier.notify(payload)

        assert result is True


# ===============================================================
# 11. sendDocument caption for MD format
# ===============================================================


class TestCaptionMD:
    @patch("output.notifiers.telegram.requests.post")
    def test_md_caption_format(
        self, mock_post: MagicMock, tmp_path
    ) -> None:
        """MD file caption should be '{topic_name} - MD report'."""
        mock_post.return_value = _mock_response(ok=True)

        md_file = tmp_path / "report.md"
        md_file.write_text("# Report")

        notifier = TelegramNotifier(bot_token="TOK", chat_id="CH")
        payload = _make_payload(file_paths={"md": str(md_file)})
        message = notifier.build_message(payload)
        notifier._send(message, payload.file_paths, payload)

        doc_call = mock_post.call_args_list[1]
        assert doc_call[1]["data"]["caption"] == "Sports Camera - MD report"
