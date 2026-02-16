"""Telegram Bot API notifier.

Sends notifications via Telegram Bot API:
- sendMessage: plain text, NO parse_mode (prevents formatting injection).
- sendDocument: HTML file, MD file SEPARATELY (2 calls).
- caption: file description per document.
- No paper titles/external text in message body.

Section refs: DevSpec 11-4.
"""

from __future__ import annotations

import os
from typing import Dict

import requests

from output.notifiers.base import NotifierBase, NotifyPayload


class TelegramNotifier(NotifierBase):
    """Telegram Bot API notifier."""

    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB Telegram limit

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token:
            raise ValueError("bot_token must not be empty")
        if not chat_id:
            raise ValueError("chat_id must not be empty")
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._api_base = f"https://api.telegram.org/bot{bot_token}"

    def _send(
        self,
        message: str,
        file_paths: Dict[str, str],
        payload: NotifyPayload,
    ) -> bool:
        """Send via Telegram Bot API.

        1. sendMessage with plain text (no parse_mode).
        2. For each file: sendDocument with caption.
        """
        # Step 1: Send text message
        resp = requests.post(
            f"{self._api_base}/sendMessage",
            json={
                "chat_id": self._chat_id,
                "text": message,
                # NO parse_mode - plain text only
            },
        )
        if not resp.ok:
            return False

        # Step 2: Send each file as a separate document
        for key, path in file_paths.items():
            caption = f"{payload.topic_name} - {key.upper()} report"
            with open(path, "rb") as f:
                resp = requests.post(
                    f"{self._api_base}/sendDocument",
                    data={"chat_id": self._chat_id, "caption": caption},
                    files={"document": (os.path.basename(path), f)},
                )
                if not resp.ok:
                    return False

        return True

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        """Send text-only with link to gh-pages."""
        text = f"{message}\n\nReport: {payload.gh_pages_url}"
        resp = requests.post(
            f"{self._api_base}/sendMessage",
            json={
                "chat_id": self._chat_id,
                "text": text,
            },
        )
        return resp.ok
