"""Discord webhook notifier for Paper Scout.

Sends notifications via Discord webhook using multipart/form-data with:
- ``payload_json``: message content with mention suppression and embed suppression
- ``files[N]``: file attachments (HTML/Markdown reports)

Section refs: DevSpec 11-5.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import requests

from output.notifiers.base import NotifierBase, NotifyPayload

logger = logging.getLogger(__name__)


class DiscordNotifier(NotifierBase):
    """Discord webhook notifier.

    Uses Discord webhook API with multipart/form-data encoding.

    Attributes:
        MAX_FILE_SIZE: Discord attachment limit (8 MB).
    """

    MAX_FILE_SIZE: int = 8 * 1024 * 1024  # 8 MB Discord limit

    def __init__(self, webhook_url: str) -> None:
        """Initialize with a Discord webhook URL.

        Args:
            webhook_url: Full Discord webhook URL.

        Raises:
            ValueError: If *webhook_url* is empty or blank.
        """
        if not webhook_url or not webhook_url.strip():
            raise ValueError("webhook_url must not be empty")
        self._webhook_url = webhook_url.strip()

    # ------------------------------------------------------------------
    # Provider hooks
    # ------------------------------------------------------------------

    def _send(
        self,
        message: str,
        file_paths: Dict[str, str],
        payload: NotifyPayload,
    ) -> bool:
        """Send via Discord webhook with optional file attachments.

        Uses multipart/form-data with:
        - ``payload_json``: JSON string containing *content*,
          *allowed_mentions* (parse=[]), and *flags* (4 = SUPPRESS_EMBEDS).
        - ``files[0]``, ``files[1]``, ...: binary file attachments.

        Query parameter ``?wait=true`` ensures Discord confirms delivery.

        Returns:
            ``True`` on HTTP 2xx, ``False`` otherwise.
        """
        payload_json = json.dumps(
            {
                "content": message,
                "allowed_mentions": {"parse": []},
                "flags": 4,
            }
        )

        url = self._webhook_url.rstrip("/") + "?wait=true"

        # Build multipart fields.
        files_to_send: Dict[str, tuple] = {
            "payload_json": (None, payload_json, "application/json"),
        }

        opened_files = []
        try:
            for idx, (key, path) in enumerate(file_paths.items()):
                fh = open(path, "rb")  # noqa: SIM115
                opened_files.append(fh)
                filename = os.path.basename(path)
                files_to_send[f"files[{idx}]"] = (
                    filename,
                    fh,
                    "application/octet-stream",
                )

            response = requests.post(url, files=files_to_send, timeout=30)
        finally:
            for fh in opened_files:
                fh.close()

        if response.ok:
            return True

        logger.warning(
            "Discord webhook returned HTTP %d: %s",
            response.status_code,
            response.text[:200],
        )
        return False

    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        """Send text-only message with gh-pages link.

        Falls back to a simple JSON POST (no files) when file delivery
        has failed.  Includes the ``gh_pages_url`` in the message body.

        Returns:
            ``True`` on HTTP 2xx, ``False`` otherwise.
        """
        link_message = message
        if payload.gh_pages_url:
            link_message = f"{message}\n{payload.gh_pages_url}"

        payload_json = json.dumps(
            {
                "content": link_message,
                "allowed_mentions": {"parse": []},
                "flags": 4,
            }
        )

        url = self._webhook_url.rstrip("/") + "?wait=true"

        response = requests.post(
            url,
            files={
                "payload_json": (None, payload_json, "application/json"),
            },
            timeout=30,
        )

        if response.ok:
            return True

        logger.warning(
            "Discord link-only webhook returned HTTP %d: %s",
            response.status_code,
            response.text[:200],
        )
        return False
