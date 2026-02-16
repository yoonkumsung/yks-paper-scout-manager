"""Abstract base class for notification providers and message formatting.

Provides:
- NotifyPayload dataclass with all fields required for notification.
- NotifierBase ABC with retry logic, file-size checking, and failure
  isolation (notification failures NEVER propagate as pipeline failures).

Section refs: DevSpec 11-1, 11-2, 11-3, 11-7, 11-8.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NotifyPayload:
    """Data needed for sending a notification.

    Attributes:
        topic_slug: URL-safe topic identifier.
        topic_name: Human-readable topic name (used in zero-result msg).
        display_date: Pre-formatted Korean date, e.g. "26년 02월 10일 화요일".
        keywords: All keyword ``name_en`` values from Agent 1 concepts.
        total_output: Number of papers collected.  0 triggers zero-result
            message (DevSpec 11-7).
        file_paths: Mapping of format key to absolute path,
            e.g. ``{"html": "/tmp/out.html", "md": "/tmp/out.md"}``.
        gh_pages_url: Optional GitHub Pages URL for link-only fallback.
    """

    topic_slug: str
    topic_name: str
    display_date: str
    keywords: List[str]
    total_output: int
    file_paths: Dict[str, str] = field(default_factory=dict)
    gh_pages_url: Optional[str] = None


class NotifierBase(ABC):
    """Abstract base for notification providers.

    Contract (DevSpec 11-1, 11-3, 11-8):
    - ``notify()`` attempts 1 delivery.
    - On failure: 1 retry (total 2 attempts).
    - After 2 failures: link-only fallback if ``gh_pages_url`` available.
    - Final failure: warning log only -- **never** raises to caller.
    """

    # Per-provider file-size ceiling in bytes.
    # Subclasses should override (e.g. Discord 8 MB, Telegram 50 MB).
    MAX_FILE_SIZE: int = 8 * 1024 * 1024  # 8 MB default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, payload: NotifyPayload) -> bool:
        """Send notification with retry and failure isolation.

        Returns ``True`` on success, ``False`` on failure (warning logged).
        **Never** raises exceptions to caller (DevSpec 11-8).
        """
        message = self.build_message(payload)

        # Determine which files to attach (pre-send size check).
        attachable_files = self._check_file_sizes(payload.file_paths)

        for attempt in range(1, 3):  # 1 try + 1 retry
            try:
                success = self._send(message, attachable_files, payload)
                if success:
                    return True
                logger.warning(
                    "Notification send returned False (attempt %d/2) for %s",
                    attempt,
                    payload.topic_slug,
                )
            except Exception as exc:
                logger.warning(
                    "Notification error (attempt %d/2) for %s: %s",
                    attempt,
                    payload.topic_slug,
                    exc,
                )

        # Both attempts failed -> try link-only fallback.
        if payload.gh_pages_url:
            try:
                return self._send_link_only(message, payload)
            except Exception as exc:
                logger.warning(
                    "Link-only fallback also failed for %s: %s",
                    payload.topic_slug,
                    exc,
                )

        logger.warning(
            "All notification attempts failed for %s",
            payload.topic_slug,
        )
        return False

    # ------------------------------------------------------------------
    # Message building  (DevSpec 11-2, 11-7)
    # ------------------------------------------------------------------

    def build_message(self, payload: NotifyPayload) -> str:
        """Build the notification message text.

        Format:
        - Normal (DevSpec 11-2):
          ``{date}, 오늘의 키워드인 "kw1", "kw2", "kw3" 외 N개에 대한
          arXiv 논문 정리입니다.``
        - Zero-result (DevSpec 11-7):
          ``{date}, 오늘은 {topic_name} 관련 신규 논문이 없습니다.``
        """
        if payload.total_output == 0:
            return self._build_zero_result_message(payload)
        return self._build_normal_message(payload)

    # ------------------------------------------------------------------
    # Abstract provider hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _send(
        self,
        message: str,
        file_paths: Dict[str, str],
        payload: NotifyPayload,
    ) -> bool:
        """Provider-specific send.  Returns ``True`` on success."""
        ...

    @abstractmethod
    def _send_link_only(
        self,
        message: str,
        payload: NotifyPayload,
    ) -> bool:
        """Provider-specific link-only fallback.  Returns ``True`` on success."""
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_normal_message(self, payload: NotifyPayload) -> str:
        """Build normal message with top-3 keywords (DevSpec 11-2)."""
        top3 = payload.keywords[:3]
        remaining = len(payload.keywords) - 3

        quoted = ", ".join(f'"{kw}"' for kw in top3)

        if remaining > 0:
            keywords_part = f"{quoted} 외 {remaining}개"
        else:
            keywords_part = quoted

        return (
            f"{payload.display_date}, "
            f"오늘의 키워드인 {keywords_part}에 대한 "
            f"arXiv 논문 정리입니다."
        )

    @staticmethod
    def _build_zero_result_message(payload: NotifyPayload) -> str:
        """Build zero-result message (DevSpec 11-7)."""
        return (
            f"{payload.display_date}, "
            f"오늘은 {payload.topic_name} 관련 신규 논문이 없습니다."
        )

    def _check_file_sizes(
        self, file_paths: Dict[str, str]
    ) -> Dict[str, str]:
        """Filter files by size limit.  Returns only attachable files."""
        result: Dict[str, str] = {}
        for key, path in file_paths.items():
            if not os.path.exists(path):
                logger.warning(
                    "File %s does not exist, skipping attachment", path
                )
                continue
            size = os.path.getsize(path)
            if size <= self.MAX_FILE_SIZE:
                result[key] = path
            else:
                logger.warning(
                    "File %s exceeds size limit (%d > %d bytes), "
                    "skipping attachment",
                    path,
                    size,
                    self.MAX_FILE_SIZE,
                )
        return result
