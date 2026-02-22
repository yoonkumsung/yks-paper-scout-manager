"""Abstract base class for notification providers and message formatting.

Provides:
- NotifyPayload dataclass with all fields required for notification.
- NotifierBase ABC with retry logic, file-size checking, and failure
  isolation (notification failures NEVER propagate as pipeline failures).

Section refs: DevSpec 11-1, 11-2, 11-3, 11-7, 11-8.
"""

from __future__ import annotations

import html
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
    notify_mode: str = "file"  # "file" = attach HTML, "link" = send URL only
    allowed_formats: List[str] = field(default_factory=lambda: ["html", "md"])
    event_type: str = "complete"  # "start" or "complete"
    categories: List[str] = field(default_factory=list)
    search_window: Optional[str] = None  # e.g. "2026-02-17 ~ 2026-02-18"
    custom_message: Optional[str] = None  # Override build_message() output


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

        # Link-only mode: send URL instead of file attachment.
        if (
            payload.notify_mode == "link"
            and payload.gh_pages_url
            and payload.event_type != "start"
        ):
            for attempt in range(1, 3):
                try:
                    success = self._send_link_only(message, payload)
                    if success:
                        return True
                    logger.warning(
                        "Link-only send returned False (attempt %d/2) for %s",
                        attempt,
                        payload.topic_slug,
                    )
                except Exception as exc:
                    logger.warning(
                        "Link-only send error (attempt %d/2) for %s: %s",
                        attempt,
                        payload.topic_slug,
                        exc,
                    )
            logger.warning(
                "All link-only attempts failed for %s",
                payload.topic_slug,
            )
            return False

        # File mode (default): attach HTML/MD files.
        # Start events are message-only -- no file attachments.
        if payload.event_type == "start":
            attachable_files: Dict[str, str] = {}
        else:
            # Determine which files to attach (pre-send size check).
            attachable_files = self._check_file_sizes(payload.file_paths)
            # Filter by configured formats (exclude json by default).
            if payload.allowed_formats:
                attachable_files = {
                    k: v for k, v in attachable_files.items()
                    if k in payload.allowed_formats
                }

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
        - Custom message: use ``payload.custom_message`` as-is.
        - Start event:
          ``{topic_name} 논문 수집을 시작합니다.``
        - Normal (DevSpec 11-2):
          ``{date}, 오늘의 키워드인 "kw1", "kw2", "kw3" 외 N개에 대한
          arXiv 논문 정리입니다.``
        - Zero-result (DevSpec 11-7):
          ``{date}, 오늘은 {topic_name} 관련 신규 논문이 없습니다.``
        """
        if payload.custom_message:
            return payload.custom_message
        if payload.event_type == "start":
            return self._build_start_message(payload)
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

    @staticmethod
    def _build_start_message(payload: NotifyPayload) -> str:
        """Build start notification message with search context."""
        topic = html.escape(payload.topic_name)
        lines = [
            f"[Paper Scout] {topic} 논문 수집을 시작합니다.",
            "",
        ]
        if payload.search_window:
            lines.append(f"📋 검색 기간: {html.escape(payload.search_window)}")
        if payload.categories:
            if len(payload.categories) <= 4:
                cats_str = ", ".join(payload.categories)
            else:
                cats_str = ", ".join(payload.categories[:3]) + f" 외 {len(payload.categories) - 3}개"
            lines.append(f"📋 카테고리: {html.escape(cats_str)}")
        return "\n".join(lines)

    def _build_normal_message(self, payload: NotifyPayload) -> str:
        """Build normal completion message.

        When ``gh_pages_url`` is set, appends it as an HTML hyperlink.
        """
        topic = html.escape(payload.topic_name)
        keyword_count = len(payload.keywords)

        lines = [
            f"[Paper Scout] {topic} 논문 수집 완료",
            "",
            f"📊 {payload.total_output}편 수집 / {keyword_count}개 키워드",
        ]

        if payload.gh_pages_url:
            url = html.escape(payload.gh_pages_url)
            slug = html.escape(payload.topic_slug)
            lines.append("")
            lines.append(f'🔗 <a href="{url}">{slug} 데일리 리포트 보기</a>')

        return "\n".join(lines)

    @staticmethod
    def _build_zero_result_message(payload: NotifyPayload) -> str:
        """Build zero-result message (DevSpec 11-7)."""
        topic = html.escape(payload.topic_name)
        return "\n".join([
            f"[Paper Scout] {topic} 논문 수집 완료",
            "",
            "📊 신규 논문이 없습니다.",
        ])

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
