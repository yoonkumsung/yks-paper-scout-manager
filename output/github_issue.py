"""GitHub Issue upsert module for Paper Scout reports.

Creates or updates GitHub Issues for paper reports using
``data/issue_map.json`` to track issue numbers across runs.

Section reference: TASK-027 from SPEC-PAPER-001 (devspec 10-3).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_MAX_BODY_LENGTH = 60000
_MAX_SUMMARY_PAPERS = 10
_TRUNCATION_NOTICE = "\n\n---\n\n> (본문이 너무 길어 일부가 잘림 처리되었습니다.)"

# Matches @username but NOT email-style patterns or lone @.
# Captures @ followed by a word character (alphanumeric or underscore).
_MENTION_RE = re.compile(r"(?<![a-zA-Z0-9._%+-])@([a-zA-Z0-9][\w-]*)")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class GitHubIssueError(Exception):
    """Raised when a GitHub Issue API operation fails."""

    pass


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GitHubIssueManager:
    """Manage GitHub Issue creation and updates for paper reports.

    Uses ``issue_map.json`` to determine whether to create (POST)
    or update (PATCH) an existing issue.

    Map key format: ``{window_end_date}_{slug}``
    Map value: ``issue_number`` (int)
    """

    def __init__(
        self,
        repo: str,
        token: str,
        issue_map_path: str = "data/issue_map.json",
    ) -> None:
        """Initialize the manager.

        Args:
            repo: GitHub repository in ``"owner/repo"`` format.
            token: GitHub personal access token or fine-grained token.
            issue_map_path: Path to the JSON file mapping keys to
                issue numbers.
        """
        self._repo = repo
        self._token = token
        self._issue_map_path = issue_map_path
        self._issue_map: Dict[str, int] = self._load_issue_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_issue(
        self,
        topic_slug: str,
        topic_name: str,
        date_str: str,
        papers: List[Dict[str, Any]],
        total_output: int,
        html_report_url: Optional[str] = None,
    ) -> int:
        """Create or update a GitHub Issue for the paper report.

        Args:
            topic_slug: URL-safe topic identifier (e.g. ``"ai-sports-device"``).
            topic_name: Human-readable topic name.
            date_str: Report date in ``YYYY-MM-DD`` format.
            papers: Top papers for summary (only first 10 are used).
            total_output: Total number of papers in the report.
            html_report_url: Optional URL to the full HTML report.

        Returns:
            The GitHub issue number (int).

        Raises:
            GitHubIssueError: If the GitHub API call fails.
        """
        key = self._map_key(date_str, topic_slug)
        title = self._build_title(date_str, topic_name, total_output)
        body = self._build_body(papers, html_report_url, topic_name, date_str)
        body = self._escape_mentions(body)
        body = self._truncate_body(body, _MAX_BODY_LENGTH)
        labels = ["paper-report", topic_slug]

        existing_number = self._issue_map.get(key)

        try:
            if existing_number is not None:
                issue_number = self._update_issue(
                    existing_number, title, body, labels
                )
            else:
                issue_number = self._create_issue(title, body, labels)
                self._issue_map[key] = issue_number
                self._save_issue_map()
        except requests.RequestException as exc:
            raise GitHubIssueError(
                "GitHub API request failed: %s" % str(exc)
            ) from exc

        return issue_number

    # ------------------------------------------------------------------
    # Title / body builders
    # ------------------------------------------------------------------

    def _build_title(
        self, date_str: str, topic_name: str, total_output: int
    ) -> str:
        """Build issue title.

        Format: ``[YYYY-MM-DD] {topic_name} 논문 리포트 (N건)``
        """
        return "[%s] %s 논문 리포트 (%d건)" % (date_str, topic_name, total_output)

    def _build_body(
        self,
        papers: List[Dict[str, Any]],
        html_report_url: Optional[str],
        topic_name: str,
        date_str: str,
    ) -> str:
        """Build issue body with top 10 papers summary.

        The body must not exceed ``_MAX_BODY_LENGTH`` characters
        (truncation is applied separately via ``_truncate_body``).
        ``@mentions`` are escaped separately via ``_escape_mentions``.
        """
        lines: List[str] = []

        # Header
        lines.append("# %s 논문 리포트" % topic_name)
        lines.append("")
        lines.append("날짜: %s" % date_str)
        lines.append("")

        # Papers summary
        top_papers = papers[:_MAX_SUMMARY_PAPERS]

        if top_papers:
            lines.append("## 상위 논문 요약")
            lines.append("")

            for idx, paper in enumerate(top_papers, start=1):
                title = paper.get("title", "")
                url = paper.get("url", "")
                final_score = paper.get("final_score", 0.0)
                summary_ko = paper.get("summary_ko", "")

                lines.append("### %d. %s" % (idx, title))
                lines.append("- arXiv: %s" % url)
                lines.append("- 점수: %s" % _format_score(final_score))
                if summary_ko:
                    lines.append("- %s" % summary_ko)
                lines.append("")
        else:
            lines.append("이번 기간에 수집된 논문이 없습니다.")
            lines.append("")

        # Separator
        lines.append("---")
        lines.append("")

        # HTML report link (only when provided)
        if html_report_url:
            lines.append(
                "[전체 리포트 보기](%s)" % html_report_url
            )
            lines.append("")
            lines.append("---")
            lines.append("")

        # Footer
        lines.append("이 Issue는 Paper Scout에 의해 자동 생성되었습니다.")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    def _escape_mentions(self, text: str) -> str:
        """Replace ``@username`` with ``@ username`` to prevent GitHub mentions.

        Preserves email addresses (e.g. ``user@example.com``) and
        lone ``@`` characters.
        """
        return _MENTION_RE.sub(r"@ \1", text)

    def _truncate_body(self, body: str, max_length: int = _MAX_BODY_LENGTH) -> str:
        """Truncate body to *max_length*, adding truncation notice if needed."""
        if len(body) <= max_length:
            return body

        # Reserve space for the truncation notice.
        notice = _TRUNCATION_NOTICE
        cut_at = max_length - len(notice)
        if cut_at < 0:
            cut_at = 0
        return body[:cut_at] + notice

    # ------------------------------------------------------------------
    # Issue map I/O
    # ------------------------------------------------------------------

    def _load_issue_map(self) -> Dict[str, int]:
        """Load ``issue_map.json`` or return empty dict."""
        if not os.path.isfile(self._issue_map_path):
            return {}

        try:
            with open(self._issue_map_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Failed to load issue map from %s, starting fresh",
                self._issue_map_path,
            )
            return {}

    def _save_issue_map(self) -> None:
        """Save ``issue_map.json``."""
        parent = os.path.dirname(self._issue_map_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._issue_map_path, "w", encoding="utf-8") as f:
            json.dump(self._issue_map, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Map key
    # ------------------------------------------------------------------

    def _map_key(self, date_str: str, topic_slug: str) -> str:
        """Build the map key: ``{date}_{slug}``."""
        return "%s_%s" % (date_str, topic_slug)

    # ------------------------------------------------------------------
    # GitHub API calls
    # ------------------------------------------------------------------

    def _create_issue(
        self, title: str, body: str, labels: List[str]
    ) -> int:
        """Create a new GitHub Issue via POST.

        Returns the issue number.

        Raises:
            GitHubIssueError: On non-201 response status.
        """
        url = "%s/repos/%s/issues" % (_GITHUB_API_BASE, self._repo)
        payload = {
            "title": title,
            "body": body,
            "labels": labels,
        }
        resp = requests.post(url, headers=self._headers(), json=payload)

        if resp.status_code != 201:
            raise GitHubIssueError(
                "Failed to create issue (HTTP %d): %s"
                % (resp.status_code, resp.text)
            )

        data = resp.json()
        issue_number: int = data["number"]
        logger.info("Created issue #%d: %s", issue_number, title)
        return issue_number

    def _update_issue(
        self,
        issue_number: int,
        title: str,
        body: str,
        labels: List[str],
    ) -> int:
        """Update an existing GitHub Issue via PATCH.

        Returns the issue number.

        Raises:
            GitHubIssueError: On non-200 response status.
        """
        url = "%s/repos/%s/issues/%d" % (
            _GITHUB_API_BASE,
            self._repo,
            issue_number,
        )
        payload = {
            "title": title,
            "body": body,
            "labels": labels,
        }
        resp = requests.patch(url, headers=self._headers(), json=payload)

        if resp.status_code != 200:
            raise GitHubIssueError(
                "Failed to update issue #%d (HTTP %d): %s"
                % (issue_number, resp.status_code, resp.text)
            )

        logger.info("Updated issue #%d: %s", issue_number, title)
        return issue_number

    def _headers(self) -> Dict[str, str]:
        """Build standard GitHub API request headers."""
        return {
            "Authorization": "Bearer %s" % self._token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_score(score: Any) -> str:
    """Format a score value for display.

    Examples:
        87.5 -> "87.5"
        70.0 -> "70.0"
    """
    if isinstance(score, float):
        formatted = "%.1f" % score
        return formatted
    return str(score)
