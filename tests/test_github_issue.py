"""Tests for output.github_issue module.

TASK-027 from SPEC-PAPER-001: GitHub Issue upsert for paper reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from output.github_issue import GitHubIssueError, GitHubIssueManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_papers() -> List[Dict[str, Any]]:
    """Return a list of paper dicts for issue body generation."""
    papers = []
    for i in range(1, 13):
        papers.append(
            {
                "rank": i,
                "title": "Paper Title %d" % i,
                "url": "https://arxiv.org/abs/2401.%05d" % i,
                "final_score": 90.0 - i,
                "summary_ko": "논문 %d 요약입니다." % i,
            }
        )
    return papers


@pytest.fixture
def few_papers() -> List[Dict[str, Any]]:
    """Return a small list of 3 papers."""
    return [
        {
            "rank": 1,
            "title": "First Paper",
            "url": "https://arxiv.org/abs/2401.00001",
            "final_score": 95.0,
            "summary_ko": "첫 번째 논문 요약",
        },
        {
            "rank": 2,
            "title": "Second Paper",
            "url": "https://arxiv.org/abs/2401.00002",
            "final_score": 88.0,
            "summary_ko": "두 번째 논문 요약",
        },
        {
            "rank": 3,
            "title": "Third Paper",
            "url": "https://arxiv.org/abs/2401.00003",
            "final_score": 80.0,
            "summary_ko": "세 번째 논문 요약",
        },
    ]


@pytest.fixture
def manager(tmp_path: Path) -> GitHubIssueManager:
    """Return a GitHubIssueManager with a temporary issue map path."""
    map_path = str(tmp_path / "issue_map.json")
    return GitHubIssueManager(
        repo="owner/repo",
        token="fake-token-123",
        issue_map_path=map_path,
    )


@pytest.fixture
def manager_with_existing_map(tmp_path: Path) -> GitHubIssueManager:
    """Return a manager with a pre-populated issue map."""
    map_path = str(tmp_path / "issue_map.json")
    existing_map = {
        "2026-02-10_ai-sports-device": 42,
        "2026-02-10_llm-optimization": 43,
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(existing_map, f)

    return GitHubIssueManager(
        repo="owner/repo",
        token="fake-token-123",
        issue_map_path=map_path,
    )


# ---------------------------------------------------------------------------
# 1. Title format
# ---------------------------------------------------------------------------


class TestTitleFormat:
    """Test issue title generation."""

    def test_title_format_basic(self, manager: GitHubIssueManager) -> None:
        """Title should follow [YYYY-MM-DD] topic 논문 리포트 (N건) format."""
        title = manager._build_title("2026-02-10", "AI Sports Device", 42)
        assert title == "[2026-02-10] AI Sports Device 논문 리포트 (42건)"

    def test_title_format_zero_papers(self, manager: GitHubIssueManager) -> None:
        """Title with zero papers should show (0건)."""
        title = manager._build_title("2026-01-01", "LLM", 0)
        assert title == "[2026-01-01] LLM 논문 리포트 (0건)"

    def test_title_format_single_paper(self, manager: GitHubIssueManager) -> None:
        """Title with one paper should show (1건)."""
        title = manager._build_title("2026-03-15", "Topic", 1)
        assert title == "[2026-03-15] Topic 논문 리포트 (1건)"


# ---------------------------------------------------------------------------
# 2. Body generation - top 10 papers
# ---------------------------------------------------------------------------


class TestBodyGeneration:
    """Test issue body generation."""

    def test_body_includes_top_10_papers(
        self,
        manager: GitHubIssueManager,
        sample_papers: List[Dict[str, Any]],
    ) -> None:
        """Body should include only top 10 papers, not 11th+."""
        body = manager._build_body(
            papers=sample_papers,
            html_report_url="https://example.com/report.html",
            topic_name="AI Sports",
            date_str="2026-02-10",
        )
        # Papers 1-10 should be present
        for i in range(1, 11):
            assert "Paper Title %d" % i in body
        # Papers 11-12 should NOT be present
        assert "Paper Title 11" not in body
        assert "Paper Title 12" not in body

    def test_body_with_fewer_than_10_papers(
        self,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """Body should include all papers when fewer than 10."""
        body = manager._build_body(
            papers=few_papers,
            html_report_url=None,
            topic_name="AI",
            date_str="2026-02-10",
        )
        assert "First Paper" in body
        assert "Second Paper" in body
        assert "Third Paper" in body


# ---------------------------------------------------------------------------
# 3. Body length limit (60,000 chars)
# ---------------------------------------------------------------------------


class TestBodyLengthLimit:
    """Test body truncation at 60,000 characters."""

    def test_body_truncated_at_60000(self, manager: GitHubIssueManager) -> None:
        """Body exceeding 60,000 chars should be truncated."""
        long_body = "x" * 70000
        result = manager._truncate_body(long_body, max_length=60000)
        assert len(result) <= 60000

    def test_truncation_adds_notice(self, manager: GitHubIssueManager) -> None:
        """Truncated body should include a truncation notice."""
        long_body = "x" * 70000
        result = manager._truncate_body(long_body, max_length=60000)
        assert "..." in result or "truncated" in result.lower() or "잘림" in result

    def test_short_body_not_truncated(self, manager: GitHubIssueManager) -> None:
        """Short body should not be modified."""
        short_body = "Hello world"
        result = manager._truncate_body(short_body, max_length=60000)
        assert result == short_body


# ---------------------------------------------------------------------------
# 4. @mention escape
# ---------------------------------------------------------------------------


class TestMentionEscape:
    """Test @mention escaping."""

    def test_at_mention_escaped(self, manager: GitHubIssueManager) -> None:
        """@username should become @ username."""
        text = "Thanks @alice and @bob for the review"
        result = manager._escape_mentions(text)
        assert "@ alice" in result
        assert "@ bob" in result
        assert "@alice" not in result
        assert "@bob" not in result


# ---------------------------------------------------------------------------
# 5. Mention escape edge cases
# ---------------------------------------------------------------------------


class TestMentionEscapeEdgeCases:
    """Test edge cases for @mention escaping."""

    def test_double_at_sign(self, manager: GitHubIssueManager) -> None:
        """@@ should be handled without crash."""
        result = manager._escape_mentions("@@test")
        # Should not crash; exact behavior may vary
        assert isinstance(result, str)

    def test_lone_at_sign(self, manager: GitHubIssueManager) -> None:
        """A lone @ should remain unchanged."""
        result = manager._escape_mentions("@ alone")
        assert result == "@ alone"

    def test_email_preserved(self, manager: GitHubIssueManager) -> None:
        """Email addresses should be preserved (no space injection)."""
        result = manager._escape_mentions("user@example.com")
        # Email should not have space added after @
        assert "user@example.com" in result or "user@ example.com" not in result

    def test_at_end_of_string(self, manager: GitHubIssueManager) -> None:
        """@ at end of string should remain unchanged."""
        result = manager._escape_mentions("trailing @")
        assert result == "trailing @"


# ---------------------------------------------------------------------------
# 6. Issue map key
# ---------------------------------------------------------------------------


class TestMapKey:
    """Test issue map key generation."""

    def test_map_key_format(self, manager: GitHubIssueManager) -> None:
        """Map key should be {date}_{slug}."""
        key = manager._map_key("2026-02-10", "ai-sports-device")
        assert key == "2026-02-10_ai-sports-device"

    def test_map_key_different_values(self, manager: GitHubIssueManager) -> None:
        """Map key adapts to different input values."""
        key = manager._map_key("2026-03-15", "llm-optimization")
        assert key == "2026-03-15_llm-optimization"


# ---------------------------------------------------------------------------
# 7. Issue map load
# ---------------------------------------------------------------------------


class TestIssueMapLoad:
    """Test issue map loading."""

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing issue_map.json should return empty dict."""
        map_path = str(tmp_path / "nonexistent.json")
        mgr = GitHubIssueManager(
            repo="owner/repo",
            token="fake",
            issue_map_path=map_path,
        )
        assert mgr._issue_map == {}

    def test_load_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """Empty file should return empty dict."""
        map_path = str(tmp_path / "empty.json")
        with open(map_path, "w") as f:
            f.write("")

        mgr = GitHubIssueManager(
            repo="owner/repo",
            token="fake",
            issue_map_path=map_path,
        )
        assert mgr._issue_map == {}

    def test_load_existing_map(
        self, manager_with_existing_map: GitHubIssueManager
    ) -> None:
        """Existing issue_map.json should be loaded correctly."""
        assert manager_with_existing_map._issue_map == {
            "2026-02-10_ai-sports-device": 42,
            "2026-02-10_llm-optimization": 43,
        }


# ---------------------------------------------------------------------------
# 8. Issue map save
# ---------------------------------------------------------------------------


class TestIssueMapSave:
    """Test issue map persistence."""

    def test_save_map_writes_json(self, manager: GitHubIssueManager) -> None:
        """_save_issue_map should persist to JSON file."""
        manager._issue_map = {"2026-02-10_test": 99}
        manager._save_issue_map()

        with open(manager._issue_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data == {"2026-02-10_test": 99}

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """_save_issue_map should create parent directories if needed."""
        nested_path = str(tmp_path / "nested" / "dir" / "issue_map.json")
        mgr = GitHubIssueManager(
            repo="owner/repo",
            token="fake",
            issue_map_path=nested_path,
        )
        mgr._issue_map = {"key": 1}
        mgr._save_issue_map()

        assert os.path.isfile(nested_path)
        with open(nested_path, "r", encoding="utf-8") as f:
            assert json.load(f) == {"key": 1}


# ---------------------------------------------------------------------------
# 9. Create new issue (POST)
# ---------------------------------------------------------------------------


class TestCreateNewIssue:
    """Test new issue creation via POST."""

    @patch("output.github_issue.requests.post")
    def test_create_calls_post(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """When no existing issue in map, POST should be called."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 55}
        mock_post.return_value = mock_response

        issue_num = manager.upsert_issue(
            topic_slug="ai-sports-device",
            topic_name="AI Sports Device",
            date_str="2026-02-10",
            papers=few_papers,
            total_output=3,
            html_report_url="https://example.com/report.html",
        )

        assert issue_num == 55
        mock_post.assert_called_once()

        # Verify POST URL
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "repos/owner/repo/issues" in url

    @patch("output.github_issue.requests.post")
    def test_create_sends_correct_payload(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """POST payload should contain title, body, and labels."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 55}
        mock_post.return_value = mock_response

        manager.upsert_issue(
            topic_slug="ai-sports-device",
            topic_name="AI Sports Device",
            date_str="2026-02-10",
            papers=few_papers,
            total_output=3,
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json", {}) if call_kwargs[1] else {}
        if not payload and len(call_kwargs) > 1:
            payload = call_kwargs[1].get("json", {})

        assert "title" in payload
        assert "body" in payload
        assert "labels" in payload


# ---------------------------------------------------------------------------
# 10. Update existing issue (PATCH)
# ---------------------------------------------------------------------------


class TestUpdateExistingIssue:
    """Test existing issue update via PATCH."""

    @patch("output.github_issue.requests.patch")
    def test_update_calls_patch(
        self,
        mock_patch: MagicMock,
        manager_with_existing_map: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """When issue exists in map, PATCH should be called."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"number": 42}
        mock_patch.return_value = mock_response

        issue_num = manager_with_existing_map.upsert_issue(
            topic_slug="ai-sports-device",
            topic_name="AI Sports Device",
            date_str="2026-02-10",
            papers=few_papers,
            total_output=3,
        )

        assert issue_num == 42
        mock_patch.assert_called_once()

        # Verify PATCH URL contains issue number
        call_args = mock_patch.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "repos/owner/repo/issues/42" in url


# ---------------------------------------------------------------------------
# 11. Map update after create
# ---------------------------------------------------------------------------


class TestMapUpdateAfterCreate:
    """Test that new issue number is saved to map after creation."""

    @patch("output.github_issue.requests.post")
    def test_map_updated_after_create(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """New issue number should be persisted to issue_map.json."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 77}
        mock_post.return_value = mock_response

        manager.upsert_issue(
            topic_slug="test-topic",
            topic_name="Test Topic",
            date_str="2026-02-10",
            papers=few_papers,
            total_output=3,
        )

        # Verify in-memory map is updated
        assert manager._issue_map.get("2026-02-10_test-topic") == 77

        # Verify persisted to disk
        with open(manager._issue_map_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved.get("2026-02-10_test-topic") == 77


# ---------------------------------------------------------------------------
# 12. Labels
# ---------------------------------------------------------------------------


class TestLabels:
    """Test issue labels."""

    @patch("output.github_issue.requests.post")
    def test_labels_include_paper_report_and_slug(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """Labels should include 'paper-report' and the topic slug."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 1}
        mock_post.return_value = mock_response

        manager.upsert_issue(
            topic_slug="ai-sports-device",
            topic_name="AI Sports Device",
            date_str="2026-02-10",
            papers=few_papers,
            total_output=3,
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json", {})
        labels = payload.get("labels", [])

        assert "paper-report" in labels
        assert "ai-sports-device" in labels


# ---------------------------------------------------------------------------
# 13. Empty papers
# ---------------------------------------------------------------------------


class TestEmptyPapers:
    """Test handling of empty papers list."""

    @patch("output.github_issue.requests.post")
    def test_empty_papers_generates_valid_body(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
    ) -> None:
        """0 papers should generate valid body with a note."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 1}
        mock_post.return_value = mock_response

        manager.upsert_issue(
            topic_slug="empty-topic",
            topic_name="Empty Topic",
            date_str="2026-02-10",
            papers=[],
            total_output=0,
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json", {})
        body = payload.get("body", "")

        # Body should be non-empty and mention no papers or zero
        assert len(body) > 0
        assert isinstance(body, str)


# ---------------------------------------------------------------------------
# 14. Special characters in title
# ---------------------------------------------------------------------------


class TestSpecialCharactersInTitle:
    """Test special characters in title."""

    def test_special_chars_in_topic_name(
        self, manager: GitHubIssueManager
    ) -> None:
        """Special characters in topic name should be handled."""
        title = manager._build_title(
            "2026-02-10", "AI/ML & NLP <Research>", 5
        )
        assert "[2026-02-10]" in title
        assert "AI/ML & NLP <Research>" in title
        assert "(5건)" in title


# ---------------------------------------------------------------------------
# 15. HTML report link
# ---------------------------------------------------------------------------


class TestHtmlReportLink:
    """Test HTML report link handling in body."""

    def test_html_report_link_included(
        self, manager: GitHubIssueManager, few_papers: List[Dict[str, Any]]
    ) -> None:
        """HTML report URL should be included in body when provided."""
        body = manager._build_body(
            papers=few_papers,
            html_report_url="https://example.com/report.html",
            topic_name="Test",
            date_str="2026-02-10",
        )
        assert "https://example.com/report.html" in body

    def test_html_report_link_omitted_when_none(
        self, manager: GitHubIssueManager, few_papers: List[Dict[str, Any]]
    ) -> None:
        """Body should be valid without HTML report URL."""
        body = manager._build_body(
            papers=few_papers,
            html_report_url=None,
            topic_name="Test",
            date_str="2026-02-10",
        )
        # Should not contain the link section phrasing
        assert "전체 리포트 보기" not in body
        assert isinstance(body, str)
        assert len(body) > 0


# ---------------------------------------------------------------------------
# 16. API error handling
# ---------------------------------------------------------------------------


class TestApiErrorHandling:
    """Test graceful handling of API failures."""

    @patch("output.github_issue.requests.post")
    def test_create_failure_raises_error(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """API failure on create should raise GitHubIssueError."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Failed"
        mock_response.json.return_value = {"message": "Validation Failed"}
        mock_post.return_value = mock_response

        with pytest.raises(GitHubIssueError):
            manager.upsert_issue(
                topic_slug="fail-topic",
                topic_name="Fail Topic",
                date_str="2026-02-10",
                papers=few_papers,
                total_output=3,
            )

    @patch("output.github_issue.requests.patch")
    def test_update_failure_raises_error(
        self,
        mock_patch: MagicMock,
        manager_with_existing_map: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """API failure on update should raise GitHubIssueError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"message": "Internal Server Error"}
        mock_patch.return_value = mock_response

        with pytest.raises(GitHubIssueError):
            manager_with_existing_map.upsert_issue(
                topic_slug="ai-sports-device",
                topic_name="AI Sports Device",
                date_str="2026-02-10",
                papers=few_papers,
                total_output=3,
            )

    @patch("output.github_issue.requests.post")
    def test_network_error_raises_error(
        self,
        mock_post: MagicMock,
        manager: GitHubIssueManager,
        few_papers: List[Dict[str, Any]],
    ) -> None:
        """Network error should raise GitHubIssueError."""
        import requests as req_lib

        mock_post.side_effect = req_lib.ConnectionError("Connection refused")

        with pytest.raises(GitHubIssueError):
            manager.upsert_issue(
                topic_slug="net-fail",
                topic_name="Net Fail",
                date_str="2026-02-10",
                papers=few_papers,
                total_output=3,
            )
