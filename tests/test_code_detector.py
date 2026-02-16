"""Tests for core.scoring.code_detector.

Covers regex-based code/repository detection in paper abstracts and
comments, URL extraction, merge logic with LLM signals, and edge cases.
"""

from __future__ import annotations

import pytest

from core.scoring.code_detector import CodeDetector


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector() -> CodeDetector:
    """Provide a fresh CodeDetector instance."""
    return CodeDetector()


# ---------------------------------------------------------------------------
# 1. GitHub URL detection
# ---------------------------------------------------------------------------


class TestGitHubURL:
    """Pattern: github_url."""

    def test_github_url_detected_in_abstract(self, detector: CodeDetector):
        """GitHub URL in abstract should be detected."""
        result = detector.detect(
            "We release code at https://github.com/user/repo"
        )
        assert result["has_code"] is True
        assert "github_url" in result["matched_patterns"]

    def test_github_url_detected_in_comment(self, detector: CodeDetector):
        """GitHub URL in comment should be detected."""
        result = detector.detect(
            "A generic abstract about ML.",
            comment="Code: https://github.com/org/project",
        )
        assert result["has_code"] is True
        assert "github_url" in result["matched_patterns"]

    def test_github_url_complex_path(self, detector: CodeDetector):
        """GitHub URL with complex org/repo-name.git should match."""
        result = detector.detect(
            "See https://github.com/my-org/repo-name.git for code."
        )
        assert result["has_code"] is True
        assert "github_url" in result["matched_patterns"]

    def test_gitlab_url_not_detected(self, detector: CodeDetector):
        """gitlab.com URL should NOT trigger the github_url pattern."""
        result = detector.detect(
            "Code at https://gitlab.com/user/project"
        )
        assert "github_url" not in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 2. "code available" variants
# ---------------------------------------------------------------------------


class TestCodeAvailable:
    """Pattern: code_available."""

    def test_code_is_available(self, detector: CodeDetector):
        """'code is available' should be detected."""
        result = detector.detect("The code is available online.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]

    def test_code_available_case_insensitive(self, detector: CodeDetector):
        """'Code Available' (mixed case) should be detected."""
        result = detector.detect("Code Available at our website.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]

    def test_source_code_is_available(self, detector: CodeDetector):
        """'source code is available' should be detected."""
        result = detector.detect("The source code is available.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]

    def test_code_available_at(self, detector: CodeDetector):
        """'code available at' should be detected."""
        result = detector.detect("Our code available at the project page.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]

    def test_code_released(self, detector: CodeDetector):
        """'code released' should be detected."""
        result = detector.detect("The code released as open source.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]

    def test_code_provided(self, detector: CodeDetector):
        """'source code provided' should be detected."""
        result = detector.detect("Source code provided in supplementary.")
        assert result["has_code"] is True
        assert "code_available" in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 3. "our code" / "our implementation"
# ---------------------------------------------------------------------------


class TestOurCode:
    """Pattern: our_code."""

    def test_our_code(self, detector: CodeDetector):
        """'our code' should be detected."""
        result = detector.detect("We benchmark our code against baselines.")
        assert result["has_code"] is True
        assert "our_code" in result["matched_patterns"]

    def test_our_implementation(self, detector: CodeDetector):
        """'our implementation' should be detected."""
        result = detector.detect("Our implementation uses PyTorch.")
        assert result["has_code"] is True
        assert "our_code" in result["matched_patterns"]

    def test_our_source_code(self, detector: CodeDetector):
        """'our source code' should be detected."""
        result = detector.detect("We release our source code publicly.")
        assert result["has_code"] is True
        assert "our_code" in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 4. "open source" variants
# ---------------------------------------------------------------------------


class TestOpenSource:
    """Pattern: open_source."""

    def test_open_source_code(self, detector: CodeDetector):
        """'open source code' should be detected."""
        result = detector.detect("We provide open source code.")
        assert result["has_code"] is True
        assert "open_source" in result["matched_patterns"]

    def test_open_source_implementation_hyphen(self, detector: CodeDetector):
        """'open-source implementation' should be detected."""
        result = detector.detect(
            "An open-source implementation is available."
        )
        assert result["has_code"] is True
        assert "open_source" in result["matched_patterns"]

    def test_open_source_software(self, detector: CodeDetector):
        """'open source software' should be detected."""
        result = detector.detect("Released as open source software.")
        assert result["has_code"] is True
        assert "open_source" in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 5. "repository:" pattern
# ---------------------------------------------------------------------------


class TestRepository:
    """Pattern: repository."""

    def test_repository_https(self, detector: CodeDetector):
        """'repository: https://...' should be detected."""
        result = detector.detect(
            "Code repository: https://github.com/user/repo"
        )
        assert result["has_code"] is True
        assert "repository" in result["matched_patterns"]

    def test_repository_http(self, detector: CodeDetector):
        """'repository: http://...' should be detected."""
        result = detector.detect(
            "repository: http://example.com/code"
        )
        assert result["has_code"] is True
        assert "repository" in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 6. No match / edge cases
# ---------------------------------------------------------------------------


class TestNoMatch:
    """Cases that should NOT trigger detection."""

    def test_no_match_returns_false(self, detector: CodeDetector):
        """Generic abstract with no code mention returns has_code=False."""
        result = detector.detect(
            "We present a novel approach to image classification."
        )
        assert result["has_code"] is False
        assert result["has_code_source"] == "none"
        assert result["code_url"] is None
        assert result["matched_patterns"] == []

    def test_empty_abstract(self, detector: CodeDetector):
        """Empty abstract should return no match."""
        result = detector.detect("")
        assert result["has_code"] is False
        assert result["matched_patterns"] == []

    def test_none_comment_handled(self, detector: CodeDetector):
        """None comment should be handled gracefully."""
        result = detector.detect(
            "A normal abstract.",
            comment=None,
        )
        assert result["has_code"] is False

    def test_code_like_words_no_match(self, detector: CodeDetector):
        """Words like 'the code runs' should NOT trigger detection."""
        result = detector.detect(
            "The code runs for 100 epochs on the training set."
        )
        assert result["has_code"] is False

    def test_abstract_with_no_code_mention_clean_result(
        self, detector: CodeDetector
    ):
        """Abstract without code mention returns clean result dict."""
        result = detector.detect(
            "We propose a transformer-based architecture for NLP."
        )
        assert result == {
            "has_code": False,
            "has_code_source": "none",
            "code_url": None,
            "matched_patterns": [],
        }


# ---------------------------------------------------------------------------
# 7. URL extraction
# ---------------------------------------------------------------------------


class TestURLExtraction:
    """Tests for extract_urls method."""

    def test_extract_github_url(self, detector: CodeDetector):
        """Should extract a standard GitHub URL."""
        urls = detector.extract_urls(
            "Visit https://github.com/user/repo for details."
        )
        assert len(urls) >= 1
        assert "https://github.com/user/repo" in urls

    def test_strips_trailing_parenthesis(self, detector: CodeDetector):
        """Trailing closing parenthesis should be stripped."""
        urls = detector.extract_urls(
            "(see https://github.com/user/repo)"
        )
        assert urls[0] == "https://github.com/user/repo"

    def test_strips_trailing_comma(self, detector: CodeDetector):
        """Trailing comma should be stripped."""
        urls = detector.extract_urls(
            "https://example.com/path, and more text"
        )
        assert urls[0] == "https://example.com/path"

    def test_strips_trailing_period(self, detector: CodeDetector):
        """Trailing period should be stripped."""
        urls = detector.extract_urls(
            "Available at https://github.com/org/repo."
        )
        assert urls[0] == "https://github.com/org/repo"

    def test_strips_trailing_colon(self, detector: CodeDetector):
        """Trailing colon should be stripped."""
        urls = detector.extract_urls(
            "See https://example.com/path:"
        )
        assert urls[0] == "https://example.com/path"

    def test_strips_trailing_semicolon(self, detector: CodeDetector):
        """Trailing semicolon should be stripped."""
        urls = detector.extract_urls(
            "Link https://example.com/path;"
        )
        assert urls[0] == "https://example.com/path"

    def test_multiple_urls_extracted(self, detector: CodeDetector):
        """Multiple URLs should all be extracted."""
        urls = detector.extract_urls(
            "Code: https://github.com/a/b and data: https://example.com/data"
        )
        assert len(urls) == 2

    def test_no_urls_returns_empty_list(self, detector: CodeDetector):
        """Text without URLs should return empty list."""
        urls = detector.extract_urls("No URLs here.")
        assert urls == []


# ---------------------------------------------------------------------------
# 8. merge_with_llm
# ---------------------------------------------------------------------------


class TestMergeWithLLM:
    """Tests for merge_with_llm provenance logic."""

    def _base_result(self, has_code: bool) -> dict:
        """Create a minimal detect result for merge testing."""
        return {
            "has_code": has_code,
            "has_code_source": "regex" if has_code else "none",
            "code_url": "https://github.com/u/r" if has_code else None,
            "matched_patterns": ["github_url"] if has_code else [],
        }

    def test_regex_true_llm_true(self, detector: CodeDetector):
        """regex=True + llm=True -> source='both'."""
        result = detector.merge_with_llm(self._base_result(True), True)
        assert result["has_code"] is True
        assert result["has_code_source"] == "both"

    def test_regex_true_llm_false(self, detector: CodeDetector):
        """regex=True + llm=False -> source='regex'."""
        result = detector.merge_with_llm(self._base_result(True), False)
        assert result["has_code"] is True
        assert result["has_code_source"] == "regex"

    def test_regex_false_llm_true(self, detector: CodeDetector):
        """regex=False + llm=True -> source='llm'."""
        result = detector.merge_with_llm(self._base_result(False), True)
        assert result["has_code"] is True
        assert result["has_code_source"] == "llm"

    def test_regex_false_llm_false(self, detector: CodeDetector):
        """regex=False + llm=False -> source='none'."""
        result = detector.merge_with_llm(self._base_result(False), False)
        assert result["has_code"] is False
        assert result["has_code_source"] == "none"

    def test_merge_preserves_other_fields(self, detector: CodeDetector):
        """merge_with_llm should preserve code_url and matched_patterns."""
        base = self._base_result(True)
        result = detector.merge_with_llm(base, True)
        assert result["code_url"] == "https://github.com/u/r"
        assert result["matched_patterns"] == ["github_url"]

    def test_merge_does_not_mutate_original(self, detector: CodeDetector):
        """merge_with_llm should not mutate the original dict."""
        base = self._base_result(True)
        original_source = base["has_code_source"]
        detector.merge_with_llm(base, True)
        assert base["has_code_source"] == original_source


# ---------------------------------------------------------------------------
# 9. Multiple patterns matched
# ---------------------------------------------------------------------------


class TestMultiplePatterns:
    """Scenarios where more than one pattern matches."""

    def test_github_url_and_our_code(self, detector: CodeDetector):
        """GitHub URL + 'our code' should both appear in matched_patterns."""
        result = detector.detect(
            "We release our code at https://github.com/user/repo."
        )
        assert result["has_code"] is True
        assert "github_url" in result["matched_patterns"]
        assert "our_code" in result["matched_patterns"]
        assert len(result["matched_patterns"]) >= 2

    def test_matched_patterns_populated_correctly(
        self, detector: CodeDetector
    ):
        """matched_patterns list should contain all matching pattern names."""
        result = detector.detect(
            "Our implementation is open-source code, "
            "available at https://github.com/user/repo."
        )
        assert "github_url" in result["matched_patterns"]
        assert "our_code" in result["matched_patterns"]
        assert "open_source" in result["matched_patterns"]


# ---------------------------------------------------------------------------
# 10. Code URL extraction from detect
# ---------------------------------------------------------------------------


class TestCodeURLFromDetect:
    """Tests for code_url field in detect() output."""

    def test_code_url_extracted_from_github(self, detector: CodeDetector):
        """code_url should be populated when a URL is present."""
        result = detector.detect(
            "Code at https://github.com/user/repo for details."
        )
        assert result["code_url"] is not None
        assert "github.com" in result["code_url"]

    def test_code_url_none_when_no_url(self, detector: CodeDetector):
        """code_url should be None when pattern matches but no URL exists."""
        result = detector.detect("Our implementation uses PyTorch.")
        assert result["has_code"] is True
        assert result["code_url"] is None

    def test_code_url_from_repository_pattern(self, detector: CodeDetector):
        """code_url should be extracted from 'repository:' pattern."""
        result = detector.detect(
            "repository: https://example.com/project/code"
        )
        assert result["code_url"] == "https://example.com/project/code"
