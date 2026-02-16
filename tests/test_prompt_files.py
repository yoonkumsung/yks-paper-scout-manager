"""Tests for prompt template files.

Validates existence, version strings, structure, and content
of all prompt template files used by Paper Scout agents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

PROMPT_FILES = {
    "agent1_keyword_expansion.txt": "agent1-v3",
    "agent2_scoring.txt": "agent2-v2",
    "agent3_summary_tier1.txt": "agent3-tier1-v1",
    "agent3_summary_tier2.txt": "agent3-tier2-v1",
}


# ---------------------------------------------------------------------------
# Test 1: All 4 prompt files exist
# ---------------------------------------------------------------------------


class TestPromptFilesExist:
    """All required prompt template files must exist."""

    @pytest.mark.parametrize("filename", list(PROMPT_FILES.keys()))
    def test_prompt_file_exists(self, filename: str) -> None:
        filepath = PROMPTS_DIR / filename
        assert filepath.exists(), f"Missing prompt file: {filepath}"


# ---------------------------------------------------------------------------
# Test 2 & 5: Each file contains correct version string
# ---------------------------------------------------------------------------


class TestPromptVersionStrings:
    """Each prompt file must contain the expected version identifier."""

    @pytest.mark.parametrize(
        "filename,expected_version",
        list(PROMPT_FILES.items()),
    )
    def test_version_string_present(
        self, filename: str, expected_version: str
    ) -> None:
        filepath = PROMPTS_DIR / filename
        content = filepath.read_text(encoding="utf-8")
        assert expected_version in content, (
            f"{filename} must contain version string '{expected_version}'"
        )


# ---------------------------------------------------------------------------
# Test 3: Each file is non-empty
# ---------------------------------------------------------------------------


class TestPromptFilesNonEmpty:
    """Prompt files must contain meaningful content."""

    @pytest.mark.parametrize("filename", list(PROMPT_FILES.keys()))
    def test_prompt_file_non_empty(self, filename: str) -> None:
        filepath = PROMPTS_DIR / filename
        content = filepath.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, f"{filename} is empty"


# ---------------------------------------------------------------------------
# Test 4: Each file contains [SYSTEM] and [USER] sections
# ---------------------------------------------------------------------------


class TestPromptSections:
    """Each prompt file must have [SYSTEM] and [USER] section markers."""

    @pytest.mark.parametrize("filename", list(PROMPT_FILES.keys()))
    def test_has_system_section(self, filename: str) -> None:
        filepath = PROMPTS_DIR / filename
        content = filepath.read_text(encoding="utf-8")
        assert "[SYSTEM]" in content, (
            f"{filename} missing [SYSTEM] section"
        )

    @pytest.mark.parametrize("filename", list(PROMPT_FILES.keys()))
    def test_has_user_section(self, filename: str) -> None:
        filepath = PROMPTS_DIR / filename
        content = filepath.read_text(encoding="utf-8")
        assert "[USER]" in content, (
            f"{filename} missing [USER] section"
        )
