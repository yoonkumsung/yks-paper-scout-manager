"""Regex-based detection of code/repository availability in paper text.

Scans paper abstracts and comments for signals that source code or
repositories are available (GitHub URLs, "code available", etc.) and
merges the regex result with the LLM-based ``mentions_code`` flag
produced by Agent 2 to determine the authoritative ``has_code`` value.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

PATTERNS: dict[str, re.Pattern[str]] = {
    "github_url": re.compile(
        r"github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
        re.IGNORECASE,
    ),
    "code_available": re.compile(
        r"(?:code|source\s+code)\s+(?:is\s+)?(?:available|released|provided)",
        re.IGNORECASE,
    ),
    "our_code": re.compile(
        r"our\s+(?:code|implementation|source\s+code)",
        re.IGNORECASE,
    ),
    "open_source": re.compile(
        r"open[- ]source\s+(?:code|implementation|software)",
        re.IGNORECASE,
    ),
    "repository": re.compile(
        r"repository\s*:\s*https?://\S+",
        re.IGNORECASE,
    ),
}

URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://[^\s\)\]>\"',;]+",
    re.IGNORECASE,
)

# Trailing punctuation that is likely sentence-ending, not part of the URL
_TRAILING_PUNCT = re.compile(r"[.,:;]+$")


class CodeDetector:
    """Detect code repository availability using regex patterns."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        abstract: str,
        comment: str | None = None,
    ) -> dict:
        """Check *abstract* and *comment* for code availability signals.

        Args:
            abstract: Paper abstract text.
            comment: Optional paper comment text.

        Returns:
            dict with keys:
                - ``has_code``  (bool): True if any pattern matched.
                - ``has_code_source`` (str): ``"regex"`` if matched,
                  ``"none"`` if not.
                - ``code_url`` (str | None): First extracted URL, if any.
                - ``matched_patterns`` (list[str]): Names of patterns
                  that matched.
        """
        combined = abstract
        if comment:
            combined = f"{abstract}\n{comment}"

        matched_patterns: list[str] = []
        for name, pattern in PATTERNS.items():
            if pattern.search(combined):
                matched_patterns.append(name)

        has_code = len(matched_patterns) > 0

        code_url: str | None = None
        if has_code:
            urls = self.extract_urls(combined)
            if urls:
                code_url = urls[0]

        return {
            "has_code": has_code,
            "has_code_source": "regex" if has_code else "none",
            "code_url": code_url,
            "matched_patterns": matched_patterns,
        }

    def merge_with_llm(
        self,
        detect_result: dict,
        mentions_code_llm: bool,
    ) -> dict:
        """Merge regex detection with the Agent 2 LLM signal.

        Updates ``has_code`` and ``has_code_source``:

        * regex=True,  llm=True  -> has_code=True,  source="both"
        * regex=True,  llm=False -> has_code=True,  source="regex"
        * regex=False, llm=True  -> has_code=True,  source="llm"
        * regex=False, llm=False -> has_code=False,  source="none"
        """
        regex_matched = detect_result["has_code"]

        if regex_matched and mentions_code_llm:
            source = "both"
        elif regex_matched:
            source = "regex"
        elif mentions_code_llm:
            source = "llm"
        else:
            source = "none"

        result = dict(detect_result)
        result["has_code"] = regex_matched or mentions_code_llm
        result["has_code_source"] = source
        return result

    def extract_urls(self, text: str) -> list[str]:
        """Extract URLs from *text*, handling common delimiters.

        Post-processes each match to strip trailing sentence-ending
        punctuation (``.``, ``,``, ``:``, ``;``).
        """
        raw_urls = URL_PATTERN.findall(text)
        cleaned: list[str] = []
        for url in raw_urls:
            url = _TRAILING_PUNCT.sub("", url)
            cleaned.append(url)
        return cleaned
