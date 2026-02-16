"""Tests for output.render.json_exporter.

TASK-024 from SPEC-PAPER-001: JSON exporter for Paper Scout reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from output.render.json_exporter import export_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_papers() -> list[dict]:
    """Return a list of ranked paper dicts matching devspec 10-4 schema."""
    return [
        {
            "rank": 1,
            "tier": 1,
            "paper_key": "arxiv:2401.12345",
            "title": "Deep Learning for Sports Analytics",
            "url": "https://arxiv.org/abs/2401.12345",
            "pdf_url": "https://arxiv.org/pdf/2401.12345",
            "authors": ["Author A", "Author B"],
            "categories": ["cs.CV", "cs.AI"],
            "published_at_utc": "2026-02-09T00:00:00Z",
            "base_score": 82,
            "bonus_score": 13,
            "llm_adjusted": 95,
            "embed_score": 0.78,
            "final_score": 87.5,
            "has_code": True,
            "code_url": "https://github.com/example/repo",
            "flags": {"is_edge": True, "is_realtime": True},
            "score_lowered": False,
            "summary_ko": "AI 스포츠 분석 논문 요약",
            "reason_ko": "관련성 높음",
            "insight_ko": "새로운 접근법 제시",
            "cluster_id": 0,
        },
        {
            "rank": 2,
            "tier": 2,
            "paper_key": "arxiv:2401.67890",
            "title": "Reinforcement Learning in Game",
            "url": "https://arxiv.org/abs/2401.67890",
            "pdf_url": "https://arxiv.org/pdf/2401.67890",
            "authors": ["Author C"],
            "categories": ["cs.LG"],
            "published_at_utc": "2026-02-08T12:00:00Z",
            "base_score": 65,
            "bonus_score": 5,
            "llm_adjusted": 70,
            "embed_score": 0.55,
            "final_score": 62.0,
            "has_code": False,
            "code_url": None,
            "flags": {"is_edge": False, "is_realtime": False},
            "score_lowered": True,
            "summary_ko": "강화학습 논문 요약",
            "reason_ko": "간접 관련",
            "insight_ko": "적용 가능성 탐색",
            "cluster_id": 1,
        },
    ]


@pytest.fixture
def sample_clusters() -> list[dict]:
    """Return sample cluster data from Clusterer."""
    return [
        {
            "cluster_id": 0,
            "representative_key": "arxiv:2401.12345",
            "member_keys": ["arxiv:2401.12345"],
            "size": 1,
        },
        {
            "cluster_id": 1,
            "representative_key": "arxiv:2401.67890",
            "member_keys": ["arxiv:2401.67890"],
            "size": 1,
        },
    ]


@pytest.fixture
def sample_remind_papers() -> list[dict]:
    """Return sample remind tab papers."""
    return [
        {
            "paper_key": "arxiv:2401.00001",
            "title": "Previously Recommended Paper",
            "url": "https://arxiv.org/abs/2401.00001",
            "recommend_count": 1,
        },
    ]


@pytest.fixture
def sample_scoring_weights() -> dict:
    """Return sample scoring weights."""
    return {"llm": 0.55, "embedding": 0.35, "recency": 0.10}


@pytest.fixture
def sample_stats() -> dict:
    """Return sample stats dict."""
    return {
        "total_collected": 347,
        "total_filtered": 156,
        "total_discarded": 23,
        "total_scored": 133,
        "total_output": 42,
    }


@pytest.fixture
def base_export_kwargs(
    sample_papers: list[dict],
    sample_clusters: list[dict],
    sample_remind_papers: list[dict],
    sample_scoring_weights: dict,
    sample_stats: dict,
    tmp_path: Path,
) -> dict[str, Any]:
    """Return common kwargs for export_json calls."""
    return {
        "topic_slug": "ai-sports-device",
        "topic_name": "AI Sports Device/Platform",
        "date_str": "20260210",
        "display_title": "26\ub144 02\uc6d4 10\uc77c \ud654\uc694\uc77c - AI \uc2a4\ud3ec\uce20 \ub514\ubc14\uc774\uc2a4 arXiv \ub17c\ubb38 \uc815\ub9ac",
        "window_start_utc": "2026-02-09T01:30:00Z",
        "window_end_utc": "2026-02-10T02:30:00Z",
        "embedding_mode": "en_synthetic",
        "scoring_weights": sample_scoring_weights,
        "stats": sample_stats,
        "threshold_used": 60,
        "threshold_lowered": False,
        "run_id": 128,
        "keywords_used": ["automatic cinematography", "sports highlight"],
        "papers": sample_papers,
        "clusters": sample_clusters,
        "remind_papers": sample_remind_papers,
        "output_dir": str(tmp_path / "reports"),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJsonExporterBasic:
    """Basic export functionality."""

    def test_basic_export_creates_json_file(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """JSON file should be created with correct structure."""
        result_path = export_json(**base_export_kwargs)

        assert os.path.isfile(result_path)
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level keys present
        assert "meta" in data
        assert "papers" in data
        assert "clusters" in data
        assert "remind_papers" in data

    def test_returns_path_string(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """export_json should return the path to the written file as str."""
        result_path = export_json(**base_export_kwargs)

        assert isinstance(result_path, str)
        assert result_path.endswith(".json")


class TestJsonExporterMeta:
    """Meta fields validation."""

    def test_all_meta_fields_present_and_correct(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """All meta fields should be present and have correct values."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["meta"]

        assert meta["topic_name"] == "AI Sports Device/Platform"
        assert meta["topic_slug"] == "ai-sports-device"
        assert meta["date"] == "2026-02-10"
        assert meta["display_title"] == base_export_kwargs["display_title"]
        assert meta["window_start_utc"] == "2026-02-09T01:30:00Z"
        assert meta["window_end_utc"] == "2026-02-10T02:30:00Z"
        assert meta["embedding_mode"] == "en_synthetic"
        assert meta["scoring_weights"] == {"llm": 0.55, "embedding": 0.35, "recency": 0.10}
        assert meta["total_collected"] == 347
        assert meta["total_filtered"] == 156
        assert meta["total_discarded"] == 23
        assert meta["total_scored"] == 133
        assert meta["total_output"] == 42
        assert meta["threshold_used"] == 60
        assert meta["threshold_lowered"] is False
        assert meta["run_id"] == 128
        assert meta["keywords_used"] == ["automatic cinematography", "sports highlight"]


class TestJsonExporterFileNaming:
    """File naming convention tests."""

    def test_file_naming_convention(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """File should be named YYYYMMDD_paper_slug.json."""
        result_path = export_json(**base_export_kwargs)

        filename = os.path.basename(result_path)
        assert filename == "20260210_paper_ai-sports-device.json"

    def test_file_naming_different_date_and_slug(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """File naming adapts to different date_str and topic_slug."""
        base_export_kwargs["date_str"] = "20260315"
        base_export_kwargs["topic_slug"] = "llm-agents"

        result_path = export_json(**base_export_kwargs)

        filename = os.path.basename(result_path)
        assert filename == "20260315_paper_llm-agents.json"


class TestJsonExporterPapers:
    """Papers array validation."""

    def test_papers_serialized_with_all_fields(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Papers array should contain all paper fields."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers = data["papers"]
        assert len(papers) == 2

        first_paper = papers[0]
        expected_keys = {
            "rank", "tier", "paper_key", "title", "url", "pdf_url",
            "authors", "categories", "published_at_utc",
            "base_score", "bonus_score", "llm_adjusted",
            "embed_score", "final_score",
            "has_code", "code_url", "flags", "score_lowered",
            "summary_ko", "reason_ko", "insight_ko", "cluster_id",
        }
        assert expected_keys.issubset(set(first_paper.keys()))

    def test_paper_values_preserved(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Paper field values should be preserved correctly."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        p = data["papers"][0]
        assert p["rank"] == 1
        assert p["tier"] == 1
        assert p["paper_key"] == "arxiv:2401.12345"
        assert p["has_code"] is True
        assert p["flags"]["is_edge"] is True
        assert p["final_score"] == 87.5
        assert p["authors"] == ["Author A", "Author B"]


class TestJsonExporterClusters:
    """Clusters array validation."""

    def test_clusters_included(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Clusters array should be included in output."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["clusters"]) == 2
        assert data["clusters"][0]["cluster_id"] == 0
        assert data["clusters"][0]["representative_key"] == "arxiv:2401.12345"
        assert data["clusters"][0]["size"] == 1


class TestJsonExporterRemind:
    """Remind papers section validation."""

    def test_remind_papers_included(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Remind papers section should be included."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["remind_papers"]) == 1
        assert data["remind_papers"][0]["paper_key"] == "arxiv:2401.00001"
        assert data["remind_papers"][0]["recommend_count"] == 1


class TestJsonExporterEmptyData:
    """Empty data handling."""

    def test_empty_papers_valid_json(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Empty papers list should produce valid JSON with empty arrays."""
        base_export_kwargs["papers"] = []
        base_export_kwargs["clusters"] = []
        base_export_kwargs["remind_papers"] = []

        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["papers"] == []
        assert data["clusters"] == []
        assert data["remind_papers"] == []
        # Meta should still be present
        assert "meta" in data
        assert data["meta"]["topic_slug"] == "ai-sports-device"


class TestJsonExporterOutputDir:
    """Output directory handling."""

    def test_output_directory_created_if_not_exists(
        self, base_export_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        """Output directory should be created if it does not exist."""
        nested_dir = str(tmp_path / "nested" / "deep" / "reports")
        base_export_kwargs["output_dir"] = nested_dir

        result_path = export_json(**base_export_kwargs)

        assert os.path.isdir(nested_dir)
        assert os.path.isfile(result_path)

    def test_default_output_dir(
        self, base_export_kwargs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default output_dir should be tmp/reports when not specified."""
        # Remove output_dir to use default
        del base_export_kwargs["output_dir"]
        # Change working directory so tmp/reports is created under tmp_path
        monkeypatch.chdir(tmp_path)

        result_path = export_json(**base_export_kwargs)

        assert "tmp/reports" in result_path or "tmp\\reports" in result_path
        assert os.path.isfile(result_path)


class TestJsonExporterValidity:
    """JSON validity and encoding tests."""

    def test_output_is_valid_json(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Output file should contain valid JSON."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should not raise
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_utf8_korean_text_properly_encoded(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Korean text should be properly encoded (ensure_ascii=False)."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Korean characters should appear directly, not as \\uXXXX escapes
        assert "\uc2a4\ud3ec\uce20" in content  # "sports" in Korean (from display_title)
        assert "AI \uc2a4\ud3ec\uce20 \ubd84\uc11d \ub17c\ubb38 \uc694\uc57d" in content  # summary_ko

    def test_no_ascii_escape_for_korean(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """Raw file should not contain \\uXXXX for Korean characters."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "rb") as f:
            raw = f.read()

        # The Korean text in summary_ko should be present as UTF-8 bytes,
        # not as \\uXXXX ASCII escapes.
        # "\\uC2A4" would appear in raw bytes if ensure_ascii=True
        assert b"\\uC2A4" not in raw
        assert b"\\uc2a4" not in raw


class TestJsonExporterDateFormats:
    """Date format preservation tests."""

    def test_iso8601_datetime_strings_preserved(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """ISO 8601 datetime strings should be preserved as-is."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["meta"]
        assert meta["window_start_utc"] == "2026-02-09T01:30:00Z"
        assert meta["window_end_utc"] == "2026-02-10T02:30:00Z"

        # Paper published_at_utc preserved
        paper = data["papers"][0]
        assert paper["published_at_utc"] == "2026-02-09T00:00:00Z"

    def test_date_field_formatted_as_yyyy_mm_dd(
        self, base_export_kwargs: dict[str, Any]
    ) -> None:
        """meta.date should be formatted as YYYY-MM-DD from YYYYMMDD input."""
        result_path = export_json(**base_export_kwargs)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # date_str input is "20260210", output meta.date should be "2026-02-10"
        assert data["meta"]["date"] == "2026-02-10"
