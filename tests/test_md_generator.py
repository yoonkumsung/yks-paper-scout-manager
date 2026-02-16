"""Comprehensive tests for output.render.md_generator.

TASK-025 from SPEC-PAPER-001: Markdown report generator for Paper Scout.

Test cases:
  1. Tier 1 format - All fields present in correct format
  2. Tier 2 format - Compact format with single line
  3. File naming - YYYYMMDD_paper_slug.md
  4. Header content - Title, stats, window info
  5. Flag display - Korean flag names correct
  6. Code link - Shown only when has_code and code_url
  7. Cluster links - "같은 클러스터: #N위" format
  8. score_lowered tag - [완화] tag present
  9. Remind section - Separate section for remind papers
  10. Empty papers - Report still generated with header
  11. arXiv notice - Present at footer
  12. UTF-8 output - Korean text properly written
  13. Bonus breakdown - "base:82 + bonus:+13" format
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from output.render.md_generator import generate_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(**overrides: Any) -> dict:
    """Build a minimal meta dict for testing."""
    base: Dict[str, Any] = {
        "topic_name": "AI 스포츠 디바이스/플랫폼",
        "topic_slug": "ai-sports-device",
        "date": "2026-02-10",
        "display_title": "26년 02월 10일 화요일 - AI 스포츠 디바이스 arXiv 논문 정리",
        "window_start_utc": "2026-02-09T01:30:00Z",
        "window_end_utc": "2026-02-10T02:30:00Z",
        "embedding_mode": "en_synthetic",
        "scoring_weights": {"llm": 0.55, "embedding": 0.35, "recency": 0.10},
        "total_collected": 347,
        "total_filtered": 156,
        "total_discarded": 23,
        "total_scored": 133,
        "total_output": 42,
        "threshold_used": 60,
        "threshold_lowered": False,
        "run_id": 128,
        "keywords_used": [
            "automatic cinematography",
            "sports highlight",
            "pose estimation",
        ],
    }
    base.update(overrides)
    return base


def _make_paper(
    rank: int = 1,
    tier: int = 1,
    title: str = "Edge-Optimized Real-Time Pose Estimation",
    paper_key: str = "arxiv:2602.12345",
    **overrides: Any,
) -> dict:
    """Build a minimal paper dict for testing."""
    base: Dict[str, Any] = {
        "paper_key": paper_key,
        "rank": rank,
        "tier": tier,
        "title": title,
        "url": "https://arxiv.org/abs/2602.12345",
        "pdf_url": "https://arxiv.org/pdf/2602.12345",
        "categories": ["cs.CV", "cs.AI"],
        "published_at_utc": "2026-02-09",
        "has_code": True,
        "code_url": "https://github.com/example/repo",
        "llm_base_score": 82,
        "bonus_score": 13,
        "llm_adjusted": 95,
        "final_score": 87.5,
        "score_lowered": False,
        "flags": {
            "is_edge": True,
            "is_realtime": True,
            "mentions_code": True,
            "is_metaphorical": False,
        },
        "summary_ko": "이 논문은 엣지 디바이스에서 실시간 포즈 추정을 위한 경량화된 모델을 제안합니다.",
        "reason_ko": "스포츠 분석 디바이스에 직접 적용 가능한 핵심 기술입니다.",
        "insight_ko": "모바일 카메라 기반 실시간 포즈 분석에 활용할 수 있습니다.",
    }
    base.update(overrides)
    return base


def _make_report_data(
    papers: List[dict] | None = None,
    clusters: List[dict] | None = None,
    remind_papers: List[dict] | None = None,
    meta_overrides: dict | None = None,
) -> dict:
    """Build a full report_data dict."""
    meta = _make_meta(**(meta_overrides or {}))
    return {
        "meta": meta,
        "papers": papers or [],
        "clusters": clusters or [],
        "remind_papers": remind_papers or [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTier1Format:
    """1. Tier 1 format: All fields present in correct format."""

    def test_tier1_all_fields_present(self, tmp_path: Any) -> None:
        """Tier 1 paper should contain all required fields."""
        paper = _make_paper(rank=1, tier=1)
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        # Title line
        assert "## 1위: Edge-Optimized Real-Time Pose Estimation" in content
        # arXiv link
        assert "- arXiv: https://arxiv.org/abs/2602.12345" in content
        # PDF link
        assert "- PDF: https://arxiv.org/pdf/2602.12345" in content
        # Published date
        assert "- 발행일: 2026-02-09" in content
        # Categories
        assert "- 카테고리: cs.CV, cs.AI" in content
        # Summary section
        assert "**개요**" in content
        assert "이 논문은 엣지 디바이스에서" in content
        # Reason section
        assert "**선정 근거**" in content
        assert "스포츠 분석 디바이스에 직접 적용" in content
        # Insight section
        assert "**활용 인사이트**" in content
        assert "모바일 카메라 기반 실시간" in content


class TestTier2Format:
    """2. Tier 2 format: Compact format with single line."""

    def test_tier2_compact_format(self, tmp_path: Any) -> None:
        """Tier 2 paper should use compact format."""
        paper = _make_paper(
            rank=31,
            tier=2,
            title="Lightweight Feature Extraction",
            paper_key="arxiv:2602.99999",
            url="https://arxiv.org/abs/2602.99999",
            final_score=68.2,
            summary_ko="경량 특징 추출 기법을 제안합니다.",
            reason_ko="효율적인 추론을 위한 기법입니다.",
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        # Tier 2 title
        assert "## 31위: Lightweight Feature Extraction" in content
        # Compact arXiv line with date and score
        assert "arXiv:" in content
        assert "2026-02-09" in content
        assert "68.2" in content
        # Summary present
        assert "경량 특징 추출 기법을 제안합니다." in content
        # Reason with arrow prefix
        assert "-> " in content
        assert "효율적인 추론을 위한 기법입니다." in content
        # Should NOT have full sections
        assert "**개요**" not in content or content.count("**개요**") == 0


class TestFileNaming:
    """3. File naming: YYYYMMDD_paper_slug.md."""

    def test_file_name_format(self, tmp_path: Any) -> None:
        """Generated file should follow YYYYMMDD_paper_slug.md pattern."""
        data = _make_report_data(
            meta_overrides={
                "date": "2026-02-10",
                "topic_slug": "ai-sports-device",
            }
        )

        path = generate_markdown(data, output_dir=str(tmp_path))

        filename = os.path.basename(path)
        assert filename == "20260210_paper_ai-sports-device.md"

    def test_output_file_exists(self, tmp_path: Any) -> None:
        """Output file must actually exist on disk."""
        data = _make_report_data()

        path = generate_markdown(data, output_dir=str(tmp_path))

        assert os.path.isfile(path)


class TestHeaderContent:
    """4. Header content: Title, stats, window info."""

    def test_header_has_title(self, tmp_path: Any) -> None:
        """Header should contain the display title."""
        data = _make_report_data()

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "26년 02월 10일 화요일 - AI 스포츠 디바이스 arXiv 논문 정리" in content

    def test_header_has_stats(self, tmp_path: Any) -> None:
        """Header should contain paper stats."""
        data = _make_report_data(
            meta_overrides={
                "total_collected": 347,
                "total_output": 42,
            }
        )

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "347" in content
        assert "42" in content

    def test_header_has_window_info(self, tmp_path: Any) -> None:
        """Header should contain window information."""
        data = _make_report_data()

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "2026-02-09" in content or "02-09" in content
        assert "2026-02-10" in content or "02-10" in content


class TestFlagDisplay:
    """5. Flag display: Korean flag names correct."""

    def test_edge_flag_korean(self, tmp_path: Any) -> None:
        """is_edge flag should display as '엣지'."""
        paper = _make_paper(
            flags={"is_edge": True, "is_realtime": False, "mentions_code": False, "is_metaphorical": False},
            has_code=False,
            code_url=None,
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "엣지" in content

    def test_realtime_flag_korean(self, tmp_path: Any) -> None:
        """is_realtime flag should display as '실시간'."""
        paper = _make_paper(
            flags={"is_edge": False, "is_realtime": True, "mentions_code": False, "is_metaphorical": False},
            has_code=False,
            code_url=None,
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "실시간" in content

    def test_code_flag_korean(self, tmp_path: Any) -> None:
        """has_code flag should display as '코드 공개'."""
        paper = _make_paper(
            flags={"is_edge": False, "is_realtime": False, "mentions_code": False, "is_metaphorical": False},
            has_code=True,
            code_url="https://github.com/example/repo",
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "코드 공개" in content

    def test_all_flags_displayed(self, tmp_path: Any) -> None:
        """All applicable flags should appear together."""
        paper = _make_paper(
            flags={"is_edge": True, "is_realtime": True, "mentions_code": True, "is_metaphorical": False},
            has_code=True,
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "엣지" in content
        assert "실시간" in content
        assert "코드 공개" in content


class TestCodeLink:
    """6. Code link: Shown only when has_code and code_url."""

    def test_code_link_shown_when_present(self, tmp_path: Any) -> None:
        """Code link should appear when has_code=True and code_url is set."""
        paper = _make_paper(has_code=True, code_url="https://github.com/example/repo")
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "https://github.com/example/repo" in content

    def test_code_link_hidden_when_no_code(self, tmp_path: Any) -> None:
        """Code link should NOT appear when has_code=False."""
        paper = _make_paper(has_code=False, code_url=None)
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        # The "코드:" line should not exist
        lines = content.split("\n")
        code_lines = [ln for ln in lines if ln.strip().startswith("- 코드:")]
        assert len(code_lines) == 0

    def test_code_link_hidden_when_has_code_but_no_url(self, tmp_path: Any) -> None:
        """Code link should NOT appear when has_code=True but code_url is None."""
        paper = _make_paper(has_code=True, code_url=None)
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        lines = content.split("\n")
        code_lines = [ln for ln in lines if ln.strip().startswith("- 코드:")]
        assert len(code_lines) == 0


class TestClusterLinks:
    """7. Cluster links: "같은 클러스터: #N위" format."""

    def test_cluster_links_present(self, tmp_path: Any) -> None:
        """Cluster members should be listed as '#N위' links."""
        papers = [
            _make_paper(rank=1, paper_key="arxiv:001"),
            _make_paper(rank=3, paper_key="arxiv:003", title="Paper Three"),
            _make_paper(rank=7, paper_key="arxiv:007", title="Paper Seven"),
        ]
        clusters = [
            {
                "cluster_id": 0,
                "member_keys": ["arxiv:001", "arxiv:003", "arxiv:007"],
                "representative_key": "arxiv:001",
                "size": 3,
            }
        ]
        data = _make_report_data(papers=papers, clusters=clusters)

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        # Paper at rank 1 should show cluster mates #3, #7
        assert "#3위" in content
        assert "#7위" in content

    def test_no_cluster_links_for_single_member(self, tmp_path: Any) -> None:
        """Single-member cluster should not show cluster links."""
        papers = [_make_paper(rank=1, paper_key="arxiv:001")]
        clusters = [
            {
                "cluster_id": 0,
                "member_keys": ["arxiv:001"],
                "representative_key": "arxiv:001",
                "size": 1,
            }
        ]
        data = _make_report_data(papers=papers, clusters=clusters)

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "같은 클러스터:" not in content


class TestScoreLowered:
    """8. score_lowered tag: [완화] tag present."""

    def test_score_lowered_tag_present(self, tmp_path: Any) -> None:
        """Paper with score_lowered=True should have [완화] tag."""
        paper = _make_paper(score_lowered=True, final_score=45.0)
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "[완화]" in content

    def test_score_lowered_tag_absent(self, tmp_path: Any) -> None:
        """Paper with score_lowered=False should NOT have [완화] tag."""
        paper = _make_paper(score_lowered=False)
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "[완화]" not in content


class TestRemindSection:
    """9. Remind section: Separate section for remind papers."""

    def test_remind_section_present(self, tmp_path: Any) -> None:
        """Remind papers should appear in a separate '다시 보기' section."""
        remind = _make_paper(
            rank=1,
            tier=1,
            title="Previously Recommended Paper",
            paper_key="arxiv:2601.11111",
        )
        remind["recommend_count"] = 2
        data = _make_report_data(remind_papers=[remind])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "다시 보기" in content
        assert "Previously Recommended Paper" in content

    def test_no_remind_section_when_empty(self, tmp_path: Any) -> None:
        """No '다시 보기' section when remind_papers is empty."""
        data = _make_report_data(remind_papers=[])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "다시 보기" not in content


class TestEmptyPapers:
    """10. Empty papers: Report still generated with header."""

    def test_empty_papers_generates_report(self, tmp_path: Any) -> None:
        """Report should still be generated even with no papers."""
        data = _make_report_data(papers=[], remind_papers=[])

        path = generate_markdown(data, output_dir=str(tmp_path))

        assert os.path.isfile(path)
        content = open(path, "r", encoding="utf-8").read()
        # Header should still be present
        assert "AI 스포츠 디바이스" in content


class TestArxivNotice:
    """11. arXiv notice: Present at footer."""

    def test_arxiv_notice_at_footer(self, tmp_path: Any) -> None:
        """Footer must contain the arXiv API usage notice."""
        data = _make_report_data()

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "arXiv API" in content
        assert "arXiv" in content
        assert "Thank you to arXiv for use of its open access interoperability" in content


class TestUtf8Output:
    """12. UTF-8 output: Korean text properly written."""

    def test_korean_text_preserved(self, tmp_path: Any) -> None:
        """Korean text in summaries should be properly written to file."""
        paper = _make_paper(
            summary_ko="한국어 요약 텍스트입니다. 유니코드 테스트 중.",
            reason_ko="한국어 근거 텍스트입니다.",
            insight_ko="한국어 인사이트 텍스트입니다.",
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "한국어 요약 텍스트입니다" in content
        assert "한국어 근거 텍스트입니다" in content
        assert "한국어 인사이트 텍스트입니다" in content


class TestBonusBreakdown:
    """13. Bonus breakdown: "base:82 + bonus:+13" format."""

    def test_bonus_breakdown_format(self, tmp_path: Any) -> None:
        """Score line should show 'base:82 + bonus:+13' breakdown."""
        paper = _make_paper(
            llm_base_score=82,
            bonus_score=13,
            llm_adjusted=95,
            final_score=87.5,
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "base:82" in content
        assert "bonus:+13" in content
        assert "llm_adjusted:95" in content
        assert "final 87.5" in content

    def test_zero_bonus_breakdown(self, tmp_path: Any) -> None:
        """Score line should correctly show zero bonus."""
        paper = _make_paper(
            llm_base_score=75,
            bonus_score=0,
            llm_adjusted=75,
            final_score=70.0,
        )
        data = _make_report_data(papers=[paper])

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "base:75" in content
        assert "bonus:+0" in content


class TestKeywordsSection:
    """Keywords section should list concepts used."""

    def test_keywords_section_present(self, tmp_path: Any) -> None:
        """Keywords section should display Agent 1 concepts."""
        data = _make_report_data(
            meta_overrides={
                "keywords_used": [
                    "automatic cinematography",
                    "sports highlight",
                    "pose estimation",
                ],
            }
        )

        path = generate_markdown(data, output_dir=str(tmp_path))
        content = open(path, "r", encoding="utf-8").read()

        assert "automatic cinematography" in content
        assert "sports highlight" in content
        assert "pose estimation" in content


class TestOutputDirectory:
    """Output directory creation."""

    def test_creates_output_dir_if_missing(self, tmp_path: Any) -> None:
        """Should create output directory if it doesn't exist."""
        out_dir = os.path.join(str(tmp_path), "nested", "reports")
        data = _make_report_data()

        path = generate_markdown(data, output_dir=out_dir)

        assert os.path.isfile(path)
        assert os.path.isdir(out_dir)
