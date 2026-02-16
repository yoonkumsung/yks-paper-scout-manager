"""Comprehensive tests for output.render.html_generator.

TASK-026 from SPEC-PAPER-001: HTML report generator with Jinja2 templates.

Test cases:
  1.  Report generation - Valid HTML generated with correct content
  2.  Autoescape verification - XSS attempt in title/abstract is escaped
  3.  Tier 1 rendering - Full paper card with all fields
  4.  Tier 2 rendering - Compact format
  5.  Tab structure - Both tabs present
  6.  Remind section - recommend_count displayed correctly
  7.  0-result handling - Empty papers generates valid HTML with message
  8.  Score bar rendering - Score values displayed correctly
  9.  Flag display - Korean flag labels rendered
  10. [완화] tag - score_lowered papers show tag
  11. Code link - Only shown when has_code AND code_url present
  12. Cluster links - Anchor links rendered for cluster mates
  13. Keyword accordion - Keywords section present
  14. Index generation - Index HTML with topic links
  15. Latest generation - latest.html generated
  16. Template loading - Templates load from correct directory
  17. File naming - {YYYYMMDD}_paper_{slug}.html format
  18. Encoding - UTF-8 Korean characters preserved
  19. No |safe filter - Verify no |safe usage in templates (security)
"""

from __future__ import annotations

import os
import glob as glob_mod
from typing import Any, Dict, List

import pytest

from output.render.html_generator import (
    generate_index_html,
    generate_latest_html,
    generate_report_html,
)


# ---------------------------------------------------------------------------
# Template directory fixture
# ---------------------------------------------------------------------------

# Templates live at project root /templates/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEMPLATE_DIR = os.path.join(_PROJECT_ROOT, "templates")


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
    papers: "List[dict] | None" = None,
    clusters: "List[dict] | None" = None,
    remind_papers: "List[dict] | None" = None,
    meta_overrides: "dict | None" = None,
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
# 1. Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """1. Report generation: Valid HTML generated with correct content."""

    def test_generates_valid_html_file(self, tmp_path: Any) -> None:
        """Should create an HTML file on disk."""
        data = _make_report_data(papers=[_make_paper()])
        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        assert os.path.isfile(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_report_contains_display_title(self, tmp_path: Any) -> None:
        """Report should contain the display title."""
        data = _make_report_data()
        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "26년 02월 10일 화요일" in content
        assert "AI 스포츠 디바이스" in content


# ---------------------------------------------------------------------------
# 2. Autoescape verification
# ---------------------------------------------------------------------------


class TestAutoescapeVerification:
    """2. Autoescape: XSS attempt in title/abstract is escaped."""

    def test_xss_in_title_is_escaped(self, tmp_path: Any) -> None:
        """Script tags in title should be escaped, not rendered."""
        malicious_title = '<script>alert("XSS")</script>Malicious Paper'
        paper = _make_paper(title=malicious_title)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        # Raw script tag must NOT appear
        assert "<script>" not in content
        # Escaped version must appear
        assert "&lt;script&gt;" in content

    def test_xss_in_summary_is_escaped(self, tmp_path: Any) -> None:
        """Script tags in summary should be escaped."""
        paper = _make_paper(summary_ko='<img src=x onerror="alert(1)">')
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert 'onerror="alert(1)"' not in content
        assert "&lt;img" in content


# ---------------------------------------------------------------------------
# 3. Tier 1 rendering
# ---------------------------------------------------------------------------


class TestTier1Rendering:
    """3. Tier 1 rendering: Full paper card with all fields."""

    def test_tier1_has_all_fields(self, tmp_path: Any) -> None:
        """Tier 1 paper card should display all required fields."""
        paper = _make_paper(rank=1, tier=1)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        # Rank
        assert "1위" in content
        # Title with arXiv link
        assert "Edge-Optimized Real-Time Pose Estimation" in content
        assert "https://arxiv.org/abs/2602.12345" in content
        # Score
        assert "87.5" in content
        # Categories
        assert "cs.CV" in content
        assert "cs.AI" in content
        # Summary accordion
        assert "개요" in content
        assert "엣지 디바이스에서" in content
        # Reason
        assert "선정 근거" in content
        assert "스포츠 분석 디바이스에 직접 적용" in content
        # Insight
        assert "활용 인사이트" in content
        assert "모바일 카메라 기반" in content
        # PDF link
        assert "https://arxiv.org/pdf/2602.12345" in content
        assert "PDF" in content


# ---------------------------------------------------------------------------
# 4. Tier 2 rendering
# ---------------------------------------------------------------------------


class TestTier2Rendering:
    """4. Tier 2 rendering: Compact format."""

    def test_tier2_compact_card(self, tmp_path: Any) -> None:
        """Tier 2 paper should use compact card format."""
        paper = _make_paper(
            rank=31,
            tier=2,
            title="Lightweight Feature Extraction",
            paper_key="arxiv:2602.99999",
            url="https://arxiv.org/abs/2602.99999",
            final_score=68.2,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "31위" in content
        assert "Lightweight Feature Extraction" in content
        assert "68.2" in content
        # Should use compact-card class
        assert "compact-card" in content


# ---------------------------------------------------------------------------
# 5. Tab structure
# ---------------------------------------------------------------------------


class TestTabStructure:
    """5. Tab structure: Both tabs present."""

    def test_both_tabs_present(self, tmp_path: Any) -> None:
        """Report should have both tabs."""
        data = _make_report_data(papers=[_make_paper()])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "오늘의 논문" in content
        assert "다시 보기" in content
        assert 'id="tab-today"' in content
        assert 'id="tab-remind"' in content


# ---------------------------------------------------------------------------
# 6. Remind section
# ---------------------------------------------------------------------------


class TestRemindSection:
    """6. Remind section: recommend_count displayed correctly."""

    def test_remind_recommend_count(self, tmp_path: Any) -> None:
        """Remind papers should show recommend count."""
        remind = {
            "paper_key": "arxiv:2601.11111",
            "title": "Previously Recommended Paper",
            "url": "https://arxiv.org/abs/2601.11111",
            "final_score": 85.0,
            "recommend_count": 2,
            "summary_ko": "이전에 추천된 논문입니다.",
            "reason_ko": "여전히 관련성이 높습니다.",
            "is_remind": True,
        }
        data = _make_report_data(remind_papers=[remind])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "Previously Recommended Paper" in content
        assert "2회째 추천" in content

    def test_remind_with_three_exposures(self, tmp_path: Any) -> None:
        """Should display correct count for multiple exposures."""
        remind = {
            "paper_key": "arxiv:2601.22222",
            "title": "Multi-Exposure Paper",
            "url": "https://arxiv.org/abs/2601.22222",
            "final_score": 78.0,
            "recommend_count": 3,
            "summary_ko": "여러 번 추천된 논문입니다.",
            "reason_ko": "핵심 연구입니다.",
            "is_remind": True,
        }
        data = _make_report_data(remind_papers=[remind])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "3회째 추천" in content


# ---------------------------------------------------------------------------
# 7. 0-result handling
# ---------------------------------------------------------------------------


class TestZeroResultHandling:
    """7. 0-result handling: Empty papers generates valid HTML with message."""

    def test_empty_papers_shows_message(self, tmp_path: Any) -> None:
        """When papers is empty, show friendly message."""
        data = _make_report_data(papers=[], remind_papers=[])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert os.path.isfile(path)
        assert "<!DOCTYPE html>" in content
        assert "오늘은 새로운 논문이 없습니다" in content

    def test_empty_papers_with_remind_still_works(self, tmp_path: Any) -> None:
        """When papers empty but remind_papers exist, both should render."""
        remind = {
            "paper_key": "arxiv:2601.33333",
            "title": "Remind Only Paper",
            "url": "https://arxiv.org/abs/2601.33333",
            "final_score": 80.0,
            "recommend_count": 1,
            "summary_ko": "리마인드 전용.",
            "reason_ko": "",
            "is_remind": True,
        }
        data = _make_report_data(papers=[], remind_papers=[remind])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "오늘은 새로운 논문이 없습니다" in content
        assert "Remind Only Paper" in content


# ---------------------------------------------------------------------------
# 8. Score bar rendering
# ---------------------------------------------------------------------------


class TestScoreBarRendering:
    """8. Score bar rendering: Score values displayed correctly."""

    def test_score_values_displayed(self, tmp_path: Any) -> None:
        """Score bar should show final score, base, and bonus."""
        paper = _make_paper(
            llm_base_score=82,
            bonus_score=13,
            llm_adjusted=95,
            final_score=87.5,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "87.5" in content
        assert "base:82" in content
        assert "bonus:+13" in content

    def test_score_bar_fill_width(self, tmp_path: Any) -> None:
        """Score bar should have width percentage based on score."""
        paper = _make_paper(final_score=75.0)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "width: 75.0%" in content


# ---------------------------------------------------------------------------
# 9. Flag display
# ---------------------------------------------------------------------------


class TestFlagDisplay:
    """9. Flag display: Korean flag labels rendered."""

    def test_edge_flag_displayed(self, tmp_path: Any) -> None:
        """is_edge flag should display as Korean label."""
        paper = _make_paper(
            flags={"is_edge": True, "is_realtime": False},
            has_code=False,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "엣지" in content

    def test_realtime_flag_displayed(self, tmp_path: Any) -> None:
        """is_realtime flag should display as Korean label."""
        paper = _make_paper(
            flags={"is_edge": False, "is_realtime": True},
            has_code=False,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "실시간" in content

    def test_code_flag_displayed(self, tmp_path: Any) -> None:
        """has_code flag should display as Korean label."""
        paper = _make_paper(
            flags={"is_edge": False, "is_realtime": False},
            has_code=True,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "코드" in content

    def test_all_flags_displayed(self, tmp_path: Any) -> None:
        """All flags should appear when all are active."""
        paper = _make_paper(
            flags={"is_edge": True, "is_realtime": True},
            has_code=True,
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "엣지" in content
        assert "실시간" in content
        assert "코드" in content


# ---------------------------------------------------------------------------
# 10. [완화] tag
# ---------------------------------------------------------------------------


class TestLoweredTag:
    """10. [완화] tag: score_lowered papers show tag."""

    def test_score_lowered_shows_tag(self, tmp_path: Any) -> None:
        """Paper with score_lowered=True should have [완화] tag."""
        paper = _make_paper(score_lowered=True, final_score=45.0)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "[완화]" in content

    def test_score_not_lowered_no_tag(self, tmp_path: Any) -> None:
        """Paper with score_lowered=False should NOT have [완화] tag."""
        paper = _make_paper(score_lowered=False)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "[완화]" not in content


# ---------------------------------------------------------------------------
# 11. Code link
# ---------------------------------------------------------------------------


class TestCodeLink:
    """11. Code link: Only shown when has_code AND code_url present."""

    def test_code_link_shown(self, tmp_path: Any) -> None:
        """Code link should appear when has_code=True and code_url is set."""
        paper = _make_paper(
            has_code=True, code_url="https://github.com/example/repo"
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "https://github.com/example/repo" in content
        assert "Code" in content

    def test_code_link_hidden_no_code(self, tmp_path: Any) -> None:
        """Code link should NOT appear when has_code=False."""
        paper = _make_paper(has_code=False, code_url=None)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        # No "Code" link should appear in the paper-links section
        assert "github.com/example/repo" not in content

    def test_code_link_hidden_has_code_but_no_url(self, tmp_path: Any) -> None:
        """Code link should NOT appear when has_code=True but code_url missing."""
        paper = _make_paper(has_code=True, code_url=None)
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        # No github link should be in the output
        assert "github.com" not in content


# ---------------------------------------------------------------------------
# 12. Cluster links
# ---------------------------------------------------------------------------


class TestClusterLinks:
    """12. Cluster links: Anchor links rendered for cluster mates."""

    def test_cluster_mates_rendered(self, tmp_path: Any) -> None:
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

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

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

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "같은 클러스터" not in content


# ---------------------------------------------------------------------------
# 13. Keyword accordion
# ---------------------------------------------------------------------------


class TestKeywordAccordion:
    """13. Keyword accordion: Keywords section present."""

    def test_keywords_in_accordion(self, tmp_path: Any) -> None:
        """Keywords should appear inside a collapsible details element."""
        data = _make_report_data(
            meta_overrides={
                "keywords_used": [
                    "automatic cinematography",
                    "sports highlight",
                ]
            }
        )

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        # Keywords section heading
        assert "검색 키워드" in content
        # Actual keywords
        assert "automatic cinematography" in content
        assert "sports highlight" in content
        # Should be in a <details> element
        assert "<details>" in content

    def test_no_keywords_no_section(self, tmp_path: Any) -> None:
        """When keywords_used is empty, no keyword section rendered."""
        data = _make_report_data(meta_overrides={"keywords_used": []})

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "검색 키워드" not in content


# ---------------------------------------------------------------------------
# 14. Index generation
# ---------------------------------------------------------------------------


class TestIndexGeneration:
    """14. Index generation: Index HTML with topic links."""

    def test_index_contains_topic_links(self, tmp_path: Any) -> None:
        """Index page should list report links."""
        reports = [
            {
                "topic_slug": "ai-sports-device",
                "topic_name": "AI 스포츠 디바이스",
                "date": "2026-02-10",
                "filepath": "reports/2026-02-10/20260210_paper_ai-sports-device.html",
            },
            {
                "topic_slug": "nlp-research",
                "topic_name": "NLP 연구",
                "date": "2026-02-09",
                "filepath": "reports/2026-02-09/20260209_paper_nlp-research.html",
            },
        ]

        path = generate_index_html(
            reports,
            output_dir=str(tmp_path),
            template_dir=_TEMPLATE_DIR,
        )
        content = open(path, "r", encoding="utf-8").read()

        assert os.path.basename(path) == "index.html"
        assert "AI 스포츠 디바이스" in content
        assert "NLP 연구" in content
        assert "2026-02-10" in content
        assert "<!DOCTYPE html>" in content

    def test_empty_index(self, tmp_path: Any) -> None:
        """Index with no reports should render without error."""
        path = generate_index_html(
            [],
            output_dir=str(tmp_path),
            template_dir=_TEMPLATE_DIR,
        )
        content = open(path, "r", encoding="utf-8").read()

        assert os.path.isfile(path)
        assert "<!DOCTYPE html>" in content


# ---------------------------------------------------------------------------
# 15. Latest generation
# ---------------------------------------------------------------------------


class TestLatestGeneration:
    """15. Latest generation: latest.html generated."""

    def test_latest_html_generated(self, tmp_path: Any) -> None:
        """latest.html should be generated."""
        data = _make_report_data(papers=[_make_paper()])

        path = generate_latest_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        assert os.path.basename(path) == "latest.html"
        assert os.path.isfile(path)
        content = open(path, "r", encoding="utf-8").read()
        assert "<!DOCTYPE html>" in content

    def test_latest_html_overwrites(self, tmp_path: Any) -> None:
        """Calling generate_latest_html twice should overwrite."""
        data1 = _make_report_data(
            papers=[_make_paper(title="First Report")],
            meta_overrides={"display_title": "First"},
        )
        data2 = _make_report_data(
            papers=[_make_paper(title="Second Report")],
            meta_overrides={"display_title": "Second"},
        )

        generate_latest_html(
            data1, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        path = generate_latest_html(
            data2, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        content = open(path, "r", encoding="utf-8").read()
        assert "Second Report" in content


# ---------------------------------------------------------------------------
# 16. Template loading
# ---------------------------------------------------------------------------


class TestTemplateLoading:
    """16. Template loading: Templates load from correct directory."""

    def test_templates_load_from_project_templates_dir(
        self, tmp_path: Any
    ) -> None:
        """Templates should load from the project templates directory."""
        data = _make_report_data()

        # This should not raise TemplateNotFound
        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        assert os.path.isfile(path)

    def test_missing_template_dir_raises(self, tmp_path: Any) -> None:
        """Missing template directory should raise."""
        data = _make_report_data()

        with pytest.raises(Exception):
            generate_report_html(
                data,
                output_dir=str(tmp_path),
                template_dir="/nonexistent/templates",
            )


# ---------------------------------------------------------------------------
# 17. File naming
# ---------------------------------------------------------------------------


class TestFileNaming:
    """17. File naming: {YYYYMMDD}_paper_{slug}.html format."""

    def test_file_name_format(self, tmp_path: Any) -> None:
        """Generated file should follow naming convention."""
        data = _make_report_data(
            meta_overrides={
                "date": "2026-02-10",
                "topic_slug": "ai-sports-device",
            }
        )

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        filename = os.path.basename(path)
        assert filename == "20260210_paper_ai-sports-device.html"

    def test_different_slug_in_filename(self, tmp_path: Any) -> None:
        """Different slug should produce different filename."""
        data = _make_report_data(
            meta_overrides={
                "date": "2026-03-15",
                "topic_slug": "nlp-research",
            }
        )

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )

        filename = os.path.basename(path)
        assert filename == "20260315_paper_nlp-research.html"


# ---------------------------------------------------------------------------
# 18. Encoding
# ---------------------------------------------------------------------------


class TestEncoding:
    """18. Encoding: UTF-8 Korean characters preserved."""

    def test_korean_characters_preserved(self, tmp_path: Any) -> None:
        """Korean text should be properly preserved in output."""
        paper = _make_paper(
            summary_ko="한국어 요약 텍스트입니다. 유니코드 테스트 중.",
            reason_ko="한국어 근거 텍스트입니다.",
            insight_ko="한국어 인사이트 텍스트입니다.",
        )
        data = _make_report_data(papers=[paper])

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "한국어 요약 텍스트입니다" in content
        assert "한국어 근거 텍스트입니다" in content
        assert "한국어 인사이트 텍스트입니다" in content

    def test_meta_charset_utf8(self, tmp_path: Any) -> None:
        """HTML should declare UTF-8 charset."""
        data = _make_report_data()

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert 'charset="UTF-8"' in content or "charset=UTF-8" in content


# ---------------------------------------------------------------------------
# 19. No |safe filter
# ---------------------------------------------------------------------------


class TestNoSafeFilter:
    """19. No |safe filter: Verify no |safe usage in templates (security)."""

    def test_no_safe_filter_in_templates(self) -> None:
        """Templates must not use the |safe filter."""
        template_files = glob_mod.glob(
            os.path.join(_TEMPLATE_DIR, "*.j2")
        )
        assert len(template_files) > 0, "No template files found"

        for template_file in template_files:
            content = open(template_file, "r", encoding="utf-8").read()
            assert "|safe" not in content, (
                "Found |safe filter in %s" % template_file
            )
            assert "| safe" not in content, (
                "Found | safe filter in %s" % template_file
            )


# ---------------------------------------------------------------------------
# Additional: Footer rendering
# ---------------------------------------------------------------------------


class TestFooterRendering:
    """Footer should contain arXiv notice and run meta."""

    def test_footer_has_arxiv_notice(self, tmp_path: Any) -> None:
        """Footer must contain the arXiv API usage notice."""
        data = _make_report_data()

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "arXiv API" in content
        assert "Thank you to arXiv" in content

    def test_footer_has_run_meta(self, tmp_path: Any) -> None:
        """Footer should show run_id and embedding_mode."""
        data = _make_report_data(
            meta_overrides={"run_id": 128, "embedding_mode": "en_synthetic"}
        )

        path = generate_report_html(
            data, output_dir=str(tmp_path), template_dir=_TEMPLATE_DIR
        )
        content = open(path, "r", encoding="utf-8").read()

        assert "run_id: 128" in content
        assert "en_synthetic" in content


# ---------------------------------------------------------------------------
# Additional: Output directory creation
# ---------------------------------------------------------------------------


class TestOutputDirectory:
    """Output directory creation."""

    def test_creates_output_dir_if_missing(self, tmp_path: Any) -> None:
        """Should create output directory if it doesn't exist."""
        out_dir = os.path.join(str(tmp_path), "nested", "reports")
        data = _make_report_data()

        path = generate_report_html(
            data, output_dir=out_dir, template_dir=_TEMPLATE_DIR
        )

        assert os.path.isfile(path)
        assert os.path.isdir(out_dir)
