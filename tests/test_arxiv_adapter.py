"""Tests for the arXiv SourceAdapter (core.sources.arxiv).

Covers paper normalization, query building integration, collection flow,
retry logic, truncation detection, in-run dedup, date filtering,
code detection integration, QueryStats recording, empty results,
and config defaults.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

import arxiv

from core.models import Paper, QueryStats
from core.scoring.code_detector import CodeDetector
from core.sources.arxiv import (
    ArxivSourceAdapter,
    _DEFAULT_MAX_RESULTS_PER_QUERY,
    _format_arxiv_date,
)
from core.sources.registry import SourceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arxiv_result(
    entry_id: str = "http://arxiv.org/abs/2401.12345v1",
    title: str = "Test Paper Title",
    summary: str = "This is a test abstract.",
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    published: datetime | None = None,
    updated: datetime | None = None,
    comment: str = "",
    pdf_url: str | None = "http://arxiv.org/pdf/2401.12345v1",
) -> arxiv.Result:
    """Create a mock ``arxiv.Result`` for testing."""
    if authors is None:
        authors = ["Author A", "Author B"]
    if categories is None:
        categories = ["cs.AI", "cs.LG"]
    if published is None:
        published = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    if updated is None:
        updated = datetime(2024, 1, 16, 12, 0, 0, tzinfo=timezone.utc)

    author_objs = [arxiv.Result.Author(name) for name in authors]
    result = arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=author_objs,
        categories=categories,
        published=published,
        updated=updated,
        comment=comment,
    )
    # Override pdf_url since it is calculated from links.
    result.pdf_url = pdf_url
    return result


def _make_agent1_output(
    concepts: list[dict] | None = None,
    cross_domain_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Create a minimal Agent 1 output dict."""
    if concepts is None:
        concepts = [
            {
                "name_en": "deep learning",
                "keywords": ["deep learning", "neural network", "CNN"],
            },
            {
                "name_en": "computer vision",
                "keywords": ["object detection", "image segmentation"],
            },
        ]
    return {
        "concepts": concepts,
        "cross_domain_keywords": cross_domain_keywords or ["transfer learning"],
        "exclude_keywords": exclude_keywords or [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure a clean registry for every test, then re-register."""
    SourceRegistry.clear()
    SourceRegistry.register(ArxivSourceAdapter)
    yield
    SourceRegistry.clear()


@pytest.fixture()
def adapter() -> ArxivSourceAdapter:
    """Create a fresh ArxivSourceAdapter instance."""
    return ArxivSourceAdapter()


@pytest.fixture()
def base_window() -> tuple[datetime, datetime]:
    """Standard date window for tests."""
    start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
    return start, end


# ---------------------------------------------------------------------------
# 1. Paper normalization: from_arxiv_result
# ---------------------------------------------------------------------------


class TestPaperNormalization:
    """Test ``Paper.from_arxiv_result`` with mock arxiv.Result objects."""

    def test_basic_normalization(self):
        """All fields are correctly mapped from arxiv.Result to Paper."""
        result = _make_arxiv_result()
        paper = Paper.from_arxiv_result(result)

        assert paper.source == "arxiv"
        assert paper.native_id == "2401.12345"
        assert paper.paper_key == "arxiv:2401.12345"
        assert paper.url == "http://arxiv.org/abs/2401.12345v1"
        assert paper.title == "Test Paper Title"
        assert paper.abstract == "This is a test abstract."
        assert paper.authors == ["Author A", "Author B"]
        assert paper.categories == ["cs.AI", "cs.LG"]
        assert paper.published_at_utc == datetime(
            2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
        )
        assert paper.updated_at_utc == datetime(
            2024, 1, 16, 12, 0, 0, tzinfo=timezone.utc
        )
        assert paper.pdf_url == "http://arxiv.org/pdf/2401.12345v1"

    def test_version_stripped_from_native_id(self):
        """Version suffix (e.g. v2) is stripped from native_id."""
        result = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.12345v2"
        )
        paper = Paper.from_arxiv_result(result)
        assert paper.native_id == "2401.12345"
        assert paper.paper_key == "arxiv:2401.12345"

    def test_legacy_arxiv_id_format(self):
        """Legacy IDs like quant-ph/0201082v1 are handled correctly."""
        result = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/quant-ph/0201082v1"
        )
        paper = Paper.from_arxiv_result(result)
        assert paper.native_id == "quant-ph/0201082"
        assert paper.paper_key == "arxiv:quant-ph/0201082"

    def test_newlines_stripped_from_title_and_abstract(self):
        """Newlines in title and abstract are replaced with spaces."""
        result = _make_arxiv_result(
            title="Title\nWith\nNewlines",
            summary="Abstract\nWith\nNewlines\nEverywhere",
        )
        paper = Paper.from_arxiv_result(result)
        assert "\n" not in paper.title
        assert "\n" not in paper.abstract

    def test_comment_none_when_empty(self):
        """Empty comment string becomes None."""
        result = _make_arxiv_result(comment="")
        paper = Paper.from_arxiv_result(result)
        assert paper.comment is None

    def test_comment_preserved_when_present(self):
        """Non-empty comment is preserved."""
        result = _make_arxiv_result(comment="10 pages, 5 figures")
        paper = Paper.from_arxiv_result(result)
        assert paper.comment == "10 pages, 5 figures"

    def test_naive_datetime_gets_utc_timezone(self):
        """Naive datetime is made UTC-aware."""
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)  # No tzinfo
        result = _make_arxiv_result(published=naive_dt, updated=naive_dt)
        paper = Paper.from_arxiv_result(result)
        assert paper.published_at_utc.tzinfo is not None
        assert paper.updated_at_utc.tzinfo is not None


# ---------------------------------------------------------------------------
# 2. Query building integration
# ---------------------------------------------------------------------------


class TestQueryBuildingIntegration:
    """Verify ArxivQueryBuilder is called with correct params."""

    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_query_builder_called_with_agent1_output(
        self, MockBuilder, adapter, base_window
    ):
        """QueryBuilder.build_queries receives agent1_output and categories."""
        mock_instance = MockBuilder.return_value
        mock_instance.build_queries.return_value = ["cat:cs.AI"]

        with patch.object(
            adapter,
            "_execute_query",
            return_value=([], QueryStats(run_id=0, query_text="cat:cs.AI")),
        ):
            agent1 = _make_agent1_output()
            adapter.collect(
                agent1, ["cs.AI"], base_window[0], base_window[1], {}
            )

        mock_instance.build_queries.assert_called_once_with(
            agent1, ["cs.AI"]
        )


# ---------------------------------------------------------------------------
# 3. Collection flow
# ---------------------------------------------------------------------------


class TestCollectionFlow:
    """Test full collection pipeline with mocked arxiv.Client."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_full_collection_pipeline(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Papers are collected, normalized, and returned."""
        # Set up query builder to return 1 query.
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        # Set up arxiv client to return 1 result.
        result = _make_arxiv_result(
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1
        assert papers[0].source == "arxiv"
        assert papers[0].paper_key == "arxiv:2401.12345"

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_multiple_queries_multiple_results(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Multiple queries yield combined results."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["q1", "q2"]

        r1 = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.00001v1",
            published=datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc),
        )
        r2 = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.00002v1",
            published=datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc),
        )

        # Each query returns one unique result.
        mock_client = MockClient.return_value
        mock_client.results.side_effect = [iter([r1]), iter([r2])]

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 2
        keys = {p.paper_key for p in papers}
        assert "arxiv:2401.00001" in keys
        assert "arxiv:2401.00002" in keys


# ---------------------------------------------------------------------------
# 4. Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test outer retry behavior for different error types."""

    @patch("core.sources.arxiv.time.sleep")
    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_http_error_retries_3_times(
        self, MockBuilder, MockClient, mock_sleep, adapter, base_window
    ):
        """HTTPError triggers up to 3 retries with exponential backoff."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = arxiv.HTTPError(
            "http://example.com", 503, "retry"
        )

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        # Query should fail gracefully (no papers, but no crash).
        assert papers == []
        # Should have retried (sleep called for backoff).
        assert mock_sleep.call_count == 3  # 3 retries after initial attempt
        # QueryStats should record failure.
        assert len(adapter.query_stats) == 1
        assert adapter.query_stats[0].exception is not None
        assert adapter.query_stats[0].retries > 0

    @patch("core.sources.arxiv.time.sleep")
    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_unexpected_empty_page_retries_2_times(
        self, MockBuilder, MockClient, mock_sleep, adapter, base_window
    ):
        """UnexpectedEmptyPageError triggers 2 retries at 3s, 9s."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = arxiv.UnexpectedEmptyPageError(
            "http://example.com", 0, ""
        )

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert papers == []
        # 2 retries after initial attempt.
        assert mock_sleep.call_count == 2
        # Verify delays: 3s (3*3^0), 9s (3*3^1).
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[0] == pytest.approx(3.0)
        assert delays[1] == pytest.approx(9.0)

    @patch("core.sources.arxiv.time.sleep")
    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_http_error_then_success(
        self, MockBuilder, MockClient, mock_sleep, adapter, base_window
    ):
        """Query succeeds after initial HTTPError."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        mock_client = MockClient.return_value
        mock_client.results.side_effect = [
            arxiv.HTTPError("http://example.com", 503, "retry"),
            iter([result]),
        ]

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1
        assert adapter.query_stats[0].retries == 1
        assert adapter.query_stats[0].exception is None

    @patch("core.sources.arxiv.time.sleep")
    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_non_retryable_error_skips_query(
        self, MockBuilder, MockClient, mock_sleep, adapter, base_window
    ):
        """Non-retryable errors skip the query without retry."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = ValueError("unexpected error")

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert papers == []
        # No retries for non-retryable errors.
        assert mock_sleep.call_count == 0
        assert adapter.query_stats[0].exception is not None


# ---------------------------------------------------------------------------
# 5. Truncation detection
# ---------------------------------------------------------------------------


class TestTruncationDetection:
    """Test truncation detection when results hit max_results."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_truncated_when_collected_equals_max_results(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """truncated=True when collected == max_results."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        # Return exactly max_results papers.
        max_results = 5
        results = [
            _make_arxiv_result(
                entry_id=f"http://arxiv.org/abs/2401.{i:05d}v1",
                published=datetime(
                    2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
                ),
            )
            for i in range(max_results)
        ]
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {"max_results_per_query": max_results},
        )

        assert adapter.query_stats[0].truncated is True

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_not_truncated_when_below_max_results(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """truncated=False when collected < max_results."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        results = [
            _make_arxiv_result(
                entry_id="http://arxiv.org/abs/2401.00001v1",
                published=datetime(
                    2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
                ),
            )
        ]
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {"max_results_per_query": 200},
        )

        assert adapter.query_stats[0].truncated is False


# ---------------------------------------------------------------------------
# 6. In-run dedup
# ---------------------------------------------------------------------------


class TestInRunDedup:
    """Test deduplication of papers within a single run."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_duplicate_paper_from_multiple_queries_kept_once(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Same paper returned by two queries is deduplicated."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["q1", "q2"]

        # Both queries return the same paper.
        same_result = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.12345v1",
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        mock_client = MockClient.return_value
        mock_client.results.side_effect = [
            iter([same_result]),
            iter([same_result]),
        ]

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        # Should be deduplicated to 1 paper.
        assert len(papers) == 1
        assert papers[0].paper_key == "arxiv:2401.12345"

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_different_papers_not_deduplicated(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Different papers from the same query are all kept."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["q1"]

        r1 = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.00001v1",
            published=datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc),
        )
        r2 = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.00002v1",
            published=datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone.utc),
        )

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([r1, r2])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 2


# ---------------------------------------------------------------------------
# 7. Date filtering
# ---------------------------------------------------------------------------


class TestDateFiltering:
    """Test that papers outside the window (even with buffer) are excluded."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_paper_within_window_included(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper within the buffered window is included."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_paper_outside_buffered_window_excluded(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper well outside the buffered window is excluded."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        # Paper published 2 days before window start.
        result = _make_arxiv_result(
            published=datetime(2024, 1, 13, 0, 0, 0, tzinfo=timezone.utc)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 0

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_paper_within_buffer_zone_included(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper within the 30-min buffer zone is included."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        # 15 minutes before window start (within 30-min buffer).
        result = _make_arxiv_result(
            published=base_window[0] - timedelta(minutes=15)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_paper_just_outside_buffer_excluded(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper just outside the 30-min buffer is excluded."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        # 31 minutes before window start (outside 30-min buffer).
        result = _make_arxiv_result(
            published=base_window[0] - timedelta(minutes=31)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 0


# ---------------------------------------------------------------------------
# 8. Code detection integration
# ---------------------------------------------------------------------------


class TestCodeDetectionIntegration:
    """Test that CodeDetector runs on each collected paper."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_code_detected_in_abstract(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper with GitHub URL in abstract gets has_code=True."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            summary="Code available at https://github.com/user/repo",
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1
        assert papers[0].has_code is True
        assert papers[0].code_url is not None
        assert "github.com" in papers[0].code_url

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_no_code_detected(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Paper without code signals gets has_code=False."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            summary="We present a theoretical analysis of algorithm X.",
            comment="",
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1
        assert papers[0].has_code is False

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_code_detected_in_comment(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Code detection also checks the comment field."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            summary="A plain abstract.",
            comment="Code: https://github.com/user/project",
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(papers) == 1
        assert papers[0].has_code is True


# ---------------------------------------------------------------------------
# 9. QueryStats recording
# ---------------------------------------------------------------------------


class TestQueryStatsRecording:
    """Test QueryStats creation for each query (success and failure)."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_stats_recorded_on_success(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """QueryStats are recorded with correct values on success."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        result = _make_arxiv_result(
            published=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        )
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([result])

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(adapter.query_stats) == 1
        stats = adapter.query_stats[0]
        assert stats.query_text == "cat:cs.AI"
        assert stats.collected == 1
        assert stats.retries == 0
        assert stats.exception is None
        assert stats.duration_ms >= 0

    @patch("core.sources.arxiv.time.sleep")
    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_stats_recorded_on_failure(
        self, MockBuilder, MockClient, mock_sleep, adapter, base_window
    ):
        """QueryStats record failure details when query fails."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = arxiv.HTTPError(
            "http://example.com", 500, "server error"
        )

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(adapter.query_stats) == 1
        stats = adapter.query_stats[0]
        assert stats.collected == 0
        assert stats.exception is not None
        assert stats.retries > 0

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_stats_per_query(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """One QueryStats is recorded per query."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["q1", "q2", "q3"]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert len(adapter.query_stats) == 3


# ---------------------------------------------------------------------------
# 10. Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    """No errors when queries return 0 results."""

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_empty_results_no_error(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """Empty result set returns empty list without error."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        papers = adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {},
        )

        assert papers == []
        assert len(adapter.query_stats) == 1
        assert adapter.query_stats[0].collected == 0
        assert adapter.query_stats[0].exception is None


# ---------------------------------------------------------------------------
# 11. Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Test that config defaults are applied correctly."""

    def test_default_max_results_per_query(self):
        """Default max_results_per_query is 200."""
        assert _DEFAULT_MAX_RESULTS_PER_QUERY == 200

    @patch("core.sources.arxiv.arxiv.Client")
    @patch("core.sources.arxiv.ArxivQueryBuilder")
    def test_config_override_max_results(
        self, MockBuilder, MockClient, adapter, base_window
    ):
        """max_results_per_query from config is passed to Search."""
        mock_builder = MockBuilder.return_value
        mock_builder.build_queries.return_value = ["cat:cs.AI"]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        adapter.collect(
            _make_agent1_output(),
            ["cs.AI"],
            base_window[0],
            base_window[1],
            {"max_results_per_query": 50},
        )

        # Verify Search was created with max_results=50.
        call_args = mock_client.results.call_args
        search_obj = call_args[0][0] if call_args[0] else call_args[1]["search"]
        assert search_obj.max_results == 50


# ---------------------------------------------------------------------------
# 12. Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Test that ArxivSourceAdapter is properly registered."""

    def test_registered_with_source_registry(self):
        """ArxivSourceAdapter is registered under 'arxiv'."""
        assert "arxiv" in SourceRegistry.list_types()
        assert SourceRegistry.get("arxiv") is ArxivSourceAdapter

    def test_source_type_property(self, adapter):
        """source_type property returns 'arxiv'."""
        assert adapter.source_type == "arxiv"


# ---------------------------------------------------------------------------
# 13. Date formatting helper
# ---------------------------------------------------------------------------


class TestDateFormatting:
    """Test the date formatting helper used for arXiv queries."""

    def test_format_arxiv_date(self):
        """Datetime is formatted as YYYYMMDDHHmm."""
        dt = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        assert _format_arxiv_date(dt) == "202401150930"

    def test_format_arxiv_date_midnight(self):
        """Midnight is formatted correctly."""
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert _format_arxiv_date(dt) == "202401010000"
