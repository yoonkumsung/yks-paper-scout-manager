"""arXiv SourceAdapter for Paper Scout.

Collects papers from arXiv using the ``arxiv`` PyPI library, applies date
filtering, code detection, and in-run deduplication.  Queries are generated
by ``ArxivQueryBuilder`` (15-25 per run).

Reference: devspec Section 6-2.
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import arxiv

from core.models import Paper, QueryStats
from core.scoring.code_detector import CodeDetector
from core.sources.arxiv_query_builder import ArxivQueryBuilder
from core.sources.base import SourceAdapter
from core.sources.registry import SourceRegistry

logger = logging.getLogger(__name__)

# Buffer applied to both sides of the date window (minutes).
_DATE_BUFFER_MINUTES = 30

# Default maximum results per query when not configured.
_DEFAULT_MAX_RESULTS_PER_QUERY = 200


def _format_arxiv_date(dt: datetime) -> str:
    """Format a datetime to arXiv ``submittedDate`` range format.

    arXiv uses ``YYYYMMDDHHmm`` (no seconds).
    """
    return dt.strftime("%Y%m%d%H%M")


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract a clean arXiv ID (without version) from an entry_id URL.

    Args:
        entry_id: Full URL like ``http://arxiv.org/abs/2401.12345v1``.

    Returns:
        Clean ID like ``2401.12345``.
    """
    # Remove the URL prefix to get the short ID.
    short = entry_id.rsplit("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id
    # Strip trailing version number (e.g. "v1", "v2").
    return re.sub(r"v\d+$", "", short)


@SourceRegistry.register
class ArxivSourceAdapter(SourceAdapter):
    """Collect papers from arXiv via the ``arxiv`` PyPI library.

    Collection policy (devspec 6-2):
    - Client: ``num_retries=3, delay_seconds=3``
    - 15-25 queries executed sequentially
    - Date filter with +-30 min buffer
    - In-run dedup by ``paper_key``
    """

    @property
    def source_type(self) -> str:  # noqa: D401
        """Source identifier."""
        return "arxiv"

    def __init__(self) -> None:
        self.query_stats: list[QueryStats] = []

    def collect(
        self,
        agent1_output: dict[str, Any],
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        config: dict[str, Any],
    ) -> list[Paper]:
        """Collect papers from arXiv.

        Args:
            agent1_output: Keyword expansion from Agent 1.
            categories: arXiv category list (e.g. ``["cs.CV", "cs.AI"]``).
            window_start: UTC start of search window (inclusive).
            window_end: UTC end of search window (inclusive).
            config: Source-specific configuration dict.

        Returns:
            Deduplicated list of normalized ``Paper`` objects.
        """
        self.query_stats = []

        max_results_per_query: int = config.get(
            "max_results_per_query", _DEFAULT_MAX_RESULTS_PER_QUERY
        )
        # Target number of queries that return at least 1 result.
        # Once reached, remaining queries are skipped to save API calls.
        target_successful: int = config.get("target_successful_queries", 30)

        # Build queries (pool of up to 50 candidates).
        builder = ArxivQueryBuilder()
        queries = builder.build_queries(agent1_output, categories)

        # Date window with buffer for the arXiv query filter.
        buf = timedelta(minutes=_DATE_BUFFER_MINUTES)
        buffered_start = window_start - buf
        buffered_end = window_end + buf

        # Strict bounds for post-collection filtering.
        filter_start = buffered_start
        filter_end = buffered_end

        # arXiv client -- library handles its own retries; no extra sleep.
        client = arxiv.Client(
            num_retries=3,
            delay_seconds=3,
        )

        # Code detector instance (shared across all papers).
        detector = CodeDetector()

        # Delay between queries to avoid arXiv rate limiting (HTTP 429).
        query_delay: float = config.get("query_delay_seconds", 4.0)

        seen_keys: set[str] = set()
        all_papers: list[Paper] = []
        successful_queries = 0
        skipped_queries = 0

        for query_idx, query_text in enumerate(queries):
            if query_idx > 0 and query_delay > 0:
                time.sleep(query_delay)

            papers, stats = self._execute_query(
                client=client,
                query_text=query_text,
                max_results=max_results_per_query,
                buffered_start=buffered_start,
                buffered_end=buffered_end,
                filter_start=filter_start,
                filter_end=filter_end,
            )

            if stats.collected == 0:
                # Skip 0-result queries; record stats but don't count as successful
                skipped_queries += 1
                self.query_stats.append(stats)
                logger.debug(
                    "Query returned 0 results, skipping: %s", query_text[:80]
                )
                continue

            successful_queries += 1

            # Apply code detection.
            for paper in papers:
                detection = detector.detect(paper.abstract, paper.comment)
                paper.has_code = detection["has_code"]
                paper.has_code_source = detection["has_code_source"]
                paper.code_url = detection["code_url"]

            # In-run dedup by paper_key.
            for paper in papers:
                if paper.paper_key not in seen_keys:
                    seen_keys.add(paper.paper_key)
                    all_papers.append(paper)

            self.query_stats.append(stats)

            # Stop early if we've hit the target number of successful queries.
            if successful_queries >= target_successful:
                logger.info(
                    "Reached target of %d successful queries, "
                    "stopping early (%d skipped, %d remaining)",
                    target_successful,
                    skipped_queries,
                    len(queries) - successful_queries - skipped_queries,
                )
                break

        logger.info(
            "arXiv collection complete: %d papers from %d queries "
            "(%d successful, %d skipped/empty)",
            len(all_papers),
            len(queries),
            successful_queries,
            skipped_queries,
        )
        return all_papers

    # ------------------------------------------------------------------
    # Query execution with outer retry
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        client: arxiv.Client,
        query_text: str,
        max_results: int,
        buffered_start: datetime,
        buffered_end: datetime,
        filter_start: datetime,
        filter_end: datetime,
    ) -> tuple[list[Paper], QueryStats]:
        """Execute a single arXiv query with outer retry logic.

        The ``arxiv.Client`` already retries page-level failures internally.
        This method provides an *outer* retry that re-runs the full query
        when the library raises after exhausting its own retries.

        Retry policy:
        - ``HTTPError``: 3 retries, exponential backoff + jitter.
        - ``UnexpectedEmptyPageError``: 2 retries, fixed delays 3 s / 9 s.
        - Other exceptions: no retry, skip query.

        Returns:
            Tuple of (papers, query_stats).
        """
        # Append submittedDate range to query.
        date_clause = (
            f"submittedDate:[{_format_arxiv_date(buffered_start)} "
            f"TO {_format_arxiv_date(buffered_end)}]"
        )
        full_query = f"{query_text} AND {date_clause}"

        start_time = time.monotonic()
        retry_count = 0
        last_error: Optional[Exception] = None

        # Determine max retries and backoff by error type.
        # We try up to 4 times total for HTTPError (1 initial + 3 retries)
        # and up to 3 times for UnexpectedEmptyPageError (1 initial + 2 retries).
        max_total_attempts = 4  # 1 + 3 retries for worst case

        for attempt in range(max_total_attempts):
            try:
                search = arxiv.Search(
                    query=full_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                # Consume the generator to collect results.
                # Wrap in try-except to preserve partial results if
                # the arXiv server drops the connection mid-iteration.
                raw_results = []
                try:
                    for result in client.results(search):
                        raw_results.append(result)
                except Exception as gen_exc:
                    logger.warning(
                        "arXiv generator interrupted after %d results "
                        "(max_results=%d, partial data returned): %s",
                        len(raw_results), max_results, gen_exc,
                    )
                    if not raw_results:
                        raise  # Re-raise if zero results

                # Normalize to Paper objects.
                papers: list[Paper] = []
                for r in raw_results:
                    try:
                        paper = Paper.from_arxiv_result(r)
                        papers.append(paper)
                    except Exception:
                        logger.warning(
                            "Failed to normalize arXiv result: %s",
                            getattr(r, "entry_id", "unknown"),
                            exc_info=True,
                        )

                # Ensure timezone-aware comparison
                if filter_start.tzinfo is None:
                    filter_start = filter_start.replace(tzinfo=timezone.utc)
                if filter_end.tzinfo is None:
                    filter_end = filter_end.replace(tzinfo=timezone.utc)

                # Post-filter by date window (strict, with buffer).
                papers = [
                    p
                    for p in papers
                    if p.published_at_utc is not None
                    and p.published_at_utc.tzinfo is not None
                    and filter_start <= p.published_at_utc <= filter_end
                ]

                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                # Truncation detection (fallback: collected == max_results).
                collected = len(papers)
                truncated = len(raw_results) >= max_results
                total_available: Optional[int] = None
                if truncated:
                    logger.warning(
                        "Query may be truncated: collected=%d == max_results=%d "
                        "for query: %s",
                        len(raw_results),
                        max_results,
                        query_text,
                    )

                stats = QueryStats(
                    run_id=0,
                    query_text=query_text,
                    collected=collected,
                    total_available=total_available,
                    truncated=truncated,
                    retries=retry_count,
                    duration_ms=elapsed_ms,
                    exception=None,
                )
                return papers, stats

            except arxiv.UnexpectedEmptyPageError as exc:
                retry_count += 1
                last_error = exc
                # UnexpectedEmptyPageError: max 2 retries, delays 3s, 9s.
                if attempt >= 2:  # Already tried 3 times (0, 1, 2).
                    break
                delay = 3.0 * (3 ** attempt)  # 3s, 9s
                logger.warning(
                    "UnexpectedEmptyPageError on attempt %d for query '%s', "
                    "retrying in %.1fs",
                    attempt + 1,
                    query_text,
                    delay,
                )
                time.sleep(delay)

            except (arxiv.HTTPError, arxiv.ArxivError) as exc:
                retry_count += 1
                last_error = exc
                if attempt >= 3:  # Already tried 4 times (0, 1, 2, 3).
                    break
                # Exponential backoff with jitter.
                base_delay = 2 ** attempt  # 1, 2, 4
                jitter = random.uniform(0, 1)
                delay = base_delay + jitter
                logger.warning(
                    "HTTPError on attempt %d for query '%s', "
                    "retrying in %.1fs: %s",
                    attempt + 1,
                    query_text,
                    delay,
                    exc,
                )
                time.sleep(delay)

            except Exception as exc:
                # Non-retryable error: skip this query.
                last_error = exc
                retry_count += 1
                logger.error(
                    "Non-retryable error for query '%s': %s",
                    query_text,
                    exc,
                    exc_info=True,
                )
                break

        # All retries exhausted -- record failure and skip query.
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        error_str = str(last_error) if last_error else "unknown error"
        logger.error(
            "Skipping query after %d retries: '%s' -- %s",
            retry_count,
            query_text,
            error_str,
        )
        stats = QueryStats(
            run_id=0,
            query_text=query_text,
            collected=0,
            total_available=None,
            truncated=False,
            retries=retry_count,
            duration_ms=elapsed_ms,
            exception=error_str,
        )
        return [], stats
