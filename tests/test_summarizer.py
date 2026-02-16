"""Comprehensive tests for Summarizer (Agent 3).

Covers:
  - Tier 1 batch: 5 papers get full summary (summary_ko, reason_ko, insight_ko)
  - Tier 2 batch: 10 papers get compact summary (summary_ko, reason_ko, no insight_ko)
  - Tier splitting: 30 Tier 1 papers + 70 Tier 2 papers split correctly
  - Batch sizes: Tier 1 uses 5/batch, Tier 2 uses 10/batch
  - Parse failure: 2x failure skips batch
  - Missing item re-call: Missing indices re-called once
  - Rate limiting: wait/record_call invoked per LLM call
  - Prompt construction: Verify topic and paper info in prompt
  - Empty papers: Returns empty list
  - Tier boundary: Paper at rank 30 is Tier 1, rank 31 is Tier 2
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.summarizer import Summarizer
from core.config import AppConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(agents: dict | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    if agents is None:
        agents = {
            "common": {"ignore_reasoning": True},
            "agent3": {
                "effort": "low",
                "max_tokens": 4096,
                "temperature": 0.4,
                "prompt_version": "agent3-tier1-v1",
                "tier1_batch_size": 5,
                "tier2_batch_size": 10,
            },
        }

    return AppConfig(
        app={},
        llm={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "test/model:free",
            "app_url": "https://github.com/test",
            "app_title": "Test",
            "retry": {"max_retries": 1, "backoff_base": 0, "jitter": False},
        },
        agents=agents,
        sources={},
        filter={},
        embedding={},
        scoring={
            "weights": {
                "embedding_on": {"llm": 0.5, "embed": 0.3, "recency": 0.2},
                "embedding_off": {"llm": 0.8, "recency": 0.2},
            },
            "discard_cutoff": 20,
            "max_output": 100,
        },
        remind={},
        clustering={},
        topics=[],
        output={},
        notifications={},
        database={"path": ":memory:"},
        weekly={},
        local_ui={},
    )


def _make_ranked_paper(
    rank: int,
    *,
    title: str | None = None,
    abstract: str | None = None,
    base_score: int = 75,
    brief_reason: str = "Relevant paper.",
) -> dict:
    """Create a test ranked paper dict."""
    return {
        "paper_key": f"arxiv:2401.{rank:05d}",
        "rank": rank,
        "title": title or f"Test Paper {rank}",
        "abstract": abstract or f"Abstract for test paper {rank}.",
        "base_score": base_score,
        "brief_reason": brief_reason,
    }


def _make_ranked_papers(count: int, start_rank: int = 1) -> list[dict]:
    """Create a list of test ranked papers."""
    return [
        _make_ranked_paper(start_rank + i) for i in range(count)
    ]


def _make_tier1_llm_item(
    index: int,
    *,
    summary_ko: str = "Tier1 summary",
    reason_ko: str = "Tier1 reason",
    insight_ko: str = "Tier1 insight",
) -> dict:
    """Create a single Tier 1 LLM output item."""
    return {
        "index": index,
        "summary_ko": summary_ko,
        "reason_ko": reason_ko,
        "insight_ko": insight_ko,
    }


def _make_tier2_llm_item(
    index: int,
    *,
    summary_ko: str = "Tier2 summary",
    reason_ko: str = "Tier2 reason",
) -> dict:
    """Create a single Tier 2 LLM output item."""
    return {
        "index": index,
        "summary_ko": summary_ko,
        "reason_ko": reason_ko,
    }


def _make_tier1_response(count: int, start_index: int = 1) -> list[dict]:
    """Create a Tier 1 LLM response list."""
    return [
        _make_tier1_llm_item(start_index + i)
        for i in range(count)
    ]


def _make_tier2_response(count: int, start_index: int = 1) -> list[dict]:
    """Create a Tier 2 LLM response list."""
    return [
        _make_tier2_llm_item(start_index + i)
        for i in range(count)
    ]


def _make_summarizer(config: AppConfig | None = None) -> Summarizer:
    """Instantiate a Summarizer with mocked client."""
    cfg = config or _make_config()
    client = MagicMock()
    return Summarizer(cfg, client)


def _make_rate_limiter() -> MagicMock:
    """Create a mock RateLimiter."""
    rl = MagicMock()
    rl.wait = MagicMock()
    rl.record_call = MagicMock()
    return rl


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestTier1Batch:
    """Test Tier 1: 5 papers get full summary (summary_ko, reason_ko, insight_ko)."""

    def test_tier1_papers_get_full_summary(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(5)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 5
        for r in results:
            assert "summary_ko" in r
            assert "reason_ko" in r
            assert "insight_ko" in r

    def test_tier1_summary_values_populated(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = [
            _make_tier1_llm_item(1, summary_ko="S1", reason_ko="R1", insight_ko="I1"),
            _make_tier1_llm_item(2, summary_ko="S2", reason_ko="R2", insight_ko="I2"),
            _make_tier1_llm_item(3, summary_ko="S3", reason_ko="R3", insight_ko="I3"),
        ]

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert results[0]["summary_ko"] == "S1"
        assert results[0]["reason_ko"] == "R1"
        assert results[0]["insight_ko"] == "I1"

    def test_tier1_prompt_version_set(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(2, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(2)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        for r in results:
            assert r["prompt_ver_summ"] == "agent3-tier1-v1"


class TestTier2Batch:
    """Test Tier 2: 10 papers get compact summary (summary_ko, reason_ko, no insight_ko)."""

    def test_tier2_papers_get_compact_summary(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(10, start_rank=31)
        rl = _make_rate_limiter()
        llm_response = _make_tier2_response(10)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 10
        for r in results:
            assert "summary_ko" in r
            assert "reason_ko" in r
            assert "insight_ko" not in r

    def test_tier2_summary_values_populated(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=31)
        rl = _make_rate_limiter()
        llm_response = [
            _make_tier2_llm_item(1, summary_ko="S1", reason_ko="R1"),
            _make_tier2_llm_item(2, summary_ko="S2", reason_ko="R2"),
            _make_tier2_llm_item(3, summary_ko="S3", reason_ko="R3"),
        ]

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert results[0]["summary_ko"] == "S1"
        assert results[0]["reason_ko"] == "R1"


class TestTierSplitting:
    """Test 30 Tier 1 papers + 70 Tier 2 papers split correctly."""

    def test_100_papers_split_into_tiers(self):
        summarizer = _make_summarizer()
        # 30 Tier 1 (rank 1-30) + 70 Tier 2 (rank 31-100)
        papers = _make_ranked_papers(100, start_rank=1)
        rl = _make_rate_limiter()

        call_count = 0
        tier1_calls = 0
        tier2_calls = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count, tier1_calls, tier2_calls
            call_count += 1
            user_msg = messages[1]["content"]
            # Determine tier by checking output format
            if "insight_ko" in user_msg:
                tier1_calls += 1
                # Tier 1 batches of 5
                return _make_tier1_response(5)
            else:
                tier2_calls += 1
                # Tier 2 batches of 10
                return _make_tier2_response(10)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        # 30 Tier 1 papers / 5 per batch = 6 Tier 1 calls
        assert tier1_calls == 6
        # 70 Tier 2 papers / 10 per batch = 7 Tier 2 calls
        assert tier2_calls == 7
        assert len(results) == 100

    def test_only_tier1_papers(self):
        """All papers with rank <= 30 go to Tier 1."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(10, start_rank=1)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            user_msg = messages[1]["content"]
            assert "insight_ko" in user_msg  # should be Tier 1 prompt
            return _make_tier1_response(5)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        # All should have insight_ko
        for r in results:
            assert "insight_ko" in r

    def test_only_tier2_papers(self):
        """All papers with rank > 30 go to Tier 2."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(10, start_rank=31)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            user_msg = messages[1]["content"]
            assert "insight_ko" not in user_msg  # should be Tier 2 prompt
            return _make_tier2_response(10)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        # None should have insight_ko
        for r in results:
            assert "insight_ko" not in r


class TestBatchSizes:
    """Test Tier 1 uses 5/batch, Tier 2 uses 10/batch."""

    def test_tier1_batch_size_5(self):
        """12 Tier 1 papers should produce 3 batches (5+5+2)."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(12, start_rank=1)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            # Return appropriate count
            if call_count <= 2:
                return _make_tier1_response(5)
            else:
                return _make_tier1_response(2)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert call_count == 3  # 12/5 = 3 batches
        assert len(results) == 12

    def test_tier2_batch_size_10(self):
        """25 Tier 2 papers should produce 3 batches (10+10+5)."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(25, start_rank=31)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_tier2_response(10)
            else:
                return _make_tier2_response(5)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert call_count == 3  # 25/10 = 3 batches
        assert len(results) == 25

    def test_custom_batch_sizes_from_config(self):
        """Batch sizes can be overridden via config."""
        config = _make_config(
            agents={
                "common": {},
                "agent3": {
                    "prompt_version": "agent3-v1",
                    "tier1_batch_size": 3,
                    "tier2_batch_size": 7,
                },
            }
        )
        summarizer = _make_summarizer(config)
        # 9 Tier 1 papers with batch_size=3 -> 3 batches
        papers = _make_ranked_papers(9, start_rank=1)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            return _make_tier1_response(3)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert call_count == 3


class TestParseFailure:
    """Test 2x failure skips batch."""

    def test_2x_failure_skips_batch(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()

        with patch.object(summarizer, "call_llm", return_value=None):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 0

    def test_first_failure_second_attempt_succeeds(self):
        """First parse returns None, second attempt returns valid data."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=1)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # first attempt fails
            else:
                return _make_tier1_response(3)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 3
        assert call_count == 2

    def test_non_list_response_triggers_retry(self):
        """LLM returns a dict instead of list -- triggers retry."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=1)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": "not a list"}  # wrong type
            else:
                return _make_tier1_response(3)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 3
        assert call_count == 2

    def test_first_batch_fails_second_succeeds(self):
        """First Tier 1 batch fails, second Tier 1 batch succeeds."""
        summarizer = _make_summarizer()
        # 10 Tier 1 papers -> 2 batches of 5
        papers = _make_ranked_papers(10, start_rank=1)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            if batch_index == 0:
                return None  # first batch fails both times
            else:
                return _make_tier1_response(5)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        # Only second batch succeeded (5 papers)
        assert len(results) == 5


class TestMissingItemReCall:
    """Test missing indices re-called once."""

    def test_missing_items_recalled(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()

        # First call returns 3 items (missing indices 2 and 4)
        first_response = [
            _make_tier1_llm_item(i)
            for i in range(1, 6)
            if i not in (2, 4)
        ]
        # Re-call returns the missing 2 items
        retry_response = [
            _make_tier1_llm_item(2),
            _make_tier1_llm_item(4),
        ]

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            else:
                return retry_response

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 5
        assert call_count == 2

    def test_missing_items_retry_fails_returns_partial(self):
        """If re-call also fails to return missing items, we get partial results."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()

        # First call returns 3 items (missing indices 2 and 4)
        first_response = [
            _make_tier1_llm_item(i)
            for i in range(1, 6)
            if i not in (2, 4)
        ]

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            else:
                return None  # retry fails

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        # Only 3 results (missing 2 not recovered)
        assert len(results) == 3


class TestRateLimiting:
    """Test wait/record_call invoked per LLM call."""

    def test_wait_and_record_call_invoked(self):
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(5)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            summarizer.summarize(papers, "Test topic", rl)

        # 1 batch, 1 LLM call -> 1 wait + 1 record_call
        assert rl.wait.call_count >= 1
        assert rl.record_call.call_count >= 1

    def test_multiple_batches_multiple_waits(self):
        summarizer = _make_summarizer()
        # 10 Tier 1 papers -> 2 batches of 5
        papers = _make_ranked_papers(10, start_rank=1)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            return _make_tier1_response(5)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            summarizer.summarize(papers, "Test topic", rl)

        # 2 batches -> at least 2 waits and 2 record_calls
        assert rl.wait.call_count >= 2
        assert rl.record_call.call_count >= 2

    def test_wait_called_before_each_llm_call(self):
        """Verify wait is called before every call_llm invocation."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(5, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(5)

        call_order = []

        def track_wait():
            call_order.append("wait")

        def track_record():
            call_order.append("record")

        rl.wait = track_wait
        rl.record_call = track_record

        def track_llm(messages, batch_index=0):
            call_order.append("llm")
            return llm_response

        with patch.object(summarizer, "call_llm", side_effect=track_llm):
            summarizer.summarize(papers, "Test topic", rl)

        # Verify pattern: wait -> llm -> record
        assert call_order[0] == "wait"
        assert call_order[1] == "llm"
        assert call_order[2] == "record"


class TestPromptConstruction:
    """Test prompt contains topic and paper info."""

    def test_topic_description_in_prompt(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1)]
        topic_desc = "Korean topic about object detection for autonomous driving"

        messages = summarizer.build_messages(
            papers=papers,
            topic_description=topic_desc,
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert topic_desc in user_msg

    def test_paper_titles_in_prompt(self):
        summarizer = _make_summarizer()
        papers = [
            _make_ranked_paper(1, title="Paper Alpha"),
            _make_ranked_paper(2, title="Paper Beta"),
        ]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert "Paper Alpha" in user_msg
        assert "Paper Beta" in user_msg

    def test_paper_abstracts_in_prompt(self):
        summarizer = _make_summarizer()
        papers = [
            _make_ranked_paper(1, abstract="Abstract about machine learning."),
        ]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert "Abstract about machine learning." in user_msg

    def test_base_score_in_prompt(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1, base_score=85)]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert "85" in user_msg

    def test_system_prompt_present(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1)]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        assert messages[0]["role"] == "system"
        assert "technical writer" in messages[0]["content"]
        assert "ONLY raw JSON" in messages[0]["content"]

    def test_writing_rules_in_prompt(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1)]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert "Writing Rules" in user_msg
        assert "Problem -> Solution -> Result" in user_msg
        assert "fps" in user_msg
        assert "mAP" in user_msg

    def test_tier1_prompt_includes_insight_ko(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1)]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=1,
        )

        user_msg = messages[1]["content"]
        assert "insight_ko" in user_msg

    def test_tier2_prompt_excludes_insight_ko(self):
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(31)]

        messages = summarizer.build_messages(
            papers=papers,
            topic_description="Test",
            tier=2,
        )

        user_msg = messages[1]["content"]
        assert "insight_ko" not in user_msg


class TestEmptyPapers:
    """Test empty papers list returns empty results."""

    def test_empty_input_returns_empty_list(self):
        summarizer = _make_summarizer()
        rl = _make_rate_limiter()

        results = summarizer.summarize([], "Test topic", rl)

        assert results == []
        rl.wait.assert_not_called()
        rl.record_call.assert_not_called()


class TestTierBoundary:
    """Test paper at rank 30 is Tier 1, rank 31 is Tier 2."""

    def test_rank_30_is_tier1(self):
        """Paper with rank 30 should be in Tier 1 (gets insight_ko)."""
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(30)]
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(1)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 1
        assert "insight_ko" in results[0]

    def test_rank_31_is_tier2(self):
        """Paper with rank 31 should be in Tier 2 (no insight_ko)."""
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(31)]
        rl = _make_rate_limiter()
        llm_response = _make_tier2_response(1)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 1
        assert "insight_ko" not in results[0]

    def test_mixed_boundary_papers(self):
        """Papers at rank 29, 30, 31, 32 split correctly at boundary."""
        summarizer = _make_summarizer()
        papers = [
            _make_ranked_paper(29),
            _make_ranked_paper(30),
            _make_ranked_paper(31),
            _make_ranked_paper(32),
        ]
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            user_msg = messages[1]["content"]
            if "insight_ko" in user_msg:
                # Tier 1 batch (rank 29, 30)
                return _make_tier1_response(2)
            else:
                # Tier 2 batch (rank 31, 32)
                return _make_tier2_response(2)

        with patch.object(summarizer, "call_llm", side_effect=side_effect):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 4
        # First 2 should have insight_ko (Tier 1)
        tier1_results = [r for r in results if "insight_ko" in r]
        tier2_results = [r for r in results if "insight_ko" not in r]
        assert len(tier1_results) == 2
        assert len(tier2_results) == 2


class TestAgentProperties:
    """Test agent_name and agent_config_key properties."""

    def test_agent_name(self):
        summarizer = _make_summarizer()
        assert summarizer.agent_name == "summarizer"

    def test_agent_config_key(self):
        summarizer = _make_summarizer()
        assert summarizer.agent_config_key == "agent3"

    def test_prompt_version(self):
        summarizer = _make_summarizer()
        assert summarizer.prompt_version == "agent3-tier1-v1"


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_paper_without_rank_defaults_to_0(self):
        """Paper without rank key defaults to rank 0, placed in Tier 1."""
        summarizer = _make_summarizer()
        paper = {"title": "No Rank", "abstract": "Test", "base_score": 50, "brief_reason": "ok"}
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(1)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize([paper], "Test topic", rl)

        assert len(results) == 1
        # rank 0 <= 30, so Tier 1
        assert "insight_ko" in results[0]

    def test_non_dict_items_in_response_skipped(self):
        """Non-dict items in the LLM response are skipped."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = [
            _make_tier1_llm_item(1),
            "invalid string item",
            _make_tier1_llm_item(3),
        ]

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 2

    def test_item_with_invalid_index_skipped(self):
        """Items with index not in the batch are skipped."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(3, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = [
            _make_tier1_llm_item(1),
            _make_tier1_llm_item(99),  # invalid index
            _make_tier1_llm_item(3),
        ]

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert len(results) == 2

    def test_original_paper_fields_preserved(self):
        """Original paper dict fields are preserved in the enriched output."""
        summarizer = _make_summarizer()
        papers = [_make_ranked_paper(1, title="Original Title", base_score=90)]
        rl = _make_rate_limiter()
        llm_response = _make_tier1_response(1)

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert results[0]["title"] == "Original Title"
        assert results[0]["base_score"] == 90
        assert results[0]["rank"] == 1

    def test_missing_summary_defaults_to_empty_string(self):
        """If LLM omits summary_ko, it defaults to empty string."""
        summarizer = _make_summarizer()
        papers = _make_ranked_papers(1, start_rank=1)
        rl = _make_rate_limiter()
        llm_response = [{"index": 1}]  # no summary fields

        with patch.object(summarizer, "call_llm", return_value=llm_response):
            results = summarizer.summarize(papers, "Test topic", rl)

        assert results[0]["summary_ko"] == ""
        assert results[0]["reason_ko"] == ""
        assert results[0]["insight_ko"] == ""
