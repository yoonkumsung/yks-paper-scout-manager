"""Comprehensive tests for Scorer (Agent 2).

Covers:
  - Single batch scoring (10 papers)
  - Multi batch scoring (25 papers -> 3 batches)
  - Score clamping (above 100 -> 100, below 0 -> 0)
  - Discard logic (is_metaphorical -> discard, base_score < 20 -> discard)
  - has_code merge (regex has_code OR mentions_code -> final has_code)
  - Missing item re-call (LLM returns 8/10, re-call gets missing 2)
  - Parse failure (2x failure -> batch skipped, other batches processed)
  - Rate limiting (wait called before each LLM call)
  - Prompt construction (topic_description and paper list present)
  - Empty papers (returns empty list)
  - Flags validation (all 4 flags present and boolean)
  - brief_reason present (each result has brief_reason string)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agents.scorer import Scorer
from core.config import AppConfig
from core.models import Paper


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(agents: dict | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    if agents is None:
        agents = {
            "common": {"ignore_reasoning": True},
            "agent2": {
                "effort": "low",
                "max_tokens": 2048,
                "temperature": 0.2,
                "prompt_version": "agent2-v2",
                "batch_size": 10,
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


def _make_paper(
    index: int,
    *,
    has_code: bool = False,
    title: str | None = None,
    abstract: str | None = None,
) -> Paper:
    """Create a test Paper."""
    return Paper(
        source="arxiv",
        native_id=f"2401.{index:05d}",
        paper_key=f"arxiv:2401.{index:05d}",
        url=f"http://arxiv.org/abs/2401.{index:05d}",
        title=title or f"Test Paper {index}",
        abstract=abstract or f"Abstract for test paper {index}.",
        authors=["Author A"],
        categories=["cs.CV"],
        published_at_utc=datetime(2024, 1, 15, tzinfo=timezone.utc),
        has_code=has_code,
    )


def _make_papers(count: int) -> list[Paper]:
    """Create a list of test Papers."""
    return [_make_paper(i + 1) for i in range(count)]


def _make_llm_item(
    index: int,
    *,
    base_score: int = 75,
    is_edge: bool = False,
    is_realtime: bool = False,
    mentions_code: bool = False,
    is_metaphorical: bool = False,
    discard: bool = False,
    brief_reason: str = "Relevant paper.",
) -> dict:
    """Create a single LLM output item."""
    return {
        "index": index,
        "base_score": base_score,
        "flags": {
            "is_edge": is_edge,
            "is_realtime": is_realtime,
            "mentions_code": mentions_code,
            "is_metaphorical": is_metaphorical,
        },
        "discard": discard,
        "brief_reason": brief_reason,
    }


def _make_llm_response(count: int, start_index: int = 1) -> list[dict]:
    """Create a full LLM response list for `count` papers."""
    return [
        _make_llm_item(start_index + i, base_score=70 + i)
        for i in range(count)
    ]


def _make_scorer(config: AppConfig | None = None) -> Scorer:
    """Instantiate a Scorer with mocked client."""
    cfg = config or _make_config()
    client = MagicMock()
    return Scorer(cfg, client)


def _make_rate_limiter() -> MagicMock:
    """Create a mock RateLimiter."""
    rl = MagicMock()
    rl.wait = MagicMock()
    rl.record_call = MagicMock()
    return rl


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSingleBatch:
    """Test scoring a single batch of 10 papers."""

    def test_single_batch_scores_all_papers(self):
        scorer = _make_scorer()
        papers = _make_papers(10)
        rl = _make_rate_limiter()
        llm_response = _make_llm_response(10)

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["paper_key"] == f"arxiv:2401.{i + 1:05d}"
            assert 0 <= result["base_score"] <= 100

    def test_single_batch_preserves_order(self):
        scorer = _make_scorer()
        papers = _make_papers(5)
        rl = _make_rate_limiter()
        llm_response = _make_llm_response(5)

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        paper_keys = [r["paper_key"] for r in results]
        assert paper_keys == [p.paper_key for p in papers]


class TestMultiBatch:
    """Test scoring 25 papers -> 3 batches (10+10+5)."""

    def test_25_papers_produces_3_batches(self):
        scorer = _make_scorer()
        papers = _make_papers(25)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            # Determine batch size from messages
            # Each batch: first 10, second 10, third 5
            if call_count <= 1:
                return _make_llm_response(10, start_index=1)
            elif call_count <= 2:
                return _make_llm_response(10, start_index=1)
            else:
                return _make_llm_response(5, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 25
        assert call_count == 3

    def test_multi_batch_all_paper_keys_present(self):
        scorer = _make_scorer()
        papers = _make_papers(15)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            if batch_index == 0:
                return _make_llm_response(10, start_index=1)
            else:
                return _make_llm_response(5, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        result_keys = {r["paper_key"] for r in results}
        expected_keys = {p.paper_key for p in papers}
        assert result_keys == expected_keys


class TestScoreClamping:
    """Test base_score clamping to 0-100 range."""

    def test_score_above_100_clamped_to_100(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=150)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["base_score"] == 100

    def test_score_below_0_clamped_to_0(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=-20)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["base_score"] == 0

    def test_score_at_boundaries(self):
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()
        llm_response = [
            _make_llm_item(1, base_score=0),
            _make_llm_item(2, base_score=100),
            _make_llm_item(3, base_score=50),
        ]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["base_score"] == 0
        assert results[1]["base_score"] == 100
        assert results[2]["base_score"] == 50


class TestDiscardLogic:
    """Test discard logic: is_metaphorical=True or base_score < 20."""

    def test_metaphorical_true_causes_discard(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=85, is_metaphorical=True)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is True

    def test_score_below_20_causes_discard(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=15)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is True

    def test_score_exactly_20_not_discarded(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=20)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is False

    def test_score_19_discarded(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=19)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is True

    def test_normal_score_not_discarded(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=75)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is False

    def test_metaphorical_and_low_score_both_discard(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, base_score=5, is_metaphorical=True)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["discard"] is True


class TestHasCodeMerge:
    """Test has_code merge: regex has_code OR mentions_code -> final has_code."""

    def test_paper_has_code_true_llm_false(self):
        scorer = _make_scorer()
        papers = [_make_paper(1, has_code=True)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, mentions_code=False)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["has_code"] is True

    def test_paper_has_code_false_llm_true(self):
        scorer = _make_scorer()
        papers = [_make_paper(1, has_code=False)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, mentions_code=True)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["has_code"] is True

    def test_both_true(self):
        scorer = _make_scorer()
        papers = [_make_paper(1, has_code=True)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, mentions_code=True)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["has_code"] is True

    def test_both_false(self):
        scorer = _make_scorer()
        papers = [_make_paper(1, has_code=False)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, mentions_code=False)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["has_code"] is False


class TestMissingItemReCall:
    """Test missing item re-call: LLM returns 8/10, re-call gets missing 2."""

    def test_missing_items_recalled(self):
        scorer = _make_scorer()
        papers = _make_papers(10)
        rl = _make_rate_limiter()

        # First call returns 8 items (missing indices 3 and 7)
        first_response = [
            _make_llm_item(i, base_score=70 + i)
            for i in range(1, 11)
            if i not in (3, 7)
        ]
        # Re-call returns the missing 2 items
        retry_response = [
            _make_llm_item(3, base_score=73),
            _make_llm_item(7, base_score=77),
        ]

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            else:
                return retry_response

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 10
        assert call_count == 2

        # Verify the re-called items are present
        result_keys = {r["paper_key"] for r in results}
        assert "arxiv:2401.00003" in result_keys
        assert "arxiv:2401.00007" in result_keys

    def test_missing_items_retry_fails_returns_partial(self):
        """If re-call also fails to return missing items, we get partial results."""
        scorer = _make_scorer()
        papers = _make_papers(10)
        rl = _make_rate_limiter()

        # First call returns 8 items (missing 3 and 7)
        first_response = [
            _make_llm_item(i, base_score=70 + i)
            for i in range(1, 11)
            if i not in (3, 7)
        ]

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            else:
                return None  # retry fails

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        # Only 8 results (missing 2 not recovered)
        assert len(results) == 8


class TestParseFailure:
    """Test 2x parse failure -> batch skipped, other batches processed."""

    def test_2x_failure_skips_batch(self):
        scorer = _make_scorer()
        papers = _make_papers(5)
        rl = _make_rate_limiter()

        with patch.object(scorer, "call_llm", return_value=None):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 0

    def test_first_batch_fails_second_succeeds(self):
        """First batch (10 papers) fails, second batch (5 papers) succeeds."""
        scorer = _make_scorer()
        papers = _make_papers(15)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if batch_index == 0:
                return None  # first batch fails
            else:
                return _make_llm_response(5, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        # Only second batch succeeded (5 papers)
        assert len(results) == 5

    def test_first_failure_second_attempt_succeeds(self):
        """First parse returns None, second attempt returns valid data."""
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # first attempt fails
            else:
                return _make_llm_response(3, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 3
        assert call_count == 2

    def test_non_list_response_triggers_retry(self):
        """LLM returns a dict instead of list -- triggers retry."""
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"error": "not a list"}  # wrong type
            else:
                return _make_llm_response(3, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 3
        assert call_count == 2


class TestRateLimiting:
    """Test rate_limiter.wait() called before each LLM call."""

    def test_wait_and_record_call_invoked(self):
        scorer = _make_scorer()
        papers = _make_papers(5)
        rl = _make_rate_limiter()
        llm_response = _make_llm_response(5)

        with patch.object(scorer, "call_llm", return_value=llm_response):
            scorer.score(papers, "Test topic", rl)

        # 1 batch, 1 LLM call -> 1 wait + 1 record_call
        assert rl.wait.call_count >= 1
        assert rl.record_call.call_count >= 1

    def test_multiple_batches_multiple_waits(self):
        scorer = _make_scorer()
        papers = _make_papers(20)
        rl = _make_rate_limiter()

        def side_effect(messages, batch_index=0):
            return _make_llm_response(10, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            scorer.score(papers, "Test topic", rl)

        # 2 batches -> at least 2 waits and 2 record_calls
        assert rl.wait.call_count >= 2
        assert rl.record_call.call_count >= 2

    def test_wait_called_before_each_llm_call(self):
        """Verify wait is called before every call_llm invocation."""
        scorer = _make_scorer()
        papers = _make_papers(5)
        rl = _make_rate_limiter()
        llm_response = _make_llm_response(5)

        call_order = []
        original_wait = rl.wait
        original_record = rl.record_call

        def track_wait():
            call_order.append("wait")

        def track_record():
            call_order.append("record")

        rl.wait = track_wait
        rl.record_call = track_record

        def track_llm(messages, batch_index=0):
            call_order.append("llm")
            return llm_response

        with patch.object(scorer, "call_llm", side_effect=track_llm):
            scorer.score(papers, "Test topic", rl)

        # Verify pattern: wait -> llm -> record
        assert call_order[0] == "wait"
        assert call_order[1] == "llm"
        assert call_order[2] == "record"


class TestPromptConstruction:
    """Test prompt contains topic_description and paper list."""

    def test_topic_description_in_prompt(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        topic_desc = "Korean topic about deep learning for sports"

        messages = scorer.build_messages(
            papers=papers,
            topic_description=topic_desc,
        )

        user_msg = messages[1]["content"]
        assert topic_desc in user_msg

    def test_paper_titles_in_prompt(self):
        scorer = _make_scorer()
        papers = [
            _make_paper(1, title="Paper Alpha"),
            _make_paper(2, title="Paper Beta"),
        ]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
        )

        user_msg = messages[1]["content"]
        assert "Paper Alpha" in user_msg
        assert "Paper Beta" in user_msg

    def test_paper_abstracts_in_prompt(self):
        scorer = _make_scorer()
        papers = [
            _make_paper(1, abstract="Abstract about machine learning."),
        ]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
        )

        user_msg = messages[1]["content"]
        assert "Abstract about machine learning." in user_msg

    def test_system_prompt_present(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
        )

        assert messages[0]["role"] == "system"
        assert "paper evaluator" in messages[0]["content"]
        assert "ONLY raw JSON" in messages[0]["content"]

    def test_scoring_rubric_in_prompt(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
        )

        user_msg = messages[1]["content"]
        assert "90~100" in user_msg
        assert "Below 20" in user_msg

    def test_flags_description_in_prompt(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
        )

        user_msg = messages[1]["content"]
        assert "is_edge" in user_msg
        assert "is_realtime" in user_msg
        assert "mentions_code" in user_msg
        assert "is_metaphorical" in user_msg

    def test_custom_indices_in_prompt(self):
        scorer = _make_scorer()
        papers = [_make_paper(1), _make_paper(2)]

        messages = scorer.build_messages(
            papers=papers,
            topic_description="Test",
            indices=[5, 9],
        )

        user_msg = messages[1]["content"]
        assert "5." in user_msg
        assert "9." in user_msg


class TestEmptyPapers:
    """Test empty papers list returns empty results."""

    def test_empty_input_returns_empty_list(self):
        scorer = _make_scorer()
        rl = _make_rate_limiter()

        results = scorer.score([], "Test topic", rl)

        assert results == []
        rl.wait.assert_not_called()
        rl.record_call.assert_not_called()


class TestFlagsValidation:
    """Test all 4 flags present and boolean."""

    def test_all_flags_present(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        flags = results[0]["flags"]
        assert "is_edge" in flags
        assert "is_realtime" in flags
        assert "mentions_code" in flags
        assert "is_metaphorical" in flags

    def test_flags_are_boolean(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1)]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        flags = results[0]["flags"]
        for key, value in flags.items():
            assert isinstance(value, bool), f"Flag '{key}' is not bool: {type(value)}"

    def test_flags_default_false_when_missing(self):
        """If LLM omits flags, they default to False."""
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        # LLM returns item with empty flags dict
        llm_response = [{"index": 1, "base_score": 60, "flags": {}, "brief_reason": "ok"}]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        flags = results[0]["flags"]
        assert flags["is_edge"] is False
        assert flags["is_realtime"] is False
        assert flags["mentions_code"] is False
        assert flags["is_metaphorical"] is False

    def test_flags_non_dict_defaults_all_false(self):
        """If LLM returns non-dict flags, all default to False."""
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [{"index": 1, "base_score": 50, "flags": "invalid", "brief_reason": "ok"}]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        flags = results[0]["flags"]
        for value in flags.values():
            assert value is False


class TestBriefReasonPresent:
    """Test each result has brief_reason string."""

    def test_brief_reason_present(self):
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [_make_llm_item(1, brief_reason="Relevant for CV.")]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["brief_reason"] == "Relevant for CV."

    def test_brief_reason_is_string(self):
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()
        llm_response = _make_llm_response(3)

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        for result in results:
            assert isinstance(result["brief_reason"], str)

    def test_brief_reason_default_empty_string(self):
        """If LLM omits brief_reason, it defaults to empty string."""
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [{"index": 1, "base_score": 60, "flags": {}}]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["brief_reason"] == ""


class TestAgentProperties:
    """Test agent_name and agent_config_key properties."""

    def test_agent_name(self):
        scorer = _make_scorer()
        assert scorer.agent_name == "scorer"

    def test_agent_config_key(self):
        scorer = _make_scorer()
        assert scorer.agent_config_key == "agent2"

    def test_prompt_version(self):
        scorer = _make_scorer()
        assert scorer.prompt_version == "agent2-v2"


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_invalid_base_score_type_defaults_to_0(self):
        """If base_score is not numeric, defaults to 0."""
        scorer = _make_scorer()
        papers = [_make_paper(1)]
        rl = _make_rate_limiter()
        llm_response = [{"index": 1, "base_score": "invalid", "flags": {}, "brief_reason": ""}]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert results[0]["base_score"] == 0
        # base_score=0 < 20, so discard should be True
        assert results[0]["discard"] is True

    def test_non_dict_items_in_response_skipped(self):
        """Non-dict items in the LLM response are skipped."""
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()
        llm_response = [
            _make_llm_item(1),
            "invalid string item",
            _make_llm_item(3),
        ]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 2

    def test_item_with_invalid_index_skipped(self):
        """Items with index not in the batch are skipped."""
        scorer = _make_scorer()
        papers = _make_papers(3)
        rl = _make_rate_limiter()
        llm_response = [
            _make_llm_item(1),
            _make_llm_item(99),  # invalid index
            _make_llm_item(3),
        ]

        with patch.object(scorer, "call_llm", return_value=llm_response):
            results = scorer.score(papers, "Test topic", rl)

        assert len(results) == 2
        result_keys = [r["paper_key"] for r in results]
        assert "arxiv:2401.00001" in result_keys
        assert "arxiv:2401.00003" in result_keys

    def test_batch_size_from_config(self):
        """batch_size is read from agent config."""
        config = _make_config(
            agents={
                "common": {},
                "agent2": {
                    "batch_size": 5,
                    "prompt_version": "agent2-v2",
                },
            }
        )
        scorer = _make_scorer(config)
        papers = _make_papers(12)
        rl = _make_rate_limiter()

        call_count = 0

        def side_effect(messages, batch_index=0):
            nonlocal call_count
            call_count += 1
            # batch sizes: 5, 5, 2
            if batch_index == 2:
                return _make_llm_response(2, start_index=1)
            return _make_llm_response(5, start_index=1)

        with patch.object(scorer, "call_llm", side_effect=side_effect):
            results = scorer.score(papers, "Test topic", rl)

        # 12 papers / batch_size 5 = 3 batches
        assert call_count == 3
