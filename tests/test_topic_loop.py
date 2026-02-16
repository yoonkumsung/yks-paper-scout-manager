"""Tests for core.pipeline.topic_loop module.

Covers the TopicLoopOrchestrator class:
- run_all_topics summary structure and edge cases
- Topic resolution (with and without slug filter)
- Error isolation between topics
- Daily limit and per-topic skip logic
- Rate limiter / usage tracker interactions
- DB status updates (running, completed, failed)
- _extract_keywords static helper
- 12-step sequence verification (mocked internal steps)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

from core.pipeline.topic_loop import TopicLoopOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockNotifyConfig:
    """Minimal stand-in for core.models.NotifyConfig."""

    def __init__(self) -> None:
        self.provider = "discord"
        self.channel_id = "123"
        self.secret_key = "secret"


class MockTopic:
    """Minimal stand-in for core.models.TopicSpec."""

    def __init__(self, slug: str, name: str = "") -> None:
        self.slug = slug
        self.name = name or slug.replace("-", " ").title()
        self.description = "Research topic about %s" % slug
        self.arxiv_categories = ["cs.AI"]
        self.notify = MockNotifyConfig()
        self.must_concepts_en: Optional[List[str]] = None
        self.should_concepts_en: Optional[List[str]] = None
        self.must_not_en: Optional[List[str]] = None


class MockConfig:
    """Minimal stand-in for core.config.AppConfig."""

    def __init__(
        self,
        topics: Optional[List[MockTopic]] = None,
    ) -> None:
        self.topics = topics if topics is not None else [
            MockTopic("topic-a"),
            MockTopic("topic-b"),
        ]
        self.scoring: Dict[str, Any] = {
            "weights": {"embedding_off": {"llm": 0.7, "recency": 0.3}},
        }
        self.llm: Dict[str, Any] = {"response_format_supported": False}
        self.agents: Dict[str, Any] = {}
        self.filter: Dict[str, Any] = {}
        self.output: Dict[str, Any] = {
            "report_dir": "tmp/reports",
            "template_dir": "templates",
        }
        self.sources: Dict[str, Any] = {
            "primary": "arxiv",
            "seen_items_path": "data/seen_items.jsonl",
        }
        self.embedding: Dict[str, Any] = {"cache_dir": "data"}
        self.notifications: Dict[str, Any] = {}
        self.remind: Dict[str, Any] = {
            "min_score": 80.0,
            "max_recommend_count": 2,
        }
        self.clustering: Dict[str, Any] = {"threshold": 0.85}


# Lazy-import patch targets used by _run_single_topic.
# These are imported inside the method body so we patch the source module.
_PATCH_EMBEDDING_RANKER = "core.embeddings.embedding_ranker.EmbeddingRanker"
_PATCH_CODE_DETECTOR = "core.scoring.code_detector.CodeDetector"
_PATCH_RANKER = "core.scoring.ranker.Ranker"
_PATCH_DEDUP_MANAGER = "core.storage.dedup.DedupManager"


def _make_orchestrator(
    config: Optional[MockConfig] = None,
    rate_limiter: Optional[MagicMock] = None,
    usage_tracker: Optional[MagicMock] = None,
    db_manager: Optional[MagicMock] = None,
    search_window: Optional[MagicMock] = None,
    run_options: Optional[Dict[str, Any]] = None,
) -> TopicLoopOrchestrator:
    """Create an orchestrator with default mocks for all dependencies."""
    if config is None:
        config = MockConfig()

    if rate_limiter is None:
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(return_value=False)
        rate_limiter.record_topic_completed = MagicMock()

    if usage_tracker is None:
        usage_tracker = MagicMock()

    if db_manager is None:
        db_manager = MagicMock()
        db_manager.create_run = MagicMock(return_value=1)

    if search_window is None:
        search_window = MagicMock()
        search_window.compute = MagicMock(
            return_value=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc),
            )
        )

    return TopicLoopOrchestrator(
        config=config,
        db_manager=db_manager,
        rate_limiter=rate_limiter,
        search_window=search_window,
        usage_tracker=usage_tracker,
        run_options=run_options,
    )


def _patch_lazy_imports():
    """Return a dict of context managers for the lazy imports in _run_single_topic.

    Returns mock objects for EmbeddingRanker, CodeDetector, Ranker, DedupManager.
    """
    mock_emb_inst = MagicMock()
    mock_emb_inst.mode = "disabled"
    mock_emb_inst.available = False

    mock_emb_cls = MagicMock(return_value=mock_emb_inst)

    mock_ranker_inst = MagicMock()
    mock_ranker_inst._default_threshold = 60
    mock_ranker_cls = MagicMock(return_value=mock_ranker_inst)

    mock_dedup_inst = MagicMock()
    mock_dedup_cls = MagicMock(return_value=mock_dedup_inst)

    mock_code_inst = MagicMock()
    mock_code_cls = MagicMock(return_value=mock_code_inst)

    return {
        "emb_cls": mock_emb_cls,
        "emb_inst": mock_emb_inst,
        "ranker_cls": mock_ranker_cls,
        "dedup_cls": mock_dedup_cls,
        "code_cls": mock_code_cls,
    }


# ---------------------------------------------------------------------------
# 1. Empty topics list -> returns empty summary
# ---------------------------------------------------------------------------


class TestEmptyTopics:
    """When no topics are configured, orchestrator returns an empty summary."""

    def test_empty_topics_returns_empty_summary(self) -> None:
        config = MockConfig(topics=[])
        orch = _make_orchestrator(config=config)

        result = orch.run_all_topics()

        assert result == {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }


# ---------------------------------------------------------------------------
# 2. Single topic completes successfully
# ---------------------------------------------------------------------------


class TestSingleTopicSuccess:
    """A single topic completing successfully shows up in topics_completed."""

    def test_single_topic_completes(self) -> None:
        config = MockConfig(topics=[MockTopic("my-topic")])
        orch = _make_orchestrator(config=config)

        fake_result = {"total_output": 5}
        with patch.object(orch, "_run_single_topic", return_value=fake_result):
            summary = orch.run_all_topics()

        assert len(summary["topics_completed"]) == 1
        assert summary["topics_completed"][0]["slug"] == "my-topic"
        assert summary["topics_completed"][0]["total_output"] == 5
        assert summary["topics_failed"] == []
        assert summary["topics_skipped"] == []


# ---------------------------------------------------------------------------
# 3. Error isolation: one topic fails, next one still runs
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    """Failure in one topic does not block subsequent topics."""

    def test_failed_topic_does_not_block_next(self) -> None:
        config = MockConfig(
            topics=[MockTopic("fail-topic"), MockTopic("ok-topic")]
        )
        orch = _make_orchestrator(config=config)

        def side_effect(topic: Any) -> Dict[str, Any]:
            if topic.slug == "fail-topic":
                raise RuntimeError("boom")
            return {"total_output": 3}

        with patch.object(orch, "_run_single_topic", side_effect=side_effect):
            summary = orch.run_all_topics()

        assert len(summary["topics_failed"]) == 1
        assert summary["topics_failed"][0]["slug"] == "fail-topic"
        assert "boom" in summary["topics_failed"][0]["error"]

        assert len(summary["topics_completed"]) == 1
        assert summary["topics_completed"][0]["slug"] == "ok-topic"


# ---------------------------------------------------------------------------
# 4. Daily limit reached -> remaining topics skipped
# ---------------------------------------------------------------------------


class TestDailyLimitReached:
    """When daily limit is reached, all remaining topics are skipped."""

    def test_daily_limit_skips_remaining_topics(self) -> None:
        config = MockConfig(
            topics=[MockTopic("topic-a"), MockTopic("topic-b")]
        )
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = True
        rate_limiter.should_skip_topic = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)
        summary = orch.run_all_topics()

        assert len(summary["topics_skipped"]) == 2
        for entry in summary["topics_skipped"]:
            assert entry["reason"] == "daily_limit_reached"
        assert summary["topics_completed"] == []
        assert summary["topics_failed"] == []


# ---------------------------------------------------------------------------
# 5. Topic already completed today -> skip with reason
# ---------------------------------------------------------------------------


class TestTopicAlreadyCompleted:
    """A topic already completed today is skipped."""

    def test_already_completed_today_skipped(self) -> None:
        config = MockConfig(topics=[MockTopic("done-topic")])
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(return_value=True)

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)
        summary = orch.run_all_topics()

        assert len(summary["topics_skipped"]) == 1
        assert summary["topics_skipped"][0]["reason"] == "already_completed_today"
        assert summary["topics_completed"] == []


# ---------------------------------------------------------------------------
# 6. run_all_topics returns correct summary structure
# ---------------------------------------------------------------------------


class TestSummaryStructure:
    """The returned summary always has the three expected keys."""

    def test_summary_keys_present(self) -> None:
        orch = _make_orchestrator(config=MockConfig(topics=[]))
        result = orch.run_all_topics()

        assert "topics_completed" in result
        assert "topics_skipped" in result
        assert "topics_failed" in result
        assert isinstance(result["topics_completed"], list)
        assert isinstance(result["topics_skipped"], list)
        assert isinstance(result["topics_failed"], list)


# ---------------------------------------------------------------------------
# 7. _resolve_topics with specific slugs -> filters
# ---------------------------------------------------------------------------


class TestResolveTopicsWithFilter:
    """When run_options.topics is specified, only matching topics returned."""

    def test_resolve_filters_to_requested_slugs(self) -> None:
        config = MockConfig(
            topics=[
                MockTopic("alpha"),
                MockTopic("beta"),
                MockTopic("gamma"),
            ]
        )
        orch = _make_orchestrator(
            config=config,
            run_options={"topics": ["beta"]},
        )

        resolved = orch._resolve_topics()

        assert len(resolved) == 1
        assert resolved[0].slug == "beta"


# ---------------------------------------------------------------------------
# 8. _resolve_topics without filter -> returns all
# ---------------------------------------------------------------------------


class TestResolveTopicsNoFilter:
    """Without a topics filter, all topics are returned."""

    def test_resolve_returns_all_topics(self) -> None:
        config = MockConfig(
            topics=[MockTopic("x"), MockTopic("y")]
        )
        orch = _make_orchestrator(config=config)

        resolved = orch._resolve_topics()

        assert len(resolved) == 2
        slugs = [t.slug for t in resolved]
        assert "x" in slugs
        assert "y" in slugs


# ---------------------------------------------------------------------------
# 9. _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    """Static helper extracts keywords from concepts + cross_domain."""

    def test_extract_from_concepts_and_cross_domain(self) -> None:
        agent1_output = {
            "concepts": [
                {"keywords": ["llm", "transformer"]},
                {"keywords": ["attention"]},
            ],
            "cross_domain_keywords": ["multimodal", "vision"],
        }

        keywords = TopicLoopOrchestrator._extract_keywords(agent1_output)

        assert keywords == ["llm", "transformer", "attention", "multimodal", "vision"]

    def test_extract_empty_output(self) -> None:
        keywords = TopicLoopOrchestrator._extract_keywords({})
        assert keywords == []

    def test_extract_missing_cross_domain(self) -> None:
        agent1_output = {
            "concepts": [{"keywords": ["a", "b"]}],
        }
        keywords = TopicLoopOrchestrator._extract_keywords(agent1_output)
        assert keywords == ["a", "b"]


# ---------------------------------------------------------------------------
# 10. 12-step sequence verification
# ---------------------------------------------------------------------------


class TestStepSequence:
    """Verify the 12-step pipeline sequence by mocking all internal steps."""

    def test_all_steps_called_in_order(self) -> None:
        config = MockConfig(topics=[MockTopic("seq-topic")])
        db = MagicMock()
        db.create_run = MagicMock(return_value=42)
        db.paper_exists = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, db_manager=db)

        call_order: List[str] = []

        with patch.object(
            orch, "_step_agent1",
            side_effect=lambda topic: (
                call_order.append("agent1"),
                {"concepts": [], "cross_domain_keywords": []},
            )[-1],
        ), patch.object(
            orch, "_step_collect",
            side_effect=lambda *a, **kw: (
                call_order.append("collect"),
                ([], []),
            )[-1],
        ), patch.object(
            orch, "_step_filter",
            side_effect=lambda *a, **kw: (
                call_order.append("filter"),
                ([], {}),
            )[-1],
        ), patch.object(
            orch, "_step_score",
            side_effect=lambda *a, **kw: (
                call_order.append("score"),
                [],
            )[-1],
        ), patch.object(
            orch, "_step_rank",
            side_effect=lambda *a, **kw: (
                call_order.append("rank"),
                [],
            )[-1],
        ), patch.object(
            orch, "_step_cluster",
            side_effect=lambda *a, **kw: (
                call_order.append("cluster"),
                [],
            )[-1],
        ), patch.object(
            orch, "_step_summarize",
            side_effect=lambda *a, **kw: (
                call_order.append("summarize"),
                [],
            )[-1],
        ), patch.object(
            orch, "_step_remind",
            side_effect=lambda *a, **kw: (
                call_order.append("remind"),
                [],
            )[-1],
        ), patch.object(
            orch, "_step_generate_reports",
            side_effect=lambda **kw: (
                call_order.append("reports"),
                {"json": "f.json", "md": "f.md", "html": "f.html"},
            )[-1],
        ), patch.object(
            orch, "_step_github_issue",
            side_effect=lambda *a, **kw: call_order.append("github_issue"),
        ), patch(
            _PATCH_EMBEDDING_RANKER,
        ) as mock_emb, patch(
            _PATCH_RANKER,
        ) as mock_ranker_cls, patch(
            _PATCH_DEDUP_MANAGER,
        ) as mock_dedup_cls, patch(
            _PATCH_CODE_DETECTOR,
        ):
            mock_emb_inst = MagicMock()
            mock_emb_inst.mode = "disabled"
            mock_emb_inst.available = False
            mock_emb.return_value = mock_emb_inst

            mock_ranker_inst = MagicMock()
            mock_ranker_inst._default_threshold = 60
            mock_ranker_cls.return_value = mock_ranker_inst

            mock_dedup_cls.return_value = MagicMock()

            summary = orch.run_all_topics()

        expected_order = [
            "agent1",
            "collect",
            "filter",
            "score",
            "rank",
            "cluster",
            "summarize",
            "remind",
            "reports",
            "github_issue",
        ]
        assert call_order == expected_order
        assert len(summary["topics_completed"]) == 1


# ---------------------------------------------------------------------------
# 11. Run creates RunMeta with status=running
# ---------------------------------------------------------------------------


class TestRunMetaCreation:
    """_run_single_topic creates a RunMeta and calls DB.create_run."""

    def test_create_run_called_with_running_status(self) -> None:
        config = MockConfig(topics=[MockTopic("run-test")])
        db = MagicMock()
        db.create_run = MagicMock(return_value=10)
        db.paper_exists = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, db_manager=db)

        with patch.object(
            orch, "_step_agent1",
            return_value={"concepts": [], "cross_domain_keywords": []},
        ), patch.object(
            orch, "_step_collect", return_value=([], []),
        ), patch.object(
            orch, "_step_filter", return_value=([], {}),
        ), patch.object(
            orch, "_step_score", return_value=[],
        ), patch.object(
            orch, "_step_rank", return_value=[],
        ), patch.object(
            orch, "_step_cluster", return_value=[],
        ), patch.object(
            orch, "_step_summarize", return_value=[],
        ), patch.object(
            orch, "_step_remind", return_value=[],
        ), patch.object(
            orch, "_step_generate_reports",
            return_value={"json": "", "md": "", "html": ""},
        ), patch.object(
            orch, "_step_github_issue",
        ), patch(
            _PATCH_EMBEDDING_RANKER,
        ) as mock_emb, patch(
            _PATCH_RANKER,
        ) as mock_ranker_cls, patch(
            _PATCH_DEDUP_MANAGER,
        ) as mock_dedup_cls, patch(
            _PATCH_CODE_DETECTOR,
        ):
            mock_inst = MagicMock()
            mock_inst.mode = "disabled"
            mock_inst.available = False
            mock_emb.return_value = mock_inst

            mock_ranker_inst = MagicMock()
            mock_ranker_inst._default_threshold = 60
            mock_ranker_cls.return_value = mock_ranker_inst

            mock_dedup_cls.return_value = MagicMock()

            orch.run_all_topics()

        db.create_run.assert_called_once()
        run_meta_arg = db.create_run.call_args[0][0]
        assert run_meta_arg.status == "running"
        assert run_meta_arg.topic_slug == "run-test"


# ---------------------------------------------------------------------------
# 12. Failed topic sets status=failed in DB
# ---------------------------------------------------------------------------


class TestFailedTopicDBStatus:
    """When _run_single_topic raises, DB status is updated to 'failed'."""

    def test_failed_run_updates_db_status(self) -> None:
        config = MockConfig(topics=[MockTopic("fail-db")])
        db = MagicMock()
        db.create_run = MagicMock(return_value=99)

        orch = _make_orchestrator(config=config, db_manager=db)

        def _explode(topic: Any) -> Dict[str, Any]:
            raise ValueError("test explosion")

        with patch.object(orch, "_run_single_topic", side_effect=_explode):
            summary = orch.run_all_topics()

        assert len(summary["topics_failed"]) == 1
        # _run_single_topic is fully mocked here, so the internal
        # db.update_run_status("failed") is not called from inside.
        # Instead we check the error was captured in the summary.
        assert "test explosion" in summary["topics_failed"][0]["error"]


class TestFailedTopicDBStatusInternal:
    """Within _run_single_topic, a step failure marks the run as failed."""

    def test_internal_failure_marks_run_failed(self) -> None:
        config = MockConfig(topics=[MockTopic("internal-fail")])
        db = MagicMock()
        db.create_run = MagicMock(return_value=77)

        orch = _make_orchestrator(config=config, db_manager=db)

        # Make _step_agent1 raise to trigger the except block inside
        # _run_single_topic (after RunMeta is created).
        with patch(
            _PATCH_EMBEDDING_RANKER,
        ) as mock_emb, patch.object(
            orch, "_step_agent1", side_effect=RuntimeError("agent1 boom"),
        ):
            mock_inst = MagicMock()
            mock_inst.mode = "disabled"
            mock_emb.return_value = mock_inst

            summary = orch.run_all_topics()

        assert len(summary["topics_failed"]) == 1
        db.update_run_status.assert_called_once()
        call_args = db.update_run_status.call_args
        assert call_args[0][0] == 77  # run_id
        assert call_args[0][1] == "failed"


# ---------------------------------------------------------------------------
# 13. Successful topic sets status=completed
# ---------------------------------------------------------------------------


class TestCompletedTopicDBStatus:
    """A successful run marks status as 'completed' in the DB."""

    def test_completed_run_updates_db_status(self) -> None:
        config = MockConfig(topics=[MockTopic("ok-status")])
        db = MagicMock()
        db.create_run = MagicMock(return_value=55)
        db.paper_exists = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, db_manager=db)

        with patch.object(
            orch, "_step_agent1",
            return_value={"concepts": [], "cross_domain_keywords": []},
        ), patch.object(
            orch, "_step_collect", return_value=([], []),
        ), patch.object(
            orch, "_step_filter", return_value=([], {}),
        ), patch.object(
            orch, "_step_score", return_value=[],
        ), patch.object(
            orch, "_step_rank", return_value=[],
        ), patch.object(
            orch, "_step_cluster", return_value=[],
        ), patch.object(
            orch, "_step_summarize", return_value=[],
        ), patch.object(
            orch, "_step_remind", return_value=[],
        ), patch.object(
            orch, "_step_generate_reports",
            return_value={"json": "", "md": "", "html": ""},
        ), patch.object(
            orch, "_step_github_issue",
        ), patch(
            _PATCH_EMBEDDING_RANKER,
        ) as mock_emb, patch(
            _PATCH_RANKER,
        ) as mock_ranker_cls, patch(
            _PATCH_DEDUP_MANAGER,
        ) as mock_dedup_cls, patch(
            _PATCH_CODE_DETECTOR,
        ):
            mock_inst = MagicMock()
            mock_inst.mode = "disabled"
            mock_inst.available = False
            mock_emb.return_value = mock_inst

            mock_ranker_inst = MagicMock()
            mock_ranker_inst._default_threshold = 60
            mock_ranker_cls.return_value = mock_ranker_inst

            mock_dedup_cls.return_value = MagicMock()

            orch.run_all_topics()

        # Verify update_run_status was called with "completed"
        found_completed = False
        for c in db.update_run_status.call_args_list:
            if c[0] == (55, "completed"):
                found_completed = True
        assert found_completed, (
            "Expected db.update_run_status(55, 'completed') to be called"
        )


# ---------------------------------------------------------------------------
# 14. Usage tracker updated on completion
# ---------------------------------------------------------------------------


class TestUsageTrackerOnCompletion:
    """record_topic_completed is called on the usage_tracker for success."""

    def test_usage_tracker_records_completion(self) -> None:
        config = MockConfig(topics=[MockTopic("track-ok")])
        usage = MagicMock()

        orch = _make_orchestrator(config=config, usage_tracker=usage)

        with patch.object(
            orch, "_run_single_topic",
            return_value={"total_output": 7},
        ):
            orch.run_all_topics()

        usage.record_topic_completed.assert_called_once_with("track-ok", 7)


# ---------------------------------------------------------------------------
# 15. Usage tracker updated on failure
# ---------------------------------------------------------------------------


class TestUsageTrackerOnFailure:
    """Failed topic appears in topics_failed (usage_tracker not called for completion)."""

    def test_failed_topic_not_in_completed(self) -> None:
        config = MockConfig(topics=[MockTopic("track-fail")])
        usage = MagicMock()

        orch = _make_orchestrator(config=config, usage_tracker=usage)

        with patch.object(
            orch, "_run_single_topic",
            side_effect=RuntimeError("fail!"),
        ):
            summary = orch.run_all_topics()

        usage.record_topic_completed.assert_not_called()
        assert len(summary["topics_failed"]) == 1
        assert summary["topics_failed"][0]["slug"] == "track-fail"


# ---------------------------------------------------------------------------
# 16. Rate limiter consulted before each topic
# ---------------------------------------------------------------------------


class TestRateLimiterConsulted:
    """is_daily_limit_reached is checked before each topic iteration."""

    def test_daily_limit_checked_per_topic(self) -> None:
        config = MockConfig(
            topics=[MockTopic("a"), MockTopic("b"), MockTopic("c")]
        )
        rate_limiter = MagicMock()
        rate_limiter.should_skip_topic = MagicMock(return_value=False)

        # Track how many times is_daily_limit_reached is read.
        # The code does: if self._rate_limiter.is_daily_limit_reached:
        # We use PropertyMock to intercept the attribute access.
        prop_mock = PropertyMock(return_value=False)
        type(rate_limiter).is_daily_limit_reached = prop_mock

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)

        with patch.object(
            orch, "_run_single_topic",
            return_value={"total_output": 0},
        ):
            orch.run_all_topics()

        # The property is accessed once per topic in the loop
        assert prop_mock.call_count == 3

    def test_should_skip_topic_called_per_topic(self) -> None:
        config = MockConfig(topics=[MockTopic("x"), MockTopic("y")])
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)

        with patch.object(
            orch, "_run_single_topic",
            return_value={"total_output": 0},
        ):
            orch.run_all_topics()

        assert rate_limiter.should_skip_topic.call_count == 2
        rate_limiter.should_skip_topic.assert_any_call("x")
        rate_limiter.should_skip_topic.assert_any_call("y")


# ---------------------------------------------------------------------------
# 17. Rate limiter record_topic_completed called on success
# ---------------------------------------------------------------------------


class TestRateLimiterRecordsCompletion:
    """rate_limiter.record_topic_completed is called on success."""

    def test_rate_limiter_records_on_success(self) -> None:
        config = MockConfig(topics=[MockTopic("rl-ok")])
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(return_value=False)

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)

        with patch.object(
            orch, "_run_single_topic",
            return_value={"total_output": 1},
        ):
            orch.run_all_topics()

        rate_limiter.record_topic_completed.assert_called_once_with("rl-ok")


# ---------------------------------------------------------------------------
# 18. Usage tracker records skipped topics
# ---------------------------------------------------------------------------


class TestUsageTrackerSkipped:
    """record_topic_skipped is called when topics are skipped."""

    def test_daily_limit_skip_records_in_tracker(self) -> None:
        config = MockConfig(topics=[MockTopic("skipped-a")])
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = True
        usage = MagicMock()

        orch = _make_orchestrator(
            config=config, rate_limiter=rate_limiter, usage_tracker=usage,
        )
        orch.run_all_topics()

        usage.record_topic_skipped.assert_called_once_with(
            "skipped-a", "daily_limit_reached",
        )

    def test_already_completed_skip_records_in_tracker(self) -> None:
        config = MockConfig(topics=[MockTopic("skipped-b")])
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(return_value=True)
        usage = MagicMock()

        orch = _make_orchestrator(
            config=config, rate_limiter=rate_limiter, usage_tracker=usage,
        )
        orch.run_all_topics()

        usage.record_topic_skipped.assert_called_once_with(
            "skipped-b", "already_completed_today",
        )


# ---------------------------------------------------------------------------
# 19. Multiple topics: mixed completed, skipped, failed
# ---------------------------------------------------------------------------


class TestMixedResults:
    """Orchestrator correctly categorizes topics across all three buckets."""

    def test_mixed_completed_skipped_failed(self) -> None:
        config = MockConfig(
            topics=[
                MockTopic("ok-1"),
                MockTopic("skip-me"),
                MockTopic("fail-me"),
                MockTopic("ok-2"),
            ]
        )
        rate_limiter = MagicMock()
        rate_limiter.is_daily_limit_reached = False
        rate_limiter.should_skip_topic = MagicMock(
            side_effect=lambda slug: slug == "skip-me"
        )

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)

        def side_effect(topic: Any) -> Dict[str, Any]:
            if topic.slug == "fail-me":
                raise RuntimeError("intentional")
            return {"total_output": 2}

        with patch.object(orch, "_run_single_topic", side_effect=side_effect):
            summary = orch.run_all_topics()

        completed_slugs = [e["slug"] for e in summary["topics_completed"]]
        skipped_slugs = [e["slug"] for e in summary["topics_skipped"]]
        failed_slugs = [e["slug"] for e in summary["topics_failed"]]

        assert completed_slugs == ["ok-1", "ok-2"]
        assert skipped_slugs == ["skip-me"]
        assert failed_slugs == ["fail-me"]


# ---------------------------------------------------------------------------
# 20. _resolve_topics with non-existent slug returns empty
# ---------------------------------------------------------------------------


class TestResolveNonExistentSlug:
    """Filtering by a slug that does not exist returns an empty list."""

    def test_non_existent_slug_returns_empty(self) -> None:
        config = MockConfig(topics=[MockTopic("alpha")])
        orch = _make_orchestrator(
            config=config,
            run_options={"topics": ["no-such-topic"]},
        )

        resolved = orch._resolve_topics()
        assert resolved == []


# ---------------------------------------------------------------------------
# 21. Daily limit transitions mid-loop
# ---------------------------------------------------------------------------


class TestDailyLimitMidLoop:
    """Daily limit becomes True after the first topic completes."""

    def test_limit_reached_after_first_topic(self) -> None:
        config = MockConfig(
            topics=[MockTopic("first"), MockTopic("second")]
        )
        rate_limiter = MagicMock()
        rate_limiter.should_skip_topic = MagicMock(return_value=False)

        # First access returns False (first topic runs),
        # second access returns True (second topic skipped).
        type(rate_limiter).is_daily_limit_reached = PropertyMock(
            side_effect=[False, True]
        )

        orch = _make_orchestrator(config=config, rate_limiter=rate_limiter)

        with patch.object(
            orch, "_run_single_topic",
            return_value={"total_output": 1},
        ):
            summary = orch.run_all_topics()

        assert len(summary["topics_completed"]) == 1
        assert summary["topics_completed"][0]["slug"] == "first"
        assert len(summary["topics_skipped"]) == 1
        assert summary["topics_skipped"][0]["slug"] == "second"
