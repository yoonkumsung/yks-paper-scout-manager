"""Tests for core.remind.remind_selector (TASK-022).

Covers basic selection, count graduation, count increment, summary reuse,
current-run exclusion, topic isolation, empty results, first-time tracking,
score boundary, discarded exclusion, multiple runs, and tracking update.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.models import Evaluation, EvaluationFlags, Paper, RemindTracking, RunMeta
from core.remind.remind_selector import RemindSelector
from core.storage.db_manager import DBManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """Provide a fresh DBManager instance backed by a temp file."""
    manager = DBManager(db_path=str(tmp_path / "test_remind.db"))
    yield manager
    manager.close()


@pytest.fixture()
def selector(db):
    """Provide a RemindSelector wired to the test DB."""
    return RemindSelector(db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_run(db: DBManager, topic_slug: str = "llm-agents", status: str = "completed") -> int:
    """Create a run and return its run_id."""
    meta = RunMeta(
        topic_slug=topic_slug,
        window_start_utc=_NOW,
        window_end_utc=_NOW + timedelta(hours=24),
        display_date_kst="2026-02-16",
        embedding_mode="disabled",
        scoring_weights={"embed": 0.3, "llm": 0.7},
        response_format_supported=True,
        prompt_versions={"agent1": "v3", "agent2": "v2"},
        status=status,
    )
    return db.create_run(meta)


def _make_paper(
    db: DBManager,
    paper_key: str = "arxiv:2401.00001",
    run_id: int = 1,
    title: str = "Test Paper",
) -> Paper:
    """Insert and return a Paper."""
    paper = Paper(
        source="arxiv",
        native_id=paper_key.split(":")[1],
        paper_key=paper_key,
        url=f"https://arxiv.org/abs/{paper_key.split(':')[1]}",
        title=title,
        abstract="An abstract.",
        authors=["Alice"],
        categories=["cs.AI", "cs.CL"],
        published_at_utc=datetime(2026, 2, 9, tzinfo=timezone.utc),
        first_seen_run_id=run_id,
        created_at=_NOW.isoformat(),
    )
    db.insert_paper(paper)
    return paper


def _make_eval(
    db: DBManager,
    run_id: int,
    paper_key: str = "arxiv:2401.00001",
    final_score: float = 85.0,
    discarded: bool = False,
    summary_ko: str | None = "Summary text",
    reason_ko: str | None = "Reason text",
    insight_ko: str | None = "Insight text",
) -> Evaluation:
    """Insert and return an Evaluation."""
    ev = Evaluation(
        run_id=run_id,
        paper_key=paper_key,
        llm_base_score=80,
        flags=EvaluationFlags(is_edge=True),
        prompt_ver_score="agent2-v2",
        final_score=final_score,
        rank=1,
        tier=1,
        discarded=discarded,
        summary_ko=summary_ko,
        reason_ko=reason_ko,
        insight_ko=insight_ko,
        prompt_ver_summ="agent3-v1",
    )
    db.insert_evaluation(ev)
    return ev


# ---------------------------------------------------------------------------
# Test 1: Basic selection - papers with score >= 80 are selected
# ---------------------------------------------------------------------------


class TestBasicSelection:
    def test_high_score_papers_selected(self, db, selector):
        """Papers with final_score >= 80 from a previous run appear in remind."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(db, past_run, "arxiv:0001", final_score=90.0)

        results = selector.select("llm-agents", current_run)

        assert len(results) == 1
        assert results[0]["paper_key"] == "arxiv:0001"
        assert results[0]["is_remind"] is True


# ---------------------------------------------------------------------------
# Test 2: Count graduation - papers with recommend_count == 2 excluded
# ---------------------------------------------------------------------------


class TestCountGraduation:
    def test_graduated_papers_excluded(self, db, selector):
        """Papers that have been recommended 2 times should graduate."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(db, past_run, "arxiv:0001", final_score=90.0)

        # Pre-set tracking to recommend_count=2 (graduated)
        db.upsert_remind_tracking(
            RemindTracking(
                paper_key="arxiv:0001",
                topic_slug="llm-agents",
                last_recommend_run_id=past_run,
                recommend_count=2,
            )
        )

        results = selector.select("llm-agents", current_run)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 3: Count increment - recommend_count increments on selection
# ---------------------------------------------------------------------------


class TestCountIncrement:
    def test_recommend_count_incremented(self, db, selector):
        """Selecting a paper should increment its recommend_count by 1."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(db, past_run, "arxiv:0001", final_score=85.0)

        results = selector.select("llm-agents", current_run)

        assert len(results) == 1
        assert results[0]["recommend_count"] == 1

        # Verify the DB tracking was updated
        tracking = db.get_remind_tracking("arxiv:0001", "llm-agents")
        assert tracking is not None
        assert tracking.recommend_count == 1
        assert tracking.last_recommend_run_id == current_run


# ---------------------------------------------------------------------------
# Test 4: Summary reuse - summaries copied from latest evaluation
# ---------------------------------------------------------------------------


class TestSummaryReuse:
    def test_summaries_reused_from_evaluation(self, db, selector):
        """summary_ko, reason_ko, insight_ko should be reused from the latest eval."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(
            db,
            past_run,
            "arxiv:0001",
            final_score=85.0,
            summary_ko="Summary KO",
            reason_ko="Reason KO",
            insight_ko="Insight KO",
        )

        results = selector.select("llm-agents", current_run)

        assert len(results) == 1
        assert results[0]["summary_ko"] == "Summary KO"
        assert results[0]["reason_ko"] == "Reason KO"
        assert results[0]["insight_ko"] == "Insight KO"


# ---------------------------------------------------------------------------
# Test 5: Current run exclusion - papers from current_run_id not included
# ---------------------------------------------------------------------------


class TestCurrentRunExclusion:
    def test_current_run_papers_excluded(self, db, selector):
        """Papers evaluated in the current run should NOT appear in remind."""
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", current_run)
        _make_eval(db, current_run, "arxiv:0001", final_score=95.0)

        results = selector.select("llm-agents", current_run)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 6: Topic isolation - papers from other topics not included
# ---------------------------------------------------------------------------


class TestTopicIsolation:
    def test_other_topic_papers_excluded(self, db, selector):
        """Papers from topic B should not appear in topic A's remind."""
        run_topic_a = _make_run(db, topic_slug="topic-a")
        run_topic_b = _make_run(db, topic_slug="topic-b")
        current_run_a = _make_run(db, topic_slug="topic-a")

        _make_paper(db, "arxiv:0001", run_topic_a)
        _make_paper(db, "arxiv:0002", run_topic_b)

        _make_eval(db, run_topic_a, "arxiv:0001", final_score=90.0)
        _make_eval(db, run_topic_b, "arxiv:0002", final_score=92.0)

        results = selector.select("topic-a", current_run_a)

        paper_keys = [r["paper_key"] for r in results]
        assert "arxiv:0001" in paper_keys
        assert "arxiv:0002" not in paper_keys


# ---------------------------------------------------------------------------
# Test 7: Empty results - no candidates returns empty list
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_no_candidates_returns_empty(self, db, selector):
        """If no papers meet the criteria, return an empty list."""
        current_run = _make_run(db)
        results = selector.select("llm-agents", current_run)
        assert results == []


# ---------------------------------------------------------------------------
# Test 8: First-time tracking - papers without tracking records initialized
# ---------------------------------------------------------------------------


class TestFirstTimeTracking:
    def test_creates_tracking_for_untracked_papers(self, db, selector):
        """Papers scoring >= 80 without a tracking record get one created."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(db, past_run, "arxiv:0001", final_score=85.0)

        # Before selection, no tracking exists
        assert db.get_remind_tracking("arxiv:0001", "llm-agents") is None

        results = selector.select("llm-agents", current_run)

        assert len(results) == 1
        tracking = db.get_remind_tracking("arxiv:0001", "llm-agents")
        assert tracking is not None
        assert tracking.recommend_count == 1


# ---------------------------------------------------------------------------
# Test 9: Score boundary - exactly 80 included, below 80 excluded
# ---------------------------------------------------------------------------


class TestScoreBoundary:
    def test_score_at_boundary(self, db, selector):
        """Papers at exactly 80 should be included; below 80 excluded."""
        past_run = _make_run(db)
        current_run = _make_run(db)

        _make_paper(db, "arxiv:at80", past_run)
        _make_paper(db, "arxiv:below80", past_run)

        _make_eval(db, past_run, "arxiv:at80", final_score=80.0)
        _make_eval(db, past_run, "arxiv:below80", final_score=79.9)

        results = selector.select("llm-agents", current_run)

        paper_keys = [r["paper_key"] for r in results]
        assert "arxiv:at80" in paper_keys
        assert "arxiv:below80" not in paper_keys


# ---------------------------------------------------------------------------
# Test 10: Discarded exclusion - discarded papers never selected
# ---------------------------------------------------------------------------


class TestDiscardedExclusion:
    def test_discarded_papers_excluded(self, db, selector):
        """Discarded papers should never appear in remind, even if score >= 80."""
        past_run = _make_run(db)
        current_run = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(
            db, past_run, "arxiv:0001", final_score=90.0, discarded=True
        )

        results = selector.select("llm-agents", current_run)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 11: Multiple runs - papers from multiple past runs considered
# ---------------------------------------------------------------------------


class TestMultipleRuns:
    def test_papers_from_multiple_runs(self, db, selector):
        """High-score papers from different past runs should all be candidates."""
        run1 = _make_run(db)
        run2 = _make_run(db)
        current_run = _make_run(db)

        _make_paper(db, "arxiv:0001", run1)
        _make_paper(db, "arxiv:0002", run2)

        _make_eval(db, run1, "arxiv:0001", final_score=85.0)
        _make_eval(db, run2, "arxiv:0002", final_score=90.0)

        results = selector.select("llm-agents", current_run)

        paper_keys = {r["paper_key"] for r in results}
        assert paper_keys == {"arxiv:0001", "arxiv:0002"}

        # Higher score should come first
        assert results[0]["paper_key"] == "arxiv:0002"
        assert results[1]["paper_key"] == "arxiv:0001"


# ---------------------------------------------------------------------------
# Test 12: Tracking update - DB tracking records properly updated
# ---------------------------------------------------------------------------


class TestTrackingUpdate:
    def test_tracking_updated_after_selection(self, db, selector):
        """After selection, tracking records should reflect incremented count."""
        past_run = _make_run(db)
        current_run1 = _make_run(db)
        _make_paper(db, "arxiv:0001", past_run)
        _make_eval(db, past_run, "arxiv:0001", final_score=85.0)

        # First selection: recommend_count goes 0 -> 1
        results1 = selector.select("llm-agents", current_run1)
        assert len(results1) == 1
        assert results1[0]["recommend_count"] == 1

        tracking1 = db.get_remind_tracking("arxiv:0001", "llm-agents")
        assert tracking1.recommend_count == 1
        assert tracking1.last_recommend_run_id == current_run1

        # Second selection: recommend_count goes 1 -> 2
        current_run2 = _make_run(db)
        results2 = selector.select("llm-agents", current_run2)
        assert len(results2) == 1
        assert results2[0]["recommend_count"] == 2

        tracking2 = db.get_remind_tracking("arxiv:0001", "llm-agents")
        assert tracking2.recommend_count == 2

        # Third selection: recommend_count is 2 -> graduated, not selected
        current_run3 = _make_run(db)
        results3 = selector.select("llm-agents", current_run3)
        assert len(results3) == 0
