"""Tests for core.storage.db_manager (P1 priority).

Covers table creation, CRUD for all five tables, purge operations,
VACUUM, db_stats, and context manager protocol.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from core.models import Evaluation, EvaluationFlags, Paper, QueryStats, RemindTracking, RunMeta
from core.storage.db_manager import DBManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test.db")


@pytest.fixture()
def db(db_path):
    """Provide a fresh DBManager instance, closed after use."""
    manager = DBManager(db_path=db_path)
    yield manager
    manager.close()


def _make_paper(
    paper_key: str = "arxiv:2401.00001",
    run_id: int = 1,
    created_at: str | None = None,
) -> Paper:
    """Create a sample Paper dataclass."""
    return Paper(
        source="arxiv",
        native_id="2401.00001",
        paper_key=paper_key,
        url="https://arxiv.org/abs/2401.00001",
        title="Test Paper",
        abstract="An abstract about testing.",
        authors=["Alice", "Bob"],
        categories=["cs.AI", "cs.CL"],
        published_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        first_seen_run_id=run_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def _make_run_meta(
    topic_slug: str = "llm-agents",
    status: str = "running",
    window_start: datetime | None = None,
) -> RunMeta:
    """Create a sample RunMeta dataclass."""
    now = window_start or datetime.now(timezone.utc)
    return RunMeta(
        topic_slug=topic_slug,
        window_start_utc=now,
        window_end_utc=now + timedelta(hours=24),
        display_date_kst="2024-01-02",
        embedding_mode="disabled",
        scoring_weights={"embed": 0.3, "llm": 0.7},
        response_format_supported=True,
        prompt_versions={"agent1": "v3", "agent2": "v2"},
        status=status,
    )


def _make_evaluation(
    run_id: int = 1,
    paper_key: str = "arxiv:2401.00001",
    llm_base_score: int = 80,
    final_score: float | None = 85.0,
) -> Evaluation:
    """Create a sample Evaluation dataclass."""
    return Evaluation(
        run_id=run_id,
        paper_key=paper_key,
        llm_base_score=llm_base_score,
        flags=EvaluationFlags(is_edge=True, mentions_code=True),
        prompt_ver_score="agent2-v2",
        final_score=final_score,
        rank=1,
        tier=1,
    )


# ---------------------------------------------------------------------------
# 1. DB creation and table schema verification
# ---------------------------------------------------------------------------


class TestDBCreation:
    """Test 1: DB file creation and table schema."""

    def test_creates_db_file(self, db_path):
        """DB file should exist after initialization."""
        manager = DBManager(db_path=db_path)
        assert os.path.exists(db_path)
        manager.close()

    def test_creates_parent_directory(self, tmp_path):
        """Parent directories should be created automatically."""
        nested_path = str(tmp_path / "a" / "b" / "test.db")
        manager = DBManager(db_path=nested_path)
        assert os.path.exists(nested_path)
        manager.close()

    def test_all_tables_exist(self, db):
        """All five expected tables should be present."""
        tables = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        expected = {
            "papers",
            "paper_evaluations",
            "runs",
            "query_stats",
            "remind_tracking",
        }
        # sqlite_sequence is auto-created by AUTOINCREMENT; exclude it
        table_names.discard("sqlite_sequence")
        assert table_names == expected


# ---------------------------------------------------------------------------
# 2. Paper insert and retrieve
# ---------------------------------------------------------------------------


class TestPaperCRUD:
    """Tests 2-4: Paper insert, retrieve, exists, and duplicate handling."""

    def test_insert_and_get_paper(self, db):
        """Insert a paper and retrieve it by key."""
        paper = _make_paper()
        db.insert_paper(paper)

        result = db.get_paper("arxiv:2401.00001")
        assert result is not None
        assert result.paper_key == "arxiv:2401.00001"
        assert result.title == "Test Paper"
        assert result.authors == ["Alice", "Bob"]
        assert result.categories == ["cs.AI", "cs.CL"]
        assert result.has_code is False
        assert result.has_code_source == "none"

    def test_get_paper_not_found(self, db):
        """get_paper returns None for non-existent key."""
        assert db.get_paper("nonexistent") is None

    def test_paper_exists(self, db):
        """paper_exists returns correct boolean."""
        db.insert_paper(_make_paper())
        assert db.paper_exists("arxiv:2401.00001") is True
        assert db.paper_exists("nonexistent") is False

    def test_duplicate_paper_key_ignored(self, db):
        """Inserting a duplicate paper_key should be silently ignored."""
        paper1 = _make_paper()
        paper2 = _make_paper()
        paper2.title = "Updated Title"  # different title

        db.insert_paper(paper1)
        db.insert_paper(paper2)

        result = db.get_paper("arxiv:2401.00001")
        assert result is not None
        # Original title preserved (INSERT OR IGNORE)
        assert result.title == "Test Paper"

    def test_update_paper_code_info(self, db):
        """update_paper_code_info modifies code-related fields."""
        db.insert_paper(_make_paper())
        db.update_paper_code_info(
            "arxiv:2401.00001",
            has_code=True,
            has_code_source="regex",
            code_url="https://github.com/example/repo",
        )

        result = db.get_paper("arxiv:2401.00001")
        assert result is not None
        assert result.has_code is True
        assert result.has_code_source == "regex"
        assert result.code_url == "https://github.com/example/repo"


# ---------------------------------------------------------------------------
# 5-7. Evaluation CRUD
# ---------------------------------------------------------------------------


class TestEvaluationCRUD:
    """Tests 5-7: Evaluation insert, retrieve by run, latest, high score."""

    def _setup_run(self, db, topic_slug="llm-agents") -> int:
        """Insert a run and return its run_id."""
        return db.create_run(_make_run_meta(topic_slug=topic_slug))

    def test_insert_and_get_by_run(self, db):
        """Insert evaluations and retrieve them by run_id."""
        run_id = self._setup_run(db)
        db.insert_paper(_make_paper("arxiv:p1", run_id=run_id))
        db.insert_paper(_make_paper("arxiv:p2", run_id=run_id))

        ev1 = _make_evaluation(run_id=run_id, paper_key="arxiv:p1")
        ev2 = _make_evaluation(run_id=run_id, paper_key="arxiv:p2", llm_base_score=60)

        db.insert_evaluation(ev1)
        db.insert_evaluation(ev2)

        results = db.get_evaluations_by_run(run_id)
        assert len(results) == 2
        keys = {e.paper_key for e in results}
        assert keys == {"arxiv:p1", "arxiv:p2"}

    def test_evaluation_flags_serialization(self, db):
        """EvaluationFlags should round-trip through JSON correctly."""
        run_id = self._setup_run(db)
        db.insert_paper(_make_paper(run_id=run_id))

        ev = _make_evaluation(run_id=run_id)
        db.insert_evaluation(ev)

        result = db.get_evaluations_by_run(run_id)[0]
        assert result.flags.is_edge is True
        assert result.flags.mentions_code is True
        assert result.flags.is_realtime is False
        assert result.flags.is_metaphorical is False

    def test_get_latest_evaluation(self, db):
        """get_latest_evaluation returns the most recent for paper+topic."""
        run_id_1 = db.create_run(_make_run_meta(topic_slug="llm-agents"))
        run_id_2 = db.create_run(_make_run_meta(topic_slug="llm-agents"))

        db.insert_paper(_make_paper("arxiv:p1", run_id=run_id_1))
        db.insert_evaluation(
            _make_evaluation(run_id=run_id_1, paper_key="arxiv:p1", llm_base_score=60)
        )
        db.insert_evaluation(
            _make_evaluation(run_id=run_id_2, paper_key="arxiv:p1", llm_base_score=90)
        )

        result = db.get_latest_evaluation("arxiv:p1", "llm-agents")
        assert result is not None
        assert result.run_id == run_id_2
        assert result.llm_base_score == 90

    def test_get_latest_evaluation_not_found(self, db):
        """get_latest_evaluation returns None when no match exists."""
        assert db.get_latest_evaluation("nonexistent", "llm-agents") is None

    def test_get_high_score_papers(self, db):
        """get_high_score_papers filters by final_score >= threshold."""
        run_id = self._setup_run(db)
        db.insert_paper(_make_paper("arxiv:high", run_id=run_id))
        db.insert_paper(_make_paper("arxiv:low", run_id=run_id))

        db.insert_evaluation(
            _make_evaluation(run_id=run_id, paper_key="arxiv:high", final_score=90.0)
        )
        db.insert_evaluation(
            _make_evaluation(run_id=run_id, paper_key="arxiv:low", final_score=40.0)
        )

        results = db.get_high_score_papers("llm-agents", 80.0)
        assert len(results) == 1
        assert results[0].paper_key == "arxiv:high"


# ---------------------------------------------------------------------------
# 8. Run create and status update
# ---------------------------------------------------------------------------


class TestRunCRUD:
    """Tests 8: Run create, status update, stats update."""

    def test_create_run_returns_auto_increment_id(self, db):
        """create_run should return an auto-incremented run_id."""
        run_id_1 = db.create_run(_make_run_meta())
        run_id_2 = db.create_run(_make_run_meta())

        assert run_id_1 == 1
        assert run_id_2 == 2

    def test_update_run_status(self, db):
        """update_run_status should change status and errors."""
        run_id = db.create_run(_make_run_meta())
        db.update_run_status(run_id, "completed")

        result = db.get_latest_completed_run("llm-agents")
        assert result is not None
        assert result.status == "completed"

    def test_update_run_status_with_error(self, db):
        """update_run_status can set error messages."""
        run_id = db.create_run(_make_run_meta())
        db.update_run_status(run_id, "failed", errors="API rate limit exceeded")

        row = db._conn.execute(
            "SELECT status, errors FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "failed"
        assert row["errors"] == "API rate limit exceeded"

    def test_update_run_stats(self, db):
        """update_run_stats should update numeric counters."""
        run_id = db.create_run(_make_run_meta())
        db.update_run_stats(
            run_id,
            total_collected=100,
            total_filtered=80,
            total_scored=70,
            total_output=50,
        )

        row = db._conn.execute(
            "SELECT total_collected, total_filtered, total_scored, total_output "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row["total_collected"] == 100
        assert row["total_filtered"] == 80
        assert row["total_scored"] == 70
        assert row["total_output"] == 50

    def test_update_run_stats_ignores_unknown_keys(self, db):
        """update_run_stats should silently ignore unknown columns."""
        run_id = db.create_run(_make_run_meta())
        # Should not raise
        db.update_run_stats(run_id, unknown_field=42, total_collected=10)

        row = db._conn.execute(
            "SELECT total_collected FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["total_collected"] == 10

    def test_get_latest_completed_run(self, db):
        """get_latest_completed_run returns most recent completed run."""
        run_id_1 = db.create_run(_make_run_meta(status="completed"))
        db.update_run_status(run_id_1, "completed")
        run_id_2 = db.create_run(_make_run_meta(status="running"))

        result = db.get_latest_completed_run("llm-agents")
        assert result is not None
        assert result.run_id == run_id_1

    def test_get_latest_completed_run_none(self, db):
        """get_latest_completed_run returns None when no completed runs."""
        db.create_run(_make_run_meta(status="running"))
        assert db.get_latest_completed_run("llm-agents") is None

    def test_run_json_fields_roundtrip(self, db):
        """JSON fields (scoring_weights, prompt_versions) should round-trip."""
        meta = _make_run_meta(status="completed")
        run_id = db.create_run(meta)
        db.update_run_status(run_id, "completed")

        result = db.get_latest_completed_run("llm-agents")
        assert result is not None
        assert result.scoring_weights == {"embed": 0.3, "llm": 0.7}
        assert result.prompt_versions == {"agent1": "v3", "agent2": "v2"}


# ---------------------------------------------------------------------------
# 9. QueryStats insert
# ---------------------------------------------------------------------------


class TestQueryStats:
    """Test 9: QueryStats insert."""

    def test_insert_query_stats(self, db):
        """insert_query_stats should persist a record."""
        run_id = db.create_run(_make_run_meta())
        stats = QueryStats(
            run_id=run_id,
            query_text="llm agents 2024",
            collected=50,
            total_available=200,
            truncated=True,
            retries=1,
            duration_ms=1200,
        )
        db.insert_query_stats(stats)

        row = db._conn.execute(
            "SELECT * FROM query_stats WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row["query_text"] == "llm agents 2024"
        assert row["collected"] == 50
        assert row["total_available"] == 200
        assert row["truncated"] == 1
        assert row["retries"] == 1
        assert row["duration_ms"] == 1200


# ---------------------------------------------------------------------------
# 10. RemindTracking upsert
# ---------------------------------------------------------------------------


class TestRemindTracking:
    """Test 10: RemindTracking upsert (insert + update)."""

    def test_upsert_insert(self, db):
        """First upsert should insert a new record."""
        tracking = RemindTracking(
            paper_key="arxiv:2401.00001",
            topic_slug="llm-agents",
            recommend_count=1,
            last_recommend_run_id=1,
        )
        db.upsert_remind_tracking(tracking)

        result = db.get_remind_tracking("arxiv:2401.00001", "llm-agents")
        assert result is not None
        assert result.recommend_count == 1

    def test_upsert_update(self, db):
        """Second upsert should update existing record."""
        tracking = RemindTracking(
            paper_key="arxiv:2401.00001",
            topic_slug="llm-agents",
            recommend_count=1,
            last_recommend_run_id=1,
        )
        db.upsert_remind_tracking(tracking)

        tracking.recommend_count = 2
        tracking.last_recommend_run_id = 5
        db.upsert_remind_tracking(tracking)

        result = db.get_remind_tracking("arxiv:2401.00001", "llm-agents")
        assert result is not None
        assert result.recommend_count == 2
        assert result.last_recommend_run_id == 5

    def test_get_remind_tracking_not_found(self, db):
        """get_remind_tracking returns None for non-existent entry."""
        assert db.get_remind_tracking("nonexistent", "test") is None

    def test_get_remind_candidates(self, db):
        """get_remind_candidates returns papers meeting criteria."""
        run_id = db.create_run(_make_run_meta())
        db.insert_paper(_make_paper("arxiv:candidate", run_id=run_id))
        db.insert_evaluation(
            _make_evaluation(
                run_id=run_id, paper_key="arxiv:candidate", final_score=90.0
            )
        )
        db.upsert_remind_tracking(
            RemindTracking(
                paper_key="arxiv:candidate",
                topic_slug="llm-agents",
                recommend_count=1,
                last_recommend_run_id=run_id,
            )
        )

        candidates = db.get_remind_candidates("llm-agents", min_score=80.0, max_count=3)
        assert len(candidates) == 1
        assert candidates[0]["paper_key"] == "arxiv:candidate"


# ---------------------------------------------------------------------------
# 11-13. Purge operations
# ---------------------------------------------------------------------------


class TestPurgeOperations:
    """Tests 11-13: Purge old evaluations, papers, and orphans."""

    def test_purge_old_evaluations(self, db):
        """Evaluations linked to old runs should be purged."""
        old_start = datetime.now(timezone.utc) - timedelta(days=100)
        run_id = db.create_run(
            _make_run_meta(status="completed", window_start=old_start)
        )
        db.insert_paper(_make_paper("arxiv:old", run_id=run_id))
        db.insert_evaluation(
            _make_evaluation(run_id=run_id, paper_key="arxiv:old")
        )

        deleted = db.purge_old_evaluations(days=90)
        assert deleted == 1
        assert db.get_evaluations_by_run(run_id) == []

    def test_purge_old_evaluations_keeps_recent(self, db):
        """Recent evaluations should not be purged."""
        run_id = db.create_run(_make_run_meta(status="completed"))
        db.insert_paper(_make_paper(run_id=run_id))
        db.insert_evaluation(_make_evaluation(run_id=run_id))

        deleted = db.purge_old_evaluations(days=90)
        assert deleted == 0
        assert len(db.get_evaluations_by_run(run_id)) == 1

    def test_purge_old_papers(self, db):
        """Papers older than 365 days should be purged."""
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        db.insert_paper(_make_paper("arxiv:ancient", created_at=old_ts))

        deleted = db.purge_old_papers(days=365)
        assert deleted == 1
        assert db.get_paper("arxiv:ancient") is None

    def test_purge_old_papers_keeps_recent(self, db):
        """Recent papers should not be purged."""
        db.insert_paper(_make_paper())

        deleted = db.purge_old_papers(days=365)
        assert deleted == 0
        assert db.paper_exists("arxiv:2401.00001") is True

    def test_purge_orphan_remind_tracking(self, db):
        """Remind entries with no evaluations should be purged."""
        db.upsert_remind_tracking(
            RemindTracking(
                paper_key="arxiv:orphan",
                topic_slug="llm-agents",
                recommend_count=1,
                last_recommend_run_id=1,
            )
        )

        deleted = db.purge_orphan_remind_tracking()
        assert deleted == 1
        assert db.get_remind_tracking("arxiv:orphan", "llm-agents") is None

    def test_purge_orphan_keeps_valid(self, db):
        """Remind entries with matching evaluations should remain."""
        run_id = db.create_run(_make_run_meta())
        db.insert_paper(_make_paper(run_id=run_id))
        db.insert_evaluation(_make_evaluation(run_id=run_id))
        db.upsert_remind_tracking(
            RemindTracking(
                paper_key="arxiv:2401.00001",
                topic_slug="llm-agents",
                recommend_count=1,
                last_recommend_run_id=run_id,
            )
        )

        deleted = db.purge_orphan_remind_tracking()
        assert deleted == 0

    def test_purge_old_runs(self, db):
        """Runs older than 90 days should be purged."""
        old_start = datetime.now(timezone.utc) - timedelta(days=100)
        db.create_run(_make_run_meta(window_start=old_start))

        deleted = db.purge_old_runs(days=90)
        assert deleted == 1

    def test_purge_old_query_stats(self, db):
        """Query stats linked to old runs should be purged."""
        old_start = datetime.now(timezone.utc) - timedelta(days=100)
        run_id = db.create_run(_make_run_meta(window_start=old_start))
        db.insert_query_stats(
            QueryStats(run_id=run_id, query_text="test query", collected=10)
        )

        deleted = db.purge_old_query_stats(days=90)
        assert deleted == 1


# ---------------------------------------------------------------------------
# 14. VACUUM execution
# ---------------------------------------------------------------------------


class TestVacuum:
    """Test 14: VACUUM does not raise an error."""

    def test_vacuum_runs_without_error(self, db):
        """VACUUM should execute without raising."""
        db.insert_paper(_make_paper())
        db.vacuum()  # should not raise


# ---------------------------------------------------------------------------
# 15. DB stats
# ---------------------------------------------------------------------------


class TestDBStats:
    """Test 15: get_db_stats returns record counts and file size."""

    def test_db_stats_empty(self, db):
        """Stats for an empty database should all be zero."""
        stats = db.get_db_stats()
        assert stats["papers"] == 0
        assert stats["paper_evaluations"] == 0
        assert stats["runs"] == 0
        assert stats["query_stats"] == 0
        assert stats["remind_tracking"] == 0
        assert stats["file_size_bytes"] > 0  # DB file exists

    def test_db_stats_with_data(self, db):
        """Stats should reflect inserted records."""
        run_id = db.create_run(_make_run_meta())
        db.insert_paper(_make_paper(run_id=run_id))
        db.insert_evaluation(_make_evaluation(run_id=run_id))

        stats = db.get_db_stats()
        assert stats["papers"] == 1
        assert stats["paper_evaluations"] == 1
        assert stats["runs"] == 1


# ---------------------------------------------------------------------------
# 16. Context manager (with statement)
# ---------------------------------------------------------------------------


class TestContextManager:
    """Test 16: Context manager support."""

    def test_context_manager_closes_connection(self, db_path):
        """Connection should be closed after exiting 'with' block."""
        with DBManager(db_path=db_path) as manager:
            manager.insert_paper(_make_paper())
            assert manager.paper_exists("arxiv:2401.00001") is True

        # Re-open to verify data was persisted
        manager2 = DBManager(db_path=db_path)
        assert manager2.paper_exists("arxiv:2401.00001") is True
        manager2.close()

    def test_context_manager_on_exception(self, db_path):
        """Connection should close even when exception occurs."""
        try:
            with DBManager(db_path=db_path) as manager:
                manager.insert_paper(_make_paper())
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # Re-open to verify committed data persists
        manager2 = DBManager(db_path=db_path)
        assert manager2.paper_exists("arxiv:2401.00001") is True
        manager2.close()
