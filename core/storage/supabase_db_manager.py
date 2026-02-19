"""PostgreSQL (Supabase) database manager for Paper Scout.

Drop-in replacement for DBManager that connects to Supabase PostgreSQL
instead of local SQLite.  All public methods match DBManager's interface.
Uses psycopg2 with RealDictCursor for dict-style row access.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from core.models import (
    Evaluation,
    EvaluationFlags,
    Paper,
    QueryStats,
    RemindTracking,
    RunMeta,
)

logger = logging.getLogger(__name__)


def _dt_to_iso(dt: datetime | None) -> str | None:
    """Convert a datetime to ISO 8601 string or return None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_dt(s: str | None) -> datetime | None:
    """Parse an ISO 8601 string back to a datetime or return None."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SupabaseDBManager:
    """PostgreSQL database manager for Paper Scout (Supabase backend)."""

    def __init__(self, connection_string: str) -> None:
        """Initialize PostgreSQL connection.

        Args:
            connection_string: PostgreSQL connection string (from SUPABASE_DB_URL).
        """
        self._connection_string = connection_string
        self._conn = psycopg2.connect(
            connection_string,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        self._conn.autocommit = False
        self._create_tables()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SupabaseDBManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Table creation
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create all tables if they do not exist."""
        cur = self._conn.cursor()
        cur.execute(_CREATE_PAPERS)
        cur.execute(_CREATE_RUNS)
        cur.execute(_CREATE_EVALUATIONS)
        cur.execute(_CREATE_QUERY_STATS)
        cur.execute(_CREATE_REMIND_TRACKING)
        self._conn.commit()

    # ==================================================================
    # Papers CRUD
    # ==================================================================

    def insert_paper(self, paper: Paper, *, commit: bool = True) -> None:
        """Insert a paper record.  Ignores duplicates on paper_key."""
        created_at = paper.created_at or datetime.now(timezone.utc).isoformat()
        self._conn.cursor().execute(
            """
            INSERT INTO papers (
                paper_key, source, native_id, canonical_id, url,
                title, abstract, authors, categories, published_at_utc,
                updated_at_utc, pdf_url, has_code, has_code_source,
                code_url, comment, first_seen_run_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_key) DO NOTHING
            """,
            (
                paper.paper_key,
                paper.source,
                paper.native_id,
                paper.canonical_id,
                paper.url,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                json.dumps(paper.categories),
                _dt_to_iso(paper.published_at_utc),
                _dt_to_iso(paper.updated_at_utc),
                paper.pdf_url,
                int(paper.has_code),
                paper.has_code_source,
                paper.code_url,
                paper.comment,
                paper.first_seen_run_id,
                created_at,
            ),
        )
        if commit:
            self._conn.commit()

    def get_paper(self, paper_key: str) -> Paper | None:
        """Retrieve a paper by its primary key."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM papers WHERE paper_key = %s", (paper_key,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def paper_exists(self, paper_key: str) -> bool:
        """Return True if a paper with the given key exists."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT 1 FROM papers WHERE paper_key = %s LIMIT 1", (paper_key,)
        )
        return cur.fetchone() is not None

    def update_paper_code_info(
        self,
        paper_key: str,
        has_code: bool,
        has_code_source: str,
        code_url: str | None,
    ) -> None:
        """Update code-related columns for an existing paper."""
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE papers
               SET has_code = %s, has_code_source = %s, code_url = %s
             WHERE paper_key = %s
            """,
            (int(has_code), has_code_source, code_url, paper_key),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_paper(row: dict) -> Paper:
        """Map a database row to a Paper dataclass."""
        return Paper(
            paper_key=row["paper_key"],
            source=row["source"],
            native_id=row["native_id"],
            canonical_id=row["canonical_id"],
            url=row["url"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]),
            categories=json.loads(row["categories"]),
            published_at_utc=_iso_to_dt(row["published_at_utc"]),  # type: ignore[arg-type]
            updated_at_utc=_iso_to_dt(row["updated_at_utc"]),
            pdf_url=row["pdf_url"],
            has_code=bool(row["has_code"]),
            has_code_source=row["has_code_source"],
            code_url=row["code_url"],
            comment=row["comment"],
            first_seen_run_id=row["first_seen_run_id"],
            created_at=row["created_at"],
        )

    # ==================================================================
    # Evaluations CRUD
    # ==================================================================

    def insert_evaluation(
        self, evaluation: Evaluation, commit: bool = True
    ) -> None:
        """Insert an evaluation record (upsert on conflict)."""
        self._conn.cursor().execute(
            """
            INSERT INTO paper_evaluations (
                run_id, paper_key, embed_score, llm_base_score, flags,
                bonus_score, final_score, rank, tier, discarded,
                score_lowered, multi_topic, is_remind,
                summary_ko, reason_ko, insight_ko, brief_reason,
                prompt_ver_score, prompt_ver_summ
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, paper_key) DO UPDATE SET
                embed_score = EXCLUDED.embed_score,
                llm_base_score = EXCLUDED.llm_base_score,
                flags = EXCLUDED.flags,
                bonus_score = EXCLUDED.bonus_score,
                final_score = EXCLUDED.final_score,
                rank = EXCLUDED.rank,
                tier = EXCLUDED.tier,
                discarded = EXCLUDED.discarded,
                score_lowered = EXCLUDED.score_lowered,
                multi_topic = EXCLUDED.multi_topic,
                is_remind = EXCLUDED.is_remind,
                summary_ko = EXCLUDED.summary_ko,
                reason_ko = EXCLUDED.reason_ko,
                insight_ko = EXCLUDED.insight_ko,
                brief_reason = EXCLUDED.brief_reason,
                prompt_ver_score = EXCLUDED.prompt_ver_score,
                prompt_ver_summ = EXCLUDED.prompt_ver_summ
            """,
            (
                evaluation.run_id,
                evaluation.paper_key,
                evaluation.embed_score,
                evaluation.llm_base_score,
                json.dumps(evaluation.flags.to_dict()),
                evaluation.bonus_score,
                evaluation.final_score,
                evaluation.rank,
                evaluation.tier,
                int(evaluation.discarded),
                int(evaluation.score_lowered) if evaluation.score_lowered is not None else None,
                evaluation.multi_topic,
                int(evaluation.is_remind),
                evaluation.summary_ko,
                evaluation.reason_ko,
                evaluation.insight_ko,
                evaluation.brief_reason,
                evaluation.prompt_ver_score,
                evaluation.prompt_ver_summ,
            ),
        )
        if commit:
            self._conn.commit()

    def get_evaluations_by_run(self, run_id: int) -> list[Evaluation]:
        """Retrieve all evaluations for a given run."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM paper_evaluations WHERE run_id = %s", (run_id,)
        )
        return [self._row_to_evaluation(r) for r in cur.fetchall()]

    def get_latest_evaluation(
        self, paper_key: str, topic_slug: str
    ) -> Evaluation | None:
        """Return the most recent evaluation for a paper within a topic."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT pe.*
              FROM paper_evaluations pe
              JOIN runs r ON pe.run_id = r.run_id
             WHERE pe.paper_key = %s
               AND r.topic_slug = %s
             ORDER BY pe.run_id DESC
             LIMIT 1
            """,
            (paper_key, topic_slug),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_evaluation(row)

    def get_high_score_papers(
        self, topic_slug: str, min_score: float
    ) -> list[Evaluation]:
        """Return evaluations with final_score >= min_score for a topic."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT pe.*
              FROM paper_evaluations pe
              JOIN runs r ON pe.run_id = r.run_id
             WHERE r.topic_slug = %s
               AND pe.final_score >= %s
               AND pe.discarded = 0
             ORDER BY pe.final_score DESC
            """,
            (topic_slug, min_score),
        )
        return [self._row_to_evaluation(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_evaluation(row: dict) -> Evaluation:
        """Map a database row to an Evaluation dataclass."""
        return Evaluation(
            run_id=row["run_id"],
            paper_key=row["paper_key"],
            embed_score=row["embed_score"],
            llm_base_score=row["llm_base_score"],
            flags=EvaluationFlags.from_dict(json.loads(row["flags"])),
            bonus_score=row["bonus_score"],
            final_score=row["final_score"],
            rank=row["rank"],
            tier=row["tier"],
            discarded=bool(row["discarded"]),
            score_lowered=bool(row["score_lowered"]) if row["score_lowered"] is not None else None,
            multi_topic=row["multi_topic"],
            is_remind=bool(row["is_remind"]),
            summary_ko=row["summary_ko"],
            reason_ko=row["reason_ko"],
            insight_ko=row["insight_ko"],
            brief_reason=row["brief_reason"],
            prompt_ver_score=row["prompt_ver_score"],
            prompt_ver_summ=row["prompt_ver_summ"],
        )

    # ==================================================================
    # Runs CRUD
    # ==================================================================

    def create_run(self, run_meta: RunMeta) -> int:
        """Insert a new run and return the auto-generated run_id."""
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                topic_slug, window_start_utc, window_end_utc,
                display_date_kst, embedding_mode, scoring_weights,
                detected_rpm, detected_daily_limit,
                response_format_supported, prompt_versions,
                topic_override_fields,
                total_collected, total_filtered, total_scored,
                total_discarded, total_output,
                threshold_used, threshold_lowered,
                status, errors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (
                run_meta.topic_slug,
                _dt_to_iso(run_meta.window_start_utc),
                _dt_to_iso(run_meta.window_end_utc),
                run_meta.display_date_kst,
                run_meta.embedding_mode,
                json.dumps(run_meta.scoring_weights),
                run_meta.detected_rpm,
                run_meta.detected_daily_limit,
                int(run_meta.response_format_supported),
                json.dumps(run_meta.prompt_versions),
                json.dumps(run_meta.topic_override_fields),
                run_meta.total_collected,
                run_meta.total_filtered,
                run_meta.total_scored,
                run_meta.total_discarded,
                run_meta.total_output,
                run_meta.threshold_used,
                int(run_meta.threshold_lowered),
                run_meta.status,
                run_meta.errors,
            ),
        )
        row = cur.fetchone()
        self._conn.commit()
        return row["run_id"]

    def update_run_status(
        self, run_id: int, status: str, errors: str | None = None
    ) -> None:
        """Update the status (and optionally errors) of a run."""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE runs SET status = %s, errors = %s WHERE run_id = %s",
            (status, errors, run_id),
        )
        self._conn.commit()

    def update_run_stats(self, run_id: int, **stats: Any) -> None:
        """Update numeric counters / stats on a run record."""
        allowed = {
            "total_collected",
            "total_filtered",
            "total_scored",
            "total_discarded",
            "total_output",
            "threshold_used",
            "threshold_lowered",
        }
        filtered = {k: v for k, v in stats.items() if k in allowed}
        if not filtered:
            return
        set_clause = ", ".join(f"{k} = %s" for k in filtered)
        values = list(filtered.values())
        values.append(run_id)
        cur = self._conn.cursor()
        cur.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = %s",  # noqa: S608
            values,
        )
        self._conn.commit()

    def get_latest_completed_run(self, topic_slug: str) -> RunMeta | None:
        """Return the most recently completed run for a topic."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT * FROM runs
             WHERE topic_slug = %s AND status = 'completed'
             ORDER BY run_id DESC
             LIMIT 1
            """,
            (topic_slug,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    @staticmethod
    def _row_to_run(row: dict) -> RunMeta:
        """Map a database row to a RunMeta dataclass."""
        return RunMeta(
            run_id=row["run_id"],
            topic_slug=row["topic_slug"],
            window_start_utc=_iso_to_dt(row["window_start_utc"]),  # type: ignore[arg-type]
            window_end_utc=_iso_to_dt(row["window_end_utc"]),  # type: ignore[arg-type]
            display_date_kst=row["display_date_kst"],
            embedding_mode=row["embedding_mode"],
            scoring_weights=json.loads(row["scoring_weights"]),
            detected_rpm=row["detected_rpm"],
            detected_daily_limit=row["detected_daily_limit"],
            response_format_supported=bool(row["response_format_supported"]),
            prompt_versions=json.loads(row["prompt_versions"]),
            topic_override_fields=json.loads(row["topic_override_fields"]),
            total_collected=row["total_collected"],
            total_filtered=row["total_filtered"],
            total_scored=row["total_scored"],
            total_discarded=row["total_discarded"],
            total_output=row["total_output"],
            threshold_used=row["threshold_used"],
            threshold_lowered=bool(row["threshold_lowered"]),
            status=row["status"],
            errors=row["errors"],
        )

    # ==================================================================
    # QueryStats CRUD
    # ==================================================================

    def insert_query_stats(self, stats: QueryStats) -> None:
        """Insert a query_stats record."""
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO query_stats (
                run_id, query_text, collected, total_available,
                truncated, retries, duration_ms, exception
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                stats.run_id,
                stats.query_text,
                stats.collected,
                stats.total_available,
                int(stats.truncated),
                stats.retries,
                stats.duration_ms,
                stats.exception,
            ),
        )
        self._conn.commit()

    # ==================================================================
    # RemindTracking CRUD
    # ==================================================================

    def get_remind_tracking(
        self, paper_key: str, topic_slug: str
    ) -> RemindTracking | None:
        """Retrieve a remind_tracking entry by composite key."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT * FROM remind_tracking
             WHERE paper_key = %s AND topic_slug = %s
            """,
            (paper_key, topic_slug),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return RemindTracking(
            paper_key=row["paper_key"],
            topic_slug=row["topic_slug"],
            recommend_count=row["recommend_count"],
            last_recommend_run_id=row["last_recommend_run_id"],
        )

    def get_remind_trackings_by_topic(self, topic_slug: str) -> dict[str, RemindTracking]:
        """Get all remind_tracking entries for a topic as a dict keyed by paper_key."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM remind_tracking WHERE topic_slug = %s",
            (topic_slug,),
        )
        result: dict[str, RemindTracking] = {}
        for row in cur.fetchall():
            tracking = RemindTracking(
                paper_key=row["paper_key"],
                topic_slug=row["topic_slug"],
                recommend_count=row["recommend_count"],
                last_recommend_run_id=row["last_recommend_run_id"],
            )
            result[tracking.paper_key] = tracking
        return result

    def upsert_remind_tracking(self, tracking: RemindTracking) -> None:
        """Insert or update a remind_tracking entry."""
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO remind_tracking (
                paper_key, topic_slug, recommend_count, last_recommend_run_id
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (paper_key, topic_slug)
            DO UPDATE SET
                recommend_count = EXCLUDED.recommend_count,
                last_recommend_run_id = EXCLUDED.last_recommend_run_id
            """,
            (
                tracking.paper_key,
                tracking.topic_slug,
                tracking.recommend_count,
                tracking.last_recommend_run_id,
            ),
        )
        self._conn.commit()

    def get_remind_candidates(
        self, topic_slug: str, min_score: float, max_count: int
    ) -> list[dict]:
        """Return candidate papers for re-recommendation."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT rt.paper_key,
                   rt.topic_slug,
                   rt.recommend_count,
                   rt.last_recommend_run_id,
                   pe.final_score
              FROM remind_tracking rt
              JOIN paper_evaluations pe
                ON rt.paper_key = pe.paper_key
               AND rt.last_recommend_run_id = pe.run_id
             WHERE rt.topic_slug = %s
               AND pe.final_score >= %s
               AND rt.recommend_count < %s
               AND pe.discarded = 0
             ORDER BY pe.final_score DESC
            """,
            (topic_slug, min_score, max_count),
        )
        return [dict(r) for r in cur.fetchall()]

    def update_evaluation_multi_topic(
        self, run_id: int, paper_key: str, multi_topic: str
    ) -> None:
        """Update the multi_topic field for an evaluation."""
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE paper_evaluations
               SET multi_topic = %s
             WHERE run_id = %s AND paper_key = %s
            """,
            (multi_topic, run_id, paper_key),
        )

    def commit(self) -> None:
        """Commit pending database changes."""
        self._conn.commit()

    # ==================================================================
    # Purge operations
    # ==================================================================

    def purge_old_evaluations(self, days: int = 90) -> int:
        """Delete evaluations whose run is older than *days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            DELETE FROM paper_evaluations
             WHERE run_id IN (
                SELECT run_id FROM runs
                 WHERE window_start_utc < %s
             )
            """,
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def purge_old_query_stats(self, days: int = 90) -> int:
        """Delete query_stats whose run is older than *days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            DELETE FROM query_stats
             WHERE run_id IN (
                SELECT run_id FROM runs
                 WHERE window_start_utc < %s
             )
            """,
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def purge_old_runs(self, days: int = 90) -> int:
        """Delete runs older than *days* days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM runs WHERE window_start_utc < %s", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    def purge_old_papers(self, days: int = 365) -> int:
        """Delete papers older than *days* days.

        Also removes orphaned paper_evaluations and remind_tracking
        entries referencing the deleted papers.
        When days is 0 or negative, skips purge (permanent retention).
        """
        if days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            DELETE FROM paper_evaluations
             WHERE paper_key IN (
                SELECT paper_key FROM papers WHERE created_at < %s
             )
            """,
            (cutoff,),
        )
        cur.execute(
            """
            DELETE FROM remind_tracking
             WHERE paper_key IN (
                SELECT paper_key FROM papers WHERE created_at < %s
             )
            """,
            (cutoff,),
        )
        cur.execute(
            "DELETE FROM papers WHERE created_at < %s", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    def purge_orphan_remind_tracking(self) -> int:
        """Delete remind_tracking entries with no matching evaluations."""
        cur = self._conn.cursor()
        cur.execute(
            """
            DELETE FROM remind_tracking
             WHERE paper_key NOT IN (
                SELECT DISTINCT paper_key FROM paper_evaluations
             )
            """
        )
        self._conn.commit()
        return cur.rowcount

    def vacuum(self) -> None:
        """No-op for Supabase (auto-managed)."""
        logger.debug("VACUUM skipped: Supabase manages vacuuming automatically")

    # ==================================================================
    # Utility
    # ==================================================================

    def get_db_stats(self) -> dict:
        """Return record counts per table."""
        tables = [
            "papers",
            "paper_evaluations",
            "runs",
            "query_stats",
            "remind_tracking",
        ]
        counts: dict[str, Any] = {}
        cur = self._conn.cursor()
        for table in tables:
            cur.execute(
                f"SELECT COUNT(*) AS cnt FROM {table}"  # noqa: S608
            )
            row = cur.fetchone()
            counts[table] = row["cnt"]
        counts["file_size_bytes"] = 0  # Not applicable for cloud DB
        return counts


# ---------------------------------------------------------------------------
# PostgreSQL DDL statements
# ---------------------------------------------------------------------------

_CREATE_PAPERS = """
CREATE TABLE IF NOT EXISTS papers (
    paper_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    native_id TEXT NOT NULL,
    canonical_id TEXT,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    published_at_utc TEXT NOT NULL,
    updated_at_utc TEXT,
    pdf_url TEXT,
    has_code INTEGER NOT NULL DEFAULT 0,
    has_code_source TEXT NOT NULL DEFAULT 'none',
    code_url TEXT,
    comment TEXT,
    first_seen_run_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id SERIAL PRIMARY KEY,
    topic_slug TEXT NOT NULL,
    window_start_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    display_date_kst TEXT NOT NULL,
    embedding_mode TEXT NOT NULL,
    scoring_weights TEXT NOT NULL,
    detected_rpm INTEGER,
    detected_daily_limit INTEGER,
    response_format_supported INTEGER NOT NULL,
    prompt_versions TEXT NOT NULL,
    topic_override_fields TEXT NOT NULL,
    total_collected INTEGER NOT NULL DEFAULT 0,
    total_filtered INTEGER NOT NULL DEFAULT 0,
    total_scored INTEGER NOT NULL DEFAULT 0,
    total_discarded INTEGER NOT NULL DEFAULT 0,
    total_output INTEGER NOT NULL DEFAULT 0,
    threshold_used INTEGER NOT NULL DEFAULT 60,
    threshold_lowered INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    errors TEXT
);
"""

_CREATE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS paper_evaluations (
    run_id INTEGER NOT NULL,
    paper_key TEXT NOT NULL,
    embed_score REAL,
    llm_base_score INTEGER NOT NULL,
    flags TEXT NOT NULL,
    bonus_score INTEGER,
    final_score REAL,
    rank INTEGER,
    tier INTEGER,
    discarded INTEGER NOT NULL DEFAULT 0,
    score_lowered INTEGER,
    multi_topic TEXT,
    is_remind INTEGER NOT NULL DEFAULT 0,
    summary_ko TEXT,
    reason_ko TEXT,
    insight_ko TEXT,
    brief_reason TEXT,
    prompt_ver_score TEXT NOT NULL,
    prompt_ver_summ TEXT,
    PRIMARY KEY (run_id, paper_key)
);
"""

_CREATE_QUERY_STATS = """
CREATE TABLE IF NOT EXISTS query_stats (
    run_id INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    collected INTEGER NOT NULL DEFAULT 0,
    total_available INTEGER,
    truncated INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    exception TEXT
);
"""

_CREATE_REMIND_TRACKING = """
CREATE TABLE IF NOT EXISTS remind_tracking (
    paper_key TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    recommend_count INTEGER NOT NULL DEFAULT 0,
    last_recommend_run_id INTEGER NOT NULL,
    PRIMARY KEY (paper_key, topic_slug)
);
"""
