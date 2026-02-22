"""End-to-end integration test for weekly intelligence pipeline.

Creates temporary test DB with fixture data, runs full pipeline,
validates all outputs, then cleans up all generated files.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestWeeklyIntelligenceE2E:
    """Full pipeline E2E test with fixture data and cleanup."""

    def setup_method(self):
        """Create temp directory and test database with fixture data."""
        self.tmp_dir = tempfile.mkdtemp(prefix="weekly_intel_test_")
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.report_dir = os.path.join(self.tmp_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self._create_test_db()
        self._insert_fixture_data()

        # Reference date: a Monday for clean week boundaries
        self.reference_date = datetime(2025, 1, 27, tzinfo=timezone.utc)
        self.test_date_str = self.reference_date.strftime("%Y%m%d")

    def teardown_method(self):
        """Remove ALL test artifacts."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _create_test_db(self):
        """Create all tables including weekly_snapshots."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
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
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            )
        """)
        conn.execute("""
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
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_stats (
                run_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                collected INTEGER NOT NULL DEFAULT 0,
                total_available INTEGER,
                truncated INTEGER NOT NULL DEFAULT 0,
                retries INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                exception TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS remind_tracking (
                paper_key TEXT NOT NULL,
                topic_slug TEXT NOT NULL,
                recommend_count INTEGER NOT NULL DEFAULT 0,
                last_recommend_run_id INTEGER NOT NULL,
                PRIMARY KEY (paper_key, topic_slug)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_snapshots (
                iso_year INTEGER NOT NULL,
                iso_week INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                section TEXT NOT NULL,
                topic_slug TEXT NOT NULL DEFAULT 'all',
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (iso_year, iso_week, section, topic_slug)
            )
        """)
        conn.commit()
        conn.close()

    def _insert_fixture_data(self):
        """Insert 4 weeks of realistic test data."""
        conn = sqlite3.connect(self.db_path)
        ref = datetime(2025, 1, 27, tzinfo=timezone.utc)

        # Create runs for 4 weeks, 2 topics each
        run_id = 1
        runs_by_week = {}
        for week_offset in range(4):
            week_start = ref - timedelta(weeks=3 - week_offset)
            week_end = week_start + timedelta(days=1)
            runs_this_week = []

            for topic in ["cv-research", "nlp-research"]:
                conn.execute(
                    """INSERT INTO runs (run_id, topic_slug, window_start_utc, window_end_utc,
                       display_date_kst, embedding_mode, scoring_weights, detected_rpm,
                       detected_daily_limit, response_format_supported, prompt_versions,
                       topic_override_fields, total_collected, total_filtered, total_scored,
                       total_discarded, total_output, threshold_used, threshold_lowered,
                       status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, topic, week_start.isoformat(), week_end.isoformat(),
                     week_start.strftime("%y-%m-%d"), "en_synthetic", "{}",
                     20, 2000, 1, "{}", "{}",
                     10, 2, 8, 0, 8, 60, 0, "completed"),
                )
                runs_this_week.append(run_id)
                run_id += 1

            runs_by_week[week_offset] = runs_this_week

        # Create papers + evaluations
        categories_options = [
            '["cs.CV", "cs.AI"]',
            '["cs.CV", "cs.LG"]',
            '["cs.AI", "cs.LG"]',
            '["cs.CL", "cs.AI"]',
        ]
        authors_options = [
            '["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince"]',
            '["Eve Wilson", "Frank Miller"]',
            '["Grace Lee", "Henry Park", "Ivy Chen"]',
        ]

        paper_idx = 0
        for week_offset in range(4):
            for run_id_val in runs_by_week[week_offset]:
                for i in range(5):
                    paper_key = f"arxiv:2501.{paper_idx:05d}"
                    score = 90 - paper_idx * 2 + i * 5
                    score = max(30, min(95, score))
                    tier = 1 if score >= 70 else 2

                    # Add keywords to some titles/abstracts
                    kw_title = ""
                    kw_abstract = ""
                    if i % 3 == 0:
                        kw_title = "transformer attention"
                        kw_abstract = "We propose a novel transformer architecture with improved attention mechanism."
                    elif i % 3 == 1:
                        kw_title = "visual search retrieval"
                        kw_abstract = "This paper presents a visual search system for image retrieval."
                    else:
                        kw_title = "pruning quantization"
                        kw_abstract = "We investigate pruning and quantization techniques for model compression."

                    # Conference comment for some papers
                    comment = ""
                    if i == 0 and week_offset == 3:
                        comment = "Accepted at CVPR 2025"

                    conn.execute(
                        """INSERT OR IGNORE INTO papers (paper_key, source, native_id, canonical_id,
                           url, title, abstract, authors, categories, published_at_utc,
                           has_code, has_code_source, comment, first_seen_run_id, created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (paper_key, "arxiv", f"2501.{paper_idx:05d}", None,
                         f"https://arxiv.org/abs/2501.{paper_idx:05d}",
                         f"Paper {paper_idx}: {kw_title}",
                         f"Abstract for paper {paper_idx}. {kw_abstract}",
                         authors_options[i % len(authors_options)],
                         categories_options[i % len(categories_options)],
                         (datetime(2025, 1, 6) + timedelta(days=paper_idx)).isoformat(),
                         0, "none", comment, run_id_val,
                         datetime.now(timezone.utc).isoformat()),
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO paper_evaluations
                           (run_id, paper_key, embed_score, llm_base_score, flags,
                            bonus_score, final_score, rank, tier, discarded,
                            is_remind, summary_ko, reason_ko, insight_ko,
                            prompt_ver_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id_val, paper_key, 0.85, score, "{}",
                         5, float(score), i + 1, tier, 0,
                         0, f"요약 {paper_idx}", f"이유 {paper_idx}", None,
                         "v1"),
                    )
                    paper_idx += 1

        # Multi-topic paper: same paper_key in different runs/topics
        multi_pk = "arxiv:2501.99999"
        conn.execute(
            """INSERT OR IGNORE INTO papers (paper_key, source, native_id, canonical_id,
               url, title, abstract, authors, categories, published_at_utc,
               has_code, has_code_source, comment, first_seen_run_id, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (multi_pk, "arxiv", "2501.99999", None,
             "https://arxiv.org/abs/2501.99999",
             "Multi-topic transformer attention paper",
             "A paper about transformer and attention that spans multiple topics.",
             '["Alice Smith", "Bob Jones"]',
             '["cs.CV", "cs.AI"]',
             datetime(2025, 1, 25).isoformat(),
             0, "none", "", runs_by_week[3][0],
             datetime.now(timezone.utc).isoformat()),
        )
        # Evaluation in topic 1 (score 85)
        conn.execute(
            """INSERT OR REPLACE INTO paper_evaluations
               (run_id, paper_key, embed_score, llm_base_score, flags,
                bonus_score, final_score, rank, tier, discarded,
                is_remind, prompt_ver_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (runs_by_week[3][0], multi_pk, 0.9, 85, "{}",
             5, 85.0, 1, 1, 0, 0, "v1"),
        )
        # Evaluation in topic 2 (score 75)
        conn.execute(
            """INSERT OR REPLACE INTO paper_evaluations
               (run_id, paper_key, embed_score, llm_base_score, flags,
                bonus_score, final_score, rank, tier, discarded,
                is_remind, prompt_ver_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (runs_by_week[3][1], multi_pk, 0.8, 75, "{}",
             5, 75.0, 2, 1, 0, 0, "v1"),
        )

        # Remind tracking: 2 graduated + 3 active
        for i in range(5):
            pk = f"arxiv:2501.{i:05d}"
            count = 2 if i < 2 else 1
            conn.execute(
                """INSERT OR REPLACE INTO remind_tracking
                   (paper_key, topic_slug, recommend_count, last_recommend_run_id)
                VALUES (?,?,?,?)""",
                (pk, "cv-research", count, runs_by_week[3][0]),
            )

        conn.commit()
        conn.close()

    def _build_test_config(
        self, llm_enabled: bool = False, intelligence_enabled: bool = True
    ):
        """Build a mock AppConfig for testing."""
        config = MagicMock()
        config.weekly = {
            "enabled": True,
            "intelligence": {
                "enabled": intelligence_enabled,
                "must_track_keywords": {
                    "groups": {
                        "model_architecture": ["transformer", "attention"],
                        "technique_optimization": ["pruning", "quantization"],
                        "domain_application": ["visual search", "image retrieval"],
                    },
                },
                "product_lines": {
                    "search_engine": ["visual search", "image retrieval"],
                    "model_compression": ["pruning", "quantization"],
                },
                "conferences": ["CVPR", "ICCV", "NeurIPS"],
                "top_papers_count": 10,
                "tfidf_top_n": 20,
                "trend_weeks": 4,
                "llm": {
                    "model": "",
                    "enabled": llm_enabled,
                    "max_input_tokens_per_call": 8000,
                    "delay_between_calls_sec": 0,
                },
            },
        }
        config.output = {"report_dir": self.report_dir, "template_dir": "templates"}
        config.database = {"path": self.db_path, "provider": "sqlite"}
        config.llm = {"model": "test-model", "base_url": "https://test.api", "provider": "openrouter"}
        return config

    # ---------------------------------------------------------------
    # Test Cases
    # ---------------------------------------------------------------

    def test_full_pipeline_no_llm(self):
        """LLM disabled: full pipeline produces valid data, HTML, and MD."""
        config = self._build_test_config(llm_enabled=False)

        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        summary_data, md_content, html_content = generate_weekly_intelligence(
            db_path=self.db_path,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )

        # Return types
        assert isinstance(summary_data, dict)
        assert isinstance(md_content, str) and len(md_content) > 0
        assert isinstance(html_content, str) and len(html_content) > 0

        # Section A
        exec_data = summary_data["sections"]["executive"]
        assert exec_data["metrics"]["total_evaluated"] > 0
        assert exec_data["metrics"]["cs_cv_count"] >= 0
        assert exec_data["llm_briefing"] is None

        # Section B
        radar = summary_data["sections"].get("tech_radar", {})
        if radar:
            assert "keyword_groups" in radar
            assert "tfidf" in radar

        # Section C
        top = summary_data["sections"]["top_papers"]
        assert len(top["papers"]) <= 10
        # Multi-topic dedup check
        paper_keys = [p["paper_key"] for p in top["papers"]]
        assert len(paper_keys) == len(set(paper_keys))

        # product_intel section removed (consolidated into intelligence)

        # Research Network
        net = summary_data["sections"].get("research_net", {})
        if net and net.get("notable_authors"):
            assert all(a["count"] >= 2 for a in net["notable_authors"])

        # HTML validity
        assert "<html" in html_content.lower()

        # Snapshot saved
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM weekly_snapshots").fetchall()
        assert len(rows) >= 2  # at least executive + top_papers
        conn.close()

    def test_full_pipeline_with_llm_mock(self):
        """LLM enabled (mocked): validates LLM integration."""
        config = self._build_test_config(llm_enabled=True)
        mock_response = "테스트 LLM 응답입니다."

        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        with patch("core.pipeline.weekly_llm_analyst.OpenRouterClient") as MockClient:
            MockClient.return_value.call.return_value = mock_response

            summary_data, md_content, html_content = generate_weekly_intelligence(
                db_path=self.db_path,
                date_str=self.test_date_str,
                config=config,
                provider="sqlite",
            )

        # LLM briefing should be set
        exec_data = summary_data["sections"]["executive"]
        assert exec_data.get("llm_briefing") == mock_response

    def test_empty_database(self):
        """Empty DB (tables only, no data) should not error."""
        empty_db = os.path.join(self.tmp_dir, "empty.db")
        conn = sqlite3.connect(empty_db)
        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (paper_key TEXT PRIMARY KEY, source TEXT NOT NULL, native_id TEXT NOT NULL, canonical_id TEXT, url TEXT NOT NULL, title TEXT NOT NULL, abstract TEXT NOT NULL, authors TEXT NOT NULL, categories TEXT NOT NULL, published_at_utc TEXT NOT NULL, updated_at_utc TEXT, pdf_url TEXT, has_code INTEGER NOT NULL DEFAULT 0, has_code_source TEXT NOT NULL DEFAULT 'none', code_url TEXT, comment TEXT, first_seen_run_id INTEGER NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, topic_slug TEXT NOT NULL, window_start_utc TEXT NOT NULL, window_end_utc TEXT NOT NULL, display_date_kst TEXT NOT NULL, embedding_mode TEXT NOT NULL, scoring_weights TEXT NOT NULL, detected_rpm INTEGER, detected_daily_limit INTEGER, response_format_supported INTEGER NOT NULL, prompt_versions TEXT NOT NULL, topic_override_fields TEXT NOT NULL, total_collected INTEGER NOT NULL DEFAULT 0, total_filtered INTEGER NOT NULL DEFAULT 0, total_scored INTEGER NOT NULL DEFAULT 0, total_discarded INTEGER NOT NULL DEFAULT 0, total_output INTEGER NOT NULL DEFAULT 0, threshold_used INTEGER NOT NULL DEFAULT 60, threshold_lowered INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'running', errors TEXT);
            CREATE TABLE IF NOT EXISTS paper_evaluations (run_id INTEGER NOT NULL, paper_key TEXT NOT NULL, embed_score REAL, llm_base_score INTEGER NOT NULL, flags TEXT NOT NULL, bonus_score INTEGER, final_score REAL, rank INTEGER, tier INTEGER, discarded INTEGER NOT NULL DEFAULT 0, score_lowered INTEGER, multi_topic TEXT, is_remind INTEGER NOT NULL DEFAULT 0, summary_ko TEXT, reason_ko TEXT, insight_ko TEXT, brief_reason TEXT, prompt_ver_score TEXT NOT NULL, prompt_ver_summ TEXT, PRIMARY KEY (run_id, paper_key));
            CREATE TABLE IF NOT EXISTS remind_tracking (paper_key TEXT NOT NULL, topic_slug TEXT NOT NULL, recommend_count INTEGER NOT NULL DEFAULT 0, last_recommend_run_id INTEGER NOT NULL, PRIMARY KEY (paper_key, topic_slug));
            CREATE TABLE IF NOT EXISTS weekly_snapshots (iso_year INTEGER NOT NULL, iso_week INTEGER NOT NULL, snapshot_date TEXT NOT NULL, section TEXT NOT NULL, topic_slug TEXT NOT NULL DEFAULT 'all', data_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY (iso_year, iso_week, section, topic_slug));
        """)
        conn.commit()
        conn.close()

        config = self._build_test_config(llm_enabled=False)
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        summary_data, md_content, html_content = generate_weekly_intelligence(
            db_path=empty_db,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )
        assert summary_data["sections"]["executive"]["metrics"]["total_evaluated"] == 0
        assert len(summary_data["sections"]["top_papers"]["papers"]) == 0

    def test_first_run_no_prev_snapshot(self):
        """First run: WoW values should be N/A, TF-IDF all NEW."""
        config = self._build_test_config(llm_enabled=False)
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        summary_data, _, _ = generate_weekly_intelligence(
            db_path=self.db_path,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )
        exec_data = summary_data["sections"]["executive"]
        assert exec_data["metrics"]["cs_cv_wow"] == "N/A"

        tfidf = summary_data["sections"].get("tech_radar", {}).get("tfidf", {}).get("keywords", [])
        if tfidf:
            assert all(kw["classification"] == "NEW" for kw in tfidf)

    def test_multi_topic_deduplication(self):
        """Same paper_key in multiple topics should be deduped (highest score)."""
        config = self._build_test_config(llm_enabled=False)
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        summary_data, _, _ = generate_weekly_intelligence(
            db_path=self.db_path,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )
        top_papers = summary_data["sections"]["top_papers"]["papers"]
        paper_keys = [p["paper_key"] for p in top_papers]
        assert len(paper_keys) == len(set(paper_keys))

    def test_intelligence_always_enabled(self):
        """Intelligence is always enabled regardless of config flag."""
        config = self._build_test_config(intelligence_enabled=False)
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        summary_data, md_content, html_content = generate_weekly_intelligence(
            db_path=self.db_path,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )
        # Should still produce valid output even when enabled=False in config
        assert "sections" in summary_data
        assert len(html_content) > 0

    def test_output_files_created(self):
        """HTML and MD output content should be non-empty."""
        config = self._build_test_config(llm_enabled=False)
        from core.pipeline.weekly_intelligence import generate_weekly_intelligence

        _, md_content, html_content = generate_weekly_intelligence(
            db_path=self.db_path,
            date_str=self.test_date_str,
            config=config,
            provider="sqlite",
        )

        md_path = os.path.join(self.report_dir, f"{self.test_date_str}_weekly_paper_report.md")
        html_path = os.path.join(self.report_dir, f"{self.test_date_str}_weekly_paper_report.html")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        assert os.path.exists(md_path)
        assert os.path.exists(html_path)
        assert os.path.getsize(md_path) > 100
        assert os.path.getsize(html_path) > 100
