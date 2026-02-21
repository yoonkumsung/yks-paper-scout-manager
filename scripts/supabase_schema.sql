-- Supabase PostgreSQL schema for Paper Scout.
-- Run this once in the Supabase SQL Editor to create all required tables.

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

CREATE TABLE IF NOT EXISTS remind_tracking (
    paper_key TEXT NOT NULL,
    topic_slug TEXT NOT NULL,
    recommend_count INTEGER NOT NULL DEFAULT 0,
    last_recommend_run_id INTEGER NOT NULL,
    PRIMARY KEY (paper_key, topic_slug)
);

CREATE TABLE IF NOT EXISTS weekly_snapshots (
    iso_year    INTEGER NOT NULL,
    iso_week    INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    section     TEXT NOT NULL,
    data_json   JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (iso_year, iso_week, section)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_runs_topic_slug ON runs (topic_slug);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (topic_slug, status);
CREATE INDEX IF NOT EXISTS idx_evaluations_paper ON paper_evaluations (paper_key);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON paper_evaluations (run_id);
CREATE INDEX IF NOT EXISTS idx_query_stats_run ON query_stats (run_id);
CREATE INDEX IF NOT EXISTS idx_remind_topic ON remind_tracking (topic_slug);
CREATE INDEX IF NOT EXISTS idx_papers_created ON papers (created_at);
CREATE INDEX IF NOT EXISTS idx_runs_window ON runs (window_start_utc);
