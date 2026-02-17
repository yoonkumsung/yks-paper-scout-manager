"""Data models for Paper Scout.

Defines all core dataclasses used across the pipeline:
Paper, Evaluation, EvaluationFlags, RunMeta, QueryStats,
RemindTracking, TopicSpec, and NotifyConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Nested / helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvaluationFlags:
    """Boolean flags assigned by Agent 2 during scoring."""

    is_edge: bool = False
    is_realtime: bool = False
    mentions_code: bool = False
    is_metaphorical: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Serialize flags to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationFlags:
        """Construct an EvaluationFlags instance from a dictionary.

        Unknown keys are silently ignored so that forward-compatible
        payloads do not break deserialization.
        """
        return cls(
            is_edge=bool(d.get("is_edge", False)),
            is_realtime=bool(d.get("is_realtime", False)),
            mentions_code=bool(d.get("mentions_code", False)),
            is_metaphorical=bool(d.get("is_metaphorical", False)),
        )


@dataclass
class NotifyConfig:
    """Notification channel configuration for a topic."""

    provider: str  # "discord" or "telegram"
    channel_id: str
    secret_key: str


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """Normalized paper record (Section 4-1).

    PK: paper_key  ("{source}:{native_id}")
    DB table: papers (365-day TTL)
    """

    source: str  # e.g. "arxiv"
    native_id: str  # source-specific ID
    paper_key: str  # PK, format "{source}:{native_id}"
    url: str
    title: str
    abstract: str
    authors: list[str]  # JSON-serialized TEXT in DB
    categories: list[str]  # JSON-serialized TEXT in DB
    published_at_utc: datetime

    canonical_id: Optional[str] = None  # DOI for cross-source dedup
    updated_at_utc: Optional[datetime] = None
    pdf_url: Optional[str] = None
    has_code: bool = False
    has_code_source: str = "none"  # "regex" | "llm" | "both" | "none"
    code_url: Optional[str] = None
    comment: Optional[str] = None
    first_seen_run_id: Optional[int] = None  # set when first inserted to DB
    created_at: Optional[str] = None  # ISO 8601 timestamp

    @classmethod
    def from_arxiv_result(cls, result: Any) -> Paper:
        """Create a Paper from an ``arxiv.Result`` object.

        Maps fields from the ``arxiv`` PyPI library's ``Result`` class to
        the normalized ``Paper`` dataclass.

        Args:
            result: An ``arxiv.Result`` instance.

        Returns:
            A normalized ``Paper`` with ``source="arxiv"``.
        """
        import re

        # Extract the arxiv ID from the entry_id URL.
        # entry_id looks like "http://arxiv.org/abs/2401.12345v1"
        # or "http://arxiv.org/abs/quant-ph/0201082v1".
        # get_short_id() returns e.g. "2401.12345v1"; strip the version.
        short_id = result.get_short_id()
        native_id = re.sub(r"v\d+$", "", short_id)

        # Ensure published datetime is UTC-aware.
        published = result.published
        if published.tzinfo is None:
            from datetime import timezone

            published = published.replace(tzinfo=timezone.utc)

        updated = result.updated
        if updated is not None and updated.tzinfo is None:
            from datetime import timezone

            updated = updated.replace(tzinfo=timezone.utc)

        return cls(
            source="arxiv",
            native_id=native_id,
            paper_key=f"arxiv:{native_id}",
            url=result.entry_id,
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.replace("\n", " ").strip(),
            authors=[a.name for a in result.authors],
            categories=list(result.categories),
            published_at_utc=published,
            updated_at_utc=updated,
            pdf_url=result.pdf_url,
            comment=result.comment if result.comment else None,
        )


@dataclass
class Evaluation:
    """Per-run evaluation of a single paper (Section 4-2).

    Composite PK: (run_id, paper_key)
    DB table: paper_evaluations (90-day purge)
    """

    run_id: int
    paper_key: str

    # Scoring
    llm_base_score: int  # 0-100
    flags: EvaluationFlags
    prompt_ver_score: str  # Agent 2 prompt version

    embed_score: Optional[float] = None  # 0-1, None when embeddings disabled
    bonus_score: Optional[int] = None  # calculated by Ranker
    final_score: Optional[float] = None
    rank: Optional[int] = None
    tier: Optional[int] = None  # 1 or 2
    discarded: bool = False
    score_lowered: Optional[bool] = None

    # Multi-topic
    multi_topic: Optional[str] = None

    # Remind (re-recommend)
    is_remind: bool = False

    # Summaries (Agent 3)
    summary_ko: Optional[str] = None
    reason_ko: Optional[str] = None
    insight_ko: Optional[str] = None
    brief_reason: Optional[str] = None  # Agent 2 one-sentence reason
    prompt_ver_summ: Optional[str] = None  # Agent 3 prompt version

    @property
    def should_null_on_discard(self) -> bool:
        """Return True when discard-sensitive fields should be NULL.

        Per Section 4-2: when ``discarded=True`` the following fields
        are expected to be NULL because Ranker / Agent 3 are not invoked:
        bonus_score, final_score, rank, tier, score_lowered,
        summary_ko, reason_ko, insight_ko, prompt_ver_summ.
        """
        return self.discarded


@dataclass
class RunMeta:
    """Execution metadata for a single pipeline run (Section 4-3).

    PK: run_id (auto-increment in DB)
    DB table: runs (90-day purge)
    """

    topic_slug: str
    window_start_utc: datetime
    window_end_utc: datetime
    display_date_kst: str
    embedding_mode: str  # "disabled" | "en_synthetic"
    scoring_weights: dict  # JSON in DB
    response_format_supported: bool
    prompt_versions: dict  # JSON in DB, e.g. {"agent1": "agent1-v3", ...}
    status: str  # "running" | "completed" | "failed"

    run_id: Optional[int] = None  # auto-increment in DB
    detected_rpm: Optional[int] = None
    detected_daily_limit: Optional[int] = None
    topic_override_fields: list = field(default_factory=list)  # JSON in DB

    # Counters
    total_collected: int = 0
    total_filtered: int = 0
    total_scored: int = 0
    total_discarded: int = 0
    total_output: int = 0

    # Threshold
    threshold_used: int = 60
    threshold_lowered: bool = False

    # Error tracking
    errors: Optional[str] = None


@dataclass
class QueryStats:
    """Per-query statistics (Section 4-4).

    DB table: query_stats (90-day purge)
    """

    run_id: int
    query_text: str

    collected: int = 0
    total_available: Optional[int] = None  # OpenSearch totalResults
    truncated: bool = False
    retries: int = 0
    duration_ms: int = 0
    exception: Optional[str] = None


@dataclass
class RemindTracking:
    """Cross-run re-recommendation tracking (Section 4-6).

    Composite PK: (paper_key, topic_slug)
    DB table: remind_tracking
    """

    paper_key: str
    topic_slug: str
    last_recommend_run_id: int

    recommend_count: int = 0  # values: 0, 1, or 2


# ---------------------------------------------------------------------------
# Configuration models (Section 5)
# ---------------------------------------------------------------------------


@dataclass
class TopicSpec:
    """Topic definition from config.yaml (Section 5).

    Required fields: slug, name, description, arxiv_categories, notify.
    Optional fields allow fine-grained control over Agent 1 keyword
    expansion.
    """

    slug: str
    name: str
    description: str
    arxiv_categories: list[str]
    notify: NotifyConfig | None = None

    must_concepts_en: Optional[list[str]] = None
    should_concepts_en: Optional[list[str]] = None
    must_not_en: Optional[list[str]] = None
