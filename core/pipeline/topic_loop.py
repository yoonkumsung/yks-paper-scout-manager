"""Topic Loop Orchestrator for Paper Scout (TASK-036).

Executes the 12-step per-topic pipeline that ties all components together.
Each topic runs independently with error isolation: a failure in one topic
does not prevent subsequent topics from executing.

DevSpec Section 9 (Pipeline Architecture).
"""

from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.config import AppConfig
from core.models import (
    Evaluation,
    EvaluationFlags,
    Paper,
    QueryStats,
    RunMeta,
)

logger = logging.getLogger(__name__)

# KST offset for display date
_KST_OFFSET = timezone(timedelta(hours=9))


class TopicLoopOrchestrator:
    """Orchestrate the 12-step pipeline for all topics.

    Receives all dependencies via constructor injection.
    Iterates over topics from config and runs the pipeline per topic.
    Error isolation ensures one topic failure does not block others.
    """

    def __init__(
        self,
        config: AppConfig,
        db_manager: Any,  # DBManager
        rate_limiter: Any,  # RateLimiter
        search_window: Any,  # SearchWindowComputer
        usage_tracker: Any,  # UsageTracker
        run_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the orchestrator with injected dependencies.

        Args:
            config: Full application configuration.
            db_manager: Database manager for CRUD operations.
            rate_limiter: Rate limiter for LLM API calls.
            search_window: Search window computer for date ranges.
            usage_tracker: Usage tracker for daily statistics.
            run_options: Optional dict with overrides:
                - date_from: Manual start datetime (UTC).
                - date_to: Manual end datetime (UTC).
                - dedup_mode: "skip_recent" or "none".
                - topics: List of specific topic slugs to run.
        """
        self._config = config
        self._db = db_manager
        self._rate_limiter = rate_limiter
        self._search_window = search_window
        self._usage_tracker = usage_tracker
        self._run_options = run_options or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_topics(self) -> Dict[str, Any]:
        """Execute the pipeline for all configured topics.

        Returns:
            Summary dict with per-topic results:
            {
                "topics_completed": [{"slug": ..., "total_output": ...}],
                "topics_skipped": [{"slug": ..., "reason": ...}],
                "topics_failed": [{"slug": ..., "error": ...}],
            }
        """
        topics = self._resolve_topics()
        summary: Dict[str, Any] = {
            "topics_completed": [],
            "topics_skipped": [],
            "topics_failed": [],
        }

        for topic in topics:
            slug = topic.slug

            # Check daily limit
            if self._rate_limiter.is_daily_limit_reached:
                reason = "daily_limit_reached"
                logger.warning(
                    "Skipping topic '%s': %s", slug, reason
                )
                self._usage_tracker.record_topic_skipped(slug, reason)
                summary["topics_skipped"].append(
                    {"slug": slug, "reason": reason}
                )
                continue

            # Check per-topic skip
            if self._rate_limiter.should_skip_topic(slug):
                reason = "already_completed_today"
                logger.info(
                    "Skipping topic '%s': %s", slug, reason
                )
                self._usage_tracker.record_topic_skipped(slug, reason)
                summary["topics_skipped"].append(
                    {"slug": slug, "reason": reason}
                )
                continue

            try:
                result = self._run_single_topic(topic)
                self._rate_limiter.record_topic_completed(slug)
                self._usage_tracker.record_topic_completed(
                    slug, result.get("total_output", 0)
                )
                summary["topics_completed"].append(
                    {"slug": slug, "total_output": result.get("total_output", 0)}
                )
            except Exception:
                error_msg = traceback.format_exc()
                logger.error(
                    "Topic '%s' failed:\n%s", slug, error_msg
                )
                summary["topics_failed"].append(
                    {"slug": slug, "error": error_msg}
                )

        return summary

    # ------------------------------------------------------------------
    # Per-topic pipeline (12 steps)
    # ------------------------------------------------------------------

    def _run_single_topic(self, topic: Any) -> Dict[str, Any]:
        """Execute the 12-step pipeline for a single topic.

        Args:
            topic: TopicSpec instance.

        Returns:
            Result dict with pipeline stats.

        Raises:
            Exception: On unrecoverable error (caught by caller).
        """
        slug = topic.slug
        logger.info("=== Starting topic: %s ===", slug)

        # Compute search window
        window_start, window_end = self._search_window.compute(
            topic_slug=slug,
            date_from=self._run_options.get("date_from"),
            date_to=self._run_options.get("date_to"),
        )

        # Send start notifications (best-effort, never blocks pipeline)
        self._send_start_notifications(topic, window_start, window_end)

        # Determine embedding mode
        from core.embeddings.embedding_ranker import EmbeddingRanker

        embedding_ranker = EmbeddingRanker(
            cache_dir=self._config.embedding.get("cache_dir", "data")
        )
        embedding_mode = embedding_ranker.mode

        # Build scoring weights reference
        weights_key = (
            "embedding_on" if embedding_mode != "disabled"
            else "embedding_off"
        )
        scoring_weights = self._config.scoring.get("weights", {}).get(
            weights_key, {}
        )

        # Display date in KST
        kst_now = datetime.now(_KST_OFFSET)
        display_date_kst = kst_now.strftime("%Y-%m-%d")
        date_compact = kst_now.strftime("%Y%m%d")

        # Prompt versions
        prompt_versions = {
            "agent1": self._config.agents.get("keyword_expander", {}).get(
                "prompt_version", "agent1-v3"
            ),
            "agent2": self._config.agents.get("scorer", {}).get(
                "prompt_version", "agent2-v3"
            ),
            "agent3": self._config.agents.get("summarizer", {}).get(
                "prompt_version", "agent3-v3"
            ),
        }

        # Response format support
        response_format_supported = self._config.llm.get(
            "response_format_supported", False
        )

        # ---- Step 1: Create RunMeta ----
        run_meta = RunMeta(
            topic_slug=slug,
            window_start_utc=window_start,
            window_end_utc=window_end,
            display_date_kst=display_date_kst,
            embedding_mode=embedding_mode,
            scoring_weights=scoring_weights,
            response_format_supported=response_format_supported,
            prompt_versions=prompt_versions,
            status="running",
        )
        try:
            run_id = self._db.create_run(run_meta)
        except Exception as exc:
            logger.warning("DB create_run failed: %s. Continuing without DB tracking.", exc)
            run_id = -1  # sentinel: DB tracking disabled for this run
        logger.info("Step 1: RunMeta created, run_id=%d", run_id)

        try:
            # ---- Step 2: Agent 1 - Keyword Expansion ----
            agent1_output = self._step_agent1(topic)
            logger.info(
                "Step 2: Agent 1 completed for '%s'", slug
            )

            # ---- Step 3: arXiv Collection ----
            papers, query_stats_list = self._step_collect(
                topic, agent1_output, window_start, window_end, run_id
            )
            total_collected = len(papers)
            logger.info(
                "Step 3: Collected %d papers for '%s'",
                total_collected, slug,
            )

            # Save query stats
            if run_id >= 0:
                for qs in query_stats_list:
                    qs.run_id = run_id
                    try:
                        self._db.insert_query_stats(qs)
                    except Exception as exc:
                        logger.warning("DB insert_query_stats failed: %s", exc)

            # Early exit if no papers collected
            if total_collected == 0:
                logger.warning(
                    "Step 3: No papers collected for '%s', skipping remaining steps",
                    slug,
                )
                if run_id >= 0:
                    try:
                        self._db.update_run_status(run_id, "completed")
                        self._db.update_run_stats(
                            run_id,
                            total_collected=0, total_filtered=0,
                            total_scored=0, total_discarded=0, total_output=0,
                            threshold_used=0, threshold_lowered=False,
                        )
                    except Exception as exc:
                        logger.warning("DB update failed (no-papers path): %s", exc)
                return {
                    "run_id": run_id,
                    "total_collected": 0, "total_filtered": 0,
                    "total_scored": 0, "total_discarded": 0,
                    "total_output": 0, "report_paths": {},
                }

            # ---- Step 4: Hybrid Filter ----
            topic_embedding_text = agent1_output.get("topic_embedding_text")
            filtered_papers, filter_stats = self._step_filter(
                papers, agent1_output, topic,
                embedding_ranker, topic_embedding_text,
            )
            total_filtered = len(filtered_papers)
            logger.info(
                "Step 4: Filtered to %d papers for '%s'",
                total_filtered, slug,
            )

            # Early exit if no papers passed filter
            if total_filtered == 0:
                logger.warning(
                    "Step 4: No papers passed filter for '%s', skipping remaining steps",
                    slug,
                )
                if run_id >= 0:
                    try:
                        self._db.update_run_status(run_id, "completed")
                        self._db.update_run_stats(
                            run_id,
                            total_collected=total_collected, total_filtered=0,
                            total_scored=0, total_discarded=0, total_output=0,
                            threshold_used=0, threshold_lowered=False,
                        )
                    except Exception as exc:
                        logger.warning("DB update failed (no-filter path): %s", exc)
                return {
                    "run_id": run_id,
                    "total_collected": total_collected, "total_filtered": 0,
                    "total_scored": 0, "total_discarded": 0,
                    "total_output": 0, "report_paths": {},
                }

            # ---- Step 5: Agent 2 - Scoring ----
            evaluations = self._step_score(
                filtered_papers, topic.description
            )
            total_scored = len(evaluations)

            # Count discarded
            total_discarded = sum(
                1 for e in evaluations if e.get("discard", False)
            )
            logger.info(
                "Step 5: Scored %d papers (%d discarded) for '%s'",
                total_scored, total_discarded, slug,
            )

            # Insert papers into DB (batch commit for efficiency)
            papers_map: Dict[str, Paper] = {}
            for paper in filtered_papers:
                if not self._db.paper_exists(paper.paper_key):
                    paper.first_seen_run_id = run_id
                    self._db.insert_paper(paper, commit=False)
                papers_map[paper.paper_key] = paper
            self._db.commit()

            # Merge code detection with LLM output
            from core.scoring.code_detector import CodeDetector

            code_detector = CodeDetector()
            for ev in evaluations:
                pk = ev["paper_key"]
                paper = papers_map.get(pk)
                if paper is None:
                    continue

                # Regex-based code detection
                regex_result = code_detector.detect(
                    paper.abstract, paper.comment
                )
                # Merge with LLM mentions_code flag
                merged = code_detector.merge_with_llm(
                    regex_result, ev.get("flags", {}).get("mentions_code", False)
                )

                # Update paper code info in DB
                if merged.get("has_code", False):
                    self._db.update_paper_code_info(
                        paper_key=pk,
                        has_code=merged["has_code"],
                        has_code_source=merged["has_code_source"],
                        code_url=merged.get("code_url"),
                    )
                    paper.has_code = merged["has_code"]
                    paper.has_code_source = merged["has_code_source"]
                    paper.code_url = merged.get("code_url")

            # Check if all papers were discarded
            non_discarded_count = sum(
                1 for e in evaluations if not e.get("discard", False)
            )
            if non_discarded_count == 0:
                logger.warning(
                    "All %d papers were discarded for topic '%s'. "
                    "Skipping ranking/summarization steps.",
                    len(evaluations), slug,
                )
                if run_id >= 0:
                    try:
                        self._db.update_run_status(run_id, "completed")
                        self._db.update_run_stats(
                            run_id,
                            total_collected=total_collected,
                            total_filtered=total_filtered,
                            total_scored=total_scored,
                            total_discarded=total_discarded,
                            total_output=0,
                            threshold_used=0, threshold_lowered=False,
                        )
                    except Exception as exc:
                        logger.warning("DB update failed (all-discarded path): %s", exc)
                return {
                    "run_id": run_id,
                    "total_collected": total_collected,
                    "total_filtered": total_filtered,
                    "total_scored": total_scored,
                    "total_discarded": total_discarded,
                    "total_output": 0, "report_paths": {},
                }

            # ---- Step 6: Ranker ----
            ranked_papers = self._step_rank(
                evaluations, papers_map, window_end, embedding_mode
            )
            total_output = len(ranked_papers)
            logger.info(
                "Step 6: Ranked %d papers for '%s'",
                total_output, slug,
            )

            # Extract threshold info from scoring config
            thresh_cfg = self._config.scoring.get("thresholds", {})
            threshold_used = thresh_cfg.get("default", 60)
            threshold_lowered = any(
                p.get("score_lowered", False) for p in ranked_papers
            )

            # ---- Step 7: Clustering ----
            clusters = self._step_cluster(
                filtered_papers, embedding_ranker, embedding_mode
            )
            logger.info(
                "Step 7: Clustering done (%d clusters) for '%s'",
                len(clusters), slug,
            )

            # ---- Step 8: Agent 3 - Summarization ----
            summarized_papers = self._step_summarize(
                ranked_papers, topic.description, papers_map
            )
            logger.info(
                "Step 8: Summarization completed for '%s'", slug
            )

            # Insert evaluations into DB
            for sp in summarized_papers:
                flags_dict = sp.get("flags", {})
                if isinstance(flags_dict, EvaluationFlags):
                    flags_obj = flags_dict
                else:
                    flags_obj = EvaluationFlags.from_dict(flags_dict)

                evaluation = Evaluation(
                    run_id=run_id,
                    paper_key=sp["paper_key"],
                    llm_base_score=sp.get("llm_base_score", sp.get("base_score", 0)),
                    flags=flags_obj,
                    prompt_ver_score=prompt_versions.get("agent2", ""),
                    embed_score=sp.get("embed_score"),
                    bonus_score=sp.get("bonus_score"),
                    final_score=sp.get("final_score"),
                    rank=sp.get("rank"),
                    tier=sp.get("tier"),
                    discarded=sp.get("discard", False),
                    score_lowered=sp.get("score_lowered"),
                    summary_ko=sp.get("summary_ko"),
                    reason_ko=sp.get("reason_ko"),
                    insight_ko=sp.get("insight_ko"),
                    brief_reason=sp.get("brief_reason"),
                    prompt_ver_summ=prompt_versions.get("agent3"),
                )
                self._db.insert_evaluation(evaluation)

            # ---- Step 9: Remind Selection ----
            remind_papers = self._step_remind(slug, run_id)
            logger.info(
                "Step 9: %d remind papers for '%s'",
                len(remind_papers), slug,
            )

            # ---- Step 10: Report Generation ----
            stats = {
                "total_collected": total_collected,
                "total_filtered": total_filtered,
                "total_discarded": total_discarded,
                "total_scored": total_scored,
                "total_output": total_output,
            }

            # Extract keywords used from agent1 output
            keywords_used = self._extract_keywords(agent1_output)

            report_paths = self._step_generate_reports(
                topic=topic,
                date_compact=date_compact,
                display_date_kst=display_date_kst,
                window_start=window_start,
                window_end=window_end,
                embedding_mode=embedding_mode,
                scoring_weights=scoring_weights,
                stats=stats,
                threshold_used=threshold_used,
                threshold_lowered=threshold_lowered,
                run_id=run_id,
                keywords_used=keywords_used,
                ranked_papers=summarized_papers,
                clusters=clusters,
                remind_papers=remind_papers,
            )
            logger.info(
                "Step 10: Reports generated for '%s': %s",
                slug, report_paths,
            )

            # ---- Step 11: Post-run updates ----
            # Save dedup seen items
            dedup_mode = self._run_options.get("dedup_mode", "skip_recent")
            from core.storage.dedup import DedupManager

            dedup = DedupManager(
                db_manager=self._db,
                seen_items_path=self._config.sources.get(
                    "seen_items_path", "data/seen_items.jsonl"
                ),
                dedup_mode=dedup_mode,
            )
            for sp in summarized_papers:
                dedup.mark_seen(sp["paper_key"], slug)
            dedup.save_seen_items()

            # Update last_success.json
            self._search_window.update_last_success(slug, window_end)

            # GitHub Issue upsert (optional, skip on missing config)
            self._step_github_issue(
                topic, display_date_kst, summarized_papers, total_output
            )

            logger.info("Step 11: Post-run updates done for '%s'", slug)

            # ---- Step 12: Mark run completed ----
            if run_id >= 0:
                try:
                    self._db.update_run_status(run_id, "completed")
                    self._db.update_run_stats(
                        run_id,
                        total_collected=total_collected,
                        total_filtered=total_filtered,
                        total_scored=total_scored,
                        total_discarded=total_discarded,
                        total_output=total_output,
                        threshold_used=threshold_used,
                        threshold_lowered=threshold_lowered,
                    )
                except Exception as exc:
                    logger.warning("DB update failed (completion): %s", exc)
            logger.info(
                "Step 12: Run completed for '%s' (run_id=%d)",
                slug, run_id,
            )

            return {
                "run_id": run_id,
                "total_collected": total_collected,
                "total_filtered": total_filtered,
                "total_scored": total_scored,
                "total_discarded": total_discarded,
                "total_output": total_output,
                "report_paths": report_paths,
            }

        except Exception:
            # Mark run as failed
            error_trace = traceback.format_exc()
            if run_id >= 0:
                try:
                    self._db.update_run_status(run_id, "failed", errors=error_trace)
                except Exception as db_exc:
                    logger.warning("DB update_run_status(failed) error: %s", db_exc)
            raise

    # ------------------------------------------------------------------
    # Start notification
    # ------------------------------------------------------------------

    @staticmethod
    def _send_start_notifications(
        topic: Any,
        window_start: Any = None,
        window_end: Any = None,
    ) -> None:
        """Send start-event notifications for a topic.

        Best-effort: failures are logged but never block the pipeline.
        """
        if not topic.notify:
            return

        try:
            from output.notifiers.base import NotifyPayload
            from output.notifiers.registry import NotifierRegistry

            registry = NotifierRegistry()
            notifiers = registry.get_notifiers_for_event(topic.notify, "start")
            if not notifiers:
                return

            # Build search window string
            search_window = None
            if window_start and window_end:
                fmt = "%Y-%m-%d %H:%M"
                search_window = f"{window_start.strftime(fmt)} ~ {window_end.strftime(fmt)}"

            # Categories from topic spec
            categories = getattr(topic, "arxiv_categories", []) or []

            payload = NotifyPayload(
                topic_slug=topic.slug,
                topic_name=topic.name,
                display_date="",
                keywords=[],
                total_output=0,
                event_type="start",
                categories=categories,
                search_window=search_window,
            )

            for notifier in notifiers:
                try:
                    notifier.notify(payload)
                except Exception as exc:
                    logger.warning(
                        "Start notification failed for '%s': %s",
                        topic.slug,
                        exc,
                    )
        except Exception as exc:
            logger.warning(
                "Start notification setup failed for '%s': %s",
                topic.slug,
                exc,
            )

    # ------------------------------------------------------------------
    # Pipeline step implementations
    # ------------------------------------------------------------------

    def _step_agent1(self, topic: Any) -> dict:
        """Step 2: Run Agent 1 (Keyword Expander)."""
        from agents.keyword_expander import KeywordExpander
        from core.llm.openrouter_client import OpenRouterClient

        client = OpenRouterClient(self._config)
        response_format_supported = self._config.llm.get(
            "response_format_supported", False
        )
        expander = KeywordExpander(
            config=self._config,
            client=client,
            response_format_supported=response_format_supported,
        )
        return expander.expand(topic)

    def _step_collect(
        self,
        topic: Any,
        agent1_output: dict,
        window_start: datetime,
        window_end: datetime,
        run_id: int,
    ) -> tuple:
        """Step 3: Collect papers from sources.

        Returns:
            Tuple of (papers: list[Paper], query_stats: list[QueryStats]).
        """
        from core.sources.registry import SourceRegistry

        source_type = self._config.sources.get("primary", "arxiv")
        adapter_cls = SourceRegistry.get(source_type)
        adapter = adapter_cls()

        papers = adapter.collect(
            agent1_output=agent1_output,
            categories=topic.arxiv_categories,
            window_start=window_start,
            window_end=window_end,
            config=self._config.sources,
        )

        # Get query stats from adapter
        query_stats: List[QueryStats] = []
        if hasattr(adapter, "query_stats"):
            query_stats = adapter.query_stats

        # Apply dedup
        dedup_mode = self._run_options.get("dedup_mode", "skip_recent")
        from core.storage.dedup import DedupManager

        dedup = DedupManager(
            db_manager=self._db,
            seen_items_path=self._config.sources.get(
                "seen_items_path", "data/seen_items.jsonl"
            ),
            dedup_mode=dedup_mode,
        )
        dedup.reset_in_run()

        deduped: List[Paper] = []
        for paper in papers:
            if not dedup.is_duplicate(paper.paper_key, topic.slug):
                deduped.append(paper)

        # UTC window filtering
        filtered_by_window: List[Paper] = []
        for paper in deduped:
            pub = paper.published_at_utc
            if pub is not None and window_start <= pub <= window_end:
                filtered_by_window.append(paper)
            else:
                # Keep papers without valid timestamps
                if pub is None:
                    filtered_by_window.append(paper)

        # Code detection
        from core.scoring.code_detector import CodeDetector

        code_detector = CodeDetector()
        for paper in filtered_by_window:
            result = code_detector.detect(paper.abstract, paper.comment)
            if result.get("has_code", False):
                paper.has_code = True
                paper.has_code_source = result.get("has_code_source", "regex")
                paper.code_url = result.get("code_url")

        return filtered_by_window, query_stats

    def _step_filter(
        self,
        papers: List[Paper],
        agent1_output: dict,
        topic: Any,
        embedding_ranker: Any,
        topic_embedding_text: Optional[str],
    ) -> tuple:
        """Step 4: Hybrid filter."""
        from core.scoring.hybrid_filter import HybridFilter

        hybrid_filter = HybridFilter(self._config.filter)

        er = embedding_ranker if embedding_ranker.available else None

        filtered_papers, filter_stats = hybrid_filter.filter(
            papers=papers,
            agent1_output=agent1_output,
            topic=topic,
            embedding_ranker=er,
            topic_embedding_text=topic_embedding_text,
        )
        return filtered_papers, filter_stats

    def _step_score(
        self,
        papers: List[Paper],
        topic_description: str,
    ) -> List[dict]:
        """Step 5: Agent 2 - Score papers."""
        from agents.scorer import Scorer
        from core.llm.openrouter_client import OpenRouterClient

        client = OpenRouterClient(self._config)
        response_format_supported = self._config.llm.get(
            "response_format_supported", False
        )
        scorer = Scorer(
            config=self._config,
            client=client,
            response_format_supported=response_format_supported,
        )
        return scorer.score(papers, topic_description, self._rate_limiter)

    def _step_rank(
        self,
        evaluations: List[dict],
        papers_map: Dict[str, Paper],
        window_end: datetime,
        embedding_mode: str,
    ) -> List[dict]:
        """Step 6: Rank papers.

        Maps Scorer output format to Ranker input format:
        - base_score -> llm_base_score
        - flags (dict) -> flags (EvaluationFlags)
        - discard -> discarded
        """
        from core.scoring.ranker import Ranker

        # Map scorer output to ranker input
        mapped_evals: List[dict] = []
        for ev in evaluations:
            mapped = dict(ev)
            # Rename base_score to llm_base_score for Ranker
            if "base_score" in mapped and "llm_base_score" not in mapped:
                mapped["llm_base_score"] = mapped.pop("base_score")
            # Convert flags dict to EvaluationFlags
            flags = mapped.get("flags", {})
            if isinstance(flags, dict):
                mapped["flags"] = EvaluationFlags.from_dict(flags)
            # Rename discard to discarded for Ranker
            if "discard" in mapped and "discarded" not in mapped:
                mapped["discarded"] = mapped.pop("discard")
            # Set default embed_score
            if "embed_score" not in mapped:
                mapped["embed_score"] = None
            mapped_evals.append(mapped)

        ranker = Ranker(self._config.scoring)
        ranked = ranker.rank(mapped_evals, papers_map, window_end, embedding_mode)

        # Convert EvaluationFlags back to plain dict for JSON serialization
        for item in ranked:
            flags_obj = item.get("flags")
            if isinstance(flags_obj, EvaluationFlags):
                item["flags"] = flags_obj.to_dict()

        return ranked

    def _step_cluster(
        self,
        papers: List[Paper],
        embedding_ranker: Any,
        embedding_mode: str,
    ) -> List[dict]:
        """Step 7: Cluster papers (embedding-only)."""
        if embedding_mode == "disabled" or not embedding_ranker.available:
            return []

        from core.clustering.clusterer import Clusterer

        threshold = self._config.clustering.get("similarity_threshold", 0.85)
        clusterer = Clusterer(threshold=threshold)
        return clusterer.cluster(papers)

    def _step_summarize(
        self,
        ranked_papers: List[dict],
        topic_description: str,
        papers_map: Optional[Dict[str, Paper]] = None,
    ) -> List[dict]:
        """Step 8: Agent 3 - Summarize papers."""
        if not ranked_papers:
            return []

        # Enrich ranked_papers with fields from papers_map for reports
        if papers_map:
            for rp in ranked_papers:
                pk = rp.get("paper_key", "")
                paper = papers_map.get(pk)
                if paper is None:
                    continue
                if "title" not in rp:
                    rp["title"] = paper.title
                    rp["abstract"] = paper.abstract
                    if "base_score" not in rp:
                        rp["base_score"] = rp.get("llm_base_score", 0)
                # Add report-essential fields from Paper
                if "url" not in rp:
                    rp["url"] = paper.url
                    rp["pdf_url"] = paper.pdf_url
                    rp["categories"] = paper.categories
                    rp["code_url"] = paper.code_url
                    if paper.published_at_utc is not None:
                        rp["published_at_utc"] = paper.published_at_utc.strftime(
                            "%Y-%m-%d"
                        )
                    else:
                        rp["published_at_utc"] = ""

        from agents.summarizer import Summarizer
        from core.llm.openrouter_client import OpenRouterClient

        client = OpenRouterClient(self._config)
        response_format_supported = self._config.llm.get(
            "response_format_supported", False
        )
        summarizer = Summarizer(
            config=self._config,
            client=client,
            response_format_supported=response_format_supported,
        )
        summaries = summarizer.summarize(
            ranked_papers, topic_description, self._rate_limiter
        )

        # Merge summaries back into ranked papers
        summary_map: Dict[str, dict] = {}
        for s in summaries:
            pk = s.get("paper_key", "")
            if pk:
                summary_map[pk] = s

        enriched: List[dict] = []
        for rp in ranked_papers:
            merged = dict(rp)
            pk = rp.get("paper_key", "")
            if pk in summary_map:
                sm = summary_map[pk]
                merged["summary_ko"] = sm.get("summary_ko", "")
                merged["reason_ko"] = sm.get("reason_ko", "")
                merged["insight_ko"] = sm.get("insight_ko", "")
            else:
                # Summarization failed for this paper - set defaults
                merged.setdefault("summary_ko", "(요약 생성 실패)")
                merged.setdefault("reason_ko", "")
                merged.setdefault("insight_ko", "")
            enriched.append(merged)

        return enriched

    def _step_remind(self, topic_slug: str, run_id: int) -> List[dict]:
        """Step 9: Select remind papers."""
        from core.remind.remind_selector import RemindSelector

        remind_config = self._config.remind
        min_score = remind_config.get("min_score", 80.0)
        max_count = remind_config.get("max_expose_count", 2)

        selector = RemindSelector(db=self._db)
        return selector.select(
            topic_slug=topic_slug,
            current_run_id=run_id,
            min_score=min_score,
            max_recommend_count=max_count,
        )

    def _step_generate_reports(
        self,
        topic: Any,
        date_compact: str,
        display_date_kst: str,
        window_start: datetime,
        window_end: datetime,
        embedding_mode: str,
        scoring_weights: dict,
        stats: dict,
        threshold_used: int,
        threshold_lowered: bool,
        run_id: int,
        keywords_used: List[str],
        ranked_papers: List[dict],
        clusters: List[dict],
        remind_papers: List[dict],
    ) -> Dict[str, str]:
        """Step 10: Generate reports (JSON, MD, HTML)."""
        from output.render.json_exporter import export_json
        from output.render.md_generator import generate_markdown
        from output.render.html_generator import generate_report_html

        output_dir = self._config.output.get("report_dir", "tmp/reports")
        date_subdir = "%s/%s" % (output_dir, display_date_kst)

        display_title = "%s 논문 리포트 (%s)" % (
            topic.name, display_date_kst
        )

        # JSON export
        json_path = export_json(
            topic_slug=topic.slug,
            topic_name=topic.name,
            date_str=date_compact,
            display_title=display_title,
            window_start_utc=window_start.isoformat(),
            window_end_utc=window_end.isoformat(),
            embedding_mode=embedding_mode,
            scoring_weights=scoring_weights,
            stats=stats,
            threshold_used=threshold_used,
            threshold_lowered=threshold_lowered,
            run_id=run_id,
            keywords_used=keywords_used,
            papers=ranked_papers,
            clusters=clusters,
            remind_papers=remind_papers,
            output_dir=date_subdir,
        )

        # Load JSON report data for MD and HTML generation
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Markdown
        md_path = generate_markdown(
            report_data=report_data,
            output_dir=date_subdir,
        )

        # HTML
        template_dir = self._config.output.get("template_dir", "templates")
        html_path = generate_report_html(
            report_data=report_data,
            output_dir=date_subdir,
            template_dir=template_dir,
        )

        return {
            "json": json_path,
            "md": md_path,
            "html": html_path,
        }

    def _step_github_issue(
        self,
        topic: Any,
        date_str: str,
        papers: List[dict],
        total_output: int,
    ) -> None:
        """Step 11 (partial): Upsert GitHub Issue if configured."""
        github_config = self._config.notifications.get("github", {})
        repo = github_config.get("repo")
        token = github_config.get("token")

        if not repo or not token:
            logger.debug("GitHub Issue: skipped (no repo/token configured)")
            return

        try:
            from output.github_issue import GitHubIssueManager

            manager = GitHubIssueManager(
                repo=repo,
                token=token,
                issue_map_path=github_config.get(
                    "issue_map_path", "data/issue_map.json"
                ),
            )
            manager.upsert_issue(
                topic_slug=topic.slug,
                topic_name=topic.name,
                date_str=date_str,
                papers=papers[:10],
                total_output=total_output,
            )
        except Exception:
            logger.warning(
                "GitHub Issue creation/update failed for '%s'. "
                "Check GITHUB_TOKEN and repo permissions.",
                topic.slug,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_topics(self) -> list:
        """Resolve which topics to run based on run_options."""
        all_topics = self._config.topics
        selected_slugs = self._run_options.get("topics")

        if selected_slugs:
            slug_set = set(selected_slugs)
            return [t for t in all_topics if t.slug in slug_set]

        return list(all_topics)

    @staticmethod
    def _extract_keywords(agent1_output: dict) -> List[str]:
        """Extract flat keyword list from Agent 1 output."""
        keywords: List[str] = []

        concepts = agent1_output.get("concepts", [])
        for concept in concepts:
            kws = concept.get("keywords", [])
            keywords.extend(kws)

        cross_domain = agent1_output.get("cross_domain_keywords", [])
        keywords.extend(cross_domain)

        return keywords
