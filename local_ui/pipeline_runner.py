"""Pipeline runner for local UI background execution.

Manages background pipeline execution for the web UI including both
dry-run (keyword expansion + query building) and full pipeline runs.

Reference: TASK-052 devspec Section 18-3.
"""

from __future__ import annotations

import glob as glob_module
import json
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

from agents.keyword_expander import KeywordExpander
from core.config import load_config
from core.llm.openrouter_client import OpenRouterClient
from core.sources.arxiv_query_builder import ArxivQueryBuilder

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Manages background pipeline execution for the UI.

    Provides two execution modes:
    1. Dry-run: Fast keyword expansion + query building (seconds)
    2. Full run: Complete pipeline with collection/scoring/summarization (minutes)

    Thread-safe status tracking allows polling from frontend.
    """

    def __init__(self, config_path: str, db_path: str) -> None:
        """Initialize pipeline runner.

        Args:
            config_path: Path to config.yaml
            db_path: Path to SQLite database
        """
        self._config_path = Path(config_path)
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._status: Dict[str, Any] = {
            "running": False,
            "run_id": None,
            "progress": "",
            "topics_completed": 0,
            "topics_total": 0,
            "current_topic": None,
            "error": None,
            "log_file": None,
            "log_lines": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_dryrun(self, topic_slug: Optional[str] = None, skip_cache: bool = False) -> Dict[str, Any]:
        """Run dry-run synchronously (fast operation).

        Executes Agent 1 (keyword expansion) + QueryBuilder for selected topics.
        Does NOT run scoring/summarization - just keywords and queries.

        Args:
            topic_slug: Specific topic slug to run, or None for all topics
            skip_cache: If True, bypass keyword cache and force LLM call

        Returns:
            dict with structure:
            {
                "success": bool,
                "topics": [
                    {
                        "slug": str,
                        "concepts": [...],
                        "cross_domain_keywords": [...],
                        "exclude_keywords": [...],
                        "queries": [str]
                    }
                ],
                "error": str | None
            }

        Raises:
            Exception: If dry-run fails
        """
        with self._lock:
            if self._status["running"]:
                return {
                    "success": False,
                    "topics": [],
                    "error": "Pipeline is already running a full execution",
                }
        try:
            # Load config
            config = load_config(str(self._config_path))

            # Determine topics to process
            if topic_slug:
                topics = [t for t in config.topics if t.slug == topic_slug]
                if not topics:
                    return {
                        "success": False,
                        "topics": [],
                        "error": f"Topic '{topic_slug}' not found in config",
                    }
            else:
                topics = config.topics

            # Initialize components
            client = OpenRouterClient(config=config)
            keyword_expander = KeywordExpander(config=config, client=client)
            keyword_expander.CACHE_FILE = str(self._keyword_cache_file())
            query_builder = ArxivQueryBuilder()

            # Process each topic
            results = []
            for topic in topics:
                try:
                    # Agent 1: Keyword expansion
                    logger.info(f"Dry-run: expanding keywords for topic '{topic.slug}'")

                    # Check if we have API key
                    api_key = config.llm.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
                    if not api_key:
                        # Try to use cached keywords if available
                        cache_data = self._load_keyword_cache_data()
                        if cache_data:
                            # Find cached entry for this topic
                            cache_key = self._compute_cache_key_simple(topic)
                            if cache_key in cache_data:
                                agent1_output = cache_data[cache_key].get("result", {})
                                logger.info(f"Using cached keywords for '{topic.slug}'")
                            else:
                                return {
                                    "success": False,
                                    "topics": [],
                                    "error": (
                                        "No API key configured and no cache available. "
                                        "Please set OPENROUTER_API_KEY environment variable."
                                    ),
                                }
                        else:
                            return {
                                "success": False,
                                "topics": [],
                                "error": (
                                    "No API key configured and no cache available. "
                                    "Please set OPENROUTER_API_KEY environment variable."
                                ),
                            }
                    else:
                        # Call Agent 1 with API
                        agent1_output = keyword_expander.expand(topic, skip_cache=skip_cache)

                    # Build queries
                    queries = query_builder.build_queries(
                        agent1_output=agent1_output,
                        categories=topic.arxiv_categories,
                    )

                    results.append(
                        {
                            "slug": topic.slug,
                            "concepts": agent1_output.get("concepts", []),
                            "cross_domain_keywords": agent1_output.get("cross_domain_keywords", []),
                            "query_must_keywords": agent1_output.get("query_must_keywords", []),
                            "exclude_keywords": agent1_output.get("exclude_keywords", []),
                            "exclude_mode": agent1_output.get("exclude_mode", "soft"),
                            "queries": queries,
                        }
                    )

                except Exception as e:
                    logger.error(f"Dry-run failed for topic '{topic.slug}': {e}")
                    logger.debug(traceback.format_exc())
                    results.append(
                        {
                            "slug": topic.slug,
                            "error": str(e),
                        }
                    )

            return {
                "success": True,
                "topics": results,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Dry-run failed: {e}")
            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "topics": [],
                "error": str(e),
            }

    def start_dryrun_streamed(
        self, topic_slug: Optional[str] = None, skip_cache: bool = False
    ) -> Generator[Tuple[str, dict], None, None]:
        """Run dry-run as a generator yielding SSE events.

        Same logic as start_dryrun() but yields (event_type, data) tuples
        for real-time progress streaming.

        Args:
            topic_slug: Specific topic slug to run, or None for all topics
            skip_cache: If True, bypass keyword cache and force LLM call

        Yields:
            Tuples of (event_type, data_dict) where event_type is
            "log", "result", or "error".
        """
        with self._lock:
            if self._status["running"]:
                yield ("error", {"message": "Pipeline is already running a full execution"})
                return

        try:
            yield ("log", {"step": "init", "message": "초기화 중...", "progress": 5})

            config = load_config(str(self._config_path))

            if topic_slug:
                topics = [t for t in config.topics if t.slug == topic_slug]
                if not topics:
                    yield ("error", {"message": f"Topic '{topic_slug}' not found in config"})
                    return
            else:
                topics = config.topics

            if not topics:
                yield ("error", {"message": "No topics configured in config.yaml"})
                return

            yield ("log", {"step": "llm_init", "message": "LLM 클라이언트 연결 중...", "progress": 10})

            api_key = config.llm.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
            # Show actual model used by keyword_expander (agent-level override takes priority)
            kw_agent_cfg = config.agents.get("keyword_expander", {})
            model_name = kw_agent_cfg.get("model", config.llm.get("model", "unknown"))

            client = OpenRouterClient(config=config)
            keyword_expander = KeywordExpander(config=config, client=client)
            keyword_expander.CACHE_FILE = str(self._keyword_cache_file())
            query_builder = ArxivQueryBuilder()

            results = []
            total_topics = len(topics)

            for i, topic in enumerate(topics):
                base_progress = 15 + int((i / total_topics) * 75)

                try:
                    if not api_key:
                        cache_data = self._load_keyword_cache_data()
                        if cache_data:
                            cache_key = self._compute_cache_key_simple(topic)
                            if cache_key in cache_data:
                                yield ("log", {
                                    "step": "keyword_cache",
                                    "message": f"캐시에서 키워드 로딩 중 ({topic.slug})...",
                                    "topic": topic.slug,
                                    "progress": base_progress,
                                })
                                agent1_output = cache_data[cache_key].get("result", {})
                            else:
                                yield ("error", {
                                    "message": "No API key configured and no cache available. "
                                    "Please set OPENROUTER_API_KEY environment variable."
                                })
                                return
                        else:
                            yield ("error", {
                                "message": "No API key configured and no cache available. "
                                "Please set OPENROUTER_API_KEY environment variable."
                            })
                            return
                    else:
                        # Show chunking info if categories exceed chunk_size
                        chunk_size = kw_agent_cfg.get("chunk_size", 15)
                        n_cats = len(topic.arxiv_categories)
                        if n_cats > chunk_size:
                            n_chunks = (n_cats + chunk_size - 1) // chunk_size
                            yield ("log", {
                                "step": "keyword_expansion",
                                "message": f"LLM 키워드 생성 중 ({topic.slug}) - {model_name} | 카테고리 {n_cats}개 → {n_chunks}회 분할 호출...",
                                "topic": topic.slug,
                                "progress": base_progress,
                            })
                        else:
                            yield ("log", {
                                "step": "keyword_expansion",
                                "message": f"LLM 키워드 생성 중 ({topic.slug}) - {model_name} | 카테고리 {n_cats}개...",
                                "topic": topic.slug,
                                "progress": base_progress,
                            })
                        agent1_output = keyword_expander.expand(topic, skip_cache=skip_cache)

                    n_concepts = len(agent1_output.get("concepts", []))
                    yield ("log", {
                        "step": "keyword_done",
                        "message": f"키워드 {n_concepts}개 컨셉 생성 완료 ({topic.slug})",
                        "topic": topic.slug,
                        "progress": base_progress + int(37 / total_topics),
                    })

                    yield ("log", {
                        "step": "query_building",
                        "message": f"검색 쿼리 생성 중 ({topic.slug})...",
                        "topic": topic.slug,
                        "progress": base_progress + int(56 / total_topics),
                    })

                    queries = query_builder.build_queries(
                        agent1_output=agent1_output,
                        categories=topic.arxiv_categories,
                    )

                    yield ("log", {
                        "step": "query_done",
                        "message": f"쿼리 {len(queries)}개 생성 완료 ({topic.slug})",
                        "topic": topic.slug,
                        "progress": base_progress + int(75 / total_topics),
                    })

                    results.append({
                        "slug": topic.slug,
                        "concepts": agent1_output.get("concepts", []),
                        "cross_domain_keywords": agent1_output.get("cross_domain_keywords", []),
                        "query_must_keywords": agent1_output.get("query_must_keywords", []),
                        "exclude_keywords": agent1_output.get("exclude_keywords", []),
                        "exclude_mode": agent1_output.get("exclude_mode", "soft"),
                        "queries": queries,
                    })

                except Exception as e:
                    logger.error(f"Dry-run failed for topic '{topic.slug}': {e}")
                    logger.debug(traceback.format_exc())
                    results.append({"slug": topic.slug, "error": str(e)})

            yield ("log", {"step": "complete", "message": "완료!", "progress": 100})
            yield ("result", {"success": True, "topics": results})

        except Exception as e:
            logger.error(f"Dry-run failed: {e}")
            logger.debug(traceback.format_exc())
            yield ("error", {"message": str(e)})

    def start_run(
        self,
        topic_slug: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        dedup: str = "skip_recent",
    ) -> Dict[str, Any]:
        """Start full pipeline run in background thread.

        Args:
            topic_slug: Specific topic slug to run, or None for all topics
            date_from: Start date (YYYY-MM-DD) or None for auto
            date_to: End date (YYYY-MM-DD) or None for auto
            dedup: Dedup mode ("skip_recent" or "none")

        Returns:
            dict with {"run_id": str, "status": "started"}
        """
        with self._lock:
            if self._status["running"]:
                return {"error": "Pipeline already running", "run_id": self._status["run_id"]}

            run_id = str(uuid.uuid4())[:8]

            # Clean up previous log files
            for old_log in glob_module.glob("/tmp/pipeline_run_*.log"):
                try:
                    os.remove(old_log)
                except OSError:
                    pass

            log_file = f"/tmp/pipeline_run_{run_id}.log"
            self._status = {
                "running": True,
                "run_id": run_id,
                "progress": "Starting...",
                "topics_completed": 0,
                "topics_total": 0,
                "current_topic": None,
                "error": None,
                "log_file": log_file,
                "log_lines": 0,
                "step_progress": 0,
                "step_name": "",
            }

        # Start background thread
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(run_id, topic_slug, date_from, date_to, dedup),
            daemon=True,
        )
        thread.start()

        return {"run_id": run_id, "status": "started"}

    def cancel(self) -> Dict[str, Any]:
        """Cancel running pipeline execution.

        Returns:
            dict with cancel result
        """
        with self._lock:
            if not self._status["running"]:
                return {"success": False, "message": "No pipeline running"}
            self._cancel_event.set()
            return {"success": True, "message": "Cancel requested"}

    def get_status(self) -> Dict[str, Any]:
        """Return current pipeline status.

        Returns:
            dict with running, run_id, progress, topics_completed, topics_total
        """
        with self._lock:
            return self._status.copy()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _write_log(self, run_id: str, step: str, msg: str, detail: Optional[Dict[str, Any]] = None) -> None:
        """Append a structured JSON log line to the run log file.

        Args:
            run_id: Current run identifier.
            step: Pipeline step name (e.g. "preflight", "collect").
            msg: Human-readable log message.
            detail: Optional dict with extra structured data.
        """
        log_file = f"/tmp/pipeline_run_{run_id}.log"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "msg": msg,
        }
        if detail:
            entry["detail"] = detail
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            with self._lock:
                self._status["log_lines"] = self._status.get("log_lines", 0) + 1
        except OSError:
            pass

    def _run_pipeline_thread(
        self,
        run_id: str,
        topic_slug: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        dedup: str,
    ) -> None:
        """Background thread for full pipeline execution.

        Runs the same pipeline as main.py:
        Preflight -> TopicLoopOrchestrator -> PostLoopProcessor
        """
        try:
            # 1. Preflight
            with self._lock:
                self._status["progress"] = "Preflight checks..."

            # Check for previous checkpoint (informational only for now)
            prev_checkpoint = self._load_checkpoint(run_id)
            if prev_checkpoint:
                logger.info(
                    "Found existing checkpoint for run %s: step=%s, topic=%s",
                    run_id,
                    prev_checkpoint.get("step_completed", "?"),
                    prev_checkpoint.get("topic_slug", "?"),
                )
                self._write_log(run_id, "checkpoint", "Previous checkpoint found", {
                    "step_completed": prev_checkpoint.get("step_completed"),
                    "topic_slug": prev_checkpoint.get("topic_slug"),
                    "timestamp": prev_checkpoint.get("timestamp"),
                })

            self._write_log(run_id, "preflight", "Preflight checks starting...")

            from core.pipeline.preflight import run_preflight

            dt_from = (
                datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if date_from else None
            )
            dt_to = (
                datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if date_to else None
            )

            preflight_result = run_preflight(
                config_path=str(self._config_path),
                date_from=dt_from,
                date_to=dt_to,
            )
            config = preflight_result.config

            warnings_list = list(preflight_result.warnings)
            for w in warnings_list:
                logger.warning("Preflight warning: %s", w)

            self._write_log(run_id, "preflight", "Preflight checks passed", {
                "warnings_count": len(warnings_list),
                "warnings": warnings_list[:5],
            })

            # 2. Initialize shared resources
            with self._lock:
                self._status["progress"] = "Initializing..."

            from core.pipeline.search_window import SearchWindowComputer
            from core.storage.db_manager import DBManager
            from core.storage.usage_tracker import UsageTracker

            db_path = config.database.get("path", str(self._db_path))
            db_manager = DBManager(db_path)
            rate_limiter = preflight_result.rate_limiter
            search_window = SearchWindowComputer(db_manager=db_manager)
            usage_tracker = UsageTracker()

            self._write_log(run_id, "init", "DB/rate limiter/search window initialized", {
                "db_path": db_path,
                "dedup_mode": dedup,
            })

            # 3. Build run_options
            run_options: Dict[str, Any] = {
                "mode": "full",
                "date_from": date_from,
                "date_to": date_to,
                "dedup_mode": dedup,
                "topics": [topic_slug] if topic_slug else None,
            }

            # 4. Determine topics for status tracking
            if topic_slug:
                topics = [t for t in config.topics if t.slug == topic_slug]
            else:
                topics = config.topics

            with self._lock:
                self._status["topics_total"] = len(topics)
                self._status["progress"] = "Running pipeline..."

            self._write_log(run_id, "init", f"Pipeline starting for {len(topics)} topic(s)", {
                "topic_slugs": [t.slug for t in topics],
            })

            # 5. Run topic loop
            from core.pipeline.topic_loop import TopicLoopOrchestrator

            orchestrator = TopicLoopOrchestrator(
                config=config,
                db_manager=db_manager,
                rate_limiter=rate_limiter,
                search_window=search_window,
                usage_tracker=usage_tracker,
                run_options=run_options,
            )

            # Patch orchestrator to update our status and log each step
            original_run_single = orchestrator._run_single_topic

            # Also patch _step_collect to log per-query stats
            original_step_collect = orchestrator._step_collect

            def _check_cancel(step_name: str = "") -> None:
                """Check cancel event and raise if set."""
                if self._cancel_event.is_set():
                    self._write_log(run_id, "cancel", f"Pipeline cancelled by user at step: {step_name}")
                    raise InterruptedError(f"Pipeline cancelled by user at step: {step_name}")

            def _logged_step_collect(topic_arg: Any, agent1_output: dict, window_start: Any, window_end: Any, topic_run_id: int) -> Any:
                """Wrap _step_collect to log ArXiv query details."""
                _check_cancel("collect")
                with self._lock:
                    self._status["step_progress"] = 20
                    self._status["step_name"] = "논문 수집"
                from core.sources.arxiv_query_builder import ArxivQueryBuilder

                # Pre-compute queries for logging
                builder = ArxivQueryBuilder()
                queries = builder.build_queries(agent1_output, topic_arg.arxiv_categories)
                n_queries = len(queries)
                self._write_log(run_id, "collect", f"ArXiv search starting: {n_queries} queries", {
                    "topic": topic_arg.slug,
                    "n_queries": n_queries,
                })

                t0 = time.time()
                result = original_step_collect(topic_arg, agent1_output, window_start, window_end, topic_run_id)
                elapsed = time.time() - t0

                papers = result[0] if isinstance(result, tuple) else result
                query_stats = result[1] if isinstance(result, tuple) else []
                n_papers = len(papers)

                # Log per-query stats if available
                for i, qs in enumerate(query_stats):
                    q_text = getattr(qs, "query_text", "")[:80]
                    n_results = getattr(qs, "collected", 0)
                    latency_ms = int(getattr(qs, "duration_ms", 0))
                    self._write_log(run_id, "collect_query", f"Query {i+1}/{len(query_stats)}: '{q_text}...' -> {n_results} papers ({latency_ms}ms)", {
                        "topic": topic_arg.slug,
                        "query_index": i + 1,
                        "query_total": len(query_stats),
                        "results_count": n_results,
                        "latency_ms": latency_ms,
                    })

                self._write_log(run_id, "collect", f"Collection complete: {n_papers} papers from {len(query_stats)} queries ({elapsed:.1f}s)", {
                    "topic": topic_arg.slug,
                    "total_papers": n_papers,
                    "total_queries": len(query_stats),
                    "elapsed_s": round(elapsed, 1),
                })

                return result

            orchestrator._step_collect = _logged_step_collect

            # Patch _step_agent1
            original_step_agent1 = orchestrator._step_agent1

            def _logged_step_agent1(topic_arg: Any) -> dict:
                with self._lock:
                    self._status["step_progress"] = 5
                    self._status["step_name"] = "키워드 확장"
                self._write_log(run_id, "agent1", f"Keyword expansion starting for {topic_arg.slug}", {
                    "topic": topic_arg.slug,
                })
                t0 = time.time()
                result = original_step_agent1(topic_arg)
                elapsed = time.time() - t0
                n_concepts = len(result.get("concepts", []))
                n_cross = len(result.get("cross_domain_keywords", []))
                self._write_log(run_id, "agent1", f"Keyword expansion complete: {n_concepts} concepts, {n_cross} cross-domain keywords ({elapsed:.1f}s)", {
                    "topic": topic_arg.slug,
                    "n_concepts": n_concepts,
                    "n_cross_domain": n_cross,
                    "elapsed_s": round(elapsed, 1),
                })
                return result

            orchestrator._step_agent1 = _logged_step_agent1

            # Patch _step_filter
            original_step_filter = orchestrator._step_filter

            def _logged_step_filter(*args: Any, **kwargs: Any) -> Any:
                _check_cancel("filter")
                with self._lock:
                    self._status["step_progress"] = 25
                    self._status["step_name"] = "필터링"
                papers_in = args[0] if args else kwargs.get("papers", [])
                n_before = len(papers_in) if hasattr(papers_in, "__len__") else 0
                t0 = time.time()
                result = original_step_filter(*args, **kwargs)
                elapsed = time.time() - t0
                filtered = result[0] if isinstance(result, tuple) else result
                filter_stats = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
                n_after = len(filtered) if hasattr(filtered, "__len__") else 0

                # Build detailed filter message
                neg = filter_stats.get("excluded_negative", 0)
                nomatch = filter_stats.get("excluded_no_match", 0)
                after_rule = filter_stats.get("after_rule_filter", n_after)
                after_cap = filter_stats.get("after_cap", n_after)

                detail_parts = []
                if neg > 0:
                    detail_parts.append(f"제외키워드 {neg}건")
                if nomatch > 0:
                    detail_parts.append(f"매칭실패 {nomatch}건")
                if after_rule > after_cap:
                    detail_parts.append(f"상한캡 {after_rule - after_cap}건")

                detail_str = ", ".join(detail_parts) if detail_parts else "변동 없음"
                msg = f"필터링: {n_before} → {n_after}건 ({detail_str}) ({elapsed:.1f}s)"

                self._write_log(run_id, "filter", msg, {
                    "n_before": n_before,
                    "n_after": n_after,
                    "excluded_negative": neg,
                    "excluded_no_match": nomatch,
                    "after_rule_filter": after_rule,
                    "after_cap": after_cap,
                    "elapsed_s": round(elapsed, 1),
                })

                # -- Additional filter detail logs --
                neg_hits = filter_stats.get("negative_keyword_hits", {})
                if neg_hits:
                    n_neg_kw = filter_stats.get("negative_keywords_count", len(neg_hits))
                    top_hits = sorted(neg_hits.items(), key=lambda x: x[1], reverse=True)[:5]
                    hits_str = ", ".join(f'"{k}"({v}건)' for k, v in top_hits)
                    self._write_log(run_id, "filter_detail",
                        f"  ├ 제외 키워드 ({n_neg_kw}개 사용): {hits_str}", {})

                excluded_samples = filter_stats.get("excluded_samples", [])
                for sample in excluded_samples[:3]:
                    title = sample.get("title", "")
                    reason = sample.get("reason", "")
                    if reason == "negative_keyword":
                        matched = sample.get("matched_keyword", "")
                        reason_str = f"제외키워드 \"{matched}\"" if matched else "제외키워드"
                    else:
                        reason_str = "매칭실패"
                    self._write_log(run_id, "filter_detail",
                        f"  ├ 제외 예시: 「{title}」← {reason_str}", {})

                sort_method = filter_stats.get("sort_method", "none")
                sort_label = "임베딩 유사도순" if sort_method == "embedding" else "최신순"
                self._write_log(run_id, "filter_detail",
                    f"  └ 정렬: {sort_label}", {})

                self._save_checkpoint(run_id, {
                    "step_completed": "filter",
                    "papers_collected": n_before,
                    "papers_filtered": n_after,
                    "topic_slug": self._status.get("current_topic", ""),
                })
                return result

            orchestrator._step_filter = _logged_step_filter

            # Patch _step_score
            original_step_score = orchestrator._step_score

            def _logged_step_score(papers: Any, topic_desc: str) -> Any:
                _check_cancel("score")
                with self._lock:
                    self._status["step_progress"] = 25
                    self._status["step_name"] = "LLM 평가"
                n_papers = len(papers) if hasattr(papers, "__len__") else 0
                scorer_cfg = config.agents.get("scorer", {})
                model = scorer_cfg.get("model", config.llm.get("model", "unknown"))
                batch_size = scorer_cfg.get("batch_size", 10)
                total_batches = (n_papers + batch_size - 1) // batch_size if n_papers > 0 else 0
                self._write_log(run_id, "score", f"LLM scoring: {n_papers} papers, model: {model}, batch_size: {batch_size} ({total_batches} batches)", {
                    "n_papers": n_papers,
                    "model": model,
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                })

                # Monkey-patch scorer to log each batch
                t0 = time.time()
                from agents.scorer import Scorer
                _orig_score_batch = Scorer._score_batch
                _batch_counter = {"done": 0}
                _runner_self = self

                def _logged_score_batch(scorer_instance, batch, topic_description, rate_limiter, batch_idx):
                    # Cancel check before each batch
                    if _runner_self._cancel_event.is_set():
                        raise InterruptedError("Pipeline cancelled by user during scoring")

                    bt0 = time.time()
                    result = _orig_score_batch(scorer_instance, batch, topic_description, rate_limiter, batch_idx)
                    bt_elapsed = time.time() - bt0
                    _batch_counter["done"] += 1
                    batch_pct = int(25 + (_batch_counter["done"] / total_batches * 40)) if total_batches > 0 else 65
                    with _runner_self._lock:
                        _runner_self._status["step_progress"] = batch_pct
                        _runner_self._status["step_name"] = f"LLM 평가 ({_batch_counter['done']}/{total_batches})"
                    n_ok = len(result) if result else 0
                    _runner_self._write_log(run_id, "score_batch",
                        f"Batch {_batch_counter['done']}/{total_batches}: {n_ok} scored ({bt_elapsed:.1f}s)", {
                        "batch_idx": _batch_counter["done"],
                        "total_batches": total_batches,
                        "n_scored": n_ok,
                        "elapsed_s": round(bt_elapsed, 1),
                    })

                    # Checkpoint after each successful score batch
                    _runner_self._save_checkpoint(run_id, {
                        "step_completed": "score_batch",
                        "score_batches_done": _batch_counter["done"],
                        "score_batches_total": total_batches,
                        "topic_slug": _runner_self._status.get("current_topic", ""),
                    })

                    return result

                Scorer._score_batch = _logged_score_batch
                try:
                    result = original_step_score(papers, topic_desc)
                finally:
                    Scorer._score_batch = _orig_score_batch

                elapsed = time.time() - t0
                n_scored = len(result) if hasattr(result, "__len__") else 0
                n_discarded = sum(1 for e in result if e.get("discard", False)) if n_scored else 0
                self._write_log(run_id, "score", f"Scoring complete: {n_scored} scored, {n_discarded} discarded ({elapsed:.1f}s)", {
                    "n_scored": n_scored,
                    "n_discarded": n_discarded,
                    "elapsed_s": round(elapsed, 1),
                })
                self._save_checkpoint(run_id, {
                    "step_completed": "score",
                    "scored_results_count": n_scored,
                    "n_discarded": n_discarded,
                    "topic_slug": self._status.get("current_topic", ""),
                })
                return result

            orchestrator._step_score = _logged_step_score

            # Patch _step_rank
            original_step_rank = orchestrator._step_rank

            def _logged_step_rank(*args: Any, **kwargs: Any) -> Any:
                _check_cancel("rank")
                with self._lock:
                    self._status["step_progress"] = 70
                    self._status["step_name"] = "순위 결정"
                t0 = time.time()
                result = original_step_rank(*args, **kwargs)
                elapsed = time.time() - t0
                n = len(result) if hasattr(result, "__len__") else 0
                self._write_log(run_id, "rank", f"Ranked {n} papers ({elapsed:.1f}s)", {
                    "n_ranked": n,
                    "elapsed_s": round(elapsed, 1),
                })
                self._save_checkpoint(run_id, {
                    "step_completed": "rank",
                    "n_ranked": n,
                    "topic_slug": self._status.get("current_topic", ""),
                })
                return result

            orchestrator._step_rank = _logged_step_rank

            # Patch _step_cluster
            original_step_cluster = orchestrator._step_cluster

            def _logged_step_cluster(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    self._status["step_progress"] = 75
                    self._status["step_name"] = "클러스터링"
                t0 = time.time()
                result = original_step_cluster(*args, **kwargs)
                elapsed = time.time() - t0
                n = len(result) if hasattr(result, "__len__") else 0
                self._write_log(run_id, "cluster", f"Clustered into {n} groups ({elapsed:.1f}s)", {
                    "n_clusters": n,
                    "elapsed_s": round(elapsed, 1),
                })
                return result

            orchestrator._step_cluster = _logged_step_cluster

            # Patch _step_summarize
            original_step_summarize = orchestrator._step_summarize

            def _logged_step_summarize(ranked: Any, desc: str, pmap: Any = None) -> Any:
                _check_cancel("summarize")
                with self._lock:
                    self._status["step_progress"] = 75
                    self._status["step_name"] = "LLM 요약"
                n = len(ranked) if hasattr(ranked, "__len__") else 0
                summ_cfg = config.agents.get("summarizer", {})
                model = summ_cfg.get("model", config.llm.get("model", "unknown"))
                t1_bs = summ_cfg.get("tier1_batch_size", 5)
                t2_bs = summ_cfg.get("tier2_batch_size", 10)
                tier1_count = min(n, 30)
                tier2_count = max(0, min(n, 100) - 30)
                total_batches = ((tier1_count + t1_bs - 1) // t1_bs) + ((tier2_count + t2_bs - 1) // t2_bs) if n > 0 else 0
                self._write_log(run_id, "summarize", f"LLM summarization: {n} papers, model: {model} ({total_batches} batches)", {
                    "n_papers": n,
                    "model": model,
                    "total_batches": total_batches,
                })

                # Monkey-patch summarizer to log each batch
                t0 = time.time()
                from agents.summarizer import Summarizer
                _orig_summ_batch = Summarizer._summarize_batch
                _summ_counter = {"done": 0}
                _runner_self = self

                def _logged_summ_batch(summ_instance, batch, topic_description, rate_limiter, batch_idx, tier=1):
                    # Cancel check before each batch
                    if _runner_self._cancel_event.is_set():
                        raise InterruptedError("Pipeline cancelled by user during summarization")

                    bt0 = time.time()
                    result = _orig_summ_batch(summ_instance, batch, topic_description, rate_limiter, batch_idx, tier=tier)
                    bt_elapsed = time.time() - bt0
                    _summ_counter["done"] += 1
                    batch_pct = int(75 + (_summ_counter["done"] / total_batches * 20)) if total_batches > 0 else 95
                    with _runner_self._lock:
                        _runner_self._status["step_progress"] = batch_pct
                        _runner_self._status["step_name"] = f"LLM 요약 ({_summ_counter['done']}/{total_batches})"
                    n_ok = len(result) if result else 0
                    _runner_self._write_log(run_id, "summarize_batch",
                        f"Batch {_summ_counter['done']}/{total_batches} (tier{tier}): {n_ok} summarized ({bt_elapsed:.1f}s)", {
                        "batch_idx": _summ_counter["done"],
                        "total_batches": total_batches,
                        "tier": tier,
                        "n_summarized": n_ok,
                        "elapsed_s": round(bt_elapsed, 1),
                    })

                    # Checkpoint after each successful summarize batch
                    _runner_self._save_checkpoint(run_id, {
                        "step_completed": "summarize_batch",
                        "summarize_batches_done": _summ_counter["done"],
                        "summarize_batches_total": total_batches,
                        "topic_slug": _runner_self._status.get("current_topic", ""),
                    })

                    return result

                Summarizer._summarize_batch = _logged_summ_batch
                try:
                    result = original_step_summarize(ranked, desc, pmap)
                finally:
                    Summarizer._summarize_batch = _orig_summ_batch

                elapsed = time.time() - t0
                self._write_log(run_id, "summarize", f"Summarization complete ({elapsed:.1f}s)", {
                    "elapsed_s": round(elapsed, 1),
                })
                self._save_checkpoint(run_id, {
                    "step_completed": "summarize",
                    "topic_slug": self._status.get("current_topic", ""),
                })
                return result

            orchestrator._step_summarize = _logged_step_summarize

            # Patch _step_remind
            original_step_remind = orchestrator._step_remind

            def _logged_step_remind(slug: str, rid: int) -> Any:
                result = original_step_remind(slug, rid)
                n = len(result) if hasattr(result, "__len__") else 0
                self._write_log(run_id, "remind", f"{n} remind papers", {
                    "topic": slug,
                    "n_remind": n,
                })
                return result

            orchestrator._step_remind = _logged_step_remind

            # Patch _step_generate_reports
            original_step_reports = orchestrator._step_generate_reports

            def _logged_step_reports(**kwargs: Any) -> Any:
                with self._lock:
                    self._status["step_progress"] = 100
                    self._status["step_name"] = "리포트 생성"
                t0 = time.time()
                result = original_step_reports(**kwargs)
                elapsed = time.time() - t0
                self._write_log(run_id, "reports", f"Reports generated ({elapsed:.1f}s)", {
                    "paths": result if isinstance(result, dict) else str(result),
                    "elapsed_s": round(elapsed, 1),
                })
                return result

            orchestrator._step_generate_reports = _logged_step_reports

            def _tracked_run_single(topic: Any) -> Any:
                if self._cancel_event.is_set():
                    raise InterruptedError("Pipeline cancelled by user")
                with self._lock:
                    self._status["current_topic"] = topic.slug
                    self._status["progress"] = f"Processing {topic.slug}..."
                    self._status["step_progress"] = 0
                    self._status["step_name"] = ""

                self._write_log(run_id, "topic_start", f"=== Starting topic: {topic.slug} ===", {
                    "topic": topic.slug,
                })

                # Log search window
                try:
                    ws, we = orchestrator._search_window.compute(
                        topic_slug=topic.slug,
                        date_from=run_options.get("date_from"),
                        date_to=run_options.get("date_to"),
                    )
                    # Convert to KST for display
                    from datetime import timedelta
                    kst = timedelta(hours=9)
                    self._write_log(run_id, "search_window", f"Search window: {ws.strftime('%Y-%m-%d %H:%M')} UTC ~ {we.strftime('%Y-%m-%d %H:%M')} UTC", {
                        "topic": topic.slug,
                        "window_start_utc": ws.isoformat(),
                        "window_end_utc": we.isoformat(),
                        "window_start_kst": (ws + kst).strftime("%Y-%m-%d %H:%M"),
                        "window_end_kst": (we + kst).strftime("%Y-%m-%d %H:%M"),
                    })
                except Exception:
                    pass  # Non-critical, don't break pipeline

                t0 = time.time()
                result = original_run_single(topic)
                elapsed = time.time() - t0

                with self._lock:
                    self._status["topics_completed"] += 1

                # Step 12 equivalent log
                tc = result.get("total_collected", 0)
                tf = result.get("total_filtered", 0)
                ts_val = result.get("total_scored", 0)
                to = result.get("total_output", 0)
                self._write_log(run_id, "topic_done", f"Topic {topic.slug} completed ({elapsed:.1f}s): collected={tc}, filtered={tf}, scored={ts_val}, output={to}", {
                    "topic": topic.slug,
                    "total_collected": tc,
                    "total_filtered": tf,
                    "total_scored": ts_val,
                    "total_output": to,
                    "elapsed_s": round(elapsed, 1),
                })
                return result

            orchestrator._run_single_topic = _tracked_run_single

            topic_results = orchestrator.run_all_topics()

            # 6. Post-loop processing
            with self._lock:
                self._status["progress"] = "Post-processing..."

            self._write_log(run_id, "post_loop", "Post-loop processing starting...")

            try:
                from core.pipeline.post_loop import PostLoopProcessor

                report_dir = config.output.get("report_dir", "tmp/reports")
                post_processor = PostLoopProcessor(
                    config=config, db_manager=db_manager, report_dir=report_dir,
                )
                post_processor.process(topic_results)
                self._write_log(run_id, "post_loop", "Dedup saved, last_success updated")
            except Exception:
                logger.error("Post-loop processing failed (non-fatal)", exc_info=True)
                self._write_log(run_id, "post_loop", "Post-loop processing failed (non-fatal)")

            # 7. Cleanup
            try:
                db_manager.close()
            except Exception:
                logger.warning("Failed to close database", exc_info=True)

            # 8. Set final status
            completed = topic_results.get("topics_completed", [])
            failed = topic_results.get("topics_failed", [])
            skipped = topic_results.get("topics_skipped", [])

            with self._lock:
                self._status["topics_completed"] = len(completed)
                if failed:
                    self._status["error"] = (
                        f"{len(failed)} topic(s) failed: "
                        + ", ".join(t["slug"] for t in failed)
                    )
                    self._status["progress"] = "Completed with errors"
                else:
                    self._status["progress"] = "Complete"
                    # Clean up checkpoint on fully successful completion
                    self._remove_checkpoint(run_id)
                self._status["running"] = False
                self._cancel_event.clear()

            self._write_log(run_id, "done", f"Run completed: {len(completed)} completed, {len(skipped)} skipped, {len(failed)} failed", {
                "completed": len(completed),
                "skipped": len(skipped),
                "failed": len(failed),
            })

            logger.info(
                "Pipeline run %s finished: %d completed, %d skipped, %d failed",
                run_id, len(completed), len(skipped), len(failed),
            )

        except InterruptedError as e:
            logger.info(f"Pipeline run cancelled: {e}")
            self._write_log(run_id, "cancel", f"Pipeline cancelled: {e}")
            with self._lock:
                self._status["error"] = "사용자에 의해 취소됨"
                self._status["progress"] = "Cancelled"
                self._status["running"] = False
                self._cancel_event.clear()
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            logger.debug(traceback.format_exc())
            self._write_log(run_id, "error", f"Pipeline failed: {e}", {
                "error": str(e),
            })
            with self._lock:
                self._status["error"] = str(e)
                self._status["running"] = False
                self._cancel_event.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_path(run_id: str) -> str:
        """Return the checkpoint file path for a given run_id."""
        return f"/tmp/pipeline_checkpoint_{run_id}.json"

    def _save_checkpoint(self, run_id: str, data: Dict[str, Any]) -> None:
        """Save checkpoint data to disk.

        The checkpoint is overwritten on each call (not appended).

        Args:
            run_id: Current run identifier.
            data: Checkpoint payload to persist.
        """
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        path = self._checkpoint_path(run_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to save checkpoint %s: %s", path, exc)

    @staticmethod
    def _load_checkpoint(run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if it exists.

        Returns:
            Checkpoint dict or None if not found / invalid.
        """
        path = f"/tmp/pipeline_checkpoint_{run_id}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _remove_checkpoint(run_id: str) -> None:
        """Delete checkpoint file on successful completion."""
        path = f"/tmp/pipeline_checkpoint_{run_id}.json"
        try:
            os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _compute_cache_key_simple(topic: Any) -> str:
        """Compute cache key matching KeywordExpander._compute_cache_key().

        Must stay in sync with KeywordExpander._compute_cache_key() which
        hashes description + optional fields (must_concepts_en, etc.).
        """
        import hashlib

        parts = [topic.description]
        if getattr(topic, "must_concepts_en", None):
            parts.append("|must:" + ",".join(sorted(topic.must_concepts_en)))
        if getattr(topic, "should_concepts_en", None):
            parts.append("|should:" + ",".join(sorted(topic.should_concepts_en)))
        if getattr(topic, "must_not_en", None):
            parts.append("|not:" + ",".join(sorted(topic.must_not_en)))

        raw = "".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _keyword_cache_file(self) -> Path:
        """Resolve keyword cache path relative to the configured config file."""
        return self._config_path.resolve().parent / "data" / "keyword_cache.json"

    def _legacy_keyword_cache_file(self) -> Path:
        """Legacy cache location resolved from process cwd."""
        return Path("data/keyword_cache.json").resolve()

    def _load_keyword_cache_data(self) -> dict:
        """Load cache, merging legacy path entries as a fallback."""
        merged: dict = {}
        primary = self._keyword_cache_file()
        legacy = self._legacy_keyword_cache_file()

        files = [primary]
        if legacy != primary:
            files.append(legacy)

        for idx, cache_file in enumerate(files):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                if idx == 0:
                    merged.update(data)
                else:
                    for k, v in data.items():
                        merged.setdefault(k, v)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue

        return merged
