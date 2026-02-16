"""Pipeline runner for local UI background execution.

Manages background pipeline execution for the web UI including both
dry-run (keyword expansion + query building) and full pipeline runs.

Reference: TASK-052 devspec Section 18-3.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.keyword_expander import KeywordExpander
from core.config import load_config
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
        self._status: Dict[str, Any] = {
            "running": False,
            "run_id": None,
            "progress": "",
            "topics_completed": 0,
            "topics_total": 0,
            "current_topic": None,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_dryrun(self, topic_slug: Optional[str] = None) -> Dict[str, Any]:
        """Run dry-run synchronously (fast operation).

        Executes Agent 1 (keyword expansion) + QueryBuilder for selected topics.
        Does NOT run scoring/summarization - just keywords and queries.

        Args:
            topic_slug: Specific topic slug to run, or None for all topics

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
            keyword_expander = KeywordExpander(config=config)
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
                        cache_file = Path("data/keyword_cache.json")
                        if cache_file.exists():
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cache_data = json.load(f)
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
                        agent1_output = keyword_expander.expand(topic)

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
                            "exclude_keywords": agent1_output.get("exclude_keywords", []),
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
            self._status = {
                "running": True,
                "run_id": run_id,
                "progress": "Starting...",
                "topics_completed": 0,
                "topics_total": 0,
                "current_topic": None,
                "error": None,
            }

        # Start background thread
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(run_id, topic_slug, date_from, date_to, dedup),
            daemon=True,
        )
        thread.start()

        return {"run_id": run_id, "status": "started"}

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

    def _run_pipeline_thread(
        self,
        run_id: str,
        topic_slug: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        dedup: str,
    ) -> None:
        """Background thread for full pipeline execution.

        This would execute the complete pipeline including:
        1. Keyword expansion
        2. Paper collection
        3. Scoring
        4. Summarization
        5. Report generation

        For now, this is a placeholder that simulates progress.
        """
        try:
            # Load config
            config = load_config(str(self._config_path))

            # Determine topics
            if topic_slug:
                topics = [t for t in config.topics if t.slug == topic_slug]
            else:
                topics = config.topics

            with self._lock:
                self._status["topics_total"] = len(topics)
                self._status["progress"] = "Collecting papers..."

            # TODO: Implement full pipeline execution
            # This would involve:
            # 1. Create TopicLoopOrchestrator with all dependencies
            # 2. Configure run_options with date_from, date_to, dedup
            # 3. Execute run_all_topics()
            # 4. Update status during execution
            # 5. Handle errors gracefully

            # For now, simulate progress
            import time

            for i, topic in enumerate(topics):
                with self._lock:
                    self._status["current_topic"] = topic.slug
                    self._status["progress"] = f"Processing {topic.slug}..."
                time.sleep(2)  # Simulate work

                with self._lock:
                    self._status["topics_completed"] = i + 1

            with self._lock:
                self._status["progress"] = "Complete"
                self._status["running"] = False

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            logger.debug(traceback.format_exc())
            with self._lock:
                self._status["error"] = str(e)
                self._status["running"] = False

    @staticmethod
    def _compute_cache_key_simple(topic: Any) -> str:
        """Compute simple cache key for topic.

        This is a simplified version - the actual KeywordExpander
        uses a more complex hash including optional fields.
        """
        import hashlib

        # Use topic slug as simple key
        return hashlib.sha256(topic.slug.encode("utf-8")).hexdigest()
