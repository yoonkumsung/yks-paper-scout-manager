"""Agent 2: Paper Scorer for Paper Scout.

Evaluates papers in batches and assigns base_score + boolean flags.
Results are used downstream by the Ranker to compute final_score.

Section 7-2 of the devspec covers the full specification.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from agents.base_agent import BaseAgent
from core.llm.rate_limiter import RateLimiter
from core.models import Paper

logger = logging.getLogger(__name__)

# System prompt for the scorer agent (devspec 7-2)
_SYSTEM_PROMPT = (
    "You are a paper evaluator. Output base_score and boolean flags.\n"
    "Do NOT apply bonuses. Output ONLY raw JSON."
)


class Scorer(BaseAgent):
    """Agent 2: Paper scoring and evaluation.

    Flow (``score``):
      1. Split papers into batches of ``batch_size`` (default 10).
      2. For each batch, build messages and call LLM.
      3. Parse JSON array response.
      4. If output count != input count, re-call once for missing items.
      5. On 2x parse failure, skip the batch and log.
      6. Clamp base_score to 0-100.
      7. Apply discard logic: discard=True if is_metaphorical or base_score < 20.
      8. Merge has_code: paper.has_code OR mentions_code flag.
      9. Collect and return all evaluation dicts.
    """

    # ------------------------------------------------------------------
    # Abstract property overrides
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        return "scorer"

    @property
    def agent_config_key(self) -> str:
        return "scorer"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        papers: list[Paper],
        topic_description: str,
        rate_limiter: RateLimiter,
    ) -> list[dict]:
        """Score papers in batches.

        Args:
            papers: List of Paper objects to evaluate.
            topic_description: Korean project description for the prompt.
            rate_limiter: RateLimiter instance for API throttling.

        Returns:
            List of evaluation dicts, one per successfully scored paper:
            {
                "paper_key": str,
                "base_score": int,      # 0-100, clamped
                "flags": {
                    "is_edge": bool,
                    "is_realtime": bool,
                    "mentions_code": bool,
                    "is_metaphorical": bool,
                },
                "discard": bool,        # True if is_metaphorical or base_score < 20
                "brief_reason": str,    # Korean 1-sentence
            }
        """
        if not papers:
            return []

        batch_size = self.agent_config.get("batch_size", 10)
        if not batch_size or batch_size <= 0:
            batch_size = 5
        try:
            batch_timeout = float(self.agent_config.get("batch_timeout_seconds", 500))
        except (TypeError, ValueError):
            batch_timeout = 500.0
        all_results: list[dict] = []

        # Sticky fallback: once triggered, all subsequent batches use
        # the fallback model to avoid repeating slow primary calls.
        active_model_override: str | None = None

        # Split into batches
        batches = [
            papers[i : i + batch_size]
            for i in range(0, len(papers), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            # Check daily API limit before each batch
            if rate_limiter.is_daily_limit_reached:
                logger.warning(
                    "scorer: daily API limit reached at batch %d/%d, "
                    "returning %d partial results",
                    batch_idx, len(batches), len(all_results),
                )
                break

            try:
                t0 = time.monotonic()
                batch_results = self._score_batch(
                    batch, topic_description, rate_limiter, batch_idx,
                    model_override=active_model_override,
                )
                elapsed = time.monotonic() - t0

                # Batch timeout check: if too slow or failed, switch model
                is_slow = elapsed > batch_timeout
                is_failed = batch_results is None or len(batch_results) == 0

                if (is_slow or is_failed) and active_model_override is None:
                    fallbacks = self.fallback_models
                    if fallbacks:
                        active_model_override = fallbacks[0]
                        logger.warning(
                            "scorer: batch %d/%d was %s "
                            "(%.1fs, timeout=%.0fs). "
                            "Switching to fallback model: %s",
                            batch_idx + 1, len(batches),
                            "slow" if is_slow else "failed",
                            elapsed, batch_timeout,
                            active_model_override,
                        )
                        # Retry this batch with fallback if it failed
                        if is_failed:
                            batch_results = self._score_batch(
                                batch, topic_description, rate_limiter,
                                batch_idx,
                                model_override=active_model_override,
                            )

                if batch_results is not None:
                    all_results.extend(batch_results)
            except InterruptedError:
                raise  # Re-raise cancel events
            except Exception as exc:
                logger.error(
                    "scorer: batch %d/%d failed unexpectedly: %s. "
                    "Returning %d partial results.",
                    batch_idx + 1, len(batches), exc, len(all_results),
                )
                break

        return all_results

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Build system + user messages for the LLM call.

        Expects keyword arguments:
            papers: list[Paper] -- papers to score in this batch.
            topic_description: str -- project description.
            indices: list[int] -- 1-based indices for the papers.
        """
        papers: list[Paper] = kwargs["papers"]
        topic_description: str = kwargs["topic_description"]
        indices: list[int] = kwargs.get("indices", list(range(1, len(papers) + 1)))

        # Build paper list section
        paper_lines: list[str] = []
        for idx, paper in zip(indices, papers):
            paper_lines.append(
                f"{idx}. [{paper.title}] {paper.abstract}"
            )

        user_content = (
            f"## Project\n{topic_description}\n\n"
            "## Scoring (base_score, pure relevance only, no bonuses)\n"
            "- 90~100: Core technology papers directly related\n"
            "- 70~89: Applicable technology/methodology included\n"
            "- 50~69: Indirectly related/reference\n"
            "- 20~49: Weakly related (discard: false, score only)\n"
            "- Below 20: Unrelated (discard: true)\n\n"
            "## flags (fact check only, do NOT reflect in score)\n"
            "- is_edge: lightweight model runnable on edge/mobile devices\n"
            "- is_realtime: real-time or near-realtime processing capable\n"
            "- mentions_code: mentions code release/github link in abstract\n"
            '- is_metaphorical: "sport"/"game" etc. used metaphorically (true -> discard)\n\n'
            "## Paper List\n"
            + "\n".join(paper_lines)
            + "\n\n## Output\n"
            '[{"index":1,"base_score":82,"flags":{...},"discard":false,"brief_reason":"Korean 1-sentence"}]'
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Internal batch processing
    # ------------------------------------------------------------------

    def _score_batch(
        self,
        batch: list[Paper],
        topic_description: str,
        rate_limiter: RateLimiter,
        batch_idx: int,
        model_override: str | None = None,
    ) -> list[dict] | None:
        """Score a single batch of papers.

        Returns list of evaluation dicts, or None if the batch was skipped
        (2x parse failure).

        Args:
            batch: Papers to score.
            topic_description: Topic description for the prompt.
            rate_limiter: Rate limiter instance.
            batch_idx: Batch index for logging.
            model_override: Override model for this batch (fallback).
        """
        indices = list(range(1, len(batch) + 1))
        messages = self.build_messages(
            papers=batch,
            topic_description=topic_description,
            indices=indices,
        )

        # --- First LLM call ---
        rate_limiter.wait()
        raw = self.call_llm(
            messages, batch_index=batch_idx, model_override=model_override,
        )
        rate_limiter.record_call()

        if raw is None or not isinstance(raw, list):
            # First parse failure -- retry once
            logger.warning(
                "scorer: parse failure 1/2 for batch %d", batch_idx
            )
            rate_limiter.wait()
            raw = self.call_llm(
                messages, batch_index=batch_idx,
                model_override=model_override,
            )
            rate_limiter.record_call()

            if raw is None or not isinstance(raw, list):
                logger.warning(
                    "scorer: parse failure 2/2 for batch %d -- skipping",
                    batch_idx,
                )
                return None

        # --- Check for missing items ---
        received_indices: set[int] = set()
        for item in raw:
            if isinstance(item, dict):
                raw_idx = item.get("index")
                if raw_idx is not None:
                    try:
                        received_indices.add(int(raw_idx))
                    except (TypeError, ValueError):
                        pass
        expected_indices = set(indices)
        missing_indices = expected_indices - received_indices

        if missing_indices and len(raw) < len(batch):
            # Re-call once for missing items
            missing_papers = [
                batch[idx - 1] for idx in sorted(missing_indices)
            ]
            missing_idx_list = sorted(missing_indices)
            retry_messages = self.build_messages(
                papers=missing_papers,
                topic_description=topic_description,
                indices=missing_idx_list,
            )

            rate_limiter.wait()
            retry_raw = self.call_llm(
                retry_messages, batch_index=batch_idx,
                model_override=model_override,
            )
            rate_limiter.record_call()

            if retry_raw is not None and isinstance(retry_raw, list):
                raw.extend(retry_raw)

        # --- Process results ---
        return self._process_batch_results(raw, batch)

    def _process_batch_results(
        self,
        raw_items: list[dict],
        batch: list[Paper],
    ) -> list[dict]:
        """Process raw LLM items into evaluation dicts.

        Applies:
        - base_score clamping to 0-100
        - Discard logic (is_metaphorical or base_score < 20)
        - has_code merge (paper.has_code OR mentions_code)
        """
        # Build index -> paper mapping (1-based)
        paper_by_index: dict[int, Paper] = {
            i + 1: paper for i, paper in enumerate(batch)
        }

        results: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            raw_index = item.get("index")
            if raw_index is None:
                continue
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index not in paper_by_index:
                continue

            paper = paper_by_index[index]

            # Extract and clamp base_score
            raw_score = item.get("base_score", 0)
            try:
                base_score = int(raw_score)
            except (TypeError, ValueError):
                base_score = 0
            base_score = max(0, min(100, base_score))

            # Extract flags with defaults
            raw_flags = item.get("flags", {})
            if not isinstance(raw_flags, dict):
                raw_flags = {}

            flags = {
                "is_edge": bool(raw_flags.get("is_edge", False)),
                "is_realtime": bool(raw_flags.get("is_realtime", False)),
                "mentions_code": bool(raw_flags.get("mentions_code", False)),
                "is_metaphorical": bool(raw_flags.get("is_metaphorical", False)),
            }

            # Discard logic: is_metaphorical OR base_score < 20
            discard = flags["is_metaphorical"] or base_score < 20

            # brief_reason
            brief_reason = str(item.get("brief_reason", ""))

            # has_code merge: paper.has_code OR mentions_code
            merged_has_code = paper.has_code or flags["mentions_code"]

            results.append(
                {
                    "paper_key": paper.paper_key,
                    "base_score": base_score,
                    "flags": flags,
                    "discard": discard,
                    "brief_reason": brief_reason,
                    "has_code": merged_has_code,
                }
            )

        return results
