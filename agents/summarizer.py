"""Agent 3: Korean Summary Generator for Paper Scout.

Generates Korean summaries for ranked papers in two tiers:
  - Tier 1 (rank 1-30): detailed summaries with insight
  - Tier 2 (rank 31-100): compact summaries without insight

Section 7-3 of the devspec covers the full specification.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from agents.base_agent import BaseAgent
from core.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def _strip_cjk_noise(text: str) -> str:
    """Remove stray Chinese characters from Korean text.

    DeepSeek and similar Chinese-trained LLMs sometimes inject
    CJK Unified Ideographs (U+4E00-U+9FFF) into Korean output.
    Replaces Chinese character runs with a space to avoid
    words being merged unnaturally after removal.
    """
    cleaned = re.sub(r"[\u4e00-\u9fff]+", " ", text)
    # Collapse multiple spaces into one
    return re.sub(r" {2,}", " ", cleaned).strip()

# Tier boundaries
_TIER1_MAX_RANK = 30

# Default batch sizes
_TIER1_BATCH_SIZE = 5
_TIER2_BATCH_SIZE = 10

# System prompt for the summarizer agent (devspec 7-3)
_SYSTEM_PROMPT = (
    "You are a technical writer for a startup CEO. Write in simple Korean.\n"
    "IMPORTANT: Write ALL text in Korean only. Do NOT use Chinese characters (汉字/漢字). "
    "Proper nouns and technical terms may use English.\n"
    "No thinking, no analysis preamble. Output ONLY raw JSON."
)


class Summarizer(BaseAgent):
    """Agent 3: Korean summary generation for ranked papers.

    Flow (``summarize``):
      1. Split papers into Tier 1 (rank 1-30) and Tier 2 (rank 31-100).
      2. Tier 1: batch of 5, output summary_ko (300-500 chars),
         reason_ko (~150 chars), insight_ko (~150 chars).
      3. Tier 2: batch of 10, output summary_ko (~200 chars),
         reason_ko (1 line), no insight_ko.
      4. Rate limiting: wait() before each call, record_call() after.
      5. Parse failure: 2x failure -> skip batch.
      6. Missing items: re-call once for missing indices.
      7. Return enriched paper dicts with summary fields added.
    """

    # ------------------------------------------------------------------
    # Abstract property overrides
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        return "summarizer"

    @property
    def agent_config_key(self) -> str:
        return "agent3"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        papers: list[dict],
        topic_description: str,
        rate_limiter: RateLimiter,
    ) -> list[dict]:
        """Generate Korean summaries for papers.

        Tier 1 (rank 1-30): 5 papers/batch
          Output: summary_ko (300-500 chars), reason_ko (~150 chars),
                  insight_ko (~150 chars)

        Tier 2 (rank 31-100): 10 papers/batch
          Output: summary_ko (~200 chars), reason_ko (1 line),
                  no insight_ko

        Args:
            papers: Ranked papers with evaluations.  Each dict must
                have at least ``rank``, ``title``, ``abstract``,
                ``base_score``, and ``brief_reason`` keys.
            topic_description: Korean project description for the prompt.
            rate_limiter: RateLimiter instance for API throttling.

        Returns:
            Enriched paper dicts with summary fields added.
        """
        if not papers:
            return []

        batch_timeout = float(self.agent_config.get("batch_timeout_seconds", 500))

        # Sticky fallback: once triggered, all subsequent batches use
        # the fallback model to avoid repeating slow primary calls.
        active_model_override: str | None = None

        # Split into tiers
        tier1 = [p for p in papers if p.get("rank", 0) <= _TIER1_MAX_RANK]
        tier2 = [p for p in papers if p.get("rank", 0) > _TIER1_MAX_RANK]

        results: list[dict] = []

        # Process Tier 1 batches (batch_size=5)
        tier1_batch_size = self.agent_config.get(
            "tier1_batch_size", _TIER1_BATCH_SIZE
        )
        tier1_batches = [
            tier1[i : i + tier1_batch_size]
            for i in range(0, len(tier1), tier1_batch_size)
        ]
        for batch_idx, batch in enumerate(tier1_batches):
            # Check daily API limit before each batch
            if rate_limiter.is_daily_limit_reached:
                logger.warning(
                    "summarizer: daily API limit reached at tier1 batch %d/%d, "
                    "returning %d partial results",
                    batch_idx, len(tier1_batches), len(results),
                )
                return results

            try:
                t0 = time.monotonic()
                batch_results = self._summarize_batch(
                    batch, topic_description, rate_limiter, batch_idx, tier=1,
                    model_override=active_model_override,
                )
                elapsed = time.monotonic() - t0

                is_slow = elapsed > batch_timeout
                is_failed = batch_results is None or len(batch_results) == 0

                if (is_slow or is_failed) and active_model_override is None:
                    fallbacks = self.fallback_models
                    if fallbacks:
                        active_model_override = fallbacks[0]
                        logger.warning(
                            "summarizer: tier1 batch %d/%d was %s "
                            "(%.1fs, timeout=%.0fs). "
                            "Switching to fallback model: %s",
                            batch_idx + 1, len(tier1_batches),
                            "slow" if is_slow else "failed",
                            elapsed, batch_timeout,
                            active_model_override,
                        )
                        if is_failed:
                            batch_results = self._summarize_batch(
                                batch, topic_description, rate_limiter,
                                batch_idx, tier=1,
                                model_override=active_model_override,
                            )

                if batch_results is not None:
                    results.extend(batch_results)
            except InterruptedError:
                raise  # Re-raise cancel events
            except Exception as exc:
                logger.error(
                    "summarizer: tier1 batch %d/%d failed unexpectedly: %s. "
                    "Returning %d partial results.",
                    batch_idx + 1, len(tier1_batches), exc, len(results),
                )
                return results

        # Process Tier 2 batches (batch_size=10)
        tier2_batch_size = self.agent_config.get(
            "tier2_batch_size", _TIER2_BATCH_SIZE
        )
        tier2_batches = [
            tier2[i : i + tier2_batch_size]
            for i in range(0, len(tier2), tier2_batch_size)
        ]
        tier2_batch_offset = len(tier1_batches)
        for batch_idx, batch in enumerate(tier2_batches):
            # Check daily API limit before each batch
            if rate_limiter.is_daily_limit_reached:
                logger.warning(
                    "summarizer: daily API limit reached at tier2 batch %d/%d, "
                    "returning %d partial results",
                    batch_idx, len(tier2_batches), len(results),
                )
                return results

            try:
                t0 = time.monotonic()
                batch_results = self._summarize_batch(
                    batch,
                    topic_description,
                    rate_limiter,
                    tier2_batch_offset + batch_idx,
                    tier=2,
                    model_override=active_model_override,
                )
                elapsed = time.monotonic() - t0

                is_slow = elapsed > batch_timeout
                is_failed = batch_results is None or len(batch_results) == 0

                if (is_slow or is_failed) and active_model_override is None:
                    fallbacks = self.fallback_models
                    if fallbacks:
                        active_model_override = fallbacks[0]
                        logger.warning(
                            "summarizer: tier2 batch %d/%d was %s "
                            "(%.1fs, timeout=%.0fs). "
                            "Switching to fallback model: %s",
                            batch_idx + 1, len(tier2_batches),
                            "slow" if is_slow else "failed",
                            elapsed, batch_timeout,
                            active_model_override,
                        )
                        if is_failed:
                            batch_results = self._summarize_batch(
                                batch, topic_description, rate_limiter,
                                tier2_batch_offset + batch_idx, tier=2,
                                model_override=active_model_override,
                            )

                if batch_results is not None:
                    results.extend(batch_results)
            except InterruptedError:
                raise  # Re-raise cancel events
            except Exception as exc:
                logger.error(
                    "summarizer: tier2 batch %d/%d failed unexpectedly: %s. "
                    "Returning %d partial results.",
                    batch_idx + 1, len(tier2_batches), exc, len(results),
                )
                return results

        return results

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Build system + user messages for the LLM call.

        Expects keyword arguments:
            papers: list[dict] -- papers to summarize in this batch.
            topic_description: str -- project description.
            indices: list[int] -- 1-based indices for the papers.
            tier: int -- 1 or 2.
        """
        papers: list[dict] = kwargs["papers"]
        topic_description: str = kwargs["topic_description"]
        indices: list[int] = kwargs.get(
            "indices", list(range(1, len(papers) + 1))
        )
        tier: int = kwargs.get("tier", 1)

        # Build paper list section
        paper_lines: list[str] = []
        for idx, paper in zip(indices, papers):
            paper_lines.append(
                f"{idx}. [{paper.get('title', '')}] "
                f"{paper.get('abstract', '')} "
                f"(base_score: {paper.get('base_score', 0)}, "
                f"brief_reason: {paper.get('brief_reason', '')})"
            )

        # Writing rules
        writing_rules = (
            "## Writing Rules\n"
            '1. "This paper proposes a method for ~. The core is ~." format\n'
            "2. Problem -> Solution -> Result order\n"
            "3. Must be able to judge applicability within 5 seconds\n"
            "4. Selection reason: why important for our project\n"
            "5. Application insight: specific application method\n"
            "6. Allowed numerics: fps, latency, inference speed, parameter count\n"
            "7. Forbidden: mAP, BLEU, ROUGE etc. academic benchmark numbers\n"
        )

        # Output format depends on tier
        if tier == 1:
            output_format = (
                "## Output\n"
                '[{"index":1,'
                '"summary_ko":"300~500chars",'
                '"reason_ko":"~150chars",'
                '"insight_ko":"~150chars"}]'
            )
        else:
            output_format = (
                "## Output\n"
                '[{"index":1,'
                '"summary_ko":"~200chars",'
                '"reason_ko":"1 line"}]'
            )

        user_content = (
            f"## Project Context\n{topic_description}\n\n"
            f"{writing_rules}\n"
            f"## Paper List\n"
            + "\n".join(paper_lines)
            + f"\n\n{output_format}"
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Internal batch processing
    # ------------------------------------------------------------------

    def _summarize_batch(
        self,
        batch: list[dict],
        topic_description: str,
        rate_limiter: RateLimiter,
        batch_idx: int,
        tier: int,
        model_override: str | None = None,
    ) -> list[dict] | None:
        """Summarize a single batch of papers.

        Returns list of enriched paper dicts, or None if the batch was
        skipped (2x parse failure).

        Args:
            batch: Papers to summarize.
            topic_description: Topic description for the prompt.
            rate_limiter: Rate limiter instance.
            batch_idx: Batch index for logging.
            tier: Summary tier (1 or 2).
            model_override: Override model for this batch (fallback).
        """
        indices = list(range(1, len(batch) + 1))
        messages = self.build_messages(
            papers=batch,
            topic_description=topic_description,
            indices=indices,
            tier=tier,
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
                "summarizer: parse failure 1/2 for batch %d (tier %d)",
                batch_idx,
                tier,
            )
            rate_limiter.wait()
            raw = self.call_llm(
                messages, batch_index=batch_idx,
                model_override=model_override,
            )
            rate_limiter.record_call()

            if raw is None or not isinstance(raw, list):
                logger.warning(
                    "summarizer: parse failure 2/2 for batch %d (tier %d)"
                    " -- skipping",
                    batch_idx,
                    tier,
                )
                return None

        # --- Check for missing items ---
        received_indices = {
            item.get("index") for item in raw if isinstance(item, dict)
        }
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
                tier=tier,
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
        return self._process_batch_results(raw, batch, tier)

    def _process_batch_results(
        self,
        raw_items: list[dict],
        batch: list[dict],
        tier: int,
    ) -> list[dict]:
        """Process raw LLM items and merge summaries into paper dicts.

        Tier 1 output: summary_ko, reason_ko, insight_ko
        Tier 2 output: summary_ko, reason_ko (no insight_ko)
        """
        # Build index -> paper mapping (1-based)
        paper_by_index: dict[int, dict] = {
            i + 1: paper for i, paper in enumerate(batch)
        }

        results: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            index = item.get("index")
            if index is None or index not in paper_by_index:
                continue

            paper = paper_by_index[index]

            # Create enriched copy with summary fields
            enriched = dict(paper)
            enriched["summary_ko"] = _strip_cjk_noise(str(item.get("summary_ko", "")))
            enriched["reason_ko"] = _strip_cjk_noise(str(item.get("reason_ko", "")))

            if tier == 1:
                enriched["insight_ko"] = _strip_cjk_noise(str(item.get("insight_ko", "")))
            # Tier 2: no insight_ko field

            enriched["prompt_ver_summ"] = self.prompt_version

            results.append(enriched)

        return results
