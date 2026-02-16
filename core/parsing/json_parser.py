"""JSON parser for LLM responses with 4-layer defense chain.

Handles malformed JSON, think blocks, and other artifacts commonly
found in LLM outputs. Implements two modes:
  - response_format mode: direct parse with fallback
  - parsing defense mode: full 4-layer chain

Layers:
  1. Think block removal (always applied)
  2. Bracket balancing extraction
  3. Standard json.loads
  4. json_repair fallback
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


class JsonParser:
    """Parse LLM responses into structured JSON data.

    Implements a multi-layer defense chain to handle the variety of
    malformed outputs that LLMs can produce.
    """

    def __init__(self, debug_dir: str = "tmp/debug") -> None:
        self._debug_dir = debug_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        raw_response: str,
        response_format_supported: bool = False,
        agent_name: str = "unknown",
        batch_index: int = 0,
    ) -> dict | list | None:
        """Parse an LLM response string into a Python dict or list.

        Args:
            raw_response: The raw text returned by the LLM.
            response_format_supported: Whether the model was called with
                ``response_format`` (i.e. it should already return valid
                JSON, but think blocks may still be present).
            agent_name: Identifier for the calling agent (used in debug
                dump filenames).
            batch_index: Index of the batch being processed (used in
                debug dump filenames).

        Returns:
            Parsed JSON as ``dict`` or ``list``, or ``None`` when every
            parsing attempt fails.  On failure the raw response is saved
            to ``{debug_dir}/{agent_name}_{batch_index}.txt``.
        """
        if not raw_response or not raw_response.strip():
            return None

        # Layer 1 -- always strip think blocks
        cleaned = self.remove_think_blocks(raw_response)

        if response_format_supported:
            # Fast path: model was instructed to return JSON via
            # response_format.  Try a direct parse first.
            result = self._try_json_loads(cleaned.strip())
            if result is not None:
                return result
            # Fall through to full defense chain if direct parse fails.

        # Layer 2 -- bracket balancing extraction
        json_str = self.extract_json_substring(cleaned)

        if json_str is not None:
            # Layer 3 -- standard json.loads
            result = self._try_json_loads(json_str)
            if result is not None:
                return result

            # Layer 4 -- json_repair fallback
            result = self._try_json_repair(json_str)
            if result is not None:
                return result

        # All layers failed -- dump raw response for debugging
        self._save_debug_dump(raw_response, agent_name, batch_index)
        return None

    # ------------------------------------------------------------------
    # Layer 1: Think block removal
    # ------------------------------------------------------------------

    def remove_think_blocks(self, text: str) -> str:
        """Remove ``<think>...</think>`` blocks from *text*.

        Handles:
        - Complete ``<think>...</think>`` pairs (including nested content)
        - Multiple think blocks in a single string
        - Unterminated ``<think>`` tags (opening without closing) --
          removes from the opening tag to either the start of actual JSON
          content or the end of the string.
        """
        # Pass 1: remove all well-formed <think>...</think> blocks.
        # Use DOTALL so '.' matches newlines inside the block.
        result = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Pass 2: handle unterminated <think> tags.
        # If an opening <think> remains, remove from it to either:
        #   (a) the start of a JSON object/array, or
        #   (b) the end of the string.
        if re.search(r"<think>", result, flags=re.IGNORECASE):
            # Try to find JSON content after the unterminated tag
            match = re.search(
                r"<think>.*?(?=[\[{])",
                result,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                result = result[: match.start()] + result[match.end() :]
            else:
                # No JSON content found -- remove from <think> to end
                result = re.sub(
                    r"<think>.*$",
                    "",
                    result,
                    flags=re.DOTALL | re.IGNORECASE,
                )

        return result.strip()

    # ------------------------------------------------------------------
    # Layer 2: Bracket balancing extraction
    # ------------------------------------------------------------------

    def extract_json_substring(self, text: str) -> str | None:
        """Extract the outermost balanced JSON object or array.

        Scans *text* for the first ``{`` or ``[`` and returns the
        substring up to the matching closing bracket.  Brackets inside
        JSON strings (delimited by ``"``) are correctly ignored.

        Returns ``None`` if no balanced JSON structure is found.
        """
        # Find the first opening bracket
        start_idx: int | None = None
        open_char: str | None = None
        close_char: str | None = None

        for i, ch in enumerate(text):
            if ch in ("{", "["):
                start_idx = i
                open_char = ch
                close_char = "}" if ch == "{" else "]"
                break

        if start_idx is None:
            return None

        # Track depth using a stack so nested mixed brackets work.
        stack: list[str] = []
        in_string = False
        escape_next = False

        _OPEN_TO_CLOSE = {"{": "}", "[": "]"}
        _CLOSE_SET = {"}", "]"}

        for i in range(start_idx, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\" and in_string:
                escape_next = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch in _OPEN_TO_CLOSE:
                stack.append(_OPEN_TO_CLOSE[ch])
            elif ch in _CLOSE_SET:
                if not stack or stack[-1] != ch:
                    # Mismatched bracket -- bail out
                    return None
                stack.pop()
                if not stack:
                    return text[start_idx : i + 1]

        # Unbalanced -- return None
        return None

    # ------------------------------------------------------------------
    # Layer 3: Standard json.loads
    # ------------------------------------------------------------------

    @staticmethod
    def _try_json_loads(text: str) -> dict | list | None:
        """Attempt ``json.loads`` and return result or ``None``."""
        try:
            result = json.loads(text)
            if isinstance(result, (dict, list)):
                return result
            return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Layer 4: json_repair fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _try_json_repair(text: str) -> dict | list | None:
        """Attempt repair via ``json_repair`` library, then parse."""
        try:
            from json_repair import repair_json  # type: ignore[import-untyped]

            repaired = repair_json(text, return_objects=False)
            result = json.loads(repaired)
            if isinstance(result, (dict, list)):
                return result
            return None
        except Exception:  # noqa: BLE001 -- catch-all for repair failures
            return None

    # ------------------------------------------------------------------
    # Debug dump
    # ------------------------------------------------------------------

    def _save_debug_dump(
        self, raw: str, agent_name: str, batch_index: int
    ) -> None:
        """Persist raw LLM response for post-mortem debugging."""
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            filename = f"{agent_name}_{batch_index}.txt"
            filepath = os.path.join(self._debug_dir, filename)
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(raw)
            logger.warning(
                "All JSON parsing failed for %s batch %d; "
                "raw response saved to %s",
                agent_name,
                batch_index,
                filepath,
            )
        except OSError:
            logger.exception(
                "Failed to write debug dump for %s batch %d",
                agent_name,
                batch_index,
            )
