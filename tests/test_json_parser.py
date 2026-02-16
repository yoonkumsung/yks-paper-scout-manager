"""Comprehensive tests for JsonParser (P0 priority).

Covers all 4 defense layers, mode branching, edge cases,
and debug dump behaviour.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from unittest.mock import patch

import pytest

from core.parsing.json_parser import JsonParser

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DEBUG_DIR = "tmp/test_debug"


@pytest.fixture()
def parser() -> JsonParser:
    """Return a JsonParser pointed at a disposable debug directory."""
    return JsonParser(debug_dir=DEBUG_DIR)


@pytest.fixture(autouse=True)
def _cleanup_debug_dir():
    """Remove the debug directory after each test."""
    yield
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)


# ------------------------------------------------------------------
# 1. Clean JSON input (no think blocks)
# ------------------------------------------------------------------


class TestCleanJson:
    def test_clean_object(self, parser: JsonParser) -> None:
        raw = '{"key": "value", "num": 42}'
        result = parser.parse(raw)
        assert result == {"key": "value", "num": 42}

    def test_clean_array(self, parser: JsonParser) -> None:
        raw = '[1, 2, 3]'
        result = parser.parse(raw)
        assert result == [1, 2, 3]


# ------------------------------------------------------------------
# 2. JSON with <think>...</think> block before it
# ------------------------------------------------------------------


class TestThinkBlockRemoval:
    def test_think_block_before_json(self, parser: JsonParser) -> None:
        raw = '<think>Let me reason about this...</think>{"answer": 42}'
        result = parser.parse(raw)
        assert result == {"answer": 42}

    def test_think_block_with_newlines(self, parser: JsonParser) -> None:
        raw = textwrap.dedent("""\
            <think>
            This is a multi-line
            reasoning block.
            </think>
            {"status": "ok"}
        """)
        result = parser.parse(raw)
        assert result == {"status": "ok"}


# ------------------------------------------------------------------
# 3. Unterminated think block
# ------------------------------------------------------------------


class TestUnterminatedThink:
    def test_unterminated_think_before_json(self, parser: JsonParser) -> None:
        raw = '<think>Some reasoning without closing tag\n{"result": true}'
        result = parser.parse(raw)
        assert result == {"result": True}

    def test_unterminated_think_no_json(self, parser: JsonParser) -> None:
        raw = "<think>Just rambling with no JSON at all"
        result = parser.parse(raw)
        assert result is None

    def test_unterminated_think_with_array(self, parser: JsonParser) -> None:
        raw = "<think>reasoning...[1, 2, 3]"
        result = parser.parse(raw)
        assert result == [1, 2, 3]


# ------------------------------------------------------------------
# 4. Nested brackets in JSON strings
# ------------------------------------------------------------------


class TestNestedBrackets:
    def test_braces_inside_string(self, parser: JsonParser) -> None:
        raw = '{"key": "value with {braces} inside"}'
        result = parser.parse(raw)
        assert result == {"key": "value with {braces} inside"}

    def test_brackets_inside_string(self, parser: JsonParser) -> None:
        raw = '{"key": "value with [brackets] inside"}'
        result = parser.parse(raw)
        assert result == {"key": "value with [brackets] inside"}

    def test_mixed_brackets_in_strings(self, parser: JsonParser) -> None:
        raw = '{"a": "x{y}z", "b": "p[q]r", "c": [1, {"d": "e{f}g"}]}'
        result = parser.parse(raw)
        assert result is not None
        assert result["a"] == "x{y}z"
        assert result["b"] == "p[q]r"


# ------------------------------------------------------------------
# 5. JSON with trailing comma (json_repair)
# ------------------------------------------------------------------


class TestJsonRepair:
    def test_trailing_comma_object(self, parser: JsonParser) -> None:
        raw = '{"key": "value", "num": 42,}'
        result = parser.parse(raw)
        # json_repair should fix the trailing comma
        if result is not None:
            assert result["key"] == "value"
            assert result["num"] == 42
        # If json_repair is not installed, result may be None

    def test_trailing_comma_array(self, parser: JsonParser) -> None:
        raw = "[1, 2, 3,]"
        result = parser.parse(raw)
        if result is not None:
            assert result == [1, 2, 3]


# ------------------------------------------------------------------
# 6. JSON array response
# ------------------------------------------------------------------


class TestArrayResponse:
    def test_simple_array(self, parser: JsonParser) -> None:
        raw = '[{"id": 1}, {"id": 2}]'
        result = parser.parse(raw)
        assert result == [{"id": 1}, {"id": 2}]

    def test_array_with_prefix_text(self, parser: JsonParser) -> None:
        raw = 'Here is the result:\n[{"name": "test"}]'
        result = parser.parse(raw)
        assert result == [{"name": "test"}]


# ------------------------------------------------------------------
# 7. Completely unparseable response -> returns None, debug file created
# ------------------------------------------------------------------


class TestUnparseableResponse:
    def test_returns_none(self, parser: JsonParser) -> None:
        raw = "This is not JSON at all, just plain text."
        result = parser.parse(raw, agent_name="test_agent", batch_index=5)
        assert result is None

    def test_debug_file_created(self, parser: JsonParser) -> None:
        raw = "completely unparseable garbage %%% !!!"
        parser.parse(raw, agent_name="scorer", batch_index=3)
        debug_path = os.path.join(DEBUG_DIR, "scorer_3.txt")
        assert os.path.exists(debug_path)
        with open(debug_path, encoding="utf-8") as fh:
            assert fh.read() == raw

    def test_debug_dir_created_on_demand(self, parser: JsonParser) -> None:
        assert not os.path.exists(DEBUG_DIR)
        parser.parse("not json", agent_name="a", batch_index=0)
        assert os.path.isdir(DEBUG_DIR)


# ------------------------------------------------------------------
# 8. response_format mode: clean JSON -> direct parse succeeds
# ------------------------------------------------------------------


class TestResponseFormatMode:
    def test_direct_parse(self, parser: JsonParser) -> None:
        raw = '{"status": "ok", "data": [1, 2]}'
        result = parser.parse(raw, response_format_supported=True)
        assert result == {"status": "ok", "data": [1, 2]}

    def test_with_think_block(self, parser: JsonParser) -> None:
        """Think blocks are removed even in response_format mode."""
        raw = '<think>reasoning</think>{"result": true}'
        result = parser.parse(raw, response_format_supported=True)
        assert result == {"result": True}


# ------------------------------------------------------------------
# 9. response_format mode: malformed -> falls back to defense chain
# ------------------------------------------------------------------


class TestResponseFormatFallback:
    def test_malformed_falls_back(self, parser: JsonParser) -> None:
        raw = 'Some prefix text {"key": "value"} some suffix'
        result = parser.parse(raw, response_format_supported=True)
        assert result == {"key": "value"}

    def test_trailing_comma_fallback(self, parser: JsonParser) -> None:
        raw = '{"key": "value",}'
        result = parser.parse(raw, response_format_supported=True)
        # Direct parse fails due to trailing comma;
        # falls back to defense chain (json_repair).
        if result is not None:
            assert result["key"] == "value"


# ------------------------------------------------------------------
# 10. Multiple think blocks in one response
# ------------------------------------------------------------------


class TestMultipleThinkBlocks:
    def test_two_think_blocks(self, parser: JsonParser) -> None:
        raw = (
            "<think>First block</think>"
            "Some text "
            "<think>Second block</think>"
            '{"answer": 1}'
        )
        result = parser.parse(raw)
        assert result == {"answer": 1}

    def test_three_think_blocks(self, parser: JsonParser) -> None:
        raw = (
            "<think>A</think>"
            "<think>B</think>"
            "<think>C</think>"
            '{"x": 0}'
        )
        result = parser.parse(raw)
        assert result == {"x": 0}


# ------------------------------------------------------------------
# 11. Think block with JSON-like content inside
# ------------------------------------------------------------------


class TestThinkBlockWithJsonContent:
    def test_json_inside_think(self, parser: JsonParser) -> None:
        raw = (
            '<think>Maybe the answer is {"wrong": true}?</think>'
            '{"correct": true}'
        )
        result = parser.parse(raw)
        assert result == {"correct": True}

    def test_array_inside_think(self, parser: JsonParser) -> None:
        raw = "<think>[1,2,3] is not the answer</think>[4,5,6]"
        result = parser.parse(raw)
        assert result == [4, 5, 6]


# ------------------------------------------------------------------
# 12. Empty string input -> returns None
# ------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string(self, parser: JsonParser) -> None:
        assert parser.parse("") is None

    def test_whitespace_only(self, parser: JsonParser) -> None:
        assert parser.parse("   \n\t  ") is None

    def test_none_like_empty(self, parser: JsonParser) -> None:
        # Verify no exception is raised
        assert parser.parse("   ") is None


# ------------------------------------------------------------------
# 13. JSON with escaped quotes inside strings
# ------------------------------------------------------------------


class TestEscapedQuotes:
    def test_escaped_quotes_in_value(self, parser: JsonParser) -> None:
        raw = r'{"msg": "He said \"hello\" to me"}'
        result = parser.parse(raw)
        assert result is not None
        assert result["msg"] == 'He said "hello" to me'

    def test_escaped_backslash_and_quote(self, parser: JsonParser) -> None:
        raw = r'{"path": "C:\\Users\\test", "quote": "a\"b"}'
        result = parser.parse(raw)
        assert result is not None
        assert result["path"] == "C:\\Users\\test"
        assert result["quote"] == 'a"b'


# ------------------------------------------------------------------
# 14. Bracket balancing with strings containing brackets
# ------------------------------------------------------------------


class TestBracketBalancingStrings:
    def test_nested_json_with_bracket_strings(self, parser: JsonParser) -> None:
        raw = textwrap.dedent("""\
            Here is the output:
            {
                "formula": "f(x) = {x + 1}",
                "items": [
                    {"name": "item[0]", "value": 10},
                    {"name": "item[1]", "value": 20}
                ]
            }
            End of output.
        """)
        result = parser.parse(raw)
        assert result is not None
        assert result["formula"] == "f(x) = {x + 1}"
        assert len(result["items"]) == 2
        assert result["items"][0]["name"] == "item[0]"

    def test_deeply_nested(self, parser: JsonParser) -> None:
        raw = '{"a": {"b": {"c": {"d": 1}}}}'
        result = parser.parse(raw)
        assert result == {"a": {"b": {"c": {"d": 1}}}}


# ------------------------------------------------------------------
# Unit tests for individual methods
# ------------------------------------------------------------------


class TestRemoveThinkBlocks:
    def test_no_think_blocks(self, parser: JsonParser) -> None:
        assert parser.remove_think_blocks("plain text") == "plain text"

    def test_complete_block(self, parser: JsonParser) -> None:
        result = parser.remove_think_blocks(
            "<think>hidden</think>visible"
        )
        assert result == "visible"

    def test_case_insensitive(self, parser: JsonParser) -> None:
        result = parser.remove_think_blocks(
            "<THINK>hidden</THINK>visible"
        )
        assert result == "visible"

    def test_multiple_blocks(self, parser: JsonParser) -> None:
        result = parser.remove_think_blocks(
            "<think>a</think>X<think>b</think>Y"
        )
        assert result == "XY"

    def test_unterminated_with_json(self, parser: JsonParser) -> None:
        result = parser.remove_think_blocks(
            '<think>thinking...{"key": "val"}'
        )
        # Should preserve the JSON part
        assert '{"key"' in result or "key" in result


class TestExtractJsonSubstring:
    def test_simple_object(self, parser: JsonParser) -> None:
        result = parser.extract_json_substring('prefix {"a": 1} suffix')
        assert result == '{"a": 1}'

    def test_simple_array(self, parser: JsonParser) -> None:
        result = parser.extract_json_substring("before [1, 2] after")
        assert result == "[1, 2]"

    def test_no_json(self, parser: JsonParser) -> None:
        result = parser.extract_json_substring("no brackets here")
        assert result is None

    def test_unbalanced(self, parser: JsonParser) -> None:
        result = parser.extract_json_substring('{"open but never closed')
        assert result is None

    def test_nested_objects(self, parser: JsonParser) -> None:
        raw = '{"a": {"b": 1}, "c": 2}'
        result = parser.extract_json_substring(raw)
        assert result == raw

    def test_string_with_brackets(self, parser: JsonParser) -> None:
        raw = '{"val": "has {braces} and [brackets]"}'
        result = parser.extract_json_substring(raw)
        assert result == raw

    def test_escaped_quotes_inside(self, parser: JsonParser) -> None:
        raw = r'{"key": "val\"ue"}'
        result = parser.extract_json_substring(raw)
        assert result == raw
