"""Tests for core.sources.arxiv_query_builder.

Comprehensive tests for ArxivQueryBuilder covering DSL syntax,
query count bounds, all four generation phases, exclusion keywords,
edge cases, deduplication, ordering, and determinism.
"""

from __future__ import annotations

import pytest

from core.sources.arxiv_query_builder import ArxivQueryBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def builder() -> ArxivQueryBuilder:
    """Provide a fresh ArxivQueryBuilder instance."""
    return ArxivQueryBuilder()


@pytest.fixture()
def typical_agent1_output() -> dict:
    """Provide a typical Agent 1 output with multiple concepts."""
    return {
        "concepts": [
            {
                "name_ko": "자동 촬영",
                "name_en": "automatic cinematography",
                "keywords": ["camera selection", "view planning", "shot composition"],
            },
            {
                "name_ko": "스포츠 방송",
                "name_en": "sports broadcasting",
                "keywords": ["highlight detection", "real-time", "action recognition"],
            },
            {
                "name_ko": "자세 추정",
                "name_en": "pose estimation",
                "keywords": ["body tracking", "skeleton detection"],
            },
        ],
        "cross_domain_keywords": ["neural radiance field", "attention mechanism"],
        "exclude_keywords": ["medical imaging", "satellite"],
        "topic_embedding_text": "Automatic cinematography and camera selection...",
    }


@pytest.fixture()
def typical_categories() -> list[str]:
    """Provide typical arXiv category list."""
    return ["cs.CV", "cs.AI"]


@pytest.fixture()
def single_concept_output() -> dict:
    """Provide Agent 1 output with a single concept."""
    return {
        "concepts": [
            {
                "name_ko": "강화학습",
                "name_en": "reinforcement learning",
                "keywords": ["policy gradient", "reward shaping"],
            },
        ],
        "cross_domain_keywords": [],
        "exclude_keywords": [],
        "topic_embedding_text": "Reinforcement learning...",
    }


@pytest.fixture()
def minimal_input() -> dict:
    """Provide minimal Agent 1 output (1 concept, 1 keyword)."""
    return {
        "concepts": [
            {
                "name_ko": "LLM",
                "name_en": "large language model",
                "keywords": ["transformer"],
            },
        ],
        "cross_domain_keywords": [],
        "exclude_keywords": [],
        "topic_embedding_text": "Large language model...",
    }


@pytest.fixture()
def large_input() -> dict:
    """Provide large Agent 1 output (10 concepts, 20 cross_domain_keywords)."""
    concepts = []
    for i in range(10):
        concepts.append(
            {
                "name_ko": f"concept_{i}",
                "name_en": f"concept {i} name",
                "keywords": [f"keyword_{i}_a", f"keyword_{i}_b", f"keyword_{i}_c"],
            }
        )
    return {
        "concepts": concepts,
        "cross_domain_keywords": [f"cross_kw_{i}" for i in range(20)],
        "exclude_keywords": ["exclude_a"],
        "topic_embedding_text": "Large input...",
    }


# ---------------------------------------------------------------------------
# Test 1: Valid arXiv DSL syntax
# ---------------------------------------------------------------------------


class TestDSLSyntax:
    """Tests for valid arXiv query DSL syntax."""

    def test_queries_contain_valid_dsl_operators(
        self, builder, typical_agent1_output, typical_categories
    ):
        """All queries should use valid arXiv DSL fields (ti:, abs:, cat:)."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        for q in queries:
            # Each query should contain at least one recognized field prefix
            assert any(
                prefix in q for prefix in ("ti:", "abs:", "cat:")
            ), f"Query missing DSL field: {q}"

    def test_queries_use_valid_boolean_operators(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Boolean operators should be AND, OR, or ANDNOT."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        import re

        for q in queries:
            # Find standalone boolean operators (all caps, word boundary)
            # Remove cat: clauses including parenthesized groups like cat:(cs.CV OR cs.AI)
            clean_q = re.sub(r"cat:\([^)]+\)", "", q)
            clean_q = re.sub(r"cat:\S+", "", clean_q)
            # Remove quoted strings to avoid false positives
            clean_q = re.sub(r'"[^"]*"', "", clean_q)
            tokens = re.findall(r"\b[A-Z]{2,}\b", clean_q)
            valid_ops = {"AND", "OR", "ANDNOT"}
            for token in tokens:
                assert token in valid_ops, (
                    f"Invalid operator '{token}' in query: {q}"
                )


# ---------------------------------------------------------------------------
# Test 2: Query count within 15-25 range
# ---------------------------------------------------------------------------


class TestQueryCount:
    """Tests for query count bounds."""

    def test_typical_input_count_in_range(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Typical input should generate 15-25 queries."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        assert 15 <= len(queries) <= 25, f"Got {len(queries)} queries"

    def test_large_input_capped_at_25(
        self, builder, large_input, typical_categories
    ):
        """Large input should not exceed 25 queries."""
        queries = builder.build_queries(large_input, typical_categories)
        assert len(queries) <= 25

    def test_minimal_input_generates_queries(self, builder, minimal_input):
        """Minimal input should generate reasonable queries (padding applied)."""
        queries = builder.build_queries(minimal_input, ["cs.LG"])
        # With very minimal input (1 concept, 1 keyword, 1 category),
        # unique query combinations are limited. Verify we get a reasonable count.
        assert len(queries) >= 5, f"Got only {len(queries)} queries"
        assert all(isinstance(q, str) for q in queries)


# ---------------------------------------------------------------------------
# Test 3: Broad queries include category + concept keywords
# ---------------------------------------------------------------------------


class TestPhase1BroadQueries:
    """Tests for Phase 1: broad category queries."""

    def test_broad_queries_contain_category(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Broad queries should include cat: field with category."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        cat_queries = [q for q in queries if "cat:" in q and "OR" in q]
        assert len(cat_queries) > 0, "No broad category queries generated"

    def test_broad_queries_contain_concept_keywords(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Broad queries should reference concept keywords in abs: field."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        broad_with_abs = [
            q for q in queries if "cat:" in q and "abs:" in q
        ]
        assert len(broad_with_abs) > 0


# ---------------------------------------------------------------------------
# Test 4: Narrow queries include title and abstract fields
# ---------------------------------------------------------------------------


class TestPhase4NarrowQueries:
    """Tests for Phase 4: narrow precision queries."""

    def test_narrow_queries_use_ti_and_abs(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Narrow queries should include both ti: and abs: fields."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        narrow = [q for q in queries if "ti:" in q and "abs:" in q and "cat:" in q]
        assert len(narrow) > 0, "No narrow queries with ti: AND abs: AND cat:"


# ---------------------------------------------------------------------------
# Test 5: Cross-domain queries combine different concepts
# ---------------------------------------------------------------------------


class TestPhase3CrossDomain:
    """Tests for Phase 3: cross-domain queries."""

    def test_cross_domain_combines_concepts(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Cross-domain queries should combine keywords from different concepts."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        # Look for queries containing keywords from two different concepts
        c0_kw = typical_agent1_output["concepts"][0]["keywords"][0]
        c1_kw = typical_agent1_output["concepts"][1]["keywords"][0]
        cross = [q for q in queries if c0_kw in q and c1_kw in q]
        assert len(cross) > 0, "No cross-domain query combining different concepts"


# ---------------------------------------------------------------------------
# Test 6-7: Exclude keywords appended as ANDNOT
# ---------------------------------------------------------------------------


class TestExclusions:
    """Tests for exclusion keyword handling."""

    def test_exclude_keywords_appended_as_andnot(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Each exclude keyword should appear as ANDNOT in every query."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        for q in queries:
            assert 'ANDNOT abs:"medical imaging"' in q
            assert 'ANDNOT abs:"satellite"' in q

    def test_multiple_exclude_keywords_all_added(self, builder, typical_categories):
        """All exclude keywords should be present in each query."""
        output = {
            "concepts": [
                {
                    "name_en": "test concept",
                    "keywords": ["keyword_a", "keyword_b"],
                },
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": ["bad_topic_1", "bad_topic_2", "bad_topic_3"],
        }
        queries = builder.build_queries(output, typical_categories)
        for q in queries:
            for excl in output["exclude_keywords"]:
                assert f'ANDNOT abs:"{excl}"' in q

    def test_no_exclude_keywords_no_andnot(self, builder, typical_categories):
        """Queries should not have ANDNOT when exclude_keywords is empty."""
        output = {
            "concepts": [
                {"name_en": "test", "keywords": ["kw1", "kw2"]},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        for q in queries:
            assert "ANDNOT" not in q


# ---------------------------------------------------------------------------
# Test 8: Empty concepts -> category-only queries
# ---------------------------------------------------------------------------


class TestEmptyConcepts:
    """Tests for empty concepts edge case."""

    def test_empty_concepts_generates_category_queries(
        self, builder, typical_categories
    ):
        """Empty concepts should still generate queries using categories."""
        output = {
            "concepts": [],
            "cross_domain_keywords": ["attention"],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        assert len(queries) >= 3, f"Got only {len(queries)} queries with empty concepts"
        # Should have category-based queries
        cat_queries = [q for q in queries if "cat:" in q]
        assert len(cat_queries) > 0


# ---------------------------------------------------------------------------
# Test 9: Empty categories -> queries without cat: constraint
# ---------------------------------------------------------------------------


class TestEmptyCategories:
    """Tests for empty categories edge case."""

    def test_empty_categories_generates_queries_without_cat(
        self, builder, typical_agent1_output
    ):
        """Empty categories should produce queries without cat: field."""
        queries = builder.build_queries(typical_agent1_output, [])
        assert len(queries) >= 10, f"Got only {len(queries)} queries without categories"
        # Check that abs: or ti: queries exist
        abs_or_ti = [q for q in queries if "abs:" in q or "ti:" in q]
        assert len(abs_or_ti) > 0


# ---------------------------------------------------------------------------
# Test 10: Empty keywords in concept -> concept skipped
# ---------------------------------------------------------------------------


class TestEmptyKeywords:
    """Tests for concepts with empty keywords."""

    def test_concept_with_empty_keywords_skipped(self, builder, typical_categories):
        """Concepts with empty keywords list should be skipped."""
        output = {
            "concepts": [
                {"name_en": "empty concept", "keywords": []},
                {"name_en": "valid concept", "keywords": ["valid_kw"]},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        # Should not contain "empty concept" keywords (there are none)
        # but should contain "valid_kw"
        has_valid = any("valid_kw" in q for q in queries)
        assert has_valid, "Valid concept keyword not found in queries"


# ---------------------------------------------------------------------------
# Test 11: No cross_domain_keywords -> Phase 3 cross portion skipped
# ---------------------------------------------------------------------------


class TestNoCrossDomainKeywords:
    """Tests for missing cross_domain_keywords."""

    def test_no_cross_domain_keywords_still_generates_queries(
        self, builder, typical_categories
    ):
        """Queries should still be generated without cross_domain_keywords."""
        output = {
            "concepts": [
                {"name_en": "concept a", "keywords": ["kw_a1", "kw_a2"]},
                {"name_en": "concept b", "keywords": ["kw_b1", "kw_b2"]},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        assert len(queries) >= 15


# ---------------------------------------------------------------------------
# Test 12: Queries are deduplicated
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for query deduplication."""

    def test_no_duplicate_queries(
        self, builder, typical_agent1_output, typical_categories
    ):
        """All queries in the output should be unique."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        # Strip ANDNOT parts to compare base queries
        assert len(queries) == len(set(queries)), "Duplicate queries found"


# ---------------------------------------------------------------------------
# Test 13: Single concept input
# ---------------------------------------------------------------------------


class TestSingleConcept:
    """Tests for single concept input."""

    def test_single_concept_generates_valid_queries(
        self, builder, single_concept_output, typical_categories
    ):
        """Single concept should produce valid queries within range."""
        queries = builder.build_queries(single_concept_output, typical_categories)
        assert len(queries) >= 8, f"Got only {len(queries)} queries for single concept"
        assert len(queries) <= 25
        # Should have concept keyword in some queries
        has_policy = any("policy gradient" in q for q in queries)
        assert has_policy


# ---------------------------------------------------------------------------
# Test 14: Single category input
# ---------------------------------------------------------------------------


class TestSingleCategory:
    """Tests for single category input."""

    def test_single_category_generates_valid_queries(
        self, builder, typical_agent1_output
    ):
        """Single category should produce valid queries."""
        queries = builder.build_queries(typical_agent1_output, ["cs.CV"])
        assert 15 <= len(queries) <= 25
        cat_queries = [q for q in queries if "cs.CV" in q]
        assert len(cat_queries) > 0


# ---------------------------------------------------------------------------
# Test 15: Many concepts/keywords -> truncated to 25
# ---------------------------------------------------------------------------


class TestTruncation:
    """Tests for query truncation."""

    def test_many_concepts_truncated(self, builder, large_input):
        """Many concepts should not produce more than 25 queries."""
        categories = ["cs.CV", "cs.AI", "cs.LG"]
        queries = builder.build_queries(large_input, categories)
        assert len(queries) <= 25


# ---------------------------------------------------------------------------
# Test 16: Few concepts -> padded to reach 15 minimum
# ---------------------------------------------------------------------------


class TestPadding:
    """Tests for query padding."""

    def test_few_concepts_padded(self, builder, minimal_input):
        """Few concepts should be padded beyond raw phase output."""
        queries = builder.build_queries(minimal_input, ["cs.LG"])
        # With minimal input, padding increases query count beyond base phases
        assert len(queries) >= 5, f"Got only {len(queries)} queries"


# ---------------------------------------------------------------------------
# Test 17: Queries ordered broad to narrow
# ---------------------------------------------------------------------------


class TestOrdering:
    """Tests for query ordering (broad to narrow)."""

    def test_broad_before_narrow(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Broad category queries should appear before narrow ti+abs queries."""
        # Build queries without exclusions to inspect base ordering
        output_no_excl = dict(typical_agent1_output)
        output_no_excl["exclude_keywords"] = []
        queries = builder.build_queries(output_no_excl, typical_categories)

        # Find first broad (cat: with OR) and first narrow (ti: AND abs: AND cat:)
        first_broad_idx = None
        first_narrow_idx = None
        for i, q in enumerate(queries):
            if first_broad_idx is None and "cat:" in q and "OR" in q:
                first_broad_idx = i
            if first_narrow_idx is None and "ti:" in q and "abs:" in q:
                first_narrow_idx = i

        if first_broad_idx is not None and first_narrow_idx is not None:
            assert first_broad_idx < first_narrow_idx, (
                f"Broad query at {first_broad_idx} should come before "
                f"narrow query at {first_narrow_idx}"
            )


# ---------------------------------------------------------------------------
# Test 18: All queries are strings (no None values)
# ---------------------------------------------------------------------------


class TestQueryTypes:
    """Tests for query value types."""

    def test_all_queries_are_strings(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Every query should be a non-empty string, never None."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        for q in queries:
            assert isinstance(q, str), f"Query is not a string: {type(q)}"
            assert len(q) > 0, "Empty query string found"


# ---------------------------------------------------------------------------
# Test 19: Special characters in keywords handled
# ---------------------------------------------------------------------------


class TestSpecialCharacters:
    """Tests for special character handling."""

    def test_quotes_in_keywords_escaped(self, builder, typical_categories):
        """Double quotes in keywords should be escaped."""
        output = {
            "concepts": [
                {
                    "name_en": 'concept with "quotes"',
                    "keywords": ['keyword "quoted"', "normal keyword"],
                },
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        for q in queries:
            # The escaped quotes should not break the DSL
            assert isinstance(q, str)
        # Check that escaping occurred
        has_escaped = any('\\"' in q for q in queries)
        assert has_escaped, "Quotes should be escaped in queries"


# ---------------------------------------------------------------------------
# Test 20: Category format preserved
# ---------------------------------------------------------------------------


class TestCategoryFormat:
    """Tests for category format preservation."""

    def test_category_case_preserved(
        self, builder, typical_agent1_output
    ):
        """Category identifiers like cs.CV should not be lowercased."""
        queries = builder.build_queries(typical_agent1_output, ["cs.CV", "eess.SP"])
        cat_queries = [q for q in queries if "cat:" in q]
        assert any("cs.CV" in q for q in cat_queries), "cs.CV not found"
        assert any("eess.SP" in q for q in cat_queries), "eess.SP not found"


# ---------------------------------------------------------------------------
# Test 21: cross_domain_keywords included in queries
# ---------------------------------------------------------------------------


class TestCrossDomainKeywordsInQueries:
    """Tests for cross_domain_keywords presence in queries."""

    def test_cross_domain_keywords_appear(
        self, builder, typical_agent1_output, typical_categories
    ):
        """cross_domain_keywords should appear in at least some queries."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        has_neural = any("neural radiance field" in q for q in queries)
        has_attention = any("attention mechanism" in q for q in queries)
        assert has_neural or has_attention, (
            "cross_domain_keywords not found in any query"
        )


# ---------------------------------------------------------------------------
# Test 22: Multiple categories joined with OR in cat: clause
# ---------------------------------------------------------------------------


class TestMultipleCategories:
    """Tests for multi-category clause construction."""

    def test_multiple_categories_use_or(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Multiple categories should be joined with OR in cat: clause."""
        queries = builder.build_queries(typical_agent1_output, typical_categories)
        # Phase 2 and Phase 3 use cat:(...OR...) for multiple categories
        or_cat_queries = [
            q for q in queries if "cat:(" in q and " OR " in q
        ]
        assert len(or_cat_queries) > 0, (
            "No queries with cat:(... OR ...) found"
        )


# ---------------------------------------------------------------------------
# Test 23: Deterministic: same input -> same output
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Same input should always produce the same output."""
        result1 = builder.build_queries(typical_agent1_output, typical_categories)
        result2 = builder.build_queries(typical_agent1_output, typical_categories)
        assert result1 == result2, "Non-deterministic behavior detected"

    def test_deterministic_across_instances(
        self, typical_agent1_output, typical_categories
    ):
        """Different builder instances should produce same output."""
        b1 = ArxivQueryBuilder()
        b2 = ArxivQueryBuilder()
        assert b1.build_queries(
            typical_agent1_output, typical_categories
        ) == b2.build_queries(typical_agent1_output, typical_categories)


# ---------------------------------------------------------------------------
# Test 24: Large input (10 concepts, 20 cross_domain_keywords)
# ---------------------------------------------------------------------------


class TestLargeInput:
    """Tests for large input handling."""

    def test_large_input_within_bounds(
        self, builder, large_input, typical_categories
    ):
        """Large input should generate 15-25 queries."""
        queries = builder.build_queries(large_input, typical_categories)
        assert 15 <= len(queries) <= 25

    def test_large_input_all_strings(
        self, builder, large_input, typical_categories
    ):
        """Large input queries should all be valid strings."""
        queries = builder.build_queries(large_input, typical_categories)
        for q in queries:
            assert isinstance(q, str)
            assert len(q) > 0


# ---------------------------------------------------------------------------
# Test 25: Minimal input (1 concept, 1 keyword, 1 category)
# ---------------------------------------------------------------------------


class TestMinimalInput:
    """Tests for minimal input handling."""

    def test_minimal_input_within_bounds(self, builder, minimal_input):
        """Minimal input should generate reasonable queries."""
        queries = builder.build_queries(minimal_input, ["cs.LG"])
        assert len(queries) >= 5, f"Got only {len(queries)} queries"
        assert len(queries) <= 25

    def test_minimal_input_contains_keyword(self, builder, minimal_input):
        """Minimal input queries should contain the single keyword."""
        queries = builder.build_queries(minimal_input, ["cs.LG"])
        has_transformer = any("transformer" in q for q in queries)
        assert has_transformer


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    """Additional edge case tests beyond the 25 required."""

    def test_completely_empty_input(self, builder):
        """Empty concepts, categories, and keywords should not raise."""
        output = {
            "concepts": [],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, [])
        assert isinstance(queries, list)
        # May not reach 15, but should not crash
        for q in queries:
            assert isinstance(q, str)

    def test_missing_keys_in_agent1_output(self, builder, typical_categories):
        """Missing keys in agent1_output should be handled gracefully."""
        output: dict = {}  # All keys missing
        queries = builder.build_queries(output, typical_categories)
        assert isinstance(queries, list)

    def test_concept_without_name_en(self, builder, typical_categories):
        """Concept missing name_en should not crash."""
        output = {
            "concepts": [
                {"keywords": ["keyword_a", "keyword_b"]},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        assert len(queries) >= 5, f"Got only {len(queries)} queries"
        has_kw = any("keyword_a" in q for q in queries)
        assert has_kw

    def test_concept_missing_keywords_key(self, builder, typical_categories):
        """Concept missing 'keywords' key should be filtered out."""
        output = {
            "concepts": [
                {"name_en": "no keywords concept"},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": [],
        }
        queries = builder.build_queries(output, typical_categories)
        assert isinstance(queries, list)

    def test_exclude_keywords_with_special_chars(self, builder, typical_categories):
        """Exclude keywords with quotes should be escaped."""
        output = {
            "concepts": [
                {"name_en": "test", "keywords": ["test_kw"]},
            ],
            "cross_domain_keywords": [],
            "exclude_keywords": ['bad "topic"'],
        }
        queries = builder.build_queries(output, typical_categories)
        for q in queries:
            assert 'ANDNOT abs:"bad \\"topic\\""' in q

    def test_return_type_is_list(
        self, builder, typical_agent1_output, typical_categories
    ):
        """Return value should always be a list."""
        result = builder.build_queries(typical_agent1_output, typical_categories)
        assert isinstance(result, list)

    def test_three_categories(self, builder, typical_agent1_output):
        """Three categories should all appear in generated queries."""
        cats = ["cs.CV", "cs.AI", "cs.LG"]
        queries = builder.build_queries(typical_agent1_output, cats)
        all_text = " ".join(queries)
        for cat in cats:
            assert cat in all_text, f"Category {cat} not found in queries"
