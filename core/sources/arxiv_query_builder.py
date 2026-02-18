"""arXiv query builder for Paper Scout.

Generates arXiv search queries from Agent 1 keyword output.
Deterministic (no LLM). Generates 15-25 queries ranging from
broad category-level to narrow keyword-combination.

Reference: devspec.md Section 6-2.
"""

from __future__ import annotations

from itertools import combinations


class ArxivQueryBuilder:
    """Generate arXiv search queries from Agent 1 keyword output.

    Deterministic (no LLM). Generates 15-25 queries ranging from
    broad category-level to narrow keyword-combination.

    Query generation follows four phases:
      Phase 1: Broad category queries (recall-first)
      Phase 2: Concept-focused queries (precision-first)
      Phase 3: Cross-domain queries (interdisciplinary discovery)
      Phase 4: Narrow precision queries (tight keyword matching)

    Exclusion keywords are appended as ANDNOT clauses to all queries.
    """

    # Target query count bounds
    _MIN_QUERIES = 15
    _MAX_QUERIES = 50

    _STOP_WORDS = frozenset({
        "a", "an", "the", "and", "or", "of", "in", "on", "for", "to",
        "with", "by", "from", "at", "is", "are", "was", "were", "be",
        "been", "as", "its", "it", "that", "this", "not", "but",
    })

    def build_queries(
        self,
        agent1_output: dict,
        categories: list[str],
    ) -> list[str]:
        """Generate arXiv query strings.

        Args:
            agent1_output: Dict with keys:
                - concepts: list[dict] with name_en, keywords
                - cross_domain_keywords: list[str]
                - exclude_keywords: list[str]
            categories: list of arXiv category strings
                (e.g., ["cs.CV", "cs.AI"])

        Returns:
            List of 15-25 arXiv query strings, broad to narrow.
        """
        concepts = agent1_output.get("concepts", [])
        cross_domain_keywords = agent1_output.get("cross_domain_keywords", [])
        exclude_keywords = agent1_output.get("exclude_keywords", [])
        exclude_mode = str(agent1_output.get("exclude_mode", "soft")).lower()
        query_must_keywords = self._normalize_keywords(
            agent1_output.get("query_must_keywords", [])
        )

        # Filter out concepts with empty or missing keywords
        valid_concepts = [
            c for c in concepts if c.get("keywords") and len(c["keywords"]) > 0
        ]

        queries: list[str] = []

        # Explicit must-keyword queries (user-selected).
        # Placed first so they survive truncation.
        must_queries = self._build_must_keyword_queries(
            query_must_keywords,
            categories,
        )
        queries.extend(must_queries)

        # Concept coverage queries:
        # Ensure each concept contributes at least one explicit query
        # before any phase-level truncation.
        coverage = self._build_concept_coverage_queries(
            valid_concepts,
            categories,
        )
        queries.extend(coverage)

        # Phase 1: Broad category queries (5-8)
        phase1 = self._build_broad_queries(valid_concepts, categories)
        queries.extend(phase1)

        # Phase 2: Concept-focused queries (5-8)
        phase2 = self._build_concept_queries(valid_concepts, categories)
        queries.extend(phase2)

        # Phase 3: Cross-domain queries (3-5)
        phase3 = self._build_cross_domain_queries(
            valid_concepts,
            cross_domain_keywords,
            categories,
        )
        queries.extend(phase3)

        # Phase 4: Narrow precision queries (2-4)
        phase4 = self._build_narrow_queries(valid_concepts, categories)
        queries.extend(phase4)

        # Deduplicate while preserving order
        queries = self._deduplicate(queries)

        # Pad if below minimum
        if len(queries) < self._MIN_QUERIES:
            padding = self._generate_padding(
                valid_concepts,
                categories,
                cross_domain_keywords,
                existing=queries,
            )
            queries.extend(padding)
            queries = self._deduplicate(queries)

        # Truncate if above maximum
        if len(queries) > self._MAX_QUERIES:
            queries = queries[: self._MAX_QUERIES]

        # Append exclusion clauses to all queries
        if exclude_keywords and exclude_mode == "strict":
            queries = [
                self._append_exclusions(q, exclude_keywords) for q in queries
            ]

        return queries

    # ------------------------------------------------------------------
    # Phase builders
    # ------------------------------------------------------------------

    def _build_broad_queries(
        self,
        concepts: list[dict],
        categories: list[str],
    ) -> list[str]:
        """Phase 1: Broad category queries (recall-first).

        Distributes queries across ALL concepts using round-robin.
        Target: 5-9 queries.
        """
        queries: list[str] = []

        if not categories and not concepts:
            return queries

        if not concepts:
            for cat in categories:
                queries.append(f"cat:{cat}")
            return queries

        if not categories:
            for concept in concepts:
                kws = concept.get("keywords", [])[:3]
                if kws:
                    or_clause = " OR ".join(
                        self._kw_to_abs(k) for k in kws
                    )
                    queries.append(f"({or_clause})")
                if len(queries) >= 9:
                    return queries
            return queries

        # Round-robin: distribute categories across ALL concepts
        cat_idx = 0
        for concept in concepts:
            if cat_idx >= len(categories):
                cat_idx = 0
            kws = concept.get("keywords", [])[:3]
            if not kws:
                continue
            or_clause = " OR ".join(
                self._kw_to_abs(k) for k in kws
            )
            queries.append(f"cat:{categories[cat_idx]} AND ({or_clause})")
            cat_idx += 1

            if len(queries) >= 9:
                return queries

        return queries

    def _build_concept_queries(
        self,
        concepts: list[dict],
        categories: list[str],
    ) -> list[str]:
        """Phase 2: Concept-focused queries (precision-first).

        For each concept, combine its keywords with AND.
        Also include title-based queries.
        Ensures ALL concepts get at least one query.
        Target: 5-10 queries.
        """
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        # Pass 1: One keyword-based query per concept (ensures coverage)
        # Split concept name into meaningful words and use all: field
        # instead of exact title match which returns 0 results.
        for concept in concepts:
            name_en = concept.get("name_en", "")
            if name_en:
                words = name_en.strip().split()
                meaningful = [
                    w.lower() for w in words
                    if len(w) >= 3 and w.lower() not in self._STOP_WORDS
                ]
                if len(meaningful) < 2:
                    continue
                selected = meaningful[:3]
                q = "(" + " AND ".join(
                    f"all:{self._escape(w)}" for w in selected
                ) + ")"
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                queries.append(q)

        # Pass 2: Keyword AND combinations sampled across ALL concepts
        # (instead of only the first concept).
        for concept in concepts:
            kws = concept.get("keywords", [])
            if len(kws) < 2:
                continue

            q = self._kws_to_abs_merged(kws[:2])
            if cat_clause:
                q = f"{q} AND {cat_clause}"
            queries.append(q)

            if len(queries) >= 10:
                return queries

        return queries

    def _build_concept_coverage_queries(
        self,
        concepts: list[dict],
        categories: list[str],
    ) -> list[str]:
        """Build one high-priority query per concept.

        This makes manual concept edits visible in final query output
        even when later phase queries are deduplicated/truncated.
        """
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        for concept in concepts:
            name_en = concept.get("name_en", "")
            kws = concept.get("keywords", [])
            if not kws:
                continue

            # Include both head and tail keywords so manually added terms
            # at the end of the list are also reflected in query text.
            selected_kws = self._select_coverage_keywords(kws, max_count=12)
            kw_clause = " OR ".join(
                self._kw_to_abs(kw) for kw in selected_kws
            )
            if name_en:
                q = (
                    f'(ti:"{self._escape(name_en)}" OR '
                    f'({kw_clause}))'
                )
            else:
                q = f"({kw_clause})"

            if cat_clause:
                q = f"{q} AND {cat_clause}"
            queries.append(q)

        return queries

    def _build_must_keyword_queries(
        self,
        must_keywords: list[str],
        categories: list[str],
    ) -> list[str]:
        """Build explicit queries for user-selected must keywords."""
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        for kw in must_keywords[:8]:
            q = self._kw_to_abs(kw)
            if cat_clause:
                q = f"{q} AND {cat_clause}"
            queries.append(q)

        return queries

    def _build_cross_domain_queries(
        self,
        concepts: list[dict],
        cross_domain_keywords: list[str],
        categories: list[str],
    ) -> list[str]:
        """Phase 3: Cross-domain queries (interdisciplinary discovery).

        Cross-combine keywords from different concepts and use
        cross_domain_keywords. Samples from across all concepts.
        Target: 3-5 queries.
        """
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        # Cross-combine keywords from different concepts (sample broadly)
        if len(concepts) >= 2:
            # Use evenly spaced concept pairs to cover all concepts
            step = max(1, len(concepts) // 4)
            sampled_indices = list(range(0, len(concepts), step))[:6]
            for i, j in combinations(sampled_indices, 2):
                if i >= len(concepts) or j >= len(concepts):
                    continue
                kws_a = concepts[i].get("keywords", [])
                kws_b = concepts[j].get("keywords", [])
                if kws_a and kws_b:
                    q = f"({self._kw_to_abs(kws_a[0])} OR {self._kw_to_abs(kws_b[0])})"
                    queries.append(q)

                    if len(queries) >= 3:
                        break

        # cross_domain_keywords with categories
        for kw in cross_domain_keywords[:3]:
            q = self._kw_to_abs(kw)
            if cat_clause:
                q = f"{q} AND {cat_clause}"
            queries.append(q)

            if len(queries) >= 5:
                break

        return queries

    def _build_narrow_queries(
        self,
        concepts: list[dict],
        categories: list[str],
    ) -> list[str]:
        """Phase 4: Narrow precision queries (tight keyword matching).

        Samples from across all concepts for even coverage.
        Combine title and abstract fields with category.
        Target: 2-5 queries.
        """
        queries: list[str] = []

        # Sample evenly across all concepts
        step = max(1, len(concepts) // 4)
        sampled = concepts[::step][:5]

        for concept in sampled:
            kws = concept.get("keywords", [])
            if not kws:
                continue

            cat = categories[0] if categories else None
            if len(kws) >= 2:
                # Use 2 different keywords in abs: field
                q = f"{self._kw_to_abs(kws[0])} AND {self._kw_to_abs(kws[1])}"
            else:
                # Single keyword: just abs: with category
                q = self._kw_to_abs(kws[0])
            if cat:
                q = f"{q} AND cat:{cat}"
            queries.append(q)

            if len(queries) >= 5:
                return queries

        return queries

    # ------------------------------------------------------------------
    # Padding & exclusion helpers
    # ------------------------------------------------------------------

    def _generate_padding(
        self,
        concepts: list[dict],
        categories: list[str],
        cross_domain_keywords: list[str],
        existing: list[str],
    ) -> list[str]:
        """Generate additional queries to reach the minimum count.

        Explores more category-concept and keyword combinations that
        were not generated in the main phases.
        """
        padding: list[str] = []
        existing_set = set(existing)
        needed = self._MIN_QUERIES - len(existing)

        cat_clause = self._build_cat_clause(categories)

        # Strategy 1: Single keyword + category combinations
        for concept in concepts:
            for kw in concept.get("keywords", []):
                q = self._kw_to_abs(kw)
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 2: Cross-domain keyword pairs (OR for broader recall)
        if len(cross_domain_keywords) >= 2:
            for kw1, kw2 in combinations(cross_domain_keywords[:6], 2):
                q = f"({self._kw_to_abs(kw1)} OR {self._kw_to_abs(kw2)})"
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 3: Concept name_en in abstract + category
        for concept in concepts:
            name_en = concept.get("name_en", "")
            if name_en:
                q = self._kw_to_abs(name_en)
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 4: Category-only queries (last resort)
        for cat in categories:
            q = f"cat:{cat}"
            if q not in existing_set:
                padding.append(q)
                existing_set.add(q)
            if len(padding) >= needed:
                return padding

        # Strategy 5: Individual cross_domain_keywords without category
        for kw in cross_domain_keywords:
            q = self._kw_to_abs(kw)
            if q not in existing_set:
                padding.append(q)
                existing_set.add(q)
            if len(padding) >= needed:
                return padding

        # Strategy 6: Title-only queries for each keyword
        for concept in concepts:
            for kw in concept.get("keywords", []):
                q = f'ti:"{self._escape(kw)}"'
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 7: Keyword + individual category combinations
        for concept in concepts:
            for kw in concept.get("keywords", []):
                for cat in categories:
                    q = f'{self._kw_to_abs(kw)} AND cat:{cat}'
                    if q not in existing_set:
                        padding.append(q)
                        existing_set.add(q)
                    if len(padding) >= needed:
                        return padding

        # Strategy 8: Title + abstract without category
        for concept in concepts:
            name_en = concept.get("name_en", "")
            for kw in concept.get("keywords", []):
                if name_en and name_en != kw:
                    q = (
                        f'ti:"{self._escape(name_en)}" AND '
                        f'{self._kw_to_abs(kw)}'
                    )
                    if q not in existing_set:
                        padding.append(q)
                        existing_set.add(q)
                    if len(padding) >= needed:
                        return padding

        # Strategy 9: Broad OR queries across all keywords
        all_kws = []
        for concept in concepts:
            all_kws.extend(concept.get("keywords", []))
        if len(all_kws) >= 2:
            for kw1, kw2 in combinations(all_kws[:8], 2):
                q = f'{self._kw_to_abs(kw1)} OR {self._kw_to_abs(kw2)}'
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        return padding

    def _append_exclusions(
        self, query: str, exclude_keywords: list[str]
    ) -> str:
        """Append ANDNOT clauses for each exclusion keyword."""
        for kw in exclude_keywords:
            query = f'{query} ANDNOT abs:"{self._escape(kw)}"'
        return query

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @classmethod
    def _kw_to_abs(cls, keyword: str) -> str:
        """Convert a keyword phrase to abs search clause.

        Single word: abs:word
        Multi-word: (abs:word1 AND abs:word2 AND ...) with stop words removed and dedup.
        """
        words = keyword.strip().split()
        meaningful = [w for w in words if w.lower() not in cls._STOP_WORDS]
        if not meaningful:
            meaningful = words  # fallback if all stop words
        seen: set[str] = set()
        unique: list[str] = []
        for w in meaningful:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(wl)
        if not unique:
            return ''
        if len(unique) == 1:
            return f'abs:{cls._escape(unique[0])}'
        return "(" + " AND ".join(f"abs:{cls._escape(w)}" for w in unique) + ")"

    @classmethod
    def _kws_to_abs_merged(cls, keywords: list[str]) -> str:
        """Merge multiple keyword phrases, deduplicate all words, AND together.

        Example: ["sports event detection", "sports video enhancement"]
        -> (abs:sports AND abs:event AND abs:detection AND abs:video AND abs:enhancement)
        """
        all_words: list[str] = []
        for kw in keywords:
            all_words.extend(kw.strip().split())
        meaningful = [w for w in all_words if w.lower() not in cls._STOP_WORDS]
        if not meaningful:
            meaningful = all_words
        seen: set[str] = set()
        unique: list[str] = []
        for w in meaningful:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(wl)
        if not unique:
            return ''
        if len(unique) == 1:
            return f'abs:{cls._escape(unique[0])}'
        return "(" + " AND ".join(f"abs:{cls._escape(w)}" for w in unique) + ")"

    def _build_cat_clause(self, categories: list[str]) -> str:
        """Build a category clause joining multiple categories with OR.

        Returns empty string if no categories provided.
        """
        if not categories:
            return ""
        if len(categories) == 1:
            return f"cat:{categories[0]}"
        return "cat:(" + " OR ".join(categories) + ")"

    @staticmethod
    def _escape(text: str) -> str:
        """Escape double quotes in query text for arXiv DSL."""
        return text.replace('"', '\\"')

    @staticmethod
    def _deduplicate(queries: list[str]) -> list[str]:
        """Remove duplicate queries while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                result.append(q)
        return result

    @staticmethod
    def _select_coverage_keywords(
        keywords: list[str],
        max_count: int = 12,
    ) -> list[str]:
        """Pick keywords from both head and tail for coverage queries."""
        if len(keywords) <= max_count:
            return keywords

        half = max_count // 2
        head = keywords[:half]
        tail = keywords[-(max_count - half):]

        merged: list[str] = []
        seen: set[str] = set()
        for kw in head + tail:
            k = kw.lower()
            if k not in seen:
                seen.add(k)
                merged.append(kw)
        return merged[:max_count]

    @staticmethod
    def _normalize_keywords(values: list[str]) -> list[str]:
        """Normalize keyword list: trim and deduplicate case-insensitively."""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            kw = value.strip()
            if not kw:
                continue
            key = kw.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(kw)
        return result
