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
    _MAX_QUERIES = 25

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

        # Filter out concepts with empty or missing keywords
        valid_concepts = [
            c for c in concepts if c.get("keywords") and len(c["keywords"]) > 0
        ]

        queries: list[str] = []

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
        if exclude_keywords:
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

        For each category, combine with top concept keywords.
        Target: 5-8 queries.
        """
        queries: list[str] = []

        if not categories and not concepts:
            return queries

        # If no concepts, generate category-only queries
        if not concepts:
            for cat in categories:
                queries.append(f"cat:{cat}")
            return queries

        # If no categories, generate concept-keyword-only broad queries
        if not categories:
            for concept in concepts[:4]:
                kws = concept.get("keywords", [])[:3]
                if kws:
                    or_clause = " OR ".join(
                        f'abs:"{self._escape(k)}"' for k in kws
                    )
                    queries.append(f"({or_clause})")
            return queries

        # Normal case: category + concept keywords
        for cat in categories:
            # Each category gets 1-2 queries depending on concept count
            for concept in concepts[:3]:
                kws = concept.get("keywords", [])[:3]
                if not kws:
                    continue
                or_clause = " OR ".join(
                    f'abs:"{self._escape(k)}"' for k in kws
                )
                queries.append(f"cat:{cat} AND ({or_clause})")

                if len(queries) >= 8:
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
        Target: 5-8 queries.
        """
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        for concept in concepts:
            name_en = concept.get("name_en", "")
            kws = concept.get("keywords", [])

            # Title-based query
            if name_en:
                q = f'ti:"{self._escape(name_en)}"'
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                queries.append(q)

            # Keyword AND combinations (pairs)
            if len(kws) >= 2:
                for kw1, kw2 in combinations(kws[:4], 2):
                    q = (
                        f'(abs:"{self._escape(kw1)}" AND '
                        f'abs:"{self._escape(kw2)}")'
                    )
                    if cat_clause:
                        q = f"{q} AND {cat_clause}"
                    queries.append(q)

                    if len(queries) >= 8:
                        return queries

        return queries

    def _build_cross_domain_queries(
        self,
        concepts: list[dict],
        cross_domain_keywords: list[str],
        categories: list[str],
    ) -> list[str]:
        """Phase 3: Cross-domain queries (interdisciplinary discovery).

        Cross-combine keywords from different concepts and use
        cross_domain_keywords.
        Target: 3-5 queries.
        """
        queries: list[str] = []
        cat_clause = self._build_cat_clause(categories)

        # Cross-combine keywords from different concepts
        if len(concepts) >= 2:
            for i, j in combinations(range(min(len(concepts), 4)), 2):
                kws_a = concepts[i].get("keywords", [])
                kws_b = concepts[j].get("keywords", [])
                if kws_a and kws_b:
                    q = (
                        f'abs:"{self._escape(kws_a[0])}" AND '
                        f'abs:"{self._escape(kws_b[0])}"'
                    )
                    queries.append(q)

                    if len(queries) >= 3:
                        break

        # cross_domain_keywords with categories
        for kw in cross_domain_keywords[:3]:
            q = f'abs:"{self._escape(kw)}"'
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

        Combine title and abstract fields with category.
        Target: 2-4 queries.
        """
        queries: list[str] = []

        for concept in concepts[:3]:
            kws = concept.get("keywords", [])
            if not kws:
                continue

            kw = kws[0]
            for cat in categories[:2]:
                q = (
                    f'ti:"{self._escape(kw)}" AND '
                    f'abs:"{self._escape(kw)}" AND cat:{cat}'
                )
                queries.append(q)

                if len(queries) >= 4:
                    return queries

            # If no categories, title+abstract only
            if not categories:
                q = (
                    f'ti:"{self._escape(kw)}" AND '
                    f'abs:"{self._escape(kw)}"'
                )
                queries.append(q)

                if len(queries) >= 4:
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
                q = f'abs:"{self._escape(kw)}"'
                if cat_clause:
                    q = f"{q} AND {cat_clause}"
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 2: Cross-domain keyword pairs
        if len(cross_domain_keywords) >= 2:
            for kw1, kw2 in combinations(cross_domain_keywords[:6], 2):
                q = (
                    f'abs:"{self._escape(kw1)}" AND '
                    f'abs:"{self._escape(kw2)}"'
                )
                if q not in existing_set:
                    padding.append(q)
                    existing_set.add(q)
                if len(padding) >= needed:
                    return padding

        # Strategy 3: Concept name_en in abstract + category
        for concept in concepts:
            name_en = concept.get("name_en", "")
            if name_en:
                q = f'abs:"{self._escape(name_en)}"'
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
            q = f'abs:"{self._escape(kw)}"'
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
                    q = f'abs:"{self._escape(kw)}" AND cat:{cat}'
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
                        f'abs:"{self._escape(kw)}"'
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
                q = (
                    f'abs:"{self._escape(kw1)}" OR '
                    f'abs:"{self._escape(kw2)}"'
                )
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
