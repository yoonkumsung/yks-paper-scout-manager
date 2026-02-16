"""Abstract base class for paper source adapters.

Each adapter collects papers from a specific source (arXiv, Semantic Scholar,
etc.) and returns them as normalized Paper objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from core.models import Paper


class SourceAdapter(ABC):
    """Abstract base for paper source adapters.

    Each adapter collects papers from a specific source (arXiv, Semantic
    Scholar, etc.) and returns them as normalized Paper objects.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source identifier string (e.g., 'arxiv')."""
        ...

    @abstractmethod
    def collect(
        self,
        agent1_output: dict[str, Any],
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        config: dict[str, Any],
    ) -> list[Paper]:
        """Collect papers from this source.

        Args:
            agent1_output: Keyword expansion result from Agent 1.
                Expected keys: concepts, cross_domain_keywords,
                exclude_keywords.
            categories: Source-specific category list (e.g.,
                arxiv_categories).
            window_start: UTC start of search window (inclusive).
            window_end: UTC end of search window (inclusive).
            config: Source-specific config from config.yaml sources
                section.

        Returns:
            List of normalized Paper objects.
        """
        ...
