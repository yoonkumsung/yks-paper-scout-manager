"""Tests for core.clustering.clusterer.Clusterer.

TASK-020 from SPEC-PAPER-001: Optional clustering module for Paper Scout.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.clustering.clusterer import Clusterer
from core.models import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(key: str) -> Paper:
    """Create a minimal Paper for testing."""
    return Paper(
        source="arxiv",
        native_id=key,
        paper_key=f"arxiv:{key}",
        url=f"https://arxiv.org/abs/{key}",
        title=f"Paper {key}",
        abstract=f"Abstract for paper {key}.",
        authors=["Author A"],
        categories=["cs.AI"],
        published_at_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClustererBasic:
    """Basic clustering behaviour."""

    def test_basic_clustering_two_similar_papers(self) -> None:
        """Two papers with cosine >= 0.85 should be in the same cluster."""
        papers = [_make_paper("001"), _make_paper("002")]
        # Two nearly identical embeddings
        emb = np.array([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]])
        scores = {"arxiv:001": 80.0, "arxiv:002": 70.0}

        clusterer = Clusterer(threshold=0.85)
        clusters = clusterer.cluster(papers, emb, scores)

        assert len(clusters) == 1
        assert clusters[0]["size"] == 2
        assert set(clusters[0]["member_keys"]) == {"arxiv:001", "arxiv:002"}

    def test_multiple_clusters(self) -> None:
        """4 papers forming 2 distinct clusters."""
        papers = [_make_paper(f"00{i}") for i in range(1, 5)]
        # Cluster A: papers 0 and 1 (similar), Cluster B: papers 2 and 3 (similar)
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.05],
        ])
        scores = {
            "arxiv:001": 90.0,
            "arxiv:002": 80.0,
            "arxiv:003": 70.0,
            "arxiv:004": 60.0,
        }

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 2
        # Check that papers are properly grouped
        cluster_keys = [set(c["member_keys"]) for c in clusters]
        assert {"arxiv:001", "arxiv:002"} in cluster_keys
        assert {"arxiv:003", "arxiv:004"} in cluster_keys

    def test_no_clustering_all_dissimilar(self) -> None:
        """All papers dissimilar (cosine < 0.85) -> each in own cluster."""
        papers = [_make_paper(f"00{i}") for i in range(1, 4)]
        # Orthogonal embeddings
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        scores = {"arxiv:001": 90.0, "arxiv:002": 80.0, "arxiv:003": 70.0}

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 3
        for c in clusters:
            assert c["size"] == 1

    def test_single_paper(self) -> None:
        """Single paper should produce a single cluster with 1 member."""
        papers = [_make_paper("001")]
        emb = np.array([[1.0, 0.0, 0.0]])
        scores = {"arxiv:001": 95.0}

        clusters = Clusterer().cluster(papers, emb, scores)

        assert len(clusters) == 1
        assert clusters[0]["size"] == 1
        assert clusters[0]["member_keys"] == ["arxiv:001"]
        assert clusters[0]["representative_key"] == "arxiv:001"


class TestClustererRepresentative:
    """Representative selection tests."""

    def test_representative_selection_highest_score(self) -> None:
        """Highest-score paper should be selected as representative."""
        papers = [_make_paper("001"), _make_paper("002"), _make_paper("003")]
        # All very similar -> one cluster
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.98, 0.1, 0.0],
        ])
        scores = {"arxiv:001": 70.0, "arxiv:002": 95.0, "arxiv:003": 80.0}

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 1
        assert clusters[0]["representative_key"] == "arxiv:002"

    def test_no_scores_defaults_to_first_paper(self) -> None:
        """When scores is None, representative defaults to first paper in cluster."""
        papers = [_make_paper("001"), _make_paper("002")]
        emb = np.array([[1.0, 0.0, 0.0], [0.99, 0.05, 0.0]])

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores=None)

        assert len(clusters) == 1
        # First paper encountered is representative when no scores
        assert clusters[0]["representative_key"] == "arxiv:001"


class TestClustererThreshold:
    """Threshold boundary tests."""

    def test_threshold_boundary_included(self) -> None:
        """Papers at exactly threshold (0.85) should be included in cluster."""
        papers = [_make_paper("001"), _make_paper("002")]
        # Construct vectors with cosine similarity exactly 0.85
        v1 = np.array([1.0, 0.0])
        # cos(theta) = 0.85 => theta ~ 31.79 degrees
        angle = np.arccos(0.85)
        v2 = np.array([np.cos(angle), np.sin(angle)])
        emb = np.vstack([v1, v2])
        scores = {"arxiv:001": 80.0, "arxiv:002": 70.0}

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 1
        assert clusters[0]["size"] == 2

    def test_threshold_boundary_excluded(self) -> None:
        """Papers at 0.849 similarity should NOT be in the same cluster."""
        papers = [_make_paper("001"), _make_paper("002")]
        # cos(theta) = 0.849 < 0.85
        angle = np.arccos(0.849)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(angle), np.sin(angle)])
        emb = np.vstack([v1, v2])
        scores = {"arxiv:001": 80.0, "arxiv:002": 70.0}

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 2

    def test_custom_threshold_lower(self) -> None:
        """Lower threshold (0.7) should produce larger clusters."""
        papers = [_make_paper(f"00{i}") for i in range(1, 5)]
        # Papers with moderate similarity (~0.75) -> separate at 0.85, grouped at 0.7
        angle = np.arccos(0.75)
        emb = np.array([
            [1.0, 0.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.99],
        ])
        scores = {
            "arxiv:001": 90.0,
            "arxiv:002": 80.0,
            "arxiv:003": 70.0,
            "arxiv:004": 60.0,
        }

        # At 0.85 threshold: papers 0 and 1 NOT grouped (cos=0.75 < 0.85)
        clusters_strict = Clusterer(threshold=0.85).cluster(papers, emb, scores)
        # At 0.7 threshold: papers 0 and 1 ARE grouped (cos=0.75 >= 0.7)
        clusters_loose = Clusterer(threshold=0.7).cluster(papers, emb, scores)

        # Loose threshold should yield fewer clusters
        assert len(clusters_loose) < len(clusters_strict)


class TestClustererEdgeCases:
    """Edge cases and disabled mode."""

    def test_embeddings_none_returns_empty(self) -> None:
        """Returns empty list when embeddings are None (disabled mode)."""
        papers = [_make_paper("001"), _make_paper("002")]
        scores = {"arxiv:001": 80.0, "arxiv:002": 70.0}

        clusters = Clusterer().cluster(papers, embeddings=None, scores=scores)

        assert clusters == []

    def test_empty_papers_returns_empty(self) -> None:
        """Returns empty list when papers list is empty."""
        emb = np.array([]).reshape(0, 3)

        clusters = Clusterer().cluster(papers=[], embeddings=emb)

        assert clusters == []

    def test_all_identical_embeddings_one_cluster(self) -> None:
        """All papers with identical embeddings should form one big cluster."""
        papers = [_make_paper(f"00{i}") for i in range(1, 6)]
        # All identical embeddings
        emb = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
        scores = {
            "arxiv:001": 50.0,
            "arxiv:002": 90.0,
            "arxiv:003": 70.0,
            "arxiv:004": 60.0,
            "arxiv:005": 80.0,
        }

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        assert len(clusters) == 1
        assert clusters[0]["size"] == 5
        # Highest score paper is representative
        assert clusters[0]["representative_key"] == "arxiv:002"

    def test_cluster_size_correct(self) -> None:
        """Cluster size field matches actual member_keys length."""
        papers = [_make_paper(f"00{i}") for i in range(1, 4)]
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ])
        scores = {"arxiv:001": 90.0, "arxiv:002": 80.0, "arxiv:003": 70.0}

        clusters = Clusterer(threshold=0.85).cluster(papers, emb, scores)

        for c in clusters:
            assert c["size"] == len(c["member_keys"])

    def test_cluster_id_is_sequential(self) -> None:
        """Cluster IDs should be sequential starting from 0."""
        papers = [_make_paper(f"00{i}") for i in range(1, 4)]
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        clusters = Clusterer(threshold=0.85).cluster(papers, emb)

        cluster_ids = [c["cluster_id"] for c in clusters]
        assert cluster_ids == list(range(len(clusters)))
