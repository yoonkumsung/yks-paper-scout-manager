"""Optional paper clustering by cosine similarity.

Groups papers with cosine similarity >= threshold (default 0.85).
Each cluster has a representative (highest final_score paper).
Display-only: does not affect ranking or scoring.
Disabled when embeddings are not available.

Section reference: TASK-020 from SPEC-PAPER-001.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from core.models import Paper

logger = logging.getLogger(__name__)


class Clusterer:
    """Optional paper clustering by cosine similarity.

    Groups papers with cosine similarity >= threshold (default 0.85).
    Each cluster has a representative (highest final_score paper).
    Display-only: does not affect ranking or scoring.
    Disabled when embeddings are not available.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        """Init with similarity threshold.

        Args:
            threshold: Minimum cosine similarity for papers to be
                       grouped into the same cluster.  Defaults to 0.85.
        """
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        papers: List[Paper],
        embeddings: Optional[np.ndarray] = None,
        scores: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """Group papers into clusters.

        Args:
            papers: Ranked papers list.
            embeddings: Paper embeddings matrix (N, D).
                        None if embeddings disabled.
            scores: paper_key -> final_score mapping for representative
                    selection.

        Returns:
            List of cluster dicts::

                [
                    {
                        "cluster_id": int,
                        "representative_key": str,
                        "member_keys": list[str],
                        "size": int,
                    }
                ]

            Returns empty list if *embeddings* is ``None`` or *papers*
            is empty.
        """
        if embeddings is None or len(papers) == 0:
            return []

        if embeddings.ndim != 2 or embeddings.shape[0] != len(papers):
            logger.warning(
                "Clusterer: embeddings shape %s does not match %d papers. "
                "Skipping clustering.",
                embeddings.shape, len(papers),
            )
            return []

        n = len(papers)

        # Compute pairwise cosine similarity matrix
        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Greedy clustering
        assigned = [False] * n
        raw_clusters: List[List[int]] = []

        for i in range(n):
            if assigned[i]:
                continue
            # Start a new cluster with paper i
            cluster_indices = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if sim_matrix[i, j] >= self._threshold:
                    cluster_indices.append(j)
                    assigned[j] = True

            raw_clusters.append(cluster_indices)

        # Build result dicts
        result: List[dict] = []
        for cluster_id, indices in enumerate(raw_clusters):
            member_keys = [papers[idx].paper_key for idx in indices]

            # Select representative: highest final_score, or first paper
            representative_key = self._select_representative(
                indices, papers, scores
            )

            result.append(
                {
                    "cluster_id": cluster_id,
                    "representative_key": representative_key,
                    "member_keys": member_keys,
                    "size": len(member_keys),
                }
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            embeddings: (N, D) matrix of paper embeddings.

        Returns:
            (N, N) similarity matrix with values in [0, 1].
        """
        # Compute L2 norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        # Dot product of normalized vectors gives cosine similarity
        sim = np.dot(normalized, normalized.T)

        # Clamp to [0, 1] (numerical precision can cause slight negatives)
        np.clip(sim, 0.0, 1.0, out=sim)

        return sim

    @staticmethod
    def _select_representative(
        indices: List[int],
        papers: List[Paper],
        scores: Optional[Dict[str, float]],
    ) -> str:
        """Select the representative paper for a cluster.

        The representative is the paper with the highest final_score.
        If scores is None, the first paper in the cluster is selected.

        Args:
            indices: List of paper indices in this cluster.
            papers: Full papers list.
            scores: paper_key -> final_score mapping, or None.

        Returns:
            paper_key of the representative paper.
        """
        if scores is None:
            return papers[indices[0]].paper_key

        best_idx = indices[0]
        best_score = scores.get(papers[best_idx].paper_key, -1.0)

        for idx in indices[1:]:
            s = scores.get(papers[idx].paper_key, -1.0)
            if s > best_score:
                best_score = s
                best_idx = idx

        return papers[best_idx].paper_key
