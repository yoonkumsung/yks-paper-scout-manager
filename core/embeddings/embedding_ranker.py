"""Optional embedding-based similarity scorer for Paper Scout.

Uses sentence-transformers/all-MiniLM-L6-v2 for computing cosine
similarity between topic description and paper text.

Gracefully handles missing dependencies (sentence-transformers, torch).

Section reference: TASK-019 from SPEC-PAPER-001.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from core.models import Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

# Default model for semantic similarity
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRanker:
    """Optional embedding-based similarity scorer.

    Uses sentence-transformers/all-MiniLM-L6-v2 for computing cosine
    similarity between topic description and paper text.

    Gracefully handles missing dependencies (sentence-transformers, torch).
    """

    def __init__(self, cache_dir: str = "data") -> None:
        """Initialize with optional dependency check.

        Args:
            cache_dir: Directory for caching topic embeddings.
                       Defaults to ``"data"``.
        """
        self._cache_dir = cache_dir
        self._model: Optional[object] = None  # lazy-loaded SentenceTransformer
        self._cache_file = os.path.join(cache_dir, "topic_embeddings.npy")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if sentence-transformers is importable."""
        return _AVAILABLE

    @property
    def mode(self) -> str:
        """Return 'en_synthetic' if available, 'disabled' otherwise."""
        return "en_synthetic" if _AVAILABLE else "disabled"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_similarity(
        self, topic_text: str, papers: List["Paper"]
    ) -> List[float]:
        """Compute cosine similarity between topic and each paper.

        Args:
            topic_text: The topic embedding text (from Agent 1 output).
            papers: List of Paper objects to score.

        Returns:
            List of float scores in [0, 1], one per paper.
            Returns empty list if sentence-transformers is not available
            or if papers is empty.
        """
        if not self.available:
            logger.debug("EmbeddingRanker: sentence-transformers not available")
            return []

        if not papers:
            return []

        try:
            # Compute topic embedding
            cache_hash = self._compute_cache_hash(topic_text)
            topic_emb = self.get_topic_embedding(topic_text, cache_hash)

            # Compute paper embeddings
            paper_embs = self.compute_paper_embeddings(papers)

            # Compute similarities
            scores: List[float] = []
            for i in range(len(papers)):
                sim = self.cosine_similarity(topic_emb, paper_embs[i])
                scores.append(sim)

            return scores
        except Exception:
            logger.exception("EmbeddingRanker: error computing similarity")
            return []

    def get_topic_embedding(
        self, topic_text: str, cache_hash: str
    ) -> np.ndarray:
        """Get or compute the topic embedding, with caching.

        Args:
            topic_text: The topic embedding text string.
            cache_hash: Hash-based cache key (same pattern as Agent 1).

        Returns:
            Embedding vector as numpy array of shape (D,).

        Raises:
            RuntimeError: If sentence-transformers is not available.
        """
        if not self.available:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "cannot compute embeddings"
            )

        # Try loading from cache
        cached = self._load_topic_cache()
        if cached is not None and cache_hash in cached:
            logger.debug("EmbeddingRanker: topic embedding cache hit")
            return cached[cache_hash]

        # Compute embedding
        model = self._get_model()
        embedding = model.encode(topic_text, convert_to_numpy=True)
        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        # Save to cache
        if cached is None:
            cached = {}
        cached[cache_hash] = embedding
        self._save_topic_cache(cached)

        return embedding

    def compute_paper_embeddings(
        self, papers: List["Paper"]
    ) -> np.ndarray:
        """Compute embeddings for paper title + abstract.

        Args:
            papers: List of Paper objects.

        Returns:
            Numpy array of shape (N, D) with one embedding per paper.

        Raises:
            RuntimeError: If sentence-transformers is not available.
        """
        if not self.available:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "cannot compute embeddings"
            )

        texts = [
            (p.title + " " + p.abstract) for p in papers
        ]
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector (1-D numpy array).
            b: Second vector (1-D numpy array).

        Returns:
            Cosine similarity clamped to [0, 1].
        """
        a_flat = a.flatten()
        b_flat = b.flatten()

        dot = float(np.dot(a_flat, b_flat))
        norm_a = float(np.linalg.norm(a_flat))
        norm_b = float(np.linalg.norm(b_flat))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        sim = dot / (norm_a * norm_b)

        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, sim)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> object:
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def _compute_cache_hash(self, topic_text: str) -> str:
        """Compute SHA-256 hash for cache key.

        Uses the same hashing approach as Agent 1 (KeywordExpander).
        """
        return hashlib.sha256(topic_text.encode("utf-8")).hexdigest()

    def _load_topic_cache(self) -> Optional[dict]:
        """Load topic embeddings cache from .npy file.

        Returns:
            Dictionary mapping cache_hash -> embedding_vector,
            or None if file does not exist or is corrupt.
        """
        try:
            if not os.path.exists(self._cache_file):
                return None
            data = np.load(self._cache_file, allow_pickle=True)
            cache_dict = data.item()
            if isinstance(cache_dict, dict):
                return cache_dict
            return None
        except Exception:
            logger.debug(
                "EmbeddingRanker: failed to load topic cache from %s",
                self._cache_file,
            )
            return None

    def _save_topic_cache(self, cache: dict) -> None:
        """Save topic embeddings cache to .npy file.

        Creates parent directories if needed.
        """
        try:
            cache_dir = os.path.dirname(self._cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            np.save(self._cache_file, cache)
        except Exception:
            logger.exception(
                "EmbeddingRanker: failed to save topic cache to %s",
                self._cache_file,
            )
