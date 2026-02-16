"""Comprehensive tests for EmbeddingRanker (TASK-019).

All tests work WITHOUT sentence-transformers installed.
Uses numpy directly for mathematical tests and mocks for
integration behavior tests.

Covers:
  - Unavailable mode: available=False, mode="disabled"
  - Graceful fallback: compute_similarity returns [] when unavailable
  - Cosine similarity calculation with numpy arrays
  - Clamping: negative cosine values clamped to 0
  - Cache hit: cached embedding returned without recomputation
  - Cache miss: new embedding computed and cached
  - Cache invalidation: different hash -> recompute
  - Paper text construction: title + " " + abstract
  - Empty papers: returns empty list
  - Multiple papers: correct number of scores returned
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test Paper dataclass (avoids importing core.models in edge cases)
# ---------------------------------------------------------------------------


@dataclass
class _FakePaper:
    """Minimal Paper stand-in for tests."""

    source: str = "arxiv"
    native_id: str = "2401.00001"
    paper_key: str = "arxiv:2401.00001"
    url: str = "http://arxiv.org/abs/2401.00001"
    title: str = "Test Paper Title"
    abstract: str = "This is a test abstract about machine learning."
    authors: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    published_at_utc: Optional[datetime] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = ["Author A"]
        if self.categories is None:
            self.categories = ["cs.AI"]
        if self.published_at_utc is None:
            self.published_at_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary directory for cache files."""
    return str(tmp_path)


@pytest.fixture
def ranker_unavailable(tmp_cache_dir):
    """Create an EmbeddingRanker with _AVAILABLE forced to False."""
    with patch(
        "core.embeddings.embedding_ranker._AVAILABLE", False
    ):
        from core.embeddings.embedding_ranker import EmbeddingRanker

        yield EmbeddingRanker(cache_dir=tmp_cache_dir)


@pytest.fixture
def ranker_available(tmp_cache_dir):
    """Create an EmbeddingRanker with _AVAILABLE forced to True."""
    with patch(
        "core.embeddings.embedding_ranker._AVAILABLE", True
    ):
        from core.embeddings.embedding_ranker import EmbeddingRanker

        yield EmbeddingRanker(cache_dir=tmp_cache_dir)


@pytest.fixture
def sample_papers():
    """Return a list of sample Paper-like objects."""
    return [
        _FakePaper(
            title="Deep Learning for NLP",
            abstract="We propose a novel transformer architecture.",
        ),
        _FakePaper(
            native_id="2401.00002",
            paper_key="arxiv:2401.00002",
            title="Reinforcement Learning in Robotics",
            abstract="This paper studies RL applications in robotic control.",
        ),
        _FakePaper(
            native_id="2401.00003",
            paper_key="arxiv:2401.00003",
            title="Graph Neural Networks",
            abstract="We introduce a new GNN layer for molecular properties.",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Unavailable mode tests
# ---------------------------------------------------------------------------


class TestUnavailableMode:
    """Tests for behavior when sentence-transformers is not installed."""

    def test_available_returns_false(self, ranker_unavailable):
        """available property returns False when import fails."""
        assert ranker_unavailable.available is False

    def test_mode_returns_disabled(self, ranker_unavailable):
        """mode property returns 'disabled' when unavailable."""
        assert ranker_unavailable.mode == "disabled"


# ---------------------------------------------------------------------------
# 2. Graceful fallback tests
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    """Tests for graceful degradation without sentence-transformers."""

    def test_compute_similarity_returns_empty(
        self, ranker_unavailable, sample_papers
    ):
        """compute_similarity returns [] when unavailable."""
        result = ranker_unavailable.compute_similarity(
            "machine learning", sample_papers
        )
        assert result == []

    def test_compute_similarity_no_exception(
        self, ranker_unavailable, sample_papers
    ):
        """compute_similarity does not raise when unavailable."""
        # Should not raise any exception
        ranker_unavailable.compute_similarity(
            "deep learning transformers", sample_papers
        )

    def test_get_topic_embedding_raises_when_unavailable(
        self, ranker_unavailable
    ):
        """get_topic_embedding raises RuntimeError when unavailable."""
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            ranker_unavailable.get_topic_embedding("test", "hash123")

    def test_compute_paper_embeddings_raises_when_unavailable(
        self, ranker_unavailable, sample_papers
    ):
        """compute_paper_embeddings raises RuntimeError when unavailable."""
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            ranker_unavailable.compute_paper_embeddings(sample_papers)


# ---------------------------------------------------------------------------
# 3. Cosine similarity calculation tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the cosine_similarity static method using numpy."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert EmbeddingRanker.cosine_similarity(a, b) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert EmbeddingRanker.cosine_similarity(a, b) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = EmbeddingRanker.cosine_similarity(a, b)
        assert 0.99 < sim <= 1.0

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        assert EmbeddingRanker.cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        """Two zero vectors should return 0.0 similarity."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        assert EmbeddingRanker.cosine_similarity(a, b) == 0.0

    def test_result_is_float(self):
        """Result should be a Python float."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        result = EmbeddingRanker.cosine_similarity(a, b)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 4. Clamping tests
# ---------------------------------------------------------------------------


class TestClamping:
    """Tests for clamping cosine similarity to [0, 1]."""

    def test_negative_cosine_clamped_to_zero(self):
        """Negative cosine similarity is clamped to 0."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        # True cosine is -1.0, should be clamped to 0.0
        sim = EmbeddingRanker.cosine_similarity(a, b)
        assert sim == 0.0

    def test_partially_negative_clamped(self):
        """Partially negative cosine is clamped to 0."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 0.5])
        b = np.array([-1.0, -0.5])
        # True cosine is -1.0, should be clamped to 0.0
        sim = EmbeddingRanker.cosine_similarity(a, b)
        assert sim == 0.0

    def test_positive_not_clamped(self):
        """Positive cosine below 1.0 is not clamped."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        a = np.array([1.0, 1.0])
        b = np.array([1.0, 0.0])
        sim = EmbeddingRanker.cosine_similarity(a, b)
        # cos(45 degrees) ~ 0.7071
        assert 0.0 < sim < 1.0
        assert sim == pytest.approx(0.7071, abs=0.01)


# ---------------------------------------------------------------------------
# 5. Cache hit tests
# ---------------------------------------------------------------------------


class TestCacheHit:
    """Tests for topic embedding cache hit behavior."""

    def test_cache_hit_returns_cached_embedding(
        self, ranker_available, tmp_cache_dir
    ):
        """Cached embedding is returned without recomputation."""
        cache_hash = "test_hash_123"
        cached_embedding = np.array([0.1, 0.2, 0.3, 0.4])

        # Pre-populate cache file
        cache_file = os.path.join(tmp_cache_dir, "topic_embeddings.npy")
        np.save(cache_file, {cache_hash: cached_embedding})

        # Mock _get_model so we can verify it is NOT called
        with patch.object(
            ranker_available, "_get_model"
        ) as mock_model:
            result = ranker_available.get_topic_embedding(
                "some topic text", cache_hash
            )

            # Model should not be called on cache hit
            mock_model.assert_not_called()

            # Result should match cached embedding
            np.testing.assert_array_almost_equal(
                result, cached_embedding
            )


# ---------------------------------------------------------------------------
# 6. Cache miss tests
# ---------------------------------------------------------------------------


class TestCacheMiss:
    """Tests for topic embedding cache miss behavior."""

    def test_cache_miss_computes_and_saves(
        self, ranker_available, tmp_cache_dir
    ):
        """New embedding is computed and cached on miss."""
        cache_hash = "new_hash_456"
        fake_embedding = np.array([0.5, 0.6, 0.7])

        # Create a mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embedding

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            result = ranker_available.get_topic_embedding(
                "novel topic text", cache_hash
            )

            # Model should have been called
            mock_model.encode.assert_called_once_with(
                "novel topic text", convert_to_numpy=True
            )

            # Result should match computed embedding
            np.testing.assert_array_almost_equal(result, fake_embedding)

            # Cache file should exist
            cache_file = os.path.join(
                tmp_cache_dir, "topic_embeddings.npy"
            )
            assert os.path.exists(cache_file)

            # Verify cache content
            loaded = np.load(cache_file, allow_pickle=True).item()
            assert cache_hash in loaded
            np.testing.assert_array_almost_equal(
                loaded[cache_hash], fake_embedding
            )


# ---------------------------------------------------------------------------
# 7. Cache invalidation tests
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """Tests for cache invalidation on hash change."""

    def test_different_hash_triggers_recompute(
        self, ranker_available, tmp_cache_dir
    ):
        """Different cache_hash triggers recomputation."""
        old_hash = "old_hash_aaa"
        new_hash = "new_hash_bbb"
        old_embedding = np.array([0.1, 0.2])
        new_embedding = np.array([0.9, 0.8])

        # Pre-populate cache with old hash
        cache_file = os.path.join(tmp_cache_dir, "topic_embeddings.npy")
        np.save(cache_file, {old_hash: old_embedding})

        # Mock model to return new embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = new_embedding

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            result = ranker_available.get_topic_embedding(
                "updated topic", new_hash
            )

            # Model should be called (cache miss for new hash)
            mock_model.encode.assert_called_once()

            # Result should be the new embedding
            np.testing.assert_array_almost_equal(result, new_embedding)

            # Cache should now contain both old and new
            loaded = np.load(cache_file, allow_pickle=True).item()
            assert old_hash in loaded
            assert new_hash in loaded

    def test_same_hash_uses_cache(
        self, ranker_available, tmp_cache_dir
    ):
        """Same cache_hash returns cached value without model call."""
        cache_hash = "same_hash_ccc"
        cached_embedding = np.array([0.3, 0.4])

        # Pre-populate cache
        cache_file = os.path.join(tmp_cache_dir, "topic_embeddings.npy")
        np.save(cache_file, {cache_hash: cached_embedding})

        mock_model = MagicMock()

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            result = ranker_available.get_topic_embedding(
                "same topic", cache_hash
            )

            # Model should NOT be called
            mock_model.encode.assert_not_called()

            np.testing.assert_array_almost_equal(
                result, cached_embedding
            )


# ---------------------------------------------------------------------------
# 8. Paper text construction tests
# ---------------------------------------------------------------------------


class TestPaperTextConstruction:
    """Tests for paper text being title + ' ' + abstract."""

    def test_paper_text_combines_title_and_abstract(
        self, ranker_available, tmp_cache_dir
    ):
        """Papers are encoded as 'title abstract'."""
        papers = [
            _FakePaper(
                title="My Title",
                abstract="My Abstract",
            ),
            _FakePaper(
                native_id="2401.00002",
                paper_key="arxiv:2401.00002",
                title="Second Title",
                abstract="Second Abstract",
            ),
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            ranker_available.compute_paper_embeddings(papers)

            # Verify the texts passed to encode
            call_args = mock_model.encode.call_args
            texts = call_args[0][0]
            assert texts == [
                "My Title My Abstract",
                "Second Title Second Abstract",
            ]


# ---------------------------------------------------------------------------
# 9. Empty papers tests
# ---------------------------------------------------------------------------


class TestEmptyPapers:
    """Tests for empty paper list input."""

    def test_empty_papers_returns_empty_list_unavailable(
        self, ranker_unavailable
    ):
        """Empty list input returns empty list when unavailable."""
        result = ranker_unavailable.compute_similarity("topic", [])
        assert result == []

    def test_empty_papers_returns_empty_list_available(
        self, ranker_available
    ):
        """Empty list input returns empty list when available."""
        result = ranker_available.compute_similarity("topic", [])
        assert result == []


# ---------------------------------------------------------------------------
# 10. Multiple papers tests
# ---------------------------------------------------------------------------


class TestMultiplePapers:
    """Tests for correct number of scores with multiple papers."""

    def test_returns_correct_count(
        self, ranker_available, tmp_cache_dir, sample_papers
    ):
        """Number of scores matches number of papers."""
        topic_text = "machine learning and deep neural networks"

        # Mock model
        dim = 4
        mock_model = MagicMock()

        # First call: topic embedding
        # Subsequent call: paper embeddings (3 papers)
        topic_emb = np.array([0.5, 0.5, 0.5, 0.5])
        paper_embs = np.array(
            [
                [0.4, 0.5, 0.6, 0.7],
                [0.1, 0.2, 0.3, 0.4],
                [0.8, 0.9, 0.7, 0.6],
            ]
        )

        # encode is called twice: once for topic, once for papers
        mock_model.encode.side_effect = [topic_emb, paper_embs]

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            scores = ranker_available.compute_similarity(
                topic_text, sample_papers
            )

            assert len(scores) == len(sample_papers)
            assert len(scores) == 3

    def test_all_scores_in_range(
        self, ranker_available, tmp_cache_dir, sample_papers
    ):
        """All scores should be in [0, 1] range."""
        topic_emb = np.array([0.5, 0.5, 0.5])
        paper_embs = np.array(
            [
                [0.4, 0.5, 0.6],
                [-0.1, -0.2, -0.3],  # negative -> clamped to 0
                [0.8, 0.9, 0.7],
            ]
        )

        mock_model = MagicMock()
        mock_model.encode.side_effect = [topic_emb, paper_embs]

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            scores = ranker_available.compute_similarity(
                "test topic", sample_papers
            )

            for score in scores:
                assert 0.0 <= score <= 1.0
                assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 11. Available mode properties tests
# ---------------------------------------------------------------------------


class TestAvailableMode:
    """Tests for properties when sentence-transformers IS available."""

    def test_available_returns_true(self, ranker_available):
        """available returns True when _AVAILABLE is True."""
        assert ranker_available.available is True

    def test_mode_returns_en_synthetic(self, ranker_available):
        """mode returns 'en_synthetic' when available."""
        assert ranker_available.mode == "en_synthetic"


# ---------------------------------------------------------------------------
# 12. Cache file management tests
# ---------------------------------------------------------------------------


class TestCacheFileManagement:
    """Tests for cache file creation and directory handling."""

    def test_cache_dir_created_on_save(self, tmp_path):
        """Cache directory is created if it does not exist."""
        nested_dir = str(tmp_path / "nested" / "cache")

        with patch(
            "core.embeddings.embedding_ranker._AVAILABLE", True
        ):
            from core.embeddings.embedding_ranker import EmbeddingRanker

            ranker = EmbeddingRanker(cache_dir=nested_dir)

            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch.object(
                ranker, "_get_model", return_value=mock_model
            ):
                ranker.get_topic_embedding("test", "hash")

            assert os.path.exists(nested_dir)
            assert os.path.exists(
                os.path.join(nested_dir, "topic_embeddings.npy")
            )

    def test_corrupt_cache_handled_gracefully(
        self, ranker_available, tmp_cache_dir
    ):
        """Corrupt cache file is handled without crash."""
        cache_file = os.path.join(tmp_cache_dir, "topic_embeddings.npy")

        # Write garbage data
        with open(cache_file, "w") as f:
            f.write("not valid npy data")

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2])

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            # Should not raise, should recompute
            result = ranker_available.get_topic_embedding(
                "test", "hash_corrupt"
            )
            np.testing.assert_array_almost_equal(
                result, np.array([0.1, 0.2])
            )

    def test_missing_cache_file_returns_none(
        self, ranker_available, tmp_cache_dir
    ):
        """Missing cache file is treated as cache miss."""
        cache_file = os.path.join(tmp_cache_dir, "topic_embeddings.npy")
        assert not os.path.exists(cache_file)

        loaded = ranker_available._load_topic_cache()
        assert loaded is None


# ---------------------------------------------------------------------------
# 13. Compute cache hash tests
# ---------------------------------------------------------------------------


class TestComputeCacheHash:
    """Tests for the _compute_cache_hash method."""

    def test_hash_is_sha256(self, tmp_cache_dir):
        """Cache hash is a SHA-256 hex digest."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        ranker = EmbeddingRanker(cache_dir=tmp_cache_dir)
        text = "machine learning topic"
        result = ranker._compute_cache_hash(text)

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_different_text_different_hash(self, tmp_cache_dir):
        """Different text produces different hash."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        ranker = EmbeddingRanker(cache_dir=tmp_cache_dir)
        h1 = ranker._compute_cache_hash("topic A")
        h2 = ranker._compute_cache_hash("topic B")
        assert h1 != h2

    def test_same_text_same_hash(self, tmp_cache_dir):
        """Same text produces same hash."""
        from core.embeddings.embedding_ranker import EmbeddingRanker

        ranker = EmbeddingRanker(cache_dir=tmp_cache_dir)
        h1 = ranker._compute_cache_hash("topic X")
        h2 = ranker._compute_cache_hash("topic X")
        assert h1 == h2


# ---------------------------------------------------------------------------
# 14. Integration-style test (mocked model)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration test with mocked model."""

    def test_full_pipeline(
        self, ranker_available, tmp_cache_dir, sample_papers
    ):
        """Full compute_similarity pipeline with mocked model."""
        topic_text = "neural network optimization"

        dim = 384  # all-MiniLM-L6-v2 dimension
        rng = np.random.RandomState(42)
        topic_emb = rng.randn(dim).astype(np.float32)
        paper_embs = rng.randn(len(sample_papers), dim).astype(
            np.float32
        )

        mock_model = MagicMock()
        mock_model.encode.side_effect = [topic_emb, paper_embs]

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            scores = ranker_available.compute_similarity(
                topic_text, sample_papers
            )

        # Verify output structure
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_exception_in_model_returns_empty(
        self, ranker_available, sample_papers
    ):
        """Exception during model.encode returns empty list."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("model error")

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            scores = ranker_available.compute_similarity(
                "test", sample_papers
            )
            assert scores == []

    def test_2d_topic_embedding_squeezed(
        self, ranker_available, tmp_cache_dir
    ):
        """2D topic embedding (1, D) is squeezed to (D,)."""
        cache_hash = "squeeze_test"
        # Model returns 2D array
        fake_2d = np.array([[0.1, 0.2, 0.3]])

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_2d

        with patch.object(
            ranker_available, "_get_model", return_value=mock_model
        ):
            result = ranker_available.get_topic_embedding(
                "test", cache_hash
            )
            assert result.ndim == 1
            np.testing.assert_array_almost_equal(
                result, np.array([0.1, 0.2, 0.3])
            )
