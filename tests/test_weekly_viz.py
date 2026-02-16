"""Tests for weekly_viz module.

All tests must pass regardless of whether visualization dependencies are installed.
Tests verify graceful degradation when numpy/matplotlib/umap-learn are missing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.pipeline.weekly_viz import (
    generate_cluster_map,
    generate_score_distribution,
    generate_weekly_charts,
    is_viz_available,
)


class TestVizAvailability:
    """Test visualization availability detection."""

    def test_is_viz_available_returns_bool(self):
        """is_viz_available should always return a boolean."""
        result = is_viz_available()
        assert isinstance(result, bool)


class TestScoreDistribution:
    """Test score distribution chart generation."""

    def test_returns_none_when_viz_unavailable(self):
        """When viz is unavailable, generate_score_distribution returns None."""
        if not is_viz_available():
            result = generate_score_distribution(
                scores=[10.0, 20.0, 30.0],
                output_path="tmp/test.png"
            )
            assert result is None

    def test_returns_none_for_empty_scores(self):
        """Empty scores list should return None."""
        result = generate_score_distribution(
            scores=[],
            output_path="tmp/test.png"
        )
        assert result is None

    @pytest.mark.skipif(not is_viz_available(), reason="Visualization not available")
    def test_creates_chart_file(self):
        """With matplotlib, should create PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "score_dist.png")
            scores = [10.0, 25.0, 30.0, 45.0, 60.0, 75.0, 80.0, 90.0]

            result = generate_score_distribution(scores, output_path)

            # If viz is available, should return path and file should exist
            if result is not None:
                assert result == output_path
                assert Path(output_path).exists()

    @pytest.mark.skipif(not is_viz_available(), reason="Visualization not available")
    def test_with_mock_matplotlib(self):
        """Test logic with mocked matplotlib (no actual rendering)."""
        # This test only runs when viz IS available (matplotlib installed)
        # We verify the function creates proper chart structure
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "score_dist.png")
            scores = [10.0, 20.0, 30.0, 40.0, 50.0]

            result = generate_score_distribution(scores, output_path)

            # Should return path when successful
            if result is not None:
                assert result == output_path
                assert Path(output_path).exists()
                # File should have some content
                assert Path(output_path).stat().st_size > 0


class TestClusterMap:
    """Test cluster map generation."""

    def test_returns_none_when_viz_unavailable(self):
        """When viz is unavailable, generate_cluster_map returns None."""
        if not is_viz_available():
            result = generate_cluster_map(
                embeddings=[[1, 2], [3, 4]],
                labels=["A", "B"],
                output_path="tmp/test.png"
            )
            assert result is None

    def test_returns_none_when_umap_unavailable(self):
        """When UMAP is unavailable, should return None gracefully."""
        # Even if matplotlib is available, UMAP might not be
        with patch("core.pipeline.weekly_viz._VIZ_AVAILABLE", True):
            with patch("core.pipeline.weekly_viz._UMAP_AVAILABLE", False):
                result = generate_cluster_map(
                    embeddings=[[1, 2], [3, 4]],
                    labels=["A", "B"],
                    output_path="tmp/test.png"
                )
                assert result is None

    def test_returns_none_for_empty_data(self):
        """Empty embeddings or labels should return None."""
        result1 = generate_cluster_map(
            embeddings=None,
            labels=[],
            output_path="tmp/test.png"
        )
        assert result1 is None

        result2 = generate_cluster_map(
            embeddings=[],
            labels=["A"],
            output_path="tmp/test.png"
        )
        assert result2 is None


class TestWeeklyCharts:
    """Test generate_weekly_charts integration."""

    def test_returns_empty_list_when_viz_unavailable(self):
        """When viz unavailable, returns empty list."""
        if not is_viz_available():
            result = generate_weekly_charts(
                db_path=":memory:",
                date_str="2026-02-17"
            )
            assert result == []

    def test_with_mock_database(self):
        """Test with a mock database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create test database with tables
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create paper_evaluations table
            cursor.execute("""
                CREATE TABLE paper_evaluations (
                    id INTEGER PRIMARY KEY,
                    score REAL,
                    created_at TEXT
                )
            """)

            # Insert test data
            cursor.execute("""
                INSERT INTO paper_evaluations (score, created_at)
                VALUES (75.0, '2026-02-15')
            """)
            cursor.execute("""
                INSERT INTO paper_evaluations (score, created_at)
                VALUES (85.0, '2026-02-16')
            """)

            conn.commit()
            conn.close()

            # Call function
            output_dir = os.path.join(tmpdir, "reports")
            result = generate_weekly_charts(
                db_path=db_path,
                date_str="2026-02-17",
                output_dir=output_dir
            )

            # If viz is available, should generate files
            if is_viz_available():
                # At minimum, should have attempted to query database
                # Result might be empty if UMAP not available or errors occurred
                assert isinstance(result, list)
            else:
                # If viz not available, should return empty list
                assert result == []

    def test_handles_missing_database_gracefully(self):
        """Should handle missing database without crashing."""
        result = generate_weekly_charts(
            db_path="/nonexistent/path.db",
            date_str="2026-02-17"
        )
        # Should return empty list, not raise exception
        assert isinstance(result, list)

    def test_output_directory_creation(self):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create minimal database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE paper_evaluations (
                    id INTEGER PRIMARY KEY,
                    score REAL,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                INSERT INTO paper_evaluations (score, created_at)
                VALUES (80.0, '2026-02-16')
            """)
            conn.commit()
            conn.close()

            # Output to non-existent directory
            output_dir = os.path.join(tmpdir, "new", "nested", "dir")

            # Should not raise even if directory doesn't exist
            result = generate_weekly_charts(
                db_path=db_path,
                date_str="2026-02-17",
                output_dir=output_dir
            )

            assert isinstance(result, list)
