"""Optional Visualization Module for Paper Scout.

Provides score distribution charts and embedding cluster maps for weekly reports.
Gracefully degrades when visualization dependencies (numpy, matplotlib, umap-learn)
are not installed.

Key Design: All visualization functions check _VIZ_AVAILABLE flag and return None
or empty list on failure, never raising exceptions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from core.storage.db_connection import get_connection

# Graceful import handling - set module-level flag
_VIZ_AVAILABLE = False
_UMAP_AVAILABLE = False

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/CI
    import matplotlib.pyplot as plt
    _VIZ_AVAILABLE = True
except ImportError:
    pass

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    pass


def is_viz_available() -> bool:
    """Check if visualization dependencies are available.

    Returns:
        True if numpy and matplotlib are installed, False otherwise.
    """
    return _VIZ_AVAILABLE


def generate_score_distribution(
    scores: List[float],
    output_path: str,
    title: str = "Score Distribution"
) -> Optional[str]:
    """Generate a histogram of paper scores.

    Args:
        scores: List of paper scores (0-100 range expected).
        output_path: Path where PNG file will be saved.
        title: Chart title.

    Returns:
        output_path on success, None on failure or if viz not available.
    """
    if not _VIZ_AVAILABLE:
        return None

    if not scores:
        return None

    try:
        # Import inside function to avoid errors if module not available
        import numpy as np
        import matplotlib.pyplot as plt

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define score ranges (0-10, 10-20, ..., 90-100)
        bins = list(range(0, 101, 10))

        # Create histogram
        ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7)

        # Formatting
        ax.set_xlabel('Score Range', fontsize=12)
        ax.set_ylabel('Paper Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    except (ValueError, OSError, TypeError):
        # Graceful failure - return None instead of raising
        return None


def generate_cluster_map(
    embeddings: Any,
    labels: List[str],
    output_path: str,
    title: str = "Paper Clusters"
) -> Optional[str]:
    """Generate UMAP 2D projection of paper embeddings with cluster coloring.

    Args:
        embeddings: Numpy array of embeddings (shape: [n_papers, embedding_dim]).
        labels: List of cluster labels for each paper.
        output_path: Path where PNG file will be saved.
        title: Chart title.

    Returns:
        output_path on success, None on failure or if viz/umap not available.
    """
    if not _VIZ_AVAILABLE:
        return None

    if not _UMAP_AVAILABLE:
        # UMAP is optional even within viz module
        return None

    if embeddings is None or len(labels) == 0:
        return None

    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import umap

        # Ensure embeddings is numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Check dimensions
        if embeddings.ndim != 2:
            return None

        if embeddings.shape[0] != len(labels):
            return None

        # UMAP reduction to 2D
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        embedding_2d = reducer.fit_transform(embeddings)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get unique labels for coloring
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(range(len(unique_labels)))

        # Plot each cluster
        for idx, label in enumerate(unique_labels):
            # Find points with this label
            mask = [l == label for l in labels]
            cluster_points = embedding_2d[mask]

            # Plot cluster
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[colors[idx]],
                label=label,
                alpha=0.6,
                s=50
            )

        # Formatting
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    except (ValueError, OSError, TypeError):
        # Graceful failure
        return None


def generate_weekly_charts(
    db_path: str,
    date_str: str,
    output_dir: str = "tmp/reports",
    provider: str = "sqlite",
    connection_string: str | None = None,
) -> List[str]:
    """Main entry: generate all available visualizations for weekly report.

    Args:
        db_path: Path to SQLite database.
        date_str: ISO date string (YYYY-MM-DD) for the week.
        output_dir: Directory where charts will be saved.
        provider: Database provider ("sqlite" or "supabase").
        connection_string: PostgreSQL connection string (when provider is "supabase").

    Returns:
        List of generated file paths (empty list if viz not available).
    """
    if not _VIZ_AVAILABLE:
        return []

    generated_files = []

    try:
        with get_connection(db_path, provider, connection_string) as (conn, ph):
            if conn is None:
                return []
            cursor = conn.cursor()

            # Query 1: Score distribution from paper_evaluations
            # Join with runs to filter by date using window_start_utc,
            # since paper_evaluations has no created_at column.
            if provider == "supabase":
                query = f"""
                    SELECT pe.final_score
                    FROM paper_evaluations pe
                    JOIN runs r ON pe.run_id = r.run_id
                    WHERE r.window_start_utc >= ({ph}::date - interval '7 days')::text
                      AND r.window_start_utc < {ph}
                      AND pe.final_score IS NOT NULL
                """
            else:
                query = f"""
                    SELECT pe.final_score
                    FROM paper_evaluations pe
                    JOIN runs r ON pe.run_id = r.run_id
                    WHERE DATE(r.window_start_utc) >= date({ph}, '-7 days')
                      AND DATE(r.window_start_utc) < date({ph})
                      AND pe.final_score IS NOT NULL
                """
            cursor.execute(query, (date_str, date_str))

            rows = cursor.fetchall()
            scores = [
                row["final_score"] if isinstance(row, dict) else row[0]
                for row in rows
            ]

        if scores:
            score_chart_path = os.path.join(
                output_dir,
                f"score_distribution_{date_str}.png"
            )
            result = generate_score_distribution(
                scores,
                score_chart_path,
                f"Score Distribution - Week of {date_str}"
            )
            if result:
                generated_files.append(result)

        # Query 2: Cluster map from embeddings
        # NOTE: paper_embeddings table does not exist in the current schema.
        # Skipping cluster map generation until embeddings storage is added.

    except (OSError, ValueError, Exception):
        # Graceful failure - return whatever was generated before error
        pass

    return generated_files
