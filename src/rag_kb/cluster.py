"""Cluster embeddings for vector-space visualization."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from rag_kb.config import get_settings

logger = logging.getLogger(__name__)

ClusterMethod = Literal["kmeans", "hdbscan"]


def cluster_embeddings(
    embeddings: list[list[float]] | np.ndarray,
    method: str | None = None,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Return integer cluster labels for each embedding row."""
    settings = get_settings()
    method = (method or settings.viz_cluster_method).lower()
    n_clusters = n_clusters or settings.viz_n_clusters

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.array([], dtype=int)
    if matrix.shape[0] == 1:
        return np.array([0], dtype=int)

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError:
            logger.warning("hdbscan not available; falling back to kmeans")
            method = "kmeans"
        else:
            # HDBSCAN degrades badly in high dimensions (measured 77% noise on
            # raw 1536-d embeddings); reduce with PCA first.
            reduced = matrix
            n_components = min(15, matrix.shape[0] - 1, matrix.shape[1])
            if matrix.shape[1] > n_components >= 2:
                from sklearn.decomposition import PCA

                reduced = PCA(n_components=n_components, random_state=42).fit_transform(matrix)
            min_cluster_size = max(2, min(15, matrix.shape[0] // 10 or 2))
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = model.fit_predict(reduced)
            # Map noise (-1) to its own bucket for coloring
            return labels.astype(int)

    k = max(1, min(n_clusters, matrix.shape[0]))
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model.fit_predict(matrix).astype(int)
