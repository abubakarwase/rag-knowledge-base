"""2D/3D visualization colored by cluster (PCA projection)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from rag_kb.cluster import cluster_embeddings
from rag_kb.config import get_settings
from rag_kb.store import fetch_all_embeddings

logger = logging.getLogger(__name__)


def _pca_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    if n_components < 2:
        padded = np.zeros((embeddings.shape[0], 2), dtype=np.float64)
        padded[:, : embeddings.shape[1]] = embeddings[:, : padded.shape[1]]
        return padded
    reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)


def build_vector_plots(
    sample_size: int | None = None,
    n_clusters: int | None = None,
    method: str | None = None,
) -> tuple[go.Figure, go.Figure, str]:
    """Return (fig_2d, fig_3d, status_message)."""
    settings = get_settings()
    sample_size = sample_size or settings.viz_sample_size

    ids, embeddings, metadatas, documents = fetch_all_embeddings(limit=sample_size)
    if not embeddings:
        empty = go.Figure()
        empty.update_layout(title="No vectors indexed yet")
        return empty, empty, "Index is empty. Ingest PDFs first."

    matrix = np.asarray(embeddings, dtype=np.float64)
    labels = cluster_embeddings(matrix, method=method, n_clusters=n_clusters)
    cluster_str = [f"cluster {int(label)}" for label in labels]
    hover = [
        f"{m.get('source', '?')} p{m.get('page', '?')}<br>{(doc or '')[:120]}"
        for m, doc in zip(metadatas, documents, strict=False)
    ]

    try:
        coords2 = _pca_reduce(matrix, 2)
        coords3 = _pca_reduce(matrix, 3)
        if coords3.shape[1] < 3:
            # Degenerate case: pad a zero axis for 3D plot.
            pad = np.zeros((coords3.shape[0], 3 - coords3.shape[1]))
            coords3 = np.hstack([coords3, pad])
    except Exception as exc:  # noqa: BLE001
        logger.exception("pca_failed")
        empty = go.Figure()
        empty.update_layout(title=f"Projection failed: {exc}")
        return empty, empty, f"Projection failed: {exc}"

    fig2 = px.scatter(
        x=coords2[:, 0],
        y=coords2[:, 1],
        color=cluster_str,
        hover_name=hover,
        title="Knowledge base in 2D (PCA + clusters)",
        labels={"x": "PC-1", "y": "PC-2", "color": "cluster"},
    )
    fig2.update_traces(marker={"size": 8, "opacity": 0.85})
    fig2.update_layout(template="plotly_white", legend_title_text="Cluster")

    fig3 = px.scatter_3d(
        x=coords3[:, 0],
        y=coords3[:, 1],
        z=coords3[:, 2],
        color=cluster_str,
        hover_name=hover,
        title="Knowledge base in 3D (PCA + clusters)",
        labels={"x": "PC-1", "y": "PC-2", "z": "PC-3", "color": "cluster"},
    )
    fig3.update_traces(marker={"size": 4, "opacity": 0.85})
    fig3.update_layout(template="plotly_white", legend_title_text="Cluster")

    status = (
        f"Plotted {len(ids)} chunks "
        f"(sample_cap={sample_size}, method={method or settings.viz_cluster_method}, "
        "projection=PCA)."
    )
    return fig2, fig3, status


def index_stats() -> dict[str, Any]:
    from rag_kb.ingest import list_pdfs
    from rag_kb.store import collection_count

    settings = get_settings()
    pdfs = list_pdfs()
    return {
        "pdf_files": len(pdfs),
        "vector_count": collection_count(),
        "knowledge_base": str(settings.knowledge_base_path),
        "chroma_path": str(settings.chroma_path),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
    }
