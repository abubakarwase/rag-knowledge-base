"""Gradio UI for Chat With Your Docs."""

from __future__ import annotations

import logging

import gradio as gr

from rag_kb.config import get_settings
from rag_kb.generate import answer_question, format_citations_markdown
from rag_kb.logging_utils import setup_logging
from rag_kb.models import ensure_dirs
from rag_kb.visualize import build_vector_plots, index_stats

logger = logging.getLogger(__name__)

CUSTOM_CSS = """
:root {
  --rk-bg: #0f1419;
  --rk-panel: #1a222c;
  --rk-accent: #3d9a7a;
  --rk-text: #e8eef4;
  --rk-muted: #9aa8b5;
}
.gradio-container {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
  background: radial-gradient(1200px 600px at 10% -10%, #1c3a32 0%, var(--rk-bg) 45%) !important;
  color: var(--rk-text) !important;
}
#app-title h1 {
  font-family: "IBM Plex Serif", Georgia, serif !important;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--rk-text) !important;
}
#app-subtitle {
  color: var(--rk-muted) !important;
  font-size: 1.05rem;
}
.panel-card {
  background: color-mix(in srgb, var(--rk-panel) 92%, transparent);
  border: 1px solid #2a3644 !important;
  border-radius: 14px !important;
  padding: 0.5rem;
}
button.primary {
  background: var(--rk-accent) !important;
}
"""


def _status_markdown() -> str:
    stats = index_stats()
    return (
        f"**PDFs on disk:** {stats['pdf_files']}  \n"
        f"**Vectors in Chroma:** {stats['vector_count']}  \n"
        f"**KB path:** `{stats['knowledge_base']}`  \n"
        f"**Chroma path:** `{stats['chroma_path']}`  \n"
        f"**Embed / chat:** `{stats['embedding_model']}` / `{stats['chat_model']}`"
    )


def _chat(message: str, history: list | None) -> tuple[list, str, str]:
    history = history or []
    settings = get_settings()
    if not settings.has_openai_key:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Set `OPENAI_API_KEY` in `.env` first."},
        ]
        return history, "_No sources._", ""

    result = answer_question(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
    refs = format_citations_markdown(result.citations)
    if result.abstained:
        refs = f"_Abstained (request `{result.request_id}`)._\n\n" + refs
    else:
        refs = f"_request `{result.request_id}` · {result.latency_s}s_\n\n" + refs
    if result.conflict:
        refs = (
            "⚠️ **Conflicting sources detected.** The documents below disagree; "
            "each value is cited so you can verify.\n\n" + refs
        )
    return history, refs, ""


def _refresh_viz(sample_size: int, n_clusters: int, method: str):
    fig2, fig3, status = build_vector_plots(
        sample_size=int(sample_size),
        n_clusters=int(n_clusters),
        method=method,
    )
    return fig2, fig3, status, _status_markdown()


def build_ui() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="Chat With Your Docs") as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Chat With Your Docs", elem_id="app-title")
                gr.Markdown(
                    "Ask questions over your local PDF knowledge base. "
                    "Answers cite source and page. Explore how chunks cluster in vector space.",
                    elem_id="app-subtitle",
                )

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes=["panel-card"]):
                        chatbot = gr.Chatbot(label="Conversation", height=420)
                        question = gr.Textbox(
                            label="Question",
                            placeholder="What does the knowledge base say about ...?",
                            lines=2,
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                    with gr.Column(scale=2, elem_classes=["panel-card"]):
                        citations = gr.Markdown("_References appear here._")
                ask_btn.click(_chat, inputs=[question, chatbot], outputs=[chatbot, citations, question])
                question.submit(_chat, inputs=[question, chatbot], outputs=[chatbot, citations, question])

            with gr.Tab("Knowledge base"):
                with gr.Column(elem_classes=["panel-card"]):
                    status = gr.Markdown(_status_markdown())
                    refresh_btn = gr.Button("Refresh status")
                    refresh_btn.click(_status_markdown, outputs=[status])
                    gr.Markdown(
                        "Indexing is a **separate local command** (not done in this UI).\n\n"
                        "1. Drop English PDFs into `knowledge-base/`\n"
                        "2. Run ingest: `uv run rag-kb-ingest`\n"
                        "3. Then start this chat: `uv run rag-kb-chat`\n\n"
                        "Re-ingest skips unchanged files via content hash. "
                        "Docker Compose is optional and meant for production deploys."
                    )

            with gr.Tab("Vector space"):
                with gr.Column(elem_classes=["panel-card"]):
                    with gr.Row():
                        sample = gr.Slider(100, 10000, value=settings.viz_sample_size, step=100, label="Sample size")
                        clusters = gr.Slider(2, 30, value=settings.viz_n_clusters, step=1, label="KMeans k")
                        method = gr.Dropdown(
                            choices=["kmeans", "hdbscan"],
                            value=settings.viz_cluster_method,
                            label="Cluster method",
                        )
                        viz_btn = gr.Button("Recompute", variant="primary")
                    viz_status = gr.Markdown("Press **Recompute** to project embeddings.")
                    with gr.Row():
                        plot2 = gr.Plot(label="2D")
                        plot3 = gr.Plot(label="3D")
                    viz_btn.click(
                        _refresh_viz,
                        inputs=[sample, clusters, method],
                        outputs=[plot2, plot3, viz_status, status],
                    )

        gr.Markdown(
            "<span style='color:#9aa8b5'>Local Chroma · OpenAI embeddings · "
            "semantic per-page chunking · run with uv</span>"
        )
    return demo


def main() -> None:
    setup_logging()
    settings = get_settings()
    ensure_dirs(settings)
    demo = build_ui()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
