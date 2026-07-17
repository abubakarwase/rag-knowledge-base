"""Application configuration from environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    knowledge_base_path: Path = Path("./knowledge-base")
    chroma_path: Path = Path("./data/chroma")
    chroma_collection: str = "rag_kb"

    retrieval_top_k: int = 5
    # Calibrated on eval/gold.jsonl: on-corpus top-1 scores start ~0.64,
    # off-corpus top out ~0.22, so 0.45 sits mid-band with margin both ways.
    retrieval_score_threshold: float = 0.45

    semantic_breakpoint_threshold: float = 95.0
    semantic_buffer_size: int = 1

    ingest_embed_batch_size: int = 64
    ingest_max_workers: int = 2

    viz_sample_size: int = 5000
    viz_n_clusters: int = 8
    viz_cluster_method: str = "kmeans"

    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860

    @property
    def has_openai_key(self) -> bool:
        key = (self.openai_api_key or "").strip()
        return bool(key) and not key.startswith("sk-your-key")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Cached: called in per-request hot paths; use get_settings.cache_clear()
    # in tests after changing env vars.
    return Settings()
