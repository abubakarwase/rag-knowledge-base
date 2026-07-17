"""Shared OpenAI embedding + chat factories and Chroma access."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_kb.config import Settings, get_settings


@lru_cache(maxsize=4)
def get_embeddings(model: str | None = None, api_key: str | None = None) -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=model or settings.embedding_model,
        api_key=api_key or settings.openai_api_key or None,
    )


def get_chat_model(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.chat_model,
        temperature=temperature,
        api_key=settings.openai_api_key or None,
    )


def ensure_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    Path(settings.knowledge_base_path).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
