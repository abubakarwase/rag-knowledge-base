"""Answer generation with citations and abstain contract."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag_kb.config import get_settings
from rag_kb.logging_utils import log_event
from rag_kb.models import get_chat_model
from rag_kb.retrieve import retrieve

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that."
)

SYSTEM_PROMPT = f"""You are a careful assistant for an English PDF knowledge base.
Answer ONLY using the provided context chunks.
Each chunk header states its file and page. A chunk only supports claims about \
the document it comes from: never answer a question about one policy using \
figures from a different policy or file.
Cite the file and page after each claim, like [auto_pol_0224_atlas_shield_mutual.pdf p4].
If different files give different values for the same fact the question asks about, \
do NOT silently pick one. Begin your answer with "Conflict found:" and list every \
value with its citation, naming which document or policy each value belongs to.
If the context does not contain the answer, reply with exactly:
{ABSTAIN_MESSAGE}
Be concise and factual. Do not invent facts outside the context.
"""

CONFLICT_MARKER = "conflict found"

# Phrasings that indicate the model declined despite retrieval returning chunks.
_SOFT_ABSTAIN_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not mentioned in the provided context",
    "not provided in the context",
)


def _is_soft_abstain(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SOFT_ABSTAIN_MARKERS)


@dataclass
class Citation:
    source: str
    page: int | str
    snippet: str
    score: float


@dataclass
class AnswerResult:
    request_id: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    conflict: bool = False
    latency_s: float = 0.0


def _format_context(documents: list[Document], scores: list[float]) -> str:
    blocks: list[str] = []
    for index, (doc, score) in enumerate(zip(documents, scores, strict=False), start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[{index}] file={source} page={page} score={score:.3f}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def _citations(documents: list[Document], scores: list[float]) -> list[Citation]:
    out: list[Citation] = []
    for doc, score in zip(documents, scores, strict=False):
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        out.append(
            Citation(
                source=str(doc.metadata.get("source", "unknown")),
                page=doc.metadata.get("page", "?"),
                snippet=snippet,
                score=round(float(score), 4),
            )
        )
    return out


def answer_question(question: str) -> AnswerResult:
    settings = get_settings()
    started = time.perf_counter()
    retrieval = retrieve(question)

    if retrieval.abstain:
        latency = round(time.perf_counter() - started, 4)
        log_event(
            logger,
            "generate",
            request_id=retrieval.request_id,
            abstain=True,
            latency_s=latency,
        )
        return AnswerResult(
            request_id=retrieval.request_id,
            answer=ABSTAIN_MESSAGE,
            citations=[],
            abstained=True,
            latency_s=latency,
        )

    context = _format_context(retrieval.documents, retrieval.scores)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Write an English answer grounded in the context with [file p<page>] citations."
    )

    llm = get_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    text = str(response.content).strip()
    # Model declined despite retrieval hits: report it as an abstention and
    # drop citations so the UI does not show irrelevant references.
    soft_abstain = _is_soft_abstain(text)
    conflict = not soft_abstain and CONFLICT_MARKER in text.lower()
    citations = [] if soft_abstain else _citations(retrieval.documents, retrieval.scores)
    latency = round(time.perf_counter() - started, 4)

    log_event(
        logger,
        "generate",
        request_id=retrieval.request_id,
        abstain=soft_abstain,
        soft_abstain=soft_abstain,
        conflict=conflict,
        model=settings.chat_model,
        citation_count=len(citations),
        latency_s=latency,
    )
    return AnswerResult(
        request_id=retrieval.request_id,
        answer=ABSTAIN_MESSAGE if soft_abstain else text,
        citations=citations,
        abstained=soft_abstain,
        conflict=conflict,
        latency_s=latency,
    )


def format_citations_markdown(citations: list[Citation]) -> str:
    if not citations:
        return "_No sources retrieved._"
    lines = ["### References"]
    for cite in citations:
        lines.append(
            f"- **{cite.source}** (page {cite.page}, score {cite.score}): {cite.snippet}"
        )
    return "\n".join(lines)
