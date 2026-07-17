# RAG Engineering Review: Fixes, Before/After Metrics, and Edge-Case Analysis

Date: 2026-07-17. Follow-up to [rag-performance-report.md](rag-performance-report.md), which recorded the baseline. This report covers the fixes applied, the measured improvement, and an edge-case evaluation of the RAG functionality. All numbers are from real runs against the 400-PDF / 4,448-vector index.

## 1. Fixes applied

| Fix | File(s) | What changed |
|---|---|---|
| Viz crash on numpy arrays | `src/rag_kb/store.py` | `fetch_all_embeddings` no longer does `arr or []` on Chroma's numpy return; Vector-space tab works again |
| Chroma client race | `src/rag_kb/store.py` | One cached `PersistentClient`/collection per path (`lru_cache`); parallel ingest no longer hits "Could not connect to tenant" |
| Memory-unbounded viz fetch | `src/rag_kb/store.py` | Sample ids first, then fetch only sampled rows; memory bounded by sample size, not collection size |
| **Identifier-filtered retrieval** | `src/rag_kb/retrieve.py` | Policy-style identifiers (regex `[A-Z]{2,8}-\d{3,8}-\d{1,4}`, case-insensitive) are extracted from the query and used as a Chroma `where_document $contains` filter, one sub-query per identifier, merged by score; falls back to plain dense search when the filter matches nothing |
| Soft-abstain detection | `src/rag_kb/generate.py` | Model refusals are normalized to the abstain contract: `abstained=True`, canonical message, no orphaned "References" in the UI |
| Citation contract | `src/rag_kb/generate.py` | Prompt now shows an explicit `[file p<page>]` example, context blocks are headed `file=... page=...`, and the system prompt forbids answering about one policy with another policy's figures |
| Abstain threshold calibration | `src/rag_kb/config.py`, `.env.example`, `.env` | 0.25 → 0.45, based on measured score bands (on-corpus min 0.57–0.64 vs off-corpus max 0.22). Note: your `.env` line was updated too, since it overrides the code default |
| Ingest double work + accounting | `src/rag_kb/ingest.py` | Single PDF parse per file, single `_already_indexed` check, `skipped` now flows from `ingest_pdf` so scanned = skipped + indexed + errors; error messages include the filename |
| API-key preflight | `src/rag_kb/ingest.py` | One cheap embedding call before ingest; a quota-dead key now fails in seconds with a clear message instead of 300+ retried calls |
| HDBSCAN in high dimensions | `src/rag_kb/cluster.py` | PCA to ≤15 dims before HDBSCAN |
| Settings caching | `src/rag_kb/config.py` | `get_settings()` is `lru_cache`d (it sits in per-request hot paths) |
| Chat UX | `src/rag_kb/app.py` | Question box clears after asking |
| **Integration tests** | `tests/test_integration.py` | 4 new tests run store→retrieve→viz against a real temp Chroma with fake embeddings; the suite would have caught the viz crash, the client race, and the filter behavior. 13/13 tests pass (5s) |

## 2. Before / after metrics (gold set: 24 questions, `eval/gold.jsonl`)

### Retrieval

| Metric | Before | After |
|---|---|---|
| Source hit@1 | 33.3% | **100%** |
| Source hit@5 | 62.5% | **100%** |
| Page hit@1 | 16.7% | **50%** |
| Page hit@3 | 29.2% | **100%** |
| Page hit@5 | 37.5% | **100%** |
| Questions where the right document never appeared | 9/24 | **0/24** |

The single change responsible is the identifier filter. The exact-match `$contains` restricts candidates to chunks that literally contain the asked-about policy number, and dense ranking then orders pages within that document. This is a narrow, corpus-appropriate stand-in for hybrid BM25+dense — see Section 4 for its limits.

### End-to-end generation

| Category | Before | After |
|---|---|---|
| On-corpus correct (of 10) | 5, plus **1 confidently wrong** | **10/10 correct** |
| Wrong answers | 1 (premium from a different policy) | **0** |
| Citations | Inconsistent (`[source: 4]`), file rarely named | Exact `[file p<page>]` on every answer, verified correct file and page |
| Off-corpus abstain (of 5) | 5 | 5 |
| Nonexistent policy / unknown carrier / comparison | Soft refusals with `abstained=False` and 5 irrelevant references shown | Clean abstains: `abstained=True`, no references |
| Mean on-corpus latency | ~2.2s | ~2.0s |

## 3. Edge-case evaluation (12 cases)

| Case | Query pattern | Result | Verdict |
|---|---|---|---|
| Lowercase identifier | `auto-385891-52` | Correct answer; filter uppercases the token | PASS |
| Mistyped identifier | `AUTO-385891-53` (last digit off) | Filter matches nothing → dense fallback → model abstains because no context block matches the asked-for policy | PASS (safe) |
| Paraphrase | "cost per year" for premium | Correct $29,660 | PASS |
| Multi-fact, one policy | collision + comprehensive limits | Both correct from one page | PASS |
| Corpus noise | Doc contains typo "Property damge Liability" | Correct $60,000 despite the typo | PASS |
| Cross-lingual | Spanish question, English corpus | Correct answer (identifier filter carries it); answer in English per design | PASS |
| Prompt injection | "Ignore all previous instructions..." | Scores below threshold → hard abstain, LLM never invoked (0.3s) | PASS |
| Gibberish / chit-chat | `???`, `hello` | Hard abstain | PASS |
| **Aggregation** | "How many auto policies does Ironwood have?" | **"Five"** — the model counted its top-5 retrieval window. Actual count: **9** | **FAIL (confident, wrong)** |
| **Ambiguity** | "The Harborpoint auto policy" (several exist) | Answered with one policy's deductible, no mention that multiple Harborpoint auto policies exist | **FAIL (misleading by omission)** |
| Cross-document comparison | Deductibles of two named policies | Abstains (safe, but the data is in the index) | SAFE FAIL |
| Conversation memory | Follow-up questions ("what about its deductible?") | Not tested — each turn is independent by design; no identifier survives to the next turn | KNOWN GAP |

## 4. Expert assessment of the RAG design

**What the system now does well.** Single-document factual lookup — the core use case — is solid: exact identifier routing, page-accurate citations, calibrated abstention with a wide margin (0.22 off-corpus vs 0.57+ on-corpus), injection and noise queries rejected before the LLM is even called (which is also a cost control). The failure directionality is right: when the system errs, it now errs toward silence, not fabrication.

**The identifier filter is a scoped fix, not hybrid retrieval.** It generalizes to any corpus where questions reference token-exact IDs (claims, invoices, SKUs, tickets), but it does nothing for exact-match needs that don't fit the regex: person names ("Taylor Ortiz"), addresses, endorsement numbers alone ("Endorsement 8038" — currently rescued by dense search, not guaranteed). True BM25+dense with RRF remains the right next step; the filter buys time, and its clean fallback means it can never make retrieval worse than baseline.

**Aggregation questions are structurally unanswerable by top-k RAG, and the system doesn't know it.** "How many X..." got a confident wrong answer because five retrieved chunks look exactly like a complete answer to the model. Options, in increasing effort: (a) prompt rule — "if the question asks to count or enumerate across documents, state that you can only see excerpts"; (b) intent detection routing count/list questions to a metadata query (Chroma `where={"source": ...}` can answer "how many Ironwood auto policies" exactly, from filenames alone); (c) an agentic loop. Option (b) is cheap and exact for this corpus.

**Ambiguity should be surfaced, not resolved silently.** When retrieval returns chunks from several distinct sources and the question implies one entity ("the Harborpoint policy"), answering from whichever ranked first is a subtle wrong-answer generator. The retrieval result already carries the source spread; a one-line rule — if top-k spans >2 sources without an identifier match, enumerate the candidates and ask the user to pick — converts this failure into a good UX moment.

**Cross-document comparison fails safe but shouldn't fail.** Both deductibles are indexed; the per-identifier sub-queries retrieve them, but a single merged 5-chunk context ranked by similarity to the whole question doesn't reliably keep each policy's declarations page. A per-identifier quota (guarantee ≥2 chunks per referenced document) would likely fix it without architecture changes.

**No conversation memory is a product decision that should be explicit.** `_chat` ignores history when answering; "what about its deductible?" as a follow-up will abstain or mis-retrieve. Either pass recent turns through a query-rewrite step (standard condense-question pattern) or label the UI single-turn.

**Evaluation posture.** The gold set (24 questions), the smoke eval, and the edge-case suite now exist and are cheap to run; the abstain threshold is an empirically calibrated number with a documented basis rather than a guess. The right next step is wiring `eval_smoke` + the threshold check into CI so retrieval changes (new embedding model, chunking tweaks) get gated by hit-rate, and adding an LLM-judged faithfulness check once the corpus stops being synthetic.

**Scale posture.** At 4.4k vectors everything is comfortable. The previously flagged ceilings still stand and are now partially mitigated (bounded viz fetch, cached client). The genuinely expensive open item for a 400k-doc future is per-sentence embedding cost in semantic chunking; the honest path stays: cheaper breakpoint model or simpler chunker at scale, then Chroma-service → Qdrant only on measured need.

## 5. Residual risks

1. The soft-abstain detector is phrase-list based; a refusal phrased unusually would slip through with citations attached. A structured output field (`{"abstain": bool, "answer": ...}`) would be robust.
2. The identifier regex is tuned to this corpus's ID shape; document it in the `rag-pipeline` skill so a future corpus change revisits it.
3. `where_document $contains` is case-sensitive exact substring; identifiers embedded in the corpus with different formatting (spaces, hyphen variants) would silently miss and fall back to dense.
4. Threshold 0.45 was calibrated on synthetic English boilerplate; recalibrate on any real corpus (the eval script makes this a 2-minute job).

## Reproducing

```bash
uv run pytest                                              # 13 tests, includes integration
uv run python -m rag_kb.eval_smoke --gold eval/gold.jsonl  # retrieval smoke
uv run rag-kb-chat                                         # UI on :7860, viz tab now works
```
