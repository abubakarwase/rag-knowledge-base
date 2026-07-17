# RAG Pipeline: Performance and Code Review Report

Date: 2026-07-17. Corpus: 400 synthetic insurance-policy PDFs (auto, home, life, health, CGL, cyber, specialty, travel, umbrella, workers comp across ~12 fictional carriers). Stack: OpenAI `text-embedding-3-small` + `gpt-4o-mini`, embedded Chroma, custom per-page semantic chunking, Gradio UI.

All numbers below were measured on a real end-to-end run, not estimated.

## 1. Run summary

| Check | Result |
|---|---|
| Unit tests | 9/9 passed (45.6s) |
| Ingest (400 PDFs, 4 workers) | 396 indexed, 4,402 chunks, 505s; **4 files failed** with Chroma client errors (see Section 5) |
| Ingest retry (1 worker) | Picked up the 4 failed files: 46 chunks, 21.4s, 0 errors |
| Final index | **4,448 vectors** across 400 files (~11 chunks/file, ~2.9 chunks/page) |
| Idempotency re-run | 0 chunks written, 6.0s. Accounting bug: 398 "skipped" + 0 "indexed" ≠ 400 scanned |
| Gradio app | Serves HTTP 200 on `:7860`; chat works end-to-end |
| Vector-space tab | **Crashes at runtime** (`ValueError` in `fetch_all_embeddings`, see Section 5) |
| Clustering (via direct call) | KMeans k=8: works, 52% purity vs document-type. HDBSCAN on raw 1536-d: **77% of points labeled noise** |

An earlier run also surfaced a robustness gap worth recording: with an OpenAI key that had no quota, ingest burned 305 failed API calls before being killed. The `insufficient_quota` error is non-retryable, but both the openai client and tenacity retried it aggressively, and the semantic-chunk fallback silently degraded every page while the final upsert still failed. There is no fail-fast preflight check of the API key.

## 2. Retrieval quality (gold set: `eval/gold.jsonl`, 24 questions)

Measured with score threshold 0 and k=5, matching by expected source file and page:

| Metric | @1 | @3 | @5 |
|---|---|---|---|
| Source hit (right file retrieved) | 33.3% | 45.8% | 62.5% |
| Page hit (right file *and* page) | 16.7% | 29.2% | 37.5% |

**9 of 24 questions never retrieved the correct document at any rank.** This is the defining failure of the system, and its cause is structural: the corpus is 400 near-duplicate boilerplate documents where the discriminating token is a policy number (`AUTO-385891-52`). Dense embeddings do not encode exact identifiers distinctively, so a query citing a specific policy number retrieves *similar-looking pages of other policies from the same carrier* with scores as high as (often higher than) the right document. Questions answerable from distinctive text (rental limits, endorsement numbers, unusual company names) did fine; identifier lookups failed.

This is precisely the case that the plan's deferred "hybrid BM25 + dense" feature solves — a keyword/BM25 leg matches `AUTO-385891-52` exactly. The deferral was reasonable process, but on this corpus hybrid retrieval is not an optimization; it is the difference between a working and a non-working system.

## 3. Abstain-threshold analysis

Top-1 similarity distributions (cosine, higher = better):

| Query set | min | mean | max |
|---|---|---|---|
| 24 gold (on-corpus) | 0.639 | 0.730 | 0.821 |
| 5 off-corpus | 0.106 | 0.147 | 0.218 |

- The configured threshold **0.25 does separate the two populations** on this corpus — every off-corpus query hard-abstained, no on-corpus query was falsely rejected. The earlier concern that the threshold was uncalibrated turned out benign *here*, though the margin above the off-corpus max (0.218 vs 0.25) is thin. A threshold around **0.45** would sit in the middle of the wide empty band and be much more robust.
- However, the threshold **cannot catch the dangerous failure mode**: wrong-document retrievals score 0.75+, far above any plausible threshold. Score-based abstention protects against off-topic questions, not against confidently retrieving the wrong policy.

## 4. End-to-end generation quality (18 questions)

| Category | Result |
|---|---|
| 10 on-corpus | **5 correct** (with citations), **4 safe soft-abstains** ("I do not have enough information..."), **1 confidently wrong** |
| 5 off-corpus | 5/5 hard abstain with the contract message |
| Nonexistent policy number | Soft-abstained (good) |
| Carrier not in KB ("State Farm") | Soft-abstained (good) |
| Cross-document comparison | Soft-abstained (retrieval never surfaced both policies' declarations pages) |

**The wrong answer is the one that matters.** Asked for the premium of policy AUTO-385891-52 ($29,660), the model answered **"$45,030"** with a confident citation — a figure belonging to a different Atlas Shield Mutual policy whose chunks dominated the retrieved context. When retrieval returns five near-identical documents from the same carrier, the LLM cannot reliably tell which chunk belongs to the asked-about policy. Grounded-but-misattributed answers are worse than abstentions for an insurance use case.

Other observations:

- **Soft abstains are invisible to the system.** When the model declines in prose, `abstained` stays `false` and the UI still renders five irrelevant "References" under the refusal. The abstain flag only tracks the retrieval threshold, not the model's own refusal.
- **Citation format is inconsistent.** The prompt asks for `[source: page]`, which the model renders variously as `[source: page 1]` and `[source: 4]`, and it almost never names the actual file. The citation panel compensates, but the inline citations don't identify documents.
- **Latency**: on-corpus mean ~2.2s (range 1.3–7.1s); abstains ~0.3–0.5s (no LLM call). Fine for interactive use.

## 5. Bugs found by running the system

1. **Vector-space tab is broken** — [store.py:120](../src/rag_kb/store.py): `data.get("embeddings") or []` raises `ValueError: The truth value of an array ... is ambiguous` because Chroma returns a numpy array. Every viz request fails with any indexed data. The mock-only test suite cannot catch this class of bug. Fix: `raw = data.get("embeddings"); raw = [] if raw is None else raw`.
2. **Concurrent ingest races the Chroma client** — [store.py](../src/rag_kb/store.py) builds a new `PersistentClient` on every call from every worker thread. With 4 workers, 4/400 files failed ("Could not connect to tenant default_tenant", "'RustBindingsAPI' object has no attribute 'bindings'"). Fix: create one client (and one collection handle) per process, e.g. an `lru_cache` factory.
3. **Ingest error messages lose the filename** — the `future.result()` exception path appends `str(exc)` only, so the summary showed "Could not connect to tenant..." with no way to know *which* file failed. The recovery worked only because the hash-skip made a full re-run cheap.
4. **Report accounting inconsistency** — on a warm re-run, 2 files were counted neither "skipped" nor "indexed" (398+0 of 400): `_work` checks `_already_indexed`, then `ingest_pdf` re-checks it and returns `(0, 0, None)`, which the caller counts as nothing. Cosmetic, but it makes ingest logs untrustworthy.
5. **HDBSCAN option is effectively unusable** — on raw 1536-d embeddings it labels 77% of chunks noise (curse of dimensionality) and took 20s for 2,000 points. Standard practice is to reduce with UMAP/PCA to ~10–20 dims before HDBSCAN. Related: the plan promises UMAP but [visualize.py](../src/rag_kb/visualize.py) ships PCA; either implement UMAP or update the docs.
6. **No API-key preflight** — a quota-dead key produced hundreds of retried calls and silent per-page chunking degradation instead of a fast, clear failure (see Section 1).

## 6. Code structure review

### What is genuinely good

- **Clean module boundaries** that match the architecture: `ingest` / `retrieve` / `generate` / `store` / `cluster` / `visualize` / `app`, with `config`, `models`, `logging_utils` as shared infrastructure. Responsibilities are where you'd look for them.
- **Config discipline**: everything tunable lives in `Settings` (pydantic-settings) with a documented `.env.example`; no magic numbers scattered in code.
- **Idempotent ingest done right in outline**: content-hash skip, `delete_by_source` before re-add (handles modified files, unlike many RAG demos), batched embedding, retry with backoff, progress file.
- **Structured logging** with request IDs, latency, sources, and abstain flags on every retrieve/generate — this made the evaluation in this report easy.
- **Custom semantic chunker** ([chunking.py](../src/rag_kb/chunking.py)) is compact, dependency-light (avoids `langchain_experimental`), and unit-testable; the <40-word short-page bypass avoids paying embedding costs for trivial pages.
- **Abstain contract** implemented and honored end-to-end (message constant, flag, UI note).

### What should improve (beyond the bugs above)

- **`get_settings()` is uncached** and re-reads `.env` on every call; it's called multiple times per request across modules. Make it `@lru_cache` (and inject settings where practical instead of re-fetching).
- **`fetch_all_embeddings()` loads the entire collection into memory** (ids + 1536-d vectors + metadata + documents) before applying the sample cap client-side. At 4,448 chunks this is fine; at the 400k-doc target it is an OOM. Sample IDs first (`collection.get(limit=..., offset=...)` pages or random ID sampling), then fetch only the sample.
- **Duplicated work in ingest**: `_already_indexed` runs twice per new file (in `_work` and again in `ingest_pdf`), and each PDF is parsed twice (`load_pdf_pages` and a second `PdfReader` just to count empty pages). Pass the page count out of `load_pdf_pages` and drop the inner re-check.
- **`retrieve()` filters after top-k**, so `top_k=5` with a threshold silently yields fewer context chunks and there is no "fetch more, then filter" compensation. Minor today; worth a comment or a `fetch_k` parameter.
- **Prompt/citation contract**: the system prompt's `[source: page]` is ambiguous (model can't tell if "source" means filename). Give an explicit example like `[auto_pol_0224.pdf p4]`, and number the context blocks so the model can cite chunk indices deterministically.
- **Soft-abstain detection**: treat model refusals as abstentions (simplest: instruct the model to output the exact `ABSTAIN_MESSAGE`; then string-match to set the flag and suppress the references panel).
- **Tests are mocks only.** All nine tests pass while the viz tab crashes on real data. Add one integration test with a tiny fixture PDF and a fake embedding class writing to a temp Chroma dir — it would have caught bugs 1, 2, and 4.
- **Chat UI details**: textbox isn't cleared after asking, no streaming, and `gr.Chatbot` without `type="messages"` is on Gradio's deprecation path.
- **`page_start`/`page_end` metadata is dead weight** — chunking is strictly per-page, so they always equal `page`. Either drop them or implement cross-page chunking (see below).

## 7. Prioritized recommendations

1. **Hybrid retrieval (BM25 + dense, RRF merge), or at minimum a policy-number regex → metadata filter.** Root cause of 9/24 total retrieval misses and of the one confidently-wrong answer. A cheap pre-step — extract `[A-Z]+-\d{6}-\d{2}` from the query and pass it as a Chroma `where` filter — would likely fix most identifier queries in an afternoon. *(High impact, low-to-medium effort.)*
2. **Fix the viz crash and the Chroma client singleton** (bugs 1–2). Both are small diffs; one restores a whole product surface, the other makes parallel ingest trustworthy. *(High impact, low effort.)*
3. **Add one real integration test** (tiny PDF + fake embeddings + temp Chroma) to stop shipping runtime crashes a mock suite can't see. *(High impact, low effort.)*
4. **Detect soft abstains and suppress references on refusals**; tighten the citation format with an explicit example. *(Medium impact, low effort.)*
5. **Raise the abstain threshold to ~0.45** based on the measured score gap (0.22 vs 0.64), and keep `eval/gold.jsonl` + the smoke eval in CI so the threshold stays calibrated as the corpus changes. *(Medium impact, trivial effort.)*
6. **Prompt the model to answer only from chunks matching the asked-about policy** (e.g., include source filename + policy number in each context block header and instruct: if no block matches the requested policy, abstain). Directly targets the misattribution failure while retrieval is dense-only. *(Medium impact, low effort.)*
7. **Scale hygiene when the corpus grows**: sample-then-fetch embeddings, cached settings, single-parse ingest, UMAP-before-HDBSCAN (or drop HDBSCAN). *(Lower urgency at 4.4k vectors.)*

## Reproducing these numbers

```bash
uv run pytest                                   # unit tests
uv run rag-kb-ingest                            # idempotent ingest
uv run python -m rag_kb.eval_smoke --gold eval/gold.jsonl --top-k 5
uv run rag-kb-chat                              # UI on http://localhost:7860
```

The gold set is `eval/gold.jsonl` (24 questions with expected source, page, and answer, authored from the actual PDF text).
