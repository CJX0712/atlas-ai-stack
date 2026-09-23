# Spec — ATLAS v1.0.0

> Generated: 2026-09-24
> Based on: system requirements + architecture review + measured baselines
> Status: **Confirmed** (frozen at v1.0.0)
> Author: 晨星

This document is the internal contract. Development, testing and the scorecard
floors all reference it. Anything not listed in section 2 is out of scope.

---

## 1. Product definition

- **One-line description** — A modular, end-to-end runnable AI system that
  answers questions over a document corpus with hybrid retrieval, guarded
  reranking, deterministic tool routing and per-claim grounding verification.
- **Target user** — engineers who need a retrieval-and-reasoning system whose
  behaviour is *auditable* and whose verification runs offline, on CPU, with no
  API key.
- **Core problem** — retrieval-augmented systems regress invisibly. A reranker
  quietly demotes correct answers, a small model does arithmetic badly, a fluent
  paragraph contains one fabricated sentence, and aggregate metrics stay green.
  ATLAS makes each of those failures detectable, testable and bounded.

---

## 2. MVP scope (locked)

| Priority | Capability | Acceptance summary | Rationale |
|---|---|---|---|
| P0 | Structure-aware chunking | base spans exactly partition the source | retrieval quality is bounded by chunk quality |
| P0 | Sparse retrieval (Robertson BM25) | IDF provably non-negative | the common IDF form inverts ranking on small corpora |
| P0 | Dense retrieval (hashing / ONNX) | self-similarity 1.0; deterministic across processes | must run without a model download |
| P0 | RRF fusion | rank-1 consensus wins; deterministic ties | cosine and BM25 scores are not comparable |
| P0 | Guarded reranking | adversarial reranker cannot displace top-1 | measured top-1 regression 11/12 to 3/12 |
| P0 | Deterministic routing | every arithmetic expression routes to the tool | measured 0.5B model failure on `12*(3+4)` |
| P0 | Safe arithmetic, unit, date tools | whitelisted AST; no `eval` | tool inputs are untrusted |
| P0 | Bounded ReAct loop | terminates; budget enforced; degraded answer on exhaustion | an unbounded agent is an outage |
| P0 | Grounding verification | fabricated sentence flagged | fluency hides fabrication |
| P0 | Offline evaluation harness | bit-identical across runs | a metric that drifts cannot gate |
| P0 | HTTP API + CLI | success and error flows asserted | the system must be operable |
| P1 | Thesaurus query expansion | unknown terms expand to nothing | measured cross-lingual recall gap |
| P1 | Provider substitution with fallback | substitution is reported, never silent | deployment degradations must be visible |
| P1 | Bounded conversation memory | compaction reduces tokens, leaves a summary | long sessions must not grow without limit |
| P1 | Vector long-term memory | recall returns the written item | episodic recall over sessions |
| P2 | Monitoring endpoints | counters, latencies, invariants | observability |

---

## 3. Explicitly out of scope (locked)

| Not doing | Reason | Revisit when |
|---|---|---|
| Persistent vector database | in-memory rebuild is exact and sufficient at this scale; a database adds an operational dependency that would break the offline guarantee | corpus exceeds ~10^5 chunks |
| User accounts / multi-tenancy | orthogonal to the retrieval and reasoning questions this system exists to answer | a deployment needs isolation |
| Fine-tuning or training | every claim here is about *composition*, not model quality | a domain-specific encoder is demonstrably required |
| Streaming token-level generation with the offline reader | the offline reader is extractive; streaming would add transport complexity with no analytical value | a generative backend is the default |
| Web UI | the deliverable is a system and its verification chain, not a front end | a demo audience requires it |
| PDF ingestion in-tree | the parser is a bounded, well-served problem and would pull a heavy dependency into the core install | `pypdf` is already pinned in `requirements-optional.txt` for callers who need it |
| Graph / agentic multi-hop retrieval | unproven benefit at this corpus scale; it would add a failure surface without a measurement to justify it | a multi-hop evaluation set exists |

---

## 4. Technology stack (locked, version-anchored)

| Layer | Technology | Verified version | Why |
|---|---|---|---|
| Language | CPython | 3.13.14 (min 3.11) | `slots=True` dataclasses, structural pattern matching, `tomllib` |
| HTTP | FastAPI | 0.141.1 | request validation + automatic OpenAPI |
| ASGI server | uvicorn | 0.53.0 | production-grade, pure Python fallback path |
| Validation | pydantic | 2.13.5 | required by FastAPI 0.141 |
| CLI | argparse (stdlib) | — | must work in an interpreter with no third-party packages |
| Logging | stdlib `logging` (+ optional loguru) | loguru 0.7.3 | zero-dependency default |
| Dense index | in-memory cosine (+ optional FAISS) | faiss-cpu 1.15.1 | exact search, bit-comparable across backends |
| Sparse retrieval | in-house Robertson BM25 | — | see section 3 of ARCHITECTURE.md |
| Embeddings | signed hashing (+ optional ONNX) | fastembed 0.8.0 | reproducible without a model download |
| Local inference | llama.cpp (optional) | llama-cpp-python 0.3.35 | CPU-only deployment |
| Test framework | pytest | 9.1.1 | |

**No lock-in.** The offline path executes with an empty site-packages. Optional
dependencies change answer quality; they never change capability.

---

## 5. API surface (locked)

| Method | Path | Purpose | Auth | Request | Response |
|---|---|---|---|---|---|
| GET | `/health` | liveness + corpus size | none | — | `HealthResponse` |
| GET | `/health/deep` | runtime invariants | none | — | `{ok, checks, guard_probe, deterministic_probe}` |
| GET | `/v1/stats` | provider identity, counters, latency | none | — | `{version, uptime_s, build, corpus, settings, counters, latency_ms}` |
| GET | `/v1/corpus` | indexed document ids | none | — | `{doc_ids, chunks, dim}` |
| POST | `/v1/ingest` | add or replace documents | none | `{documents: {id: text}, replace: bool}` | `IngestResponse` |
| POST | `/v1/query` | answer one question | none | `{question, k?}` | `QueryResponse` |
| POST | `/v1/query/stream` | answer as SSE | none | `{question, k?}` | `text/event-stream` |
| POST | `/v1/eval` | run the offline evaluation set | none | `{k?, include_cases?}` | `EvalReport` |
| POST | `/v1/guard/probe` | reranking regression probe | none | — | `{baseline_top1, guarded_top1, naive_top1, spearman, ...}` |

Machine-readable contract: [`openapi.yaml`](openapi.yaml).

Validation rules: `question` is required, 1–4000 characters; `k` is optional and
constrained to `1..50`. Violations produce `422` with a structured body.

---

## 6. Data model (locked)

No external database. The document store is the single source of truth; both
indexes are derived and rebuilt from it on every ingest.

| Structure | Key | Fields | Index | Notes |
|---|---|---|---|---|
| `documents` | `doc_id` | `text` | — | replaced wholesale on re-ingest |
| `chunks` | `chunk_id = "{doc_id}#{ordinal}"` | `text, ordinal, heading, doc_id, meta` | sorted by `chunk_id` | ids are stable, so re-ingest is idempotent |
| dense index | `chunk_id` | L2-normalised float vector (512) | exact cosine | rebuilt from `chunks` |
| sparse index | `chunk_id` | term counts + document frequency | inverted by term | rebuilt from `chunks` |

Invariant, asserted in tests and reported by `/health/deep`:

```text
len(dense_index) == len(sparse_index) == count(chunks)
```

---

## 7. Surface inventory (locked)

| Surface | Entry point | Core components | Backing API | Configuration theme |
|---|---|---|---|---|
| CLI | `python -m atlas` | `cli.main`, subcommands | in-process container | `ATLAS_*` env |
| HTTP | `atlas.api.app:app` | routers in `api/app.py` | in-process container | `ATLAS_*` env |
| Programmatic | `AtlasContainer` | `ingest`, `ask`, `evaluate`, `selfcheck` | — | `Settings` |
| Verification | `scripts/verify.py` | 8 stages | container + HTTP | `SCORE_FLOORS` |

---

## 8. Configuration contract (locked)

All defaults select the offline path. Nothing must be set to run.

| Group | Keys | Defaults | Locked because |
|---|---|---|---|
| Backends | `LLM_PROVIDER`, `EMBED_PROVIDER`, `RERANK_PROVIDER`, `INDEX_BACKEND` | `mock`, `hash`, `lexical`, `memory` | offline guarantee |
| CPU | `CPU_THREADS` | `4` | measured: `cpu_count-1` is ~4x slower for small quantised models |
| Retrieval | `TOP_K`=6, `RRF_K`=60, `QUERY_EXPANSION`=true | — | measured on the packaged corpus |
| Guard | `GUARD_ENABLED`=true, `GUARD_ALPHA`=0.5, `GUARD_SPEARMAN_FLOOR`=0.2 | — | measured reranking regression |
| Grounding | `GROUNDING_FLOOR`=0.18 | — | tuned for this corpus |
| Chunking | `CHUNK_SIZE`=800, `CHUNK_OVERLAP`=120 | — | 19 chunks over 6 documents |
| Reasoning | `MAX_STEPS`=6, `DET_ROUTER`=true | — | cost and correctness bound |

---

## 9. Acceptance criteria (EARS, locked)

| ID | Capability | Criterion | Priority |
|---|---|---|---|
| AC-01 | Chunking | *The system **must** partition the source into base spans that are strictly increasing, non-overlapping and gapless.* | P0 |
| AC-02 | Chunking | *When a document is longer than the chunk size, the system **must** emit at least two chunks.* | P0 |
| AC-03 | Chunking | *The system **must** attach the heading in force at each chunk's start, defaulting to the first heading for the opening chunk.* | P1 |
| AC-04 | Sparse | *For all `0 <= n <= N`, the IDF **must** be greater than or equal to zero.* | P0 |
| AC-05 | Dense | *If a document has been indexed, then querying with its own text **must** return it at rank 1.* | P0 |
| AC-06 | Dense | *The system **must** produce identical embeddings across processes for identical input.* | P0 |
| AC-07 | Fusion | *When both retrievers rank a document first, it **must** rank first after fusion.* | P0 |
| AC-08 | Fusion | *When scores tie, the system **must** order deterministically.* | P0 |
| AC-09 | Guard | *While the guard is enabled and the reranker is negatively correlated with the retriever, the system **must** retain the retriever's top-1 at rank 1.* | P0 |
| AC-10 | Guard | *If the reranker scores every candidate identically, the system **must** treat its rank correlation as zero.* | P0 |
| AC-11 | Routing | *When a query parses as a pure arithmetic expression, the system **must** route it to the calculator and **must not** invoke the language model.* | P0 |
| AC-12 | Routing | *If a query contains digits but is not a pure arithmetic expression, the system **must not** route it to a tool.* | P0 |
| AC-13 | Tools | *If an expression contains a call, attribute access or import, the system **must** raise a tool error.* | P0 |
| AC-14 | Tools | *If an arithmetic operation divides by zero or overflows, the system **must** return a failed tool result rather than raising.* | P0 |
| AC-15 | Reasoning | *The reasoning loop **must** terminate within the configured step budget.* | P0 |
| AC-16 | Reasoning | *When the budget is exhausted, the system **must** still return a non-empty answer.* | P0 |
| AC-17 | Reasoning | *The system **must not** treat a protocol directive line as a final answer.* | P0 |
| AC-18 | Grounding | *When an answer sentence is unsupported by every evidence sentence, the system **must** list it as unsupported.* | P0 |
| AC-19 | Grounding | *The system **must not** count provenance markers as claims.* | P0 |
| AC-20 | Ingest | *When a document is re-ingested with fewer chunks, the system **must** remove the surplus chunks of the previous revision.* | P0 |
| AC-21 | Ingest | *After ingest, `len(dense) == len(sparse) == count(chunks)` **must** hold.* | P0 |
| AC-22 | Evaluation | *The system **must** produce bit-identical scorecards across runs, excluding wall-clock fields.* | P0 |
| AC-23 | API | *If a required field is missing or out of range, the system **must** return 422.* | P0 |
| AC-24 | API | *If a path is unknown, the system **must** return 404.* | P0 |
| AC-25 | API | *The streaming endpoint **must** emit `data:` frames terminated by `[DONE]`.* | P1 |
| AC-26 | Providers | *If an optional dependency is absent, the system **must** fall back and report the substitution in `/v1/stats`.* | P0 |
| AC-27 | Repository | *The repository **must** contain no pictographic characters.* | P0 |

---

## 10. Boundaries and constraints

- **Offline by default.** The verification chain performs no network access. If
  it ever did, the guarantee that this system is reproducible would be void.
- **No GPU required.** Designed for CPU-only deployment; the CPU thread count is
  an explicit, audited setting rather than a derived one.
- **Windows-first.** Developed and verified on Windows 11 / CPython 3.13. The
  process-reaping path uses `taskkill /T /F`; the POSIX branch is present but is
  not part of the verified evidence.
- **Performance envelope**, from `python -m atlas eval` and the packed corpus:

  | Operation | p50 | p95 |
  |---|---|---|
  | end-to-end query | ~21 ms | ~33 ms |
  | dense search | ~1.5 ms | ~3 ms |
  | sparse search | ~1.5 ms | ~3 ms |
  | ingest 6 documents (19 chunks) | ~33 ms | — |

  These are wall-clock and therefore excluded from the determinism assertion.

- **Corpus scale.** The packaged evaluation corpus is 6 documents / 19 chunks.
  The numbers in the README characterise *this* corpus; they are not a claim
  about arbitrary corpora.
- **Browsers.** Not applicable — the deliverable has no web UI.

---

## 11. Embedded known pitfalls

Pulled from the failure table in `ARCHITECTURE.md` section 8, restricted to the
defects that are easy to reintroduce.

| Pitfall | Fingerprint | Root cause | Fix |
|---|---|---|---|
| Reranker replaces the fused order | rerank | reranker treated as an oracle | blend under a correlation guard |
| Constant reranker reads as perfect agreement | fusion, spearman | rank correlation undefined for constants | return 0.0 when either input is constant |
| BM25 IDF goes negative | bm25 | wrong IDF transcription | Robertson form `ln(1 + ...)` |
| Two-document BM25 corpus gives zero IDF | bm25, tests | target term in half the corpus | test corpora must have at least three documents |
| Only the first line of a chunk reaches the model | reasoner, mock | line-oriented evidence parsing | flatten on render, accumulate on parse |
| Prompt delimiter collides with instruction prose | reasoner | delimiter name appears in the instructions | use a globally unique tag, never in prose |
| `hash()` used for feature hashing | embeddings | per-process salt | BLAKE2b |
| `eval()` used for arithmetic | tools | convenience | whitelisted AST walk |
| Sub-selection with an empty relevant set scores zero | evaluator | retrieval metrics applied to tool-routed cases | exclude them and count separately |
| Wall-clock fields break reproducibility assertions | evaluator | latency in the compared payload | `deterministic_view()` |
| Routes registered outside `create_app` | api | split responsibility | register inside `create_app` |
| Literal symbols in a regex character class | any | GBK code page corruption | code-point ranges only |
| Test data shorter than the chunk threshold | tests, e2e | multi-chunk path never executed | probe documents must exceed the threshold |
| Test corpus perfectly periodic | chunking tests | every boundary lands on the same heading | non-uniform test data |

---

## 12. End-to-end verification steps (locked)

```bash
# 1. Verify. Runs offline; exits non-zero on any failure.
python scripts/verify.py
# expect: RESULT: ALL GREEN, 8 stages passed

# 2. Start the service.
python -m atlas serve --host 127.0.0.1 --port 8077 &

# 3. Core success flow: retrieval with citations.
curl -s http://127.0.0.1:8077/v1/query -H 'Content-Type: application/json' \
  -d '{"question":"混合检索里为什么不能直接对分数做加权求和？"}'
# assert: HTTP 200, .route == "retrieve", .citations non-empty, .grounding.support_rate == 1.0

# 4. Core success flow: deterministic tool bypass.
curl -s http://127.0.0.1:8077/v1/query -H 'Content-Type: application/json' \
  -d '{"question":"计算 12*(3+4)+18/3 等于多少？"}'
# assert: HTTP 200, .route == "arithmetic", .steps == 0, "90" in .answer

# 5. Key error flow: validation.
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8077/v1/query \
  -H 'Content-Type: application/json' -d '{"question":""}'
# assert: 422

# 6. Key error flow: unknown route.
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8077/v1/nope
# assert: 404

# 7. Scorecard.
curl -s http://127.0.0.1:8077/v1/eval -H 'Content-Type: application/json' \
  -d '{"include_cases":false}'
# assert: .route_accuracy == 1.0, .doc_hit_rate >= 0.95, .grounding_rate >= 0.95
```

---

## 13. Change log

| Date | Change | Reason | Impact |
|---|---|---|---|
| 2026-09-24 | v1.0.0 frozen | initial release | — |
| 2026-09-24 | Added thesaurus query expansion (P1) | measured cross-lingual recall gap: a Chinese CPU question missed its English source document | `engines/expansion.py`, `query_pipeline` sparse branch |
| 2026-09-24 | Raised the hash embedding dimension 256 to 512 | measured recall@6 0.739 to 0.755 | `providers/registry.py` |
| 2026-09-24 | Added document-level retrieval metrics | chunk-level relevance is over-broad without passage annotations | `engines/evaluator.py`, README scorecard |
| 2026-09-24 | Excluded tool-routed cases from retrieval metrics | they retrieve nothing by design; counting them as failures was wrong | `engines/evaluator.py` |
| 2026-09-24 | Added `strip_citations` to grounding | the guard was flagging its own citations | `engines/grounding.py` |
