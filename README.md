# ATLAS

<p align="center">
  <a href="https://github.com/CJX0712/atlas-ai-stack/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/atlas-ai-stack/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/atlas-ai-stack/releases"><img src="https://img.shields.io/github/v/release/CJX0712/atlas-ai-stack?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/atlas-ai-stack/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/atlas-ai-stack" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**Autonomous Task-oriented Layered Agentic System** — a modular, end-to-end runnable AI system.

Hybrid retrieval · guarded cross-encoder reranking · deterministic tool routing · ReAct reasoning · grounding verification · reproducible evaluation.

Author: **晨星** · License: MIT

---

## Why this is not another RAG demo

Most retrieval-augmented systems fail in ways that aggregate metrics hide. ATLAS
was built around four failure modes that were measured, not assumed, and each one
is committed as an executable test:

| Failure mode | What actually happens | How ATLAS handles it |
|---|---|---|
| **Reranker regression** | A cross-encoder with no signal for this query overrules a correct retriever. Measured: top-1 accuracy fell from 11/12 to 3/12 when the reranker replaced the fused order. | `atlas.engines.rerank_guard` never replaces — it blends, and modulates the blend weight by the rank correlation between the reranker and the retriever. Anti-correlated means ignored. |
| **Arithmetic by language model** | A 0.5B model answered `12*(3+4)` with `72`. Not a prompt problem — a capability boundary. | `atlas.engines.router` intercepts computable queries before the model is ever called. `steps == 0` is asserted. |
| **Cross-lingual retrieval blindness** | A Chinese question about CPU thread tuning cannot find its English source document: the query shares no token with the passage. | `atlas.engines.expansion` widens the **sparse branch only** with a curated bilingual thesaurus, keeping dense and sparse genuinely complementary. |
| **Ungrounded answers** | One fabricated sentence inside a fluent paragraph. Whole-answer similarity does not notice. | `atlas.engines.grounding` verifies claim by claim, sentence by sentence, against the evidence. |

---

## Verified scorecard

Produced by `python scripts/verify.py`, offline, with no API key and no network:

```text
==================================================================
ATLAS verification
==================================================================
[PASS] environment                        0.00s  python 3.13.14 on win32
[PASS] p0-scan                            1.26s  scanned 71 files, 0 finding(s)
[PASS] imports                            0.22s  27 modules
[PASS] unit-and-integration-tests         5.54s  157 passed
[PASS] end-to-end                         4.07s  e2e summary: passed=14 failed=0
[PASS] scorecard                          0.36s  recall@0.8512 mrr=0.9028 doc_hit=1.0000 grounding=1.0000
[PASS] invariants                         0.04s  guarded_top1=probe#0 naive_top1=probe#4 rho=-1.0
[PASS] determinism                        0.71s  two independent runs produced identical output
------------------------------------------------------------------
RESULT: ALL GREEN
```

| Metric | Workstation (NumPy index) | Clean room (pure Python) | Floor enforced |
|---|---|---|---|
| `doc_hit_rate` — right document in top-k | **1.000** | **1.000** | 0.95 |
| `doc_mrr` — reciprocal rank of the right document | **0.903** | **0.917** | 0.85 |
| `recall@6` (chunk level, over-broad ground truth) | **0.851** | **0.872** | 0.80 |
| `mrr` (chunk level) | **0.903** | **0.917** | 0.85 |
| `nDCG@6` | **0.822** | **0.839** | 0.75 |
| `grounding_rate` | **1.000** | **1.000** | 0.95 |
| `route_accuracy` | **1.000** | **1.000** | 1.00 |

Both columns are produced by `python scripts/verify.py`. The "clean room" column
comes from a freshly created virtual environment containing only
`requirements.txt` and `requirements-dev.txt` (21 packages, no NumPy, no FAISS).
Both runs report `RESULT: ALL GREEN`.

The two columns differ slightly, and the reason is worth stating: without NumPy
the in-memory index uses a pure-Python dot product, which differs from the
BLAS path in the last bits. Near-ties in the dense ranking can therefore order
differently, and RRF converts order into score. Both paths are internally
deterministic — that is asserted by the determinism stage — and both clear every
floor. Run `python -m atlas eval` to reproduce either column.

---

## Quick start

```bash
git clone https://github.com/CJX0712/atlas-ai-stack
cd atlas-ai-stack

# Optional but recommended: verify before trusting anything.
python scripts/verify.py

# Start the HTTP service.
python -m atlas serve --host 127.0.0.1 --port 8077
```

The verification chain and the default service both run with **zero third-party
packages installed**. The offline providers (deterministic extractive reader,
hashed embeddings, in-memory index) are real implementations, not stubs.

### Command line

```bash
python -m atlas doctor                       # what does this interpreter have?
python -m atlas ingest --demo                # index the packaged corpus
python -m atlas ingest ./docs ./notes        # index your own .md/.txt/.rst files
python -m atlas ask "混合检索为什么不能直接加权求和？"
python -m atlas ask "计算 12*(3+4)+18/3 等于多少？"
python -m atlas eval --cases                 # scorecard with per-case detail
python -m atlas selfcheck                    # runtime invariants
python -m atlas verify                       # full chain
```

### HTTP API

```bash
curl -s localhost:8077/health
curl -s localhost:8077/v1/query -H 'Content-Type: application/json' \
     -d '{"question":"接地核验为什么按句核验？"}'
curl -s localhost:8077/v1/eval  -H 'Content-Type: application/json' -d '{"include_cases":false}'
```

Interactive schema: <http://127.0.0.1:8077/docs>. The machine-readable contract is
[`docs/openapi.yaml`](docs/openapi.yaml).

### Docker

```bash
docker compose up --build     # serves on 127.0.0.1:8077
```

---

## Architecture

Dependencies point strictly downward. A layer may import from the layer below it
and from `contracts`; it never imports from above.

```text
L5  cli            argparse front end (stdlib only, so it works in a bare interpreter)
L4  api            FastAPI transport - thin, no logic
L3  pipeline       composition: indexing, querying, container
L2  engines        pure algorithms: no I/O, no providers, no network
L1  providers      swappable implementations of the L0 ports
L0  contracts      Protocol declarations + value objects. Zero dependencies.
```

```text
                 +------------------ L5 cli / L4 api ------------------+
                 |                                                    |
                 v                                                    |
   documents --> [IndexingPipeline]                                   |
                 |   chunk -> embed -> dense index + sparse index     |
                 v                                                    |
   question --> [router] --deterministic--> [tool] --------------------+
                 |                                                    
                 | retrieve                                           
                 v                                                    
        dense search ---+                                            
        sparse search --+--> [RRF fusion] --> [rerank guard] --> evidence
                        |                                            |
                        +--------------------------------------------+
                                                                     |
                                          [ReAct reasoner] <---------+
                                                 |
                                          [grounding verify]
                                                 |
                                                 v
                                              Answer
```

### Module responsibilities

| Module | Single responsibility | Independently verifiable invariant |
|---|---|---|
| `engines.text` | CJK bigram + ASCII tokenisation, sentence splitting | bigram/unigram behaviour on mixed script |
| `engines.chunking` | structure-aware splitting | base spans are an **exact gapless partition** of the source |
| `engines.bm25` | Robertson-IDF sparse retrieval | IDF is provably non-negative for all `0 ≤ n ≤ N` |
| `engines.fusion` | RRF over N ranked lists | rank-1 consensus wins; deterministic tie-break |
| `engines.rerank_guard` | blend reranker under a correlation guard | adversarial reranker cannot displace the retriever's top-1 |
| `engines.expansion` | thesaurus query expansion | unknown terms expand to nothing |
| `engines.router` | deterministic pre-routing | every arithmetic expression routes to the calculator |
| `engines.tools` | safe AST arithmetic, units, dates | `__import__`, division by zero and huge exponents rejected |
| `engines.reasoner` | bounded ReAct loop | terminates; step budget enforced; tool-routed queries never call the model |
| `engines.grounding` | per-claim support verification | fabricated sentence is flagged; citations are not claims |
| `engines.memory` | windowed transcript + vector recall | compaction reduces tokens and leaves a summary |
| `engines.evaluator` | metrics harness | bit-identical results across runs |
| `providers.*` | concrete backends | fake and real implementations are interchangeable |
| `pipeline.*` | composition | full chain runs offline |

Full detail, including the interface definitions and the call graph, is in
[`ARCHITECTURE.md`](ARCHITECTURE.md). The locked specification is
[`docs/SPEC.md`](docs/SPEC.md).

---

## Configuration

Every knob is an environment variable prefixed `ATLAS_`. Defaults are the offline
path; nothing needs to be set to run.

### Backend selection

| Variable | Values | Default | Effect |
|---|---|---|---|
| `ATLAS_LLM_PROVIDER` | `mock` \| `llama_cpp` \| `ollama` | `mock` | reasoning backend |
| `ATLAS_EMBED_PROVIDER` | `hash` \| `onnx` | `hash` | dense encoder |
| `ATLAS_RERANK_PROVIDER` | `lexical` \| `cross` \| `none` | `lexical` | reranker |
| `ATLAS_INDEX_BACKEND` | `memory` \| `faiss` | `memory` | vector index |
| `ATLAS_LLAMA_MODEL_PATH` | path | *(empty)* | local `.gguf` file |
| `ATLAS_OLLAMA_HOST` / `ATLAS_OLLAMA_MODEL` | url / name | `127.0.0.1:11434` / `qwen2.5:0.5b` | Ollama endpoint |

Missing optional packages do **not** break startup. The provider factory logs a
fallback event and reports the substitution in `GET /v1/stats`, so a degraded
deployment is visible rather than silent.

### Retrieval and reasoning

| Variable | Default | Notes |
|---|---|---|
| `ATLAS_TOP_K` | `6` | candidates returned to the reasoner |
| `ATLAS_RRF_K` | `60` | RRF smoothing constant |
| `ATLAS_QUERY_EXPANSION` | `true` | thesaurus expansion on the sparse branch |
| `ATLAS_GUARD_ENABLED` | `true` | rerank correlation guard |
| `ATLAS_GUARD_ALPHA` | `0.5` | blend weight when the reranker agrees |
| `ATLAS_GUARD_SPEARMAN_FLOOR` | `0.2` | below this the blend weight is halved |
| `ATLAS_GROUNDING_FLOOR` | `0.18` | per-claim support threshold |
| `ATLAS_MAX_STEPS` | `6` | ReAct step budget |
| `ATLAS_DET_ROUTER` | `true` | deterministic pre-routing |
| `ATLAS_CHUNK_SIZE` / `ATLAS_CHUNK_OVERLAP` | `800` / `120` | chunking |

### CPU inference

| Variable | Default | Notes |
|---|---|---|
| `ATLAS_CPU_THREADS` | `4` | **do not raise this to `cpu_count - 1`** |

Small quantised models on CPU are **memory-bandwidth bound, not compute bound**.
`llama.cpp` defaults to `cpu_count - 1` worker threads, which measured roughly
four times slower than a fixed four threads on this class of machine: beyond a
handful of workers the cores spend their time in cache-coherence stalls rather
than arithmetic. `ATLAS_CPU_THREADS` is also passed to `fastembed`.

---

## Verification model

Three independent layers of evidence, all runnable offline:

1. **Invariants** — `pytest`. 157 assertions on properties that must hold if the
   implementation is correct: exact coverage of the chunker, non-negative IDF,
   guard survival under an adversarial reranker, step-budget termination,
   fabricated-sentence detection.
2. **End-to-end** — `scripts/e2e.py`. Starts the real ASGI server as a child
   process, polls `/health` until ready, drives the public HTTP surface through
   the success flows **and every error flow**, then reaps the process tree. The
   ingested probe document is deliberately longer than the chunk threshold, so
   the multi-chunk path is genuinely exercised rather than assumed.
3. **Determinism** — two independent runs must produce identical scorecards and
   identical answers. Wall-clock fields are excluded from that comparison; a
   regression that changes retrieval or routing still fails it.

Plus a P0 gate (`tools/scan_emoji.py`) that rejects pictographic characters
anywhere in the repository, using code-point ranges only.

---

## Reproducing in a clean environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python scripts/verify.py
```

`requirements.lock.txt` is the **complete transitive closure** generated by
`pip freeze` inside a freshly created virtual environment that installed exactly
`requirements.txt` plus `requirements-dev.txt`. It is evidence of installability,
not a transcription. The lock intentionally contains no package-index URLs and no
Windows-only packages, so it installs on Linux containers unchanged.

[`requirements-optional.txt`](requirements-optional.txt) pins the heavyweight
backends (`fastembed`, `llama-cpp-python`) that are **not** part of the clean-room
chain — they are large, and every one of them has a working fallback.

---

## Known limitations

Stated plainly, because a scorecard without caveats is marketing.

- **The offline reader is extractive and monolingual in effect.** It answers by
  selecting the best evidence sentence. A Chinese question against an English
  document retrieves correctly (expansion bridges the gap) but the extracted
  answer sentence will be English. Set `ATLAS_LLM_PROVIDER=ollama` or
  `llama_cpp` for generative answers, and `ATLAS_EMBED_PROVIDER=onnx` for a
  multilingual encoder.
- **Chunk-level recall is measured against an over-broad ground truth.** Without
  human-annotated passages, every chunk of a source document is marked relevant,
  including chunks that never contained the answer. That is why document-level
  `doc_hit_rate` is the headline metric and chunk recall is reported alongside it.
- **Query expansion is a curated lexicon**, not a translation model. It covers
  this domain's technical vocabulary and expands unknown terms to nothing.
- **The grounding guard errs toward flagging.** With bigram tokenisation, a
  unigram stopword list cannot neutralise filler such as "这个和那个都是的了",
  so that sentence is reported as unsupported. A visible false alarm is cheaper
  than a silent pass.

---

## Project layout

```text
atlas-ai-stack/
  atlas/
    contracts.py          L0  Protocol declarations and value objects
    settings.py           L0  environment-backed configuration
    telemetry.py          L0  structured logging, counters, latencies
    engines/              L2  pure algorithms
    providers/            L1  swappable backends
    pipeline/             L3  composition root
    api/                  L4  HTTP surface
    cli.py                L5  command line
    resources/                packaged corpus + offline evaluation set
  tests/                      invariant and integration tests
  scripts/                    e2e.py, verify.py
  tools/scan_emoji.py         P0 gate
  docs/                       SPEC.md, openapi.yaml, decisions/
  ARCHITECTURE.md
```

## License

MIT. See [LICENSE](LICENSE).
