# ATLAS — Architecture

> Companion to [`README.md`](README.md). This document covers the layer contract,
> every interface definition, the call graph, the swap points and the failure
> modes designed against.

Author: 晨星

---

## 1. Layer contract

The system is organised as six layers. The rule is absolute: **a layer imports
only from layers below it and from `atlas.contracts`.** Nothing below `pipeline`
knows that `pipeline` exists; nothing below `api` knows about HTTP.

```text
+---------------------------------------------------------------+
| L5  atlas.cli            argparse          (stdlib only)       |
+---------------------------------------------------------------+
| L4  atlas.api            FastAPI transport (no business logic) |
+---------------------------------------------------------------+
| L3  atlas.pipeline       composition root                      |
|       indexing.py   ingest -> chunks -> embeddings -> indexes  |
|       querying.py   route -> retrieve -> fuse -> rerank -> ... |
|       container.py  the single object graph                    |
+---------------------------------------------------------------+
| L2  atlas.engines        pure algorithms: no I/O, no network   |
|       text  chunking  bm25  fusion  rerank_guard  expansion    |
|       router  tools  reasoner  grounding  memory  evaluator    |
+---------------------------------------------------------------+
| L1  atlas.providers      swappable backends                    |
|       hash_embed  onnx_embed  lexical_rerank  cross_rerank     |
|       mock_llm  llama_cpp_llm  ollama_llm                      |
|       memory_index  faiss_index  memory_store  registry        |
+---------------------------------------------------------------+
| L0  atlas.contracts  atlas.settings  atlas.telemetry           |
|       Protocol declarations, value objects, configuration      |
+---------------------------------------------------------------+
```

Why this matters in practice: because `engines` never imports a provider, every
algorithm can be unit-tested with a three-line fake. Because `providers` only
implements `contracts`, swapping a hashed encoder for a transformer is a
one-environment-variable change with no caller edits.

---

## 2. Interface definitions

All ports live in `atlas/contracts.py`. They are structural `Protocol`s —
implementations do not inherit from them, they simply satisfy them.

### 2.1 Value objects

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str          # "{doc_id}#{ordinal}" - stable across re-ingest
    doc_id: str
    text: str
    ordinal: int = 0
    heading: str = ""      # structural context, inherited downwards
    meta: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class Scored:
    chunk_id: str
    score: float
    source: str = ""       # "dense" | "sparse"

@dataclass(slots=True)
class GroundingReport:
    support_rate: float
    unsupported: list[str]
    checked: int
    supported: int

@dataclass(slots=True)
class Answer:
    question: str
    text: str
    citations: list[str]
    route: str             # "retrieve" | "arithmetic" | "unit" | "date"
    steps: int             # model calls actually made
    trace: list[str]       # the ReAct trace, for auditability
    grounding: GroundingReport
    latency_ms: float
```

`Chunk` and `Scored` are frozen and slotted so they are hashable and cheap; they
travel through sets, dict keys and sort keys.

### 2.2 Ports

```python
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...
    def complete(self, messages, *, temperature: float = 0.0,
                 max_tokens: int = 512) -> str: ...
    def stream(self, messages, *, temperature: float = 0.0,
               max_tokens: int = 512) -> Iterable[str]: ...

class RerankerProvider(Protocol):
    @property
    def name(self) -> str: ...
    def rerank(self, query: str, candidates: Sequence[str],
               top_k: int) -> list[tuple[int, float]]: ...

class VectorIndex(Protocol):
    def add(self, ids, vectors) -> None: ...
    def search(self, vector, k: int) -> list[Scored]: ...
    def reset(self) -> None: ...
    def __len__(self) -> int: ...

class DocumentStore(Protocol):
    def put_chunks(self, chunks) -> None: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def all_chunks(self) -> list[Chunk]: ...
    def put_document(self, doc_id: str, text: str) -> None: ...
    def documents(self) -> list[str]: ...
```

Two contract details are load-bearing:

- `RerankerProvider.rerank` returns **`(index, score)` pairs**, not reordered
  text. Indices let the guard align reranker output with the fused list and
  detect disagreement; a reordered list would already have destroyed the
  information needed to police the reranker.
- `reset()` on `VectorIndex` exists because ingestion rebuilds indexes from the
  document store. Without it, re-ingesting a shorter document leaves orphan
  vectors behind — a corruption that is invisible until the system returns
  content the operator believes was deleted.

---

## 3. Call graph

### 3.1 Ingest

```text
AtlasContainer.ingest(documents)
  -> IndexingPipeline.ingest
       for each (doc_id, text):
           store.drop_document(doc_id)          # remove the previous revision
           store.put_document(doc_id, text)
           engines.chunking.split(text)         # -> RawChunk list with spans
           store.put_chunks(Chunk list)
       IndexingPipeline._rebuild
           sparse.reset(); index.reset()
           for chunk: sparse.add(chunk_id, text)          # BM25
           index.add(ids, embedder.embed(texts))          # dense
```

`_rebuild` is a full rebuild on every ingest. At this scale that is cheaper and
far safer than incremental deletion, and it guarantees the three structures — the
document store, the dense index and the sparse index — are in exact sync. The
invariant `len(index) == len(sparse) == store.count_chunks()` is asserted in
`tests/test_pipeline_integration.py`.

### 3.2 Query

```text
QueryPipeline.query(question)
  -> engines.router.route(question, deterministic=True)
       |
       +-- hit --> Reasoner.run(route)  -- deterministic tool --> Answer(steps=0)
       |             (the LLM is NEVER called)
       |
       +-- miss -> QueryPipeline.retrieve(question, k)
                     dense  = index.search(embedder.embed_query(q), k)
                     sparse = sparse_index.search(expanded(q), k)
                       ^ thesaurus expansion applies to the SPARSE branch only
                     fused  = reciprocal_rank_fusion([dense, sparse], k=60)
                     fused  = rerank_guard.guarded_rerank(q, fused, texts, reranker)
                     chunks = store.get_chunk(...) for the top k
                  -> Reasoner.run(question, route, chunks)
                       ReAct loop, bounded by max_steps
                  -> engines.grounding.verify(answer, evidence)
                  -> Answer
```

The guard sits **between** fusion and the reasoner, and its output is the only
ordering the reasoner ever sees. There is no code path in which a reranker
replaces the fused order.

---

## 4. The rerank guard, precisely

```text
rho = spearman(fused_scores, reranker_scores)

rho < 0                      -> alpha_eff = 0.0            reranker ignored
0 <= rho < spearman_floor    -> alpha_eff = alpha / 2      signal halved
rho >= spearman_floor        -> alpha_eff = alpha          blended

final_i = (1 - alpha_eff) * minmax(fused)_i
        +      alpha_eff  * minmax(reranker)_i
tie-break: fused index (preserves the deterministic retriever order)
```

Two subtleties that cost real debugging time:

1. **A constant reranker must not read as agreement.** Rank correlation is
   undefined for a constant input. With deterministic index tie-breaking, a
   reranker returning the same score for every candidate produced
   `spearman = +1.0` — i.e. a useless reranker was classified as fully
   trustworthy. `spearman()` now returns `0.0` when either input is constant.
2. **Blending must normalise both sides.** Cosine scores live in `[0, 1]`;
   reranker logits are unbounded. `minmax` maps both onto the same scale, and a
   constant vector maps to all-ones (fully trusted) rather than dividing by zero.

Reproducible evidence, in-process and offline:

```python
container.guard_probe()
# {'baseline_top1': 'probe#0', 'guarded_top1': 'probe#0', 'naive_top1': 'probe#4',
#  'spearman': -1.0, 'guarded_preserves_top1': True, 'naive_breaks_top1': True}
```

`naive_rerank` is deliberately kept in the tree so the regression it represents
stays reproducible rather than becoming folklore.

---

## 5. Tokenisation and sparse retrieval

Chinese has no word delimiters. A whitespace tokeniser makes BM25 useless, so
`engines.text.tokenize` emits **CJK character bigrams** plus ASCII word unigrams.
Character classification uses `ord()` code-point ranges only — no literal symbol
appears in the source, because a regex character class containing literal BMP
symbols is corrupted when written through a GBK code page and fails at import
time with `bad character range`.

IDF uses the Robertson/Sparck-Jones form:

```text
idf = ln(1 + (N - n + 0.5) / (n + 0.5))
```

This is strictly positive for every `0 <= n <= N`. The more commonly transcribed
`ln((N - n + 0.5) / (n + 0.5))` goes **negative** when a term appears in more
than half the documents, letting a high-frequency term depress a document's
score. The property is asserted across the full `(N, n)` grid in
`tests/test_bm25_fusion.py`.

---

## 6. Reasoning and grounding

### 6.1 Bounded ReAct

```text
for step in 1..max_steps:
    reply = llm.complete(prompt(question, evidence, scratchpad))
    if reply contains "Final Answer:"   -> stop
    if reply contains "Action: tool[..]"-> run tool, append Observation, continue
    otherwise                           -> stop (unparsable)
otherwise: budget exhausted             -> degraded extractive answer
```

Termination is structural, not hopeful: the loop is a `for` with a hard bound,
and the budget-exhausted branch still returns a well-formed `Answer` built from
the evidence. Two guards were added after observed failures:

- A bare `Action:` line is **not** a final answer. `parse_final` rejects outputs
  starting with `action:`, `thought:` or `observation:`, so an agent that never
  stops acting cannot leak its last directive to the user as an answer.
- Tool-routed queries never enter the loop at all. `steps == 0` and
  `llm.calls == 0` are both asserted.

The step budget is a *cost* control as much as a safety control: every iteration
is a model call.

### 6.2 Prompt delimiter discipline

The evidence block is wrapped in `<kb-context>` / `</kb-context>`. The literal
tag **never appears in the instruction prose**. This is not cosmetic: when the
delimiter name collided with a word used in the instructions, a mock reader
extracted the instruction text as if it were a retrieved passage.

Evidence lines are flattened to one line per chunk. A multi-line chunk rendered
naively is read line-by-line by anything that consumes the block format, so only
the first line of each chunk reached the model — roughly 90 % of the context was
silently discarded. Both the renderer and the reader parser now handle it, and
the round trip is asserted directly.

### 6.3 Grounding

Per answer sentence:

```text
support = max over evidence sentences of  |Q ∩ S| / |Q|
          where Q = content tokens of the answer sentence
                S = content tokens of an evidence sentence
claim   = len >= 6 chars, not a question, not a boilerplate hedge
```

The denominator is the **answer** sentence, deliberately. Using the union, or
the evidence sentence, would let a long evidence passage "support" anything,
which defeats the check.

`strip_citations` removes provenance markers before verification. Without it the
guard flagged its own citation — `(依据：hybrid-retrieval#0)` shares no token with
the evidence — and halved the support rate of perfectly grounded answers.

---

## 7. Provider substitution

`atlas/providers/registry.py` is the only module that names a concrete backend.

```text
requested backend          unavailable?             resolved to
----------------------------------------------------------------
onnx embeddings      ->    import fails       ->   HashEmbeddingProvider(512)
cross reranker       ->    import fails       ->   LexicalReranker
llama_cpp / ollama   ->    import fails       ->   MockLLM
faiss index          ->    import fails       ->   MemoryVectorIndex
```

Substitutions are logged as `provider.fallback` events and reported by
`GET /v1/stats` under `build`. A degraded deployment is visible, never silent.

The hashed encoder uses **BLAKE2b**, not the builtin `hash()`. Python salts string
hashing per process, so embeddings — and therefore every retrieval result — would
differ between runs. Determinism here is a correctness requirement, not a nicety:
`scripts/verify.py` asserts it.

---

## 8. Failure modes designed against

Each row is a defect that was observed during construction, with the fix and the
test that now guards it.

| # | Symptom | Root cause | Fix | Guard |
|---|---|---|---|---|
| 1 | Reranker demoted correct answers | Reranker replaced the fused order | Blend under a correlation guard | `test_rerank_guard.py` |
| 2 | Constant reranker trusted as perfect | Rank correlation undefined for constants | `spearman` returns 0.0 | `test_constant_reranker_is_not_treated_as_agreement` |
| 3 | Arithmetic answered by the model, wrongly | No deterministic interception | Router + tool bypass | `test_arithmetic_flow_bypasses_the_model` |
| 4 | Chinese query missed its English document | Lexical retriever is monolingual | Thesaurus expansion, sparse branch only | `test_expansion_makes_a_cross_lingual_query_retrievable` |
| 5 | Only the first line of each chunk reached the model | Line-oriented evidence parsing | Flatten on render, accumulate on parse | `test_rendered_evidence_round_trips_through_the_reader_parser` |
| 6 | Model never committed, looped to budget | Query-coverage scoring collapses for long questions | Cosine over token sets | `test_mock_provider_halts_within_two_calls` |
| 7 | Last `Action:` line returned as the answer | Single-line fallback accepted directives | Reject protocol lines in `parse_final` | `test_step_budget_is_enforced_and_degrades_gracefully` |
| 8 | Guard flagged its own citations | Citations scored as claims | `strip_citations` | `test_citation_markers_are_not_treated_as_claims` |
| 9 | Re-ingested shorter document left orphans | Incremental update without deletion | `drop_document` before re-insert | `test_reingesting_a_shorter_document_drops_stale_chunks` |
| 10 | Report cited the previous question's hits | Runner read stale `last_retrieval` | Capture retrieval locally | `test_tool_routed_cases_report_no_retrieval` |
| 11 | `create_app()` answered 404 everywhere | Routes registered by the caller only | `create_app` registers them | `tests/test_api_cli.py` |
| 12 | Division by zero escaped as `ZeroDivisionError` | Missing exception normalisation | `safe_eval` raises `ToolError` | `test_safe_eval_rejects_unsafe_or_invalid_input` |
| 13 | Headings unlabelled on the first chunk | First block starts *after* its heading | `heading_at` defaults to the first block | `test_heading_is_inherited_by_chunks` |
| 14 | Latency fields broke reproducibility assertions | Wall clock in the compared payload | `deterministic_view()` | `test_evaluate_is_bit_reproducible` |

---

## 9. Extension points

| To change | Do this | Do not |
|---|---|---|
| Add a backend | Implement the `Protocol` in `providers/`, register it in `registry.py` | Import the new provider from `engines/` or `pipeline/` |
| Add a tool | Add a handler to `engines/tools.REGISTRY` and a branch to `engines/router.route` | Put tool logic in the reasoner |
| Change the scoring formula | Edit `engines/`, keep the invariant test green | Tune it inside `querying.py` |
| Add a metric | Add it to `engines/evaluator` **and** to `SCORE_FLOORS` in `scripts/verify.py` | Report a metric nothing gates |
| Support a new corpus language | Extend `engines/text` CJK ranges and `engines/expansion.LEXICON` | Special-case language inside the pipeline |

Two rules keep the design honest:

1. **A new capability needs a new invariant.** If a module cannot state a property
   that must hold when it is correct, its contract is not yet understood.
2. **A number in the README must be reproducible by a script.** If `verify.py`
   cannot regenerate it, it does not belong in the documentation.
