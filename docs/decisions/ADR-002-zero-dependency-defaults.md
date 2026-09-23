# ADR-002: Zero-dependency defaults with loud fallback

## Status
Accepted (2026-09-24)

## Background

The verification story depends on being able to run the entire system with no
network and no API key. Two designs were available:

- **A.** Test doubles only: fakes live under `tests/`, production always needs
  real backends.
- **B.** Real default implementations: every port has a genuine implementation
  that requires nothing, selected when no backend is configured.

Option A is conventional, but it means the untested configuration is the one
users get on a fresh clone. Option B means the default configuration is the one
that is continuously verified.

## Decision

Every port has a real, dependency-free default:

| Port | Default | Not a stub because |
|---|---|---|
| `LLMProvider` | `MockLLM` | extracts answers from the evidence block and drives the real ReAct protocol |
| `EmbeddingProvider` | `HashEmbeddingProvider` | signed-hashing embeddings with sub-linear term weighting |
| `RerankerProvider` | `LexicalReranker` | independent IDF-weighted ranking signal |
| `VectorIndex` | `MemoryVectorIndex` | exact cosine search |
| `DocumentStore` | `InMemoryStore` | source of truth, with JSONL persistence |

Additionally: when a **requested** backend is unavailable, the registry falls
back to the default, logs a `provider.fallback` event, and reports the
substitution under `build` in `GET /v1/stats`.

## Consequences

**Positive**

- `git clone && python scripts/verify.py` works on a machine with a bare
  CPython. No install step is on the critical path for correctness.
- The default configuration is the verified configuration.
- Degraded deployments are auditable: the substitution is queryable.

**Negative**

- The defaults are weaker than the backends they stand in for. A reader could
  mistake the offline scorecard for a claim about production quality. Mitigated
  by stating the limitation explicitly in the README and by keeping the
  scorecard's corpus size (6 documents / 19 chunks) in the same table.
- Two code paths for embedding and indexing must stay numerically consistent.
  They are compared in `scripts/_probe_dims.py`, and the determinism stage runs
  the same workload twice to catch divergence.

## Related
ADR-001, ADR-003
