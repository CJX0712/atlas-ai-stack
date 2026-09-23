# ADR-003: Blend the reranker under a correlation guard; never let it replace the fused order

## Status
Accepted (2026-09-24). Supersedes the original design, which let the reranker dictate the final order.

## Background

The first implementation reranked the fused candidate list and used the
reranker's ordering as the final ordering. This is the standard two-stage
retrieval recipe and it is what every tutorial shows.

It was measured, and it regressed the system. With an English-trained
cross-encoder and Chinese queries, top-1 accuracy on a twelve-query set fell
from 11/12 to 3/12. The reranker had no usable signal about the query language,
so it ranked by surface artefacts of the passage. Crucially, recall at k barely
moved — the correct passage was usually still in the list. Only top-1 and MRR
revealed the damage.

The lesson is not "use a better reranker". A reranker's competence is
query-dependent, and there is no query-time signal that says "I am out of
distribution" other than disagreement between two independent rankers.

## Decision

Blend, and modulate the blend weight by the rank correlation between the
reranker and the retriever:

```
rho = spearman(fused_scores, reranker_scores)

rho < 0                    -> alpha_eff = 0        reranker ignored entirely
0 <= rho < spearman_floor  -> alpha_eff = alpha/2  signal treated as noisy
rho >= spearman_floor      -> alpha_eff = alpha    blended

final = (1 - alpha_eff) * minmax(fused) + alpha_eff * minmax(reranker)
tie-break: fused index
```

Two implementation rules fall out of this:

- Rank correlation of a **constant** input is undefined. Returning the
  deterministic tie-break rank order instead produced `rho = +1.0` for a
  reranker that scored every candidate identically — a useless reranker
  classified as fully trustworthy. `spearman` returns `0.0` for constant input.
- Both score families must be normalised. Cosine lives in `[0, 1]`; reranker
  logits are unbounded.

The naive path is kept in the tree as `naive_rerank` so the regression stays
reproducible rather than becoming folklore, and
`AtlasContainer.guard_probe()` reports both outcomes.

## Consequences

**Positive**

- A wrong reranker can no longer destroy an answer the retriever got right. This
  is a structural guarantee, not a tuning outcome.
- The guard is a runtime diagnostic: a persistently zero `alpha_eff` means the
  reranker disagrees with the retriever on this query distribution.
- The regression is testable offline with a ten-line lambda.

**Negative**

- The reranker can never fully correct the retriever. If the retriever is wrong
  and the reranker is right, an anti-correlated reranker is ignored and the error
  survives. Accepted: an unhelpful reranker is far more common than a
  consistently better-than-retriever one, and the conservative direction fails
  visibly rather than silently.
- Two hyperparameters (`alpha`, `spearman_floor`) now require tuning per corpus.

## Related
ADR-002, `atlas/engines/rerank_guard.py`, `tests/test_rerank_guard.py`
