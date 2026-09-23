# ADR-005: Robertson IDF written in-house instead of using `rank_bm25`

## Status
Accepted (2026-09-24)

## Background

`rank_bm25` is the obvious dependency for sparse retrieval. It was evaluated
first, and it produced a wrong ranking on this corpus.

`BM25Okapi` computes `idf = ln((N - n + 0.5) / (n + 0.5))` and then applies a
floor of `epsilon * average_idf`. The expression goes **negative** when a term
appears in more than half the documents, and the floor rule does not repair it —
on a two-document corpus with the target term in one document the measured
scores were negative and the correct document ranked last.

Two things were already wrong before that: a two-document corpus is a broken test
corpus for any IDF-based method, and the library's floor behaviour makes the
failure mode non-obvious rather than loud.

## Decision

Write the sparse retriever in-house, using the Robertson/Sparck-Jones form:

```
idf = ln(1 + (N - n + 0.5) / (n + 0.5))
```

This is strictly positive for every `0 <= n <= N`, so term frequency can only
increase a document's score. The property is asserted across the full `(N, n)`
grid.

Supporting decisions:

- CJK tokenisation uses character bigrams, because a whitespace tokeniser makes
  BM25 useless on Chinese.
- Unit-test corpora must contain **at least three documents** and the target term
  must not appear in half of them. Otherwise IDF degenerates to `ln(1) = 0` and
  the test passes vacuously. This is documented in the test module's docstring.
- `rank_bm25` is not a dependency of this project. A note explaining why is kept
  in the module docstring so the decision is discoverable from the code.

## Consequences

**Positive**

- The ranking is provably monotone in term frequency for the whole parameter
  range, and the invariant is executable.
- No dependency, and no third-party IDF convention to reconcile.
- The tokeniser and the scorer share one representation, so an IDF change and a
  tokenisation change cannot drift apart.

**Negative**

- Reimplementing a well-known component is a maintenance cost, and this
  implementation has not been benchmarked against tuned production indexes at
  scale. Scope is stated: at the packaged corpus size (19 chunks) it is exact and
  fast; larger corpora should use a real index.
- No BM25 variant support (BM25L, BM25+, language models). Not needed here.

## Related
`atlas/engines/bm25.py`, `tests/test_bm25_fusion.py`
