# Cross-encoder reranking: when it helps and when it silently destroys recall

A cross-encoder scores a (query, passage) pair jointly. It is strictly more
expressive than a bi-encoder because the query and the passage attend to each
other. This is why reranking is the default second stage in almost every
production retrieval stack.

It is also the stage that most often regresses a system, because of one
architectural mistake: letting the reranker *replace* the retrieved order.

## The failure mode, measured

Take a well-tuned hybrid retriever and a cross-encoder that was trained on
English query-passage pairs. Evaluate on Chinese queries. The reranker has no
usable signal about the query language, so it ranks largely by surface
artefacts of the passage. In a controlled twelve-query study, replacing the
fused order with the reranker order dropped top-1 accuracy from 11 out of 12 to
3 out of 12. The retriever was right; the reranker overruled it and was wrong.

The dangerous part is that aggregate metrics can look fine. The reranked list
still contains the right passage, so recall at k moves very little. Only top-1
and mean reciprocal rank reveal the damage.

## The guard

Never replace. Blend, and modulate the blend by how much you can trust the
reranker on this particular query.

1. Score every candidate with the reranker.
2. Compute the Spearman rank correlation between the reranker's scores and the
   fused retriever scores.
3. If the correlation is negative, the reranker disagrees with a component you
   already trust on this query. Ignore it completely: set the blend weight to
   zero.
4. If the correlation is non-negative but below a floor, the signal is noisy.
   Halve the blend weight.
5. Otherwise blend at the configured weight.

The blended score is (1 - alpha) times the normalised fused score plus alpha
times the normalised reranker score. Normalise both to [0, 1] first, and keep
the fused rank as the deterministic tie-break.

## Why a guard and not a better model

Choosing a multilingual reranker is also correct, and it should be done. But a
guard is still required, because the reranker's competence is query dependent.
A model that is strong on average can still be confidently wrong on an
out-of-distribution query, and there is no query-time signal that says so other
than the disagreement between two independent rankers. The guard converts that
disagreement into a conservative default.

## Verification

The regression is committed as a test, not as folklore. The test constructs an
adversarial reranker that reverses the candidate order. With the guard enabled
the retriever's top-1 must remain at rank 1. With the naive path it must not.
Both assertions run offline with no model download.
