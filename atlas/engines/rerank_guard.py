"""Guarded cross-encoder reranking - the anti-regression wrapper.

The problem
-----------
Cross-encoder rerankers routinely **hurt** a well-tuned hybrid retriever when
they are allowed to dictate the final order.  The canonical failure: an English
trained reranker scored against Chinese queries.  It has no signal on the query
language, so it ranks by surface artefacts and silently demotes the correct
passage.  Measured in the companion study: top-1 accuracy fell from 11/12 to
3/12 when the reranker replaced the fused order outright.

The fix
-------
Never let the reranker *replace* the fused ranking - always **blend** it, and
modulate the blend weight by the rank correlation between the reranker and the
retriever that we already trust:

* ``spearman >= floor``  -> the reranker broadly agrees; blend at ``alpha``.
* ``0 <= spearman < floor`` -> noisy; blend at ``alpha / 2``.
* ``spearman < 0``       -> the reranker is anti-correlated; ignore it (alpha=0).

Guarantee asserted in ``tests/test_rerank_guard.py``: with an adversarial
(order-reversing) reranker the guarded pipeline keeps the retriever's top-1 at
rank 1, whereas the naive pipeline pushes it to last.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .fusion import FusedHit, minmax, spearman

RerankFn = Callable[[str, Sequence[str]], list[tuple[int, float]]]


@dataclass(slots=True)
class GuardDecision:
    applied: bool
    alpha_effective: float
    spearman: float
    reason: str


def _order_scores(hits: Sequence[FusedHit], scores: Sequence[float]) -> list[float]:
    """Align ``scores`` (indexed by candidate position) with ``hits`` order."""
    return [scores[i] for i in range(len(hits))]


def guarded_rerank(
    query: str,
    hits: Sequence[FusedHit],
    texts: Sequence[str],
    rerank_fn: RerankFn,
    *,
    alpha: float = 0.5,
    spearman_floor: float = 0.2,
    enabled: bool = True,
) -> tuple[list[FusedHit], GuardDecision]:
    """Blend a reranker into the fused ranking under a correlation guard."""
    if not hits:
        return [], GuardDecision(False, 0.0, 0.0, "empty candidate set")
    if not enabled:
        return list(hits), GuardDecision(False, 0.0, 0.0, "guard disabled by config")

    pairs = rerank_fn(query, list(texts))
    by_index = {idx: score for idx, score in pairs}
    if not by_index:
        return list(hits), GuardDecision(False, 0.0, 0.0, "reranker returned no scores")

    rerank_raw = [float(by_index.get(i, 0.0)) for i in range(len(hits))]
    fused_raw = [float(h.score) for h in hits]

    rho = spearman(fused_raw, rerank_raw)
    if rho < 0.0:
        alpha_eff, reason = 0.0, f"reranker anti-correlated (rho={rho:.3f}); ignored"
    elif rho < spearman_floor:
        alpha_eff = alpha / 2.0
        reason = f"weak agreement (rho={rho:.3f} < {spearman_floor}); alpha halved"
    else:
        alpha_eff = alpha
        reason = f"reranker agrees with retriever (rho={rho:.3f}); blended"

    fused_norm = minmax(fused_raw)
    rerank_norm = minmax(rerank_raw)
    blended = [
        (1.0 - alpha_eff) * fused_norm[i] + alpha_eff * rerank_norm[i]
        for i in range(len(hits))
    ]

    order = sorted(
        range(len(hits)),
        key=lambda i: (-blended[i], i),  # index tie-break == fused rank tie-break
    )
    out: list[FusedHit] = []
    for new_rank, idx in enumerate(order, start=1):
        hit = hits[idx]
        ranks = dict(hit.ranks)
        ranks["blend"] = new_rank
        out.append(
            FusedHit(
                chunk_id=hit.chunk_id,
                score=blended[idx],
                ranks=ranks,
                sources=sorted(set(hit.sources + ["rerank"])),
            )
        )

    applied = alpha_eff > 0.0
    return out, GuardDecision(applied, alpha_eff, rho, reason)


def naive_rerank(
    hits: Sequence[FusedHit],
    texts: Sequence[str],
    rerank_fn: RerankFn,
    query: str,
) -> list[FusedHit]:
    """Baseline that trusts the reranker completely - the failure mode we guard
    against. Kept in-tree so the regression is reproducible, not folklore."""
    pairs = rerank_fn(query, list(texts))
    scores = {idx: score for idx, score in pairs}
    order = sorted(range(len(hits)), key=lambda i: (-scores.get(i, 0.0), i))
    out: list[FusedHit] = []
    for new_rank, idx in enumerate(order, start=1):
        hit = hits[idx]
        ranks = dict(hit.ranks)
        ranks["naive"] = new_rank
        out.append(FusedHit(hit.chunk_id, scores.get(idx, 0.0), ranks, list(hit.sources)))
    return out


def top1(hits: Sequence[FusedHit]) -> str | None:
    return hits[0].chunk_id if hits else None
