"""Reciprocal Rank Fusion (RRF, Cormack et al. 2009) over N ranked lists.

RRF was chosen over score-normalisation fusion deliberately: dense cosine
scores and BM25 scores live on incomparable scales, and min-max normalising
them per query is unstable when a list has one dominant outlier.

``rrf(d) = sum_r  w_r / (k + rank_r(d))`` with 1-based ranks.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import Scored


@dataclass(slots=True)
class FusedHit:
    chunk_id: str
    score: float
    ranks: dict[str, int]
    sources: list[str]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[Scored]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Fuse ``ranked_lists`` where each element is ``(name, ranked_hits)``."""
    if k < 1:
        raise ValueError("rrf k must be >= 1")
    weights = weights or {}
    acc: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    seen_order: dict[str, int] = {}

    for list_index, (name, hits) in enumerate(ranked_lists):
        weight = float(weights.get(name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            acc[hit.chunk_id] = acc.get(hit.chunk_id, 0.0) + weight / (k + rank)
            ranks.setdefault(hit.chunk_id, {})[name] = rank
            seen_order.setdefault(hit.chunk_id, list_index * 100000 + rank)

    hits_out = [
        FusedHit(
            chunk_id=cid,
            score=score,
            ranks=ranks[cid],
            sources=sorted(ranks[cid]),
        )
        for cid, score in acc.items()
    ]
    # Deterministic tie-break: score desc, then original appearance order.
    hits_out.sort(key=lambda h: (-h.score, seen_order[h.chunk_id]))
    if top_k is not None:
        hits_out = hits_out[: max(0, top_k)]
    return hits_out


def minmax(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1]. A constant vector maps to all-ones (fully trusted)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation. Returns 0.0 for degenerate input.

    A constant input has no defined rank correlation. Without this guard,
    deterministic index tie-breaking would assign it ascending ranks and report
    a perfect agreement of +1.0 - i.e. a reranker that returned the same score
    for every candidate would be treated as fully trustworthy.
    """
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    if len(set(a)) < 2 or len(set(b)) < 2:
        return 0.0

    def ranks_of(xs: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: (-xs[i], i))
        out = [0.0] * n
        for rank, idx in enumerate(order, start=1):
            out[idx] = float(rank)
        return out

    ra, rb = ranks_of(a), ranks_of(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = sum((v - mean_a) ** 2 for v in ra) ** 0.5
    db = sum((v - mean_b) ** 2 for v in rb) ** 0.5
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return num / (da * db)
