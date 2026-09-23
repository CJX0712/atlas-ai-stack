"""The reranking regression, committed as a test rather than as folklore.

An adversarial reranker that reverses candidate order stands in for the real
failure mode: a cross-encoder that has no usable signal for this query and
therefore ranks by artefact. The guarded pipeline must not regress.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas.engines.fusion import FusedHit
from atlas.engines.rerank_guard import guarded_rerank, naive_rerank, top1

TEXTS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _hits(n: int = 5) -> list[FusedHit]:
    return [
        FusedHit(f"c{i}", score=float(n - i), ranks={"dense": i + 1}, sources=["dense"])
        for i in range(n)
    ]


def _reversing(_query: str, candidates: list[str]) -> list[tuple[int, float]]:
    """Adversarial: scores increase with position, i.e. exactly reversed."""
    return [(i, float(i)) for i in range(len(candidates))]


def _agreeing(_query: str, candidates: list[str]) -> list[tuple[int, float]]:
    """Aligned with the fused order."""
    return [(i, float(len(candidates) - i)) for i in range(len(candidates))]


def _weakly_agreeing(_query: str, candidates: list[str]) -> list[tuple[int, float]]:
    """Permutation chosen so that rho(FUSED, RERANK) = 1 - 32/35 = 0.0857.

    Ranks produced: [2, 3, 6, 5, 1, 4] against fused ranks [1, 2, 3, 4, 5, 6].
    """
    return [(0, 5.0), (1, 4.0), (2, 1.0), (3, 2.0), (4, 6.0), (5, 3.0)]


def _constant(_query: str, candidates: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.0) for i in range(len(candidates))]


# --------------------------------------------------------------------------


def test_adversarial_reranker_breaks_the_naive_path() -> None:
    hits = _hits()
    naive = naive_rerank(hits, TEXTS, _reversing, "q")
    assert top1(naive) == "c4", "the naive path must be provably broken for this regression to matter"


def test_guard_preserves_top1_under_an_adversarial_reranker() -> None:
    hits = _hits()
    guarded, decision = guarded_rerank("q", hits, TEXTS, _reversing, alpha=0.5)
    assert top1(guarded) == top1(hits)
    assert decision.applied is False
    assert decision.alpha_effective == 0.0
    assert "anti-correlated" in decision.reason


def test_guard_blends_when_the_reranker_agrees() -> None:
    hits = _hits()
    guarded, decision = guarded_rerank("q", hits, TEXTS, _agreeing, alpha=0.5)
    assert decision.applied is True
    assert decision.alpha_effective == pytest.approx(0.5)
    assert top1(guarded) == top1(hits)


def test_guard_halves_alpha_on_weak_agreement() -> None:
    hits = _hits(6)
    guarded, decision = guarded_rerank(
        "q", hits, TEXTS + ["zeta"], _weakly_agreeing, alpha=0.5, spearman_floor=0.2
    )
    assert 0.0 < decision.spearman < 0.2
    assert decision.alpha_effective == pytest.approx(0.25)
    assert "halved" in decision.reason
    assert len(guarded) == 6


def test_constant_reranker_is_not_treated_as_agreement() -> None:
    hits = _hits()
    _, decision = guarded_rerank("q", hits, TEXTS, _constant, alpha=0.5)
    assert decision.spearman == 0.0
    assert decision.alpha_effective == pytest.approx(0.25)


def test_guard_disabled_is_a_passthrough() -> None:
    hits = _hits()
    guarded, decision = guarded_rerank("q", hits, TEXTS, _reversing, enabled=False)
    assert [h.chunk_id for h in guarded] == [h.chunk_id for h in hits]
    assert decision.applied is False


def test_empty_candidate_set_is_handled() -> None:
    guarded, decision = guarded_rerank("q", [], [], _reversing)
    assert guarded == []
    assert decision.applied is False


def test_reranker_returning_nothing_is_handled() -> None:
    hits = _hits()
    guarded, decision = guarded_rerank("q", hits, TEXTS, lambda _q, _c: [])
    assert [h.chunk_id for h in guarded] == [h.chunk_id for h in hits]
    assert "no scores" in decision.reason


def test_guard_output_is_deterministic() -> None:
    first, _ = guarded_rerank("q", _hits(6), TEXTS + ["zeta"], _weakly_agreeing)
    second, _ = guarded_rerank("q", _hits(6), TEXTS + ["zeta"], _weakly_agreeing)
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]
