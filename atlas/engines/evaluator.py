"""Offline, deterministic evaluation harness.

Metrics: Recall@k, Precision@k, MRR, nDCG@k, answer grounding rate and latency
percentiles.  The harness takes a ``runner`` callable so it can evaluate either
the in-process pipeline or a remote HTTP endpoint without changing this file.

Determinism contract: the same dataset run twice against the same pipeline must
produce bit-identical ranks and metrics. ``tests/test_evaluator.py`` asserts it.

Author: 晨星
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Case:
    """One evaluation item.

    Relevance is recorded at two granularities on purpose. ``relevant`` holds
    chunk ids; ``relevant_docs`` holds document ids. Without human-annotated
    *passages*, marking every chunk of a source document as relevant over-counts
    the denominator (a four-chunk document contributes three chunks that never
    contained the answer), which depresses chunk recall without saying anything
    about retrieval quality. Document-level hit rate is therefore the headline
    metric; chunk-level recall is reported alongside it as a diagnostic.
    """

    question: str
    relevant: tuple[str, ...]
    expected_route: str | None = None
    relevant_docs: tuple[str, ...] = ()


@dataclass(slots=True)
class CaseResult:
    question: str
    retrieved: list[str]
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float
    route: str
    grounded: float
    latency_ms: float
    doc_hit: float = 0.0
    doc_reciprocal_rank: float = 0.0


@dataclass(slots=True)
class EvalReport:
    n: int = 0
    retrieval_cases: int = 0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    doc_hit_rate: float = 0.0
    doc_mrr: float = 0.0
    grounding_rate: float = 0.0
    route_accuracy: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cases: list[CaseResult] = field(default_factory=list)

    def as_dict(self, include_cases: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n": self.n,
            "retrieval_cases": self.retrieval_cases,
            "recall_at_k": round(self.recall_at_k, 6),
            "precision_at_k": round(self.precision_at_k, 6),
            "mrr": round(self.mrr, 6),
            "ndcg_at_k": round(self.ndcg_at_k, 6),
            "doc_hit_rate": round(self.doc_hit_rate, 6),
            "doc_mrr": round(self.doc_mrr, 6),
            "grounding_rate": round(self.grounding_rate, 6),
            "route_accuracy": round(self.route_accuracy, 6),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
        }
        if include_cases:
            payload["cases"] = [
                {
                    "question": c.question,
                    "retrieved": c.retrieved,
                    "recall": round(c.recall, 6),
                    "reciprocal_rank": round(c.reciprocal_rank, 6),
                    "ndcg": round(c.ndcg, 6),
                    "route": c.route,
                    "grounded": round(c.grounded, 6),
                    "latency_ms": round(c.latency_ms, 3),
                }
                for c in self.cases
            ]
        return payload

    def deterministic_view(self, include_cases: bool = False) -> dict[str, Any]:
        """The report without wall-clock fields.

        Latency is the one quantity that legitimately differs between two runs
        of an identical workload. The reproducibility assertion compares this
        view; a regression that changes retrieval or routing still fails it.
        """
        volatile = {"latency_p50_ms", "latency_p95_ms"}
        payload = {k: v for k, v in self.as_dict(include_cases=False).items()
                   if k not in volatile}
        if include_cases:
            payload["cases"] = [
                {
                    "question": c.question,
                    "retrieved": c.retrieved,
                    "recall": round(c.recall, 6),
                    "reciprocal_rank": round(c.reciprocal_rank, 6),
                    "ndcg": round(c.ndcg, 6),
                    "route": c.route,
                    "grounded": round(c.grounded, 6),
                }
                for c in self.cases
            ]
        return payload


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    wanted = set(relevant)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    wanted = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in wanted
    )
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(k, len(wanted)) + 1))
    if ideal <= 0.0:
        return 0.0
    return dcg / ideal


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * max(0.0, min(1.0, q))
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def evaluate(
    cases: Sequence[Case],
    runner: Callable[[str], dict[str, Any]],
    *,
    k: int = 6,
) -> EvalReport:
    """Run ``runner`` for every case and aggregate metrics."""
    report = EvalReport(n=len(cases))
    if not cases:
        return report

    latencies: list[float] = []
    route_hits = 0
    route_total = 0
    retrieval: list[CaseResult] = []
    for case in cases:
        outcome = runner(case.question) or {}
        retrieved = list(outcome.get("retrieved", []))
        route = str(outcome.get("route", ""))
        latency = float(outcome.get("latency_ms", 0.0))
        grounded = float(outcome.get("grounded", 0.0))

        result = CaseResult(
            question=case.question,
            retrieved=retrieved,
            recall=recall_at_k(retrieved, case.relevant, k),
            precision=precision_at_k(retrieved, case.relevant, k),
            reciprocal_rank=reciprocal_rank(retrieved, case.relevant),
            ndcg=ndcg_at_k(retrieved, case.relevant, k),
            route=route,
            grounded=grounded,
            latency_ms=latency,
        )
        if case.relevant_docs:
            wanted_docs = set(case.relevant_docs)
            retrieved_docs = [cid.split("#", 1)[0] for cid in retrieved[:k]]
            result.doc_hit = 1.0 if set(retrieved_docs) & wanted_docs else 0.0
            result.doc_reciprocal_rank = next(
                (1.0 / rank for rank, doc in enumerate(retrieved_docs, start=1)
                 if doc in wanted_docs),
                0.0,
            )

        report.cases.append(result)
        latencies.append(latency)
        if case.expected_route:
            route_total += 1
            if route == case.expected_route:
                route_hits += 1
        if case.relevant_docs:
            retrieval.append(result)

    # Cases answered by a deterministic tool expect no evidence and retrieve
    # nothing. Averaging them into the retrieval metrics would score them as
    # total failures for a behaviour that is correct by construction, so they
    # are excluded from the retrieval aggregates and counted separately.
    report.retrieval_cases = len(retrieval)
    total = len(report.cases)
    if retrieval:
        count = len(retrieval)
        report.recall_at_k = sum(c.recall for c in retrieval) / count
        report.precision_at_k = sum(c.precision for c in retrieval) / count
        report.mrr = sum(c.reciprocal_rank for c in retrieval) / count
        report.ndcg_at_k = sum(c.ndcg for c in retrieval) / count
        report.doc_hit_rate = sum(c.doc_hit for c in retrieval) / count
        report.doc_mrr = sum(c.doc_reciprocal_rank for c in retrieval) / count
    report.grounding_rate = sum(c.grounded for c in report.cases) / total
    report.route_accuracy = (route_hits / route_total) if route_total else 1.0
    report.latency_p50_ms = _percentile(latencies, 0.50)
    report.latency_p95_ms = _percentile(latencies, 0.95)
    return report
