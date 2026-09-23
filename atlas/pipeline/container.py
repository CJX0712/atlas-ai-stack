"""Composition root - the single place where concrete providers are assembled.

Nothing above this module imports a provider; nothing below it knows about HTTP.
That is what keeps every layer independently testable.

Author: 晨星
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import Chunk
from ..engines import router as router_engine
from ..engines.bm25 import SparseIndex
from ..engines.evaluator import Case, EvalReport, evaluate
from ..engines.fusion import FusedHit
from ..engines.rerank_guard import guarded_rerank, naive_rerank, top1
from ..engines.tools import safe_eval
from ..providers import registry
from ..resources import load_cases, load_corpus
from ..settings import Settings
from ..telemetry import LATENCY, METRICS, log_event
from .indexing import IngestReport, IndexingPipeline
from .querying import QueryPipeline


@dataclass(slots=True)
class BuildInfo:
    llm: str
    embedder: str
    reranker: str
    index: str
    dim: int
    cpu_threads: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm": self.llm,
            "embedder": self.embedder,
            "reranker": self.reranker,
            "index": self.index,
            "dim": self.dim,
            "cpu_threads": self.cpu_threads,
        }


class AtlasContainer:
    """Owns the object graph for one ATLAS instance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.store = registry.build_store(self.settings)
        self.embedder = registry.build_embedder(self.settings)
        self.index = registry.build_index(self.settings, self.embedder.dim)
        self.sparse = SparseIndex()
        self.reranker = registry.build_reranker(self.settings)
        self.llm = registry.build_llm(self.settings)

        self.indexing = IndexingPipeline(
            self.store, self.embedder, self.index, self.sparse, self.settings
        )
        self.querying = QueryPipeline(
            settings=self.settings,
            store=self.store,
            embedder=self.embedder,
            index=self.index,
            sparse=self.sparse,
            reranker=self.reranker,
            llm=self.llm,
        )
        self.started_at = time.time()
        log_event("container.built", **self.build_info().as_dict())

    # -- identity ---------------------------------------------------------
    def build_info(self) -> BuildInfo:
        return BuildInfo(
            llm=getattr(self.llm, "name", type(self.llm).__name__),
            embedder=getattr(self.embedder, "name", type(self.embedder).__name__),
            reranker=(getattr(self.reranker, "name", None) or "none")
            if self.reranker
            else "none",
            index=getattr(self.index, "backend", type(self.index).__name__),
            dim=self.embedder.dim,
            cpu_threads=self.settings.cpu_threads,
        )

    # -- ingest -----------------------------------------------------------
    def ingest(self, documents: Mapping[str, str]) -> IngestReport:
        return self.indexing.ingest(documents)

    def load_demo_corpus(self) -> IngestReport:
        return self.indexing.ingest(load_corpus())

    def reset(self) -> None:
        for doc_id in list(self.store.documents()):
            self.store.drop_document(doc_id)
        self.indexing._rebuild()

    # -- query ------------------------------------------------------------
    def ask(self, question: str, k: int | None = None) -> dict[str, Any]:
        return self.querying.query_payload(question, k)

    def retrieve(self, question: str, k: int | None = None) -> list[Chunk]:
        return self.querying.retrieve(question, k).chunks

    # -- evaluation -------------------------------------------------------
    def evaluate(self, k: int | None = None) -> EvalReport:
        top_k = k or self.settings.top_k
        by_doc: dict[str, list[str]] = {}
        for chunk in self.store.all_chunks():
            by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

        cases: list[Case] = []
        for raw in load_cases():
            doc_ids = [str(d) for d in raw.get("relevant_docs", [])]
            relevant: list[str] = []
            for doc_id in doc_ids:
                relevant.extend(by_doc.get(doc_id, []))
            cases.append(
                Case(
                    question=str(raw["question"]),
                    relevant=tuple(relevant),
                    expected_route=raw.get("expected_route"),
                    relevant_docs=tuple(doc_ids),
                )
            )
        return evaluate(cases, self.querying.as_runner(), k=top_k)

    # -- verification helpers --------------------------------------------
    def guard_probe(self) -> dict[str, Any]:
        """Reproduce the reranking regression offline, in-process.

        An adversarial reranker reverses the candidate order. The guarded path
        must keep the fused top-1 at rank 1; the naive path must not.
        """
        hits = [
            FusedHit(f"probe#{i}", score=10.0 - i, ranks={"dense": i + 1}, sources=["dense"])
            for i in range(5)
        ]
        texts = [f"probe passage {i}" for i in range(5)]

        def adversarial(_query: str, candidates: list[str]) -> list[tuple[int, float]]:
            return [(i, float(i)) for i in range(len(candidates))]

        guarded, decision = guarded_rerank(
            "probe",
            hits,
            texts,
            adversarial,
            alpha=self.settings.guard_alpha,
            spearman_floor=self.settings.guard_spearman_floor,
            enabled=True,
        )
        naive = naive_rerank(hits, texts, adversarial, "probe")
        baseline_top = top1(hits)
        return {
            "baseline_top1": baseline_top,
            "guarded_top1": top1(guarded),
            "naive_top1": top1(naive),
            "guard_applied": decision.applied,
            "spearman": round(decision.spearman, 6),
            "reason": decision.reason,
            "guarded_preserves_top1": top1(guarded) == baseline_top,
            "naive_breaks_top1": top1(naive) != baseline_top,
        }

    def deterministic_probe(self) -> dict[str, Any]:
        """Every arithmetic route must be answered by the tool, not the model."""
        samples = ["12*(3+4)+18/3", "(128+64)/8-100", "计算 99*101 等于多少", "2^10"]
        routes = []
        all_ok = True
        for question in samples:
            route = router_engine.route(question, deterministic=True)
            ok = route.kind == router_engine.ARITHMETIC
            if ok:
                try:
                    safe_eval(route.payload)
                except Exception:
                    ok = False
            all_ok = all_ok and ok
            routes.append({"question": question, "kind": route.kind, "ok": ok})
        return {"all_arithmetic_routed": all_ok, "samples": routes}

    def selfcheck(self) -> dict[str, Any]:
        """Runtime invariants reported by ``/health`` and ``atlas verify``."""
        guard = self.guard_probe()
        deterministic = self.deterministic_probe()
        checks = {
            "corpus_loaded": self.store.count_chunks() > 0,
            "deterministic_routing": bool(deterministic["all_arithmetic_routed"]),
            "rerank_guard_holds": bool(guard["guarded_preserves_top1"]),
            "rerank_regression_reproducible": bool(guard["naive_breaks_top1"]),
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "guard_probe": guard,
            "deterministic_probe": deterministic,
        }

    # -- telemetry --------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "version": _version(),
            "uptime_s": round(time.time() - self.started_at, 2),
            "build": self.build_info().as_dict(),
            "corpus": {
                "documents": len(self.store.documents()),
                "chunks": self.store.count_chunks(),
                "doc_ids": self.store.documents(),
            },
            "settings": self.settings.as_dict(),
            "counters": METRICS.snapshot(),
            "latency_ms": LATENCY.summary(),
        }


def _version() -> str:
    from .. import __version__

    return __version__


__all__ = ["AtlasContainer", "BuildInfo"]
