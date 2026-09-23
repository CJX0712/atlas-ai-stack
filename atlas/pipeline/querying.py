"""Query pipeline: route -> hybrid retrieve -> fused -> guarded rerank -> reason.

Every stage is a pure function over injected components, so the pipeline can be
driven with fakes in tests and with real backends in production without a single
branch inside this file.

The reasoner is constructed here (rather than injected) because its ``search``
tool needs a back-reference to this pipeline. Building it internally removes a
circular construction dependency between the toolbox and the retriever.

Author: 晨星
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from ..contracts import Answer, Chunk
from ..engines import expansion, rerank_guard, router
from ..engines.fusion import FusedHit, reciprocal_rank_fusion
from ..engines.reasoner import Reasoner, Toolbox
from ..telemetry import LATENCY, METRICS, log_event, timed


@dataclass(slots=True)
class RetrievalResult:
    hits: list[FusedHit]
    chunks: list[Chunk]
    guard_reason: str = ""
    guard_alpha: float = 0.0
    sparse_hits: int = 0
    dense_hits: int = 0

    def ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]


class QueryPipeline:
    """End-to-end question answering over the indexed corpus."""

    def __init__(
        self,
        *,
        settings,
        store,
        embedder,
        index,
        sparse,
        reranker,
        llm,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.index = index
        self.sparse = sparse
        self.reranker = reranker
        self.llm = llm
        self.reasoner = Reasoner(
            llm,
            toolbox=Toolbox(search_fn=self._tool_search),
            max_steps=settings.max_steps,
            deterministic_router=settings.det_router,
        )
        self.last_retrieval: RetrievalResult | None = None

    # -- retrieval --------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        top_k = max(1, k or self.settings.top_k)
        with timed("dense_search"):
            dense = self.index.search(self.embedder.embed_query(query), top_k)
        with timed("sparse_search"):
            # Expansion applies only to the sparse branch. Feeding the expanded
            # query to the dense branch too would collapse the two retrievers
            # onto one representation and destroy the complementarity that makes
            # fusion worth doing.
            sparse_query = (
                expansion.expand(query).query if self.settings.query_expansion else query
            )
            sparse = self.sparse.search(sparse_query, top_k)

        if not dense and not sparse:
            self.last_retrieval = RetrievalResult([], [])
            return self.last_retrieval

        fused = reciprocal_rank_fusion(
            [("dense", dense), ("sparse", sparse)],
            k=self.settings.rrf_k,
            weights={"dense": 1.0, "sparse": 1.0},
            top_k=top_k * 2,
        )

        guard_reason, guard_alpha = "", 0.0
        if self.reranker is not None and fused:
            texts = [self._text(h.chunk_id) for h in fused]
            fused, decision = rerank_guard.guarded_rerank(
                query,
                fused,
                texts,
                self.reranker.rerank,
                alpha=self.settings.guard_alpha,
                spearman_floor=self.settings.guard_spearman_floor,
                enabled=self.settings.guard_enabled,
            )
            guard_reason, guard_alpha = decision.reason, decision.alpha_effective
            METRICS.inc("guard.applied" if decision.applied else "guard.skipped")

        fused = fused[:top_k]
        chunks = [c for c in (self.store.get_chunk(h.chunk_id) for h in fused) if c]
        self.last_retrieval = RetrievalResult(
            hits=list(fused),
            chunks=chunks,
            guard_reason=guard_reason,
            guard_alpha=guard_alpha,
            sparse_hits=len(sparse),
            dense_hits=len(dense),
        )
        return self.last_retrieval

    # -- question answering ----------------------------------------------
    def query(self, question: str, k: int | None = None) -> Answer:
        started = time.perf_counter()
        route = router.route(question, deterministic=self.settings.det_router)
        METRICS.inc(f"route.{route.kind}")

        evidence: list[Chunk] = []
        if not route.is_deterministic:
            evidence = self.retrieve(question, k).chunks

        answer = self.reasoner.run(
            question,
            route=route,
            evidence=evidence,
            grounding_floor=self.settings.grounding_floor,
        )
        if not route.is_deterministic:
            answer.citations = [c.chunk_id for c in evidence]
        answer.latency_ms = (time.perf_counter() - started) * 1000.0
        LATENCY.observe("query", answer.latency_ms)
        log_event(
            "query.done",
            route=answer.route,
            steps=answer.steps,
            support=round(answer.grounding.support_rate, 3),
            latency_ms=round(answer.latency_ms, 2),
        )
        return answer

    def query_payload(self, question: str, k: int | None = None) -> dict:
        answer = self.query(question, k)
        return {
            "question": answer.question,
            "answer": answer.text,
            "citations": answer.citations,
            "route": answer.route,
            "steps": answer.steps,
            "trace": answer.trace,
            "grounding": {
                "support_rate": round(answer.grounding.support_rate, 6),
                "checked": answer.grounding.checked,
                "supported": answer.grounding.supported,
                "unsupported": answer.grounding.unsupported,
            },
            "latency_ms": round(answer.latency_ms, 3),
        }

    def stream(self, question: str, k: int | None = None) -> Iterator[str]:
        """Chunked text stream suitable for server-sent events."""
        text = self.query(question, k).text
        step = 24
        for i in range(0, len(text), step):
            yield text[i: i + step]

    # -- adapters ---------------------------------------------------------
    def as_runner(self):
        """Shaped for :func:`atlas.engines.evaluator.evaluate`."""

        def runner(question: str) -> dict:
            started = time.perf_counter()
            route = router.route(question, deterministic=self.settings.det_router)
            if route.is_deterministic:
                # Do not read self.last_retrieval here: it still holds whatever
                # the previous case retrieved, and reporting stale hits makes a
                # tool-routed case look like a retrieval success.
                hits = []
                evidence: list[Chunk] = []
            else:
                result = self.retrieve(question, self.settings.top_k)
                hits = list(result.hits)
                evidence = list(result.chunks)
            answer = self.reasoner.run(
                question,
                route=route,
                evidence=evidence,
                grounding_floor=self.settings.grounding_floor,
            )
            return {
                "retrieved": [h.chunk_id for h in hits],
                "route": answer.route,
                "grounded": answer.grounding.support_rate,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "answer": answer.text,
            }

        return runner

    # -- internals --------------------------------------------------------
    def _tool_search(self, query: str, k: int) -> list[Chunk]:
        return self.retrieve(query, k).chunks

    def _text(self, chunk_id: str) -> str:
        chunk = self.store.get_chunk(chunk_id)
        return chunk.text if chunk else ""
