"""Ingest pipeline: raw documents -> chunks -> embeddings -> dense + sparse index.

Rebuild semantics
-----------------
Chunk ids are ``"{doc_id}#{ordinal}"``. Re-ingesting a document that now has
*fewer* chunks would leave orphans behind under an incremental update, so both
indexes are rebuilt from the document store on every ingest. The store is the
single source of truth and the three structures stay in exact sync.

Author: 晨星
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..contracts import Chunk
from ..engines import chunking
from ..telemetry import log_event


@dataclass(slots=True)
class IngestReport:
    documents: int = 0
    chunks: int = 0
    dim: int = 0
    elapsed_ms: float = 0.0
    doc_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "dim": self.dim,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "doc_ids": self.doc_ids,
        }


class IndexingPipeline:
    """Turns a mapping of ``doc_id -> text`` into a searchable corpus."""

    def __init__(self, store, embedder, index, sparse, settings) -> None:
        self.store = store
        self.embedder = embedder
        self.index = index
        self.sparse = sparse
        self.settings = settings

    def ingest(self, documents: Mapping[str, str]) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        for raw_id, text in documents.items():
            doc_id = str(raw_id)
            body = "" if text is None else str(text)
            if not body.strip():
                continue
            # Drop the previous revision first. Without this, re-ingesting a
            # document that now produces fewer chunks leaves the surplus chunks
            # of the old revision in the store, and they stay retrievable
            # forever - a silent corruption that only shows up as "the index
            # returns content I deleted".
            self.store.drop_document(doc_id)
            self.store.put_document(doc_id, body)
            self.store.put_chunks(self._chunk_document(doc_id, body))

        self._rebuild()
        report.documents = len(self.store.documents())
        report.chunks = self.store.count_chunks()
        report.dim = self.embedder.dim
        report.doc_ids = self.store.documents()
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            "ingest.done",
            documents=report.documents,
            chunks=report.chunks,
            elapsed_ms=round(report.elapsed_ms, 2),
        )
        return report

    def ingest_texts(self, pairs: Sequence[tuple[str, str]]) -> IngestReport:
        return self.ingest({doc_id: text for doc_id, text in pairs})

    def rebuild(self) -> IngestReport:
        """Re-index everything currently in the store."""
        started = time.perf_counter()
        self._rebuild()
        return IngestReport(
            documents=len(self.store.documents()),
            chunks=self.store.count_chunks(),
            dim=self.embedder.dim,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            doc_ids=self.store.documents(),
        )

    # -- internals --------------------------------------------------------
    def _chunk_document(self, doc_id: str, body: str) -> list[Chunk]:
        raw = chunking.split(
            body,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        return [
            Chunk(
                chunk_id=f"{doc_id}#{item.ordinal}",
                doc_id=doc_id,
                text=item.text,
                ordinal=item.ordinal,
                heading=item.heading,
            )
            for item in raw
        ]

    def _rebuild(self) -> None:
        chunks = self.store.all_chunks()
        self.sparse.reset()
        self.index.reset()
        if not chunks:
            return
        for chunk in chunks:
            self.sparse.add(chunk.chunk_id, chunk.text)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.add([c.chunk_id for c in chunks], vectors)
