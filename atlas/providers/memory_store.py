"""In-memory document store with optional JSONL persistence.

Author: 晨星
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ..contracts import Chunk


class InMemoryStore:
    """Dict-backed chunk + document registry."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._documents: dict[str, str] = {}

    # -- chunks -----------------------------------------------------------
    def put_chunks(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in sorted(self._chunks)]

    def count_chunks(self) -> int:
        return len(self._chunks)

    # -- documents --------------------------------------------------------
    def put_document(self, doc_id: str, text: str) -> None:
        self._documents[doc_id] = text

    def documents(self) -> list[str]:
        return sorted(self._documents)

    def get_document(self, doc_id: str) -> str | None:
        return self._documents.get(doc_id)

    def drop_document(self, doc_id: str) -> int:
        removed = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in removed:
            del self._chunks[cid]
        self._documents.pop(doc_id, None)
        return len(removed)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.all_chunks():
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        return len(self._chunks)

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryStore":
        store = cls()
        source = Path(path)
        if not source.exists():
            return store
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                store.put_chunks([Chunk(**payload)])
        return store
