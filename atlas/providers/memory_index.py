"""In-memory cosine vector index - the zero-dependency default backend.

Uses NumPy when available for a single GEMV per query, and falls back to a pure
Python dot product otherwise. Both paths are covered by the test-suite.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from ..contracts import Scored

try:  # optional accelerator
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


class MemoryVectorIndex:
    """Brute-force exact cosine search. Deterministic tie-break by chunk id."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._matrix = None
        self._dirty = True

    @property
    def backend(self) -> str:
        return "numpy" if _np is not None else "python"

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have equal length")
        existing = {cid: pos for pos, cid in enumerate(self._ids)}
        for cid, vector in zip(ids, vectors):
            row = [float(v) for v in vector]
            if cid in existing:
                self._vectors[existing[cid]] = row
                continue
            existing[cid] = len(self._ids)
            self._ids.append(cid)
            self._vectors.append(row)
        self._dirty = True

    def search(self, vector: Sequence[float], k: int) -> list[Scored]:
        if not self._ids or k <= 0:
            return []
        query = [float(v) for v in vector]
        if _np is not None:
            self._refresh()
            scores = self._matrix @ _np.asarray(query, dtype="float64")
            order = sorted(range(len(self._ids)), key=lambda i: (-float(scores[i]), self._ids[i]))
        else:
            scores = [self._dot(query, row) for row in self._vectors]
            order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        out: list[Scored] = []
        for idx in order[:k]:
            value = float(scores[idx])
            if value <= 0.0:
                continue
            out.append(Scored(self._ids[idx], value, source="dense"))
        return out

    def __len__(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids.clear()
        self._vectors.clear()
        self._matrix = None
        self._dirty = True

    def ids(self) -> list[str]:
        return list(self._ids)

    def scores_against(self, vector: Sequence[float]) -> dict[str, float]:
        query = [float(v) for v in vector]
        return {cid: self._dot(query, row) for cid, row in zip(self._ids, self._vectors)}

    def _refresh(self) -> None:
        if not self._dirty or _np is None:
            return
        self._matrix = _np.asarray(self._vectors, dtype="float64")
        self._dirty = False

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(y * y for y in b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)
