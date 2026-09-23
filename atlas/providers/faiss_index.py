"""FAISS dense index (optional production backend).

Uses ``IndexFlatIP`` over L2-normalised vectors, which is exact cosine search -
no approximation, so results are bit-comparable with the in-memory backend and
the test-suite can cross-validate the two.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts import Scored


class FaissVectorIndex:
    """Exact inner-product index over normalised vectors."""

    def __init__(self, dim: int) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "faiss-cpu is not installed. Install it or set ATLAS_INDEX_BACKEND=memory."
            ) from exc
        import faiss
        import numpy as np

        self._np = np
        self._faiss = faiss
        self._dim = int(dim)
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids: list[str] = []

    @property
    def backend(self) -> str:
        return "faiss"

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        matrix = self._np.asarray(vectors, dtype="float32")
        if matrix.shape[1] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {matrix.shape[1]}")
        self._faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._ids.extend(str(cid) for cid in ids)

    def search(self, vector: Sequence[float], k: int) -> list[Scored]:
        if len(self._index) == 0 or k <= 0:
            return []
        query = self._np.asarray([list(vector)], dtype="float32")
        self._faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(k, len(self._ids)))
        out: list[Scored] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            out.append(Scored(self._ids[int(idx)], float(score), source="dense"))
        return out

    def __len__(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._index = self._faiss.IndexFlatIP(self._dim)
        self._ids.clear()
