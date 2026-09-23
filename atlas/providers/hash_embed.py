"""Dependency-free embedding provider: signed hashing of CJK-aware features.

Hashing trick with a **stable** digest. Python's built-in ``hash()`` is salted
per process (``PYTHONHASHSEED``), which would make embeddings - and therefore
every retrieval result - differ across runs. We use BLAKE2b instead, so the
whole system is bit-reproducible across processes and machines.

Sub-linear term weighting (``1 + log tf``) plus signed buckets keeps the
collision bias symmetric, which matters far more than raw accuracy for a
fallback encoder.

Author: 晨星
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

from ..engines.text import tokenize

_DIGEST_BYTES = 8


def stable_hash(token: str) -> int:
    """Process-independent 63-bit hash of ``token``."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") >> 1


class HashEmbeddingProvider:
    """Signed random-projection embeddings over bag-of-features.

    Parameters
    ----------
    dim:
        Output dimensionality. 256 is a good CPU-memory compromise.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hash-{self._dim}"

    # -- API --------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Symmetric by design: same encoder for queries and documents.
        return self._encode(text)

    # -- internals --------------------------------------------------------
    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        counts = Counter(tokenize(text))
        for token, tf in counts.items():
            h = stable_hash(token)
            index = h % self._dim
            sign = 1.0 if (h >> 20) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm < 1e-12:
            return vector
        return [v / norm for v in vector]
