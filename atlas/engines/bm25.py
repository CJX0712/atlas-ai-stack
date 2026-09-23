"""BM25 sparse retrieval with a strictly non-negative Robertson IDF.

Why not ``rank_bm25``?
----------------------
``rank_bm25.BM25Okapi`` computes ``idf = ln((N - n + 0.5) / (n + 0.5))`` and
then applies a floor of ``epsilon * average_idf``.  On small corpora that
expression goes **negative** (a term appearing in more than half the documents
scores below zero) and the floor rule can invert the ranking.  This module uses
the Robertson/Sparck-Jones form::

    idf = ln(1 + (N - n + 0.5) / (n + 0.5))

which is strictly positive for every ``0 <= n <= N``, so scores are monotone in
term frequency by construction.  Property-tested in ``tests/test_bm25.py``.

Author: 晨星
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..contracts import Scored
from .text import tokenize

K1 = 1.5
B = 0.75


def robertson_idf(n_docs: int, doc_freq: int) -> float:
    """Strictly positive IDF. ``doc_freq`` is clamped into ``[0, n_docs]``."""
    if n_docs <= 0:
        return 0.0
    df = max(0, min(int(doc_freq), n_docs))
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass
class SparseIndex:
    """In-memory BM25 index over ``(chunk_id, text)`` pairs."""

    k1: float = K1
    b: float = B
    _docs: dict[str, Counter[str]] = field(default_factory=dict)
    _length: dict[str, int] = field(default_factory=dict)
    _df: Counter[str] = field(default_factory=Counter)
    _order: list[str] = field(default_factory=list)

    # -- mutation ---------------------------------------------------------
    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        if chunk_id in self._docs:
            self._remove(chunk_id)
        self._docs[chunk_id] = counts
        self._length[chunk_id] = max(1, len(tokens))
        self._order.append(chunk_id)
        for term in counts:
            self._df[term] += 1

    def _remove(self, chunk_id: str) -> None:
        old = self._docs.pop(chunk_id, None)
        self._length.pop(chunk_id, None)
        if chunk_id in self._order:
            self._order.remove(chunk_id)
        if old:
            for term in old:
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]

    def extend(self, pairs: list[tuple[str, str]]) -> None:
        for chunk_id, text in pairs:
            self.add(chunk_id, text)

    def reset(self) -> None:
        """Drop every document, keeping the configured ``k1`` / ``b``."""
        self._docs.clear()
        self._length.clear()
        self._df.clear()
        self._order.clear()

    # -- query ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def average_length(self) -> float:
        if not self._docs:
            return 1.0
        return sum(self._length.values()) / len(self._docs)

    def idf(self, term: str) -> float:
        return robertson_idf(len(self._docs), self._df.get(term, 0))

    def score(self, query: str, chunk_id: str) -> float:
        counts = self._docs.get(chunk_id)
        if not counts:
            return 0.0
        dl = self._length.get(chunk_id, 1)
        avgdl = self.average_length or 1.0
        total = 0.0
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        for term in tokenize(query):
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            total += self.idf(term) * (tf * (self.k1 + 1.0)) / (tf + denom_norm)
        return total

    def search(self, query: str, k: int = 10) -> list[Scored]:
        scored: list[Scored] = []
        for chunk_id in self._order:
            value = self.score(query, chunk_id)
            if value > 0.0:
                scored.append(Scored(chunk_id, value, source="sparse"))
        scored.sort(key=lambda s: (-s.score, s.chunk_id))
        return scored[: max(0, k)]

    def terms(self) -> list[str]:
        return sorted(self._df)
