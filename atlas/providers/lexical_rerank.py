"""Dependency-free reranker: IDF-weighted lexical overlap.

This is a genuine, independent ranking signal (not a no-op), so the guard in
:mod:`atlas.engines.rerank_guard` can be exercised end-to-end without a
transformer.  It is also useful in production as a cheap second-stage filter.

Author: 晨星
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log

from ..engines.bm25 import robertson_idf
from ..engines.text import tokenize


class LexicalReranker:
    """Scores candidates by IDF-weighted coverage of the query tokens."""

    def __init__(self, purpose: str = "lexical") -> None:
        self._name = purpose

    @property
    def name(self) -> str:
        return self._name

    def rerank(
        self, query: str, candidates: Sequence[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return [(i, 0.0) for i in range(len(candidates))]

        doc_tokens = [tokenize(c) for c in candidates]
        n_docs = len(doc_tokens)
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] += 1

        unique_query = list(dict.fromkeys(query_tokens))
        scored: list[tuple[int, float]] = []
        for idx, tokens in enumerate(doc_tokens):
            counts = Counter(tokens)
            length = max(1, len(tokens))
            total = 0.0
            for term in unique_query:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                total += robertson_idf(n_docs, df[term]) * (1.0 + log(tf))
            scored.append((idx, total / (length ** 0.5)))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(0, top_k)]
