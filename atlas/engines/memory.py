"""Memory: bounded conversation window with summarisation, plus vector recall.

Two independent responsibilities, kept separate on purpose:

* :class:`ConversationMemory` - a token-budgeted transcript that folds the
  oldest turns into a running summary once the budget is exceeded.
* :class:`VectorMemory` - a small long-term store recalled by embedding
  similarity.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import sqrt

from ..contracts import Message

Summarizer = Callable[[Sequence[str]], str]


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: CJK counts per character, latin per 4 chars."""
    cjk = sum(1 for ch in text if 0x3400 <= ord(ch) <= 0x9FFF)
    latin = max(0, len(text) - cjk)
    return cjk + latin // 4 + 1


@dataclass
class ConversationMemory:
    """Sliding transcript with summarisation of the evicted prefix."""

    budget_tokens: int = 1024
    summarizer: Summarizer | None = None
    turns: list[Message] = field(default_factory=list)
    summary: str = ""
    evicted: int = 0

    def append(self, role: str, content: str) -> None:
        self.turns.append(Message(role, content))
        self._compact()

    def _total(self) -> int:
        return _approx_tokens(self.summary) + sum(
            _approx_tokens(m.content) for m in self.turns
        )

    def _compact(self) -> None:
        while self.turns and self._total() > self.budget_tokens:
            pair = self.turns[:2] if len(self.turns) >= 2 else self.turns[:1]
            self.turns = self.turns[len(pair):]
            self.evicted += len(pair)
            chunk_texts = [m.content for m in pair]
            if self.summarizer is not None:
                self.summary = self.summarizer(chunk_texts)
            else:
                joined = " ".join(chunk_texts)
                keep = max(0, self.budget_tokens // 4)
                self.summary = (self.summary + " " + joined)[-keep:].strip() \
                    if keep else ""

    def context(self) -> list[Message]:
        head = [Message("system", f"Earlier conversation summary: {self.summary}")] \
            if self.summary else []
        return head + list(self.turns)

    def tokens(self) -> int:
        return self._total()


@dataclass
class MemoryItem:
    key: str
    text: str
    vector: list[float]
    meta: dict[str, str] = field(default_factory=dict)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


@dataclass
class VectorMemory:
    """Long-term episodic memory recalled by vector similarity."""

    embedder: object | None = None
    items: list[MemoryItem] = field(default_factory=list)

    def remember(self, key: str, text: str, **meta: str) -> None:
        vector = self._embed(text)
        for idx, item in enumerate(self.items):
            if item.key == key:
                self.items[idx] = MemoryItem(key, text, vector, dict(meta))
                return
        self.items.append(MemoryItem(key, text, vector, dict(meta)))

    def recall(self, query: str, k: int = 3) -> list[tuple[MemoryItem, float]]:
        if not self.items:
            return []
        qv = self._embed(query)
        scored = [(item, cosine(qv, item.vector)) for item in self.items]
        scored.sort(key=lambda pair: (-pair[1], pair[0].key))
        return [(item, score) for item, score in scored[: max(0, k)] if score > 0.0]

    def __len__(self) -> int:
        return len(self.items)

    def _embed(self, text: str) -> list[float]:
        if self.embedder is None:
            import hashlib

            from .text import tokenize  # local import keeps the engine layer pure

            buckets: dict[int, float] = {}
            for token in tokenize(text):
                # BLAKE2b, not hash(): builtin hash() is salted per process and
                # would make recall non-reproducible across runs.
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest, "big") % 256
                buckets[slot] = buckets.get(slot, 0.0) + 1.0
            return [buckets.get(i, 0.0) for i in range(256)]
        return list(self.embedder.embed_query(text))
