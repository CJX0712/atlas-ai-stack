"""L0 contracts - the only module every other layer may depend on.

Every external capability (LLM, embedding, reranker, vector index, document
store) is declared here as a structural Protocol. Implementations live in
``atlas.providers`` and are injected at runtime, so the whole pipeline can be
exercised with zero-dependency fakes (offline, no API key, no database).

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


# --------------------------------------------------------------------------
# Value objects (frozen -> hashable, safe to put in sets/dicts)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text with provenance."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int = 0
    heading: str = ""
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Scored:
    """A ranked hit produced by any retriever."""

    chunk_id: str
    score: float
    source: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a chat-style prompt."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GroundingReport:
    """Result of checking an answer against its supporting evidence."""

    support_rate: float = 0.0
    unsupported: list[str] = field(default_factory=list)
    checked: int = 0
    supported: int = 0


@dataclass(slots=True)
class Answer:
    """Final system output for one question."""

    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    route: str = "retrieve"
    steps: int = 0
    trace: list[str] = field(default_factory=list)
    grounding: GroundingReport = field(default_factory=GroundingReport)
    latency_ms: float = 0.0


# --------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """Turns text into L2-normalised dense vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LLMProvider(Protocol):
    """Chat completion backend. Implementations must be deterministic when
    ``temperature`` is 0 so that evaluations are reproducible."""

    @property
    def name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterable[str]: ...


class RerankerProvider(Protocol):
    """Scores (query, candidate) pairs. Returns ``(index, score)`` pairs
    ordered by descending score. Higher score means more relevant."""

    @property
    def name(self) -> str: ...

    def rerank(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> list[tuple[int, float]]: ...


class VectorIndex(Protocol):
    """Dense nearest-neighbour index over chunk ids."""

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[Scored]: ...

    def __len__(self) -> int: ...


class DocumentStore(Protocol):
    """Key-value storage for chunks plus a document registry."""

    def put_chunks(self, chunks: Sequence[Chunk]) -> None: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def all_chunks(self) -> list[Chunk]: ...

    def put_document(self, doc_id: str, text: str) -> None: ...

    def documents(self) -> list[str]: ...


class Retriever(Protocol):
    """Anything that can answer a query with ranked chunk ids."""

    def retrieve(self, query: str, k: int) -> list[Scored]: ...
