"""Provider factory - the only place that knows about concrete backends.

Reads :class:`atlas.settings.Settings`, tries the requested backend, and
**degrades loudly** to the zero-dependency default when an optional package is
missing. A missing optional dependency therefore never breaks the system; it
just changes the quality of one component, and the choice is reported by
``GET /v1/stats``.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

from ..settings import Settings
from ..telemetry import log_event
from .hash_embed import HashEmbeddingProvider
from .lexical_rerank import LexicalReranker
from .memory_index import MemoryVectorIndex
from .memory_store import InMemoryStore
from .mock_llm import MockLLM


def _fallback(kind: str, requested: str, error: Exception) -> None:
    log_event("provider.fallback", kind=kind, requested=requested, error=str(error)[:200])


def build_embedder(settings: Settings):
    if settings.embed_provider == "onnx":
        try:
            from .onnx_embed import OnnxEmbeddingProvider

            return OnnxEmbeddingProvider(threads=settings.cpu_threads)
        except Exception as exc:  # pragma: no cover - depends on environment
            _fallback("embedder", "onnx", exc)
    # 512 dimensions measured best on the packaged bilingual corpus:
    # recall@6 0.755 vs 0.739 at 256 and 0.739 at 1024 (see scripts/_probe_dims.py).
    return HashEmbeddingProvider(dim=512)


def build_reranker(settings: Settings):
    if settings.rerank_provider in {"none", ""}:
        return None
    if settings.rerank_provider == "cross":
        try:
            from .cross_rerank import CrossEncoderReranker

            return CrossEncoderReranker()
        except Exception as exc:  # pragma: no cover - depends on environment
            _fallback("reranker", "cross", exc)
    return LexicalReranker()


def build_llm(settings: Settings):
    if settings.llm_provider == "llama_cpp":
        try:
            from .llama_cpp_llm import LlamaCppLLM

            return LlamaCppLLM(
                settings.llama_model_path,
                n_ctx=settings.llama_context,
                n_threads=settings.cpu_threads,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            _fallback("llm", "llama_cpp", exc)
    elif settings.llm_provider == "ollama":
        try:
            from .ollama_llm import OllamaLLM

            return OllamaLLM(settings.ollama_host, settings.ollama_model)
        except Exception as exc:  # pragma: no cover - depends on environment
            _fallback("llm", "ollama", exc)
    return MockLLM()


def build_index(settings: Settings, dim: int):
    if settings.index_backend == "faiss":
        try:
            from .faiss_index import FaissVectorIndex

            return FaissVectorIndex(dim)
        except Exception as exc:  # pragma: no cover - depends on environment
            _fallback("index", "faiss", exc)
    return MemoryVectorIndex()


def build_store(settings: Settings):
    del settings
    return InMemoryStore()


def describe(settings: Settings, resolved: dict[str, Any]) -> dict[str, Any]:
    """Provider identity card exposed by ``/v1/stats``."""
    return {
        "requested": {
            "llm": settings.llm_provider,
            "embed": settings.embed_provider,
            "rerank": settings.rerank_provider,
            "index": settings.index_backend,
        },
        "resolved": resolved,
        "cpu_threads": settings.cpu_threads,
    }
