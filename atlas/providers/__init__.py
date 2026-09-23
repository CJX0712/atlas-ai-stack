"""L1 provider implementations of the L0 ports.

Every capability has a **zero-dependency default** (``hash`` embeddings,
``mock`` LLM, ``lexical`` reranker, in-memory index/store) and an optional
production backend (ONNX embeddings, llama.cpp / Ollama LLMs, FAISS index).
The :mod:`atlas.providers.registry` module wires them from
:class:`atlas.settings.Settings`, falling back gracefully when an optional
dependency is missing.
"""

from __future__ import annotations

__all__ = [
    "hash_embed",
    "mock_llm",
    "lexical_rerank",
    "memory_index",
    "memory_store",
    "onnx_embed",
    "cross_rerank",
    "llama_cpp_llm",
    "ollama_llm",
    "faiss_index",
    "registry",
]
