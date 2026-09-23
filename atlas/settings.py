"""Runtime configuration - plain dataclass over environment variables.

Deliberately dependency-free (no pydantic-settings) so that ``import atlas``
never requires a third-party package. Every knob has a safe offline default.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

ENV_PREFIX = "ATLAS_"


def _get(name: str, default: Any) -> Any:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return default
    return raw


@dataclass
class Settings:
    """All ATLAS tunables. Override with ``ATLAS_<UPPER_FIELD>`` env vars."""

    # --- providers -------------------------------------------------------
    llm_provider: str = "mock"          # mock | llama_cpp | ollama
    embed_provider: str = "hash"        # hash | onnx
    rerank_provider: str = "lexical"    # lexical | cross | none
    index_backend: str = "memory"       # memory | faiss

    # --- CPU inference (memory-bandwidth bound, NOT compute bound) -------
    # llama.cpp defaults to cpu_count-1 threads which is ~4x SLOWER for small
    # quantised models. Lock to a small fixed number unless overridden.
    cpu_threads: int = 4
    llama_model_path: str = ""
    llama_context: int = 4096

    # --- ollama ----------------------------------------------------------
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:0.5b"

    # --- retrieval -------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 6
    rrf_k: int = 60
    query_expansion: bool = True
    guard_enabled: bool = True
    guard_alpha: float = 0.5
    guard_spearman_floor: float = 0.2
    grounding_floor: float = 0.18

    # --- reasoning -------------------------------------------------------
    max_steps: int = 6
    det_router: bool = True

    # --- serving ---------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8077
    log_level: str = "INFO"

    # --- persistence -----------------------------------------------------
    data_dir: str = "data"
    workspace_dir: str = ".atlas"

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "extra":
                continue
            kwargs[f.name] = _get(f.name.upper(), f.default)
        kwargs.update(overrides)
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "extra"
        }
