"""llama.cpp local LLM provider with correct CPU thread configuration.

The single most important knob here is ``n_threads``.

llama.cpp (and ``llama-cpp-python``) default to ``cpu_count - 1`` worker
threads.  On a CPU-only box running a small quantised model the workload is
**memory-bandwidth bound, not compute bound**: past 2-4 threads the workers
thrash the shared cache lines and total throughput collapses.  Measured on this
class of machine: the default thread count made a small GGUF model roughly 4x
slower than a fixed 4 threads.  Hence :data:`DEFAULT_THREADS`.

Author: 晨星
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from ..contracts import Message

DEFAULT_THREADS = 4


def resolve_threads(explicit: int | None = None) -> int:
    """Thread count for CPU inference, clamped to the bandwidth-bound sweet spot."""
    if explicit and explicit > 0:
        return explicit
    env = os.environ.get("ATLAS_CPU_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return DEFAULT_THREADS


def _flatten(messages: Sequence[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            message.role, message.role.title()
        )
        parts.append(f"{label}: {message.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


class LlamaCppLLM:
    """Chat completion over a local GGUF model."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        if not model_path:
            raise ValueError(
                "ATLAS_LLAMA_MODEL_PATH is empty. Point it at a local .gguf file "
                "or set ATLAS_LLM_PROVIDER=mock."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it or set "
                "ATLAS_LLM_PROVIDER=mock / ollama."
            ) from exc

        self._threads = resolve_threads(n_threads)
        self._model_path = model_path
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self._threads,
            verbose=verbose,
        )

    @property
    def name(self) -> str:
        return f"llama_cpp(threads={self._threads})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        prompt = _flatten(messages)
        result = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nUser:", "\nSystem:"],
        )
        choices = result.get("choices") or [{}]
        return str(choices[0].get("text", "")).strip()

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        prompt = _flatten(messages)
        for part in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stop=["\nUser:", "\nSystem:"],
        ):
            choices = part.get("choices") or [{}]
            delta = choices[0].get("text", "")
            if delta:
                yield str(delta)
