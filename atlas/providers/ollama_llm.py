"""Ollama LLM provider.

Critical detail: ``httpx.Client(trust_env=False)``.

httpx honours ``trust_env`` by default, which means requests to
``http://127.0.0.1:11434`` are routed through whatever ``http_proxy`` /
``ALL_PROXY`` the shell exports. On a developer machine running a SOCKS5 tunnel
(ssh -D / frpc) that produces ``WinError 10054`` connection resets against a
daemon that is demonstrably up. Loopback traffic must never see a proxy.

Author: 晨星
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from ..contracts import Message


def _client(host: str, timeout: float = 120.0):
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("httpx is required for the ollama provider") from exc
    return httpx.Client(base_url=host, timeout=timeout, trust_env=False)


class OllamaLLM:
    """Chat completion against a local (or remote) Ollama daemon."""

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen2.5:0.5b") -> None:
        self._host = host.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def _payload(self, messages: Sequence[Message], temperature: float, max_tokens: int) -> dict:
        return {
            "model": self._model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        with _client(self._host) as client:
            response = client.post("/api/chat", json=self._payload(messages, temperature, max_tokens))
            response.raise_for_status()
            data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        payload = self._payload(messages, temperature, max_tokens)
        payload["stream"] = True
        with _client(self._host) as client:
            with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield str(piece)
                    if chunk.get("done"):
                        break

    def health(self) -> bool:
        try:
            with _client(self._host, timeout=5.0) as client:
                return client.get("/api/tags").status_code == 200
        except Exception:
            return False
