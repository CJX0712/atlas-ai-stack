"""ONNX embedding provider backed by ``fastembed`` (Qdrant).

Verified model choices for win_amd64 / CPU:

* ``Xenova/bge-small-zh-v1.5`` - 512-dim Chinese-friendly encoder, ships an
  ``onnx/model_quantized.onnx`` (int8, ~24 MB).
* ``BAAI/bge-small-en-v1.5`` - English counterpart.

Note: the ModelScope mirrors (``AI-ModelScope/bge-*``) only publish
``safetensors``, so they cannot be used by fastembed. Use the ``Xenova/*``
mirrors, which carry the ONNX export.

The import is lazy: this module can be imported without ``fastembed``
installed, and only fails when instantiated.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "Xenova/bge-small-zh-v1.5"


class OnnxEmbeddingProvider:
    """Thin, cached wrapper around ``fastembed.TextEmbedding``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "fastembed is not installed. Install it with "
                "`pip install fastembed` or set ATLAS_EMBED_PROVIDER=hash."
            ) from exc

        self._model_name = model_name
        kwargs: dict[str, object] = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if threads:
            kwargs["threads"] = threads
        self._model = TextEmbedding(model_name, **kwargs)
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"onnx:{self._model_name}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            vector = self.embed_query("warmup")
            self._dim = len(vector)
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [list(map(float, v)) for v in self._model.embed(list(texts))]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0] if vectors else []
