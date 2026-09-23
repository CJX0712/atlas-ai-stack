"""Cross-encoder reranker via ``fastembed.TextCrossEncoder`` (optional).

Default model: ``Xenova/bge-reranker-base`` (int8 ONNX, ~267 MB).

Important operational note, mirrored from the reranking study that this
repository's guard was built from: a cross-encoder trained on English queries
produces near-arbitrary orderings for Chinese queries. Always run it through
:func:`atlas.engines.rerank_guard.guarded_rerank` rather than letting it dictate
the final order.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "Xenova/bge-reranker-base"


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a transformer cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache_dir: str | None = None) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "fastembed cross-encoder support is unavailable. "
                "Install `fastembed` or set ATLAS_RERANK_PROVIDER=lexical."
            ) from exc

        self._model_name = model_name
        kwargs: dict[str, object] = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self._model = TextCrossEncoder(model_name, **kwargs)

    @property
    def name(self) -> str:
        return f"cross:{self._model_name}"

    def rerank(
        self, query: str, candidates: Sequence[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        scores = [float(s) for s in self._model.rerank(query, list(candidates))]
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[: max(0, top_k)]
