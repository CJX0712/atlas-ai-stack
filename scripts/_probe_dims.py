"""Development probe: embedding-dimension sensitivity and per-case diagnostics.

Not part of the shipped test chain - kept in-tree so the reported recall numbers
can be re-derived rather than trusted.

Author: 晨星
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas.pipeline.container import AtlasContainer  # noqa: E402
from atlas.providers.hash_embed import HashEmbeddingProvider  # noqa: E402
from atlas.settings import Settings  # noqa: E402


def main() -> int:
    report: dict[str, object] = {"dims": {}, "variants": {}}

    for dim in (128, 256, 512, 1024):
        container = AtlasContainer(Settings.from_env())
        container.embedder = HashEmbeddingProvider(dim=dim)
        container.indexing.embedder = container.embedder
        container.querying.embedder = container.embedder
        container.index.reset()
        container.load_demo_corpus()
        report["dims"][str(dim)] = container.evaluate().as_dict()

    variants = {
        "baseline": {},
        "no_expansion": {"query_expansion": False},
        "no_rerank": {"rerank_provider": "none"},
        "guard_off": {"guard_enabled": False},
        "k3": {"top_k": 3},
        "k6": {"top_k": 6},
        "k10": {"top_k": 10},
    }
    for label, overrides in variants.items():
        container = AtlasContainer(Settings.from_env(**overrides))
        container.load_demo_corpus()
        report["variants"][label] = container.evaluate().as_dict()

    container = AtlasContainer(Settings.from_env())
    container.load_demo_corpus()
    evaluation = container.evaluate()
    report["per_case"] = [
        {
            "question": case.question[:40],
            "recall": round(case.recall, 3),
            "rr": round(case.reciprocal_rank, 3),
            "ndcg": round(case.ndcg, 3),
            "route": case.route,
            "retrieved": len(case.retrieved),
        }
        for case in evaluation.cases
    ]

    target = ROOT / "_probe_dims.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
