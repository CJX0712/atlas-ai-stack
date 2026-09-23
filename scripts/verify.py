"""One command that proves the system works, offline.

Stages, in order - each one is a gate, and the chain stops at the first failure:

1. **P0 scan** - no pictographic characters anywhere in the repository.
2. **Import check** - every package module imports in this interpreter.
3. **Unit and integration tests** - ``pytest``, 150+ assertions on invariants.
4. **End-to-end** - start the real HTTP server and drive it over the wire.
5. **Scorecard** - evaluation floors for retrieval, routing and grounding.
6. **Determinism** - two independent runs must produce identical results.

The whole chain runs with no network, no API key and no database, because the
default providers are offline implementations. Exit code is non-zero if any
stage fails.

Usage::

    python scripts/verify.py [--skip-e2e] [--json] [--report PATH]

Author: 晨星
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "atlas",
    "atlas.contracts",
    "atlas.settings",
    "atlas.telemetry",
    "atlas.engines.text",
    "atlas.engines.chunking",
    "atlas.engines.bm25",
    "atlas.engines.fusion",
    "atlas.engines.rerank_guard",
    "atlas.engines.router",
    "atlas.engines.tools",
    "atlas.engines.reasoner",
    "atlas.engines.grounding",
    "atlas.engines.memory",
    "atlas.engines.evaluator",
    "atlas.engines.expansion",
    "atlas.providers.registry",
    "atlas.providers.hash_embed",
    "atlas.providers.mock_llm",
    "atlas.providers.lexical_rerank",
    "atlas.providers.memory_index",
    "atlas.providers.memory_store",
    "atlas.pipeline.indexing",
    "atlas.pipeline.querying",
    "atlas.pipeline.container",
    "atlas.resources",
    "atlas.cli",
]

SCORE_FLOORS = {
    "route_accuracy": 1.0,
    "doc_hit_rate": 0.95,
    "doc_mrr": 0.85,
    "recall_at_k": 0.80,
    "mrr": 0.85,
    "ndcg_at_k": 0.75,
    "grounding_rate": 0.95,
}


@dataclass
class Stage:
    name: str
    ok: bool
    seconds: float = 0.0
    detail: str = ""


@dataclass
class VerificationReport:
    stages: list[Stage] = field(default_factory=list)
    scorecard: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(stage.ok for stage in self.stages) and not self.missing_floors()

    def missing_floors(self) -> list[str]:
        return [
            f"{key}={self.scorecard.get(key)} < {floor}"
            for key, floor in SCORE_FLOORS.items()
            if float(self.scorecard.get(key, 0.0)) < floor
        ]

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stages": [
                {"name": s.name, "ok": s.ok, "seconds": round(s.seconds, 2), "detail": s.detail}
                for s in self.stages
            ],
            "scorecard": self.scorecard,
            "environment": self.environment,
        }


def _stage(report: VerificationReport, name: str, func) -> bool:
    started = time.perf_counter()
    try:
        ok, detail = func()
    except Exception as exc:  # noqa: BLE001 - a gate must report, never raise
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    report.stages.append(Stage(name, bool(ok), time.perf_counter() - started, detail))
    return bool(ok)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def stage_p0_scan() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "tools/scan_emoji.py", "."],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    summary = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    return result.returncode == 0, summary


def stage_imports() -> tuple[bool, str]:
    failures: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return not failures, (f"{len(MODULES)} modules" if not failures else "; ".join(failures[:3]))


def stage_tests() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--no-header"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    # Prefer the pytest summary line. The literal last line is often a warnings
    # or docs banner, which would make the stage detail useless.
    summary = next(
        (line for line in reversed(lines) if "passed" in line or "failed" in line or "error" in line),
        lines[-1] if lines else (result.stderr or "")[-200:],
    )
    return result.returncode == 0, summary


def stage_e2e() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "scripts/e2e.py"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    tail = [line for line in (result.stdout or "").splitlines() if line.startswith("e2e summary")]
    detail = tail[0] if tail else (result.stdout or result.stderr or "")[-300:]
    return result.returncode == 0, detail


def stage_scorecard(report: VerificationReport) -> tuple[bool, str]:
    from atlas.pipeline.container import AtlasContainer

    container = AtlasContainer()
    container.load_demo_corpus()
    scorecard = container.evaluate().as_dict()
    report.scorecard = scorecard
    missing = report.missing_floors()
    detail = (
        f"recall@{scorecard['recall_at_k']:.4f} mrr={scorecard['mrr']:.4f} "
        f"doc_hit={scorecard['doc_hit_rate']:.4f} grounding={scorecard['grounding_rate']:.4f}"
    )
    return not missing, detail if not missing else "; ".join(missing)


def stage_determinism() -> tuple[bool, str]:
    from atlas.pipeline.container import AtlasContainer

    question = "为什么 CPU 上的线程数不能设置得太大？"
    fingerprints = []
    for _ in range(2):
        container = AtlasContainer()
        container.load_demo_corpus()
        evaluation = container.evaluate()
        answer = container.ask(question)
        fingerprints.append(
            json.dumps(
                {
                    "scorecard": evaluation.deterministic_view(),
                    "answer": answer["answer"],
                    "citations": answer["citations"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return fingerprints[0] == fingerprints[1], (
        "two independent runs produced identical output"
        if fingerprints[0] == fingerprints[1]
        else "output diverged between runs"
    )


def stage_invariants(report: VerificationReport) -> tuple[bool, str]:
    from atlas.pipeline.container import AtlasContainer

    container = AtlasContainer()
    container.load_demo_corpus()
    selfcheck = container.selfcheck()
    guard = selfcheck["guard_probe"]
    detail = (
        f"guarded_top1={guard['guarded_top1']} naive_top1={guard['naive_top1']} "
        f"rho={guard['spearman']}"
    )
    return selfcheck["ok"], detail


def stage_environment(report: VerificationReport) -> tuple[bool, str]:
    from atlas import __version__

    report.environment = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "atlas_version": __version__,
        "platform": sys.platform,
    }
    return True, f"python {report.environment['python']} on {sys.platform}"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-e2e", action="store_true", help="skip the HTTP end-to-end stage")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--report", default="verify_report.json", help="where to write the report")
    args = parser.parse_args(argv)

    report = VerificationReport()
    _stage(report, "environment", lambda: stage_environment(report))
    if not _stage(report, "p0-scan", stage_p0_scan):
        return _finish(report, args, "P0 gate failed: blocked characters present")
    if not _stage(report, "imports", stage_imports):
        return _finish(report, args, "import failure")
    if not _stage(report, "unit-and-integration-tests", stage_tests):
        return _finish(report, args, "test suite failed")
    if not args.skip_e2e and not _stage(report, "end-to-end", stage_e2e):
        return _finish(report, args, "end-to-end failed")
    if not _stage(report, "scorecard", lambda: stage_scorecard(report)):
        return _finish(report, args, "scorecard below floors")
    if not _stage(report, "invariants", lambda: stage_invariants(report)):
        return _finish(report, args, "runtime invariants violated")
    if not _stage(report, "determinism", stage_determinism):
        return _finish(report, args, "non-deterministic output")
    return _finish(report, args, None)


def _finish(report: VerificationReport, args: argparse.Namespace, failure: str | None) -> int:
    payload = report.as_dict()
    target = ROOT / args.report
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print("=" * 66)
        print("ATLAS verification")
        print("=" * 66)
        for stage in report.stages:
            mark = "PASS" if stage.ok else "FAIL"
            print(f"[{mark}] {stage.name:32s} {stage.seconds:6.2f}s  {stage.detail}")
        if report.scorecard:
            print("-" * 66)
            print(
                "scorecard: "
                f"n={report.scorecard.get('n')} "
                f"recall@k={report.scorecard.get('recall_at_k')} "
                f"mrr={report.scorecard.get('mrr')} "
                f"ndcg@k={report.scorecard.get('ndcg_at_k')} "
                f"doc_hit={report.scorecard.get('doc_hit_rate')} "
                f"grounding={report.scorecard.get('grounding_rate')} "
                f"route_acc={report.scorecard.get('route_accuracy')}"
            )
        print("-" * 66)
        print("RESULT:", "ALL GREEN" if not failure else f"FAILED ({failure})")
        print("report written to:", target.name)

    return 0 if not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
