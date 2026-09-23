"""L5 command line interface.

Implemented on :mod:`argparse` rather than Typer on purpose: the CLI is the
entry point a first-time user runs, so it must work in an interpreter that has
**no** third-party packages installed. ``atlas doctor`` in particular is
designed to be the first command after ``git clone``.

Author: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .pipeline.container import AtlasContainer
from .settings import Settings
from .telemetry import configure

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _print(payload: object) -> None:
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _read_paths(paths: list[str]) -> dict[str, str]:
    docs: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".md", ".txt", ".rst"}:
                    docs[child.stem] = child.read_text(encoding="utf-8")
        elif path.is_file():
            docs[path.stem] = path.read_text(encoding="utf-8")
        else:
            print(f"skip missing path: {raw}", file=sys.stderr)
    return docs


def _container(settings: Settings | None = None) -> AtlasContainer:
    return AtlasContainer(settings or Settings.from_env())


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    container = _container()
    if args.demo or not args.paths:
        report = container.load_demo_corpus()
        source = "packaged demo corpus"
    else:
        report = container.ingest(_read_paths(args.paths))
        source = ", ".join(args.paths)
    _print({"source": source, **report.as_dict()})
    return EXIT_OK


def cmd_ask(args: argparse.Namespace) -> int:
    container = _container()
    container.load_demo_corpus()
    _print(container.ask(args.question, args.k))
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    container = _container()
    container.load_demo_corpus()
    report = container.evaluate(args.k)
    _print(report.as_dict(include_cases=args.cases))
    if args.min_recall is not None and report.recall_at_k < args.min_recall:
        print(
            f"recall@k {report.recall_at_k:.4f} below threshold {args.min_recall:.4f}",
            file=sys.stderr,
        )
        return EXIT_FAIL
    return EXIT_OK


def cmd_selfcheck(args: argparse.Namespace) -> int:
    container = _container()
    container.load_demo_corpus()
    report = container.selfcheck()
    _print(report)
    return EXIT_OK if report["ok"] else EXIT_FAIL


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which optional capabilities are available in this interpreter."""
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "version": __version__,
    }
    for module, purpose in (
        ("numpy", "fast vector math for the memory index"),
        ("faiss", "exact FAISS inner-product index"),
        ("fastembed", "ONNX embeddings and cross-encoder reranking"),
        ("llama_cpp", "local GGUF inference"),
        ("httpx", "Ollama transport"),
        ("fastapi", "HTTP service"),
        ("uvicorn", "ASGI server"),
        ("pytest", "test suite"),
    ):
        try:
            __import__(module)
            checks[module] = {"available": True, "purpose": purpose}
        except Exception as exc:  # noqa: BLE001 - report any import failure
            checks[module] = {"available": False, "purpose": purpose, "error": str(exc)[:120]}

    try:
        container = _container()
        checks["providers"] = container.build_info().as_dict()
        checks["offline_mode"] = {
            "llm": getattr(container.llm, "name", "unknown"),
            "note": (
                "mock LLM selected: the full pipeline runs with no network and "
                "no API key"
                if getattr(container.llm, "name", "") == "mock"
                else "a real model backend is configured"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        checks["providers"] = {"error": str(exc)[:200]}

    checks["ok"] = bool(checks.get("providers")) and "error" not in checks["providers"]
    _print(checks)
    return EXIT_OK if checks["ok"] else EXIT_FAIL


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        return EXIT_FAIL

    settings = Settings.from_env(host=args.host, port=args.port)
    configure(settings.log_level)
    uvicorn.run(
        "atlas.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the packaged verification chain (emoji gate, unit tests, E2E)."""
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "verify.py"
    if not runner.exists():
        print(
            "scripts/verify.py not found. This command is available in a source "
            "checkout, not in an installed wheel.",
            file=sys.stderr,
        )
        return EXIT_FAIL

    import subprocess

    command = [sys.executable, str(runner)]
    if args.skip_e2e:
        command.append("--skip-e2e")
    if args.json:
        command.append("--json")
    return subprocess.call(command, cwd=str(root))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description=(
            "ATLAS - hybrid retrieval, guarded reranking, deterministic routing "
            "and grounding verification."
        ),
    )
    parser.add_argument("--version", action="version", version=f"atlas {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index documents or the packaged corpus")
    p_ingest.add_argument("paths", nargs="*", help="files or directories to ingest")
    p_ingest.add_argument("--demo", action="store_true", help="use the packaged corpus")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the offline evaluation set")
    p_eval.add_argument("--k", type=int, default=None)
    p_eval.add_argument("--cases", action="store_true", help="include per-case detail")
    p_eval.add_argument("--min-recall", type=float, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_check = sub.add_parser("selfcheck", help="assert runtime invariants")
    p_check.set_defaults(func=cmd_selfcheck)

    p_doctor = sub.add_parser("doctor", help="report capability availability")
    p_doctor.set_defaults(func=cmd_doctor)

    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8077)
    p_serve.set_defaults(func=cmd_serve)

    p_verify = sub.add_parser("verify", help="run the full verification chain")
    p_verify.add_argument("--skip-e2e", action="store_true")
    p_verify.add_argument("--json", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
