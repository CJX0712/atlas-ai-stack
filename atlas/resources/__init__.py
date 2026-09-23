"""Packaged resources: the demo corpus and the offline evaluation set.

Keeping them inside the package means ``atlas eval`` and ``atlas ingest`` work
from any working directory, including after ``pip install``.

Author: 晨星
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = RESOURCE_DIR / "corpus"
CASES_FILE = RESOURCE_DIR / "eval_cases.json"


def corpus_files() -> list[Path]:
    if not CORPUS_DIR.exists():
        return []
    return sorted(p for p in CORPUS_DIR.glob("*.md") if p.is_file())


def load_corpus() -> dict[str, str]:
    """Return ``{doc_id: text}`` for every packaged corpus file."""
    return {p.stem: p.read_text(encoding="utf-8") for p in corpus_files()}


def load_cases() -> list[dict[str, Any]]:
    if not CASES_FILE.exists():
        return []
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


__all__ = ["RESOURCE_DIR", "CORPUS_DIR", "CASES_FILE", "corpus_files",
           "load_corpus", "load_cases"]
