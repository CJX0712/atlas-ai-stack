"""P0 gate: reject emoji and pictographic characters anywhere in the repository.

Emoji must never be used as functional UI icons. The check uses **code-point
ranges only** - no literal symbol appears in this file. That is deliberate: a
regex character class containing literal BMP symbols is corrupted when the file
is written through a GBK code page, and the failure surfaces as
``bad character range`` at import time rather than as a finding.

Typographic characters (dashes, curly quotes, CJK punctuation, arrows) are
explicitly permitted. The gate targets pictographs, dingbats and emoji
presentation machinery, not legitimate typesetting.

Usage::

    python tools/scan_emoji.py [root ...]

Exit codes: 0 = clean, 1 = findings, 2 = usage error.

Author: 晨星
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inclusive code-point ranges that must not appear in the repository.
BLOCKED_RANGES: tuple[tuple[int, int], ...] = (
    (0x2600, 0x26FF),    # miscellaneous symbols (sun, cloud, lightning, ...)
    (0x2700, 0x27BF),    # dingbats
    (0x2B00, 0x2BFF),    # miscellaneous symbols and arrows (star, ...)
    (0xFE00, 0xFE0F),    # variation selectors (emoji presentation)
    (0x1F000, 0x1F02F),  # mahjong / dominoes
    (0x1F0A0, 0x1F0FF),
    (0x1F100, 0x1F1FF),
    (0x1F200, 0x1F2FF),  # enclosed ideographic supplement
    (0x1F300, 0x1F5FF),  # misc symbols and pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport and map symbols
    (0x1F900, 0x1F9FF),  # supplemental symbols and pictographs
    (0x1FA00, 0x1FAFF),  # symbols and pictographs extended-A
    (0x200D, 0x200D),    # zero-width joiner (emoji sequence glue)
    (0x20E3, 0x20E3),    # combining enclosing keycap
    (0xE0020, 0xE007F),  # tag characters
)

SCAN_SUFFIXES = {
    ".py", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml",
    ".html", ".htm", ".css", ".js", ".ts", ".tsx", ".jsx", ".cfg", ".ini",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".atlas", ".mypy",
}


def blocked(ch: str) -> bool:
    """True when ``ch`` is a pictograph that must not appear in the repository."""
    code = ord(ch)
    return any(low <= code <= high for low, high in BLOCKED_RANGES)


def scan_file(path: Path) -> list[tuple[int, int, str]]:
    findings: list[tuple[int, int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            if blocked(ch):
                findings.append((line_no, col, f"U+{ord(ch):04X}"))
    return findings


def iter_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            out.append(path)
    return out


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path(".")]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        print(f"usage error: path not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    files = iter_files(roots)
    total = 0
    for path in files:
        findings = scan_file(path)
        if not findings:
            continue
        total += len(findings)
        for line_no, col, code in findings[:20]:
            print(f"{path}:{line_no}:{col}: blocked character {code}")

    print(f"scanned {len(files)} files, {total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
