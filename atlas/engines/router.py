"""Deterministic pre-routing: send computable questions to tools, never to the
language model.

Rationale
---------
Small quantised models (0.5B class) are unreliable at arithmetic and date
reasoning.  Measured: a 0.5B ReAct agent answered ``12*(3+4)`` with ``72``.
Routing those queries to a deterministic tool *before* the reasoning loop both
fixes the answer and saves an LLM round-trip.

Invariant asserted in ``tests/test_router.py``: every expression that parses as
arithmetic routes to the calculator and the reasoner never invokes the LLM.

Author: 晨星
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARITHMETIC = "arithmetic"
UNIT = "unit"
DATE = "date"
RETRIEVE = "retrieve"

_ALLOWED_CHARS = set("0123456789+-*/%^(). \t")
_OPERATORS = set("+-*/%^")
_MATH_WORDS = ("计算", "算一下", "等于多少", "的结果", "what is", "calculate", "compute")

# Trailing/leading noise that survives word removal. Must NOT contain any
# operator character, or a valid expression would be corrupted.
_TRIM = " \t=?:.。！？，、；;!！\u3000"

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"([A-Za-z]{1,6})"
_UNIT_PATTERN = re.compile(
    rf"{_NUM}\s*{_UNIT}\s*(?:to|->|=>|in|转|换成|换算成|等于|=)\s*{_UNIT}",
    re.IGNORECASE,
)
_DATE_ISO = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_DATE_WORDS = ("相差", "间隔", "多少天", "几天", "距离", "days between", "diff", "相差几天")


@dataclass(frozen=True, slots=True)
class Route:
    kind: str
    tool: str = ""
    payload: str = ""
    reason: str = ""

    @property
    def is_deterministic(self) -> bool:
        return self.kind in {ARITHMETIC, UNIT, DATE}


_DEGREE = chr(176)      # the degree sign, kept out of source literals
_CELSIUS_GLYPH = chr(8451)
_FAHRENHEIT_GLYPH = chr(8457)


def normalise(query: str) -> str:
    """Fold temperature glyphs into ASCII so regexes stay ASCII-only."""
    text = query
    for glyph in (_CELSIUS_GLYPH, _FAHRENHEIT_GLYPH):
        text = text.replace(glyph, "deg")
    text = text.replace(_DEGREE + "C", "degC").replace(_DEGREE + "F", "degF")
    text = text.replace(_DEGREE + "c", "degC").replace(_DEGREE + "f", "degF")
    text = text.replace(_DEGREE, "deg")
    return text


def _strip_math_words(text: str) -> str:
    out = text.lower()
    for word in _MATH_WORDS:
        out = out.replace(word, " ")
    # Strip question/terminal punctuation from both ends. Full-width marks are
    # included because CJK users type "？", not "?", and an unstripped glyph
    # makes an otherwise valid expression fail the allowed-character gate.
    return out.strip().strip(_TRIM).strip()


def _looks_arithmetic(raw: str) -> str | None:
    text = _strip_math_words(raw)
    if not text or len(text) > 200:
        return None
    if not any(ch.isdigit() for ch in text):
        return None
    if not (set(text) & _OPERATORS):
        return None
    if not set(text) <= _ALLOWED_CHARS:
        return None
    # Must not be a bare number, and parentheses must balance.
    if text.count("(") != text.count(")"):
        return None
    return text


def route(query: str, *, deterministic: bool = True) -> Route:
    """Classify a query. Returns ``Route(kind=RETRIEVE)`` when nothing applies."""
    if not deterministic:
        return Route(RETRIEVE, reason="deterministic routing disabled")

    text = normalise(query or "")

    expression = _looks_arithmetic(text)
    if expression:
        return Route(ARITHMETIC, tool="calculator", payload=expression,
                     reason="expression parses as pure arithmetic")

    lowered = text.lower()
    match = _UNIT_PATTERN.search(text)
    if match and any(k in lowered for k in ("convert", "to", "转", "换", "=")):
        value, src, dst = match.group(1), match.group(2), match.group(3)
        if src.lower() != dst.lower():
            return Route(UNIT, tool="unit_convert", payload=f"{value} {src} {dst}",
                         reason=f"unit conversion {src} -> {dst}")

    dates = _DATE_ISO.findall(text)
    if len(dates) >= 2 and any(word in lowered for word in _DATE_WORDS):
        a = "-".join(dates[0])
        b = "-".join(dates[1])
        return Route(DATE, tool="date_diff", payload=f"{a} {b}",
                     reason="two ISO dates plus an interval keyword")

    return Route(RETRIEVE, reason="no deterministic tool matched")
