"""Deterministic tool suite: safe arithmetic, unit conversion, date arithmetic.

``safe_eval`` walks a whitelisted AST instead of calling ``eval``.  Only numeric
constants and the five arithmetic operators plus ``//``, ``%`` and ``**`` are
permitted; the exponent is bounded so a malicious input cannot exhaust memory.

Author: 晨星
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
from dataclasses import dataclass
from typing import Callable

MAX_EXPONENT = 64

_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class ToolError(ValueError):
    """Raised when a tool cannot satisfy the request."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("exponent too large")
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unary operator {type(node.op).__name__} is not allowed")
        return float(op(_eval_node(node.operand)))
    raise ToolError(f"node {type(node).__name__} is not allowed")


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression. Raises ``ToolError`` otherwise."""
    text = (expression or "").replace("^", "**").strip()
    if not text:
        raise ToolError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"cannot parse expression: {exc.msg}") from exc
    try:
        value = _eval_node(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except OverflowError as exc:
        raise ToolError("numeric overflow") from exc
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        raise ToolError("result is not finite")
    return value


def format_number(value: float, digits: int = 10) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}g}"


# --------------------------------------------------------------------------
# Unit conversion - explicit tables, no implicit prefix magic
# --------------------------------------------------------------------------

# factor -> metres
_LENGTH = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
           "mile": 1609.344, "miles": 1609.344, "ft": 0.3048, "feet": 0.3048,
           "inch": 0.0254, "in": 0.0254, "yd": 0.9144, "海里": 1852.0,
           "nm": 1852.0}
# factor -> kilograms
_MASS = {"kg": 1.0, "g": 0.001, "mg": 1e-06, "t": 1000.0, "ton": 1000.0,
         "lb": 0.45359237, "lbs": 0.45359237, "oz": 0.028349523125,
         "斤": 0.5}
_TEMP = {"degc": "C", "degf": "F", "c": "C", "f": "F", "k": "K", "kelvin": "K"}


def _to_celsius(value: float, unit: str) -> float:
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    if unit == "K":
        return value - 273.15
    raise ToolError(f"unknown temperature unit {unit}")


def _from_celsius(value: float, unit: str) -> float:
    if unit == "C":
        return value
    if unit == "F":
        return value * 9.0 / 5.0 + 32.0
    if unit == "K":
        return value + 273.15
    raise ToolError(f"unknown temperature unit {unit}")


def convert_unit(value: float, src: str, dst: str) -> float:
    s, d = src.strip().lower(), dst.strip().lower()
    if not s or not d:
        raise ToolError("missing unit")
    if s in _TEMP and d in _TEMP:
        return _from_celsius(_to_celsius(value, _TEMP[s]), _TEMP[d])
    if s in _LENGTH and d in _LENGTH:
        return value * _LENGTH[s] / _LENGTH[d]
    if s in _MASS and d in _MASS:
        return value * _MASS[s] / _MASS[d]
    raise ToolError(f"no conversion from {src} to {dst}")


# --------------------------------------------------------------------------
# Date arithmetic
# --------------------------------------------------------------------------


def parse_date(token: str) -> _dt.date:
    token = token.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return _dt.datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    raise ToolError(f"unsupported date format: {token}")


def days_between(a: str, b: str) -> int:
    """Signed day count from ``a`` to ``b``."""
    return (parse_date(b) - parse_date(a)).days


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolResult:
    ok: bool
    value: str = ""
    error: str = ""


def _calculator(payload: str) -> ToolResult:
    try:
        return ToolResult(True, format_number(safe_eval(payload)))
    except ToolError as exc:
        return ToolResult(False, error=str(exc))


def _unit_convert(payload: str) -> ToolResult:
    parts = payload.split()
    if len(parts) != 3:
        return ToolResult(False, error="expected '<value> <from> <to>'")
    try:
        value = float(parts[0])
        result = convert_unit(value, parts[1], parts[2])
    except (ToolError, ValueError) as exc:
        return ToolResult(False, error=str(exc))
    return ToolResult(True, f"{format_number(result)} {parts[2].lower()}")


def _date_diff(payload: str) -> ToolResult:
    parts = payload.split()
    if len(parts) != 2:
        return ToolResult(False, error="expected '<date-a> <date-b>'")
    try:
        delta = days_between(parts[0], parts[1])
    except ToolError as exc:
        return ToolResult(False, error=str(exc))
    return ToolResult(True, f"{abs(delta)} 天")


REGISTRY: dict[str, Callable[[str], ToolResult]] = {
    "calculator": _calculator,
    "unit_convert": _unit_convert,
    "date_diff": _date_diff,
}


def invoke(tool: str, payload: str) -> ToolResult:
    handler = REGISTRY.get(tool)
    if handler is None:
        return ToolResult(False, error=f"unknown tool: {tool}")
    return handler(payload)
