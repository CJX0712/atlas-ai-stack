"""Deterministic routing and the safe arithmetic tool.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas.engines import router
from atlas.engines.tools import (
    ToolError,
    convert_unit,
    days_between,
    format_number,
    invoke,
    safe_eval,
)

ARITHMETIC_QUERIES = [
    "12*(3+4)+18/3",
    "计算 12*(3+4)+18/3 等于多少？",
    "(128+64)/8-100",
    "2^10",
    "calculate 3 * (4 + 5)",
    "99*101 =?",
]

RETRIEVAL_QUERIES = [
    "2024 年的营收是多少",
    "报告中提到了 3 个主要风险，分别是什么",
    "BM25 的 IDF 应该写成什么形式？",
    "第 2 章的结论是什么",
]


@pytest.mark.parametrize("query", ARITHMETIC_QUERIES)
def test_arithmetic_queries_route_to_the_calculator(query: str) -> None:
    route = router.route(query, deterministic=True)
    assert route.kind == router.ARITHMETIC
    assert route.tool == "calculator"
    # The payload must itself be evaluable, or the route is useless.
    safe_eval(route.payload)


@pytest.mark.parametrize("query", RETRIEVAL_QUERIES)
def test_natural_language_with_digits_does_not_route(query: str) -> None:
    assert router.route(query, deterministic=True).kind == router.RETRIEVE


def test_fullwidth_question_mark_is_stripped() -> None:
    route = router.route("计算 6*7 等于多少？", deterministic=True)
    assert route.kind == router.ARITHMETIC
    assert route.payload == "6*7"


def test_unbalanced_parentheses_do_not_route() -> None:
    assert router.route("(1+2))", deterministic=True).kind == router.RETRIEVE
    assert router.route("1+2)*(3)+4)", deterministic=True).kind == router.RETRIEVE


def test_unit_conversion_routes() -> None:
    route = router.route("1500 m to km 换算成多少？", deterministic=True)
    assert route.kind == router.UNIT
    assert route.tool == "unit_convert"
    assert route.payload.split() == ["1500", "m", "km"]


def test_identical_units_do_not_route_as_conversion() -> None:
    assert router.route("5 kg to kg", deterministic=True).kind == router.RETRIEVE


def test_date_interval_routes() -> None:
    route = router.route("2026-01-01 到 2026-09-24 相差多少天？", deterministic=True)
    assert route.kind == router.DATE
    assert route.payload == "2026-01-01 2026-09-24"


def test_two_dates_without_an_interval_keyword_do_not_route() -> None:
    assert router.route(
        "2026-01-01 发布了什么，2026-09-24 又发布了什么", deterministic=True
    ).kind == router.RETRIEVE


def test_router_can_be_disabled() -> None:
    assert router.route("1+1", deterministic=False).kind == router.RETRIEVE


def test_temperature_glyphs_are_normalised() -> None:
    normalised = router.normalise("100" + chr(176) + "C to F")
    assert "degC" in normalised
    assert chr(176) not in normalised
    assert chr(8451) not in router.normalise("20" + chr(8451))


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def test_safe_eval_arithmetic() -> None:
    assert safe_eval("12*(3+4)+18/3") == pytest.approx(90.0)
    assert safe_eval("2^10") == pytest.approx(1024.0)
    assert safe_eval("-(-5)") == pytest.approx(5.0)
    assert safe_eval("7 % 4") == pytest.approx(3.0)


@pytest.mark.parametrize(
    "expression",
    [
        "1/0",
        "2**500",
        "__import__('os').system('echo hi')",
        "1; import os",
        "open('x')",
        "(1).__class__",
        "abs(-1)",
        "",
        "((",
    ],
)
def test_safe_eval_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(ToolError):
        safe_eval(expression)


def test_tool_registry_dispatch() -> None:
    ok = invoke("calculator", "2+2")
    assert ok.ok and ok.value == "4"

    failed = invoke("calculator", "1/0")
    assert failed.ok is False and "zero" in failed.error

    assert invoke("nope", "x").ok is False


def test_unit_conversion_tables() -> None:
    assert convert_unit(1500.0, "m", "km") == pytest.approx(1.5)
    assert convert_unit(1.0, "mile", "km") == pytest.approx(1.609344)
    assert convert_unit(0.0, "C", "F") == pytest.approx(32.0)
    assert convert_unit(100.0, "C", "K") == pytest.approx(373.15)
    with pytest.raises(ToolError):
        convert_unit(1.0, "kg", "km")


def test_days_between() -> None:
    assert days_between("2026-01-01", "2026-09-24") == 266
    assert days_between("2026-09-24", "2026-01-01") == -266
    with pytest.raises(ToolError):
        days_between("not-a-date", "2026-01-01")


def test_format_number() -> None:
    assert format_number(90.0) == "90"
    assert format_number(1.5) == "1.5"


def test_unit_tool_roundtrip_through_registry() -> None:
    result = invoke("unit_convert", "1500 m km")
    assert result.ok and result.value.startswith("1.5")

    date_result = invoke("date_diff", "2026-01-01 2026-09-24")
    assert date_result.ok and date_result.value == "266 天"
