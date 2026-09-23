"""Bounded conversation memory, vector recall and the evaluation harness.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas.engines.evaluator import (
    Case,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from atlas.engines.memory import ConversationMemory, VectorMemory, cosine

# --------------------------------------------------------------------------
# conversation memory
# --------------------------------------------------------------------------


def test_conversation_memory_compacts_and_summarises() -> None:
    memory = ConversationMemory(budget_tokens=60)
    for i in range(8):
        memory.append("user", f"这是第 {i} 轮用户消息，内容需要足够长才会触发压缩机制。")
        memory.append("assistant", f"这是第 {i} 轮助手回复，同样需要足够长以占用预算。")
    assert memory.evicted > 0
    assert memory.summary, "evicted turns must leave a summary behind"
    assert memory.tokens() <= 60 + 40, "compaction must bring the window near the budget"


def test_conversation_memory_within_budget_keeps_everything() -> None:
    memory = ConversationMemory(budget_tokens=100000)
    memory.append("user", "短消息")
    assert memory.evicted == 0
    assert memory.summary == ""
    assert len(memory.context()) == 1


def test_conversation_memory_context_prepends_the_summary() -> None:
    memory = ConversationMemory(budget_tokens=40)
    memory.append("user", "很长的一段用户消息，用来强制驱逐最早的一轮对话内容。")
    memory.append("assistant", "很长的一段助手回复，同样用于触发压缩逻辑。")
    memory.append("user", "还有更多内容，确保超过预算上限。")
    context = memory.context()
    if memory.summary:
        assert context[0].role == "system"
        assert "summary" in context[0].content.lower()


def test_conversation_memory_uses_a_custom_summariser() -> None:
    memory = ConversationMemory(budget_tokens=40, summarizer=lambda texts: "SUMMARY")
    for _ in range(6):
        memory.append("user", "很长的一段内容，用来触发压缩逻辑并调用摘要函数。")
    assert memory.summary == "SUMMARY"


# --------------------------------------------------------------------------
# vector memory
# --------------------------------------------------------------------------


def test_vector_memory_roundtrip() -> None:
    memory = VectorMemory()
    memory.remember("lesson-1", "线程数应当固定在二到四之间。")
    memory.remember("lesson-2", "倒数排名融合只使用排名。")
    assert len(memory) == 2
    recalled = memory.recall("线程数应当固定在二到四之间。", k=1)
    assert recalled and recalled[0][0].key == "lesson-1"
    assert recalled[0][1] == pytest.approx(1.0)


def test_vector_memory_upsert_replaces_by_key() -> None:
    memory = VectorMemory()
    memory.remember("k", "第一次写入的内容")
    memory.remember("k", "第二次写入的内容")
    assert len(memory) == 1
    assert memory.items[0].text == "第二次写入的内容"


def test_vector_memory_empty_recall() -> None:
    assert VectorMemory().recall("anything") == []


def test_cosine_degenerate_inputs() -> None:
    assert cosine([], []) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_recall_precision_and_rr() -> None:
    assert recall_at_k(["a", "b", "c"], ["a"], 2) == 1.0
    assert recall_at_k(["c", "b", "a"], ["a"], 2) == 0.0
    assert recall_at_k(["a"], [], 2) == 1.0
    assert precision_at_k(["a", "z"], ["a"], 2) == pytest.approx(0.5)
    assert precision_at_k([], ["a"], 2) == 0.0
    assert reciprocal_rank(["z", "a"], ["a"]) == pytest.approx(0.5)
    assert reciprocal_rank(["z"], ["a"]) == 0.0


def test_ndcg_bounds() -> None:
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert ndcg_at_k(["z", "y"], ["a"], 2) == 0.0
    assert ndcg_at_k([], [], 2) == 0.0


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

CASES = [
    Case("q1", ("d1#0", "d1#1"), "retrieve", ("d1",)),
    Case("q2", ("d2#5",), "retrieve", ("d2",)),
    Case("q3", (), "arithmetic"),
]

RETRIEVED = {
    "q1": ["d1#0", "z#0"],
    "q2": ["z#0", "d2#5"],
    "q3": [],
}


def _runner(question: str) -> dict:
    return {
        "retrieved": RETRIEVED[question],
        "route": "arithmetic" if question == "q3" else "retrieve",
        "grounded": 1.0,
        "latency_ms": 10.0,
    }


def test_evaluate_known_metrics() -> None:
    report = evaluate(CASES, _runner, k=6)
    assert report.n == 3
    assert report.retrieval_cases == 2, "the tool-routed case is excluded from retrieval metrics"
    assert report.recall_at_k == pytest.approx((0.5 + 1.0) / 2)
    assert report.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert report.doc_hit_rate == pytest.approx(1.0)
    assert report.doc_mrr == pytest.approx((1.0 + 0.5) / 2)
    assert report.route_accuracy == pytest.approx(1.0)

    expected_ndcg = (
        ndcg_at_k(RETRIEVED["q1"], CASES[0].relevant, 6)   # d1#0 found at rank 1
        + ndcg_at_k(RETRIEVED["q2"], CASES[1].relevant, 6)  # d2#5 found at rank 2
    ) / 2
    assert report.ndcg_at_k == pytest.approx(expected_ndcg)


def test_route_mismatch_lowers_route_accuracy() -> None:
    cases = [Case("q1", (), "unit")]
    report = evaluate(cases, lambda _q: {"route": "retrieve"}, k=3)
    assert report.route_accuracy == pytest.approx(0.0)


def test_evaluate_is_bit_reproducible() -> None:
    first = evaluate(CASES, _runner, k=6).deterministic_view(include_cases=True)
    second = evaluate(CASES, _runner, k=6).deterministic_view(include_cases=True)
    assert first == second


def test_deterministic_view_excludes_latency() -> None:
    view = evaluate(CASES, _runner, k=6).deterministic_view()
    assert "latency_p50_ms" not in view
    assert "latency_p95_ms" not in view
    assert "recall_at_k" in view


def test_evaluate_handles_an_empty_dataset() -> None:
    report = evaluate([], _runner, k=6)
    assert report.n == 0
    assert report.recall_at_k == 0.0
