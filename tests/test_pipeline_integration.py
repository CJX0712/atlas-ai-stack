"""Integration tests for the assembled system, fully offline.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas.pipeline.container import AtlasContainer
from atlas.settings import Settings


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def test_demo_corpus_indexes(loaded_container: AtlasContainer) -> None:
    assert len(loaded_container.store.documents()) >= 5
    assert loaded_container.store.count_chunks() >= 10
    assert len(loaded_container.index) == loaded_container.store.count_chunks()
    assert len(loaded_container.sparse) == loaded_container.store.count_chunks()


def test_reingesting_a_shorter_document_drops_stale_chunks(
    empty_container: AtlasContainer,
) -> None:
    long_text = "这是一段很长的正文内容，包含多个句子以产生多个分块。" * 120
    empty_container.ingest({"doc": long_text})
    many = empty_container.store.count_chunks()
    assert many > 1

    empty_container.ingest({"doc": "只有一个分块的短文。"})
    assert empty_container.store.count_chunks() == 1
    assert len(empty_container.index) == 1
    assert len(empty_container.sparse) == 1


def test_blank_documents_are_ignored(empty_container: AtlasContainer) -> None:
    report = empty_container.ingest({"a": "   ", "b": ""})
    assert report.documents == 0
    assert report.chunks == 0


def test_reset_clears_the_corpus(loaded_container: AtlasContainer) -> None:
    container = AtlasContainer(loaded_container.settings)
    container.load_demo_corpus()
    assert container.store.count_chunks() > 0
    container.reset()
    assert container.store.count_chunks() == 0
    assert len(container.index) == 0


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------


def test_retrieval_question_returns_answer_and_citations(
    loaded_container: AtlasContainer,
) -> None:
    payload = loaded_container.ask("混合检索里为什么不能直接对分数做加权求和？")
    assert payload["route"] == "retrieve"
    assert payload["answer"].strip()
    assert payload["citations"], "a retrieval answer must cite its evidence"
    assert payload["grounding"]["support_rate"] > 0.0
    assert payload["steps"] >= 1


def test_arithmetic_question_is_answered_without_the_model(
    loaded_container: AtlasContainer,
) -> None:
    payload = loaded_container.ask("计算 12*(3+4)+18/3 等于多少？")
    assert payload["route"] == "arithmetic"
    assert payload["steps"] == 0
    assert "90" in payload["answer"]
    assert payload["citations"] == []


def test_date_question_uses_the_date_tool(loaded_container: AtlasContainer) -> None:
    payload = loaded_container.ask("2026-01-01 到 2026-09-24 相差多少天？")
    assert payload["route"] == "date"
    assert "266" in payload["answer"]


def test_query_on_an_empty_corpus_does_not_crash(empty_container: AtlasContainer) -> None:
    payload = empty_container.ask("语料为空时会怎样？")
    assert payload["answer"].strip()
    assert payload["citations"] == []


def test_retrieval_is_deterministic_across_instances(offline_settings: Settings) -> None:
    first = AtlasContainer(offline_settings)
    first.load_demo_corpus()
    second = AtlasContainer(offline_settings)
    second.load_demo_corpus()

    question = "为什么 CPU 上的线程数不能设置得太大？"
    a = first.ask(question)
    b = second.ask(question)
    assert a["answer"] == b["answer"]
    assert a["citations"] == b["citations"]


def test_streaming_yields_the_same_text(loaded_container: AtlasContainer) -> None:
    question = "接地核验为什么按句核验？"
    whole = loaded_container.querying.query(question).text
    streamed = "".join(loaded_container.querying.stream(question))
    assert streamed == whole


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


def test_guard_probe_reproduces_the_regression(loaded_container: AtlasContainer) -> None:
    probe = loaded_container.guard_probe()
    assert probe["guarded_preserves_top1"] is True
    assert probe["naive_breaks_top1"] is True
    assert probe["spearman"] < 0.0


def test_deterministic_probe_routes_every_arithmetic_sample(
    loaded_container: AtlasContainer,
) -> None:
    probe = loaded_container.deterministic_probe()
    assert probe["all_arithmetic_routed"] is True


def test_selfcheck_is_all_green(loaded_container: AtlasContainer) -> None:
    report = loaded_container.selfcheck()
    assert report["ok"] is True, report["checks"]
    assert all(report["checks"].values())


def test_stats_reports_provider_identity(loaded_container: AtlasContainer) -> None:
    stats = loaded_container.stats()
    assert stats["build"]["llm"] == "mock"
    assert stats["build"]["dim"] > 0
    assert stats["corpus"]["chunks"] > 0


# --------------------------------------------------------------------------
# evaluation floors
# --------------------------------------------------------------------------


def test_evaluation_meets_the_published_floors(loaded_container: AtlasContainer) -> None:
    """Regression gate for the published scorecard.

    Floors are set just below the measured baseline so that any real retrieval,
    routing or grounding regression fails the build, while ordinary numeric
    drift does not. Baselines are recorded in ``docs/SPEC.md``.
    """
    report = loaded_container.evaluate()
    assert report.n >= 12
    assert report.retrieval_cases >= 10
    assert report.route_accuracy == pytest.approx(1.0), "every routable case must route"
    # Document-level retrieval is the headline: chunk-level relevance is
    # over-broad here because every chunk of a source document is marked
    # relevant (see atlas.engines.evaluator.Case).
    assert report.doc_hit_rate >= 0.95, f"doc_hit_rate={report.doc_hit_rate:.4f}"
    assert report.doc_mrr >= 0.85, f"doc_mrr={report.doc_mrr:.4f}"
    assert report.recall_at_k >= 0.80, f"recall@k={report.recall_at_k:.4f}"
    assert report.mrr >= 0.85, f"mrr={report.mrr:.4f}"
    assert report.ndcg_at_k >= 0.75, f"ndcg@k={report.ndcg_at_k:.4f}"
    assert report.grounding_rate >= 0.95, f"grounding={report.grounding_rate:.4f}"


def test_tool_routed_cases_report_no_retrieval(loaded_container: AtlasContainer) -> None:
    runner = loaded_container.querying.as_runner()
    assert runner("计算 2+2 等于多少？")["retrieved"] == []


def test_evaluation_is_reproducible(offline_settings: Settings) -> None:
    first = AtlasContainer(offline_settings)
    first.load_demo_corpus()
    second = AtlasContainer(offline_settings)
    second.load_demo_corpus()
    assert (
        first.evaluate().deterministic_view(include_cases=True)
        == second.evaluate().deterministic_view(include_cases=True)
    )
