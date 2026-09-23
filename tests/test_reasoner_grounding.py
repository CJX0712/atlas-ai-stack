"""Reasoning loop bounds and grounding verification.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from atlas.contracts import Chunk, Message
from atlas.engines import grounding
from atlas.engines.reasoner import (
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    Reasoner,
    Toolbox,
    parse_action,
    parse_final,
    render_evidence,
)
from atlas.engines.router import ARITHMETIC, RETRIEVE, Route
from atlas.engines.tools import REGISTRY
from atlas.providers.mock_llm import MockLLM, _split_prompt


class CountingLLM:
    """Records how often it was called."""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting"

    def complete(self, messages: Sequence[Message], **_kwargs) -> str:
        self.calls += 1
        if not self.replies:
            return "Final Answer: done"
        index = min(self.calls - 1, len(self.replies) - 1)
        return self.replies[index]

    def stream(self, messages: Sequence[Message], **kwargs) -> Iterable[str]:
        yield self.complete(messages, **kwargs)


def _chunk(text: str, chunk_id: str = "doc#0") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc", text=text)


# --------------------------------------------------------------------------
# evidence rendering / parsing
# --------------------------------------------------------------------------


def test_render_evidence_flattens_multiline_chunks() -> None:
    rendered = render_evidence([_chunk("first line\nsecond line\nthird line")])
    assert "first line second line third line" in rendered
    assert EVIDENCE_OPEN in rendered and EVIDENCE_CLOSE in rendered
    # Exactly three lines: open tag, one chunk, close tag.
    assert len(rendered.splitlines()) == 3


def test_render_evidence_with_no_chunks() -> None:
    rendered = render_evidence([])
    assert "no evidence retrieved" in rendered


def test_rendered_evidence_round_trips_through_the_reader_parser() -> None:
    """The render/parse pair must not silently drop multi-line chunk text.

    Regression: the parser originally read line by line, so every chunk
    contributed only its first line to the model's context.
    """
    chunks = [
        _chunk("alpha sentence one\n\nbeta sentence two", "a#0"),
        _chunk("gamma sentence three", "b#0"),
    ]
    prompt = f"Question: beta\n\n{render_evidence(chunks)}"
    question, parsed, _scratch = _split_prompt(prompt)
    assert question == "beta"
    assert len(parsed) == 2
    assert parsed[0][1] == "alpha sentence one beta sentence two"
    assert parsed[1][1] == "gamma sentence three"


# --------------------------------------------------------------------------
# protocol parsing
# --------------------------------------------------------------------------


def test_parse_action() -> None:
    assert parse_action("Action: calculator[2+2]") == ("calculator", "2+2")
    assert parse_action("Thought: x\nAction: search[线程数]") == ("search", "线程数")
    assert parse_action("no action here") is None
    assert parse_action("Action: broken") is None


def test_parse_final() -> None:
    assert parse_final("Final Answer: 42") == "42"
    assert parse_final("Final Answer: line one\nline two") == "line one\nline two"
    assert parse_final("single line") == "single line"
    assert parse_final("line one\nline two\nline three") is None


# --------------------------------------------------------------------------
# reasoner
# --------------------------------------------------------------------------


def test_deterministic_route_never_calls_the_model() -> None:
    llm = CountingLLM()
    reasoner = Reasoner(llm, toolbox=Toolbox(handlers=dict(REGISTRY)), max_steps=4)
    answer = reasoner.run(
        "12*(3+4)+18/3",
        route=Route(ARITHMETIC, "calculator", "12*(3+4)+18/3"),
        evidence=[],
    )
    assert llm.calls == 0
    assert answer.steps == 0
    assert answer.route == ARITHMETIC
    assert "90" in answer.text


def test_deterministic_route_reports_tool_failure_without_crashing() -> None:
    reasoner = Reasoner(CountingLLM(), max_steps=2)
    answer = reasoner.run("1/0", route=Route(ARITHMETIC, "calculator", "1/0"), evidence=[])
    assert answer.steps == 0
    assert "无法计算" in answer.text


def test_react_loop_terminates_after_a_search_observation() -> None:
    llm = CountingLLM(["Thought: need more\nAction: search[线程数]", "Final Answer: 固定在二到四"])
    evidence = [_chunk("线程数应当固定在二到四之间。")]
    reasoner = Reasoner(llm, toolbox=Toolbox(search_fn=lambda q, k: evidence), max_steps=5)
    answer = reasoner.run("线程数固定在多少", route=Route(RETRIEVE), evidence=evidence)
    assert answer.steps == 2
    assert llm.calls == 2
    assert "固定在二到四" in answer.text
    assert answer.trace, "the tool call must be recorded in the trace"


def test_step_budget_is_enforced_and_degrades_gracefully() -> None:
    llm = CountingLLM(["Action: search[永远不结束]"])
    evidence = [_chunk("这是一段可以兜底的证据文本，用来验证预算耗尽后的降级回答。")]
    reasoner = Reasoner(llm, toolbox=Toolbox(search_fn=lambda q, k: evidence), max_steps=3)
    answer = reasoner.run("永不停歇的问题", route=Route(RETRIEVE), evidence=evidence)
    assert answer.steps == 3
    assert answer.text, "a degraded answer must still be returned"
    assert any("budget" in step for step in answer.trace)


def test_unparsable_output_stops_the_loop() -> None:
    llm = CountingLLM(["line one\nline two\nline three"])
    evidence = [_chunk("兜底证据文本，长度足够形成一句可核验的断言。")]
    reasoner = Reasoner(llm, max_steps=5)
    answer = reasoner.run("问题", route=Route(RETRIEVE), evidence=evidence)
    assert answer.steps == 1
    assert any("unparsable" in step for step in answer.trace)


def test_mock_provider_halts_within_two_calls() -> None:
    llm = MockLLM()
    reasoner = Reasoner(llm, max_steps=6)
    answer = reasoner.run(
        "线程数固定在多少",
        route=Route(RETRIEVE),
        evidence=[_chunk("线程数应当固定在二到四之间，这是内存带宽受限下的最优解。")],
    )
    assert llm.calls <= 2
    assert answer.steps <= 2


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------


def test_fully_supported_answer() -> None:
    evidence = ["线程数应当固定在二到四之间。"]
    report = grounding.verify("线程数应当固定在二到四之间。", evidence)
    assert report.support_rate == pytest.approx(1.0)
    assert report.unsupported == []
    assert report.checked == 1


def test_fabricated_sentence_is_flagged() -> None:
    evidence = ["线程数应当固定在二到四之间。"]
    answer = "线程数应当固定在二到四之间。量子计算机需要液氦冷却才能维持超导。"
    report = grounding.verify(answer, evidence, floor=0.2)
    assert report.checked == 2
    assert report.supported == 1
    assert report.support_rate == pytest.approx(0.5)
    assert len(report.unsupported) == 1
    assert "液氦" in report.unsupported[0]


def test_boilerplate_hedge_is_not_penalised() -> None:
    report = grounding.verify("证据不足，无法回答该问题。", ["随便一段证据。"])
    assert report.support_rate == pytest.approx(1.0)
    assert report.checked == 0


def test_empty_answer_is_vacuously_grounded() -> None:
    report = grounding.verify("", ["evidence"])
    assert report.support_rate == pytest.approx(1.0)


def test_citation_markers_are_not_treated_as_claims() -> None:
    """Regression: the guard used to flag its own citations.

    "(依据：doc#0)" shares no token with the evidence text, so it was scored as
    an unsupported assertion and halved the support rate of a fully grounded
    answer.
    """
    evidence = ["线程数应当固定在二到四之间。"]
    report = grounding.verify("线程数应当固定在二到四之间。 (依据：cpu#3)", evidence)
    assert report.checked == 1
    assert report.support_rate == pytest.approx(1.0)


def test_strip_citations_only_removes_provenance_markers() -> None:
    assert grounding.strip_citations("结论 [doc#1] 成立") == "结论   成立"
    assert grounding.strip_citations("区间 [0, 1] 内") == "区间 [0, 1] 内"


def test_contentless_sentence_is_flagged_rather_than_silently_passed() -> None:
    """The lexical guard errs toward flagging.

    Bigram tokenisation means unigram stopword lists cannot neutralise filler
    like "这个和那个都是的了" - the *bigrams* are not stopwords. Rather than
    hand-maintaining a bigram stop list, the guard reports the sentence as
    unsupported. That is the conservative direction: a false alarm is visible
    and cheap, a silent pass is neither.
    """
    report = grounding.verify("这个和那个都是的了。", ["完全不相干的一段文字内容。"], floor=0.2)
    assert report.checked == 1
    assert report.support_rate == pytest.approx(0.0)
    assert len(report.unsupported) == 1
