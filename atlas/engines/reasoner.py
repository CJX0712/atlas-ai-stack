"""ReAct reasoning kernel with a bounded step budget and deterministic pre-route.

Flow
----
1. If the router matched a deterministic tool, the tool answers directly and the
   language model is **never invoked** (``steps == 0``).
2. Otherwise run a ReAct loop: Thought -> Action -> Observation, capped at
   ``max_steps``. Exhausting the budget is not an exception: the loop returns a
   degraded-but-grounded answer built from the evidence block, so the caller
   always receives a well-formed :class:`Answer`.

Delimiter discipline
--------------------
The evidence block is wrapped in a **globally unique** tag so that a mock or
weak model cannot confuse instructions with context. The literal tag never
appears in the instruction prose - that exact mistake once made a mock provider
quote the instruction text back as if it were a retrieved passage.

Author: 晨星
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..contracts import Answer, Chunk, Message
from . import grounding as grounding_engine
from .router import RETRIEVE, Route
from .text import best_excerpt
from .tools import ToolResult, invoke as invoke_tool

EVIDENCE_OPEN = "<kb-context>"
EVIDENCE_CLOSE = "</kb-context>"

_SYSTEM = (
    "You are ATLAS, a retrieval-grounded assistant. Answer strictly from the "
    "evidence block provided in the user turn. If the evidence is insufficient, "
    "say so instead of inventing facts. You may call a tool by emitting exactly "
    "one line of the form 'Action: tool_name[argument]' where tool_name is one "
    "of calculator, unit_convert, date_diff, search. When you are done, emit a "
    "line beginning with 'Final Answer:' followed by the answer and the ids of "
    "the evidence you used."
)

SearchFn = Callable[[str, int], list[Chunk]]


@dataclass
class Toolbox:
    """Tool surface available to the reasoning loop."""

    handlers: dict[str, Callable[[str], ToolResult]] = field(default_factory=dict)
    search_fn: SearchFn | None = None

    def invoke(self, name: str, payload: str) -> ToolResult:
        if name == "search" and self.search_fn is not None:
            hits = self.search_fn(payload, 3)
            if not hits:
                return ToolResult(False, error="no documents matched the search")
            joined = " | ".join(
                f"[{c.chunk_id}] {best_excerpt(payload, c.text, 160)}" for c in hits
            )
            return ToolResult(True, joined)
        handler = self.handlers.get(name)
        if handler is not None:
            return handler(payload)
        return invoke_tool(name, payload)


def render_evidence(chunks: Sequence[Chunk]) -> str:
    """Render the evidence block, one chunk per line.

    Chunk text is flattened to a single line **on purpose**. A line-oriented
    reader (including any model that copies the block format) would otherwise
    see only the first line of each chunk and silently discard the rest - a
    failure we hit while building the offline mock provider.
    """
    if not chunks:
        return EVIDENCE_OPEN + "\n(no evidence retrieved)\n" + EVIDENCE_CLOSE
    lines = [EVIDENCE_OPEN]
    for chunk in chunks:
        flat = " ".join(chunk.text.split())
        lines.append(f"[{chunk.chunk_id}] {flat}")
    lines.append(EVIDENCE_CLOSE)
    return "\n".join(lines)


def build_prompt(question: str, chunks: Sequence[Chunk], scratchpad: str) -> list[Message]:
    body = [
        f"Question: {question}",
        "",
        render_evidence(chunks),
    ]
    if scratchpad:
        body += ["", "Previous steps:", scratchpad]
    body += [
        "",
        "Reply with either one Action line or one Final Answer line.",
    ]
    return [Message("system", _SYSTEM), Message("user", "\n".join(body))]


def parse_action(text: str) -> tuple[str, str] | None:
    """Extract ``('tool', 'payload')`` from an ``Action:`` line."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.lower().startswith("action:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if "[" not in payload or not payload.endswith("]"):
            continue
        name, _, rest = payload.partition("[")
        return name.strip(), rest[:-1].strip()
    return None


def parse_final(text: str) -> str | None:
    lines = (text or "").splitlines()
    for idx, raw in enumerate(lines):
        if raw.strip().lower().startswith("final answer:"):
            first = raw.split(":", 1)[1].strip()
            tail = [ln.strip() for ln in lines[idx + 1 :] if ln.strip()]
            return "\n".join([first] + tail).strip()
    if len(lines) == 1 and lines[0].strip():
        candidate = lines[0].strip()
        # A bare protocol directive is not an answer. Without this check an
        # agent that never stops emitting "Action:" lines would have its last
        # action line returned to the user as the final answer.
        lowered = candidate.lower()
        if not lowered.startswith(("action:", "thought:", "observation:")):
            return candidate
    return None


class Reasoner:
    """Bounded ReAct loop over an injected LLM provider."""

    def __init__(
        self,
        llm,
        *,
        toolbox: Toolbox | None = None,
        max_steps: int = 6,
        deterministic_router: bool = True,
    ) -> None:
        self.llm = llm
        self.toolbox = toolbox or Toolbox()
        self.max_steps = max(1, max_steps)
        self.deterministic_router = deterministic_router

    # -- public -----------------------------------------------------------
    def run(
        self,
        question: str,
        *,
        route: Route,
        evidence: Sequence[Chunk],
        grounding_floor: float = 0.18,
    ) -> Answer:
        started = time.perf_counter()
        if route.is_deterministic:
            answer = self._answer_from_route(question, route)
            answer.latency_ms = (time.perf_counter() - started) * 1000.0
            return answer

        trace: list[str] = []
        scratchpad = ""
        text = ""
        steps = 0
        for step in range(1, self.max_steps + 1):
            steps = step
            messages = build_prompt(question, evidence, scratchpad)
            text = self.llm.complete(messages, temperature=0.0, max_tokens=512)
            final = parse_final(text)
            if final is not None:
                break
            action = parse_action(text)
            if action is None:
                trace.append(f"step {step}: unparsable model output, stopping")
                break
            name, payload = action
            result = self.toolbox.invoke(name, payload)
            observation = result.value if result.ok else f"ERROR: {result.error}"
            trace.append(f"step {step}: {name}[{payload}] -> {observation[:120]}")
            scratchpad += (
                f"Thought: {text.splitlines()[0].strip()}\n"
                f"Action: {name}[{payload}]\n"
                f"Observation: {observation}\n"
            )
        else:
            trace.append(f"step budget of {self.max_steps} exhausted")

        answer_text = parse_final(text) or self._fallback(question, evidence)
        cites = [c.chunk_id for c in evidence]
        report = grounding_engine.verify(answer_text, [c.text for c in evidence],
                                        floor=grounding_floor)
        return Answer(
            question=question,
            text=answer_text,
            citations=cites,
            route=RETRIEVE,
            steps=steps,
            trace=trace,
            grounding=report,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # -- internals --------------------------------------------------------
    def _answer_from_route(self, question: str, route: Route) -> Answer:
        result = self.toolbox.invoke(route.tool, route.payload)
        if result.ok:
            text = f"{question.strip().rstrip('？?')} = {result.value}"
            trace = [f"deterministic route: {route.tool}[{route.payload}] -> {result.value}"]
        else:
            text = f"无法计算结果：{result.error}"
            trace = [f"deterministic route failed: {route.tool} -> {result.error}"]
        return Answer(
            question=question,
            text=text,
            citations=[],
            route=route.kind,
            steps=0,
            trace=trace + [route.reason],
            grounding=grounding_engine.GroundingReport(
                support_rate=1.0, checked=1, supported=1
            ) if result.ok else grounding_engine.GroundingReport(),
        )

    @staticmethod
    def _fallback(question: str, evidence: Sequence[Chunk]) -> str:
        if not evidence:
            return "证据不足，无法回答该问题。"
        head = evidence[0]
        snippet = head.text.strip().replace("\n", " ")[:200]
        return f"根据已有资料，最相关的依据是 [{head.chunk_id}]：{snippet}"
