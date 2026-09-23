"""Deterministic mock LLM - the reason ``verify`` runs with no network and no key.

Contract
--------
This provider is an **extractive** reader, not a chatbot. It implements the
ReAct protocol with a provably bounded step count:

1. Score every evidence sentence (and every observation excerpt) by cosine
   similarity over content-token sets.
2. If the best score clears ``_MIN_OVERLAP``, or an observation is already
   present, or the loop is not allowed to act - commit to a ``Final Answer``.
3. Otherwise emit exactly one ``Action: search[...]`` line.

Because step 2 commits whenever an observation exists, the loop terminates in at
most two model calls. This is asserted in ``tests/test_reasoner.py``.

Two failure modes found while building this, both silently fatal:

* Evidence rendered across multiple lines was parsed line by line, so only the
  first line of every chunk reached the model - roughly 90 percent of the
  context was discarded. Both render and parse now handle multi-line chunks.
* Scoring by *query coverage* (``|Q n S| / |Q|``) collapses for multi-clause
  questions, so the reader never committed and looped until the step budget ran
  out. Scoring by cosine over token sets is length-robust.

Author: 晨星
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ..contracts import Message
from ..engines.grounding import content_tokens
from ..engines.reasoner import EVIDENCE_CLOSE, EVIDENCE_OPEN
from ..engines.text import split_sentences

_EVIDENCE_LINE = re.compile(r"^\[([^\]]+)\]\s*(.+)$")
_EXCERPT = re.compile(r"\[([^\]]+)\]\s*([^|]+)")
_QUESTION_LINE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_ACTION_HINT = "Action:"
_FINAL_HINT = "Final Answer:"
_MIN_OVERLAP = 0.28
_MIN_SENTENCE_CHARS = 8


def _split_prompt(user_content: str) -> tuple[str, list[tuple[str, str]], str]:
    """Return ``(question, [(chunk_id, text)], scratchpad)``."""
    question = ""
    match = _QUESTION_LINE.search(user_content)
    if match:
        question = match.group(1).strip()

    evidence: list[tuple[str, str]] = []
    start = user_content.find(EVIDENCE_OPEN)
    end = user_content.find(EVIDENCE_CLOSE)
    if start >= 0 and end > start:
        block = user_content[start + len(EVIDENCE_OPEN): end]
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith("("):
                continue
            hit = _EVIDENCE_LINE.match(line)
            if hit:
                evidence.append((hit.group(1).strip(), hit.group(2).strip()))
            elif evidence:
                chunk_id, prior = evidence[-1]
                evidence[-1] = (chunk_id, f"{prior} {line}".strip())

    scratchpad = ""
    marker = "Previous steps:"
    pos = user_content.find(marker)
    if pos >= 0:
        scratchpad = user_content[pos + len(marker):]
    return question, evidence, scratchpad


def _parse_observations(scratchpad: str) -> list[tuple[str, str]]:
    """Extract ``(chunk_id, excerpt)`` pairs from search observations."""
    out: list[tuple[str, str]] = []
    for raw in scratchpad.splitlines():
        line = raw.strip()
        if not line.startswith("Observation:"):
            continue
        body = line.split(":", 1)[1]
        for chunk_id, excerpt in _EXCERPT.findall(body):
            text = excerpt.strip()
            if text:
                out.append((chunk_id.strip(), text))
    return out


def _score(question_tokens: set[str], sentence: str) -> float:
    tokens = set(content_tokens(sentence))
    if not tokens or not question_tokens:
        return 0.0
    return len(question_tokens & tokens) / ((len(question_tokens) * len(tokens)) ** 0.5)


def _best(
    question: str, pool: Sequence[tuple[str, str]]
) -> tuple[float, str, str]:
    """Highest-scoring sentence across ``pool``; headings are skipped."""
    q_tokens = set(content_tokens(question))
    if not q_tokens:
        return 0.0, "", ""
    best = (0.0, "", "")
    for chunk_id, text in pool:
        for sentence in split_sentences(text) or [text]:
            stripped = sentence.strip()
            if len(stripped) < _MIN_SENTENCE_CHARS or stripped.lstrip().startswith("#"):
                continue
            score = _score(q_tokens, stripped)
            if score > best[0]:
                best = (score, stripped, chunk_id)
    return best


class MockLLM:
    """Offline, deterministic extractive reader."""

    def __init__(self, react: bool = True) -> None:
        self.react = react
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        del temperature, max_tokens
        self.calls += 1
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question, evidence, scratchpad = _split_prompt(user)
        observations = _parse_observations(scratchpad)

        pool = list(evidence) + observations
        score, sentence, chunk_id = _best(question, pool)

        enough = score >= _MIN_OVERLAP or bool(observations) or not self.react
        if enough:
            if not sentence:
                return f"{_FINAL_HINT} 证据不足，无法回答该问题。"
            return f"{_FINAL_HINT} {sentence} (依据：{chunk_id})"

        return f"Thought: 现有证据不足，先扩展检索。\n{_ACTION_HINT} search[{question}]"

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterable[str]:
        text = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        step = 16
        for i in range(0, len(text), step):
            yield text[i: i + step]
