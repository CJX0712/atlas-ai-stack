"""Grounding / hallucination guard.

Every declarative sentence in the answer must be traceable to an evidence
sentence.  We measure ``|answer_tokens n evidence_tokens| / |answer_tokens|``
per answer sentence and take the best evidence sentence as its support.

The asymmetric overlap is intentional: a long evidence passage must not be
rewarded merely for being long.

Author: 晨星
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..contracts import GroundingReport
from .text import normalise_ws, overlap_coefficient, split_sentences, tokenize

_ASCII_STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "to", "and", "or", "in",
    "on", "for", "that", "this", "it", "as", "with", "by", "be", "at", "from",
    "which", "but", "not", "have", "has", "had", "will", "would", "can", "could",
}
_CJK_STOP = {"的", "了", "是", "在", "和", "与", "就", "都", "而", "及", "或",
             "也", "有", "被", "把", "对", "为", "以", "之", "其", "这", "那"}

# Citation markers are provenance metadata, not factual claims. Leaving them in
# makes the guard flag its own citations: "(依据：hybrid-retrieval#0)" shares no
# token with the evidence and was scored as an unsupported assertion, halving
# the support rate of otherwise perfect answers.
_CITATION_PAREN = re.compile(r"[（(]\s*(?:依据|来源|出处|source|citation)\s*[:：][^）)]*[）)]")
_CITATION_ID = re.compile(r"\[[A-Za-z0-9_.\-]+#\d+\]")

MIN_CLAIM_CHARS = 6
MIN_CONTENT_TOKENS = 2


def strip_citations(text: str) -> str:
    """Remove provenance markers so they are not scored as claims."""
    return _CITATION_ID.sub(" ", _CITATION_PAREN.sub(" ", text or ""))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _ASCII_STOP and t not in _CJK_STOP]


def _is_claim(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < MIN_CLAIM_CHARS:
        return False
    if stripped.endswith("?") or stripped.endswith("？"):
        return False
    # Boilerplate hedges carry no factual content and must not fail the check.
    return not stripped.startswith(("证据不足", "无法", "根据已有资料，最相关"))


def verify(
    answer: str, evidence: Sequence[str], *, floor: float = 0.18
) -> GroundingReport:
    """Return a grounding report for ``answer`` against ``evidence``."""
    cleaned = normalise_ws(strip_citations(answer))
    sentences = [s for s in split_sentences(cleaned) if _is_claim(s)]
    if not sentences:
        return GroundingReport(support_rate=1.0, checked=0, supported=0)

    evidence_tokens: list[list[str]] = []
    for block in evidence:
        for sentence in split_sentences(normalise_ws(block)) or [block]:
            toks = content_tokens(sentence)
            if toks:
                evidence_tokens.append(toks)

    unsupported: list[str] = []
    supported = 0
    for sentence in sentences:
        toks = content_tokens(sentence)
        if len(toks) < MIN_CONTENT_TOKENS:
            supported += 1  # too short to be a factual claim
            continue
        best = 0.0
        for toks_e in evidence_tokens:
            score = overlap_coefficient(toks, toks_e)
            if score > best:
                best = score
            if best >= floor:
                break
        if best >= floor:
            supported += 1
        else:
            unsupported.append(sentence)

    checked = len(sentences)
    return GroundingReport(
        support_rate=(supported / checked) if checked else 1.0,
        unsupported=unsupported,
        checked=checked,
        supported=supported,
    )
