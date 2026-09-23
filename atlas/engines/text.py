"""CJK-aware tokenisation and lexical similarity primitives.

Design notes
------------
* Chinese has no word delimiters, so a plain ``str.split()`` tokeniser makes
  BM25 useless. We emit **CJK character bigrams** (which is what the classic
  CJK BM25 baseline does) plus ASCII word unigrams.
* Character classification uses ``ord()`` code-point ranges only. Source files
  never contain literal symbol characters, because this repository is authored
  on a Windows/GBK-adjacent pipeline where non-ASCII literals in regex
  character classes have historically been mangled into ``bad character
  range`` errors.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# CJK Unified Ideographs (+ Extension A, compatibility) and kana/hangul.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)

_PUNCT_SKIP = set("、。！？；：“”‘’（）《》【】…—·　")


def is_cjk(ch: str) -> bool:
    """True when ``ch`` is a CJK ideograph or kana/hangul syllable."""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def split_latin_runs(text: str) -> list[str]:
    """ASCII/alphanumeric word unigrams, lowercased."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if is_cjk(ch) or ch in _PUNCT_SKIP:
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if ch.isalnum() or ch in {"_", "-", "+", "#", "."}:
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [tok for tok in out if tok and not tok.isspace()]


def cjk_bigrams(text: str) -> list[str]:
    """CJK character bigrams; isolated characters degrade to unigrams."""
    runs: list[list[str]] = []
    cur: list[str] = []
    for ch in text:
        if is_cjk(ch):
            cur.append(ch)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out: list[str] = []
    for run in runs:
        if len(run) == 1:
            out.append(run[0])
        else:
            out.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
    return out


def tokenize(text: str) -> list[str]:
    """Mixed Chinese/English tokeniser: CJK bigrams + ASCII word unigrams."""
    if not text:
        return []
    return cjk_bigrams(text) + split_latin_runs(text)


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_coefficient(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    """|Q n D| / |Q| - asymmetric, which is what we want for grounding checks.

    A long evidence sentence should not be rewarded for merely being long.
    """
    q = set(query_tokens)
    if not q:
        return 0.0
    return len(q & set(doc_tokens)) / len(q)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter for Chinese + English punctuation."""
    text = text.strip()
    if not text:
        return []
    terminators = set("。！？!?;\n")
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in terminators:
            buf.append(ch)
            candidate = "".join(buf).strip()
            if candidate:
                out.append(candidate)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    # Merge fragments shorter than 4 chars into the previous sentence.
    merged: list[str] = []
    for sent in out:
        if merged and len(sent) < 4:
            merged[-1] = merged[-1] + sent
        else:
            merged.append(sent)
    return merged


def normalise_ws(text: str) -> str:
    return " ".join(text.split())


def best_excerpt(query: str, text: str, limit: int = 200) -> str:
    """Return the sentence of ``text`` that best covers ``query``.

    Used by the retrieval tool so that a reasoning step observes a relevant
    sentence rather than the first ``limit`` characters of a chunk - the head of
    a chunk is almost always boilerplate or a heading.
    """
    flat = normalise_ws(text)
    if not flat:
        return ""
    q = token_set(query)
    if not q:
        return flat[:limit]
    best, best_score = "", -1.0
    for sentence in split_sentences(flat) or [flat]:
        score = len(q & token_set(sentence)) / len(q)
        if score > best_score:
            best, best_score = sentence, score
    return (best or flat)[:limit]

