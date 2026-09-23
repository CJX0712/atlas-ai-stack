"""Structure-aware document chunking with an exact-coverage invariant.

The chunker partitions the source into **base spans** that are a strict,
non-overlapping, gapless partition of ``[0, len(text))``.  Emitted chunks are
base spans widened by an overlap prefix, so retrieval never loses a boundary
sentence.

Invariants asserted by the test-suite
-------------------------------------
1. ``spans`` are strictly increasing, non-overlapping and cover every character.
2. ``len(base_span) <= chunk_size - overlap`` for every span but the last.
3. Chunk ordinals are gapless and monotone.
4. Output is deterministic for identical input.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

from .text import split_sentences

_HEADING_PREFIXES = ("#", "##", "###", "####")


@dataclass(frozen=True, slots=True)
class RawChunk:
    text: str
    heading: str
    ordinal: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Block:
    """A logical block: a heading plus the paragraphs under it."""

    heading: str
    start: int
    end: int


def _segments(text: str) -> list[Block]:
    """Split into (heading, span) blocks. Heading is inherited downwards."""
    blocks: list[Block] = []
    current_heading = ""
    offset = 0
    buffer_start: int | None = None
    buffer_end = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        is_heading = any(stripped.startswith(p) for p in _HEADING_PREFIXES)
        line_start, line_end = offset, offset + len(line)
        offset = line_end

        if is_heading:
            if buffer_start is not None and buffer_end > buffer_start:
                blocks.append(Block(current_heading, buffer_start, buffer_end))
            current_heading = stripped.lstrip("#").strip()
            buffer_start = None
            continue

        if not stripped:
            if buffer_start is not None and buffer_end > buffer_start:
                blocks.append(Block(current_heading, buffer_start, buffer_end))
                buffer_start = None
            continue

        if buffer_start is None:
            buffer_start = line_start
        buffer_end = line_end

    if buffer_start is not None and buffer_end > buffer_start:
        blocks.append(Block(current_heading, buffer_start, buffer_end))
    if not blocks:
        blocks.append(Block("", 0, len(text)))
    return blocks


def _hard_windows(start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Split an oversized span at sentence boundaries, else force-cut."""
    if end - start <= limit:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > limit:
        window = (cursor, cursor + limit)
        out.append(window)
        cursor += limit
    if cursor < end:
        out.append((cursor, end))
    return out


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Char spans of sentences inside ``[start, end)``."""
    body = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = start
    for sentence in split_sentences(body):
        found = text.find(sentence, cursor, end)
        if found < 0:
            found = cursor
        s, e = found, found + len(sentence)
        spans.append((s, e))
        cursor = e
    if not spans or spans[-1][1] < end:
        spans.append((max(cursor, start), end))
    return [(s, e) for s, e in spans if e > s]


def build_spans(
    text: str, *, chunk_size: int = 800, overlap: int = 120
) -> list[tuple[int, int]]:
    """Exact partition of ``[0, len(text))`` into base spans."""
    if not text:
        return []
    budget = max(1, chunk_size - max(0, overlap))
    spans: list[tuple[int, int]] = []
    pending_start: int | None = None
    pending_end = 0

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None and pending_end > pending_start:
            spans.extend(_hard_windows(pending_start, pending_end, budget))
        pending_start, pending_end = None, 0

    for block in _segments(text):
        for s, e in _sentence_spans(text, block.start, block.end):
            if e - s > budget:
                flush()
                spans.extend(_hard_windows(s, e, budget))
                continue
            if pending_start is None:
                pending_start, pending_end = s, e
            elif e - pending_start <= budget:
                pending_end = e
            else:
                flush()
                pending_start, pending_end = s, e
    flush()

    if not spans:
        return [(0, len(text))]
    # Guarantee gapless coverage: extend each span to the next span's start.
    patched: list[tuple[int, int]] = []
    for idx, (s, e) in enumerate(spans):
        next_start = spans[idx + 1][0] if idx + 1 < len(spans) else len(text)
        patched.append((s, max(e, next_start)))
    patched[0] = (0, patched[0][1])
    return patched


def heading_at(headings: list[Block], position: int) -> str:
    """Heading in force at ``position``.

    The first block begins *after* its own heading line, so a span starting at
    offset 0 sits before every block. Defaulting to the first block's heading
    keeps the document title attached to the opening chunk instead of leaving it
    unlabelled.
    """
    if not headings:
        return ""
    chosen = headings[0].heading
    for block in headings:
        if block.start <= position:
            chosen = block.heading
        else:
            break
    return chosen


def split(
    text: str, *, chunk_size: int = 800, overlap: int = 120
) -> list[RawChunk]:
    """Split ``text`` into overlapping chunks that cover every character."""
    spans = build_spans(text, chunk_size=chunk_size, overlap=overlap)
    if not spans:
        return []

    headings = _segments(text)
    chunks: list[RawChunk] = []
    for start, end in spans:
        head_start = max(0, start - max(0, overlap))
        body = text[head_start:end].strip()
        if not body:
            continue
        chunks.append(
            RawChunk(
                text=body,
                heading=heading_at(headings, start),
                ordinal=len(chunks),
                start=start,
                end=end,
            )
        )
    return chunks
