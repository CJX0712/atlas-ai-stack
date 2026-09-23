"""Text primitives and the chunker's exact-coverage invariant.

Author: 晨星
"""

from __future__ import annotations

from atlas.engines import chunking, text

_BLOCK = (
    "# 标题\n\n"
    "第一段中文内容，用来验证分块与分词。它包含若干句子。{pad}\n\n"
    "## 二级标题\n\n"
    "Second paragraph with ASCII words and numbers 12345.{pad}\n"
)

# Deliberately non-uniform: a perfectly periodic document makes every chunk
# boundary land on the same heading, so heading inheritance would never be
# exercised even though the implementation is correct.
SAMPLE = "".join(_BLOCK.format(pad="补充说明文字" * (i % 5)) for i in range(30))


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def test_tokenize_mixes_cjk_bigrams_and_ascii_words() -> None:
    tokens = text.tokenize("混合检索 Hybrid Retrieval")
    assert "混合" in tokens and "合检" in tokens and "检索" in tokens
    assert "hybrid" in tokens and "retrieval" in tokens


def test_tokenize_isolated_cjk_char_degrades_to_unigram() -> None:
    assert "猫" in text.tokenize("猫")


def test_tokenize_empty() -> None:
    assert text.tokenize("") == []


def test_split_sentences_chinese_and_english() -> None:
    sentences = text.split_sentences("第一句。第二句！Third one? Fourth.")
    assert len(sentences) == 4
    assert sentences[0].endswith("。")


def test_split_sentences_merges_tiny_fragments() -> None:
    assert len(text.split_sentences("很长的一个句子内容。好。")) == 1


def test_jaccard_and_overlap_coefficient() -> None:
    assert text.jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert text.jaccard([], []) == 1.0
    assert text.jaccard(["a"], []) == 0.0
    assert text.overlap_coefficient(["a", "b"], ["a"]) == 0.5


def test_best_excerpt_selects_the_relevant_sentence() -> None:
    passage = "无关的一句话。线程数应当固定在二到四之间。另一个无关句。"
    excerpt = text.best_excerpt("线程数应该固定在多少", passage)
    assert "线程数" in excerpt


def test_best_excerpt_empty_input() -> None:
    assert text.best_excerpt("q", "") == ""


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_spans_are_an_exact_gapless_partition() -> None:
    spans = chunking.build_spans(SAMPLE, chunk_size=800, overlap=120)
    assert spans, "expected a non-empty partition"
    assert spans[0][0] == 0
    assert spans[-1][1] == len(SAMPLE)
    for (s0, e0), (s1, _e1) in zip(spans, spans[1:]):
        assert s1 == e0, "spans must be contiguous"
        assert e0 > s0
    rebuilt = "".join(SAMPLE[s:e] for s, e in spans)
    assert rebuilt == SAMPLE


def test_spans_respect_budget_except_the_final_one() -> None:
    budget = 800 - 120
    spans = chunking.build_spans(SAMPLE, chunk_size=800, overlap=120)
    for index, (start, end) in enumerate(spans):
        assert end - start <= budget or index == len(spans) - 1


def test_split_produces_multiple_chunks_for_long_input() -> None:
    chunks = chunking.split(SAMPLE, chunk_size=800, overlap=120)
    assert len(chunks) >= 4, "the coverage path for multi-chunk documents must be exercised"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunk_text_never_exceeds_chunk_size() -> None:
    for chunk in chunking.split(SAMPLE, chunk_size=800, overlap=120):
        assert len(chunk.text) <= 800


def test_chunking_is_deterministic() -> None:
    first = chunking.split(SAMPLE, chunk_size=800, overlap=120)
    second = chunking.split(SAMPLE, chunk_size=800, overlap=120)
    assert [c.text for c in first] == [c.text for c in second]


def test_heading_is_inherited_by_chunks() -> None:
    chunks = chunking.split(SAMPLE, chunk_size=800, overlap=120)
    headings = {c.heading for c in chunks}
    assert "标题" in headings
    assert "二级标题" in headings


def test_empty_and_whitespace_input() -> None:
    assert chunking.split("", chunk_size=800, overlap=120) == []
    assert chunking.build_spans("", chunk_size=800, overlap=120) == []


def test_short_input_is_a_single_chunk() -> None:
    chunks = chunking.split("只有一句话。", chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert chunks[0].text == "只有一句话。"
