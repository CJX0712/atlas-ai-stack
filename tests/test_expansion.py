"""Thesaurus query expansion, including the mechanism it exists to fix.

Author: 晨星
"""

from __future__ import annotations

from atlas.engines import expansion
from atlas.engines.bm25 import SparseIndex
from atlas.engines.chunking import split
from atlas.resources import load_corpus


def test_expansion_adds_known_english_terms() -> None:
    result = expansion.expand("线程数应该固定在多少")
    assert "线程数" in result.matched
    assert any("thread" in term for term in result.terms)
    assert "thread" in result.query


def test_expansion_is_silent_for_unknown_terms() -> None:
    result = expansion.expand("完全不认识的词组xyz")
    assert result.terms == ()
    assert result.query == "完全不认识的词组xyz"


def test_expansion_of_empty_query() -> None:
    result = expansion.expand("")
    assert result.query == ""
    assert result.terms == ()


def test_expansion_prefers_the_longest_key() -> None:
    result = expansion.expand("线程数")
    assert "线程数" in result.matched


def test_expansion_does_not_duplicate_terms_already_present() -> None:
    result = expansion.expand("thread 线程")
    assert len(set(term.lower() for term in result.terms)) == len(result.terms)
    assert "thread" not in [term.lower() for term in result.terms]


def test_expansion_is_deterministic() -> None:
    assert expansion.expand("重排与检索").query == expansion.expand("重排与检索").query


def test_expansion_respects_the_term_cap() -> None:
    result = expansion.expand("检索 重排 融合 排名 语料 文档", max_terms=3)
    assert len(result.terms) == 3


# --------------------------------------------------------------------------
# mechanism: cross-lingual sparse retrieval
# --------------------------------------------------------------------------


def _corpus_index() -> SparseIndex:
    index = SparseIndex()
    for doc_id, body in load_corpus().items():
        for chunk in split(body, chunk_size=800, overlap=120):
            index.add(f"{doc_id}#{chunk.ordinal}", chunk.text)
    return index


def test_expansion_makes_a_cross_lingual_query_retrievable() -> None:
    """A Chinese question must reach its English source document.

    Without expansion the query shares no token with the English passage, so the
    sparse retriever is blind to it. This is the concrete failure the module was
    written for, asserted rather than asserted-by-comment.
    """
    index = _corpus_index()
    question = "线程数应该固定在多少，为什么不能用满所有核心？"

    plain = [hit.chunk_id for hit in index.search(question, 5)]
    expanded = [hit.chunk_id for hit in index.search(expansion.expand(question).query, 5)]

    assert any(cid.startswith("cpu-inference") for cid in expanded), (
        "expanded query must reach the English CPU-inference document"
    )
    assert not any(cid.startswith("cpu-inference") for cid in plain), (
        "the unexpanded query is expected to miss it entirely"
    )
