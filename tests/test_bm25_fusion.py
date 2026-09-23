"""Sparse retrieval and rank fusion invariants.

The corpus used here has **three** documents and the target term appears in
exactly one of them. Two documents would put the target term in half the corpus,
which drives the classic BM25 IDF to ``ln(1) = 0`` and makes the test pass
vacuously.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas.contracts import Scored
from atlas.engines.bm25 import SparseIndex, robertson_idf
from atlas.engines.fusion import minmax, reciprocal_rank_fusion, spearman

DOCS = [
    ("d1", "线程数应当固定在二到四之间，这是内存带宽受限下的最优解。"),
    ("d2", "倒数排名融合只使用排名，不使用分数，因此无需校准。"),
    ("d3", "接地核验按句计算证据覆盖度。"),
]


def _index() -> SparseIndex:
    index = SparseIndex()
    index.extend(DOCS)
    return index


# --------------------------------------------------------------------------
# IDF
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_docs,doc_freq", [(3, 0), (3, 1), (3, 2), (3, 3), (10, 5), (1, 1), (0, 0)])
def test_idf_is_never_negative(n_docs: int, doc_freq: int) -> None:
    assert robertson_idf(n_docs, doc_freq) >= 0.0


def test_idf_decreases_with_document_frequency() -> None:
    assert robertson_idf(10, 1) > robertson_idf(10, 5) > robertson_idf(10, 9)


def test_idf_clamps_out_of_range_frequency() -> None:
    assert robertson_idf(5, 99) == robertson_idf(5, 5)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_exact_term_hits_the_right_document_first() -> None:
    hits = _index().search("线程数固定在多少", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk_id == "d1"


def test_scores_are_strictly_positive_for_returned_hits() -> None:
    for hit in _index().search("融合 排名", k=3):
        assert hit.score > 0.0


def test_unmatched_query_returns_nothing() -> None:
    assert _index().search("完全不存在的术语zzz", k=3) == []


def test_reset_clears_the_index() -> None:
    index = _index()
    assert len(index) == 3
    index.reset()
    assert len(index) == 0
    assert index.search("线程数", k=3) == []


def test_readding_a_document_replaces_it() -> None:
    index = _index()
    index.add("d1", "完全改写的文档内容，不再包含原词。")
    assert len(index) == 3
    hits = [h.chunk_id for h in index.search("线程数", k=3)]
    assert "d1" not in hits


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def _hits(*ids: str) -> list[Scored]:
    return [Scored(cid, float(len(ids) - i)) for i, cid in enumerate(ids)]


def test_rrf_preserves_a_single_list_order() -> None:
    fused = reciprocal_rank_fusion([("dense", _hits("a", "b", "c"))], k=60)
    assert [h.chunk_id for h in fused] == ["a", "b", "c"]


def test_rrf_ranks_a_document_agreed_on_by_both_lists_highest() -> None:
    fused = reciprocal_rank_fusion(
        [("dense", _hits("a", "b", "c")), ("sparse", _hits("b", "c", "a"))], k=60
    )
    assert fused[0].chunk_id == "b"
    assert fused[0].ranks == {"dense": 2, "sparse": 1}


def test_rrf_k_one_strongly_favours_rank_one() -> None:
    small = reciprocal_rank_fusion(
        [("x", _hits("a", "b")), ("y", _hits("b", "a"))], k=1
    )
    assert small[0].score == small[1].score  # symmetric disagreement


def test_rrf_output_is_deterministic_on_ties() -> None:
    lists = [("x", _hits("a", "b")), ("y", _hits("b", "a"))]
    first = [h.chunk_id for h in reciprocal_rank_fusion(lists, k=60)]
    second = [h.chunk_id for h in reciprocal_rank_fusion(lists, k=60)]
    assert first == second


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([("x", _hits("a"))], k=0)


def test_rrf_weights_shift_the_result() -> None:
    lists = [("dense", _hits("a", "b")), ("sparse", _hits("b", "a"))]
    fused = reciprocal_rank_fusion(lists, k=60, weights={"dense": 5.0, "sparse": 0.1})
    assert fused[0].chunk_id == "a"


def test_minmax_handles_constant_input() -> None:
    assert minmax([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]
    assert minmax([]) == []
    assert minmax([0.0, 10.0]) == [0.0, 1.0]


def test_spearman_extremes_and_degenerate_input() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1], [1]) == 0.0
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0
