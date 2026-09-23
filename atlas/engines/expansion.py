"""Domain thesaurus query expansion for the sparse retriever.

Why this exists
---------------
The corpus is bilingual, but a purely lexical sparse retriever cannot bridge
languages: the Chinese query "线程数" shares no token with the English heading
"thread count". A hash-based dense encoder does not bridge it either, because it
is trained on nothing. Measured: a Chinese question about CPU thread tuning
ranked the correct English document outside the top three.

The fix is the classical IR one: expand the query with a curated domain
thesaurus before handing it to the sparse retriever. Expansion deliberately
applies **only to the sparse branch** - the dense branch keeps the raw query, so
the two retrievers stay genuinely complementary instead of collapsing onto the
same representation.

Scope discipline: this is a small, explicit, reviewable lexicon of technical
terms, not a general translation model. Unknown terms expand to nothing, which
is asserted in the test-suite.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical bilingual technical lexicon for this domain.
LEXICON: dict[str, tuple[str, ...]] = {
    "线程": ("thread", "threads"),
    "线程数": ("thread count", "threads", "n_threads"),
    "小模型": ("small model", "quantised", "quantized"),
    "推理": ("inference", "decoding"),
    "吞吐": ("throughput", "tokens per second"),
    "核心": ("core", "cores", "cpu count"),
    "量化": ("quantised", "quantized", "int8"),
    "检索": ("retrieval", "retriever", "search"),
    "召回": ("recall", "retrieve"),
    "融合": ("fusion", "fuse", "rrf", "reciprocal rank fusion"),
    "排名": ("rank", "ranking"),
    "重排": ("rerank", "reranker", "reranking", "cross-encoder"),
    "交叉编码器": ("cross-encoder", "cross encoder"),
    "归一化": ("normalisation", "normalization", "min-max"),
    "分块": ("chunk", "chunks", "chunking"),
    "索引": ("index", "indexing"),
    "语料": ("corpus", "document set"),
    "文档": ("document", "documents", "passage"),
    "停用词": ("stopword", "stopwords"),
    "阈值": ("threshold", "floor"),
    "接地": ("grounding", "grounded"),
    "核验": ("verify", "verification", "check"),
    "幻觉": ("hallucination", "hallucinated", "fabricated"),
    "证据": ("evidence", "support"),
    "断言": ("claim", "assertion", "sentence"),
    "路由": ("route", "routing", "router"),
    "确定性": ("deterministic", "deterministic routing"),
    "计算器": ("calculator", "arithmetic"),
    "算术": ("arithmetic", "expression"),
    "单位": ("unit", "conversion"),
    "复现": ("reproducible", "reproducibility", "deterministic"),
    "依赖": ("dependency", "dependencies", "lock"),
    "锁定": ("lock", "pinned", "pin", "freeze"),
    "干净环境": ("clean environment", "clean room", "fresh install"),
    "单元测试": ("unit test", "unit tests", "test suite"),
    "端到端": ("end-to-end", "e2e", "integration"),
    "锁文件": ("lockfile", "requirements.lock", "freeze"),
    "哈希": ("hash", "hashing", "blake2b"),
    "评估": ("evaluation", "eval", "metrics"),
    "排序": ("ranking", "order", "ordering"),
}

MAX_EXPANSIONS = 12


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    query: str
    terms: tuple[str, ...]
    matched: tuple[str, ...]


def expand(query: str, *, max_terms: int = MAX_EXPANSIONS) -> ExpansionResult:
    """Return ``query`` widened with thesaurus terms. Deterministic ordering."""
    if not query:
        return ExpansionResult("", (), ())

    lowered = query.lower()
    extras: list[str] = []
    matched: list[str] = []
    seen: set[str] = set()

    # Longest keys first so that "线程数" wins over "线程".
    for key in sorted(LEXICON, key=len, reverse=True):
        if key not in query:
            continue
        matched.append(key)
        for term in LEXICON[key]:
            low = term.lower()
            if low in seen or low in lowered:
                continue
            seen.add(low)
            extras.append(term)
            if len(extras) >= max_terms:
                break
        if len(extras) >= max_terms:
            break

    if not extras:
        return ExpansionResult(query, (), tuple(matched))
    return ExpansionResult(f"{query} {' '.join(extras)}", tuple(extras), tuple(matched))
