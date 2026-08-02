from typing import cast

import pytest

from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.hybrid import WeightedRetrieverSource
from supportbench.retrieval.parent_hybrid import (
    ParentAggregation,
    ParentCandidateChunkRetriever,
    ParentCandidateSubsetRetriever,
    ParentWeightedRRFHybrid,
)


class RankingRetriever:
    def __init__(self, doc_ids: tuple[str, ...]) -> None:
        self._doc_ids = doc_ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(doc_id=doc_id, score=1.0 / rank, rank=rank)
            for rank, doc_id in enumerate(self._doc_ids[:top_k], start=1)
        ]


def _build_parent_retriever(
    *,
    aggregation: str = "best_chunk_rank",
) -> ParentWeightedRRFHybrid:
    return ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                name="bm25",
                retriever=RankingRetriever(("a_bm25", "b_bm25", "a_other")),
            ),
            WeightedRetrieverSource(
                name="dense",
                retriever=RankingRetriever(("c_dense", "a_dense", "b_dense")),
            ),
        ),
        parent_by_chunk_id={
            "a_bm25": "parent_a",
            "a_other": "parent_a",
            "a_dense": "parent_a",
            "b_bm25": "parent_b",
            "b_dense": "parent_b",
            "c_dense": "parent_c",
        },
        source_candidate_k=3,
        rrf_k=10,
        aggregation=cast(ParentAggregation, aggregation),
        representative_chunks_per_parent=2,
    )


def test_fuses_different_chunks_at_parent_level() -> None:
    results = _build_parent_retriever().search_with_chunks("query", top_k=3)

    assert [result.parent_id for result in results] == [
        "parent_a",
        "parent_b",
        "parent_c",
    ]
    assert results[0].representative_chunk_ids == ("a_bm25", "a_dense")


def test_capped_sum_uses_second_chunk_from_same_source() -> None:
    best = _build_parent_retriever().search("query", top_k=3)
    capped = _build_parent_retriever(aggregation="capped_top_2_sum").search(
        "query",
        top_k=3,
    )

    best_scores = {result.doc_id: result.score for result in best}
    capped_scores = {result.doc_id: result.score for result in capped}

    assert capped_scores["parent_a"] > best_scores["parent_a"]
    assert capped_scores["parent_b"] == pytest.approx(best_scores["parent_b"])


def test_materializes_representative_chunks_for_reranking() -> None:
    retriever = ParentCandidateChunkRetriever(
        _build_parent_retriever(),
        parent_candidate_k=2,
        chunks_per_parent=2,
    )

    results = retriever.search("query", top_k=4)

    assert [result.doc_id for result in results] == [
        "a_bm25",
        "a_dense",
        "b_bm25",
        "b_dense",
    ]
    assert [result.rank for result in results] == [1, 2, 3, 4]


def test_selects_nested_parent_candidate_subset() -> None:
    retriever = ParentCandidateSubsetRetriever(
        RankingRetriever(("a_1", "a_2", "a_3", "b_1", "b_2", "c_1")),
        parent_by_chunk_id={
            "a_1": "parent_a",
            "a_2": "parent_a",
            "a_3": "parent_a",
            "b_1": "parent_b",
            "b_2": "parent_b",
            "c_1": "parent_c",
        },
        source_candidate_k=6,
        parent_candidate_k=2,
        chunks_per_parent=2,
    )

    results = retriever.search("query", top_k=4)

    assert [result.doc_id for result in results] == ["a_1", "a_2", "b_1", "b_2"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
