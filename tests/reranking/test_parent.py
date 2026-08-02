from supportbench.reranking.parent import ParentEvidenceRerankingRetriever
from supportbench.retrieval.base import SearchResult


class ScoredChunkRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return list(self._results[:top_k])


def test_ranks_parents_only_by_aggregated_cross_encoder_scores() -> None:
    retriever = ParentEvidenceRerankingRetriever(
        ScoredChunkRetriever(
            (
                SearchResult("b_1", score=0.9, rank=1),
                SearchResult("a_1", score=0.8, rank=2),
                SearchResult("a_2", score=0.8, rank=3),
                SearchResult("b_2", score=0.1, rank=4),
            )
        ),
        parent_by_chunk_id={
            "a_1": "parent_a",
            "a_2": "parent_a",
            "b_1": "parent_b",
            "b_2": "parent_b",
        },
        chunk_candidate_k=4,
        second_evidence_weight=1.0,
    )

    results = retriever.search("query", top_k=2)

    assert [result.doc_id for result in results] == ["parent_a", "parent_b"]
    assert [result.score for result in results] == [0.8, 0.5]
    assert [result.rank for result in results] == [1, 2]


def test_zero_second_evidence_weight_matches_best_chunk_collapse() -> None:
    retriever = ParentEvidenceRerankingRetriever(
        ScoredChunkRetriever(
            (
                SearchResult("a_1", score=0.9, rank=1),
                SearchResult("b_1", score=0.8, rank=2),
                SearchResult("b_2", score=0.7, rank=3),
            )
        ),
        parent_by_chunk_id={
            "a_1": "parent_a",
            "b_1": "parent_b",
            "b_2": "parent_b",
        },
        chunk_candidate_k=3,
        second_evidence_weight=0.0,
    )

    results = retriever.search("query", top_k=2)

    assert [result.doc_id for result in results] == ["parent_a", "parent_b"]
    assert [result.score for result in results] == [0.9, 0.8]
