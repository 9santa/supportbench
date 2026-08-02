from supportbench.reranking.parent import ParentEvidenceRerankingRetriever
from supportbench.retrieval.base import SearchResult


class StubChunkReranker:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = [
            SearchResult("a1", 0.8, 1),
            SearchResult("b1", 0.7, 2),
            SearchResult("b2", 0.6, 3),
            SearchResult("a2", 0.1, 4),
        ]
        return results[:top_k]


def test_parent_reranker_is_ranked_only_by_cross_encoder_evidence() -> None:
    retriever = ParentEvidenceRerankingRetriever(
        StubChunkReranker(),
        parent_by_chunk_id={"a1": "A", "a2": "A", "b1": "B", "b2": "B"},
        chunk_candidate_k=4,
        second_evidence_weight=0.5,
    )

    results = retriever.search("query", top_k=2)

    assert [result.doc_id for result in results] == ["B", "A"]
    assert results[0].score > results[1].score
