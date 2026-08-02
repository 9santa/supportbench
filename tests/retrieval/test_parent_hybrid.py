from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.hybrid import WeightedRetrieverSource
from supportbench.retrieval.parent_hybrid import (
    ParentCandidateChunkRetriever,
    ParentWeightedRRFHybrid,
)


class StubRetriever:
    def __init__(self, document_ids: list[str]) -> None:
        self._document_ids = document_ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(document_id, 1.0 / rank, rank)
            for rank, document_id in enumerate(self._document_ids[:top_k], start=1)
        ]


def test_parent_wrrf_caps_evidence_and_retains_representative_chunks() -> None:
    sparse = StubRetriever(["a1", "a2", "b1"])
    dense = StubRetriever(["b1", "a2", "a3"])
    retriever = ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("sparse", sparse, 1.0),
            WeightedRetrieverSource("dense", dense, 1.0),
        ),
        parent_by_chunk_id={"a1": "A", "a2": "A", "a3": "A", "b1": "B"},
        source_candidate_k=3,
        rrf_k=10,
        aggregation="capped_top_2_sum",
        representative_chunks_per_parent=2,
    )

    results = retriever.search_with_chunks("query", top_k=2)

    assert [result.parent_id for result in results] == ["A", "B"]
    assert results[0].representative_chunk_ids == ("a2", "a1")


def test_parent_candidate_retriever_expands_only_representative_chunks() -> None:
    parent_retriever = ParentWeightedRRFHybrid(
        sources=(WeightedRetrieverSource("source", StubRetriever(["a1", "a2", "b1"])),),
        parent_by_chunk_id={"a1": "A", "a2": "A", "b1": "B"},
        source_candidate_k=3,
        rrf_k=10,
        aggregation="capped_top_2_sum",
        representative_chunks_per_parent=2,
    )
    retriever = ParentCandidateChunkRetriever(
        parent_retriever,
        parent_candidate_k=2,
        chunks_per_parent=2,
    )

    results = retriever.search("query", top_k=4)

    assert [result.doc_id for result in results] == ["a1", "a2", "b1"]
