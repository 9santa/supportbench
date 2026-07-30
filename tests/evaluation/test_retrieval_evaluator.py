import pytest

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import evaluate_retriever
from supportbench.retrieval.base import SearchResult


class RecordingRetriever:
    def __init__(self) -> None:
        self.requested_top_k: int | None = None

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.requested_top_k = top_k
        return [
            SearchResult(
                doc_id=f"doc_{rank}",
                score=1.0 / rank,
                rank=rank,
            )
            for rank in range(1, top_k + 1)
        ]


class EmptyRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return []


def test_evaluation_passes_top_k_to_retriever() -> None:
    retriever = RecordingRetriever()
    queries = [
        QueryExample(
            query_id="query_1",
            query="find the last document",
            relevant_doc_ids=("doc_10",),
            split="dev",
        )
    ]

    result = evaluate_retriever(retriever, queries, top_k=50)

    assert retriever.requested_top_k == 50
    assert len(result.queries[0].retrieved_doc_ids) == 50
    assert result.recall_at_50 == 1.0


def test_rejects_recall_cutoff_above_top_k() -> None:
    query = QueryExample(
        query_id="q1",
        query="example",
        relevant_doc_ids=("doc1",),
        split="dev",
    )

    with pytest.raises(
        ValueError,
        match="largest recall cutoff",
    ):
        evaluate_retriever(
            EmptyRetriever(),
            [query],
            top_k=10,
        )


def test_accepts_top_k_covering_all_cutoffs() -> None:
    query = QueryExample(
        query_id="q1",
        query="example",
        relevant_doc_ids=("doc1",),
        split="dev",
    )

    result = evaluate_retriever(
        EmptyRetriever(),
        [query],
        top_k=50,
    )

    assert result.query_count == 1
    assert result.labeled_query_count == 1
