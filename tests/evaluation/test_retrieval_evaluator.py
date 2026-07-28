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

    result = evaluate_retriever(retriever, queries, top_k=10)

    assert retriever.requested_top_k == 10
    assert len(result.queries[0].retrieved_doc_ids) == 10
    assert result.recall_at_10 == 1.0
