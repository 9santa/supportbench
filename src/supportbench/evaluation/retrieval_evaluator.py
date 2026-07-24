from dataclasses import dataclass

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from supportbench.retrieval.base import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


def evaluate_retriever(
    retriever: Retriever, queries: list[QueryExample]
) -> RetrievalEvaluationResult:
    """Evaluate a retriever."""
    if not queries:
        return RetrievalEvaluationResult(
            query_count=0,
            recall_at_1=0,
            recall_at_3=0,
            recall_at_5=0,
            mrr=0.0,
        )

    recall_at_1_values: list[float] = []
    recall_at_3_values: list[float] = []
    recall_at_5_values: list[float] = []
    reciprocal_ranks: list[float] = []

    for query in queries:
        relevant_doc_ds = set(query.relevant_doc_ids)

        if not relevant_doc_ds:
            raise ValueError(
                f"query {query.query_id!r} must contain at least one relevant document id"
            )

        results = retriever.search(query.query, top_k=5)

        retrieved_doc_ids = [result.doc_id for result in results]

        recall_at_1_values.append(recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=1))
        recall_at_3_values.append(recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=3))
        recall_at_5_values.append(recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=5))

        reciprocal_ranks.append(reciprocal_rank(retrieved_doc_ids, relevant_doc_ds))

    query_count = len(queries)

    return RetrievalEvaluationResult(
        query_count=query_count,
        recall_at_1=sum(recall_at_1_values) / query_count,
        recall_at_3=sum(recall_at_3_values) / query_count,
        recall_at_5=sum(recall_at_5_values) / query_count,
        mrr=mean_reciprocal_rank(reciprocal_ranks),
    )
