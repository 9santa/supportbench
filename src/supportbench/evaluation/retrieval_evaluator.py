from collections.abc import Iterable
from dataclasses import dataclass

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from supportbench.retrieval.base import Retriever


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    scores: tuple[float, ...]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float

    @property
    def first_relevant_rank(self) -> int | None:
        relevant_doc_ids = self.relevant_doc_ids

        for rank, doc_id in enumerate(self.retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return rank

        return None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    queries: tuple[QueryEvaluation, ...]


def evaluate_retriever(
    retriever: Retriever,
    queries: list[QueryExample],
    *,
    top_k: int = 5,
) -> RetrievalEvaluationResult:
    """Evaluate a retriever."""
    if not queries:
        return RetrievalEvaluationResult(
            query_count=0,
            recall_at_1=0,
            recall_at_3=0,
            recall_at_5=0,
            mrr=0.0,
            queries=tuple(),
        )

    if top_k < 5:
        raise ValueError("top_k must be at least 5 to compute Recall@5")

    query_evals: list[QueryEvaluation] = []

    for query in queries:
        relevant_doc_ds = set(query.relevant_doc_ids)

        if not relevant_doc_ds:
            raise ValueError(
                f"query {query.query_id!r} must contain at least one relevant document id"
            )

        results = retriever.search(query.query, top_k=5)

        retrieved_doc_ids = [result.doc_id for result in results]

        scores = [result.score for result in results]

        recall_1 = recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=1)
        recall_3 = recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=3)
        recall_5 = recall_at_k(retrieved_doc_ids, relevant_doc_ds, k=5)
        r_rank = reciprocal_rank(retrieved_doc_ids, relevant_doc_ds)

        query_evals.append(
            QueryEvaluation(
                query_id=query.query_id,
                query=query.query,
                relevant_doc_ids=tuple(sorted(relevant_doc_ds)),
                retrieved_doc_ids=tuple(retrieved_doc_ids),
                scores=tuple(scores),
                recall_at_1=recall_1,
                recall_at_3=recall_3,
                recall_at_5=recall_5,
                reciprocal_rank=r_rank,
            )
        )

    evaluations = tuple(query_evals)
    query_count = len(evaluations)

    if not evaluations:
        return RetrievalEvaluationResult(
            query_count=0,
            recall_at_1=0,
            recall_at_3=0,
            recall_at_5=0,
            mrr=0.0,
            queries=tuple(),
        )

    return RetrievalEvaluationResult(
        query_count=query_count,
        # Passing generator to _mean()
        recall_at_1=_mean(item.recall_at_1 for item in evaluations),
        recall_at_3=_mean(item.recall_at_3 for item in evaluations),
        recall_at_5=_mean(item.recall_at_5 for item in evaluations),
        mrr=mean_reciprocal_rank([item.reciprocal_rank for item in evaluations]),
        queries=evaluations,
    )


def _mean(values: Iterable[float]) -> float:
    # No list to save memory
    total = 0
    count = 0
    for v in values:
        count += 1
        total += v

    return total / count if count else 0.0
