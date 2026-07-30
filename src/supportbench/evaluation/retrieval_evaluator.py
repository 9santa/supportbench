from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
)
from supportbench.retrieval.base import Retriever

DEFAULT_RECALL_CUTOFFS = (1, 3, 5, 10, 20, 50)
DEFAULT_MRR_CUTOFF = 10


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    scores: tuple[float, ...]
    is_labeled: bool

    recalls: tuple[tuple[int, float], ...]
    reciprocal_rank: float
    mrr_cutoff: int

    def recall_at(
        self,
        cutoff: int,
    ) -> float:
        return _metric_at(
            self.recalls,
            cutoff=cutoff,
            metric_name="Recall",
        )

    @property
    def recall_at_1(self) -> float:
        return self.recall_at(1)

    @property
    def recall_at_3(self) -> float:
        return self.recall_at(3)

    @property
    def recall_at_5(self) -> float:
        return self.recall_at(5)

    @property
    def recall_at_10(self) -> float:
        return self.recall_at(10)

    @property
    def recall_at_20(self) -> float:
        return self.recall_at(20)

    @property
    def recall_at_50(self) -> float:
        return self.recall_at(50)

    @property
    def first_relevant_rank(self) -> int | None:
        if not self.is_labeled:
            return None

        relevant_doc_ids = set(self.relevant_doc_ids)

        for rank, doc_id in enumerate(self.retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return rank

        return None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    query_count: int
    labeled_query_count: int
    unlabeled_query_count: int

    evaluation_top_k: int
    recall_cutoffs: tuple[int, ...]
    mrr_cutoff: int

    recalls: tuple[tuple[int, float], ...]
    mrr: float

    queries: tuple[QueryEvaluation, ...]

    def recall_at(
        self,
        cutoff: int,
    ) -> float:
        return _metric_at(
            self.recalls,
            cutoff=cutoff,
            metric_name="Recall",
        )

    @property
    def recall_at_1(self) -> float:
        return self.recall_at(1)

    @property
    def recall_at_3(self) -> float:
        return self.recall_at(3)

    @property
    def recall_at_5(self) -> float:
        return self.recall_at(5)

    @property
    def recall_at_10(self) -> float:
        return self.recall_at(10)

    @property
    def recall_at_20(self) -> float:
        return self.recall_at(20)

    @property
    def recall_at_50(self) -> float:
        return self.recall_at(50)


def evaluate_retriever(
    retriever: Retriever,
    queries: Sequence[QueryExample],
    *,
    top_k: int = 10,
    recall_cutoffs: Sequence[int] = (DEFAULT_RECALL_CUTOFFS),
    mrr_cutoff: int = DEFAULT_MRR_CUTOFF,
) -> RetrievalEvaluationResult:
    """
    Evaluate a retriever.

    All queries are searched and included in the
    returned per-query results.

    Aggregate recall and MRR are calculated only on labeled queries.
    """
    validated_cutoffs = _validate_evalution_config(
        top_k=top_k,
        recall_cutoffs=recall_cutoffs,
        mrr_cutoff=mrr_cutoff,
    )

    query_items = tuple(queries)
    query_evaluations: list[QueryEvaluation] = []

    for query in query_items:
        relevant_doc_ids = set(query.relevant_doc_ids)
        is_labeled = bool(relevant_doc_ids)

        results = retriever.search(
            query.query,
            top_k=top_k,
        )

        results = results[:top_k]

        retrieved_doc_ids = tuple(result.doc_id for result in results)
        scores = tuple(result.score for result in results)

        if is_labeled:
            recalls = tuple(
                (
                    cutoff,
                    recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=cutoff),
                )
                for cutoff in validated_cutoffs
            )

            query_reciprocal_rank = reciprocal_rank(
                retrieved_doc_ids[:mrr_cutoff],
                relevant_doc_ids,
            )
        else:
            # Unlabeled queries remain in the output,
            # but do not affect aggregate metrics.
            recalls = tuple((cutoff, 0.0) for cutoff in validated_cutoffs)
            query_reciprocal_rank = 0.0

        query_evaluations.append(
            QueryEvaluation(
                query_id=query.query_id,
                query=query.query,
                relevant_doc_ids=tuple(sorted(relevant_doc_ids)),
                retrieved_doc_ids=(retrieved_doc_ids),
                scores=scores,
                is_labeled=is_labeled,
                recalls=recalls,
                reciprocal_rank=(query_reciprocal_rank),
                mrr_cutoff=mrr_cutoff,
            )
        )

    evaluations = tuple(query_evaluations)

    labeled_evaluations = tuple(evaluation for evaluation in evaluations if evaluation.is_labeled)

    aggregate_recalls = tuple(
        (
            cutoff,
            _mean(evaluation.recall_at(cutoff) for evaluation in labeled_evaluations),
        )
        for cutoff in recall_cutoffs
    )

    aggregate_mrr = _mean(evaluation.reciprocal_rank for evaluation in labeled_evaluations)

    query_count = len(evaluations)
    labeled_query_count = len(labeled_evaluations)

    return RetrievalEvaluationResult(
        query_count=query_count,
        labeled_query_count=(labeled_query_count),
        unlabeled_query_count=(query_count - labeled_query_count),
        evaluation_top_k=top_k,
        recall_cutoffs=tuple(validated_cutoffs),
        mrr_cutoff=mrr_cutoff,
        recalls=aggregate_recalls,
        mrr=aggregate_mrr,
        queries=evaluations,
    )


def _metric_at(
    values: tuple[tuple[int, float], ...],
    *,
    cutoff: int,
    metric_name: str,
) -> float:
    for current_cutoff, value in values:
        if current_cutoff == cutoff:
            return value

    available = ", ".join(str(current_cutoff) for current_cutoff, _ in values)

    raise ValueError(
        f"{metric_name}@{cutoff} was not evaluated; available cutoffs: {available}",
    )


def _mean(values: Iterable[float]) -> float:
    # No list to save memory
    total: float = 0.0
    count: int = 0
    for v in values:
        count += 1
        total += v

    return total / count if count else 0.0

def _validate_evalution_config(
    *,
        top_k: int,
        recall_cutoffs: Sequence[int],
    mrr_cutoff: int,
) -> tuple[int, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    cutoffs = tuple(recall_cutoffs)

    if not cutoffs:
        raise ValueError(
            "recall_cutoffs must not be empty"
        )

    if any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError(
            "recall cutoffs must be positive"
        )

    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError(
            "recall cutoffs must not contain duplicates"
        )

    if cutoffs != tuple(sorted(cutoffs)):
        raise ValueError(
            "recall cutoffs must be sorted"
        )

    if cutoffs[-1] > top_k:
        raise ValueError(
            "largest recall cutoff must not exceed top_k"
        )

    if mrr_cutoff <= 0:
        raise ValueError(
            "mrr_cutoff must be positive"
        )

    if mrr_cutoff > top_k:
        raise ValueError(
            "mrr_cutoff must not exceed top_k"
        )

    return cutoffs
