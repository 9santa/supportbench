from collections.abc import Mapping
from dataclasses import dataclass

from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
)


@dataclass(frozen=True, slots=True)
class NamedQueryEvaluation:
    retriever_name: str
    evaluation: QueryEvaluation


@dataclass(frozen=True, slots=True)
class QueryComparison:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    evaluations: tuple[NamedQueryEvaluation, ...]

    def successful_retrievers(self, *, k: int) -> tuple[str, ...]:
        return tuple(
            item.retriever_name for item in self.evaluations if succeeds_at_k(item.evaluation, k=k)
        )

    def failed_retrievers(self, *, k: int) -> tuple[str, ...]:
        return tuple(
            item.retriever_name
            for item in self.evaluations
            if not succeeds_at_k(item.evaluation, k=k)
        )

    def all_succeeded(self, *, k: int) -> bool:
        return not self.failed_retrievers(k=k)

    def all_failed(self, *, k: int) -> bool:
        return not self.successful_retrievers(k=k)

    @property
    def best_retrievers(self) -> tuple[str, ...]:
        best_score = max(item.evaluation.reciprocal_rank for item in self.evaluations)

        if best_score == 0.0:
            return ()

        return tuple(
            item.retriever_name
            for item in self.evaluations
            if (item.evaluation.reciprocal_rank == best_score)
        )


def succeeds_at_k(evaluation: QueryEvaluation, *, k: int) -> bool:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_doc_ids = set(evaluation.relevant_doc_ids)

    return any(doc_id in relevant_doc_ids for doc_id in evaluation.retrieved_doc_ids[:k])


def failures_at_k(
    result: RetrievalEvaluationResult,
    *,
    k: int,
) -> tuple[QueryEvaluation, ...]:
    return tuple(
        evaluation
        for evaluation in result.queries
        if evaluation.is_labeled and not succeeds_at_k(evaluation, k=k)
    )


def compare_evaluation_results(
    results: Mapping[str, RetrievalEvaluationResult],
) -> tuple[QueryComparison, ...]:
    if len(results) < 2:
        raise ValueError("at least two retrievers are required for comparison")

    # Convert items to a list so we can refer to the first one easily
    retriever_items = list(results.items())

    first_name, first_result = retriever_items[0]

    # Mapping: retriever_name -> {query_id -> QueryEvaluation}
    query_id_to_query = {
        retriever_name: {q.query_id: q for q in result.queries}
        for retriever_name, result in retriever_items
    }

    # All retrievers must have exactly the same set of evaluated query ids
    expected_ids = set(query_id_to_query[first_name])
    for name, mapping in query_id_to_query.items():
        if set(mapping) != expected_ids:
            raise ValueError(f"Query ID mismatch between {first_name!r} and {name!r} retrievers")

    # Build comparisons, using the frist retriever's order and reference data
    comparisons = []
    for ref_query in first_result.queries:
        qid = ref_query.query_id
        evaluations = []

        for name, _ in retriever_items:
            query_obj = query_id_to_query[name][qid]
            _validate_same_query(ref_query, query_obj, retriever_name=name)
            evaluations.append(
                NamedQueryEvaluation(
                    retriever_name=name,
                    evaluation=query_obj,
                )
            )

        comparisons.append(
            QueryComparison(
                query_id=qid,
                query=ref_query.query,
                relevant_doc_ids=ref_query.relevant_doc_ids,
                evaluations=tuple(evaluations),
            )
        )

    return tuple(comparisons)


def _validate_same_query(
    reference: QueryEvaluation,
    candidate: QueryEvaluation,
    *,
    retriever_name: str,
) -> None:
    if candidate.query != reference.query:
        raise ValueError(f"query text differs for {reference.query_id!r} in {retriever_name!r}")

    if candidate.relevant_doc_ids != reference.relevant_doc_ids:
        raise ValueError(f"gold labels differ for {reference.query_id!r} in {retriever_name!r}")
