import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from supportbench.data.models import QueryExample
from supportbench.retrieval.base import Retriever


@dataclass(frozen=True, slots=True)
class BeirQueryEvaluation:
    query_id: str
    retrieved_doc_ids: tuple[str, ...]
    scores: tuple[float, ...]
    ndcg: tuple[tuple[int, float], ...]
    average_precision: tuple[tuple[int, float], ...]
    recall: tuple[tuple[int, float], ...]
    precision: tuple[tuple[int, float], ...]
    reciprocal_rank: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class BeirEvaluationResult:
    query_count: int
    top_k: int
    cutoffs: tuple[int, ...]
    ndcg: tuple[tuple[int, float], ...]
    mean_average_precision: tuple[tuple[int, float], ...]
    recall: tuple[tuple[int, float], ...]
    precision: tuple[tuple[int, float], ...]
    mrr: tuple[tuple[int, float], ...]
    queries: tuple[BeirQueryEvaluation, ...]


def evaluate_beir_retriever(
    retriever: Retriever,
    queries: Sequence[QueryExample],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    top_k: int,
    cutoffs: Sequence[int],
    ignore_identical_ids: bool = True,
) -> BeirEvaluationResult:
    validated_cutoffs = _validate_configuration(top_k=top_k, cutoffs=cutoffs)
    query_evaluations: list[BeirQueryEvaluation] = []

    for query in queries:
        relevance_by_document = qrels.get(query.query_id)

        if relevance_by_document is None:
            raise ValueError(f"missing qrels for query {query.query_id!r}")

        results = retriever.search(query.query, top_k=top_k)

        if ignore_identical_ids:
            results = [result for result in results if result.doc_id != query.query_id]

        results = results[:top_k]
        retrieved_document_ids = tuple(result.doc_id for result in results)
        scores = tuple(result.score for result in results)
        query_evaluations.append(
            _evaluate_query(
                query.query_id,
                retrieved_document_ids,
                scores,
                relevance_by_document,
                validated_cutoffs,
            )
        )

    query_items = tuple(query_evaluations)

    return BeirEvaluationResult(
        query_count=len(query_items),
        top_k=top_k,
        cutoffs=validated_cutoffs,
        ndcg=_mean_metrics(query_items, "ndcg", validated_cutoffs),
        mean_average_precision=_mean_metrics(
            query_items,
            "average_precision",
            validated_cutoffs,
        ),
        recall=_mean_metrics(query_items, "recall", validated_cutoffs),
        precision=_mean_metrics(query_items, "precision", validated_cutoffs),
        mrr=_mean_metrics(query_items, "reciprocal_rank", validated_cutoffs),
        queries=query_items,
    )


def _evaluate_query(
    query_id: str,
    retrieved_document_ids: tuple[str, ...],
    scores: tuple[float, ...],
    relevance_by_document: Mapping[str, int],
    cutoffs: tuple[int, ...],
) -> BeirQueryEvaluation:
    positive_relevances = {
        document_id: relevance
        for document_id, relevance in relevance_by_document.items()
        if relevance > 0
    }

    if not positive_relevances:
        raise ValueError(f"query {query_id!r} has no positive relevance judgments")

    return BeirQueryEvaluation(
        query_id=query_id,
        retrieved_doc_ids=retrieved_document_ids,
        scores=scores,
        ndcg=tuple(
            (cutoff, _ndcg_at_k(retrieved_document_ids, positive_relevances, cutoff))
            for cutoff in cutoffs
        ),
        average_precision=tuple(
            (cutoff, _average_precision_at_k(retrieved_document_ids, positive_relevances, cutoff))
            for cutoff in cutoffs
        ),
        recall=tuple(
            (cutoff, _recall_at_k(retrieved_document_ids, positive_relevances, cutoff))
            for cutoff in cutoffs
        ),
        precision=tuple(
            (cutoff, _precision_at_k(retrieved_document_ids, positive_relevances, cutoff))
            for cutoff in cutoffs
        ),
        reciprocal_rank=tuple(
            (cutoff, _reciprocal_rank_at_k(retrieved_document_ids, positive_relevances, cutoff))
            for cutoff in cutoffs
        ),
    )


def _ndcg_at_k(
    retrieved_document_ids: tuple[str, ...],
    relevance_by_document: Mapping[str, int],
    cutoff: int,
) -> float:
    seen_document_ids: set[str] = set()
    dcg = 0.0

    for rank, document_id in enumerate(retrieved_document_ids[:cutoff], start=1):
        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)
        dcg += relevance_by_document.get(document_id, 0) / math.log2(rank + 1)

    ideal_relevances = sorted(relevance_by_document.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _average_precision_at_k(
    retrieved_document_ids: tuple[str, ...],
    relevance_by_document: Mapping[str, int],
    cutoff: int,
) -> float:
    relevant_seen = 0
    precision_sum = 0.0
    seen_document_ids: set[str] = set()

    for rank, document_id in enumerate(retrieved_document_ids[:cutoff], start=1):
        if document_id in seen_document_ids or document_id not in relevance_by_document:
            continue

        seen_document_ids.add(document_id)
        relevant_seen += 1
        precision_sum += relevant_seen / rank

    denominator = min(len(relevance_by_document), cutoff)
    return precision_sum / denominator if denominator else 0.0


def _recall_at_k(
    retrieved_document_ids: tuple[str, ...],
    relevance_by_document: Mapping[str, int],
    cutoff: int,
) -> float:
    retrieved_relevant = len(set(retrieved_document_ids[:cutoff]) & set(relevance_by_document))
    return retrieved_relevant / len(relevance_by_document)


def _precision_at_k(
    retrieved_document_ids: tuple[str, ...],
    relevance_by_document: Mapping[str, int],
    cutoff: int,
) -> float:
    retrieved_relevant = len(set(retrieved_document_ids[:cutoff]) & set(relevance_by_document))
    return retrieved_relevant / cutoff


def _reciprocal_rank_at_k(
    retrieved_document_ids: tuple[str, ...],
    relevance_by_document: Mapping[str, int],
    cutoff: int,
) -> float:
    for rank, document_id in enumerate(retrieved_document_ids[:cutoff], start=1):
        if document_id in relevance_by_document:
            return 1.0 / rank

    return 0.0


def _mean_metrics(
    queries: tuple[BeirQueryEvaluation, ...],
    attribute: str,
    cutoffs: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (
            cutoff,
            sum(dict(getattr(query, attribute))[cutoff] for query in queries) / len(queries)
            if queries
            else 0.0,
        )
        for cutoff in cutoffs
    )


def _validate_configuration(*, top_k: int, cutoffs: Sequence[int]) -> tuple[int, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    values = tuple(cutoffs)

    if not values or any(cutoff <= 0 for cutoff in values):
        raise ValueError("cutoffs must contain positive values")

    if values != tuple(sorted(set(values))):
        raise ValueError("cutoffs must be unique and sorted")

    if values[-1] > top_k:
        raise ValueError("largest cutoff must not exceed top_k")

    return values
