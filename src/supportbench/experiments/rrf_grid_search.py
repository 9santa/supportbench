import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.experiments.rrf_grid_config import (
    RRFGridDefinition,
    RRFGridPoint,
)
from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)
from supportbench.retrieval.hybrid import (
    WeightedRetrieverSource,
    WeightedRRFHybrid,
)
from supportbench.retrieval.cached import (
    CachedRetriever,
    cache_retriever_results,
)

GRID_RECALL_CUTOFFS = (1, 3, 5, 10, 20, 50)
GRID_EVALUATION_TOP_K = max(GRID_RECALL_CUTOFFS)
STANDALONE_MRR_CUTOFF = 10


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    query_count: int
    labeled_query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    mrr: float

    @classmethod
    def from_evaluation(
        cls,
        result: RetrievalEvaluationResult,
    ) -> Self:
        return cls(
            query_count=result.query_count,
            labeled_query_count=sum(bool(query.relevant_doc_ids) for query in result.queries),
            recall_at_1=result.recall_at_1,
            recall_at_3=result.recall_at_3,
            recall_at_5=result.recall_at_5,
            recall_at_10=result.recall_at_10,
            recall_at_20=_mean_recall_at_k(result, k=20),
            recall_at_50=_mean_recall_at_k(result, k=50),
            mrr=_mean_reciprocal_rank_at_k(result, k=STANDALONE_MRR_CUTOFF),
        )


@dataclass(frozen=True, slots=True)
class RetrievalMetricDelta:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    mrr: float

    @classmethod
    def between(
        cls,
        candidate: RetrievalMetrics,
        baseline: RetrievalMetrics,
    ) -> Self:
        return cls(
            recall_at_1=(candidate.recall_at_1 - baseline.recall_at_1),
            recall_at_3=(candidate.recall_at_3 - baseline.recall_at_3),
            recall_at_5=(candidate.recall_at_5 - baseline.recall_at_5),
            recall_at_10=(candidate.recall_at_10 - baseline.recall_at_10),
            recall_at_20=(candidate.recall_at_20 - baseline.recall_at_20),
            recall_at_50=(candidate.recall_at_50 - baseline.recall_at_50),
            mrr=candidate.mrr - baseline.mrr,
        )


@dataclass(frozen=True, slots=True)
class RelevantDocumentComparison:
    queries_improved: int
    queries_degraded: int
    queries_tied: int
    relevant_documents_gained: int
    relevant_documents_lost: int


@dataclass(frozen=True, slots=True)
class DenseComparisonStats:
    better_by_rr: int
    worse_by_rr: int
    tied_by_rr: int

    dense_rank_1_count: int
    dense_rank_1_preserved: int
    dense_rank_1_degraded: int

    hybrid_only_hit_at_3: int
    dense_only_hit_at_3: int

    hybrid_only_hit_at_5: int
    dense_only_hit_at_5: int

    hybrid_only_hit_at_10: int
    dense_only_hit_at_10: int

    relevant_documents_at_1: RelevantDocumentComparison
    relevant_documents_at_3: RelevantDocumentComparison
    relevant_documents_at_5: RelevantDocumentComparison
    relevant_documents_at_10: RelevantDocumentComparison
    relevant_documents_at_20: RelevantDocumentComparison
    relevant_documents_at_50: RelevantDocumentComparison

    def relevant_documents_at(self, k: int) -> RelevantDocumentComparison:
        comparisons = {
            1: self.relevant_documents_at_1,
            3: self.relevant_documents_at_3,
            5: self.relevant_documents_at_5,
            10: self.relevant_documents_at_10,
            20: self.relevant_documents_at_20,
            50: self.relevant_documents_at_50,
        }

        try:
            return comparisons[k]
        except KeyError as error:
            raise ValueError(f"unsupported comparison cutoff: {k}") from error


@dataclass(frozen=True, slots=True)
class RRFGridRun:
    config: RRFGridPoint
    evaluation: RetrievalEvaluationResult
    metrics: RetrievalMetrics
    delta_vs_dense: RetrievalMetricDelta
    comparison_vs_dense: DenseComparisonStats


@dataclass(frozen=True, slots=True)
class RRFGridSearchResult:
    definition: RRFGridDefinition
    bm25_baseline: RetrievalEvaluationResult
    dense_baseline: RetrievalEvaluationResult
    runs: tuple[RRFGridRun, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("grid search must contain at least one run")

        expected_query_count = self.dense_baseline.query_count
        query_counts = {
            self.bm25_baseline.query_count,
            *(run.metrics.query_count for run in self.runs),
        }

        if query_counts != {expected_query_count}:
            raise ValueError("all grid evaluations must contain the same number of queries")

    @property
    def best_standalone(self) -> RRFGridRun:
        return max(
            self.runs,
            key=_standalone_sort_key,
        )

    @property
    def best_candidate(self) -> RRFGridRun:
        return max(
            self.runs,
            key=_candidate_sort_key,
        )

    @property
    def pareto_config_names(
        self,
    ) -> frozenset[str]:
        pareto_names = {
            run.config.name
            for run in self.runs
            if not any(_dominates(other, run) for other in self.runs if other is not run)
        }

        return frozenset(pareto_names)


def run_rrf_grid_search(
    *,
    queries: Sequence[QueryExample],
    bm25: Retriever,
    dense: Retriever,
    definition: RRFGridDefinition,
) -> RRFGridSearchResult:
    query_items = list(queries)

    if not query_items:
        raise ValueError("queries must not be empty")

    evaluation_top_k = max(definition.final_top_k, GRID_EVALUATION_TOP_K)
    cache_top_k = max(definition.max_candidate_k, evaluation_top_k)

    cached_bm25 = cache_retriever_results(
        bm25,
        query_items,
        top_k=cache_top_k,
    )
    cached_dense = cache_retriever_results(
        dense,
        query_items,
        top_k=cache_top_k,
    )

    bm25_baseline = evaluate_retriever(
        cached_bm25,
        query_items,
        top_k=evaluation_top_k,
    )
    dense_baseline = evaluate_retriever(
        cached_dense,
        query_items,
        top_k=evaluation_top_k,
    )
    dense_metrics = RetrievalMetrics.from_evaluation(dense_baseline)

    runs: list[RRFGridRun] = []

    for config in definition.points:
        hybrid = WeightedRRFHybrid(
            sources=(
                WeightedRetrieverSource(
                    name="bm25",
                    retriever=cached_bm25,
                    weight=config.bm25_weight,
                ),
                WeightedRetrieverSource(
                    name="dense",
                    retriever=cached_dense,
                    weight=config.dense_weight,
                ),
            ),
            candidate_k=config.candidate_k,
            rrf_k=config.rrf_k,
        )

        evaluation = evaluate_retriever(
            hybrid,
            query_items,
            top_k=evaluation_top_k,
        )
        metrics = RetrievalMetrics.from_evaluation(evaluation)

        runs.append(
            RRFGridRun(
                config=config,
                evaluation=evaluation,
                metrics=metrics,
                delta_vs_dense=(
                    RetrievalMetricDelta.between(
                        metrics,
                        dense_metrics,
                    )
                ),
                comparison_vs_dense=(
                    compare_with_dense(
                        hybrid=evaluation,
                        dense=dense_baseline,
                    )
                ),
            )
        )

    return RRFGridSearchResult(
        definition=definition,
        bm25_baseline=bm25_baseline,
        dense_baseline=dense_baseline,
        runs=tuple(runs),
    )


def compare_with_dense(
    *,
    hybrid: RetrievalEvaluationResult,
    dense: RetrievalEvaluationResult,
) -> DenseComparisonStats:
    hybrid_queries = {item.query_id: item for item in hybrid.queries}
    dense_queries = {item.query_id: item for item in dense.queries}

    if set(hybrid_queries) != set(dense_queries):
        raise ValueError("hybrid and dense evaluations must contain the same query IDs")

    better_by_rr = 0
    worse_by_rr = 0
    tied_by_rr = 0

    dense_rank_1_count = 0
    dense_rank_1_preserved = 0
    dense_rank_1_degraded = 0

    hybrid_only_hit_at_3 = 0
    dense_only_hit_at_3 = 0

    hybrid_only_hit_at_5 = 0
    dense_only_hit_at_5 = 0

    hybrid_only_hit_at_10 = 0
    dense_only_hit_at_10 = 0

    for query_id, dense_query in dense_queries.items():
        hybrid_query = hybrid_queries[query_id]

        if hybrid_query.relevant_doc_ids != dense_query.relevant_doc_ids:
            raise ValueError(f"relevant document IDs differ for query {query_id!r}")

        hybrid_reciprocal_rank = _reciprocal_rank_at_k(
            hybrid_query,
            k=STANDALONE_MRR_CUTOFF,
        )
        dense_reciprocal_rank = _reciprocal_rank_at_k(
            dense_query,
            k=STANDALONE_MRR_CUTOFF,
        )

        if math.isclose(
            hybrid_reciprocal_rank,
            dense_reciprocal_rank,
            abs_tol=1e-12,
        ):
            tied_by_rr += 1
        elif hybrid_reciprocal_rank > dense_reciprocal_rank:
            better_by_rr += 1
        else:
            worse_by_rr += 1

        if dense_query.first_relevant_rank == 1:
            dense_rank_1_count += 1

            if hybrid_query.first_relevant_rank == 1:
                dense_rank_1_preserved += 1
            else:
                dense_rank_1_degraded += 1

        hybrid_hit_3 = _has_hit(
            hybrid_query,
            k=3,
        )
        dense_hit_3 = _has_hit(
            dense_query,
            k=3,
        )

        hybrid_hit_5 = _has_hit(
            hybrid_query,
            k=5,
        )
        dense_hit_5 = _has_hit(
            dense_query,
            k=5,
        )

        hybrid_hit_10 = _has_hit(
            hybrid_query,
            k=10,
        )
        dense_hit_10 = _has_hit(
            dense_query,
            k=10,
        )

        if hybrid_hit_3 and not dense_hit_3:
            hybrid_only_hit_at_3 += 1
        elif dense_hit_3 and not hybrid_hit_3:
            dense_only_hit_at_3 += 1

        if hybrid_hit_5 and not dense_hit_5:
            hybrid_only_hit_at_5 += 1
        elif dense_hit_5 and not hybrid_hit_5:
            dense_only_hit_at_5 += 1

        if hybrid_hit_10 and not dense_hit_10:
            hybrid_only_hit_at_10 += 1
        elif dense_hit_10 and not hybrid_hit_10:
            dense_only_hit_at_10 += 1

    return DenseComparisonStats(
        better_by_rr=better_by_rr,
        worse_by_rr=worse_by_rr,
        tied_by_rr=tied_by_rr,
        dense_rank_1_count=dense_rank_1_count,
        dense_rank_1_preserved=(dense_rank_1_preserved),
        dense_rank_1_degraded=(dense_rank_1_degraded),
        hybrid_only_hit_at_3=(hybrid_only_hit_at_3),
        dense_only_hit_at_3=dense_only_hit_at_3,
        hybrid_only_hit_at_5=(hybrid_only_hit_at_5),
        dense_only_hit_at_5=dense_only_hit_at_5,
        hybrid_only_hit_at_10=(hybrid_only_hit_at_10),
        dense_only_hit_at_10=(dense_only_hit_at_10),
        relevant_documents_at_1=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=1,
        ),
        relevant_documents_at_3=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=3,
        ),
        relevant_documents_at_5=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=5,
        ),
        relevant_documents_at_10=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=10,
        ),
        relevant_documents_at_20=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=20,
        ),
        relevant_documents_at_50=_compare_relevant_documents(
            hybrid_queries,
            dense_queries,
            k=50,
        ),
    )


def relevant_documents_at_k(
    evaluation: QueryEvaluation,
    *,
    k: int,
) -> set[str]:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(evaluation.relevant_doc_ids)
    retrieved = set(evaluation.retrieved_doc_ids[:k])

    return relevant & retrieved


def _compare_relevant_documents(
    hybrid_queries: Mapping[str, QueryEvaluation],
    dense_queries: Mapping[str, QueryEvaluation],
    *,
    k: int,
) -> RelevantDocumentComparison:
    queries_improved = 0
    queries_degraded = 0
    queries_tied = 0
    relevant_documents_gained = 0
    relevant_documents_lost = 0

    for query_id, dense_query in dense_queries.items():
        hybrid_query = hybrid_queries[query_id]
        dense_relevant = relevant_documents_at_k(dense_query, k=k)
        hybrid_relevant = relevant_documents_at_k(hybrid_query, k=k)

        relevant_documents_gained += len(hybrid_relevant - dense_relevant)
        relevant_documents_lost += len(dense_relevant - hybrid_relevant)

        if len(hybrid_relevant) > len(dense_relevant):
            queries_improved += 1
        elif len(hybrid_relevant) < len(dense_relevant):
            queries_degraded += 1
        else:
            queries_tied += 1

    return RelevantDocumentComparison(
        queries_improved=queries_improved,
        queries_degraded=queries_degraded,
        queries_tied=queries_tied,
        relevant_documents_gained=relevant_documents_gained,
        relevant_documents_lost=relevant_documents_lost,
    )


def _has_hit(
    evaluation: QueryEvaluation,
    *,
    k: int,
) -> bool:
    relevant = set(evaluation.relevant_doc_ids)

    return any(doc_id in relevant for doc_id in (evaluation.retrieved_doc_ids[:k]))


def _standalone_sort_key(
    run: RRFGridRun,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    float,
]:
    metrics = run.metrics

    return (
        metrics.mrr,
        metrics.recall_at_1,
        metrics.recall_at_3,
        metrics.recall_at_5,
        metrics.recall_at_10,
        -run.config.candidate_k,
        -run.config.rrf_k,
        -run.config.dense_weight,
    )


def _candidate_sort_key(
    run: RRFGridRun,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    int,
    float,
]:
    metrics = run.metrics

    return (
        metrics.recall_at_50,
        metrics.recall_at_20,
        metrics.recall_at_10,
        metrics.recall_at_5,
        metrics.recall_at_3,
        metrics.mrr,
        metrics.recall_at_1,
        -run.config.candidate_k,
        -run.config.rrf_k,
        -run.config.dense_weight,
    )


def _dominates(
    candidate: RRFGridRun,
    other: RRFGridRun,
) -> bool:
    candidate_metrics = (
        candidate.metrics.recall_at_1,
        candidate.metrics.recall_at_3,
        candidate.metrics.recall_at_5,
        candidate.metrics.recall_at_10,
        candidate.metrics.recall_at_20,
        candidate.metrics.recall_at_50,
        candidate.metrics.mrr,
    )
    other_metrics = (
        other.metrics.recall_at_1,
        other.metrics.recall_at_3,
        other.metrics.recall_at_5,
        other.metrics.recall_at_10,
        other.metrics.recall_at_20,
        other.metrics.recall_at_50,
        other.metrics.mrr,
    )

    at_least_as_good = all(
        candidate_value >= other_value
        for candidate_value, other_value in zip(
            candidate_metrics,
            other_metrics,
            strict=True,
        )
    )

    strictly_better = any(
        candidate_value > other_value
        for candidate_value, other_value in zip(
            candidate_metrics,
            other_metrics,
            strict=True,
        )
    )

    return at_least_as_good and strictly_better


def _mean_recall_at_k(
    result: RetrievalEvaluationResult,
    *,
    k: int,
) -> float:
    if not result.queries:
        return 0.0

    total = sum(
        (
            len(relevant_documents_at_k(query, k=k)) / len(query.relevant_doc_ids)
            if query.relevant_doc_ids
            else 0.0
        )
        for query in result.queries
    )

    return total / result.query_count


def _reciprocal_rank_at_k(
    evaluation: QueryEvaluation,
    *,
    k: int,
) -> float:
    relevant = set(evaluation.relevant_doc_ids)

    for rank, doc_id in enumerate(evaluation.retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def _mean_reciprocal_rank_at_k(
    result: RetrievalEvaluationResult,
    *,
    k: int,
) -> float:
    if not result.queries:
        return 0.0

    return sum(_reciprocal_rank_at_k(query, k=k) for query in result.queries) / result.query_count
