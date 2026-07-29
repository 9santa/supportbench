import math
from collections.abc import Sequence
from dataclasses import dataclass

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.reranking.factory import (
    RerankingFactory,
)
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.cached import (
    cache_retriever_results,
)
from supportbench.retrieval.hybrid import (
    WeightedRetrieverSource,
    WeightedRRFHybrid,
)


@dataclass(frozen=True, slots=True)
class RRFProfile:
    name: str
    bm25_weight: float
    dense_weight: float
    rrf_k: int
    candidate_k: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RRF profile name must be non-empty")

        for name, weight in (
            ("bm25_weight", self.bm25_weight),
            ("dense_weight", self.dense_weight),
        ):
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")


@dataclass(frozen=True, slots=True)
class PipelineEvaluation:
    name: str
    candidate_evaluation: RetrievalEvaluationResult
    reranked_evaluation: RetrievalEvaluationResult


@dataclass(frozen=True, slots=True)
class RerankerComparisonResult:
    query_count: int
    reranker_candidate_k: int
    final_top_k: int
    pipelines: tuple[
        PipelineEvaluation,
        ...,
    ]


DEFAULT_STANDALONE_RRF_PROFILE = RRFProfile(
    name="rrf_standalone",
    bm25_weight=1.0,
    dense_weight=3.0,
    rrf_k=10,
    candidate_k=100,
)

DEFAULT_CANDIDATE_RRF_PROFILE = RRFProfile(
    name="rrf_candidate",
    bm25_weight=1.0,
    dense_weight=1.5,
    rrf_k=10,
    candidate_k=100,
)


def run_reranker_comparison(
    *,
    queries: Sequence[QueryExample],
    bm25: Retriever,
    dense: Retriever,
    reranking_factory: RerankingFactory,
    standalone_profile: RRFProfile = (DEFAULT_STANDALONE_RRF_PROFILE),
    candidate_profile: RRFProfile = (DEFAULT_CANDIDATE_RRF_PROFILE),
    reranker_candidate_k: int = 50,
    final_top_k: int = 10,
    mrr_cutoff: int = 10,
) -> RerankerComparisonResult:
    query_items = tuple(queries)

    if not query_items:
        raise ValueError("queries must not be empty")

    if reranker_candidate_k < 10:
        raise ValueError("reranker_candidate_k must be at least 10")

    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")

    if final_top_k > reranker_candidate_k:
        raise ValueError("final_top_k must not be greater than reranker_candidate_k")

    max_base_candidate_k = max(
        standalone_profile.candidate_k,
        candidate_profile.candidate_k,
        reranker_candidate_k,
    )

    # Real BM25 and Dense are executed only once
    # per distinct query text.
    cached_bm25 = cache_retriever_results(
        bm25,
        query_items,
        top_k=max_base_candidate_k,
    )
    cached_dense = cache_retriever_results(
        dense,
        query_items,
        top_k=max_base_candidate_k,
    )

    standalone_rrf = _create_rrf(
        profile=standalone_profile,
        bm25=cached_bm25,
        dense=cached_dense,
    )

    candidate_rrf = _create_rrf(
        profile=candidate_profile,
        bm25=cached_bm25,
        dense=cached_dense,
    )

    candidate_sources: tuple[
        tuple[str, Retriever],
        ...,
    ] = (
        ("dense", cached_dense),
        (
            standalone_profile.name,
            standalone_rrf,
        ),
        (
            candidate_profile.name,
            candidate_rrf,
        ),
    )

    pipeline_results: list[PipelineEvaluation] = []

    for source_name, source in candidate_sources:
        # Cache retrieved top-50 pool once
        cached_source = cache_retriever_results(
            source,
            query_items,
            top_k=reranker_candidate_k,
        )

        candidate_evaluation = evaluate_retriever(
            cached_source,
            query_items,
            top_k=reranker_candidate_k,
            recall_cutoffs=(
                1,
                3,
                5,
                10,
                20,
                50,
            ),
            mrr_cutoff=mrr_cutoff,
        )

        reranking_retriever = reranking_factory.create(
            candidate_retriever=(cached_source),
            candidate_k=(reranker_candidate_k),
        )

        reranked_evaluation = evaluate_retriever(
            reranking_retriever,
            query_items,
            top_k=final_top_k,
            recall_cutoffs=(
                1,
                3,
                5,
                10,
            ),
            mrr_cutoff=mrr_cutoff,
        )

        pipeline_results.append(
            PipelineEvaluation(
                name=source_name,
                candidate_evaluation=(candidate_evaluation),
                reranked_evaluation=(reranked_evaluation),
            )
        )

    return RerankerComparisonResult(
        query_count=len(query_items),
        reranker_candidate_k=(reranker_candidate_k),
        final_top_k=final_top_k,
        pipelines=tuple(pipeline_results),
    )


def _create_rrf(
    *,
    profile: RRFProfile,
    bm25: Retriever,
    dense: Retriever,
) -> WeightedRRFHybrid:
    return WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                name="bm25",
                retriever=bm25,
                weight=profile.bm25_weight,
            ),
            WeightedRetrieverSource(
                name="dense",
                retriever=dense,
                weight=profile.dense_weight,
            ),
        ),
        candidate_k=profile.candidate_k,
        rrf_k=profile.rrf_k,
    )
