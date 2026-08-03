import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from supportbench.data.models import QueryExample
from supportbench.reranking.performance import (
    RerankingSearchMetrics,
)
from supportbench.reranking.retriever import (
    RerankingRetriever,
)


@dataclass(frozen=True, slots=True)
class LatencySummary:
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float


@dataclass(frozen=True, slots=True)
class PipelinePerformanceSummary:
    name: str
    query_count: int
    warmup_query_count: int

    reranker_batch_size: int
    candidate_pair_count: int
    effective_batch_count: int

    candidate_retrieval_latency: LatencySummary
    reranking_latency: LatencySummary
    total_latency: LatencySummary

    pairs_per_second: float
    batches_per_second: float
    queries_per_second: float

    peak_allocated_bytes: int
    peak_reserved_bytes: int

    peak_reranking_allocated_bytes: int
    peak_reranking_reserved_bytes: int
    peak_reranking_incremental_bytes: int


def benchmark_reranking_retriever(
    *,
    name: str,
    retriever: RerankingRetriever,
    queries: Sequence[QueryExample],
    top_k: int,
    reranker_batch_size: int,
    warmup_query_count: int = 5,
    benchmark_query_count: int | None = None,
) -> PipelinePerformanceSummary:
    if not name.strip():
        raise ValueError("pipeline name must be non-empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size must be positive")

    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must not be negative")

    query_items = tuple(queries)

    if benchmark_query_count is not None:
        if benchmark_query_count <= 0:
            raise ValueError("benchmark_query_count must be positive")

        query_items = query_items[:benchmark_query_count]

    if not query_items:
        raise ValueError("benchmark queries must not be empty")

    warmup_count = min(
        warmup_query_count,
        len(query_items),
    )

    for query in query_items[:warmup_count]:
        retriever.search(
            query.query,
            top_k=top_k,
        )

    samples: list[RerankingSearchMetrics] = []

    for query in query_items:
        response = retriever.search_with_metrics(
            query.query,
            top_k=top_k,
        )

        samples.append(response.metrics)

    candidate_pair_count = sum(sample.candidate_count for sample in samples)

    effective_batch_count = sum(
        math.ceil(sample.candidate_count / reranker_batch_size)
        for sample in samples
        if sample.candidate_count > 0
    )

    total_reranking_seconds = sum(sample.reranking_seconds for sample in samples)
    total_pipeline_seconds = sum(sample.total_seconds for sample in samples)

    return PipelinePerformanceSummary(
        name=name,
        query_count=len(samples),
        warmup_query_count=warmup_count,
        reranker_batch_size=(reranker_batch_size),
        candidate_pair_count=(candidate_pair_count),
        effective_batch_count=(effective_batch_count),
        candidate_retrieval_latency=(
            _summarize_latency(sample.candidate_retrieval_seconds for sample in samples)
        ),
        reranking_latency=_summarize_latency(sample.reranking_seconds for sample in samples),
        total_latency=_summarize_latency(sample.total_seconds for sample in samples),
        pairs_per_second=_safe_rate(
            candidate_pair_count,
            total_reranking_seconds,
        ),
        batches_per_second=_safe_rate(
            effective_batch_count,
            total_reranking_seconds,
        ),
        queries_per_second=_safe_rate(
            len(samples),
            total_pipeline_seconds,
        ),
        peak_allocated_bytes=max(sample.peak_allocated_bytes for sample in samples),
        peak_reserved_bytes=max(sample.peak_reserved_bytes for sample in samples),
        peak_reranking_allocated_bytes=max(
            sample.reranking_peak_allocated_bytes for sample in samples
        ),
        peak_reranking_reserved_bytes=max(
            sample.reranking_peak_reserved_bytes for sample in samples
        ),
        peak_reranking_incremental_bytes=max(
            sample.reranking_incremental_peak_bytes for sample in samples
        ),
    )


def get_gpu_name(device: str) -> str | None:
    torch_device = torch.device(device)

    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None

    device_index = torch_device.index

    # None if single GPU
    if device_index is None:
        device_index = torch.cuda.current_device()

    return torch.cuda.get_device_name(device_index)


def _summarize_latency(
    values: Iterable[float],
) -> LatencySummary:
    items = sorted(tuple(values))

    if not items:
        return LatencySummary(
            mean_seconds=0.0,
            p50_seconds=0.0,
            p95_seconds=0.0,
        )

    return LatencySummary(
        mean_seconds=sum(items) / len(items),
        p50_seconds=_percentile(
            items,
            quantile=0.50,
        ),
        p95_seconds=_percentile(
            items,
            quantile=0.95,
        ),
    )


def _percentile(
    sorted_values: Sequence[float],
    *,
    quantile: float,
) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * quantile

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]

    if lower_index == upper_index:
        return lower

    fraction = position - lower_index

    return lower + (upper - lower) * fraction


def _safe_rate(
    count: int,
    seconds: float,
) -> float:
    if seconds <= 0.0:
        return 0.0

    return count / seconds
