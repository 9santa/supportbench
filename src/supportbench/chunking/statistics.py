from dataclasses import dataclass
from math import ceil
from statistics import fmean, median


@dataclass(frozen=True, slots=True)
class ChunkingStatistics:
    document_count: int
    documents_with_chunks: int
    documents_without_chunks: int

    total_chunks: int

    mean_chunks_per_document: float
    median_chunks_per_document: float
    p95_chunks_per_document: float

    mean_tokens_per_chunk: float
    median_tokens_per_chunk: float
    p95_tokens_per_chunk: float
    max_tokens_per_chunk: int

    chunks_under_50_tokens: int
    chunks_under_50_tokens_rate: float

    mean_formatted_tokens_per_chunk: float
    p95_formatted_tokens_per_chunk: float
    max_formatted_tokens_per_chunk: int

    formatted_over_budget_chunks: int
    formatted_over_budget_rate: float  # over encoder limit with added title, section etc

    indexable_empty_chunks: int


def build_chunking_statistics(
    *,
    chunks_per_document: list[int],
    body_token_counts: list[int],
    formatted_token_counts: list[int],
    max_input_tokens: int,
    special_token_reserve: int,
    indexable_empty_chunks: int,
) -> ChunkingStatistics:
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    if special_token_reserve < 0:
        raise ValueError("special_token_reserve must be non-negative")

    if special_token_reserve >= max_input_tokens:
        raise ValueError("special_token_reserve must be smaller than max_input_tokens")

    if len(body_token_counts) != len(formatted_token_counts):
        raise ValueError("body and formatted token counts must have the same length")

    document_count = len(chunks_per_document)
    total_chunks = len(body_token_counts)

    documents_with_chunks = sum(count > 0 for count in chunks_per_document)

    documents_without_chunks = document_count - documents_with_chunks

    chunks_under_50_tokens = sum(count < 50 for count in body_token_counts)

    usable_input_budget = max_input_tokens - special_token_reserve

    formatted_over_budget_chunks = sum(
        count > usable_input_budget for count in formatted_token_counts
    )

    return ChunkingStatistics(
        document_count=document_count,
        documents_with_chunks=(documents_with_chunks),
        documents_without_chunks=(documents_without_chunks),
        total_chunks=total_chunks,
        mean_chunks_per_document=_mean(chunks_per_document),
        median_chunks_per_document=_median(chunks_per_document),
        p95_chunks_per_document=_percentile(
            chunks_per_document,
            0.95,
        ),
        mean_tokens_per_chunk=_mean(body_token_counts),
        median_tokens_per_chunk=_median(body_token_counts),
        p95_tokens_per_chunk=_percentile(
            body_token_counts,
            0.95,
        ),
        max_tokens_per_chunk=max(
            body_token_counts,
            default=0,
        ),
        chunks_under_50_tokens=(chunks_under_50_tokens),
        chunks_under_50_tokens_rate=_rate(
            chunks_under_50_tokens,
            total_chunks,
        ),
        mean_formatted_tokens_per_chunk=_mean(formatted_token_counts),
        p95_formatted_tokens_per_chunk=(
            _percentile(
                formatted_token_counts,
                0.95,
            )
        ),
        max_formatted_tokens_per_chunk=max(
            formatted_token_counts,
            default=0,
        ),
        formatted_over_budget_chunks=(formatted_over_budget_chunks),
        formatted_over_budget_rate=_rate(
            formatted_over_budget_chunks,
            total_chunks,
        ),
        indexable_empty_chunks=(indexable_empty_chunks),
    )


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0

    return float(fmean(values))


def _median(values: list[int]) -> float:
    if not values:
        return 0.0

    return float(median(values))


def _percentile(
    values: list[int],
    percentile: float,
) -> float:
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")

    if not values:
        return 0.0

    ordered = sorted(values)

    # Nearest-rank percentile.
    index = max(
        0,
        ceil(percentile * len(ordered)) - 1,
    )

    return float(ordered[index])


def _rate(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return count / total
