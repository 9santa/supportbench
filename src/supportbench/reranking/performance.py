from dataclasses import dataclass

from supportbench.retrieval.base import SearchResult


@dataclass(frozen=True, slots=True)
class RerankingSearchMetrics:
    candidate_count: int

    candidate_retrieval_seconds: float
    reranking_seconds: float
    total_seconds: float

    allocated_before_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    reranking_allocated_before_bytes: int
    reranking_peak_allocated_bytes: int
    reranking_peak_reserved_bytes: int
    reranking_incremental_peak_bytes: int

    @property
    def pairs_per_second(self) -> float:
        if self.reranking_seconds <= 0.0:
            return 0.0

        return self.candidate_count / self.reranking_seconds


@dataclass(frozen=True, slots=True)
class RerankingSearchResponse:
    results: tuple[SearchResult, ...]
    metrics: RerankingSearchMetrics
