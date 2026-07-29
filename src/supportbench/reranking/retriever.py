from time import perf_counter
from collections.abc import Callable, Sequence

import torch

from supportbench.data.models import Document
from supportbench.reranking.base import (
    RerankCandidate,
    Reranker,
)
from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)
from supportbench.reranking.performance import (
    RerankingSearchMetrics,
    RerankingSearchResponse,
)

type DocumentFormatter = Callable[
    [Document],
    str,
]


def format_document_for_reranking(document: Document) -> str:
    return f"{document.title}\n{document.text}"


class RerankingRetriever(Retriever):
    def __init__(
        self,
        *,
        candidate_retriever: Retriever,
        reranker: Reranker,
        documents: Sequence[Document],
        candidate_k: int = 50,
        document_formatter: DocumentFormatter = format_document_for_reranking,
        performance_device: str = "cuda",
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        documents_by_id: dict[str, Document] = {}

        for document in documents:
            if document.doc_id in documents_by_id:
                raise ValueError("documents contain duplicate doc_id:", repr(document.doc_id))

            documents_by_id[document.doc_id] = document

        if not documents_by_id:
            raise ValueError("documents must not be empty")

        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._documents_by_id = documents_by_id
        self._candidate_k = candidate_k
        self._document_formatter = document_formatter
        self._performance_tracker = _CudaPerformanceTracker(performance_device)

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        response = self.search_with_metrics(
            query,
            top_k=top_k,
        )

        return list(response.results)

        # if top_k <= 0:
        #     raise ValueError("top_k must be positive")
        #
        # if top_k > self._candidate_k:
        #     raise ValueError("top_k must not be greater than candidate_k")
        #
        # if not query.strip():
        #     return []
        #
        # retrieved = self._candidate_retriever.search(
        #     query,
        #     top_k=self._candidate_k,
        # )
        #
        # candidates = self._build_candidates(retrieved)
        #
        # reranked = self._reranker.rerank(
        #     query,
        #     candidates,
        #     top_k=top_k,
        # )
        #
        # self._validate_reranker_results(
        #     reranked_doc_ids=[result.doc_id for result in reranked],
        #     candidate_doc_ids={candidate.doc_id for candidate in candidates},
        # )
        #
        # return [
        #     SearchResult(
        #         doc_id=result.doc_id,
        #         score=result.score,
        #         rank=rank,
        #     )
        #     for rank, result in enumerate(reranked, start=1)
        # ]

    def search_with_metrics(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> RerankingSearchResponse:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self._candidate_k:
            raise ValueError("top_k must not be greater than candidate_k")

        if not query.strip():
            return RerankingSearchResponse(
                results=(),
                metrics=RerankingSearchMetrics(
                    candidate_count=0,
                    candidate_retrieval_seconds=0.0,
                    reranking_seconds=0.0,
                    total_seconds=0.0,
                    allocated_before_bytes=0,
                    peak_allocated_bytes=0,
                    peak_reserved_bytes=0,
                    reranking_allocated_before_bytes=0,
                    reranking_peak_allocated_bytes=0,
                    reranking_peak_reserved_bytes=0,
                    reranking_incremental_peak_bytes=0,
                ),
            )

        tracker = self._performance_tracker

        tracker.synchronize()
        tracker.reset_peak_memory_stats()

        allocated_before = tracker.memory_allocated()

        total_started = perf_counter()

        retrieval_started = perf_counter()

        retrieved = self._candidate_retriever.search(query, top_k=self._candidate_k)

        # Dense retrieval runs on CUDA by default, so need to sync
        tracker.synchronize()

        candidate_retrieval_seconds = perf_counter() - retrieval_started
        candidate_peak_allocated = tracker.max_memory_allocated()
        candidate_peak_reserved = tracker.max_memory_reserved()
        candidates = self._build_candidates(retrieved)

        # Start a new peak-memory counter for reranking
        tracker.synchronize()

        reranking_allocated_before = tracker.memory_allocated()

        reranking_started = perf_counter()

        reranked = self._reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )

        tracker.synchronize()

        reranking_seconds = perf_counter() - retrieval_started
        reranking_peak_allocated = tracker.max_memory_allocated()
        reranking_peak_reserved = tracker.max_memory_reserved()

        self._validate_reranker_results(
            reranked_doc_ids=[result.doc_id for result in reranked],
            candidate_doc_ids={candidate.doc_id for candidate in candidates},
        )

        results = tuple(
            SearchResult(
                doc_id=result.doc_id,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(reranked, start=1)
        )

        total_seconds = perf_counter() - total_started

        peak_allocated = max(candidate_peak_allocated, reranking_peak_allocated)
        peak_reserved = max(candidate_peak_reserved, reranking_peak_reserved)

        incremental_reranking_peak = max(0, reranking_peak_allocated - reranking_allocated_before)

        return RerankingSearchResponse(
            results=results,
            metrics=RerankingSearchMetrics(
                candidate_count=len(candidates),
                candidate_retrieval_seconds=(candidate_retrieval_seconds),
                reranking_seconds=(reranking_seconds),
                total_seconds=total_seconds,
                allocated_before_bytes=(allocated_before),
                peak_allocated_bytes=(peak_allocated),
                peak_reserved_bytes=(peak_reserved),
                reranking_allocated_before_bytes=(reranking_allocated_before),
                reranking_peak_allocated_bytes=(reranking_peak_allocated),
                reranking_peak_reserved_bytes=(reranking_peak_reserved),
                reranking_incremental_peak_bytes=(incremental_reranking_peak),
            ),
        )

    def _build_candidates(
        self,
        retrieved: Sequence[SearchResult],
    ) -> tuple[RerankCandidate, ...]:
        candidates: list[RerankCandidate] = []
        seen_doc_ids: set[str] = set()

        for result in retrieved:
            if result.doc_id in seen_doc_ids:
                continue

            seen_doc_ids.add(result.doc_id)

            document = self._documents_by_id.get(result.doc_id)

            if document is None:
                raise ValueError(
                    f"candidate retriever returned an unknown document: {result.doc_id!r}"
                )

            candidates.append(
                RerankCandidate(
                    doc_id=result.doc_id,
                    text=self._document_formatter(document),
                    retrieval_score=result.score,
                    retrieval_rank=result.rank,
                )
            )

        return tuple(candidates)

    @staticmethod
    def _validate_reranker_results(
        *,
        reranked_doc_ids: Sequence[str],
        candidate_doc_ids: set[str],
    ) -> None:
        if len(reranked_doc_ids) != len(set(reranked_doc_ids)):
            raise ValueError("reranker returned duplcate document IDs")

        unknown_doc_ids = set(reranked_doc_ids) - candidate_doc_ids

        if unknown_doc_ids:
            formatted = ", ".join(sorted(unknown_doc_ids))

            raise ValueError(f"reranker returned documents outside the candidate set: {formatted}")


class _CudaPerformanceTracker:
    def __init__(self, device: str) -> None:
        self._device = torch.device(device)

        self._enabled = self._device.type == "cuda" and torch.cuda.is_available()

    def synchronize(self) -> None:
        if self._enabled:
            torch.cuda.synchronize(self._device)

    def reset_peak_memory_stats(self) -> None:
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(self._device)

    def memory_allocated(self) -> int:
        if not self._enabled:
            return 0

        return torch.cuda.memory_allocated(self._device)

    def max_memory_allocated(self) -> int:
        """Reports peak tensor memory tracker by PyTorch."""
        if not self._enabled:
            return 0

        return torch.cuda.max_memory_allocated(self._device)

    def max_memory_reserved(self) -> int:
        """Reports peak memory retained by PyTorch's caching allocator."""
        if not self._enabled:
            return 0

        return torch.cuda.max_memory_reserved(self._device)
