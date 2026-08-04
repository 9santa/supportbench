import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from supportbench.rag.document_store import DocumentStore
from supportbench.reranking.base import RerankCandidate, Reranker, RerankResult
from supportbench.reranking.parent import rank_parents_from_chunk_scores
from supportbench.reranking.retriever import format_document_for_reranking
from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.hybrid import weighted_rrf_fusion
from supportbench.retrieval.parent_hybrid import ParentSearchResult


class RepresentativeParentRetriever(Protocol):
    def search_with_chunks(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[ParentSearchResult]: ...


@dataclass(frozen=True, slots=True)
class ParentRetrievalRun:
    candidate_parents: tuple[SearchResult, ...]
    representative_chunks_by_parent: Mapping[str, tuple[str, ...]]
    reranked_parents: tuple[SearchResult, ...]
    fused_parents: tuple[SearchResult, ...]


class ParentRetrievalService:
    """Execute the online parent retrieval path once for a single query."""

    def __init__(
        self,
        *,
        parent_retriever: RepresentativeParentRetriever,
        reranker: Reranker,
        chunk_store: DocumentStore,
        parent_by_chunk_id: Mapping[str, str],
        parent_candidate_k: int,
        chunks_per_parent: int,
        candidate_prior_weight: float,
        fusion_rrf_k: int,
        second_evidence_weight: float = 0.0,
    ) -> None:
        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        if parent_candidate_k <= 0:
            raise ValueError("parent_candidate_k must be positive")

        if chunks_per_parent <= 0:
            raise ValueError("chunks_per_parent must be positive")

        if not math.isfinite(candidate_prior_weight) or candidate_prior_weight < 0.0:
            raise ValueError("candidate_prior_weight must be finite and non-negative")

        if fusion_rrf_k <= 0:
            raise ValueError("fusion_rrf_k must be positive")

        if not math.isfinite(second_evidence_weight) or not 0.0 <= second_evidence_weight <= 1.0:
            raise ValueError("second_evidence_weight must be between 0 and 1")

        self._parent_retriever = parent_retriever
        self._reranker = reranker
        self._chunk_store = chunk_store
        self._parent_by_chunk_id = dict(parent_by_chunk_id)
        self._parent_candidate_k = parent_candidate_k
        self._chunks_per_parent = chunks_per_parent
        self._candidate_prior_weight = candidate_prior_weight
        self._fusion_rrf_k = fusion_rrf_k
        self._second_evidence_weight = second_evidence_weight

    def retrieve(self, query: str) -> ParentRetrievalRun:
        if not query.strip():
            return _empty_run()

        parent_results = self._parent_retriever.search_with_chunks(
            query,
            top_k=self._parent_candidate_k,
        )
        self._validate_parent_results(parent_results)

        if not parent_results:
            return _empty_run()

        candidate_parents = tuple(
            SearchResult(
                doc_id=result.parent_id,
                score=result.score,
                rank=result.rank,
            )
            for result in parent_results
        )
        representatives = MappingProxyType(
            {
                result.parent_id: result.representative_chunk_ids[: self._chunks_per_parent]
                for result in parent_results
            }
        )
        rerank_candidates = self._build_rerank_candidates(parent_results)
        reranked_chunks = self._reranker.rerank(
            query,
            rerank_candidates,
            top_k=len(rerank_candidates),
        )
        self._validate_reranker_results(
            reranked_chunks,
            candidate_ids={candidate.doc_id for candidate in rerank_candidates},
        )
        reranked_parents = tuple(
            rank_parents_from_chunk_scores(
                tuple((result.doc_id, result.score) for result in reranked_chunks),
                parent_by_chunk_id=self._parent_by_chunk_id,
                top_k=self._parent_candidate_k,
                second_evidence_weight=self._second_evidence_weight,
            )
        )
        fused_parents = tuple(
            weighted_rrf_fusion(
                (
                    (candidate_parents, self._candidate_prior_weight),
                    (reranked_parents, 1.0),
                ),
                top_k=self._parent_candidate_k,
                rrf_k=self._fusion_rrf_k,
            )
        )

        return ParentRetrievalRun(
            candidate_parents=candidate_parents,
            representative_chunks_by_parent=representatives,
            reranked_parents=reranked_parents,
            fused_parents=fused_parents,
        )

    def _build_rerank_candidates(
        self,
        parent_results: Sequence[ParentSearchResult],
    ) -> tuple[RerankCandidate, ...]:
        candidates: list[RerankCandidate] = []
        seen_chunk_ids: set[str] = set()

        for parent in parent_results:
            for chunk_id in parent.representative_chunk_ids[: self._chunks_per_parent]:
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"duplicate representative chunk ID: {chunk_id!r}")

                expected_parent_id = self._parent_by_chunk_id.get(chunk_id)

                if expected_parent_id is None:
                    raise ValueError(f"unknown representative chunk ID: {chunk_id!r}")

                if expected_parent_id != parent.parent_id:
                    raise ValueError(
                        f"chunk {chunk_id!r} belongs to {expected_parent_id!r}, "
                        f"not {parent.parent_id!r}"
                    )

                try:
                    document = self._chunk_store.get(chunk_id)
                except KeyError as error:
                    raise ValueError(
                        f"representative chunk is missing from the runtime corpus: {chunk_id!r}"
                    ) from error

                seen_chunk_ids.add(chunk_id)
                candidates.append(
                    RerankCandidate(
                        doc_id=chunk_id,
                        text=format_document_for_reranking(document),
                        retrieval_score=parent.score,
                        retrieval_rank=len(candidates) + 1,
                    )
                )

        return tuple(candidates)

    @staticmethod
    def _validate_parent_results(results: Sequence[ParentSearchResult]) -> None:
        seen_parent_ids: set[str] = set()

        for expected_rank, result in enumerate(results, start=1):
            if not result.parent_id.strip():
                raise ValueError("parent retriever returned an empty parent ID")

            if result.parent_id in seen_parent_ids:
                raise ValueError(f"parent retriever returned duplicate ID: {result.parent_id!r}")

            if result.rank != expected_rank:
                raise ValueError(
                    "parent ranks must be consecutive starting at 1; "
                    f"expected {expected_rank}, received {result.rank}"
                )

            if not math.isfinite(result.score):
                raise ValueError("parent score must be finite")

            if not result.representative_chunk_ids:
                raise ValueError(
                    f"parent {result.parent_id!r} has no representative chunks"
                )

            seen_parent_ids.add(result.parent_id)

    @staticmethod
    def _validate_reranker_results(
        results: Sequence[RerankResult],
        *,
        candidate_ids: set[str],
    ) -> None:
        result_ids = [result.doc_id for result in results]

        if len(result_ids) != len(set(result_ids)):
            raise ValueError("reranker returned duplicate chunk IDs")

        if set(result_ids) != candidate_ids:
            raise ValueError("reranker must return every representative chunk exactly once")


def _empty_run() -> ParentRetrievalRun:
    return ParentRetrievalRun(
        candidate_parents=(),
        representative_chunks_by_parent=MappingProxyType({}),
        reranked_parents=(),
        fused_parents=(),
    )
