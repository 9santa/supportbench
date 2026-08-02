from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from supportbench.retrieval.base import Retriever, SearchResult
from supportbench.retrieval.hybrid import WeightedRetrieverSource

type ParentAggregation = Literal["best_chunk_rank", "capped_top_2_sum"]


@dataclass(frozen=True, slots=True)
class ParentSearchResult:
    parent_id: str
    score: float
    rank: int
    representative_chunk_ids: tuple[str, ...]


class ParentWeightedRRFHybrid(Retriever):
    def __init__(
        self,
        *,
        sources: Sequence[WeightedRetrieverSource],
        parent_by_chunk_id: Mapping[str, str],
        source_candidate_k: int,
        rrf_k: int,
        aggregation: ParentAggregation,
        representative_chunks_per_parent: int = 2,
    ) -> None:
        if not sources:
            raise ValueError("at least one retriever source is required")

        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        if source_candidate_k <= 0:
            raise ValueError("source_candidate_k must be positive")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        if aggregation not in ("best_chunk_rank", "capped_top_2_sum"):
            raise ValueError(f"unknown parent aggregation: {aggregation!r}")

        if representative_chunks_per_parent <= 0:
            raise ValueError("representative_chunks_per_parent must be positive")

        names = [source.name for source in sources]

        if len(names) != len(set(names)):
            raise ValueError("retriever source names must be unique")

        self._sources = tuple(sources)
        self._parent_by_chunk_id = dict(parent_by_chunk_id)
        self._source_candidate_k = source_candidate_k
        self._rrf_k = rrf_k
        self._aggregation = aggregation
        self._representative_chunks_per_parent = representative_chunks_per_parent

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                doc_id=result.parent_id,
                score=result.score,
                rank=result.rank,
            )
            for result in self.search_with_chunks(query, top_k=top_k)
        ]

    def search_with_chunks(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[ParentSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        source_parent_chunks: dict[str, dict[str, list[tuple[int, str]]]] = {}
        chunk_evidence: dict[str, dict[str, float]] = {}
        chunk_best_rank: dict[str, dict[str, int]] = {}

        for source in self._sources:
            source_results = source.retriever.search(query, top_k=self._source_candidate_k)
            seen_chunk_ids: set[str] = set()

            for rank, result in enumerate(source_results, start=1):
                if result.doc_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(result.doc_id)
                parent_id = self._parent_by_chunk_id.get(result.doc_id)

                if parent_id is None:
                    raise ValueError(
                        f"retriever source {source.name!r} returned an unknown chunk ID: "
                        f"{result.doc_id!r}"
                    )

                source_parent_chunks.setdefault(parent_id, {}).setdefault(source.name, []).append(
                    (rank, result.doc_id)
                )
                contribution = source.weight / (self._rrf_k + rank)
                evidence = chunk_evidence.setdefault(parent_id, {})
                evidence[result.doc_id] = evidence.get(result.doc_id, 0.0) + contribution
                best_ranks = chunk_best_rank.setdefault(parent_id, {})
                best_ranks[result.doc_id] = min(best_ranks.get(result.doc_id, rank), rank)

        parent_items: list[tuple[str, float, tuple[str, ...]]] = []
        contribution_limit = 1 if self._aggregation == "best_chunk_rank" else 2

        for parent_id, chunks_by_source in source_parent_chunks.items():
            score = 0.0
            source_best_chunks: set[str] = set()

            for source in self._sources:
                ranked_chunks = chunks_by_source.get(source.name, ())

                for rank, _ in ranked_chunks[:contribution_limit]:
                    score += source.weight / (self._rrf_k + rank)

                if ranked_chunks:
                    source_best_chunks.add(ranked_chunks[0][1])

            evidence = chunk_evidence[parent_id]
            best_ranks = chunk_best_rank[parent_id]

            def chunk_key(
                chunk_id: str,
                evidence: dict[str, float] = evidence,
                best_ranks: dict[str, int] = best_ranks,
            ) -> tuple[float, int, str]:
                return (-evidence[chunk_id], best_ranks[chunk_id], chunk_id)

            representatives = sorted(source_best_chunks, key=chunk_key)

            if len(representatives) < self._representative_chunks_per_parent:
                remaining = sorted(
                    (chunk_id for chunk_id in evidence if chunk_id not in source_best_chunks),
                    key=chunk_key,
                )
                representatives.extend(
                    remaining[: self._representative_chunks_per_parent - len(representatives)]
                )

            parent_items.append(
                (
                    parent_id,
                    score,
                    tuple(representatives[: self._representative_chunks_per_parent]),
                )
            )

        parent_items.sort(key=lambda item: (-item[1], item[0]))

        return [
            ParentSearchResult(
                parent_id=parent_id,
                score=score,
                rank=rank,
                representative_chunk_ids=representatives,
            )
            for rank, (parent_id, score, representatives) in enumerate(
                parent_items[:top_k],
                start=1,
            )
        ]


class ParentCandidateChunkRetriever(Retriever):
    def __init__(
        self,
        parent_retriever: ParentWeightedRRFHybrid,
        *,
        parent_candidate_k: int,
        chunks_per_parent: int,
    ) -> None:
        if parent_candidate_k <= 0:
            raise ValueError("parent_candidate_k must be positive")

        if chunks_per_parent <= 0:
            raise ValueError("chunks_per_parent must be positive")

        self._parent_retriever = parent_retriever
        self._parent_candidate_k = parent_candidate_k
        self._chunks_per_parent = chunks_per_parent

    @property
    def candidate_k(self) -> int:
        return self._parent_candidate_k * self._chunks_per_parent

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self.candidate_k:
            raise ValueError("top_k must not be greater than candidate_k")

        parents = self._parent_retriever.search_with_chunks(
            query,
            top_k=self._parent_candidate_k,
        )
        results: list[SearchResult] = []

        for parent in parents:
            for chunk_id in parent.representative_chunk_ids[: self._chunks_per_parent]:
                results.append(
                    SearchResult(
                        doc_id=chunk_id,
                        score=parent.score,
                        rank=len(results) + 1,
                    )
                )

        return results[:top_k]


class ParentCandidateSubsetRetriever(Retriever):
    """Select a smaller parent/chunk pool from a cached maximal candidate pool."""

    def __init__(
        self,
        chunk_retriever: Retriever,
        *,
        parent_by_chunk_id: Mapping[str, str],
        source_candidate_k: int,
        parent_candidate_k: int,
        chunks_per_parent: int,
    ) -> None:
        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        if source_candidate_k <= 0:
            raise ValueError("source_candidate_k must be positive")

        if parent_candidate_k <= 0:
            raise ValueError("parent_candidate_k must be positive")

        if chunks_per_parent <= 0:
            raise ValueError("chunks_per_parent must be positive")

        self._chunk_retriever = chunk_retriever
        self._parent_by_chunk_id = dict(parent_by_chunk_id)
        self._source_candidate_k = source_candidate_k
        self._parent_candidate_k = parent_candidate_k
        self._chunks_per_parent = chunks_per_parent

    @property
    def candidate_k(self) -> int:
        return self._parent_candidate_k * self._chunks_per_parent

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self.candidate_k:
            raise ValueError("top_k must not be greater than candidate_k")

        source_results = self._chunk_retriever.search(
            query,
            top_k=self._source_candidate_k,
        )
        parent_order: dict[str, int] = {}
        chunks_seen_by_parent: dict[str, int] = {}
        selected: list[SearchResult] = []

        for result in source_results:
            parent_id = self._parent_by_chunk_id.get(result.doc_id)

            if parent_id is None:
                raise ValueError(
                    f"candidate retriever returned an unknown chunk ID: {result.doc_id!r}"
                )

            if parent_id not in parent_order:
                if len(parent_order) == self._parent_candidate_k:
                    continue

                parent_order[parent_id] = len(parent_order) + 1

            chunk_count = chunks_seen_by_parent.get(parent_id, 0)

            if chunk_count == self._chunks_per_parent:
                continue

            chunks_seen_by_parent[parent_id] = chunk_count + 1
            selected.append(
                SearchResult(
                    doc_id=result.doc_id,
                    score=result.score,
                    rank=len(selected) + 1,
                )
            )

        return selected[:top_k]
