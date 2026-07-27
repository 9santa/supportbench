import math
from collections.abc import Sequence
from dataclasses import dataclass

from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)


@dataclass(frozen=True, slots=True)
class WeightedRetrieverSource:
    name: str
    retriever: Retriever
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("retriever source name must be non-empty")

        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("retriever weight must be finite and non-negative")


class WeightedRRFHybrid:
    def __init__(
        self,
        *,
        sources: Sequence[WeightedRetrieverSource],
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        if not sources:
            raise ValueError("at least one retriever source is required")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        names = [source.name for source in sources]

        if len(names) != len(set(names)):
            raise ValueError("retriever source names must be unique")

        self._sources = sources
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        scores: dict[str, float] = {}

        for source in self._sources:
            source_result = source.retriever.search(
                query,
                top_k=self._candidate_k,
            )

            seen_doc_ids: set[str] = set()

            for rank, result in enumerate(source_result, start=1):
                if result.doc_id in seen_doc_ids:
                    continue

                seen_doc_ids.add(result.doc_id)

                contribution = source.weight / (self._rrf_k + rank)

                scores[result.doc_id] = scores.get(result.doc_id, 0.0) + contribution

        ranked_documents = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
            )
            for rank, (doc_id, score) in enumerate(ranked_documents[:top_k], start=1)
        ]
