import math
from collections import defaultdict
from collections.abc import Mapping

from supportbench.retrieval.base import Retriever, SearchResult


class ParentEvidenceRerankingRetriever(Retriever):
    """Aggregate cross-encoder chunk scores into an independent parent ranking."""

    def __init__(
        self,
        chunk_reranker: Retriever,
        *,
        parent_by_chunk_id: Mapping[str, str],
        chunk_candidate_k: int,
        second_evidence_weight: float = 0.0,
    ) -> None:
        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        if chunk_candidate_k <= 0:
            raise ValueError("chunk_candidate_k must be positive")

        if not math.isfinite(second_evidence_weight) or not 0.0 <= second_evidence_weight <= 1.0:
            raise ValueError("second_evidence_weight must be between 0 and 1")

        self._chunk_reranker = chunk_reranker
        self._parent_by_chunk_id = dict(parent_by_chunk_id)
        self._chunk_candidate_k = chunk_candidate_k
        self._second_evidence_weight = second_evidence_weight

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunk_results = self._chunk_reranker.search(query, top_k=self._chunk_candidate_k)
        scores_by_parent: defaultdict[str, list[float]] = defaultdict(list)

        for result in chunk_results:
            parent_id = self._parent_by_chunk_id.get(result.doc_id)

            if parent_id is None:
                raise ValueError(
                    f"chunk reranker returned an unknown chunk ID: {result.doc_id!r}"
                )

            scores_by_parent[parent_id].append(result.score)

        parent_scores = [
            (parent_id, self._aggregate_scores(scores))
            for parent_id, scores in scores_by_parent.items()
        ]
        parent_scores.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(parent_id, score, rank)
            for rank, (parent_id, score) in enumerate(parent_scores[:top_k], start=1)
        ]

    def _aggregate_scores(self, scores: list[float]) -> float:
        scores.sort(reverse=True)
        best_score = scores[0]

        if len(scores) == 1 or self._second_evidence_weight == 0.0:
            return best_score

        weight = self._second_evidence_weight
        return (best_score + weight * scores[1]) / (1.0 + weight)
