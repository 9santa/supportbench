import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

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

        if top_k > self._chunk_candidate_k:
            raise ValueError("top_k must not be greater than chunk_candidate_k")

        chunk_results = self._chunk_reranker.search(
            query,
            top_k=self._chunk_candidate_k,
        )
        return rank_parents_from_chunk_scores(
            tuple((result.doc_id, result.score) for result in chunk_results),
            parent_by_chunk_id=self._parent_by_chunk_id,
            top_k=top_k,
            second_evidence_weight=self._second_evidence_weight,
        )


def rank_parents_from_chunk_scores(
    chunk_scores: Sequence[tuple[str, float]],
    *,
    parent_by_chunk_id: Mapping[str, str],
    top_k: int,
    second_evidence_weight: float = 0.0,
) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not math.isfinite(second_evidence_weight) or not 0.0 <= second_evidence_weight <= 1.0:
        raise ValueError("second_evidence_weight must be between 0 and 1")

    scores_by_parent: defaultdict[str, list[float]] = defaultdict(list)

    for chunk_id, score in chunk_scores:
        parent_id = parent_by_chunk_id.get(chunk_id)

        if parent_id is None:
            raise ValueError(f"chunk reranker returned an unknown chunk ID: {chunk_id!r}")

        if not math.isfinite(score):
            raise ValueError("chunk reranker returned a non-finite score")

        scores_by_parent[parent_id].append(score)

    parent_scores = [
        (
            parent_id,
            _aggregate_scores(scores, second_evidence_weight=second_evidence_weight),
        )
        for parent_id, scores in scores_by_parent.items()
    ]
    parent_scores.sort(key=lambda item: (-item[1], item[0]))

    return [
        SearchResult(doc_id=parent_id, score=score, rank=rank)
        for rank, (parent_id, score) in enumerate(parent_scores[:top_k], start=1)
    ]


def _aggregate_scores(
    scores: list[float],
    *,
    second_evidence_weight: float,
) -> float:
    scores.sort(reverse=True)
    best_score = scores[0]

    if len(scores) == 1 or second_evidence_weight == 0.0:
        return best_score

    return (best_score + second_evidence_weight * scores[1]) / (
        1.0 + second_evidence_weight
    )
