from supportbench.retrieval.base import Retriever, SearchResult


class CandidateSetRestrictedRetriever(Retriever):
    """Restrict a ranking to the document IDs produced by a candidate retriever."""

    def __init__(
        self,
        ranking_retriever: Retriever,
        candidate_retriever: Retriever,
        *,
        ranking_candidate_k: int,
        candidate_k: int,
    ) -> None:
        if ranking_candidate_k <= 0:
            raise ValueError("ranking_candidate_k must be positive")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        self._ranking_retriever = ranking_retriever
        self._candidate_retriever = candidate_retriever
        self._ranking_candidate_k = ranking_candidate_k
        self._candidate_k = candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self._candidate_k:
            raise ValueError("top_k must not be greater than candidate_k")

        allowed_doc_ids = {
            result.doc_id
            for result in self._candidate_retriever.search(query, top_k=self._candidate_k)
        }
        ranked = self._ranking_retriever.search(
            query,
            top_k=self._ranking_candidate_k,
        )
        selected = [result for result in ranked if result.doc_id in allowed_doc_ids]

        return [
            SearchResult(doc_id=result.doc_id, score=result.score, rank=rank)
            for rank, result in enumerate(selected[:top_k], start=1)
        ]
