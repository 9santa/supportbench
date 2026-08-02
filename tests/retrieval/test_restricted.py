from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.restricted import CandidateSetRestrictedRetriever


class RankingRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return list(self._results[:top_k])


def test_restricts_ranking_to_candidate_set() -> None:
    ranking = RankingRetriever(
        (
            SearchResult("outside", 0.99, 1),
            SearchResult("allowed_b", 0.9, 2),
            SearchResult("allowed_a", 0.8, 3),
        )
    )
    candidates = RankingRetriever(
        (
            SearchResult("allowed_a", 2.0, 1),
            SearchResult("allowed_b", 1.0, 2),
        )
    )
    retriever = CandidateSetRestrictedRetriever(
        ranking,
        candidates,
        ranking_candidate_k=3,
        candidate_k=2,
    )

    results = retriever.search("query", top_k=2)

    assert [result.doc_id for result in results] == ["allowed_b", "allowed_a"]
    assert [result.score for result in results] == [0.9, 0.8]
    assert [result.rank for result in results] == [1, 2]
