from collections.abc import Mapping, Sequence
from types import MappingProxyType

from supportbench.data.models import QueryExample
from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)


class CachedRetriever:
    def __init__(
        self,
        results_by_query: Mapping[
            str,
            tuple[SearchResult, ...],
        ],
    ) -> None:
        self._results_by_query = MappingProxyType(dict(results_by_query))

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        return list(
            self._results_by_query.get(
                query,
                (),
            )[:top_k]
        )


def cache_retriever_results(
    retriever: Retriever,
    queries: Sequence[QueryExample],
    *,
    top_k: int,
) -> CachedRetriever:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results_by_query: dict[
        str,
        tuple[SearchResult, ...],
    ] = {}

    for query in queries:
        if query.query in results_by_query:
            continue

        results_by_query[query.query] = tuple(
            retriever.search(
                query.query,
                top_k=top_k,
            )
        )

    return CachedRetriever(results_by_query)
