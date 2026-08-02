from collections.abc import Mapping

from supportbench.retrieval.base import (
    Retriever,
    SearchResult,
)


class ParentDocumentRetriever:
    """
    Map chunk retrieval results to parent document IDs.

    Parent duplicates are preserved on purpose because
    Recall@k is measured over the raw chunking results.
    """

    def __init__(
        self,
        chunk_retriever: Retriever,
        *,
        parent_by_chunk_id: Mapping[str, str],
    ) -> None:
        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        self._chunk_retriever = chunk_retriever
        self._parent_by_chunk_id = dict(parent_by_chunk_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        chunk_results = self._chunk_retriever.search(query, top_k=top_k)

        parent_results: list[SearchResult] = []

        for result in chunk_results:
            parent_id = self._parent_by_chunk_id.get(result.doc_id)

            if parent_id is None:
                raise ValueError(f"retriever returned an unknown chunk ID: {result.doc_id!r}")

            parent_results.append(
                SearchResult(
                    doc_id=parent_id,
                    score=result.score,
                    rank=result.rank,
                )
            )

        return parent_results


class UniqueParentDocumentRetriever:
    """Collapse ranked chunks to their first occurrence per parent document."""

    def __init__(
        self,
        chunk_retriever: Retriever,
        *,
        parent_by_chunk_id: Mapping[str, str],
        chunk_candidate_k: int,
    ) -> None:
        if not parent_by_chunk_id:
            raise ValueError("parent_by_chunk_id must not be empty")

        if chunk_candidate_k <= 0:
            raise ValueError("chunk_candidate_k must be positive")

        self._chunk_retriever = chunk_retriever
        self._parent_by_chunk_id = dict(parent_by_chunk_id)
        self._chunk_candidate_k = chunk_candidate_k

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunk_results = self._chunk_retriever.search(
            query,
            top_k=self._chunk_candidate_k,
        )
        parent_results: list[SearchResult] = []
        seen_parent_ids: set[str] = set()

        for result in chunk_results:
            parent_id = self._parent_by_chunk_id.get(result.doc_id)

            if parent_id is None:
                raise ValueError(
                    f"retriever returned an unknown chunk ID: {result.doc_id!r}"
                )

            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)
            parent_results.append(
                SearchResult(
                    doc_id=parent_id,
                    score=result.score,
                    rank=len(parent_results) + 1,
                )
            )

            if len(parent_results) == top_k:
                break

        return parent_results
