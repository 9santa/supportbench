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
                raise ValueError("retriever returned an unknown document ID:", repr(result.doc_id))

            parent_results.append(
                SearchResult(
                    doc_id=parent_id,
                    score=result.score,
                    rank=result.rank,
                )
            )

        return parent_results
