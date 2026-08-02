import pytest

from supportbench.evaluation.parent_document import (
    ParentDocumentRetriever,
    UniqueParentDocumentRetriever,
)
from supportbench.retrieval.base import (
    SearchResult,
)


class StubChunkRetriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        results = [
            SearchResult(
                doc_id="D1::ft4o1::chunk_0000",
                score=0.9,
                rank=1,
            ),
            SearchResult(
                doc_id="D1::ft4o1::chunk_0001",
                score=0.8,
                rank=2,
            ),
            SearchResult(
                doc_id="D2::ft4o1::chunk_0000",
                score=0.7,
                rank=3,
            ),
        ]

        return results[:top_k]


def test_maps_chunks_to_parent_documents() -> None:
    retriever = ParentDocumentRetriever(
        StubChunkRetriever(),
        parent_by_chunk_id={
            "D1::ft4o1::chunk_0000": "D1",
            "D1::ft4o1::chunk_0001": "D1",
            "D2::ft4o1::chunk_0000": "D2",
        },
    )

    results = retriever.search(
        "example query",
        top_k=3,
    )

    assert [result.doc_id for result in results] == [
        "D1",
        "D1",
        "D2",
    ]

    assert [result.rank for result in results] == [1, 2, 3]


def test_rejects_unknown_chunk_id() -> None:
    retriever = ParentDocumentRetriever(
        StubChunkRetriever(),
        parent_by_chunk_id={
            "D1::ft4o1::chunk_0000": "D1",
        },
    )

    with pytest.raises(
        ValueError,
        match="unknown chunk ID",
    ):
        retriever.search(
            "example query",
            top_k=2,
        )


def test_unique_parent_retriever_uses_deeper_chunk_pool() -> None:
    retriever = UniqueParentDocumentRetriever(
        StubChunkRetriever(),
        parent_by_chunk_id={
            "D1::ft4o1::chunk_0000": "D1",
            "D1::ft4o1::chunk_0001": "D1",
            "D2::ft4o1::chunk_0000": "D2",
        },
        chunk_candidate_k=3,
    )

    results = retriever.search("example query", top_k=2)

    assert [result.doc_id for result in results] == ["D1", "D2"]
    assert [result.rank for result in results] == [1, 2]
