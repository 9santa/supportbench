import math

import pytest

from supportbench.data.models import Document
from supportbench.experiments.synthetic_v2.rag.retrieval_pipeline import (
    RetrievalPipeline,
)
from supportbench.rag.document_store import (
    InMemoryDocumentStore,
)
from supportbench.retrieval.base import (
    SearchResult,
)


class StubRetriever:
    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self._results = results
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.received_query = query
        self.received_top_k = top_k

        return self._results[:top_k]


def make_document(
    doc_id: str,
) -> Document:
    return Document(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Text {doc_id}",
        category="support",
    )


def make_pipeline(
    *,
    results: list[SearchResult],
    documents: list[Document],
) -> tuple[RetrievalPipeline, StubRetriever]:
    retriever = StubRetriever(results)

    pipeline = RetrievalPipeline(
        retriever=retriever,
        document_store=InMemoryDocumentStore(documents),
    )

    return pipeline, retriever


def test_query_is_forwarded_to_retriever() -> None:
    pipeline, retriever = make_pipeline(
        results=[],
        documents=[],
    )

    pipeline.retrieve("reset gitlab 2fa")

    assert retriever.received_query == "reset gitlab 2fa"


def test_top_k_is_forwarded_to_retriever() -> None:
    pipeline, retriever = make_pipeline(
        results=[],
        documents=[],
    )

    pipeline.retrieve(
        "query",
        top_k=7,
    )

    assert retriever.received_top_k == 7


def test_results_are_resolved_to_documents() -> None:
    document = make_document("doc_a")

    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_a",
                score=0.9,
                rank=1,
            )
        ],
        documents=[document],
    )

    result = pipeline.retrieve("query")

    assert result[0].doc_id == "doc_a"
    assert result[0].title == document.title
    assert result[0].text == document.text
    assert result[0].category == document.category


def test_scores_and_ranks_are_preserved() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_a",
                score=0.81,
                rank=1,
            )
        ],
        documents=[make_document("doc_a")],
    )

    result = pipeline.retrieve("query")

    assert result[0].score == 0.81
    assert result[0].rank == 1


def test_retrieval_order_is_preserved() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_b",
                score=0.9,
                rank=1,
            ),
            SearchResult(
                doc_id="doc_a",
                score=0.8,
                rank=2,
            ),
        ],
        documents=[
            make_document("doc_a"),
            make_document("doc_b"),
        ],
    )

    result = pipeline.retrieve("query")

    assert [document.doc_id for document in result] == [
        "doc_b",
        "doc_a",
    ]


def test_empty_query_returns_no_documents() -> None:
    pipeline, retriever = make_pipeline(
        results=[],
        documents=[],
    )

    assert pipeline.retrieve("   ") == []
    assert retriever.received_query is None


def test_invalid_top_k_is_rejected() -> None:
    pipeline, _ = make_pipeline(
        results=[],
        documents=[],
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        pipeline.retrieve(
            "query",
            top_k=0,
        )


def test_unknown_retrieved_document_is_rejected() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="unknown",
                score=1.0,
                rank=1,
            )
        ],
        documents=[],
    )

    with pytest.raises(
        ValueError,
        match="unknown document",
    ):
        pipeline.retrieve("query")


def test_duplicate_retrieved_document_is_rejected() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_a",
                score=1.0,
                rank=1,
            ),
            SearchResult(
                doc_id="doc_a",
                score=0.9,
                rank=2,
            ),
        ],
        documents=[make_document("doc_a")],
    )

    with pytest.raises(
        ValueError,
        match="duplicate document ID",
    ):
        pipeline.retrieve("query")


def test_non_consecutive_ranks_are_rejected() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_a",
                score=1.0,
                rank=1,
            ),
            SearchResult(
                doc_id="doc_b",
                score=0.9,
                rank=3,
            ),
        ],
        documents=[
            make_document("doc_a"),
            make_document("doc_b"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="ranks must be consecutive",
    ):
        pipeline.retrieve("query")


def test_non_finite_score_is_rejected() -> None:
    pipeline, _ = make_pipeline(
        results=[
            SearchResult(
                doc_id="doc_a",
                score=math.nan,
                rank=1,
            )
        ],
        documents=[make_document("doc_a")],
    )

    with pytest.raises(
        ValueError,
        match="score must be finite",
    ):
        pipeline.retrieve("query")
