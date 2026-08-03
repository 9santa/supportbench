from collections.abc import Sequence

import pytest

from supportbench.data.models import Document
from supportbench.reranking.base import (
    RerankCandidate,
    RerankResult,
)
from supportbench.reranking.retriever import (
    RerankingRetriever,
)
from supportbench.retrieval.base import (
    SearchResult,
)


class FakeCandidateRetriever:
    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self._results[:top_k]


class ReverseReranker:
    def __init__(self) -> None:
        self.received_candidates: tuple[
            RerankCandidate,
            ...,
        ] = ()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        self.received_candidates = tuple(candidates)

        reversed_candidates = list(reversed(candidates))

        return [
            RerankResult(
                doc_id=candidate.doc_id,
                score=float(len(reversed_candidates) - index),
                retrieval_score=(candidate.retrieval_score),
                retrieval_rank=(candidate.retrieval_rank),
            )
            for index, candidate in enumerate(reversed_candidates[:top_k])
        ]


def test_retrieves_candidates_then_reranks() -> None:
    candidate_retriever = FakeCandidateRetriever(
        [
            SearchResult(
                doc_id="doc_a",
                score=0.9,
                rank=1,
            ),
            SearchResult(
                doc_id="doc_b",
                score=0.8,
                rank=2,
            ),
            SearchResult(
                doc_id="doc_c",
                score=0.7,
                rank=3,
            ),
        ]
    )
    reranker = ReverseReranker()

    documents = [
        Document(
            doc_id="doc_a",
            title="Title A",
            text="Text A",
            category="",
        ),
        Document(
            doc_id="doc_b",
            title="Title B",
            text="Text B",
            category="",
        ),
        Document(
            doc_id="doc_c",
            title="Title C",
            text="Text C",
            category="",
        ),
    ]

    retriever = RerankingRetriever(
        candidate_retriever=(candidate_retriever),
        reranker=reranker,
        documents=documents,
        candidate_k=3,
    )

    results = retriever.search(
        "query",
        top_k=2,
    )

    assert candidate_retriever.calls == [("query", 3)]

    assert [result.doc_id for result in results] == [
        "doc_c",
        "doc_b",
    ]

    assert [result.rank for result in results] == [1, 2]

    assert reranker.received_candidates[0].text == "Title A\nText A"


def test_rejects_top_k_above_candidate_k() -> None:
    retriever = RerankingRetriever(
        candidate_retriever=(FakeCandidateRetriever([])),
        reranker=ReverseReranker(),
        documents=[
            Document(
                doc_id="doc",
                title="Title",
                text="Text",
                category="",
            )
        ],
        candidate_k=10,
    )

    with pytest.raises(
        ValueError,
        match=("top_k must not be greater than candidate_k"),
    ):
        retriever.search(
            "query",
            top_k=11,
        )


def test_rejects_unknown_candidate_document() -> None:
    retriever = RerankingRetriever(
        candidate_retriever=(
            FakeCandidateRetriever(
                [
                    SearchResult(
                        doc_id="unknown",
                        score=1.0,
                        rank=1,
                    )
                ]
            )
        ),
        reranker=ReverseReranker(),
        documents=[
            Document(
                doc_id="known",
                title="Title",
                text="Text",
                category="",
            )
        ],
        candidate_k=10,
    )

    with pytest.raises(
        ValueError,
        match=("candidate retriever returned an unknown document"),
    ):
        retriever.search(
            "query",
            top_k=5,
        )
