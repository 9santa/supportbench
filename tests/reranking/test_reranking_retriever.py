import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from supportbench.data.loaders import load_documents
from supportbench.data.models import Document
from supportbench.reranking.base import (
    RerankCandidate,
    RerankResult,
)
from supportbench.reranking.cross_encoder import SentenceTransformerCrossEncoderReranker
from supportbench.reranking.retriever import (
    RerankingRetriever,
)
from supportbench.retrieval.base import (
    SearchResult,
)
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents_v2.jsonl"
DENSE_INDEX_PATH = PROJECT_ROOT / "artifacts" / "dense_v2"


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


@pytest.mark.smoke
def test_reranking_retriever_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the reranking smoke test")

    documents = load_documents(DOCUMENTS_PATH)
    document_ids = {document.doc_id for document in documents}

    factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=DENSE_INDEX_PATH,
            dense_model_name="intfloat/multilingual-e5-base",
            dense_device="cuda",
            dense_batch_size=16,
            bm25_weight=1.0,
            dense_weight=1.5,
            candidate_k=100,
            rrf_k=10,
        ),
    )

    reranker = SentenceTransformerCrossEncoderReranker(
        "BAAI/bge-reranker-v2-m3",
        device="cuda",
        batch_size=16,
        max_length=512,
    )

    reranking_retriever = RerankingRetriever(
        candidate_retriever=factory.create("hybrid"),
        reranker=reranker,
        documents=documents,
        candidate_k=50,
    )

    results = reranking_retriever.search(
        "не работает vpn на ubuntu",
        top_k=10,
    )

    result_doc_ids = [result.doc_id for result in results]

    assert len(results) == 10
    assert len(set(result_doc_ids)) == 10
    assert set(result_doc_ids) <= document_ids
    assert "vpn_ubuntu_troubleshooting" in result_doc_ids
    assert [result.rank for result in results] == list(range(1, 11))
    assert all(math.isfinite(result.score) for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )
