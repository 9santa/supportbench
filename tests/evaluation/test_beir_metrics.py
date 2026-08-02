import math

import pytest

from supportbench.data.models import QueryExample
from supportbench.evaluation.beir import evaluate_beir_retriever
from supportbench.retrieval.base import SearchResult


class FixedRetriever:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        document_ids = ("relevant-low", "noise", "relevant-high")
        return [
            SearchResult(doc_id=document_id, score=1.0 / rank, rank=rank)
            for rank, document_id in enumerate(document_ids[:top_k], start=1)
        ]


def test_computes_standard_beir_metrics() -> None:
    result = evaluate_beir_retriever(
        FixedRetriever(),
        (
            QueryExample(
                query_id="query",
                query="claim",
                relevant_doc_ids=("relevant-high", "relevant-low"),
                split="test",
            ),
        ),
        {"query": {"relevant-high": 2, "relevant-low": 1}},
        top_k=3,
        cutoffs=(1, 3),
    )

    ndcg_at_3 = (1.0 + 2.0 / math.log2(4)) / (2.0 + 1.0 / math.log2(3))

    assert dict(result.ndcg)[1] == pytest.approx(0.5)
    assert dict(result.ndcg)[3] == ndcg_at_3
    assert dict(result.mean_average_precision)[3] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert dict(result.recall)[1] == 0.5
    assert dict(result.recall)[3] == 1.0
    assert dict(result.precision)[3] == pytest.approx(2.0 / 3.0)
    assert dict(result.mrr)[3] == 1.0


def test_duplicate_parent_hits_do_not_receive_repeated_gain() -> None:
    class DuplicateRetriever:
        def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
            return [
                SearchResult("relevant", 1.0, 1),
                SearchResult("relevant", 0.9, 2),
                SearchResult("noise", 0.8, 3),
            ][:top_k]

    result = evaluate_beir_retriever(
        DuplicateRetriever(),
        (
            QueryExample(
                query_id="query",
                query="claim",
                relevant_doc_ids=("relevant",),
                split="test",
            ),
        ),
        {"query": {"relevant": 1}},
        top_k=3,
        cutoffs=(3,),
    )

    assert dict(result.ndcg)[3] == 1.0
    assert dict(result.mean_average_precision)[3] == 1.0
