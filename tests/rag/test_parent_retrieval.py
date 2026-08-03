from collections.abc import Sequence

from supportbench.data.models import Document
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.rag.parent_retrieval import ParentRetrievalOrchestrator
from supportbench.reranking.base import RerankCandidate, RerankResult
from supportbench.retrieval.parent_hybrid import ParentSearchResult


class CountingParentRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search_with_chunks(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[ParentSearchResult]:
        self.calls += 1
        return [
            ParentSearchResult("parent_a", 0.7, 1, ("a_1", "a_2")),
            ParentSearchResult("parent_b", 0.6, 2, ("b_1", "b_2")),
        ][:top_k]


class RecordingReranker:
    def __init__(self) -> None:
        self.calls = 0
        self.candidates: tuple[RerankCandidate, ...] = ()

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        self.calls += 1
        self.candidates = tuple(candidates)
        scores = {
            "a_1": 0.8,
            "a_2": 0.2,
            "b_1": 0.9,
            "b_2": 0.1,
        }
        results = [
            RerankResult(
                doc_id=candidate.doc_id,
                score=scores[candidate.doc_id],
                retrieval_score=candidate.retrieval_score,
                retrieval_rank=candidate.retrieval_rank,
            )
            for candidate in candidates
        ]
        results.sort(key=lambda result: (-result.score, result.retrieval_rank, result.doc_id))
        return results[:top_k]


def _orchestrator(
    parent_retriever: CountingParentRetriever,
    reranker: RecordingReranker,
) -> ParentRetrievalOrchestrator:
    documents = [
        Document(chunk_id, f"Title {chunk_id}", f"Text {chunk_id}", "support")
        for chunk_id in ("a_1", "a_2", "b_1", "b_2")
    ]
    return ParentRetrievalOrchestrator(
        parent_retriever=parent_retriever,
        reranker=reranker,
        chunk_store=InMemoryDocumentStore(documents),
        parent_by_chunk_id={
            "a_1": "parent_a",
            "a_2": "parent_a",
            "b_1": "parent_b",
            "b_2": "parent_b",
        },
        parent_candidate_k=2,
        chunks_per_parent=2,
        candidate_prior_weight=1.25,
        fusion_rrf_k=10,
    )


def test_executes_first_stage_and_reranker_once_per_query() -> None:
    parent_retriever = CountingParentRetriever()
    reranker = RecordingReranker()

    run = _orchestrator(parent_retriever, reranker).run("query")

    assert parent_retriever.calls == 1
    assert reranker.calls == 1
    assert [candidate.doc_id for candidate in reranker.candidates] == [
        "a_1",
        "a_2",
        "b_1",
        "b_2",
    ]
    assert [result.doc_id for result in run.candidate_parents] == ["parent_a", "parent_b"]
    assert [result.doc_id for result in run.reranked_parents] == ["parent_b", "parent_a"]
    assert [result.doc_id for result in run.fused_parents] == ["parent_a", "parent_b"]
    assert run.representative_chunks_by_parent == {
        "parent_a": ("a_1", "a_2"),
        "parent_b": ("b_1", "b_2"),
    }


def test_empty_query_does_not_execute_retrieval() -> None:
    parent_retriever = CountingParentRetriever()
    reranker = RecordingReranker()

    run = _orchestrator(parent_retriever, reranker).run("  ")

    assert parent_retriever.calls == 0
    assert reranker.calls == 0
    assert not run.fused_parents
