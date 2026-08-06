from collections.abc import Sequence

from supportbench.chunking.models import Chunk
from supportbench.data.models import Document
from supportbench.rag.context import RepresentativeChunkResolver
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.rag.retrieval import ParentRetrievalRun
from supportbench.reranking.base import RerankCandidate, RerankResult
from supportbench.retrieval.base import SearchResult


def _chunk(chunk_id: str, parent_id: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=parent_id,
        document_title=f"Title {parent_id}",
        text=f"Text {chunk_id}",
        ordinal=ordinal,
        token_count=2,
        section_path=("Resolution",),
        start_char=ordinal * 10,
        end_char=ordinal * 10 + 9,
    )


class ScoreReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
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
        results = [
            RerankResult(
                doc_id=candidate.doc_id,
                score=self._scores[candidate.doc_id],
                retrieval_score=candidate.retrieval_score,
                retrieval_rank=candidate.retrieval_rank,
            )
            for candidate in candidates
        ]
        results.sort(key=lambda result: (-result.score, result.retrieval_rank))
        return results[:top_k]


def test_materializes_representatives_in_final_parent_order() -> None:
    chunks = {
        "a_1": _chunk("a_1", "parent_a", 1),
        "b_1": _chunk("b_1", "parent_b", 1),
        "b_2": _chunk("b_2", "parent_b", 2),
    }
    documents = [
        Document(chunk.chunk_id, "Runtime title", chunk.text, "support")
        for chunk in chunks.values()
    ]
    resolver = RepresentativeChunkResolver(
        chunk_store=InMemoryDocumentStore(documents),
        chunks_by_id=chunks,
    )
    run = ParentRetrievalRun(
        candidate_parents=(
            SearchResult("parent_a", 1.0, 1),
            SearchResult("parent_b", 0.9, 2),
        ),
        representative_chunks_by_parent={
            "parent_a": ("a_1",),
            "parent_b": ("b_2", "b_1"),
        },
        reranked_parents=(
            SearchResult("parent_b", 0.9, 1),
            SearchResult("parent_a", 0.8, 2),
        ),
        fused_parents=(
            SearchResult("parent_b", 0.7, 1),
            SearchResult("parent_a", 0.6, 2),
        ),
    )

    retrieved = resolver.resolve(run, top_k=2)

    assert [chunk.chunk_id for chunk in retrieved] == ["b_2", "b_1", "a_1"]
    assert [chunk.parent_rank for chunk in retrieved] == [1, 1, 2]
    assert [chunk.evidence_rank for chunk in retrieved] == [1, 2, 1]
    assert retrieved[0].section_path == ("Resolution",)


def test_reranks_all_chunks_within_final_parents() -> None:
    chunks = {
        "a_1": _chunk("a_1", "parent_a", 1),
        "a_2": _chunk("a_2", "parent_a", 2),
        "a_3": _chunk("a_3", "parent_a", 3),
        "b_1": _chunk("b_1", "parent_b", 1),
        "b_2": _chunk("b_2", "parent_b", 2),
        "c_1": _chunk("c_1", "parent_c", 1),
    }
    documents = [
        Document(chunk.chunk_id, "Runtime title", chunk.text, "support")
        for chunk in chunks.values()
    ]
    reranker = ScoreReranker(
        {
            "a_1": 0.1,
            "a_2": 0.8,
            "a_3": 0.9,
            "b_1": 0.7,
            "b_2": 1.0,
            "c_1": 2.0,
        }
    )
    resolver = RepresentativeChunkResolver(
        chunk_store=InMemoryDocumentStore(documents),
        chunks_by_id=chunks,
        reranker=reranker,
        chunks_per_parent=2,
        evidence_selection="within_parent_rerank",
    )
    run = ParentRetrievalRun(
        candidate_parents=(
            SearchResult("parent_a", 1.0, 1),
            SearchResult("parent_b", 0.9, 2),
            SearchResult("parent_c", 0.8, 3),
        ),
        representative_chunks_by_parent={
            "parent_a": ("a_1",),
            "parent_b": ("b_1",),
            "parent_c": ("c_1",),
        },
        reranked_parents=(
            SearchResult("parent_b", 0.9, 1),
            SearchResult("parent_a", 0.8, 2),
            SearchResult("parent_c", 0.7, 3),
        ),
        fused_parents=(
            SearchResult("parent_b", 0.7, 1),
            SearchResult("parent_a", 0.6, 2),
            SearchResult("parent_c", 0.5, 3),
        ),
    )

    baseline = resolver.resolve(
        run,
        top_k=2,
        evidence_selection="retrieval_representatives",
    )
    retrieved = resolver.resolve(
        run,
        query="query",
        top_k=2,
        evidence_selection="within_parent_rerank",
    )

    assert reranker.calls == 1
    assert [chunk.chunk_id for chunk in baseline] == ["b_1", "a_1"]
    assert {candidate.doc_id for candidate in reranker.candidates} == set(chunks) - {"c_1"}
    assert [chunk.chunk_id for chunk in retrieved] == ["b_2", "b_1", "a_3", "a_2"]
    assert [chunk.parent_rank for chunk in retrieved] == [1, 1, 2, 2]
    assert [chunk.evidence_rank for chunk in retrieved] == [1, 2, 1, 2]
