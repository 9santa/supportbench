from supportbench.chunking.models import Chunk
from supportbench.data.models import Document
from supportbench.rag.chunk_retrieval_pipeline import (
    RepresentativeChunkRetrievalPipeline,
)
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.retrieval.base import SearchResult
from supportbench.retrieval.parent_hybrid import ParentSearchResult


class StubParentRetriever:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult("parent_b", 0.9, 1),
            SearchResult("parent_a", 0.8, 2),
        ][:top_k]


class StubRepresentativeRetriever:
    def search_with_chunks(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[ParentSearchResult]:
        return [
            ParentSearchResult("parent_a", 1.0, 1, ("a_1",)),
            ParentSearchResult("parent_b", 0.9, 2, ("b_2", "b_1")),
        ][:top_k]


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
    pipeline = RepresentativeChunkRetrievalPipeline(
        parent_retriever=StubParentRetriever(),
        representative_retriever=StubRepresentativeRetriever(),
        representative_candidate_k=2,
        chunk_store=InMemoryDocumentStore(documents),
        chunks_by_id=chunks,
    )

    retrieved = pipeline.retrieve("query", top_k=2)

    assert [chunk.chunk_id for chunk in retrieved] == ["b_2", "b_1", "a_1"]
    assert [chunk.parent_rank for chunk in retrieved] == [1, 1, 2]
    assert [chunk.evidence_rank for chunk in retrieved] == [1, 2, 1]
    assert retrieved[0].section_path == ("Resolution",)
