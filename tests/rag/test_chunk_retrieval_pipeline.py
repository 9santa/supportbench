from supportbench.chunking.models import Chunk
from supportbench.data.models import Document
from supportbench.rag.chunk_retrieval_pipeline import (
    RepresentativeChunkRetrievalPipeline,
)
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.rag.parent_retrieval import ParentRetrievalRun
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

    retrieved = pipeline.retrieve(run, top_k=2)

    assert [chunk.chunk_id for chunk in retrieved] == ["b_2", "b_1", "a_1"]
    assert [chunk.parent_rank for chunk in retrieved] == [1, 1, 2]
    assert [chunk.evidence_rank for chunk in retrieved] == [1, 2, 1]
    assert retrieved[0].section_path == ("Resolution",)
