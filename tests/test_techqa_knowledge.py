from collections.abc import Sequence

from supportbench.chunking.models import Chunk
from supportbench.knowledge.techqa import TechQAKnowledgeService
from supportbench.rag.context import EvidenceSelection
from supportbench.rag.models import RetrievedChunk
from supportbench.rag.retrieval import ParentRetrievalRun


class CharacterCodec:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class UnusedRetriever:
    def retrieve(self, query: str) -> ParentRetrievalRun:
        raise AssertionError("read() must not run retrieval")


class UnusedResolver:
    def resolve(
        self,
        run: ParentRetrievalRun,
        *,
        query: str | None = None,
        top_k: int = 5,
        evidence_selection: EvidenceSelection | None = None,
    ) -> list[RetrievedChunk]:
        raise AssertionError("read() must not resolve retrieval chunks")


def _chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1::chunk-{ordinal}",
        document_id="doc-1",
        document_title="Test document",
        text=text,
        ordinal=ordinal,
        token_count=len(text),
        section_path=(),
        start_char=None,
        end_char=None,
    )


def _service(*, read_max_chunks: int = 8) -> TechQAKnowledgeService:
    chunks = (_chunk(0, "first chunk"), _chunk(1, "second chunk"))
    return TechQAKnowledgeService(
        retrieval_service=UnusedRetriever(),
        chunk_resolver=UnusedResolver(),
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
        token_codec=CharacterCodec(),
        read_max_tokens=100,
        read_max_chunks=read_max_chunks,
    )


def test_requested_chunk_is_not_marked_truncated_when_fully_read() -> None:
    result = _service().read(
        document_id="doc-1",
        chunk_ids=("doc-1::chunk-1",),
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["doc-1::chunk-1"]
    assert not result.chunks[0].truncated
    assert not result.truncated


def test_requested_chunks_are_truncated_only_when_chunk_limit_drops_one() -> None:
    result = _service(read_max_chunks=1).read(
        document_id="doc-1",
        chunk_ids=("doc-1::chunk-0", "doc-1::chunk-1"),
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["doc-1::chunk-0"]
    assert result.truncated
