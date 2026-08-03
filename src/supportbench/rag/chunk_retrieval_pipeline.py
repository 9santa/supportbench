import math
from collections.abc import Mapping, Sequence
from numbers import Real

from supportbench.chunking.models import Chunk
from supportbench.rag.document_store import DocumentStore
from supportbench.rag.models import RetrievedChunk
from supportbench.rag.parent_retrieval import ParentRetrievalRun
from supportbench.retrieval.base import SearchResult


class RepresentativeChunkRetrievalPipeline:
    """Resolve final parent ranks to the chunks that supplied retrieval evidence."""

    def __init__(
        self,
        *,
        chunk_store: DocumentStore,
        chunks_by_id: Mapping[str, Chunk],
    ) -> None:
        if not chunks_by_id:
            raise ValueError("chunks_by_id must not be empty")

        self._chunk_store = chunk_store
        self._chunks_by_id = dict(chunks_by_id)

    def retrieve(
        self,
        run: ParentRetrievalRun,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not run.fused_parents:
            return []

        parent_results = run.fused_parents[:top_k]
        self._validate_parent_results(parent_results)
        retrieved_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for parent_result in parent_results:
            chunk_ids = run.representative_chunks_by_parent.get(parent_result.doc_id)

            if chunk_ids is None:
                raise ValueError(
                    "final parent retriever returned a parent without representative "
                    f"chunks: {parent_result.doc_id!r}"
                )

            for evidence_rank, chunk_id in enumerate(chunk_ids, start=1):
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"duplicate representative chunk ID: {chunk_id!r}")

                metadata = self._chunks_by_id.get(chunk_id)

                if metadata is None:
                    raise ValueError(f"unknown representative chunk ID: {chunk_id!r}")

                if metadata.document_id != parent_result.doc_id:
                    raise ValueError(
                        f"chunk {chunk_id!r} belongs to {metadata.document_id!r}, "
                        f"not {parent_result.doc_id!r}"
                    )

                try:
                    document = self._chunk_store.get(chunk_id)
                except KeyError as error:
                    raise ValueError(
                        f"representative chunk is missing from the runtime corpus: {chunk_id!r}"
                    ) from error

                if document.text != metadata.text:
                    raise ValueError(
                        f"runtime text and chunk metadata differ for {chunk_id!r}"
                    )

                seen_chunk_ids.add(chunk_id)
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        parent_doc_id=parent_result.doc_id,
                        document_title=metadata.document_title,
                        text=metadata.text,
                        category=document.category,
                        section_path=metadata.section_path,
                        ordinal=metadata.ordinal,
                        start_char=metadata.start_char,
                        end_char=metadata.end_char,
                        parent_score=float(parent_result.score),
                        parent_rank=parent_result.rank,
                        evidence_rank=evidence_rank,
                    )
                )

        return retrieved_chunks

    @staticmethod
    def _validate_parent_results(results: Sequence[SearchResult]) -> None:
        seen_parent_ids: set[str] = set()

        for expected_rank, result in enumerate(results, start=1):
            if not result.doc_id.strip():
                raise ValueError("parent retriever returned an empty document ID")

            if result.doc_id in seen_parent_ids:
                raise ValueError(f"parent retriever returned duplicate ID: {result.doc_id!r}")

            if result.rank != expected_rank:
                raise ValueError(
                    "parent ranks must be consecutive starting at 1; "
                    f"expected {expected_rank}, received {result.rank}"
                )

            if not isinstance(result.score, Real) or not math.isfinite(float(result.score)):
                raise ValueError("parent score must be finite")

            seen_parent_ids.add(result.doc_id)
