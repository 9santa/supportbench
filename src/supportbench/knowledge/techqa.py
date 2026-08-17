from collections.abc import Mapping, Sequence
from typing import Protocol

from supportbench.chunking.base import TokenCodec
from supportbench.chunking.models import Chunk
from supportbench.knowledge.errors import (
    SupportChunkNotFoundError,
    SupportDocumentNotFoundError,
)
from supportbench.knowledge.models import (
    SupportDocumentMatch,
    SupportDocumentRead,
    SupportEvidenceChunk,
)
from supportbench.rag.models import RetrievedChunk
from supportbench.rag.retrieval import (
    ParentRetrievalRun,
)


class KnowledgeParentRetriever(Protocol):
    def retrieve(
        self,
        query: str,
    ) -> ParentRetrievalRun: ...


class KnowledgeChunkResolver(Protocol):
    def resolve(
        self,
        run: ParentRetrievalRun,
        *,
        query: str | None = None,
        top_k: int = 5,
        evidence_selection=None,
    ) -> list[RetrievedChunk]: ...


class TechQAKnowledgeService:
    def __init__(
        self,
        *,
        retrieval_service: KnowledgeParentRetriever,
        chunk_resolver: KnowledgeChunkResolver,
        chunks_by_id: Mapping[str, Chunk],
        token_codec: TokenCodec,
        search_top_parents: int = 4,
        search_snippet_tokens: int = 192,
        read_max_tokens: int = 2048,
        read_max_chunks: int = 8,
    ) -> None:
        if not chunks_by_id:
            raise ValueError("chunks_by_id must not be empty")

        if search_top_parents <= 0:
            raise ValueError("search_top_parents must be positive")

        if search_snippet_tokens <= 0:
            raise ValueError("search_snippet_tokens must be positive")

        if read_max_tokens <= 0:
            raise ValueError("read_max_tokens must be positive")

        if read_max_chunks <= 0:
            raise ValueError("read_max_chunks must be positive")

        self._retrieval_service = retrieval_service
        self._chunk_resolver = chunk_resolver
        self._chunks_by_id = dict(chunks_by_id)
        self._token_codec = token_codec

        self._search_top_parents = search_top_parents
        self._search_snippet_tokens = search_snippet_tokens
        self._read_max_tokens = read_max_tokens
        self._read_max_chunks = read_max_chunks

        chunks_by_parent: dict[str, list[Chunk]] = {}

        for chunk in self._chunks_by_id.values():
            chunks_by_parent.setdefault(chunk.document_id, []).append(chunk)

        self._chunks_by_parent = {
            document_id: tuple(
                sorted(
                    chunks,
                    key=lambda item: (item.ordinal, item.chunk_id),
                ),
            )
            for document_id, chunks in chunks_by_parent.items()
        }

    def search(
        self,
        *,
        query: str,
    ) -> tuple[SupportDocumentMatch, ...]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        retrieval = self._retrieval_service.retrieve(normalized_query)

        retrieved_chunks = self._chunk_resolver.resolve(
            retrieval,
            query=normalized_query,
            top_k=self._search_top_parents,
        )

        grouped: dict[str, list[RetrievedChunk]] = {}

        for chunk in retrieved_chunks:
            grouped.setdefault(chunk.parent_doc_id, []).append(chunk)

        matches: list[SupportDocumentMatch] = []

        for parent_id, chunks in grouped.items():
            first = chunks[0]

            matches.append(
                SupportDocumentMatch(
                    document_id=parent_id,
                    title=first.document_title,
                    rank=first.parent_rank,
                    evidence=tuple(
                        SupportEvidenceChunk(
                            chunk_id=chunk.chunk_id,
                            section_path=(chunk.section_path),
                            text=self._snippet(chunk.text),
                            truncated=(
                                self._is_truncated(
                                    chunk.text,
                                    self._search_snippet_tokens,
                                )
                            ),
                        )
                        for chunk in chunks
                    ),
                )
            )

        return tuple(
            sorted(
                matches,
                key=lambda item: item.rank,
            )
        )

    def _snippet(self, text: str) -> str:
        token_ids = self._token_codec.encode(text)

        if len(token_ids) <= self._search_snippet_tokens:
            return text

        return self._token_codec.decode(token_ids[: self._search_snippet_tokens]).strip()

    def _is_truncated(
        self,
        text: str,
        limit: int,
    ) -> bool:
        return len(self._token_codec.encode(text)) > limit

    def read(
        self,
        *,
        document_id: str,
        chunk_ids: tuple[str, ...] | None = None,
    ) -> SupportDocumentRead:
        normalized_id = document_id.strip()

        if not normalized_id:
            raise ValueError("document_id must be non-empty")

        document_chunks = self._chunks_by_parent.get(normalized_id)

        if document_chunks is None:
            raise SupportDocumentNotFoundError(document_id=normalized_id)

        if chunk_ids is None:
            selected = document_chunks
        else:
            requested = set(chunk_ids)

            selected = tuple(chunk for chunk in document_chunks if chunk.chunk_id in requested)

            found = {chunk.chunk_id for chunk in selected}

            missing = requested - found

            if missing:
                raise SupportChunkNotFoundError(
                    document_id=normalized_id,
                    chunk_id=sorted(missing)[0],
                )

        selected = selected[: self._read_max_chunks]

        packed: list[SupportEvidenceChunk] = []
        used_tokens = 0
        truncated: bool = bool(
            len(selected) < len(document_chunks) if chunk_ids is None else len(chunk_ids)
        )

        for chunk in selected:
            remaining = self._read_max_tokens - used_tokens

            if remaining <= 0:
                truncated = True
                break

            token_ids = self._token_codec.encode(chunk.text)

            chunk_truncated = len(token_ids) > remaining

            included_ids = token_ids[:remaining]

            text = self._token_codec.decode(included_ids).strip()

            if not text:
                break

            packed.append(
                SupportEvidenceChunk(
                    chunk_id=chunk.chunk_id,
                    section_path=chunk.section_path,
                    text=text,
                    truncated=chunk_truncated,
                )
            )

            used_tokens += len(included_ids)

            # Only the last chunk can be truncated, thus break
            if chunk_truncated:
                truncated = True
                break

        if not packed:
            raise RuntimeError("document read budget produced no readable chunks")

        return SupportDocumentRead(
            document_id=normalized_id,
            title=document_chunks[0].document_title,
            chunks=tuple(packed),
            truncated=truncated,
        )
