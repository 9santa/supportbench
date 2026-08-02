from collections.abc import Mapping

from supportbench.chunking.base import (
    TokenCodec,
    build_chunk_id,
)
from supportbench.chunking.models import Chunk
from supportbench.data.models import Document


class FixedTokenChunker:
    def __init__(
        self,
        *,
        token_codec: TokenCodec,
        chunk_size: int = 384,
        overlap: int = 64,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._token_codec = token_codec
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    @property
    def chunking_key(self) -> str:
        return f"ft{self._chunk_size}o{self._overlap}"

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "strategy": "fixed_token",
            "version": 1,
            "chunk_size": self._chunk_size,
            "overlap": self._overlap,
        }

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        token_ids = self._token_codec.encode(document.text)

        if not token_ids:
            return []

        stride = self._chunk_size - self._overlap

        chunks: list[Chunk] = []
        start = 0
        ordinal = 0

        while start < len(token_ids):
            end = min(start + self._chunk_size, len(token_ids))

            chunk_token_ids = token_ids[start:end]

            chunk_text = self._token_codec.decode(chunk_token_ids)
            if isinstance(chunk_text, str):
                chunk_text = chunk_text.strip()

            if not chunk_text:
                raise ValueError(
                    "token codec decoded a non-empty "
                    "token range into empty text for "
                    f"document {document.doc_id!r}"
                )

            chunks.append(
                Chunk(
                    chunk_id=build_chunk_id(
                        document_id=document.doc_id,
                        chunking_key=self.chunking_key,
                        ordinal=ordinal,
                    ),
                    document_id=document.doc_id,
                    document_title=document.title,
                    text=chunk_text,
                    ordinal=ordinal,
                    token_count=len(chunk_token_ids),
                    section_path=(),
                    start_char=None,
                    end_char=None,
                )
            )

            if end == len(token_ids):
                break

            start += stride
            ordinal += 1

        return chunks
