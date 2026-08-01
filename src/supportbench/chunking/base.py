from collections.abc import Sequence
from typing import Protocol, Self, cast

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from supportbench.chunking.models import Chunk
from supportbench.data.models import Document


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into text."""
        ...


class Chunker(Protocol):
    @property
    def chunking_key(self) -> str:
        """
        Stable identifier for the chunking configuration.
        Example: `ft384o64` means:
            ft = fixed-token,
            384 = chunk size,
            o64 = overlap 64
        It is used inside Chunk.chunk_id: swg21996508::ft384o64::chunk_0003.
        So the same document can be chunked in different ways without chunk_id collisions.
        """
        ...

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """Chunk (split) one document deterministically."""
        ...


class HuggingFaceTokenCodec:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
    ) -> Self:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        encoded = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return cast(list[int], encoded)

    def decode(self, token_ids: Sequence[int]) -> str | list[str]:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def build_chunk_id(
    *,
    document_id: str,
    chunking_key: str,
    ordinal: int,
) -> str:
    normalized_document_id = document_id.strip()
    normalized_key = chunking_key.strip()

    if not normalized_document_id:
        raise ValueError("document_id must be non-empty")

    if not normalized_key:
        raise ValueError("chunking_key must be non-empty")

    if "::" in normalized_key:
        raise ValueError("chunking_key must not contain '::'")

    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")

    return f"{normalized_document_id}::{normalized_key}::chunk_{ordinal:04d}"
