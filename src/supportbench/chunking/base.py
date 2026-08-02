from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from typing import Protocol, Self, cast

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from supportbench.chunking.models import Chunk
from supportbench.data.models import Document


@dataclass(frozen=True, slots=True)
class TokenOffset:
    token_id: int
    start_char: int
    end_char: int

    def __post_init__(self) -> None:
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")

        if self.end_char < self.start_char:
            raise ValueError("end_char must not be smaller than start_char")


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str | list[str]:
        """Decode token IDs into text."""
        ...


class OffsetTokenCodec(TokenCodec, Protocol):
    def encode_with_offsets(
        self,
        text: str,
    ) -> list[TokenOffset]:
        """Encode text and preserve character offsets."""
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

    @property
    def configuration(self) -> Mapping[str, object]: ...

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """Chunk (split) one document deterministically."""
        ...


class HuggingFaceTokenCodec(TokenCodec):
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
            truncation=False,
            verbose=False,
        )

        return cast(list[int], encoded)

    def encode_with_offsets(self, text: str) -> list[TokenOffset]:
        if not self._tokenizer.is_fast:
            raise ValueError("character offsets require a fast Hugging Face tokenizer")

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            verbose=False,
        )

        token_ids = cast(list[int], encoded["input_ids"])
        raw_offsets = cast(list[tuple[int, int]], encoded["offset_mapping"])

        if len(token_ids) != len(raw_offsets):
            raise ValueError("token IDs and offset mappings have different lengths")

        return [
            TokenOffset(
                token_id=token_id,
                start_char=int(start_char),
                end_char=int(end_char),
            )
            for token_id, (
                start_char,
                end_char,
            ) in zip(
                token_ids,
                raw_offsets,
                strict=True,
            )
        ]

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
