import re
from collections.abc import Sequence

from supportbench.chunking.base import (
    TokenOffset,
)
from supportbench.chunking.heading_aware import (
    HeadingAwareChunker,
)
from supportbench.data.models import Document


class WhitespaceOffsetCodec:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(
        self,
        text: str,
    ) -> list[int]:
        return [token.token_id for token in self.encode_with_offsets(text)]

    def encode_with_offsets(
        self,
        text: str,
    ) -> list[TokenOffset]:
        result: list[TokenOffset] = []

        for match in re.finditer(
            r"\S+",
            text,
        ):
            value = match.group(0)

            token_id = self._token_to_id.get(value)

            if token_id is None:
                token_id = len(self._token_to_id)
                self._token_to_id[value] = token_id
                self._id_to_token[token_id] = value

            result.append(
                TokenOffset(
                    token_id=token_id,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )

        return result

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def test_preserves_sections_and_offsets() -> None:
    text = (
        "Introductory text.\n\n"
        "Troubleshooting\n\n"
        "Stop the deployment manager.\n\n"
        "Error ABC123:\n\n"
        "Remove the cached files."
    )

    document = Document(
        doc_id="doc-1",
        title="IBM Product startup failure",
        text=text,
        category="support",
    )

    chunker = HeadingAwareChunker(
        token_codec=WhitespaceOffsetCodec(),
        target_tokens=20,
        oversized_overlap=4,
        max_input_tokens=64,
        special_token_reserve=2,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3

    assert chunks[0].section_path == ()
    assert chunks[0].text == ("Introductory text.")

    assert chunks[1].section_path == ("Troubleshooting",)
    assert chunks[1].text == ("Stop the deployment manager.")

    assert chunks[2].section_path == (
        "Troubleshooting",
        "Error ABC123",
    )
    assert chunks[2].text == ("Remove the cached files.")

    for chunk in chunks:
        assert chunk.start_char is not None
        assert chunk.end_char is not None

        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_packs_paragraphs_in_same_section() -> None:
    text = "Cause\n\nFirst paragraph has three tokens.\n\nSecond paragraph has three tokens."

    document = Document(
        doc_id="doc-1",
        title="Example",
        text=text,
        category="support",
    )

    chunker = HeadingAwareChunker(
        token_codec=WhitespaceOffsetCodec(),
        target_tokens=20,
        oversized_overlap=4,
        max_input_tokens=64,
        special_token_reserve=2,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section_path == ("Cause",)

    assert chunks[0].text == (
        "First paragraph has three tokens.\n\nSecond paragraph has three tokens."
    )


def test_splits_only_oversized_paragraph() -> None:
    text = "Troubleshooting\n\nzero one two three four five six seven eight nine"

    document = Document(
        doc_id="doc-1",
        title="Example",
        text=text,
        category="support",
    )

    chunker = HeadingAwareChunker(
        token_codec=WhitespaceOffsetCodec(),
        target_tokens=4,
        oversized_overlap=1,
        max_input_tokens=32,
        special_token_reserve=2,
    )

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "zero one two three",
        "three four five six",
        "six seven eight nine",
    ]

    assert all(chunk.section_path == ("Troubleshooting",) for chunk in chunks)

    for chunk in chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
