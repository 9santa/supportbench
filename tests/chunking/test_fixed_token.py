import pytest
from conftest import WhitespaceTokenCodec

from supportbench.chunking.fixed_token import (
    FixedTokenChunker,
)
from supportbench.data.models import Document


def test_chunks_document_with_overlap() -> None:
    document = Document(
        doc_id="swg12345678",
        title="Example document",
        text=("zero one two three four five six seven eight nine"),
        category="technical_support",
    )

    chunker = FixedTokenChunker(
        token_codec=WhitespaceTokenCodec(),
        chunk_size=4,
        overlap=1,
    )

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "zero one two three",
        "three four five six",
        "six seven eight nine",
    ]

    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]

    assert [chunk.chunk_id for chunk in chunks] == [
        ("swg12345678::ft4o1::chunk_0000"),
        ("swg12345678::ft4o1::chunk_0001"),
        ("swg12345678::ft4o1::chunk_0002"),
    ]

    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]

    assert all(chunk.document_id == "swg12345678" for chunk in chunks)

    assert all(chunk.section_path == () for chunk in chunks)

    assert all(chunk.start_char is None and chunk.end_char is None for chunk in chunks)


def test_does_not_create_extra_terminal_chunk() -> None:
    document = Document(
        doc_id="doc-1",
        title="Title",
        text="one two three four",
        category="test",
    )

    chunker = FixedTokenChunker(
        token_codec=WhitespaceTokenCodec(),
        chunk_size=4,
        overlap=1,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == ("one two three four")


def test_returns_empty_list_for_no_tokens() -> None:
    document = Document(
        doc_id="doc-1",
        title="Title",
        text="   ",
        category="test",
    )

    chunker = FixedTokenChunker(
        token_codec=WhitespaceTokenCodec(),
        chunk_size=4,
        overlap=1,
    )

    assert chunker.chunk(document) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (
            0,
            0,
            "chunk_size must be positive",
        ),
        (
            4,
            -1,
            "overlap must be non-negative",
        ),
        (
            4,
            4,
            "overlap must be smaller",
        ),
        (
            4,
            5,
            "overlap must be smaller",
        ),
    ],
)
def test_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        FixedTokenChunker(
            token_codec=WhitespaceTokenCodec(),
            chunk_size=chunk_size,
            overlap=overlap,
        )
