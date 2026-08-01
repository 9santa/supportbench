from supportbench.chunking.formatting import (
    format_chunk_for_embedding,
)
from supportbench.chunking.models import Chunk


def test_formats_title_and_text() -> None:
    chunk = Chunk(
        chunk_id="doc-1::ft4o1::chunk_0000",
        document_id="doc-1",
        document_title="IBM Product startup failure",
        text="Stop the deployment manager.",
        ordinal=0,
        token_count=4,
        section_path=(),
        start_char=None,
        end_char=None,
    )

    assert format_chunk_for_embedding(chunk) == (
        "Title: IBM Product startup failure\n\nStop the deployment manager."
    )


def test_formats_section_path() -> None:
    chunk = Chunk(
        chunk_id="doc-1::ha384::chunk_0000",
        document_id="doc-1",
        document_title="IBM Product startup failure",
        text="Remove the cached files.",
        ordinal=0,
        token_count=4,
        section_path=(
            "Troubleshooting",
            "Error ABC123",
        ),
        start_char=100,
        end_char=124,
    )

    assert format_chunk_for_embedding(chunk) == (
        "Title: IBM Product startup failure\n"
        "Section: Troubleshooting > Error ABC123"
        "\n\n"
        "Remove the cached files."
    )
