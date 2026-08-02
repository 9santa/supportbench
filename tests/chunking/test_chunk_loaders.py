import json
from pathlib import Path

import pytest

from supportbench.chunking.loaders import (
    ChunkDatasetValidationError,
    load_chunks,
)


def _write_chunk(path: Path, **overrides: object) -> None:
    record: dict[str, object] = {
        "chunk_id": "doc::ft4o1::chunk_0000",
        "document_id": "doc",
        "document_title": "Document title",
        "text": "Chunk text",
        "ordinal": 0,
        "token_count": 2,
        "section_path": ["Troubleshooting", "Resolution"],
        "start_char": 10,
        "end_char": 20,
    }
    record.update(overrides)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_loads_complete_chunk_metadata(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_chunk(path)

    chunk = load_chunks(path)["doc::ft4o1::chunk_0000"]

    assert chunk.document_id == "doc"
    assert chunk.document_title == "Document title"
    assert chunk.section_path == ("Troubleshooting", "Resolution")
    assert (chunk.start_char, chunk.end_char) == (10, 20)


def test_rejects_invalid_section_path(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    _write_chunk(path, section_path=[""])

    with pytest.raises(ChunkDatasetValidationError, match="section_path"):
        load_chunks(path)
