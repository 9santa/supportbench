import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from supportbench.chunking.base import (
    Chunker,
    TokenCodec,
)
from supportbench.chunking.formatting import (
    format_chunk_for_embedding,
)
from supportbench.chunking.models import Chunk
from supportbench.chunking.statistics import (
    ChunkingStatistics,
    build_chunking_statistics,
)
from supportbench.data.models import Document
from supportbench.retrieval.tokenization import (
    tokenize,
)


@dataclass(frozen=True, slots=True)
class ChunkCorpusBuildResult:
    output_directory: Path
    chunks_path: Path
    documents_path: Path
    statistics_path: Path
    manifest_path: Path
    statistics: ChunkingStatistics


def build_chunk_corpus(
    *,
    documents: Sequence[Document],
    chunker: Chunker,
    token_codec: TokenCodec,
    tokenizer_name: str,
    source_documents_path: Path,
    output_directory: Path,
    max_input_tokens: int = 512,
    special_token_reserve: int = 2,
) -> ChunkCorpusBuildResult:
    if not documents:
        raise ValueError("cannot build chunks from an empty corpus")

    if not tokenizer_name.strip():
        raise ValueError("tokenizer_name must be non-empty")

    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    if special_token_reserve < 0:
        raise ValueError("special_token_reserve must be non-negative")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_path = output_directory / "chunks.jsonl"
    documents_path = output_directory / "documents.jsonl"
    statistics_path = output_directory / "statistics.json"
    manifest_path = output_directory / "manifest.json"

    chunks_per_document: list[int] = []
    body_token_counts: list[int] = []
    formatted_token_counts: list[int] = []

    indexable_empty_chunks = 0
    seen_chunk_ids: set[str] = set()

    section_paths: list[tuple[str, ...]] = []

    with (
        chunks_path.open(
            mode="w",
            encoding="utf-8",
        ) as chunks_file,
        documents_path.open(
            mode="w",
            encoding="utf-8",
        ) as documents_file,
    ):
        for document in documents:
            chunks = chunker.chunk(document)

            chunks_per_document.append(len(chunks))

            for chunk in chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    raise ValueError(f"duplicate chunk_id generated: {chunk.chunk_id!r}")

                seen_chunk_ids.add(chunk.chunk_id)

                formatted_text = format_chunk_for_embedding(chunk)

                formatted_token_count = len(token_codec.encode(formatted_text))

                body_token_counts.append(chunk.token_count)
                formatted_token_counts.append(formatted_token_count)

                if not tokenize(f"{chunk.document_title} {chunk.text}"):
                    indexable_empty_chunks += 1

                section_paths.append(chunk.section_path)

                _write_jsonl_record(
                    chunks_file,
                    _chunk_to_record(
                        chunk,
                        category=document.category,
                        formatted_token_count=(formatted_token_count),
                    ),
                )

                _write_jsonl_record(
                    documents_file,
                    _chunk_to_runtime_document_record(
                        chunk,
                        category=document.category,
                    ),
                )

    statistics = build_chunking_statistics(
        chunks_per_document=(chunks_per_document),
        body_token_counts=body_token_counts,
        formatted_token_counts=(formatted_token_counts),
        max_input_tokens=max_input_tokens,
        special_token_reserve=(special_token_reserve),
        indexable_empty_chunks=(indexable_empty_chunks),
        section_paths=section_paths,
    )

    _write_json(
        statistics_path,
        asdict(statistics),
    )

    manifest = {
        "source": {
            "documents_path": str(source_documents_path),
            "documents_sha256": _sha256_file(source_documents_path),
            "document_count": len(documents),
        },
        "chunking": {
            "chunking_key": (chunker.chunking_key),
            "parameters": dict(chunker.configuration),
            "tokenizer": tokenizer_name,
            "add_special_tokens": False,
            "max_input_tokens": (max_input_tokens),
            "special_token_reserve": (special_token_reserve),
            "chunk_id_format": ("{document_id}::{chunking_key}::chunk_{ordinal:04d}"),
            "character_offsets": ("not_available_for_fixed_token"),
        },
        "formatting": {
            "chunk_text_contains_title": False,
            "embedding_format": ("title_section_blankline_text"),
            "runtime_document_title": ("source document title"),
            "runtime_document_text": ("chunk body only"),
        },
        "outputs": {
            "chunks": "chunks.jsonl",
            "runtime_documents": ("documents.jsonl"),
            "statistics": "statistics.json",
        },
        "statistics": asdict(statistics),
    }

    _write_json(
        manifest_path,
        manifest,
    )

    return ChunkCorpusBuildResult(
        output_directory=output_directory,
        chunks_path=chunks_path,
        documents_path=documents_path,
        statistics_path=statistics_path,
        manifest_path=manifest_path,
        statistics=statistics,
    )


def _chunk_to_record(
    chunk: Chunk,
    *,
    category: str,
    formatted_token_count: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_title": (chunk.document_title),
        "text": chunk.text,
        "category": category,
        "ordinal": chunk.ordinal,
        "token_count": chunk.token_count,
        "formatted_token_count": (formatted_token_count),
        "section_path": list(chunk.section_path),
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def _chunk_to_runtime_document_record(
    chunk: Chunk,
    *,
    category: str,
) -> dict[str, Any]:
    """
    Create a record compatible with the existing load_documents().

    Additional parent/chunk fields are intentionally preserved;
    the old loader ignores them.
    """
    return {
        "doc_id": chunk.chunk_id,
        "title": chunk.document_title,
        "text": chunk.text,
        "category": category,
        "parent_document_id": (chunk.document_id),
        "ordinal": chunk.ordinal,
        "section_path": list(chunk.section_path),
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def _write_jsonl_record(
    file: Any,
    record: dict[str, Any],
) -> None:
    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )
    file.write("\n")


def _write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()
