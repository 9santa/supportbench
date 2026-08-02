import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from supportbench.chunking.models import Chunk


class ChunkDatasetValidationError(ValueError):
    """Raised when a chunk dataset is invalid."""


def load_chunk_parent_ids(
    path: Path,
) -> dict[str, str]:
    parent_by_chunk_id: dict[str, str] = {}

    for line_num, record in _read_jsonl(path):
        chunk_id = _require_non_empty_string(
            record,
            "chunk_id",
            path=path,
            line_num=line_num,
        )
        document_id = _require_non_empty_string(
            record,
            "document_id",
            path=path,
            line_num=line_num,
        )

        if chunk_id in parent_by_chunk_id:
            raise ChunkDatasetValidationError(f"{path}:{line_num}: duplicate chunk_id {chunk_id!r}")

        parent_by_chunk_id[chunk_id] = document_id

    if not parent_by_chunk_id:
        raise ChunkDatasetValidationError(f"{path}: chunk metadata is empty")

    return parent_by_chunk_id


def load_chunks(path: Path) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}

    for line_num, record in _read_jsonl(path):
        chunk_id = _require_non_empty_string(
            record,
            "chunk_id",
            path=path,
            line_num=line_num,
        )

        if chunk_id in chunks:
            raise ChunkDatasetValidationError(f"{path}:{line_num}: duplicate chunk_id {chunk_id!r}")

        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            document_id=_require_non_empty_string(
                record,
                "document_id",
                path=path,
                line_num=line_num,
            ),
            document_title=_require_non_empty_string(
                record,
                "document_title",
                path=path,
                line_num=line_num,
            ),
            text=_require_non_empty_string(
                record,
                "text",
                path=path,
                line_num=line_num,
            ),
            ordinal=_require_non_negative_int(
                record,
                "ordinal",
                path=path,
                line_num=line_num,
            ),
            token_count=_require_positive_int(
                record,
                "token_count",
                path=path,
                line_num=line_num,
            ),
            section_path=_require_section_path(
                record,
                path=path,
                line_num=line_num,
            ),
            start_char=_require_optional_non_negative_int(
                record,
                "start_char",
                path=path,
                line_num=line_num,
            ),
            end_char=_require_optional_non_negative_int(
                record,
                "end_char",
                path=path,
                line_num=line_num,
            ),
        )

    if not chunks:
        raise ChunkDatasetValidationError(f"{path}: chunk metadata is empty")

    return chunks


def _read_jsonl(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        for line_num, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ChunkDatasetValidationError(
                    f"{path}:{line_num}: invalid JSON: {error}"
                ) from error

            if not isinstance(value, dict):
                raise ChunkDatasetValidationError(
                    f"{path}:{line_num}: each line must be a JSON object"
                )

            yield line_num, value


def _require_non_empty_string(
    record: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_num: int,
) -> str:
    value = record.get(field)

    if not isinstance(value, str):
        raise ChunkDatasetValidationError(f"{path}:{line_num}: {field!r} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ChunkDatasetValidationError(f"{path}:{line_num}: {field!r} must be non-empty")

    return normalized


def _require_non_negative_int(
    record: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_num: int,
) -> int:
    value = record.get(field)

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChunkDatasetValidationError(
            f"{path}:{line_num}: {field!r} must be a non-negative integer"
        )

    return value


def _require_positive_int(
    record: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_num: int,
) -> int:
    value = _require_non_negative_int(
        record,
        field,
        path=path,
        line_num=line_num,
    )

    if value == 0:
        raise ChunkDatasetValidationError(
            f"{path}:{line_num}: {field!r} must be a positive integer"
        )

    return value


def _require_optional_non_negative_int(
    record: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_num: int,
) -> int | None:
    value = record.get(field)

    if value is None:
        return None

    return _require_non_negative_int(
        record,
        field,
        path=path,
        line_num=line_num,
    )


def _require_section_path(
    record: dict[str, Any],
    *,
    path: Path,
    line_num: int,
) -> tuple[str, ...]:
    value = record.get("section_path")

    if not isinstance(value, list):
        raise ChunkDatasetValidationError(
            f"{path}:{line_num}: 'section_path' must be a list"
        )

    sections: list[str] = []

    for index, section in enumerate(value):
        if not isinstance(section, str) or not section.strip():
            raise ChunkDatasetValidationError(
                f"{path}:{line_num}: section_path[{index}] must be a non-empty string"
            )

        sections.append(section.strip())

    return tuple(sections)
