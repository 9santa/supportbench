import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


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
