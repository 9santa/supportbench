import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from supportbench.data.models import DatasetSplit, Document, QueryExample


class DatasetValidationError(ValueError):
    """Raised when a dataset file violates the SupportBench schema."""


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Yield (line_number, parsed_object) for every non-empty line in JSONL file.
    """
    with path.open(mode="r", encoding="utf-8") as file:
        for line_num, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:  # allow empty lines (skip them)
                continue

            # Must be valid JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"{path}:{line_num}: invalid JSON - {exc}") from exc

            if not isinstance(obj, dict):
                raise DatasetValidationError(
                    f"{path}:{line_num}: each non-empty line must be a JSON object"
                )

            yield line_num, obj


def _require_non_empty_string(
    obj: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_num: int,
) -> str:
    if field not in obj:
        raise DatasetValidationError(f"{path}:{line_num}: missing required field '{field}'")

    val = obj[field]

    # Check the field is a string
    if not isinstance(val, str):
        raise DatasetValidationError(f"{path}:{line_num}: field '{field}' must be a string")

    # Strip and check it is not empty
    stripped = val.strip()
    if not stripped:
        raise DatasetValidationError(f"{path}:{line_num}: field '{field}' must be non-empty")

    return stripped


REQUIRED_DOCUMENT_FIELDS = frozenset(["doc_id", "title", "text", "category"])
REQUIRED_QUERY_FIELDS = frozenset(["query_id", "query", "relevant_doc_ids", "split"])
ALLOWED_SPLITS = ("train", "dev", "test", "frozen_test")


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()

    for line_num, obj in _read_jsonl(path):
        for field in REQUIRED_DOCUMENT_FIELDS:
            obj[field] = _require_non_empty_string(obj, field, path=path, line_num=line_num)

        doc_id = obj["doc_id"]

        # Duplicate id check
        if doc_id in seen_ids:
            raise DatasetValidationError(f"{path}:{line_num}: duplicate doc_id '{doc_id}'")
        seen_ids.add(doc_id)

        documents.append(
            Document(
                doc_id=doc_id,
                title=obj["title"],
                text=obj["text"],
                category=obj["category"],
            )
        )

    return documents


def load_queries(
    path: Path,
    known_doc_ids: set[str],
) -> list[QueryExample]:
    queries: list[QueryExample] = []
    seen_qids: set[str] = set()

    for line_num, obj in _read_jsonl(path):
        for field in ("query_id", "query"):
            obj[field] = _require_non_empty_string(obj, field, path=path, line_num=line_num)

        qid = obj["query_id"]
        if qid in seen_qids:
            raise DatasetValidationError(f"{path}:{line_num}: duplicate query_id '{qid}'")
        seen_qids.add(qid)

        if "relevant_doc_ids" not in obj:
            raise DatasetValidationError(
                f"{path}:{line_num}: missing required field 'relevant_doc_ids'"
            )
        ids_list = obj["relevant_doc_ids"]
        if not isinstance(ids_list, list):
            raise DatasetValidationError(f"{path}:{line_num}: 'relevant_doc_ids' must be a list")

        # Empty labels represent unsupported or unanswerable benchmark queries.

        # Verify ids are strings and non-empty
        doc_ids: list[str] = []
        for i, item in enumerate(ids_list):
            if not isinstance(item, str):
                raise DatasetValidationError(
                    f"{path}:{line_num}: item {i} in 'relevant_doc_ids' must be a string"
                )
            stripped_item = item.strip()
            if not stripped_item:
                raise DatasetValidationError(
                    f"{path}:{line_num}: item {i} in 'relevant_doc_ids' must not be empty"
                )
            doc_ids.append(stripped_item)

        # Duplicates within a list
        if len(set(doc_ids)) != len(doc_ids):
            seen = set()
            for d in doc_ids:
                if d in seen:
                    raise DatasetValidationError(
                        f"{path}:{line_num}: duplicate doc_id '{d}' in relevant_doc_ids"
                    )
                seen.add(d)

        # Existance relevant_doc_ids in the known_doc_ids database
        missing = [d for d in doc_ids if d not in known_doc_ids]
        if missing:
            raise DatasetValidationError(
                f"{path}:{line_num}: unknown doc_id(s) in relevant_doc_ids: "
                f"{', '.join(repr(m) for m in missing)}"
            )

        if "split" not in obj:
            raise DatasetValidationError(f"{path}:{line_num}: missing required field 'split'")

        split_val = obj["split"]
        if not isinstance(split_val, str):
            raise DatasetValidationError(f"{path}:{line_num}: field 'split' must be a string")
        split_val = split_val.strip()

        if split_val not in ALLOWED_SPLITS:
            raise DatasetValidationError(
                f"{path}:{line_num}: invalid split '{split_val}'; must be one of {ALLOWED_SPLITS}"
            )

        queries.append(
            QueryExample(
                query_id=qid,
                query=obj["query"],
                relevant_doc_ids=tuple(doc_ids),
                split=cast(DatasetSplit, split_val),
            )
        )

    return queries
