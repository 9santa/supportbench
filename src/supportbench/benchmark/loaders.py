from pathlib import Path
from typing import Any, cast

from supportbench.benchmark.models import (
    Answerability,
    BenchmarkQuery,
)
from supportbench.data.loaders import (
    ALLOWED_SPLITS,
    DatasetValidationError,
    _read_jsonl,
    _require_non_empty_string,
)
from supportbench.data.models import DatasetSplit


def load_benchmark_queries(
    path: Path,
    *,
    known_doc_ids: set[str],
) -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []
    seen_query_ids: set[str] = set()

    for line_num, obj in _read_jsonl(path):
        query_id = _require_non_empty_string(
            obj,
            "query_id",
            path=path,
            line_num=line_num,
        )

        query = _require_non_empty_string(
            obj,
            "query",
            path=path,
            line_num=line_num,
        )

        if query_id in seen_query_ids:
            raise DatasetValidationError(f"{path}:{line_num}: duplicate query_id {query_id!r}")

        seen_query_ids.add(query_id)

        split = _load_split(
            obj,
            path=path,
            line_num=line_num,
        )

        relevant_doc_ids = _load_relevant_doc_ids(
            obj,
            known_doc_ids=known_doc_ids,
            path=path,
            line_num=line_num,
        )

        is_impossible = _load_is_impossible(
            obj,
            path=path,
            line_num=line_num,
        )

        answer = _load_answer(
            obj,
            path=path,
            line_num=line_num,
        )

        if is_impossible:
            answerability = "unanswerable"
            reference_answer = None
        else:
            answerability = "answerable"
            reference_answer = answer.strip() or None

        try:
            benchmark_query = BenchmarkQuery(
                query_id=query_id,
                query=query,
                relevant_doc_ids=(relevant_doc_ids),
                split=split,
                answerability=cast(Answerability, answerability),
                reference_answer=reference_answer,
            )
        except ValueError as error:
            raise DatasetValidationError(f"{path}:{line_num}: {error}") from error

        queries.append(benchmark_query)

    return queries


def _load_split(
    obj: dict[str, Any],
    *,
    path: Path,
    line_num: int,
) -> DatasetSplit:
    value = _require_non_empty_string(
        obj,
        "split",
        path=path,
        line_num=line_num,
    )

    if value not in ALLOWED_SPLITS:
        raise DatasetValidationError(
            f"{path}:{line_num}: invalid split {value!r}; must be one of {ALLOWED_SPLITS}"
        )

    return cast(DatasetSplit, value)


def _load_relevant_doc_ids(
    obj: dict[str, Any],
    *,
    known_doc_ids: set[str],
    path: Path,
    line_num: int,
) -> tuple[str, ...]:
    value = obj.get("relevant_doc_ids")

    if not isinstance(value, list):
        raise DatasetValidationError(f"{path}:{line_num}: 'relevant_doc_ids' must be a list")

    document_ids: list[str] = []

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise DatasetValidationError(
                f"{path}:{line_num}: item {index} in relevant_doc_ids must be a string"
            )

        document_id = item.strip()

        if not document_id:
            raise DatasetValidationError(
                f"{path}:{line_num}: item {index} in relevant_doc_ids must be non-empty"
            )

        document_ids.append(document_id)

    if len(document_ids) != len(set(document_ids)):
        raise DatasetValidationError(
            f"{path}:{line_num}: relevant_doc_ids must not contain duplicates"
        )

    unknown_ids = [document_id for document_id in document_ids if document_id not in known_doc_ids]

    if unknown_ids:
        raise DatasetValidationError(
            f"{path}:{line_num}: "
            "unknown relevant document IDs: "
            + ", ".join(repr(document_id) for document_id in unknown_ids)
        )

    return tuple(document_ids)


def _load_is_impossible(
    obj: dict[str, Any],
    *,
    path: Path,
    line_num: int,
) -> bool:
    value = obj.get("is_impossible")

    if not isinstance(value, bool):
        raise DatasetValidationError(f"{path}:{line_num}: 'is_impossible' must be a boolean")

    return value


def _load_answer(
    obj: dict[str, Any],
    *,
    path: Path,
    line_num: int,
) -> str:
    value = obj.get("answer")

    if not isinstance(value, str):
        raise DatasetValidationError(f"{path}:{line_num}: 'answer' must be a string")

    return value
