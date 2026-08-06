import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import ZipFile

type DatasetSplit = Literal["train", "dev"]


class NvidiaTechQAError(RuntimeError):
    """Raised when the source archives violate the expected dataset contract."""


@dataclass(frozen=True, slots=True)
class NvidiaTechQADocument:
    doc_id: str
    title: str
    text: str
    category: str
    source: str
    source_filename: str
    license_name: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class NvidiaTechQAQuery:
    query_id: str
    split: DatasetSplit
    query: str
    answer: str
    is_impossible: bool
    relevant_doc_ids: tuple[str, ...]
    source_context_filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    document_count: int
    query_count: int
    answerable_query_count: int
    impossible_query_count: int
    train_query_count: int
    dev_query_count: int
    unique_gold_document_count: int
    duplicate_question_group_count: int
    conflicting_answerability_group_count: int
    answerable_empty_answer_count: int
    answerable_answer_not_in_context_count: int


@dataclass(frozen=True, slots=True)
class _SourceContext:
    filename: str
    text: str


@dataclass(frozen=True, slots=True)
class _SourceQuery:
    query_id: str
    question: str
    answer: str
    is_impossible: bool
    contexts: tuple[_SourceContext, ...]


def prepare_nvidia_techqa(
    *,
    dataset_zip: Path,
    corpus_zip: Path,
    output_dir: Path,
) -> PreparationSummary:
    """Normalize Nvidia TechQA without silently repairing labels."""
    source_queries = _load_source_queries(dataset_zip)
    documents, corpus_text_by_filename = _load_documents(corpus_zip)

    _validate_context_integrity(source_queries, corpus_text_by_filename)

    queries = tuple(_to_query(query) for query in source_queries)
    anomaly_report = _build_anomaly_report(source_queries)
    summary = _build_summary(documents, queries, anomaly_report)

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_dir / "documents.jsonl", (asdict(document) for document in documents))

    _write_jsonl(output_dir / "queries.jsonl", (_query_to_dict(query) for query in queries))

    _write_json(output_dir / "anomalies.json", anomaly_report)

    _write_json(
        output_dir / "manifest.json",
        {
            "dataset": "nvidia/TechQA-RAG-Eval",
            "source_dataset_zip_sha256": _sha256_file(dataset_zip),
            "source_corpus_zip_sha256": _sha256_file(corpus_zip),
            "summary": asdict(summary),
            "outputs": {
                "documents": "documents.jsonl",
                "queries": "queries.jsonl",
                "anomalies": "anomalies.json",
            },
        },
    )

    return summary


def _load_source_queries(dataset_zip: Path) -> tuple[_SourceQuery, ...]:
    with ZipFile(dataset_zip) as archive:
        train_json = next(
            name for name in archive.namelist() if PurePosixPath(name).name == "train.json"
        )
        raw = json.loads(archive.read(train_json))

    return tuple(
        _SourceQuery(
            query_id=item["id"],
            question=item["question"],
            answer=item["answer"],
            is_impossible=item["is_impossible"],
            contexts=tuple(
                _SourceContext(
                    filename=PurePosixPath(context["filename"]).name,
                    text=context["text"],
                )
                for context in item["contexts"]
            ),
        )
        for item in raw
    )


def _load_documents(
    corpus_zip: Path,
) -> tuple[tuple[NvidiaTechQADocument, ...], dict[str, str]]:
    documents = []
    text_by_filename = {}

    with ZipFile(corpus_zip) as archive:
        for archive_name in sorted(name for name in archive.namelist() if name.endswith(".txt")):
            filename = PurePosixPath(archive_name).name
            source_text = archive.read(archive_name).decode("utf-8")

            documents.append(_parse_document(filename, source_text))
            text_by_filename[filename] = source_text

    return tuple(documents), text_by_filename


def _parse_document(
    filename: str,
    source_text: str,
) -> NvidiaTechQADocument:
    title, body = source_text.split("\n\nText:\n", maxsplit=1)

    return NvidiaTechQADocument(
        doc_id=Path(filename).stem,
        title=title.removeprefix("Title: ").strip(),
        text=body.strip(),
        category="technical_support",
        source="nvidia_techqa_rag_eval",
        source_filename=filename,
        license_name="Apache-2.0",
        content_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
    )


def _validate_context_integrity(
    source_queries: tuple[_SourceQuery, ...],
    corpus_text_by_filename: dict[str, str],
) -> None:
    for query in source_queries:
        for context in query.contexts:
            if corpus_text_by_filename.get(context.filename) != context.text:
                raise NvidiaTechQAError(
                    f"Query {query.query_id}: context {context.filename} does not match the corpus"
                )


def _to_query(source_query: _SourceQuery) -> NvidiaTechQAQuery:
    filenames = tuple(context.filename for context in source_query.contexts)

    return NvidiaTechQAQuery(
        query_id=source_query.query_id,
        split=_split_from_query_id(source_query.query_id),
        query=source_query.question.strip(),
        answer=source_query.answer,
        is_impossible=source_query.is_impossible,
        relevant_doc_ids=tuple(Path(filename).stem for filename in filenames),
        source_context_filenames=filenames,
    )


def _split_from_query_id(query_id: str) -> DatasetSplit:
    if query_id.startswith("DEV_"):
        return "dev"

    if query_id.startswith("TRAIN_"):
        return "train"

    raise NvidiaTechQAError(f"query ID must start with 'TRAIN_' or 'DEV_': {query_id!r}")


def _build_anomaly_report(source_queries: tuple[_SourceQuery, ...]) -> dict[str, Any]:
    groups: dict[str, list[_SourceQuery]] = defaultdict(list)

    for query in source_queries:
        groups[_normalize(query.question)].append(query)

    duplicate_groups = []
    # Groups with impossible and possible at the same time
    conflicting_groups = []

    for normalized_question, group in sorted(groups.items()):
        if len(group) < 2:
            continue

        item = {
            "normalized_question": normalized_question,
            "query_ids": [query.query_id for query in group],
            "is_impossible": [query.is_impossible for query in group],
            "relevant_filenames": [
                [context.filename for context in query.contexts] for query in group
            ],
        }

        duplicate_groups.append(item)

        answerability_values = {query.is_impossible for query in group}

        if len(answerability_values) > 1:
            conflicting_groups.append(item)

    answerable = [query for query in source_queries if not query.is_impossible]

    return {
        "duplicate_question_groups": duplicate_groups,
        "conflicting_answerability_groups": conflicting_groups,
        "answerable_empty_answer_query_ids": sorted(
            query.query_id for query in answerable if not query.answer.strip()
        ),
        "answerable_answer_not_in_context_query_ids": sorted(
            query.query_id
            for query in answerable
            if query.answer.strip()
            and not any(
                _normalize(query.answer) in _normalize(context.text) for context in query.contexts
            )
        ),
        "policy": {
            "source_rows_modified": False,
            "note": (
                "All source queries are preserved in queries.jsonl. "
                "Anomalies are reported but not silently repaired."
            ),
        },
    }


def _build_summary(
    documents: tuple[NvidiaTechQADocument, ...],
    queries: tuple[NvidiaTechQAQuery, ...],
    anomaly_report: dict[str, Any],
) -> PreparationSummary:
    split_counts = Counter(query.split for query in queries)
    answerable_count = sum(not query.is_impossible for query in queries)
    gold_ids = {doc_id for query in queries for doc_id in query.relevant_doc_ids}

    return PreparationSummary(
        document_count=len(documents),
        query_count=len(queries),
        answerable_query_count=answerable_count,
        impossible_query_count=len(queries) - answerable_count,
        train_query_count=split_counts["train"],
        dev_query_count=split_counts["dev"],
        unique_gold_document_count=len(gold_ids),
        duplicate_question_group_count=len(anomaly_report["duplicate_question_groups"]),
        conflicting_answerability_group_count=len(
            anomaly_report["conflicting_answerability_groups"]
        ),
        answerable_empty_answer_count=len(anomaly_report["answerable_empty_answer_query_ids"]),
        answerable_answer_not_in_context_count=len(
            anomaly_report["answerable_answer_not_in_context_query_ids"]
        ),
    )


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _query_to_dict(query: NvidiaTechQAQuery) -> dict[str, object]:
    value = asdict(query)
    value["relevant_doc_ids"] = list(query.relevant_doc_ids)
    value["source_context_filenames"] = list(query.source_context_filenames)
    return value


def _write_jsonl(path: Path, records: Any) -> None:
    with path.open(mode="w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
