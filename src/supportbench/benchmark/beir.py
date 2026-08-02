import csv
import hashlib
import json
import os
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from supportbench.data.models import DatasetSplit, Document, QueryExample

BEIR_DATASET_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True, slots=True)
class BeirDatasetSpec:
    name: str
    archive_md5: str
    default_split: DatasetSplit

    @property
    def archive_url(self) -> str:
        return f"{BEIR_DATASET_BASE_URL}/{self.name}.zip"


BEIR_DATASETS: Mapping[str, BeirDatasetSpec] = MappingProxyType(
    {
        "scifact": BeirDatasetSpec(
            name="scifact",
            archive_md5="5f7d1de60b170fc8027bb7898e2efca1",
            default_split="test",
        ),
        "nfcorpus": BeirDatasetSpec(
            name="nfcorpus",
            archive_md5="a89dba18a62ef92f7d323ec890a0d38d",
            default_split="test",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class BeirDataset:
    name: str
    split: DatasetSplit
    documents: tuple[Document, ...]
    queries: tuple[QueryExample, ...]
    qrels: Mapping[str, Mapping[str, int]]


def download_beir_dataset(
    spec: BeirDatasetSpec,
    *,
    output_root: Path,
) -> Path:
    dataset_directory = output_root / spec.name

    if _is_dataset_directory(dataset_directory):
        return dataset_directory

    downloads = output_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / f"{spec.name}.zip"

    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(spec.archive_url, partial_path)
        os.replace(partial_path, archive_path)

    actual_md5 = _file_md5(archive_path)

    if actual_md5 != spec.archive_md5:
        raise ValueError(
            f"BEIR archive checksum mismatch for {spec.name!r}: "
            f"expected {spec.archive_md5}, got {actual_md5}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        _validate_archive_members(archive, output_root)
        archive.extractall(output_root)

    if not _is_dataset_directory(dataset_directory):
        raise ValueError(
            f"BEIR archive {archive_path} did not contain the expected {spec.name!r} dataset"
        )

    return dataset_directory


def load_beir_dataset(
    directory: Path,
    *,
    name: str,
    split: DatasetSplit,
) -> BeirDataset:
    documents = _load_corpus(directory / "corpus.jsonl", dataset_name=name)
    query_text_by_id = _load_queries(directory / "queries.jsonl")
    qrels = _load_qrels(directory / "qrels" / f"{split}.tsv")
    known_document_ids = {document.doc_id for document in documents}
    query_examples: list[QueryExample] = []

    for query_id, relevance_by_document in qrels.items():
        query_text = query_text_by_id.get(query_id)

        if query_text is None:
            raise ValueError(f"qrels reference unknown query ID: {query_id!r}")

        unknown_document_ids = set(relevance_by_document) - known_document_ids

        if unknown_document_ids:
            unknown = ", ".join(sorted(unknown_document_ids))
            raise ValueError(f"qrels reference unknown document IDs: {unknown}")

        relevant_document_ids = tuple(
            sorted(
                document_id
                for document_id, relevance in relevance_by_document.items()
                if relevance > 0
            )
        )

        if not relevant_document_ids:
            continue

        query_examples.append(
            QueryExample(
                query_id=query_id,
                query=query_text,
                relevant_doc_ids=relevant_document_ids,
                split=split,
            )
        )

    frozen_qrels = MappingProxyType(
        {
            query_id: MappingProxyType(dict(relevance_by_document))
            for query_id, relevance_by_document in qrels.items()
            if any(relevance > 0 for relevance in relevance_by_document.values())
        }
    )

    return BeirDataset(
        name=name,
        split=split,
        documents=tuple(documents),
        queries=tuple(query_examples),
        qrels=frozen_qrels,
    )


def _load_corpus(path: Path, *, dataset_name: str) -> list[Document]:
    documents: list[Document] = []
    seen_document_ids: set[str] = set()

    for line_number, record in _read_jsonl(path):
        document_id = _required_string(record, "_id", path=path, line_number=line_number)
        text = _required_string(record, "text", path=path, line_number=line_number)
        title_value = record.get("title", "")

        if not isinstance(title_value, str):
            raise ValueError(f"{path}:{line_number}: field 'title' must be a string")

        if document_id in seen_document_ids:
            raise ValueError(f"{path}:{line_number}: duplicate document ID {document_id!r}")

        seen_document_ids.add(document_id)
        documents.append(
            Document(
                doc_id=document_id,
                title=title_value.strip(),
                text=text,
                category=dataset_name,
            )
        )

    if not documents:
        raise ValueError(f"BEIR corpus is empty: {path}")

    return documents


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}

    for line_number, record in _read_jsonl(path):
        query_id = _required_string(record, "_id", path=path, line_number=line_number)
        text = _required_string(record, "text", path=path, line_number=line_number)

        if query_id in queries:
            raise ValueError(f"{path}:{line_number}: duplicate query ID {query_id!r}")

        queries[query_id] = text

    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)

    with path.open(mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        expected_fields = {"query-id", "corpus-id", "score"}

        if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
            raise ValueError(f"unexpected BEIR qrels header in {path}: {reader.fieldnames!r}")

        for line_number, row in enumerate(reader, start=2):
            query_id = row["query-id"].strip()
            document_id = row["corpus-id"].strip()

            if not query_id or not document_id:
                raise ValueError(f"{path}:{line_number}: qrels IDs must be non-empty")

            try:
                relevance = int(row["score"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: qrels score must be an integer"
                ) from error

            if relevance < 0:
                raise ValueError(f"{path}:{line_number}: qrels score must be non-negative")

            if document_id in qrels[query_id]:
                raise ValueError(
                    f"{path}:{line_number}: duplicate qrel for {query_id!r}/{document_id!r}"
                )

            qrels[query_id][document_id] = relevance

    return dict(qrels)


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []

    with path.open(mode="r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error

            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            records.append((line_number, cast(dict[str, Any], value)))

    return records


def _required_string(
    record: dict[str, Any],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: field {field!r} must be a non-empty string")

    return value.strip()


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)

    with path.open(mode="rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _validate_archive_members(archive: zipfile.ZipFile, output_root: Path) -> None:
    resolved_root = output_root.resolve()

    for member in archive.infolist():
        destination = (output_root / member.filename).resolve()

        if not destination.is_relative_to(resolved_root):
            raise ValueError(f"unsafe path in BEIR archive: {member.filename!r}")


def _is_dataset_directory(path: Path) -> bool:
    return (path / "corpus.jsonl").is_file() and (path / "queries.jsonl").is_file()
