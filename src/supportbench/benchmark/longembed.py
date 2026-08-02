import hashlib
import json
import os
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from supportbench.data.models import Document, QueryExample

LONGEMBED_REVISION = "10039a580487dacecf79db69166e17ace3ede392"
LONGEMBED_BASE_URL = "https://huggingface.co/datasets/dwzhu/LongEmbed/resolve"


@dataclass(frozen=True, slots=True)
class LongEmbedFile:
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LongEmbedTaskSpec:
    name: str
    files: tuple[LongEmbedFile, ...]


LONGEMBED_TASKS: Mapping[str, LongEmbedTaskSpec] = MappingProxyType(
    {
        "2wikimqa": LongEmbedTaskSpec(
            name="2wikimqa",
            files=(
                LongEmbedFile(
                    "corpus.jsonl",
                    "f50e7df1dbd5bdf27d8c16a7e85390a1485c2c13b7e4f3e147d7718d52ae787b",
                ),
                LongEmbedFile(
                    "queries.jsonl",
                    "68ce1d47c018ceb0b47ad4bfb970d835df882c16e0687534cddea9b7fe10fc26",
                ),
                LongEmbedFile(
                    "qrels.jsonl",
                    "ae65eec5dfbcd0169d0af92b19e732ab4c07d875061343b317804ed4dc226aea",
                ),
            ),
        )
    }
)


@dataclass(frozen=True, slots=True)
class LongEmbedDataset:
    name: str
    revision: str
    documents: tuple[Document, ...]
    queries: tuple[QueryExample, ...]
    qrels: Mapping[str, Mapping[str, int]]


def download_longembed_task(
    spec: LongEmbedTaskSpec,
    *,
    output_root: Path,
) -> Path:
    output_directory = output_root / spec.name
    output_directory.mkdir(parents=True, exist_ok=True)

    for file_spec in spec.files:
        output = output_directory / file_spec.name

        if output.exists():
            _validate_checksum(output, expected=file_spec.sha256)
            continue

        partial = output.with_suffix(f"{output.suffix}.part")
        url = (
            f"{LONGEMBED_BASE_URL}/{LONGEMBED_REVISION}/"
            f"{spec.name}/{file_spec.name}"
        )
        urllib.request.urlretrieve(url, partial)
        _validate_checksum(partial, expected=file_spec.sha256)
        os.replace(partial, output)

    manifest = {
        "dataset": "dwzhu/LongEmbed",
        "task": spec.name,
        "revision": LONGEMBED_REVISION,
        "files": {file.name: file.sha256 for file in spec.files},
    }
    (output_directory / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_directory


def load_longembed_task(directory: Path, *, name: str) -> LongEmbedDataset:
    documents = _load_corpus(directory / "corpus.jsonl", task_name=name)
    query_text_by_id = _load_queries(directory / "queries.jsonl")
    qrels = _load_qrels(directory / "qrels.jsonl")
    known_document_ids = {document.doc_id for document in documents}
    queries: list[QueryExample] = []

    for query_id, relevance_by_document in qrels.items():
        query_text = query_text_by_id.get(query_id)

        if query_text is None:
            raise ValueError(f"qrels reference unknown query ID: {query_id!r}")

        unknown_document_ids = set(relevance_by_document) - known_document_ids

        if unknown_document_ids:
            unknown = ", ".join(sorted(unknown_document_ids))
            raise ValueError(f"qrels reference unknown document IDs: {unknown}")

        queries.append(
            QueryExample(
                query_id=query_id,
                query=query_text,
                relevant_doc_ids=tuple(sorted(relevance_by_document)),
                split="test",
            )
        )

    frozen_qrels = MappingProxyType(
        {
            query_id: MappingProxyType(dict(relevance_by_document))
            for query_id, relevance_by_document in qrels.items()
        }
    )
    return LongEmbedDataset(
        name=name,
        revision=LONGEMBED_REVISION,
        documents=tuple(documents),
        queries=tuple(queries),
        qrels=frozen_qrels,
    )


def _load_corpus(path: Path, *, task_name: str) -> list[Document]:
    documents: list[Document] = []
    seen_document_ids: set[str] = set()

    for line_number, record in _read_jsonl(path):
        document_id = _required_string(record, "doc_id", path=path, line_number=line_number)
        text = _required_string(record, "text", path=path, line_number=line_number)

        if document_id in seen_document_ids:
            raise ValueError(f"{path}:{line_number}: duplicate document ID {document_id!r}")

        seen_document_ids.add(document_id)
        documents.append(
            Document(
                doc_id=document_id,
                title=_document_title(text),
                text=text,
                category=f"longembed/{task_name}",
            )
        )

    if not documents:
        raise ValueError(f"LongEmbed corpus is empty: {path}")

    return documents


def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}

    for line_number, record in _read_jsonl(path):
        query_id = _required_string(record, "qid", path=path, line_number=line_number)
        text = _required_string(record, "text", path=path, line_number=line_number)

        if query_id in queries:
            raise ValueError(f"{path}:{line_number}: duplicate query ID {query_id!r}")

        queries[query_id] = text

    return queries


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)

    for line_number, record in _read_jsonl(path):
        query_id = _required_string(record, "qid", path=path, line_number=line_number)
        document_id = _required_string(record, "doc_id", path=path, line_number=line_number)

        if document_id in qrels[query_id]:
            raise ValueError(
                f"{path}:{line_number}: duplicate qrel for {query_id!r}/{document_id!r}"
            )

        qrels[query_id][document_id] = 1

    if not qrels:
        raise ValueError(f"LongEmbed qrels are empty: {path}")

    return dict(qrels)


def _document_title(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())

    for line in lines:
        if line and not line.startswith("Passage "):
            return line

    return "LongEmbed document"


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
    record: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: field {field!r} must be a non-empty string")

    return value.strip()


def _validate_checksum(path: Path, *, expected: str) -> None:
    actual = _sha256_file(path)

    if actual != expected:
        raise ValueError(
            f"LongEmbed checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
