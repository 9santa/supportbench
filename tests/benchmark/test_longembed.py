import hashlib
import json
from pathlib import Path

from supportbench.benchmark.longembed import (
    LONGEMBED_REVISION,
    LongEmbedFile,
    LongEmbedTaskSpec,
    download_longembed_task,
    load_longembed_task,
)


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_longembed_task_maps_gold_parent_documents(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {
                "doc_id": "doc_1",
                "text": "Passage 1:\nActual title\nLong document body",
                "qid": "",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"qid": "query_1", "text": "question", "doc_id": ""}],
    )
    _write_jsonl(
        tmp_path / "qrels.jsonl",
        [{"qid": "query_1", "doc_id": "doc_1", "text": ""}],
    )

    dataset = load_longembed_task(tmp_path, name="2wikimqa")

    assert dataset.revision == LONGEMBED_REVISION
    assert dataset.documents[0].title == "Actual title"
    assert dataset.queries[0].relevant_doc_ids == ("doc_1",)
    assert dataset.qrels == {"query_1": {"doc_1": 1}}


def test_download_longembed_task_reuses_checksum_verified_files(tmp_path: Path) -> None:
    task_directory = tmp_path / "fixture"
    task_directory.mkdir()
    contents = b'{"doc_id":"doc_1"}\n'
    (task_directory / "corpus.jsonl").write_bytes(contents)
    spec = LongEmbedTaskSpec(
        name="fixture",
        files=(LongEmbedFile("corpus.jsonl", hashlib.sha256(contents).hexdigest()),),
    )

    result = download_longembed_task(spec, output_root=tmp_path)

    assert result == task_directory
    manifest = json.loads((task_directory / "download_manifest.json").read_text())
    assert manifest["revision"] == LONGEMBED_REVISION
