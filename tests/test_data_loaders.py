import json
import re
from pathlib import Path
from typing import Any

import pytest

from supportbench.data.loaders import (
    DatasetValidationError,
    load_documents,
    load_queries,
)
from supportbench.data.models import (
    Document,
    QueryExample,
)


def write_jsonl(path: Path, objects: list[dict[str, Any]]) -> None:
    """Записать список объектов как JSONL файл."""
    lines = [json.dumps(obj, ensure_ascii=False) for obj in objects]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Tests for document loading
# ----------------------------------------------------------------------


def test_load_valid_documents(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [
            {
                "doc_id": "vpn_linux",
                "title": "VPN на Linux",
                "text": "Установите OpenVPN.",
                "category": "network",
            },
            {
                "doc_id": "gitlab_2fa",
                "title": "Восстановление 2FA",
                "text": "Создайте обращение в поддержку.",
                "category": "access",
            },
        ],
    )

    docs = load_documents(path)

    assert docs == [
        Document(
            doc_id="vpn_linux",
            title="VPN на Linux",
            text="Установите OpenVPN.",
            category="network",
        ),
        Document(
            doc_id="gitlab_2fa",
            title="Восстановление 2FA",
            text="Создайте обращение в поддержку.",
            category="access",
        ),
    ]


def test_document_fields_are_stripped(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    write_jsonl(
        path,
        [
            {
                "doc_id": "  vpn_linux  ",
                "title": "  VPN на Linux  ",
                "text": "  Установите OpenVPN.  ",
                "category": "  network  ",
            }
        ],
    )

    docs = load_documents(path)

    assert docs == [
        Document(
            doc_id="vpn_linux",
            title="VPN на Linux",
            text="Установите OpenVPN.",
            category="network",
        )
    ]


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    content = (
        '{"doc_id": "id1", "title": "t", "text": "x", "category": "c"}\n'
        "\n"
        '{"doc_id": "id2", "title": "t", "text": "x", "category": "c"}\n'
    )
    path.write_text(content, encoding="utf-8")

    docs = load_documents(path)

    assert len(docs) == 2
    assert docs[0].doc_id == "id1"
    assert docs[1].doc_id == "id2"


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    content = (
        '{"doc_id": "id1", "title": "t", "text": "x", "category": "c"}\n'
        '{"doc_id": "id2", "title": "t", "text": "x"\n'  # не закрыт объект
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=re.escape(f"{path}:2: invalid JSON")):
        load_documents(path)


def test_non_object_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('"just a string"\n', encoding="utf-8")

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: each non-empty line must be a JSON object"),
    ):
        load_documents(path)


def test_missing_document_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [{"doc_id": "id1", "title": "t", "text": "x"}],  # нет category
    )

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: missing required field 'category'"),
    ):
        load_documents(path)


def test_non_string_document_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [{"doc_id": 123, "title": "t", "text": "x", "category": "c"}],
    )

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: field 'doc_id' must be a string"),
    ):
        load_documents(path)


def test_empty_document_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [{"doc_id": "id1", "title": "   ", "text": "x", "category": "c"}],
    )

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: field 'title' must be non-empty"),
    ):
        load_documents(path)


def test_duplicate_doc_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    write_jsonl(
        path,
        [
            {"doc_id": "dup", "title": "t1", "text": "x", "category": "c"},
            {"doc_id": "dup", "title": "t2", "text": "y", "category": "c"},
        ],
    )

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:2: duplicate doc_id 'dup'"),
    ):
        load_documents(path)


# ----------------------------------------------------------------------
# Тесты для запросов
# ----------------------------------------------------------------------

def test_load_valid_queries(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [
            {
                "query_id": "q1",
                "query": "Как настроить VPN?",
                "relevant_doc_ids": ["vpn_linux", "gitlab_2fa"],
                "split": "train",
            }
        ],
    )

    known_ids: set[str] = {"vpn_linux", "gitlab_2fa"}
    queries = load_queries(path, known_ids)

    assert queries == [
        QueryExample(
            query_id="q1",
            query="Как настроить VPN?",
            relevant_doc_ids=("vpn_linux", "gitlab_2fa"),
            split="train",
        )
    ]


def test_query_fields_are_stripped(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [
            {
                "query_id": "  q1  ",
                "query": "  Как настроить VPN?  ",
                "relevant_doc_ids": ["  vpn_linux  "],
                "split": "  dev  ",
            }
        ],
    )

    known_ids: set[str] = {"vpn_linux"}
    queries = load_queries(path, known_ids)

    assert queries == [
        QueryExample(
            query_id="q1",
            query="Как настроить VPN?",
            relevant_doc_ids=("vpn_linux",),
            split="dev",
        )
    ]


def test_duplicate_query_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [
            {"query_id": "q1", "query": "q", "relevant_doc_ids": ["a"], "split": "train"},
            {"query_id": "q1", "query": "q2", "relevant_doc_ids": ["a"], "split": "test"},
        ],
    )
    known_ids: set[str] = {"a"}

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:2: duplicate query_id 'q1'"),
    ):
        load_queries(path, known_ids)


def test_relevant_doc_ids_must_be_a_list(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": "not a list", "split": "train"}],
    )
    known_ids: set[str] = set()

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: 'relevant_doc_ids' must be a list"),
    ):
        load_queries(path, known_ids)


def test_empty_relevant_documents_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": [], "split": "train"}],
    )
    known_ids: set[str] = set()

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: 'relevant_doc_ids' must not be empty"),
    ):
        load_queries(path, known_ids)


def test_non_string_relevant_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": ["a", 123], "split": "train"}],
    )
    known_ids: set[str] = {"a"}

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: item 1 in 'relevant_doc_ids' must be a string"),
    ):
        load_queries(path, known_ids)


def test_duplicate_relevant_documents_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": ["a", "a"], "split": "train"}],
    )
    known_ids: set[str] = {"a"}

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(f"{path}:1: duplicate doc_id 'a' in relevant_doc_ids"),
    ):
        load_queries(path, known_ids)


def test_unknown_relevant_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": ["missing"], "split": "train"}],
    )
    known_ids: set[str] = {"a", "b"}  # missing нет

    with pytest.raises(
        DatasetValidationError,
        match=r".*\.jsonl:1: unknown doc_id\(s\) in relevant_doc_ids: 'missing'",
    ):
        load_queries(path, known_ids)


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    write_jsonl(
        path,
        [{"query_id": "q1", "query": "q", "relevant_doc_ids": ["a"], "split": "production"}],
    )
    known_ids: set[str] = {"a"}

    with pytest.raises(
        DatasetValidationError,
        match=re.escape(
            f"{path}:1: invalid split 'production'; must be one of ('train', 'dev', 'test')"
        ),
    ):
        load_queries(path, known_ids)
