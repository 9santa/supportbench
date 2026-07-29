import pytest

from supportbench.data.models import Document
from supportbench.rag.document_store import (
    InMemoryDocumentStore,
)


def make_document(
    doc_id: str,
) -> Document:
    return Document(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Text {doc_id}",
        category="test",
    )


def test_document_can_be_retrieved_by_id() -> None:
    document = make_document("doc_a")

    store = InMemoryDocumentStore([document])

    assert store.get("doc_a") == document


def test_get_many_preserves_input_order() -> None:
    doc_a = make_document("doc_a")
    doc_b = make_document("doc_b")

    store = InMemoryDocumentStore([doc_a, doc_b])

    assert store.get_many(["doc_b", "doc_a", "doc_b"]) == [
        doc_b,
        doc_a,
        doc_b,
    ]


def test_get_many_supports_empty_input() -> None:
    store = InMemoryDocumentStore([make_document("doc_a")])

    assert store.get_many([]) == []


def test_unknown_document_is_rejected() -> None:
    store = InMemoryDocumentStore([])

    with pytest.raises(KeyError):
        store.get("unknown")


def test_duplicate_document_id_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate document ID",
    ):
        InMemoryDocumentStore(
            [
                make_document("doc_a"),
                make_document("doc_a"),
            ]
        )


def test_contains_known_document() -> None:
    store = InMemoryDocumentStore([make_document("doc_a")])

    assert "doc_a" in store


def test_contains_unknown_document() -> None:
    store = InMemoryDocumentStore([make_document("doc_a")])

    assert "unknown" not in store
