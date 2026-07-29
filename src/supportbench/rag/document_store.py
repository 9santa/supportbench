from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Protocol

from supportbench.data.models import Document

# Document Store doesn't know anything about JSONL, index or retrieval score


class DocumentStore(Protocol):
    def get(self, doc_id: str) -> Document: ...

    def get_many(self, doc_ids: Iterable[str]) -> list[Document]: ...

    def __contains__(self, doc_id: object) -> bool: ...


class InMemoryDocumentStore(DocumentStore):
    def __init__(
        self,
        documents: Iterable[Document],
    ) -> None:
        documents_by_id: dict[str, Document] = {}

        for document in documents:
            if document.doc_id in documents_by_id:
                raise ValueError(f"duplicate document ID: {document.doc_id!r}")

            documents_by_id[document.doc_id] = document

        self._documents: Mapping[str, Document] = MappingProxyType(documents_by_id)

    def get(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    def get_many(self, doc_ids: Iterable[str]) -> list[Document]:
        return [self.get(doc_id) for doc_id in doc_ids]

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._documents
