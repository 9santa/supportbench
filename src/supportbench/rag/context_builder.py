from collections.abc import Sequence
from dataclasses import replace

from supportbench.rag.models import (
    RAGContext,
    RetrievedDocument,
)

DOCUMENT_SEPARATOR = "\n\n"
DOCUMENT_FOOTER = "\n[/DOCUMENT]"
TRUNCATION_MARKER = "\n[TRUNCATED]"


class ContextBuilder:
    def __init__(
        self,
        *,
        max_documents: int = 5,
        max_characters: int = 12_000,
    ) -> None:
        if max_documents <= 0:
            raise ValueError("max_documents must be positive")

        if max_characters <= 0:
            raise ValueError("max_characters must be positive")

        self._max_documents = max_documents
        self._max_characters = max_characters

    def build(
        self,
        documents: Sequence[RetrievedDocument],
    ) -> RAGContext:
        document_items = tuple(documents)

        if not document_items:
            return RAGContext(
                documents=(),
                formatted_text="",
                truncated=False,
            )

        self._validate_documents(document_items)

        added_documents: list[RetrievedDocument] = []
        formatted_blocks: list[str] = []

        document_was_truncated = False

        for document in document_items:
            if len(added_documents) >= self._max_documents:
                break

            separator = DOCUMENT_SEPARATOR if formatted_blocks else ""

            full_block = _format_document(document)

            length_after = (
                len(separator) + sum(len(block) for block in formatted_blocks) + len(full_block)
            )

            if length_after <= self._max_characters:
                formatted_blocks.append(full_block)
                added_documents.append(document)
                continue

            # Case: length_after > max_characters
            # Only the first document may be truncated.
            # For later documents we stop.
            if formatted_blocks:
                break

            truncated_result = self._truncate_first_document(document)

            if truncated_result is None:
                break

            truncated_document, block = truncated_result

            added_documents.append(truncated_document)
            formatted_blocks.append(block)
            document_was_truncated = True
            break

        formatted_text = DOCUMENT_SEPARATOR.join(formatted_blocks)

        truncated = document_was_truncated or len(added_documents) < len(document_items)

        if len(formatted_text) > self._max_characters:
            raise RuntimeError("context builder exceeded the character budget")

        return RAGContext(
            documents=tuple(added_documents),
            formatted_text=formatted_text,
            truncated=truncated,
        )

    def _truncate_first_document(
        self,
        document: RetrievedDocument,
    ) -> tuple[RetrievedDocument, str] | None:
        header = _format_header(document)

        fixed_character_count = len(header) + len(TRUNCATION_MARKER) + len(DOCUMENT_FOOTER)

        available_content_characters = self._max_characters - fixed_character_count

        if available_content_characters < 0:
            return None

        truncated_text = document.text[:available_content_characters]

        truncated_document = replace(document, text=truncated_text)

        formatter_block = header + truncated_text + TRUNCATION_MARKER + DOCUMENT_FOOTER

        return (truncated_document, formatter_block)

    @staticmethod
    def _validate_documents(
        documents: tuple[RetrievedDocument, ...],
    ) -> None:
        seen_doc_ids: set[str] = set()

        for expected_rank, document in enumerate(documents, start=1):
            if not document.doc_id.strip():
                raise ValueError("document ID must be non-empty")

            if document.doc_id in seen_doc_ids:
                raise ValueError(f"duplicate retrieved document ID: {document.doc_id!r}")

            if document.rank != expected_rank:
                raise ValueError(
                    "retrieved document ranks must "
                    "be consecutive starting at 1; "
                    f"expected {expected_rank}, "
                    f"received {document.rank}"
                )

            seen_doc_ids.add(document.doc_id)


def _format_header(document: RetrievedDocument) -> str:
    return (
        "[DOCUMENT]\n"
        f"doc_id: {document.doc_id}\n"
        f"title: {document.title}\n"
        f"category: {document.category}\n"
        "content:\n"
    )


def _format_document(document: RetrievedDocument) -> str:
    return _format_header(document) + document.text + DOCUMENT_FOOTER
