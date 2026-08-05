from dataclasses import replace

from supportbench.rag.generation.models import (
    GeneratedAnswer,
    LLMResponse,
)
from supportbench.rag.models import RAGContext


class CitationValidationError(ValueError):
    """Raised when generated citations are invalid."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        llm_response: LLMResponse | None = None,
        citation_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.llm_response = llm_response
        self.citation_ids = citation_ids


def validate_generated_answer(
    answer: GeneratedAnswer,
    context: RAGContext,
) -> GeneratedAnswer:
    citation_ids = answer.citation_ids

    if answer.decision == "answer" and not citation_ids:
        raise CitationValidationError("answered response must contain at least one citation")

    if answer.decision != "answer" and citation_ids:
        raise CitationValidationError(f"{answer.decision} response must not contain citations")

    parent_ids = {document.doc_id for document in context.documents}
    parent_by_handle = {parent_id: parent_id for parent_id in parent_ids}
    parent_by_handle.update(
        {
            provenance.chunk_id: provenance.parent_doc_id
            for provenance in context.provenance
            if provenance.parent_doc_id in parent_ids
        }
    )
    canonical_ids: list[str] = []
    unknown_ids: set[str] = set()

    for citation_id in citation_ids:
        parent_id = _resolve_citation_handle(citation_id, parent_by_handle)

        if parent_id is None:
            unknown_ids.add(citation_id)
        else:
            canonical_ids.append(parent_id)

    if unknown_ids:
        formatted_ids = ", ".join(sorted(unknown_ids))

        raise CitationValidationError(
            f"response contains citations outside the supplied context: {formatted_ids}"
        )

    return replace(
        answer,
        citation_ids=tuple(dict.fromkeys(canonical_ids)),
    )


def _resolve_citation_handle(
    citation_id: str,
    parent_by_handle: dict[str, str],
) -> str | None:
    direct_match = parent_by_handle.get(citation_id)

    if direct_match is not None:
        return direct_match

    label, separator, value = citation_id.partition(":")

    if separator and label.strip() in {"doc_id", "chunk_id"}:
        return parent_by_handle.get(value.strip())

    return None
