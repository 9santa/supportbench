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
        parsed_answer: GeneratedAnswer | None = None,
        raw_citation_ids: tuple[str, ...] = (),
        citation_ids: tuple[str, ...] = (),
        contract_violations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.llm_response = llm_response
        self.parsed_answer = parsed_answer
        self.raw_citation_ids = raw_citation_ids
        self.citation_ids = citation_ids
        self.contract_violations = contract_violations


class CitationResolutionError(CitationValidationError):
    """Raised when a citation handle cannot be resolved to supplied evidence."""


class CitationContractError(CitationValidationError):
    """Raised when resolved citations conflict with the generated decision."""


def resolve_generated_answer_citations(
    answer: GeneratedAnswer,
    context: RAGContext,
) -> GeneratedAnswer:
    parent_by_handle = _parent_by_citation_handle(context)
    canonical_ids: list[str] = []
    unknown_ids: set[str] = set()

    for citation_id in answer.citation_ids:
        parent_id = parent_by_handle.get(citation_id)

        if parent_id is None:
            unknown_ids.add(citation_id)
        else:
            canonical_ids.append(parent_id)

    if unknown_ids:
        formatted_ids = ", ".join(sorted(unknown_ids))

        raise CitationResolutionError(
            f"response contains unknown citation source IDs: {formatted_ids}",
            parsed_answer=answer,
            raw_citation_ids=answer.citation_ids,
            citation_ids=tuple(dict.fromkeys(canonical_ids)),
        )

    return replace(
        answer,
        citation_ids=tuple(dict.fromkeys(canonical_ids)),
    )


def validate_generated_answer_contract(
    answer: GeneratedAnswer,
    *,
    raw_citation_ids: tuple[str, ...] | None = None,
) -> GeneratedAnswer:
    raw_ids = answer.citation_ids if raw_citation_ids is None else raw_citation_ids
    parsed_answer = replace(
        answer,
        citation_ids=raw_ids,
    )

    if answer.decision == "answer" and not answer.citation_ids:
        raise CitationContractError(
            "answered response must contain at least one citation",
            parsed_answer=parsed_answer,
            raw_citation_ids=raw_ids,
            citation_ids=answer.citation_ids,
            contract_violations=("answer_has_no_citations",),
        )

    if answer.decision != "answer" and answer.citation_ids:
        raise CitationContractError(
            f"{answer.decision} response must not contain citations",
            parsed_answer=parsed_answer,
            raw_citation_ids=raw_ids,
            citation_ids=answer.citation_ids,
            contract_violations=("non_answer_has_citations",),
        )

    return answer


def validate_generated_answer(
    answer: GeneratedAnswer,
    context: RAGContext,
) -> GeneratedAnswer:
    """Resolve and validate citations for historical callers."""
    resolved = resolve_generated_answer_citations(answer, context)
    return validate_generated_answer_contract(
        resolved,
        raw_citation_ids=answer.citation_ids,
    )


def _parent_by_citation_handle(context: RAGContext) -> dict[str, str]:
    parent_ids = {document.doc_id for document in context.documents}
    source_handles = {
        provenance.source_id: provenance.parent_doc_id
        for provenance in context.provenance
        if provenance.source_id is not None and provenance.parent_doc_id in parent_ids
    }

    if source_handles:
        return source_handles

    # Historical contexts exposed parent and chunk IDs directly to the model.
    parent_by_handle = {parent_id: parent_id for parent_id in parent_ids}
    parent_by_handle.update(
        {
            provenance.chunk_id: provenance.parent_doc_id
            for provenance in context.provenance
            if provenance.parent_doc_id in parent_ids
        }
    )
    return parent_by_handle
