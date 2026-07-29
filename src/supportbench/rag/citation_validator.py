from supportbench.rag.generation.models import (
    GeneratedAnswer,
)
from supportbench.rag.models import RAGContext


class CitationValidationError(ValueError):
    """Raised when generated citations are invalid."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def validate_generated_answer(
    answer: GeneratedAnswer,
    context: RAGContext,
) -> None:
    citation_ids = answer.citation_ids

    if answer.decision == "answer" and not citation_ids:
        raise CitationValidationError("answered response must contain at least one citation")

    if answer.decision != "answer" and citation_ids:
        raise CitationValidationError(f"{answer.decision} response must not contain citations")

    if len(citation_ids) != len(set(citation_ids)):
        raise CitationValidationError("citation IDs must not contain duplicates")

    available_ids = {document.doc_id for document in context.documents}

    unknown_ids = set(citation_ids) - available_ids

    if unknown_ids:
        formatted_ids = ", ".join(sorted(unknown_ids))

        raise CitationValidationError(
            f"response contains citations outside the supplied context: {formatted_ids}"
        )
