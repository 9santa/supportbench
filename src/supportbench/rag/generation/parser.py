import json
from typing import cast

from supportbench.rag.generation.models import (
    AnswerDecision,
    GeneratedAnswer,
    LLMResponse,
)

REQUIRED_FIELDS = {
    "decision",
    "answer",
    "citation_ids",
}

ALLOWED_DECISIONS = {
    "answer",
    "abstain",
    "clarify",
}


class GeneratedAnswerParseError(ValueError):
    """Raised when a model response violates the schema."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        llm_response: LLMResponse | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.llm_response = llm_response


def parse_generated_answer(
    raw_response: str,
) -> GeneratedAnswer:
    if not isinstance(raw_response, str):
        raise GeneratedAnswerParseError(
            "model response must be a string", raw_response=repr(raw_response)
        )

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise GeneratedAnswerParseError(
            "model response is not valid JSON",
            raw_response=raw_response,
        ) from error

    if not isinstance(parsed, dict):
        raise GeneratedAnswerParseError(
            "model response must be a valid JSON object",
            raw_response=raw_response,
        )

    actual_fields = set(parsed)

    if actual_fields != REQUIRED_FIELDS:
        missing_fields = REQUIRED_FIELDS - actual_fields
        unknown_fields = actual_fields - REQUIRED_FIELDS

        mistakes: list[str] = []

        if missing_fields:
            mistakes.append("missing fields: " + ", ".join(sorted(missing_fields)))

        if unknown_fields:
            mistakes.append("unknown fields: " + ", ".join(sorted(unknown_fields)))

        raise GeneratedAnswerParseError(
            "; ".join(mistakes),
            raw_response=raw_response,
        )

    # Decision validation
    decision_value = parsed["decision"]

    if not isinstance(decision_value, str) or decision_value not in ALLOWED_DECISIONS:
        raise GeneratedAnswerParseError(
            "decision must be one of: answer, abstain, clarify",
            raw_response=raw_response,
        )

    # Answer validation
    answer_value = parsed["answer"]

    if not isinstance(answer_value, str):
        raise GeneratedAnswerParseError(
            "answer must be a string",
            raw_response=raw_response,
        )

    normalized_answer = answer_value.strip()

    if not normalized_answer:
        raise GeneratedAnswerParseError(
            "answer must be non-empty",
            raw_response=raw_response,
        )

    # Citations validation
    citation_values = parsed["citation_ids"]

    if not isinstance(citation_values, list):
        raise GeneratedAnswerParseError(
            "citation_ids must be a list",
            raw_response=raw_response,
        )

    normalized_citations: list[str] = []

    for citation in citation_values:
        if not isinstance(citation, str):
            raise GeneratedAnswerParseError(
                "every citation ID must be a string",
                raw_response=raw_response,
            )

        normalized_citation = citation.strip()

        if not normalized_citation:
            raise GeneratedAnswerParseError(
                "citation IDs must be non-empty",
                raw_response=raw_response,
            )

        normalized_citations.append(normalized_citation)

    decision = cast(AnswerDecision, decision_value)

    return GeneratedAnswer(
        decision=decision,
        answer=normalized_answer,
        citation_ids=tuple(normalized_citations),
    )
