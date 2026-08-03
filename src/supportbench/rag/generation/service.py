from dataclasses import dataclass

from supportbench.rag.citation_validator import (
    CitationValidationError,
    validate_generated_answer,
)
from supportbench.rag.generation.client import LLMClient
from supportbench.rag.generation.models import ChatMessage, GeneratedAnswer
from supportbench.rag.generation.parser import parse_generated_answer
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.models import RAGContext

EMPTY_CONTEXT_ABSTENTION = GeneratedAnswer(
    decision="abstain",
    answer="В базе знаний не найдено документов, достаточных для ответа.",
    citation_ids=(),
)


@dataclass(frozen=True, slots=True)
class GroundedGenerationRun:
    messages: tuple[ChatMessage, ...]
    raw_response: str | None
    answer: GeneratedAnswer


class GroundedAnswerGenerator:
    """Generate and validate an answer from an already constructed context."""

    def __init__(
        self,
        *,
        prompt_builder: GroundedPromptBuilder,
        llm_client: LLMClient,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client

    def run(
        self,
        *,
        query: str,
        context: RAGContext,
    ) -> GroundedGenerationRun:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        if not context.documents:
            return GroundedGenerationRun(
                messages=(),
                raw_response=None,
                answer=EMPTY_CONTEXT_ABSTENTION,
            )

        messages = self._prompt_builder.build(
            query=normalized_query,
            context=context,
        )
        raw_response = self._llm_client.generate(messages)
        generated_answer = parse_generated_answer(raw_response)

        try:
            validate_generated_answer(generated_answer, context)
        except CitationValidationError as error:
            raise CitationValidationError(
                str(error),
                raw_response=raw_response,
            ) from error

        return GroundedGenerationRun(
            messages=messages,
            raw_response=raw_response,
            answer=generated_answer,
        )
