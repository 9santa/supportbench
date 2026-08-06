from dataclasses import dataclass, replace

from supportbench.rag.citations import (
    CitationValidationError,
    resolve_generated_answer_citations,
    validate_generated_answer_contract,
)
from supportbench.rag.generation.client import LLMClient
from supportbench.rag.generation.models import ChatMessage, GeneratedAnswer, LLMResponse
from supportbench.rag.generation.parser import GeneratedAnswerParseError, parse_generated_answer
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.models import RAGContext

EMPTY_CONTEXT_ABSTENTION = GeneratedAnswer(
    decision="abstain",
    answer="No documents were found in the knowledge base that "
    "were sufficient to answer this question.",
    citation_ids=(),
)


@dataclass(frozen=True, slots=True)
class GroundedGenerationRun:
    messages: tuple[ChatMessage, ...]
    raw_response: str | None
    answer: GeneratedAnswer
    raw_citation_ids: tuple[str, ...] = ()
    resolved_citation_ids: tuple[str, ...] = ()
    contract_repaired: bool = False
    strict_contract_valid: bool = True
    contract_violations: tuple[str, ...] = ()
    llm_response: LLMResponse | None = None


class GenerationTruncatedError(ValueError):
    """Raised when the model reaches its output-token limit."""

    def __init__(self, response: LLMResponse) -> None:
        super().__init__("model response was truncated at the output-token limit")
        self.raw_response = response.content
        self.llm_response = response


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

    def generate(
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
        llm_response = self._llm_client.generate(messages)

        if llm_response.truncated:
            raise GenerationTruncatedError(llm_response)

        try:
            generated_answer = parse_generated_answer(llm_response.content)
        except GeneratedAnswerParseError as error:
            raise GeneratedAnswerParseError(
                str(error),
                raw_response=error.raw_response,
                llm_response=llm_response,
            ) from error

        parsed_answer = generated_answer
        raw_citation_ids = parsed_answer.citation_ids

        try:
            generated_answer = resolve_generated_answer_citations(parsed_answer, context)
            resolved_citation_ids = generated_answer.citation_ids
            contract_violations: tuple[str, ...]

            if generated_answer.decision in {"abstain", "clarify"} and resolved_citation_ids:
                generated_answer = replace(
                    generated_answer,
                    citation_ids=(),
                )
                contract_repaired = True
                strict_contract_valid = False
                contract_violations = ("non_answer_has_citations",)
            else:
                generated_answer = validate_generated_answer_contract(
                    generated_answer,
                    raw_citation_ids=raw_citation_ids,
                )
                contract_repaired = False
                strict_contract_valid = True
                contract_violations = ()
        except CitationValidationError as error:
            error_type = type(error)
            raise error_type(
                str(error),
                raw_response=llm_response.content,
                llm_response=llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                citation_ids=error.citation_ids,
                contract_violations=error.contract_violations,
            ) from error

        return GroundedGenerationRun(
            messages=messages,
            raw_response=llm_response.content,
            answer=generated_answer,
            raw_citation_ids=raw_citation_ids,
            resolved_citation_ids=resolved_citation_ids,
            contract_repaired=contract_repaired,
            strict_contract_valid=strict_contract_valid,
            contract_violations=contract_violations,
            llm_response=llm_response,
        )

    def run(
        self,
        *,
        query: str,
        context: RAGContext,
    ) -> GroundedGenerationRun:
        """Retain the historical synthetic_v2 generation entry point."""
        return self.generate(query=query, context=context)
