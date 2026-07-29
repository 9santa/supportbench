from dataclasses import dataclass

from supportbench.rag import retrieval_pipeline
from supportbench.rag.citation_validator import (
    CitationValidationError,
    validate_generated_answer,
)
from supportbench.rag.context_builder import (
    ContextBuilder,
)
from supportbench.rag.generation.client import (
    LLMClient,
)
from supportbench.rag.generation.models import (
    ChatMessage,
    GeneratedAnswer,
)
from supportbench.rag.generation.parser import (
    parse_generated_answer,
)
from supportbench.rag.generation.prompt import (
    GroundedPromptBuilder,
)
from supportbench.rag.models import RAGContext
from supportbench.rag.retrieval_pipeline import (
    RetrievalPipeline,
)


EMPTY_CONTEXT_ABSTENTION = GeneratedAnswer(
    decision="abstain",
    answer=("В базе знаний не найдено документов, достаточных для ответа."),
    citation_ids=(),
)


@dataclass(frozen=True, slots=True)
class GroundedRAGRun:
    context: RAGContext
    messages: tuple[ChatMessage, ...]
    raw_response: str | None
    answer: GeneratedAnswer


class GroundedRAGPipeline:
    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_builder: GroundedPromptBuilder,
        llm_client: LLMClient,
        retrieval_top_k: int = 5,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")

        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._retrieval_top_k = retrieval_top_k

    def answer(
        self,
        query: str,
    ) -> GeneratedAnswer:
        """Production API."""
        return self.run(query).answer

    def run(
        self,
        query: str,
    ) -> GroundedRAGRun:
        """CLI, Tests, Diagnostics API."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        retrieved_documents = self._retrieval_pipeline.retrieve(
            normalized_query,
            top_k=self._retrieval_top_k,
        )

        context = self._context_builder.build(
            retrieved_documents,
        )

        if not context.documents:
            return GroundedRAGRun(
                context=context,
                messages=(),
                raw_response=None,
                answer=EMPTY_CONTEXT_ABSTENTION,
            )

        messages = self._prompt_builder.build(
            query=normalized_query,
            context=context,
        )

        raw_response = self._llm_client.generate(
            messages,
        )

        generated_answer = parse_generated_answer(
            raw_response,
        )

        try:
            validate_generated_answer(
                generated_answer,
                context,
            )
        except CitationValidationError as error:
            raise CitationValidationError(
                str(error),
                raw_response=raw_response,
            ) from error

        return GroundedRAGRun(
            context=context,
            messages=messages,
            raw_response=raw_response,
            answer=generated_answer,
        )
