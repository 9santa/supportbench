from dataclasses import dataclass

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
from supportbench.rag.generation.prompt import (
    GroundedPromptBuilder,
)
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import RAGContext
from supportbench.rag.retrieval_pipeline import (
    RetrievalPipeline,
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
        self._answer_generator = GroundedAnswerGenerator(
            prompt_builder=prompt_builder,
            llm_client=llm_client,
        )
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

        generation = self._answer_generator.run(
            query=normalized_query,
            context=context,
        )

        return GroundedRAGRun(
            context=context,
            messages=generation.messages,
            raw_response=generation.raw_response,
            answer=generation.answer,
        )
