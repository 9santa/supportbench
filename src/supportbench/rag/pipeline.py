from dataclasses import dataclass

from supportbench.rag.context import ContextPreparationService
from supportbench.rag.generation.models import ChatMessage, GeneratedAnswer
from supportbench.rag.generation.prompt import PromptBudget
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import RAGContext, RetrievedChunk
from supportbench.rag.retrieval import ParentRetrievalRun


@dataclass(frozen=True, slots=True)
class RAGRun:
    retrieval: ParentRetrievalRun
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context: RAGContext
    messages: tuple[ChatMessage, ...]
    raw_response: str | None
    answer: GeneratedAnswer
    prompt_budget: PromptBudget | None = None
    prompt_token_count: int = 0


class RAGPipeline:
    """Run the current end-to-end grounded question-answering flow."""

    def __init__(
        self,
        *,
        context_service: ContextPreparationService,
        answer_generator: GroundedAnswerGenerator,
    ) -> None:
        self._context_service = context_service
        self._answer_generator = answer_generator

    def answer(self, query: str) -> GeneratedAnswer:
        return self.run(query).answer

    def run(self, query: str) -> RAGRun:
        context_run = self._context_service.prepare(query)
        generation = self._answer_generator.generate(
            query=query,
            context=context_run.context,
        )

        return RAGRun(
            retrieval=context_run.retrieval,
            retrieved_chunks=context_run.retrieved_chunks,
            context=context_run.context,
            messages=generation.messages,
            raw_response=generation.raw_response,
            answer=generation.answer,
            prompt_budget=context_run.prompt_budget,
            prompt_token_count=context_run.prompt_token_count,
        )
