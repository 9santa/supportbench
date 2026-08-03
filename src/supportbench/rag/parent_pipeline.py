from dataclasses import dataclass
from typing import Protocol

from supportbench.rag.chunk_context_builder import RepresentativeChunkContextBuilder
from supportbench.rag.chunk_retrieval_pipeline import RepresentativeChunkRetrievalPipeline
from supportbench.rag.generation.models import ChatMessage, GeneratedAnswer
from supportbench.rag.generation.prompt import PromptBudget, PromptBudgetCalculator
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import RAGContext, RetrievedChunk
from supportbench.rag.parent_retrieval import ParentRetrievalOrchestrator, ParentRetrievalRun


@dataclass(frozen=True, slots=True)
class ParentContextRun:
    retrieval: ParentRetrievalRun
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context: RAGContext
    prompt_budget: PromptBudget | None = None
    prompt_token_count: int = 0


class ParentContextPipeline:
    def __init__(
        self,
        *,
        retrieval_orchestrator: ParentRetrievalOrchestrator,
        chunk_pipeline: RepresentativeChunkRetrievalPipeline,
        context_builder: RepresentativeChunkContextBuilder,
        prompt_budget_calculator: PromptBudgetCalculator,
        top_parents: int,
    ) -> None:
        if top_parents <= 0:
            raise ValueError("top_parents must be positive")

        self._retrieval_orchestrator = retrieval_orchestrator
        self._chunk_pipeline = chunk_pipeline
        self._context_builder = context_builder
        self._prompt_budget_calculator = prompt_budget_calculator
        self._top_parents = top_parents

    def run(self, query: str) -> ParentContextRun:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        prompt_budget = self._prompt_budget_calculator.calculate(normalized_query)
        retrieval = self._retrieval_orchestrator.run(normalized_query)
        retrieved_chunks = tuple(
            self._chunk_pipeline.retrieve(
                retrieval,
                top_k=self._top_parents,
            )
        )
        context_token_budget = prompt_budget.available_context_tokens
        context = self._context_builder.build(
            retrieved_chunks,
            max_tokens=context_token_budget,
        )
        prompt_token_count = self._prompt_budget_calculator.count_prompt(
            query=normalized_query,
            context=context,
        )
        overflow = (
            prompt_token_count
            + prompt_budget.reserved_output_tokens
            - prompt_budget.model_context_window
        )

        if overflow > 0:
            context_token_budget -= overflow

            if context_token_budget <= 0:
                raise ValueError("full prompt leaves no room for knowledge context")

            context = self._context_builder.build(
                retrieved_chunks,
                max_tokens=context_token_budget,
            )
            prompt_token_count = self._prompt_budget_calculator.count_prompt(
                query=normalized_query,
                context=context,
            )

        if (
            prompt_token_count + prompt_budget.reserved_output_tokens
            > prompt_budget.model_context_window
        ):
            raise RuntimeError("full prompt exceeded the model context window")

        return ParentContextRun(
            retrieval=retrieval,
            retrieved_chunks=retrieved_chunks,
            context=context,
            prompt_budget=prompt_budget,
            prompt_token_count=prompt_token_count,
        )


class ParentContextRunner(Protocol):
    def run(self, query: str) -> ParentContextRun: ...


@dataclass(frozen=True, slots=True)
class ParentGroundedRAGRun:
    retrieval: ParentRetrievalRun
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context: RAGContext
    messages: tuple[ChatMessage, ...]
    raw_response: str | None
    answer: GeneratedAnswer
    prompt_budget: PromptBudget | None = None
    prompt_token_count: int = 0


class ParentGroundedRAGPipeline:
    def __init__(
        self,
        *,
        context_pipeline: ParentContextRunner,
        answer_generator: GroundedAnswerGenerator,
    ) -> None:
        self._context_pipeline = context_pipeline
        self._answer_generator = answer_generator

    def answer(self, query: str) -> GeneratedAnswer:
        return self.run(query).answer

    def run(self, query: str) -> ParentGroundedRAGRun:
        context_run = self._context_pipeline.run(query)
        generation = self._answer_generator.run(
            query=query,
            context=context_run.context,
        )

        return ParentGroundedRAGRun(
            retrieval=context_run.retrieval,
            retrieved_chunks=context_run.retrieved_chunks,
            context=context_run.context,
            messages=generation.messages,
            raw_response=generation.raw_response,
            answer=generation.answer,
            prompt_budget=context_run.prompt_budget,
            prompt_token_count=context_run.prompt_token_count,
        )
