from dataclasses import dataclass
from typing import Protocol

from supportbench.rag.chunk_context_builder import RepresentativeChunkContextBuilder
from supportbench.rag.chunk_retrieval_pipeline import RepresentativeChunkRetrievalPipeline
from supportbench.rag.generation.models import ChatMessage, GeneratedAnswer
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import RAGContext, RetrievedChunk
from supportbench.rag.parent_retrieval import ParentRetrievalOrchestrator, ParentRetrievalRun


@dataclass(frozen=True, slots=True)
class ParentContextRun:
    retrieval: ParentRetrievalRun
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context: RAGContext


class ParentContextPipeline:
    def __init__(
        self,
        *,
        retrieval_orchestrator: ParentRetrievalOrchestrator,
        chunk_pipeline: RepresentativeChunkRetrievalPipeline,
        context_builder: RepresentativeChunkContextBuilder,
        top_parents: int,
    ) -> None:
        if top_parents <= 0:
            raise ValueError("top_parents must be positive")

        self._retrieval_orchestrator = retrieval_orchestrator
        self._chunk_pipeline = chunk_pipeline
        self._context_builder = context_builder
        self._top_parents = top_parents

    def run(self, query: str) -> ParentContextRun:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must be non-empty")

        retrieval = self._retrieval_orchestrator.run(normalized_query)
        retrieved_chunks = tuple(
            self._chunk_pipeline.retrieve(
                retrieval,
                top_k=self._top_parents,
            )
        )
        context = self._context_builder.build(retrieved_chunks)

        return ParentContextRun(
            retrieval=retrieval,
            retrieved_chunks=retrieved_chunks,
            context=context,
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
        )
