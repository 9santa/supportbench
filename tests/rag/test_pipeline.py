from supportbench.rag.context import ContextPreparationRun, ContextPreparationService
from supportbench.rag.generation.models import ChatMessage, LLMResponse
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import ChunkProvenance, RAGContext, RetrievedDocument
from supportbench.rag.pipeline import RAGPipeline
from supportbench.rag.retrieval import ParentRetrievalRun


class StaticContextService(ContextPreparationService):
    def __init__(self, run: ContextPreparationRun) -> None:
        self._run = run

    def prepare(self, query: str) -> ContextPreparationRun:
        return self._run


class StubLLMClient:
    def generate(self, messages: tuple[ChatMessage, ...]) -> LLMResponse:
        return LLMResponse(
            content=(
                '{"decision":"answer","answer":"Use A.","citation_ids":["S1"]}'
            ),
            done_reason="stop",
        )


def test_preserves_parent_retrieval_diagnostics_through_generation() -> None:
    retrieval = ParentRetrievalRun(
        candidate_parents=(),
        representative_chunks_by_parent={},
        reranked_parents=(),
        fused_parents=(),
    )
    context = RAGContext(
        documents=(
            RetrievedDocument(
                doc_id="parent_a",
                title="Document A",
                text="Use A.",
                category="support",
                score=1.0,
                rank=1,
            ),
        ),
        formatted_text="[DOCUMENT]\nsource_id: S1\ncontent:\nUse A.\n[/DOCUMENT]",
        truncated=False,
        provenance=(
            ChunkProvenance(
                parent_doc_id="parent_a",
                chunk_id="parent_a::chunk_0001",
                parent_rank=1,
                evidence_rank=1,
                document_title="Document A",
                section_path=("Resolution",),
                ordinal=1,
                source_start_char=0,
                source_end_char=6,
                included_start_char=0,
                included_end_char=6,
                removed_prefix_tokens=0,
                included_tokens=2,
                truncated=False,
                source_id="S1",
            ),
        ),
    )
    context_run = ContextPreparationRun(
        retrieval=retrieval,
        retrieved_chunks=(),
        context=context,
    )
    pipeline = RAGPipeline(
        context_service=StaticContextService(context_run),
        answer_generator=GroundedAnswerGenerator(
            prompt_builder=GroundedPromptBuilder(),
            llm_client=StubLLMClient(),
        ),
    )

    run = pipeline.run("question")

    assert run.retrieval is retrieval
    assert run.context is context
    assert run.answer.citation_ids == ("parent_a",)
    assert run.resolved_citation_ids == ("parent_a",)
    assert run.contract_repaired is False
    assert run.strict_contract_valid is True
