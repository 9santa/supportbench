from supportbench.evaluation.grounded_generation import evaluate_grounded_generation
from supportbench.rag.generation.models import GeneratedAnswer, LLMResponse
from supportbench.rag.generation.service import GroundedGenerationRun
from supportbench.rag.models import RAGContext


class StubGenerator:
    def generate(self, *, query: str, context: RAGContext) -> GroundedGenerationRun:
        assert query == "How do I fix it?"
        assert context.formatted_text == "evidence"
        return GroundedGenerationRun(
            messages=(),
            raw_response="raw JSON",
            answer=GeneratedAnswer(
                decision="answer",
                answer="Restart the service.",
                citation_ids=("gold",),
            ),
            raw_citation_ids=("S1",),
            resolved_citation_ids=("gold",),
            llm_response=LLMResponse(
                content="raw JSON",
                done_reason="stop",
                prompt_eval_count=20,
                eval_count=5,
            ),
        )


def test_evaluates_generation_from_an_existing_context() -> None:
    result = evaluate_grounded_generation(
        query="How do I fix it?",
        reference_answer="Restart the service.",
        relevant_doc_ids=("gold",),
        context=RAGContext(
            documents=(),
            formatted_text="evidence",
            truncated=False,
        ),
        generator=StubGenerator(),
        retries=0,
    )

    assert result["status"] == "success"
    assert result["decision"] == "answer"
    assert result["citation_ids"] == ["gold"]
    assert result["gold_document_cited"] is True
    assert result["reference_token_f1"] == 1.0
    assert result["prompt_eval_count"] == 20
