import pytest

from supportbench.rag.citations import CitationValidationError
from supportbench.rag.generation.models import ChatMessage, LLMResponse
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.generation.service import (
    GenerationTruncatedError,
    GroundedAnswerGenerator,
)
from supportbench.rag.models import ChunkProvenance, RAGContext, RetrievedDocument


class StubLLMClient:
    def __init__(
        self,
        response: str,
        *,
        done_reason: str = "stop",
        prompt_eval_count: int = 100,
        eval_count: int = 20,
    ) -> None:
        self._response = LLMResponse(
            content=response,
            done_reason=done_reason,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )
        self.calls = 0
        self.messages: tuple[ChatMessage, ...] = ()

    def generate(self, messages: tuple[ChatMessage, ...]) -> LLMResponse:
        self.calls += 1
        self.messages = messages
        return self._response


def _context() -> RAGContext:
    return RAGContext(
        documents=(
            RetrievedDocument(
                doc_id="parent_a",
                title="Document A",
                text="Restart the service.",
                category="support",
                score=0.9,
                rank=1,
            ),
        ),
        formatted_text="[DOCUMENT]\ndoc_id: parent_a\ncontent:\nRestart the service.\n[/DOCUMENT]",
        truncated=False,
        token_count=12,
    )


def _generator(client: StubLLMClient) -> GroundedAnswerGenerator:
    return GroundedAnswerGenerator(
        prompt_builder=GroundedPromptBuilder(),
        llm_client=client,
    )


def test_generates_and_validates_grounded_answer() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.","citation_ids":["parent_a"]}'
    )

    run = _generator(client).generate(query="What should I do?", context=_context())

    assert client.calls == 1
    assert run.answer.citation_ids == ("parent_a",)
    assert run.raw_response is not None
    assert run.llm_response is not None
    assert run.llm_response.eval_count == 20
    assert "Отвечай на языке запроса пользователя" in client.messages[0].content
    assert "doc_id: parent_a" in client.messages[1].content


def test_empty_context_abstains_without_calling_llm() -> None:
    client = StubLLMClient("not used")
    context = RAGContext(documents=(), formatted_text="", truncated=False)

    run = _generator(client).generate(query="question", context=context)

    assert client.calls == 0
    assert run.answer.decision == "abstain"
    assert run.messages == ()
    assert run.raw_response is None


def test_invalid_citation_preserves_raw_response() -> None:
    raw_response = (
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["unknown_parent"]}'
    )
    client = StubLLMClient(raw_response)

    with pytest.raises(CitationValidationError) as caught:
        _generator(client).generate(query="question", context=_context())

    assert caught.value.raw_response == raw_response


def test_chunk_citation_is_normalized_to_parent_id() -> None:
    context = RAGContext(
        documents=_context().documents,
        formatted_text=_context().formatted_text,
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
                source_end_char=10,
                included_start_char=0,
                included_end_char=10,
                removed_prefix_tokens=0,
                included_tokens=4,
                truncated=False,
            ),
        ),
    )
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["parent_a::chunk_0001"]}'
    )

    run = _generator(client).generate(query="question", context=context)

    assert run.raw_citation_ids == ("parent_a::chunk_0001",)
    assert run.answer.citation_ids == ("parent_a",)


def test_labeled_citation_handle_is_normalized_to_parent_id() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["doc_id: parent_a"]}'
    )

    run = _generator(client).generate(query="question", context=_context())

    assert run.raw_citation_ids == ("doc_id: parent_a",)
    assert run.answer.citation_ids == ("parent_a",)


def test_output_limit_is_reported_as_generation_truncated() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"unfinished',
        done_reason="length",
        eval_count=512,
    )

    with pytest.raises(GenerationTruncatedError) as caught:
        _generator(client).generate(query="question", context=_context())

    assert caught.value.raw_response.endswith("unfinished")
    assert caught.value.llm_response.eval_count == 512
