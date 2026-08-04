import pytest

from supportbench.rag.citations import CitationValidationError
from supportbench.rag.generation.models import ChatMessage
from supportbench.rag.generation.prompt import GroundedPromptBuilder
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import RAGContext, RetrievedDocument


class StubLLMClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0
        self.messages: tuple[ChatMessage, ...] = ()

    def generate(self, messages: tuple[ChatMessage, ...]) -> str:
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
