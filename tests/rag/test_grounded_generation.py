from dataclasses import replace

import pytest

from supportbench.rag.citations import (
    CitationContractError,
    CitationResolutionError,
)
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
        formatted_text=(
            "[DOCUMENT]\nsource_id: S1\ntitle: Document A\n"
            "section: Resolution\ncontent:\nRestart the service.\n[/DOCUMENT]"
        ),
        truncated=False,
        token_count=12,
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
                source_end_char=20,
                included_start_char=0,
                included_end_char=20,
                removed_prefix_tokens=0,
                included_tokens=4,
                truncated=False,
                source_id="S1",
            ),
        ),
    )


def _generator(client: StubLLMClient) -> GroundedAnswerGenerator:
    return GroundedAnswerGenerator(
        prompt_builder=GroundedPromptBuilder(),
        llm_client=client,
    )


def test_generates_and_validates_grounded_answer() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.","citation_ids":["S1"]}'
    )

    run = _generator(client).generate(query="What should I do?", context=_context())

    assert client.calls == 1
    assert run.answer.citation_ids == ("parent_a",)
    assert run.resolved_citation_ids == ("parent_a",)
    assert run.contract_repaired is False
    assert run.strict_contract_valid is True
    assert run.contract_violations == ()
    assert run.raw_response is not None
    assert run.llm_response is not None
    assert run.llm_response.eval_count == 20
    assert "Always answer in English" in client.messages[0].content
    assert "source_id: S1" in client.messages[1].content
    assert "parent_a" not in client.messages[1].content
    assert "chunk_0001" not in client.messages[1].content


def test_empty_context_abstains_without_calling_llm() -> None:
    client = StubLLMClient("not used")
    context = RAGContext(documents=(), formatted_text="", truncated=False)

    run = _generator(client).generate(query="question", context=context)

    assert client.calls == 0
    assert run.answer.decision == "abstain"
    assert run.messages == ()
    assert run.raw_response is None


def test_unknown_source_preserves_parsed_response() -> None:
    raw_response = (
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["S9"]}'
    )
    client = StubLLMClient(raw_response)

    with pytest.raises(CitationResolutionError) as caught:
        _generator(client).generate(query="question", context=_context())

    assert caught.value.raw_response == raw_response
    assert caught.value.parsed_answer is not None
    assert caught.value.parsed_answer.decision == "answer"
    assert caught.value.raw_citation_ids == ("S9",)
    assert caught.value.citation_ids == ()


def test_source_citation_is_resolved_to_parent_id() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["S1"]}'
    )

    run = _generator(client).generate(query="question", context=_context())

    assert run.raw_citation_ids == ("S1",)
    assert run.answer.citation_ids == ("parent_a",)


def test_multiple_sources_from_one_parent_are_deduplicated() -> None:
    context = _context()
    second_source = replace(
        context.provenance[0],
        source_id="S2",
        chunk_id="parent_a::chunk_0002",
    )
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["S1","S2"]}'
    )

    run = _generator(client).generate(
        query="question",
        context=replace(
            context,
            provenance=(*context.provenance, second_source),
        ),
    )

    assert run.raw_citation_ids == ("S1", "S2")
    assert run.answer.citation_ids == ("parent_a",)


def test_legacy_context_can_still_resolve_parent_id() -> None:
    context = replace(
        _context(),
        formatted_text=(
            "[DOCUMENT]\ndoc_id: parent_a\ncontent:\n"
            "Restart the service.\n[/DOCUMENT]"
        ),
        provenance=(),
    )
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.",'
        '"citation_ids":["parent_a"]}'
    )

    run = _generator(client).generate(query="question", context=context)

    assert run.answer.citation_ids == ("parent_a",)


def test_abstain_with_citations_is_safely_repaired() -> None:
    client = StubLLMClient(
        '{"decision":"abstain","answer":"Not enough evidence.",'
        '"citation_ids":["S1"]}'
    )

    run = _generator(client).generate(query="question", context=_context())

    assert run.answer.decision == "abstain"
    assert run.answer.answer == "Not enough evidence."
    assert run.raw_citation_ids == ("S1",)
    assert run.resolved_citation_ids == ("parent_a",)
    assert run.answer.citation_ids == ()
    assert run.contract_repaired is True
    assert run.strict_contract_valid is False
    assert run.contract_violations == ("non_answer_has_citations",)


def test_clarify_with_citations_is_safely_repaired() -> None:
    client = StubLLMClient(
        '{"decision":"clarify","answer":"Which version is installed?",'
        '"citation_ids":["S1"]}'
    )

    run = _generator(client).generate(query="question", context=_context())

    assert run.answer.decision == "clarify"
    assert run.raw_citation_ids == ("S1",)
    assert run.resolved_citation_ids == ("parent_a",)
    assert run.answer.citation_ids == ()
    assert run.contract_repaired is True
    assert run.strict_contract_valid is False
    assert run.contract_violations == ("non_answer_has_citations",)


def test_abstain_without_citations_is_strictly_valid() -> None:
    client = StubLLMClient(
        '{"decision":"abstain","answer":"Not enough evidence.","citation_ids":[]}'
    )

    run = _generator(client).generate(query="question", context=_context())

    assert run.answer.citation_ids == ()
    assert run.raw_citation_ids == ()
    assert run.resolved_citation_ids == ()
    assert run.contract_repaired is False
    assert run.strict_contract_valid is True
    assert run.contract_violations == ()


def test_answer_without_citations_remains_contract_error() -> None:
    client = StubLLMClient(
        '{"decision":"answer","answer":"Restart it.","citation_ids":[]}'
    )

    with pytest.raises(CitationContractError) as caught:
        _generator(client).generate(query="question", context=_context())

    assert caught.value.parsed_answer is not None
    assert caught.value.parsed_answer.decision == "answer"
    assert caught.value.raw_citation_ids == ()
    assert caught.value.citation_ids == ()
    assert caught.value.contract_violations == ("answer_has_no_citations",)


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
