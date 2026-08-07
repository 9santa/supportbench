from typing import Any

from supportbench.rag.generation.prompt import (
    GroundedPromptBuilder,
    PromptBudgetCalculator,
)
from supportbench.rag.models import RAGContext


class WhitespaceChatTokenizer:
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        rendered = "<bos> " + " ".join(
            f"<start_of_turn>{message['role']} {message['content']} <end_of_turn>"
            for message in messages
        )
        rendered += " <start_of_turn>model"
        input_ids = self.encode(rendered)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


def test_gemma_prompt_uses_one_user_turn() -> None:
    messages = GroundedPromptBuilder().build(
        query="How do I restart it?",
        context=RAGContext(
            documents=(),
            formatted_text="source_id: S1",
            truncated=False,
        ),
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "Always answer in English" in messages[0].content
    assert "[USER_QUERY]\nHow do I restart it?\n[/USER_QUERY]" in messages[0].content
    assert "source_id: S1" in messages[0].content


def test_legacy_prompt_layout_remains_available_for_paired_evaluation() -> None:
    messages = GroundedPromptBuilder(layout="legacy_system_user").build(
        query="question",
        context=RAGContext(documents=(), formatted_text="context", truncated=False),
    )

    assert tuple(message.role for message in messages) == ("system", "user")


def test_legacy_prompt_budget_counts_two_ollama_user_turns() -> None:
    tokenizer = WhitespaceChatTokenizer()
    builder = GroundedPromptBuilder(layout="legacy_system_user")
    context = RAGContext(documents=(), formatted_text="alpha beta", truncated=False)
    messages = builder.build(query="question", context=context)
    calculator = PromptBudgetCalculator(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt_builder=builder,
        model_context_window=10_000,
        reserved_output_tokens=100,
        max_context_tokens=200,
    )
    rendered = "<bos>" + "".join(
        f"<start_of_turn>user\n{message.content}<end_of_turn>\n" for message in messages
    )
    rendered += "<start_of_turn>model\n"

    assert calculator.count_prompt(query="question", context=context) == len(
        tokenizer.encode(rendered)
    )


def test_prompt_budget_reserves_fixed_prompt_and_output_tokens() -> None:
    tokenizer = WhitespaceChatTokenizer()
    initial = PromptBudgetCalculator(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt_builder=GroundedPromptBuilder(),
        model_context_window=10_000,
        reserved_output_tokens=100,
        max_context_tokens=200,
    ).calculate("How do I restart it?")
    calculator = PromptBudgetCalculator(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt_builder=GroundedPromptBuilder(),
        model_context_window=initial.fixed_prompt_tokens + 100 + 37,
        reserved_output_tokens=100,
        max_context_tokens=200,
    )

    budget = calculator.calculate("How do I restart it?")

    assert budget.available_context_tokens == 37
    assert (
        budget.fixed_prompt_tokens
        + budget.available_context_tokens
        + budget.reserved_output_tokens
        == budget.model_context_window
    )


def test_full_prompt_count_includes_knowledge_context() -> None:
    calculator = PromptBudgetCalculator(
        tokenizer=WhitespaceChatTokenizer(),  # type: ignore[arg-type]
        prompt_builder=GroundedPromptBuilder(),
        model_context_window=10_000,
        reserved_output_tokens=100,
        max_context_tokens=200,
    )
    budget = calculator.calculate("question")
    context = RAGContext(
        documents=(),
        formatted_text="alpha beta",
        truncated=False,
    )

    assert calculator.count_prompt(query="question", context=context) == (
        budget.fixed_prompt_tokens + 2
    )
