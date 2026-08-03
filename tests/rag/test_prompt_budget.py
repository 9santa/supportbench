from typing import Any

from supportbench.rag.generation.prompt import (
    GroundedPromptBuilder,
    PromptBudgetCalculator,
)
from supportbench.rag.models import RAGContext


class WhitespaceChatTokenizer:
    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text.split())))


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
