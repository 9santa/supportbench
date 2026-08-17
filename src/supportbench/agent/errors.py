class AgentOrchestrationError(Exception):
    """Agent orchestration invariant was violated."""


class EmptyAssistantTurnError(AgentOrchestrationError):
    def __init__(
        self,
        *,
        step_index: int,
        finish_reason: str | None = None,
        output_token_count: int | None = None,
    ) -> None:
        self.step_index = step_index
        self.finish_reason = finish_reason
        self.output_token_count = output_token_count

        super().__init__(
            "assistant returned neither "
            "tool calls nor a non-empty "
            "final answer at "
            f"step {step_index}; "
            f"finish_reason={finish_reason!r}; "
            "output_token_count="
            f"{output_token_count!r}"
        )
