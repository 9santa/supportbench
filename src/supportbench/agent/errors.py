class AgentOrchestrationError(Exception):
    """Agent orchestration invariant was violated."""


class EmptyAssistantTurnError(AgentOrchestrationError):
    def __init__(self, *, step_index: int) -> None:
        self.step_index = step_index

        super().__init__(
            "assistant returned neither tool calls nor a non-empty final aswer at "
            f"step {step_index}"
        )
