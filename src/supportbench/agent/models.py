from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from supportbench.tools.models import (
    ToolCall,
    ToolResult,
)

type AgentRunStatus = Literal[
    "completed",
    "approval_required",
    "max_steps_exceeded",
]


@dataclass(frozen=True, slots=True)
class AgentToolExecution:
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class AgentStep:
    step_index: int
    assistant_content: str
    tool_executions: tuple[AgentToolExecution, ...]


@dataclass(frozen=True, slots=True)
class AgentApprovalRequest:
    approval_id: str
    call: ToolCall
    remaining_calls: tuple[ToolCall, ...]  # calls after pending approval call
    step_index: int


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    status: AgentRunStatus
    final_answer: str | None
    steps: tuple[AgentStep, ...]
    messages: tuple[Mapping[str, object], ...]
    pending_approval: AgentApprovalRequest | None

    def __post_init__(self) -> None:
        if self.status == "completed":
            if self.final_answer is None or not self.final_answer.strip():
                raise ValueError("completed agent run must have a non-empty final answer")

            if self.pending_approval is not None:
                raise ValueError("completed agent run cannot have a pending approval")

        elif self.status == "approval_required":
            if self.final_answer is not None:
                raise ValueError("approval-required run cannot have a final answer")

            if self.pending_approval is None:
                raise ValueError("approval-required run must contain a pending approval")

        elif self.status == "max_steps_exceeded":
            if self.final_answer is not None:
                raise ValueError("max-steps run cannot have a final answer")

            if self.pending_approval is not None:
                raise ValueError("max-steps run cannot have a pending approval")
