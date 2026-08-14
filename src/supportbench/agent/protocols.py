from collections.abc import Mapping, Sequence
from typing import Protocol

from supportbench.llm.models import AssistantModelTurn
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import (
    ToolCall,
    ToolExecutionContext,
    ToolResult,
)


class AgentModelClient(Protocol):
    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[ToolDefinition],
        request_id: str,
        assistant_turn_index: int,
    ) -> AssistantModelTurn: ...

    def tool_result_message(
        self,
        result: ToolResult,
    ) -> Mapping[str, object]: ...


class AgentToolGateway(Protocol):
    @property
    def definitions(self) -> tuple[ToolDefinition, ...]: ...

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult: ...
