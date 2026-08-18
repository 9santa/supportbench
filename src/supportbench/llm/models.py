from collections.abc import Mapping
from dataclasses import dataclass

from supportbench.tools.models import ToolCall


@dataclass(frozen=True, slots=True)
class AssistantModelTurn:
    content: str
    tool_calls: tuple[ToolCall, ...]

    # Ollama-form assistant message that must be
    # appended to conversation history before tool results.
    history_message: Mapping[str, object]
    finish_reason: str | None = None
    output_token_count: int | None = None
