from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal


type ToolStatus = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_non_empty("call_id", self.call_id)
        _require_non_empty("name", self.name)


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    world_id: str
    actor_user_id: str
    request_id: str

    def __post_init__(self) -> None:
        _require_non_empty("world_id", self.world_id)
        _require_non_empty("actor_user_id", self.actor_user_id)
        _require_non_empty("request_id", self.request_id)


@dataclass(frozen=True, slots=True)
class ToolErrorInfo:
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolStatus
    data: Mapping[str, object] | None
    error: ToolErrorInfo | None

    def __post_init__(self) -> None:
        if self.status == "success":
            if self.data is None:
                raise ValueError("successful tool result must contain data")

            if self.error is not None:
                raise ValueError("successful tool result cannot contain error")

        elif self.status == "error":
            if self.data is not None:
                raise ValueError("failed tool result cannot contain data")

            if self.error is None:
                raise ValueError("failed tool result must contain error")


def _require_non_empty(
    name: str,
    value: str,
) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
