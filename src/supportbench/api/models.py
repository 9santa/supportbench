from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


WorldScenarioName = Literal[
    "healthy",
    "dash_outage",
    "old_dash_version",
    "access_denied",
]


class CreateWorldRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    scenario: WorldScenarioName


class CreateWorldResponse(BaseModel):
    world_id: str
    scenario: WorldScenarioName


class DeleteWorldResponse(BaseModel):
    world_id: str
    deleted: bool


class CreateAgentRunRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    world_id: str = Field(
        min_length=1,
        max_length=256,
    )

    message: str = Field(
        min_length=1,
        max_length=10_000,
    )


class ToolExecutionResponse(BaseModel):
    call_id: str
    tool_name: str

    arguments: dict[str, object]

    status: Literal[
        "success",
        "error",
    ]

    error_code: str | None = None


class PendingApprovalResponse(BaseModel):
    tool_name: str

    arguments: dict[str, object]


class AgentRunResponse(BaseModel):
    run_id: str
    world_id: str

    status: Literal[
        "completed",
        "approval_required",
        "max_steps_exceeded",
    ]

    final_answer: str | None

    pending_approval: PendingApprovalResponse | None = None

    tool_executions: tuple[ToolExecutionResponse, ...] = ()


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
