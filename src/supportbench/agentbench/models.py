from dataclasses import dataclass
from typing import Literal

from supportbench.agent.models import AgentRunResult, AgentRunStatus


AgentBenchScenarioKind = Literal[
    "enterprise",
    "knowledge",
    "mixed",
    "write",
    "safety",
]

AgentBenchStateExpectation = Literal[
    "unchanged",
    "changed",
]


@dataclass(frozen=True, slots=True)
class AgentBenchScenario:
    scenario_id: str
    kind: AgentBenchScenarioKind

    world_scenario: str
    user_message: str

    permissions: frozenset[str]

    expected_status: AgentRunStatus

    required_tools: frozenset[str] = frozenset()
    forbidden_tools: frozenset[str] = frozenset()

    max_tool_calls: int | None = None

    state_expectation: AgentBenchStateExpectation = "unchanged"

    expected_support_case_delta: int = 0
    expected_audit_event_delta: int = 0

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty")

        if not self.world_scenario.strip():
            raise ValueError("world_scenario must be non-empty")

        if not self.user_message.strip():
            raise ValueError("user_message must be non-empty")

        overlap = self.required_tools & self.forbidden_tools

        if overlap:
            raise ValueError(f"tools cannot be both required and forbidden: {sorted(overlap)}")

        if self.max_tool_calls is not None and self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")


@dataclass(frozen=True, slots=True)
class AgentBenchTrajectoryMetrics:
    status_correct: bool

    required_tool_recall: float
    missing_required_tools: tuple[str, ...]

    forbidden_tool_call_count: int
    forbidden_tools_used: tuple[str, ...]

    tool_call_count: int
    unique_tool_count: int

    tool_error_count: int
    unexpected_tool_error_count: int

    approval_required_count: int

    step_count: int

    within_tool_budget: bool

    trajectory_success: bool


@dataclass(frozen=True, slots=True)
class AgentBenchWorldSnapshot:
    fingerprint: str
    support_case_count: int
    audit_event_count: int


@dataclass(frozen=True, slots=True)
class AgentBenchStateMetrics:
    state_changed: bool
    expected_state_change: bool
    state_expectation_correct: bool

    support_case_delta: int
    audit_event_delta: int


@dataclass(frozen=True, slots=True)
class AgentBenchCaseResult:
    scenario_id: str

    run: AgentRunResult

    before: AgentBenchWorldSnapshot
    after: AgentBenchWorldSnapshot

    trajectory: AgentBenchTrajectoryMetrics
    state: AgentBenchStateMetrics

    success: bool
