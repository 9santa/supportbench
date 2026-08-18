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

AgentBenchApprovalMode = Literal[
    "none",  # do not approve
    "approve",  # auto approve exact pending call
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

    approval_mode: AgentBenchApprovalMode = "none"

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
    gateway_execution_count: int
    logical_tool_call_count: int
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
class AgentBenchApprovalMetrics:
    approval_requested: bool

    pre_approval_state_unchanged: bool | None

    approval_resumed: bool

    approval_flow_correct: bool


@dataclass(frozen=True, slots=True)
class AgentBenchCaseFailure:
    scenario_id: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class AgentBenchCaseResult:
    scenario_id: str

    run: AgentRunResult

    before: AgentBenchWorldSnapshot
    pre_approval: AgentBenchWorldSnapshot | None
    after: AgentBenchWorldSnapshot

    trajectory: AgentBenchTrajectoryMetrics
    state: AgentBenchStateMetrics
    approval: AgentBenchApprovalMetrics

    success: bool


@dataclass(frozen=True, slots=True)
class AgentBenchSuiteResult:
    case_results: tuple[AgentBenchCaseResult, ...]
    case_failures: tuple[AgentBenchCaseFailure, ...]

    @property
    def total_count(self) -> int:
        return len(self.case_results) + len(self.case_failures)

    @property
    def successful_count(self) -> int:
        return sum(1 for result in self.case_results if result.success)

    @property
    def unsuccessful_count(self) -> int:
        return self.total_count - self.successful_count

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0

        return self.successful_count / self.total_count


@dataclass(frozen=True, slots=True)
class AgentBenchSuiteMetrics:
    total_cases: int
    successful_cases: int
    execution_failures: int

    success_rate: float

    mean_required_tool_recall: float
    mean_tool_calls: float
    mean_steps: float

    forbidden_tool_call_count: int
    unexpected_tool_error_count: int
    approval_flow_failure_count: int


@dataclass(frozen=True, slots=True)
class AgentBenchRunConfig:
    suite_name: str
    model_name: str
    think: bool

    prompt_version: str
    retrieval_config: str

    max_steps: int
