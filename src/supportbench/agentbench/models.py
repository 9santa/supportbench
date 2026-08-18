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
class ExpectedAnswerFact:
    fact_id: str
    accepted_phrases: tuple[str, ...] = ()
    source_tool: str | None = None
    source_result_field: str | None = None

    def __post_init__(self) -> None:
        if not self.fact_id.strip():
            raise ValueError("fact_id must be non-empty")

        if any(not phrase.strip() for phrase in self.accepted_phrases):
            raise ValueError("accepted_phrases must contain only non-empty strings")

        source_fields = (self.source_tool, self.source_result_field)

        if (source_fields[0] is None) != (source_fields[1] is None):
            raise ValueError("source_tool and source_result_field must be set together")

        if not self.accepted_phrases and self.source_tool is None:
            raise ValueError("expected fact must define phrases or a tool result source")


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

    expected_answer_facts: tuple[ExpectedAnswerFact, ...] = ()
    expected_evidence_doc_ids: frozenset[str] = frozenset()
    forbidden_answer_claims: tuple[str, ...] = ()

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

        fact_ids = [fact.fact_id for fact in self.expected_answer_facts]

        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("expected answer fact IDs must be unique")

        if any(not document_id.strip() for document_id in self.expected_evidence_doc_ids):
            raise ValueError("expected evidence document IDs must be non-empty")

        if any(not claim.strip() for claim in self.forbidden_answer_claims):
            raise ValueError("forbidden answer claims must be non-empty")


@dataclass(frozen=True, slots=True)
class AgentBenchTrajectoryMetrics:
    status_correct: bool

    required_tool_call_recall: float
    required_tool_success_recall: float
    required_tool_expected_outcome_recall: float
    missing_required_tools: tuple[str, ...]

    forbidden_tool_call_count: int
    forbidden_tools_used: tuple[str, ...]

    gateway_execution_count: int
    logical_tool_call_count: int
    unique_tool_count: int

    tool_error_count: int
    unexpected_tool_error_count: int
    policy_forbidden_error_count: int

    approval_required_count: int

    step_count: int

    prompt_token_count: int | None
    output_token_count: int | None
    model_total_duration_ms: float | None
    model_generation_duration_ms: float | None

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
class AgentBenchAnswerMetrics:
    applicable: bool
    final_answer_present: bool

    expected_fact_recall: float
    missing_expected_facts: tuple[str, ...]

    expected_evidence_recall: float
    missing_expected_evidence_doc_ids: tuple[str, ...]

    forbidden_claim_count: int
    forbidden_claims_found: tuple[str, ...]

    answer_success: bool


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
    answer: AgentBenchAnswerMetrics

    trajectory_state_success: bool
    success: bool


@dataclass(frozen=True, slots=True)
class AgentBenchSuiteResult:
    case_results: tuple[AgentBenchCaseResult, ...]
    case_failures: tuple[AgentBenchCaseFailure, ...]

    @property
    def total_count(self) -> int:
        return len(self.case_results) + len(self.case_failures)

    @property
    def trajectory_state_successful_count(self) -> int:
        return sum(
            1 for result in self.case_results if result.trajectory_state_success
        )

    @property
    def task_successful_count(self) -> int:
        return sum(1 for result in self.case_results if result.success)

    @property
    def unsuccessful_count(self) -> int:
        return self.total_count - self.task_successful_count

    @property
    def trajectory_state_success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0

        return self.trajectory_state_successful_count / self.total_count

    @property
    def task_success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0

        return self.task_successful_count / self.total_count


@dataclass(frozen=True, slots=True)
class AgentBenchSuiteMetrics:
    total_cases: int
    trajectory_state_successful_cases: int
    task_successful_cases: int
    execution_failures: int

    trajectory_state_success_rate: float
    task_success_rate: float

    answer_evaluated_cases: int
    answer_success_rate: float | None
    mean_expected_answer_fact_recall: float | None
    mean_expected_evidence_recall: float | None
    forbidden_answer_claim_count: int

    mean_required_tool_call_recall: float
    mean_required_tool_success_recall: float
    mean_required_tool_expected_outcome_recall: float
    mean_logical_tool_calls: float
    mean_steps: float
    mean_prompt_tokens: float | None
    mean_output_tokens: float | None
    mean_model_total_duration_ms: float | None
    mean_model_generation_duration_ms: float | None

    forbidden_tool_call_count: int
    policy_forbidden_error_count: int
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
