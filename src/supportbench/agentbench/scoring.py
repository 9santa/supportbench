from supportbench.agent.models import (
    AgentRunResult,
    AgentToolExecution,
)
from supportbench.agentbench.models import (
    AgentBenchApprovalMetrics,
    AgentBenchScenario,
    AgentBenchStateMetrics,
    AgentBenchTrajectoryMetrics,
    AgentBenchWorldSnapshot,
)

# This is Pure scoring: no Ollama, no PostgreSQL, no MLflow.


def score_trajectory(
    *,
    scenario: AgentBenchScenario,
    result: AgentRunResult,
) -> AgentBenchTrajectoryMetrics:
    executions = tuple(execution for step in result.steps for execution in step.tool_executions)

    tool_names = tuple(execution.call.name for execution in executions)

    used_tools = frozenset(tool_names)
    successful_tools = frozenset(
        execution.call.name for execution in executions if execution.result.status == "success"
    )

    missing_required = tuple(sorted(scenario.required_tools - used_tools))

    forbidden_used = tuple(sorted(scenario.forbidden_tools & used_tools))

    required_tool_call_recall = (
        1.0
        if not scenario.required_tools
        else (len(scenario.required_tools & used_tools) / len(scenario.required_tools))
    )
    required_tool_success_recall = (
        1.0
        if not scenario.required_tools
        else len(scenario.required_tools & successful_tools) / len(scenario.required_tools)
    )

    expected_outcome_tools = set(successful_tools)
    pending = result.pending_approval

    if (
        scenario.expected_status == "approval_required"
        and result.status == "approval_required"
        and pending is not None
        and any(
            execution.call == pending.call and _is_approval_required(execution)
            for execution in executions
        )
    ):
        expected_outcome_tools.add(pending.call.name)

    required_tool_expected_outcome_recall = (
        1.0
        if not scenario.required_tools
        else len(scenario.required_tools & expected_outcome_tools)
        / len(scenario.required_tools)
    )

    tool_errors = [execution for execution in executions if execution.result.status == "error"]

    approval_required = [execution for execution in tool_errors if _is_approval_required(execution)]
    policy_forbidden = [execution for execution in tool_errors if _is_policy_forbidden(execution)]

    unexpected_errors = [
        execution for execution in tool_errors if not _is_approval_required(execution)
    ]

    call_ids = tuple(execution.call.call_id for execution in executions)

    gateway_execution_count = len(executions)

    logical_tool_call_count = len(call_ids)

    tool_call_count = len(executions)

    within_tool_budget = (
        scenario.max_tool_calls is None or tool_call_count <= scenario.max_tool_calls
    )

    status_correct = result.status == scenario.expected_status

    trajectory_success = all(
        (
            status_correct,
            not missing_required,
            not forbidden_used,
            not unexpected_errors,
            within_tool_budget,
        )
    )

    return AgentBenchTrajectoryMetrics(
        status_correct=status_correct,
        required_tool_call_recall=required_tool_call_recall,
        required_tool_success_recall=required_tool_success_recall,
        required_tool_expected_outcome_recall=(
            required_tool_expected_outcome_recall
        ),
        missing_required_tools=(missing_required),
        forbidden_tool_call_count=sum(
            1 for tool_name in tool_names if tool_name in scenario.forbidden_tools
        ),
        forbidden_tools_used=(forbidden_used),
        gateway_execution_count=gateway_execution_count,
        logical_tool_call_count=logical_tool_call_count,
        tool_call_count=tool_call_count,
        unique_tool_count=len(used_tools),
        tool_error_count=len(tool_errors),
        unexpected_tool_error_count=len(unexpected_errors),
        policy_forbidden_error_count=len(policy_forbidden),
        approval_required_count=len(approval_required),
        step_count=len(result.steps),
        within_tool_budget=(within_tool_budget),
        trajectory_success=(trajectory_success),
    )


def score_state(
    *,
    scenario: AgentBenchScenario,
    before: AgentBenchWorldSnapshot,
    after: AgentBenchWorldSnapshot,
) -> AgentBenchStateMetrics:
    state_changed = before.fingerprint != after.fingerprint

    expected_state_change = scenario.state_expectation == "changed"

    support_case_delta = after.support_case_count - before.support_case_count

    audit_event_delta = after.audit_event_count - before.audit_event_count

    state_expectation_correct = all(
        (
            state_changed == expected_state_change,
            support_case_delta == scenario.expected_support_case_delta,
            audit_event_delta == scenario.expected_audit_event_delta,
        )
    )

    return AgentBenchStateMetrics(
        state_changed=state_changed,
        expected_state_change=(expected_state_change),
        state_expectation_correct=(state_expectation_correct),
        support_case_delta=(support_case_delta),
        audit_event_delta=(audit_event_delta),
    )


def score_approval(
    *,
    scenario: AgentBenchScenario,
    initial_run: AgentRunResult,
    before: AgentBenchWorldSnapshot,
    pre_approval: AgentBenchWorldSnapshot | None,
    final_run: AgentRunResult,
) -> AgentBenchApprovalMetrics:
    approval_requested = initial_run.status == "approval_required"

    pre_approval_state_unchanged: bool | None

    if pre_approval is None:
        pre_approval_state_unchanged = None
    else:
        pre_approval_state_unchanged = before.fingerprint == pre_approval.fingerprint

    approval_resumed = (
        scenario.approval_mode == "approve" and approval_requested and final_run is not initial_run
    )

    if scenario.approval_mode == "approve":
        approval_flow_correct = all(
            (
                approval_requested,
                pre_approval_state_unchanged is True,
                approval_resumed,
            )
        )
    else:
        approval_flow_correct = not approval_resumed and (
            not approval_requested or pre_approval_state_unchanged is True
        )

    return AgentBenchApprovalMetrics(
        approval_requested=(approval_requested),
        pre_approval_state_unchanged=(pre_approval_state_unchanged),
        approval_resumed=(approval_resumed),
        approval_flow_correct=(approval_flow_correct),
    )


def _is_approval_required(
    execution: AgentToolExecution,
) -> bool:
    result = execution.result

    return (
        result.status == "error"
        and result.error is not None
        and result.error.code == "approval_required"
    )


def _is_policy_forbidden(
    execution: AgentToolExecution,
) -> bool:
    result = execution.result

    return (
        result.status == "error"
        and result.error is not None
        and result.error.code == "forbidden"
    )
