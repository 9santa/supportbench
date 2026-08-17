from supportbench.agent.models import (
    AgentRunResult,
    AgentToolExecution,
)
from supportbench.agentbench.models import (
    AgentBenchScenario,
    AgentBenchTrajectoryMetrics,
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

    missing_required = tuple(sorted(scenario.required_tools - used_tools))

    forbidden_used = tuple(sorted(scenario.forbidden_tools & used_tools))

    required_tool_recall = (
        1.0
        if not scenario.required_tools
        else (len(scenario.required_tools & used_tools) / len(scenario.required_tools))
    )

    tool_errors = [execution for execution in executions if execution.result.status == "error"]

    approval_required = [execution for execution in tool_errors if _is_approval_required(execution)]

    unexpected_errors = [
        execution for execution in tool_errors if not _is_approval_required(execution)
    ]

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
        required_tool_recall=(required_tool_recall),
        missing_required_tools=(missing_required),
        forbidden_tool_call_count=sum(
            1 for tool_name in tool_names if tool_name in scenario.forbidden_tools
        ),
        forbidden_tools_used=(forbidden_used),
        tool_call_count=tool_call_count,
        unique_tool_count=len(used_tools),
        tool_error_count=len(tool_errors),
        unexpected_tool_error_count=len(unexpected_errors),
        approval_required_count=len(approval_required),
        step_count=len(result.steps),
        within_tool_budget=(within_tool_budget),
        trajectory_success=(trajectory_success),
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
