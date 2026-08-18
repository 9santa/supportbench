from collections.abc import Mapping

from supportbench.agent.models import (
    AgentApprovalRequest,
    AgentRunResult,
    AgentStep,
    AgentToolExecution,
)
from supportbench.agentbench.models import AgentBenchScenario
from supportbench.agentbench.scenarios import (
    AGENTBENCH_V1,
    AGENTBENCH_V2,
    MIXED_DASH_WEBGUI,
)
from supportbench.agentbench.scoring import score_trajectory
from supportbench.tools.models import (
    ToolCall,
    ToolErrorInfo,
    ToolResult,
)


def _success_result(
    *executions: AgentToolExecution,
) -> AgentRunResult:
    return AgentRunResult(
        status="completed",
        final_answer="Done.",
        steps=(
            AgentStep(
                step_index=0,
                assistant_content="",
                tool_executions=tuple(executions),
            ),
            AgentStep(
                step_index=1,
                assistant_content="Done.",
                tool_executions=(),
            ),
        ),
        messages=(),
        pending_approval=None,
    )


def _approval_result() -> AgentRunResult:
    call = ToolCall(
        call_id="call-create-support-case",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Web GUI outage",
        },
    )

    approval_execution = _error_execution(
        "create_support_case",
        code="approval_required",
        call_id=call.call_id,
        arguments=call.arguments,
    )

    return AgentRunResult(
        status="approval_required",
        final_answer=None,
        steps=(
            AgentStep(
                step_index=0,
                assistant_content="",
                tool_executions=(approval_execution,),
            ),
        ),
        messages=(),
        pending_approval=AgentApprovalRequest(
            approval_id="approval-001",
            call=call,
            remaining_calls=(),
            step_index=0,
        ),
    )


def _success_execution(
    tool_name: str,
    *,
    call_id: str | None = None,
    arguments: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
) -> AgentToolExecution:
    resolved_call_id = call_id or f"call-{tool_name}"

    call = ToolCall(
        call_id=resolved_call_id,
        name=tool_name,
        arguments=arguments or {},
    )

    result = ToolResult(
        call_id=resolved_call_id,
        tool_name=tool_name,
        status="success",
        data=data or {},
        error=None,
    )

    return AgentToolExecution(
        call=call,
        result=result,
    )


def _error_execution(
    tool_name: str,
    *,
    code: str,
    message: str = "tool failed",
    call_id: str | None = None,
    arguments: Mapping[str, object] | None = None,
) -> AgentToolExecution:
    resolved_call_id = call_id or f"call-{tool_name}"

    call = ToolCall(
        call_id=resolved_call_id,
        name=tool_name,
        arguments=dict(arguments or {}),
    )

    result = ToolResult(
        call_id=resolved_call_id,
        tool_name=tool_name,
        status="error",
        data=None,
        error=ToolErrorInfo(
            code=code,
            message=message,
        ),
    )

    return AgentToolExecution(
        call=call,
        result=result,
    )


def test_mixed_trajectory_succeeds_when_required_tools_are_used() -> None:
    result = _success_result(
        _success_execution("search_products"),
        _success_execution("get_installed_product"),
        _success_execution("search_support_docs"),
    )

    metrics = score_trajectory(
        scenario=MIXED_DASH_WEBGUI,
        result=result,
    )

    assert metrics.status_correct
    assert metrics.required_tool_call_recall == 1.0
    assert metrics.required_tool_success_recall == 1.0
    assert metrics.required_tool_expected_outcome_recall == 1.0
    assert metrics.missing_required_tools == ()
    assert metrics.forbidden_tool_call_count == 0
    assert metrics.unexpected_tool_error_count == 0
    assert metrics.trajectory_success


def test_v2_keeps_scenario_ids_but_relaxes_redundant_deep_read_requirement() -> None:
    assert [scenario.scenario_id for scenario in AGENTBENCH_V2] == [
        scenario.scenario_id for scenario in AGENTBENCH_V1
    ]
    v1 = {scenario.scenario_id: scenario for scenario in AGENTBENCH_V1}
    v2 = {scenario.scenario_id: scenario for scenario in AGENTBENCH_V2}

    assert v1["knowledge-search-and-read"].required_tools == frozenset(
        {"search_support_docs", "read_support_doc"}
    )
    assert v2["knowledge-search-and-read"].required_tools == frozenset(
        {"search_support_docs"}
    )


def test_failed_required_tool_is_not_successfully_recalled() -> None:
    scenario = AgentBenchScenario(
        scenario_id="failed-service-lookup",
        kind="enterprise",
        world_scenario="healthy",
        user_message="Check the service.",
        permissions=frozenset({"enterprise:read"}),
        expected_status="completed",
        required_tools=frozenset({"get_service_status"}),
    )
    result = _success_result(
        _error_execution("get_service_status", code="service_not_found"),
    )

    metrics = score_trajectory(scenario=scenario, result=result)

    assert metrics.required_tool_call_recall == 1.0
    assert metrics.required_tool_success_recall == 0.0
    assert metrics.required_tool_expected_outcome_recall == 0.0
    assert not metrics.trajectory_success


def test_policy_forbidden_errors_are_separate_from_scenario_forbidden_tools() -> None:
    scenario = AgentBenchScenario(
        scenario_id="policy-denied-tool",
        kind="knowledge",
        world_scenario="healthy",
        user_message="Search support docs.",
        permissions=frozenset({"support_docs:read"}),
        expected_status="completed",
        required_tools=frozenset({"search_support_docs"}),
        forbidden_tools=frozenset({"create_support_case"}),
    )
    result = _success_result(
        _error_execution("search_products", code="forbidden"),
        _success_execution("search_support_docs"),
    )

    metrics = score_trajectory(scenario=scenario, result=result)

    assert metrics.forbidden_tool_call_count == 0
    assert metrics.policy_forbidden_error_count == 1
    assert metrics.unexpected_tool_error_count == 1


def test_forbidden_mutation_fails_trajectory() -> None:
    result = _success_result(
        _success_execution("get_installed_product"),
        _success_execution("search_support_docs"),
        _success_execution("create_support_case"),
    )

    metrics = score_trajectory(
        scenario=MIXED_DASH_WEBGUI,
        result=result,
    )

    assert metrics.forbidden_tool_call_count == 1

    assert not metrics.trajectory_success


def test_approval_required_is_not_an_unexpected_tool_error() -> None:
    scenario = AgentBenchScenario(
        scenario_id="write-case",
        kind="write",
        world_scenario="dash_outage",
        user_message=("Open a support case."),
        permissions=frozenset(
            {
                "enterprise:read",
                "support_case:create",
            }
        ),
        expected_status="approval_required",
        required_tools=frozenset(
            {
                "create_support_case",
            }
        ),
    )

    result = _approval_result()

    metrics = score_trajectory(
        scenario=scenario,
        result=result,
    )

    assert metrics.approval_required_count == 1

    assert metrics.required_tool_call_recall == 1.0

    assert metrics.required_tool_success_recall == 0.0

    assert metrics.required_tool_expected_outcome_recall == 1.0

    assert metrics.tool_error_count == 1

    assert metrics.unexpected_tool_error_count == 0

    assert metrics.trajectory_success


def test_unexpected_approval_does_not_satisfy_expected_outcome_recall() -> None:
    scenario = AgentBenchScenario(
        scenario_id="write-case-without-expected-pause",
        kind="write",
        world_scenario="dash_outage",
        user_message="Open a support case.",
        permissions=frozenset({"support_case:create"}),
        expected_status="completed",
        required_tools=frozenset({"create_support_case"}),
    )

    metrics = score_trajectory(
        scenario=scenario,
        result=_approval_result(),
    )

    assert metrics.required_tool_call_recall == 1.0
    assert metrics.required_tool_success_recall == 0.0
    assert metrics.required_tool_expected_outcome_recall == 0.0
    assert not metrics.trajectory_success
