from collections.abc import Mapping
from dataclasses import replace

from supportbench.agent.models import (
    AgentApprovalRequest,
    AgentRunResult,
    AgentStep,
    AgentToolExecution,
)
from supportbench.agentbench.models import (
    AgentBenchScenario,
    ExpectedAnswerFact,
)
from supportbench.agentbench.scenarios import (
    AGENTBENCH_V1,
    AGENTBENCH_V2,
    AGENTBENCH_V3,
    MIXED_DASH_WEBGUI,
)
from supportbench.agentbench.scoring import score_answer, score_trajectory
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
    assert all(not scenario.expected_answer_facts for scenario in AGENTBENCH_V2)


def test_v3_adds_answer_evaluation_without_changing_scenario_ids() -> None:
    assert [scenario.scenario_id for scenario in AGENTBENCH_V3] == [
        scenario.scenario_id for scenario in AGENTBENCH_V2
    ]
    v3 = {scenario.scenario_id: scenario for scenario in AGENTBENCH_V3}

    assert v3["enterprise-installed-dash-version"].expected_answer_facts
    assert v3["knowledge-ssl-mutual-auth"].expected_evidence_doc_ids == frozenset(
        {"swg21179559"}
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


def test_trajectory_aggregates_available_model_usage() -> None:
    scenario = AgentBenchScenario(
        scenario_id="usage",
        kind="enterprise",
        world_scenario="healthy",
        user_message="Check status.",
        permissions=frozenset({"enterprise:read"}),
        expected_status="completed",
    )
    base = _success_result()
    result = replace(
        base,
        steps=(
            replace(
                base.steps[0],
                prompt_token_count=100,
                output_token_count=20,
                total_duration_ns=10_000_000,
                generation_duration_ns=4_000_000,
            ),
            replace(
                base.steps[1],
                prompt_token_count=150,
                output_token_count=30,
                total_duration_ns=20_000_000,
                generation_duration_ns=6_000_000,
            ),
        ),
    )

    metrics = score_trajectory(scenario=scenario, result=result)

    assert metrics.prompt_token_count == 250
    assert metrics.output_token_count == 50
    assert metrics.model_total_duration_ms == 30.0
    assert metrics.model_generation_duration_ms == 10.0


def test_answer_scoring_checks_facts_evidence_and_forbidden_claims() -> None:
    scenario = AgentBenchScenario(
        scenario_id="grounded-answer",
        kind="mixed",
        world_scenario="old_dash_version",
        user_message="Check DASH and the documentation.",
        permissions=frozenset({"enterprise:read", "support_docs:read"}),
        expected_status="completed",
        expected_answer_facts=(
            ExpectedAnswerFact(
                fact_id="installed_version",
                accepted_phrases=("3.1.0.3",),
            ),
        ),
        expected_evidence_doc_ids=frozenset({"swg21681385"}),
        forbidden_answer_claims=("documentation proves compatibility",),
    )
    result = replace(
        _success_result(
            _success_execution(
                "search_support_docs",
                data={"matches": [{"document_id": "swg21681385"}]},
            ),
        ),
        final_answer="DASH 3.1.0.3 is installed; the evidence states prerequisites.",
    )

    metrics = score_answer(scenario=scenario, result=result)

    assert metrics.expected_fact_recall == 1.0
    assert metrics.expected_evidence_recall == 1.0
    assert metrics.forbidden_claim_count == 0
    assert metrics.answer_success


def test_answer_scoring_resolves_dynamic_fact_from_tool_result() -> None:
    scenario = AgentBenchScenario(
        scenario_id="created-case",
        kind="write",
        world_scenario="dash_outage",
        user_message="Create a case.",
        permissions=frozenset({"support_case:create"}),
        expected_status="completed",
        expected_answer_facts=(
            ExpectedAnswerFact(
                fact_id="created_case_id",
                source_tool="create_support_case",
                source_result_field="case_id",
            ),
        ),
    )
    result = replace(
        _success_result(
            _success_execution(
                "create_support_case",
                data={"case_id": "CASE-123"},
            ),
        ),
        final_answer="Support case CASE-123 was created.",
    )

    metrics = score_answer(scenario=scenario, result=result)

    assert metrics.missing_expected_facts == ()
    assert metrics.answer_success


def test_answer_scoring_fails_missing_fact_and_forbidden_claim() -> None:
    scenario = AgentBenchScenario(
        scenario_id="unsupported-answer",
        kind="mixed",
        world_scenario="old_dash_version",
        user_message="Check compatibility.",
        permissions=frozenset({"enterprise:read"}),
        expected_status="completed",
        expected_answer_facts=(
            ExpectedAnswerFact(
                fact_id="installed_version",
                accepted_phrases=("3.1.0.3",),
            ),
        ),
        forbidden_answer_claims=("documentation proves compatibility",),
    )
    result = replace(
        _success_result(),
        final_answer="Documentation proves compatibility.",
    )

    metrics = score_answer(scenario=scenario, result=result)

    assert metrics.missing_expected_facts == ("installed_version",)
    assert metrics.forbidden_claims_found == (
        "documentation proves compatibility",
    )
    assert not metrics.answer_success


def test_answer_scoring_is_not_applicable_to_expected_approval_pause() -> None:
    scenario = AgentBenchScenario(
        scenario_id="approval-pause",
        kind="write",
        world_scenario="dash_outage",
        user_message="Create a case.",
        permissions=frozenset({"support_case:create"}),
        expected_status="approval_required",
    )

    metrics = score_answer(scenario=scenario, result=_approval_result())

    assert not metrics.applicable
    assert not metrics.final_answer_present
    assert metrics.answer_success
