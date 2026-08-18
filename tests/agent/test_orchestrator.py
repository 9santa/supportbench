from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from supportbench.agent.errors import AgentOrchestrationError, EmptyAssistantTurnError
from supportbench.agent.orchestrator import AgentOrchestrator
from supportbench.llm.models import AssistantModelTurn
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import (
    ToolCall,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
)


class FakeModelClient:
    def __init__(
        self,
        turns: Sequence[AssistantModelTurn],
    ) -> None:
        self.turns = deque(turns)

        self.seen_messages: list[list[Mapping[str, object]]] = []

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[ToolDefinition],
        request_id: str,
        assistant_turn_index: int,
    ) -> AssistantModelTurn:
        self.seen_messages.append(list(messages))

        return self.turns.popleft()

    def tool_result_message(
        self,
        result: ToolResult,
    ) -> Mapping[str, object]:
        return {
            "role": "tool",
            "tool_name": result.tool_name,
            "content": result.status,
        }


class FakeToolGateway:
    def __init__(
        self,
        results: Sequence[ToolResult],
    ) -> None:
        self.results = deque(results)
        self.calls: list[ToolCall] = []
        self.contexts: list[ToolExecutionContext] = []

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return ()

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult:
        self.calls.append(call)
        self.contexts.append(context)

        return self.results.popleft()


def _tool_turn(
    *,
    call_id: str = "tc-001",
    tool_name: str = "get_service_status",
) -> AssistantModelTurn:
    call = ToolCall(
        call_id=call_id,
        name=tool_name,
        arguments={
            "service_id": "webgui-noc-prod",
        },
    )

    return AssistantModelTurn(
        content="",
        tool_calls=(call,),
        history_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": {
                            "service_id": ("webgui-noc-prod"),
                        },
                    }
                }
            ],
        },
    )


def _final_turn(
    content: str,
) -> AssistantModelTurn:
    return AssistantModelTurn(
        content=content,
        tool_calls=(),
        history_message={
            "role": "assistant",
            "content": content,
        },
    )


def _context(
    *,
    world_id: str = "world-001",
    approved_tool_calls: frozenset[str] = frozenset(),
) -> ToolExecutionContext:
    return ToolExecutionContext(
        world_id=world_id,
        actor_user_id="alice",
        request_id="req-001",
        permissions=frozenset({"enterprise:read"}),
        approved_tool_calls=approved_tool_calls,
    )


def _approval_required_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        status="error",
        data=None,
        error=ToolErrorInfo(
            code="approval_required",
            message="This tool call requires approval.",
            details={"approval_id": f"approval:{call.call_id}"},
        ),
    )


def _successful_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        status="success",
        data={"ok": True},
        error=None,
    )


def test_tool_is_followed_by_final_turn() -> None:
    model = FakeModelClient(
        (
            _tool_turn(),
            _final_turn("The service is degraded."),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-001",
                tool_name="get_service_status",
                status="success",
                data={
                    "status": "degraded",
                },
                error=None,
            ),
        )
    )

    orch = AgentOrchestrator(
        model=model,
        gateway=gateway,
        max_steps=4,
    )

    result = orch.run(
        messages=[
            {
                "role": "user",
                "content": "What is the service status?",
            }
        ],
        context=_context(),
    )

    assert result.status == "completed"

    assert result.final_answer == "The service is degraded."

    assert len(result.steps) == 2

    tool_data = result.steps[0].tool_executions[0].result.data
    assert tool_data is not None
    assert tool_data["status"] == "degraded"


def test_model_usage_is_preserved_on_agent_step() -> None:
    turn = replace(
        _final_turn("Done."),
        finish_reason="stop",
        prompt_token_count=120,
        output_token_count=8,
        total_duration_ns=15_000_000,
        load_duration_ns=1_000_000,
        prompt_eval_duration_ns=4_000_000,
        generation_duration_ns=10_000_000,
    )
    result = AgentOrchestrator(
        model=FakeModelClient((turn,)),
        gateway=FakeToolGateway(()),
    ).run(
        messages=[{"role": "user", "content": "Finish."}],
        context=_context(),
    )

    step = result.steps[0]
    assert step.finish_reason == "stop"
    assert step.prompt_token_count == 120
    assert step.output_token_count == 8
    assert step.total_duration_ns == 15_000_000
    assert step.load_duration_ns == 1_000_000
    assert step.prompt_eval_duration_ns == 4_000_000
    assert step.generation_duration_ns == 10_000_000


def test_model_sees_tool_result_on_next_turn() -> None:
    model = FakeModelClient(
        (
            _tool_turn(),
            _final_turn("Done."),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-001",
                tool_name="get_service_status",
                status="success",
                data={
                    "status": "operational",
                },
                error=None,
            ),
        )
    )

    orchestrator = AgentOrchestrator(
        model=model,
        gateway=gateway,
    )

    orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "Status?",
            }
        ],
        context=_context(),
    )

    second_messages = model.seen_messages[1]

    assert len(second_messages) == 3

    assert second_messages[0]["role"] == "user"
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[2]["role"] == "tool"


def test_approval_required_stops_agent() -> None:
    model = FakeModelClient(
        (
            _tool_turn(
                call_id="tc-create-001",
                tool_name="create_support_case",
            ),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-create-001",
                tool_name="create_support_case",
                status="error",
                data=None,
                error=ToolErrorInfo(
                    code="approval_required",
                    message=("This tool call requires approval."),
                    details={"approval_id": "approval:tc-create-001"},
                ),
            ),
        )
    )

    orchestrator = AgentOrchestrator(
        model=model,
        gateway=gateway,
    )

    result = orchestrator.run(
        messages=[
            {
                "role": "user",
                "content": "Open a support case.",
            }
        ],
        context=_context(),
    )

    assert result.status == "approval_required"

    assert result.final_answer is None
    assert result.pending_approval is not None

    assert result.pending_approval.call.call_id == "tc-create-001"
    assert result.pending_approval.approval_id == "approval:tc-create-001"

    assert len(model.seen_messages) == 1


def test_approval_required_without_authoritative_id_is_rejected() -> None:
    tool_turn = _tool_turn(
        call_id="tc-create-001",
        tool_name="create_support_case",
    )
    result_without_id = ToolResult(
        call_id="tc-create-001",
        tool_name="create_support_case",
        status="error",
        data=None,
        error=ToolErrorInfo(
            code="approval_required",
            message="This tool call requires approval.",
        ),
    )
    orchestrator = AgentOrchestrator(
        model=FakeModelClient((tool_turn,)),
        gateway=FakeToolGateway((result_without_id,)),
    )

    with pytest.raises(AgentOrchestrationError, match="has no approval details"):
        orchestrator.run(
            messages=[{"role": "user", "content": "Open a support case."}],
            context=_context(),
        )


def test_normal_tool_error_does_not_stop_agent() -> None:
    model = FakeModelClient(
        (
            _tool_turn(),
            _final_turn("That service does not exist."),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-001",
                tool_name="get_service_status",
                status="error",
                data=None,
                error=ToolErrorInfo(
                    code="service_not_found",
                    message="Service was not found.",
                ),
            ),
        )
    )

    result = AgentOrchestrator(
        model=model,
        gateway=gateway,
    ).run(
        messages=[
            {
                "role": "user",
                "content": "Status?",
            }
        ],
        context=_context(),
    )

    assert result.status == "completed"
    assert len(model.seen_messages) == 2


def test_max_steps_stops_infinite_tool_loop() -> None:
    model = FakeModelClient(
        (
            _tool_turn(call_id="tc-1"),
            _tool_turn(call_id="tc-2"),
            _tool_turn(call_id="tc-3"),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-1",
                tool_name="get_service_status",
                status="success",
                data={"status": "operational"},
                error=None,
            ),
            ToolResult(
                call_id="tc-2",
                tool_name="get_service_status",
                status="success",
                data={"status": "operational"},
                error=None,
            ),
        )
    )

    result = AgentOrchestrator(
        model=model,
        gateway=gateway,
        max_steps=2,
    ).run(
        messages=[
            {
                "role": "user",
                "content": "Keep checking.",
            }
        ],
        context=_context(),
    )

    assert result.status == "max_steps_exceeded"

    assert result.final_answer is None
    assert len(result.steps) == 2
    assert len(gateway.calls) == 2


def test_empty_final_turn_is_rejected() -> None:
    model = FakeModelClient((_final_turn("  "),))

    orch = AgentOrchestrator(
        model=model,
        gateway=FakeToolGateway(()),
    )

    with pytest.raises(EmptyAssistantTurnError):
        orch.run(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            context=_context(),
        )


def test_multiple_tool_calls_execute_sequentially() -> None:
    first = ToolCall(
        call_id="tc-1",
        name="get_service_status",
        arguments={
            "service_id": "webgui-noc-prod",
        },
    )

    second = ToolCall(
        call_id="tc-2",
        name="check_user_entitlement",
        arguments={
            "user_id": "alice",
            "service_id": "webgui-noc-prod",
        },
    )

    model = FakeModelClient(
        (
            AssistantModelTurn(
                content="",
                tool_calls=(
                    first,
                    second,
                ),
                history_message={
                    "role": "assistant",
                    "content": "",
                },
            ),
            _final_turn("Done."),
        )
    )

    gateway = FakeToolGateway(
        (
            ToolResult(
                call_id="tc-1",
                tool_name="get_service_status",
                status="success",
                data={
                    "status": "operational",
                },
                error=None,
            ),
            ToolResult(
                call_id="tc-2",
                tool_name="check_user_entitlement",
                status="success",
                data={
                    "granted": True,
                },
                error=None,
            ),
        )
    )

    result = AgentOrchestrator(
        model=model,
        gateway=gateway,
    ).run(
        messages=[
            {
                "role": "user",
                "content": "Check everything.",
            }
        ],
        context=_context(),
    )

    assert result.status == "completed"

    assert [call.call_id for call in gateway.calls] == [
        "tc-1",
        "tc-2",
    ]


def test_resume_after_approval_executes_call_and_continues() -> None:
    tool_turn = replace(
        _tool_turn(
            call_id="tc-create-001",
            tool_name="create_support_case",
        ),
        prompt_token_count=100,
        output_token_count=20,
        total_duration_ns=10_000_000,
        generation_duration_ns=5_000_000,
    )
    call = tool_turn.tool_calls[0]
    model = FakeModelClient(
        (
            tool_turn,
            _final_turn("The support case was created."),
        )
    )
    gateway = FakeToolGateway(
        (
            _approval_required_result(call),
            _successful_result(call),
        )
    )
    orchestrator = AgentOrchestrator(model=model, gateway=gateway)
    context = _context()

    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Open a support case."}],
        context=context,
    )
    assert paused.pending_approval is not None

    approved_context = replace(
        context,
        approved_tool_calls=frozenset({paused.pending_approval.approval_id}),
    )
    resumed = orchestrator.resume_after_approval(
        previous=paused,
        context=approved_context,
    )

    assert resumed.status == "completed"
    assert resumed.final_answer == "The support case was created."
    assert [execution.result.status for execution in resumed.steps[0].tool_executions] == [
        "error",
        "success",
    ]
    assert resumed.steps[0].tool_executions[0].result.error is not None
    assert resumed.steps[0].tool_executions[0].result.error.code == "approval_required"
    assert resumed.steps[0].prompt_token_count == 100
    assert resumed.steps[0].output_token_count == 20
    assert resumed.steps[0].total_duration_ns == 10_000_000
    assert resumed.steps[0].generation_duration_ns == 5_000_000
    assert [message["role"] for message in model.seen_messages[1]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert gateway.calls == [call, call]
    assert paused.pending_approval.approval_id in gateway.contexts[1].approved_tool_calls


def test_resume_after_approval_requires_granted_approval() -> None:
    tool_turn = _tool_turn(
        call_id="tc-create-001",
        tool_name="create_support_case",
    )
    call = tool_turn.tool_calls[0]
    orchestrator = AgentOrchestrator(
        model=FakeModelClient((tool_turn,)),
        gateway=FakeToolGateway((_approval_required_result(call),)),
    )
    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Open a support case."}],
        context=_context(),
    )

    with pytest.raises(AgentOrchestrationError, match="exactly the pending approval"):
        orchestrator.resume_after_approval(
            previous=paused,
            context=_context(),
        )


def test_resume_after_approval_rejects_additional_approvals() -> None:
    tool_turn = _tool_turn(
        call_id="tc-create-001",
        tool_name="create_support_case",
    )
    call = tool_turn.tool_calls[0]
    gateway = FakeToolGateway((_approval_required_result(call),))
    orchestrator = AgentOrchestrator(
        model=FakeModelClient((tool_turn,)),
        gateway=gateway,
    )
    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Open a support case."}],
        context=_context(),
    )
    assert paused.pending_approval is not None

    with pytest.raises(AgentOrchestrationError, match="exactly the pending approval"):
        orchestrator.resume_after_approval(
            previous=paused,
            context=_context(
                approved_tool_calls=frozenset(
                    {paused.pending_approval.approval_id, "approval:unrelated"}
                )
            ),
        )

    assert gateway.calls == [call]


def test_resume_after_approval_rejects_changed_execution_context() -> None:
    tool_turn = _tool_turn(
        call_id="tc-create-001",
        tool_name="create_support_case",
    )
    call = tool_turn.tool_calls[0]
    orchestrator = AgentOrchestrator(
        model=FakeModelClient((tool_turn,)),
        gateway=FakeToolGateway(
            (
                _approval_required_result(call),
                _approval_required_result(call),
            )
        ),
    )
    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Open a support case."}],
        context=_context(),
    )
    assert paused.pending_approval is not None

    changed_context = _context(
        world_id="different-world",
        approved_tool_calls=frozenset({paused.pending_approval.approval_id}),
    )

    with pytest.raises(AgentOrchestrationError, match="still requires approval"):
        orchestrator.resume_after_approval(
            previous=paused,
            context=changed_context,
        )


def test_resume_finishes_remaining_calls_from_paused_turn() -> None:
    first = ToolCall(
        call_id="tc-read-001",
        name="get_service_status",
        arguments={"service_id": "webgui-noc-prod"},
    )
    pending_call = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={"summary": "Service degraded"},
    )
    remaining = ToolCall(
        call_id="tc-read-002",
        name="check_user_entitlement",
        arguments={"user_id": "alice", "service_id": "webgui-noc-prod"},
    )
    tool_turn = AssistantModelTurn(
        content="",
        tool_calls=(first, pending_call, remaining),
        history_message={"role": "assistant", "content": ""},
    )
    model = FakeModelClient((tool_turn, _final_turn("Done.")))
    gateway = FakeToolGateway(
        (
            _successful_result(first),
            _approval_required_result(pending_call),
            _successful_result(pending_call),
            _successful_result(remaining),
        )
    )
    orchestrator = AgentOrchestrator(model=model, gateway=gateway)
    context = _context()

    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Check and create a case."}],
        context=context,
    )
    assert paused.pending_approval is not None
    assert paused.pending_approval.remaining_calls == (remaining,)

    resumed = orchestrator.resume_after_approval(
        previous=paused,
        context=replace(
            context,
            approved_tool_calls=frozenset({paused.pending_approval.approval_id}),
        ),
    )

    assert resumed.status == "completed"
    assert [execution.call for execution in resumed.steps[0].tool_executions] == [
        first,
        pending_call,
        pending_call,
        remaining,
    ]
    assert [execution.result.status for execution in resumed.steps[0].tool_executions] == [
        "success",
        "error",
        "success",
        "success",
    ]
    assert [message["role"] for message in model.seen_messages[1]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
    ]


def test_sequential_approvals_preserve_every_gateway_attempt() -> None:
    first = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={"summary": "First case"},
    )
    second = ToolCall(
        call_id="tc-create-002",
        name="create_support_case",
        arguments={"summary": "Second case"},
    )
    tool_turn = AssistantModelTurn(
        content="",
        tool_calls=(first, second),
        history_message={"role": "assistant", "content": ""},
    )
    model = FakeModelClient((tool_turn, _final_turn("Done.")))
    gateway = FakeToolGateway(
        (
            _approval_required_result(first),
            _successful_result(first),
            _approval_required_result(second),
            _successful_result(second),
        )
    )
    orchestrator = AgentOrchestrator(model=model, gateway=gateway)
    context = _context()

    first_pause = orchestrator.run(
        messages=[{"role": "user", "content": "Create both cases."}],
        context=context,
    )
    assert first_pause.pending_approval is not None

    second_pause = orchestrator.resume_after_approval(
        previous=first_pause,
        context=replace(
            context,
            approved_tool_calls=frozenset({first_pause.pending_approval.approval_id}),
        ),
    )
    assert second_pause.status == "approval_required"
    assert second_pause.pending_approval is not None
    assert second_pause.pending_approval.call == second
    assert len(model.seen_messages) == 1

    completed = orchestrator.resume_after_approval(
        previous=second_pause,
        context=replace(
            context,
            approved_tool_calls=frozenset({second_pause.pending_approval.approval_id}),
        ),
    )

    assert completed.status == "completed"
    assert [execution.call for execution in completed.steps[0].tool_executions] == [
        first,
        first,
        second,
        second,
    ]
    assert [execution.result.status for execution in completed.steps[0].tool_executions] == [
        "error",
        "success",
        "error",
        "success",
    ]
    assert len(model.seen_messages) == 2


def test_resume_preserves_original_step_budget() -> None:
    tool_turn = _tool_turn(
        call_id="tc-create-001",
        tool_name="create_support_case",
    )
    call = tool_turn.tool_calls[0]
    model = FakeModelClient((tool_turn,))
    gateway = FakeToolGateway(
        (
            _approval_required_result(call),
            _successful_result(call),
        )
    )
    orchestrator = AgentOrchestrator(
        model=model,
        gateway=gateway,
        max_steps=1,
    )
    context = _context()
    paused = orchestrator.run(
        messages=[{"role": "user", "content": "Open a support case."}],
        context=context,
    )
    assert paused.pending_approval is not None

    resumed = orchestrator.resume_after_approval(
        previous=paused,
        context=replace(
            context,
            approved_tool_calls=frozenset({paused.pending_approval.approval_id}),
        ),
    )

    assert resumed.status == "max_steps_exceeded"
    assert len(resumed.steps) == 1
    assert not model.turns
