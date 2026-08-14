import pytest

from collections import deque
from collections.abc import Mapping, Sequence

from supportbench.llm.models import AssistantModelTurn
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import (
    ToolCall,
    ToolResult,
    ToolExecutionContext,
    ToolErrorInfo,
)
from supportbench.agent.orchestrator import AgentOrchestrator
from supportbench.agent.errors import EmptyAssistantTurnError


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
        messages,
        tools,
        request_id,
        assistant_turn_index,
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
        self.calls = []

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return ()

    def execute(
        self,
        call,
        *,
        context,
    ) -> ToolResult:
        self.calls.append(call)

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


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        world_id="world-001",
        actor_user_id="alice",
        request_id="req-001",
        permissions=frozenset({"enterprise:read"}),
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

    assert result.steps[0].tool_executions[0].result.data["status"] == "degraded"


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

    assert len(model.seen_messages) == 1


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
