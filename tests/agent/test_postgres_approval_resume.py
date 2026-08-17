import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from supportbench.agent.orchestrator import AgentOrchestrator
from supportbench.applications.enterprise_simulator import (
    EnterpriseSimulatorRuntime,
    build_enterprise_simulator,
)
from supportbench.llm.models import AssistantModelTurn
from supportbench.simulator.postgres.lifecycle import delete_world, reset_world
from supportbench.simulator.postgres.schema import audit_events, support_cases
from supportbench.simulator.scenarios import build_scenario
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import ToolCall, ToolExecutionContext, ToolResult
from supportbench.tools.policies import CREATE_SUPPORT_CASE_PERMISSION

pytestmark = pytest.mark.postgres


@dataclass(frozen=True, slots=True)
class ModelChatCall:
    messages: tuple[Mapping[str, object], ...]
    request_id: str
    assistant_turn_index: int
    mutation_counts: tuple[int, int]


class FakeModelClient:
    def __init__(
        self,
        *,
        turns: Sequence[AssistantModelTurn],
        mutation_counts: Callable[[], tuple[int, int]],
    ) -> None:
        self._turns = deque(turns)
        self._mutation_counts = mutation_counts
        self.chat_calls: list[ModelChatCall] = []

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[ToolDefinition],
        request_id: str,
        assistant_turn_index: int,
    ) -> AssistantModelTurn:
        assert any(tool.name == "create_support_case" for tool in tools)
        self.chat_calls.append(
            ModelChatCall(
                messages=tuple(dict(message) for message in messages),
                request_id=request_id,
                assistant_turn_index=assistant_turn_index,
                mutation_counts=self._mutation_counts(),
            )
        )
        return self._turns.popleft()

    def tool_result_message(
        self,
        result: ToolResult,
    ) -> Mapping[str, object]:
        return {
            "role": "tool",
            "tool_name": result.tool_name,
            "content": result.status,
        }


def _database_url() -> str:
    value = os.environ.get("SUPPORTBENCH_SIMULATOR_DATABASE_URL", "").strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def _mutation_counts(
    runtime: EnterpriseSimulatorRuntime,
    *,
    world_id: str,
) -> tuple[int, int]:
    with runtime.session_factory() as session:
        case_count = session.scalar(
            select(func.count())
            .select_from(support_cases)
            .where(support_cases.c.world_id == world_id)
        )
        audit_count = session.scalar(
            select(func.count())
            .select_from(audit_events)
            .where(audit_events.c.world_id == world_id)
        )

    return int(case_count or 0), int(audit_count or 0)


def _tool_turn(call: ToolCall) -> AssistantModelTurn:
    return AssistantModelTurn(
        content="",
        tool_calls=(call,),
        history_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.call_id,
                    "function": {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    },
                }
            ],
        },
    )


def _final_turn() -> AssistantModelTurn:
    content = "The support case was created."
    return AssistantModelTurn(
        content=content,
        tool_calls=(),
        history_message={"role": "assistant", "content": content},
    )


def test_approved_agent_resume_executes_pending_mutation_once_in_postgres() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())
    world_id = f"agent-approval-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(name="healthy", world_id=world_id),
        )
        call = ToolCall(
            call_id="tc-create-001",
            name="create_support_case",
            arguments={
                "user_id": "alice",
                "service_id": "webgui-noc-prod",
                "summary": "Cannot access Web GUI",
                "description": "Alice cannot access the production Web GUI.",
                "severity": "high",
            },
        )
        context = ToolExecutionContext(
            world_id=world_id,
            actor_user_id="alice",
            request_id="req-agent-approval-001",
            permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
        )

        def counts() -> tuple[int, int]:
            return _mutation_counts(runtime, world_id=world_id)

        model = FakeModelClient(
            turns=(_tool_turn(call), _final_turn(), _final_turn()),
            mutation_counts=counts,
        )
        orchestrator = AgentOrchestrator(model=model, gateway=runtime.tool_gateway)

        paused = orchestrator.run(
            messages=({"role": "user", "content": "Open a support case."},),
            context=context,
        )

        assert paused.status == "approval_required"
        assert paused.pending_approval is not None
        assert paused.pending_approval.call == call
        assert counts() == (0, 0)
        assert len(model.chat_calls) == 1
        assert model.chat_calls[0].assistant_turn_index == 0
        assert model.chat_calls[0].mutation_counts == (0, 0)
        assert [message["role"] for message in model.chat_calls[0].messages] == ["user"]

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
        assert [
            execution.result.status for execution in resumed.steps[0].tool_executions
        ] == ["error", "success"]
        assert resumed.steps[0].tool_executions[0].result.error is not None
        assert (
            resumed.steps[0].tool_executions[0].result.error.code == "approval_required"
        )
        assert resumed.steps[0].tool_executions[1].call is paused.pending_approval.call
        assert counts() == (1, 1)
        assert len(model.chat_calls) == 2
        assert model.chat_calls[1].assistant_turn_index == 1
        assert model.chat_calls[1].mutation_counts == (1, 1)
        assert [message["role"] for message in model.chat_calls[1].messages] == [
            "user",
            "assistant",
            "tool",
        ]

        retried = orchestrator.resume_after_approval(
            previous=paused,
            context=approved_context,
        )

        assert retried.status == "completed"
        assert counts() == (1, 1)
        assert len(model.chat_calls) == 3
        assert model.chat_calls[2].mutation_counts == (1, 1)
    finally:
        delete_world(session_factory=runtime.session_factory, world_id=world_id)
        runtime.close()
