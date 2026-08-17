import os
from collections import deque
from collections.abc import Mapping, Sequence

import pytest

from supportbench.agent.orchestrator import AgentOrchestrator
from supportbench.agentbench.models import AgentBenchScenario
from supportbench.agentbench.postgres import PostgresAgentBenchSnapshotter
from supportbench.agentbench.runner import AgentBenchRunner
from supportbench.applications.enterprise_simulator import build_enterprise_simulator
from supportbench.llm.models import AssistantModelTurn
from supportbench.llm.ollama_tools import tool_result_to_ollama_message
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import ToolCall, ToolResult

pytestmark = pytest.mark.postgres


class FakeModelClient:
    def __init__(self, turns: Sequence[AssistantModelTurn]) -> None:
        self._turns = deque(turns)
        self.chat_calls: list[tuple[Mapping[str, object], ...]] = []

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[ToolDefinition],
        request_id: str,
        assistant_turn_index: int,
    ) -> AssistantModelTurn:
        assert any(tool.name == "get_installed_product" for tool in tools)
        self.chat_calls.append(tuple(dict(message) for message in messages))
        return self._turns.popleft()

    def tool_result_message(self, result: ToolResult) -> Mapping[str, object]:
        return tool_result_to_ollama_message(result)


def _database_url() -> str:
    value = os.environ.get("SUPPORTBENCH_SIMULATOR_DATABASE_URL", "").strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def _installed_product_turn() -> AssistantModelTurn:
    call = ToolCall(
        call_id="tc-installed-dash-001",
        name="get_installed_product",
        arguments={
            "asset_id": "dash-host-01",
            "product_key": "dash",
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
    content = "DASH 3.1.0.3 is installed on dash-host-01."
    return AssistantModelTurn(
        content=content,
        tool_calls=(),
        history_message={"role": "assistant", "content": content},
    )


def test_enterprise_read_case_preserves_frozen_postgres_world() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    try:
        model = FakeModelClient((_installed_product_turn(), _final_turn()))
        runner = AgentBenchRunner(
            orchestrator=AgentOrchestrator(
                model=model,
                gateway=runtime.tool_gateway,
            ),
            session_factory=runtime.session_factory,
            snapshotter=PostgresAgentBenchSnapshotter(runtime.session_factory),
            system_prompt="Use tools for current enterprise state.",
        )
        scenario = AgentBenchScenario(
            scenario_id="read-installed-dash",
            kind="enterprise",
            world_scenario="old_dash_version",
            user_message="Which DASH version is installed on dash-host-01?",
            permissions=frozenset({"enterprise:read"}),
            expected_status="completed",
            required_tools=frozenset({"get_installed_product"}),
            forbidden_tools=frozenset({"create_support_case"}),
            state_expectation="unchanged",
            expected_support_case_delta=0,
            expected_audit_event_delta=0,
        )

        result = runner.run_case(scenario)

        assert result.success
        assert result.trajectory.required_tool_recall == 1.0
        assert result.trajectory.forbidden_tool_call_count == 0
        assert not result.state.state_changed
        assert result.state.support_case_delta == 0
        assert result.state.audit_event_delta == 0
        assert result.before == result.after

        execution = result.run.steps[0].tool_executions[0]
        assert execution.call.name == "get_installed_product"
        assert execution.result.status == "success"
        assert execution.result.data is not None
        assert execution.result.data["version"] == "3.1.0.3"

        assert len(model.chat_calls) == 2
        assert [message["role"] for message in model.chat_calls[0]] == [
            "system",
            "user",
        ]
        assert [message["role"] for message in model.chat_calls[1]] == [
            "system",
            "user",
            "assistant",
            "tool",
        ]
    finally:
        runtime.close()
