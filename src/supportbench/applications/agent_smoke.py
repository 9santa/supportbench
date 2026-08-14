import json
import os
from dataclasses import dataclass
from uuid import uuid4

from supportbench.agent.orchestrator import (
    AgentOrchestrator,
)
from supportbench.applications.enterprise_simulator import (
    build_enterprise_simulator,
)
from supportbench.llm.ollama_client import (
    OllamaToolCallingClient,
)
from supportbench.simulator.postgres.lifecycle import (
    delete_world,
    reset_world,
)
from supportbench.simulator.scenarios import (
    ScenarioName,
    build_scenario,
)
from supportbench.tools.models import (
    ToolExecutionContext,
)
from supportbench.tools.policies import (
    ENTERPRISE_READ_PERMISSION,
)


def _get_env(name: str) -> str:
    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


@dataclass(frozen=True, slots=True)
class SmokeScenario:
    scenario_name: ScenarioName
    user_message: str
    expected_tool_name: str


SMOKE_SCENARIOS = (
    SmokeScenario(
        scenario_name="dash_outage",
        user_message=("What is the current operational status of service webgui-noc-prod?"),
        expected_tool_name="get_service_status",
    ),
    SmokeScenario(
        scenario_name="old_dash_version",
        user_message=("What version of DASH is currently installed on asset dash-host-01?"),
        expected_tool_name="search_products",
    ),
    SmokeScenario(
        scenario_name="access_denied",
        user_message=("Does user alice currently have access to service webgui-noc-prod?"),
        expected_tool_name="check_user_entitlement",
    ),
)


SYSTEM_PROMPT = """
You are an enterprise support agent.

Use the available tools whenever the user asks about current
enterprise state, including service status, installed product
versions, or user access.

Do not invent current enterprise state.

After receiving tool results, answer the user's question
concisely using the observed tool data.
"""


def _run_scenario(
    *,
    runtime,
    orchestrator: AgentOrchestrator,
    scenario: SmokeScenario,
) -> None:
    world_id = f"agent-smoke-{scenario.scenario_name}-{uuid4()}"

    request_id = f"req-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name=scenario.scenario_name,
                world_id=world_id,
            ),
        )

        result = orchestrator.run(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": scenario.user_message,
                },
            ],
            context=ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id=request_id,
                permissions=frozenset(
                    {
                        ENTERPRISE_READ_PERMISSION,
                    }
                ),
            ),
        )

        _print_result(
            scenario=scenario,
            result=result,
        )

        _validate_result(
            scenario=scenario,
            result=result,
        )

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )


def _print_result(
    *,
    scenario: SmokeScenario,
    result,
) -> None:
    print()
    print("=" * 72)
    print(f"scenario: {scenario.scenario_name}")
    print(f"status:   {result.status}")

    for step in result.steps:
        print()
        print(f"step {step.step_index}")

        if step.assistant_content.strip():
            print(
                "assistant:",
                step.assistant_content,
            )

        for execution in step.tool_executions:
            print("tool call:")
            print(
                json.dumps(
                    {
                        "call_id": (execution.call.call_id),
                        "name": (execution.call.name),
                        "arguments": dict(execution.call.arguments),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print("tool result:")
            print(
                json.dumps(
                    {
                        "status": (execution.result.status),
                        "data": (execution.result.data),
                        "error": (
                            {
                                "code": (execution.result.error.code),
                                "message": (execution.result.error.message),
                            }
                            if (execution.result.error is not None)
                            else None
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

    print()
    print("final answer:")
    print(result.final_answer)


def _validate_result(
    *,
    scenario: SmokeScenario,
    result,
) -> None:
    if result.status != "completed":
        raise RuntimeError(f"agent smoke did not complete: {result.status}")

    if result.final_answer is None or not result.final_answer.strip():
        raise RuntimeError("agent smoke returned no final answer")

    executions = [execution for step in result.steps for execution in step.tool_executions]

    if not executions:
        raise RuntimeError("agent completed without using a tool")

    tool_names = [execution.call.name for execution in executions]

    if scenario.expected_tool_name not in tool_names:
        raise RuntimeError(
            f"expected tool was not used: {scenario.expected_tool_name!r}; observed={tool_names!r}"
        )

    failed = [execution for execution in executions if execution.result.status != "success"]

    if failed:
        raise RuntimeError("one or more smoke tool calls failed")


def main() -> int:
    database_url = _get_env("SUPPORTBENCH_SIMULATOR_DATABASE_URL")

    model_name = os.environ.get(
        "SUPPORTBENCH_AGENT_MODEL",
        "qwen3:4b",
    ).strip()

    # model_name = "qwen3-4b-8096"

    ollama_base_url = os.environ.get(
        "SUPPORTBENCH_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip()

    runtime = build_enterprise_simulator(database_url=database_url)

    model = OllamaToolCallingClient(
        model_name=model_name,
        base_url=ollama_base_url,
        temperature=0.0,
        max_output_tokens=512,
        think=False,
    )

    orchestrator = AgentOrchestrator(
        model=model,
        gateway=runtime.tool_gateway,
        max_steps=6,
    )

    try:
        for scenario in SMOKE_SCENARIOS:
            _run_scenario(
                runtime=runtime,
                orchestrator=orchestrator,
                scenario=scenario,
            )

        print()
        print("All agent smoke scenarios passed.")

        return 0

    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
