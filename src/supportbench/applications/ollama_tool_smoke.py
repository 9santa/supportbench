import json
import os

from supportbench.applications.enterprise_simulator import (
    build_enterprise_simulator,
)
from supportbench.llm.ollama_client import (
    OllamaToolCallingClient,
)
from supportbench.llm.ollama_tools import (
    tool_result_to_ollama_message,
)
from supportbench.simulator.postgres.lifecycle import (
    reset_world,
)
from supportbench.simulator.scenarios import (
    build_scenario,
)
from supportbench.tools.models import (
    ToolExecutionContext,
)


WORLD_ID = "ollama-tool-smoke"


def _required_env(name: str) -> str:
    value = os.environ.get(
        name,
        "",
    ).strip()

    if not value:
        raise RuntimeError(f"{name} is not set")

    return value


def main() -> int:
    database_url = _required_env("SUPPORTBENCH_SIMULATOR_DATABASE_URL")

    model_name = os.environ.get(
        "SUPPORTBENCH_AGENT_MODEL",
        "qwen3:4b",
    ).strip()

    ollama_base_url = os.environ.get(
        "SUPPORTBENCH_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip()

    runtime = build_enterprise_simulator(database_url=database_url)

    client = OllamaToolCallingClient(
        model_name=model_name,
        base_url=ollama_base_url,
        temperature=0.0,
        max_output_tokens=512,
        think=False,
    )

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="dash_outage",
                world_id=WORLD_ID,
            ),
        )

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "You are an enterprise support agent. "
                    "Use the available tools when current "
                    "enterprise state is needed. "
                    "Do not invent service state."
                ),
            },
            {
                "role": "user",
                "content": ("What is the current status of webgui-noc-prod?"),
            },
        ]

        turn = client.chat(
            messages=messages,
            tools=runtime.tool_gateway.definitions,
            request_id="smoke-request-001",
            assistant_turn_index=0,
        )

        print("assistant content:")
        print(repr(turn.content))

        print("tool calls:")
        for call in turn.tool_calls:
            print(
                json.dumps(
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    },
                    indent=2,
                )
            )

        if len(turn.tool_calls) != 1:
            raise RuntimeError("expected exactly one tool call")

        call = turn.tool_calls[0]

        result = runtime.tool_gateway.execute(
            call,
            context=ToolExecutionContext(
                world_id=WORLD_ID,
                actor_user_id="alice",
                request_id="smoke-request-001",
                permissions=frozenset({"enterprise:read"}),
            ),
        )

        print("tool result:")
        print(
            json.dumps(
                {
                    "call_id": result.call_id,
                    "tool_name": result.tool_name,
                    "status": result.status,
                    "data": result.data,
                    "error": (
                        {
                            "code": result.error.code,
                            "message": result.error.message,
                        }
                        if result.error
                        else None
                    ),
                },
                indent=2,
                default=str,
            )
        )

        tool_message = tool_result_to_ollama_message(result)

        print("ollama tool message:")
        print(
            json.dumps(
                tool_message,
                indent=2,
            )
        )

        return 0

    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
