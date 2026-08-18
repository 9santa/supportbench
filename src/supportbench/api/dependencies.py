from fastapi import Request

from supportbench.api.runtime import ApiRuntime
from supportbench.api.worlds import WorldService
from supportbench.api.agent_service import AgentRunService


def get_api_runtime(request: Request) -> ApiRuntime:
    runtime = getattr(
        request.app.state,
        "runtime",
        None,
    )

    if runtime is None:
        raise RuntimeError("API runtime is not initialized")

    return runtime


def get_world_service(request: Request) -> WorldService:
    return get_api_runtime(request).world_service


def get_agent_run_service(request: Request) -> AgentRunService:
    return get_api_runtime(request).agent_run_service
