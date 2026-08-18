from collections.abc import Callable
from dataclasses import dataclass

from supportbench.api.worlds import WorldService
from supportbench.api.agent_service import AgentRunService


@dataclass(slots=True)
class ApiRuntime:
    world_service: WorldService
    agent_run_service: AgentRunService

    close_callbacks: tuple[Callable[[], None], ...] = ()

    def close(self) -> None:
        for callback in self.close_callbacks:
            callback()


RuntimeFactory = Callable[
    [],
    ApiRuntime,
]
