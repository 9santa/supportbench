from typing import cast
from uuid import uuid4

from sqlalchemy.orm import Session, sessionmaker

from supportbench.agent.orchestrator import (
    AgentOrchestrator,
)
from supportbench.agentbench.models import (
    AgentBenchCaseResult,
    AgentBenchScenario,
)
from supportbench.agentbench.postgres import (
    PostgresAgentBenchSnapshotter,
)
from supportbench.agentbench.scoring import (
    score_state,
    score_trajectory,
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


class AgentBenchRunner:
    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        session_factory: sessionmaker[Session],
        snapshotter: PostgresAgentBenchSnapshotter,
        system_prompt: str,
    ) -> None:
        self._orchestrator = orchestrator
        self._session_factory = session_factory
        self._snapshotter = snapshotter
        self._system_prompt = system_prompt

    def run_case(
        self,
        scenario: AgentBenchScenario,
    ) -> AgentBenchCaseResult:
        world_id = f"agentbench-{scenario.scenario_id}-{uuid4()}"

        request_id = f"agentbench-{uuid4()}"

        try:
            reset_world(
                session_factory=self._session_factory,
                scenario=build_scenario(
                    name=cast(ScenarioName, scenario.world_scenario),
                    world_id=world_id,
                ),
            )

            before = self._snapshotter.snapshot(world_id=world_id)

            context = ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id=request_id,
                permissions=scenario.permissions,
            )

            run = self._orchestrator.run(
                messages=(
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    },
                    {
                        "role": "user",
                        "content": scenario.user_message,
                    },
                ),
                context=context,
            )

            after = self._snapshotter.snapshot(world_id=world_id)

            trajectory = score_trajectory(
                scenario=scenario,
                result=run,
            )

            state = score_state(
                scenario=scenario,
                before=before,
                after=after,
            )

            return AgentBenchCaseResult(
                scenario_id=(scenario.scenario_id),
                run=run,
                before=before,
                after=after,
                trajectory=trajectory,
                state=state,
                success=(trajectory.trajectory_success and state.state_expectation_correct),
            )

        finally:
            delete_world(
                session_factory=self._session_factory,
                world_id=world_id,
            )
