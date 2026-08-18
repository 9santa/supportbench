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
    score_answer,
    score_approval,
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

            initial_run = self._orchestrator.run(
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

            pre_approval = None
            final_run = initial_run

            if initial_run.status == "approval_required":
                pre_approval = self._snapshotter.snapshot(world_id=world_id)

                if initial_run.pending_approval is None:
                    raise RuntimeError("approval_required run has no pending approval")

                if scenario.approval_mode == "approve":
                    pending = initial_run.pending_approval

                    approved_context = ToolExecutionContext(
                        world_id=context.world_id,
                        actor_user_id=context.actor_user_id,
                        request_id=context.request_id,
                        permissions=context.permissions,
                        approved_tool_calls=frozenset({pending.approval_id}),
                    )

                    final_run = self._orchestrator.resume_after_approval(
                        previous=initial_run,
                        context=approved_context,
                    )

            after = self._snapshotter.snapshot(world_id=world_id)

            trajectory = score_trajectory(
                scenario=scenario,
                result=final_run,
            )

            state = score_state(
                scenario=scenario,
                before=before,
                after=after,
            )

            approval = score_approval(
                scenario=scenario,
                initial_run=initial_run,
                before=before,
                pre_approval=pre_approval,
                final_run=final_run,
            )

            answer = score_answer(
                scenario=scenario,
                result=final_run,
            )

            trajectory_state_success = all(
                (
                    trajectory.trajectory_success,
                    state.state_expectation_correct,
                    approval.approval_flow_correct,
                )
            )

            return AgentBenchCaseResult(
                scenario_id=(scenario.scenario_id),
                run=final_run,
                before=before,
                pre_approval=pre_approval,
                after=after,
                trajectory=trajectory,
                state=state,
                approval=approval,
                answer=answer,
                trajectory_state_success=trajectory_state_success,
                success=trajectory_state_success and answer.answer_success,
            )

        finally:
            delete_world(
                session_factory=self._session_factory,
                world_id=world_id,
            )
