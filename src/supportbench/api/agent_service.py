from datetime import UTC, datetime
from uuid import uuid4

from supportbench.agent.orchestrator import (
    AgentOrchestrator,
)
from supportbench.api.runs import (
    InMemoryAgentRunStore,
    StoredAgentRun,
)
from supportbench.api.worlds import WorldService
from supportbench.tools.models import (
    ToolExecutionContext,
)


class AgentRunService:
    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        store: InMemoryAgentRunStore,
        world_service: WorldService,
        system_prompt: str,
        default_permissions: frozenset[str],
        actor_user_id: str = "alice",
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")

        if not actor_user_id.strip():
            raise ValueError("actor_user_id must be non-empty")

        self._orchestrator = orchestrator
        self._store = store
        self._world_service = world_service
        self._system_prompt = system_prompt
        self._default_permissions = default_permissions
        self._actor_user_id = actor_user_id

    def create_run(
        self,
        *,
        world_id: str,
        message: str,
    ) -> StoredAgentRun:
        self._world_service.get(world_id=world_id)

        run_id = str(uuid4())

        request_id = f"agent-run-{run_id}"

        context = ToolExecutionContext(
            world_id=world_id,
            actor_user_id=self._actor_user_id,
            request_id=request_id,
            permissions=self._default_permissions,
        )

        result = self._orchestrator.run(
            messages=(
                {
                    "role": "system",
                    "content": self._system_prompt,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ),
            context=context,
        )

        now = datetime.now(UTC)

        stored = StoredAgentRun(
            run_id=run_id,
            world_id=world_id,
            actor_user_id=(context.actor_user_id),
            request_id=(context.request_id),
            permissions=(context.permissions),
            result=result,
            api_state="ready",
            created_at=now,
            updated_at=now,
        )

        self._store.put(stored)

        return stored

    def get(
        self,
        *,
        run_id: str,
    ) -> StoredAgentRun:
        return self._store.get(run_id)

    def approve(
        self,
        *,
        run_id: str,
    ) -> StoredAgentRun:
        stored = self._store.claim_for_approval(run_id=run_id)

        try:
            pending = stored.result.pending_approval

            if pending is None:
                raise RuntimeError("approval_required run has no pending approval")

            approved_context = ToolExecutionContext(
                world_id=(stored.world_id),
                actor_user_id=(stored.actor_user_id),
                request_id=(stored.request_id),
                permissions=(stored.permissions),
                approved_tool_calls=(frozenset({pending.approval_id})),
            )

            result = self._orchestrator.resume_after_approval(
                previous=stored.result,
                context=(approved_context),
            )

        except Exception:
            self._store.release_approval_claim(run_id=run_id)
            raise

        return self._store.complete_approval(
            run_id=run_id,
            result=result,
        )
