from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock
from typing import Literal

from supportbench.agent.models import AgentRunResult


AgentApiRunState = Literal[
    "ready",
    "approving",
]


@dataclass(frozen=True, slots=True)
class StoredAgentRun:
    run_id: str

    world_id: str
    actor_user_id: str
    request_id: str

    permissions: frozenset[str]

    result: AgentRunResult

    api_state: AgentApiRunState

    created_at: datetime
    updated_at: datetime


class AgentRunNotFoundError(LookupError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"agent run not found: {run_id}")
        self.run_id = run_id


class AgentRunConflictError(RuntimeError):
    pass


class InMemoryAgentRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, StoredAgentRun] = {}

        self._lock = RLock()

    def put(
        self,
        run: StoredAgentRun,
    ) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def get(
        self,
        run_id: str,
    ) -> StoredAgentRun:
        with self._lock:
            run = self._runs.get(run_id)

        if run is None:
            raise AgentRunNotFoundError(run_id)

        return run

    def claim_for_approval(
        self,
        *,
        run_id: str,
    ) -> StoredAgentRun:
        with self._lock:
            current = self._runs.get(run_id)

            if current is None:
                raise AgentRunNotFoundError(run_id)

            if current.api_state != "ready" or current.result.status != "approval_required":
                raise AgentRunConflictError("agent run cannot be approved")

            claimed = replace(
                current,
                api_state="approving",
                updated_at=datetime.now(UTC),
            )

            self._runs[run_id] = claimed

            return claimed

    def complete_approval(
        self,
        *,
        run_id: str,
        result: AgentRunResult,
    ) -> StoredAgentRun:
        with self._lock:
            current = self._runs.get(run_id)

            if current is None:
                raise AgentRunNotFoundError(run_id)

            if current.api_state != "approving":
                raise AgentRunConflictError("agent run has not active approval claim")

            updated = replace(
                current,
                result=result,
                api_state="ready",
                updated_at=datetime.now(UTC),
            )

            self._runs[run_id] = updated

            return updated

    def release_approval_claim(
        self,
        *,
        run_id: str,
    ) -> None:
        """
        This is in case `resume_after_approval()` fails.
        Otherwise run will forever be left in "approving".
        """
        with self._lock:
            current = self._runs.get(run_id)

            if current is None:
                raise AgentRunNotFoundError(run_id)

            if current.api_state != "approving":
                return

            self._runs[run_id] = replace(
                current,
                api_state="ready",
                updated_at=datetime.now(UTC),
            )
