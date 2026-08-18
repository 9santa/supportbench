from datetime import UTC, datetime

import pytest

from supportbench.agent.models import (
    AgentApprovalRequest,
    AgentRunResult,
)
from supportbench.api.runs import StoredAgentRun
from supportbench.tools.models import ToolCall


def _stored_run(
    *,
    result: AgentRunResult,
) -> StoredAgentRun:
    now = datetime.now(UTC)

    return StoredAgentRun(
        run_id="run-001",
        world_id="world-001",
        actor_user_id="alice",
        request_id="request-001",
        permissions=frozenset(),
        result=result,
        api_state="ready",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def completed_stored_run() -> StoredAgentRun:
    return _stored_run(
        result=AgentRunResult(
            status="completed",
            final_answer="The task is complete.",
            steps=(),
            messages=(),
            pending_approval=None,
        )
    )


@pytest.fixture
def approval_stored_run() -> StoredAgentRun:
    call = ToolCall(
        call_id="call-001",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Production Web GUI outage",
        },
    )

    return _stored_run(
        result=AgentRunResult(
            status="approval_required",
            final_answer=None,
            steps=(),
            messages=(),
            pending_approval=AgentApprovalRequest(
                approval_id="internal-approval-id",
                call=call,
                remaining_calls=(),
                step_index=0,
            ),
        )
    )
