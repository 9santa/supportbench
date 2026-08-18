import pytest

from supportbench.api.runs import (
    AgentRunConflictError,
    AgentRunNotFoundError,
    InMemoryAgentRunStore,
)


def test_missing_run_raises_not_found():
    store = InMemoryAgentRunStore()

    with pytest.raises(AgentRunNotFoundError):
        store.get("missing")


def test_completed_run_cannot_be_claimed_for_approval(
    completed_stored_run,
):
    store = InMemoryAgentRunStore()
    store.put(completed_stored_run)

    with pytest.raises(AgentRunConflictError):
        store.claim_for_approval(run_id=(completed_stored_run.run_id))


def test_approval_claim_is_exclusive(
    approval_stored_run,
):
    store = InMemoryAgentRunStore()
    store.put(approval_stored_run)

    claimed = store.claim_for_approval(run_id=(approval_stored_run.run_id))

    assert claimed.api_state == "approving"

    with pytest.raises(AgentRunConflictError):
        store.claim_for_approval(run_id=(approval_stored_run.run_id))


def test_release_restores_approval_run(
    approval_stored_run,
):
    store = InMemoryAgentRunStore()
    store.put(approval_stored_run)

    store.claim_for_approval(run_id=(approval_stored_run.run_id))

    store.release_approval_claim(run_id=(approval_stored_run.run_id))

    restored = store.get(approval_stored_run.run_id)

    assert restored.api_state == "ready"

    assert restored.result.status == "approval_required"
