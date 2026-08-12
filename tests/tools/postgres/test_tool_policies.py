import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from supportbench.applications.enterprise_simulator import (
    build_enterprise_simulator,
)
from supportbench.simulator.postgres.lifecycle import (
    delete_world,
    reset_world,
)
from supportbench.simulator.postgres.schema import (
    audit_events,
    support_cases,
)
from supportbench.simulator.scenarios import (
    build_scenario,
)
from supportbench.tools.models import (
    ToolCall,
    ToolExecutionContext,
)
from supportbench.tools.policies import (
    CREATE_SUPPORT_CASE_PERMISSION,
    tool_approval_id,
)


pytestmark = pytest.mark.postgres


def _database_url() -> str:
    value = os.environ.get(
        "SUPPORTBENCH_SIMULATOR_DATABASE_URL",
        "",
    ).strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def _create_case_call(
    *,
    call_id: str = "tc-create-001",
    severity: str = "high",
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="create_support_case",
        arguments={
            "user_id": "alice",
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": ("Alice cannot access the production Web GUI."),
            "severity": severity,
        },
    )


def _create_context(
    *,
    world_id: str,
    request_id: str = "req-001",
) -> ToolExecutionContext:
    return ToolExecutionContext(
        world_id=world_id,
        actor_user_id="alice",
        request_id=request_id,
        permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
    )


def _db_mutation_counts(
    runtime,
    *,
    world_id: str,
) -> tuple[int, int]:
    with runtime.session_factory() as session:
        case_count = session.scalar(
            select(func.count())
            .select_from(support_cases)
            .where(support_cases.c.world_id == world_id)
        )

        audit_count = session.scalar(
            select(func.count())
            .select_from(audit_events)
            .where(audit_events.c.world_id == world_id)
        )

    return (
        int(case_count or 0),
        int(audit_count or 0),
    )


def test_forbidden_mutation_does_not_touch_database() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = _create_case_call()

        context = ToolExecutionContext(
            world_id=world_id,
            actor_user_id="alice",
            request_id="req-001",
            # deliberately no permission
            permissions=frozenset(),
        )

        result = runtime.tool_gateway.execute(
            call,
            context=context,
        )

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "forbidden"

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (0, 0)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_unapproved_mutation_does_not_touch_database() -> None:
    """
    authorization = yes
    approval      = no

    → mutation = no
    """
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = _create_case_call()
        context = _create_context(
            world_id=world_id,
        )

        result = runtime.tool_gateway.execute(
            call,
            context=context,
        )

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "approval_required"

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (0, 0)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def _approved_context(
    call: ToolCall,
    *,
    world_id: str = "scenario-0042",
    actor_user_id: str = "alice",
    request_id: str = "req-900",
) -> ToolExecutionContext:
    context = ToolExecutionContext(
        world_id=world_id,
        actor_user_id=actor_user_id,
        request_id=request_id,
        permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
    )
    return replace(
        context,
        approved_tool_calls=frozenset({tool_approval_id(call=call, context=context)}),
    )


def test_approved_mutation_creates_case_and_audit() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = _create_case_call()

        approved_context = _approved_context(call=call, world_id=world_id)

        result = runtime.tool_gateway.execute(
            call,
            context=approved_context,
        )

        assert result.status == "success"
        assert result.error is None
        assert result.data is not None

        assert result.data["world_id"] == world_id
        assert result.data["actor_user_id"] == "alice"
        assert result.data["status"] == "open"
        assert result.data["assigned_team"] == "noc-platform"

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (1, 1)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_approved_tool_call_retry_is_idempotent() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = _create_case_call(
            call_id="tc-create-017",
        )

        context = _approved_context(call=call, world_id=world_id)

        first = runtime.tool_gateway.execute(
            call,
            context=context,
        )

        second = runtime.tool_gateway.execute(
            call,
            context=context,
        )

        assert first.status == "success"
        assert second.status == "success"

        assert first.data is not None
        assert second.data is not None

        assert first.data["case_id"] == second.data["case_id"]

        assert first.data["idempotency_key"] == second.data["idempotency_key"]

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (1, 1)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_changed_arguments_cannot_reuse_approval() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        original_call = _create_case_call(
            call_id="tc-create-001",
            severity="high",
        )

        approved_context = _approved_context(call=original_call, world_id=world_id)

        changed_call = _create_case_call(
            call_id="tc-create-001",
            severity="critical",
        )

        result = runtime.tool_gateway.execute(
            changed_call,
            context=approved_context,
        )

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "approval_required"

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (0, 0)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_approval_cannot_be_replayed_in_another_world() -> None:
    """approval(world-a) != approval(world-b)"""

    runtime = build_enterprise_simulator(database_url=_database_url())

    world_a = f"policy-test-{uuid4()}"
    world_b = f"policy-test-{uuid4()}"

    try:
        for world_id in (world_a, world_b):
            reset_world(
                session_factory=runtime.session_factory,
                scenario=build_scenario(
                    name="healthy",
                    world_id=world_id,
                ),
            )

        call = _create_case_call()

        context_a = _create_context(
            world_id=world_a,
        )

        approval_id = tool_approval_id(
            call=call,
            context=context_a,
        )

        context_b = replace(
            _create_context(
                world_id=world_b,
            ),
            approved_tool_calls=frozenset({approval_id}),
        )

        result = runtime.tool_gateway.execute(
            call,
            context=context_b,
        )

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "approval_required"

        assert _db_mutation_counts(
            runtime,
            world_id=world_a,
        ) == (0, 0)

        assert _db_mutation_counts(
            runtime,
            world_id=world_b,
        ) == (0, 0)

    finally:
        for world_id in (world_a, world_b):
            delete_world(
                session_factory=runtime.session_factory,
                world_id=world_id,
            )

        runtime.close()


def test_approval_does_not_replace_permission() -> None:
    """approval = yes && permission = no -> forbidden"""

    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = _create_case_call()

        authorized_context = _create_context(
            world_id=world_id,
        )

        approval_id = tool_approval_id(
            call=call,
            context=authorized_context,
        )

        no_permission_context = ToolExecutionContext(
            world_id=world_id,
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset(),
            approved_tool_calls=frozenset({approval_id}),
        )

        result = runtime.tool_gateway.execute(
            call,
            context=no_permission_context,
        )

        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "forbidden"

        assert _db_mutation_counts(
            runtime,
            world_id=world_id,
        ) == (0, 0)

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_authorized_read_reaches_postgres() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"policy-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="dash_outage",
                world_id=world_id,
            ),
        )

        result = runtime.tool_gateway.execute(
            ToolCall(
                call_id="tc-read-001",
                name="get_service_status",
                arguments={
                    "service_id": "webgui-noc-prod",
                },
            ),
            context=ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id="req-read",
                permissions=frozenset({"enterprise:read"}),
            ),
        )

        assert result.status == "success"
        assert result.data is not None
        assert result.data["status"] == "degraded"

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()
