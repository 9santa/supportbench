import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import select

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
    ENTERPRISE_READ_PERMISSION,
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


def test_get_service_status_through_gateway() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"tool-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        result = runtime.tool_gateway.execute(
            ToolCall(
                call_id="tc-001",
                name="get_service_status",
                arguments={
                    "service_id": "webgui-noc-prod",
                },
            ),
            context=ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id="req-001",
                permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
            ),
        )

        assert result.status == "success"
        assert result.error is None
        assert result.data is not None

        assert result.data["service_id"] == "webgui-noc-prod"
        assert result.data["status"] == "operational"
        assert result.data["world_id"] == world_id

        assert result.call_id == "tc-001"
        assert result.tool_name == "get_service_status"

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_entitlement_result_depends_on_trusted_world() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    healthy_world = f"tool-test-{uuid4()}"
    denied_world = f"tool-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=healthy_world,
            ),
        )

        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="access_denied",
                world_id=denied_world,
            ),
        )

        call = ToolCall(
            call_id="tc-001",
            name="check_user_entitlement",
            arguments={
                "user_id": "alice",
                "service_id": "webgui-noc-prod",
            },
        )

        healthy = runtime.tool_gateway.execute(
            call,
            context=ToolExecutionContext(
                world_id=healthy_world,
                actor_user_id="alice",
                request_id="req-healthy",
                permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
            ),
        )

        denied = runtime.tool_gateway.execute(
            call,
            context=ToolExecutionContext(
                world_id=denied_world,
                actor_user_id="alice",
                request_id="req-denied",
                permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
            ),
        )

        assert healthy.status == "success"
        assert denied.status == "success"

        assert healthy.data is not None
        assert denied.data is not None

        assert healthy.data["granted"] is True
        assert denied.data["granted"] is False

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=healthy_world,
        )
        delete_world(
            session_factory=runtime.session_factory,
            world_id=denied_world,
        )

        runtime.close()


def test_installed_product_exposes_scenario_version() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"tool-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="old_dash_version",
                world_id=world_id,
            ),
        )

        result = runtime.tool_gateway.execute(
            ToolCall(
                call_id="tc-002",
                name="get_installed_product",
                arguments={
                    "asset_id": "dash-host-01",
                    "product_key": "dash",
                },
            ),
            context=ToolExecutionContext(
                world_id=world_id,
                actor_user_id="alice",
                request_id="req-001",
                permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
            ),
        )

        assert result.status == "success"
        assert result.data is not None

        assert result.data["version"] == "3.1.0.3"

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()


def test_create_support_case_through_gateway() -> None:
    runtime = build_enterprise_simulator(database_url=_database_url())

    world_id = f"tool-test-{uuid4()}"

    try:
        reset_world(
            session_factory=runtime.session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        call = ToolCall(
            call_id="tc-create-001",
            name="create_support_case",
            arguments={
                "user_id": "alice",
                "service_id": "webgui-noc-prod",
                "summary": "Cannot access Web GUI",
                "description": ("Alice cannot access production Web GUI."),
                "severity": "high",
            },
        )
        context = ToolExecutionContext(
            world_id=world_id,
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
        )
        context = replace(
            context,
            approved_tool_calls=frozenset({tool_approval_id(call=call, context=context)}),
        )

        result = runtime.tool_gateway.execute(call, context=context)

        assert result.status == "success"
        assert result.error is None
        assert result.data is not None

        case_id = result.data["case_id"]

        assert result.data["world_id"] == world_id
        assert result.data["actor_user_id"] == "alice"
        assert result.data["status"] == "open"
        assert result.data["assigned_team"] == "noc-platform"

        with runtime.session_factory() as session:
            case_rows = (
                session.execute(select(support_cases).where(support_cases.c.world_id == world_id))
                .mappings()
                .all()
            )

            audit_rows = (
                session.execute(select(audit_events).where(audit_events.c.world_id == world_id))
                .mappings()
                .all()
            )

        assert len(case_rows) == 1
        assert len(audit_rows) == 1

        assert case_rows[0]["case_id"] == case_id

        assert audit_rows[0]["event_type"] == "support_case.created"
        assert audit_rows[0]["entity_id"] == case_id

    finally:
        delete_world(
            session_factory=runtime.session_factory,
            world_id=world_id,
        )

        runtime.close()
