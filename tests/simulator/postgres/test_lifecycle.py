import os
from uuid import uuid4

import pytest
from sqlalchemy import select

from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.simulator.postgres.lifecycle import delete_world, reset_world
from supportbench.simulator.postgres.schema import audit_events, support_cases
from supportbench.simulator.postgres.seed import seed_scenario
from supportbench.simulator.postgres.session import build_engine, build_session_factory
from supportbench.simulator.postgres.unit_of_work import PostgresUnitOfWork
from supportbench.simulator.scenarios import build_scenario
from supportbench.simulator.service import EnterpriseService


def _database_url() -> str:
    value = os.environ.get("SUPPORTBENCH_SIMULATOR_DATABASE_URL", "").strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def test_reset_world_replaces_all_world_state() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        enterprise.create_support_case(
            CreateSupportCaseCommand(
                world_id=world_id,
                idempotency_key="old-request",
                actor_user_id="alice",
                user_id="alice",
                service_id="webgui-noc-prod",
                summary="Old case",
                description="Must disappear after reset.",
                severity="high",
            )
        )

        reset_world(
            session_factory=session_factory,
            scenario=build_scenario(
                name="access_denied",
                world_id=world_id,
            ),
        )

        entitlement = enterprise.check_user_entitlement(
            world_id=world_id,
            user_id="alice",
            service_id="webgui-noc-prod",
        )

        assert entitlement.granted is False

        with session_factory() as session:
            old_cases = session.execute(
                select(support_cases).where(support_cases.c.world_id == world_id)
            ).all()

            old_audit = session.execute(
                select(audit_events).where(audit_events.c.world_id == world_id)
            ).all()

        assert old_cases == []
        assert old_audit == []

    finally:
        delete_world(
            session_factory=session_factory,
            world_id=world_id,
        )

        engine.dispose()
