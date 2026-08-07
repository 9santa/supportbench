import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import delete

from supportbench.simulator.postgres.schema import simulator_worlds
from supportbench.simulator.postgres.seed import seed_scenario
from supportbench.simulator.postgres.session import (
    build_engine,
    build_session_factory,
)
from supportbench.simulator.postgres.unit_of_work import (
    PostgresUnitOfWork,
)
from supportbench.simulator.scenarios import (
    ScenarioDefinition,
    build_healthy_scenario,
)
from supportbench.simulator.service import EnterpriseService

# All tests in this file will automatically get the 'postgres' mark
pytestmark = pytest.mark.postgres


def _database_url() -> str:
    value = os.environ.get("SUPPORTBENCH_SIMULATOR_DATABASE_URL", "").strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def test_get_service_status_from_postgres() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory, scenario=build_healthy_scenario(world_id=world_id)
        )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        result = enterprise.get_service_status(
            world_id=world_id,
            service_id="dash-noc-prod",
        )

        assert result.world_id == world_id
        assert result.service_id == "dash-noc-prod"
        assert result.version == "3.1.2.1"
        assert result.status == "operational"

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    # ON DELETE CASCADE should delete all attached service_instances to this world
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()


def test_same_service_can_have_different_state_per_world() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    healthy_world_id = f"test-{uuid4()}"
    old_world_id = f"test-{uuid4()}"

    healthy = build_healthy_scenario(world_id=healthy_world_id)

    old_dash = build_healthy_scenario(world_id=old_world_id)

    old_dash = ScenarioDefinition(
        world=replace(
            old_dash.world,
            scenario_name="old_dash_version",
        ),
        products=old_dash.products,
        services=tuple(
            replace(
                service,
                version="3.1.0.3",
            )
            if service.service_id == "dash-noc-prod"
            else service
            for service in old_dash.services
        ),
    )

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=healthy,
        )

        seed_scenario(
            session_factory=session_factory,
            scenario=old_dash,
        )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        healthy_dash = enterprise.get_service_status(
            world_id=healthy_world_id,
            service_id="dash-noc-prod",
        )

        old_dash_result = enterprise.get_service_status(
            world_id=old_world_id,
            service_id="dash-noc-prod",
        )

        assert healthy_dash.version == "3.1.2.1"
        assert old_dash_result.version == "3.1.0.3"

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_(
                            [
                                healthy_world_id,
                                old_world_id,
                            ]
                        )
                    )
                )
