import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.exc import IntegrityError

from supportbench.simulator.postgres.schema import (
    assets,
    installed_products,
    service_instances,
    simulator_worlds,
    user_entitlements,
    users,
)
from supportbench.simulator.postgres.seed import seed_scenario
from supportbench.simulator.postgres.session import (
    build_engine,
    build_session_factory,
)
from supportbench.simulator.scenarios import (
    build_healthy_scenario,
    build_scenario,
)
from supportbench.simulator.service import EnterpriseService
from supportbench.simulator.postgres.unit_of_work import PostgresUnitOfWork


pytestmark = pytest.mark.postgres


def _database_url() -> str:
    value = os.environ.get(
        "SUPPORTBENCH_SIMULATOR_DATABASE_URL",
        "",
    ).strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def test_installed_product_cannot_reference_asset_from_other_world() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_a = f"test-{uuid4()}"
    world_b = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_a,
            ),
        )
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_b,
            ),
        )

        with session_factory() as session:
            with session.begin():
                session.execute(
                    insert(assets).values(
                        world_id=world_a,
                        asset_id="world-a-only-asset",
                        hostname="dash-host-01.example.test",
                        operating_system="RHEL 9",
                        environment="production",
                    )
                )

        with pytest.raises(IntegrityError):
            with session_factory() as session:
                with session.begin():
                    session.execute(
                        insert(installed_products).values(
                            world_id=world_b,
                            asset_id="world-a-only-asset",
                            product_key="dash",
                            version="3.1.2.1",
                            patch_level="FP1",
                        )
                    )

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_([world_a, world_b])
                    )
                )

        engine.dispose()


def test_entitlement_cannot_reference_user_from_other_world() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_a = f"test-{uuid4()}"
    world_b = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_a,
            ),
        )
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_b,
            ),
        )

        with session_factory() as session:
            with session.begin():
                session.execute(
                    insert(users).values(
                        world_id=world_a,
                        user_id="world-a-only-user",
                        display_name="World A User",
                        department="Operations",
                    )
                )

        with pytest.raises(IntegrityError):
            with session_factory() as session:
                with session.begin():
                    session.execute(
                        insert(user_entitlements).values(
                            world_id=world_b,
                            user_id="world-a-only-user",
                            service_id="webgui-noc-prod",
                            granted=True,
                            role="viewer",
                        )
                    )

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_([world_a, world_b])
                    )
                )

        engine.dispose()


def test_entitlement_cannot_reference_service_from_other_world() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_a = f"test-{uuid4()}"
    world_b = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_a,
            ),
        )
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_b,
            ),
        )

        with session_factory() as session:
            with session.begin():
                session.execute(
                    insert(service_instances).values(
                        world_id=world_a,
                        service_id="world-a-only-service",
                        display_name="World A Only Service",
                        product_key="dash",
                        version="3.1.2.1",
                        environment="production",
                        status="operational",
                        owner_team="noc-platform",
                    )
                )

        with pytest.raises(IntegrityError):
            with session_factory() as session:
                with session.begin():
                    session.execute(
                        insert(user_entitlements).values(
                            world_id=world_b,
                            user_id="alice",
                            service_id="world-a-only-service",
                            granted=True,
                            role="viewer",
                        )
                    )

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_([world_a, world_b])
                    )
                )

        engine.dispose()


def test_deleting_world_cascades_world_state() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_healthy_scenario(
                world_id=world_id,
            ),
        )

        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        with session_factory() as session:
            assert (
                session.execute(assets.select().where(assets.c.world_id == world_id)).first()
                is None
            )

            assert (
                session.execute(
                    installed_products.select().where(installed_products.c.world_id == world_id)
                ).first()
                is None
            )

            assert (
                session.execute(users.select().where(users.c.world_id == world_id)).first() is None
            )

            assert (
                session.execute(
                    user_entitlements.select().where(user_entitlements.c.world_id == world_id)
                ).first()
                is None
            )

    finally:
        # harmless if world was already deleted
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()


def test_entitlement_differs_between_worlds() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    healthy_world = f"test-{uuid4()}"
    denied_world = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=healthy_world,
            ),
        )

        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="access_denied",
                world_id=denied_world,
            ),
        )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        healthy = enterprise.check_user_entitlement(
            world_id=healthy_world,
            user_id="alice",
            service_id="webgui-noc-prod",
        )

        denied = enterprise.check_user_entitlement(
            world_id=denied_world,
            user_id="alice",
            service_id="webgui-noc-prod",
        )

        assert healthy.granted is True
        assert denied.granted is False

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_([healthy_world, denied_world])
                    )
                )

        engine.dispose()
