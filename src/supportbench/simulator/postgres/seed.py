from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from supportbench.simulator.postgres.schema import (
    products,
    service_instances,
    simulator_worlds,
    installed_products,
    user_entitlements,
    users,
    assets,
)
from supportbench.simulator.postgres.session import SessionFactory
from supportbench.simulator.scenarios import ScenarioDefinition


def _seed_scenario(
    *,
    session: Session,
    scenario: ScenarioDefinition,
) -> None:
    """Works within the passed session, doesn't start a new one."""
    for product in scenario.products:
        session.execute(
            pg_insert(products)
            .values(
                product_key=product.product_key,
                display_name=product.display_name,
            )
            .on_conflict_do_nothing(index_elements=[products.c.product_key])
        )

    session.execute(
        insert(simulator_worlds).values(
            world_id=scenario.world.world_id,
            scenario_name=scenario.world.scenario_name,
        )
    )

    if scenario.services:
        session.execute(
            insert(service_instances),
            [
                {
                    "world_id": item.world_id,
                    "service_id": item.service_id,
                    "display_name": item.display_name,
                    "product_key": item.product_key,
                    "version": item.version,
                    "environment": item.environment,
                    "status": item.status,
                    "owner_team": item.owner_team,
                }
                for item in scenario.services
            ],
        )

    if scenario.assets:
        session.execute(
            insert(assets),
            [
                {
                    "world_id": item.world_id,
                    "asset_id": item.asset_id,
                    "hostname": item.hostname,
                    "operating_system": item.operating_system,
                    "environment": item.environment,
                }
                for item in scenario.assets
            ],
        )

    if scenario.users:
        session.execute(
            insert(users),
            [
                {
                    "world_id": item.world_id,
                    "user_id": item.user_id,
                    "display_name": item.display_name,
                    "department": item.department,
                }
                for item in scenario.users
            ],
        )

    if scenario.installed_products:
        session.execute(
            insert(installed_products),
            [
                {
                    "world_id": item.world_id,
                    "asset_id": item.asset_id,
                    "product_key": item.product_key,
                    "version": item.version,
                    "patch_level": item.patch_level,
                }
                for item in scenario.installed_products
            ],
        )

    if scenario.entitlements:
        session.execute(
            insert(user_entitlements),
            [
                {
                    "world_id": item.world_id,
                    "user_id": item.user_id,
                    "service_id": item.service_id,
                    "granted": item.granted,
                    "role": item.role,
                }
                for item in scenario.entitlements
            ],
        )


def seed_scenario(
    *,
    session_factory: SessionFactory,
    scenario: ScenarioDefinition,
) -> None:
    with session_factory() as session:
        with session.begin():
            _seed_scenario(
                session=session,
                scenario=scenario,
            )
