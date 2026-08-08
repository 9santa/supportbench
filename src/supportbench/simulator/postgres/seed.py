from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

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


def seed_scenario(
    *,
    session_factory: SessionFactory,
    scenario: ScenarioDefinition,
) -> None:
    with session_factory() as session:
        with session.begin():
            for product in scenario.products:
                statement = (
                    pg_insert(products)
                    .values(
                        product_key=product.product_key,
                        display_name=product.display_name,
                    )
                    .on_conflict_do_nothing(index_elements=[products.c.product_key])
                )

                session.execute(statement)

            session.execute(
                insert(simulator_worlds).values(
                    world_id=scenario.world.world_id,
                    scenario_name=scenario.world.scenario_name,
                )
            )

            session.execute(
                insert(service_instances),
                [
                    {
                        "world_id": service.world_id,
                        "service_id": service.service_id,
                        "display_name": service.display_name,
                        "product_key": service.product_key,
                        "version": service.version,
                        "environment": service.environment,
                        "status": service.status,
                        "owner_team": service.owner_team,
                    }
                    for service in scenario.services
                ],
            )

            if scenario.assets:
                session.execute(
                    insert(assets),
                    [
                        {
                            "world_id": asset.world_id,
                            "asset_id": asset.asset_id,
                            "hostname": asset.hostname,
                            "operating_system": asset.operating_system,
                            "environment": asset.environment,
                        }
                        for asset in scenario.assets
                    ],
                )

            if scenario.installed_products:
                session.execute(
                    insert(installed_products),
                    [
                        {
                            "world_id": installed.world_id,
                            "asset_id": installed.asset_id,
                            "product_key": installed.product_key,
                            "version": installed.version,
                            "patch_level": installed.patch_level,
                        }
                        for installed in scenario.installed_products
                    ],
                )

            if scenario.users:
                session.execute(
                    insert(users),
                    [
                        {
                            "world_id": user.world_id,
                            "user_id": user.user_id,
                            "display_name": user.display_name,
                            "department": user.department,
                        }
                        for user in scenario.users
                    ],
                )

            if scenario.entitlements:
                session.execute(
                    insert(user_entitlements),
                    [
                        {
                            "world_id": entitlement.world_id,
                            "user_id": entitlement.user_id,
                            "service_id": entitlement.service_id,
                            "granted": entitlement.granted,
                            "role": entitlement.role,
                        }
                        for entitlement in scenario.entitlements
                    ],
                )
