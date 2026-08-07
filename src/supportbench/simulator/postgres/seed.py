from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from supportbench.simulator.postgres.schema import (
    products,
    service_instances,
    simulator_worlds,
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
