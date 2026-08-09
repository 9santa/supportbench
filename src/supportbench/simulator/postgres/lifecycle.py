from sqlalchemy import delete

from supportbench.simulator.postgres.schema import simulator_worlds
from supportbench.simulator.postgres.seed import _seed_scenario
from supportbench.simulator.postgres.session import SessionFactory
from supportbench.simulator.scenarios import ScenarioDefinition


def delete_world(
    *,
    session_factory: SessionFactory,
    world_id: str,
) -> bool:
    normalized_world_id = world_id.strip()

    if not normalized_world_id:
        raise ValueError("world_id must be non-empty")

    with session_factory() as session:
        with session.begin():
            deleted_world_id = session.execute(
                delete(simulator_worlds)
                .where(simulator_worlds.c.world_id == normalized_world_id)
                .returning(simulator_worlds.c.world_id)
            ).scalar_one_or_none()

    return deleted_world_id is not None


def reset_world(
    *,
    session_factory: SessionFactory,
    scenario: ScenarioDefinition,
) -> None:
    with session_factory() as session:
        with session.begin():
            session.execute(
                delete(simulator_worlds).where(
                    simulator_worlds.c.world_id == scenario.world.world_id
                )
            )

            _seed_scenario(
                session=session,
                scenario=scenario,
            )
