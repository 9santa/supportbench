from dataclasses import dataclass
from threading import RLock
from typing import Protocol
from uuid import uuid4

from supportbench.api.models import (
    WorldScenarioName,
)
from supportbench.simulator.postgres.lifecycle import (
    delete_world,
    reset_world,
)
from supportbench.simulator.scenarios import (
    build_scenario,
)


@dataclass(frozen=True, slots=True)
class DemoWorld:
    world_id: str
    scenario: WorldScenarioName


class DemoWorldNotFoundError(LookupError):
    def __init__(self, world_id: str) -> None:
        super().__init__(f"demo world not found: {world_id}")
        self.world_id = world_id


class WorldService(Protocol):
    def create(
        self,
        *,
        scenario: WorldScenarioName,
    ) -> DemoWorld: ...

    def get(
        self,
        *,
        world_id: str,
    ) -> DemoWorld: ...

    def delete(
        self,
        *,
        world_id: str,
    ) -> None: ...

    def close(self) -> None: ...


class PostgresDemoWorldService:
    def __init__(
        self,
        *,
        session_factory,
    ) -> None:
        self._session_factory = session_factory
        self._worlds: dict[str, DemoWorld] = {}
        self._lock = RLock()

    def create(
        self,
        *,
        scenario: WorldScenarioName,
    ) -> DemoWorld:
        world_id = f"demo-{scenario}-{uuid4()}"

        simulator_scenario = build_scenario(
            name=scenario,
            world_id=world_id,
        )

        reset_world(
            session_factory=self._session_factory,
            scenario=simulator_scenario,
        )

        world = DemoWorld(
            world_id=world_id,
            scenario=scenario,
        )

        with self._lock:
            self._worlds[world_id] = world

        return world

    def get(
        self,
        *,
        world_id: str,
    ) -> DemoWorld:
        with self._lock:
            world = self._worlds.get(world_id)

        if world is None:
            raise DemoWorldNotFoundError(world_id)

        return world

    def delete(
        self,
        *,
        world_id: str,
    ) -> None:
        with self._lock:
            world = self._worlds.get(world_id)

        if world_id is None:
            raise DemoWorldNotFoundError(world_id)

        delete_world(
            session_factory=self._session_factory,
            world_id=world_id,
        )

        with self._lock:
            self._worlds.pop(world_id, None)

    def close(self) -> None:
        with self._lock:
            world_ids = tuple(self._worlds)

        for world_id in world_ids:
            try:
                self.delete(world_id=world_id)
            except Exception:
                continue
