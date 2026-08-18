from typing import cast

from fastapi.testclient import (
    TestClient,
)

from supportbench.api.agent_service import AgentRunService
from supportbench.api.app import (
    create_app,
)
from supportbench.api.models import (
    WorldScenarioName,
)
from supportbench.api.runtime import (
    ApiRuntime,
)
from supportbench.api.worlds import (
    DemoWorld,
    DemoWorldNotFoundError,
)


class FakeWorldService:
    def __init__(self) -> None:
        self.worlds: dict[
            str,
            DemoWorld,
        ] = {}

        self.closed = False

    def create(
        self,
        *,
        scenario: WorldScenarioName,
    ) -> DemoWorld:
        world = DemoWorld(
            world_id="demo-world-001",
            scenario=scenario,
        )

        self.worlds[world.world_id] = world

        return world

    def get(
        self,
        *,
        world_id: str,
    ) -> DemoWorld:
        try:
            return self.worlds[world_id]
        except KeyError as exc:
            raise DemoWorldNotFoundError(world_id) from exc

    def delete(
        self,
        *,
        world_id: str,
    ) -> None:
        if world_id not in self.worlds:
            raise DemoWorldNotFoundError(world_id)

        del self.worlds[world_id]

    def close(self) -> None:
        self.closed = True


def test_health_and_world_lifecycle() -> None:
    worlds = FakeWorldService()

    app = create_app(
        runtime_factory=lambda: ApiRuntime(
            world_service=worlds,
            agent_run_service=cast(AgentRunService, object()),
            close_callbacks=(worlds.close,),
        )
    )

    with TestClient(app) as client:
        health = client.get("/health")

        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
        }

        created = client.post(
            "/worlds",
            json={
                "scenario": ("old_dash_version"),
            },
        )

        assert created.status_code == 201

        assert created.json() == {
            "world_id": ("demo-world-001"),
            "scenario": ("old_dash_version"),
        }

        deleted = client.delete("/worlds/demo-world-001")

        assert deleted.status_code == 200

        assert deleted.json() == {
            "world_id": ("demo-world-001"),
            "deleted": True,
        }

    assert worlds.closed
