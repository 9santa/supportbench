from types import TracebackType

import pytest

from supportbench.simulator.errors import ServiceNotFoundError
from supportbench.simulator.models import ServiceInstance
from supportbench.simulator.service import EnterpriseService


class FakeServiceRepository:
    def __init__(
        self,
        services: tuple[ServiceInstance, ...],
    ) -> None:
        self._services = {(service.world_id, service.service_id): service for service in services}

    def get(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance | None:
        return self._services.get((world_id, service_id))


class FakeUnitOfWork:
    def __init__(
        self,
        services: tuple[ServiceInstance, ...],
    ) -> None:
        self.services = FakeServiceRepository(services)

    def __enter__(self) -> "FakeUnitOfWork":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


def test_get_service_status_returns_service() -> None:
    expected = ServiceInstance(
        world_id="world-a",
        service_id="webgui-noc-prod",
        display_name="NOC Web GUI",
        product_key="netcool_webgui",
        version="8.1 FP7",
        environment="production",
        status="operational",
        owner_team="noc-platform",
    )

    service = EnterpriseService(uow_factory=lambda: FakeUnitOfWork((expected,)))

    actual = service.get_service_status(
        world_id="world-a",
        service_id="webgui-noc-prod",
    )

    assert actual == expected


def test_get_service_status_is_world_isolated() -> None:
    world_a = ServiceInstance(
        world_id="world-a",
        service_id="dash-noc-prod",
        display_name="NOC DASH",
        product_key="dash",
        version="3.1.2.1",
        environment="production",
        status="operational",
        owner_team="noc-platform",
    )

    world_b = ServiceInstance(
        world_id="world-b",
        service_id="dash-noc-prod",
        display_name="NOC DASH",
        product_key="dash",
        version="3.1.0.3",
        environment="production",
        status="operational",
        owner_team="noc-platform",
    )

    service = EnterpriseService(uow_factory=lambda: FakeUnitOfWork((world_a, world_b)))

    result_a = service.get_service_status(
        world_id="world-a",
        service_id="dash-noc-prod",
    )

    result_b = service.get_service_status(
        world_id="world-b",
        service_id="dash-noc-prod",
    )

    assert result_a.version == "3.1.2.1"
    assert result_b.version == "3.1.0.3"


def test_get_service_status_raises_when_missing() -> None:
    service = EnterpriseService(uow_factory=lambda: FakeUnitOfWork(()))

    with pytest.raises(ServiceNotFoundError):
        service.get_service_status(
            world_id="world-a",
            service_id="missing-service",
        )
