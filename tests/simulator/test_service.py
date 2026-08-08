from types import TracebackType

import pytest

from supportbench.simulator.errors import ServiceNotFoundError
from supportbench.simulator.models import (
    ServiceInstance,
    InstalledProduct,
    UserEntitlement,
)
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


class FakeInstalledProductRepository:
    def __init__(
        self,
        products: tuple[InstalledProduct, ...],
    ) -> None:
        self._products = {
            (
                product.world_id,
                product.asset_id,
                product.product_key,
            ): product
            for product in products
        }

    def get(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct | None:
        return self._products.get((world_id, asset_id, product_key))


class FakeUserEntitlementRepository:
    def __init__(
        self,
        entitlements: tuple[UserEntitlement, ...],
    ) -> None:
        self._entitlements = {
            (
                entitlement.world_id,
                entitlement.user_id,
                entitlement.service_id,
            ): entitlement
            for entitlement in entitlements
        }

    def get(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement | None:
        return self._entitlements.get((world_id, user_id, service_id))


class FakeUnitOfWork:
    def __init__(
        self,
        services: tuple[ServiceInstance, ...],
        installed_products: tuple[InstalledProduct, ...],
        entitlements: tuple[UserEntitlement, ...],
    ) -> None:
        self.services = FakeServiceRepository(services)
        self.installed_products = FakeInstalledProductRepository(installed_products)
        self.user_entitlements = FakeUserEntitlementRepository(entitlements)

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

    service = EnterpriseService(uow_factory=lambda: FakeUnitOfWork((expected,), (), ()))

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

    service = EnterpriseService(
        uow_factory=lambda: FakeUnitOfWork(
            services=(world_a, world_b),
            installed_products=(),
            entitlements=(),
        )
    )

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
    service = EnterpriseService(uow_factory=lambda: FakeUnitOfWork((), (), ()))

    with pytest.raises(ServiceNotFoundError):
        service.get_service_status(
            world_id="world-a",
            service_id="missing-service",
        )


def test_get_installed_product() -> None:
    expected = InstalledProduct(
        world_id="world-a",
        asset_id="dash-host-01",
        product_key="dash",
        version="3.1.2.1",
        patch_level="FP1",
    )

    service = EnterpriseService(
        uow_factory=lambda: FakeUnitOfWork(
            services=(),
            installed_products=(expected,),
            entitlements=(),
        )
    )

    actual = service.get_installed_product(
        world_id="world-a",
        asset_id="dash-host-01",
        product_key="dash",
    )

    assert actual == expected


def test_check_user_entitlement_preserves_explicit_denial() -> None:
    expected = UserEntitlement(
        world_id="world-a",
        user_id="alice",
        service_id="webgui-noc-prod",
        granted=False,
        role="viewer",
    )

    service = EnterpriseService(
        uow_factory=lambda: FakeUnitOfWork(
            services=(),
            installed_products=(),
            entitlements=(expected,),
        )
    )

    actual = service.check_user_entitlement(
        world_id="world-a",
        user_id="alice",
        service_id="webgui-noc-prod",
    )

    assert actual.granted is False
    assert actual.role == "viewer"
