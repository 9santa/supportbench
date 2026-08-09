from types import TracebackType
from datetime import datetime, timezone

import pytest

from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.simulator.errors import ServiceNotFoundError
from supportbench.simulator.models import (
    ServiceInstance,
    InstalledProduct,
    UserEntitlement,
    SupportCase,
    AuditEvent,
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


class FakeSupportCaseRepository:
    def __init__(self) -> None:
        self.items: list[SupportCase] = []

    def get_by_idempotency_key(
        self,
        *,
        world_id: str,
        idempotency_key: str,
    ) -> SupportCase | None:
        for item in self.items:
            if item.world_id == world_id and item.idempotency_key == idempotency_key:
                return item

        return None

    def add(
        self,
        support_case: SupportCase,
    ) -> None:
        self.items.append(support_case)


class FakeAuditEventRepository:
    def __init__(self) -> None:
        self.items: list[AuditEvent] = []

    def add(
        self,
        event: AuditEvent,
    ) -> None:
        self.items.append(event)


class FakeUnitOfWork:
    def __init__(
        self,
        services: tuple[ServiceInstance, ...] = (),
        installed_products: tuple[InstalledProduct, ...] = (),
        entitlements: tuple[UserEntitlement, ...] = (),
    ) -> None:
        self.services = FakeServiceRepository(services)
        self.installed_products = FakeInstalledProductRepository(installed_products)
        self.user_entitlements = FakeUserEntitlementRepository(entitlements)
        self.support_cases = FakeSupportCaseRepository()
        self.audit_events = FakeAuditEventRepository()

        self.commit_count = 0
        self.rollback_count = 0

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
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


class FixedClock:
    def __init__(
        self,
        value: datetime,
    ) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


class SequenceIdGenerator:
    def __init__(
        self,
        *values: str,
    ) -> None:
        self._values = iter(values)

    def new_id(self) -> str:
        return next(self._values)


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


def test_create_support_case() -> None:
    service_instance = ServiceInstance(
        world_id="world-a",
        service_id="webgui-noc-prod",
        display_name="NOC Web GUI",
        product_key="netcool_webgui",
        version="8.1 FP7",
        environment="production",
        status="operational",
        owner_team="noc-platform",
    )

    uow = FakeUnitOfWork(
        services=(service_instance,),
    )

    now = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)

    enterprise = EnterpriseService(
        uow_factory=lambda: uow,
        clock=FixedClock(now),
        id_generator=SequenceIdGenerator(
            "CASE-001",
            "EVENT-001",
        ),
    )

    result = enterprise.create_support_case(
        CreateSupportCaseCommand(
            world_id="world-a",
            idempotency_key="request-001",
            actor_user_id="alice",
            user_id="alice",
            service_id="webgui-noc-prod",
            summary="Cannot access Web GUI",
            description=("Alice cannot access the production Web GUI."),
            severity="high",
        )
    )

    assert result.case_id == "CASE-001"
    assert result.status == "open"

    assert result.assigned_team == "noc-platform"

    assert result.created_at == now
    assert result.updated_at == now

    assert len(uow.support_cases.items) == 1
    assert len(uow.audit_events.items) == 1

    audit = uow.audit_events.items[0]

    assert audit.event_id == "EVENT-001"
    assert audit.event_type == "support_case.created"
    assert audit.entity_id == "CASE-001"
    assert audit.occurred_at == now

    assert uow.commit_count == 1


def test_create_support_case_is_idempotent() -> None:
    service_instance = ServiceInstance(
        world_id="world-a",
        service_id="webgui-noc-prod",
        display_name="NOC Web GUI",
        product_key="netcool_webgui",
        version="8.1 FP7",
        environment="production",
        status="operational",
        owner_team="noc-platform",
    )

    uow = FakeUnitOfWork(
        services=(service_instance,),
    )

    now = datetime(
        2026,
        8,
        8,
        12,
        0,
        tzinfo=timezone.utc,
    )

    enterprise = EnterpriseService(
        uow_factory=lambda: uow,
        clock=FixedClock(now),
        id_generator=SequenceIdGenerator(
            "CASE-001",
            "EVENT-001",
        ),
    )

    command = CreateSupportCaseCommand(
        world_id="world-a",
        idempotency_key="request-001",
        actor_user_id="alice",
        user_id="alice",
        service_id="webgui-noc-prod",
        summary="Cannot access Web GUI",
        description="Cannot access Web GUI.",
        severity="high",
    )

    first = enterprise.create_support_case(command)

    second = enterprise.create_support_case(command)

    assert first == second

    assert len(uow.support_cases.items) == 1
    assert len(uow.audit_events.items) == 1

    assert uow.commit_count == 1
