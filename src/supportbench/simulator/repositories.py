from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from supportbench.simulator.models import (
    AuditEvent,
    InstalledProduct,
    Product,
    ServiceInstance,
    SupportCase,
    UserEntitlement,
)


class ProductRepository(Protocol):
    def search(
        self,
        *,
        query: str,
        limit: int,
    ) -> tuple[Product, ...]: ...


class ServiceRepository(Protocol):
    def get(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance | None: ...


class InstalledProductRepository(Protocol):
    def get(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct | None: ...


class UserEntitlementRepository(Protocol):
    def get(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement | None: ...


class SupportCaseRepository(Protocol):
    def get_by_idempotency_key(
        self,
        *,
        world_id: str,
        idempotency_key: str,
    ) -> SupportCase | None: ...

    def add(
        self,
        support_case: SupportCase,
    ) -> None: ...

    def add_if_absent(
        self,
        support_case: SupportCase,
    ) -> bool: ...

    def get(
        self,
        *,
        world_id: str,
        case_id: str,
    ) -> SupportCase | None: ...


class AuditEventRepository(Protocol):
    def add(
        self,
        event: AuditEvent,
    ) -> None: ...


class UnitOfWork(Protocol):
    products: ProductRepository
    services: ServiceRepository
    installed_products: InstalledProductRepository
    user_entitlements: UserEntitlementRepository

    support_cases: SupportCaseRepository
    audit_events: AuditEventRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


type UnitOfWorkFactory = Callable[[], UnitOfWork]
