from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from supportbench.simulator.models import (
    ServiceInstance,
    InstalledProduct,
    UserEntitlement,
)


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


class UnitOfWork(Protocol):
    services: ServiceRepository
    installed_products: InstalledProductRepository
    user_entitlements: UserEntitlementRepository

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
