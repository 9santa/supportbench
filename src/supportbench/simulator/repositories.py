from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self

from supportbench.simulator.models import ServiceInstance


class ServiceRepository(Protocol):
    def get(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance | None: ...


class UnitOfWork(Protocol):
    services: ServiceRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


type UnitOfWordFactory = Callable[[], UnitOfWork]
