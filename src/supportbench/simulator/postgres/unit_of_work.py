from types import TracebackType
from typing import Self

from sqlalchemy.orm import Session

from supportbench.simulator.postgres.repositories import (
    PostgresServiceRepository,
    PostgresInstalledProductRepository,
    PostgresUserEntitlementRepository,
)
from supportbench.simulator.postgres.session import SessionFactory
from supportbench.simulator.repositories import (
    ServiceRepository,
    InstalledProductRepository,
    UserEntitlementRepository,
)


class PostgresUnitOfWork:
    services: ServiceRepository
    installed_products: InstalledProductRepository
    user_entitlements: UserEntitlementRepository

    def __init__(
        self,
        session_factory: SessionFactory,
    ) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> Self:
        session = self._session_factory()

        self._session = session
        self.services = PostgresServiceRepository(session)
        self.installed_products = PostgresInstalledProductRepository(session)
        self.user_entitlements = PostgresUserEntitlementRepository(session)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return

        try:
            self._session.rollback()
        finally:
            self._session.close()
            self._session = None

    def commit(self) -> None:
        self._require_session().commit()

    def rollback(self) -> None:
        self._require_session().rollback()

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("unit of work is not active")

        return self._session
