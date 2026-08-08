from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from supportbench.simulator.models import (
    Environment,
    ServiceInstance,
    ServiceStatus,
    InstalledProduct,
    UserEntitlement,
)
from supportbench.simulator.postgres.schema import (
    service_instances,
    installed_products,
    user_entitlements,
)


class PostgresServiceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance | None:
        statement = select(service_instances).where(
            service_instances.c.world_id == world_id,
            service_instances.c.service_id == service_id,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return ServiceInstance(
            world_id=row["world_id"],
            service_id=row["service_id"],
            display_name=row["display_name"],
            product_key=row["product_key"],
            version=row["version"],
            environment=cast(
                Environment,
                row["environment"],
            ),
            status=cast(
                ServiceStatus,
                row["status"],
            ),
            owner_team=row["owner_team"],
        )


class PostgresInstalledProductRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def get(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct | None:
        statement = select(installed_products).where(
            installed_products.c.world_id == world_id,
            installed_products.c.asset_id == asset_id,
            installed_products.c.product_key == product_key,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return InstalledProduct(
            world_id=row["world_id"],
            asset_id=row["asset_id"],
            product_key=row["product_key"],
            version=row["version"],
            patch_level=row["patch_level"],
        )


class PostgresUserEntitlementRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def get(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement | None:
        statement = select(user_entitlements).where(
            user_entitlements.c.world_id == world_id,
            user_entitlements.c.user_id == user_id,
            user_entitlements.c.service_id == service_id,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return UserEntitlement(
            world_id=row["world_id"],
            user_id=row["user_id"],
            service_id=row["service_id"],
            granted=row["granted"],
            role=row["role"],
        )
