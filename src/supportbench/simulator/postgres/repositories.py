from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from supportbench.simulator.models import (
    Environment,
    ServiceInstance,
    ServiceStatus,
)
from supportbench.simulator.postgres.schema import service_instances


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
