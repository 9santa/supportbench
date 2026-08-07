from supportbench.simulator.errors import ServiceNotFoundError
from supportbench.simulator.models import ServiceInstance
from supportbench.simulator.repositories import UnitOfWordFactory


class EnterpriseService:
    def __init__(
        self,
        *,
        uow_factory: UnitOfWordFactory,
    ) -> None:
        self._uow_factory = uow_factory

    def get_service_status(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance:
        normalized_world_id = world_id.strip()
        normalized_service_id = service_id.strip()

        if not normalized_world_id:
            raise ValueError("world_id must be non-empty")

        if not normalized_service_id:
            raise ValueError("service_id must be non-empty")

        with self._uow_factory() as uow:
            service = uow.services.get(
                world_id=normalized_world_id,
                service_id=normalized_service_id,
            )

        if service is None:
            raise ServiceNotFoundError(
                world_id=normalized_world_id,
                service_id=normalized_service_id,
            )

        return service
