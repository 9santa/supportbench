class SimulatorError(Exception):
    """Base error for enterprise simulator."""


class ServiceNotFoundError(SimulatorError):
    def __init__(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> None:
        self.world_id = world_id
        self.service_id = service_id

        super().__init__(f"service not found: world_id={world_id!r}, service_id={service_id!r}")
