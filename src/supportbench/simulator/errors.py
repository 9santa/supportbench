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


class InstalledProductNotFoundError(SimulatorError):
    def __init__(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> None:
        self.world_id = world_id
        self.asset_id = asset_id
        self.product_key = product_key

        super().__init__(
            "installed product not found: "
            f"world_id={world_id!r}, "
            f"asset_id={asset_id!r}, "
            f"product_key={product_key!r}"
        )


class UserEntitlementNotFoundError(SimulatorError):
    def __init__(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> None:
        self.world_id = world_id
        self.user_id = user_id
        self.service_id = service_id

        super().__init__(
            "user entitlement not found: "
            f"world_id={world_id!r}, "
            f"user_id={user_id!r}, "
            f"service_id={service_id!r}"
        )
