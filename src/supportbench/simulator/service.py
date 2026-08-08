from supportbench.simulator.errors import (
    ServiceNotFoundError,
    InstalledProductNotFoundError,
    UserEntitlementNotFoundError,
)
from supportbench.simulator.models import (
    InstalledProduct,
    ServiceInstance,
    UserEntitlement,
)
from supportbench.simulator.repositories import UnitOfWorkFactory


class EnterpriseService:
    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
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

    def get_installed_product(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct:
        normalized_world_id = world_id.strip()
        normalized_asset_id = asset_id.strip()
        normalized_product_key = product_key.strip()

        if not normalized_world_id:
            raise ValueError("world_id must be non-empty")

        if not normalized_asset_id:
            raise ValueError("asset_id must be non-empty")

        if not normalized_product_key:
            raise ValueError("product_key must be non-empty")

        with self._uow_factory() as uow:
            installed_product = uow.installed_products.get(
                world_id=normalized_world_id,
                asset_id=normalized_asset_id,
                product_key=normalized_product_key,
            )

        if installed_product is None:
            raise InstalledProductNotFoundError(
                world_id=normalized_world_id,
                asset_id=normalized_asset_id,
                product_key=normalized_product_key,
            )

        return installed_product

    def check_user_entitlement(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement:
        normalized_world_id = world_id.strip()
        normalized_user_id = user_id.strip()
        normalized_service_id = service_id.strip()

        if not normalized_world_id:
            raise ValueError("world_id must be non-empty")

        if not normalized_user_id:
            raise ValueError("user_id must be non-empty")

        if not normalized_service_id:
            raise ValueError("service_id must be non-empty")

        with self._uow_factory() as uow:
            entitlement = uow.user_entitlements.get(
                world_id=normalized_world_id,
                user_id=normalized_user_id,
                service_id=normalized_service_id,
            )

        if entitlement is None:
            raise UserEntitlementNotFoundError(
                world_id=normalized_world_id,
                user_id=normalized_user_id,
                service_id=normalized_service_id,
            )

        return entitlement
