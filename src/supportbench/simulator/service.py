from supportbench.simulator.clock import Clock, SystemClock
from supportbench.simulator.ids import IdGenerator, UuidGenerator
from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.simulator.errors import (
    ServiceNotFoundError,
    InstalledProductNotFoundError,
    SupportCaseNotFoundError,
    UserEntitlementNotFoundError,
)
from supportbench.simulator.models import (
    InstalledProduct,
    ServiceInstance,
    UserEntitlement,
    AuditEvent,
    SupportCase,
)
from supportbench.simulator.repositories import UnitOfWorkFactory


class EnterpriseService:
    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock or SystemClock()
        self._id_generator = id_generator or UuidGenerator()

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

    def create_support_case(
        self,
        command: CreateSupportCaseCommand,
    ) -> SupportCase:
        """
        1. idempotency lookup
        2. service lookup
        3. construct case
        4. construct audit
        5. add case
        6. add audit
        7. one commit
        """
        world_id = command.world_id.strip()
        idempotency_key = command.idempotency_key.strip()
        actor_user_id = command.actor_user_id.strip()
        user_id = command.user_id.strip()
        service_id = command.service_id.strip()
        summary = command.summary.strip()
        description = command.description.strip()

        if not world_id:
            raise ValueError("world_id must be non-empty")

        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")

        if not actor_user_id:
            raise ValueError("actor_user_id must be non-empty")

        if not user_id:
            raise ValueError("user_id must be non-empty")

        if not service_id:
            raise ValueError("service_id must be non-empty")

        if not summary:
            raise ValueError("summary must be non-empty")

        if not description:
            raise ValueError("description must be non-empty")

        with self._uow_factory() as uow:
            existing = uow.support_cases.get_by_idempotency_key(
                world_id=world_id,
                idempotency_key=idempotency_key,
            )

            # This case already exists
            if existing is not None:
                return existing

            service = uow.services.get(
                world_id=world_id,
                service_id=service_id,
            )

            if service is None:
                raise ServiceNotFoundError(
                    world_id=world_id,
                    service_id=service_id,
                )

            now = self._clock.now()

            support_case = SupportCase(
                world_id=world_id,
                case_id=self._id_generator.new_id(),
                idempotency_key=idempotency_key,
                actor_user_id=actor_user_id,
                user_id=user_id,
                service_id=service_id,
                summary=summary,
                description=description,
                severity=command.severity,
                status="open",
                assigned_team=service.owner_team,
                created_at=now,
                updated_at=now,
            )

            inserted = uow.support_cases.add_if_absent(support_case)

            if not inserted:
                existing = uow.support_cases.get_by_idempotency_key(
                    world_id=world_id,
                    idempotency_key=idempotency_key,
                )

                if existing is None:
                    raise RuntimeError(
                        "support case idempotency conflict "
                        "was detected, but the existing case could not be loaded"
                    )

                return existing

            audit_event = AuditEvent(
                world_id=world_id,
                event_id=self._id_generator.new_id(),
                event_type="support_case.created",
                actor_user_id=actor_user_id,
                entity_type="support_case",
                entity_id=support_case.case_id,
                occurred_at=now,
                metadata={
                    "service_id": service_id,
                    "user_id": user_id,
                    "severity": command.severity,
                    "status": "open",
                    "assigned_team": service.owner_team,
                    "idempotency_key": idempotency_key,
                },
            )

            uow.audit_events.add(audit_event)

            uow.commit()

            return support_case

    def get_support_case(
        self,
        *,
        world_id: str,
        case_id: str,
    ) -> SupportCase:
        normalized_world_id = world_id.strip()
        normalized_case_id = case_id.strip()

        if not normalized_world_id:
            raise ValueError("world_id must be non-empty")

        if not normalized_case_id:
            raise ValueError("case_id must be non-empty")

        with self._uow_factory() as uow:
            support_case = uow.support_cases.get(
                world_id=normalized_world_id,
                case_id=normalized_case_id,
            )

        if support_case is None:
            raise SupportCaseNotFoundError(
                world_id=normalized_world_id,
                case_id=normalized_case_id,
            )

        return support_case
