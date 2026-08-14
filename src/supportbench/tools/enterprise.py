from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from typing import Protocol

from supportbench.simulator.commands import (
    CreateSupportCaseCommand,
)
from supportbench.simulator.models import (
    InstalledProduct,
    Product,
    ServiceInstance,
    SupportCase,
    UserEntitlement,
)
from supportbench.tools.definitions import (
    CHECK_USER_ENTITLEMENT,
    CREATE_SUPPORT_CASE,
    GET_INSTALLED_PRODUCT,
    GET_SERVICE_STATUS,
    SEARCH_PRODUCTS,
    CheckUserEntitlementArguments,
    CreateSupportCaseArguments,
    GetInstalledProductArguments,
    GetServiceStatusArguments,
    SearchProductsArguments,
    ToolDefinition,
)
from supportbench.tools.handlers import ToolHandler
from supportbench.tools.models import (
    ToolExecutionContext,
)


class EnterpriseToolService(Protocol):
    def search_products(
        self,
        *,
        query: str,
    ) -> tuple[Product, ...]: ...

    def get_service_status(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance: ...

    def get_installed_product(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct: ...

    def check_user_entitlement(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement: ...

    def create_support_case(
        self,
        command: CreateSupportCaseCommand,
    ) -> SupportCase: ...


def _tool_idempotency_key(
    *,
    request_id: str,
    call_id: str,
) -> str:
    """
    Idempotency key is created with hashing, because
    its length is restricted and this way we don't care
    how long `call_id` is.
    """
    payload = f"{request_id}\0create_support_case\0{call_id}"

    digest = sha256(payload.encode("utf-8")).hexdigest()

    return f"agent-tool:{digest}"


class GetServiceStatusHandler:
    def __init__(
        self,
        service: EnterpriseToolService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return GET_SERVICE_STATUS

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = GetServiceStatusArguments.model_validate(arguments)

        result = self._service.get_service_status(
            world_id=context.world_id,
            service_id=args.service_id,
        )

        return asdict(result)


class GetInstalledProductHandler:
    def __init__(
        self,
        service: EnterpriseToolService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return GET_INSTALLED_PRODUCT

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = GetInstalledProductArguments.model_validate(arguments)

        result = self._service.get_installed_product(
            world_id=context.world_id,
            asset_id=args.asset_id,
            product_key=args.product_key,
        )

        return asdict(result)


class SearchProductsHandler:
    def __init__(
        self,
        service: EnterpriseToolService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return SEARCH_PRODUCTS

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = SearchProductsArguments.model_validate(arguments)

        matches = self._service.search_products(
            query=args.query,
        )

        return {
            "matches": [asdict(product) for product in matches],
        }


class CheckUserEntitlementHandler:
    def __init__(
        self,
        service: EnterpriseToolService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return CHECK_USER_ENTITLEMENT

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = CheckUserEntitlementArguments.model_validate(arguments)

        result = self._service.check_user_entitlement(
            world_id=context.world_id,
            user_id=args.user_id,
            service_id=args.service_id,
        )

        return asdict(result)


class CreateSupportCaseHandler:
    def __init__(
        self,
        service: EnterpriseToolService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return CREATE_SUPPORT_CASE

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = CreateSupportCaseArguments.model_validate(arguments)

        command = CreateSupportCaseCommand(
            world_id=context.world_id,
            idempotency_key=_tool_idempotency_key(
                request_id=context.request_id,
                call_id=call_id,
            ),
            actor_user_id=context.actor_user_id,
            user_id=args.user_id,
            service_id=args.service_id,
            summary=args.summary,
            description=args.description,
            severity=args.severity,
        )

        result = self._service.create_support_case(command)

        data = asdict(result)

        data["created_at"] = result.created_at.isoformat()
        data["updated_at"] = result.updated_at.isoformat()

        return data


def build_enterprise_tool_handlers(
    service: EnterpriseToolService,
) -> tuple[ToolHandler, ...]:
    return (
        GetServiceStatusHandler(service),
        SearchProductsHandler(service),
        GetInstalledProductHandler(service),
        CheckUserEntitlementHandler(service),
        CreateSupportCaseHandler(service),
    )
