from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.simulator.models import (
    InstalledProduct,
    Product,
    ServiceInstance,
    SupportCase,
    UserEntitlement,
)
from supportbench.tools.definitions import (
    GET_SERVICE_STATUS,
    ToolDefinition,
)
from supportbench.tools.enterprise import (
    build_enterprise_tool_handlers,
)
from supportbench.tools.errors import (
    DuplicateToolNameError,
    MissingToolPolicyError,
    UnknownToolPolicyError,
)
from supportbench.tools.gateway import ToolGateway
from supportbench.tools.models import (
    ToolCall,
    ToolExecutionContext,
)
from supportbench.tools.policies import (
    CREATE_SUPPORT_CASE_PERMISSION,
    ENTERPRISE_READ_PERMISSION,
    StaticToolPolicyEngine,
    ToolPolicyRule,
    build_enterprise_tool_policy_engine,
    tool_approval_id,
)


class FakeHandler:
    def __init__(
        self,
        *,
        definition: ToolDefinition = GET_SERVICE_STATUS,
    ) -> None:
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        return {
            "world_id_seen": context.world_id,
        }


def _context(
    *,
    permissions: frozenset[str] = frozenset({ENTERPRISE_READ_PERMISSION}),
) -> ToolExecutionContext:
    return ToolExecutionContext(
        world_id="trusted-world",
        actor_user_id="alice",
        request_id="req-001",
        permissions=permissions,
    )


def test_unknown_tool_returns_error() -> None:
    gateway = ToolGateway(
        (),
        policy_engine=StaticToolPolicyEngine({}),
    )

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="does_not_exist",
            arguments={},
        ),
        context=_context(),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "unknown_tool"
    assert result.call_id == "tc-001"


def test_duplicate_tool_names_are_rejected() -> None:
    handler = FakeHandler()

    with pytest.raises(DuplicateToolNameError):
        ToolGateway(
            (
                handler,
                handler,
            ),
            policy_engine=StaticToolPolicyEngine({}),
        )


def test_missing_tool_policy_is_rejected() -> None:
    with pytest.raises(MissingToolPolicyError, match="get_service_status"):
        ToolGateway(
            (FakeHandler(),),
            policy_engine=StaticToolPolicyEngine({}),
        )


def test_policy_for_unknown_tool_is_rejected() -> None:
    with pytest.raises(UnknownToolPolicyError, match="unknown_tool"):
        ToolGateway(
            (),
            policy_engine=StaticToolPolicyEngine(
                {
                    "unknown_tool": ToolPolicyRule(
                        required_permissions=frozenset(),
                        requires_approval=False,
                    )
                }
            ),
        )


class FakeEnterpriseService:
    def __init__(self) -> None:
        self.last_world_id: str | None = None
        self.last_command: CreateSupportCaseCommand | None = None

    def search_products(
        self,
        *,
        query: str,
    ) -> tuple[Product, ...]:
        return (
            Product(
                product_key="dash",
                display_name="IBM Dashboard Application Services Hub",
            ),
        )

    def search_services(
        self,
        *,
        world_id: str,
        query: str,
    ) -> tuple[ServiceInstance, ...]:
        self.last_world_id = world_id
        return (
            ServiceInstance(
                world_id=world_id,
                service_id="webgui-noc-prod",
                display_name="NOC Web GUI",
                product_key="netcool_webgui",
                version="8.1 FP7",
                environment="production",
                status="operational",
                owner_team="noc-platform",
            ),
        )

    def get_service_status(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance:
        self.last_world_id = world_id

        return ServiceInstance(
            world_id=world_id,
            service_id=service_id,
            display_name="NOC Web GUI",
            product_key="netcool_webgui",
            version="8.1 FP7",
            environment="production",
            status="operational",
            owner_team="noc-platform",
        )

    def get_installed_product(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct:
        self.last_world_id = world_id

        return InstalledProduct(
            world_id=world_id,
            asset_id=asset_id,
            product_key=product_key,
            version="3.1.2.1",
            patch_level="FP1",
        )

    def check_user_entitlement(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement:
        self.last_world_id = world_id

        return UserEntitlement(
            world_id=world_id,
            user_id=user_id,
            service_id=service_id,
            granted=True,
            role="viewer",
        )

    def create_support_case(
        self,
        command: CreateSupportCaseCommand,
    ) -> SupportCase:
        self.last_command = command

        now = datetime(
            2026,
            8,
            11,
            12,
            0,
            tzinfo=UTC,
        )

        return SupportCase(
            world_id=command.world_id,
            case_id="CASE-001",
            idempotency_key=command.idempotency_key,
            actor_user_id=command.actor_user_id,
            user_id=command.user_id,
            service_id=command.service_id,
            summary=command.summary,
            description=command.description,
            severity=command.severity,
            status="open",
            assigned_team="noc-platform",
            created_at=now,
            updated_at=now,
        )


def _enterprise_gateway(
    service: FakeEnterpriseService,
) -> ToolGateway:
    return ToolGateway(
        build_enterprise_tool_handlers(service),
        policy_engine=build_enterprise_tool_policy_engine(),
    )


def _approved_context(
    call: ToolCall,
    *,
    world_id: str = "scenario-0042",
    actor_user_id: str = "alice",
    request_id: str = "req-900",
) -> ToolExecutionContext:
    context = ToolExecutionContext(
        world_id=world_id,
        actor_user_id=actor_user_id,
        request_id=request_id,
        permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
    )
    return replace(
        context,
        approved_tool_calls=frozenset({tool_approval_id(call=call, context=context)}),
    )


def test_world_id_cannot_be_controlled_by_tool_arguments() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="get_service_status",
            arguments={
                "service_id": "webgui-noc-prod",
                "world_id": "victim-world",
            },
        ),
        context=ToolExecutionContext(
            world_id="trusted-world",
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        ),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_arguments"

    assert service.last_world_id is None


def test_service_status_uses_trusted_world() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="get_service_status",
            arguments={
                "service_id": "webgui-noc-prod",
            },
        ),
        context=ToolExecutionContext(
            world_id="trusted-world",
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        ),
    )

    assert result.status == "success"
    assert service.last_world_id == "trusted-world"

    assert result.data is not None
    assert result.data == {
        "service_id": "webgui-noc-prod",
        "display_name": "NOC Web GUI",
        "product_key": "netcool_webgui",
        "version": "8.1 FP7",
        "environment": "production",
        "status": "operational",
        "owner_team": "noc-platform",
    }


def test_create_support_case_uses_trusted_actor_as_requester() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)
    call = ToolCall(
        call_id="tc-017",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": ("Alice cannot access the Web GUI."),
            "severity": "high",
        },
    )

    result = gateway.execute(
        call,
        context=_approved_context(call),
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data == {
        "case_id": "CASE-001",
        "service_id": "webgui-noc-prod",
        "summary": "Cannot access Web GUI",
        "description": "Alice cannot access the Web GUI.",
        "severity": "high",
        "status": "open",
        "assigned_team": "noc-platform",
        "created_at": "2026-08-11T12:00:00+00:00",
        "updated_at": "2026-08-11T12:00:00+00:00",
    }

    command = service.last_command

    assert command is not None

    assert command.world_id == "scenario-0042"
    assert command.actor_user_id == "alice"

    assert command.user_id == "alice"


def test_same_tool_call_gets_same_idempotency_key() -> None:
    first_service = FakeEnterpriseService()
    second_service = FakeEnterpriseService()

    call = ToolCall(
        call_id="tc-017",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": "Cannot access Web GUI.",
            "severity": "high",
        },
    )

    context = _approved_context(call)

    _enterprise_gateway(first_service).execute(
        call,
        context=context,
    )

    _enterprise_gateway(second_service).execute(
        call,
        context=context,
    )

    assert first_service.last_command is not None
    assert second_service.last_command is not None
    assert first_service.last_command.idempotency_key == second_service.last_command.idempotency_key


def test_read_tool_is_denied_without_permission() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="get_service_status",
            arguments={"service_id": "webgui-noc-prod"},
        ),
        context=_context(permissions=frozenset()),
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "forbidden"
    assert service.last_world_id is None


def test_read_tool_is_allowed_with_permission() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="get_service_status",
            arguments={"service_id": "webgui-noc-prod"},
        ),
        context=_context(),
    )

    assert result.status == "success"
    assert service.last_world_id == "trusted-world"


def test_mutating_tool_with_permission_requires_approval() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)
    call = ToolCall(
        call_id="tc-017",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": "Cannot access Web GUI.",
            "severity": "high",
        },
    )
    context = ToolExecutionContext(
        world_id="scenario-0042",
        actor_user_id="alice",
        request_id="req-900",
        permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
    )

    result = gateway.execute(
        call,
        context=context,
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "approval_required"
    assert result.error.details == {
        "approval_id": tool_approval_id(
            call=call,
            context=context,
        )
    }
    assert service.last_command is None


def test_approved_create_case_is_executed() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    call = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": "Cannot access Web GUI.",
            "severity": "high",
        },
    )

    context = _approved_context(call=call)

    result = gateway.execute(
        call,
        context=context,
    )

    assert result.status == "success"

    assert service.last_command is not None
    assert service.last_command.world_id == "scenario-0042"


def test_approval_is_bound_to_call_arguments() -> None:
    """
    Approved severity=high, after that changed call arguments to
    severity=critical, old approval should not work.
    """
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    original_call = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": "Cannot access Web GUI.",
            "severity": "high",
        },
    )

    context = ToolExecutionContext(
        world_id="trusted-world",
        actor_user_id="alice",
        request_id="req-001",
        permissions=frozenset({"support_case:create"}),
    )

    approval_id = tool_approval_id(
        call=original_call,
        context=context,
    )

    context = replace(
        context,
        approved_tool_calls=frozenset({approval_id}),
    )

    changed_call = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={
            **original_call.arguments,
            "severity": "critical",
        },
    )

    result = gateway.execute(
        changed_call,
        context=context,
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "approval_required"

    assert service.last_command is None


def test_all_registered_tools_have_explicit_policy() -> None:
    service = FakeEnterpriseService()

    gateway = ToolGateway(
        build_enterprise_tool_handlers(service),
        policy_engine=build_enterprise_tool_policy_engine(),
    )

    assert len(gateway.definitions) == 6


def test_tool_definitions_have_strict_json_schemas() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    definitions = {definition.name: definition for definition in gateway.definitions}

    assert set(definitions) == {
        "get_service_status",
        "search_services",
        "get_installed_product",
        "search_products",
        "check_user_entitlement",
        "create_support_case",
    }

    schema = definitions["get_service_status"].arguments_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    properties = schema["properties"]
    assert isinstance(properties, Mapping)

    assert "service_id" in properties
    assert "world_id" not in properties
    assert "actor_user_id" not in properties

    create_case_properties = definitions["create_support_case"].arguments_schema[
        "properties"
    ]
    assert isinstance(create_case_properties, Mapping)
    assert "user_id" not in create_case_properties


def test_create_support_case_rejects_model_supplied_user_id() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)
    call = ToolCall(
        call_id="tc-create-001",
        name="create_support_case",
        arguments={
            "user_id": "current_user",
            "service_id": "webgui-noc-prod",
            "summary": "Cannot access Web GUI",
            "description": "Cannot access Web GUI.",
            "severity": "high",
        },
    )

    result = gateway.execute(call, context=_approved_context(call))

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "invalid_arguments"
    assert service.last_command is None


def test_search_products_returns_canonical_keys() -> None:
    result = _enterprise_gateway(FakeEnterpriseService()).execute(
        ToolCall(
            call_id="tc-search-001",
            name="search_products",
            arguments={"query": "DASH"},
        ),
        context=ToolExecutionContext(
            world_id="world-a",
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        ),
    )

    assert result.status == "success"
    assert result.data == {
        "matches": [
            {
                "product_key": "dash",
                "display_name": "IBM Dashboard Application Services Hub",
            }
        ]
    }


def test_search_services_returns_identity_without_operational_state() -> None:
    result = _enterprise_gateway(FakeEnterpriseService()).execute(
        ToolCall(
            call_id="tc-search-service-001",
            name="search_services",
            arguments={"query": "production Web GUI"},
        ),
        context=ToolExecutionContext(
            world_id="world-a",
            actor_user_id="alice",
            request_id="req-001",
            permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        ),
    )

    assert result.status == "success"
    assert result.data == {
        "matches": [
            {
                "service_id": "webgui-noc-prod",
                "display_name": "NOC Web GUI",
                "product_key": "netcool_webgui",
                "environment": "production",
            }
        ]
    }


class ExplodingEnterpriseService(FakeEnterpriseService):
    def get_service_status(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance:
        raise RuntimeError("postgresql://admin:secret@db/internal")


def test_internal_error_does_not_leak_details() -> None:
    gateway = _enterprise_gateway(ExplodingEnterpriseService())

    result = gateway.execute(
        ToolCall(
            call_id="tc-001",
            name="get_service_status",
            arguments={
                "service_id": "webgui-noc-prod",
            },
        ),
        context=_context(),
    )

    assert result.status == "error"
    assert result.error is not None

    assert result.error.code == "internal_error"

    assert "secret" not in result.error.message
    assert "postgresql" not in result.error.message
