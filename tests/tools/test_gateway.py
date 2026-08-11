from collections.abc import Mapping

import pytest

from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.tools.definitions import (
    GET_SERVICE_STATUS,
    ToolDefinition,
)
from supportbench.tools.errors import (
    DuplicateToolNameError,
)
from supportbench.tools.gateway import ToolGateway
from supportbench.tools.models import (
    ToolCall,
    ToolExecutionContext,
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


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        world_id="trusted-world",
        actor_user_id="alice",
        request_id="req-001",
    )


def test_unknown_tool_returns_error() -> None:
    gateway = ToolGateway(())

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
            )
        )


from datetime import datetime, timezone

from supportbench.simulator.models import (
    InstalledProduct,
    ServiceInstance,
    SupportCase,
    UserEntitlement,
)
from supportbench.tools.enterprise import (
    build_enterprise_tool_handlers,
)


class FakeEnterpriseService:
    def __init__(self) -> None:
        self.last_world_id: str | None = None
        self.last_command: CreateSupportCaseCommand | None = None

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
        command,
    ) -> SupportCase:
        self.last_command = command

        now = datetime(
            2026,
            8,
            11,
            12,
            0,
            tzinfo=timezone.utc,
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
    return ToolGateway(build_enterprise_tool_handlers(service))


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
        ),
    )

    assert result.status == "success"
    assert service.last_world_id == "trusted-world"

    assert result.data is not None
    assert result.data["service_id"] == "webgui-noc-prod"


def test_create_support_case_uses_trusted_actor() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    result = gateway.execute(
        ToolCall(
            call_id="tc-017",
            name="create_support_case",
            arguments={
                "user_id": "bob",
                "service_id": "webgui-noc-prod",
                "summary": "Cannot access Web GUI",
                "description": ("Bob cannot access the Web GUI."),
                "severity": "high",
            },
        ),
        context=ToolExecutionContext(
            world_id="scenario-0042",
            actor_user_id="alice",
            request_id="req-900",
        ),
    )

    assert result.status == "success"

    command = service.last_command

    assert command is not None

    assert command.world_id == "scenario-0042"
    assert command.actor_user_id == "alice"

    assert command.user_id == "bob"


def test_same_tool_call_gets_same_idempotency_key() -> None:
    first_service = FakeEnterpriseService()
    second_service = FakeEnterpriseService()

    call = ToolCall(
        call_id="tc-017",
        name="create_support_case",
        arguments={
            "user_id": "alice",
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
    )

    _enterprise_gateway(first_service).execute(
        call,
        context=context,
    )

    _enterprise_gateway(second_service).execute(
        call,
        context=context,
    )

    assert first_service.last_command.idempotency_key == second_service.last_command.idempotency_key


def test_tool_definitions_have_strict_json_schemas() -> None:
    service = FakeEnterpriseService()
    gateway = _enterprise_gateway(service)

    definitions = {definition.name: definition for definition in gateway.definitions}

    assert set(definitions) == {
        "get_service_status",
        "get_installed_product",
        "check_user_entitlement",
        "create_support_case",
    }

    schema = definitions["get_service_status"].arguments_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    properties = schema["properties"]

    assert "service_id" in properties
    assert "world_id" not in properties
    assert "actor_user_id" not in properties


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
