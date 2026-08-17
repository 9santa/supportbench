import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.errors import (
    MissingToolPolicyError,
    UnknownToolPolicyError,
)
from supportbench.tools.models import ToolCall, ToolExecutionContext

type ToolPolicyOutcome = Literal[
    "allow",
    "deny",
    "approval_required",
]


@dataclass(frozen=True, slots=True)
class ToolPolicyDecision:
    outcome: ToolPolicyOutcome
    code: str | None = None
    message: str | None = None
    details: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ToolPolicyRule:
    required_permissions: frozenset[str]
    requires_approval: bool


ENTERPRISE_READ_PERMISSION = "enterprise:read"
CREATE_SUPPORT_CASE_PERMISSION = "support_case:create"


def tool_approval_id(
    *,
    call: ToolCall,
    context: ToolExecutionContext,
) -> str:
    payload = {
        "world_id": context.world_id,
        "actor_user_id": context.actor_user_id,
        "request_id": context.request_id,
        "call_id": call.call_id,
        "tool_name": call.name,
        "arguments": dict(call.arguments),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    digest = sha256(encoded).hexdigest()

    return f"tool-approval:{digest}"


class ToolPolicyEngine(Protocol):
    def validate_registered_tools(
        self,
        tool_names: Collection[str],
    ) -> None:
        """
        Fail-closed: if new handler was added without
        policy, ToolGateway won't start.
        """
        ...

    def evaluate(
        self,
        *,
        call: ToolCall,
        definition: ToolDefinition,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision: ...


class StaticToolPolicyEngine(ToolPolicyEngine):
    def __init__(
        self,
        rules: Mapping[str, ToolPolicyRule],
    ) -> None:
        self._rules = rules

    def validate_registered_tools(
        self,
        tool_names: Collection[str],
    ) -> None:
        registered = set(tool_names)
        configured = set(self._rules)

        missing = registered - configured

        if missing:
            raise MissingToolPolicyError(
                tool_name=sorted(missing)[0],
            )

        unknown = configured - registered

        if unknown:
            raise UnknownToolPolicyError(
                tool_name=sorted(unknown)[0],
            )

    def evaluate(
        self,
        *,
        call: ToolCall,
        definition: ToolDefinition,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        rule = self._rules[definition.name]

        missing_permissions = rule.required_permissions - context.permissions

        if missing_permissions:
            return ToolPolicyDecision(
                outcome="deny",
                code="forbidden",
                message=("The caller does not have permissions to execute this tool."),
            )

        if rule.requires_approval:
            approval_id = tool_approval_id(
                call=call,
                context=context,
            )

            if approval_id not in context.approved_tool_calls:
                return ToolPolicyDecision(
                    outcome="approval_required",
                    code="approval_required",
                    message=("This tool call requires approval before it can be executed."),
                    details={"approval_id": approval_id},
                )

        return ToolPolicyDecision(
            outcome="allow",
        )


ENTERPRISE_TOOL_POLICIES = {
    "get_service_status": ToolPolicyRule(
        required_permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        requires_approval=False,
    ),
    "get_installed_product": ToolPolicyRule(
        required_permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        requires_approval=False,
    ),
    "search_products": ToolPolicyRule(
        required_permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        requires_approval=False,
    ),
    "check_user_entitlement": ToolPolicyRule(
        required_permissions=frozenset({ENTERPRISE_READ_PERMISSION}),
        requires_approval=False,
    ),
    "create_support_case": ToolPolicyRule(
        required_permissions=frozenset({CREATE_SUPPORT_CASE_PERMISSION}),
        requires_approval=True,
    ),
}


SUPPORT_DOCS_READ_PERMISSION = "support_docs:read"

KNOWLEDGE_TOOL_POLICIES = {
    "search_support_docs": ToolPolicyRule(
        required_permissions=frozenset({SUPPORT_DOCS_READ_PERMISSION}),
        requires_approval=False,
    ),
    "read_support_doc": ToolPolicyRule(
        required_permissions=frozenset({SUPPORT_DOCS_READ_PERMISSION}),
        requires_approval=False,
    ),
}

SUPPORT_AGENT_TOOL_POLICIES = {
    **ENTERPRISE_TOOL_POLICIES,
    **KNOWLEDGE_TOOL_POLICIES,
}


def build_support_agent_tool_policy_engine() -> StaticToolPolicyEngine:
    return StaticToolPolicyEngine(SUPPORT_AGENT_TOOL_POLICIES)


def build_enterprise_tool_policy_engine() -> StaticToolPolicyEngine:
    return StaticToolPolicyEngine(ENTERPRISE_TOOL_POLICIES)
