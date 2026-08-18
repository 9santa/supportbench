from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments_schema: Mapping[str, object]
    mutating: bool


class GetServiceStatusArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    service_id: str = Field(
        min_length=1,
    )


class GetInstalledProductArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    asset_id: str = Field(
        min_length=1,
    )
    product_key: str = Field(
        min_length=1,
    )


class SearchProductsArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
    )


class SearchServicesArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=500,
    )


class CheckUserEntitlementArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str = Field(
        min_length=1,
    )
    service_id: str = Field(
        min_length=1,
    )


class CreateSupportCaseArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str = Field(
        min_length=1,
    )
    service_id: str = Field(
        min_length=1,
    )
    summary: str = Field(
        min_length=1,
        max_length=200,
    )
    description: str = Field(
        min_length=1,
        max_length=4000,
    )
    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]


GET_SERVICE_STATUS = ToolDefinition(
    name="get_service_status",
    description=(
        "Return the current operational state and deployed version of an enterprise service."
    ),
    arguments_schema=(GetServiceStatusArguments.model_json_schema()),
    mutating=False,
)


GET_INSTALLED_PRODUCT = ToolDefinition(
    name="get_installed_product",
    description=(
        "Return the installed version and patch level for an exact canonical product_key. "
        "When the user provides a product name instead of a known key, call "
        "search_products first."
    ),
    arguments_schema=(GetInstalledProductArguments.model_json_schema()),
    mutating=False,
)


SEARCH_PRODUCTS = ToolDefinition(
    name="search_products",
    description=(
        "Search the product catalog by a user-facing name or partial name and return "
        "matching canonical product_key values. Use this before tools that require an "
        "exact product_key."
    ),
    arguments_schema=(SearchProductsArguments.model_json_schema()),
    mutating=False,
)


SEARCH_SERVICES = ToolDefinition(
    name="search_services",
    description=(
        "Resolve a human-readable enterprise service name to canonical service_id values. "
        "Use this before tools that require an exact service_id."
    ),
    arguments_schema=(SearchServicesArguments.model_json_schema()),
    mutating=False,
)


CHECK_USER_ENTITLEMENT = ToolDefinition(
    name="check_user_entitlement",
    description=("Return whether a user is entitled to access an enterprise service."),
    arguments_schema=(CheckUserEntitlementArguments.model_json_schema()),
    mutating=False,
)


CREATE_SUPPORT_CASE = ToolDefinition(
    name="create_support_case",
    description=("Create a support case for a user and enterprise service."),
    arguments_schema=(CreateSupportCaseArguments.model_json_schema()),
    mutating=True,
)
