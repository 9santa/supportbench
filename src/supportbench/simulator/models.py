from dataclasses import dataclass
from typing import Literal, Any
from datetime import datetime

type Environment = Literal[
    "production",
    "staging",
    "development",
]

type ServiceStatus = Literal[
    "operational",
    "degraded",
    "outage",
    "maintenance",
]

type CaseSeverity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]

type CaseStatus = Literal[
    "open",
    "in_progress",
    "escalated",
    "resolved",
]

_ENVIRONMENTS = frozenset(
    {
        "production",
        "staging",
        "development",
    }
)

_SERVICE_STATUSES = frozenset(
    {
        "operational",
        "degraded",
        "outage",
        "maintenance",
    }
)

_CASE_SEVERITIES = frozenset(
    {
        "low",
        "medium",
        "high",
        "critical",
    }
)

_CASE_STATUSES = frozenset(
    {
        "open",
        "in_progress",
        "escalated",
        "resolved",
    }
)

# Runtime validation even though DB will have its own constraints,
# because domain validation + database constraints are different guard lines.


def _require_non_empty(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class SimulatorWorld:
    world_id: str
    scenario_name: str

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.scenario_name, field_name="scenario_name")


@dataclass(frozen=True, slots=True)
class Product:
    product_key: str
    display_name: str

    def __post_init__(self) -> None:
        _require_non_empty(self.product_key, field_name="product_key")
        _require_non_empty(self.display_name, field_name="display_name")


@dataclass(frozen=True, slots=True)
class ServiceInstance:
    world_id: str
    service_id: str
    display_name: str
    product_key: str
    version: str
    environment: Environment
    status: ServiceStatus
    owner_team: str

    def __post_init__(self) -> None:
        text_values = {
            "world_id": self.world_id,
            "service_id": self.service_id,
            "display_name": self.display_name,
            "product_key": self.product_key,
            "version": self.version,
            "owner_team": self.owner_team,
        }

        for field_name, value in text_values.items():
            _require_non_empty(
                value,
                field_name=field_name,
            )

        if self.environment not in _ENVIRONMENTS:
            raise ValueError(f"unknown environment: {self.environment!r}")

        if self.status not in _SERVICE_STATUSES:
            raise ValueError(f"unknown service status: {self.status!r}")


@dataclass(frozen=True, slots=True)
class Asset:
    world_id: str
    asset_id: str
    hostname: str
    operating_system: str
    environment: Environment

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.asset_id, field_name="asset_id")
        _require_non_empty(self.hostname, field_name="hostname")
        _require_non_empty(
            self.operating_system,
            field_name="operating_system",
        )

        if self.environment not in _ENVIRONMENTS:
            raise ValueError(f"invalid environment: {self.environment!r}")


@dataclass(frozen=True, slots=True)
class InstalledProduct:
    world_id: str
    asset_id: str
    product_key: str
    version: str
    patch_level: str

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.asset_id, field_name="asset_id")
        _require_non_empty(
            self.product_key,
            field_name="product_key",
        )
        _require_non_empty(self.version, field_name="version")
        _require_non_empty(
            self.patch_level,
            field_name="patch_level",
        )


@dataclass(frozen=True, slots=True)
class User:
    world_id: str
    user_id: str
    display_name: str
    department: str

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.user_id, field_name="user_id")
        _require_non_empty(
            self.display_name,
            field_name="display_name",
        )
        _require_non_empty(
            self.department,
            field_name="department",
        )


@dataclass(frozen=True, slots=True)
class UserEntitlement:
    world_id: str
    user_id: str
    service_id: str
    granted: bool  # important for `access_denied`
    role: str

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.user_id, field_name="user_id")
        _require_non_empty(
            self.service_id,
            field_name="service_id",
        )
        _require_non_empty(self.role, field_name="role")


@dataclass(frozen=True, slots=True)
class SupportCase:
    world_id: str
    case_id: str
    idempotency_key: str

    # For example: actor_user_id = alice, user_id = bob
    # means that Alice opens a case from Bob's name.
    actor_user_id: str
    user_id: str
    service_id: str

    summary: str
    description: str
    severity: CaseSeverity
    status: CaseStatus
    assigned_team: str

    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.case_id, field_name="case_id")
        _require_non_empty(
            self.idempotency_key,
            field_name="idempotency_key",
        )
        _require_non_empty(
            self.actor_user_id,
            field_name="actor_user_id",
        )
        _require_non_empty(self.user_id, field_name="user_id")
        _require_non_empty(
            self.service_id,
            field_name="service_id",
        )
        _require_non_empty(self.summary, field_name="summary")
        _require_non_empty(
            self.description,
            field_name="description",
        )
        _require_non_empty(
            self.assigned_team,
            field_name="assigned_team",
        )

        if self.severity not in _CASE_SEVERITIES:
            raise ValueError(f"invalid case severity: {self.severity!r}")

        if self.status not in _CASE_STATUSES:
            raise ValueError(f"invalid case status: {self.status!r}")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        if self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    world_id: str
    event_id: str

    event_type: str
    actor_user_id: str

    entity_type: str
    entity_id: str

    occurred_at: datetime
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty(self.world_id, field_name="world_id")
        _require_non_empty(self.event_id, field_name="event_id")
        _require_non_empty(
            self.event_type,
            field_name="event_type",
        )
        _require_non_empty(
            self.actor_user_id,
            field_name="actor_user_id",
        )
        _require_non_empty(
            self.entity_type,
            field_name="entity_type",
        )
        _require_non_empty(
            self.entity_id,
            field_name="entity_id",
        )

        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
