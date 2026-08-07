from dataclasses import dataclass
from typing import Literal

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

_ENVIRONMENTS = {
    "production",
    "staging",
    "development",
}

_SERVICE_STATUSES = {
    "operational",
    "degraded",
    "outage",
    "maintenance",
}

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
