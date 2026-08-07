from dataclasses import dataclass, replace
from typing import Literal

from supportbench.simulator.models import (
    Product,
    ServiceInstance,
    SimulatorWorld,
)

type ScenarioName = Literal[
    "healthy",
    "dash_outage",
    "old_dash_version",
]


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    world: SimulatorWorld
    products: tuple[Product, ...]
    services: tuple[ServiceInstance, ...]


"""
healthy
  webgui = operational
  dash   = operational, 3.1.2.1

dash_outage
  webgui = degraded
  dash   = outage, 3.1.2.1

old_dash_version
  webgui = operational
  dash   = operational, 3.1.0.3
"""


def build_healthy_scenario(
    *,
    world_id: str,
) -> ScenarioDefinition:
    return ScenarioDefinition(
        world=SimulatorWorld(world_id=world_id, scenario_name="healthy"),
        products=(
            Product(
                product_key="netcool_webgui",
                display_name="IBM Netcool/OMNIbus Web GUI",
            ),
            Product(
                product_key="dash",
                display_name="IBM Dashboard Application Services Hub",
            ),
        ),
        services=(
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
            ServiceInstance(
                world_id=world_id,
                service_id="dash-noc-prod",
                display_name="NOC DASH",
                product_key="dash",
                version="3.1.2.1",
                environment="production",
                status="operational",
                owner_team="noc-platform",
            ),
        ),
    )


def build_dash_outage_scenario(
    *,
    world_id: str,
) -> ScenarioDefinition:
    base = build_healthy_scenario(
        world_id=world_id,
    )

    return ScenarioDefinition(
        world=replace(
            base.world,
            scenario_name="dash_outage",
        ),
        products=base.products,
        services=tuple(
            replace(
                service,
                status=("degraded" if service.service_id == "webgui-noc-prod" else "outage"),
            )
            if service.service_id
            in {
                "webgui-noc-prod",
                "dash-noc-prod",
            }
            else service
            for service in base.services
        ),
    )


def build_old_dash_version_scenario(
    *,
    world_id: str,
) -> ScenarioDefinition:
    base = build_healthy_scenario(
        world_id=world_id,
    )

    return ScenarioDefinition(
        world=replace(
            base.world,
            scenario_name="old_dash_version",
        ),
        products=base.products,
        services=tuple(
            replace(
                service,
                version="3.1.0.3",
            )
            if service.service_id == "dash-noc-prod"
            else service
            for service in base.services
        ),
    )


def build_scenario(
    *,
    name: ScenarioName,
    world_id: str,
) -> ScenarioDefinition:
    match name:
        case "healthy":
            return build_healthy_scenario(world_id=world_id)
        case "dash_outage":
            return build_dash_outage_scenario(world_id=world_id)
        case "old_dash_version":
            return build_old_dash_version_scenario(world_id=world_id)
