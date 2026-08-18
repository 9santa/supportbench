from dataclasses import dataclass, replace
from typing import Literal

from supportbench.simulator.models import (
    Asset,
    InstalledProduct,
    Product,
    ServiceInstance,
    SimulatorWorld,
    User,
    UserEntitlement,
)

type ScenarioName = Literal[
    "healthy",
    "dash_outage",
    "old_dash_version",
    "access_denied",
]


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    world: SimulatorWorld
    products: tuple[Product, ...]
    services: tuple[ServiceInstance, ...]
    assets: tuple[Asset, ...]
    installed_products: tuple[InstalledProduct, ...]
    users: tuple[User, ...]
    entitlements: tuple[UserEntitlement, ...]


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
        assets=(
            Asset(
                world_id=world_id,
                asset_id="dash-host-01",
                hostname="dash-host-01.example.test",
                operating_system="RHEL 9",
                environment="production",
            ),
        ),
        installed_products=(
            InstalledProduct(
                world_id=world_id,
                asset_id="dash-host-01",
                product_key="dash",
                version="3.1.2.1",
                patch_level="FP1",
            ),
        ),
        users=(
            User(
                world_id=world_id,
                user_id="alice",
                display_name="Alice",
                department="Operations",
            ),
        ),
        entitlements=(
            UserEntitlement(
                world_id=world_id,
                user_id="alice",
                service_id="webgui-noc-prod",
                granted=True,
                role="viewer",
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
        assets=base.assets,
        installed_products=base.installed_products,
        users=base.users,
        entitlements=base.entitlements,
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
        assets=base.assets,
        installed_products=tuple(
            replace(
                installed,
                version="3.1.0.3",
                patch_level="FP3",
            )
            if installed.product_key == "dash"
            else installed
            for installed in base.installed_products
        ),
        users=base.users,
        entitlements=base.entitlements,
    )


def build_access_denied_scenario(
    *,
    world_id: str,
) -> ScenarioDefinition:
    base = build_healthy_scenario(world_id=world_id)

    return ScenarioDefinition(
        world=replace(
            base.world,
            scenario_name="access_denied",
        ),
        products=base.products,
        services=base.services,
        assets=base.assets,
        installed_products=base.installed_products,
        users=base.users,
        entitlements=tuple(
            replace(
                entitlement,
                granted=False,
            )
            if (entitlement.user_id == "alice" and entitlement.service_id == "webgui-noc-prod")
            else entitlement
            for entitlement in base.entitlements
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
        case "access_denied":
            return build_access_denied_scenario(world_id=world_id)
