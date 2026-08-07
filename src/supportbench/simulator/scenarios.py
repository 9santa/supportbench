from dataclasses import dataclass

from supportbench.simulator.models import (
    Product,
    ServiceInstance,
    SimulatorWorld,
)


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    world: SimulatorWorld
    products: tuple[Product, ...]
    services: tuple[ServiceInstance, ...]


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
