from supportbench.simulator.scenarios import build_scenario


def test_dash_outage_scenario() -> None:
    scenario = build_scenario(
        name="dash_outage",
        world_id="world-a",
    )

    services = {service.service_id: service for service in scenario.services}

    assert scenario.world.scenario_name == "dash_outage"
    assert services["webgui-noc-prod"].status == "degraded"
    assert services["dash-noc-prod"].status == "outage"


def test_old_dash_version_is_operational() -> None:
    scenario = build_scenario(
        name="old_dash_version",
        world_id="world-a",
    )

    services = {service.service_id: service for service in scenario.services}

    dash = services["dash-noc-prod"]

    assert dash.status == "operational"
    assert dash.version == "3.1.0.3"
