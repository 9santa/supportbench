from supportbench.agentbench.models import AgentBenchScenario

MIXED_DASH_WEBGUI = AgentBenchScenario(
    scenario_id=("mixed-dash-webgui-requirements"),
    kind="mixed",
    world_scenario="old_dash_version",
    user_message=(
        "Determine which DASH version is "
        "currently installed on dash-host-01. "
        "Then search support documentation "
        "for Web GUI 8.1 requirements and "
        "determine whether the installed "
        "version is sufficient. Do not make "
        "a compatibility claim unless the "
        "documentation establishes it."
    ),
    permissions=frozenset(
        {
            "enterprise:read",
            "support_docs:read",
        }
    ),
    expected_status="completed",
    # search_products() tool call is not required, the model can
    # call get_installed_product(product_key="dash") right away.
    required_tools=frozenset(
        {
            "get_installed_product",
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=6,
)
