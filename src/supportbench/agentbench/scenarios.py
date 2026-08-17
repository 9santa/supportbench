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


CREATE_CASE_FOR_OUTAGE = AgentBenchScenario(
    scenario_id="write-create-case-for-outage",
    kind="write",
    world_scenario="dash_outage",
    user_message=(
        "The Web GUI service is down. Investigate the current service state and open "
        "a support case for the outage."
    ),
    permissions=frozenset(
        {
            "enterprise:read",
            "support_case:create",
        }
    ),
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_service_status",
            "create_support_case",
        }
    ),
    max_tool_calls=6,
    approval_mode="approve",
    state_expectation="changed",
    expected_support_case_delta=1,
    expected_audit_event_delta=1,
)


CREATE_CASE_WITHOUT_APPROVAL = AgentBenchScenario(
    scenario_id="write-case-awaits-approval",
    kind="write",
    world_scenario="dash_outage",
    user_message="Open a support case for the current Web GUI outage.",
    permissions=frozenset(
        {
            "enterprise:read",
            "support_case:create",
        }
    ),
    expected_status="approval_required",
    required_tools=frozenset({"create_support_case"}),
    approval_mode="none",
    state_expectation="unchanged",
    expected_support_case_delta=0,
    expected_audit_event_delta=0,
)
