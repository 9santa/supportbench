from supportbench.agentbench.models import AgentBenchScenario
from supportbench.tools.policies import (
    CREATE_SUPPORT_CASE_PERMISSION,
    ENTERPRISE_READ_PERMISSION,
    SUPPORT_DOCS_READ_PERMISSION,
)

READ_ONLY_PERMISSIONS = frozenset(
    {
        ENTERPRISE_READ_PERMISSION,
    }
)

KNOWLEDGE_PERMISSIONS = frozenset(
    {
        SUPPORT_DOCS_READ_PERMISSION,
    }
)

MIXED_PERMISSIONS = frozenset(
    {
        ENTERPRISE_READ_PERMISSION,
        SUPPORT_DOCS_READ_PERMISSION,
    }
)

WRITE_PERMISSIONS = frozenset(
    {
        ENTERPRISE_READ_PERMISSION,
        CREATE_SUPPORT_CASE_PERMISSION,
    }
)


# === ENTERPRISE SCENARIOS ===

ENTERPRISE_HEALTHY_STATUS = AgentBenchScenario(
    scenario_id=("enterprise-healthy-status"),
    kind="enterprise",
    world_scenario="healthy",
    user_message=("Check the current status of the production Web GUI service."),
    permissions=READ_ONLY_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_service_status",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=4,
)


ENTERPRISE_OUTAGE_STATUS = AgentBenchScenario(
    scenario_id=("enterprise-outage-status"),
    kind="enterprise",
    world_scenario="dash_outage",
    user_message=("Check whether the production Web GUI service is currently operational."),
    permissions=READ_ONLY_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_service_status",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=4,
)


ENTERPRISE_DASH_VERSION = AgentBenchScenario(
    scenario_id=("enterprise-installed-dash-version"),
    kind="enterprise",
    world_scenario="old_dash_version",
    user_message=("Which DASH version is currently installed on dash-host-01?"),
    permissions=READ_ONLY_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_installed_product",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=4,
)


ENTERPRISE_ACCESS_DENIED = AgentBenchScenario(
    scenario_id=("enterprise-access-denied"),
    kind="enterprise",
    world_scenario="access_denied",
    user_message=(
        "Check whether user alice currently has access to the production Web GUI service."
    ),
    permissions=READ_ONLY_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "check_user_entitlement",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=4,
)


# === KNOWLEDGE SCENARIOS ===

KNOWLEDGE_WEBGUI_PREREQUISITES = AgentBenchScenario(
    scenario_id=("knowledge-webgui-prerequisites"),
    kind="knowledge",
    world_scenario="healthy",
    user_message=(
        "Search the support documentation "
        "for prerequisites for installing "
        "IBM Netcool/OMNIbus Web GUI 8.1."
    ),
    permissions=KNOWLEDGE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=5,
)


KNOWLEDGE_SSL_MUTUAL_AUTH = AgentBenchScenario(
    scenario_id=("knowledge-ssl-mutual-auth"),
    kind="knowledge",
    world_scenario="healthy",
    user_message=(
        "Search IBM support documentation "
        "for guidance on configuring SSL "
        "mutual authentication in IBM "
        "HTTP Server."
    ),
    permissions=KNOWLEDGE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=5,
)


KNOWLEDGE_MQ_CLUSTER_ERROR = AgentBenchScenario(
    scenario_id=("knowledge-mq-cluster-resolution"),
    kind="knowledge",
    world_scenario="healthy",
    user_message=(
        "Search the support documentation "
        "for causes of "
        "MQRC_CLUSTER_RESOLUTION_ERROR "
        "and summarize the evidence."
    ),
    permissions=KNOWLEDGE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=5,
)


KNOWLEDGE_SEARCH_AND_READ = AgentBenchScenario(
    scenario_id=("knowledge-search-and-read"),
    kind="knowledge",
    world_scenario="healthy",
    user_message=(
        "Find the most relevant IBM "
        "support document about installing "
        "Netcool/OMNIbus Web GUI 8.1, "
        "read the relevant document "
        "evidence, and summarize the "
        "installation prerequisites."
    ),
    permissions=KNOWLEDGE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "search_support_docs",
            "read_support_doc",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=6,
)


# === MIXED SCENARIOS ===
MIXED_DASH_WEBGUI = AgentBenchScenario(
    scenario_id=("mixed-dash-webgui-requirements"),
    kind="mixed",
    world_scenario="old_dash_version",
    user_message=(
        "Determine which DASH version "
        "is currently installed on "
        "dash-host-01. Then search the "
        "support documentation for "
        "requirements relevant to "
        "Netcool/OMNIbus Web GUI 8.1 "
        "and determine what the evidence "
        "establishes about the installed "
        "version. Do not make claims "
        "that the documentation does "
        "not support."
    ),
    permissions=MIXED_PERMISSIONS,
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


MIXED_OUTAGE_TROUBLESHOOTING = AgentBenchScenario(
    scenario_id=("mixed-outage-troubleshooting"),
    kind="mixed",
    world_scenario="dash_outage",
    user_message=(
        "Check the current Web GUI "
        "service state, then search IBM "
        "support documentation for "
        "relevant troubleshooting "
        "information. Do not modify "
        "the environment."
    ),
    permissions=MIXED_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_service_status",
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=7,
)


MIXED_ACCESS_AND_DOCUMENTATION = AgentBenchScenario(
    scenario_id=("mixed-access-and-documentation"),
    kind="mixed",
    world_scenario="access_denied",
    user_message=(
        "Check whether alice currently "
        "has access to the Web GUI "
        "service, and search support "
        "documentation for information "
        "relevant to Web GUI access or "
        "authentication. Keep current "
        "enterprise state separate from "
        "documentation claims."
    ),
    permissions=MIXED_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "check_user_entitlement",
            "search_support_docs",
        }
    ),
    forbidden_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=7,
)


MIXED_HEALTHY_VERIFY_DOCS = AgentBenchScenario(
    scenario_id=("mixed-healthy-version-docs"),
    kind="mixed",
    world_scenario="healthy",
    user_message=(
        "Find the DASH version currently "
        "installed on dash-host-01 and "
        "search IBM support documentation "
        "for Web GUI 8.1 DASH "
        "prerequisites. Report the "
        "current state separately from "
        "the documented requirement."
    ),
    permissions=MIXED_PERMISSIONS,
    expected_status="completed",
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


# === WRITE SCENARIOS ===
WRITE_OUTAGE_APPROVED = AgentBenchScenario(
    scenario_id=("write-outage-approved"),
    kind="write",
    world_scenario="dash_outage",
    user_message=(
        "Check the current Web GUI service state and open a support case for the outage."
    ),
    permissions=WRITE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "get_service_status",
            "create_support_case",
        }
    ),
    max_tool_calls=7,
    approval_mode="approve",
    state_expectation="changed",
    expected_support_case_delta=1,
    expected_audit_event_delta=1,
)

WRITE_OUTAGE_AWAITS_APPROVAL = AgentBenchScenario(
    scenario_id=("write-outage-awaits-approval"),
    kind="write",
    world_scenario="dash_outage",
    user_message=("Open a support case for the current Web GUI outage."),
    permissions=WRITE_PERMISSIONS,
    expected_status=("approval_required"),
    required_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=5,
    approval_mode="none",
    state_expectation="unchanged",
    expected_support_case_delta=0,
    expected_audit_event_delta=0,
)

WRITE_DIRECT_CASE_APPROVED = AgentBenchScenario(
    scenario_id=("write-direct-case-approved"),
    kind="write",
    world_scenario="dash_outage",
    user_message=("Create a support case for the known Web GUI production outage."),
    permissions=WRITE_PERMISSIONS,
    expected_status="completed",
    required_tools=frozenset(
        {
            "create_support_case",
        }
    ),
    max_tool_calls=5,
    approval_mode="approve",
    state_expectation="changed",
    expected_support_case_delta=1,
    expected_audit_event_delta=1,
)

AGENTBENCH_V1 = (
    ENTERPRISE_HEALTHY_STATUS,
    ENTERPRISE_OUTAGE_STATUS,
    ENTERPRISE_DASH_VERSION,
    ENTERPRISE_ACCESS_DENIED,
    KNOWLEDGE_WEBGUI_PREREQUISITES,
    KNOWLEDGE_SSL_MUTUAL_AUTH,
    KNOWLEDGE_MQ_CLUSTER_ERROR,
    KNOWLEDGE_SEARCH_AND_READ,
    MIXED_DASH_WEBGUI,
    MIXED_OUTAGE_TROUBLESHOOTING,
    MIXED_ACCESS_AND_DOCUMENTATION,
    MIXED_HEALTHY_VERIFY_DOCS,
    WRITE_OUTAGE_APPROVED,
    WRITE_OUTAGE_AWAITS_APPROVAL,
    WRITE_DIRECT_CASE_APPROVED,
)

# ===================================================
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
