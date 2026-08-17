from typing import cast

from supportbench.agent.protocols import (
    AgentModelClient,
)
from supportbench.applications.support_agent import (
    build_support_agent,
)
from supportbench.knowledge.protocols import (
    SupportKnowledgeService,
)
from supportbench.tools.enterprise import (
    EnterpriseToolService,
)


def test_support_agent_exposes_unified_tool_surface() -> None:
    runtime = build_support_agent(
        enterprise_service=cast(
            EnterpriseToolService,
            object(),
        ),
        knowledge_service=cast(
            SupportKnowledgeService,
            object(),
        ),
        model=cast(
            AgentModelClient,
            object(),
        ),
    )

    names = {definition.name for definition in runtime.tool_gateway.definitions}

    assert names == {
        "search_products",
        "get_service_status",
        "get_installed_product",
        "check_user_entitlement",
        "create_support_case",
        "search_support_docs",
        "read_support_doc",
    }
