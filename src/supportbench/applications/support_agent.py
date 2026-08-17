from dataclasses import dataclass

from supportbench.agent.orchestrator import (
    AgentOrchestrator,
)
from supportbench.agent.protocols import (
    AgentModelClient,
)
from supportbench.knowledge.protocols import (
    SupportKnowledgeService,
)
from supportbench.tools.enterprise import (
    EnterpriseToolExceptionMapper,
    EnterpriseToolService,
    build_enterprise_tool_handlers,
)
from supportbench.tools.gateway import (
    ToolGateway,
)
from supportbench.tools.knowledge import (
    KnowledgeToolExceptionMapper,
    build_knowledge_tool_handlers,
)
from supportbench.tools.policies import (
    build_support_agent_tool_policy_engine,
)


@dataclass(slots=True)
class SupportAgentRuntime:
    tool_gateway: ToolGateway
    orchestrator: AgentOrchestrator


def build_support_agent(
    *,
    enterprise_service: EnterpriseToolService,
    knowledge_service: SupportKnowledgeService,
    model: AgentModelClient,
    max_steps: int = 8,
) -> SupportAgentRuntime:
    handlers = (
        *build_enterprise_tool_handlers(enterprise_service),
        *build_knowledge_tool_handlers(knowledge_service),
    )

    gateway = ToolGateway(
        handlers,
        policy_engine=(build_support_agent_tool_policy_engine()),
        exception_mappers=(
            EnterpriseToolExceptionMapper(),
            KnowledgeToolExceptionMapper(),
        ),
    )

    orchestrator = AgentOrchestrator(
        model=model,
        gateway=gateway,
        max_steps=max_steps,
    )

    return SupportAgentRuntime(
        tool_gateway=gateway,
        orchestrator=orchestrator,
    )
