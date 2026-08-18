from supportbench.api.models import (
    AgentRunResponse,
    PendingApprovalResponse,
    ToolExecutionResponse,
)
from supportbench.api.runs import (
    StoredAgentRun,
)


def agent_run_response(
    stored: StoredAgentRun,
) -> AgentRunResponse:
    result = stored.result

    executions = tuple(
        ToolExecutionResponse(
            call_id=(execution.call.call_id),
            tool_name=(execution.call.name),
            arguments=dict(execution.call.arguments),
            status=(execution.result.status),
            error_code=(
                execution.result.error.code if (execution.result.error is not None) else None
            ),
        )
        for step in result.steps
        for execution in step.tool_executions
    )

    pending_response = None

    if result.pending_approval is not None:
        pending = result.pending_approval

        pending_response = PendingApprovalResponse(
            tool_name=(pending.call.name),
            arguments=dict(pending.call.arguments),
        )

    return AgentRunResponse(
        run_id=stored.run_id,
        world_id=stored.world_id,
        status=result.status,
        final_answer=(result.final_answer),
        pending_approval=(pending_response),
        tool_executions=executions,
    )
