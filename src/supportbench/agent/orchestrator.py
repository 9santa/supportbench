from collections.abc import Mapping, Sequence

from supportbench.agent.errors import (
    EmptyAssistantTurnError,
)
from supportbench.agent.models import (
    AgentApprovalRequest,
    AgentRunResult,
    AgentStep,
    AgentToolExecution,
)
from supportbench.agent.protocols import (
    AgentModelClient,
    AgentToolGateway,
)
from supportbench.tools.models import (
    ToolExecutionContext,
)
from supportbench.tools.policies import (
    tool_approval_id,
)


class AgentOrchestrator:
    def __init__(
        self,
        *,
        model: AgentModelClient,
        gateway: AgentToolGateway,
        max_steps: int = 8,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self._model = model
        self._gateway = gateway
        self._max_steps = max_steps

    def run(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        context: ToolExecutionContext,
    ) -> AgentRunResult:
        history: list[Mapping[str, object]] = [dict(message) for message in messages]

        steps: list[AgentStep] = []

        for step_index in range(self._max_steps):
            turn = self._model.chat(
                messages=history,
                tools=self._gateway.definitions,
                request_id=context.request_id,
                assistant_turn_index=step_index,
            )

            if not turn.tool_calls:
                final_answer = turn.content.strip()

                if not final_answer:
                    raise EmptyAssistantTurnError(
                        step_index=step_index,
                        finish_reason=turn.finish_reason,
                        output_token_count=turn.output_token_count,
                    )

                history.append(dict(turn.history_message))

                steps.append(
                    AgentStep(
                        step_index=step_index,
                        assistant_content=turn.content,
                        tool_executions=(),
                    )
                )

                return AgentRunResult(
                    status="completed",
                    final_answer=final_answer,
                    steps=tuple(steps),
                    messages=tuple(history),
                    pending_approval=None,
                )

            history.append(dict(turn.history_message))

            executions: list[AgentToolExecution] = []

            for call in turn.tool_calls:
                result = self._gateway.execute(call, context=context)

                execution = AgentToolExecution(
                    call=call,
                    result=result,
                )

                executions.append(execution)

                if (
                    result.status == "error"
                    and result.error is not None
                    and result.error.code == "approval_required"
                ):
                    steps.append(
                        AgentStep(
                            step_index=step_index,
                            assistant_content=turn.content,
                            tool_executions=tuple(executions),
                        ),
                    )

                    return AgentRunResult(
                        status="approval_required",
                        final_answer=None,
                        steps=tuple(steps),
                        messages=tuple(history),
                        pending_approval=AgentApprovalRequest(
                            approval_id=(tool_approval_id(call=call, context=context)),
                            call=call,
                        ),
                    )

                history.append(dict(self._model.tool_result_message(result)))

            steps.append(
                AgentStep(
                    step_index=step_index,
                    assistant_content=turn.content,
                    tool_executions=tuple(executions),
                )
            )

        return AgentRunResult(
            status="max_steps_exceeded",
            final_answer=None,
            steps=tuple(steps),
            messages=tuple(history),
            pending_approval=None,
        )
