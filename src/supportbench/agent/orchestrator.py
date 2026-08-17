from collections.abc import Mapping, Sequence

from supportbench.agent.errors import (
    AgentOrchestrationError,
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
    ToolCall,
    ToolExecutionContext,
    ToolResult,
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

        return self._continue(
            history=history,
            steps=steps,
            context=context,
            next_step_index=0,
        )

    def resume_after_approval(
        self,
        *,
        previous: AgentRunResult,
        context: ToolExecutionContext,
    ) -> AgentRunResult:
        if previous.status != "approval_required":
            raise AgentOrchestrationError("only approval_required runs can be resumed")

        pending = previous.pending_approval

        if pending is None:
            raise AgentOrchestrationError("approval_required run has no pending approval")

        expected_approvals = frozenset({pending.approval_id})

        if context.approved_tool_calls != expected_approvals:
            raise AgentOrchestrationError(
                "resume context must grant exactly the pending approval"
            )

        if not previous.steps:
            raise AgentOrchestrationError("approval_required run has no paused step")

        paused_step = previous.steps[-1]

        if paused_step.step_index != pending.step_index:
            raise AgentOrchestrationError("pending approval does not match the paused step")

        executions = list(paused_step.tool_executions)

        if (
            not executions
            or executions[-1].call != pending.call
            or not _is_approval_required(executions[-1].result)
        ):
            raise AgentOrchestrationError("paused step does not end with the pending approval")

        history: list[Mapping[str, object]] = [dict(message) for message in previous.messages]
        steps = list(previous.steps[:-1])

        approved_result = self._gateway.execute(
            pending.call,
            context=context,
        )

        if _is_approval_required(approved_result):
            raise AgentOrchestrationError("approved tool call still requires approval")

        executions.append(
            AgentToolExecution(
                call=pending.call,
                result=approved_result,
            )
        )
        history.append(dict(self._model.tool_result_message(approved_result)))

        next_pending = self._execute_calls(
            calls=pending.remaining_calls,
            context=context,
            history=history,
            executions=executions,
            step_index=pending.step_index,
        )

        steps.append(
            AgentStep(
                step_index=pending.step_index,
                assistant_content=paused_step.assistant_content,
                tool_executions=tuple(executions),
            )
        )

        if next_pending is not None:
            return _approval_required_result(
                history=history,
                steps=steps,
                pending=next_pending,
            )

        return self._continue(
            history=history,
            steps=steps,
            context=context,
            next_step_index=pending.step_index + 1,
        )

    def _continue(
        self,
        *,
        history: list[Mapping[str, object]],
        steps: list[AgentStep],
        context: ToolExecutionContext,
        next_step_index: int,
    ) -> AgentRunResult:
        for step_index in range(next_step_index, self._max_steps):
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

            pending = self._execute_calls(
                calls=turn.tool_calls,
                context=context,
                history=history,
                executions=executions,
                step_index=step_index,
            )

            steps.append(
                AgentStep(
                    step_index=step_index,
                    assistant_content=turn.content,
                    tool_executions=tuple(executions),
                )
            )

            if pending is not None:
                return _approval_required_result(
                    history=history,
                    steps=steps,
                    pending=pending,
                )

        return AgentRunResult(
            status="max_steps_exceeded",
            final_answer=None,
            steps=tuple(steps),
            messages=tuple(history),
            pending_approval=None,
        )

    def _execute_calls(
        self,
        *,
        calls: Sequence[ToolCall],
        context: ToolExecutionContext,
        history: list[Mapping[str, object]],
        executions: list[AgentToolExecution],
        step_index: int,
    ) -> AgentApprovalRequest | None:
        for call_index, call in enumerate(calls):
            result = self._gateway.execute(call, context=context)
            executions.append(
                AgentToolExecution(
                    call=call,
                    result=result,
                )
            )

            if _is_approval_required(result):
                return AgentApprovalRequest(
                    approval_id=_approval_id_from_result(result),
                    call=call,
                    remaining_calls=tuple(calls[call_index + 1 :]),
                    step_index=step_index,
                )

            history.append(dict(self._model.tool_result_message(result)))

        return None


def _is_approval_required(result: ToolResult) -> bool:
    return (
        result.status == "error"
        and result.error is not None
        and result.error.code == "approval_required"
    )


def _approval_id_from_result(result: ToolResult) -> str:
    if (
        result.status != "error"
        or result.error is None
        or result.error.code != "approval_required"
    ):
        raise AgentOrchestrationError("tool result does not require approval")

    details = result.error.details

    if details is None:
        raise AgentOrchestrationError("approval_required result has no approval details")

    approval_id = details.get("approval_id")

    if not isinstance(approval_id, str) or not approval_id.strip():
        raise AgentOrchestrationError("approval_required result has no valid approval_id")

    return approval_id


def _approval_required_result(
    *,
    history: Sequence[Mapping[str, object]],
    steps: Sequence[AgentStep],
    pending: AgentApprovalRequest,
) -> AgentRunResult:
    return AgentRunResult(
        status="approval_required",
        final_answer=None,
        steps=tuple(steps),
        messages=tuple(history),
        pending_approval=pending,
    )
