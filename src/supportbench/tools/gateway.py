import logging
from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from supportbench.tools.definitions import (
    ToolDefinition,
)
from supportbench.tools.errors import (
    DuplicateToolNameError,
)
from supportbench.tools.exception_mapping import ToolExceptionMapper
from supportbench.tools.handlers import ToolHandler
from supportbench.tools.models import (
    ToolCall,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
)
from supportbench.tools.policies import ToolPolicyEngine

logger = logging.getLogger(__name__)


class ToolGateway:
    def __init__(
        self,
        handlers: Iterable[ToolHandler],
        *,
        policy_engine: ToolPolicyEngine,
        exception_mappers: Iterable[ToolExceptionMapper] = (),
    ) -> None:
        registry: dict[str, ToolHandler] = {}

        for handler in handlers:
            name = handler.definition.name

            if name in registry:
                raise DuplicateToolNameError(
                    tool_name=name,
                )

            registry[name] = handler

        policy_engine.validate_registered_tools(
            registry.keys(),
        )

        self._handlers = registry
        self._policy_engine = policy_engine
        self._exception_mappers = tuple(exception_mappers)

    @property
    def definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        return tuple(handler.definition for handler in self._handlers.values())

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
    ) -> ToolResult:
        handler = self._handlers.get(call.name)

        if handler is None:
            return _error_result(
                call=call,
                code="unknown_tool",
                message=f"Unknown tool {call.name!r}.",
            )

        decision = self._policy_engine.evaluate(
            call=call,
            definition=handler.definition,
            context=context,
        )

        if decision.outcome != "allow":
            return _error_result(
                call=call,
                code=decision.code or "forbidden",
                message=decision.message or "Tool execution is not allowed.",
                details=decision.details,
            )

        try:
            data = handler.execute(
                call_id=call.call_id,
                arguments=call.arguments,
                context=context,
            )

        except ValidationError as exc:
            return _error_result(
                call=call,
                code="invalid_arguments",
                message=_validation_message(exc),
            )

        except Exception as exc:
            for mapper in self._exception_mappers:
                mapped = mapper.map_exception(exc)

                if mapped is not None:
                    return _error_result(
                        call=call,
                        code=mapped.code,
                        message=mapped.message,
                    )

            logger.exception(
                "Unexpected tool execution failure",
                extra={
                    "tool_name": call.name,
                    "call_id": call.call_id,
                    "request_id": (context.request_id),
                    "world_id": context.world_id,
                },
            )

            return _error_result(
                call=call,
                code="internal_error",
                message=("The tool could not be executed."),
            )

        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            status="success",
            data=data,
            error=None,
        )


def _error_result(
    *,
    call: ToolCall,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        status="error",
        data=None,
        error=ToolErrorInfo(
            code=code,
            message=message,
            details=details,
        ),
    )


def _validation_message(
    exc: ValidationError,
) -> str:
    problems: list[str] = []

    for error in exc.errors(
        include_input=False,
        include_url=False,
    ):
        location = ".".join(str(part) for part in error["loc"])

        message = error["msg"]

        if location:
            problems.append(f"{location}: {message}")
        else:
            problems.append(message)

    if not problems:
        return "Tool arguments failed validation."

    return "Tool arguments failed validation: " + "; ".join(problems)
