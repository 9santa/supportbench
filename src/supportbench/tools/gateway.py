import logging
from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from supportbench.simulator.errors import (
    InstalledProductNotFoundError,
    ServiceNotFoundError,
    UserEntitlementNotFoundError,
)
from supportbench.tools.definitions import (
    ToolDefinition,
)
from supportbench.tools.errors import (
    DuplicateToolNameError,
)
from supportbench.tools.handlers import ToolHandler
from supportbench.tools.models import (
    ToolCall,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolResult,
)


logger = logging.getLogger(__name__)


class ToolGateway:
    def __init__(
        self,
        handlers: Iterable[ToolHandler],
    ) -> None:
        registry: dict[str, ToolHandler] = {}

        for handler in handlers:
            name = handler.definition.name

            if name in registry:
                raise DuplicateToolNameError(
                    tool_name=name,
                )

            registry[name] = handler

        self._handlers = registry

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

        except ServiceNotFoundError as exc:
            return _error_result(
                call=call,
                code="service_not_found",
                message=(f"Service {exc.service_id!r} was not found."),
            )

        except InstalledProductNotFoundError as exc:
            return _error_result(
                call=call,
                code="installed_product_not_found",
                message=(
                    "Installed product "
                    f"{exc.product_key!r} was not found "
                    f"on asset {exc.asset_id!r}."
                ),
            )

        except UserEntitlementNotFoundError as exc:
            return _error_result(
                call=call,
                code="user_entitlement_not_found",
                message=(
                    "No entitlement was found for user "
                    f"{exc.user_id!r} and service "
                    f"{exc.service_id!r}."
                ),
            )

        except Exception:
            logger.exception(
                "Unexpected tool execution failure",
                extra={
                    "tool_name": call.name,
                    "call_id": call.call_id,
                    "request_id": context.request_id,
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
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        status="error",
        data=None,
        error=ToolErrorInfo(
            code=code,
            message=message,
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
