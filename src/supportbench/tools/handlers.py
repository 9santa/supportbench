from collections.abc import Mapping
from typing import Protocol

from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import ToolExecutionContext


# Gateway doesn't know what operation is behind the handler
class ToolHandler(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]: ...
