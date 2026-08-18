from typing import Protocol

from supportbench.tools.models import ToolErrorInfo

# This is preferable because without this, ToolGateway directly imports
# simulator exceptions, mixing domains together.
# With an exception mapper, generic gateway truly doesn't know what simulator is.


class ToolExceptionMapper(Protocol):
    def map_exception(
        self,
        exc: Exception,
    ) -> ToolErrorInfo | None: ...
