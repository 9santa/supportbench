class ToolGatewayConfigurationError(Exception):
    """Invalid tool gateway configuration."""


class DuplicateToolNameError(ToolGatewayConfigurationError):
    def __init__(
        self,
        *,
        tool_name: str,
    ) -> None:
        self.tool_name = tool_name

        super().__init__(f"duplicate tool name: {tool_name!r}")
