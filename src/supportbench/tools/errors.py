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


class MissingToolPolicyError(ToolGatewayConfigurationError):
    def __init__(
        self,
        *,
        tool_name: str,
    ) -> None:
        self.tool_name = tool_name

        super().__init__(f"missing policy for tool: {tool_name!r}")


class UnknownToolPolicyError(ToolGatewayConfigurationError):
    def __init__(
        self,
        *,
        tool_name: str,
    ) -> None:
        self.tool_name = tool_name

        super().__init__(f"policy configured for unknown tool: {tool_name!r}")
