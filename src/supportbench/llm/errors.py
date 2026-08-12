class ModelProtocolError(Exception):
    """The model provider returned an invalid protocol response."""


class ModelTransportError(Exception):
    """Communication with the model provider failed."""


class OllamaProtocolError(ModelProtocolError):
    """Ollama response does not match the expected API contract."""


class OllamaTransportError(ModelTransportError):
    """Communication with Ollama failed."""


"""
Error distinction:

connection refused / timeout
→ OllamaTransportError

HTTP succeeded,
but response violates protocol
→ OllamaProtocolError

valid tool call,
but service doesn't exist
→ ToolResult(service_not_found)
"""
