class ModelProtocolError(Exception):
    """The model provider returned an invalid protocol response."""


class OllamaProtocolError(ModelProtocolError):
    """Ollama response does not match the expected API contract."""
