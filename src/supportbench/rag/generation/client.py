from typing import Protocol

from supportbench.rag.generation.models import (
    ChatMessage,
    LLMResponse,
)


class LLMClient(Protocol):
    def generate(
        self,
        messages: tuple[ChatMessage, ...],
    ) -> LLMResponse:
        """Return model content together with generation metadata."""
        ...
