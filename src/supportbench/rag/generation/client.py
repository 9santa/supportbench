from typing import Protocol

from supportbench.rag.generation.models import (
    ChatMessage,
)


class LLMClient(Protocol):
    def generate(
        self,
        messages: tuple[ChatMessage, ...],
    ) -> str:
        """Return the raw text model response."""
        ...
