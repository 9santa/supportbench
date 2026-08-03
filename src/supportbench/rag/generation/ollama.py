import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from supportbench.rag.generation.models import (
    ChatMessage,
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "answer",
                "abstain",
                "clarify",
            ],
        },
        "answer": {
            "type": "string",
        },
        "citation_ids": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "decision",
        "answer",
        "citation_ids",
    ],
    "additionalProperties": False,
}


class OllamaClientError(RuntimeError):
    """Raised when communication with Ollama fails."""


class OllamaLLMClient:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")

        if not base_url.strip():
            raise ValueError("base_url must be non-empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if context_window is not None and context_window <= 0:
            raise ValueError("context_window must be positive")

        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        if (
            context_window is not None
            and max_output_tokens is not None
            and max_output_tokens >= context_window
        ):
            raise ValueError("max_output_tokens must be smaller than context_window")

        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
    ) -> str:
        options: dict[str, float | int] = {
            "temperature": self._temperature,
        }

        if self._context_window is not None:
            options["num_ctx"] = self._context_window

        if self._max_output_tokens is not None:
            options["num_predict"] = self._max_output_tokens

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
            "format": ANSWER_SCHEMA,
            "options": options,
        }

        request = Request(
            url=(f"{self._base_url}/api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": ("application/json"),
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_data = json.load(response)
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OllamaClientError("Ollama request failed") from error

        if not isinstance(response_data, dict):
            raise OllamaClientError("Ollama returned an invalid response envelope")

        message = response_data.get("message")

        if not isinstance(message, dict):
            raise OllamaClientError("Ollama response does not contain a message object")

        content = message.get("content")

        if not isinstance(content, str):
            raise OllamaClientError("Ollama response does not contain textual message content")

        return content
