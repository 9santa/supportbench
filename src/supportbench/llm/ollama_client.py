import json
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from supportbench.llm.errors import (
    OllamaProtocolError,
    OllamaTransportError,
)
from supportbench.llm.models import AssistantModelTurn
from supportbench.llm.ollama_tools import (
    parse_ollama_chat_response,
    tool_definitions_to_ollama,
)
from supportbench.tools.definitions import ToolDefinition


class OllamaToolCallingClient:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        think: bool = False,
    ) -> None:
        normalized_model = model_name.strip()
        normalized_base_url = base_url.strip()

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if context_window is not None and context_window <= 0:
            raise ValueError("context_window must be positive")

        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        self._model_name = normalized_model
        self._base_url = normalized_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._think = think

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[ToolDefinition],
        request_id: str,
        assistant_turn_index: int,
    ) -> AssistantModelTurn:
        options: dict[str, object] = {"temperature": self._temperature}

        if self._context_window is not None:
            options["num_ctx"] = self._context_window

        if self._max_output_tokens is not None:
            options["num_predict"] = self._max_output_tokens

        payload: dict[str, object] = {
            "model": self._model_name,
            "messages": [dict(message) for message in messages],
            "tools": tool_definitions_to_ollama(tools),
            "stream": False,
            "think": self._think,
            "options": options,
        }

        response = self._post_json(payload)

        return parse_ollama_chat_response(
            response,
            request_id=request_id,
            assistant_turn_index=assistant_turn_index,
        )

    def _post_json(
        self,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        request = Request(
            url=f"{self._base_url}/api/chat",
            data=json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                response_data = json.load(response)

        except HTTPError as exc:
            raise OllamaTransportError(f"Ollama returned HTTP {exc.code}") from exc

        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaTransportError("Ollama request failed") from exc

        if not isinstance(response_data, Mapping):
            raise OllamaProtocolError("Ollama returned an invalid response envelope")

        return response_data
