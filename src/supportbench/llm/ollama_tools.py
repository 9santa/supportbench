from hashlib import sha256
import json
from collections.abc import Iterable, Mapping, Sequence

from supportbench.llm.errors import OllamaProtocolError
from supportbench.llm.models import AssistantModelTurn
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.models import ToolCall, ToolResult


# Ollama's expected structure:
# type=function
# function.name
# function.description
# function.parameters (json schema)


def tool_definition_to_ollama(
    definition: ToolDefinition,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": _mapping_to_json(definition.arguments_schema),
        },
    }


def tool_definitions_to_ollama(
    definitions: Iterable[ToolDefinition],
) -> list[dict[str, object]]:
    return [tool_definition_to_ollama(definition) for definition in definitions]


def _mapping_to_json(value: object) -> object:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise OllamaProtocolError("value is not JSON serializable") from exc


def _tool_call_id(
    *,
    request_id: str,
    assistant_turn_index: int,
    tool_index: int,
    tool_name: str,
    arguments: Mapping[str, object],
) -> str:
    payload = {
        "request_id": request_id,
        "assistant_turn_index": assistant_turn_index,
        "tool_index": tool_index,
        "tool_name": tool_name,
        "arguments": dict(arguments),
    }

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OllamaProtocolError("tool call arguments are not valid JSON") from exc

    digest = sha256(encoded).hexdigest()

    return f"ollama-call:{digest}"


def parse_ollama_chat_response(
    response: Mapping[str, object],
    *,
    request_id: str,
    assistant_turn_index: int,
) -> AssistantModelTurn:
    raw_message = response.get("message")

    if not isinstance(raw_message, Mapping):
        raise OllamaProtocolError("Ollama response does not contain a valid message object")

    raw_content = raw_message.get("content", "")

    if raw_content is None:
        raw_content = ""

    if not isinstance(raw_content, str):
        raise OllamaProtocolError("Ollama assistant content must be a string")

    raw_thinking = raw_message.get("thinking", "")

    if raw_thinking is None:
        raw_thinking = ""

    if not isinstance(raw_thinking, str):
        raise OllamaProtocolError("Ollama assistant thinking must be a string")

    raw_done_reason = response.get("done_reason")

    if raw_done_reason is not None and not isinstance(raw_done_reason, str):
        raise OllamaProtocolError("Ollama done_reason must be a string")

    raw_eval_count = response.get("eval_count")

    if raw_eval_count is not None and not isinstance(raw_eval_count, int):
        raise OllamaProtocolError("Ollama eval_count must be an integer")

    raw_tool_calls = raw_message.get("tool_calls", ())

    if raw_tool_calls is None:
        raw_tool_calls = ()

    if not isinstance(raw_tool_calls, (list, tuple)):
        raise OllamaProtocolError("Ollama tool_calls must be an array")

    tool_calls: list[ToolCall] = []

    for index, raw_call in enumerate(raw_tool_calls):
        tool_calls.append(
            _parse_tool_call(
                raw_call,
                request_id=request_id,
                assistant_turn_index=assistant_turn_index,
                tool_index=index,
            )
        )

    history_message: dict[str, object] = {
        "role": "assistant",
        "content": raw_content,
    }

    if raw_thinking:
        history_message["thinking"] = raw_thinking

    if raw_tool_calls:
        history_message["tool_calls"] = _mapping_to_json(raw_tool_calls)

    return AssistantModelTurn(
        content=raw_content,
        tool_calls=tuple(tool_calls),
        history_message=history_message,
        finish_reason=raw_done_reason,
        output_token_count=raw_eval_count,
    )


def _parse_tool_call(
    raw_call: object,
    *,
    request_id: str,
    assistant_turn_index: int,
    tool_index: int,
) -> ToolCall:
    if not isinstance(raw_call, Mapping):
        raise OllamaProtocolError("Ollama tool call must be an object")

    raw_function = raw_call.get("function")

    if not isinstance(raw_function, Mapping):
        raise OllamaProtocolError("Ollama tool call does not contain a valid function object")

    name = raw_function.get("name")

    if not isinstance(name, str) or not name.strip():
        raise OllamaProtocolError("Ollama tool call function name must be a non-empty string")

    raw_arguments = raw_function.get(
        "arguments",
        {},
    )

    if not isinstance(
        raw_arguments,
        Mapping,
    ):
        raise OllamaProtocolError("Ollama tool call arguments must be an object")

    arguments = dict(raw_arguments)

    call_id = _tool_call_id(
        request_id=request_id,
        assistant_turn_index=assistant_turn_index,
        tool_index=tool_index,
        tool_name=name,
        arguments=arguments,
    )

    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def tool_result_to_ollama_message(
    result: ToolResult,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "call_id": result.call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "data": (dict(result.data) if result.data is not None else None),
        "error": (
            {
                "code": result.error.code,
                "message": result.error.message,
            }
            if result.error is not None
            else None
        ),
    }

    try:
        content = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OllamaProtocolError("ToolResult is not JSON serializable") from exc

    return {
        "role": "tool",
        "tool_name": result.tool_name,
        "content": content,
    }


def build_ollama_tool_to_followup_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    assistant_turn: AssistantModelTurn,
    tool_results: Sequence[ToolResult],
) -> list[dict[str, object]]:
    expected_calls = assistant_turn.tool_calls

    if len(tool_results) != len(expected_calls):
        raise OllamaProtocolError(
            "number of tool results does not match the number of assistant tool calls"
        )

    result_messages = [dict(message) for message in messages]

    result_messages.append(dict(assistant_turn.history_message))

    for call, result in zip(
        expected_calls,
        tool_results,
        strict=True,
    ):
        if result.call_id != call.call_id:
            raise OllamaProtocolError("tool result call_id does not match the assistant tool call")

        if result.tool_name != call.name:
            raise OllamaProtocolError("tool result name does not match the assistant tool call")

        result_messages.append(tool_result_to_ollama_message(result))

    return result_messages
