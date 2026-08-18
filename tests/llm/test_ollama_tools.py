import json

import pytest

from supportbench.llm.errors import OllamaProtocolError
from supportbench.llm.ollama_tools import (
    parse_ollama_chat_response,
    tool_definitions_to_ollama,
    tool_result_to_ollama_message,
)
from supportbench.tools.definitions import (
    GET_SERVICE_STATUS,
)
from supportbench.tools.models import (
    ToolErrorInfo,
    ToolResult,
)


def test_tool_definition_is_converted_to_ollama_schema() -> None:
    tools = tool_definitions_to_ollama([GET_SERVICE_STATUS])

    assert len(tools) == 1

    tool = tools[0]

    assert tool["type"] == "function"

    function = tool["function"]

    assert function["name"] == "get_service_status"

    parameters = function["parameters"]

    assert parameters["type"] == "object"

    assert parameters["additionalProperties"] is False

    assert "service_id" in (parameters["properties"])

    assert "world_id" not in (parameters["properties"])


def test_parse_text_response_without_tools() -> None:
    turn = parse_ollama_chat_response(
        {
            "message": {
                "role": "assistant",
                "content": "The service is operational.",
            }
        },
        request_id="req-001",
        assistant_turn_index=0,
    )

    assert turn.content == "The service is operational."
    assert turn.tool_calls == ()

    assert turn.history_message == {
        "role": "assistant",
        "content": "The service is operational.",
    }


def test_parse_tool_call() -> None:
    response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_service_status",
                        "arguments": {
                            "service_id": ("webgui-noc-prod"),
                        },
                    }
                }
            ],
        }
    }

    turn = parse_ollama_chat_response(
        response,
        request_id="req-001",
        assistant_turn_index=0,
    )

    assert len(turn.tool_calls) == 1

    call = turn.tool_calls[0]

    assert call.name == "get_service_status"

    assert call.arguments == {
        "service_id": "webgui-noc-prod",
    }

    assert call.call_id.startswith("ollama-call:")


def test_same_tool_call_gets_same_call_id() -> None:
    response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_service_status",
                        "arguments": {
                            "service_id": "webgui-noc-prod",
                        },
                    }
                }
            ],
        }
    }

    first = parse_ollama_chat_response(
        response,
        request_id="req-001",
        assistant_turn_index=2,
    )

    second = parse_ollama_chat_response(
        response,
        request_id="req-001",
        assistant_turn_index=2,
    )

    assert first.tool_calls[0].call_id == second.tool_calls[0].call_id


def test_changed_arguments_change_call_id() -> None:
    first = parse_ollama_chat_response(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_service_status",
                            "arguments": {
                                "service_id": "service-a",
                            },
                        }
                    }
                ],
            }
        },
        request_id="req-001",
        assistant_turn_index=2,
    )

    second = parse_ollama_chat_response(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_service_status",
                            "arguments": {
                                "service_id": "service-b",
                            },
                        }
                    }
                ],
            }
        },
        request_id="req-001",
        assistant_turn_index=2,
    )

    assert first.tool_calls[0].call_id != second.tool_calls[0].call_id


def test_multiple_tool_calls_get_distinct_ids() -> None:
    turn = parse_ollama_chat_response(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_service_status",
                            "arguments": {
                                "service_id": "service-a",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "get_service_status",
                            "arguments": {
                                "service_id": "service-b",
                            },
                        }
                    },
                ],
            }
        },
        request_id="req-001",
        assistant_turn_index=0,
    )

    assert len(turn.tool_calls) == 2

    assert turn.tool_calls[0].call_id != turn.tool_calls[1].call_id


def test_success_tool_result_becomes_ollama_message() -> None:
    message = tool_result_to_ollama_message(
        ToolResult(
            call_id="tc-001",
            tool_name="get_service_status",
            status="success",
            data={
                "service_id": "webgui-noc-prod",
                "status": "operational",
            },
            error=None,
        )
    )

    assert message["role"] == "tool"
    assert message["tool_name"] == "get_service_status"

    payload = json.loads(message["content"])

    assert payload["call_id"] == "tc-001"
    assert payload["status"] == "success"
    assert payload["error"] is None


def test_error_tool_result_becomes_ollama_message() -> None:
    message = tool_result_to_ollama_message(
        ToolResult(
            call_id="tc-404",
            tool_name="get_service_status",
            status="error",
            data=None,
            error=ToolErrorInfo(
                code="service_not_found",
                message=("Service was not found."),
            ),
        )
    )

    payload = json.loads(message["content"])

    assert payload["call_id"] == "tc-404"
    assert payload["status"] == "error"

    assert payload["error"]["code"] == "service_not_found"


def test_non_object_tool_arguments_are_rejected() -> None:
    with pytest.raises(OllamaProtocolError):
        parse_ollama_chat_response(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": ("get_service_status"),
                                "arguments": ("not-an-object"),
                            }
                        }
                    ],
                }
            },
            request_id="req-001",
            assistant_turn_index=0,
        )


def test_qwen3_thinking_tool_call_response() -> None:
    response = {
        "message": {
            "role": "assistant",
            "thinking": ("I should inspect the installed product."),
            "content": "",
            "tool_calls": [
                {
                    "id": "call_j35synbb",
                    "function": {
                        "index": 0,
                        "name": ("get_installed_product"),
                        "arguments": {
                            "asset_id": ("dash-host-01"),
                            "product_key": "dash",
                        },
                    },
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "eval_count": 123,
    }

    turn = parse_ollama_chat_response(
        response,
        request_id="req-001",
        assistant_turn_index=0,
    )

    assert turn.content == ""

    assert len(turn.tool_calls) == 1

    call = turn.tool_calls[0]

    assert call.name == "get_installed_product"

    assert call.arguments == {
        "asset_id": "dash-host-01",
        "product_key": "dash",
    }

    assert turn.finish_reason == "stop"
    assert turn.output_token_count == 123

    assert turn.history_message["thinking"] == ("I should inspect the installed product.")
