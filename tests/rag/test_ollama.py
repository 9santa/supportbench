import io
import json
from unittest.mock import patch

from supportbench.rag.generation.models import ChatMessage
from supportbench.rag.generation.ollama import OllamaLLMClient


def test_sends_context_window_and_output_limit() -> None:
    response = io.BytesIO(
        json.dumps(
            {
                "message": {
                    "content": (
                        '{"decision":"abstain","answer":"No context",'
                        '"citation_ids":[]}'
                    )
                }
            }
        ).encode()
    )

    with patch("supportbench.rag.generation.ollama.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = response
        OllamaLLMClient(
            model_name="gemma3:4b",
            context_window=8_192,
            max_output_tokens=512,
        ).generate((ChatMessage(role="user", content="question"),))

    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)

    assert payload["options"]["num_ctx"] == 8_192
    assert payload["options"]["num_predict"] == 512
