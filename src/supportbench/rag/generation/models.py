from dataclasses import dataclass
from typing import Literal

type AnswerDecision = Literal[
    "answer",
    "abstain",
    "clarify",
]

type ChatRole = Literal[
    "system",
    "user",
]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    decision: AnswerDecision
    answer: str
    citation_ids: tuple[str, ...]
