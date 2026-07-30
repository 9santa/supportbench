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
class GeneratedAnswer:
    decision: AnswerDecision
    answer: str
    citation_ids: tuple[str, ...]
