from dataclasses import dataclass
from typing import Literal, TypeAlias


AnswerDecision: TypeAlias = Literal[
    "answer",
    "abstain",
    "clarify",
]

ChatRole: TypeAlias = Literal[
    "system",
    "user",
]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class GeneratedAnser:
    decision: AnswerDecision
    answer: str
    citation_ids: tuple[str, ...]
