from dataclasses import dataclass
from typing import Literal

from supportbench.data.models import DatasetSplit

type Answerability = Literal[
    "answerable",
    "unanswerable",
]


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """Query metadata for retrieval and generation detailed evalution."""

    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    split: DatasetSplit
    answerability: Answerability
    reference_answer: str | None

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must be non-empty")

        if not self.query.strip():
            raise ValueError("query must be non-empty")

        if len(self.relevant_doc_ids) != len(set(self.relevant_doc_ids)):
            raise ValueError("relevant_doc_ids must not contain duplicates")

        if any(not doc_id.strip() for doc_id in self.relevant_doc_ids):
            raise ValueError("relevant_doc_ids must contain only non-empty strings")

        if self.answerability == "answerable":
            if not self.relevant_doc_ids:
                raise ValueError("answerable query must have at least one relevant document")
            return

        if self.relevant_doc_ids:
            raise ValueError("unanswerable query must not have relevant documents")

        if self.reference_answer is not None:
            raise ValueError("unanswerable query must not have a reference answer")
