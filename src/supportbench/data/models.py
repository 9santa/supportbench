from dataclasses import dataclass
from typing import Literal

DatasetSplit = Literal["train", "dev", "test", "frozen_test"]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    category: str


@dataclass(frozen=True, slots=True)
class QueryExample:
    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]
    split: DatasetSplit
