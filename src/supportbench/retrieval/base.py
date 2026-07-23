from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: str
    score: float
    rank: int


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return documents ordered by descending relevance"""
        ...
