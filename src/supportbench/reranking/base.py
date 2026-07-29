import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    doc_id: str
    text: str
    retrieval_score: float
    retrieval_rank: int

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("candidate doc_id must be non-empty")

        if not self.text.strip():
            raise ValueError("candidate text must be non-empty")

        if not math.isfinite(self.retrieval_score):
            raise ValueError("candidate retrieval score must be finite")

        if self.retrieval_rank <= 0:
            raise ValueError("candidate retrieval rank must be positive")


@dataclass(frozen=True, slots=True)
class RerankResult:
    doc_id: str
    score: float
    retrieval_score: float
    retrieval_rank: int

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("rerank result doc_id must be non-empty")

        if not math.isfinite(self.score):
            raise ValueError("reranker score must be finite")

        if not math.isfinite(self.retrieval_score):
            raise ValueError("retrieval score must be finite")

        if self.retrieval_rank <= 0:
            raise ValueError("retriaval rank must be positive")


class Reranker(Protocol):
    """The reranker receives text + original retrieval information.
    The original rank and score are useful for failure analysis."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> list[RerankResult]: ...
