from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    doc_id: str
    title: str
    text: str
    category: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RAGContext:
    documents: tuple[RetrievedDocument, ...]
    formatted_text: str
    truncated: bool  # Flag in case the text length exceeds max context length
