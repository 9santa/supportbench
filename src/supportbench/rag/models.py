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
class RetrievedChunk:
    chunk_id: str
    parent_doc_id: str
    document_title: str
    text: str
    category: str
    section_path: tuple[str, ...]
    ordinal: int
    start_char: int | None
    end_char: int | None
    parent_score: float
    parent_rank: int
    evidence_rank: int


@dataclass(frozen=True, slots=True)
class ChunkProvenance:
    parent_doc_id: str
    chunk_id: str
    parent_rank: int
    evidence_rank: int
    document_title: str
    section_path: tuple[str, ...]
    ordinal: int
    source_start_char: int | None
    source_end_char: int | None
    included_start_char: int | None
    included_end_char: int | None
    removed_prefix_tokens: int
    included_tokens: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class RAGContext:
    documents: tuple[RetrievedDocument, ...]
    formatted_text: str
    truncated: bool  # Flag in case the text length exceeds max context length
    token_count: int = 0
    provenance: tuple[ChunkProvenance, ...] = ()
