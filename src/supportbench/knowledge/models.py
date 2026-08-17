from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SupportEvidenceChunk:
    chunk_id: str
    section_path: tuple[str, ...]
    text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")

        if not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(frozen=True, slots=True)
class SupportDocumentMatch:
    document_id: str
    title: str
    rank: int
    evidence: tuple[SupportEvidenceChunk, ...]

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must be non-empty")

        if not self.title.strip():
            raise ValueError("title must be non-empty")

        if self.rank <= 0:
            raise ValueError("rank must be positive")

        if not self.evidence:
            raise ValueError("evidence must not be empty")


@dataclass(frozen=True, slots=True)
class SupportDocumentRead:
    document_id: str
    title: str
    chunks: tuple[SupportEvidenceChunk, ...]
    truncated: bool

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must be non-empty")

        if not self.title.strip():
            raise ValueError("title must be non-empty")

        if not self.chunks:
            raise ValueError("chunks must not be empty")
