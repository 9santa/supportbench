from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    ordinal: int
    token_count: int
    section_path: tuple[str, ...]
    start_char: int | None
    end_char: int | None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")

        if not self.document_id.strip():
            raise ValueError("document_id must be non-empty")

        if not self.document_title.strip():
            raise ValueError("document_title must be non-empty")

        if not self.text.strip():
            raise ValueError("chunk text must be non-empty")

        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")

        if self.token_count <= 0:
            raise ValueError("token_count must be positive")

        if any(not section.strip() for section in self.section_path):
            raise ValueError("section_path must contain only non-empty strings")

        offsets = (self.start_char, self.end_char)

        if offsets == (None, None):
            return

        if self.start_char is None or self.end_char is None:
            raise ValueError("start_char and end_char must either both be None or both be set")

        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")

        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
