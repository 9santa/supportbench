import re
from collections.abc import Sequence

from supportbench.rag.context_builder import (
    RepresentativeChunkContextBuilder,
)
from supportbench.rag.models import RetrievedChunk


class WhitespaceTokenCodec:
    def __init__(self) -> None:
        self._token_ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}
        self.encode_calls = 0

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        result: list[int] = []

        for token in re.findall(r"\S+", text):
            token_id = self._token_ids.setdefault(token, len(self._token_ids))
            self._tokens[token_id] = token
            result.append(token_id)

        return result

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self._tokens[token_id] for token_id in token_ids)


def _chunk(
    chunk_id: str,
    *,
    text: str,
    parent_id: str = "parent_a",
    parent_rank: int = 1,
    evidence_rank: int = 1,
    ordinal: int = 0,
    start_char: int | None = None,
    end_char: int | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_doc_id=parent_id,
        document_title=f"Title {parent_id}",
        text=text,
        category="support",
        section_path=("Troubleshooting", "Resolution"),
        ordinal=ordinal,
        start_char=start_char,
        end_char=end_char,
        parent_score=0.9,
        parent_rank=parent_rank,
        evidence_rank=evidence_rank,
    )


def test_exposes_local_source_id_and_preserves_real_provenance() -> None:
    context = RepresentativeChunkContextBuilder(
        token_codec=WhitespaceTokenCodec(),
        max_tokens=200,
    ).build([_chunk("chunk_a", text="Restart the service.")])

    assert [document.doc_id for document in context.documents] == ["parent_a"]
    assert "source_id: S1" in context.formatted_text
    assert "doc_id:" not in context.formatted_text
    assert "chunk_id:" not in context.formatted_text
    assert "section: Troubleshooting > Resolution" in context.formatted_text
    assert context.provenance[0].source_id == "S1"
    assert context.provenance[0].parent_doc_id == "parent_a"
    assert context.provenance[0].chunk_id == "chunk_a"
    assert context.provenance[0].section_path == ("Troubleshooting", "Resolution")


def test_removes_character_offset_overlap() -> None:
    context = RepresentativeChunkContextBuilder(
        token_codec=WhitespaceTokenCodec(),
        max_tokens=200,
    ).build(
        [
            _chunk(
                "chunk_1",
                text="alpha beta gamma",
                evidence_rank=1,
                ordinal=0,
                start_char=0,
                end_char=16,
            ),
            _chunk(
                "chunk_2",
                text="gamma delta",
                evidence_rank=2,
                ordinal=1,
                start_char=11,
                end_char=22,
            ),
        ]
    )

    assert context.formatted_text.count("gamma") == 1
    assert "delta" in context.formatted_text
    assert "source_span" not in context.formatted_text
    assert "included_span" not in context.formatted_text
    assert context.provenance[1].source_start_char == 11
    assert context.provenance[1].source_end_char == 22
    assert context.provenance[1].included_start_char == 17
    assert context.provenance[1].included_end_char == 22
    assert [item.source_id for item in context.provenance] == ["S1", "S2"]


def test_removes_token_overlap_without_character_offsets() -> None:
    context = RepresentativeChunkContextBuilder(
        token_codec=WhitespaceTokenCodec(),
        max_tokens=200,
        minimum_token_overlap=2,
    ).build(
        [
            _chunk("chunk_1", text="alpha beta gamma delta", evidence_rank=1, ordinal=0),
            _chunk("chunk_2", text="gamma delta epsilon zeta", evidence_rank=2, ordinal=1),
        ]
    )

    assert context.formatted_text.count("gamma delta") == 1
    assert context.provenance[1].removed_prefix_tokens == 2


def test_context_never_exceeds_token_budget() -> None:
    codec = WhitespaceTokenCodec()
    text = " ".join(f"token_{i}" for i in range(100))
    context = RepresentativeChunkContextBuilder(
        token_codec=codec,
        max_tokens=200,
    ).build(
        [_chunk("chunk_a", text=text, start_char=0, end_char=len(text))],
        max_tokens=26,
    )

    assert context.documents
    assert context.token_count == len(codec.encode(context.formatted_text))
    assert context.token_count <= 26
    assert context.truncated is True
    assert context.provenance[0].truncated is True
    assert context.provenance[0].source_start_char == 0
    assert context.provenance[0].source_end_char == len(text)
    assert context.provenance[0].included_start_char == 0
    assert context.provenance[0].included_end_char is None
    assert "[TRUNCATED]" in context.formatted_text

def test_parent_limit_keeps_highest_ranked_parents() -> None:
    context = RepresentativeChunkContextBuilder(
        token_codec=WhitespaceTokenCodec(),
        max_tokens=200,
        max_parents=1,
    ).build(
        [
            _chunk("chunk_a", text="alpha"),
            _chunk(
                "chunk_b",
                text="beta",
                parent_id="parent_b",
                parent_rank=2,
            ),
        ]
    )

    assert [document.doc_id for document in context.documents] == ["parent_a"]
    assert context.truncated is True
