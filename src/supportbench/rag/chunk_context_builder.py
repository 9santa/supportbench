import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from supportbench.chunking.base import TokenCodec
from supportbench.rag.models import (
    ChunkProvenance,
    RAGContext,
    RetrievedChunk,
    RetrievedDocument,
)

DOCUMENT_SEPARATOR = "\n\n"
TRUNCATION_MARKER = "\n[TRUNCATED]"


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    source: RetrievedChunk
    text: str
    token_ids: tuple[int, ...]
    included_start_char: int | None
    included_end_char: int | None
    removed_prefix_tokens: int
    truncated: bool = False


class RepresentativeChunkContextBuilder:
    def __init__(
        self,
        *,
        token_codec: TokenCodec,
        max_tokens: int,
        max_parents: int = 5,
        minimum_token_overlap: int = 8,
        maximum_token_overlap: int = 256,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if max_parents <= 0:
            raise ValueError("max_parents must be positive")

        if minimum_token_overlap <= 0:
            raise ValueError("minimum_token_overlap must be positive")

        if maximum_token_overlap < minimum_token_overlap:
            raise ValueError("maximum_token_overlap must be at least minimum_token_overlap")

        self._token_codec = token_codec
        self._max_tokens = max_tokens
        self._max_parents = max_parents
        self._minimum_token_overlap = minimum_token_overlap
        self._maximum_token_overlap = maximum_token_overlap

    def build(self, chunks: Sequence[RetrievedChunk]) -> RAGContext:
        chunk_items = tuple(chunks)

        if not chunk_items:
            return RAGContext(
                documents=(),
                formatted_text="",
                truncated=False,
                token_count=0,
                provenance=(),
            )

        parent_groups = self._validate_and_group(chunk_items)
        selected_groups = parent_groups[: self._max_parents]
        prepared = [
            chunk for group in selected_groups for chunk in self._deduplicate_parent_chunks(group)
        ]
        packed: list[_PreparedChunk] = []
        budget_truncated = False

        for chunk in prepared:
            candidate = [*packed, chunk]

            if self._token_count(_format_context(candidate)) <= self._max_tokens:
                packed.append(chunk)
                continue

            truncated_chunk = self._truncate_chunk_to_budget(packed, chunk)

            if truncated_chunk is not None:
                packed.append(truncated_chunk)

            budget_truncated = True
            break

        formatted_text = _format_context(packed)
        token_count = self._token_count(formatted_text)

        if token_count > self._max_tokens:
            raise RuntimeError("context builder exceeded the token budget")

        return RAGContext(
            documents=_build_documents(packed),
            formatted_text=formatted_text,
            truncated=(
                budget_truncated
                or len(selected_groups) < len(parent_groups)
                or len(packed) < len(prepared)
            ),
            token_count=token_count,
            provenance=tuple(_build_provenance(chunk) for chunk in packed),
        )

    def _validate_and_group(
        self,
        chunks: tuple[RetrievedChunk, ...],
    ) -> list[tuple[RetrievedChunk, ...]]:
        groups: list[list[RetrievedChunk]] = []
        seen_chunk_ids: set[str] = set()
        seen_parent_ids: set[str] = set()

        for chunk in chunks:
            self._validate_chunk(chunk)

            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(f"duplicate retrieved chunk ID: {chunk.chunk_id!r}")

            seen_chunk_ids.add(chunk.chunk_id)

            if not groups or groups[-1][0].parent_doc_id != chunk.parent_doc_id:
                expected_parent_rank = len(groups) + 1

                if chunk.parent_doc_id in seen_parent_ids:
                    raise ValueError(
                        f"chunks for parent {chunk.parent_doc_id!r} must be contiguous"
                    )

                if chunk.parent_rank != expected_parent_rank:
                    raise ValueError(
                        "parent ranks must be consecutive starting at 1; "
                        f"expected {expected_parent_rank}, received {chunk.parent_rank}"
                    )

                seen_parent_ids.add(chunk.parent_doc_id)
                groups.append([chunk])
                continue

            first = groups[-1][0]

            if chunk.parent_rank != first.parent_rank:
                raise ValueError("chunks from one parent must have the same parent rank")

            if (
                chunk.document_title != first.document_title
                or chunk.category != first.category
                or chunk.parent_score != first.parent_score
            ):
                raise ValueError("chunks from one parent have inconsistent metadata")

            groups[-1].append(chunk)

        frozen_groups: list[tuple[RetrievedChunk, ...]] = []

        for group in groups:
            evidence_ranks = sorted(chunk.evidence_rank for chunk in group)

            if evidence_ranks != list(range(1, len(group) + 1)):
                raise ValueError(
                    "evidence ranks must be consecutive starting at 1 within each parent"
                )

            frozen_groups.append(tuple(group))

        return frozen_groups

    @staticmethod
    def _validate_chunk(chunk: RetrievedChunk) -> None:
        string_fields = (
            chunk.chunk_id,
            chunk.parent_doc_id,
            chunk.document_title,
            chunk.text,
            chunk.category,
        )

        if any(not value.strip() for value in string_fields):
            raise ValueError("retrieved chunk string fields must be non-empty")

        if chunk.ordinal < 0:
            raise ValueError("chunk ordinal must be non-negative")

        if chunk.parent_rank <= 0 or chunk.evidence_rank <= 0:
            raise ValueError("parent and evidence ranks must be positive")

        if not math.isfinite(chunk.parent_score):
            raise ValueError("parent score must be finite")

        if (chunk.start_char is None) != (chunk.end_char is None):
            raise ValueError("chunk offsets must either both be set or both be absent")

        if chunk.start_char is not None and (
            chunk.start_char < 0 or chunk.end_char is None or chunk.end_char <= chunk.start_char
        ):
            raise ValueError("chunk offsets are invalid")

    def _deduplicate_parent_chunks(
        self,
        chunks: tuple[RetrievedChunk, ...],
    ) -> list[_PreparedChunk]:
        ordered = sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.evidence_rank))
        prepared: list[_PreparedChunk] = []
        previous_source_tokens: tuple[int, ...] = ()
        covered_end_char: int | None = None

        for chunk in ordered:
            text = chunk.text
            source_tokens = tuple(self._token_codec.encode(text))
            included_start_char = chunk.start_char
            included_end_char = chunk.end_char
            removed_prefix_tokens = 0

            if (
                chunk.start_char is not None
                and chunk.end_char is not None
                and covered_end_char is not None
            ):
                if chunk.end_char <= covered_end_char:
                    previous_source_tokens = source_tokens
                    continue

                if chunk.start_char < covered_end_char:
                    removed_characters = covered_end_char - chunk.start_char
                    removed_text = text[:removed_characters]
                    remaining_text = text[removed_characters:]
                    text = remaining_text.lstrip()
                    removed_whitespace = len(remaining_text) - len(text)
                    removed_prefix_tokens = len(self._token_codec.encode(removed_text))
                    included_start_char = covered_end_char + removed_whitespace

            elif previous_source_tokens:
                overlap = _longest_token_overlap(
                    previous_source_tokens,
                    source_tokens,
                    minimum=self._minimum_token_overlap,
                    maximum=self._maximum_token_overlap,
                )

                if overlap:
                    text = self._decode(source_tokens[overlap:]).lstrip()
                    removed_prefix_tokens = overlap

            if chunk.end_char is not None:
                covered_end_char = max(covered_end_char or 0, chunk.end_char)

            previous_source_tokens = source_tokens

            if not text.strip():
                continue

            token_ids = tuple(self._token_codec.encode(text))

            if not token_ids:
                continue

            prepared.append(
                _PreparedChunk(
                    source=chunk,
                    text=text,
                    token_ids=token_ids,
                    included_start_char=included_start_char,
                    included_end_char=included_end_char,
                    removed_prefix_tokens=removed_prefix_tokens,
                )
            )

        return prepared

    def _truncate_chunk_to_budget(
        self,
        packed: list[_PreparedChunk],
        chunk: _PreparedChunk,
    ) -> _PreparedChunk | None:
        low = 1
        high = len(chunk.token_ids)
        best: _PreparedChunk | None = None

        while low <= high:
            mid = low + (high - low) // 2
            token_ids = chunk.token_ids[:mid]
            text = self._decode(token_ids).strip()

            if not text:
                low = mid + 1
                continue

            candidate = replace(
                chunk,
                text=text,
                token_ids=token_ids,
                included_end_char=None,
                truncated=True,
            )

            if self._token_count(_format_context([*packed, candidate])) <= self._max_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _token_count(self, text: str) -> int:
        return len(self._token_codec.encode(text)) if text else 0

    def _decode(self, token_ids: Sequence[int]) -> str:
        decoded = self._token_codec.decode(token_ids)

        if not isinstance(decoded, str):
            raise TypeError("token codec must decode token IDs to a string")

        return decoded


def _longest_token_overlap(
    previous: tuple[int, ...],
    current: tuple[int, ...],
    *,
    minimum: int,
    maximum: int,
) -> int:
    upper = min(len(previous), len(current), maximum)

    for size in range(upper, minimum - 1, -1):
        if previous[-size:] == current[:size]:
            return size

    return 0


def _format_context(chunks: Sequence[_PreparedChunk]) -> str:
    if not chunks:
        return ""

    groups: list[list[_PreparedChunk]] = []

    for chunk in chunks:
        if not groups or groups[-1][0].source.parent_doc_id != chunk.source.parent_doc_id:
            groups.append([chunk])
        else:
            groups[-1].append(chunk)

    return DOCUMENT_SEPARATOR.join(_format_parent(group) for group in groups)


def _format_parent(chunks: list[_PreparedChunk]) -> str:
    first = chunks[0].source
    source_blocks = "\n\n".join(_format_chunk(chunk) for chunk in chunks)
    return (
        "[DOCUMENT]\n"
        f"doc_id: {first.parent_doc_id}\n"
        f"title: {first.document_title}\n"
        f"category: {first.category}\n"
        f"{source_blocks}\n"
        "[/DOCUMENT]"
    )


def _format_chunk(chunk: _PreparedChunk) -> str:
    source = chunk.source
    section = " > ".join(source.section_path) if source.section_path else "<root>"
    offsets = (
        f"{source.start_char}:{source.end_char}"
        if source.start_char is not None and source.end_char is not None
        else "unknown"
    )
    marker = TRUNCATION_MARKER if chunk.truncated else ""
    return (
        "[CHUNK]\n"
        f"chunk_id: {source.chunk_id}\n"
        f"section: {section}\n"
        f"ordinal: {source.ordinal}\n"
        f"source_chars: {offsets}\n"
        "content:\n"
        f"{chunk.text}{marker}\n"
        "[/CHUNK]"
    )


def _build_documents(chunks: Sequence[_PreparedChunk]) -> tuple[RetrievedDocument, ...]:
    groups: list[list[_PreparedChunk]] = []

    for chunk in chunks:
        if not groups or groups[-1][0].source.parent_doc_id != chunk.source.parent_doc_id:
            groups.append([chunk])
        else:
            groups[-1].append(chunk)

    return tuple(
        RetrievedDocument(
            doc_id=group[0].source.parent_doc_id,
            title=group[0].source.document_title,
            text="\n\n".join(chunk.text for chunk in group),
            category=group[0].source.category,
            score=group[0].source.parent_score,
            rank=rank,
        )
        for rank, group in enumerate(groups, start=1)
    )


def _build_provenance(chunk: _PreparedChunk) -> ChunkProvenance:
    source = chunk.source
    return ChunkProvenance(
        parent_doc_id=source.parent_doc_id,
        chunk_id=source.chunk_id,
        parent_rank=source.parent_rank,
        evidence_rank=source.evidence_rank,
        document_title=source.document_title,
        section_path=source.section_path,
        ordinal=source.ordinal,
        source_start_char=source.start_char,
        source_end_char=source.end_char,
        included_start_char=chunk.included_start_char,
        included_end_char=chunk.included_end_char,
        removed_prefix_tokens=chunk.removed_prefix_tokens,
        included_tokens=len(chunk.token_ids),
        truncated=chunk.truncated,
    )
