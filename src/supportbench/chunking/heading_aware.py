import re
from collections.abc import Mapping
from dataclasses import dataclass

from supportbench.chunking.base import (
    OffsetTokenCodec,
    build_chunk_id,
)
from supportbench.chunking.formatting import (
    format_chunk_header,
)
from supportbench.chunking.models import Chunk
from supportbench.data.models import Document

_MARKDOWN_HEADING = re.compile(
    r"^(?P<marks>#{1,6})\s+"
    r"(?P<title>.+?)"
    r"\s*#*\s*$"
)

_HIERARCHICAL_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+)+)"
    r"(?:[.)])?\s+"
    r"(?P<title>\S.*)$"
)

_WORD_PATTERN = re.compile(r"\S+")

_KNOWN_HEADINGS = {
    "abstract",
    "additional information",
    "apars",
    "cause",
    "causes",
    "comments",
    "diagnosing the problem",
    "diagnosis",
    "document information",
    "environment",
    "error description",
    "example",
    "examples",
    "installation instructions",
    "local fix",
    "note",
    "notes",
    "prerequisites",
    "problem",
    "problem abstract",
    "problem conclusion",
    "problem summary",
    "procedure",
    "related information",
    "resolution",
    "resolving the problem",
    "solution",
    "steps",
    "symptom",
    "symptoms",
    "technical details",
    "temporary fix",
    "troubleshooting",
    "workaround",
}


@dataclass(frozen=True, slots=True)
class _SourceBlock:
    text: str
    start_char: int
    end_char: int
    heading_level: int | None


@dataclass(frozen=True, slots=True)
class _Paragraph:
    start_char: int
    end_char: int
    section_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PackedSpan:
    start_char: int
    end_char: int
    section_path: tuple[str, ...]


class HeadingAwareChunker:
    def __init__(
        self,
        *,
        token_codec: OffsetTokenCodec,
        target_tokens: int = 384,
        oversized_overlap: int = 64,
        max_input_tokens: int = 512,
        special_token_reserve: int = 2,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")

        if oversized_overlap < 0:
            raise ValueError("oversized_overlap must be non-negative")

        if oversized_overlap >= target_tokens:
            raise ValueError("oversized_overlap must be smaller than target_tokens")

        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")

        if special_token_reserve < 0:
            raise ValueError("special_token_reserve must be non-negative")

        if special_token_reserve >= max_input_tokens:
            raise ValueError("special_token_reserve must be smaller than max_input_tokens")

        self._token_codec = token_codec
        self._target_tokens = target_tokens
        self._oversized_overlap = oversized_overlap
        self._max_input_tokens = max_input_tokens
        self._special_token_reserve = special_token_reserve

    @property
    def chunking_key(self) -> str:
        return (
            f"ha{self._target_tokens}"
            f"o{self._oversized_overlap}"
            f"m{self._max_input_tokens}"
            f"r{self._special_token_reserve}"
            "v2"
        )

    @property
    def configuration(
        self,
    ) -> Mapping[str, object]:
        return {
            "strategy": ("heading_paragraph_aware"),
            "version": 2,
            "target_tokens": (self._target_tokens),
            "oversized_overlap": (self._oversized_overlap),
            "max_input_tokens": (self._max_input_tokens),
            "special_token_reserve": (self._special_token_reserve),
            "heading_detection": ("conservative_rules_v2"),
            "packing": ("whole_paragraphs_within_section"),
        }

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        paragraphs = _extract_paragraphs(document.text)

        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        ordinal = 0
        packed: _PackedSpan | None = None

        def emit(
            span: _PackedSpan,
        ) -> None:
            nonlocal ordinal

            start_char, end_char = _trim_span(
                document.text,
                span.start_char,
                span.end_char,
            )

            if end_char <= start_char:
                return

            chunk_text = document.text[start_char:end_char]

            token_count = len(self._token_codec.encode(chunk_text))

            if token_count <= 0:
                return

            chunks.append(
                Chunk(
                    chunk_id=build_chunk_id(
                        document_id=(document.doc_id),
                        chunking_key=(self.chunking_key),
                        ordinal=ordinal,
                    ),
                    document_id=document.doc_id,
                    document_title=(document.title),
                    text=chunk_text,
                    ordinal=ordinal,
                    token_count=token_count,
                    section_path=(span.section_path),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

            ordinal += 1

        for paragraph in paragraphs:
            body_budget = self._body_budget(
                document_title=document.title,
                section_path=(paragraph.section_path),
            )

            paragraph_text = document.text[paragraph.start_char : paragraph.end_char]

            paragraph_tokens = len(self._token_codec.encode(paragraph_text))

            if paragraph_tokens > body_budget:
                if packed is not None:
                    emit(packed)
                    packed = None

                oversized_spans = self._split_oversized_paragraph(
                    source_text=document.text,
                    paragraph=paragraph,
                    body_budget=body_budget,
                )

                for oversized_span in oversized_spans:
                    emit(oversized_span)

                continue

            if packed is None:
                packed = _PackedSpan(
                    start_char=(paragraph.start_char),
                    end_char=(paragraph.end_char),
                    section_path=(paragraph.section_path),
                )
                continue

            if packed.section_path != paragraph.section_path:
                emit(packed)

                packed = _PackedSpan(
                    start_char=(paragraph.start_char),
                    end_char=(paragraph.end_char),
                    section_path=(paragraph.section_path),
                )
                continue

            combined_text = document.text[packed.start_char : paragraph.end_char]

            combined_token_count = len(self._token_codec.encode(combined_text))

            if combined_token_count <= body_budget:
                packed = _PackedSpan(
                    start_char=packed.start_char,
                    end_char=(paragraph.end_char),
                    section_path=(packed.section_path),
                )
            else:
                emit(packed)

                packed = _PackedSpan(
                    start_char=(paragraph.start_char),
                    end_char=(paragraph.end_char),
                    section_path=(paragraph.section_path),
                )

        if packed is not None:
            emit(packed)

        return chunks

    def _body_budget(
        self,
        *,
        document_title: str,
        section_path: tuple[str, ...],
    ) -> int:
        header = format_chunk_header(
            document_title=document_title,
            section_path=section_path,
        )

        header_tokens = len(self._token_codec.encode(f"{header}\n\n"))

        available_tokens = self._max_input_tokens - self._special_token_reserve - header_tokens

        if available_tokens <= 0:
            raise ValueError("document title and section path consume the full encoder budget")

        return min(
            self._target_tokens,
            available_tokens,
        )

    def _split_oversized_paragraph(
        self,
        *,
        source_text: str,
        paragraph: _Paragraph,
        body_budget: int,
    ) -> list[_PackedSpan]:
        paragraph_text = source_text[paragraph.start_char : paragraph.end_char]

        token_offsets = self._token_codec.encode_with_offsets(paragraph_text)

        if not token_offsets:
            return []

        overlap = min(
            self._oversized_overlap,
            body_budget - 1,
        )
        stride = body_budget - overlap

        spans: list[_PackedSpan] = []
        token_start = 0

        while token_start < len(token_offsets):
            token_end = min(
                token_start + body_budget,
                len(token_offsets),
            )

            window = token_offsets[token_start:token_end]

            usable_offsets = [token for token in window if token.end_char > token.start_char]

            if not usable_offsets:
                raise ValueError("tokenizer returned a token window without usable offsets")

            local_start = usable_offsets[0].start_char
            local_end = usable_offsets[-1].end_char

            global_start = paragraph.start_char + local_start
            global_end = paragraph.start_char + local_end

            global_start, global_end = _trim_span(
                source_text,
                global_start,
                global_end,
            )

            if global_end > global_start:
                spans.append(
                    _PackedSpan(
                        start_char=global_start,
                        end_char=global_end,
                        section_path=(paragraph.section_path),
                    )
                )

            if token_end == len(token_offsets):
                break

            token_start += stride

        return spans


def _extract_paragraphs(
    source_text: str,
) -> list[_Paragraph]:
    """Scans source text line by line to build blocks: heading blocks or paragraph blocks."""
    blocks = _scan_source_blocks(source_text)

    section_stack: list[tuple[int, str]] = []

    paragraphs: list[_Paragraph] = []

    for block in blocks:
        if block.heading_level is not None:
            while section_stack and section_stack[-1][0] >= block.heading_level:
                section_stack.pop()

            section_stack.append(
                (
                    block.heading_level,
                    block.text,
                )
            )
            continue

        paragraphs.append(
            _Paragraph(
                start_char=block.start_char,
                end_char=block.end_char,
                section_path=tuple(title for _, title in section_stack),
            )
        )

    return paragraphs


def _scan_source_blocks(
    source_text: str,
) -> list[_SourceBlock]:
    """
    Splits text into lines, flushes paragraphs on blank lines.

    For each line, if it's a heading, adds a heading block;
    else accumulates lines into a paragraph block (stripped leading/trailing whitespace,
    preserving indentation offset).
    """
    blocks: list[_SourceBlock] = []

    paragraph_start: int | None = None
    paragraph_end: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_start
        nonlocal paragraph_end

        if paragraph_start is not None and paragraph_end is not None:
            start_char, end_char = _trim_span(
                source_text,
                paragraph_start,
                paragraph_end,
            )

            if end_char > start_char:
                blocks.append(
                    _SourceBlock(
                        text=source_text[start_char:end_char],
                        start_char=start_char,
                        end_char=end_char,
                        heading_level=None,
                    )
                )

        paragraph_start = None
        paragraph_end = None

    cursor = 0

    for raw_line in source_text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw_line)

        content = raw_line.rstrip("\r\n")

        stripped = content.strip()

        if not stripped:
            flush_paragraph()
            continue

        left_padding = len(content) - len(content.lstrip())

        right_trimmed = content.rstrip()

        line_content_start = line_start + left_padding
        line_content_end = line_start + len(right_trimmed)

        heading = _detect_heading(stripped)

        if heading is not None:
            flush_paragraph()

            heading_title, heading_level = heading

            blocks.append(
                _SourceBlock(
                    text=heading_title,
                    start_char=(line_content_start),
                    end_char=line_content_end,
                    heading_level=(heading_level),
                )
            )
            continue

        if paragraph_start is None:
            paragraph_start = line_content_start

        paragraph_end = line_content_end

    # splitlines() covers the final line even
    # when the document has no trailing newline.
    flush_paragraph()

    return blocks


def _build_heading(
    title: str,
    level: int,
) -> tuple[str, int] | None:
    normalized_title = title.strip().rstrip(":").strip()

    if not normalized_title:
        return None

    # Reject separators and punctuation-only lines:
    # ":", "---", "====", "***".
    if not any(character.isalnum() for character in normalized_title):
        return None

    return normalized_title, level


def _detect_heading(
    text: str,
) -> tuple[str, int] | None:
    normalized_text = text.strip()

    if not normalized_text:
        return None

    if len(normalized_text) > 120:
        return None

    words = _WORD_PATTERN.findall(normalized_text)

    if len(words) > 14:
        return None

    markdown_match = _MARKDOWN_HEADING.fullmatch(normalized_text)

    if markdown_match is not None:
        return _build_heading(
            markdown_match.group("title"),
            len(markdown_match.group("marks")),
        )

    normalized_label = _normalize_heading(normalized_text.rstrip(":"))

    if normalized_label in _KNOWN_HEADINGS:
        return _build_heading(
            normalized_text,
            1,
        )

    hierarchical_match = _HIERARCHICAL_HEADING.fullmatch(normalized_text)

    if hierarchical_match is not None:
        title = hierarchical_match.group("title").strip()

        # Sentence-like lines are probably content,
        # not section headings.
        if title.endswith((".", "?", "!")):
            return None

        if len(_WORD_PATTERN.findall(title)) > 10:
            return None

        depth = len(hierarchical_match.group("number").split("."))

        return _build_heading(
            title,
            min(depth, 6),
        )

    return None


def _normalize_heading(text: str) -> str:
    """Lowercase, replace non-alphanumeric with spaces, strip."""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.casefold(),
    ).strip()


def _trim_span(
    source_text: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    """Trim whitespace from start and end."""
    while start_char < end_char and source_text[start_char].isspace():
        start_char += 1

    while end_char > start_char and source_text[end_char - 1].isspace():
        end_char -= 1

    return start_char, end_char
