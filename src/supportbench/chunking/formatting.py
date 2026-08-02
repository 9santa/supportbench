from collections.abc import Sequence

from supportbench.chunking.models import Chunk


def format_chunk_header(
    *,
    document_title: str,
    section_path: Sequence[str],
) -> str:
    parts = [f"Title: {document_title}"]

    if section_path:
        parts.append("Section: " + " > ".join(section_path))

    return "\n".join(parts)


def format_chunk_for_embedding(chunk: Chunk) -> str:
    metadata = [f"Title: {chunk.document_title}"]

    if chunk.section_path:
        metadata.append("Section: " + " > ".join(chunk.section_path))

    header = "\n".join(metadata)

    return f"{header}\n\n{chunk.text}"


def format_chunk_title_for_retrieval(chunk: Chunk) -> str:
    if not chunk.section_path:
        return chunk.document_title

    section = " > ".join(chunk.section_path)

    return f"{chunk.document_title}\nSection: {section}"


def format_chunk_for_display(chunk: Chunk) -> str:
    parts = [f"title: {chunk.document_title}"]

    if chunk.section_path:
        parts.append("section: " + " > ".join(chunk.section_path))

    parts.append(f"content:\n{chunk.text}")

    return "\n".join(parts)
