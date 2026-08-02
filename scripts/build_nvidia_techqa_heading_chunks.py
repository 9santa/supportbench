from pathlib import Path

from supportbench.chunking import (
    HeadingAwareChunker,
    HuggingFaceTokenCodec,
    build_chunk_corpus,
)
from supportbench.data.loaders import (
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"

TOKENIZER_NAME = "intfloat/multilingual-e5-base"


def main() -> None:
    documents = load_documents(DOCUMENTS_PATH)

    print(f"Loaded {len(documents):,} documents")

    token_codec = HuggingFaceTokenCodec.from_pretrained(TOKENIZER_NAME)

    chunker = HeadingAwareChunker(
        token_codec=token_codec,
        target_tokens=384,
        oversized_overlap=64,
        max_input_tokens=512,
        special_token_reserve=2,
    )

    output_directory = OUTPUT_ROOT / chunker.chunking_key

    print(f"Building {chunker.chunking_key}...")

    result = build_chunk_corpus(
        documents=documents,
        chunker=chunker,
        token_codec=token_codec,
        tokenizer_name=TOKENIZER_NAME,
        source_documents_path=(DOCUMENTS_PATH),
        output_directory=(output_directory),
        max_input_tokens=512,
        special_token_reserve=2,
    )

    stats = result.statistics

    print(f"Chunks: {stats.total_chunks:,}")
    print(f"Mean chunks/document: {stats.mean_chunks_per_document:.2f}")
    print(f"Median chunks/document: {stats.median_chunks_per_document:.2f}")
    print(f"P95 chunks/document: {stats.p95_chunks_per_document:.0f}")
    print(f"Mean body tokens/chunk: {stats.mean_tokens_per_chunk:.2f}")
    print(f"P95 body tokens/chunk: {stats.p95_tokens_per_chunk:.0f}")
    print(
        "Chunks under 50 tokens: "
        f"{stats.chunks_under_50_tokens:,} "
        f"({stats.chunks_under_50_tokens_rate:.2%})"
    )
    print(
        "Chunks with section path: "
        f"{stats.chunks_with_section_path:,} "
        f"({stats.chunks_with_section_path_rate:.2%})"
    )
    print(f"Unique section paths: {stats.unique_section_paths:,}")
    print(f"Maximum section depth: {stats.max_section_depth}")
    print(
        "Formatted chunks over budget: "
        f"{stats.formatted_over_budget_chunks:,} "
        f"({stats.formatted_over_budget_rate:.2%})"
    )
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
