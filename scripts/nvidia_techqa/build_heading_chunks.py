import argparse
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.chunking import (
    HeadingAwareChunker,
    HuggingFaceTokenCodec,
    build_chunk_corpus,
)
from supportbench.data.loaders import load_documents

DEFAULT_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"
DEFAULT_TOKENIZER = "intfloat/multilingual-e5-base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a heading-aware chunk corpus for NVIDIA TechQA."
    )
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=384)
    parser.add_argument("--oversized-overlap", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--special-token-reserve", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    if args.oversized_overlap < 0:
        parser.error("--oversized-overlap must be non-negative")
    if args.oversized_overlap >= args.target_tokens:
        parser.error("--oversized-overlap must be smaller than --target-tokens")
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive")
    if args.special_token_reserve < 0:
        parser.error("--special-token-reserve must be non-negative")
    if args.special_token_reserve >= args.max_input_tokens:
        parser.error("--special-token-reserve must be smaller than --max-input-tokens")

    documents = load_documents(args.documents)
    if not documents:
        parser.error("documents corpus is empty")

    print(f"Loaded {len(documents):,} documents")
    print(f"Tokenizer: {args.tokenizer}")

    token_codec = HuggingFaceTokenCodec.from_pretrained(args.tokenizer)
    chunker = HeadingAwareChunker(
        token_codec=token_codec,
        target_tokens=args.target_tokens,
        oversized_overlap=args.oversized_overlap,
        max_input_tokens=args.max_input_tokens,
        special_token_reserve=args.special_token_reserve,
    )
    output_directory = args.output_root / chunker.chunking_key

    print(f"Building {chunker.chunking_key}...")
    result = build_chunk_corpus(
        documents=documents,
        chunker=chunker,
        token_codec=token_codec,
        tokenizer_name=args.tokenizer,
        source_documents_path=args.documents,
        output_directory=output_directory,
        max_input_tokens=args.max_input_tokens,
        special_token_reserve=args.special_token_reserve,
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
