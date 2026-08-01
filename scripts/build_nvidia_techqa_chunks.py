import argparse
from dataclasses import dataclass
from pathlib import Path

from supportbench.chunking import (
    FixedTokenChunker,
    HuggingFaceTokenCodec,
    build_chunk_corpus,
)
from supportbench.data.loaders import (
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"

DEFAULT_TOKENIZER = "intfloat/multilingual-e5-base"


@dataclass(frozen=True, slots=True)
class FixedTokenConfig:
    chunk_size: int
    overlap: int


DEFAULT_CONFIGS = (
    FixedTokenConfig(
        chunk_size=256,
        overlap=32,
    ),
    FixedTokenConfig(
        chunk_size=384,
        overlap=64,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build fixed-token chunk corpora for NVIDIA TechQA.")
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--special-token-reserve",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="SIZE:OVERLAP",
        help=("Fixed-token configuration. May be repeated. Default: 256:32 and 384:64."),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configs = (
        tuple(
            _parse_config(
                value,
                parser=parser,
            )
            for value in args.config
        )
        if args.config
        else DEFAULT_CONFIGS
    )

    if len(configs) != len(set(configs)):
        parser.error("--config values must be unique")

    documents = load_documents(args.documents)

    if not documents:
        parser.error("documents corpus is empty")

    print(f"Loaded {len(documents):,} documents")
    print(f"Tokenizer: {args.tokenizer}")

    token_codec = HuggingFaceTokenCodec.from_pretrained(args.tokenizer)

    for config in configs:
        chunker = FixedTokenChunker(
            token_codec=token_codec,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )

        output_directory = args.output_root / chunker.chunking_key

        print()
        print(f"Building {chunker.chunking_key}...")

        result = build_chunk_corpus(
            documents=documents,
            chunker=chunker,
            token_codec=token_codec,
            tokenizer_name=args.tokenizer,
            source_documents_path=(args.documents),
            output_directory=(output_directory),
            max_input_tokens=(args.max_input_tokens),
            special_token_reserve=(args.special_token_reserve),
        )

        statistics = result.statistics

        print(f"Chunks: {statistics.total_chunks:,}")
        print(f"Mean chunks/document: {statistics.mean_chunks_per_document:.2f}")
        print(f"Median chunks/document: {statistics.median_chunks_per_document:.2f}")
        print(f"P95 chunks/document: {statistics.p95_chunks_per_document:.0f}")
        print(f"Mean body tokens/chunk: {statistics.mean_tokens_per_chunk:.2f}")
        print(f"P95 body tokens/chunk: {statistics.p95_tokens_per_chunk:.0f}")
        print(
            "Chunks under 50 tokens: "
            f"{statistics.chunks_under_50_tokens:,} "
            f"({statistics.chunks_under_50_tokens_rate:.2%})"
        )
        print(
            "Formatted chunks over budget: "
            f"{statistics.formatted_over_budget_chunks:,} "
            f"({statistics.formatted_over_budget_rate:.2%})"
        )
        print(f"Output: {output_directory}")


def _parse_config(
    value: str,
    *,
    parser: argparse.ArgumentParser,
) -> FixedTokenConfig:
    parts = value.split(":", maxsplit=1)

    if len(parts) != 2:
        parser.error(f"--config must use SIZE:OVERLAP format, got {value!r}")

    try:
        chunk_size = int(parts[0])
        overlap = int(parts[1])
    except ValueError:
        parser.error(f"--config size and overlap must be integers, got {value!r}")

    if chunk_size <= 0:
        parser.error("chunk size must be positive")

    if overlap < 0:
        parser.error("overlap must be non-negative")

    if overlap >= chunk_size:
        parser.error("overlap must be smaller than chunk size")

    return FixedTokenConfig(
        chunk_size=chunk_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    main()
