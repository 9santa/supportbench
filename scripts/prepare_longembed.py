import argparse
import json
from dataclasses import asdict
from pathlib import Path

from supportbench.benchmark.longembed import (
    LONGEMBED_TASKS,
    download_longembed_task,
    load_longembed_task,
)
from supportbench.chunking.base import HuggingFaceTokenCodec
from supportbench.chunking.build import build_chunk_corpus
from supportbench.chunking.fixed_token import FixedTokenChunker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = "intfloat/multilingual-e5-base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a pinned LongEmbed task and build fixed-token chunks."
    )
    parser.add_argument("--task", choices=tuple(LONGEMBED_TASKS), default="2wikimqa")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "longembed",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    if args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("--overlap must be non-negative and smaller than --chunk-size")

    spec = LONGEMBED_TASKS[args.task]
    raw_directory = download_longembed_task(spec, output_root=args.data_root)
    dataset = load_longembed_task(raw_directory, name=spec.name)
    chunking_key = f"ft{args.chunk_size}o{args.overlap}"
    output_directory = raw_directory / "chunks" / chunking_key
    expected_outputs = (
        output_directory / "chunks.jsonl",
        output_directory / "documents.jsonl",
        output_directory / "statistics.json",
        output_directory / "manifest.json",
    )

    if all(path.is_file() for path in expected_outputs):
        print(f"Reusing prepared chunks: {output_directory}")
        print((output_directory / "statistics.json").read_text(encoding="utf-8"))
        return

    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(
            f"incomplete chunk output exists: {output_directory}; remove it or use new settings"
        )

    token_codec = HuggingFaceTokenCodec.from_pretrained(args.tokenizer)
    chunker = FixedTokenChunker(
        token_codec=token_codec,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    result = build_chunk_corpus(
        documents=dataset.documents,
        chunker=chunker,
        token_codec=token_codec,
        tokenizer_name=args.tokenizer,
        source_documents_path=raw_directory / "corpus.jsonl",
        output_directory=output_directory,
    )
    print(
        json.dumps(
            {
                "task": spec.name,
                "documents": len(dataset.documents),
                "queries": len(dataset.queries),
                "output": str(result.output_directory),
                "statistics": asdict(result.statistics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
