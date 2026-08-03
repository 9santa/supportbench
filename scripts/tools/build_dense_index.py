import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import load_documents
from supportbench.retrieval.dense_build import (
    DenseIndexBuildResult,
    build_dense_index,
)
from supportbench.retrieval.dense_encoder import (
    SentenceTransformerDenceEncoder,
)

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "v1" / "documents.jsonl"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "synthetic"
    / "v1"
    / "dense"
    / "multilingual-e5-base"
)
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


@dataclass(frozen=True, slots=True)
class CLIArguments:
    documents_path: Path
    output_path: Path
    model_name: str
    device: str
    batch_size: int


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description="Encode documents and build an exact FAISS dense index."
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help="path to documents JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="directory for dense index artifacts",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers model name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="encoder device, for example cuda or cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="document encoding batch size",
    )

    args = parser.parse_args()

    batch_size = cast(int, args.batch_size)

    if batch_size <= 0:
        parser.error("--batch-size must be positive")

    return CLIArguments(
        documents_path=cast(
            Path,
            args.documents,
        ),
        output_path=cast(
            Path,
            args.output,
        ),
        model_name=cast(str, args.model),
        device=cast(str, args.device),
        batch_size=batch_size,
    )


def print_result(
    *,
    args: CLIArguments,
    result: DenseIndexBuildResult,
) -> None:
    text = "Dense index built successfully"
    width = len(text) + 4  # 2 spaces padding on each side
    print("+" + "-" * (width - 2) + "+")
    print(f"| {text} |")
    print("+" + "-" * (width - 2) + "+")

    print()
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Documents: {result.document_count}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Batch size: {args.batch_size}")
    print(f"Encoding time: {result.encoding_seconds:.3f} s")
    print(f"Index build time: {result.index_build_seconds:.3f} s")
    print(f"Output: {result.output_directory}")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    encoder = SentenceTransformerDenceEncoder(
        args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    result = build_dense_index(
        documents=documents,
        encoder=encoder,
        model_name=args.model_name,
        output_directory=args.output_path,
    )

    print_result(args=args, result=result)


if __name__ == "__main__":
    main()
