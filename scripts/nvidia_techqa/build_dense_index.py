import argparse
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import load_documents
from supportbench.retrieval.dense_build import (
    build_dense_index,
)
from supportbench.retrieval.dense_encoder import (
    SentenceTransformerDenceEncoder,
)

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes" / "multilingual_e5_base"
)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Build a dense FAISS index for NVIDIA TechQA."))

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    documents = load_documents(args.documents)

    if not documents:
        parser.error("documents corpus is empty")

    print(f"Loaded {len(documents):,} documents")
    print(f"Embedding model: {args.model_name}")
    print(f"Device: {args.device}")

    encoder = SentenceTransformerDenceEncoder(
        args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    result = build_dense_index(
        documents=documents,
        encoder=encoder,
        model_name=args.model_name,
        output_directory=args.output_dir,
    )

    print()
    print(f"Documents: {result.document_count:,}")
    print(f"Embedding dimension: {result.embedding_dimension}")
    print(f"Encoding time: {result.encoding_seconds:.2f}s")
    print(f"Index build time: {result.index_build_seconds:.2f}s")
    print(f"Index saved to: {result.output_directory}")


if __name__ == "__main__":
    main()
