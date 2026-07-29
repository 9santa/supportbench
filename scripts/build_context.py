import argparse
from pathlib import Path

from supportbench.data.loaders import (
    load_documents,
)
from supportbench.rag.context_builder import (
    ContextBuilder,
)
from supportbench.rag.document_store import (
    InMemoryDocumentStore,
)
from supportbench.rag.retrieval_pipeline import (
    RetrievalPipeline,
)
from supportbench.reranking.factory import (
    CrossEncoderConfig,
    RerankingFactory,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)
from supportbench.retrieval.hybrid import (
    WeightedRetrieverSource,
    WeightedRRFHybrid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents_v2.jsonl"

DEFAULT_DENSE_INDEX_PATH = PROJECT_ROOT / "artifacts" / "dense_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build RAG context using RRF candidate retrieval and cross-encoder reranking.")
    )

    parser.add_argument("query")

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_DENSE_INDEX_PATH,
    )

    parser.add_argument(
        "--dense-model",
        default=("intfloat/multilingual-e5-base"),
    )
    parser.add_argument(
        "--reranker-model",
        default=("BAAI/bge-reranker-v2-m3"),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--source-candidate-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=12_000,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.top_k > args.reranker_candidate_k:
        parser.error("--top-k must not exceed --reranker-candidate-k")

    return args


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents)

    retrieval_factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=args.dense_index,
            dense_model_name=args.dense_model,
            dense_device=args.device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=0.5,
            bm25_b=1.0,
        ),
    )

    bm25 = retrieval_factory.create("bm25")
    dense = retrieval_factory.create("dense")

    candidate_rrf = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                name="bm25",
                retriever=bm25,
                weight=1.0,
            ),
            WeightedRetrieverSource(
                name="dense",
                retriever=dense,
                weight=1.5,
            ),
        ),
        candidate_k=args.source_candidate_k,
        rrf_k=20,
    )

    reranking_factory = RerankingFactory(
        documents,
        config=CrossEncoderConfig(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=(args.reranker_batch_size),
            max_length=512,
        ),
    )

    reranked_retriever = reranking_factory.create(
        candidate_retriever=(candidate_rrf),
        candidate_k=(args.reranker_candidate_k),
    )

    retrieval_pipeline = RetrievalPipeline(
        retriever=reranked_retriever,
        document_store=(InMemoryDocumentStore(documents)),
    )

    retrieved_documents = retrieval_pipeline.retrieve(
        args.query,
        top_k=args.top_k,
    )

    context = ContextBuilder(
        max_documents=args.max_documents,
        max_characters=args.max_characters,
    ).build(retrieved_documents)

    print(f"Query: {args.query}")
    print("Pipeline: rrf-candidate -> reranker")
    print(f"Retrieved documents: {len(retrieved_documents)}")
    print(f"Context documents: {len(context.documents)}")
    print(f"Context characters: {len(context.formatted_text)}")
    print(f"Truncated: {str(context.truncated).lower()}")
    print()

    print(context.formatted_text)


if __name__ == "__main__":
    main()
