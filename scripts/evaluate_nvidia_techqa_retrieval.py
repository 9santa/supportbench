import argparse
import json
from pathlib import Path

from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.evaluation.retrieval_evaluator import (
    evaluate_retriever,
)
from supportbench.experiments.evaluation_export import (
    export_query_evaluations,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)
from supportbench.chunking.loaders import (
    load_chunk_parent_ids,
)
from supportbench.evaluation.parent_document import (
    ParentDocumentRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "artifacts" / "indexes" / "nvidia_techqa" / "multilingual_e5_base"
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "evaluations" / "nvidia_techqa"

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Evaluate Dense or Hybrid retrieval on NVIDIA TechQA.")
    )

    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        required=True,
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--dense-model",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--dense-device",
        default="cuda",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--chunk-metadata",
        type=Path,
        default=None,
        help=(
            "Optional chunks.jsonl."
            "When provided, metrics are calculated against parent document IDs"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 50:
        parser.error("--top-k must be at least 50")

    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be greater than or equal to --top-k")

    documents = load_documents(args.documents)

    runtime_document_ids = {document.doc_id for document in documents}

    parent_by_chunk_id: dict[str, str] | None = None

    if args.chunk_metadata is not None:
        parent_by_chunk_id = load_chunk_parent_ids(args.chunk_metadata)

        metadata_chunk_ids = set(parent_by_chunk_id)

        if metadata_chunk_ids != runtime_document_ids:
            missing_metadata = runtime_document_ids - metadata_chunk_ids
            missing_documents = metadata_chunk_ids - runtime_document_ids

            parser.error(
                "chunk metadata and runtime corpus "
                "contain different chunk IDs; "
                f"missing metadata={len(missing_metadata)}, "
                f"missing documents={len(missing_documents)}"
            )

        known_query_document_ids = set(parent_by_chunk_id.values())
    else:
        known_query_document_ids = runtime_document_ids

    all_queries = load_queries(
        args.queries,
        known_query_document_ids,
    )

    queries = [query for query in all_queries if query.split == args.split]

    if not queries:
        parser.error(f"no queries for split {args.split!r}")

    config = RetrieverConfig(
        dense_index_path=args.dense_index,
        dense_model_name=args.dense_model,
        dense_device=args.dense_device,
        dense_batch_size=args.dense_batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )

    factory = RetrieverFactory(
        documents,
        config=config,
    )

    retriever = factory.create(args.retriever)

    if parent_by_chunk_id is not None:
        retriever = ParentDocumentRetriever(
            retriever,
            parent_by_chunk_id=parent_by_chunk_id,
        )

    print(f"Retriever: {args.retriever}")
    print(f"Documents: {len(documents):,}")
    print(f"Queries: {len(queries):,}")
    print(f"Split: {args.split}")

    result = evaluate_retriever(
        retriever,
        queries,
        top_k=args.top_k,
    )

    output_dir = args.output_root / f"{args.retriever}_document_baseline" / args.split

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_query_evaluations(
        result,
        output_dir / "queries.jsonl",
    )

    summary = {
        "dataset": "nvidia_techqa",
        "retriever": args.retriever,
        "split": args.split,
        "document_count": len(documents),
        "evaluation_unit": (
            "parent_document_from_raw_chunks" if parent_by_chunk_id is not None else "document"
        ),
        "runtime_document_count": len(documents),
        "parent_document_count": (
            len(set(parent_by_chunk_id.values()))
            if parent_by_chunk_id is not None
            else len(documents)
        ),
        "query_count": result.query_count,
        "labeled_query_count": (result.labeled_query_count),
        "unlabeled_query_count": (result.unlabeled_query_count),
        "top_k": args.top_k,
        "recall_at_1": result.recall_at_1,
        "recall_at_3": result.recall_at_3,
        "recall_at_5": result.recall_at_5,
        "recall_at_10": result.recall_at_10,
        "recall_at_20": result.recall_at_20,
        "recall_at_50": result.recall_at_50,
        "mrr_at_10": result.mrr,
        "configuration": {
            "dense_model": args.dense_model,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "bm25_weight": args.bm25_weight,
            "dense_weight": args.dense_weight,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Queries: "
        f"{result.query_count} total, "
        f"{result.labeled_query_count} labeled, "
        f"{result.unlabeled_query_count} unlabeled"
    )
    print(f"Recall@1:  {result.recall_at_1:.4f}")
    print(f"Recall@3:  {result.recall_at_3:.4f}")
    print(f"Recall@5:  {result.recall_at_5:.4f}")
    print(f"Recall@10: {result.recall_at_10:.4f}")
    print(f"Recall@20: {result.recall_at_20:.4f}")
    print(f"Recall@50: {result.recall_at_50:.4f}")
    print(f"MRR@10:    {result.mrr:.4f}")
    print()
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
