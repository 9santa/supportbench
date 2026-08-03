import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scripts._paths import PROJECT_ROOT
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.data.loaders import load_documents, load_queries
from supportbench.evaluation.parent_document import (
    ParentDocumentRetriever,
    UniqueParentDocumentRetriever,
)
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.experiments.evaluation_export import export_query_evaluations
from supportbench.retrieval.cached import cache_retriever_results
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory

DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"
DEFAULT_CHUNKS_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "candidate_recall"
)
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CUTOFFS = (20, 50, 100, 200)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Hybrid candidate recall over raw chunks and unique parent documents."
        )
    )
    parser.add_argument("--chunk-config", default=DEFAULT_CHUNK_CONFIG)
    parser.add_argument("--chunks-root", type=Path, default=DEFAULT_CHUNKS_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument(
        "--unique-parent-chunk-k",
        type=int,
        default=1000,
        help="raw Hybrid chunks inspected to produce the unique-parent ranking",
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--bm25-k1", type=float, default=0.5)
    parser.add_argument("--bm25-b", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--source-candidate-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=20)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cutoffs = tuple(sorted(set(args.cutoffs)))
    _validate_arguments(parser, args=args, cutoffs=cutoffs)

    chunk_directory = args.chunks_root / args.chunk_config
    documents_path = chunk_directory / "documents.jsonl"
    metadata_path = chunk_directory / "chunks.jsonl"
    dense_index_path = args.index_root / args.chunk_config

    documents = load_documents(documents_path)
    parent_by_chunk_id = load_chunk_parent_ids(metadata_path)
    runtime_chunk_ids = {document.doc_id for document in documents}

    if runtime_chunk_ids != set(parent_by_chunk_id):
        parser.error("documents.jsonl and chunks.jsonl contain different chunk IDs")

    all_queries = load_queries(args.queries, set(parent_by_chunk_id.values()))
    queries = [
        query for query in all_queries if query.split == args.split and query.relevant_doc_ids
    ]

    if args.limit is not None:
        queries = queries[: args.limit]

    if not queries:
        parser.error(f"no labeled queries found for split {args.split!r}")

    factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=dense_index_path,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            candidate_k=args.source_candidate_k,
            rrf_k=args.rrf_k,
        ),
    )

    cached_chunks = cache_retriever_results(
        factory.create("hybrid"),
        queries,
        top_k=args.unique_parent_chunk_k,
    )
    raw_parent_retriever = ParentDocumentRetriever(
        cached_chunks,
        parent_by_chunk_id=parent_by_chunk_id,
    )
    unique_parent_retriever = UniqueParentDocumentRetriever(
        cached_chunks,
        parent_by_chunk_id=parent_by_chunk_id,
        chunk_candidate_k=args.unique_parent_chunk_k,
    )

    max_cutoff = max(cutoffs)
    raw_result = evaluate_retriever(
        raw_parent_retriever,
        queries,
        top_k=max_cutoff,
        recall_cutoffs=cutoffs,
        mrr_cutoff=min(10, max_cutoff),
    )
    unique_result = evaluate_retriever(
        unique_parent_retriever,
        queries,
        top_k=max_cutoff,
        recall_cutoffs=cutoffs,
        mrr_cutoff=min(10, max_cutoff),
    )

    output_directory = args.output_root / args.chunk_config / args.split
    export_query_evaluations(raw_result, output_directory / "raw_chunk_queries.jsonl")
    export_query_evaluations(unique_result, output_directory / "unique_parent_queries.jsonl")

    summary = {
        "dataset": "nvidia_techqa",
        "split": args.split,
        "chunk_config": args.chunk_config,
        "indexed_chunk_count": len(documents),
        "parent_document_count": len(set(parent_by_chunk_id.values())),
        "labeled_query_count": len(queries),
        "cutoffs": list(cutoffs),
        "hybrid": {
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "bm25_weight": args.bm25_weight,
            "dense_weight": args.dense_weight,
            "source_candidate_k": args.source_candidate_k,
            "rrf_k": args.rrf_k,
            "unique_parent_chunk_k": args.unique_parent_chunk_k,
        },
        "raw_chunks": _result_summary(raw_result),
        "unique_parents": _result_summary(unique_result),
    }
    _write_json(output_directory / "summary.json", summary)

    print(f"Chunk config: {args.chunk_config}")
    print(f"Labeled queries: {len(queries):,}")
    print(f"Unique-parent chunk pool: {args.unique_parent_chunk_k:,}")
    print()
    print(f"{'Cutoff':>8} {'Raw chunks':>12} {'Unique parents':>16}")

    for cutoff in cutoffs:
        print(
            f"{cutoff:>8} "
            f"{raw_result.recall_at(cutoff):>12.4f} "
            f"{unique_result.recall_at(cutoff):>16.4f}"
        )

    unique_counts = [len(query.retrieved_doc_ids) for query in unique_result.queries]
    print()
    print(
        "Unique parents returned: "
        f"min={min(unique_counts)}, "
        f"mean={mean(unique_counts):.1f}, "
        f"max={max(unique_counts)}"
    )
    print(f"Results: {output_directory}")


def _validate_arguments(
    parser: argparse.ArgumentParser,
    *,
    args: argparse.Namespace,
    cutoffs: tuple[int, ...],
) -> None:
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        parser.error("--cutoffs must contain positive values")

    if args.unique_parent_chunk_k < max(cutoffs):
        parser.error("--unique-parent-chunk-k must cover the largest cutoff")

    if args.source_candidate_k < args.unique_parent_chunk_k:
        parser.error("--source-candidate-k must cover --unique-parent-chunk-k")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")


def _result_summary(result: RetrievalEvaluationResult) -> dict[str, Any]:
    result_counts = [len(query.retrieved_doc_ids) for query in result.queries]

    return {
        "recalls": {str(cutoff): value for cutoff, value in result.recalls},
        "result_count": {
            "min": min(result_counts),
            "mean": mean(result_counts),
            "max": max(result_counts),
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
