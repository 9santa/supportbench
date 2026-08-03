import argparse
import json
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from supportbench.chunking.loaders import (
    load_chunk_parent_ids,
)
from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.evaluation.parent_document import (
    ParentDocumentRetriever,
)
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.experiments.evaluation_export import (
    export_query_evaluations,
)
from supportbench.reranking.factory import (
    CrossEncoderConfig,
    RerankingFactory,
)
from supportbench.retrieval.cached import (
    cache_retriever_results,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"

DEFAULT_CHUNKS_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"

DEFAULT_INDEX_ROOT = PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "chunk_reranker"
)

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chunk-level Hybrid retrieval and reranking with parent-document metrics."
        )
    )

    parser.add_argument(
        "--chunk-config",
        required=True,
        help=("Chunk corpus directory name, for example ft256o32 or ha384o64m512r2v2."),
    )

    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=DEFAULT_CHUNKS_ROOT,
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "dev",
        ),
        default="train",
    )
    parser.add_argument(
        "--candidate-pools",
        type=int,
        nargs="+",
        default=(20, 50),
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Optional number of labeled queries for a smoke test."),
    )

    parser.add_argument(
        "--dense-model",
        default=DEFAULT_DENSE_MODEL,
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
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
    )
    parser.add_argument(
        "--reranker-device",
        default="cuda",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
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
        default=1.0,
    )
    parser.add_argument(
        "--source-candidate-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=20,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    candidate_pools = tuple(sorted(set(args.candidate_pools)))

    _validate_arguments(
        parser=parser,
        candidate_pools=candidate_pools,
        final_top_k=args.final_top_k,
        source_candidate_k=(args.source_candidate_k),
        limit=args.limit,
    )

    chunk_directory = args.chunks_root / args.chunk_config

    documents_path = chunk_directory / "documents.jsonl"
    chunk_metadata_path = chunk_directory / "chunks.jsonl"
    dense_index_path = args.index_root / args.chunk_config

    output_directory = args.output_root / args.chunk_config / args.split

    if not dense_index_path.exists():
        parser.error(f"dense index does not exist: {dense_index_path}")

    print(f"Chunk config: {args.chunk_config}")
    print(f"Documents: {documents_path}")
    print(f"Dense index: {dense_index_path}")

    chunk_documents = load_documents(documents_path)

    parent_by_chunk_id = load_chunk_parent_ids(chunk_metadata_path)

    _validate_chunk_corpus(
        parser=parser,
        chunk_documents=chunk_documents,
        parent_by_chunk_id=(parent_by_chunk_id),
    )

    parent_document_ids = set(parent_by_chunk_id.values())

    all_queries = load_queries(
        args.queries,
        parent_document_ids,
    )

    # Reranking quality is measured only on
    # queries that have gold document labels.
    queries = [
        query for query in all_queries if query.split == args.split and query.relevant_doc_ids
    ]

    if args.limit is not None:
        queries = queries[: args.limit]

    if not queries:
        parser.error(f"no labeled queries found for split {args.split!r}")

    print(f"Indexed chunks: {len(chunk_documents):,}")
    print(f"Parent documents: {len(parent_document_ids):,}")
    print(f"Labeled queries: {len(queries):,}")
    print(
        "Hybrid config: "
        f"BM25={args.bm25_weight}, "
        f"Dense={args.dense_weight}, "
        f"source candidate_k="
        f"{args.source_candidate_k}, "
        f"RRF k={args.rrf_k}"
    )
    print("Candidate pools: " + ", ".join(str(pool) for pool in candidate_pools))
    print(f"Reranker: {args.reranker_model}")

    retriever_config = RetrieverConfig(
        dense_index_path=dense_index_path,
        dense_model_name=args.dense_model,
        dense_device=args.dense_device,
        dense_batch_size=(args.dense_batch_size),
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        candidate_k=(args.source_candidate_k),
        rrf_k=args.rrf_k,
    )

    retriever_factory = RetrieverFactory(
        chunk_documents,
        config=retriever_config,
    )

    hybrid_chunk_retriever = retriever_factory.create("hybrid")

    max_candidate_pool = max(candidate_pools)

    print()
    print(f"Caching Hybrid chunk results up to top-{max_candidate_pool}...")

    cached_chunk_retriever = cache_retriever_results(
        hybrid_chunk_retriever,
        queries,
        top_k=max_candidate_pool,
    )

    # This wrapper is used only for metrics.
    # Cached results remain chunk-level.
    parent_candidate_retriever = ParentDocumentRetriever(
        cached_chunk_retriever,
        parent_by_chunk_id=(parent_by_chunk_id),
    )

    reranking_factory = RerankingFactory(
        chunk_documents,
        config=CrossEncoderConfig(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=(args.reranker_batch_size),
            max_length=(args.reranker_max_length),
        ),
    )

    summaries: list[dict[str, Any]] = []

    for candidate_pool in candidate_pools:
        print()
        print("=" * 60)
        print(f"Candidate pool: {candidate_pool}")

        candidate_cutoffs = tuple(
            cutoff
            for cutoff in (
                1,
                3,
                5,
                10,
                20,
                50,
            )
            if cutoff <= candidate_pool
        )

        # Hybrid chunk ranking mapped to parent IDs
        # only for evaluation.
        candidate_evaluation = evaluate_retriever(
            parent_candidate_retriever,
            queries,
            top_k=candidate_pool,
            recall_cutoffs=(candidate_cutoffs),
            mrr_cutoff=min(
                10,
                candidate_pool,
            ),
        )

        # The reranker receives chunk IDs and chunk
        # texts, not parent document IDs.
        reranked_chunk_retriever = reranking_factory.create(
            candidate_retriever=(cached_chunk_retriever),
            candidate_k=candidate_pool,
        )

        # Parent mapping happens only after
        # cross-encoder reranking.
        parent_reranked_retriever = ParentDocumentRetriever(
            reranked_chunk_retriever,
            parent_by_chunk_id=(parent_by_chunk_id),
        )

        reranked_cutoffs = tuple(
            cutoff
            for cutoff in (
                1,
                3,
                5,
                10,
            )
            if cutoff <= args.final_top_k
        )

        reranked_evaluation = evaluate_retriever(
            parent_reranked_retriever,
            queries,
            top_k=args.final_top_k,
            recall_cutoffs=(reranked_cutoffs),
            mrr_cutoff=min(
                10,
                args.final_top_k,
            ),
        )

        pool_output_directory = output_directory / f"pool_{candidate_pool}"

        pool_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        export_query_evaluations(
            candidate_evaluation,
            pool_output_directory / "candidate_queries.jsonl",
        )

        export_query_evaluations(
            reranked_evaluation,
            pool_output_directory / "reranked_queries.jsonl",
        )

        candidate_coverage = candidate_evaluation.recall_at(candidate_pool)

        summary = {
            "dataset": "nvidia_techqa",
            "split": args.split,
            "evaluation_unit": ("parent_document_from_raw_chunks"),
            "chunk_config": (args.chunk_config),
            "indexed_chunk_count": (len(chunk_documents)),
            "parent_document_count": (len(parent_document_ids)),
            "labeled_query_count": (len(queries)),
            "candidate_pool": (candidate_pool),
            "final_top_k": (args.final_top_k),
            "hybrid": {
                "bm25_k1": args.bm25_k1,
                "bm25_b": args.bm25_b,
                "bm25_weight": (args.bm25_weight),
                "dense_weight": (args.dense_weight),
                "source_candidate_k": (args.source_candidate_k),
                "rrf_k": args.rrf_k,
            },
            "reranker": {
                "model": (args.reranker_model),
                "device": (args.reranker_device),
                "batch_size": (args.reranker_batch_size),
                "max_length": (args.reranker_max_length),
            },
            "candidate": _metrics_dict(candidate_evaluation),
            "reranked": _metrics_dict(reranked_evaluation),
            "candidate_coverage": (candidate_coverage),
            "oracle_gap_at_1": (candidate_coverage - reranked_evaluation.recall_at_1),
            "delta": {
                "recall_at_1": (reranked_evaluation.recall_at_1 - candidate_evaluation.recall_at_1),
                "recall_at_3": (reranked_evaluation.recall_at_3 - candidate_evaluation.recall_at_3),
                "recall_at_5": (reranked_evaluation.recall_at_5 - candidate_evaluation.recall_at_5),
                "recall_at_10": (
                    reranked_evaluation.recall_at_10 - candidate_evaluation.recall_at_10
                ),
                "mrr_at_10": (reranked_evaluation.mrr - candidate_evaluation.mrr),
            },
        }

        _write_json(
            pool_output_directory / "summary.json",
            summary,
        )

        summaries.append(summary)

        _print_result(
            candidate_pool=(candidate_pool),
            candidate_evaluation=(candidate_evaluation),
            reranked_evaluation=(reranked_evaluation),
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json(
        output_directory / "comparison.json",
        {
            "chunk_config": (args.chunk_config),
            "split": args.split,
            "experiments": summaries,
        },
    )


def _validate_arguments(
    *,
    parser: argparse.ArgumentParser,
    candidate_pools: tuple[int, ...],
    final_top_k: int,
    source_candidate_k: int,
    limit: int | None,
) -> None:
    if not candidate_pools:
        parser.error("--candidate-pools must not be empty")

    if any(pool <= 0 for pool in candidate_pools):
        parser.error("candidate pools must be positive")

    if final_top_k <= 0:
        parser.error("--final-top-k must be positive")

    if any(pool < final_top_k for pool in candidate_pools):
        parser.error("every candidate pool must be greater than or equal to --final-top-k")

    if source_candidate_k < max(candidate_pools):
        parser.error(
            "--source-candidate-k must be "
            "greater than or equal to the "
            "largest reranker candidate pool"
        )

    if limit is not None and limit <= 0:
        parser.error("--limit must be positive")


def _validate_chunk_corpus(
    *,
    parser: argparse.ArgumentParser,
    chunk_documents: list[Any],
    parent_by_chunk_id: dict[str, str],
) -> None:
    runtime_chunk_ids = {document.doc_id for document in chunk_documents}

    metadata_chunk_ids = set(parent_by_chunk_id)

    if runtime_chunk_ids == metadata_chunk_ids:
        return

    missing_metadata = runtime_chunk_ids - metadata_chunk_ids
    missing_runtime_documents = metadata_chunk_ids - runtime_chunk_ids

    parser.error(
        "documents.jsonl and chunks.jsonl "
        "contain different chunk IDs; "
        f"missing metadata="
        f"{len(missing_metadata)}, "
        f"missing runtime documents="
        f"{len(missing_runtime_documents)}"
    )


def _metrics_dict(
    result: RetrievalEvaluationResult,
) -> dict[str, Any]:
    return {
        "query_count": (result.query_count),
        "evaluation_top_k": (result.evaluation_top_k),
        "recalls": {str(cutoff): value for cutoff, value in result.recalls},
        "mrr": result.mrr,
        "mrr_cutoff": (result.mrr_cutoff),
    }


def _print_result(
    *,
    candidate_pool: int,
    candidate_evaluation: (RetrievalEvaluationResult),
    reranked_evaluation: (RetrievalEvaluationResult),
) -> None:
    candidate_coverage = candidate_evaluation.recall_at(candidate_pool)

    print(f"Candidate coverage: {candidate_coverage:.4f}")

    print("Before reranking:")
    _print_metrics(candidate_evaluation)

    print("After reranking:")
    _print_metrics(reranked_evaluation)

    print("Delta:")
    print(f"  ΔR@1:  {reranked_evaluation.recall_at_1 - candidate_evaluation.recall_at_1:+.4f}")
    print(f"  ΔR@3:  {reranked_evaluation.recall_at_3 - candidate_evaluation.recall_at_3:+.4f}")
    print(f"  ΔR@5:  {reranked_evaluation.recall_at_5 - candidate_evaluation.recall_at_5:+.4f}")
    print(f"  ΔR@10: {reranked_evaluation.recall_at_10 - candidate_evaluation.recall_at_10:+.4f}")
    print(f"  ΔMRR:  {reranked_evaluation.mrr - candidate_evaluation.mrr:+.4f}")


def _print_metrics(
    result: RetrievalEvaluationResult,
) -> None:
    available_recalls = dict(result.recalls)

    for cutoff in (
        1,
        3,
        5,
        10,
        20,
        50,
    ):
        value = available_recalls.get(cutoff)

        if value is not None:
            print(f"  R@{cutoff:<2}: {value:.4f}")

    print(f"  MRR:  {result.mrr:.4f}")


def _write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
