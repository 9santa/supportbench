import argparse
import json
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import (
    load_documents,
    load_queries,
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

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"

DEFAULT_DENSE_INDEX_PATH = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes" / "multilingual_e5_base"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "reranker_document_baseline"
)

DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Hybrid retrieval followed by cross-encoder reranking on NVIDIA TechQA."
        )
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
        default=DEFAULT_DENSE_INDEX_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--split",
        choices=("train", "dev"),
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
        help=("Optional number of labeled queries for a smoke run."),
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    candidate_pools = tuple(sorted(set(args.candidate_pools)))

    if not candidate_pools:
        parser.error("--candidate-pools must not be empty")

    if any(pool < args.final_top_k for pool in candidate_pools):
        parser.error("every candidate pool must be greater than or equal to --final-top-k")

    if args.final_top_k <= 0:
        parser.error("--final-top-k must be positive")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    documents = load_documents(args.documents)

    known_doc_ids = {document.doc_id for document in documents}

    all_queries = load_queries(
        args.queries,
        known_doc_ids,
    )

    # Reranker retrieval quality is evaluated only
    # on queries with gold document labels.
    queries = [
        query for query in all_queries if query.split == args.split and query.relevant_doc_ids
    ]

    if args.limit is not None:
        queries = queries[: args.limit]

    if not queries:
        parser.error("no labeled queries found")

    print(f"Documents: {len(documents):,}")
    print(f"Labeled {args.split} queries: {len(queries):,}")
    print("Hybrid: BM25=1.0, Dense=1.0, RRF k=20, source candidate_k=100")
    print(f"Reranker: {args.reranker_model}")
    print(f"Reranker device: {args.reranker_device}")
    print("Candidate pools: " + ", ".join(str(pool) for pool in candidate_pools))

    retriever_config = RetrieverConfig(
        dense_index_path=args.dense_index,
        dense_model_name=args.dense_model,
        dense_device=args.dense_device,
        dense_batch_size=(args.dense_batch_size),
        bm25_k1=0.5,
        bm25_b=1.0,
        bm25_weight=1.0,
        dense_weight=1.0,
        candidate_k=100,
        rrf_k=20,
    )

    retriever_factory = RetrieverFactory(
        documents,
        config=retriever_config,
    )

    hybrid = retriever_factory.create("hybrid")

    max_candidate_pool = max(candidate_pools)

    print()
    print(f"Caching Hybrid candidates up to top-{max_candidate_pool}...")

    cached_hybrid = cache_retriever_results(
        hybrid,
        queries,
        top_k=max_candidate_pool,
    )

    reranking_factory = RerankingFactory(
        documents,
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
        print(f"Evaluating candidate pool {candidate_pool}...")

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

        candidate_evaluation = evaluate_retriever(
            cached_hybrid,
            queries,
            top_k=candidate_pool,
            recall_cutoffs=(candidate_cutoffs),
            mrr_cutoff=min(
                10,
                candidate_pool,
            ),
        )

        reranking_retriever = reranking_factory.create(
            candidate_retriever=(cached_hybrid),
            candidate_k=candidate_pool,
        )

        reranked_evaluation = evaluate_retriever(
            reranking_retriever,
            queries,
            top_k=args.final_top_k,
            recall_cutoffs=(
                1,
                3,
                5,
                10,
            ),
            mrr_cutoff=min(
                10,
                args.final_top_k,
            ),
        )

        pool_output_dir = args.output_dir / args.split / f"pool_{candidate_pool}"

        export_query_evaluations(
            candidate_evaluation,
            pool_output_dir / "candidate_queries.jsonl",
        )

        export_query_evaluations(
            reranked_evaluation,
            pool_output_dir / "reranked_queries.jsonl",
        )

        candidate_coverage = candidate_evaluation.recall_at(candidate_pool)

        summary = {
            "dataset": "nvidia_techqa",
            "split": args.split,
            "document_count": len(documents),
            "query_count": len(queries),
            "candidate_pool": candidate_pool,
            "final_top_k": args.final_top_k,
            "hybrid": {
                "bm25_k1": 0.5,
                "bm25_b": 1.0,
                "bm25_weight": 1.0,
                "dense_weight": 1.0,
                "source_candidate_k": 100,
                "rrf_k": 20,
            },
            "reranker": {
                "model": args.reranker_model,
                "device": args.reranker_device,
                "batch_size": (args.reranker_batch_size),
                "max_length": (args.reranker_max_length),
            },
            "candidate": _metrics_dict(candidate_evaluation),
            "reranked": _metrics_dict(reranked_evaluation),
            "candidate_coverage": (candidate_coverage),
            "oracle_gap_at_1": (candidate_coverage - reranked_evaluation.recall_at_1),
        }

        pool_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (pool_output_dir / "summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        summaries.append(summary)

        _print_result(
            candidate_pool=candidate_pool,
            candidate_evaluation=(candidate_evaluation),
            reranked_evaluation=(reranked_evaluation),
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (args.output_dir / args.split / "comparison.json").write_text(
        json.dumps(
            summaries,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _metrics_dict(
    result: RetrievalEvaluationResult,
) -> dict[str, Any]:
    return {
        "query_count": result.query_count,
        "evaluation_top_k": (result.evaluation_top_k),
        "recalls": {str(cutoff): value for cutoff, value in result.recalls},
        "mrr": result.mrr,
        "mrr_cutoff": result.mrr_cutoff,
    }


def _print_result(
    *,
    candidate_pool: int,
    candidate_evaluation: (RetrievalEvaluationResult),
    reranked_evaluation: (RetrievalEvaluationResult),
) -> None:
    print()
    print(f"Candidate pool: {candidate_pool}")
    print(f"Candidate coverage: {candidate_evaluation.recall_at(candidate_pool):.4f}")
    print("Before reranking:")
    print(f"  R@1:  {candidate_evaluation.recall_at_1:.4f}")
    print(f"  R@3:  {candidate_evaluation.recall_at_3:.4f}")
    print(f"  R@5:  {candidate_evaluation.recall_at_5:.4f}")
    print(f"  R@10: {candidate_evaluation.recall_at_10:.4f}")
    print(f"  MRR:  {candidate_evaluation.mrr:.4f}")

    print("After reranking:")
    print(f"  R@1:  {reranked_evaluation.recall_at_1:.4f}")
    print(f"  R@3:  {reranked_evaluation.recall_at_3:.4f}")
    print(f"  R@5:  {reranked_evaluation.recall_at_5:.4f}")
    print(f"  R@10: {reranked_evaluation.recall_at_10:.4f}")
    print(f"  MRR:  {reranked_evaluation.mrr:.4f}")

    print("Delta:")
    print(f"  ΔR@1:  {reranked_evaluation.recall_at_1 - candidate_evaluation.recall_at_1:+.4f}")
    print(f"  ΔR@10: {reranked_evaluation.recall_at_10 - candidate_evaluation.recall_at_10:+.4f}")
    print(f"  ΔMRR:  {reranked_evaluation.mrr - candidate_evaluation.mrr:+.4f}")


if __name__ == "__main__":
    main()
