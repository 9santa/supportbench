import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.experiments.rrf_grid_config import (
    RRFGridDefinition,
    load_rrf_grid_definition,
)
from supportbench.experiments.rrf_grid_export import (
    RRFGridExperimentMetadata,
    export_rrf_grid_search,
)
from supportbench.experiments.rrf_grid_search import (
    GRID_RECALL_CUTOFFS,
    RetrievalMetrics,
    RRFGridRun,
    RRFGridSearchResult,
    run_rrf_grid_search,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rrf_grid.yaml"
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents_v2.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_v2_dev.jsonl"
DEFAULT_DENSE_INDEX_PATH = PROJECT_ROOT / "artifacts" / "dense_v2"
# DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "rrf_grid"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "rrf_grid_v2"


@dataclass(frozen=True, slots=True)
class CliArguments:
    config_path: Path
    documents_path: Path
    queries_path: Path
    split: str
    output_path: Path

    retriever_config: RetrieverConfig


def parse_args() -> CliArguments:
    parser = argparse.ArgumentParser(
        description=("Grid-search Weighted RRF using cached BM25 and Dense rankings.")
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
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
        "--split",
        default="dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
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
        "--dense-index",
        type=Path,
        default=DEFAULT_DENSE_INDEX_PATH,
    )
    parser.add_argument(
        "--dense-model",
        default=("intfloat/multilingual-e5-base"),
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

    namespace = parser.parse_args()

    try:
        retriever_config = RetrieverConfig(
            dense_index_path=cast(Path, namespace.dense_index),
            dense_model_name=cast(str, namespace.dense_model),
            dense_device=cast(str, namespace.dense_device),
            dense_batch_size=cast(int, namespace.dense_batch_size),
            bm25_k1=cast(float, namespace.bm25_k1),
            bm25_b=cast(float, namespace.bm25_b),
        )
    except ValueError as error:
        parser.error(str(error))

    return CliArguments(
        config_path=cast(
            Path,
            namespace.config,
        ),
        documents_path=cast(
            Path,
            namespace.documents,
        ),
        queries_path=cast(
            Path,
            namespace.queries,
        ),
        split=cast(str, namespace.split),
        output_path=cast(
            Path,
            namespace.output,
        ),
        retriever_config=retriever_config,
    )


def print_grid_definition(
    definition: RRFGridDefinition,
) -> None:
    print("Weighted RRF grid search")
    print()
    print(f"Configurations: {len(definition.points)}")
    print(f"Maximum candidate_k: {definition.max_candidate_k}")
    print(f"Final top_k: {definition.final_top_k}")
    print("Evaluation cutoffs: " + ", ".join(str(k) for k in GRID_RECALL_CUTOFFS))
    print()


def print_run(
    label: str,
    run: RRFGridRun,
) -> None:
    metrics = run.metrics
    comparison = run.comparison_vs_dense

    print(f"{label}: {run.config.name}")
    print(f"  R@1:  {metrics.recall_at_1:.4f}")
    print(f"  R@3:  {metrics.recall_at_3:.4f}")
    print(f"  R@5:  {metrics.recall_at_5:.4f}")
    print(f"  R@10: {metrics.recall_at_10:.4f}")
    print(f"  R@20: {metrics.recall_at_20:.4f}")
    print(f"  R@50: {metrics.recall_at_50:.4f}")
    print(f"  MRR@10: {metrics.mrr:.4f}")
    print(
        "  Better/worse/tied by RR@10: "
        f"{comparison.better_by_rr}/"
        f"{comparison.worse_by_rr}/"
        f"{comparison.tied_by_rr}"
    )
    print(f"  Dense rank-1 degraded: {comparison.dense_rank_1_degraded}")
    print(f"  Hybrid-only hit@10: {comparison.hybrid_only_hit_at_10}")
    print(f"  Dense-only hit@10: {comparison.dense_only_hit_at_10}")

    for k in GRID_RECALL_CUTOFFS:
        relevant = comparison.relevant_documents_at(k)
        print(
            f"  Relevant@{k} queries +/-/=: "
            f"{relevant.queries_improved}/"
            f"{relevant.queries_degraded}/"
            f"{relevant.queries_tied}; "
            f"documents +/-: "
            f"{relevant.relevant_documents_gained}/"
            f"{relevant.relevant_documents_lost}"
        )


def print_summary(
    result: RRFGridSearchResult,
    *,
    output_path: Path,
) -> None:
    dense = RetrievalMetrics.from_evaluation(result.dense_baseline)

    print("Dense baseline:")
    print(f"  Queries: {dense.query_count}")
    print(f"  Labeled queries: {dense.labeled_query_count}")
    print(f"  Unlabeled queries: {dense.query_count - dense.labeled_query_count}")
    print(f"  R@1:  {dense.recall_at_1:.4f}")
    print(f"  R@3:  {dense.recall_at_3:.4f}")
    print(f"  R@5:  {dense.recall_at_5:.4f}")
    print(f"  R@10: {dense.recall_at_10:.4f}")
    print(f"  R@20: {dense.recall_at_20:.4f}")
    print(f"  R@50: {dense.recall_at_50:.4f}")
    print(f"  MRR@10: {dense.mrr:.4f}")
    print()

    print_run(
        "Best standalone",
        result.best_standalone,
    )
    print()

    print_run(
        "Best candidate",
        result.best_candidate,
    )
    print()

    print(f"Pareto configurations: {len(result.pareto_config_names)}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()

    definition = load_rrf_grid_definition(args.config_path)

    print_grid_definition(definition)

    documents = load_documents(args.documents_path)

    factory = RetrieverFactory(
        documents,
        config=args.retriever_config,
    )

    queries = load_queries(
        args.queries_path,
        {document.doc_id for document in documents},
    )

    selected_queries = [query for query in queries if query.split == args.split]

    if not selected_queries:
        raise SystemExit(f"no queries found for split {args.split!r}")

    bm25 = factory.create("bm25")
    dense = factory.create("dense")

    result = run_rrf_grid_search(
        queries=selected_queries,
        bm25=bm25,
        dense=dense,
        definition=definition,
    )

    export_rrf_grid_search(
        result,
        output_directory=args.output_path,
        metadata=RRFGridExperimentMetadata(
            split=args.split,
            documents_path=args.documents_path,
            queries_path=args.queries_path,
            dense_index_path=args.retriever_config.dense_index_path,
            dense_model_name=args.retriever_config.dense_model_name,
            bm25_k1=args.retriever_config.bm25_k1,
            bm25_b=args.retriever_config.bm25_b,
        ),
    )

    print_summary(
        result,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
