import argparse
import json
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.experiments.synthetic_v2.reranker_benchmark import (
    LatencySummary,
    PipelinePerformanceSummary,
    benchmark_reranking_retriever,
    get_gpu_name,
)
from supportbench.experiments.synthetic_v2.reranker_report import (
    render_performance_table,
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

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "documents.jsonl"

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "queries_dev.jsonl"

DEFAULT_DENSE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "synthetic"
    / "v2"
    / "dense"
    / "multilingual-e5-base"
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "synthetic" / "v2" / "reranker" / "benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Benchmark reranked retrieval pipelines."))

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

    parser.add_argument(
        "--reranker-model",
        default=("BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--reranker-device",
        default="cuda",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
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
        "--final-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--standalone-dense-weight",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--standalone-rrf-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--candidate-dense-weight",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--candidate-rrf-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--benchmark-queries",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gpu-power-limit-watts",
        type=float,
        default=None,
        help=("Power limit used during the run. Recorded as metadata only."),
    )

    args = parser.parse_args()

    if args.source_candidate_k <= 0:
        parser.error("--source-candidate-k must be positive")

    if args.reranker_candidate_k <= 0:
        parser.error("--reranker-candidate-k must be positive")

    if args.reranker_candidate_k > args.source_candidate_k:
        parser.error("--reranker-candidate-k must not exceed --source-candidate-k")

    if args.final_top_k <= 0:
        parser.error("--final-top-k must be positive")

    if args.final_top_k > args.reranker_candidate_k:
        parser.error("--final-top-k must not exceed --reranker-candidate-k")

    if args.benchmark_queries <= 0:
        parser.error("--benchmark-queries must be positive")

    if args.warmup_queries < 0:
        parser.error("--warmup-queries must not be negative")

    return args


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents)

    retrieval_factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=args.dense_index,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )

    queries = load_queries(
        args.queries,
        {document.doc_id for document in documents},
    )

    selected_queries = [query for query in queries if query.split == args.split]

    if not selected_queries:
        raise SystemExit(f"No queries found for split {args.split!r}")

    bm25 = retrieval_factory.create("bm25")
    dense = retrieval_factory.create("dense")

    standalone_rrf = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                name="bm25",
                retriever=bm25,
                weight=1.0,
            ),
            WeightedRetrieverSource(
                name="dense",
                retriever=dense,
                weight=(args.standalone_dense_weight),
            ),
        ),
        candidate_k=args.source_candidate_k,
        rrf_k=args.standalone_rrf_k,
    )

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
                weight=(args.candidate_dense_weight),
            ),
        ),
        candidate_k=args.source_candidate_k,
        rrf_k=args.candidate_rrf_k,
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

    pipelines = (
        (
            "dense",
            reranking_factory.create(
                candidate_retriever=dense,
                candidate_k=(args.reranker_candidate_k),
            ),
        ),
        (
            "rrf_standalone",
            reranking_factory.create(
                candidate_retriever=(standalone_rrf),
                candidate_k=(args.reranker_candidate_k),
            ),
        ),
        (
            "rrf_candidate",
            reranking_factory.create(
                candidate_retriever=(candidate_rrf),
                candidate_k=(args.reranker_candidate_k),
            ),
        ),
    )

    summaries = tuple(
        benchmark_reranking_retriever(
            name=name,
            retriever=retriever,
            queries=selected_queries,
            top_k=args.final_top_k,
            reranker_batch_size=(args.reranker_batch_size),
            warmup_query_count=(args.warmup_queries),
            benchmark_query_count=(args.benchmark_queries),
        )
        for name, retriever in pipelines
    )

    gpu_name = get_gpu_name(args.reranker_device)

    print(
        render_performance_table(
            summaries,
            gpu_name=gpu_name,
            gpu_power_limit_watts=(args.gpu_power_limit_watts),
        )
    )

    export_benchmark(
        path=args.output,
        summaries=summaries,
        gpu_name=gpu_name,
        gpu_power_limit_watts=(args.gpu_power_limit_watts),
        reranker_model_name=(args.reranker_model),
        reranker_batch_size=(args.reranker_batch_size),
        reranker_candidate_k=(args.reranker_candidate_k),
        source_candidate_k=(args.source_candidate_k),
        final_top_k=args.final_top_k,
    )

    print()
    print(f"Saved: {args.output}")


def export_benchmark(
    *,
    path: Path,
    summaries: tuple[
        PipelinePerformanceSummary,
        ...,
    ],
    gpu_name: str | None,
    gpu_power_limit_watts: float | None,
    reranker_model_name: str,
    reranker_batch_size: int,
    reranker_candidate_k: int,
    source_candidate_k: int,
    final_top_k: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "environment": {
            "gpu_name": gpu_name,
            "gpu_power_limit_watts": (gpu_power_limit_watts),
            "manual_gpu_pauses": False,
        },
        "reranker": {
            "model_name": reranker_model_name,
            "batch_size": reranker_batch_size,
            "candidate_k": (reranker_candidate_k),
            "final_top_k": final_top_k,
        },
        "retrieval": {
            "source_candidate_k": (source_candidate_k),
        },
        "pipelines": [performance_summary_to_dict(summary) for summary in summaries],
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def performance_summary_to_dict(
    summary: PipelinePerformanceSummary,
) -> dict[str, object]:
    return {
        "name": summary.name,
        "query_count": summary.query_count,
        "warmup_query_count": (summary.warmup_query_count),
        "candidate_pair_count": (summary.candidate_pair_count),
        "effective_batch_count": (summary.effective_batch_count),
        "candidate_retrieval_latency_ms": (latency_to_dict(summary.candidate_retrieval_latency)),
        "reranking_latency_ms": (latency_to_dict(summary.reranking_latency)),
        "total_latency_ms": latency_to_dict(summary.total_latency),
        "pairs_per_second": (summary.pairs_per_second),
        "batches_per_second": (summary.batches_per_second),
        "queries_per_second": (summary.queries_per_second),
        "peak_allocated_gib": bytes_to_gib(summary.peak_allocated_bytes),
        "peak_reserved_gib": bytes_to_gib(summary.peak_reserved_bytes),
        "reranking_peak_allocated_gib": (bytes_to_gib(summary.peak_reranking_allocated_bytes)),
        "reranking_peak_reserved_gib": (bytes_to_gib(summary.peak_reranking_reserved_bytes)),
        "reranking_incremental_peak_gib": (bytes_to_gib(summary.peak_reranking_incremental_bytes)),
    }


def latency_to_dict(
    latency: LatencySummary,
) -> dict[str, float]:
    return {
        "mean": (latency.mean_seconds * 1_000),
        "p50": (latency.p50_seconds * 1_000),
        "p95": (latency.p95_seconds * 1_000),
    }


def bytes_to_gib(
    byte_count: int,
) -> float:
    return byte_count / (1024**3)


if __name__ == "__main__":
    main()
