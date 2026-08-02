import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.data.loaders import load_documents, load_queries
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.cached import cache_retriever_results
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource
from supportbench.retrieval.parent_hybrid import (
    ParentAggregation,
    ParentWeightedRRFHybrid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"
DEFAULT_CHUNKS_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks"
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "artifacts" / "indexes" / "nvidia_techqa"
DEFAULT_QUERIES = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "nvidia_techqa"
    / "parent_rrf_grid"
    / DEFAULT_CHUNK_CONFIG
    / "result.json"
)


@dataclass(frozen=True, slots=True)
class GridProfile:
    aggregation: ParentAggregation
    bm25_weight: float
    dense_weight: float
    rrf_k: int
    source_candidate_k: int


@dataclass(frozen=True, slots=True)
class GridRun:
    profile: GridProfile
    recall_at_20: float
    recall_at_50: float
    mrr_at_20: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune parent-level Weighted RRF on train and freeze the best profile on dev."
    )
    parser.add_argument("--chunk-config", default=DEFAULT_CHUNK_CONFIG)
    parser.add_argument("--chunks-root", type=Path, default=DEFAULT_CHUNKS_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--bm25-k1", type=float, default=0.5)
    parser.add_argument("--bm25-b", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunk_directory = args.chunks_root / args.chunk_config
    documents = load_documents(chunk_directory / "documents.jsonl")
    parent_by_chunk_id = load_chunk_parent_ids(chunk_directory / "chunks.jsonl")

    if {document.doc_id for document in documents} != set(parent_by_chunk_id):
        raise SystemExit("documents.jsonl and chunks.jsonl contain different chunk IDs")

    queries = load_queries(args.queries, set(parent_by_chunk_id.values()))
    train_queries = [
        query for query in queries if query.split == "train" and query.relevant_doc_ids
    ]
    dev_queries = [query for query in queries if query.split == "dev" and query.relevant_doc_ids]

    if args.limit_train is not None:
        train_queries = train_queries[: args.limit_train]

    if args.limit_dev is not None:
        dev_queries = dev_queries[: args.limit_dev]

    if not train_queries or not dev_queries:
        raise SystemExit("both train and dev must contain labeled queries")

    source_depths = (100, 200, 500)
    max_source_depth = max(source_depths)
    all_queries = tuple(train_queries + dev_queries)
    factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=args.index_root / args.chunk_config,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )

    print(f"Caching BM25 and Dense top-{max_source_depth} for {len(all_queries)} queries...")
    cached_bm25 = cache_retriever_results(
        factory.create("bm25"),
        all_queries,
        top_k=max_source_depth,
    )
    cached_dense = cache_retriever_results(
        factory.create("dense"),
        all_queries,
        top_k=max_source_depth,
    )

    profiles = (
        GridProfile(
            aggregation=cast(ParentAggregation, aggregation),
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            source_candidate_k=source_candidate_k,
        )
        for aggregation, bm25_weight, dense_weight, rrf_k, source_candidate_k in product(
            ("best_chunk_rank", "capped_top_2_sum"),
            (0.5, 1.0, 1.5),
            (0.5, 1.0, 1.5, 2.0),
            (10, 20, 40),
            source_depths,
        )
    )
    runs: list[GridRun] = []

    for profile in profiles:
        retriever = _create_parent_retriever(
            profile,
            bm25=cached_bm25,
            dense=cached_dense,
            parent_by_chunk_id=parent_by_chunk_id,
        )
        evaluation = evaluate_retriever(
            retriever,
            train_queries,
            top_k=50,
            recall_cutoffs=(20, 50),
            mrr_cutoff=20,
        )
        runs.append(
            GridRun(
                profile=profile,
                recall_at_20=evaluation.recall_at_20,
                recall_at_50=evaluation.recall_at_50,
                mrr_at_20=evaluation.mrr,
            )
        )

    runs.sort(
        key=lambda run: (-run.recall_at_20, -run.recall_at_50, -run.mrr_at_20, str(run.profile))
    )
    best = runs[0]
    best_retriever = _create_parent_retriever(
        best.profile,
        bm25=cached_bm25,
        dense=cached_dense,
        parent_by_chunk_id=parent_by_chunk_id,
    )
    train_curve = _evaluate_curve(best_retriever, train_queries)
    dev_curve = _evaluate_curve(best_retriever, dev_queries)

    payload = {
        "chunk_config": args.chunk_config,
        "selection_split": "train",
        "validation_split": "dev",
        "train_query_count": len(train_queries),
        "dev_query_count": len(dev_queries),
        "best_profile": asdict(best.profile),
        "train_curve": _metrics(train_curve),
        "dev_curve": _metrics(dev_curve),
        "runs": [
            {
                "profile": asdict(run.profile),
                "recall_at_20": run.recall_at_20,
                "recall_at_50": run.recall_at_50,
                "mrr_at_20": run.mrr_at_20,
            }
            for run in runs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    print("Best profile:")
    print(json.dumps(asdict(best.profile), ensure_ascii=False, indent=2))
    _print_curve("Train", train_curve)
    _print_curve("Dev", dev_curve)
    print(f"Saved: {args.output}")


def _create_parent_retriever(
    profile: GridProfile,
    *,
    bm25: Any,
    dense: Any,
    parent_by_chunk_id: dict[str, str],
) -> ParentWeightedRRFHybrid:
    return ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", bm25, profile.bm25_weight),
            WeightedRetrieverSource("dense", dense, profile.dense_weight),
        ),
        parent_by_chunk_id=parent_by_chunk_id,
        source_candidate_k=profile.source_candidate_k,
        rrf_k=profile.rrf_k,
        aggregation=profile.aggregation,
    )


def _evaluate_curve(
    retriever: ParentWeightedRRFHybrid,
    queries: list[Any],
) -> RetrievalEvaluationResult:
    return evaluate_retriever(
        retriever,
        queries,
        top_k=200,
        recall_cutoffs=(20, 50, 100, 200),
        mrr_cutoff=20,
    )


def _metrics(result: RetrievalEvaluationResult) -> dict[str, Any]:
    return {
        "recalls": {str(cutoff): value for cutoff, value in result.recalls},
        "mrr_at_20": result.mrr,
    }


def _print_curve(name: str, result: RetrievalEvaluationResult) -> None:
    recalls = dict(result.recalls)
    print(name + ": " + ", ".join(f"R@{cutoff}={value:.4f}" for cutoff, value in recalls.items()))


if __name__ == "__main__":
    main()
