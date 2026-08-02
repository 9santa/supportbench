import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from supportbench.benchmark.beir import BEIR_DATASETS, load_beir_dataset
from supportbench.data.models import Document, QueryExample
from supportbench.evaluation.beir import BeirEvaluationResult, evaluate_beir_retriever
from supportbench.reranking.factory import CrossEncoderConfig, RerankingFactory
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.cached import CachedRetriever, cache_retriever_results
from supportbench.retrieval.dense_build import build_dense_index
from supportbench.retrieval.dense_encoder import SentenceTransformerDenceEncoder
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource, WeightedRRFHybrid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 50, 100)


@dataclass(frozen=True, slots=True)
class SystemRun:
    name: str
    retriever: CachedRetriever
    retrieval_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SupportBench retriever/reranker combinations on BEIR."
    )
    parser.add_argument("--dataset", choices=tuple(BEIR_DATASETS), default="scifact")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "beir",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "indexes" / "beir",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "beir",
    )
    parser.add_argument("--experiment-name", default="e5base_bgev2m3_wrrf_c100_v1")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--bm25-k1", type=float, default=0.5)
    parser.add_argument("--bm25-b", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.5)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    spec = BEIR_DATASETS[args.dataset]
    dataset = load_beir_dataset(
        args.data_root / spec.name,
        name=spec.name,
        split=spec.default_split,
    )
    queries = dataset.queries if args.limit is None else dataset.queries[: args.limit]
    experiment_root = args.output_root / spec.name / args.experiment_name

    if experiment_root.exists() and any(experiment_root.iterdir()):
        parser.error(
            f"experiment output already exists: {experiment_root}; "
            "use a new --experiment-name"
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "dataset": spec.name,
        "split": spec.default_split,
        "document_count": len(dataset.documents),
        "query_count": len(queries),
        "dense_model": args.dense_model,
        "reranker_model": args.reranker_model,
        "candidate_k": args.candidate_k,
        "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
        "wrrf": {
            "bm25_weight": args.bm25_weight,
            "dense_weight": args.dense_weight,
            "rrf_k": args.rrf_k,
        },
        "final_fusion": {
            "candidate_weight": args.candidate_prior_weight,
            "cross_encoder_weight": 1.0,
            "rrf_k": args.rrf_k,
        },
    }
    _write_json(experiment_root / "manifest.json", manifest)
    dense_index = args.index_root / spec.name / _path_label(args.dense_model)
    index_build = _ensure_dense_index(
        dataset.documents,
        output=dense_index,
        model_name=args.dense_model,
        device=args.dense_device,
        batch_size=args.dense_batch_size,
    )
    factory = RetrieverFactory(
        dataset.documents,
        config=RetrieverConfig(
            dense_index_path=dense_index,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )

    print(f"Caching BM25 top-{args.candidate_k}...")
    bm25 = _cache_run("bm25", factory.create("bm25"), queries, args.candidate_k)
    _update_manifest(experiment_root, manifest, completed_system="bm25")
    print(f"Caching Dense top-{args.candidate_k}...")
    dense = _cache_run("dense", factory.create("dense"), queries, args.candidate_k)
    _update_manifest(experiment_root, manifest, completed_system="dense")
    print(f"Caching Weighted RRF top-{args.candidate_k}...")
    wrrf_retriever = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", bm25.retriever, args.bm25_weight),
            WeightedRetrieverSource("dense", dense.retriever, args.dense_weight),
        ),
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    wrrf = _cache_run("wrrf", wrrf_retriever, queries, args.candidate_k)
    base_runs = (bm25, dense, wrrf)
    systems: list[SystemRun] = list(base_runs)
    reranking_factory = RerankingFactory(
        dataset.documents,
        config=CrossEncoderConfig(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=512,
        ),
    )

    for base_run in base_runs:
        print(f"Cross-encoding {base_run.name} top-{args.candidate_k}...")
        pure_reranker = reranking_factory.create(
            candidate_retriever=base_run.retriever,
            candidate_k=args.candidate_k,
        )
        reranked = _cache_run(
            f"{base_run.name}_reranked",
            pure_reranker,
            queries,
            args.candidate_k,
        )
        fusion = WeightedRRFHybrid(
            sources=(
                WeightedRetrieverSource(
                    "candidate",
                    base_run.retriever,
                    args.candidate_prior_weight,
                ),
                WeightedRetrieverSource("cross_encoder", reranked.retriever, 1.0),
            ),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
        fused = _cache_run(
            f"{base_run.name}_fused",
            fusion,
            queries,
            args.candidate_k,
        )
        systems.extend((reranked, fused))
        _update_manifest(
            experiment_root,
            manifest,
            completed_system=fused.name,
        )

    evaluations: dict[str, BeirEvaluationResult] = {}

    for system in systems:
        print(f"Evaluating {system.name}...")
        evaluation = evaluate_beir_retriever(
            system.retriever,
            queries,
            dataset.qrels,
            top_k=args.candidate_k,
            cutoffs=tuple(cutoff for cutoff in DEFAULT_CUTOFFS if cutoff <= args.candidate_k),
        )
        evaluations[system.name] = evaluation
        _export_system(
            system,
            evaluation,
            experiment_root / "systems" / system.name,
        )

    result = {
        **manifest,
        "status": "complete",
        "dense_index_build": index_build,
        "systems": {
            system.name: {
                "retrieval_seconds": system.retrieval_seconds,
                "metrics": _metrics(evaluations[system.name]),
            }
            for system in systems
        },
        "reference": _reference_metrics(spec.name, evaluations["bm25"]),
    }
    _write_json(experiment_root / "result.json", result)
    _write_json(experiment_root / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {experiment_root}")


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.experiment_name.strip() or Path(args.experiment_name).name != args.experiment_name:
        parser.error("--experiment-name must be a non-empty path segment")

    for name in ("dense_batch_size", "reranker_batch_size", "candidate_k", "rrf_k"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.candidate_k < max(DEFAULT_CUTOFFS):
        parser.error(f"--candidate-k must be at least {max(DEFAULT_CUTOFFS)}")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.candidate_prior_weight < 0.0:
        parser.error("--candidate-prior-weight must be non-negative")


def _ensure_dense_index(
    documents: Sequence[Document],
    *,
    output: Path,
    model_name: str,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        return {"built": False, "output": str(output)}

    encoder = SentenceTransformerDenceEncoder(
        model_name,
        device=device,
        batch_size=batch_size,
    )
    result = build_dense_index(
        documents=documents,
        encoder=encoder,
        model_name=model_name,
        output_directory=output,
    )
    return {"built": True, **asdict(result), "output_directory": str(result.output_directory)}


def _cache_run(
    name: str,
    retriever: Retriever,
    queries: Sequence[QueryExample],
    top_k: int,
) -> SystemRun:
    started = perf_counter()
    cached = cache_retriever_results(retriever, queries, top_k=top_k)
    return SystemRun(
        name=name,
        retriever=cached,
        retrieval_seconds=perf_counter() - started,
    )


def _export_system(
    system: SystemRun,
    evaluation: BeirEvaluationResult,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "summary.json",
        {
            "system": system.name,
            "retrieval_seconds": system.retrieval_seconds,
            "metrics": _metrics(evaluation),
        },
    )

    with (output / "queries.jsonl").open(mode="w", encoding="utf-8") as file:
        for query in evaluation.queries:
            file.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")

    with (output / "run.trec").open(mode="w", encoding="utf-8") as file:
        for query in evaluation.queries:
            for rank, (document_id, score) in enumerate(
                zip(query.retrieved_doc_ids, query.scores, strict=True),
                start=1,
            ):
                file.write(
                    f"{query.query_id} Q0 {document_id} {rank} {score:.12g} {system.name}\n"
                )


def _metrics(evaluation: BeirEvaluationResult) -> dict[str, dict[str, float]]:
    return {
        "ndcg": {str(cutoff): value for cutoff, value in evaluation.ndcg},
        "map": {
            str(cutoff): value for cutoff, value in evaluation.mean_average_precision
        },
        "recall": {str(cutoff): value for cutoff, value in evaluation.recall},
        "precision": {str(cutoff): value for cutoff, value in evaluation.precision},
        "mrr": {str(cutoff): value for cutoff, value in evaluation.mrr},
    }


def _reference_metrics(
    dataset_name: str,
    bm25_evaluation: BeirEvaluationResult,
) -> dict[str, object] | None:
    if dataset_name != "scifact":
        return None

    official_ndcg_at_10 = 0.665
    official_recall_at_100 = 0.908
    actual_ndcg_at_10 = dict(bm25_evaluation.ndcg)[10]
    actual_recall_at_100 = dict(bm25_evaluation.recall)[100]
    return {
        "name": "BEIR paper Lucene BM25",
        "source": "https://openreview.net/forum?id=wCu6T5xFjeJ",
        "metrics": {
            "ndcg_at_10": official_ndcg_at_10,
            "recall_at_100": official_recall_at_100,
        },
        "supportbench_bm25_delta": {
            "ndcg_at_10": actual_ndcg_at_10 - official_ndcg_at_10,
            "recall_at_100": actual_recall_at_100 - official_recall_at_100,
        },
        "comparability_note": (
            "The reference uses Lucene BM25; tokenizer and BM25 implementation differ."
        ),
    }


def _update_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    completed_system: str,
) -> None:
    completed = list(manifest.get("completed_systems", []))
    completed.append(completed_system)
    manifest["completed_systems"] = completed
    _write_json(root / "manifest.json", manifest)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _path_label(model_name: str) -> str:
    return model_name.rsplit("/", maxsplit=1)[-1]


if __name__ == "__main__":
    main()
