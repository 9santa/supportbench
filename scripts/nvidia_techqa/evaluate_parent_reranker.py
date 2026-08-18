import argparse
import json
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from scripts.experiments.tracking import (
    add_tracking_arguments,
    resolve_tracker,
)
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.data.loaders import load_documents, load_queries
from supportbench.evaluation.parent_document import UniqueParentDocumentRetriever
from supportbench.evaluation.retrieval_evaluator import evaluate_retriever
from supportbench.experiments.evaluation_export import export_query_evaluations
from supportbench.experiments.fingerprints import (
    read_git_state,
    sha256_file,
)
from supportbench.experiments.metrics import retrieval_metrics
from supportbench.reranking.factory import CrossEncoderConfig, RerankingFactory
from supportbench.reranking.parent import ParentEvidenceRerankingRetriever
from supportbench.retrieval.cached import cache_retriever_results
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource, WeightedRRFHybrid
from supportbench.retrieval.parent_hybrid import (
    ParentCandidateChunkRetriever,
    ParentWeightedRRFHybrid,
)

DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Evaluate parent-level RRF candidates with cross-encoder reranking."
    )

    add_tracking_arguments(
        parser,
        default_experiment="supportbench-retrieval",
    )

    parser.add_argument("--chunk-config", default=DEFAULT_CHUNK_CONFIG)
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "parent_reranker"),
    )
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parent-candidate-k", type=int, default=20)
    parser.add_argument("--chunks-per-parent", type=int, default=2)
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--second-evidence-weight", type=float, default=0.0)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--fusion-rrf-k", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.parent_candidate_k < 20:
        parser.error("--parent-candidate-k must be at least 20")

    if args.chunks_per_parent <= 0:
        parser.error("--chunks-per-parent must be positive")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if not 0.0 <= args.second_evidence_weight <= 1.0:
        parser.error("--second-evidence-weight must be between 0 and 1")

    if args.candidate_prior_weight < 0.0:
        parser.error("--candidate-prior-weight must be non-negative")

    if args.fusion_rrf_k <= 0:
        parser.error("--fusion-rrf-k must be positive")

    weight_label = str(args.second_evidence_weight).replace(".", "_")
    output_name = args.output_name or f"parent_evidence_w{weight_label}"

    if not output_name.strip() or Path(output_name).name != output_name:
        parser.error("--output-name must be a non-empty path segment")

    output = args.output_root / args.chunk_config / "experiments" / output_name / args.split

    if output.exists() and any(output.iterdir()):
        parser.error(f"experiment output already exists: {output}; use a new --experiment-name")

    tracker = resolve_tracker(parser, args)

    run_name = args.mlflow_run_name or (f"{args.chunk_config}-{output_name}-{args.split}")

    chunk_directory = args.chunks_root / args.chunk_config
    documents = load_documents(chunk_directory / "documents.jsonl")
    parent_by_chunk_id = load_chunk_parent_ids(chunk_directory / "chunks.jsonl")
    queries = load_queries(args.queries, set(parent_by_chunk_id.values()))
    selected_queries = [
        query for query in queries if query.split == args.split and query.relevant_doc_ids
    ]

    if args.limit is not None:
        selected_queries = selected_queries[: args.limit]

    git_state = read_git_state(PROJECT_ROOT)

    chunk_manifest_path = chunk_directory / "manifest.json"
    chunk_statistics_path = chunk_directory / "statistics.json"
    index_manifest_path = args.index_root / args.chunk_config / "manifest.json"

    with tracker.start_run(
        experiment_name=args.mlflow_experiment,
        run_name=run_name,
        tags={
            "stage": "reranking",
            "dataset": "nvidia_techqa",
            "split": args.split,
            "chunk_config": args.chunk_config,
            "implementation": ("independent_parent_evidence_reranker"),
            "git_commit": git_state.commit,
            "git_branch": git_state.branch,
            "git_dirty": str(git_state.dirty).lower(),
            "queries_sha256": sha256_file(args.queries),
            "chunk_manifest_sha256": (sha256_file(chunk_manifest_path)),
            "dense_index_manifest_sha256": (sha256_file(index_manifest_path)),
        },
    ) as run:
        run.log_params(
            {
                "query_count": len(selected_queries),
                "parent_candidate_k": (args.parent_candidate_k),
                "chunks_per_parent": (args.chunks_per_parent),
                "source_candidate_k": 500,
                "parent_aggregation": ("capped_top_2_sum"),
                "bm25_k1": 0.5,
                "bm25_b": 1.0,
                "bm25_weight": 1.0,
                "dense_model": ("intfloat/multilingual-e5-base"),
                "dense_normalized": True,
                "dense_weight": 1.5,
                "source_rrf_k": 10,
                "reranker_model": ("BAAI/bge-reranker-v2-m3"),
                "reranker_batch_size": (args.reranker_batch_size),
                "reranker_max_length": 512,
                "second_evidence_weight": (args.second_evidence_weight),
                "candidate_prior_weight": (args.candidate_prior_weight),
                "fusion_method": ("weighted_rrf"),
                "fusion_rrf_k": (args.fusion_rrf_k),
            }
        )

        factory = RetrieverFactory(
            documents,
            config=RetrieverConfig(
                dense_index_path=args.index_root / args.chunk_config,
                dense_model_name="intfloat/multilingual-e5-base",
                dense_device=args.dense_device,
                dense_batch_size=16,
                bm25_k1=0.5,
                bm25_b=1.0,
            ),
        )
        parent_retriever = ParentWeightedRRFHybrid(
            sources=(
                WeightedRetrieverSource("bm25", factory.create("bm25"), 1.0),
                WeightedRetrieverSource("dense", factory.create("dense"), 1.5),
            ),
            parent_by_chunk_id=parent_by_chunk_id,
            source_candidate_k=500,
            rrf_k=10,
            aggregation="capped_top_2_sum",
            representative_chunks_per_parent=args.chunks_per_parent,
        )
        chunk_candidates = ParentCandidateChunkRetriever(
            parent_retriever,
            parent_candidate_k=args.parent_candidate_k,
            chunks_per_parent=args.chunks_per_parent,
        )
        cached_chunk_candidates = cache_retriever_results(
            chunk_candidates,
            selected_queries,
            top_k=chunk_candidates.candidate_k,
        )
        parent_candidates = UniqueParentDocumentRetriever(
            cached_chunk_candidates,
            parent_by_chunk_id=parent_by_chunk_id,
            chunk_candidate_k=chunk_candidates.candidate_k,
        )
        reranking_retriever = RerankingFactory(
            documents,
            config=CrossEncoderConfig(
                model_name="BAAI/bge-reranker-v2-m3",
                device=args.reranker_device,
                batch_size=args.reranker_batch_size,
                max_length=512,
            ),
        ).create(
            candidate_retriever=cached_chunk_candidates,
            candidate_k=chunk_candidates.candidate_k,
        )
        cached_reranked_chunks = cache_retriever_results(
            reranking_retriever,
            selected_queries,
            top_k=chunk_candidates.candidate_k,
        )
        best_chunk_parent_reranker = ParentEvidenceRerankingRetriever(
            cached_reranked_chunks,
            parent_by_chunk_id=parent_by_chunk_id,
            chunk_candidate_k=chunk_candidates.candidate_k,
            second_evidence_weight=0.0,
        )
        parent_reranker = ParentEvidenceRerankingRetriever(
            cached_reranked_chunks,
            parent_by_chunk_id=parent_by_chunk_id,
            chunk_candidate_k=chunk_candidates.candidate_k,
            second_evidence_weight=args.second_evidence_weight,
        )
        cached_parent_candidates = cache_retriever_results(
            parent_candidates,
            selected_queries,
            top_k=args.parent_candidate_k,
        )
        cached_best_chunk_parents = cache_retriever_results(
            best_chunk_parent_reranker,
            selected_queries,
            top_k=args.parent_candidate_k,
        )
        cached_reranked_parents = cache_retriever_results(
            parent_reranker,
            selected_queries,
            top_k=args.parent_candidate_k,
        )
        fused_parent_retriever = WeightedRRFHybrid(
            sources=(
                WeightedRetrieverSource(
                    "candidate",
                    cached_parent_candidates,
                    args.candidate_prior_weight,
                ),
                WeightedRetrieverSource(
                    "cross_encoder",
                    cached_best_chunk_parents,
                    1.0,
                ),
            ),
            candidate_k=args.parent_candidate_k,
            rrf_k=args.fusion_rrf_k,
        )

        candidate_evaluation = evaluate_retriever(
            cached_parent_candidates,
            selected_queries,
            top_k=args.parent_candidate_k,
            recall_cutoffs=(1, 3, 5, 10, 20),
            mrr_cutoff=10,
        )
        best_chunk_evaluation = evaluate_retriever(
            cached_best_chunk_parents,
            selected_queries,
            top_k=args.parent_candidate_k,
            recall_cutoffs=(1, 3, 5, 10, 20),
            mrr_cutoff=10,
        )
        reranked_evaluation = evaluate_retriever(
            cached_reranked_parents,
            selected_queries,
            top_k=args.parent_candidate_k,
            recall_cutoffs=(1, 3, 5, 10, 20),
            mrr_cutoff=10,
        )
        fused_evaluation = evaluate_retriever(
            fused_parent_retriever,
            selected_queries,
            top_k=args.parent_candidate_k,
            recall_cutoffs=(1, 3, 5, 10, 20),
            mrr_cutoff=10,
        )

        metrics: dict[str, float] = {}

        metrics.update(
            retrieval_metrics(
                candidate_evaluation,
                prefix="candidate_parent",
            )
        )
        metrics.update(
            retrieval_metrics(
                best_chunk_evaluation,
                prefix="best_chunk_parent",
            )
        )
        metrics.update(
            retrieval_metrics(
                reranked_evaluation,
                prefix="reranked_parent",
            )
        )
        metrics.update(
            retrieval_metrics(
                fused_evaluation,
                prefix="fused_parent",
            )
        )

        run.log_metrics(metrics)

        export_query_evaluations(candidate_evaluation, output / "candidate_queries.jsonl")
        export_query_evaluations(
            best_chunk_evaluation,
            output / "best_chunk_queries.jsonl",
        )
        export_query_evaluations(reranked_evaluation, output / "reranked_queries.jsonl")
        export_query_evaluations(fused_evaluation, output / "fused_queries.jsonl")
        summary = {
            "output_name": output_name,
            "implementation": "independent_parent_evidence_reranker",
            "split": args.split,
            "query_count": len(selected_queries),
            "parent_candidate_k": args.parent_candidate_k,
            "chunks_per_parent": args.chunks_per_parent,
            "second_evidence_weight": args.second_evidence_weight,
            "candidate_prior_weight": args.candidate_prior_weight,
            "fusion_rrf_k": args.fusion_rrf_k,
            "candidate_recalls": dict(candidate_evaluation.recalls),
            "best_chunk_recalls": dict(best_chunk_evaluation.recalls),
            "reranked_recalls": dict(reranked_evaluation.recalls),
            "fused_recalls": dict(fused_evaluation.recalls),
            "candidate_mrr": candidate_evaluation.mrr,
            "best_chunk_mrr": best_chunk_evaluation.mrr,
            "reranked_mrr": reranked_evaluation.mrr,
            "fused_mrr": fused_evaluation.mrr,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        for artifact in (
            output / "summary.json",
            output / "candidate_queries.jsonl",
            output / "best_chunk_queries.jsonl",
            output / "reranked_queries.jsonl",
            output / "fused_queries.jsonl",
        ):
            run.log_artifact(
                artifact,
                artifact_path="evaluation",
            )

        for artifact in (
            chunk_manifest_path,
            chunk_statistics_path,
            index_manifest_path,
        ):
            run.log_artifact(
                artifact,
                artifact_path="manifests",
            )

        print(f"MLflow run ID: {run.run_id or 'tracking disabled'}")


if __name__ == "__main__":
    main()
