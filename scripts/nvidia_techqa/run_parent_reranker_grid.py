import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

from scripts._paths import PROJECT_ROOT
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.data.loaders import load_documents, load_queries
from supportbench.data.models import QueryExample
from supportbench.evaluation.parent_document import UniqueParentDocumentRetriever
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.experiments.evaluation_export import export_query_evaluations
from supportbench.reranking.factory import CrossEncoderConfig, RerankingFactory
from supportbench.reranking.parent import ParentEvidenceRerankingRetriever
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.cached import CachedRetriever, cache_retriever_results
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource, WeightedRRFHybrid
from supportbench.retrieval.parent_hybrid import (
    ParentCandidateChunkRetriever,
    ParentCandidateSubsetRetriever,
    ParentWeightedRRFHybrid,
)
from supportbench.retrieval.restricted import CandidateSetRestrictedRetriever

DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"
PARENT_CANDIDATE_DEPTHS = (20, 30, 50)
CHUNKS_PER_PARENT_VALUES = (1, 2, 4)
MAX_PARENT_CANDIDATE_K = max(PARENT_CANDIDATE_DEPTHS)
MAX_CHUNKS_PER_PARENT = max(CHUNKS_PER_PARENT_VALUES)


@dataclass(frozen=True, slots=True)
class GridProfile:
    parent_candidate_k: int
    chunks_per_parent: int

    @property
    def chunk_candidate_k(self) -> int:
        return self.parent_candidate_k * self.chunks_per_parent

    @property
    def name(self) -> str:
        return f"parents_{self.parent_candidate_k}_chunks_{self.chunks_per_parent}"


@dataclass(frozen=True, slots=True)
class ProfileEvaluation:
    profile: GridProfile
    candidate: RetrievalEvaluationResult
    independent: RetrievalEvaluationResult
    fused: RetrievalEvaluationResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune parent candidate depth and evidence coverage on train."
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
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "evaluations"
            / "nvidia_techqa"
            / "parent_reranker_grid"
        ),
    )
    parser.add_argument("--experiment-name", default="candidate_pool_evidence_grid_v1")
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--fusion-rrf-k", type=int, default=10)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--skip-dev", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)

    experiment_root = args.output_root / args.chunk_config / args.experiment_name

    if experiment_root.exists() and any(experiment_root.iterdir()):
        parser.error(
            f"experiment output already exists: {experiment_root}; "
            "use a new --experiment-name"
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        experiment_root / "manifest.json",
        {
            "status": "running",
            "chunk_config": args.chunk_config,
            "selection_split": "train",
            "validation_split": None if args.skip_dev else "dev",
            "parent_candidate_depths": PARENT_CANDIDATE_DEPTHS,
            "chunks_per_parent_values": CHUNKS_PER_PARENT_VALUES,
            "candidate_prior_weight": args.candidate_prior_weight,
            "fusion_rrf_k": args.fusion_rrf_k,
        },
    )

    chunk_directory = args.chunks_root / args.chunk_config
    documents = load_documents(chunk_directory / "documents.jsonl")
    parent_by_chunk_id = load_chunk_parent_ids(chunk_directory / "chunks.jsonl")
    queries = load_queries(args.queries, set(parent_by_chunk_id.values()))
    train_queries = _select_queries(queries, split="train", limit=args.limit_train)
    dev_queries = _select_queries(queries, split="dev", limit=args.limit_dev)

    if not train_queries:
        parser.error("train split does not contain labeled queries")

    if not args.skip_dev and not dev_queries:
        parser.error("dev split does not contain labeled queries")

    all_queries = tuple(train_queries + ([] if args.skip_dev else dev_queries))
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
        representative_chunks_per_parent=MAX_CHUNKS_PER_PARENT,
    )
    maximal_candidates = ParentCandidateChunkRetriever(
        parent_retriever,
        parent_candidate_k=MAX_PARENT_CANDIDATE_K,
        chunks_per_parent=MAX_CHUNKS_PER_PARENT,
    )

    candidate_started = perf_counter()
    print(
        f"Caching maximal {MAX_PARENT_CANDIDATE_K}x{MAX_CHUNKS_PER_PARENT} "
        f"candidate pool for {len(all_queries)} queries..."
    )
    cached_maximal_candidates = cache_retriever_results(
        maximal_candidates,
        all_queries,
        top_k=maximal_candidates.candidate_k,
    )
    candidate_seconds = perf_counter() - candidate_started
    reranking_factory = RerankingFactory(
        documents,
        config=CrossEncoderConfig(
            model_name="BAAI/bge-reranker-v2-m3",
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=512,
        ),
    )

    print(f"Cross-encoding train maximal pool ({len(train_queries)} queries)...")
    train_ce_started = perf_counter()
    cached_train_ranking = _cache_cross_encoder_ranking(
        reranking_factory,
        cached_maximal_candidates,
        train_queries,
        maximal_candidates.candidate_k,
    )
    train_ce_seconds = perf_counter() - train_ce_started
    profiles = tuple(
        GridProfile(parent_candidate_k=parent_k, chunks_per_parent=chunks_per_parent)
        for parent_k, chunks_per_parent in product(
            PARENT_CANDIDATE_DEPTHS,
            CHUNKS_PER_PARENT_VALUES,
        )
    )
    train_runs = [
        _evaluate_profile(
            profile,
            queries=train_queries,
            cached_maximal_candidates=cached_maximal_candidates,
            cached_cross_encoder_ranking=cached_train_ranking,
            parent_by_chunk_id=parent_by_chunk_id,
            candidate_prior_weight=args.candidate_prior_weight,
            fusion_rrf_k=args.fusion_rrf_k,
        )
        for profile in profiles
    ]

    for run in train_runs:
        _export_profile(run, experiment_root / "profiles" / run.profile.name / "train")

    best_independent = max(train_runs, key=lambda run: _selection_key(run.independent, run.profile))
    best_fused = max(train_runs, key=lambda run: _selection_key(run.fused, run.profile))
    selected_profiles = tuple(
        {
            best_independent.profile.name: best_independent.profile,
            best_fused.profile.name: best_fused.profile,
        }.values()
    )
    _write_json(
        experiment_root / "manifest.json",
        {
            "status": "train_complete_dev_pending" if not args.skip_dev else "train_complete",
            "chunk_config": args.chunk_config,
            "train_query_count": len(train_queries),
            "best_independent_profile": best_independent.profile.name,
            "best_fused_profile": best_fused.profile.name,
            "candidate_generation_seconds": candidate_seconds,
            "train_cross_encoder_seconds": train_ce_seconds,
            "train_runs": [_profile_summary(run) for run in train_runs],
        },
    )
    dev_runs: list[ProfileEvaluation] = []
    dev_ce_seconds: float | None = None

    if not args.skip_dev:
        print(f"Cross-encoding frozen dev maximal pool ({len(dev_queries)} queries)...")
        dev_ce_started = perf_counter()
        cached_dev_ranking = _cache_cross_encoder_ranking(
            reranking_factory,
            cached_maximal_candidates,
            dev_queries,
            maximal_candidates.candidate_k,
        )
        dev_ce_seconds = perf_counter() - dev_ce_started
        dev_runs = [
            _evaluate_profile(
                profile,
                queries=dev_queries,
                cached_maximal_candidates=cached_maximal_candidates,
                cached_cross_encoder_ranking=cached_dev_ranking,
                parent_by_chunk_id=parent_by_chunk_id,
                candidate_prior_weight=args.candidate_prior_weight,
                fusion_rrf_k=args.fusion_rrf_k,
            )
            for profile in selected_profiles
        ]

        for run in dev_runs:
            _export_profile(run, experiment_root / "profiles" / run.profile.name / "dev")

    result = {
        "status": "complete",
        "chunk_config": args.chunk_config,
        "train_query_count": len(train_queries),
        "dev_query_count": 0 if args.skip_dev else len(dev_queries),
        "selection_metric": "MRR@10, then R@1/R@3/R@5/R@10, then lower CE pair count",
        "best_independent_profile": best_independent.profile.name,
        "best_fused_profile": best_fused.profile.name,
        "candidate_generation_seconds": candidate_seconds,
        "train_cross_encoder_seconds": train_ce_seconds,
        "dev_cross_encoder_seconds": dev_ce_seconds,
        "train_runs": [_profile_summary(run) for run in train_runs],
        "dev_runs": [_profile_summary(run) for run in dev_runs],
    }
    _write_json(experiment_root / "result.json", result)
    _write_json(experiment_root / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {experiment_root}")


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.experiment_name.strip() or Path(args.experiment_name).name != args.experiment_name:
        parser.error("--experiment-name must be a non-empty path segment")

    if args.reranker_batch_size <= 0:
        parser.error("--reranker-batch-size must be positive")

    if args.candidate_prior_weight < 0.0:
        parser.error("--candidate-prior-weight must be non-negative")

    if args.fusion_rrf_k <= 0:
        parser.error("--fusion-rrf-k must be positive")

    for name in ("limit_train", "limit_dev"):
        value = getattr(args, name)

        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def _select_queries(
    queries: list[QueryExample],
    *,
    split: str,
    limit: int | None,
) -> list[QueryExample]:
    selected = [
        query for query in queries if query.split == split and query.relevant_doc_ids
    ]
    return selected if limit is None else selected[:limit]


def _cache_cross_encoder_ranking(
    factory: RerankingFactory,
    candidates: Retriever,
    queries: list[QueryExample],
    candidate_k: int,
) -> CachedRetriever:
    reranker = factory.create(candidate_retriever=candidates, candidate_k=candidate_k)
    return cache_retriever_results(reranker, queries, top_k=candidate_k)


def _evaluate_profile(
    profile: GridProfile,
    *,
    queries: list[QueryExample],
    cached_maximal_candidates: Retriever,
    cached_cross_encoder_ranking: Retriever,
    parent_by_chunk_id: dict[str, str],
    candidate_prior_weight: float,
    fusion_rrf_k: int,
) -> ProfileEvaluation:
    chunk_candidates = ParentCandidateSubsetRetriever(
        cached_maximal_candidates,
        parent_by_chunk_id=parent_by_chunk_id,
        source_candidate_k=MAX_PARENT_CANDIDATE_K * MAX_CHUNKS_PER_PARENT,
        parent_candidate_k=profile.parent_candidate_k,
        chunks_per_parent=profile.chunks_per_parent,
    )
    reranked_chunks = CandidateSetRestrictedRetriever(
        cached_cross_encoder_ranking,
        chunk_candidates,
        ranking_candidate_k=MAX_PARENT_CANDIDATE_K * MAX_CHUNKS_PER_PARENT,
        candidate_k=profile.chunk_candidate_k,
    )
    candidate_parents = UniqueParentDocumentRetriever(
        chunk_candidates,
        parent_by_chunk_id=parent_by_chunk_id,
        chunk_candidate_k=profile.chunk_candidate_k,
    )
    independent_parents = ParentEvidenceRerankingRetriever(
        reranked_chunks,
        parent_by_chunk_id=parent_by_chunk_id,
        chunk_candidate_k=profile.chunk_candidate_k,
        second_evidence_weight=0.0,
    )
    cached_candidate_parents = cache_retriever_results(
        candidate_parents,
        queries,
        top_k=profile.parent_candidate_k,
    )
    cached_independent_parents = cache_retriever_results(
        independent_parents,
        queries,
        top_k=profile.parent_candidate_k,
    )
    fused_parents = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                "candidate",
                cached_candidate_parents,
                candidate_prior_weight,
            ),
            WeightedRetrieverSource(
                "cross_encoder",
                cached_independent_parents,
                1.0,
            ),
        ),
        candidate_k=profile.parent_candidate_k,
        rrf_k=fusion_rrf_k,
    )
    cutoffs = (1, 3, 5, 10, profile.parent_candidate_k)

    return ProfileEvaluation(
        profile=profile,
        candidate=evaluate_retriever(
            cached_candidate_parents,
            queries,
            top_k=profile.parent_candidate_k,
            recall_cutoffs=cutoffs,
            mrr_cutoff=10,
        ),
        independent=evaluate_retriever(
            cached_independent_parents,
            queries,
            top_k=profile.parent_candidate_k,
            recall_cutoffs=cutoffs,
            mrr_cutoff=10,
        ),
        fused=evaluate_retriever(
            fused_parents,
            queries,
            top_k=profile.parent_candidate_k,
            recall_cutoffs=cutoffs,
            mrr_cutoff=10,
        ),
    )


def _selection_key(
    evaluation: RetrievalEvaluationResult,
    profile: GridProfile,
) -> tuple[float, float, float, float, float, int]:
    recalls = dict(evaluation.recalls)
    return (
        evaluation.mrr,
        recalls[1],
        recalls[3],
        recalls[5],
        recalls[10],
        -profile.chunk_candidate_k,
    )


def _profile_summary(run: ProfileEvaluation) -> dict[str, object]:
    return {
        "profile": asdict(run.profile),
        "chunk_candidate_k": run.profile.chunk_candidate_k,
        "candidate": _metrics(run.candidate),
        "independent": _metrics(run.independent),
        "fused": _metrics(run.fused),
    }


def _metrics(evaluation: RetrievalEvaluationResult) -> dict[str, object]:
    return {
        "recalls": {str(cutoff): value for cutoff, value in evaluation.recalls},
        "mrr_at_10": evaluation.mrr,
    }


def _export_profile(run: ProfileEvaluation, output: Path) -> None:
    export_query_evaluations(run.candidate, output / "candidate_queries.jsonl")
    export_query_evaluations(run.independent, output / "independent_queries.jsonl")
    export_query_evaluations(run.fused, output / "fused_queries.jsonl")
    _write_json(output / "summary.json", _profile_summary(run))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
