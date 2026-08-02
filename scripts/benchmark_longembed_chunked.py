import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from supportbench.benchmark.longembed import LONGEMBED_TASKS, load_longembed_task
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.data.loaders import load_documents
from supportbench.data.models import Document, QueryExample
from supportbench.evaluation.beir import BeirEvaluationResult, evaluate_beir_retriever
from supportbench.evaluation.parent_document import (
    ParentDocumentRetriever,
    UniqueParentDocumentRetriever,
)
from supportbench.reranking.factory import CrossEncoderConfig, RerankingFactory
from supportbench.reranking.parent import ParentEvidenceRerankingRetriever
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.cached import CachedRetriever, cache_retriever_results
from supportbench.retrieval.dense_build import build_dense_index
from supportbench.retrieval.dense_encoder import SentenceTransformerDenceEncoder
from supportbench.retrieval.dense_index import (
    FaissFlatVectorIndex,
    compute_document_fingerprint,
)
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource, WeightedRRFHybrid
from supportbench.retrieval.parent_hybrid import (
    ParentCandidateChunkRetriever,
    ParentWeightedRRFHybrid,
)

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
        description="Benchmark chunk and parent retrieval on a pinned LongEmbed task."
    )
    parser.add_argument("--task", choices=tuple(LONGEMBED_TASKS), default="2wikimqa")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "longembed",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "indexes" / "longembed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "longembed",
    )
    parser.add_argument("--chunking-key", default="ft384o64")
    parser.add_argument("--experiment-name", default="e5base_bgev2m3_parent_wrrf_c100_v1")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--parent-candidate-k", type=int, default=100)
    parser.add_argument("--source-candidate-k", type=int, default=500)
    parser.add_argument("--representative-chunks", type=int, default=2)
    parser.add_argument("--bm25-k1", type=float, default=0.5)
    parser.add_argument("--bm25-b", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.5)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--second-evidence-weight", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    spec = LONGEMBED_TASKS[args.task]
    raw_directory = args.data_root / spec.name
    dataset = load_longembed_task(raw_directory, name=spec.name)
    queries = dataset.queries if args.limit is None else dataset.queries[: args.limit]
    chunk_directory = raw_directory / "chunks" / args.chunking_key
    chunks = load_documents(chunk_directory / "documents.jsonl")
    parent_by_chunk_id = load_chunk_parent_ids(chunk_directory / "chunks.jsonl")
    experiment_root = args.output_root / spec.name / args.chunking_key / args.experiment_name

    if experiment_root.exists() and any(experiment_root.iterdir()):
        parser.error(
            f"experiment output already exists: {experiment_root}; "
            "use a new --experiment-name"
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "benchmark": "LongEmbed",
        "dataset": spec.name,
        "dataset_revision": dataset.revision,
        "document_count": len(dataset.documents),
        "chunk_count": len(chunks),
        "query_count": len(queries),
        "chunking_key": args.chunking_key,
        "dense_model": args.dense_model,
        "reranker_model": args.reranker_model,
        "source_candidate_k": args.source_candidate_k,
        "parent_candidate_k": args.parent_candidate_k,
        "representative_chunks_per_parent": args.representative_chunks,
        "bm25": {"k1": args.bm25_k1, "b": args.bm25_b},
        "parent_wrrf": {
            "aggregation": "capped_top_2_sum",
            "bm25_weight": args.bm25_weight,
            "dense_weight": args.dense_weight,
            "rrf_k": args.rrf_k,
        },
        "parent_reranker": {
            "independent_cross_encoder_ranking": True,
            "second_evidence_weight": args.second_evidence_weight,
        },
        "final_fusion": {
            "candidate_weight": args.candidate_prior_weight,
            "cross_encoder_weight": 1.0,
            "rrf_k": args.rrf_k,
        },
    }
    _write_json(experiment_root / "manifest.json", manifest)
    dense_index = (
        args.index_root / spec.name / args.chunking_key / _path_label(args.dense_model)
    )
    index_build = _ensure_dense_index(
        chunks,
        output=dense_index,
        model_name=args.dense_model,
        device=args.dense_device,
        batch_size=args.dense_batch_size,
    )
    chunk_factory = RetrieverFactory(
        chunks,
        config=RetrieverConfig(
            dense_index_path=dense_index,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )
    document_factory = RetrieverFactory(
        dataset.documents,
        config=RetrieverConfig(
            dense_index_path=Path("unused-document-dense-index"),
            dense_model_name=args.dense_model,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )

    systems: list[SystemRun] = []
    document_bm25 = _cache_run(
        "document_bm25",
        document_factory.create("bm25"),
        queries,
        args.parent_candidate_k,
    )
    systems.append(document_bm25)
    _persist_system(
        document_bm25,
        queries=queries,
        qrels=dataset.qrels,
        top_k=args.parent_candidate_k,
        output_root=experiment_root,
        manifest=manifest,
    )

    print(f"Caching chunk BM25 top-{args.source_candidate_k}...")
    bm25_chunks = _cache_run(
        "bm25_chunks",
        chunk_factory.create("bm25"),
        queries,
        args.source_candidate_k,
    )
    print(f"Caching chunk Dense top-{args.source_candidate_k}...")
    dense_chunks = _cache_run(
        "dense_chunks",
        chunk_factory.create("dense"),
        queries,
        args.source_candidate_k,
    )
    chunk_wrrf_retriever = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", bm25_chunks.retriever, args.bm25_weight),
            WeightedRetrieverSource("dense", dense_chunks.retriever, args.dense_weight),
        ),
        candidate_k=args.source_candidate_k,
        rrf_k=args.rrf_k,
    )
    print(f"Caching chunk Weighted RRF top-{args.source_candidate_k}...")
    wrrf_chunks = _cache_run(
        "wrrf_chunks",
        chunk_wrrf_retriever,
        queries,
        args.source_candidate_k,
    )

    for name, chunk_run in (
        ("chunk_bm25", bm25_chunks),
        ("chunk_dense", dense_chunks),
        ("chunk_wrrf", wrrf_chunks),
    ):
        raw = _cache_run(
            f"{name}_raw_parent",
            ParentDocumentRetriever(
                chunk_run.retriever,
                parent_by_chunk_id=parent_by_chunk_id,
            ),
            queries,
            args.parent_candidate_k,
        )
        unique = _cache_run(
            f"{name}_unique_parent",
            UniqueParentDocumentRetriever(
                chunk_run.retriever,
                parent_by_chunk_id=parent_by_chunk_id,
                chunk_candidate_k=args.source_candidate_k,
            ),
            queries,
            args.parent_candidate_k,
        )
        systems.extend((raw, unique))

        for run in (raw, unique):
            _persist_system(
                run,
                queries=queries,
                qrels=dataset.qrels,
                top_k=args.parent_candidate_k,
                output_root=experiment_root,
                manifest=manifest,
            )

    parent_wrrf_retriever = ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", bm25_chunks.retriever, args.bm25_weight),
            WeightedRetrieverSource("dense", dense_chunks.retriever, args.dense_weight),
        ),
        parent_by_chunk_id=parent_by_chunk_id,
        source_candidate_k=args.source_candidate_k,
        rrf_k=args.rrf_k,
        aggregation="capped_top_2_sum",
        representative_chunks_per_parent=args.representative_chunks,
    )
    parent_wrrf = _cache_run(
        "parent_wrrf",
        parent_wrrf_retriever,
        queries,
        args.parent_candidate_k,
    )
    systems.append(parent_wrrf)
    _persist_system(
        parent_wrrf,
        queries=queries,
        qrels=dataset.qrels,
        top_k=args.parent_candidate_k,
        output_root=experiment_root,
        manifest=manifest,
    )

    parent_chunk_candidates = ParentCandidateChunkRetriever(
        parent_wrrf_retriever,
        parent_candidate_k=args.parent_candidate_k,
        chunks_per_parent=args.representative_chunks,
    )
    chunk_candidate_k = parent_chunk_candidates.candidate_k
    reranking_factory = RerankingFactory(
        chunks,
        config=CrossEncoderConfig(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            max_length=512,
        ),
    )
    chunk_reranker = reranking_factory.create(
        candidate_retriever=parent_chunk_candidates,
        candidate_k=chunk_candidate_k,
    )
    independent_parent_reranker = ParentEvidenceRerankingRetriever(
        chunk_reranker,
        parent_by_chunk_id=parent_by_chunk_id,
        chunk_candidate_k=chunk_candidate_k,
        second_evidence_weight=args.second_evidence_weight,
    )
    print(
        f"Cross-encoding up to {chunk_candidate_k} representative chunks per query..."
    )
    parent_reranked = _cache_run(
        "parent_wrrf_reranked",
        independent_parent_reranker,
        queries,
        args.parent_candidate_k,
    )
    systems.append(parent_reranked)
    _persist_system(
        parent_reranked,
        queries=queries,
        qrels=dataset.qrels,
        top_k=args.parent_candidate_k,
        output_root=experiment_root,
        manifest=manifest,
    )
    parent_fusion = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                "candidate",
                parent_wrrf.retriever,
                args.candidate_prior_weight,
            ),
            WeightedRetrieverSource("cross_encoder", parent_reranked.retriever, 1.0),
        ),
        candidate_k=args.parent_candidate_k,
        rrf_k=args.rrf_k,
    )
    parent_fused = _cache_run(
        "parent_wrrf_fused",
        parent_fusion,
        queries,
        args.parent_candidate_k,
    )
    systems.append(parent_fused)
    _persist_system(
        parent_fused,
        queries=queries,
        qrels=dataset.qrels,
        top_k=args.parent_candidate_k,
        output_root=experiment_root,
        manifest=manifest,
    )

    evaluations: dict[str, BeirEvaluationResult] = {}

    for system in systems:
        print(f"Evaluating {system.name}...")
        evaluation = evaluate_beir_retriever(
            system.retriever,
            queries,
            dataset.qrels,
            top_k=args.parent_candidate_k,
            cutoffs=tuple(
                cutoff for cutoff in DEFAULT_CUTOFFS if cutoff <= args.parent_candidate_k
            ),
        )
        evaluations[system.name] = evaluation
        _export_system(system, evaluation, experiment_root / "systems" / system.name)

    chunk_statistics = json.loads(
        (chunk_directory / "statistics.json").read_text(encoding="utf-8")
    )
    result = {
        **manifest,
        "status": "complete",
        "dense_index_build": index_build,
        "chunk_statistics": chunk_statistics,
        "systems": {
            system.name: {
                "retrieval_seconds": system.retrieval_seconds,
                "metrics": _metrics(evaluations[system.name]),
            }
            for system in systems
        },
    }
    _write_json(experiment_root / "result.json", result)
    _write_json(experiment_root / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {experiment_root}")


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    path_segments = (args.chunking_key, args.experiment_name)

    if any(not value.strip() or Path(value).name != value for value in path_segments):
        parser.error("--chunking-key and --experiment-name must be path segments")

    positive_arguments = (
        "dense_batch_size",
        "reranker_batch_size",
        "parent_candidate_k",
        "source_candidate_k",
        "representative_chunks",
        "rrf_k",
    )

    for name in positive_arguments:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.parent_candidate_k < max(DEFAULT_CUTOFFS):
        parser.error(f"--parent-candidate-k must be at least {max(DEFAULT_CUTOFFS)}")

    if args.source_candidate_k < args.parent_candidate_k:
        parser.error("--source-candidate-k must be at least --parent-candidate-k")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if not 0.0 <= args.second_evidence_weight <= 1.0:
        parser.error("--second-evidence-weight must be between 0 and 1")


def _ensure_dense_index(
    documents: Sequence[Document],
    *,
    output: Path,
    model_name: str,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    fingerprint = compute_document_fingerprint(documents)

    if output.exists() and any(output.iterdir()):
        index = FaissFlatVectorIndex.load(
            output,
            expected_document_fingerprint=fingerprint,
            expected_model_name=model_name,
        )
        return {
            "built": False,
            "output": str(output),
            "document_count": index.size,
            "document_fingerprint": fingerprint,
        }

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
    return {
        "built": True,
        **asdict(result),
        "document_fingerprint": fingerprint,
        "output_directory": str(result.output_directory),
    }


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
                    f"{query.query_id} Q0 {document_id} {rank} "
                    f"{score:.12g} {system.name}\n"
                )


def _persist_system(
    system: SystemRun,
    *,
    queries: Sequence[QueryExample],
    qrels: Mapping[str, Mapping[str, int]],
    top_k: int,
    output_root: Path,
    manifest: dict[str, Any],
) -> None:
    evaluation = evaluate_beir_retriever(
        system.retriever,
        queries,
        qrels,
        top_k=top_k,
        cutoffs=tuple(cutoff for cutoff in DEFAULT_CUTOFFS if cutoff <= top_k),
    )
    _export_system(
        system,
        evaluation,
        output_root / "systems" / system.name,
    )
    _update_manifest(output_root, manifest, completed_system=system.name)


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
