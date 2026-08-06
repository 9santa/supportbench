import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from scripts.experiments.tracking import add_tracking_arguments, resolve_tracker
from scripts.nvidia_techqa._context_cli import (
    add_context_config_arguments,
    context_config_from_args,
)
from supportbench.applications.nvidia_techqa import build_nvidia_techqa_context_service
from supportbench.benchmark.loaders import load_benchmark_queries
from supportbench.benchmark.models import BenchmarkQuery
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.evaluation.context_comparison import summarize_context_comparison
from supportbench.evaluation.rag_evaluator import (
    flatten_numeric_summary,
    reference_is_in_text,
)
from supportbench.experiments.fingerprints import read_git_state, sha256_file
from supportbench.rag.context import (
    ContextPreparationRun,
    ContextPreparationService,
    EvidenceSelection,
)

EVALUATION_VERSION = "techqa_evidence_selection_ab_v1"
BASELINE_SELECTION: EvidenceSelection = "retrieval_representatives"
CANDIDATE_SELECTION: EvidenceSelection = "within_parent_rerank"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original retrieval representatives with within-parent chunk reranking "
            "using one shared parent retrieval run per answerable query."
        )
    )
    add_context_config_arguments(
        parser,
        include_evidence_selection=False,
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit answerable queries after split filtering.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "nvidia_techqa"
            / "evaluations"
            / "evidence_selection"
        ),
    )
    parser.add_argument(
        "--output-name",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    add_tracking_arguments(
        parser,
        default_experiment="supportbench-rag",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = context_config_from_args(parser, args)
    output = args.output_root / args.output_name / args.split
    config_path = output / "evaluation_config.json"
    results_path = output / "per_query_results.jsonl"
    summary_path = output / "summary.json"
    failures_path = output / "failures.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    chunk_directory = config.chunks_root / config.chunk_config
    chunks_path = chunk_directory / "chunks.jsonl"
    index_manifest_path = config.index_root / config.chunk_config / "manifest.json"
    chunk_manifest_path = chunk_directory / "manifest.json"
    config_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "split": args.split,
        "limit": args.limit,
        "baseline_evidence_selection": BASELINE_SELECTION,
        "candidate_evidence_selection": CANDIDATE_SELECTION,
        "context": _jsonable(asdict(config)),
        "fingerprints": {
            "queries_sha256": sha256_file(args.queries),
            "chunks_sha256": sha256_file(chunks_path),
            "chunk_manifest_sha256": sha256_file(chunk_manifest_path),
            "dense_index_manifest_sha256": sha256_file(index_manifest_path),
        },
    }
    _prepare_output(
        parser=parser,
        config_path=config_path,
        results_path=results_path,
        config_payload=config_payload,
        resume=args.resume,
    )
    results_path.touch(exist_ok=True)

    known_parent_ids = set(load_chunk_parent_ids(chunks_path).values())
    queries = [
        query
        for query in load_benchmark_queries(
            args.queries,
            known_doc_ids=known_parent_ids,
        )
        if query.split == args.split and query.answerability == "answerable"
    ]

    if args.limit is not None:
        queries = queries[: args.limit]

    completed_query_ids = _load_completed_query_ids(results_path) if args.resume else set()
    pending_queries = [
        query for query in queries if query.query_id not in completed_query_ids
    ]
    git_state = read_git_state(PROJECT_ROOT)
    tracker = resolve_tracker(parser, args)
    context_service = build_nvidia_techqa_context_service(config)
    interrupted = False

    print(f"Answerable queries: {len(queries):,}", flush=True)
    print(f"Already completed: {len(completed_query_ids):,}", flush=True)
    print(f"Pending: {len(pending_queries):,}", flush=True)

    with tracker.start_run(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or args.output_name,
        tags={
            "stage": "context_evaluation",
            "dataset": "nvidia_techqa",
            "split": args.split,
            "chunk_config": config.chunk_config,
            "git_commit": git_state.commit,
            "git_branch": git_state.branch,
            "git_dirty": str(git_state.dirty).lower(),
            **{
                f"fingerprint.{name}": value
                for name, value in config_payload["fingerprints"].items()
            },
        },
    ) as run:
        try:
            for index, query in enumerate(pending_queries, start=1):
                result = compare_query_contexts(
                    query=query,
                    context_service=context_service,
                )
                _append_jsonl(results_path, result)
                baseline_reference = result.get("baseline", {}).get(
                    "reference_answer_in_context"
                )
                candidate_reference = result.get("candidate", {}).get(
                    "reference_answer_in_context"
                )
                print(
                    f"[{index:,}/{len(pending_queries):,}] {query.query_id}: "
                    f"{result['status']} reference {baseline_reference} -> "
                    f"{candidate_reference}",
                    flush=True,
                )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            results = _load_jsonl(results_path)
            summary = summarize_context_comparison(results)
            summary.update(
                {
                    "evaluation_version": EVALUATION_VERSION,
                    "output_name": args.output_name,
                    "split": args.split,
                    "selected_query_count": len(queries),
                    "completed_query_count": len(results),
                    "interrupted": interrupted,
                    "baseline_evidence_selection": BASELINE_SELECTION,
                    "candidate_evidence_selection": CANDIDATE_SELECTION,
                }
            )
            _write_json(summary_path, summary)
            _write_jsonl(
                failures_path,
                (result for result in results if result.get("status") != "success"),
            )
            run.log_params(
                {
                    "query_count": len(queries),
                    "chunk_config": config.chunk_config,
                    "baseline_evidence_selection": BASELINE_SELECTION,
                    "candidate_evidence_selection": CANDIDATE_SELECTION,
                    "top_parents": config.top_parents,
                    "chunks_per_parent": config.chunks_per_parent,
                    "max_context_tokens": config.max_context_tokens,
                    "reranker_model": config.reranker_model_name,
                }
            )
            run.log_metrics(flatten_numeric_summary(summary))

            for artifact in (
                config_path,
                summary_path,
                results_path,
                failures_path,
            ):
                run.log_artifact(artifact, artifact_path="evaluation")

            print()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"MLflow run ID: {run.run_id}")

    if interrupted:
        raise KeyboardInterrupt


def compare_query_contexts(
    *,
    query: BenchmarkQuery,
    context_service: ContextPreparationService,
) -> dict[str, Any]:
    baseline_started = time.perf_counter()

    try:
        baseline = context_service.prepare(
            query.query,
            evidence_selection=BASELINE_SELECTION,
        )
        baseline_latency_ms = (time.perf_counter() - baseline_started) * 1_000.0
        candidate_started = time.perf_counter()
        candidate = context_service.prepare(
            query.query,
            retrieval=baseline.retrieval,
            evidence_selection=CANDIDATE_SELECTION,
        )
        candidate_latency_ms = (time.perf_counter() - candidate_started) * 1_000.0
    except ValueError as error:
        return {
            "query_id": query.query_id,
            "query": query.query,
            "split": query.split,
            "status": "context_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    return {
        "query_id": query.query_id,
        "query": query.query,
        "split": query.split,
        "status": "success",
        "reference_answer": query.reference_answer,
        "relevant_doc_ids": list(query.relevant_doc_ids),
        "candidate_parent_ids": [
            result.doc_id for result in baseline.retrieval.candidate_parents
        ],
        "reranked_parent_ids": [
            result.doc_id for result in baseline.retrieval.reranked_parents
        ],
        "fused_parent_ids": [
            result.doc_id for result in baseline.retrieval.fused_parents
        ],
        "baseline": _context_result(query=query, run=baseline),
        "candidate": _context_result(query=query, run=candidate),
        "retrieval_and_baseline_latency_ms": baseline_latency_ms,
        "candidate_latency_ms": candidate_latency_ms,
        "error_type": None,
        "error_message": None,
    }


def _context_result(
    *,
    query: BenchmarkQuery,
    run: ContextPreparationRun,
) -> dict[str, Any]:
    context_parent_ids = [document.doc_id for document in run.context.documents]

    return {
        "selected_parent_ids": list(
            dict.fromkeys(chunk.parent_doc_id for chunk in run.retrieved_chunks)
        ),
        "selected_chunk_ids": [chunk.chunk_id for chunk in run.retrieved_chunks],
        "context_parent_ids": context_parent_ids,
        "context_chunk_ids": [item.chunk_id for item in run.context.provenance],
        "context_token_count": run.context.token_count,
        "prompt_token_count": run.prompt_token_count,
        "context_truncated": run.context.truncated,
        "gold_document_in_context": bool(
            set(query.relevant_doc_ids) & set(context_parent_ids)
        ),
        "reference_answer_in_context": reference_is_in_text(
            query.reference_answer,
            run.context.formatted_text,
        ),
    }


def _prepare_output(
    *,
    parser: argparse.ArgumentParser,
    config_path: Path,
    results_path: Path,
    config_payload: dict[str, Any],
    resume: bool,
) -> None:
    if results_path.exists() and not resume:
        parser.error(f"results already exist: {results_path}; use --resume or a new name")

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

        if existing != config_payload:
            parser.error("existing evaluation config does not match this run")
    else:
        _write_json(config_path, config_payload)


def _load_completed_query_ids(path: Path) -> set[str]:
    return {
        str(result["query_id"])
        for result in _load_jsonl(path)
        if result.get("query_id") is not None
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, payloads: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for payload in payloads:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


if __name__ == "__main__":
    main()
