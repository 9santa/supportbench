import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from scripts.experiments.tracking import add_tracking_arguments, resolve_tracker
from scripts.nvidia_techqa._context_cli import (
    add_context_config_arguments,
    context_config_from_args,
)
from scripts.nvidia_techqa._evaluation_io import (
    append_jsonl,
    jsonable,
    load_completed_query_ids,
    load_jsonl,
    prepare_output,
    write_json,
    write_jsonl,
)
from supportbench.applications.nvidia_techqa import build_nvidia_techqa_context_service
from supportbench.benchmark.loaders import load_benchmark_queries
from supportbench.benchmark.models import BenchmarkQuery
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.evaluation.context_variants import summarize_context_variants
from supportbench.evaluation.rag_evaluator import (
    flatten_numeric_summary,
    reference_is_in_text,
)
from supportbench.experiments.fingerprints import read_git_state, sha256_file
from supportbench.rag.context import ContextPreparationRun, ContextPreparationService

EVALUATION_VERSION = "techqa_parent_count_context_v1"
DEFAULT_PARENT_COUNTS = (1, 2, 3, 4, 5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired NVIDIA TechQA contexts for several parent-count prefixes "
            "using one shared retrieval and evidence-selection run per query."
        )
    )
    add_context_config_arguments(parser)
    parser.add_argument(
        "--parent-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_PARENT_COUNTS),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl",
    )
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "parent_count_context"
        ),
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--resume", action="store_true")
    add_tracking_arguments(parser, default_experiment="supportbench-rag")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parent_counts = tuple(sorted(set(args.parent_counts)))
    _validate_args(parser, args, parent_counts=parent_counts)
    config = context_config_from_args(parser, args)
    modes = tuple(f"top_{count}" for count in parent_counts)

    output = args.output_root / args.output_name / args.split
    config_path = output / "evaluation_config.json"
    results_path = output / "per_query_results.jsonl"
    summary_path = output / "summary.json"
    failures_path = output / "failures.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    chunk_directory = config.chunks_root / config.chunk_config
    chunks_path = chunk_directory / "chunks.jsonl"
    config_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "split": args.split,
        "limit": args.limit,
        "parent_counts": list(parent_counts),
        "modes": list(modes),
        "context": jsonable(asdict(config)),
        "fingerprints": {
            "queries_sha256": sha256_file(args.queries),
            "chunks_sha256": sha256_file(chunks_path),
            "chunk_manifest_sha256": sha256_file(chunk_directory / "manifest.json"),
            "dense_index_manifest_sha256": sha256_file(
                config.index_root / config.chunk_config / "manifest.json"
            ),
        },
    }
    prepare_output(
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
        for query in load_benchmark_queries(args.queries, known_doc_ids=known_parent_ids)
        if query.split == args.split and query.answerability == "answerable"
    ]
    if args.limit is not None:
        queries = queries[: args.limit]

    completed = load_completed_query_ids(results_path) if args.resume else set()
    pending = [query for query in queries if query.query_id not in completed]
    context_service = build_nvidia_techqa_context_service(config)
    git_state = read_git_state(PROJECT_ROOT)
    tracker = resolve_tracker(parser, args)
    interrupted = False

    print(f"Answerable queries: {len(queries):,}", flush=True)
    print(f"Parent counts: {parent_counts}", flush=True)
    print(f"Already completed: {len(completed):,}", flush=True)
    print(f"Pending: {len(pending):,}", flush=True)

    with tracker.start_run(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or args.output_name,
        tags={
            "stage": "parent_count_context_evaluation",
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
            for index, query in enumerate(pending, start=1):
                result = compare_parent_count_contexts(
                    query=query,
                    context_service=context_service,
                    parent_counts=parent_counts,
                )
                append_jsonl(results_path, result)
                references = [
                    result.get(mode, {}).get("reference_answer_in_context") for mode in modes
                ]
                print(
                    f"[{index:,}/{len(pending):,}] {query.query_id}: "
                    f"{result['status']} reference {references}",
                    flush=True,
                )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            results = load_jsonl(results_path)
            summary = summarize_context_variants(results, modes=modes)
            summary.update(
                {
                    "evaluation_version": EVALUATION_VERSION,
                    "output_name": args.output_name,
                    "split": args.split,
                    "selected_query_count": len(queries),
                    "completed_query_count": len(results),
                    "interrupted": interrupted,
                }
            )
            write_json(summary_path, summary)
            write_jsonl(
                failures_path,
                (result for result in results if result.get("status") != "success"),
            )
            run.log_params(
                {
                    "query_count": len(queries),
                    "parent_counts": ",".join(str(value) for value in parent_counts),
                    "chunk_config": config.chunk_config,
                    "chunks_per_parent": config.chunks_per_parent,
                    "evidence_selection": config.evidence_selection,
                    "max_context_tokens": config.max_context_tokens,
                    "reserved_output_tokens": config.reserved_output_tokens,
                }
            )
            run.log_metrics(flatten_numeric_summary(summary))
            for artifact in (config_path, summary_path, results_path, failures_path):
                run.log_artifact(artifact, artifact_path="evaluation")

            print()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"MLflow run ID: {run.run_id}")

    if interrupted:
        raise KeyboardInterrupt


def compare_parent_count_contexts(
    *,
    query: BenchmarkQuery,
    context_service: ContextPreparationService,
    parent_counts: tuple[int, ...],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        maximum_run = context_service.prepare(query.query)
        variants: dict[str, ContextPreparationRun] = {}

        for parent_count in parent_counts:
            mode = f"top_{parent_count}"

            if parent_count == parent_counts[-1]:
                variants[mode] = maximum_run
                continue

            selected_chunks = tuple(
                chunk for chunk in maximum_run.retrieved_chunks if chunk.parent_rank <= parent_count
            )
            variants[mode] = context_service.prepare_from_chunks(
                query.query,
                retrieved_chunks=selected_chunks,
                retrieval=maximum_run.retrieval,
                prompt_budget=maximum_run.prompt_budget,
            )
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
        "reference_answer": query.reference_answer,
        "relevant_doc_ids": list(query.relevant_doc_ids),
        "status": "success",
        "shared_retrieval": {
            "candidate_parent_ids": [
                result.doc_id for result in maximum_run.retrieval.candidate_parents
            ],
            "reranked_parent_ids": [
                result.doc_id for result in maximum_run.retrieval.reranked_parents
            ],
            "fused_parent_ids": [result.doc_id for result in maximum_run.retrieval.fused_parents],
        },
        **{mode: _context_payload(query=query, run=variant) for mode, variant in variants.items()},
        "latency_ms": (time.perf_counter() - started) * 1_000.0,
        "error_type": None,
        "error_message": None,
    }


def _context_payload(
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
        "gold_document_in_context": bool(set(query.relevant_doc_ids) & set(context_parent_ids)),
        "reference_answer_in_context": reference_is_in_text(
            query.reference_answer,
            run.context.formatted_text,
        ),
        "formatted_context": run.context.formatted_text,
        "documents": [asdict(document) for document in run.context.documents],
        "provenance": [asdict(item) for item in run.context.provenance],
        "source_to_parent": {
            item.source_id: item.parent_doc_id
            for item in run.context.provenance
            if item.source_id is not None
        },
    }


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    parent_counts: tuple[int, ...],
) -> None:
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not parent_counts or parent_counts[0] <= 0:
        parser.error("--parent-counts must contain positive values")
    if parent_counts[-1] != args.top_parents:
        parser.error("largest --parent-counts value must equal --top-parents")


if __name__ == "__main__":
    main()
