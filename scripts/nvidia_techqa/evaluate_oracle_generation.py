import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from scripts.experiments.tracking import add_tracking_arguments, resolve_tracker
from scripts.nvidia_techqa._evaluation_io import (
    append_jsonl,
    load_jsonl,
    prepare_output,
    write_json,
    write_jsonl,
)
from supportbench.applications.nvidia_techqa import FROZEN_PROMPT_LAYOUT
from supportbench.evaluation.context_variants import summarize_generation_variants
from supportbench.evaluation.grounded_generation import evaluate_grounded_generation
from supportbench.evaluation.rag_evaluator import (
    flatten_numeric_summary,
    summarize_rag_results,
)
from supportbench.experiments.fingerprints import read_git_state, sha256_file
from supportbench.rag.generation.ollama import OllamaLLMClient
from supportbench.rag.generation.prompt import (
    SYSTEM_PROMPT,
    GroundedPromptBuilder,
)
from supportbench.rag.generation.service import GroundedAnswerGenerator
from supportbench.rag.models import ChunkProvenance, RAGContext, RetrievedDocument

EVALUATION_VERSION = "techqa_oracle_generation_v2"
CONTEXT_VARIANT_EVALUATION_VERSION = "techqa_context_variant_generation_v2"
PROMPT_VERSIONS = {
    "legacy_system_user": "grounded_source_ids_v4",
    "gemma_single_user": "grounded_source_ids_gemma_single_user_v5",
}
MODES = (
    "current",
    "gold_injected",
    "gold_only_selected",
    "oracle_source",
)
DEFAULT_CONTEXTS = (
    PROJECT_ROOT
    / "artifacts"
    / "nvidia_techqa"
    / "evaluations"
    / "oracle_context"
    / "ha384o64m512r2v2-oracle-context-v1"
    / "dev"
    / "per_query_results.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers from saved paired NVIDIA TechQA contexts without repeating retrieval."
        )
    )
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "oracle_generation"
        ),
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-model", default="gemma3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llm-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument(
        "--prompt-layout",
        choices=tuple(PROMPT_VERSIONS),
        default=FROZEN_PROMPT_LAYOUT,
    )
    parser.add_argument(
        "--context-run-id",
        default=None,
        help="MLflow run ID of the saved oracle-context evaluation.",
    )
    add_tracking_arguments(parser, default_experiment="supportbench-rag")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    source_config_path = args.contexts.parent / "evaluation_config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    context_config = source_config["context"]
    context_window = int(context_config["model_context_window"])
    reserved_output_tokens = int(context_config["reserved_output_tokens"])
    prompt_version = PROMPT_VERSIONS[args.prompt_layout]
    configured_modes = tuple(str(mode) for mode in source_config.get("modes", MODES))
    modes = tuple(dict.fromkeys(args.modes or configured_modes))
    unknown_modes = set(modes) - set(configured_modes)

    if unknown_modes:
        parser.error(
            "requested modes are missing from the context artifact: "
            + ", ".join(sorted(unknown_modes))
        )

    evaluation_version = (
        EVALUATION_VERSION
        if source_config.get("evaluation_version") == "techqa_oracle_context_v1"
        else CONTEXT_VARIANT_EVALUATION_VERSION
    )

    output = args.output_root / args.output_name / str(source_config["split"])
    config_path = output / "evaluation_config.json"
    results_path = output / "per_query_results.jsonl"
    summary_path = output / "summary.json"
    failures_path = output / "failures.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "evaluation_version": evaluation_version,
        "split": source_config["split"],
        "limit": args.limit,
        "modes": list(modes),
        "generation": {
            "model": args.llm_model,
            "backend": "ollama",
            "url": args.ollama_url,
            "temperature": args.temperature,
            "timeout_seconds": args.llm_timeout_seconds,
            "retries": args.llm_retries,
            "model_context_window": context_window,
            "reserved_output_tokens": reserved_output_tokens,
        },
        "prompt": {
            "version": prompt_version,
            "layout": args.prompt_layout,
            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "source": {
            "contexts_path": str(args.contexts.resolve()),
            "contexts_sha256": sha256_file(args.contexts),
            "config_sha256": sha256_file(source_config_path),
            "context_evaluation_version": source_config["evaluation_version"],
            "context_run_id": args.context_run_id,
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

    context_results = [
        result for result in load_jsonl(args.contexts) if result.get("status") == "success"
    ]
    if args.limit is not None:
        context_results = context_results[: args.limit]

    completed = _load_completed_pairs(results_path) if args.resume else set()
    pending = [
        (result, mode)
        for result in context_results
        for mode in modes
        if (str(result["query_id"]), mode) not in completed
    ]
    answer_generator = GroundedAnswerGenerator(
        prompt_builder=GroundedPromptBuilder(layout=args.prompt_layout),
        llm_client=OllamaLLMClient(
            model_name=args.llm_model,
            base_url=args.ollama_url,
            timeout_seconds=args.llm_timeout_seconds,
            temperature=args.temperature,
            context_window=context_window,
            max_output_tokens=reserved_output_tokens,
        ),
    )
    git_state = read_git_state(PROJECT_ROOT)
    tracker = resolve_tracker(parser, args)
    interrupted = False

    print(f"Queries: {len(context_results):,}", flush=True)
    print(f"Mode-query pairs: {len(context_results) * len(modes):,}", flush=True)
    print(f"Already completed: {len(completed):,}", flush=True)
    print(f"Pending: {len(pending):,}", flush=True)

    with tracker.start_run(
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or args.output_name,
        tags={
            "stage": "oracle_generation_evaluation",
            "dataset": "nvidia_techqa",
            "split": str(source_config["split"]),
            "lineage.context_run_id": args.context_run_id or "none",
            "git_commit": git_state.commit,
            "git_branch": git_state.branch,
            "git_dirty": str(git_state.dirty).lower(),
            "fingerprint.contexts": config_payload["source"]["contexts_sha256"],
        },
    ) as run:
        try:
            for index, (source_result, mode) in enumerate(pending, start=1):
                mode_payload = source_result[mode]
                context = _rag_context_from_payload(mode_payload)
                generation_result = evaluate_grounded_generation(
                    query=str(source_result["query"]),
                    reference_answer=_optional_string(source_result.get("reference_answer")),
                    relevant_doc_ids=tuple(str(item) for item in source_result["relevant_doc_ids"]),
                    context=context,
                    generator=answer_generator,
                    retries=args.llm_retries,
                )
                result = _result_payload(
                    source_result=source_result,
                    mode=mode,
                    mode_payload=mode_payload,
                    generation_result=generation_result,
                )
                append_jsonl(results_path, result)
                print(
                    f"[{index:,}/{len(pending):,}] {result['query_id']} {mode}: "
                    f"{result['status']} {result.get('decision')}",
                    flush=True,
                )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            results = load_jsonl(results_path)
            summary = summarize_generation_variants(results, modes=modes)

            if "oracle_source" in modes:
                oracle_verifiable = [
                    result
                    for result in results
                    if result.get("mode") == "oracle_source"
                    and result.get("reference_answer_in_context") is True
                ]
                summary["oracle_source_reference_present"] = summarize_rag_results(
                    oracle_verifiable
                )

            summary.update(
                {
                    "evaluation_version": evaluation_version,
                    "output_name": args.output_name,
                    "split": source_config["split"],
                    "selected_query_count": len(context_results),
                    "completed_pair_count": len(results),
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
                    "query_count": len(context_results),
                    "mode_count": len(modes),
                    "generation_model": args.llm_model,
                    "temperature": args.temperature,
                    "prompt_version": prompt_version,
                    "prompt_layout": args.prompt_layout,
                    "model_context_window": context_window,
                    "reserved_output_tokens": reserved_output_tokens,
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


def _rag_context_from_payload(payload: dict[str, Any]) -> RAGContext:
    return RAGContext(
        documents=tuple(
            RetrievedDocument(
                doc_id=str(document["doc_id"]),
                title=str(document["title"]),
                text=str(document["text"]),
                category=str(document["category"]),
                score=float(document["score"]),
                rank=int(document["rank"]),
            )
            for document in payload["documents"]
        ),
        formatted_text=str(payload["formatted_context"]),
        truncated=bool(payload["context_truncated"]),
        token_count=int(payload["context_token_count"]),
        provenance=tuple(
            ChunkProvenance(
                parent_doc_id=str(item["parent_doc_id"]),
                chunk_id=str(item["chunk_id"]),
                parent_rank=int(item["parent_rank"]),
                evidence_rank=int(item["evidence_rank"]),
                document_title=str(item["document_title"]),
                section_path=tuple(str(value) for value in item["section_path"]),
                ordinal=int(item["ordinal"]),
                source_start_char=_optional_int(item.get("source_start_char")),
                source_end_char=_optional_int(item.get("source_end_char")),
                included_start_char=_optional_int(item.get("included_start_char")),
                included_end_char=_optional_int(item.get("included_end_char")),
                removed_prefix_tokens=int(item["removed_prefix_tokens"]),
                included_tokens=int(item["included_tokens"]),
                truncated=bool(item["truncated"]),
                source_id=_optional_string(item.get("source_id")),
            )
            for item in payload["provenance"]
        ),
    )


def _result_payload(
    *,
    source_result: dict[str, Any],
    mode: str,
    mode_payload: dict[str, Any],
    generation_result: dict[str, object],
) -> dict[str, Any]:
    return {
        "query_id": source_result["query_id"],
        "mode": mode,
        "query": source_result["query"],
        "split": source_result["split"],
        "benchmark_reference_status": "answerable",
        "reference_answer": source_result.get("reference_answer"),
        "relevant_doc_ids": source_result["relevant_doc_ids"],
        "context_parent_ids": mode_payload["context_parent_ids"],
        "context_chunk_ids": mode_payload["context_chunk_ids"],
        "source_to_parent": mode_payload["source_to_parent"],
        "context_token_count": mode_payload["context_token_count"],
        "source_prompt_token_count": mode_payload["prompt_token_count"],
        "prompt_token_count": generation_result.get("prompt_eval_count")
        or mode_payload["prompt_token_count"],
        "context_truncated": mode_payload["context_truncated"],
        "gold_document_in_context": mode_payload["gold_document_in_context"],
        "reference_answer_in_context": mode_payload["reference_answer_in_context"],
        "context_latency_ms": 0.0,
        **generation_result,
    }


def _load_completed_pairs(path: Path) -> set[tuple[str, str]]:
    return {
        (str(result["query_id"]), str(result["mode"]))
        for result in load_jsonl(path)
        if result.get("query_id") is not None and result.get("mode") is not None
    }


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.contexts.is_file():
        parser.error(f"context results do not exist: {args.contexts}")
    if not (args.contexts.parent / "evaluation_config.json").is_file():
        parser.error("context evaluation_config.json does not exist beside --contexts")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.llm_timeout_seconds <= 0:
        parser.error("--llm-timeout-seconds must be positive")
    if args.llm_retries < 0:
        parser.error("--llm-retries must be non-negative")


if __name__ == "__main__":
    main()
