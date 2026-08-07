import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._paths import PROJECT_ROOT
from scripts.experiments.tracking import (
    add_tracking_arguments,
    resolve_tracker,
)
from scripts.nvidia_techqa._context_cli import (
    add_context_config_arguments,
    context_config_from_args,
)
from supportbench.applications.nvidia_techqa import (
    FROZEN_PROMPT_LAYOUT,
    build_nvidia_techqa_context_service,
)
from supportbench.benchmark.loaders import load_benchmark_queries
from supportbench.benchmark.models import BenchmarkQuery
from supportbench.chunking.loaders import load_chunk_parent_ids
from supportbench.evaluation.rag_evaluator import (
    flatten_numeric_summary,
    lexical_token_scores,
    output_contract_diagnostics,
    reference_is_in_text,
    summarize_rag_results,
)
from supportbench.rag.citations import (
    CitationContractError,
    CitationResolutionError,
    CitationValidationError,
)
from supportbench.rag.generation.models import GeneratedAnswer, LLMResponse
from supportbench.rag.generation.ollama import (
    OllamaClientError,
    OllamaLLMClient,
)
from supportbench.rag.generation.parser import (
    GeneratedAnswerParseError,
)
from supportbench.rag.generation.prompt import (
    SYSTEM_PROMPT,
    GroundedPromptBuilder,
)
from supportbench.rag.generation.service import (
    GenerationTruncatedError,
    GroundedAnswerGenerator,
)

EVALUATION_VERSION = "techqa_rag_eval_v6"
PROMPT_VERSION = "grounded_source_ids_v4"
PARSER_VERSION = "strict_json_v1"
CITATION_VALIDATOR_VERSION = "source_resolution_contract_repair_v5"
STRUCTURED_OUTPUT_VERSION = "ollama_json_schema_v2"

DEFAULT_LLM_MODEL = "gemma3:4b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run end-to-end RAG evaluation on NVIDIA TechQA.")
    )

    add_context_config_arguments(parser)

    parser.add_argument(
        "--queries",
        type=Path,
        default=(PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
    )
    parser.add_argument(
        "--limit-answerable",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-unanswerable",
        type=int,
        default=None,
        help=(
            "Limit NVIDIA is_impossible queries. These have no benchmark reference; "
            "they are not guaranteed unanswerable against the full corpus."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "rag"),
    )
    parser.add_argument(
        "--output-name",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--retriever-run-id",
        default=None,
        help=("MLflow run ID of the frozen retrieval configuration."),
    )

    add_tracking_arguments(
        parser,
        default_experiment="supportbench-rag",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _validate_args(parser, args)

    config = context_config_from_args(
        parser,
        args,
    )

    output = args.output_root / args.output_name / args.split
    results_path = output / "per_query_results.jsonl"
    summary_path = output / "summary.json"
    config_path = output / "evaluation_config.json"
    failures_path = output / "failures.jsonl"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_payload = {
        "evaluation_version": (EVALUATION_VERSION),
        "split": args.split,
        "generation": {
            "model": args.llm_model,
            "backend": "ollama",
            "url": args.ollama_url,
            "temperature": args.temperature,
            "timeout_seconds": (args.llm_timeout_seconds),
            "retries": args.llm_retries,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "layout": FROZEN_PROMPT_LAYOUT,
            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "parser_version": PARSER_VERSION,
        "citation_validator_version": (CITATION_VALIDATOR_VERSION),
        "structured_output_version": STRUCTURED_OUTPUT_VERSION,
        "benchmark_label_semantics": {
            "answerable": "reference answer and gold parent are available",
            "benchmark_reference_missing": (
                "NVIDIA is_impossible example without a reference answer or source context; "
                "not guaranteed unanswerable against the full corpus"
            ),
        },
        "retrieval": _jsonable(asdict(config)),
        "retriever_run_id": (args.retriever_run_id),
    }

    _prepare_output(
        parser=parser,
        config_path=config_path,
        results_path=results_path,
        config_payload=config_payload,
        resume=args.resume,
    )

    parent_ids = set(
        load_chunk_parent_ids(config.chunks_root / config.chunk_config / "chunks.jsonl").values()
    )

    all_queries = load_benchmark_queries(
        args.queries,
        known_doc_ids=parent_ids,
    )
    selected_queries = _select_queries(
        all_queries,
        split=args.split,
        limit_answerable=(args.limit_answerable),
        limit_unanswerable=(args.limit_unanswerable),
    )

    completed_query_ids = _load_completed_query_ids(results_path) if args.resume else set()

    pending_queries = [
        query for query in selected_queries if query.query_id not in completed_query_ids
    ]

    print(f"Selected queries: {len(selected_queries):,}", flush=True)
    print(f"Already completed: {len(completed_query_ids):,}", flush=True)
    print(f"Pending: {len(pending_queries):,}", flush=True)

    context_service = build_nvidia_techqa_context_service(config)
    answer_generator = GroundedAnswerGenerator(
        prompt_builder=GroundedPromptBuilder(layout=FROZEN_PROMPT_LAYOUT),
        llm_client=OllamaLLMClient(
            model_name=args.llm_model,
            base_url=args.ollama_url,
            timeout_seconds=(args.llm_timeout_seconds),
            temperature=args.temperature,
            context_window=(config.model_context_window),
            max_output_tokens=(config.reserved_output_tokens),
        ),
    )

    tracker = resolve_tracker(
        parser,
        args,
    )

    interrupted = False

    with tracker.start_run(
        experiment_name=args.mlflow_experiment,
        run_name=(args.mlflow_run_name or args.output_name),
        tags={
            "stage": "rag",
            "dataset": "nvidia_techqa",
            "split": args.split,
            "chunk_config": (config.chunk_config),
            "lineage.retriever_run_id": (args.retriever_run_id or "none"),
            "evaluation_version": (EVALUATION_VERSION),
        },
    ) as run:
        try:
            for index, query in enumerate(
                pending_queries,
                start=1,
            ):
                result = _evaluate_query(
                    query=query,
                    context_service=(context_service),
                    answer_generator=(answer_generator),
                    llm_retries=(args.llm_retries),
                )

                _append_jsonl(
                    results_path,
                    result,
                )

                print(
                    f"[{index:,}/"
                    f"{len(pending_queries):,}] "
                    f"{query.query_id}: "
                    f"{result['status']} "
                    f"{result.get('decision')}",
                    flush=True,
                )

        except KeyboardInterrupt:
            interrupted = True

        finally:
            results = _load_jsonl(results_path)

            summary = summarize_rag_results(results)
            summary["evaluation_version"] = EVALUATION_VERSION
            summary["output_name"] = args.output_name
            summary["split"] = args.split
            summary["selected_query_count"] = len(selected_queries)
            summary["completed_query_count"] = len(results)
            summary["interrupted"] = interrupted

            summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            failures = [result for result in results if result["status"] != "success"]
            _write_jsonl(
                failures_path,
                failures,
            )

            print()
            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            run.log_params(
                {
                    "query_count": len(selected_queries),
                    "generation_model": args.llm_model,
                    "generation_backend": "ollama",
                    "temperature": args.temperature,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_layout": FROZEN_PROMPT_LAYOUT,
                    "prompt_hash": (config_payload["prompt"]["sha256"]),
                    "context_builder_version": ("representative_sources_v2"),
                    "max_context_tokens": (config.max_context_tokens),
                    "reserved_output_tokens": (config.reserved_output_tokens),
                    "top_parents": (config.top_parents),
                    "chunks_per_parent": (config.chunks_per_parent),
                    "evidence_selection": config.evidence_selection,
                }
            )

            run.log_metrics(flatten_numeric_summary(summary))

            for artifact in (
                config_path,
                summary_path,
                results_path,
                failures_path,
            ):
                run.log_artifact(
                    artifact,
                    artifact_path="evaluation",
                )

            print(f"MLflow run ID: {run.run_id}")

        if interrupted:
            raise KeyboardInterrupt


def _evaluate_query(
    *,
    query: BenchmarkQuery,
    context_service: Any,
    answer_generator: GroundedAnswerGenerator,
    llm_retries: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    context_started = time.perf_counter()

    try:
        context_run = context_service.prepare(query.query)
    except ValueError as error:
        return _error_result(
            query=query,
            status="context_error",
            error=error,
            total_started=total_started,
        )

    context_latency_ms = (time.perf_counter() - context_started) * 1_000.0

    context = context_run.context
    context_parent_ids = tuple(document.doc_id for document in context.documents)
    gold_document_in_context = (
        bool(set(query.relevant_doc_ids) & set(context_parent_ids))
        if query.answerability == "answerable"
        else None
    )
    reference_answer_in_context = (
        reference_is_in_text(
            query.reference_answer,
            context.formatted_text,
        )
        if query.answerability == "answerable"
        else None
    )

    generation_started = time.perf_counter()
    generation_run = None
    final_error: Exception | None = None

    for attempt in range(llm_retries + 1):
        try:
            generation_run = answer_generator.generate(
                query=query.query,
                context=context,
            )
            final_error = None
            break
        except OllamaClientError as error:
            final_error = error

            if attempt >= llm_retries:
                break

            time.sleep(
                min(
                    2.0**attempt,
                    8.0,
                )
            )
        except GeneratedAnswerParseError as error:
            return _generation_error_result(
                query=query,
                context_run=context_run,
                status="parse_error",
                error=error,
                context_latency_ms=(context_latency_ms),
                generation_started=(generation_started),
                total_started=total_started,
                raw_response=(error.raw_response),
                llm_response=error.llm_response,
                parsed_answer=None,
                raw_citation_ids=(),
                resolved_citation_ids=(),
            )
        except CitationResolutionError as error:
            return _generation_error_result(
                query=query,
                context_run=context_run,
                status="citation_resolution_error",
                error=error,
                context_latency_ms=context_latency_ms,
                generation_started=generation_started,
                total_started=total_started,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
            )
        except CitationContractError as error:
            return _generation_error_result(
                query=query,
                context_run=context_run,
                status="citation_contract_error",
                error=error,
                context_latency_ms=context_latency_ms,
                generation_started=generation_started,
                total_started=total_started,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
            )
        except CitationValidationError as error:
            return _generation_error_result(
                query=query,
                context_run=context_run,
                status="citation_error",
                error=error,
                context_latency_ms=(context_latency_ms),
                generation_started=(generation_started),
                total_started=total_started,
                raw_response=(error.raw_response),
                llm_response=error.llm_response,
                parsed_answer=error.parsed_answer,
                raw_citation_ids=error.raw_citation_ids,
                resolved_citation_ids=error.citation_ids,
            )
        except GenerationTruncatedError as error:
            return _generation_error_result(
                query=query,
                context_run=context_run,
                status="generation_truncated",
                error=error,
                context_latency_ms=context_latency_ms,
                generation_started=generation_started,
                total_started=total_started,
                raw_response=error.raw_response,
                llm_response=error.llm_response,
                parsed_answer=None,
                raw_citation_ids=(),
                resolved_citation_ids=(),
            )

    if generation_run is None:
        assert final_error is not None

        return _generation_error_result(
            query=query,
            context_run=context_run,
            status="llm_error",
            error=final_error,
            context_latency_ms=(context_latency_ms),
            generation_started=(generation_started),
            total_started=total_started,
            raw_response=None,
            llm_response=None,
            parsed_answer=None,
            raw_citation_ids=(),
            resolved_citation_ids=(),
        )

    generation_latency_ms = (time.perf_counter() - generation_started) * 1_000.0
    answer = generation_run.answer
    output_diagnostics = output_contract_diagnostics(
        decision=answer.decision,
        answer=answer.answer,
    )

    precision, recall, f1 = (
        lexical_token_scores(
            answer.answer,
            query.reference_answer,
        )
        if (query.answerability == "answerable" and answer.decision == "answer")
        else (None, None, None)
    )

    gold_document_cited = (
        bool(set(answer.citation_ids) & set(query.relevant_doc_ids))
        if (query.answerability == "answerable" and answer.decision == "answer")
        else None
    )

    result = _context_payload(
        query=query,
        context_run=context_run,
    )
    result.update(
        {
            "status": "success",
            "parsed_decision": answer.decision,
            "parsed_answer": answer.answer,
            "decision": answer.decision,
            "answer": answer.answer,
            "citation_ids": list(answer.citation_ids),
            "raw_citation_ids": list(generation_run.raw_citation_ids),
            "resolved_citation_ids": list(generation_run.resolved_citation_ids),
            "contract_repaired": generation_run.contract_repaired,
            "strict_contract_valid": generation_run.strict_contract_valid,
            "contract_violations": list(generation_run.contract_violations),
            **output_diagnostics,
            "full_output_contract_valid": (
                generation_run.strict_contract_valid
                and not _has_output_contract_violation(output_diagnostics)
            ),
            "raw_response": (generation_run.raw_response),
            "llm_called": (generation_run.raw_response is not None),
            **_llm_metadata(generation_run.llm_response),
            "gold_document_in_context": (gold_document_in_context),
            "reference_answer_in_context": (reference_answer_in_context),
            "gold_document_cited": (gold_document_cited),
            "reference_token_precision": (precision),
            "reference_token_recall": (recall),
            "reference_token_f1": f1,
            "context_latency_ms": (context_latency_ms),
            "generation_latency_ms": (generation_latency_ms),
            "total_latency_ms": ((time.perf_counter() - total_started) * 1_000.0),
            "error_type": None,
            "error_message": None,
        }
    )

    return result


def _context_payload(
    *,
    query: BenchmarkQuery,
    context_run: Any,
) -> dict[str, Any]:
    retrieval = context_run.retrieval
    context = context_run.context

    return {
        "query_id": query.query_id,
        "query": query.query,
        "split": query.split,
        "benchmark_reference_status": _benchmark_reference_status(query),
        "reference_answer": (query.reference_answer),
        "relevant_doc_ids": list(query.relevant_doc_ids),
        "candidate_parent_ids": [result.doc_id for result in retrieval.candidate_parents],
        "reranked_parent_ids": [result.doc_id for result in retrieval.reranked_parents],
        "fused_parent_ids": [result.doc_id for result in retrieval.fused_parents],
        "context_parent_ids": [document.doc_id for document in context.documents],
        "context_chunk_ids": [item.chunk_id for item in context.provenance],
        "source_to_parent": {
            item.source_id: item.parent_doc_id
            for item in context.provenance
            if item.source_id is not None
        },
        "context_token_count": (context.token_count),
        "prompt_token_count": (context_run.prompt_token_count),
        "context_truncated": (context.truncated),
        "context_text": (context.formatted_text),
        "provenance": [asdict(item) for item in context.provenance],
    }


def _generation_error_result(
    *,
    query: BenchmarkQuery,
    context_run: Any,
    status: str,
    error: Exception,
    context_latency_ms: float,
    generation_started: float,
    total_started: float,
    raw_response: str | None,
    llm_response: LLMResponse | None,
    parsed_answer: GeneratedAnswer | None,
    raw_citation_ids: tuple[str, ...],
    resolved_citation_ids: tuple[str, ...],
) -> dict[str, Any]:
    parsed_decision = parsed_answer.decision if parsed_answer is not None else None
    parsed_answer_text = parsed_answer.answer if parsed_answer is not None else None
    output_diagnostics = output_contract_diagnostics(
        decision=parsed_decision,
        answer=parsed_answer_text,
    )
    result = _context_payload(
        query=query,
        context_run=context_run,
    )
    result.update(
        {
            "status": status,
            "parsed_decision": parsed_decision,
            "parsed_answer": parsed_answer_text,
            "decision": None,
            "answer": None,
            "citation_ids": [],
            "raw_citation_ids": list(raw_citation_ids),
            "resolved_citation_ids": list(resolved_citation_ids),
            "contract_repaired": False,
            "strict_contract_valid": (
                False if status == "citation_contract_error" else None
            ),
            "contract_violations": (
                list(error.contract_violations)
                if isinstance(error, CitationValidationError)
                else []
            ),
            **output_diagnostics,
            "full_output_contract_valid": False,
            "raw_response": raw_response,
            "llm_called": True,
            **_llm_metadata(llm_response),
            "gold_document_in_context": (
                bool(
                    set(query.relevant_doc_ids)
                    & {document.doc_id for document in context_run.context.documents}
                )
                if query.answerability == "answerable"
                else None
            ),
            "reference_answer_in_context": (
                reference_is_in_text(
                    query.reference_answer,
                    context_run.context.formatted_text,
                )
                if query.answerability == "answerable"
                else None
            ),
            "gold_document_cited": None,
            "reference_token_precision": None,
            "reference_token_recall": None,
            "reference_token_f1": None,
            "context_latency_ms": (context_latency_ms),
            "generation_latency_ms": ((time.perf_counter() - generation_started) * 1_000.0),
            "total_latency_ms": ((time.perf_counter() - total_started) * 1_000.0),
            "error_type": (type(error).__name__),
            "error_message": str(error),
        }
    )
    return result


def _error_result(
    *,
    query: BenchmarkQuery,
    status: str,
    error: Exception,
    total_started: float,
) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query": query.query,
        "split": query.split,
        "benchmark_reference_status": _benchmark_reference_status(query),
        "reference_answer": (query.reference_answer),
        "relevant_doc_ids": list(query.relevant_doc_ids),
        "status": status,
        "parsed_decision": None,
        "parsed_answer": None,
        "decision": None,
        "answer": None,
        "citation_ids": [],
        "raw_citation_ids": [],
        "resolved_citation_ids": [],
        "contract_repaired": False,
        "strict_contract_valid": None,
        "contract_violations": [],
        "answer_source_id_leak": False,
        "answer_embedded_citation_list": False,
        "answer_word_count": 0,
        "answer_over_120_words": False,
        "decision_content_mismatch": False,
        "full_output_contract_valid": False,
        "raw_response": None,
        "llm_called": False,
        **_llm_metadata(None),
        "candidate_parent_ids": [],
        "reranked_parent_ids": [],
        "fused_parent_ids": [],
        "context_parent_ids": [],
        "context_chunk_ids": [],
        "source_to_parent": {},
        "context_token_count": None,
        "prompt_token_count": None,
        "context_truncated": None,
        "context_text": None,
        "provenance": [],
        "gold_document_in_context": None,
        "reference_answer_in_context": None,
        "gold_document_cited": None,
        "reference_token_precision": None,
        "reference_token_recall": None,
        "reference_token_f1": None,
        "context_latency_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": ((time.perf_counter() - total_started) * 1_000.0),
        "error_type": (type(error).__name__),
        "error_message": str(error),
    }


def _select_queries(
    queries: list[BenchmarkQuery],
    *,
    split: str,
    limit_answerable: int | None,
    limit_unanswerable: int | None,
) -> list[BenchmarkQuery]:
    selected: list[BenchmarkQuery] = []
    answerable_count = 0
    unanswerable_count = 0

    for query in queries:
        if query.split != split:
            continue

        if query.answerability == "answerable":
            if limit_answerable is not None and answerable_count >= limit_answerable:
                continue

            answerable_count += 1
        else:
            if limit_unanswerable is not None and unanswerable_count >= limit_unanswerable:
                continue

            unanswerable_count += 1

        selected.append(query)

    return selected


def _benchmark_reference_status(query: BenchmarkQuery) -> str:
    return "answerable" if query.answerability == "answerable" else "benchmark_reference_missing"


def _llm_metadata(response: LLMResponse | None) -> dict[str, object]:
    return {
        "done_reason": response.done_reason if response is not None else None,
        "prompt_eval_count": response.prompt_eval_count if response is not None else None,
        "eval_count": response.eval_count if response is not None else None,
    }


def _has_output_contract_violation(
    diagnostics: dict[str, bool | int],
) -> bool:
    return any(
        bool(diagnostics[key])
        for key in (
            "answer_source_id_leak",
            "answer_embedded_citation_list",
            "answer_over_120_words",
            "decision_content_mismatch",
        )
    )


def _prepare_output(
    *,
    parser: argparse.ArgumentParser,
    config_path: Path,
    results_path: Path,
    config_payload: dict[str, Any],
    resume: bool,
) -> None:
    if results_path.exists() and not resume:
        parser.error(f"results already exist: {results_path}; use --resume or a new --output-name")

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

        if existing != config_payload:
            parser.error("existing evaluation config does not match this run")
    else:
        config_path.write_text(
            json.dumps(
                config_payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _append_jsonl(
    path: Path,
    value: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                value,
                ensure_ascii=False,
            )
        )
        file.write("\n")
        file.flush()


def _write_jsonl(
    path: Path,
    values: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for value in values:
            file.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    values: list[dict[str, Any]] = []

    with path.open(
        encoding="utf-8",
    ) as file:
        for line_num, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_num}: invalid JSONL checkpoint") from error

            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_num}: record must be an object")

            values.append(value)

    return values


def _load_completed_query_ids(
    path: Path,
) -> set[str]:
    return {str(value["query_id"]) for value in _load_jsonl(path)}


def _jsonable(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.output_name.strip() or Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a non-empty path segment")

    for name in (
        "limit_answerable",
        "limit_unanswerable",
    ):
        value = getattr(args, name)

        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if not args.llm_model.strip():
        parser.error("--llm-model must be non-empty")

    if args.llm_timeout_seconds <= 0:
        parser.error("--llm-timeout-seconds must be positive")

    if args.llm_retries < 0:
        parser.error("--llm-retries must be non-negative")

    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature must be finite and non-negative")


if __name__ == "__main__":
    main()
