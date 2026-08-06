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
from supportbench.corpus.nvidia_techqa import (
    NvidiaTechQAOracleContext,
    load_nvidia_techqa_oracle_contexts,
)
from supportbench.evaluation.oracle_context import summarize_oracle_contexts
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
from supportbench.rag.models import RetrievedChunk
from supportbench.rag.retrieval import ParentRetrievalRun
from supportbench.retrieval.base import SearchResult

EVALUATION_VERSION = "techqa_oracle_context_v1"
CURRENT_SELECTION: EvidenceSelection = "within_parent_rerank"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired current, gold-injected, gold-only selected, and raw oracle "
            "contexts for answerable NVIDIA TechQA queries without calling an LLM."
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
        "--dataset-zip",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "raw" / "NvidiaTechQA.zip",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "nvidia_techqa"
            / "evaluations"
            / "oracle_context"
        ),
    )
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--resume", action="store_true")
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
    config_payload = {
        "evaluation_version": EVALUATION_VERSION,
        "split": args.split,
        "limit": args.limit,
        "modes": [
            "current",
            "gold_injected",
            "gold_only_selected",
            "oracle_source",
        ],
        "current_evidence_selection": CURRENT_SELECTION,
        "gold_injection": "rank_1_then_keep_current_parents_up_to_top_k",
        "context": jsonable(asdict(config)),
        "fingerprints": {
            "queries_sha256": sha256_file(args.queries),
            "dataset_zip_sha256": sha256_file(args.dataset_zip),
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
        for query in load_benchmark_queries(
            args.queries,
            known_doc_ids=known_parent_ids,
        )
        if query.split == args.split and query.answerability == "answerable"
    ]

    if args.limit is not None:
        queries = queries[: args.limit]

    oracle_contexts = load_nvidia_techqa_oracle_contexts(args.dataset_zip)
    completed_query_ids = load_completed_query_ids(results_path) if args.resume else set()
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
            "stage": "oracle_context_evaluation",
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
                result = compare_oracle_contexts(
                    query=query,
                    source_contexts=oracle_contexts[query.query_id],
                    context_service=context_service,
                    top_parents=config.top_parents,
                )
                append_jsonl(results_path, result)
                references = [
                    result.get(mode, {}).get("reference_answer_in_context")
                    for mode in (
                        "current",
                        "gold_injected",
                        "gold_only_selected",
                        "oracle_source",
                    )
                ]
                print(
                    f"[{index:,}/{len(pending_queries):,}] {query.query_id}: "
                    f"{result['status']} reference {references}",
                    flush=True,
                )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            results = load_jsonl(results_path)
            summary = summarize_oracle_contexts(results)
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
                    "chunk_config": config.chunk_config,
                    "current_evidence_selection": CURRENT_SELECTION,
                    "top_parents": config.top_parents,
                    "chunks_per_parent": config.chunks_per_parent,
                    "max_context_tokens": config.max_context_tokens,
                    "reserved_output_tokens": config.reserved_output_tokens,
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


def compare_oracle_contexts(
    *,
    query: BenchmarkQuery,
    source_contexts: tuple[NvidiaTechQAOracleContext, ...],
    context_service: ContextPreparationService,
    top_parents: int,
) -> dict[str, Any]:
    if {context.document_id for context in source_contexts} != set(
        query.relevant_doc_ids
    ):
        raise ValueError(f"oracle contexts do not match gold parents for {query.query_id}")

    try:
        current_started = time.perf_counter()
        current = context_service.prepare(
            query.query,
            evidence_selection=CURRENT_SELECTION,
        )
        current_latency_ms = _elapsed_ms(current_started)

        injected_retrieval = _force_gold_parents(
            current.retrieval,
            gold_parent_ids=query.relevant_doc_ids,
            top_k=top_parents,
        )
        injected_started = time.perf_counter()
        gold_injected = context_service.prepare(
            query.query,
            retrieval=injected_retrieval,
            evidence_selection=CURRENT_SELECTION,
        )
        gold_injected_latency_ms = _elapsed_ms(injected_started)

        gold_only_retrieval = _gold_only_retrieval(
            injected_retrieval,
            gold_parent_ids=query.relevant_doc_ids,
        )
        gold_only_chunks = tuple(
            chunk
            for chunk in gold_injected.retrieved_chunks
            if chunk.parent_doc_id in query.relevant_doc_ids
        )
        gold_only_started = time.perf_counter()
        gold_only = context_service.prepare_from_chunks(
            query.query,
            retrieved_chunks=gold_only_chunks,
            retrieval=gold_only_retrieval,
        )
        gold_only_latency_ms = _elapsed_ms(gold_only_started)

        oracle_retrieval = _oracle_retrieval(source_contexts)
        oracle_started = time.perf_counter()
        oracle = context_service.prepare_from_chunks(
            query.query,
            retrieved_chunks=_oracle_chunks(source_contexts),
            retrieval=oracle_retrieval,
        )
        oracle_latency_ms = _elapsed_ms(oracle_started)
    except ValueError as error:
        return {
            "query_id": query.query_id,
            "query": query.query,
            "split": query.split,
            "status": "context_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    oracle_source_text = "\n\n".join(context.source_text for context in source_contexts)

    return {
        "query_id": query.query_id,
        "query": query.query,
        "split": query.split,
        "status": "success",
        "reference_answer": query.reference_answer,
        "relevant_doc_ids": list(query.relevant_doc_ids),
        "retrieval": {
            "candidate_parent_ids": [
                result.doc_id for result in current.retrieval.candidate_parents
            ],
            "reranked_parent_ids": [
                result.doc_id for result in current.retrieval.reranked_parents
            ],
            "fused_parent_ids": [
                result.doc_id for result in current.retrieval.fused_parents
            ],
        },
        "current": _context_payload(query=query, run=current),
        "gold_injected": _context_payload(query=query, run=gold_injected),
        "gold_only_selected": _context_payload(query=query, run=gold_only),
        "oracle_source": {
            **_context_payload(query=query, run=oracle),
            "source_filenames": [
                context.source_filename for context in source_contexts
            ],
            "source_char_count": len(oracle_source_text),
            "reference_in_full_source": reference_is_in_text(
                query.reference_answer,
                oracle_source_text,
            ),
        },
        "latency_ms": {
            "current_retrieval_and_context": current_latency_ms,
            "gold_injected_context": gold_injected_latency_ms,
            "gold_only_context": gold_only_latency_ms,
            "oracle_source_context": oracle_latency_ms,
        },
        "error_type": None,
        "error_message": None,
    }


def _force_gold_parents(
    retrieval: ParentRetrievalRun,
    *,
    gold_parent_ids: tuple[str, ...],
    top_k: int,
) -> ParentRetrievalRun:
    current_by_id = {result.doc_id: result for result in retrieval.fused_parents}
    fallback_score = retrieval.fused_parents[0].score if retrieval.fused_parents else 1.0
    parent_ids = list(gold_parent_ids)
    parent_ids.extend(
        result.doc_id
        for result in retrieval.fused_parents
        if result.doc_id not in gold_parent_ids
    )
    fused = tuple(
        SearchResult(
            doc_id=parent_id,
            score=current_by_id.get(
                parent_id,
                SearchResult(parent_id, fallback_score, 1),
            ).score,
            rank=rank,
        )
        for rank, parent_id in enumerate(parent_ids[:top_k], start=1)
    )
    return ParentRetrievalRun(
        candidate_parents=retrieval.candidate_parents,
        representative_chunks_by_parent=retrieval.representative_chunks_by_parent,
        reranked_parents=retrieval.reranked_parents,
        fused_parents=fused,
    )


def _gold_only_retrieval(
    retrieval: ParentRetrievalRun,
    *,
    gold_parent_ids: tuple[str, ...],
) -> ParentRetrievalRun:
    fused_by_id = {result.doc_id: result for result in retrieval.fused_parents}
    fused = tuple(
        SearchResult(
            doc_id=parent_id,
            score=fused_by_id[parent_id].score,
            rank=rank,
        )
        for rank, parent_id in enumerate(gold_parent_ids, start=1)
    )
    return ParentRetrievalRun(
        candidate_parents=retrieval.candidate_parents,
        representative_chunks_by_parent=retrieval.representative_chunks_by_parent,
        reranked_parents=retrieval.reranked_parents,
        fused_parents=fused,
    )


def _oracle_retrieval(
    source_contexts: tuple[NvidiaTechQAOracleContext, ...],
) -> ParentRetrievalRun:
    fused = tuple(
        SearchResult(context.document_id, 1.0, rank)
        for rank, context in enumerate(source_contexts, start=1)
    )
    return ParentRetrievalRun(
        candidate_parents=(),
        representative_chunks_by_parent={},
        reranked_parents=(),
        fused_parents=fused,
    )


def _oracle_chunks(
    source_contexts: tuple[NvidiaTechQAOracleContext, ...],
) -> tuple[RetrievedChunk, ...]:
    return tuple(
        RetrievedChunk(
            chunk_id=f"{context.document_id}::oracle_source",
            parent_doc_id=context.document_id,
            document_title=context.title,
            text=context.text,
            category="technical_support",
            section_path=("Oracle source context",),
            ordinal=0,
            start_char=None,
            end_char=None,
            parent_score=1.0,
            parent_rank=rank,
            evidence_rank=1,
        )
        for rank, context in enumerate(source_contexts, start=1)
    )


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
        "gold_document_in_context": bool(
            set(query.relevant_doc_ids) & set(context_parent_ids)
        ),
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


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000.0


if __name__ == "__main__":
    main()
