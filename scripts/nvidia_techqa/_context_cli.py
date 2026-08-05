import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.applications.nvidia_techqa import (
    DEFAULT_CHUNK_CONFIG,
    DEFAULT_DENSE_MODEL,
    DEFAULT_GENERATION_TOKENIZER,
    DEFAULT_RERANKER_MODEL,
    NvidiaTechQAContextConfig,
)
from supportbench.rag.context import ContextPreparationRun


def add_context_config_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--context-tokenizer",
        default=DEFAULT_GENERATION_TOKENIZER,
        help="Hugging Face tokenizer matching the generation model",
    )
    parser.add_argument("--dense-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--source-candidate-k", type=int, default=500)
    parser.add_argument("--parent-candidate-k", type=int, default=20)
    parser.add_argument("--chunks-per-parent", type=int, default=2)
    parser.add_argument("--top-parents", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=4_096)
    parser.add_argument("--model-context-window", type=int, default=8_192)
    parser.add_argument("--reserved-output-tokens", type=int, default=512)
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--fusion-rrf-k", type=int, default=10)
    parser.add_argument("--minimum-overlap-tokens", type=int, default=8)
    parser.add_argument("--maximum-overlap-tokens", type=int, default=256)


def add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    add_context_config_arguments(parser)

    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")


def context_config_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> NvidiaTechQAContextConfig:
    try:
        return NvidiaTechQAContextConfig(
            chunks_root=args.chunks_root,
            index_root=args.index_root,
            chunk_config=args.chunk_config,
            dense_model_name=args.dense_model,
            reranker_model_name=args.reranker_model,
            context_tokenizer_name=args.context_tokenizer,
            dense_device=args.dense_device,
            reranker_device=args.reranker_device,
            dense_batch_size=args.dense_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            source_candidate_k=args.source_candidate_k,
            parent_candidate_k=args.parent_candidate_k,
            chunks_per_parent=args.chunks_per_parent,
            top_parents=args.top_parents,
            max_context_tokens=args.max_context_tokens,
            model_context_window=args.model_context_window,
            reserved_output_tokens=args.reserved_output_tokens,
            candidate_prior_weight=args.candidate_prior_weight,
            fusion_rrf_k=args.fusion_rrf_k,
            minimum_overlap_tokens=args.minimum_overlap_tokens,
            maximum_overlap_tokens=args.maximum_overlap_tokens,
        )
    except ValueError as error:
        parser.error(str(error))


def parse_context_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> NvidiaTechQAContextConfig:
    if not args.query.strip():
        parser.error("query must be non-empty")

    return context_config_from_args(parser, args)


def validate_output_path(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite to replace it")


def parent_context_payload(
    *,
    query: str,
    config: NvidiaTechQAContextConfig,
    run: ContextPreparationRun,
) -> dict[str, object]:
    retrieval = run.retrieval
    return {
        "query": query,
        "chunk_config": config.chunk_config,
        "retrieval": {
            "source_candidate_k": config.source_candidate_k,
            "parent_candidate_k": config.parent_candidate_k,
            "chunks_per_parent": config.chunks_per_parent,
            "top_parents": config.top_parents,
            "candidate_prior_weight": config.candidate_prior_weight,
            "fusion_rrf_k": config.fusion_rrf_k,
            "bm25_weight": config.bm25_weight,
            "dense_weight": config.dense_weight,
            "source_rrf_k": config.source_rrf_k,
            "parent_aggregation": config.parent_aggregation,
        },
        "retrieval_run": {
            "candidate_parents": [asdict(result) for result in retrieval.candidate_parents],
            "representative_chunks_by_parent": dict(retrieval.representative_chunks_by_parent),
            "reranked_parents": [asdict(result) for result in retrieval.reranked_parents],
            "fused_parents": [asdict(result) for result in retrieval.fused_parents],
        },
        "max_context_tokens": config.max_context_tokens,
        "context_tokenizer": config.context_tokenizer_name,
        "prompt_budget": (asdict(run.prompt_budget) if run.prompt_budget is not None else None),
        "prompt_token_count": run.prompt_token_count,
        "context": asdict(run.context),
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
