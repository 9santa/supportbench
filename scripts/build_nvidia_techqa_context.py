import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from supportbench.chunking.base import HuggingFaceTokenCodec
from supportbench.chunking.loaders import load_chunks
from supportbench.data.loaders import load_documents
from supportbench.rag.chunk_context_builder import RepresentativeChunkContextBuilder
from supportbench.rag.chunk_retrieval_pipeline import (
    RepresentativeChunkRetrievalPipeline,
)
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.rag.parent_pipeline import ParentContextPipeline, ParentContextRun
from supportbench.rag.parent_retrieval import ParentRetrievalOrchestrator
from supportbench.reranking.cross_encoder import SentenceTransformerCrossEncoderReranker
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource
from supportbench.retrieval.parent_hybrid import ParentWeightedRRFHybrid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def build_parser(
    *,
    description: str = "Build a token-budgeted RAG context from fused parent retrieval.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument("query")
    parser.add_argument("--chunk-config", default=DEFAULT_CHUNK_CONFIG)
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "nvidia_techqa" / "chunks",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "indexes" / "nvidia_techqa",
    )
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--context-tokenizer",
        default=DEFAULT_DENSE_MODEL,
        help="Hugging Face tokenizer used to enforce the context token budget",
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
    parser.add_argument("--candidate-prior-weight", type=float, default=1.25)
    parser.add_argument("--fusion-rrf-k", type=int, default=10)
    parser.add_argument("--minimum-overlap-tokens", type=int, default=8)
    parser.add_argument("--maximum-overlap-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    validate_output_path(parser, args)

    try:
        pipeline = build_context_pipeline(args)
        run = pipeline.run(args.query)
    except ValueError as error:
        parser.error(str(error))

    context = run.context

    print(f"Query: {args.query}")
    print("Pipeline: parent WRRF -> independent cross-encoder -> fusion")
    print(f"Retrieved parents: {len({chunk.parent_doc_id for chunk in run.retrieved_chunks})}")
    print(f"Representative chunks: {len(run.retrieved_chunks)}")
    print(f"Context parents: {len(context.documents)}")
    print(f"Context chunks: {len(context.provenance)}")
    print(f"Context tokens: {context.token_count:,} / {args.max_context_tokens:,}")
    print(f"Truncated: {str(context.truncated).lower()}")
    print()
    print(context.formatted_text)

    if args.output is not None:
        save_json(args.output, parent_context_payload(args, run))
        print()
        print(f"Saved: {args.output}")


def build_context_pipeline(args: argparse.Namespace) -> ParentContextPipeline:
    chunk_directory = args.chunks_root / args.chunk_config
    runtime_documents = load_documents(chunk_directory / "documents.jsonl")
    chunks_by_id = load_chunks(chunk_directory / "chunks.jsonl")
    runtime_ids = {document.doc_id for document in runtime_documents}

    if runtime_ids != set(chunks_by_id):
        raise ValueError("documents.jsonl and chunks.jsonl contain different chunk IDs")

    factory = RetrieverFactory(
        runtime_documents,
        config=RetrieverConfig(
            dense_index_path=args.index_root / args.chunk_config,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=0.5,
            bm25_b=1.0,
        ),
    )
    parent_wrrf = ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", factory.create("bm25"), 1.0),
            WeightedRetrieverSource("dense", factory.create("dense"), 1.5),
        ),
        parent_by_chunk_id={
            chunk_id: chunk.document_id for chunk_id, chunk in chunks_by_id.items()
        },
        source_candidate_k=args.source_candidate_k,
        rrf_k=10,
        aggregation="capped_top_2_sum",
        representative_chunks_per_parent=args.chunks_per_parent,
    )
    parent_by_chunk_id = {chunk_id: chunk.document_id for chunk_id, chunk in chunks_by_id.items()}
    chunk_store = InMemoryDocumentStore(runtime_documents)
    reranker = SentenceTransformerCrossEncoderReranker(
        args.reranker_model,
        device=args.reranker_device,
        batch_size=args.reranker_batch_size,
        max_length=512,
    )
    retrieval_orchestrator = ParentRetrievalOrchestrator(
        parent_retriever=parent_wrrf,
        reranker=reranker,
        chunk_store=chunk_store,
        parent_by_chunk_id=parent_by_chunk_id,
        parent_candidate_k=args.parent_candidate_k,
        chunks_per_parent=args.chunks_per_parent,
        candidate_prior_weight=args.candidate_prior_weight,
        fusion_rrf_k=args.fusion_rrf_k,
        second_evidence_weight=0.0,
    )
    chunk_pipeline = RepresentativeChunkRetrievalPipeline(
        chunk_store=chunk_store,
        chunks_by_id=chunks_by_id,
    )
    token_codec = HuggingFaceTokenCodec.from_pretrained(args.context_tokenizer)
    context_builder = RepresentativeChunkContextBuilder(
        token_codec=token_codec,
        max_tokens=args.max_context_tokens,
        max_parents=args.top_parents,
        minimum_token_overlap=args.minimum_overlap_tokens,
        maximum_token_overlap=args.maximum_overlap_tokens,
    )

    return ParentContextPipeline(
        retrieval_orchestrator=retrieval_orchestrator,
        chunk_pipeline=chunk_pipeline,
        context_builder=context_builder,
        top_parents=args.top_parents,
    )


def parent_context_payload(
    args: argparse.Namespace,
    run: ParentContextRun,
) -> dict[str, object]:
    retrieval = run.retrieval
    return {
        "query": args.query,
        "chunk_config": args.chunk_config,
        "retrieval": {
            "source_candidate_k": args.source_candidate_k,
            "parent_candidate_k": args.parent_candidate_k,
            "chunks_per_parent": args.chunks_per_parent,
            "top_parents": args.top_parents,
            "candidate_prior_weight": args.candidate_prior_weight,
            "fusion_rrf_k": args.fusion_rrf_k,
        },
        "retrieval_run": {
            "candidate_parents": [asdict(result) for result in retrieval.candidate_parents],
            "representative_chunks_by_parent": dict(retrieval.representative_chunks_by_parent),
            "reranked_parents": [asdict(result) for result in retrieval.reranked_parents],
            "fused_parents": [asdict(result) for result in retrieval.fused_parents],
        },
        "max_context_tokens": args.max_context_tokens,
        "context_tokenizer": args.context_tokenizer,
        "context": asdict(run.context),
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.query.strip():
        parser.error("query must be non-empty")

    if not args.chunk_config.strip() or Path(args.chunk_config).name != args.chunk_config:
        parser.error("--chunk-config must be a non-empty path segment")

    if not args.context_tokenizer.strip():
        parser.error("--context-tokenizer must be non-empty")

    positive_arguments = (
        "dense_batch_size",
        "reranker_batch_size",
        "source_candidate_k",
        "parent_candidate_k",
        "chunks_per_parent",
        "top_parents",
        "max_context_tokens",
        "fusion_rrf_k",
        "minimum_overlap_tokens",
        "maximum_overlap_tokens",
    )

    for name in positive_arguments:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.top_parents > args.parent_candidate_k:
        parser.error("--top-parents must not exceed --parent-candidate-k")

    if args.source_candidate_k < args.parent_candidate_k:
        parser.error("--source-candidate-k must be at least --parent-candidate-k")

    if args.maximum_overlap_tokens < args.minimum_overlap_tokens:
        parser.error("--maximum-overlap-tokens must be at least --minimum-overlap-tokens")

    if not math.isfinite(args.candidate_prior_weight) or args.candidate_prior_weight < 0.0:
        parser.error("--candidate-prior-weight must be finite and non-negative")


def validate_output_path(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite to replace it")


if __name__ == "__main__":
    main()
