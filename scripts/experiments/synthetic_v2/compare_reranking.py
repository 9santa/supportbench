import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.experiments.synthetic_v2.reranker_comparison import (
    RRFProfile,
    run_reranker_comparison,
)
from supportbench.experiments.synthetic_v2.reranker_report import (
    export_reranker_comparison,
    render_reranker_comparison,
)
from supportbench.reranking.factory import (
    CrossEncoderConfig,
    RerankingFactory,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "queries_dev.jsonl"
DEFAULT_DENSE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "synthetic"
    / "v2"
    / "dense"
    / "multilingual-e5-base"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "synthetic" / "v2" / "reranker" / "summary.json"


@dataclass(frozen=True, slots=True)
class CliArguments:
    documents_path: Path
    queries_path: Path
    split: str
    output_path: Path

    bm25_k1: float
    bm25_b: float

    dense_index_path: Path
    dense_model_name: str
    dense_device: str
    dense_batch_size: int

    reranker_model_name: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int

    source_candidate_k: int
    reranker_candidate_k: int
    final_top_k: int

    standalone_dense_weight: float
    standalone_rrf_k: int

    candidate_dense_weight: float
    candidate_rrf_k: int


def parse_args() -> CliArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Dense and Weighted RRF "
            "candidate sources before and after "
            "cross-encoder reranking."
        )
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
    )
    parser.add_argument(
        "--split",
        default="dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_DENSE_INDEX_PATH,
    )
    parser.add_argument(
        "--dense-model",
        default=("intfloat/multilingual-e5-base"),
    )
    parser.add_argument(
        "--dense-device",
        default="cuda",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--reranker-model",
        default=("BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--reranker-device",
        default="cuda",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--source-candidate-k",
        type=int,
        default=100,
        help=("documents requested from BM25 and Dense inside RRF"),
    )
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=50,
        help=("fused documents passed to the cross-encoder"),
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--standalone-dense-weight",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--standalone-rrf-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--candidate-dense-weight",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--candidate-rrf-k",
        type=int,
        default=10,
    )

    namespace = parser.parse_args()

    arguments = CliArguments(
        documents_path=cast(
            Path,
            namespace.documents,
        ),
        queries_path=cast(
            Path,
            namespace.queries,
        ),
        split=cast(str, namespace.split),
        output_path=cast(
            Path,
            namespace.output,
        ),
        bm25_k1=cast(
            float,
            namespace.bm25_k1,
        ),
        bm25_b=cast(
            float,
            namespace.bm25_b,
        ),
        dense_index_path=cast(
            Path,
            namespace.dense_index,
        ),
        dense_model_name=cast(
            str,
            namespace.dense_model,
        ),
        dense_device=cast(
            str,
            namespace.dense_device,
        ),
        dense_batch_size=cast(
            int,
            namespace.dense_batch_size,
        ),
        reranker_model_name=cast(
            str,
            namespace.reranker_model,
        ),
        reranker_device=cast(
            str,
            namespace.reranker_device,
        ),
        reranker_batch_size=cast(
            int,
            namespace.reranker_batch_size,
        ),
        reranker_max_length=cast(
            int,
            namespace.reranker_max_length,
        ),
        source_candidate_k=cast(
            int,
            namespace.source_candidate_k,
        ),
        reranker_candidate_k=cast(
            int,
            namespace.reranker_candidate_k,
        ),
        final_top_k=cast(
            int,
            namespace.final_top_k,
        ),
        standalone_dense_weight=cast(
            float,
            namespace.standalone_dense_weight,
        ),
        standalone_rrf_k=cast(
            int,
            namespace.standalone_rrf_k,
        ),
        candidate_dense_weight=cast(
            float,
            namespace.candidate_dense_weight,
        ),
        candidate_rrf_k=cast(
            int,
            namespace.candidate_rrf_k,
        ),
    )

    _validate_args(parser, arguments)

    return arguments


def _validate_args(
    parser: argparse.ArgumentParser,
    args: CliArguments,
) -> None:
    for name, value in (
        ("--bm25-k1", args.bm25_k1),
        (
            "--standalone-dense-weight",
            args.standalone_dense_weight,
        ),
        (
            "--candidate-dense-weight",
            args.candidate_dense_weight,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")

    if not math.isfinite(args.bm25_b) or not 0.0 <= args.bm25_b <= 1.0:
        parser.error("--bm25-b must be between 0 and 1")

    for name, value in (
        (
            "--dense-batch-size",
            args.dense_batch_size,
        ),
        (
            "--reranker-batch-size",
            args.reranker_batch_size,
        ),
        (
            "--reranker-max-length",
            args.reranker_max_length,
        ),
        (
            "--source-candidate-k",
            args.source_candidate_k,
        ),
        (
            "--reranker-candidate-k",
            args.reranker_candidate_k,
        ),
        (
            "--final-top-k",
            args.final_top_k,
        ),
        (
            "--standalone-rrf-k",
            args.standalone_rrf_k,
        ),
        (
            "--candidate-rrf-k",
            args.candidate_rrf_k,
        ),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")

    if args.reranker_candidate_k > args.source_candidate_k:
        parser.error("--reranker-candidate-k must not be greater than --source-candidate-k")

    if args.final_top_k > args.reranker_candidate_k:
        parser.error("--final-top-k must not be greater than --reranker-candidate-k")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    retrieval_factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=args.dense_index_path,
            dense_model_name=args.dense_model_name,
            dense_device=args.dense_device,
            dense_batch_size=args.dense_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        ),
    )

    queries = load_queries(
        args.queries_path,
        {document.doc_id for document in documents},
    )

    selected_queries = [query for query in queries if query.split == args.split]

    if not selected_queries:
        raise SystemExit(f"no queries found for split {args.split!r}")

    reranking_factory = RerankingFactory(
        documents,
        config=CrossEncoderConfig(
            model_name=(args.reranker_model_name),
            device=args.reranker_device,
            batch_size=(args.reranker_batch_size),
            max_length=(args.reranker_max_length),
        ),
    )

    result = run_reranker_comparison(
        queries=selected_queries,
        bm25=retrieval_factory.create("bm25"),
        dense=retrieval_factory.create("dense"),
        reranking_factory=reranking_factory,
        standalone_profile=RRFProfile(
            name="rrf_standalone",
            bm25_weight=1.0,
            dense_weight=(args.standalone_dense_weight),
            rrf_k=args.standalone_rrf_k,
            candidate_k=(args.source_candidate_k),
        ),
        candidate_profile=RRFProfile(
            name="rrf_candidate",
            bm25_weight=1.0,
            dense_weight=(args.candidate_dense_weight),
            rrf_k=args.candidate_rrf_k,
            candidate_k=(args.source_candidate_k),
        ),
        reranker_candidate_k=(args.reranker_candidate_k),
        final_top_k=args.final_top_k,
    )

    print(render_reranker_comparison(result))

    export_reranker_comparison(
        result,
        path=args.output_path,
    )

    print()
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
