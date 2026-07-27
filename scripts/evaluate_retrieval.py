import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.data.models import Document, QueryExample
from supportbench.evaluation.retrieval_analysis import failures_at_k
from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.dense import DenseRetriever
from supportbench.retrieval.dense_build import (
    compute_document_fingerprint,
)
from supportbench.retrieval.dense_encoder import SentenceTransformerDenceEncoder
from supportbench.retrieval.dense_index import FaissFlatVectorIndex
from supportbench.retrieval.hybrid import WeightedRRFHybrid, WeightedRetrieverSource
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever

type RetrieverName = Literal[
    "tfidf",
    "bm25",
    "dense",
    "hybrid",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_dev.jsonl"


@dataclass(frozen=True, slots=True)
class CliArguments:
    retriever: RetrieverName
    split: str
    documents_path: Path
    queries_path: Path
    top_k: int
    show_errors: bool
    failure_k: int
    dense_index_path: Path
    dense_model_name: str
    dense_device: str
    dense_batch_size: int
    bm25_weight: float
    dense_weight: float
    candidate_k: int
    rrf_k: int


def parse_args() -> CliArguments:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality.",
    )
    parser.add_argument(
        "--retriever",
        choices=("tfidf", "bm25", "dense", "hybrid"),
        default="bm25",
        help="retrieval algorithm (default: bm25)",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="query split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help=(f"path to documents.jsonl (default: {DEFAULT_DOCUMENTS_PATH})"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=(f"path to queries.jsonl (default: {DEFAULT_QUERIES_PATH})"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=("number of retrieved documents per query (default: 5)"),
    )

    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="show queries with no relevant document in top-k",
    )

    parser.add_argument(
        "--failure-k",
        type=int,
        choices=(1, 3, 5),
        default=3,
        help=("cutoff used to classify retrieval failures (default: 3)"),
    )

    parser.add_argument(
        "--dense-index",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "dense" / "multilingual-e5-base"),
    )

    parser.add_argument(
        "--dense-model",
        type=str,
        default="intfloat/multilingual-e5-base",
    )

    parser.add_argument(
        "--dense-device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="BM25 retriever weight in weighted RRF",
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense retriever weight in weighted RRF",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="candidate count requested from each retriever",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank smoothing constant",
    )

    args = parser.parse_args()
    top_k = cast(int, args.top_k)
    failure_k = cast(int, args.failure_k)

    if top_k < 5:
        parser.error("--top-k must be at least 5 to compute Recall@5")

    if failure_k > top_k:
        parser.error("--failure-k must not be greater than --top-k")

    if not math.isfinite(args.bm25_weight) or args.bm25_weight < 0.0:
        parser.error("--bm25-weight must be finite and non-negative")

    if not math.isfinite(args.dense_weight) or args.dense_weight < 0.0:
        parser.error("--dense-weight must be finite and non-negative")

    return CliArguments(
        retriever=cast(
            RetrieverName,
            args.retriever,
        ),
        split=cast(str, args.split),
        documents_path=cast(Path, args.documents),
        queries_path=cast(Path, args.queries),
        top_k=top_k,
        show_errors=cast(bool, args.show_errors),
        failure_k=failure_k,
        dense_index_path=cast(Path, args.dense_index),
        dense_model_name=cast(str, args.dense_model),
        dense_device=cast(str, args.dense_device),
        dense_batch_size=cast(int, args.dense_batch_size),
        bm25_weight=cast(float, args.bm25_weight),
        dense_weight=cast(float, args.dense_weight),
        candidate_k=cast(int, args.candidate_k),
        rrf_k=cast(int, args.rrf_k),
    )


def create_dense_retriver(
    args: CliArguments,
    *,
    documents: Sequence[Document],
) -> DenseRetriever:
    fingerprint = compute_document_fingerprint(documents)

    vector_index = FaissFlatVectorIndex.load(
        args.dense_index_path,
        expected_document_fingerprint=fingerprint,
        expected_model_name=args.dense_model_name,
    )

    encoder = SentenceTransformerDenceEncoder(
        args.dense_model_name,
        device=args.dense_device,
        batch_size=args.dense_batch_size,
    )

    return DenseRetriever(
        encoder,
        vector_index,
    )


def create_retriever(
    args: CliArguments,
    *,
    documents: Sequence[Document],
) -> Retriever:
    name: str = args.retriever
    if name == "tfidf":
        index = InvertedIndex.build(documents)
        return TfidfRetriever(index)

    if name == "bm25":
        index = InvertedIndex.build(documents)
        return BM25Retriever(index)

    if name == "dense":
        return create_dense_retriver(args, documents=documents)

    if name == "hybrid":
        index = InvertedIndex.build(documents)

        bm25 = BM25Retriever(
            index,
            k1=0.5,
            b=1.0,
        )

        dense = create_dense_retriver(args, documents=documents)

        return WeightedRRFHybrid(
            sources=(
                WeightedRetrieverSource(
                    name="bm25",
                    retriever=bm25,
                    weight=args.bm25_weight,
                ),
                WeightedRetrieverSource(
                    name="dense",
                    retriever=dense,
                    weight=args.dense_weight,
                ),
            ),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )

    raise ValueError(f"unknown retriever: {name!r}")


def select_queries(
    queries: list[QueryExample],
    *,
    split: str,
) -> list[QueryExample]:
    return [query for query in queries if query.split == split]


def print_evaluation(
    *,
    retriever_name: RetrieverName,
    split: str,
    result: RetrievalEvaluationResult,
) -> None:
    print(f"Retriever: {retriever_name}")
    print(f"Split: {split}")
    print(f"Queries: {result.query_count}")
    print()
    print(f"Recall@1: {result.recall_at_1:.4f}")
    print(f"Recall@3: {result.recall_at_3:.4f}")
    print(f"Recall@5: {result.recall_at_5:.4f}")
    print(f"Recall@10: {result.recall_at_10:.4f}")
    print(f"MRR:      {result.mrr:.4f}")


def print_failures(
    failures: tuple[QueryEvaluation, ...],
    *,
    failure_k: int,
) -> None:
    print()
    print(f"Failures: {len(failures)} (no relevant document in top {failure_k})")

    for failure in failures:
        print()
        print(f"[FAIL] {failure.query_id}")
        print(f"Query: {failure.query}")
        print("Relevant: " + ", ".join(failure.relevant_doc_ids))

        print("Retrieved:")
        if not failure.retrieved_doc_ids:
            print("  No results.")

        for rank, doc_id in enumerate(failure.retrieved_doc_ids, start=1):
            print(f"  {rank}. {doc_id}")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    document_ids = set(doc.doc_id for doc in documents)

    queries = load_queries(args.queries_path, document_ids)

    selected_queries = select_queries(
        queries,
        split=args.split,
    )

    if not selected_queries:
        raise SystemExit(f"no queries found for split {args.split!r}")

    retriever = create_retriever(
        args,
        documents=documents,
    )

    evaluation = evaluate_retriever(
        retriever,
        selected_queries,
        top_k=args.top_k,
    )

    print_evaluation(
        retriever_name=args.retriever,
        split=args.split,
        result=evaluation,
    )

    if args.show_errors:
        failures = failures_at_k(
            evaluation,
            k=args.failure_k,
        )

        print_failures(
            failures,
            failure_k=args.failure_k,
        )


if __name__ == "__main__":
    main()
