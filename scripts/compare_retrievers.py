import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from collections.abc import Sequence

from scripts.evaluate_retrieval import select_queries
from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.data.models import QueryExample, Document
from supportbench.evaluation.retrieval_analysis import (
    QueryComparison,
    compare_evaluation_results,
)
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever
from supportbench.retrieval.dense import DenseRetriever
from supportbench.retrieval.dense_build import (
    compute_document_fingerprint,
)
from supportbench.retrieval.dense_encoder import SentenceTransformerDenceEncoder
from supportbench.retrieval.dense_index import FaissFlatVectorIndex

type RetrieverName = Literal["tfidf", "bm25", "dense"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_dev.jsonl"


@dataclass(frozen=True, slots=True)
class CLIArguments:
    retrievers: tuple[RetrieverName, ...]
    split: str
    documents_path: Path
    queries_path: Path
    top_k: int
    failure_k: int



@dataclass(frozen=True, slots=True)
class ComparisonStats:
    query_count: int
    all_failed_count: int
    all_succeeded_count: int
    mixed_count: int
    better_counts: dict[str, int]
    tied_count: int


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser(description="Compare retrieval algorithms query by query.")

    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=("tfidf", "bm25", "dense"),
        default=("tfidf", "bm25", "dense"),
        help=("retrieval algorithms to compare (default: tfidf bm25 dense)"),
    )

    parser.add_argument("--split", default="dev", help="query split to evaluate (default: dev)")

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
        default=10,
        help=("number of retrieved documents per query (default: 10)"),
    )

    parser.add_argument(
        "--failure-k",
        type=int,
        choices=(1, 3, 5),
        default=3,
        help=("cutoff used to classify retrieval failures (default: 3)"),
    )

    args = parser.parse_args()

    retrievers = tuple(cast(list[RetrieverName], args.retrievers))
    top_k = cast(int, args.top_k)
    failure_k = cast(int, args.failure_k)

    if len(retrievers) < 2:
        parser.error("at least two retrievers must be provided")

    if len(set(retrievers)) != len(retrievers):
        parser.error("retrievers must be different")

    if top_k < 10:
        parser.error("--top-k must be at least 10 to compute Recall@10")

    if failure_k > top_k:
        parser.error("--failure-k must not be greater than --top-k")

    return CLIArguments(
        retrievers=retrievers,
        split=cast(str, args.split),
        documents_path=cast(Path, args.documents),
        queries_path=cast(Path, args.queries),
        top_k=top_k,
        failure_k=failure_k,
    )


def create_retriever(
    name: RetrieverName,
    *,
    documents: Sequence[Document],
) -> Retriever:
    if name == "tfidf":
        index = InvertedIndex.build(documents)
        return TfidfRetriever(index)

    if name == "bm25":
        index = InvertedIndex.build(documents)
        return BM25Retriever(index)

    if name == "dense":
        fingerprint = compute_document_fingerprint(documents)

        vector_index = FaissFlatVectorIndex.load(
            (PROJECT_ROOT / "artifacts" / "dense" / "multilingual-e5-base"),
            expected_document_fingerprint=fingerprint,
            expected_model_name="intfloat/multilingual-e5-base",
        )

        encoder = SentenceTransformerDenceEncoder(
            "intfloat/multilingual-e5-base",
            device="cuda",
            batch_size=16,
        )

        return DenseRetriever(
            encoder,
            vector_index,
        )

    raise ValueError(f"unknown retriever: {name!r}")



def evaluate_retrievers(
    retriever_names: tuple[RetrieverName, ...],
    *,
    index: InvertedIndex,
    queries: list[QueryExample],
    top_k: int,
    documents: Sequence[Document],
) -> dict[str, RetrievalEvaluationResult]:
    results: dict[str, RetrievalEvaluationResult] = {}

    for name in retriever_names:
        retriever = create_retriever(name, documents=documents)

        results[name] = evaluate_retriever(retriever, queries, top_k=top_k)

    return results


def print_metric_table(results: dict[str, RetrievalEvaluationResult]) -> None:
    print("Aggregate metrics:")
    print()
    print(f"{'Retriever':<12}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>10}")

    for retriever_name, result in results.items():
        print(
            f"{retriever_name:<20}"
            f"{result.recall_at_1:>9.4f}"
            f"{result.recall_at_3:>9.4f}"
            f"{result.recall_at_5:>9.4f}"
            f"{result.recall_at_10:>9.4f}"
            f"{result.mrr:>9.4f}"
        )


def print_better(
    comparison: QueryComparison,
    *,
    winner: str,
) -> None:
    print()
    print(f"[{winner.upper()} BETTER] {comparison.query_id}")
    print(f"Query: {comparison.query}")
    print("Relevant: " + ", ".join(comparison.relevant_doc_ids))

    for item in comparison.evaluations:
        print(f"{item.retriever_name.upper()} rank: {item.evaluation.first_relevant_rank}")


def print_all_failed(
    comparison: QueryComparison,
    *,
    failure_k: int,
) -> None:
    print()
    print(f"[ALL FAILED@{failure_k}] {comparison.query_id}")
    print(f"Query: {comparison.query}")
    print("Relevant: " + ", ".join(comparison.relevant_doc_ids))

    for item in comparison.evaluations:
        retrieved = ", ".join(item.evaluation.retrieved_doc_ids)

        if not retrieved:
            retrieved = "no results"

        print(f"{item.retriever_name.upper()}: {retrieved}")


def calculate_stats(
    comparisons: tuple[QueryComparison, ...],
    *,
    retriever_names: tuple[RetrieverName, ...],
    failure_k: int,
) -> ComparisonStats:
    better_counts = {retriever_name: 0 for retriever_name in retriever_names}

    all_failed_count = 0
    all_succeeded_count = 0
    mixed_count = 0
    tied_count = 0

    for comparison in comparisons:
        if comparison.all_failed(k=failure_k):
            all_failed_count += 1
        elif comparison.all_succeeded(k=failure_k):
            all_succeeded_count += 1
        else:
            mixed_count += 1

        best_retrievers = comparison.best_retrievers

        if len(best_retrievers) == 1:
            better_counts[best_retrievers[0]] += 1
        else:
            tied_count += 1

    return ComparisonStats(
        query_count=len(comparisons),
        all_failed_count=all_failed_count,
        all_succeeded_count=all_succeeded_count,
        mixed_count=mixed_count,
        better_counts=better_counts,
        tied_count=tied_count,
    )


def print_comparison_details(
    comparisons: tuple[QueryComparison, ...],
    *,
    failure_k: int,
) -> None:
    print()
    print("Per-query differences:")

    printed_anything = False

    for comparison in comparisons:
        if comparison.all_failed(k=failure_k):
            print_all_failed(
                comparison,
                failure_k=failure_k,
            )
            printed_anything = True
            continue

        best_retrievers = comparison.best_retrievers

        if len(best_retrievers) == 1:
            print_better(
                comparison,
                winner=best_retrievers[0],
            )
            printed_anything = True

    if not printed_anything:
        print()
        print("No ranking differences or shared failures found.")


def print_summary(
    statistics: ComparisonStats,
    *,
    retriever_names: tuple[RetrieverName, ...],
    failure_k: int,
) -> None:
    print()
    print("Comparison summary:")
    print(f"Queries: {statistics.query_count}")
    print(f"All succeeded@{failure_k}: {statistics.all_succeeded_count}")
    print(f"Mixed results@{failure_k}: {statistics.mixed_count}")
    print(f"All failed@{failure_k}: {statistics.all_failed_count}")

    for retriever_name in retriever_names:
        print(
            f"{retriever_name.upper()} better by RR in: {statistics.better_counts[retriever_name]} "
            f"queries"
        )

    print(f"Tied by RR: {statistics.tied_count}")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    index = InvertedIndex.build(documents)

    queries = load_queries(args.queries_path, set(index.document_ids))

    selected_queries = select_queries(
        queries,
        split=args.split,
    )

    if not select_queries:
        raise SystemExit(f"no queries found for split {args.split!r}")

    evaluation_results = evaluate_retrievers(
        args.retrievers,
        index=index,
        queries=selected_queries,
        top_k=args.top_k,
        documents=documents,
    )

    comparisons = compare_evaluation_results(evaluation_results)

    stats = calculate_stats(
        comparisons,
        retriever_names=args.retrievers,
        failure_k=args.failure_k,
    )

    print("Retrievers: " + ", ".join(args.retrievers))
    print(f"Split: {args.split}")
    print(f"Failure cutoff: top {args.failure_k}")
    print()

    print_metric_table(evaluation_results)

    print_comparison_details(
        comparisons,
        failure_k=args.failure_k,
    )

    print_summary(
        stats,
        retriever_names=args.retrievers,
        failure_k=args.failure_k,
    )


if __name__ == "__main__":
    main()
