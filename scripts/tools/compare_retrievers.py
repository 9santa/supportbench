import argparse
from dataclasses import dataclass
from typing import cast

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_analysis import (
    QueryComparison,
    compare_evaluation_results,
)
from supportbench.evaluation.retrieval_cli import (
    EvaluationArguments,
    add_evaluation_arguments,
    add_retriever_config_arguments,
    load_evaluation_data,
    parse_evaluation_arguments,
    parse_retriever_config,
)
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.factory import (
    RETRIEVER_NAMES,
    RetrieverConfig,
    RetrieverFactory,
    RetrieverName,
)


@dataclass(frozen=True, slots=True)
class CLIArguments:
    retrievers: tuple[RetrieverName, ...]
    evaluation: EvaluationArguments
    retriever_config: RetrieverConfig


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
        choices=RETRIEVER_NAMES,
        default=("tfidf", "bm25", "dense"),
        help=("retrieval algorithms to compare (default: tfidf bm25 dense)"),
    )
    add_evaluation_arguments(parser)
    add_retriever_config_arguments(parser)

    args = parser.parse_args()

    retrievers = tuple(cast(list[RetrieverName], args.retrievers))

    if len(retrievers) < 2:
        parser.error("at least two retrievers must be provided")

    if len(set(retrievers)) != len(retrievers):
        parser.error("retrievers must be different")

    return CLIArguments(
        retrievers=retrievers,
        evaluation=parse_evaluation_arguments(parser, args),
        retriever_config=parse_retriever_config(parser, args),
    )


def evaluate_retrievers(
    retriever_names: tuple[RetrieverName, ...],
    *,
    factory: RetrieverFactory,
    queries: list[QueryExample],
    top_k: int,
) -> dict[str, RetrievalEvaluationResult]:
    results: dict[str, RetrievalEvaluationResult] = {}

    for name in retriever_names:
        retriever = factory.create(name)

        results[name] = evaluate_retriever(retriever, queries, top_k=top_k)

    return results


def print_metric_table(results: dict[str, RetrievalEvaluationResult]) -> None:
    print("Aggregate metrics:")
    print()
    print(
        f"{'Retriever':<12}"
        f"{'Recall@1':>10}"
        f"{'Recall@3':>10}"
        f"{'Recall@5':>10}"
        f"{'Recall@10':>10}"
        f"{'MRR':>10}"
    )

    for retriever_name, result in results.items():
        print(
            f"{retriever_name:<12}"
            f"{result.recall_at_1:>10.4f}"
            f"{result.recall_at_3:>10.4f}"
            f"{result.recall_at_5:>10.4f}"
            f"{result.recall_at_10:>10.4f}"
            f"{result.mrr:>10.4f}"
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
    better_counts: dict[str, int] = {retriever_name: 0 for retriever_name in retriever_names}

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
    data = load_evaluation_data(args.evaluation)

    if not data.queries:
        raise SystemExit(f"no queries found for split {args.evaluation.split!r}")

    factory = RetrieverFactory(
        data.documents,
        config=args.retriever_config,
    )

    evaluation_results = evaluate_retrievers(
        args.retrievers,
        factory=factory,
        queries=data.queries,
        top_k=args.evaluation.top_k,
    )

    comparisons = compare_evaluation_results(evaluation_results)

    stats = calculate_stats(
        comparisons,
        retriever_names=args.retrievers,
        failure_k=args.evaluation.failure_k,
    )

    print("Retrievers: " + ", ".join(args.retrievers))
    print(f"Split: {args.evaluation.split}")
    print(f"Failure cutoff: top {args.evaluation.failure_k}")
    print()

    print_metric_table(evaluation_results)

    print_comparison_details(
        comparisons,
        failure_k=args.evaluation.failure_k,
    )

    print_summary(
        stats,
        retriever_names=args.retrievers,
        failure_k=args.evaluation.failure_k,
    )


if __name__ == "__main__":
    main()
