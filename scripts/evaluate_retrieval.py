import argparse
from dataclasses import dataclass
from typing import cast

from supportbench.evaluation.retrieval_analysis import failures_at_k
from supportbench.evaluation.retrieval_cli import (
    EvaluationArguments,
    add_evaluation_arguments,
    add_retriever_config_arguments,
    load_evaluation_data,
    parse_evaluation_arguments,
    parse_retriever_config,
)
from supportbench.evaluation.retrieval_evaluator import (
    QueryEvaluation,
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
    retriever: RetrieverName
    evaluation: EvaluationArguments
    retriever_config: RetrieverConfig
    show_errors: bool


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality.",
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="bm25",
        help="retrieval algorithm (default: bm25)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="show queries with no relevant document in top-k",
    )
    add_evaluation_arguments(parser)
    add_retriever_config_arguments(parser)

    args = parser.parse_args()

    return CLIArguments(
        retriever=cast(RetrieverName, args.retriever),
        evaluation=parse_evaluation_arguments(parser, args),
        retriever_config=parse_retriever_config(parser, args),
        show_errors=cast(bool, args.show_errors),
    )


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
    data = load_evaluation_data(args.evaluation)

    if not data.queries:
        raise SystemExit(f"no queries found for split {args.evaluation.split!r}")

    factory = RetrieverFactory(
        data.documents,
        config=args.retriever_config,
    )
    retriever = factory.create(args.retriever)

    evaluation = evaluate_retriever(
        retriever,
        data.queries,
        top_k=args.evaluation.top_k,
    )

    print_evaluation(
        retriever_name=args.retriever,
        split=args.evaluation.split,
        result=evaluation,
    )

    if args.show_errors:
        failures = failures_at_k(
            evaluation,
            k=args.evaluation.failure_k,
        )

        print_failures(
            failures,
            failure_k=args.evaluation.failure_k,
        )


if __name__ == "__main__":
    main()
