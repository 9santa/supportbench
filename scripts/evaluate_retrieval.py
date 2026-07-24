import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever

type RetrieverName = Literal["tfidf", "bm25"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_test.jsonl"


@dataclass(frozen=True, slots=True)
class CliArguments:
    retriever: RetrieverName
    split: str
    documents_path: Path
    queries_path: Path


def parse_args() -> CliArguments:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality.",
    )
    parser.add_argument(
        "--retriever",
        choices=("tfidf", "bm25"),
        default="bm25",
        help="retrieval algorithm (default: bm25)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="query split to evaluate (default: test)",
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

    args = parser.parse_args()

    return CliArguments(
        retriever=cast(
            RetrieverName,
            args.retriever,
        ),
        split=cast(str, args.split),
        documents_path=cast(Path, args.documents),
        queries_path=cast(Path, args.queries),
    )


def create_retriever(
    name: RetrieverName,
    index: InvertedIndex,
) -> Retriever:
    if name == "tfidf":
        return TfidfRetriever(index)

    if name == "bm25":
        return BM25Retriever(index)

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
    print(f"MRR:      {result.mrr:.4f}")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    index = InvertedIndex.build(documents)

    queries = load_queries(args.queries_path, set(index.document_ids))

    selected_queries = select_queries(
        queries,
        split=args.split,
    )

    if not selected_queries:
        raise SystemExit(f"no queries found for split {args.split!r}")

    retriever = create_retriever(
        args.retriever,
        index,
    )

    result = evaluate_retriever(
        retriever,
        selected_queries,
    )

    print_evaluation(
        retriever_name=args.retriever,
        split=args.split,
        result=result,
    )


if __name__ == "__main__":
    main()
