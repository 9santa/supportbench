import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from supportbench.data.loaders import load_documents
from supportbench.retrieval.base import (
    Retriever,
)
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever

type RetrieverName = Literal["tfidf", "bm25"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"


@dataclass(frozen=True, slots=True)
class CLIArguments:
    query: str
    documents: Path
    top_k: int
    retriever: RetrieverName


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser(description="Search the SupportBench corpus.")

    parser.add_argument(
        "query",
        help="search query",
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help=(f"path to documents.jsonl (default: {DEFAULT_DOCUMENTS_PATH})"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="maximum number of results (default: 5)",
    )

    parser.add_argument(
        "--retriever",
        choices=("tfidf", "bm25"),
        default="bm25",
        help="retreval algorithm (default: bm25)",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    return CLIArguments(
        query=cast(str, args.query),
        documents=cast(Path, args.documents),
        top_k=cast(int, args.top_k),
        retriever=cast(RetrieverName, args.retriever),
    )


def create_retriever(name: RetrieverName, index: InvertedIndex) -> Retriever:
    if name == "tfidf":
        return TfidfRetriever(index)

    if name == "bm25":
        return BM25Retriever(index)

    raise ValueError(f"unknown retriever: {name!r}")


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents)
    index = InvertedIndex.build(documents)
    retriever = create_retriever(args.retriever, index)

    search_result = retriever.search(args.query, top_k=args.top_k)

    doc_id_width = max(len(result.doc_id) for result in search_result)

    print(f"Query: {args.query}\n")
    for result in search_result:
        print(f"{result.rank}. {result.doc_id:<{doc_id_width}}      score={result.score:.4f}")


if __name__ == "__main__":
    main()
