import argparse
from pathlib import Path

from supportbench.data.loaders import load_documents
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the SupportBench corpus using TF-IDF.")

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

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    return args


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents)
    index = InvertedIndex.build(documents)
    retriever = TfidfRetriever(index)

    search_result = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    for result in search_result:
        print(f"{result.rank}. {result.doc_id}      score={result.score:.4f}")


if __name__ == "__main__":
    main()
