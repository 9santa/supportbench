import argparse
from pathlib import Path

from supportbench.data.loaders import load_documents
from supportbench.retrieval.inverted_index import InvertedIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"


def most_common_terms(index: InvertedIndex, *, limit: int) -> list[tuple[str, int]]:
    terms_with_frequence = [(term, index.document_frequency(term)) for term in index.terms]

    terms_with_frequence.sort(key=lambda item: (-item[1], item[0]))

    return terms_with_frequence[:limit]


def print_stats(index: InvertedIndex, *, top_k: int) -> None:
    stats = index.statistics

    print(f"Documents: {stats.document_count}")
    print(f"Vocabulary size: {stats.vocab_size}")
    print(f"Average document length: {stats.avg_doc_len:.2f}")
    print()
    print("Top terms by document frequency:")

    top_terms = most_common_terms(
        index,
        limit=top_k,
    )

    if not top_terms:
        print("No terms found.")
        return

    for position, (term, document_frequency) in enumerate(top_terms, start=1):
        print(f"{position}. {term}, df={document_frequency}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect corpus statistics.")

    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help=(f"path to documents.jsonl (default: {DEFAULT_DOCUMENTS_PATH})"),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of terms to show (default: 20)",
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be greater than zero")

    return args


def main() -> None:
    args = parse_args()

    documents = load_documents(args.path)
    index = InvertedIndex.build(documents)

    print_stats(index, top_k=args.top)


if __name__ == "__main__":
    main()
