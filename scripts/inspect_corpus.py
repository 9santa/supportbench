import argparse
from pathlib import Path

from supportbench.data.corpus_statistics import (
    FullCorpusStats,
    compute_full_corpus_statistics,
)
from supportbench.data.loaders import load_documents
from supportbench.retrieval.inverted_index import InvertedIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"


def most_common_terms(index: InvertedIndex, *, limit: int) -> list[tuple[str, int]]:
    terms_with_frequence = [(term, index.document_frequency(term)) for term in index.terms]

    terms_with_frequence.sort(key=lambda item: (-item[1], item[0]))

    return terms_with_frequence[:limit]


def print_stats(statistics: FullCorpusStats) -> None:
    lengths = statistics.document_lengths
    frequencies = statistics.posting_frequencies

    print("Corpus statistics:")
    print()
    print("Document lengths:")
    print(f"  min:    {lengths.minimum}")
    print(f"  median: {lengths.median:.2f}")
    print(f"  mean:   {lengths.mean:.2f}")
    print(f"  p90:    {lengths.p90:.2f}")
    print(f"  max:    {lengths.maximum}")
    print(f"  std:    {lengths.standard_deviation:.2f}")
    print(f"  CV:     {lengths.coefficient_of_variation:.4f}")

    print()
    print("Posting frequencies:")
    print(f"  postings: {frequencies.posting_count}")
    print(f"  tf=1:     {frequencies.share_tf_1:.2%}")
    print(f"  tf=2:     {frequencies.share_tf_2:.2%}")
    print(f"  tf>=3:   {frequencies.share_tf_3_or_more:.2%}")
    print(f"  mean tf:  {frequencies.mean:.2f}")
    print(f"  max tf:   {frequencies.maximum}")


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

    corpus_stats = compute_full_corpus_statistics(index)
    print_stats(corpus_stats)
    print()


if __name__ == "__main__":
    main()
