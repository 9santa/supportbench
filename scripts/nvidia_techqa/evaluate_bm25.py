import argparse
from pathlib import Path
from typing import cast

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.evaluation.retrieval_evaluator import (
    DEFAULT_RECALL_CUTOFFS,
    evaluate_retriever,
)
from supportbench.experiments.evaluation_export import (
    build_bm25_experiment_summary,
    export_bm25_experiment_summary,
    export_query_evaluations,
)
from supportbench.retrieval.bm25 import (
    BM25Retriever,
)
from supportbench.retrieval.inverted_index import (
    InvertedIndex,
)

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "documents.jsonl"

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "nvidia_techqa" / "normalized" / "queries.jsonl"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "evaluations" / "bm25_document_baseline"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Evaluate document-level BM25 on NVIDIA TechQA-RAG-Eval.")
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="dev",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    documents_path = cast(
        Path,
        args.documents,
    )
    queries_path = cast(
        Path,
        args.queries,
    )
    output_dir = cast(
        Path,
        args.output_dir,
    )
    split = cast(str, args.split)
    top_k = cast(int, args.top_k)
    k1 = cast(float, args.k1)
    b = cast(float, args.b)

    minimum_top_k = max(DEFAULT_RECALL_CUTOFFS)

    if top_k < minimum_top_k:
        parser.error(f"--top-k must be at least {minimum_top_k}")

    if k1 <= 0.0:
        parser.error("--k1 must be positive")

    if not 0.0 <= b <= 1.0:
        parser.error("--b must be between 0 and 1")

    documents = load_documents(documents_path)

    document_ids = {document.doc_id for document in documents}

    all_queries = load_queries(
        queries_path,
        document_ids,
    )

    queries = [query for query in all_queries if query.split == split]

    if not queries:
        parser.error(f"no queries found for split {split!r}")

    print(f"Building BM25 index for {len(documents):,} documents...")

    index = InvertedIndex.build(documents)

    print(
        "Index built: "
        f"{index.statistics.document_count:,} documents, "
        f"{index.statistics.vocab_size:,} terms, "
        f"avg length "
        f"{index.statistics.avg_doc_len:.1f} tokens"
    )

    retriever = BM25Retriever(
        index,
        k1=k1,
        b=b,
    )

    evaluation = evaluate_retriever(
        retriever,
        queries,
        top_k=top_k,
    )

    experiment_name = f"bm25_k1_{k1:g}_b_{b:g}"

    summary = build_bm25_experiment_summary(
        experiment=experiment_name,
        k1=k1,
        b=b,
        split=split,
        top_k=top_k,
        result=evaluation,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_bm25_experiment_summary(
        summary,
        output_dir / "summary.json",
    )

    export_query_evaluations(
        evaluation,
        output_dir / "queries.jsonl",
    )

    print()
    print(f"Split: {split}")
    print(
        "Queries: "
        f"{evaluation.query_count} total, "
        f"{evaluation.labeled_query_count} labeled, "
        f"{evaluation.unlabeled_query_count} unlabeled"
    )
    print(f"Recall@1:  {evaluation.recall_at_1:.4f}")
    print(f"Recall@3:  {evaluation.recall_at_3:.4f}")
    print(f"Recall@5:  {evaluation.recall_at_5:.4f}")
    print(f"Recall@10: {evaluation.recall_at_10:.4f}")
    print(f"Recall@20: {evaluation.recall_at_20:.4f}")
    print(f"Recall@50: {evaluation.recall_at_50:.4f}")
    print(f"MRR@10:    {evaluation.mrr:.4f}")
    print()
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
