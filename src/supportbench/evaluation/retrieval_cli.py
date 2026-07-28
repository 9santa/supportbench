import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from supportbench.data.loaders import load_documents, load_queries
from supportbench.data.models import Document, QueryExample
from supportbench.retrieval.factory import RetrieverConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_dev.jsonl"
DEFAULT_DENSE_INDEX_PATH = PROJECT_ROOT / "artifacts" / "dense" / "multilingual-e5-base"
DEFAULT_DENSE_MODEL_NAME = "intfloat/multilingual-e5-base"


@dataclass(frozen=True, slots=True)
class EvaluationArguments:
    split: str
    documents_path: Path
    queries_path: Path
    top_k: int
    failure_k: int


@dataclass(frozen=True, slots=True)
class EvaluationData:
    documents: list[Document]
    queries: list[QueryExample]


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default="dev",
        help="query split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help=f"path to documents.jsonl (default: {DEFAULT_DOCUMENTS_PATH})",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"path to queries.jsonl (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="number of retrieved documents per query (default: 10)",
    )
    parser.add_argument(
        "--failure-k",
        type=int,
        choices=(1, 3, 5),
        default=3,
        help="cutoff used to classify retrieval failures (default: 3)",
    )


def add_retriever_config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("retriever configuration")
    group.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_DENSE_INDEX_PATH,
        help=f"path to dense index (default: {DEFAULT_DENSE_INDEX_PATH})",
    )
    group.add_argument(
        "--dense-model",
        default=DEFAULT_DENSE_MODEL_NAME,
        help=f"Sentence Transformers model name (default: {DEFAULT_DENSE_MODEL_NAME})",
    )
    group.add_argument(
        "--dense-device",
        default="cuda",
        help="dense encoder device (default: cuda)",
    )
    group.add_argument(
        "--dense-batch-size",
        type=int,
        default=16,
        help="dense query encoding batch size (default: 16)",
    )
    group.add_argument(
        "--bm25-k1",
        type=float,
        default=0.5,
        help="BM25 term-frequency saturation parameter (default: 0.5)",
    )
    group.add_argument(
        "--bm25-b",
        type=float,
        default=1.0,
        help="BM25 document-length normalization parameter (default: 1.0)",
    )
    group.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="BM25 retriever weight in weighted RRF (default: 1.0)",
    )
    group.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="dense retriever weight in weighted RRF (default: 1.0)",
    )
    group.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="candidate count requested from each hybrid source (default: 50)",
    )
    group.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank smoothing constant (default: 60)",
    )


def parse_evaluation_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> EvaluationArguments:
    top_k = cast(int, args.top_k)
    failure_k = cast(int, args.failure_k)

    if top_k < 10:
        parser.error("--top-k must be at least 10 to compute Recall@10")

    if failure_k > top_k:
        parser.error("--failure-k must not be greater than --top-k")

    return EvaluationArguments(
        split=cast(str, args.split),
        documents_path=cast(Path, args.documents),
        queries_path=cast(Path, args.queries),
        top_k=top_k,
        failure_k=failure_k,
    )


def parse_retriever_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> RetrieverConfig:
    try:
        return RetrieverConfig(
            dense_index_path=cast(Path, args.dense_index),
            dense_model_name=cast(str, args.dense_model),
            dense_device=cast(str, args.dense_device),
            dense_batch_size=cast(int, args.dense_batch_size),
            bm25_k1=cast(float, args.bm25_k1),
            bm25_b=cast(float, args.bm25_b),
            bm25_weight=cast(float, args.bm25_weight),
            dense_weight=cast(float, args.dense_weight),
            candidate_k=cast(int, args.candidate_k),
            rrf_k=cast(int, args.rrf_k),
        )
    except ValueError as error:
        parser.error(str(error))


def load_evaluation_data(args: EvaluationArguments) -> EvaluationData:
    documents = load_documents(args.documents_path)
    document_ids = {document.doc_id for document in documents}
    queries = load_queries(args.queries_path, document_ids)
    selected_queries = [query for query in queries if query.split == args.split]

    return EvaluationData(
        documents=documents,
        queries=selected_queries,
    )
