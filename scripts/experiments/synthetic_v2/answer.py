import argparse
import sys
from pathlib import Path

from scripts._paths import PROJECT_ROOT
from supportbench.data.loaders import load_documents
from supportbench.experiments.synthetic_v2.rag.context_builder import ContextBuilder
from supportbench.experiments.synthetic_v2.rag.pipeline import GroundedRAGPipeline
from supportbench.experiments.synthetic_v2.rag.retrieval_pipeline import RetrievalPipeline
from supportbench.rag.citations import (
    CitationValidationError,
)
from supportbench.rag.document_store import (
    InMemoryDocumentStore,
)
from supportbench.rag.generation.ollama import (
    OllamaLLMClient,
)
from supportbench.rag.generation.parser import (
    GeneratedAnswerParseError,
)
from supportbench.rag.generation.prompt import (
    GroundedPromptBuilder,
)
from supportbench.reranking.factory import (
    CrossEncoderConfig,
    RerankingFactory,
)
from supportbench.retrieval.factory import (
    RetrieverConfig,
    RetrieverFactory,
)
from supportbench.retrieval.hybrid import (
    WeightedRetrieverSource,
    WeightedRRFHybrid,
)

DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "documents.jsonl"

DEFAULT_DENSE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "synthetic"
    / "v2"
    / "dense"
    / "multilingual-e5-base"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Answer a query using grounded RAG."))

    parser.add_argument("query")

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_DENSE_INDEX_PATH,
    )

    parser.add_argument(
        "--dense-device",
        default="cuda",
    )
    parser.add_argument(
        "--reranker-device",
        default="cuda",
    )
    parser.add_argument(
        "--dense-model",
        default=("intfloat/multilingual-e5-base"),
    )
    parser.add_argument(
        "--reranker-model",
        default=("BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3:4b",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=12_000,
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.top_k > args.reranker_candidate_k:
        parser.error("--top-k must not exceed --reranker-candidate-k")

    return args


def build_pipeline(args: argparse.Namespace) -> GroundedRAGPipeline:
    documents = load_documents(args.documents)

    retrieval_factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=args.dense_index,
            dense_model_name=args.dense_model,
            dense_device=args.dense_device,
            dense_batch_size=16,
            bm25_k1=0.5,
            bm25_b=1.0,
        ),
    )

    bm25 = retrieval_factory.create("bm25")
    dense = retrieval_factory.create("dense")

    candidate_rrf = WeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource(
                name="bm25",
                retriever=bm25,
                weight=1.0,
            ),
            WeightedRetrieverSource(
                name="dense",
                retriever=dense,
                weight=1.5,
            ),
        ),
        candidate_k=100,
        rrf_k=20,
    )

    reranking_factory = RerankingFactory(
        documents,
        config=CrossEncoderConfig(
            model_name=args.reranker_model,
            device=args.reranker_device,
            batch_size=16,
            max_length=512,
        ),
    )

    reranked_retriever = reranking_factory.create(
        candidate_retriever=candidate_rrf,
        candidate_k=args.reranker_candidate_k,
    )

    retrieval_pipeline = RetrievalPipeline(
        retriever=reranked_retriever,
        document_store=InMemoryDocumentStore(documents),
    )

    return GroundedRAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_builder=ContextBuilder(
            max_documents=args.max_documents,
            max_characters=args.max_characters,
        ),
        prompt_builder=GroundedPromptBuilder(layout="legacy_system_user"),
        llm_client=OllamaLLMClient(
            model_name=args.llm_model,
            base_url=args.ollama_url,
        ),
        retrieval_top_k=args.top_k,
    )


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args)

    try:
        run = pipeline.run(args.query)
    except GeneratedAnswerParseError as error:
        print(
            f"Generation contract error: {error}",
            file=sys.stderr,
        )

        if args.show_raw_response:
            print()
            print("Raw response:")
            print(error.raw_response)

        raise SystemExit(2) from error
    except CitationValidationError as error:
        print(
            f"Citation validation error: {error}",
            file=sys.stderr,
        )

        if args.show_raw_response and error.raw_response is not None:
            print()
            print("Raw response:")
            print(error.raw_response)

        raise SystemExit(3) from error

    answer = run.answer

    print(f"Decision: {answer.decision}")
    print("Answer:")
    print(answer.answer)
    print("Citations:")

    if answer.citation_ids:
        for citation_id in answer.citation_ids:
            print(f"- {citation_id}")
    else:
        print("- none")

    if args.show_context:
        print()
        print("Context:")
        print(run.context.formatted_text)

    if args.show_raw_response:
        print()
        print("Raw response:")

        if run.raw_response is None:
            print("<LLM was not called>")
        else:
            print(run.raw_response)


if __name__ == "__main__":
    main()
