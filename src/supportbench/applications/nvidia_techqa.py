import math
from dataclasses import dataclass
from pathlib import Path

from supportbench.chunking.base import HuggingFaceTokenCodec
from supportbench.chunking.loaders import load_chunks
from supportbench.data.loaders import load_documents
from supportbench.rag.chunk_context_builder import RepresentativeChunkContextBuilder
from supportbench.rag.chunk_retrieval_pipeline import RepresentativeChunkRetrievalPipeline
from supportbench.rag.document_store import InMemoryDocumentStore
from supportbench.rag.parent_pipeline import ParentContextPipeline
from supportbench.rag.parent_retrieval import ParentRetrievalOrchestrator
from supportbench.reranking.cross_encoder import SentenceTransformerCrossEncoderReranker
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory
from supportbench.retrieval.hybrid import WeightedRetrieverSource
from supportbench.retrieval.parent_hybrid import ParentAggregation, ParentWeightedRRFHybrid

DEFAULT_CHUNK_CONFIG = "ha384o64m512r2v2"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True, slots=True)
class NvidiaTechQAContextConfig:
    chunks_root: Path
    index_root: Path
    chunk_config: str = DEFAULT_CHUNK_CONFIG
    dense_model_name: str = DEFAULT_DENSE_MODEL
    reranker_model_name: str = DEFAULT_RERANKER_MODEL
    context_tokenizer_name: str = DEFAULT_DENSE_MODEL
    dense_device: str = "cuda"
    reranker_device: str = "cuda"
    dense_batch_size: int = 16
    reranker_batch_size: int = 16
    source_candidate_k: int = 500
    parent_candidate_k: int = 20
    chunks_per_parent: int = 2
    top_parents: int = 5
    max_context_tokens: int = 4_096
    candidate_prior_weight: float = 1.25
    fusion_rrf_k: int = 10
    minimum_overlap_tokens: int = 8
    maximum_overlap_tokens: int = 256
    bm25_k1: float = 0.5
    bm25_b: float = 1.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.5
    source_rrf_k: int = 10
    parent_aggregation: ParentAggregation = "capped_top_2_sum"
    reranker_max_length: int = 512
    second_evidence_weight: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.chunk_config.strip()
            or Path(self.chunk_config).name != self.chunk_config
            or self.chunk_config in {".", ".."}
        ):
            raise ValueError("chunk_config must be a non-empty path segment")

        named_values = {
            "dense_model_name": self.dense_model_name,
            "reranker_model_name": self.reranker_model_name,
            "context_tokenizer_name": self.context_tokenizer_name,
            "dense_device": self.dense_device,
            "reranker_device": self.reranker_device,
        }
        for name, text_value in named_values.items():
            if not text_value.strip():
                raise ValueError(f"{name} must be non-empty")

        positive_values = {
            "dense_batch_size": self.dense_batch_size,
            "reranker_batch_size": self.reranker_batch_size,
            "source_candidate_k": self.source_candidate_k,
            "parent_candidate_k": self.parent_candidate_k,
            "chunks_per_parent": self.chunks_per_parent,
            "top_parents": self.top_parents,
            "max_context_tokens": self.max_context_tokens,
            "fusion_rrf_k": self.fusion_rrf_k,
            "minimum_overlap_tokens": self.minimum_overlap_tokens,
            "maximum_overlap_tokens": self.maximum_overlap_tokens,
            "source_rrf_k": self.source_rrf_k,
            "reranker_max_length": self.reranker_max_length,
        }
        for name, integer_value in positive_values.items():
            if integer_value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.top_parents > self.parent_candidate_k:
            raise ValueError("top_parents must not exceed parent_candidate_k")
        if self.source_candidate_k < self.parent_candidate_k:
            raise ValueError("source_candidate_k must be at least parent_candidate_k")
        if self.maximum_overlap_tokens < self.minimum_overlap_tokens:
            raise ValueError(
                "maximum_overlap_tokens must be at least minimum_overlap_tokens"
            )

        non_negative_weights = {
            "candidate_prior_weight": self.candidate_prior_weight,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
        }
        for name, weight in non_negative_weights.items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if (
            not math.isfinite(self.second_evidence_weight)
            or not 0.0 <= self.second_evidence_weight <= 1.0
        ):
            raise ValueError("second_evidence_weight must be between 0 and 1")


def build_nvidia_techqa_context_pipeline(
    config: NvidiaTechQAContextConfig,
) -> ParentContextPipeline:
    chunk_directory = config.chunks_root / config.chunk_config
    runtime_documents = load_documents(chunk_directory / "documents.jsonl")
    chunks_by_id = load_chunks(chunk_directory / "chunks.jsonl")
    runtime_ids = {document.doc_id for document in runtime_documents}

    if runtime_ids != set(chunks_by_id):
        raise ValueError("documents.jsonl and chunks.jsonl contain different chunk IDs")

    factory = RetrieverFactory(
        runtime_documents,
        config=RetrieverConfig(
            dense_index_path=config.index_root / config.chunk_config,
            dense_model_name=config.dense_model_name,
            dense_device=config.dense_device,
            dense_batch_size=config.dense_batch_size,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
        ),
    )
    parent_by_chunk_id = {
        chunk_id: chunk.document_id for chunk_id, chunk in chunks_by_id.items()
    }
    parent_wrrf = ParentWeightedRRFHybrid(
        sources=(
            WeightedRetrieverSource("bm25", factory.create("bm25"), config.bm25_weight),
            WeightedRetrieverSource("dense", factory.create("dense"), config.dense_weight),
        ),
        parent_by_chunk_id=parent_by_chunk_id,
        source_candidate_k=config.source_candidate_k,
        rrf_k=config.source_rrf_k,
        aggregation=config.parent_aggregation,
        representative_chunks_per_parent=config.chunks_per_parent,
    )
    chunk_store = InMemoryDocumentStore(runtime_documents)
    reranker = SentenceTransformerCrossEncoderReranker(
        config.reranker_model_name,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
    )
    retrieval_orchestrator = ParentRetrievalOrchestrator(
        parent_retriever=parent_wrrf,
        reranker=reranker,
        chunk_store=chunk_store,
        parent_by_chunk_id=parent_by_chunk_id,
        parent_candidate_k=config.parent_candidate_k,
        chunks_per_parent=config.chunks_per_parent,
        candidate_prior_weight=config.candidate_prior_weight,
        fusion_rrf_k=config.fusion_rrf_k,
        second_evidence_weight=config.second_evidence_weight,
    )
    chunk_pipeline = RepresentativeChunkRetrievalPipeline(
        chunk_store=chunk_store,
        chunks_by_id=chunks_by_id,
    )
    context_builder = RepresentativeChunkContextBuilder(
        token_codec=HuggingFaceTokenCodec.from_pretrained(config.context_tokenizer_name),
        max_tokens=config.max_context_tokens,
        max_parents=config.top_parents,
        minimum_token_overlap=config.minimum_overlap_tokens,
        maximum_token_overlap=config.maximum_overlap_tokens,
    )

    return ParentContextPipeline(
        retrieval_orchestrator=retrieval_orchestrator,
        chunk_pipeline=chunk_pipeline,
        context_builder=context_builder,
        top_parents=config.top_parents,
    )
