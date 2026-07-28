import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from supportbench.data.models import Document
from supportbench.retrieval.base import Retriever
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.dense import DenseRetriever
from supportbench.retrieval.dense_encoder import SentenceTransformerDenceEncoder
from supportbench.retrieval.dense_index import (
    FaissFlatVectorIndex,
    compute_document_fingerprint,
)
from supportbench.retrieval.hybrid import WeightedRetrieverSource, WeightedRRFHybrid
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.retrieval.tfidf import TfidfRetriever

type RetrieverName = Literal["tfidf", "bm25", "dense", "hybrid"]

RETRIEVER_NAMES: tuple[RetrieverName, ...] = (
    "tfidf",
    "bm25",
    "dense",
    "hybrid",
)


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    dense_index_path: Path
    dense_model_name: str
    dense_device: str = "cuda"
    dense_batch_size: int = 16
    bm25_k1: float = 0.5
    bm25_b: float = 1.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    candidate_k: int = 50
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if not self.dense_model_name.strip():
            raise ValueError("dense_model_name must be non-empty")

        if not self.dense_device.strip():
            raise ValueError("dense_device must be non-empty")

        if self.dense_batch_size <= 0:
            raise ValueError("dense_batch_size must be positive")

        if not math.isfinite(self.bm25_k1) or self.bm25_k1 <= 0.0:
            raise ValueError("bm25_k1 must be finite and positive")

        if not math.isfinite(self.bm25_b) or not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("bm25_b must be finite and between 0 and 1")

        if not math.isfinite(self.bm25_weight) or self.bm25_weight < 0.0:
            raise ValueError("bm25_weight must be finite and non-negative")

        if not math.isfinite(self.dense_weight) or self.dense_weight < 0.0:
            raise ValueError("dense_weight must be finite and non-negative")

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


class RetrieverFactory:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        config: RetrieverConfig,
    ) -> None:
        self._documents = tuple(documents)
        self._config = config
        self._inverted_index: InvertedIndex | None = None
        self._dense_retriever: DenseRetriever | None = None

    def create(self, name: RetrieverName) -> Retriever:
        if name == "tfidf":
            return TfidfRetriever(self._get_inverted_index())

        if name == "bm25":
            return self._create_bm25_retriever()

        if name == "dense":
            return self._get_dense_retriever()

        if name == "hybrid":
            return WeightedRRFHybrid(
                sources=(
                    WeightedRetrieverSource(
                        name="bm25",
                        retriever=self._create_bm25_retriever(),
                        weight=self._config.bm25_weight,
                    ),
                    WeightedRetrieverSource(
                        name="dense",
                        retriever=self._get_dense_retriever(),
                        weight=self._config.dense_weight,
                    ),
                ),
                candidate_k=self._config.candidate_k,
                rrf_k=self._config.rrf_k,
            )

        raise ValueError(f"unknown retriever: {name!r}")

    def _create_bm25_retriever(self) -> BM25Retriever:
        return BM25Retriever(
            self._get_inverted_index(),
            k1=self._config.bm25_k1,
            b=self._config.bm25_b,
        )

    def _get_inverted_index(self) -> InvertedIndex:
        if self._inverted_index is None:
            self._inverted_index = InvertedIndex.build(self._documents)

        return self._inverted_index

    def _get_dense_retriever(self) -> DenseRetriever:
        if self._dense_retriever is None:
            fingerprint = compute_document_fingerprint(self._documents)
            vector_index = FaissFlatVectorIndex.load(
                self._config.dense_index_path,
                expected_document_fingerprint=fingerprint,
                expected_model_name=self._config.dense_model_name,
            )
            encoder = SentenceTransformerDenceEncoder(
                self._config.dense_model_name,
                device=self._config.dense_device,
                batch_size=self._config.dense_batch_size,
            )
            self._dense_retriever = DenseRetriever(encoder, vector_index)

        return self._dense_retriever
