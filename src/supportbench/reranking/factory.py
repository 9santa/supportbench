from collections.abc import Sequence
from dataclasses import dataclass

from supportbench.data.models import Document
from supportbench.reranking.cross_encoder import (
    SentenceTransformerCrossEncoderReranker,
)
from supportbench.reranking.retriever import (
    RerankingRetriever,
)
from supportbench.retrieval.base import Retriever


@dataclass(frozen=True, slots=True)
class CrossEncoderConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("reranker model name must be non-empty")

        if not self.device.strip():
            raise ValueError("reranker device must be non-empty")

        if self.batch_size <= 0:
            raise ValueError("reranker batch size must be positive")

        if self.max_length <= 0:
            raise ValueError("reranker max length must be positive")


class RerankingFactory:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        config: CrossEncoderConfig,
    ) -> None:
        self._documents = documents

        if not self._documents:
            raise ValueError("documents must not be empty")

        self._config = config

        self._reranker: SentenceTransformerCrossEncoderReranker | None = None

    def create(
        self,
        *,
        candidate_retriever: Retriever,
        candidate_k: int,
    ) -> RerankingRetriever:
        return RerankingRetriever(
            candidate_retriever=candidate_retriever,
            reranker=self._get_reranker(),
            documents=self._documents,
            candidate_k=candidate_k,
            performance_device=self._config.device,
        )

    def _get_reranker(self) -> SentenceTransformerCrossEncoderReranker:
        if self._reranker is None:
            self._reranker = SentenceTransformerCrossEncoderReranker(
                self._config.model_name,
                device=self._config.device,
                batch_size=self._config.batch_size,
                max_length=self._config.max_length,
            )

        return self._reranker
