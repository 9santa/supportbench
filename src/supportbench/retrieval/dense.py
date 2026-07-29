import numpy as np

from supportbench.retrieval.base import Retriever, SearchResult
from supportbench.retrieval.dense_encoder import (
    DenseEncoder,
    FloatMatrix,
)
from supportbench.retrieval.dense_index import (
    DenseVectorIndex,
)


class DenseRetriever(Retriever):
    def __init__(
        self,
        encoder: DenseEncoder,
        index: DenseVectorIndex,
    ) -> None:
        if encoder.dimension != index.dimension:
            raise ValueError(
                "encoder and index dimensions do not match: "
                f"encoder={encoder.dimension}, "
                f"index={index.dimension}"
            )

        self._encoder = encoder
        self._index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        if self._index.size == 0:
            return []

        embeddings = self._encoder.encode_queries([query])

        self._validate_query_embeddings(embeddings)

        query_vector = embeddings[0]

        vector_results = self._index.search(
            query_vector,
            top_k=top_k,
        )

        return [
            SearchResult(
                doc_id=result.doc_id,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(vector_results, start=1)
        ]

    def _validate_query_embeddings(
        self,
        embeddings: FloatMatrix,
    ) -> None:
        expected_shape = (1, self._encoder.dimension)

        if embeddings.shape != expected_shape:
            raise ValueError(
                "unexpected query embedding shape: "
                f"expected {expected_shape}, "
                f"got {embeddings.shape}"
            )

        if embeddings.dtype != np.float32:
            raise ValueError("query embeddings must have dtype=np.float32")

        if not np.isfinite(embeddings).all():
            raise ValueError("query embeddings contain NaN or infinity")
