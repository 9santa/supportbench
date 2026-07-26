from typing import Protocol

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from supportbench.data.models import Document

FloatMatrix = npt.NDArray[np.float32]


class DenseEncoder(Protocol):
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    def encode_documents(self, texts: list[str]) -> FloatMatrix:
        """Encode document texts into normalized vectors."""
        ...

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        """Encode query texts into normalized vectors."""
        ...


class SentenceTransformerDenceEncoder(DenseEncoder):
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        batch_size: int,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")

        if not device.strip():
            raise ValueError("device must be non-empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None or dimension <= 0:
            raise ValueError("model must have a positive sentence embedding dimension")

        self._dimension: int = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str]) -> FloatMatrix:
        return self._encode(
            texts,
            prefix="passage: ",
        )

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        return self._encode(
            texts,
            prefix="query: ",
        )

    def _encode(self, texts: list[str], *, prefix: str) -> FloatMatrix:
        if not texts:
            return np.empty(shape=(0, self.dimension), dtype=np.float32)

        prefixed_texts = [f"{prefix}{text}" for text in texts]

        raw_embeddings = self._model.encode(
            prefixed_texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        self._validate_embeddings(embeddings, expected_rows=len(texts))

        return embeddings

    def _validate_embeddings(
        self,
        embeddings: FloatMatrix,
        *,
        expected_rows: int,
    ) -> None:
        expected_shape = (expected_rows, self.dimension)

        if embeddings.shape != expected_shape:
            raise ValueError(
                f"unexpected embedding shape: expected: {expected_shape}, got: {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinity")


def format_document(document: Document) -> str:
    return f"{document.title}\n{document.text}"
