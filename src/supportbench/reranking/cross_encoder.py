from collections.abc import Sequence
from re import L
import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder

from supportbench.reranking.base import (
    RerankCandidate,
    RerankResult,
    Reranker,
)


type FloatVector = npt.NDArray[np.float32]


class SentenseTransformerCrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")

        if not device.strip():
            raise ValueError("device must be non-empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self._model_name = model_name
        self._batch_size = batch_size

        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not query.strip():
            return []

        candidate_items = tuple(candidates)

        if not candidate_items:
            return []

        pairs = [(query, candidate.text) for candidate in candidate_items]

        raw_scores: object = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = self._normalize_scores(
            raw_scores,
            expected_count=len(candidate_items),
        )

        scored_candidates = [
            RerankResult(
                doc_id=candidate.doc_id,
                score=float(score),
                retrieval_score=candidate.retrieval_score,
                retrieval_rank=candidate.retrieval_rank,
            )
            for candidate, score in zip(candidate_items, scores, strict=True)
        ]

        # Tie-break order
        scored_candidates.sort(
            key=lambda result: (
                -result.score,
                result.retrieval_rank,
                result.doc_id,
            )
        )

        return scored_candidates[:top_k]

    @staticmethod
    def _normalize_scores(
        raw_scores: object,
        *,
        expected_count: int,
    ) -> FloatVector:
        scores = np.asarray(raw_scores, dtype=np.float32)

        # Safety guard. predict() should typically return 1D array anyway.
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        if scores.ndim != 1:
            raise ValueError(
                "cross-encoder scores must be a one-dimensional array "
                "(one score per query-document text pair)"
            )

        if scores.shape[0] != expected_count:
            raise ValueError("cross-encoder returned an unexpected number of scores")

        if not np.isfinite(scores).all():
            raise ValueError("cross-encoder returned non-finite scores")

        return scores
