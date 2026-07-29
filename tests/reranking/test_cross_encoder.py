from collections.abc import Sequence

import numpy as np
import pytest

import supportbench.reranking.cross_encoder as module
from supportbench.reranking.base import (
    RerankCandidate,
)
from supportbench.reranking.cross_encoder import (
    SentenceTransformerCrossEncoderReranker,
)


class FakeCrossEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.received_pairs: Sequence[tuple[str, str]] | None = None

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.received_pairs = pairs

        assert batch_size == 8
        assert show_progress_bar is False
        assert convert_to_numpy is True

        return np.array(
            [0.2, 0.9, 0.4],
            dtype=np.float32,
        )


def test_reranks_by_cross_encoder_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = SentenceTransformerCrossEncoderReranker(
        "fake-model",
        device="cpu",
        batch_size=8,
        max_length=256,
    )

    candidates = [
        RerankCandidate(
            doc_id="doc_a",
            text="Document A",
            retrieval_score=0.9,
            retrieval_rank=1,
        ),
        RerankCandidate(
            doc_id="doc_b",
            text="Document B",
            retrieval_score=0.8,
            retrieval_rank=2,
        ),
        RerankCandidate(
            doc_id="doc_c",
            text="Document C",
            retrieval_score=0.7,
            retrieval_rank=3,
        ),
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert [result.doc_id for result in results] == [
        "doc_b",
        "doc_c",
    ]

    model = reranker._model

    assert model.received_pairs == [
        ("query", "Document A"),
        ("query", "Document B"),
        ("query", "Document C"),
    ]


def test_empty_candidates_return_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = SentenceTransformerCrossEncoderReranker(
        "fake-model",
        device="cpu",
    )

    assert (
        reranker.rerank(
            "query",
            [],
            top_k=10,
        )
        == []
    )
