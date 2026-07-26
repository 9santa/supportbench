from typing import Any

import numpy as np
import pytest

import supportbench.retrieval.dense_encoder as encoder_module
from supportbench.retrieval.dense_encoder import (
    SentenceTransformerDenceEncoder,
)


class FakeSentenceTransformer:
    def __init__(
        self,
        *,
        dimension: int = 3,
        output: object | None = None,
    ) -> None:
        self._dimension = dimension
        self._output = output
        self.calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        if self._output is not None:
            return self._output

        return np.ones(shape=(len(sentences), self._dimension), dtype=np.float32)


def build_encoder(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeSentenceTransformer,
    *,
    batch_size: int = 16,
) -> SentenceTransformerDenceEncoder:
    def fake_constructor(
        model_name: str,
        *,
        device: str,
    ) -> FakeSentenceTransformer:
        assert model_name == "intfloat/multilingual-e5-base"
        assert device == "cuda"

        return model

    monkeypatch.setattr(
        encoder_module,
        "SentenceTransformer",
        fake_constructor,
    )

    return SentenceTransformerDenceEncoder(
        "intfloat/multilingual-e5-base",
        device="cuda",
        batch_size=batch_size,
    )


def test_query_encoding_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    encoder = build_encoder(
        monkeypatch,
        model,
        batch_size=16,
    )

    embeddings = encoder.encode_queries(
        [
            "потерял доступ",
            "не работает vpn",
        ]
    )

    assert model.calls == [
        {
            "sentences": [
                "query: потерял доступ",
                "query: не работает vpn",
            ],
            "batch_size": 16,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32


def test_document_prefix_is_added(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    encoder = build_encoder(monkeypatch, model)

    encoder.encode_documents(
        [
            "Настройка VPN",
            "Восстановление GitLab",
        ]
    )

    assert model.calls[0]["sentences"] == [
        "passage: Настройка VPN",
        "passage: Восстановление GitLab",
    ]


def test_empty_input_returns_empty_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    encoder = build_encoder(monkeypatch, model)

    embeddings = encoder.encode_documents([])

    assert embeddings.shape == (0, 3)
    assert embeddings.dtype == np.float32
    assert model.calls == []
