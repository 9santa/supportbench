import json
from pathlib import Path

import numpy as np
import pytest

from supportbench.data.models import Document
from supportbench.retrieval.dense_build import (
    build_dense_index,
)
from supportbench.retrieval.dense_encoder import (
    FloatMatrix,
)
from supportbench.retrieval.dense_index import (
    FaissFlatVectorIndex,
    compute_document_fingerprint,
)


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.encoded_documents: list[str] = []

    @property
    def dimension(self) -> int:
        return 3

    def encode_queries(
        self,
        texts: list[str],
    ) -> FloatMatrix:
        raise AssertionError("encode_queries must not be called while building an index")

    def encode_documents(
        self,
        texts: list[str],
    ) -> FloatMatrix:
        self.encoded_documents = texts

        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )


def test_builds_and_saves_dense_index(
    tmp_path: Path,
) -> None:
    documents = [
        Document(
            doc_id="vpn_setup",
            title="Настройка VPN",
            text="Инструкция для Ubuntu.",
            category="network",
        ),
        Document(
            doc_id="gitlab_recovery",
            title="Восстановление GitLab",
            text="Сброс двухфакторной авторизации.",
            category="access",
        ),
    ]
    encoder = FakeDenseEncoder()
    output_directory = tmp_path / "dense-index"

    result = build_dense_index(
        documents=documents,
        encoder=encoder,
        model_name="fake-model",
        output_directory=output_directory,
    )

    assert encoder.encoded_documents == [
        "Настройка VPN\nИнструкция для Ubuntu.",
        "Восстановление GitLab\nСброс двухфакторной авторизации.",
    ]

    assert result.document_count == 2
    assert result.embedding_dimension == 3
    assert result.encoding_seconds >= 0.0
    assert result.index_build_seconds >= 0.0

    assert (output_directory / "index.faiss").exists()
    assert (output_directory / "doc_ids.json").exists()
    assert (output_directory / "manifest.json").exists()


def test_saved_metadata_is_correct(
    tmp_path: Path,
) -> None:
    documents = [
        Document(
            doc_id="vpn_setup",
            title="Настройка VPN",
            text="Инструкция.",
            category="network",
        ),
        Document(
            doc_id="gitlab_recovery",
            title="GitLab 2FA",
            text="Восстановление доступа.",
            category="access",
        ),
    ]

    output_directory = tmp_path / "dense-index"

    build_dense_index(
        documents=documents,
        encoder=FakeDenseEncoder(),
        model_name="fake-model",
        output_directory=output_directory,
    )

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest == {
        "format_version": 1,
        "model_name": "fake-model",
        "embedding_dimension": 3,
        "normalized": True,
        "document_count": 2,
        "document_fingerprint": (compute_document_fingerprint(documents)),
        "document_format": ("title_newline_text"),
    }


def test_saved_index_can_be_loaded(
    tmp_path: Path,
) -> None:
    documents = [
        Document(
            doc_id="vpn_setup",
            title="Настройка VPN",
            text="Инструкция.",
            category="network",
        ),
        Document(
            doc_id="gitlab_recovery",
            title="GitLab 2FA",
            text="Восстановление.",
            category="access",
        ),
    ]
    output_directory = tmp_path / "dense-index"

    build_dense_index(
        documents=documents,
        encoder=FakeDenseEncoder(),
        model_name="fake-model",
        output_directory=output_directory,
    )

    loaded = FaissFlatVectorIndex.load(
        output_directory,
        expected_model_name="fake-model",
        expected_document_fingerprint=(compute_document_fingerprint(documents)),
    )

    assert loaded.size == 2
    assert loaded.dimension == 3
    assert loaded.doc_ids == (
        "vpn_setup",
        "gitlab_recovery",
    )


def test_rejects_empty_document_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=("cannot build dense index on empty documents corpus"),
    ):
        build_dense_index(
            documents=[],
            encoder=FakeDenseEncoder(),
            model_name="fake-model",
            output_directory=tmp_path,
        )
