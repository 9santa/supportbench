from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from supportbench.data.models import Document
from supportbench.retrieval.dense_index import (
    FaissFlatVectorIndex,
    compute_document_fingerprint,
)


type FloatMatrix = NDArray[np.float32]
type FloatVector = NDArray[np.float32]


@pytest.fixture
def document_vectors() -> FloatMatrix:
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def vector_index(
    document_vectors: FloatMatrix,
) -> FaissFlatVectorIndex:
    return FaissFlatVectorIndex.build(
        document_vectors,
        (
            "horizontal",
            "vertical",
            "diagonal",
        ),
        model_name="fake-model",
        document_fingerprint="fingerprint-1",
    )


def test_exact_vector_is_ranked_first(
    vector_index: FaissFlatVectorIndex,
) -> None:
    query: FloatVector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = vector_index.search(
        query,
        top_k=3,
    )

    assert [result.doc_id for result in results] == [
        "horizontal",
        "diagonal",
        "vertical",
    ]

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.70710677)
    assert results[2].score == pytest.approx(0.0)


def test_top_k_is_applied(
    vector_index: FaissFlatVectorIndex,
) -> None:
    query: FloatVector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = vector_index.search(
        query,
        top_k=2,
    )

    assert len(results) == 2


def test_top_k_above_index_size_is_supported(
    vector_index: FaissFlatVectorIndex,
) -> None:
    query: FloatVector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = vector_index.search(
        query,
        top_k=100,
    )

    assert len(results) == 3


def test_rejects_invalid_query_dimension(
    vector_index: FaissFlatVectorIndex,
) -> None:
    query: FloatVector = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        vector_index.search(
            query,
            top_k=5,
        )


def test_rejects_non_normalized_query(
    vector_index: FaissFlatVectorIndex,
) -> None:
    query: FloatVector = np.array(
        [2.0, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="must be L2-normalized",
    ):
        vector_index.search(
            query,
            top_k=5,
        )


def test_rejects_duplicate_doc_ids(
    document_vectors: FloatMatrix,
) -> None:
    with pytest.raises(
        ValueError,
        match="duplicate document IDs",
    ):
        FaissFlatVectorIndex.build(
            document_vectors,
            (
                "duplicate",
                "duplicate",
                "other",
            ),
            model_name="fake-model",
            document_fingerprint="fingerprint",
        )


def test_rejects_vector_count_mismatch() -> None:
    vectors: FloatMatrix = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="vector count must match",
    ):
        FaissFlatVectorIndex.build(
            vectors,
            ("first", "second"),
            model_name="fake-model",
            document_fingerprint="fingerprint",
        )


def test_save_and_load_roundtrip(
    tmp_path: Path,
    vector_index: FaissFlatVectorIndex,
) -> None:
    directory = tmp_path / "dense-index"

    vector_index.save(directory)

    loaded = FaissFlatVectorIndex.load(
        directory,
        expected_document_fingerprint=("fingerprint-1"),
        expected_model_name="fake-model",
    )

    assert loaded.size == 3
    assert loaded.dimension == 2
    assert loaded.doc_ids == (
        "horizontal",
        "vertical",
        "diagonal",
    )

    query: FloatVector = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    original_results = vector_index.search(
        query,
        top_k=3,
    )
    loaded_results = loaded.search(
        query,
        top_k=3,
    )

    assert [result.doc_id for result in loaded_results] == [
        result.doc_id for result in original_results
    ]

    assert [result.score for result in loaded_results] == pytest.approx(
        [result.score for result in original_results]
    )


def test_rejects_document_fingerprint_mismatch(
    tmp_path: Path,
    vector_index: FaissFlatVectorIndex,
) -> None:
    directory = tmp_path / "dense-index"
    vector_index.save(directory)

    with pytest.raises(
        ValueError,
        match="document fingerprint mismatch",
    ):
        FaissFlatVectorIndex.load(
            directory,
            expected_document_fingerprint=("different-fingerprint"),
        )


def test_document_fingerprint_is_order_independent() -> None:
    first = Document(
        doc_id="first",
        title="First",
        text="First text",
        category="test",
    )
    second = Document(
        doc_id="second",
        title="Second",
        text="Second text",
        category="test",
    )

    assert compute_document_fingerprint([first, second]) == compute_document_fingerprint(
        [second, first]
    )


def test_document_fingerprint_changes_with_text() -> None:
    original = Document(
        doc_id="document",
        title="Title",
        text="Original text",
        category="test",
    )
    changed = Document(
        doc_id="document",
        title="Title",
        text="Changed text",
        category="test",
    )

    assert compute_document_fingerprint([original]) != compute_document_fingerprint([changed])
