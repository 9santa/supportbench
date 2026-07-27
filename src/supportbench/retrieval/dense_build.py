from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from supportbench.data.models import Document
from supportbench.retrieval.dense_encoder import (
    DenseEncoder,
    format_document,
)
from supportbench.retrieval.dense_index import (
    FaissFlatVectorIndex,
    compute_document_fingerprint,
)

DOCUMENT_FORMAT = "title_newline_text"


@dataclass(frozen=True, slots=True)
class DenseIndexBuildResult:
    document_count: int
    embedding_dimension: int
    encoding_seconds: float
    index_build_seconds: float
    output_directory: Path


def build_dense_index(
    *,
    documents: Sequence[Document],
    encoder: DenseEncoder,
    model_name: str,
    output_directory: Path,
) -> DenseIndexBuildResult:
    if not documents:
        raise ValueError("cannot build dense index on empty documents corpus")

    if not model_name.strip():
        raise ValueError("model_name must be non-empty")

    formatted_docs = [format_document(doc) for doc in documents]

    doc_ids = tuple(doc.doc_id for doc in documents)

    encoding_started = perf_counter()

    embeddings = encoder.encode_documents(formatted_docs)

    encoding_seconds = perf_counter() - encoding_started

    expected_shape = (
        len(documents),
        encoder.dimension,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            "unexpected document embedding shape: "
            f"expected: {expected_shape}, "
            f"got: {embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError("document embeddings must have dtype=np.float32")

    if not np.isfinite(embeddings).all():
        raise ValueError("document embeddings contain NaN or infinity")

    fingerprint = compute_document_fingerprint(documents)

    index_build_started = perf_counter()

    index = FaissFlatVectorIndex.build(
        embeddings,
        doc_ids,
        model_name=model_name,
        document_fingerprint=fingerprint,
        document_format=DOCUMENT_FORMAT,
    )

    index_build_seconds = perf_counter() - index_build_started

    index.save(output_directory)

    return DenseIndexBuildResult(
        document_count=len(documents),
        embedding_dimension=index.dimension,
        encoding_seconds=encoding_seconds,
        index_build_seconds=index_build_seconds,
        output_directory=output_directory,
    )
