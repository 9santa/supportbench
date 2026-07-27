import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast

import faiss
import numpy as np
import numpy.typing as npt

from supportbench.data.models import Document
from supportbench.retrieval.dense_encoder import FloatMatrix

type FloatVector = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    doc_id: str
    score: float


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    format_version: int
    model_name: str
    embedding_dimension: int
    normalized: bool
    document_count: int
    document_fingerprint: str
    document_format: str

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported manifest format version")

        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")

        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        if not self.normalized:
            raise ValueError("dense index required vectors to be normalized")

        if self.document_count < 0:
            raise ValueError("document_count must not be negative")

        if not self.document_fingerprint.strip():
            raise ValueError("document_fingerprint must be non-empty")

        if not self.document_format.strip():
            raise ValueError("document_format must be non-empty")

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> Self:
        if not isinstance(value, dict):
            raise ValueError("manifest root must be a dict")

        data = cast(dict[str, object], value)

        required_fields = {
            "format_version",
            "model_name",
            "embedding_dimension",
            "normalized",
            "document_count",
            "document_fingerprint",
            "document_format",
        }

        missing_fields = required_fields - set(data)
        unknown_fields = set(data) - required_fields

        if missing_fields:
            raise ValueError("manifest is missing fields: " + ", ".join(sorted(missing_fields)))

        if unknown_fields:
            raise ValueError(
                "manifest contains unknown fields: " + ", ".join(sorted(unknown_fields))
            )

        return cls(
            format_version=_require_int(
                data,
                "format_version",
            ),
            model_name=_require_string(
                data,
                "model_name",
            ),
            embedding_dimension=_require_int(
                data,
                "embedding_dimension",
            ),
            normalized=_require_bool(
                data,
                "normalized",
            ),
            document_count=_require_int(
                data,
                "document_count",
            ),
            document_fingerprint=_require_string(
                data,
                "document_fingerprint",
            ),
            document_format=_require_string(
                data,
                "document_format",
            ),
        )


class DenseVectorIndex(Protocol):
    """Dense Index Contract."""

    @property
    def size(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def search(
        self,
        query_vector: FloatVector,
        top_k: int,
    ) -> list[VectorSearchResult]: ...


class FaissFlatVectorIndex:
    def __init__(
        self,
        *,
        index: Any,
        doc_ids: tuple[str, ...],
        manifest: DenseIndexManifest,
    ) -> None:
        self._index = index
        self._doc_ids = doc_ids
        self._manifest = manifest

        self._validate_internal_state()

    @classmethod
    def build(
        cls,
        vectors: FloatMatrix,
        doc_ids: Sequence[str],
        *,
        model_name: str,
        document_fingerprint: str,
        document_format: str = "title_newline_text",
    ) -> Self:
        matrix = _validate_document_vectors(vectors, doc_ids)
        normalized_doc_ids = _validate_doc_ids(doc_ids)

        dimension = int(matrix.shape[1])

        index = faiss.IndexFlatIP(dimension)

        if matrix.shape[0] > 0:
            index.add(matrix)

        manifest = DenseIndexManifest(
            format_version=1,
            model_name=model_name,
            embedding_dimension=dimension,
            normalized=True,
            document_count=len(normalized_doc_ids),
            document_fingerprint=(document_fingerprint),
            document_format=document_format,
        )

        return cls(index=index, doc_ids=normalized_doc_ids, manifest=manifest)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_document_fingerprint: str | None = None,
        expected_model_name: str | None = None,
    ) -> Self:
        manifest_path = directory / "manifest.json"
        doc_ids_path = directory / "doc_ids.json"
        index_path = directory / "index.faiss"

        try:
            raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_doc_ids: object = json.loads(doc_ids_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
        except OSError as error:
            raise ValueError(f"cannot load dense index from {directory}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid dense index json metadata in {directory}") from error
        except RuntimeError as error:
            raise ValueError(f"invalid FAISS index in {directory}") from error

        manifest = DenseIndexManifest.from_dict(raw_manifest)
        doc_ids = _parse_doc_ids(raw_doc_ids)

        if (
            expected_document_fingerprint is not None
            and manifest.document_fingerprint != expected_document_fingerprint
        ):
            raise ValueError(
                "document fingerprint mismatch: this dense index was build for different documents"
            )

        if expected_model_name is not None and manifest.model_name != expected_model_name:
            raise ValueError(
                "embedding model mismatch: "
                f"expected: {expected_model_name!r}, got: {manifest.model_name!r}"
            )

        return cls(
            index=index,
            doc_ids=doc_ids,
            manifest=manifest,
        )

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    @property
    def dimension(self) -> int:
        return self._manifest.embedding_dimension

    @property
    def manifest(self) -> DenseIndexManifest:
        return self._manifest

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    def search(
        self,
        query_vector: FloatVector,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vector = np.asarray(query_vector, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"query_vector dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )

        if not np.isfinite(vector).all():
            raise ValueError("query_vector contains NaN or infinity")

        norm = float(np.linalg.norm(vector))

        if not np.allclose(norm, 1.0, 1e-4):
            raise ValueError(
                "query vector must be L2-normalized, something went wrong with normalization"
            )

        if self.size == 0:
            return []

        result_count = min(top_k, self.size)

        query_matrix = np.ascontiguousarray(
            vector.reshape(1, -1),
            dtype=np.float32,
        )

        scores, positions = self._index.search(
            query_matrix,
            result_count,
        )

        results = [
            VectorSearchResult(
                doc_id=self._doc_ids[int(position)],
                score=float(score),
            )
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]

        return sorted(
            results,
            key=lambda result: (-result.score, result.doc_id),
        )

    def save(
        self,
        directory: Path,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        (directory / "doc_ids.json").write_text(
            json.dumps(
                list(self._doc_ids),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (directory / "manifest.json").write_text(
            json.dumps(
                asdict(self._manifest),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _validate_internal_state(self) -> None:
        index_dimension = int(self._index.d)
        index_size = int(self._index.ntotal)

        if index_dimension != self.dimension:
            raise ValueError("FAISS index dimension does not match the manifest dimension")

        if index_size != self.size:
            raise ValueError("FAISS index size does not match doc_ids count")

        if self._manifest.document_count != self.size:
            raise ValueError("manifest document_count does not match doc_ids count")


def compute_document_fingerprint(
    documents: Sequence[Document],
) -> str:
    digest = hashlib.sha256()

    for document in sorted(
        documents,
        key=lambda item: item.doc_id,
    ):
        record = {
            "doc_id": document.doc_id,
            "title": document.title,
            "text": document.text,
        }

        canonical_json = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        digest.update(canonical_json.encode(encoding="utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def _validate_document_vectors(
    vectors: FloatMatrix,
    doc_ids: Sequence[str],
) -> FloatMatrix:
    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("document embeddings matrix must be two-dimensional")

    if matrix.shape[1] <= 0:
        raise ValueError("embedding dimension must be positive")

    if matrix.shape[0] != len(doc_ids):
        raise ValueError("document embeddings matrix vector count must match doc_ids")

    if not np.isfinite(matrix).all():
        raise ValueError("document embeddings matrix contains NaN or infinity")

    if matrix.shape[0] > 0:
        norms = np.linalg.norm(matrix, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                "document vectors must be L2-normalized, something went wrong with normalization"
            )

    return np.ascontiguousarray(
        matrix,
        dtype=np.float32,
    )


def _validate_doc_ids(doc_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(doc_ids)

    for i, doc_id in enumerate(normalized):
        if not doc_id.strip():
            raise ValueError("doc_id is empty at pos", i)

    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate document IDs are not allowed")

    return normalized


def _parse_doc_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("doc_ids.json must contain a list")

    if not all(isinstance(item, str) for item in value):
        raise ValueError("all document IDs must be strings")

    return _validate_doc_ids(cast(list[str], value))


def _require_string(
    data: dict[str, object],
    field: str,
) -> str:
    value = data[field]

    if not isinstance(value, str):
        raise ValueError(f"manifest field {field!r} must be a string")

    return value


def _require_int(
    data: dict[str, object],
    field: str,
) -> int:
    value = data[field]

    if not isinstance(value, int):
        raise ValueError(f"manifest field {field!r} must be an integer")

    return value


def _require_bool(
    data: dict[str, object],
    field: str,
) -> bool:
    value = data[field]

    if not isinstance(value, bool):
        raise ValueError(f"manifest field {field!r} must be a boolean")

    return value
