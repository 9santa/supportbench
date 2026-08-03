import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True, slots=True)
class RRFGridPoint:
    bm25_weight: float
    dense_weight: float
    rrf_k: int
    candidate_k: int
    final_top_k: int

    def __post_init__(self) -> None:
        _validate_weight(self.bm25_weight, name="bm25_weight")
        _validate_weight(self.dense_weight, name="dense_weight")

        if self.bm25_weight == 0.0 and self.dense_weight == 0.0:
            raise ValueError("at least one retriever weight must be positive")

        _validate_positive_int(self.rrf_k, name="rrf_k")
        _validate_positive_int(self.candidate_k, name="candidate_k")
        _validate_positive_int(self.final_top_k, name="final_top_k")

        if self.final_top_k < 10:
            raise ValueError("final_top_k must be at least 10 to compute Recall@10")

    @property
    def name(self) -> str:
        return (
            f"bm25_{self.bm25_weight:g}"
            f"_dense_{self.dense_weight:g}"
            f"_rrf_{self.rrf_k}"
            f"_candidates_{self.candidate_k}"
            f"_top_{self.final_top_k}"
        )


@dataclass(frozen=True, slots=True)
class RRFGridDefinition:
    bm25_weight: float
    dense_weights: tuple[float, ...]
    rrf_k_values: tuple[int, ...]
    candidate_k_values: tuple[int, ...]
    final_top_k: int

    def __post_init__(self) -> None:
        if not self.dense_weights:
            raise ValueError("dense_weights must not be empty")

        if not self.rrf_k_values:
            raise ValueError("rrf_k_values must not be empty")

        if not self.candidate_k_values:
            raise ValueError("candidate_k_values must not be empty")

        _validate_unique(self.dense_weights, name="dense_weights")
        _validate_unique(self.rrf_k_values, name="rrf_k_values")
        _validate_unique(self.candidate_k_values, name="candidate_k_values")

        for point in self.points:
            # Constructing the points applies all cross-field validation.
            _ = point

    @property
    def points(self) -> tuple[RRFGridPoint, ...]:
        return tuple(
            RRFGridPoint(
                bm25_weight=self.bm25_weight,
                dense_weight=dense_weight,
                rrf_k=rrf_k,
                candidate_k=candidate_k,
                final_top_k=self.final_top_k,
            )
            for dense_weight in self.dense_weights
            for rrf_k in self.rrf_k_values
            for candidate_k in self.candidate_k_values
        )

    @property
    def max_candidate_k(self) -> int:
        return max(self.candidate_k_values)


type RRFGridConfig = RRFGridDefinition


def load_rrf_grid_definition(
    path: Path,
) -> RRFGridDefinition:
    try:
        with path.open(mode="r", encoding="utf-8") as file:
            raw_config: object = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid RRF grid YAML in {path}") from error

    if not isinstance(raw_config, dict):
        raise ValueError("RRF grid config root must be a mapping")

    config = cast(Mapping[object, object], raw_config)
    required_fields = {
        "bm25_weight",
        "dense_weights",
        "rrf_k_values",
        "candidate_k_values",
        "final_top_k",
    }
    actual_fields = set(config)

    missing_fields = required_fields - actual_fields
    unknown_fields = actual_fields - required_fields

    if missing_fields:
        raise ValueError("RRF grid config is missing fields: " + _format_fields(missing_fields))

    if unknown_fields:
        raise ValueError(
            "RRF grid config contains unknown fields: " + _format_fields(unknown_fields)
        )

    return RRFGridDefinition(
        bm25_weight=_require_float(config, "bm25_weight"),
        dense_weights=_require_float_tuple(config, "dense_weights"),
        rrf_k_values=_require_int_tuple(config, "rrf_k_values"),
        candidate_k_values=_require_int_tuple(config, "candidate_k_values"),
        final_top_k=_require_int(config, "final_top_k"),
    )


def load_rrf_grid_config(path: Path) -> RRFGridConfig:
    return load_rrf_grid_definition(path)


def _require_float(config: Mapping[object, object], field: str) -> float:
    value = config[field]

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")

    return float(value)


def _require_int(config: Mapping[object, object], field: str) -> int:
    value = config[field]

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")

    return value


def _require_float_tuple(
    config: Mapping[object, object],
    field: str,
) -> tuple[float, ...]:
    value = config[field]

    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")

    values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{field}[{index}] must be a number")
        values.append(float(item))

    return tuple(values)


def _require_int_tuple(
    config: Mapping[object, object],
    field: str,
) -> tuple[int, ...]:
    value = config[field]

    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")

    values: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field}[{index}] must be an integer")
        values.append(item)

    return tuple(values)


def _validate_weight(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_unique(values: tuple[object, ...], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _format_fields(fields: Collection[object]) -> str:
    return ", ".join(sorted(repr(field) for field in fields))
