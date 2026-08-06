from pathlib import Path

import pytest

from supportbench.applications.nvidia_techqa import NvidiaTechQAContextConfig


def make_config(**overrides: object) -> NvidiaTechQAContextConfig:
    values: dict[str, object] = {
        "chunks_root": Path("chunks"),
        "index_root": Path("indexes"),
    }
    values.update(overrides)
    return NvidiaTechQAContextConfig(**values)  # type: ignore[arg-type]


def test_defaults_capture_frozen_online_retrieval_profile() -> None:
    config = make_config()

    assert config.bm25_weight == 1.0
    assert config.dense_weight == 1.5
    assert config.source_rrf_k == 10
    assert config.parent_aggregation == "capped_top_2_sum"
    assert config.parent_candidate_k == 20
    assert config.chunks_per_parent == 2
    assert config.evidence_selection == "within_parent_rerank"
    assert config.candidate_prior_weight == 1.25
    assert config.second_evidence_weight == 0.0
    assert config.context_tokenizer_name != config.dense_model_name
    assert config.model_context_window == 8_192
    assert config.reserved_output_tokens == 512


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_parents": 21}, "top_parents must not exceed parent_candidate_k"),
        (
            {"source_candidate_k": 19},
            "source_candidate_k must be at least parent_candidate_k",
        ),
        ({"candidate_prior_weight": float("nan")}, "must be finite and non-negative"),
        ({"chunk_config": "../chunks"}, "must be a non-empty path segment"),
        ({"evidence_selection": "unknown"}, "unknown evidence selection"),
        (
            {"reserved_output_tokens": 8_192},
            "reserved_output_tokens must be smaller than model_context_window",
        ),
    ],
)
def test_rejects_invalid_online_retrieval_profile(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)
