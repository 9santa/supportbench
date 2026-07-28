from pathlib import Path

import pytest

from supportbench.experiments.rrf_grid_config import load_rrf_grid_definition


def test_loads_rrf_grid_and_builds_cartesian_product(tmp_path: Path) -> None:
    path = tmp_path / "rrf_grid.yaml"
    path.write_text(
        """
bm25_weight: 1.0
dense_weights: [1.0, 2.0]
rrf_k_values: [10, 20]
candidate_k_values: [10, 50]
final_top_k: 10
""",
        encoding="utf-8",
    )

    definition = load_rrf_grid_definition(path)

    assert len(definition.points) == 8
    assert definition.max_candidate_k == 50
    assert definition.points[0].name == "bm25_1_dense_1_rrf_10_candidates_10_top_10"
    assert definition.points[-1].name == "bm25_1_dense_2_rrf_20_candidates_50_top_10"


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        (
            """
bm25_weight: 1.0
dense_weights: []
rrf_k_values: [10]
candidate_k_values: [10]
final_top_k: 10
""",
            "dense_weights must not be empty",
        ),
        (
            """
bm25_weight: 0.0
dense_weights: [0.0]
rrf_k_values: [10]
candidate_k_values: [10]
final_top_k: 10
""",
            "at least one retriever weight must be positive",
        ),
        (
            """
bm25_weight: 1.0
dense_weights: [1.0]
rrf_k_values: [10, 10]
candidate_k_values: [10]
final_top_k: 10
""",
            "rrf_k_values must not contain duplicates",
        ),
        (
            """
bm25_weight: 1.0
dense_weights: [1.0]
rrf_k_values: [10]
candidate_k_values: [10]
final_top_k: 5
""",
            "final_top_k must be at least 10 to compute Recall@10",
        ),
    ],
)
def test_rejects_invalid_grid_definition(
    tmp_path: Path,
    yaml_text: str,
    message: str,
) -> None:
    path = tmp_path / "rrf_grid.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_rrf_grid_definition(path)
