from pathlib import Path

from supportbench.experiments.config_loader import (
    load_bm25_ablation_configs,
)


def test_loads_selected_ablation_group(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ablation.yaml"
    path.write_text(
        """
b:
  - name: b_0_00
    k1: 1.5
    b: 0.0

  - name: b_0_75
    k1: 1.5
    b: 0.75
    split: dev
    top_k: 5

k1:
  - name: k1_0_5
    k1: 0.5
    b: 0.75
""",
        encoding="utf-8",
    )

    configs = load_bm25_ablation_configs(
        path,
        parameter="b",
    )

    assert [config.name for config in configs] == [
        "b_0_00",
        "b_0_75",
    ]
    assert [config.b for config in configs] == [
        0.0,
        0.75,
    ]
    assert all(config.k1 == 1.5 for config in configs)
    assert all(config.split == "dev" for config in configs)
    assert all(config.top_k == 5 for config in configs)


