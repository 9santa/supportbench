from pathlib import Path
from typing import Literal

import yaml

from supportbench.experiments.bm25_ablation import BM25ExperimentConfig

type AblationParameter = Literal["b", "k1"]


def load_bm25_ablation_configs(
    path: Path,
    *,
    parameter: AblationParameter,
) -> tuple[BM25ExperimentConfig, ...]:
    with path.open(mode="r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiments = [BM25ExperimentConfig(**d) for d in config[parameter]]

    return tuple(experiments)
