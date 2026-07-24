from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25ExperimentConfig:
    name: str
    k1: float
    b: float
    split: str = "dev"
    top_k: int = 5

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must be non-empty")

        if self.k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        if not self.split.strip():
            raise ValueError("split must be non-empty")

        if self.top_k < 5:
            raise ValueError("top_k must be at least 5 to compute Recall@5")
