from dataclasses import dataclass
from pathlib import Path

from supportbench.data.models import QueryExample
from supportbench.evaluation.retrieval_evaluator import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from supportbench.retrieval.bm25 import BM25Retriever
from supportbench.retrieval.inverted_index import InvertedIndex
from supportbench.experiments.evaluation_export import (
    build_bm25_experiment_summary,
    export_bm25_experiment_summary,
    export_query_evaluations,
)


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


@dataclass(frozen=True, slots=True)
class BM25ExperimentRun:
    config: BM25ExperimentConfig
    evaluation: RetrievalEvaluationResult


def run_bm25_experiment(
    *,
    index: InvertedIndex,
    queries: list[QueryExample],
    config: BM25ExperimentConfig,
) -> BM25ExperimentRun:
    selected_queries = [query for query in queries if query.split == config.split]

    if not selected_queries:
        raise ValueError(f"no queries found for split {config.split!r}")

    retriever = BM25Retriever(index, k1=config.k1, b=config.b)

    evaluation = evaluate_retriever(retriever, selected_queries, top_k=config.top_k)

    return BM25ExperimentRun(
        config=config,
        evaluation=evaluation,
    )


def same_bm25_experiment_run(
    run: BM25ExperimentRun,
    *,
    output_root: Path,
) -> Path:
    output_dir = output_root / run.config.name

    summary = build_bm25_experiment_summary(
        experiment=run.config.name,
        k1=run.config.k1,
        b=run.config.b,
        split=run.config.split,
        top_k=run.config.top_k,
        result=run.evaluation,
    )

    export_bm25_experiment_summary(
        summary,
        output_dir / "summary.json",
    )
    export_query_evaluations(
        run.evaluation,
        output_dir / "queries.jsonl",
    )

    return output_dir


def run_bm25_ablation(
    *,
    index: InvertedIndex,
    queries: list[QueryExample],
    configs: tuple[BM25ExperimentConfig, ...],
    output_root: Path,
) -> tuple[BM25ExperimentRun, ...]:
    if not configs:
        raise ValueError("ablation must contain at least one configuration")

    runs: list[BM25ExperimentRun] = []

    for config in configs:
        run = run_bm25_experiment(
            index=index,
            queries=queries,
            config=config,
        )

        same_bm25_experiment_run(
            run,
            output_root=output_root,
        )

        runs.append(run)

    return tuple(runs)
