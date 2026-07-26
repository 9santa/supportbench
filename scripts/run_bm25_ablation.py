import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from supportbench.data.loaders import (
    load_documents,
    load_queries,
)
from supportbench.experiments.bm25_ablation import (
    BM25ExperimentRun,
    run_bm25_ablation,
)
from supportbench.experiments.config_loader import (
    AblationParameter,
    load_bm25_ablation_configs,
)
from supportbench.retrieval.inverted_index import InvertedIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "bm25_ablation.yaml"
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "raw" / "documents.jsonl"
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "benchmark" / "queries_dev.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "bm25_ablation"


@dataclass(frozen=True, slots=True)
class CLIArguments:
    parameter: AblationParameter
    config_path: Path
    documents_path: Path
    queries_path: Path
    output_path: Path


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser(description="Run BM25 parameter ablation.")

    parser.add_argument(
        "--parameter",
        choices=("b", "k1"),
        required=True,
        help="BM25 parameter to change for ablation",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to BM25 ablation YAML config",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help="path to documents.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="path to queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="directory for experiment results",
    )

    args = parser.parse_args()

    return CLIArguments(
        parameter=cast(AblationParameter, args.parameter),
        config_path=cast(Path, args.config),
        documents_path=cast(
            Path,
            args.documents,
        ),
        queries_path=cast(
            Path,
            args.queries,
        ),
        output_path=cast(Path, args.output),
    )


def print_runs(
    runs: tuple[BM25ExperimentRun, ...],
) -> None:
    print(f"{'Experiment':<14}{'k1':>7}{'b':>7}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}")

    for run in runs:
        config = run.config
        result = run.evaluation

        print(
            f"{config.name:<14}"
            f"{config.k1:>7.2f}"
            f"{config.b:>7.2f}"
            f"{result.recall_at_1:>9.4f}"
            f"{result.recall_at_3:>9.4f}"
            f"{result.recall_at_5:>9.4f}"
            f"{result.mrr:>9.4f}"
        )


def main() -> None:
    args = parse_args()

    documents = load_documents(args.documents_path)

    index = InvertedIndex.build(documents)

    queries = load_queries(args.queries_path, set(index.document_ids))

    configs = load_bm25_ablation_configs(
        args.config_path,
        parameter=args.parameter,
    )

    output_root = args.output_path / args.parameter

    runs = run_bm25_ablation(
        index=index,
        queries=queries,
        configs=configs,
        output_root=output_root,
    )

    print(f"Parameter: {args.parameter}")
    print(f"Configurations: {len(runs)}")
    print(f"Output: {output_root}")
    print()

    print_runs(runs)


if __name__ == "__main__":
    main()
