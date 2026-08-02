import argparse
from pathlib import Path

from supportbench.benchmark.beir import (
    BEIR_DATASETS,
    download_beir_dataset,
    load_beir_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate an official BEIR dataset.")
    parser.add_argument("--dataset", choices=tuple(BEIR_DATASETS), default="scifact")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "beir",
    )
    args = parser.parse_args()
    spec = BEIR_DATASETS[args.dataset]
    directory = download_beir_dataset(spec, output_root=args.output_root)
    dataset = load_beir_dataset(
        directory,
        name=spec.name,
        split=spec.default_split,
    )
    print(f"Dataset: {dataset.name}")
    print(f"Split: {dataset.split}")
    print(f"Documents: {len(dataset.documents):,}")
    print(f"Queries: {len(dataset.queries):,}")
    print(f"Directory: {directory}")


if __name__ == "__main__":
    main()
