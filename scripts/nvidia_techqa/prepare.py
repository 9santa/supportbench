import argparse
from dataclasses import asdict
from pathlib import Path

from supportbench.corpus.nvidia_techqa import prepare_nvidia_techqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NVIDIA TechQA-RAG-Eval for SupportBench.")

    parser.add_argument(
        "--dataset-zip",
        type=Path,
        required=True,
        help="ZIP containing NVIDIA train.json",
    )

    parser.add_argument(
        "--corpus-zip",
        type=Path,
        required=True,
        help="NVIDIA corpus.zip containing IBM Technotes",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for parsed, normalized JSONL and reports",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_nvidia_techqa(
        dataset_zip=args.dataset_zip,
        corpus_zip=args.corpus_zip,
        output_dir=args.output_dir,
    )

    print("NVIDIA TechQA preparation completed")
    for name, value in asdict(summary).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
