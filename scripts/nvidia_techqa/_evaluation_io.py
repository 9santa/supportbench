import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def prepare_output(
    *,
    parser: argparse.ArgumentParser,
    config_path: Path,
    results_path: Path,
    config_payload: dict[str, Any],
    resume: bool,
) -> None:
    if results_path.exists() and not resume:
        parser.error(f"results already exist: {results_path}; use --resume or a new name")

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))

        if existing != config_payload:
            parser.error("existing evaluation config does not match this run")
    else:
        write_json(config_path, config_payload)


def load_completed_query_ids(path: Path) -> set[str]:
    return {
        str(result["query_id"])
        for result in load_jsonl(path)
        if result.get("query_id") is not None
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, payloads: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for payload in payloads:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]

    return value
