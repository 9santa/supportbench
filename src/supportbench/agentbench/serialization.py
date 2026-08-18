import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_json_artifact(
    *,
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            _to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _to_jsonable(value: object) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items() if key != "thinking"}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return sorted(_to_jsonable(item) for item in value)

    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)
