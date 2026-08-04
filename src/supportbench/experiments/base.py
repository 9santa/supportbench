from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

type Scalar = str | int | float | bool


class ExperimentRun(Protocol):
    @property
    def run_id(self) -> str | None: ...

    def log_params(
        self,
        params: Mapping[str, Scalar],
    ) -> None: ...

    def log_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> None: ...

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None: ...

    def log_artifact(
        self,
        path: Path,
        *,
        artifact_path: str | None = None,
    ) -> None: ...


class ExperimentTracker(Protocol):
    def start_run(
        self, *, experiment_name: str, run_name: str, tags: Mapping[str, str] | None = None
    ) -> AbstractContextManager[ExperimentRun]: ...
