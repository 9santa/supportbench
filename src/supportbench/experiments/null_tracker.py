from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from supportbench.experiments.base import (
    ExperimentRun,
    Scalar,
)


class NullExperimentRun(ExperimentRun):
    @property
    def run_id(self) -> str | None:
        return None

    def log_params(
        self,
        params: Mapping[str, Scalar],
    ) -> None:
        pass

    def log_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        pass

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        pass

    def log_artifact(
        self,
        path: Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        pass


class NullExperimentTracker:
    @contextmanager
    def start_run(
        self,
        *,
        experiment_name: str,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[ExperimentRun]:
        yield NullExperimentRun()
