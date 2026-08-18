from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from supportbench.experiments.base import (
    ExperimentRun,
    Scalar,
)


class MLflowExperimentRun(ExperimentRun):
    def __init__(
        self,
        *,
        mlflow_module: ModuleType,
        run_id: str,
    ) -> None:
        self._mlflow = mlflow_module
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_params(
        self,
        params: Mapping[str, Scalar],
    ) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        self._mlflow.log_metrics({name: float(value) for name, value in metrics.items()})

    def set_tags(
        self,
        tags: Mapping[str, str],
    ) -> None:
        self._mlflow.set_tags(dict(tags))

    def log_artifact(
        self,
        path: Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        if not path.is_file():
            raise ValueError(f"artifact is not a file: {path}")

        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)


class MLflowExperimentTracker:
    def __init__(
        self,
        *,
        tracking_uri: str,
        log_system_metrics: bool = True,  # CPU, memory, GPU metrics
    ) -> None:
        if not tracking_uri.strip():
            raise ValueError("tracking_uri must be non-empty")

        self._tracking_uri = tracking_uri
        self._log_system_metrics = log_system_metrics

    @contextmanager
    def start_run(
        self,
        *,
        experiment_name: str,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[ExperimentRun]:
        mlflow = _import_mlflow()

        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=run_name,
            tags=dict(tags or {}),
            log_system_metrics=self._log_system_metrics,
        ) as active_run:
            yield MLflowExperimentRun(
                mlflow_module=mlflow,
                run_id=active_run.info.run_id,
            )


def _import_mlflow() -> ModuleType:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow tracking required the 'tracking' optional dependency") from error

    return mlflow
