import argparse
import os

from supportbench.experiments.base import ExperimentTracker
from supportbench.experiments.mlflow_tracker import MLflowExperimentTracker
from supportbench.experiments.null_tracker import NullExperimentTracker


def add_tracking_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_experiment: str,
) -> None:
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--tracking",
        dest="tracking",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    group.add_argument(
        "--no-tracking",
        dest="tracking",
        action="store_false",
        help="Disable MLflow experiment tracking.",
    )

    parser.set_defaults(tracking=None)

    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=default_experiment,
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
    )
    parser.add_argument(
        "--no-system-metrics",
        action="store_true",
    )


def resolve_tracker(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> ExperimentTracker:
    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    enabled = bool(tracking_uri) if args.tracking is None else args.tracking

    if not enabled:
        return NullExperimentTracker()

    if not tracking_uri:
        parser.error(
            "MLflow tracking is enabled but MLflow tracking URI was not provided "
            "and does not exist as env variable"
        )

    return MLflowExperimentTracker(
        tracking_uri=tracking_uri,
        log_system_metrics=(not args.no_system_metrics),
    )
