import argparse
import json
import os
from dataclasses import asdict, dataclass
from collections.abc import Sequence

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from supportbench.simulator.postgres.seed import seed_scenario
from supportbench.simulator.postgres.session import (
    build_engine,
    build_session_factory,
)
from supportbench.simulator.postgres.unit_of_work import (
    PostgresUnitOfWork,
)
from supportbench.simulator.scenarios import (
    ScenarioName,
    build_scenario,
)
from supportbench.simulator.service import EnterpriseService


DATABASE_URL_ENV = "SUPPORTBENCH_SIMULATOR_DATABASE_URL"
DEFAULT_WORLD_ID = "techqa-demo-v1"


@dataclass(slots=True)
class EnterpriseSimulatorRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    service: EnterpriseService

    def close(self) -> None:
        self.engine.dispose()


def build_enterprise_simulator(
    *,
    database_url: str,
) -> EnterpriseSimulatorRuntime:
    engine = build_engine(database_url)
    session_factory = build_session_factory(engine)

    service = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

    return EnterpriseSimulatorRuntime(
        engine=engine,
        session_factory=session_factory,
        service=service,
    )


def _database_url_from_env() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV, "").strip()

    if not database_url:
        raise RuntimeError(f"{DATABASE_URL_ENV} environment variable is not set")

    return database_url


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enterprise-simulator",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    seed_parser = subparsers.add_parser(
        "seed",
        help="Seed an isolated simulator world.",
    )
    seed_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )
    seed_parser.add_argument(
        "--scenario",
        choices=(
            "healthy",
            "dash_outage",
            "old_dash_version",
        ),
        default="healthy",
    )

    status_parser = subparsers.add_parser(
        "service-status",
        help="Get service state from a simulator world.",
    )
    status_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )
    status_parser.add_argument(
        "--service-id",
        required=True,
    )

    return parser


def _run_seed(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    scenario_name: ScenarioName,
) -> int:
    scenario = build_scenario(name=scenario_name, world_id=world_id)

    seed_scenario(
        session_factory=runtime.session_factory,
        scenario=scenario,
    )

    print(
        json.dumps(
            {
                "world_id": scenario.world.world_id,
                "scenario_name": scenario.world.scenario_name,
                "services": len(scenario.services),
            },
            indent=2,
        )
    )

    return 0


def _run_service_status(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    service_id: str,
) -> int:
    service = runtime.service.get_service_status(
        world_id=world_id,
        service_id=service_id,
    )

    print(
        json.dumps(
            asdict(service),
            indent=2,
        )
    )

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runtime = build_enterprise_simulator(database_url=_database_url_from_env())

    try:
        match args.command:
            case "seed":
                return _run_seed(
                    runtime=runtime,
                    world_id=args.world_id,
                    scenario_name=args.scenario,
                )

            case "service-status":
                return _run_service_status(
                    runtime=runtime,
                    world_id=args.world_id,
                    service_id=args.service_id,
                )

            case _:
                parser.error(f"unknown command: {args.command}")

    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
