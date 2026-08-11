import argparse
import json
import os
from dataclasses import asdict, dataclass
from collections.abc import Sequence
from typing import cast

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from supportbench.simulator.models import CaseSeverity
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
from supportbench.simulator.commands import CreateSupportCaseCommand
from supportbench.simulator.service import EnterpriseService
from supportbench.simulator.postgres.lifecycle import (
    reset_world,
    delete_world,
)
from supportbench.tools.enterprise import (
    build_enterprise_tool_handlers,
)
from supportbench.tools.gateway import ToolGateway


DATABASE_URL_ENV = "SUPPORTBENCH_SIMULATOR_DATABASE_URL"
DEFAULT_WORLD_ID = "techqa-demo-v1"


@dataclass(slots=True)
class EnterpriseSimulatorRuntime:
    engine: Engine
    session_factory: sessionmaker[Session]
    service: EnterpriseService
    tool_gateway: ToolGateway

    def close(self) -> None:
        self.engine.dispose()


def build_enterprise_simulator(
    *,
    database_url: str,
) -> EnterpriseSimulatorRuntime:
    engine = build_engine(database_url)
    session_factory = build_session_factory(engine)

    service = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

    tool_gateway = ToolGateway(build_enterprise_tool_handlers(service))

    return EnterpriseSimulatorRuntime(
        engine=engine,
        session_factory=session_factory,
        service=service,
        tool_gateway=tool_gateway,
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

    # === SEED PARSER ===
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
            "access_denied",
        ),
        default="healthy",
    )

    # === STATUS PARSER ===
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

    # === INSTALLED PRODUCT PARSER ===
    installed_parser = subparsers.add_parser(
        "installed-product",
        help="Read an installed product from an asset.",
    )
    installed_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )
    installed_parser.add_argument(
        "--asset-id",
        required=True,
    )
    installed_parser.add_argument(
        "--product-key",
        required=True,
    )

    # === USER ENTITLEMENT PARSER ===
    entitlement_parser = subparsers.add_parser(
        "user-entitlement",
        help="Read a user's entitlement for a service.",
    )
    entitlement_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )
    entitlement_parser.add_argument(
        "--user-id",
        required=True,
    )
    entitlement_parser.add_argument(
        "--service-id",
        required=True,
    )

    # === CREATE SUPPORT CASE PARSER ===
    create_case_parser = subparsers.add_parser(
        "create-support-case",
        help="Create a support case in a simulator world.",
    )

    create_case_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )

    create_case_parser.add_argument(
        "--idempotency-key",
        required=True,
    )

    create_case_parser.add_argument(
        "--actor-user-id",
        required=True,
    )

    create_case_parser.add_argument(
        "--user-id",
        required=True,
    )

    create_case_parser.add_argument(
        "--service-id",
        required=True,
    )

    create_case_parser.add_argument(
        "--summary",
        required=True,
    )

    create_case_parser.add_argument(
        "--description",
        required=True,
    )

    create_case_parser.add_argument(
        "--severity",
        choices=(
            "low",
            "medium",
            "high",
            "critical",
        ),
        required=True,
    )

    # === RESET WORLD PARSER ===
    reset_parser = subparsers.add_parser(
        "reset-world",
        help="Atomically replace a simulator world with a fresh scenario.",
    )
    reset_parser.add_argument(
        "--world-id",
        required=True,
    )
    reset_parser.add_argument(
        "--scenario",
        choices=(
            "healthy",
            "dash_outage",
            "old_dash_version",
            "access_denied",
        ),
        required=True,
    )

    # === DELETE WORLD PARSER ===
    delete_parser = subparsers.add_parser(
        "delete-world",
        help="Delete a simulator world and all world-dependent stuff.",
    )
    delete_parser.add_argument(
        "--world-id",
        required=True,
    )

    # === GET SUPPORT CASE PARSER ===
    get_case_parser = subparsers.add_parser(
        "get-support-case",
        help="Read a support case from a simulator world.",
    )
    get_case_parser.add_argument(
        "--world-id",
        default=DEFAULT_WORLD_ID,
    )
    get_case_parser.add_argument(
        "--case-id",
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


def _run_installed_product(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    asset_id: str,
    product_key: str,
) -> int:
    installed_product = runtime.service.get_installed_product(
        world_id=world_id,
        asset_id=asset_id,
        product_key=product_key,
    )

    print(
        json.dumps(
            asdict(installed_product),
            indent=2,
        )
    )

    return 0


def _run_user_entitlement(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    user_id: str,
    service_id: str,
) -> int:
    entitlement = runtime.service.check_user_entitlement(
        world_id=world_id,
        user_id=user_id,
        service_id=service_id,
    )

    print(
        json.dumps(
            asdict(entitlement),
            indent=2,
        )
    )

    return 0


def _run_create_support_case(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    idempotency_key: str,
    actor_user_id: str,
    user_id: str,
    service_id: str,
    summary: str,
    description: str,
    severity: CaseSeverity,
) -> int:
    support_case = runtime.service.create_support_case(
        CreateSupportCaseCommand(
            world_id=world_id,
            idempotency_key=idempotency_key,
            actor_user_id=actor_user_id,
            user_id=user_id,
            service_id=service_id,
            summary=summary,
            description=description,
            severity=severity,
        )
    )

    print(
        json.dumps(
            asdict(support_case),
            indent=2,
            default=str,
        )
    )

    return 0


def _run_reset_world(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    scenario_name: ScenarioName,
) -> int:
    scenario = build_scenario(
        name=scenario_name,
        world_id=world_id,
    )

    reset_world(
        session_factory=runtime.session_factory,
        scenario=scenario,
    )

    print(
        json.dumps(
            {
                "world_id": world_id,
                "scenario_name": scenario_name,
                "reset": True,
            },
            indent=2,
        )
    )

    return 0


def _run_delete_world(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
) -> int:
    deleted = delete_world(
        session_factory=runtime.session_factory,
        world_id=world_id,
    )

    print(
        json.dumps(
            {
                "world_id": world_id,
                "deleted": deleted,
            },
            indent=2,
        )
    )

    return 0


def _run_get_support_case(
    *,
    runtime: EnterpriseSimulatorRuntime,
    world_id: str,
    case_id: str,
) -> int:
    support_case = runtime.service.get_support_case(
        world_id=world_id,
        case_id=case_id,
    )

    print(
        json.dumps(
            asdict(support_case),
            indent=2,
            default=str,
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

            case "installed-product":
                return _run_installed_product(
                    runtime=runtime,
                    world_id=args.world_id,
                    asset_id=args.asset_id,
                    product_key=args.product_key,
                )

            case "user-entitlement":
                return _run_user_entitlement(
                    runtime=runtime,
                    world_id=args.world_id,
                    user_id=args.user_id,
                    service_id=args.service_id,
                )

            case "create-support-case":
                return _run_create_support_case(
                    runtime=runtime,
                    world_id=args.world_id,
                    idempotency_key=args.idempotency_key,
                    actor_user_id=args.actor_user_id,
                    user_id=args.user_id,
                    service_id=args.service_id,
                    summary=args.summary,
                    description=args.description,
                    severity=cast(
                        CaseSeverity,
                        args.severity,
                    ),
                )

            case "reset-world":
                return _run_reset_world(
                    runtime=runtime,
                    world_id=args.world_id,
                    scenario_name=args.scenario,
                )

            case "delete-world":
                return _run_delete_world(
                    runtime=runtime,
                    world_id=args.world_id,
                )

            case "get-support-case":
                return _run_get_support_case(
                    runtime=runtime,
                    world_id=args.world_id,
                    case_id=args.case_id,
                )

            case _:
                parser.error(f"unknown command: {args.command}")

    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
