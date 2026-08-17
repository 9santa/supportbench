import hashlib
import json
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session, sessionmaker

from supportbench.agentbench.models import (
    AgentBenchWorldSnapshot,
)
from supportbench.simulator.postgres.schema import (
    assets,
    audit_events,
    installed_products,
    products,
    service_instances,
    simulator_worlds,
    support_cases,
    user_entitlements,
    users,
)

_WORLD_TABLES = (
    simulator_worlds,
    products,
    service_instances,
    assets,
    installed_products,
    users,
    user_entitlements,
    support_cases,
    audit_events,
)


class PostgresAgentBenchSnapshotter:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def snapshot(
        self,
        *,
        world_id: str,
    ) -> AgentBenchWorldSnapshot:
        """Saves (snapshots) the current state of the world `world_id`."""

        payload: dict[
            str,
            list[dict[str, object]],
        ] = {}

        with self._session_factory() as session:
            for table in _WORLD_TABLES:
                statement = select(table)
                world_id_column = table.c.get("world_id")

                if world_id_column is not None:
                    statement = statement.where(world_id_column == world_id)

                rows = session.execute(statement).mappings()

                payload[table.name] = sorted(
                    (_normalize_row(row) for row in rows),
                    key=_canonical_sort_key,
                )

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

        fingerprint = hashlib.sha256(encoded).hexdigest()

        return AgentBenchWorldSnapshot(
            fingerprint=fingerprint,
            support_case_count=len(payload["support_cases"]),
            audit_event_count=len(payload["audit_events"]),
        )


def _normalize_row(
    row: RowMapping,
) -> dict[str, object]:
    return {str(key): value for key, value in row.items()}


def _canonical_sort_key(
    row: Mapping[str, object],
) -> str:
    """Important for deterministic fingerprint hashing"""
    return json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
