import os
from datetime import datetime, timezone
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from supportbench.simulator.models import (
    AuditEvent,
    SupportCase,
)
from supportbench.simulator.commands import (
    CreateSupportCaseCommand,
)
from supportbench.simulator.postgres.schema import (
    audit_events,
    simulator_worlds,
    support_cases,
)
from supportbench.simulator.postgres.repositories import (
    PostgresSupportCaseRepository,
)
from supportbench.simulator.postgres.seed import seed_scenario
from supportbench.simulator.postgres.session import (
    build_engine,
    build_session_factory,
)
from supportbench.simulator.postgres.unit_of_work import (
    PostgresUnitOfWork,
)
from supportbench.simulator.scenarios import build_scenario
from supportbench.simulator.service import EnterpriseService


pytestmark = pytest.mark.postgres


def _database_url() -> str:
    value = os.environ.get("SUPPORTBENCH_SIMULATOR_DATABASE_URL", "").strip()

    if not value:
        pytest.skip("SUPPORTBENCH_SIMULATOR_DATABASE_URL is not set")

    return value


def test_create_support_case_persists_case_and_audit() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        enterprise = EnterpriseService(
            uow_factory=lambda: PostgresUnitOfWork(session_factory),
        )

        result = enterprise.create_support_case(
            CreateSupportCaseCommand(
                world_id=world_id,
                idempotency_key="request-001",
                actor_user_id="alice",
                user_id="alice",
                service_id="webgui-noc-prod",
                summary="Cannot access Web GUI",
                description=("Alice cannot access production Web GUI."),
                severity="high",
            )
        )

        assert result.status == "open"
        assert result.assigned_team == "noc-platform"

        with session_factory() as session:
            case_rows = (
                session.execute(select(support_cases).where(support_cases.c.world_id == world_id))
                .mappings()
                .all()
            )

            audit_rows = (
                session.execute(select(audit_events).where(audit_events.c.world_id == world_id))
                .mappings()
                .all()
            )

        assert len(case_rows) == 1
        assert len(audit_rows) == 1

        assert audit_rows[0]["entity_id"] == result.case_id

        assert audit_rows[0]["event_type"] == "support_case.created"

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()


def test_create_support_case_is_idempotent_in_postgres() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        command = CreateSupportCaseCommand(
            world_id=world_id,
            idempotency_key="request-001",
            actor_user_id="alice",
            user_id="alice",
            service_id="webgui-noc-prod",
            summary="Cannot access Web GUI",
            description="Cannot access Web GUI.",
            severity="high",
        )

        first = enterprise.create_support_case(command)
        second = enterprise.create_support_case(command)

        assert second == first

        with session_factory() as session:
            cases = (
                session.execute(select(support_cases).where(support_cases.c.world_id == world_id))
                .mappings()
                .all()
            )

            events = (
                session.execute(select(audit_events).where(audit_events.c.world_id == world_id))
                .mappings()
                .all()
            )

        assert len(cases) == 1
        assert len(events) == 1

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()


def test_idempotency_key_is_scoped_to_world() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_a = f"test-{uuid4()}"
    world_b = f"test-{uuid4()}"

    try:
        for world_id in (world_a, world_b):
            seed_scenario(
                session_factory=session_factory,
                scenario=build_scenario(
                    name="healthy",
                    world_id=world_id,
                ),
            )

        enterprise = EnterpriseService(uow_factory=lambda: PostgresUnitOfWork(session_factory))

        def command(world_id: str) -> CreateSupportCaseCommand:
            return CreateSupportCaseCommand(
                world_id=world_id,
                idempotency_key="request-001",
                actor_user_id="alice",
                user_id="alice",
                service_id="webgui-noc-prod",
                summary="Cannot access Web GUI",
                description="Cannot access Web GUI.",
                severity="high",
            )

        case_a = enterprise.create_support_case(command(world_a))
        case_b = enterprise.create_support_case(command(world_b))

        assert case_a.world_id == world_a
        assert case_b.world_id == world_b
        assert case_a.case_id != case_b.case_id

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(
                        simulator_worlds.c.world_id.in_([world_a, world_b])
                    )
                )

        engine.dispose()


def test_case_is_rolled_back_when_audit_insert_fails() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        now = datetime.now(timezone.utc)

        support_case = SupportCase(
            world_id=world_id,
            case_id="CASE-ROLLBACK",
            idempotency_key="request-rollback",
            actor_user_id="alice",
            user_id="alice",
            service_id="webgui-noc-prod",
            summary="Test rollback",
            description="Test rollback.",
            severity="high",
            status="open",
            assigned_team="noc-platform",
            created_at=now,
            updated_at=now,
        )

        invalid_audit = AuditEvent(
            world_id=world_id,
            event_id="EVENT-ROLLBACK",
            event_type="support_case.created",
            # deliberately does not exist
            actor_user_id="missing-user",
            entity_type="support_case",
            entity_id=support_case.case_id,
            occurred_at=now,
            metadata={},
        )

        with pytest.raises(IntegrityError):
            with PostgresUnitOfWork(session_factory) as uow:
                inserted = uow.support_cases.add_if_absent(support_case)

                assert inserted is True

                uow.audit_events.add(invalid_audit)

                uow.commit()

        with session_factory() as session:
            persisted = (
                session.execute(
                    select(support_cases).where(
                        support_cases.c.world_id == world_id,
                        support_cases.c.case_id == "CASE-ROLLBACK",
                    )
                )
                .mappings()
                .one_or_none()
            )

        assert persisted is None

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()


def test_support_case_concurrent_conflict() -> None:
    engine = build_engine(_database_url())
    session_factory = build_session_factory(engine)

    world_id = f"test-{uuid4()}"

    def _try_insert(
        *,
        session_factory,
        barrier: Barrier,
        support_case: SupportCase,
    ) -> bool:
        with session_factory() as session:
            repository = PostgresSupportCaseRepository(session)

            with session.begin():
                barrier.wait()

                return repository.add_if_absent(support_case)

    try:
        seed_scenario(
            session_factory=session_factory,
            scenario=build_scenario(
                name="healthy",
                world_id=world_id,
            ),
        )

        now = datetime.now(timezone.utc)

        case_a = SupportCase(
            world_id=world_id,
            case_id="CASE-A",
            idempotency_key="same",
            actor_user_id="alice",
            user_id="alice",
            service_id="webgui-noc-prod",
            summary="Test conflict",
            description="Test conflict.",
            severity="high",
            status="open",
            assigned_team="noc-platform",
            created_at=now,
            updated_at=now,
        )

        case_b = SupportCase(
            world_id=world_id,
            case_id="CASE-B",
            idempotency_key="same",
            actor_user_id="alice",
            user_id="alice",
            service_id="webgui-noc-prod",
            summary="Test conflict",
            description="Test conflict.",
            severity="high",
            status="open",
            assigned_team="noc-platform",
            created_at=now,
            updated_at=now,
        )

        barrier = Barrier(2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda candidate: _try_insert(
                        session_factory=session_factory,
                        barrier=barrier,
                        support_case=candidate,
                    ),
                    [case_a, case_b],
                )
            )

        assert sorted(results) == [False, True]

        with session_factory() as session:
            cases = (
                session.execute(
                    select(support_cases).where(
                        support_cases.c.world_id == world_id,
                        support_cases.c.idempotency_key == "same",
                    )
                )
                .mappings()
                .all()
            )

        assert len(cases) == 1

    finally:
        with session_factory() as session:
            with session.begin():
                session.execute(
                    delete(simulator_worlds).where(simulator_worlds.c.world_id == world_id)
                )

        engine.dispose()
