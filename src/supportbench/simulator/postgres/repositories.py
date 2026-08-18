import re
from typing import cast

from sqlalchemy import and_, case, func, insert, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session

from supportbench.simulator.models import (
    AuditEvent,
    CaseSeverity,
    CaseStatus,
    Environment,
    InstalledProduct,
    Product,
    ServiceInstance,
    ServiceStatus,
    SupportCase,
    UserEntitlement,
)
from supportbench.simulator.postgres.schema import (
    audit_events,
    installed_products,
    products,
    service_instances,
    support_cases,
    user_entitlements,
)


class PostgresProductRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def search(
        self,
        *,
        query: str,
        limit: int,
    ) -> tuple[Product, ...]:
        normalized_query = query.lower()
        key = func.lower(products.c.product_key)
        name = func.lower(products.c.display_name)

        statement = (
            select(products)
            .where(
                or_(
                    key == normalized_query,
                    name == normalized_query,
                    key.contains(normalized_query, autoescape=True),
                    name.contains(normalized_query, autoescape=True),
                )
            )
            .order_by(
                case(
                    (key == normalized_query, 0),
                    (name == normalized_query, 1),
                    else_=2,
                ),
                key,
            )
            .limit(limit)
        )

        rows = self._session.execute(statement).mappings().all()

        return tuple(
            Product(
                product_key=row["product_key"],
                display_name=row["display_name"],
            )
            for row in rows
        )


class PostgresServiceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def search(
        self,
        *,
        world_id: str,
        query: str,
        limit: int,
    ) -> tuple[ServiceInstance, ...]:
        terms = _service_search_terms(query)
        searchable_columns = (
            func.lower(service_instances.c.service_id),
            func.lower(service_instances.c.display_name),
            func.lower(service_instances.c.product_key),
            func.lower(service_instances.c.environment),
        )
        term_matches = tuple(
            or_(*(column.contains(term, autoescape=True) for column in searchable_columns))
            for term in terms
        )
        normalized_query = query.casefold().strip()
        service_id = func.lower(service_instances.c.service_id)
        display_name = func.lower(service_instances.c.display_name)

        statement = (
            select(service_instances)
            .where(
                service_instances.c.world_id == world_id,
                and_(*term_matches),
            )
            .order_by(
                case(
                    (service_id == normalized_query, 0),
                    (display_name == normalized_query, 1),
                    else_=2,
                ),
                service_id,
            )
            .limit(limit)
        )

        rows = self._session.execute(statement).mappings().all()

        return tuple(_service_instance_from_row(row) for row in rows)

    def get(
        self,
        *,
        world_id: str,
        service_id: str,
    ) -> ServiceInstance | None:
        statement = select(service_instances).where(
            service_instances.c.world_id == world_id,
            service_instances.c.service_id == service_id,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return _service_instance_from_row(row)


_SERVICE_SEARCH_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "current",
        "instance",
        "instances",
        "service",
        "services",
        "the",
    }
)


def _service_search_terms(query: str) -> tuple[str, ...]:
    terms = tuple(
        dict.fromkeys(
            term
            for term in re.findall(r"[a-z0-9]+", query.casefold())
            if term not in _SERVICE_SEARCH_STOP_WORDS
        )
    )
    return terms or (query.casefold().strip(),)


def _service_instance_from_row(row: RowMapping) -> ServiceInstance:
    return ServiceInstance(
        world_id=row["world_id"],
        service_id=row["service_id"],
        display_name=row["display_name"],
        product_key=row["product_key"],
        version=row["version"],
        environment=cast(
            Environment,
            row["environment"],
        ),
        status=cast(
            ServiceStatus,
            row["status"],
        ),
        owner_team=row["owner_team"],
    )


class PostgresInstalledProductRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def get(
        self,
        *,
        world_id: str,
        asset_id: str,
        product_key: str,
    ) -> InstalledProduct | None:
        statement = select(installed_products).where(
            installed_products.c.world_id == world_id,
            installed_products.c.asset_id == asset_id,
            installed_products.c.product_key == product_key,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return InstalledProduct(
            world_id=row["world_id"],
            asset_id=row["asset_id"],
            product_key=row["product_key"],
            version=row["version"],
            patch_level=row["patch_level"],
        )


class PostgresUserEntitlementRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def get(
        self,
        *,
        world_id: str,
        user_id: str,
        service_id: str,
    ) -> UserEntitlement | None:
        statement = select(user_entitlements).where(
            user_entitlements.c.world_id == world_id,
            user_entitlements.c.user_id == user_id,
            user_entitlements.c.service_id == service_id,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return UserEntitlement(
            world_id=row["world_id"],
            user_id=row["user_id"],
            service_id=row["service_id"],
            granted=row["granted"],
            role=row["role"],
        )


class PostgresSupportCaseRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def get_by_idempotency_key(
        self,
        *,
        world_id: str,
        idempotency_key: str,
    ) -> SupportCase | None:
        statement = select(support_cases).where(
            support_cases.c.world_id == world_id,
            support_cases.c.idempotency_key == idempotency_key,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return self._from_row(row)

    def get(
        self,
        *,
        world_id: str,
        case_id: str,
    ) -> SupportCase | None:
        statement = select(support_cases).where(
            support_cases.c.world_id == world_id,
            support_cases.c.case_id == case_id,
        )

        row = self._session.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return self._from_row(row)

    def add(
        self,
        support_case: SupportCase,
    ) -> None:
        self._session.execute(
            insert(support_cases).values(
                world_id=support_case.world_id,
                case_id=support_case.case_id,
                idempotency_key=(support_case.idempotency_key),
                actor_user_id=(support_case.actor_user_id),
                user_id=support_case.user_id,
                service_id=support_case.service_id,
                summary=support_case.summary,
                description=support_case.description,
                severity=support_case.severity,
                status=support_case.status,
                assigned_team=(support_case.assigned_team),
                created_at=support_case.created_at,
                updated_at=support_case.updated_at,
            )
        )

    def add_if_absent(
        self,
        support_case: SupportCase,
    ) -> bool:
        statement = (
            pg_insert(support_cases)
            .values(
                world_id=support_case.world_id,
                case_id=support_case.case_id,
                idempotency_key=support_case.idempotency_key,
                actor_user_id=support_case.actor_user_id,
                user_id=support_case.user_id,
                service_id=support_case.service_id,
                summary=support_case.summary,
                description=support_case.description,
                severity=support_case.severity,
                status=support_case.status,
                assigned_team=support_case.assigned_team,
                created_at=support_case.created_at,
                updated_at=support_case.updated_at,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    support_cases.c.world_id,
                    support_cases.c.idempotency_key,
                ]
            )
            .returning(support_cases.c.case_id)
        )

        inserted_case_id = self._session.execute(statement).scalar_one_or_none()

        return inserted_case_id is not None

    @staticmethod
    def _from_row(row: RowMapping) -> SupportCase:
        return SupportCase(
            world_id=row["world_id"],
            case_id=row["case_id"],
            idempotency_key=row["idempotency_key"],
            actor_user_id=row["actor_user_id"],
            user_id=row["user_id"],
            service_id=row["service_id"],
            summary=row["summary"],
            description=row["description"],
            severity=cast(
                CaseSeverity,
                row["severity"],
            ),
            status=cast(
                CaseStatus,
                row["status"],
            ),
            assigned_team=row["assigned_team"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresAuditEventRepository:
    def __init__(
        self,
        session: Session,
    ) -> None:
        self._session = session

    def add(
        self,
        event: AuditEvent,
    ) -> None:
        self._session.execute(
            insert(audit_events).values(
                world_id=event.world_id,
                event_id=event.event_id,
                event_type=event.event_type,
                actor_user_id=event.actor_user_id,
                entity_type=event.entity_type,
                entity_id=event.entity_id,
                occurred_at=event.occurred_at,
                metadata=event.metadata,
            )
        )
