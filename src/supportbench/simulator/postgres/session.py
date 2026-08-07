from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


type SessionFactory = sessionmaker[Session]


def build_engine(database_url: str) -> Engine:
    normalized_url = database_url.strip()

    if not normalized_url:
        raise ValueError("database_url must be non-empty")

    return create_engine(
        normalized_url,
        pool_pre_ping=True,
    )


def build_session_factory(engine: Engine) -> SessionFactory:
    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
