from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    MetaData,
    String,
    Table,
    Boolean,
    ForeignKeyConstraint,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": ("fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"),
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


simulator_worlds = Table(
    "simulator_worlds",  # name
    metadata,
    Column(
        "world_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "scenario_name",
        String(128),
        nullable=False,
    ),
)


products = Table(
    "products",
    metadata,
    Column(
        "product_key",
        String(64),
        primary_key=True,
    ),
    Column(
        "display_name",
        String(255),
        nullable=False,
    ),
)


service_instances = Table(
    "service_instances",
    metadata,
    Column(
        "world_id",
        String(128),
        ForeignKey(
            "simulator_worlds.world_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
    Column(
        "service_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "display_name",
        String(255),
        nullable=False,
    ),
    Column(
        "product_key",
        String(64),
        ForeignKey("products.product_key"),
        nullable=False,
    ),
    Column(
        "version",
        String(128),
        nullable=False,
    ),
    Column(
        "environment",
        String(32),
        nullable=False,
    ),
    Column(
        "status",
        String(32),
        nullable=False,
    ),
    Column(
        "owner_team",
        String(128),
        nullable=False,
    ),
    CheckConstraint(
        "environment IN ('production', 'staging', 'development')",
        name="environment",
    ),
    CheckConstraint(
        "status IN ('operational', 'degraded', 'outage', 'maintenance')",
        name="status",
    ),
)


assets = Table(
    "assets",
    metadata,
    Column(
        "world_id",
        String(128),
        ForeignKey(
            "simulator_worlds.world_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
    Column(
        "asset_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "hostname",
        String(255),
        nullable=False,
    ),
    Column(
        "operating_system",
        String(255),
        nullable=False,
    ),
    Column(
        "environment",
        String(32),
        nullable=False,
    ),
    CheckConstraint(
        ("environment IN ('production','staging','development')"),
        name="environment",
    ),
)


installed_products = Table(
    "installed_products",
    metadata,
    Column(
        "world_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "asset_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "product_key",
        String(64),
        ForeignKey("products.product_key"),
        primary_key=True,
    ),
    Column(
        "version",
        String(128),
        nullable=False,
    ),
    Column(
        "patch_level",
        String(128),
        nullable=False,
    ),
    ForeignKeyConstraint(
        [
            "world_id",
            "asset_id",
        ],
        [
            "assets.world_id",
            "assets.asset_id",
        ],
        ondelete="CASCADE",
    ),
)


users = Table(
    "users",
    metadata,
    Column(
        "world_id",
        String(128),
        ForeignKey(
            "simulator_worlds.world_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    ),
    Column(
        "user_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "display_name",
        String(255),
        nullable=False,
    ),
    Column(
        "department",
        String(128),
        nullable=False,
    ),
)


user_entitlements = Table(
    "user_entitlements",
    metadata,
    Column(
        "world_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "user_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "service_id",
        String(128),
        primary_key=True,
    ),
    Column(
        "granted",
        Boolean,
        nullable=False,
    ),
    Column(
        "role",
        String(64),
        nullable=False,
    ),
    ForeignKeyConstraint(
        [
            "world_id",
            "user_id",
        ],
        [
            "users.world_id",
            "users.user_id",
        ],
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        [
            "world_id",
            "service_id",
        ],
        [
            "service_instances.world_id",
            "service_instances.service_id",
        ],
        ondelete="CASCADE",
    ),
)
