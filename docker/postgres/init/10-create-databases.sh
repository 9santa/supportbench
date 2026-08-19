#!/bin/sh
set -eu

create_role() {
    role_name="$1"
    role_password="$2"

    psql \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" \
        --set=ON_ERROR_STOP=1 \
        --set=role_name="$role_name" \
        --set=role_password="$role_password" <<'SQL'
SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L',
    :'role_name',
    :'role_password'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_roles
    WHERE rolname = :'role_name'
)
\gexec
SQL
}


create_database() {
    db_name="$1"
    owner_name="$2"

    psql \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" \
        --set=ON_ERROR_STOP=1 \
        --set=db_name="$db_name" \
        --set=owner_name="$owner_name" <<'SQL'
SELECT format(
    'CREATE DATABASE %I OWNER %I',
    :'db_name',
    :'owner_name'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_database
    WHERE datname = :'db_name'
)
\gexec
SQL
}


create_role \
    "$MLFLOW_DB_USER" \
    "$MLFLOW_DB_PASSWORD"

create_database \
    "$MLFLOW_DB" \
    "$MLFLOW_DB_USER"


create_role \
    "$SUPPORTBENCH_SIMULATOR_DB_USER" \
    "$SUPPORTBENCH_SIMULATOR_DB_PASSWORD"

create_database \
    "$SUPPORTBENCH_SIMULATOR_DB" \
    "$SUPPORTBENCH_SIMULATOR_DB_USER"
