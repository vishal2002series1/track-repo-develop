# src/db/database.py
"""
🟢 AZURE POSTGRESQL MIGRATION
=============================
All Agent / Workflow CRUD now targets Azure PostgreSQL in the "Agents" schema.

Previously this module pointed at a local SQLite file
(`data_local/aeon_factory.db`). That file is no longer used by the API.

The schema name is configurable via env var AGENTS_PG_SCHEMA (default: "Agents").
The schema is auto-created on import if it does not yet exist.
"""

import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base

from src.db.pg_connection import get_pg_connection_string

# 🟢 Schema where Agent/Workflow metadata lives in Azure PostgreSQL.
AGENTS_PG_SCHEMA = os.getenv("AGENTS_PG_SCHEMA", "Agents") # Give name to set in PostgreSQL.

SQLALCHEMY_DATABASE_URL = get_pg_connection_string()

# pool_pre_ping avoids stale Azure connections being reused after idle timeouts.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)


# Ensure every new connection sets search_path to the Agents schema so
# unqualified table names resolve correctly for ORM queries.
@event.listens_for(engine, "connect")
def _set_search_path(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f'SET search_path TO "{AGENTS_PG_SCHEMA}"')
    finally:
        cursor.close()


def ensure_agents_schema_exists() -> None:
    """Create the Agents schema if it does not already exist."""
    with engine.connect() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{AGENTS_PG_SCHEMA}"'))
        conn.commit()


# Create schema eagerly so Base.metadata.create_all (called by main.py) can
# place tables inside it on the first run.
ensure_agents_schema_exists()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()