from __future__ import annotations

import os

import pytest
from typing import Any

from config import settings
from src.indexing.schema import init_schema


def _schema_exists(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                to_regclass('public.documents') IS NOT NULL,
                to_regclass('public.chunks') IS NOT NULL,
                to_regclass('public.link_candidates') IS NOT NULL
            """
        )
        row = cur.fetchone()
    return bool(row and row[0] and row[1] and row[2])


@pytest.fixture(scope="session")
def db_conn() -> Any:
    psycopg_module = pytest.importorskip(
        "psycopg",
        reason="Skipping DB integration tests: psycopg is not installed in this environment.",
    )
    connect = psycopg_module.connect

    pgvector_psycopg = pytest.importorskip(
        "pgvector.psycopg",
        reason="Skipping DB integration tests: pgvector is not installed in this environment.",
    )
    register_vector = pgvector_psycopg.register_vector

    database_url = os.getenv("DATABASE_URL") or settings.database_url
    if not database_url:
        pytest.skip("Skipping retrieval integration tests: DATABASE_URL is not set.")

    conn = connect(database_url, autocommit=True)
    with conn.cursor() as cur:
        # Prevent indefinite hangs when stale sessions hold DDL locks.
        cur.execute("SET lock_timeout = '5s';")
        cur.execute("SET statement_timeout = '120s';")
        # Local test hygiene: clear abandoned sessions that would block schema setup.
        cur.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
                WHERE datname = current_database()
                  AND pid <> pg_backend_pid()
                  AND state = 'idle in transaction'
            """
        )
    if not _schema_exists(conn):
        init_schema(conn)
    register_vector(conn)
    # autocommit=True means every statement auto-commits immediately.
    # Without this, psycopg3's default transaction mode opens implicit
    # transactions on each query that stay open until an explicit
    # conn.commit().  Those open transactions hold read locks that block
    # index_batch's separate connection — causing indefinite hangs
    # whenever a test reads from db_conn before calling index_batch.
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def reset_index_tables(db_conn: Any):
    def _reset() -> None:
        with db_conn.transaction():
            with db_conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE link_candidates CASCADE;")
                cur.execute("TRUNCATE TABLE chunks CASCADE;")
                cur.execute("TRUNCATE TABLE documents CASCADE;")

    return _reset
