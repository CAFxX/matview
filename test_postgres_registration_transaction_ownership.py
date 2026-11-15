#!/usr/bin/env python3
"""Native PostgreSQL regression for registration transaction ownership.

The manager's atomic DDL contract requires an idle connection with autocommit
disabled. Verify both unsafe modes against a real psycopg2 connection and an
independent observer, not only a fake DB-API adapter.
"""
from __future__ import annotations

import os
import uuid

import psycopg2
from psycopg2 import extensions

from matview_sync import Dialect, MaterializedViewManager


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError(
            "POSTGRES_DSN is required; this test must run against native PostgreSQL"
        )

    conn = psycopg2.connect(dsn)
    observer = psycopg2.connect(dsn)
    suffix = uuid.uuid4().hex[:16]
    table = "mv_tx_owner_" + suffix
    view = "mv_tx_owner_view_" + suffix
    try:
        manager = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)

        # With autocommit enabled, each DDL statement would commit separately.
        # The guard must reject before issuing any manager query or DDL.
        conn.autocommit = True
        for operation in (
            lambda: manager.register_view(
                f"CREATE VIEW {view} AS SELECT 1 AS value;"
            ),
            lambda: manager.sync_views(
                f"CREATE VIEW {view} AS SELECT 1 AS value;"
            ),
        ):
            try:
                operation()
            except RuntimeError as exc:
                assert "autocommit=False" in str(exc), str(exc)
            else:
                raise AssertionError("registration accepted an autocommit connection")
        with observer.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"public.{view}",))
            assert cur.fetchone()[0] is None
        conn.autocommit = False

        # Successful and no-op registration/sync must close transactions they
        # opened for introspection, otherwise the next operation would mistake
        # manager-owned reads for caller-owned work.
        view_sql = f"CREATE VIEW {view} AS SELECT 1 AS value;"
        manager.register_view(view_sql)
        assert conn.get_transaction_status() == extensions.TRANSACTION_STATUS_IDLE
        manager.register_view(view_sql)
        assert conn.get_transaction_status() == extensions.TRANSACTION_STATUS_IDLE
        manager.sync_views(view_sql)
        assert conn.get_transaction_status() == extensions.TRANSACTION_STATUS_IDLE

        # A caller-owned transaction must not be committed or rolled back by
        # registration/sync. Its uncommitted row stays invisible to another
        # connection, and the transaction remains active after both rejections.
        with conn.cursor() as cur:
            cur.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute(f'INSERT INTO "{table}" VALUES (1)')
        assert conn.get_transaction_status() == extensions.TRANSACTION_STATUS_INTRANS

        for operation in (
            lambda: manager.register_view(
                f"CREATE VIEW {view} AS SELECT 1 AS value;"
            ),
            lambda: manager.sync_views(
                f"CREATE VIEW {view} AS SELECT 1 AS value;"
            ),
        ):
            try:
                operation()
            except RuntimeError as exc:
                assert "active caller transaction" in str(exc), str(exc)
            else:
                raise AssertionError("registration accepted a caller-owned transaction")

            assert (
                conn.get_transaction_status() == extensions.TRANSACTION_STATUS_INTRANS
            ), "manager changed the caller's transaction state"
            with observer.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                assert cur.fetchone()[0] == 0, "manager committed caller-owned DML"

        conn.rollback()
        with observer.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            assert cur.fetchone()[0] == 0, "caller rollback did not undo its own DML"
        print(
            "PASS: native PostgreSQL registration rejects unsafe transaction ownership "
            "without committing or rolling back caller work"
        )
    finally:
        try:
            if not conn.autocommit:
                conn.rollback()
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table}"')
                cur.execute(f'DROP VIEW IF EXISTS "{view}"')
            conn.commit()
        finally:
            observer.close()
            conn.close()


if __name__ == "__main__":
    main()
