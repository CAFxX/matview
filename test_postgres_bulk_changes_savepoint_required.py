#!/usr/bin/env python3
"""Native PostgreSQL regression for SET LOCAL no-op outside a transaction."""
from __future__ import annotations

import os
import uuid

import psycopg2

from matview_sync import Dialect, MaterializedViewManager


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError(
            "POSTGRES_DSN is required; this test must run against native PostgreSQL"
        )

    conn = psycopg2.connect(dsn)
    schema = "mv_bulk_savepoint_" + uuid.uuid4().hex[:14]
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(
                "CREATE TABLE source_rows (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)"
            )
            cur.execute("INSERT INTO source_rows VALUES (1, 10), (2, 20)")
        conn.commit()

        manager = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        view_sql = (
            "CREATE VIEW savepoint_guard_reader AS "
            "SELECT id, value FROM source_rows;"
        )
        plan = manager.register_view(view_sql)
        original_catalog = manager.get_catalog()[plan.view_name]
        conn.commit()

        # Do not commit an already-active application transaction. The
        # trigger-maintained reader change must remain rollback-able by caller.
        with conn.cursor() as cur:
            cur.execute("UPDATE source_rows SET value = 11 WHERE id = 1")
        try:
            with manager.bulk_changes():
                raise AssertionError("bulk_changes joined a caller-owned PostgreSQL transaction")
        except RuntimeError as exc:
            assert "active caller transaction" in str(exc), str(exc)
        assert conn.get_transaction_status() != 0
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT id, value FROM savepoint_guard_reader ORDER BY id")
            assert cur.fetchall() == [(1, 10), (2, 20)]

        # In autocommit mode, SAVEPOINT fails and SET LOCAL outside a
        # transaction emits a warning/no-op rather than raising. The manager
        # must reject before entering the user block or dropping functions.
        conn.autocommit = True
        try:
            try:
                with manager.bulk_changes():
                    raise AssertionError(
                        "bulk_changes entered user block without a transactional savepoint"
                    )
            except RuntimeError as exc:
                assert "requires a transactional savepoint" in str(exc), str(exc)
        finally:
            conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, value FROM savepoint_guard_reader ORDER BY id"
            )
            assert cur.fetchall() == [(1, 10), (2, 20)]

            # The original maintenance trigger remains installed and active.
            cur.execute("INSERT INTO source_rows VALUES (3, 30)")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, value FROM savepoint_guard_reader ORDER BY id"
            )
            assert cur.fetchall() == [(1, 10), (2, 20), (3, 30)]

        assert manager.get_catalog()[plan.view_name]["signature_hash"] == (
            original_catalog["signature_hash"]
        )
        conn.commit()
        print(
            "PASS: PostgreSQL SET LOCAL no-op without a savepoint fails closed "
            "and leaves native maintenance active"
        )
    finally:
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
