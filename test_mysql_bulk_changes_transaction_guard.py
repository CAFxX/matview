#!/usr/bin/env python3
"""Native MySQL/MariaDB regression for bulk_changes caller-transaction safety."""
from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse

import pymysql

from matview_sync import Dialect, MaterializedViewManager


def connect(dsn: str, *, autocommit: bool = False):
    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=autocommit,
    )


def main() -> None:
    dsn = os.environ.get("MYSQL_DSN")
    if not dsn:
        raise RuntimeError("MYSQL_DSN is required; this test must run on native MySQL/MariaDB")

    suffix = uuid.uuid4().hex[:12]
    table = f"mv_bulk_tx_guard_{suffix}"
    conn = connect(dsn)
    observer = connect(dsn, autocommit=True)
    autocommit_conn = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {table} (id BIGINT PRIMARY KEY) ENGINE=InnoDB"
            )
        conn.commit()

        # Construct the manager before beginning the caller-owned transaction;
        # constructor/catalog setup is not part of the operation being tested.
        manager = MaterializedViewManager(conn, dialect=Dialect.MYSQL)

        # An autocommit observer gets a fresh read for each statement. If entry
        # into bulk_changes accidentally commits caller work (including via
        # implicit DDL), this observer will see the uncommitted row.
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {table} VALUES (1)")
        assert conn.in_transaction, "fixture failed to establish caller transaction"
        assert manager._mysql_transaction_state() is True
        assert observer.cursor().execute(f"SELECT COUNT(*) FROM {table}") == 1
        with observer.cursor() as cur:
            assert cur.fetchone()[0] == 0

        try:
            with manager.bulk_changes():
                raise AssertionError("bulk_changes unexpectedly entered caller transaction")
        except RuntimeError as exc:
            assert "active caller transaction" in str(exc), str(exc)
        else:
            raise AssertionError("active caller transaction was not rejected")

        assert conn.in_transaction, "rejected entry ended the caller transaction"
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 1, "caller transaction lost its own write"
        with observer.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 0, "bulk_changes committed caller-owned work"

        # The caller still owns the transaction and can decide its outcome.
        conn.rollback()
        with observer.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 0

        # Autocommit mode cannot provide the rollback contract either and must
        # be rejected before bulk_changes issues its first query/DDL.
        autocommit_conn = connect(dsn, autocommit=True)
        autocommit_manager = MaterializedViewManager(
            autocommit_conn, dialect=Dialect.MYSQL
        )
        try:
            with autocommit_manager.bulk_changes():
                raise AssertionError("bulk_changes unexpectedly accepted autocommit")
        except RuntimeError as exc:
            assert "autocommit to be disabled" in str(exc), str(exc)
        else:
            raise AssertionError("autocommit connection was not rejected")

        print("PASS: native MySQL/MariaDB bulk_changes preserves caller transactions and rejects autocommit")
    finally:
        if autocommit_conn is not None:
            autocommit_conn.close()
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
        finally:
            observer.close()
            conn.close()


if __name__ == "__main__":
    main()
