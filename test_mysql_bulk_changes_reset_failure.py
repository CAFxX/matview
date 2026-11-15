#!/usr/bin/env python3
"""Native MySQL/MariaDB regression for poisoned session-bypass reset failure.

The wrapper deliberately drops the SET @__matview_skip_triggers = 0 response
before it reaches the server. Its close hook reads the still-set session
variable, records it, and closes the underlying physical DB-API connection.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

from matview_sync import Dialect, MaterializedViewManager


class ResetFailingCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def execute(self, statement: str, *args: Any, **kwargs: Any) -> Any:
        if str(statement).strip() == "SET @__matview_skip_triggers = 0;":
            raise RuntimeError("injected lost reset response")
        return self._cursor.execute(statement, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class ResetFailingConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.closed = False
        self.flag_before_close: Any = None

    def cursor(self, *args: Any, **kwargs: Any) -> ResetFailingCursor:
        return ResetFailingCursor(self._connection.cursor(*args, **kwargs))

    def close(self) -> None:
        # Observe the server-side session variable before disconnecting the
        # physical session. Transaction rollback cannot clear user variables.
        cur = self._connection.cursor()
        try:
            cur.execute("SELECT @__matview_skip_triggers")
            row = cur.fetchone()
            self.flag_before_close = row[0] if row else None
        finally:
            cur.close()
        self._connection.close()
        self.closed = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def main() -> None:
    dsn = os.environ.get("MYSQL_DSN")
    if not dsn:
        raise RuntimeError(
            "MYSQL_DSN is required; this test must run against native MySQL/MariaDB"
        )

    conn, dialect = MaterializedViewManager._connect_database_url(dsn)
    if dialect != Dialect.MYSQL:
        raise RuntimeError(f"expected MySQL-family dialect, got {dialect.value}")

    suffix = uuid.uuid4().hex[:12]
    table = f"mv_bulk_reset_rows_{suffix}"
    view = f"mv_bulk_reset_view_{suffix}"
    cleanup_conn = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)"
            )
            cur.execute(f"INSERT INTO {table} VALUES (1, 10), (2, 20)")
        conn.commit()

        manager = MaterializedViewManager(conn, dialect=Dialect.MYSQL)
        manager.register_view(
            f"CREATE VIEW {view} AS SELECT id, value FROM {table};"
        )

        wrapped = ResetFailingConnection(conn)
        manager.conn = wrapped
        try:
            with manager.bulk_changes():
                pass
        except RuntimeError as exc:
            assert "discard this connection" in str(exc), str(exc)
        else:
            raise AssertionError("reset failure did not invalidate the MySQL session")

        assert wrapped.closed, "physical MySQL/MariaDB connection was not closed"
        assert wrapped.flag_before_close == 1, (
            "the injected lost reset response did not leave the session bypass flag set "
            f"before close: {wrapped.flag_before_close!r}"
        )

        # A fresh connection must still see a consistent registered reader; the
        # failed context must not leave the persisted materialized view corrupted.
        cleanup_conn, _ = MaterializedViewManager._connect_database_url(dsn)
        cleanup_manager = MaterializedViewManager(cleanup_conn, dialect=Dialect.MYSQL)
        rows = cleanup_conn.cursor()
        try:
            rows.execute(f"SELECT id, value FROM {view} ORDER BY id")
            assert rows.fetchall() == [(1, 10), (2, 20)]
        finally:
            rows.close()
        cleanup_manager.drop_view(view)
        with cleanup_conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        cleanup_conn.commit()
        print("PASS: failed MySQL/MariaDB session-flag reset closes the poisoned physical connection")
    finally:
        if cleanup_conn is not None:
            try:
                cleanup_conn.rollback()
                cleanup_conn.close()
            except Exception:
                pass
        else:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
