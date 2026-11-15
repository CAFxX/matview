#!/usr/bin/env python3
"""Verify a failed single-shot replacement leaves the prior MV usable.

SQLite's transactional DDL requires an explicit BEGIN IMMEDIATE before dropping
the old generation. PostgreSQL's DDL is transactional under the connection's
normal transaction. MySQL/MariaDB and chunk-committing incremental replacement
are deliberately outside this regression's claimed scope.
"""
from __future__ import annotations

import sqlite3

from matview_sync import Dialect, MaterializedViewManager, MaterializedViewPlan


def main() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE replacement_rows (id INTEGER PRIMARY KEY, value INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO replacement_rows VALUES (?, ?);",
            [(1, 10), (2, 20), (3, 30)],
        )
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)

        view_sql = (
            "CREATE VIEW replacement_reader AS "
            "SELECT id, value FROM replacement_rows;"
        )
        plan = manager.register_view(view_sql)
        original_catalog = manager.get_catalog()[plan.view_name]
        original_rows = conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert original_rows == [(1, 10), (2, 20), (3, 30)]

        # Fail exactly when the replacement reaches its population phase. At
        # this point the old generation has already been dropped and the new
        # structure, triggers, and catalog row have been installed.
        original_populate = MaterializedViewPlan.get_populate_statements

        def fail_population(self, *args, **kwargs):
            raise RuntimeError("injected replacement population failure")

        MaterializedViewPlan.get_populate_statements = fail_population
        try:
            try:
                manager.register_view(view_sql, force_rebuild=True)
            except RuntimeError as exc:
                assert "injected replacement population failure" in str(exc)
            else:
                raise AssertionError("replacement unexpectedly succeeded")
        finally:
            MaterializedViewPlan.get_populate_statements = original_populate

        restored_rows = conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert restored_rows == original_rows, (original_rows, restored_rows)

        restored_catalog = manager.get_catalog()[plan.view_name]
        assert restored_catalog["signature_hash"] == original_catalog["signature_hash"]
        assert conn.execute(
            "SELECT COUNT(*) FROM replacement_rows;"
        ).fetchone()[0] == 3

        for trigger_name in original_catalog["trigger_names"]:
            found = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?;",
                (trigger_name,),
            ).fetchone()[0]
            assert found == 1, f"old maintenance trigger {trigger_name} was not restored"

        conn.execute("INSERT INTO replacement_rows VALUES (4, 40);")
        after_write = conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert after_write == [(1, 10), (2, 20), (3, 30), (4, 40)], after_write

        # Exercise the earlier failure boundary too: fail only after drop_view
        # has removed the reader/backing objects but before it deletes the
        # catalog row. The whole DDL sequence must roll back on the same
        # connection, and that connection must remain usable afterwards.
        raw_conn = conn
        executed = []
        injected = {"fail": True}

        class CursorProxy:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                normalized = " ".join(sql.upper().split())
                if (
                    injected["fail"]
                    and normalized.startswith(
                        f"DELETE FROM {manager.catalog_table.upper()}"
                    )
                ):
                    assert any(
                        item.upper().startswith(
                            f"DROP VIEW IF EXISTS {plan.view_name.upper()}"
                        )
                        for item in executed
                    ), executed
                    injected["fail"] = False
                    raise sqlite3.OperationalError(
                        "injected failure after reader/backing drop"
                    )
                if params is None:
                    return self._cursor.execute(statement)
                return self._cursor.execute(statement, params)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        class ConnectionProxy:
            def cursor(self, *args, **kwargs):
                return CursorProxy(raw_conn.cursor(*args, **kwargs))

            def __getattr__(self, name):
                return getattr(raw_conn, name)

        manager.conn = ConnectionProxy()
        try:
            manager.register_view(view_sql, force_rebuild=True)
        except sqlite3.OperationalError as exc:
            assert "injected failure after reader/backing drop" in str(exc), exc
        else:
            raise AssertionError("drop-stage replacement unexpectedly succeeded")
        finally:
            manager.conn = raw_conn

        assert not injected["fail"], "fault injection did not reach catalog cleanup"
        restored_rows = raw_conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert restored_rows == [(1, 10), (2, 20), (3, 30), (4, 40)], restored_rows
        restored_catalog = manager.get_catalog()[plan.view_name]
        assert restored_catalog["signature_hash"] == original_catalog["signature_hash"]

        # A restored reader/catalog entry is insufficient if its physical
        # generation is missing. Check the backing and every auxiliary table.
        physical_tables = {
            row[0]
            for row in raw_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert plan.backing_table_name in physical_tables, (
            f"backing table {plan.backing_table_name} was not restored"
        )
        for auxiliary_name in plan.auxiliary_table_names:
            assert auxiliary_name in physical_tables, (
                f"auxiliary table {auxiliary_name} was not restored"
            )

        for trigger_name in original_catalog["trigger_names"]:
            found = raw_conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?;",
                (trigger_name,),
            ).fetchone()[0]
            assert found == 1, f"old maintenance trigger {trigger_name} was not restored"

        # Prove the failed DDL transaction did not poison the connection.
        raw_conn.execute("BEGIN;")
        raw_conn.execute("INSERT INTO replacement_rows VALUES (5, 50);")
        raw_conn.commit()
        after_recovery = raw_conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert after_recovery == [
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)
        ], after_recovery

        print("PASS: failed replacement rolled back to the prior live materialized view")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
