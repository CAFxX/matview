#!/usr/bin/env python3
"""Verify a failed single-shot replacement leaves the prior MV usable.

SQLite's transactional DDL requires an explicit BEGIN IMMEDIATE before dropping
the old generation. PostgreSQL's DDL is transactional under the connection's
normal transaction. MySQL/MariaDB and chunk-committing incremental replacement
are deliberately outside this regression's claimed scope.
"""
from __future__ import annotations

import sqlite3

from matview_sync import Dialect, MaterializedViewManager


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
        conn.create_function("maybe_fail", 1, lambda value: value)
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)

        view_sql = (
            "CREATE VIEW replacement_reader AS "
            "SELECT id, maybe_fail(value) AS value FROM replacement_rows;"
        )
        plan = manager.register_view(view_sql)
        original_catalog = manager.get_catalog()[plan.view_name]
        original_rows = conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert original_rows == [(1, 10), (2, 20), (3, 30)]

        def fail_population(_value: int) -> int:
            raise RuntimeError("injected replacement population failure")

        conn.create_function("maybe_fail", 1, fail_population)
        try:
            manager.register_view(view_sql, force_rebuild=True)
        except (sqlite3.Error, RuntimeError):
            pass
        else:
            raise AssertionError("replacement unexpectedly succeeded")

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

        conn.create_function("maybe_fail", 1, lambda value: value)
        conn.execute("INSERT INTO replacement_rows VALUES (4, 40);")
        after_write = conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall()
        assert after_write == [(1, 10), (2, 20), (3, 30), (4, 40)], after_write
        print("PASS: failed replacement rolled back to the prior live materialized view")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
