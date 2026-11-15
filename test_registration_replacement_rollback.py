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
        # A failure partway through dropping the old generation must roll
        # back the already-dropped trigger and leave the same connection usable.
        original_drop = manager.drop_view

        def fail_mid_drop(view_name, _commit_transaction=True):
            conn.execute(f"DROP TRIGGER IF EXISTS {original_catalog['trigger_names'][0]};")
            raise RuntimeError("injected mid-drop failure")

        manager.drop_view = fail_mid_drop
        try:
            try:
                manager.register_view(view_sql, force_rebuild=True)
            except RuntimeError as exc:
                assert "injected mid-drop failure" in str(exc)
            else:
                raise AssertionError("mid-drop failure was not injected")
        finally:
            manager.drop_view = original_drop

        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name=?;",
            (original_catalog["trigger_names"][0],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall() == original_rows
        conn.execute("INSERT INTO replacement_rows VALUES (5, 50);")
        conn.commit()
        assert conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall() == [
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)
        ]

        # Caller-owned transactions are rejected before schema/catalog reads or
        # DDL; registration must never commit unrelated caller work.
        conn.execute("BEGIN;")
        conn.execute("INSERT INTO replacement_rows VALUES (6, 60);")
        try:
            manager.register_view(view_sql, force_rebuild=True)
        except RuntimeError as exc:
            assert "idle SQLite connection" in str(exc)
        else:
            raise AssertionError("caller-owned SQLite transaction was accepted")
        conn.rollback()
        assert conn.execute(
            "SELECT COUNT(*) FROM replacement_rows WHERE id=6;"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT id, value FROM replacement_reader ORDER BY id;"
        ).fetchall() == [
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)
        ]

        # Unsupported replacement modes fail before drop_view is called.
        original_dialect = manager.dialect
        original_drop = manager.drop_view
        def unexpected_drop(*args, **kwargs):
            raise AssertionError("unsupported replacement attempted to drop old generation")
        manager.drop_view = unexpected_drop
        try:
            for dialect, incremental in (
                (Dialect.MYSQL, False),
                (Dialect.SQLITE, True),
            ):
                manager.dialect = dialect
                try:
                    manager._register_single_view_spec(
                        plan.view_spec,
                        schemas=manager.fetch_schema(),
                        populate=True,
                        incremental_backfill=incremental,
                        batch_size=500,
                        on_progress=None,
                        force_rebuild=True,
                        effective_ignore=False,
                        effective_strict=False,
                        compiled_plan=plan,
                    )
                except RuntimeError as exc:
                    assert "Replacing an existing materialized view is unsupported" in str(exc)
                else:
                    raise AssertionError("unsupported replacement mode was accepted")
        finally:
            manager.dialect = original_dialect
            manager.drop_view = original_drop

        print("PASS: failed replacement and mid-drop faults preserve the prior materialized view")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
