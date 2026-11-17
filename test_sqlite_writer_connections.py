#!/usr/bin/env python3
"""SQLite writer connections must enable recursive triggers.

SQLite fires DELETE triggers for rows that ``INSERT OR REPLACE`` /
``REPLACE`` removes implicitly only when ``PRAGMA recursive_triggers`` is ON
on the connection executing the statement, and the PRAGMA is per connection.
``MaterializedViewManager.configure_sqlite_connection`` enables and verifies
it for any writer connection (and the manager applies it to its own
connection, failing closed). These tests use file-backed databases with an
independent writer connection.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from typing import Any, Optional

from matview_sync import MaterializedViewManager

SCHEMA = """
    CREATE TABLE source_rows (
        id INTEGER PRIMARY KEY,
        unique_code TEXT NOT NULL UNIQUE,
        unique_tag TEXT NOT NULL UNIQUE,
        bucket TEXT NOT NULL
    );
    INSERT INTO source_rows VALUES (1, 'code-1', 'tag-1', 'old');
"""
VIEW_SQL = "CREATE VIEW source_rows_materialized AS SELECT id, unique_code, unique_tag, bucket FROM source_rows;"
COUNTS_SQL = "CREATE VIEW source_bucket_counts AS SELECT bucket, COUNT(*) AS n FROM source_rows GROUP BY bucket;"


class WriterInitializerTests(unittest.TestCase):
    def _replace_through_writer(self, replacement: tuple, initialized: bool) -> tuple:
        fd, path = tempfile.mkstemp(prefix="matview-writer-", suffix=".sqlite")
        os.close(fd)
        setup: Optional[sqlite3.Connection] = None
        writer: Optional[sqlite3.Connection] = None
        try:
            setup = sqlite3.connect(path)
            setup.executescript(SCHEMA)
            setup.commit()
            manager = MaterializedViewManager(setup)
            manager.register_view(VIEW_SQL)
            manager.register_view(COUNTS_SQL)

            writer = sqlite3.connect(path)
            if initialized:
                self.assertIs(MaterializedViewManager.configure_sqlite_connection(writer), writer)
                # Idempotent, for connection-pool initialization hooks.
                MaterializedViewManager.configure_sqlite_connection(writer)
            else:
                # Force the unsafe setting independently of the build default.
                writer.execute("PRAGMA recursive_triggers = OFF;")
                self.assertEqual(writer.execute("PRAGMA recursive_triggers;").fetchone()[0], 0)
            writer.execute("INSERT OR REPLACE INTO source_rows VALUES (?, ?, ?, ?);", replacement)
            writer.commit()
            return (
                sorted(setup.execute("SELECT * FROM source_rows_materialized;").fetchall()),
                sorted(setup.execute("SELECT * FROM source_rows;").fetchall()),
                sorted(setup.execute("SELECT bucket, n FROM source_bucket_counts;").fetchall()),
                sorted(setup.execute("SELECT bucket, COUNT(*) FROM source_rows GROUP BY bucket;").fetchall()),
            )
        finally:
            for c in (writer, setup):
                if c is not None:
                    c.close()
            os.unlink(path)

    def test_initialized_writer_maintains_replace_on_every_conflict_path(self) -> None:
        # Every uniqueness path: rowid primary key and both UNIQUE constraints.
        for target, replacement in (
            ("primary key", (1, "code-2", "tag-2", "new")),
            ("unique_code", (2, "code-1", "tag-2", "new")),
            ("unique_tag", (2, "code-2", "tag-1", "new")),
        ):
            with self.subTest(conflict=target):
                mv, live, mv_counts, live_counts = self._replace_through_writer(replacement, initialized=True)
                self.assertEqual(mv, live)
                self.assertEqual(live, [replacement])
                self.assertEqual(mv_counts, live_counts)

    def test_uninitialized_writer_diverges(self) -> None:
        # Documents why the initializer is required: with recursive triggers
        # OFF SQLite skips the implicit-conflict DELETE trigger, so the
        # aggregate keeps counting the replaced row on every conflict path.
        # (The keyed projection happens to be overwritten on a PRIMARY KEY
        # conflict, but keeps the stale row on UNIQUE conflicts.)
        for target, replacement in (
            ("primary key", (1, "code-2", "tag-2", "new")),
            ("unique_code", (2, "code-1", "tag-2", "new")),
            ("unique_tag", (2, "code-2", "tag-1", "new")),
        ):
            with self.subTest(conflict=target):
                mv, live, mv_counts, live_counts = self._replace_through_writer(replacement, initialized=False)
                self.assertEqual(live_counts, [("new", 1)])
                self.assertEqual(mv_counts, [("new", 1), ("old", 1)])
                if target != "primary key":
                    self.assertEqual(mv, sorted([(1, "code-1", "tag-1", "old"), replacement]))

    def test_initializer_fails_closed_and_closes_its_cursor(self) -> None:
        class Cursor:
            def __init__(self, readback: Any, fail: bool = False) -> None:
                self.readback, self.fail, self.closed = readback, fail, False

            def execute(self, sql: str) -> None:
                return None  # DB-API does not require returning the cursor

            def fetchone(self) -> Any:
                if self.fail:
                    raise sqlite3.OperationalError("injected PRAGMA readback failure")
                return self.readback

            def close(self) -> None:
                self.closed = True

        class Connection:
            def __init__(self, cursor: Cursor) -> None:
                self._cursor = cursor

            def cursor(self) -> Cursor:
                return self._cursor

        ok = Cursor((1,))
        conn = Connection(ok)
        self.assertIs(MaterializedViewManager.configure_sqlite_connection(conn), conn)
        self.assertTrue(ok.closed)
        for label, cursor, error in (
            ("readback 0", Cursor((0,)), RuntimeError),
            ("no row", Cursor(None), RuntimeError),
            ("readback error", Cursor((1,), fail=True), sqlite3.OperationalError),
        ):
            with self.subTest(case=label):
                with self.assertRaises(error):
                    MaterializedViewManager.configure_sqlite_connection(Connection(cursor))
                self.assertTrue(cursor.closed)
        with self.assertRaises(TypeError):
            MaterializedViewManager.configure_sqlite_connection(object())

    def test_manager_configures_its_own_connection_and_fails_closed(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA recursive_triggers = OFF;")
            MaterializedViewManager(conn)
            self.assertEqual(conn.execute("PRAGMA recursive_triggers;").fetchone()[0], 1)
        finally:
            conn.close()

        class RefusesPragma:
            """A SQLite-compatible connection on which the PRAGMA does not take effect."""

            in_transaction = False

            def __init__(self) -> None:
                self._raw = sqlite3.connect(":memory:")

            def cursor(self) -> Any:
                raw = self._raw.cursor()

                class Cursor:
                    def execute(self, sql: str, *args: Any) -> Any:
                        if sql.strip().upper().startswith("PRAGMA RECURSIVE_TRIGGERS ="):
                            return None
                        return raw.execute(sql, *args)

                    def __getattr__(self, name: str) -> Any:
                        return getattr(raw, name)

                return Cursor()

            def commit(self) -> None:
                self._raw.commit()

            def rollback(self) -> None:
                self._raw.rollback()

        fake = RefusesPragma()
        try:
            with self.assertRaisesRegex(RuntimeError, "recursive_triggers"):
                MaterializedViewManager(fake, dialect="sqlite")
        finally:
            fake._raw.close()


if __name__ == "__main__":
    unittest.main()
