#!/usr/bin/env python3
"""Non-transactional MV replacement fails before mutating the published generation."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from matview_sync import Dialect, MaterializedViewManager


class CatalogCursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=None):
        self.connection.executed_sql.append(sql)
        return self

    def fetchall(self):
        return [
            (
                "reader",
                "CREATE VIEW reader AS SELECT 1",
                "strategy",
                "old-signature",
                "reader_backing",
                "[]",
                "[]",
                "[]",
                "[]",
                None,
                None,
                0,
            )
        ]


class CatalogConnection:
    def __init__(self):
        self.cursor_calls = 0
        self.executed_sql = []
        self.rollback_calls = 0
        self.in_transaction = False

    def cursor(self):
        self.cursor_calls += 1
        return CatalogCursor(self)

    def rollback(self):
        self.rollback_calls += 1
        self.in_transaction = False


class NonTransactionalReplacementTests(unittest.TestCase):
    def _assert_refused(self, dialect, incremental_backfill):
        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        conn = CatalogConnection()
        manager.conn = conn
        manager.dialect = dialect
        manager.catalog_table = "matview_catalog"
        manager._compute_signature = lambda plan, schemas: "new-signature"
        drop_calls = []
        manager.drop_view = lambda *args, **kwargs: drop_calls.append((args, kwargs))

        plan = SimpleNamespace(view_name="reader")
        with self.assertRaisesRegex(
            RuntimeError,
            "Replacing an existing materialized view is unsupported",
        ):
            manager._register_single_view_spec(
                v_spec=None,
                schemas={},
                populate=True,
                incremental_backfill=incremental_backfill,
                batch_size=10,
                on_progress=None,
                force_rebuild=True,
                effective_ignore=False,
                compiled_plan=plan,
            )

        # The actual get_catalog() path executed one SELECT. The refusal then
        # happened before a replacement cursor, DROP, DDL, or rollback.
        self.assertEqual(conn.cursor_calls, 1)
        self.assertEqual(len(conn.executed_sql), 1)
        self.assertTrue(conn.executed_sql[0].lstrip().upper().startswith("SELECT"))
        self.assertEqual(drop_calls, [])
        self.assertEqual(conn.rollback_calls, 0)
        self.assertFalse(conn.in_transaction)

    def test_mysql_replacement_is_refused_after_catalog_read_only(self):
        self._assert_refused(Dialect.MYSQL, incremental_backfill=False)

    def test_sqlite_incremental_replacement_is_refused_after_catalog_read_only(self):
        self._assert_refused(Dialect.SQLITE, incremental_backfill=True)

    def test_postgres_incremental_replacement_is_refused_after_catalog_read_only(self):
        self._assert_refused(Dialect.POSTGRES, incremental_backfill=True)


if __name__ == "__main__":
    unittest.main()
