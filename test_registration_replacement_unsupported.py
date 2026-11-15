#!/usr/bin/env python3
"""Non-transactional MV replacement must fail before touching the old generation."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from matview_sync import Dialect, MaterializedViewManager


class NoSQLConnection:
    def __init__(self):
        self.cursor_calls = 0

    def cursor(self):
        self.cursor_calls += 1
        raise AssertionError("replacement refusal must happen before cursor/SQL access")


class NonTransactionalReplacementTests(unittest.TestCase):
    def _assert_refused(self, dialect, incremental_backfill):
        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        conn = NoSQLConnection()
        manager.conn = conn
        manager.dialect = dialect
        manager._compute_signature = lambda plan, schemas: "new-signature"
        manager.get_catalog = lambda: {
            "reader": {"signature_hash": "old-signature"}
        }
        drop_calls = []
        manager.drop_view = lambda *args, **kwargs: drop_calls.append((args, kwargs))

        plan = SimpleNamespace(view_name="reader")
        with self.assertRaisesRegex(RuntimeError, "Replacing an existing materialized view is unsupported"):
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
        self.assertEqual(drop_calls, [])
        self.assertEqual(conn.cursor_calls, 0)

    def test_mysql_replacement_is_refused_before_any_sql(self):
        self._assert_refused(Dialect.MYSQL, incremental_backfill=False)

    def test_sqlite_incremental_replacement_is_refused_before_any_sql(self):
        self._assert_refused(Dialect.SQLITE, incremental_backfill=True)

    def test_postgres_incremental_replacement_is_refused_before_any_sql(self):
        self._assert_refused(Dialect.POSTGRES, incremental_backfill=True)


if __name__ == "__main__":
    unittest.main()