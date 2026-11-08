#!/usr/bin/env python3
"""Independent-session tests for incremental materialized-view backfill."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest

from matview_sync import MaterializedViewManager
from test_matview_sync import _register_sqlite_compat_udfs


class IndependentSessionBackfillTests(unittest.TestCase):
    def test_committed_chunk_allows_concurrent_second_session_dml(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="matview-backfill-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE ledger (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO ledger VALUES (?, ?, ?);",
            [(i, "B1" if i % 2 else "B2", i * 10) for i in range(1, 11)],
        )
        conn.commit()

        started = threading.Event()
        finished = threading.Event()
        worker_errors: list[BaseException] = []
        worker_thread_ids: list[int] = []
        progress_calls: list[tuple[int, int]] = []

        def write_from_independent_session() -> None:
            try:
                second = sqlite3.connect(db_path, timeout=10)
                try:
                    second.execute("PRAGMA busy_timeout=10000;")
                    _register_sqlite_compat_udfs(second)
                    worker_thread_ids.append(threading.get_ident())
                    started.set()

                    # The first backfill chunk has committed and advanced its
                    # watermark. These writes exercise both sides of that cursor
                    # using a genuinely independent SQLite connection.
                    second.execute("INSERT INTO ledger VALUES (11, 'B1', 110);")
                    second.execute("DELETE FROM ledger WHERE id = 2;")
                    second.execute("DELETE FROM ledger WHERE id = 8;")
                    second.execute("UPDATE ledger SET amt = 999 WHERE id = 1;")
                    second.execute("UPDATE ledger SET amt = 888 WHERE id = 9;")
                    second.commit()

                    # A rolled-back write must roll back its trigger effects too.
                    second.execute("BEGIN;")
                    second.execute("INSERT INTO ledger VALUES (12, 'B2', 120);")
                    second.rollback()
                finally:
                    second.close()
            except BaseException as exc:
                worker_errors.append(exc)
            finally:
                finished.set()

        def on_progress(batch_idx: int, rows_done: int) -> None:
            progress_calls.append((batch_idx, rows_done))
            if batch_idx != 1:
                return
            worker = threading.Thread(target=write_from_independent_session, daemon=True)
            worker.start()
            if not started.wait(timeout=10):
                raise AssertionError("independent writer did not start during backfill")
            if not finished.wait(timeout=10):
                raise AssertionError("independent writer did not finish before backfill resumed")
            worker.join(timeout=1)
            if worker.is_alive():
                raise AssertionError("independent writer thread did not terminate")
            if worker_errors:
                raise worker_errors[0]

        try:
            manager = MaterializedViewManager(conn)
            plan = manager.register_view(
                "CREATE VIEW ledger_totals AS "
                "SELECT bucket, COUNT(*) AS c, SUM(amt) AS s "
                "FROM ledger GROUP BY bucket;",
                incremental_backfill=True,
                batch_size=3,
                on_progress=on_progress,
            )

            self.assertTrue(progress_calls, "backfill did not report chunk progress")
            self.assertTrue(worker_thread_ids)
            self.assertNotEqual(worker_thread_ids[0], threading.get_ident())

            actual = conn.execute(
                "SELECT bucket, c, s FROM __matview_ledger_totals ORDER BY bucket;"
            ).fetchall()
            expected = conn.execute(
                "SELECT bucket, COUNT(*) AS c, SUM(amt) AS s "
                "FROM ledger GROUP BY bucket ORDER BY bucket;"
            ).fetchall()
            self.assertEqual(actual, expected)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM ledger WHERE id = 12;").fetchone()[0],
                0,
                "rolled-back insert unexpectedly persisted",
            )
            catalog = manager.get_catalog()[plan.view_name]
            self.assertEqual(catalog["backfill_active"], 0)
        finally:
            conn.close()
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)