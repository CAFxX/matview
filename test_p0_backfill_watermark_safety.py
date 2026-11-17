#!/usr/bin/env python3
"""
Dedicated unit and integration tests for P0 Backfill Watermark Safety & Trigger Coverage.

Verifies:
1. Finding #2: Catalog/watermark update failures fail closed (exceptions are never
   silently swallowed, active savepoints/transactions are rolled back, and watermark
   reset prevents stale trigger gating).
2. Finding #3: Views lacking watermark trigger gating (join views, union_all_agg,
   Tier 2, Tier 3 window functions, group_key_batch, offset_batch, or un-watermarked plans)
   explicitly reject `backfill_incrementally(..., snapshot_isolation=False)` with
   `UnsupportedMaterializedViewError` / `ValueError`, while safely allowing
   `snapshot_isolation=True` and non-chunked backfill. Single-table projection views
   with an integer PK safely support watermark-gated triggers and pk_range_fold.
3. Finding #1: Backfill watermark concurrency safety under `snapshot_isolation=False`
   with chunk-by-chunk serialization and trigger gating across primary key boundaries.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest

from matview_sync import (
    BackfillInterruptedError,
    Dialect,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
)
from test_matview_sync import _register_sqlite_compat_udfs


class BackfillWatermarkSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fd, self.db_path = tempfile.mkstemp(prefix="p0-backfill-safety-", suffix=".sqlite")
        os.close(self.fd)
        self.conn = sqlite3.connect(self.db_path, timeout=10)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(self.conn)

    def tearDown(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    # -------------------------------------------------------------------------
    # Finding #2: Catalog/Watermark Update Fail Closed & Exception Propagation
    # -------------------------------------------------------------------------

    def test_watermark_update_failure_propagates_and_rolls_back_under_snapshot_isolation(self) -> None:
        """Catalog update failure during snapshot backfill must propagate and roll back savepoint."""
        self.conn.execute(
            "CREATE TABLE t_src (id INTEGER PRIMARY KEY, category TEXT, val INTEGER);"
        )
        self.conn.executemany(
            "INSERT INTO t_src VALUES (?, ?, ?);",
            [(i, f"c{i % 2}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            "CREATE VIEW v_totals AS SELECT category, COUNT(*) AS c, SUM(val) AS s FROM t_src GROUP BY category;",
            populate=False,
        )

        # Inject failure into _update_backfill_watermark to simulate database/catalog corruption or lock
        original_update = manager._update_backfill_watermark
        call_count = 0

        def failing_update(cur, view_name, cursor_pk, max_pk, active):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise sqlite3.OperationalError("simulated catalog write failure")
            return original_update(cur, view_name, cursor_pk, max_pk, active)

        manager._update_backfill_watermark = failing_update

        with self.assertRaisesRegex(sqlite3.OperationalError, "simulated catalog write failure"):
            manager.backfill_incrementally(plan, batch_size=3, snapshot_isolation=True)

        # Fail closed: connection transaction/savepoint must be cleanly closed
        self.assertFalse(self.conn.in_transaction, "savepoint transaction was left unclosed")

        # Stale active watermark must not remain in catalog
        catalog = manager.get_catalog()[plan.view_name]
        self.assertEqual(catalog["backfill_active"], 0, "backfill_active was left as 1")
        self.assertIsNone(catalog["backfill_cursor_pk"])
        self.assertIsNone(catalog["backfill_max_pk"])

    def test_watermark_update_failure_propagates_and_keeps_watermark_under_non_snapshot(self) -> None:
        """Catalog update failure under non-snapshot backfill must propagate and fail closed.

        The active watermark was already committed, so triggers have been
        excluding the unfolded range. Resetting it to inactive would let a
        re-run double-count committed chunks and let triggers apply deltas for
        never-folded rows. It stays active, a re-run is refused, and an
        explicit rebuild restores exact equivalence with the original view.
        """
        self.conn.execute(
            "CREATE TABLE t_src2 (id INTEGER PRIMARY KEY, category TEXT, val INTEGER);"
        )
        self.conn.executemany(
            "INSERT INTO t_src2 VALUES (?, ?, ?);",
            [(i, f"c{i % 2}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            "CREATE VIEW v_totals2 AS SELECT category, COUNT(*) AS c, SUM(val) AS s FROM t_src2 GROUP BY category;",
            populate=False,
        )

        original_update = manager._update_backfill_watermark
        call_count = 0

        def failing_update(cur, view_name, cursor_pk, max_pk, active):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise sqlite3.OperationalError("simulated mid-batch watermark disk error")
            return original_update(cur, view_name, cursor_pk, max_pk, active)

        manager._update_backfill_watermark = failing_update

        with self.assertRaisesRegex(sqlite3.OperationalError, "simulated mid-batch watermark disk error"):
            manager.backfill_incrementally(plan, batch_size=3, snapshot_isolation=False)

        self.assertFalse(self.conn.in_transaction, "transaction was left unclosed after error")

        # Fail closed: batch (3, 6] was rolled back; the committed watermark
        # after batch (0, 3] stays active.
        catalog = manager.get_catalog()[plan.view_name]
        self.assertEqual(
            (catalog["backfill_active"], catalog["backfill_cursor_pk"], catalog["backfill_max_pk"]),
            (1, 3, 10),
        )
        manager._update_backfill_watermark = original_update
        self.conn.execute("DELETE FROM t_src2 WHERE id IN (1, 7);")
        self.conn.execute("INSERT INTO t_src2 VALUES (20, 'c0', 5);")
        self.conn.commit()
        # A re-run is refused without touching the watermark.
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan, batch_size=3, snapshot_isolation=False)
        self.assertEqual(manager.get_catalog()[plan.view_name]["backfill_active"], 1)
        # Explicit rebuild repairs the view.
        plan = manager.register_view(
            "CREATE VIEW v_totals2 AS SELECT category, COUNT(*) AS c, SUM(val) AS s FROM t_src2 GROUP BY category;",
            force_rebuild=True,
        )
        self.assertEqual(
            self.conn.execute(f"SELECT category, c, s FROM {plan.view_name} ORDER BY category;").fetchall(),
            self.conn.execute(
                "SELECT category, COUNT(*), SUM(val) FROM t_src2 GROUP BY category ORDER BY category;"
            ).fetchall(),
        )

    # -------------------------------------------------------------------------
    # Finding #3: Trigger Coverage & Explicit Safety Checks for Non-Snapshot Backfill
    # -------------------------------------------------------------------------

    def test_unsupported_modes_reject_non_snapshot_incremental_backfill(self) -> None:
        """Plans lacking watermark-gated triggers must explicitly reject snapshot_isolation=False."""
        self.conn.execute(
            "CREATE TABLE j_left (id INTEGER PRIMARY KEY, ref_val TEXT);"
        )
        self.conn.execute(
            "CREATE TABLE j_right (id INTEGER PRIMARY KEY, left_id INTEGER, amt INTEGER);"
        )
        self.conn.execute(
            "CREATE TABLE comp_pk (k1 INT, k2 INT, val TEXT, PRIMARY KEY (k1, k2));"
        )
        self.conn.execute(
            "CREATE TABLE win_tbl (id INTEGER PRIMARY KEY, val INT);"
        )
        self.conn.executemany(
            "INSERT INTO win_tbl VALUES (?, ?);",
            [(i, i * 10) for i in range(1, 6)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)

        # 1. Join view (cannot use watermark triggers across multi-table joins)
        plan_join = manager.register_view(
            "CREATE VIEW v_join AS SELECT l.ref_val, SUM(r.amt) AS total "
            "FROM j_left l JOIN j_right r ON l.id = r.left_id GROUP BY l.ref_val;",
            populate=False,
        )
        self.assertFalse(plan_join.has_watermark_triggers)
        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan_join, batch_size=2, snapshot_isolation=False)

        # 2. Window function / Tier 3 view (uses offset_batch, no watermark triggers)
        plan_win = manager.register_view(
            "CREATE VIEW v_win AS SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM win_tbl;",
            populate=False,
        )
        self.assertFalse(plan_win.has_watermark_triggers)
        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan_win, batch_size=2, snapshot_isolation=False)

        # 3. Composite primary key projection (uses offset_batch, no single integer PK)
        plan_comp = manager.register_view(
            "CREATE VIEW v_comp AS SELECT k1, k2, val FROM comp_pk;",
            populate=False,
        )
        self.assertFalse(plan_comp.has_watermark_triggers)
        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan_comp, batch_size=2, snapshot_isolation=False)

        # 4. View compiled without watermark_catalog_table
        schemas = manager.fetch_schema()
        from matview_sync import generate_materialized_view, parse_view_definitions
        specs = parse_view_definitions("CREATE VIEW v_no_wm AS SELECT val, COUNT(*) AS c FROM win_tbl GROUP BY val;")
        plan_no_wm = generate_materialized_view(
            schemas, specs[0], dialect=Dialect.SQLITE, watermark_catalog_table=None
        )
        self.assertFalse(plan_no_wm.has_watermark_triggers)
        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan_no_wm, batch_size=2, snapshot_isolation=False)

        # Under snapshot_isolation=True, un-gated plans execute safely
        batches_snap = manager.backfill_incrementally(plan_win, batch_size=2, snapshot_isolation=True)
        self.assertGreaterEqual(batches_snap, 1)

        # Non-chunked backfill (batch_size <= 0) executes safely without snapshot isolation
        batches_single = manager.backfill_incrementally(plan_win, batch_size=0, snapshot_isolation=False)
        self.assertEqual(batches_single, 1)

    def test_single_table_projection_with_int_pk_supports_watermark_and_pk_range_fold(self) -> None:
        """Single-table projection with integer PK must emit watermark triggers and support pk_range_fold."""
        self.conn.execute("CREATE TABLE proj_items (id INTEGER PRIMARY KEY, name TEXT, score INT);")
        self.conn.executemany(
            "INSERT INTO proj_items VALUES (?, ?, ?);",
            [(i, f"item_{i}", i * 100) for i in range(1, 11)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            "CREATE VIEW v_proj AS SELECT id, name, score FROM proj_items WHERE score >= 200;",
            populate=False,
        )

        self.assertTrue(plan.has_watermark_triggers, "projection view did not emit watermark triggers")
        self.assertIsNotNone(plan.incremental_backfill_spec)
        self.assertEqual(plan.incremental_backfill_spec.mode, "pk_range_fold")
        self.assertEqual(plan.incremental_backfill_spec.pk_column, "id")

        # Must execute cleanly under snapshot_isolation=False
        batches = manager.backfill_incrementally(plan, batch_size=3, snapshot_isolation=False)
        self.assertGreaterEqual(batches, 2)

        actual = self.conn.execute(f"SELECT id, name, score FROM {plan.view_name} ORDER BY id;").fetchall()
        expected = self.conn.execute("SELECT id, name, score FROM proj_items WHERE score >= 200 ORDER BY id;").fetchall()
        self.assertEqual(actual, expected)

    # -------------------------------------------------------------------------
    # Finding #1: Backfill Watermark Concurrency & Chunk Safety
    # -------------------------------------------------------------------------

    def test_projection_watermark_gates_concurrent_independent_dml(self) -> None:
        """Concurrent writes during pk_range_fold projection backfill must gate correctly on watermark."""
        self.conn.execute("CREATE TABLE live_ledger (id INTEGER PRIMARY KEY, note TEXT, amt INT);")
        self.conn.executemany(
            "INSERT INTO live_ledger VALUES (?, ?, ?);",
            [(i, f"note_{i}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()

        started = threading.Event()
        finished = threading.Event()
        worker_errors: list[BaseException] = []

        def independent_worker() -> None:
            try:
                second = sqlite3.connect(self.db_path, timeout=10)
                try:
                    second.execute("PRAGMA busy_timeout=10000;")
                    _register_sqlite_compat_udfs(second)
                    started.set()

                    # Write below cursor (id=1): trigger must fire and update MV
                    second.execute("UPDATE live_ledger SET amt = 999 WHERE id = 1;")
                    # Write in upcoming range (id=8): trigger must suppress; chunk fold will incorporate it
                    second.execute("UPDATE live_ledger SET amt = 888 WHERE id = 8;")
                    # Delete below cursor (id=2): trigger must fire and delete from MV
                    second.execute("DELETE FROM live_ledger WHERE id = 2;")
                    # Insert beyond max_pk (id=12): trigger must fire
                    second.execute("INSERT INTO live_ledger VALUES (12, 'note_12', 120);")
                    second.commit()
                finally:
                    second.close()
            except BaseException as exc:
                worker_errors.append(exc)
            finally:
                finished.set()

        def on_progress(batch_idx: int, rows_done: int) -> None:
            if batch_idx != 1:
                return
            t = threading.Thread(target=independent_worker, daemon=True)
            t.start()
            if not started.wait(timeout=10):
                raise AssertionError("independent worker did not start")
            if not finished.wait(timeout=10):
                raise AssertionError("independent worker did not finish")
            t.join(timeout=1)
            if worker_errors:
                raise worker_errors[0]

        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            "CREATE VIEW v_live_ledger AS SELECT id, note, amt FROM live_ledger;",
            populate=False,
        )

        batches = manager.backfill_incrementally(
            plan,
            batch_size=3,
            on_progress=on_progress,
            snapshot_isolation=False,
        )
        self.assertGreaterEqual(batches, 2)

        actual = self.conn.execute("SELECT id, note, amt FROM v_live_ledger ORDER BY id;").fetchall()
        expected = self.conn.execute("SELECT id, note, amt FROM live_ledger ORDER BY id;").fetchall()
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 10)  # 10 initial - 1 deleted (id=2) + 1 inserted (id=12) = 10


if __name__ == "__main__":
    unittest.main(verbosity=2)
