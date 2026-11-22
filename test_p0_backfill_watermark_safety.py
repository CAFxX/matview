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
import re
import sqlite3
import tempfile
import threading
import unittest
import warnings

from matview.manager import MATVIEW_GENERATOR_VERSION
from matview_sync import (
    BackfillInterruptedError,
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
    generate_materialized_view,
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

        # Inject failure into the per-batch watermark advance to simulate database/catalog corruption or lock
        original_update = manager._advance_backfill_watermark
        call_count = 0

        def failing_update(cur, view_name, cursor_pk, upper_pk, max_pk, active):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise sqlite3.OperationalError("simulated catalog write failure")
            return original_update(cur, view_name, cursor_pk, upper_pk, max_pk, active)

        manager._advance_backfill_watermark = failing_update

        with self.assertRaisesRegex(sqlite3.OperationalError, "simulated catalog write failure"):
            manager.backfill_incrementally(plan, batch_size=3, snapshot_isolation=True)

        # Fail closed: connection transaction/savepoint must be cleanly closed
        self.assertFalse(self.conn.in_transaction, "savepoint transaction was left unclosed")

        # Fail closed: unpopulated view stays unclaimed with active=1 so triggers
        # remain gated and check_schema_drift() reports backfill_incomplete until
        # a retry succeeds or repair_view()/force_rebuild completes it.
        catalog = manager.get_catalog()[plan.view_name]
        self.assertEqual(catalog["backfill_active"], 1, "unpopulated view must stay fail-closed (active=1)")
        self.assertIsNone(catalog["backfill_cursor_pk"])
        self.assertIsNone(catalog["backfill_max_pk"])
        self.assertEqual(manager.check_schema_drift()[plan.view_name], "backfill_incomplete")

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

        original_update = manager._advance_backfill_watermark
        call_count = 0

        def failing_update(cur, view_name, cursor_pk, upper_pk, max_pk, active):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise sqlite3.OperationalError("simulated mid-batch watermark disk error")
            return original_update(cur, view_name, cursor_pk, upper_pk, max_pk, active)

        manager._advance_backfill_watermark = failing_update

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
        manager._advance_backfill_watermark = original_update
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
        # Already populated: a second backfill must fail closed instead of double-counting.
        from matview_sync import BackfillInterruptedError
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan_win, batch_size=0, snapshot_isolation=False)

        # Non-chunked backfill (batch_size <= 0) executes safely without snapshot isolation on an unclaimed view
        plan_win_single = manager.register_view(
            "CREATE VIEW v_win_single AS SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM win_tbl;",
            populate=False,
        )
        batches_single = manager.backfill_incrementally(plan_win_single, batch_size=0, snapshot_isolation=False)
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

    def _run_projection_inclusive_cursor_boundary(
        self,
        table_name: str,
        view_name: str,
        worker_sqls: tuple[str, ...],
        *,
        where_clause: str = "",
    ) -> None:
        self.conn.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, note TEXT, amt INT);")
        self.conn.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?, ?);",
            [(i, f"note_{i}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()

        observed_watermark: list[tuple[int, int, int]] = []
        worker_errors: list[BaseException] = []

        def on_progress(batch_idx: int, _rows_done: int) -> None:
            if batch_idx != 1:
                return
            done = threading.Event()

            def independent_worker() -> None:
                try:
                    second = sqlite3.connect(self.db_path, timeout=10)
                    try:
                        second.execute("PRAGMA busy_timeout=10000;")
                        _register_sqlite_compat_udfs(second)
                        wm = second.execute(
                            "SELECT backfill_active, backfill_cursor_pk, backfill_max_pk "
                            "FROM __matview_catalog WHERE view_name = ?;",
                            (view_name,),
                        ).fetchone()
                        self.assertEqual(wm, (1, 3, 10))
                        observed_watermark.append(wm)
                        for stmt in worker_sqls:
                            second.execute(stmt)
                        second.commit()
                    finally:
                        second.close()
                except BaseException as exc:
                    worker_errors.append(exc)
                finally:
                    done.set()

            th = threading.Thread(target=independent_worker, daemon=True)
            th.start()
            if not done.wait(timeout=10):
                raise AssertionError("independent cursor-boundary worker timed out")
            th.join(timeout=5)
            self.assertFalse(th.is_alive(), "independent cursor-boundary worker thread did not exit")
            if worker_errors:
                raise worker_errors[0]

        where_sql = f" WHERE {where_clause}" if where_clause else ""
        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            f"CREATE VIEW {view_name} AS SELECT id, note, amt FROM {table_name}{where_sql};",
            populate=False,
        )
        batches = manager.backfill_incrementally(
            plan,
            batch_size=3,
            on_progress=on_progress,
            snapshot_isolation=False,
        )
        self.assertGreaterEqual(batches, 2)
        self.assertEqual(observed_watermark, [(1, 3, 10)])

        actual = self.conn.execute(f"SELECT id, note, amt FROM {view_name} ORDER BY id;").fetchall()
        expected = self.conn.execute(
            f"SELECT id, note, amt FROM {table_name}{where_sql} ORDER BY id;"
        ).fetchall()
        self.assertEqual(actual, expected)

    def test_projection_watermark_updates_exact_inclusive_cursor_row(self) -> None:
        """Updating the exact inclusive cursor row (id = backfill_cursor_pk) during
        projection backfill must fire NEW-row maintenance and match the live SELECT."""
        self._run_projection_inclusive_cursor_boundary(
            "proj_wm_upd",
            "v_proj_wm_upd",
            ("UPDATE proj_wm_upd SET note = 'cursor_updated', amt = 9999 WHERE id = 3;",),
        )

    def test_projection_watermark_deletes_exact_inclusive_cursor_row(self) -> None:
        """Deleting the exact inclusive cursor row (id = backfill_cursor_pk) during
        projection backfill must fire OLD-row retraction and match the live SELECT."""
        self._run_projection_inclusive_cursor_boundary(
            "proj_wm_del",
            "v_proj_wm_del",
            ("DELETE FROM proj_wm_del WHERE id = 3;",),
        )

    def test_projection_watermark_filtered_boundaries_and_max_pk(self) -> None:
        """Filtered projection (with OR predicate) during pk_range_fold backfill
        must stay exact across cursor_pk, max_pk, and max_pk + 1 boundaries."""
        self._run_projection_inclusive_cursor_boundary(
            "proj_wm_filt",
            "v_proj_wm_filt",
            (
                # Exact inclusive cursor row exits WHERE filter
                "UPDATE proj_wm_filt SET amt = 5, note = 'dropped' WHERE id = 3;",
                # Exact inclusive max_pk row updated inside pending range (suppressed, folded later)
                "UPDATE proj_wm_filt SET amt = 1005, note = 'max_updated' WHERE id = 10;",
                # Row above max_pk inserted (trigger must fire)
                "INSERT INTO proj_wm_filt VALUES (11, 'above_max', 110);",
            ),
            where_clause="amt >= 20 OR note = 'keep_extra'",
        )

    def test_backfill_unreferenced_pk_update_across_watermark_boundaries(self) -> None:
        """Updating an unreferenced integer PK across the active backfill watermark
        range (unbackfilled -> above max_pk, unbackfilled -> <= cursor, <= cursor
        -> unbackfilled, <= cursor -> above max_pk) must fire the UPDATE trigger
        and stay exact across grouped, global, filtered, DISTINCT, and native-PK
        projection views whose SELECT lists do NOT reference `*` or the PK column."""
        cases = (
            (
                "agg_grp_src",
                "v_agg_grp",
                "SELECT grp, SUM(val) AS s FROM agg_grp_src GROUP BY grp",
                "ORDER BY grp",
                [(i, f"g{i % 3}", i * 10, f"n{i}") for i in range(1, 11)],
            ),
            (
                "agg_glob_src",
                "v_agg_glob",
                "SELECT SUM(val) AS s FROM agg_glob_src",
                "",
                [(i, f"g{i % 3}", i * 10, f"n{i}") for i in range(1, 11)],
            ),
            (
                "agg_filt_src",
                "v_agg_filt",
                "SELECT grp, SUM(val) AS s FROM agg_filt_src WHERE val >= 20 GROUP BY grp",
                "ORDER BY grp",
                [(i, f"g{i % 3}", i * 10, f"n{i}") for i in range(1, 11)],
            ),
            (
                "dist_src",
                "v_dist_unref_pk",
                "SELECT DISTINCT grp FROM dist_src",
                "ORDER BY grp",
                [(i, f"g{i:02d}", i * 10, f"n{i}") for i in range(1, 11)],
            ),
            (
                "proj_unref_src",
                "v_proj_unref_pk",
                "SELECT grp, val FROM proj_unref_src",
                "ORDER BY grp, val",
                [(i, f"g{i:02d}", i * 10, f"n{i}") for i in range(1, 11)],
            ),
        )
        for tbl, vname, select_sql, order_sql, rows in cases:
            with self.subTest(view=vname):
                self.conn.execute(
                    f"CREATE TABLE {tbl} (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL, note TEXT);"
                )
                self.conn.executemany(f"INSERT INTO {tbl} VALUES (?, ?, ?, ?);", rows)
                self.conn.commit()

                manager = MaterializedViewManager(self.conn)
                plan = manager.register_view(f"CREATE VIEW {vname} AS {select_sql};", populate=False)
                worker_errors: list[BaseException] = []

                def on_progress(batch_idx: int, _rows_done: int) -> None:
                    if batch_idx != 1:
                        return
                    done = threading.Event()

                    def worker() -> None:
                        try:
                            second = sqlite3.connect(self.db_path, timeout=10)
                            try:
                                second.execute("PRAGMA busy_timeout=10000;")
                                _register_sqlite_compat_udfs(second)
                                # Non-PK unreferenced update should remain harmless
                                second.execute(f"UPDATE {tbl} SET note = 'touched' WHERE id = 3;")
                                # 1. Unbackfilled (id=8 in (3,10]) -> above max_pk (100): must fold NEW
                                second.execute(f"UPDATE {tbl} SET id = 100 WHERE id = 8;")
                                # 2. Unbackfilled (id=7 in (3,10]) -> <= cursor (0): must fold NEW
                                second.execute(f"UPDATE {tbl} SET id = 0 WHERE id = 7;")
                                # 3. Already backfilled (id=2 <= 3) -> unbackfilled (id=7 in (3,10]): must retract OLD
                                second.execute(f"UPDATE {tbl} SET id = 7 WHERE id = 2;")
                                # 4. Already backfilled (id=1 <= 3) -> above max_pk (101): retract OLD + fold NEW
                                second.execute(f"UPDATE {tbl} SET id = 101 WHERE id = 1;")
                                second.commit()
                            finally:
                                second.close()
                        except BaseException as exc:
                            worker_errors.append(exc)
                        finally:
                            done.set()

                    t = threading.Thread(target=worker, daemon=True)
                    t.start()
                    self.assertTrue(done.wait(timeout=10), "worker timed out")
                    t.join(timeout=1)
                    if worker_errors:
                        raise worker_errors[0]

                batches = manager.backfill_incrementally(
                    plan,
                    batch_size=3,
                    on_progress=on_progress,
                    snapshot_isolation=False,
                )
                self.assertGreaterEqual(batches, 2)
                # Behavioral check runs BEFORE static DDL check so mutation testing
                # proves runtime MV != LIVE sensitivity.
                actual = self.conn.execute(f"SELECT * FROM {vname} {order_sql};").fetchall()
                expected = self.conn.execute(f"{select_sql} {order_sql};").fetchall()
                self.assertEqual(actual, expected, f"{vname}: MV={actual} LIVE={expected}")

                au_ddls = [d for d in plan.trigger_ddls if "_au" in d]
                self.assertTrue(au_ddls, f"{vname}: missing AFTER UPDATE trigger")
                m_of = re.search(
                    r"AFTER\s+UPDATE\s+OF\s+(.+?)\s+ON\b",
                    au_ddls[0],
                    flags=re.IGNORECASE | re.DOTALL,
                )
                self.assertIsNotNone(m_of, f"{vname}: expected AFTER UPDATE OF clause: {au_ddls[0]}")
                assert m_of is not None
                of_cols = {c.strip().strip('"').lower() for c in m_of.group(1).split(",")}
                self.assertIn("id", of_cols, f"{vname}: unreferenced PK `id` missing from OF list: {of_cols}")
                self.assertNotIn("note", of_cols, f"{vname}: unreferenced non-PK `note` should stay out of OF list")

    def test_backfill_unreferenced_rowid_update_on_text_pk_table(self) -> None:
        """On a SQLite TEXT PRIMARY KEY table (`wm_pk_col == 'rowid'`), UPDATE triggers
        for aggregates, bag projections, and NOT NULL native-PK projections must omit
        invalid `OF rowid`, guard via `OLD.rowid = NEW.rowid`, retract moved rows
        without phantom retention when subsequently deleted, and remain exact."""
        self.conn.execute(
            "CREATE TABLE kv_text_pk (k TEXT PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        self.conn.executemany(
            "INSERT INTO kv_text_pk VALUES (?, ?, ?);",
            [(f"k{i}", f"g{i % 2}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        agg_sql = "SELECT grp, SUM(val) AS s FROM kv_text_pk GROUP BY grp"
        plan_agg = manager.register_view(f"CREATE VIEW v_kv_rowid AS {agg_sql};", populate=False)

        def _make_on_progress(moves: tuple[str, ...]):
            worker_errors: list[BaseException] = []

            def on_progress(batch_idx: int, _rows_done: int) -> None:
                if batch_idx != 1:
                    return
                done = threading.Event()

                def worker() -> None:
                    try:
                        second = sqlite3.connect(self.db_path, timeout=10)
                        try:
                            second.execute("PRAGMA busy_timeout=10000;")
                            _register_sqlite_compat_udfs(second)
                            for stmt in moves:
                                second.execute(stmt)
                            second.commit()
                        finally:
                            second.close()
                    except BaseException as exc:
                        worker_errors.append(exc)
                    finally:
                        done.set()

                t = threading.Thread(target=worker, daemon=True)
                t.start()
                self.assertTrue(done.wait(timeout=10), "worker timed out")
                t.join(timeout=1)
                if worker_errors:
                    raise worker_errors[0]

            return on_progress

        batches_agg = manager.backfill_incrementally(
            plan_agg,
            batch_size=3,
            on_progress=_make_on_progress(
                (
                    "UPDATE kv_text_pk SET rowid = 100 WHERE rowid = 8;",
                    "UPDATE kv_text_pk SET rowid = 8 WHERE rowid = 2;",
                )
            ),
            snapshot_isolation=False,
        )
        self.assertGreaterEqual(batches_agg, 2)
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_kv_rowid ORDER BY grp;").fetchall(),
            self.conn.execute(f"{agg_sql} ORDER BY grp;").fetchall(),
        )

        # 2. Nullable TEXT PK projection moving rowid across watermark
        self.conn.execute("CREATE TABLE kv_proj_src (k TEXT PRIMARY KEY, val INTEGER NOT NULL);")
        self.conn.executemany(
            "INSERT INTO kv_proj_src VALUES (?, ?);",
            [(f"k{i}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()
        p2 = manager.register_view("CREATE VIEW v_kv_proj2 AS SELECT k, val FROM kv_proj_src;", populate=False)
        batches_proj = manager.backfill_incrementally(
            p2,
            batch_size=3,
            on_progress=_make_on_progress(
                (
                    "UPDATE kv_proj_src SET rowid = 100 WHERE rowid = 8;",
                    "UPDATE kv_proj_src SET rowid = 8 WHERE rowid = 2;",
                )
            ),
            snapshot_isolation=False,
        )
        self.assertGreaterEqual(batches_proj, 2)
        self.assertEqual(
            self.conn.execute("SELECT k, val FROM v_kv_proj2 ORDER BY k;").fetchall(),
            self.conn.execute("SELECT k, val FROM kv_proj_src ORDER BY k;").fetchall(),
        )

        # 3. NOT NULL TEXT PK native projection: moving backfilled rowid=2 into pending
        #    range (rowid=8) and then deleting rowid=8 must retract immediately (no phantom row).
        self.conn.execute("CREATE TABLE kv_nn_proj (k TEXT PRIMARY KEY NOT NULL, val INTEGER NOT NULL);")
        self.conn.executemany(
            "INSERT INTO kv_nn_proj VALUES (?, ?);",
            [(f"k{i}", i * 10) for i in range(1, 11)],
        )
        self.conn.commit()
        p3 = manager.register_view("CREATE VIEW v_kv_nn_proj AS SELECT val FROM kv_nn_proj;", populate=False)
        batches_nn = manager.backfill_incrementally(
            p3,
            batch_size=3,
            on_progress=_make_on_progress(
                (
                    "UPDATE kv_nn_proj SET rowid = 100 WHERE rowid = 8;",
                    "UPDATE kv_nn_proj SET rowid = 8 WHERE rowid = 2;",
                    "DELETE FROM kv_nn_proj WHERE rowid = 8;",
                )
            ),
            snapshot_isolation=False,
        )
        self.assertGreaterEqual(batches_nn, 2)
        self.assertEqual(
            self.conn.execute("SELECT val FROM v_kv_nn_proj ORDER BY val;").fetchall(),
            self.conn.execute("SELECT val FROM kv_nn_proj ORDER BY val;").fetchall(),
        )

        for p in (plan_agg, p2, p3):
            au_ddls = [d for d in p.trigger_ddls if "_au" in d]
            self.assertTrue(au_ddls)
            self.assertNotIn("AFTER UPDATE OF", au_ddls[0])
            self.assertIn("OLD.rowid = NEW.rowid", au_ddls[0])

    def test_backfill_sqlite_without_rowid_shadowed_rowid_strict_and_desc_pk(self) -> None:
        """SQLite WITHOUT ROWID, STRICT, lowercase integer PK, INTEGER PRIMARY KEY DESC,
        non-STRICT INT/BIGINT PK with REAL/TEXT values, shadowed rowid aliases, and
        nullable INT PRIMARY KEY tables must all backfill and maintain exact results."""
        self.conn.executescript(
            """
            CREATE TABLE t_wor_txt (k TEXT PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL) WITHOUT ROWID;
            INSERT INTO t_wor_txt VALUES ('a', 'g1', 10), ('b', 'g1', 20), ('c', 'g2', 30);

            CREATE TABLE t_wor_nonstrict (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL) WITHOUT ROWID;
            INSERT INTO t_wor_nonstrict VALUES (1, 'g1', 10), (2.5, 'g1', 25), ('abc', 'g2', 30);

            CREATE TABLE t_wor_strict (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL) WITHOUT ROWID, STRICT;
            INSERT INTO t_wor_strict VALUES (1, 'g1', 10), (2, 'g1', 20), (3, 'g2', 30),
                                            (4, 'g2', 40), (5, 'g1', 50), (6, 'g2', 60),
                                            (7, 'g1', 70), (8, 'g2', 80);

            CREATE TABLE t_lower_int (id integer PRIMARY KEY, rowid TEXT, _rowid_ TEXT, oid TEXT, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_lower_int VALUES (1, 'a', 'b', 'c', 'g1', 10), (2, 'd', 'e', 'f', 'g2', 20),
                                           (3, 'g', 'h', 'i', 'g1', 30), (4, 'j', 'k', 'l', 'g2', 40);

            CREATE TABLE t_desc_inline (id INTEGER PRIMARY KEY DESC, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_desc_inline VALUES (NULL, 'g1', 100), (1, 'g1', 10), (2, 'g2', 20);

            CREATE TABLE t_desc_tbl (id INTEGER, grp TEXT NOT NULL, val INTEGER NOT NULL, PRIMARY KEY (id DESC));
            INSERT INTO t_desc_tbl VALUES (1, 'g1', 10), (2, 'g2', 20), (3, 'g1', 30);

            CREATE TABLE t_nonstrict_nn (id INT PRIMARY KEY NOT NULL, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_nonstrict_nn VALUES (1, 'g1', 10), (2.5, 'g1', 25), ('abc', 'g2', 30);

            CREATE TABLE t_shadow_rowid (rowid TEXT PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_shadow_rowid VALUES ('r1', 'g1', 10), ('r2', 'g1', 20), ('r3', 'g2', 30),
                                              ('r4', 'g2', 40), ('r5', 'g1', 50), ('r6', 'g2', 60),
                                              ('r7', 'g1', 70), ('r8', 'g2', 80);

            CREATE TABLE t_shadow_all (rowid TEXT, _rowid_ TEXT, oid TEXT, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_shadow_all VALUES ('a', 'b', 'c', 'g1', 10), ('d', 'e', 'f', 'g2', 20);

            CREATE TABLE t_null_int_pk (id INT PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);
            INSERT INTO t_null_int_pk VALUES (NULL, 'g1', 100), (1, 'g1', 10), (2, 'g2', 20);
            """
        )
        self.conn.commit()
        manager = MaterializedViewManager(self.conn)

        # 1. WITHOUT ROWID + TEXT PK: no watermark triggers, group_key_batch
        p_wor_txt = manager.register_view(
            "CREATE VIEW v_wor_txt AS SELECT grp, SUM(val) AS s FROM t_wor_txt GROUP BY grp;",
            populate=True,
        )
        self.assertFalse(p_wor_txt.has_watermark_triggers)
        assert p_wor_txt.incremental_backfill_spec is not None
        self.assertEqual(p_wor_txt.incremental_backfill_spec.mode, "group_key_batch")
        self.conn.execute("UPDATE t_wor_txt SET val = 99 WHERE k = 'a';")
        self.conn.commit()
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_wor_txt ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_wor_txt GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 2. WITHOUT ROWID + non-STRICT INTEGER PK (allows 2.5 and 'abc'): must NOT use pk_range_fold
        p_wor_ns = manager.register_view(
            "CREATE VIEW v_wor_ns AS SELECT grp, SUM(val) AS s FROM t_wor_nonstrict GROUP BY grp;",
            populate=True,
            incremental_backfill=True,
            batch_size=2,
        )
        self.assertFalse(p_wor_ns.has_watermark_triggers)
        assert p_wor_ns.incremental_backfill_spec is not None
        self.assertEqual(p_wor_ns.incremental_backfill_spec.mode, "group_key_batch")
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_wor_ns ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_wor_nonstrict GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 3. WITHOUT ROWID, STRICT + INTEGER PK: uses `id` and survives mid-backfill unreferenced PK UPDATE
        p_wor_s = manager.register_view(
            "CREATE VIEW v_wor_strict AS SELECT grp, SUM(val) AS s FROM t_wor_strict GROUP BY grp;",
            populate=False,
        )
        self.assertTrue(p_wor_s.has_watermark_triggers)
        assert p_wor_s.incremental_backfill_spec is not None
        self.assertEqual(p_wor_s.incremental_backfill_spec.pk_column, "id")
        wor_errors: list[BaseException] = []

        def on_prog_wor(batch_idx: int, _done: int) -> None:
            if batch_idx != 1:
                return
            ev = threading.Event()

            def wrk() -> None:
                try:
                    c2 = sqlite3.connect(self.db_path, timeout=10)
                    try:
                        c2.execute("PRAGMA busy_timeout=10000;")
                        _register_sqlite_compat_udfs(c2)
                        c2.execute("UPDATE t_wor_strict SET id = 100 WHERE id = 7;")
                        c2.execute("UPDATE t_wor_strict SET id = 7 WHERE id = 1;")
                        c2.commit()
                    finally:
                        c2.close()
                except BaseException as exc:
                    wor_errors.append(exc)
                finally:
                    ev.set()

            th = threading.Thread(target=wrk, daemon=True)
            th.start()
            self.assertTrue(ev.wait(timeout=10))
            th.join(timeout=1)
            if wor_errors:
                raise wor_errors[0]

        manager.backfill_incrementally(p_wor_s, batch_size=3, on_progress=on_prog_wor, snapshot_isolation=False)
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_wor_strict ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_wor_strict GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 4. Lowercase `integer PRIMARY KEY` even when rowid/_rowid_/oid are all shadowed: picks `id`
        p_lower = manager.register_view(
            "CREATE VIEW v_lower_int AS SELECT grp, SUM(val) AS s FROM t_lower_int GROUP BY grp;",
            populate=False,
        )
        self.assertTrue(p_lower.has_watermark_triggers)
        assert p_lower.incremental_backfill_spec is not None
        self.assertEqual(p_lower.incremental_backfill_spec.pk_column, "id")
        manager.backfill_incrementally(p_lower, batch_size=2, snapshot_isolation=False)
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_lower_int ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_lower_int GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 5. INTEGER PRIMARY KEY DESC (inline and table-level): NOT a rowid alias, preserves NULL-PK rows
        for tbl, vn in (("t_desc_inline", "v_desc_inline"), ("t_desc_tbl", "v_desc_tbl")):
            p_desc = manager.register_view(
                f"CREATE VIEW {vn} AS SELECT grp, SUM(val) AS s FROM {tbl} GROUP BY grp;",
                populate=True,
                incremental_backfill=True,
                batch_size=2,
            )
            assert p_desc.incremental_backfill_spec is not None
            self.assertEqual(p_desc.incremental_backfill_spec.pk_column, "rowid")
            self.assertEqual(
                self.conn.execute(f"SELECT grp, s FROM {vn} ORDER BY grp;").fetchall(),
                self.conn.execute(f"SELECT grp, SUM(val) FROM {tbl} GROUP BY grp ORDER BY grp;").fetchall(),
            )

        # 6. Non-STRICT `INT PRIMARY KEY NOT NULL` with REAL (2.5) and TEXT ('abc') PKs: uses `rowid`
        p_ns_nn = manager.register_view(
            "CREATE VIEW v_ns_nn AS SELECT grp, SUM(val) AS s FROM t_nonstrict_nn GROUP BY grp;",
            populate=True,
            incremental_backfill=True,
            batch_size=2,
        )
        assert p_ns_nn.incremental_backfill_spec is not None
        self.assertEqual(p_ns_nn.incremental_backfill_spec.pk_column, "rowid")
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_ns_nn ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_nonstrict_nn GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 7. User column named `rowid TEXT PRIMARY KEY`: falls back to `_rowid_`
        p_sh1 = manager.register_view(
            "CREATE VIEW v_shadow1 AS SELECT grp, SUM(val) AS s FROM t_shadow_rowid GROUP BY grp;",
            populate=False,
        )
        self.assertTrue(p_sh1.has_watermark_triggers)
        assert p_sh1.incremental_backfill_spec is not None
        self.assertEqual(p_sh1.incremental_backfill_spec.pk_column, "_rowid_")
        au1 = [d for d in p_sh1.trigger_ddls if "_au" in d][0]
        self.assertIn("OLD._rowid_ = NEW._rowid_", au1)

        worker_errors: list[BaseException] = []

        def on_prog(batch_idx: int, _done: int) -> None:
            if batch_idx != 1:
                return
            ev = threading.Event()

            def wrk() -> None:
                try:
                    c2 = sqlite3.connect(self.db_path, timeout=10)
                    try:
                        c2.execute("PRAGMA busy_timeout=10000;")
                        _register_sqlite_compat_udfs(c2)
                        c2.execute("UPDATE t_shadow_rowid SET _rowid_ = 100 WHERE _rowid_ = 7;")
                        c2.commit()
                    finally:
                        c2.close()
                except BaseException as exc:
                    worker_errors.append(exc)
                finally:
                    ev.set()

            th = threading.Thread(target=wrk, daemon=True)
            th.start()
            self.assertTrue(ev.wait(timeout=10))
            th.join(timeout=1)
            if worker_errors:
                raise worker_errors[0]

        manager.backfill_incrementally(p_sh1, batch_size=3, on_progress=on_prog, snapshot_isolation=False)
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_shadow1 ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_shadow_rowid GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 8. All three aliases (`rowid`, `_rowid_`, `oid`) shadowed: no pk_range_fold
        p_sh_all = manager.register_view(
            "CREATE VIEW v_shadow_all AS SELECT grp, SUM(val) AS s FROM t_shadow_all GROUP BY grp;",
            populate=True,
        )
        self.assertFalse(p_sh_all.has_watermark_triggers)
        assert p_sh_all.incremental_backfill_spec is not None
        self.assertEqual(p_sh_all.incremental_backfill_spec.mode, "group_key_batch")
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_shadow_all ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_shadow_all GROUP BY grp ORDER BY grp;").fetchall(),
        )

        # 9. Nullable SQLite `INT PRIMARY KEY`: uses `rowid`, never drops NULL-PK rows
        p_null_pk = manager.register_view(
            "CREATE VIEW v_null_int_pk AS SELECT grp, SUM(val) AS s FROM t_null_int_pk GROUP BY grp;",
            populate=True,
            incremental_backfill=True,
            batch_size=2,
        )
        assert p_null_pk.incremental_backfill_spec is not None
        self.assertEqual(p_null_pk.incremental_backfill_spec.pk_column, "rowid")
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_null_int_pk ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_null_int_pk GROUP BY grp ORDER BY grp;").fetchall(),
        )

    def test_aliased_projection_get_populate_statements_sqlite_execution(self) -> None:
        """Aliased projection views must emit valid SQLite incremental populate CTEs
        with zero syntax errors or unbound `?` placeholders."""
        self.conn.executescript(
            """
            CREATE TABLE alias_src (id INTEGER PRIMARY KEY, val INTEGER NOT NULL);
            INSERT INTO alias_src VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50);
            """
        )
        self.conn.commit()
        manager = MaterializedViewManager(self.conn)
        plan = manager.register_view(
            "CREATE VIEW v_alias_proj AS SELECT s.val FROM alias_src AS s WHERE s.val >= 20;",
            populate=False,
        )
        pop_sqls = plan.get_populate_statements(incremental_backfill=True)
        for sql in pop_sqls:
            self.assertNotIn("?", sql, f"unbound placeholder in SQLite incremental populate SQL:\n{sql}")
            self.conn.execute(sql)
        self.conn.commit()
        self.assertEqual(
            self.conn.execute("SELECT val FROM v_alias_proj ORDER BY val;").fetchall(),
            self.conn.execute("SELECT s.val FROM alias_src AS s WHERE s.val >= 20 ORDER BY s.val;").fetchall(),
        )

    def test_register_view_incremental_backfill_and_generator_version_drift(self) -> None:
        """`register_view(..., populate=True, incremental_backfill=True)` works on
        unreferenced-PK views, and changing `MATVIEW_GENERATOR_VERSION` changes the
        view signature so `sync_views` rebuilds stale trigger installations."""
        import matview.manager as _mgr_mod

        self.assertGreaterEqual(int(MATVIEW_GENERATOR_VERSION), 8)
        self.conn.execute("CREATE TABLE t_inc_reg (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INT NOT NULL);")
        self.conn.executemany(
            "INSERT INTO t_inc_reg VALUES (?, ?, ?);",
            [(i, f"g{i % 2}", i * 5) for i in range(1, 9)],
        )
        self.conn.commit()
        manager = MaterializedViewManager(self.conn)
        view_sql = "CREATE VIEW v_inc_reg AS SELECT grp, SUM(val) AS s FROM t_inc_reg GROUP BY grp;"
        plan = manager.register_view(view_sql, populate=True, incremental_backfill=True, batch_size=3)
        self.assertEqual(
            self.conn.execute("SELECT grp, s FROM v_inc_reg ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, SUM(val) FROM t_inc_reg GROUP BY grp ORDER BY grp;").fetchall(),
        )
        schemas_now = manager.fetch_schema()
        sig_current = manager._compute_signature(plan, schemas_now)
        orig_ver = _mgr_mod.MATVIEW_GENERATOR_VERSION
        try:
            _mgr_mod.MATVIEW_GENERATOR_VERSION = "7"
            sig_prev = manager._compute_signature(plan, schemas_now)
        finally:
            _mgr_mod.MATVIEW_GENERATOR_VERSION = orig_ver
        self.assertNotEqual(sig_current, sig_prev, "MATVIEW_GENERATOR_VERSION bump did not change signature")
        self.conn.execute(
            "UPDATE __matview_catalog SET signature_hash = ? WHERE view_name = 'v_inc_reg';",
            (sig_prev,),
        )
        self.conn.commit()
        status = manager.sync_views([view_sql])
        self.assertEqual(status["v_inc_reg"], "updated")

    # -------------------------------------------------------------------------
    # Fail-Closed Incremental Backfill (unclaimed gate, completed markers, non-PK modes)
    # -------------------------------------------------------------------------

    def test_populate_false_gates_writes_until_backfill_claims_and_reports_incomplete(self) -> None:
        """register_view(populate=False) must gate triggers immediately via (NULL, NULL, 1),
        report backfill_incomplete, refuse silent populate=True re-registration, and fold
        exactly once when backfill_incrementally() claims the range."""
        self.conn.execute(
            "CREATE TABLE t_gate (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        self.conn.executemany(
            "INSERT INTO t_gate VALUES (?, ?, ?);",
            [(1, "a", 10), (2, "a", 20), (3, "b", 30)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        agg_sql = "CREATE VIEW v_gate_agg AS SELECT grp, COUNT(*) AS c, SUM(val) AS s FROM t_gate GROUP BY grp;"
        proj_sql = "CREATE VIEW v_gate_proj AS SELECT id, grp, val FROM t_gate WHERE val >= 15;"
        plan_agg = manager.register_view(agg_sql, populate=False)
        plan_proj = manager.register_view(proj_sql, populate=False)

        for p in (plan_agg, plan_proj):
            entry = manager.get_catalog()[p.view_name]
            self.assertEqual(
                (entry["backfill_cursor_pk"], entry["backfill_max_pk"], entry["backfill_active"]),
                (None, None, 1),
            )
            self.assertEqual(manager.check_schema_drift()[p.view_name], "backfill_incomplete")

        # Re-registering with populate=False is idempotent; populate=True must fail closed.
        manager.register_view(agg_sql, populate=False)
        with self.assertRaises(BackfillInterruptedError):
            manager.register_view(agg_sql, populate=True)
        with self.assertRaises(BackfillInterruptedError):
            manager.register_view(agg_sql, populate=True, incremental_backfill=True)

        # DML before backfill_incrementally() must be gated out (backing tables stay empty).
        self.conn.execute("INSERT INTO t_gate VALUES (4, 'a', 40);")
        self.conn.execute("UPDATE t_gate SET val = 25 WHERE id = 2;")
        self.conn.execute("DELETE FROM t_gate WHERE id = 1;")
        self.conn.commit()

        self.assertEqual(
            self.conn.execute(f"SELECT COUNT(*) FROM {plan_agg.backing_table_name};").fetchone()[0],
            0,
            "aggregate trigger fired while populate=False watermark was unclaimed",
        )
        self.assertEqual(
            self.conn.execute(f"SELECT COUNT(*) FROM {plan_proj.backing_table_name};").fetchone()[0],
            0,
            "projection trigger fired while populate=False watermark was unclaimed",
        )

        # Backfill claims the post-DML range and folds every row exactly once.
        manager.backfill_incrementally(plan_agg, batch_size=2, snapshot_isolation=False)
        manager.backfill_incrementally(plan_proj, batch_size=2, snapshot_isolation=False)

        self.assertEqual(manager.check_schema_drift()[plan_agg.view_name], "ok")
        self.assertEqual(manager.check_schema_drift()[plan_proj.view_name], "ok")
        self.assertEqual(
            self.conn.execute("SELECT grp, c, s FROM v_gate_agg ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, COUNT(*), SUM(val) FROM t_gate GROUP BY grp ORDER BY grp;").fetchall(),
        )
        self.assertEqual(
            self.conn.execute("SELECT id, grp, val FROM v_gate_proj ORDER BY id;").fetchall(),
            self.conn.execute("SELECT id, grp, val FROM t_gate WHERE val >= 15 ORDER BY id;").fetchall(),
        )

        # Post-backfill writes must now be maintained by triggers.
        self.conn.execute("INSERT INTO t_gate VALUES (5, 'c', 50);")
        self.conn.execute("UPDATE t_gate SET val = 5 WHERE id = 4;")
        self.conn.commit()
        self.assertEqual(
            self.conn.execute("SELECT grp, c, s FROM v_gate_agg ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, COUNT(*), SUM(val) FROM t_gate GROUP BY grp ORDER BY grp;").fetchall(),
        )
        self.assertEqual(
            self.conn.execute("SELECT id, grp, val FROM v_gate_proj ORDER BY id;").fetchall(),
            self.conn.execute("SELECT id, grp, val FROM t_gate WHERE val >= 15 ORDER BY id;").fetchall(),
        )

        # Duplicate backfill must refuse in both isolation modes.
        for snap in (False, True):
            with self.assertRaises(BackfillInterruptedError):
                manager.backfill_incrementally(plan_agg, batch_size=2, snapshot_isolation=snap)

    def test_populate_true_and_force_rebuild_record_completed_marker_and_refuse_backfill(self) -> None:
        """populate=True (non-incremental) and force_rebuild=True must record a completed
        watermark marker so a later backfill_incrementally() refuses fail-closed."""
        self.conn.execute(
            "CREATE TABLE t_done (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        self.conn.executemany(
            "INSERT INTO t_done VALUES (?, ?, ?);",
            [(1, "x", 10), (2, "y", 20), (3, "x", 30)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        view_sql = "CREATE VIEW v_done AS SELECT grp, COUNT(*) AS c, SUM(val) AS s FROM t_done GROUP BY grp;"
        plan = manager.register_view(view_sql, populate=True, incremental_backfill=False)
        entry = manager.get_catalog()[plan.view_name]
        self.assertIsNotNone(entry["backfill_cursor_pk"])
        self.assertIsNotNone(entry["backfill_max_pk"])
        self.assertEqual(entry["backfill_active"], 0)

        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan, batch_size=2, snapshot_isolation=False)
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan, batch_size=2, snapshot_isolation=True)
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan, batch_size=0, snapshot_isolation=False)

        # After force_rebuild=True, the completed marker must still be recorded and refuse backfill.
        plan_rebuilt = manager.register_view(view_sql, populate=True, force_rebuild=True)
        entry2 = manager.get_catalog()[plan_rebuilt.view_name]
        self.assertIsNotNone(entry2["backfill_cursor_pk"])
        self.assertIsNotNone(entry2["backfill_max_pk"])
        self.assertEqual(entry2["backfill_active"], 0)
        with self.assertRaises(UnsupportedMaterializedViewError):
            manager.backfill_incrementally(plan_rebuilt, batch_size=2, snapshot_isolation=False)
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan_rebuilt, batch_size=2, snapshot_isolation=True)
        with self.assertRaises(BackfillInterruptedError):
            manager.backfill_incrementally(plan_rebuilt, batch_size=0, snapshot_isolation=False)

        # Verify contents were never double-counted.
        self.assertEqual(
            self.conn.execute("SELECT grp, c, s FROM v_done ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, COUNT(*), SUM(val) FROM t_done GROUP BY grp ORDER BY grp;").fetchall(),
        )

    def test_pre_claim_failure_stays_fail_closed_and_repairs_cleanly(self) -> None:
        """If incremental backfill fails before claiming the watermark, the catalog stays
        at (NULL, NULL, 1), drift reports backfill_incomplete, writes stay gated, and
        repair_view() restores exact contents."""
        self.conn.execute(
            "CREATE TABLE t_pre (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        self.conn.executemany(
            "INSERT INTO t_pre VALUES (?, ?, ?);",
            [(1, "a", 10), (2, "b", 20)],
        )
        self.conn.commit()

        manager = MaterializedViewManager(self.conn)
        orig_claim = manager._claim_non_snapshot_pk_range

        def boom(*_args, **_kwargs):
            raise RuntimeError("injected pre-claim crash")

        manager._claim_non_snapshot_pk_range = boom  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(RuntimeError, "injected pre-claim crash"):
                manager.register_view(
                    "CREATE VIEW v_pre AS SELECT grp, COUNT(*) AS c, SUM(val) AS s FROM t_pre GROUP BY grp;",
                    populate=True,
                    incremental_backfill=True,
                    batch_size=1,
                )
        finally:
            manager._claim_non_snapshot_pk_range = orig_claim  # type: ignore[method-assign]

        entry = manager.get_catalog()["v_pre"]
        self.assertEqual(
            (entry["backfill_cursor_pk"], entry["backfill_max_pk"], entry["backfill_active"]),
            (None, None, 1),
        )
        self.assertEqual(manager.check_schema_drift()["v_pre"], "backfill_incomplete")

        # Writes while unclaimed stay gated (backing table stays empty).
        self.conn.execute("INSERT INTO t_pre VALUES (3, 'a', 30);")
        self.conn.commit()
        self.assertEqual(
            self.conn.execute(f"SELECT COUNT(*) FROM {entry['backing_table']};").fetchone()[0],
            0,
        )

        # repair_view() populates and marks complete.
        self.assertEqual(manager.repair_view("v_pre"), {"v_pre": "repaired"})
        self.assertEqual(manager.check_schema_drift()["v_pre"], "ok")
        self.assertEqual(
            self.conn.execute("SELECT grp, c, s FROM v_pre ORDER BY grp;").fetchall(),
            self.conn.execute("SELECT grp, COUNT(*), SUM(val) FROM t_pre GROUP BY grp ORDER BY grp;").fetchall(),
        )

    def test_non_pk_backfill_modes_record_completed_marker_and_refuse_duplicate_backfill(self) -> None:
        """group_key_batch, offset_batch, and batch_size<=0 modes must stay backfill_incomplete
        until backfilled, record a completed marker, and refuse duplicate backfills."""
        self.conn.execute(
            "CREATE TABLE t_nonpk (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        self.conn.executemany(
            "INSERT INTO t_nonpk VALUES (?, ?, ?);",
            [(i, f"g{i % 2}", i * 10) for i in range(1, 7)],
        )
        self.conn.commit()
        manager = MaterializedViewManager(self.conn)
        cases = (
            ("group_key_batch", "CREATE VIEW v_np_gk AS SELECT grp, COUNT(DISTINCT val) AS cd FROM t_nonpk GROUP BY grp;"),
            ("offset_batch", "CREATE VIEW v_np_off AS SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t_nonpk;"),
        )
        for mode, view_sql in cases:
            with self.subTest(mode=mode):
                plan = manager.register_view(view_sql, populate=False)
                assert plan.incremental_backfill_spec is not None
                self.assertEqual(plan.incremental_backfill_spec.mode, mode)
                self.assertEqual(manager.check_schema_drift()[plan.view_name], "backfill_incomplete")
                manager.backfill_incrementally(plan, batch_size=2, snapshot_isolation=True)
                self.assertEqual(manager.check_schema_drift()[plan.view_name], "ok")
                entry = manager.get_catalog()[plan.view_name]
                self.assertEqual(entry["backfill_active"], 0)
                self.assertIsNotNone(entry["backfill_cursor_pk"])
                with self.assertRaises(BackfillInterruptedError):
                    manager.backfill_incrementally(plan, batch_size=2, snapshot_isolation=True)
                with self.assertRaises(BackfillInterruptedError):
                    manager.backfill_incrementally(plan, batch_size=0, snapshot_isolation=False)


class BackfillWatermarkCodegenTests(unittest.TestCase):
    """Pure codegen tests (no temporary SQLite database required)."""

    def test_mysql_and_mariadb_guard_tracks_unreferenced_and_reserved_pk_columns(self) -> None:
        """On MySQL 8.4/9.7 and MariaDB, the outer AFTER UPDATE unchanged-row IF guard
        must include an unreferenced integer PK (including reserved-keyword PKs) across
        aggregates, DISTINCT, and projections even without COUNT(*)."""
        cases = (
            (
                "agg_mysql84",
                "mysql:8.4.0",
                "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INT NOT NULL, note TEXT);",
                "CREATE VIEW v_my_unref_pk AS SELECT grp, SUM(val) AS s FROM items GROUP BY grp;",
                "OLD.id",
                "NEW.id",
            ),
            (
                "agg_mysql97",
                "mysql:9.7.0",
                "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INT NOT NULL, note TEXT);",
                "CREATE VIEW v_my_unref_pk97 AS SELECT grp, SUM(val) AS s FROM items GROUP BY grp;",
                "OLD.id",
                "NEW.id",
            ),
            (
                "reserved_mariadb",
                "mariadb:11.8.0",
                "CREATE TABLE orders (`order` INT UNSIGNED PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INT NOT NULL);",
                "CREATE VIEW v_ma_reserved_pk AS SELECT grp, SUM(val) AS s FROM orders GROUP BY grp;",
                "OLD.`order`",
                "NEW.`order`",
            ),
            (
                "distinct_mysql84",
                "mysql:8.4.0",
                "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, note TEXT);",
                "CREATE VIEW v_my_dist AS SELECT DISTINCT grp FROM items;",
                "OLD.id",
                "NEW.id",
            ),
            (
                "projection_mariadb",
                "mariadb:11.8.0",
                "CREATE TABLE items (id INT PRIMARY KEY, val INT NOT NULL, note TEXT);",
                "CREATE VIEW v_ma_proj AS SELECT val FROM items;",
                "OLD.id",
                "NEW.id",
            ),
        )
        for label, ver_str, ddl, view_sql, old_tok, new_tok in cases:
            with self.subTest(case=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        ddl,
                        view_sql,
                        dialect=Dialect.MYSQL,
                        engine_version=ver_str,
                        watermark_catalog_table="__matview_catalog",
                        ignore_limitations=True,
                    )
                self.assertTrue(caught, "expected EngineLimitationWarning on MySQL/MariaDB TRUNCATE limitation")
                au_ddls = [d for d in plan.trigger_ddls if "_au" in d]
                self.assertTrue(au_ddls)
                m_guard = re.search(r"IF\s*\(\s*NOT\s*\((.+?)\)\s*\)\s*THEN", au_ddls[0], flags=re.DOTALL)
                self.assertIsNotNone(m_guard, f"missing outer unchanged-row guard: {au_ddls[0]}")
                assert m_guard is not None
                guard_expr = m_guard.group(1)
                self.assertIn(old_tok, guard_expr)
                self.assertIn(new_tok, guard_expr)
                self.assertNotIn("OLD.note", guard_expr)

    def test_aliased_projection_get_populate_statements_postgres_and_mysql(self) -> None:
        """Aliased projection incremental populate SQL on PostgreSQL and MySQL must
        replace every `%s` bound placeholder with procedural loop variables."""
        for dialect, ver_str in ((Dialect.POSTGRES, "postgres:15.0"), (Dialect.MYSQL, "mysql:8.4.0")):
            with self.subTest(dialect=dialect.value):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        "CREATE TABLE items (id INT PRIMARY KEY, val INT NOT NULL);",
                        "CREATE VIEW v_aliased_p AS SELECT s.val FROM items AS s WHERE s.val > 0;",
                        dialect=dialect,
                        engine_version=ver_str,
                        incremental_backfill=True,
                        ignore_limitations=True,
                    )
                pop_sql = "\n".join(plan.get_populate_statements(incremental_backfill=True))
                self.assertNotIn("%s", pop_sql, f"unbound %s in {dialect.value} incremental populate SQL:\n{pop_sql}")
                self.assertIn("__mv_cur_pk", pop_sql)
                self.assertIn("__mv_top_pk", pop_sql)

    def test_non_integer_composite_and_holistic_views_do_not_use_pk_range_watermark(self) -> None:
        """PostgreSQL INTERVAL/POINT PKs, composite PKs, and DISTINCT/holistic aggregate
        views must NOT emit pk_range_fold or watermark-gated triggers."""
        pg_cases = (
            (
                "interval_pk",
                "CREATE TABLE t_iv (id INTERVAL PRIMARY KEY, grp TEXT NOT NULL, val INT NOT NULL);",
                "CREATE VIEW v_iv AS SELECT grp, SUM(val) AS s FROM t_iv GROUP BY grp;",
            ),
            (
                "point_pk",
                "CREATE TABLE t_pt (id POINT PRIMARY KEY, grp TEXT NOT NULL, val INT NOT NULL);",
                "CREATE VIEW v_pt AS SELECT grp, SUM(val) AS s FROM t_pt GROUP BY grp;",
            ),
            (
                "composite_pk",
                "CREATE TABLE t_cp (a INT, b INT, grp TEXT NOT NULL, val INT NOT NULL, PRIMARY KEY (a, b));",
                "CREATE VIEW v_cp AS SELECT grp, SUM(val) AS s FROM t_cp GROUP BY grp;",
            ),
            (
                "distinct_agg_group_key_batch",
                "CREATE TABLE t_da (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INT NOT NULL);",
                "CREATE VIEW v_da AS SELECT grp, COUNT(DISTINCT val) AS cd FROM t_da GROUP BY grp;",
            ),
        )
        for label, ddl, view_sql in pg_cases:
            with self.subTest(case=label):
                plan = generate_materialized_view(
                    ddl,
                    view_sql,
                    dialect=Dialect.POSTGRES,
                    watermark_catalog_table="__matview_catalog",
                )
                self.assertFalse(plan.has_watermark_triggers, f"{label}: unexpected watermark triggers")
                assert plan.incremental_backfill_spec is not None
                self.assertNotEqual(plan.incremental_backfill_spec.mode, "pk_range_fold")

    def test_watermark_codegen_emits_null_cursor_gate_and_pg_share_lock(self) -> None:
        """All three dialects must gate on backfill_cursor_pk IS NULL across aggregate and
        projection views; PostgreSQL adds FOR SHARE to coordinate skipped rows with chunk
        advances, while MySQL omits LOCK IN SHARE MODE to avoid catalog AB-BA deadlocks
        and MariaDB snapshot-isolation (1020) aborts."""
        ddl = "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(16) NOT NULL, val INT NOT NULL);"
        views = (
            ("agg", "CREATE VIEW v_wm AS SELECT grp, SUM(val) AS s FROM items GROUP BY grp;"),
            ("proj", "CREATE VIEW v_wm_p AS SELECT id, grp, val FROM items WHERE val > 0;"),
        )
        for label, view_sql in views:
            for dialect, ver in (
                (Dialect.SQLITE, None),
                (Dialect.POSTGRES, "postgres:15.0"),
                (Dialect.MYSQL, "mysql:8.4.0"),
            ):
                with self.subTest(view=label, dialect=dialect.value):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = generate_materialized_view(
                            ddl,
                            view_sql,
                            dialect=dialect,
                            engine_version=ver,
                            watermark_catalog_table="__matview_catalog",
                            ignore_limitations=True,
                        )
                    joined = "\n".join(plan.trigger_ddls)
                    self.assertIn("backfill_cursor_pk IS NULL OR (", joined)
                    if dialect == Dialect.POSTGRES:
                        self.assertIn("FOR SHARE)", joined)
                    else:
                        self.assertNotIn("FOR SHARE)", joined)
                        self.assertNotIn("LOCK IN SHARE MODE", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
