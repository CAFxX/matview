#!/usr/bin/env python3
"""
Regression and isolation test suite for P0 Findings #4 and #5:
- Finding #4: Disruptive live auxiliary table mutation & view drop in `use_shadow_swap=True`
- Finding #5: `bulk_changes()` trigger suspension safety, SQLite transaction/write-lock
  isolation, autocommit trigger restoration, and PostgreSQL/MySQL atomic table swap emission.
"""

import os
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from matview_sync import (
    Dialect,
    EngineVersion,
    MaterializedViewManager,
    generate_materialized_view,
    parse_engine_version,
    parse_table_schema,
)


class TestP0BulkChangesAtomicIsolation(unittest.TestCase):
    """Test suite for bulk_changes() shadow swap isolation, aux table safety, and trigger safety."""

    # -------------------------------------------------------------------------
    # Finding #4: Live auxiliary table preservation during shadow swap
    # -------------------------------------------------------------------------

    def test_shadow_swap_exception_in_user_block_preserves_live_backing_and_aux_tables(self) -> None:
        """When user code inside bulk_changes(use_shadow_swap=True) raises an exception,

        the live backing table AND live distinct auxiliary tables must remain 100% intact.
        """
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            mgr = MaterializedViewManager(f"sqlite:///{tmp.name}")
            schema = "CREATE TABLE items (id INT PRIMARY KEY, cat TEXT, val INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO items VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30);")
            mgr.conn.commit()

            v_sql = "CREATE VIEW v_cat_agg AS SELECT cat, COUNT(DISTINCT val) AS dv, SUM(val) AS total FROM items GROUP BY cat;"
            mgr.register_view(v_sql)

            catalog = mgr.get_catalog()
            self.assertIn("v_cat_agg", catalog)
            aux_tables = catalog["v_cat_agg"]["aux_tables"]
            self.assertTrue(len(aux_tables) > 0, "View with COUNT(DISTINCT) must have auxiliary table(s)")
            backing_tbl = catalog["v_cat_agg"]["backing_table"]

            # Snapshot baseline rows
            orig_view_rows = sorted(mgr.conn.execute("SELECT * FROM v_cat_agg;").fetchall())
            orig_backing_rows = sorted(mgr.conn.execute(f"SELECT * FROM {backing_tbl};").fetchall())
            orig_aux_rows = {
                aux: sorted(mgr.conn.execute(f"SELECT * FROM {aux};").fetchall())
                for aux in aux_tables
            }
            self.assertTrue(all(len(rows) > 0 for rows in orig_aux_rows.values()))

            # Raise exception inside with bulk_changes(use_shadow_swap=True)
            with self.assertRaises(RuntimeError):
                with mgr.bulk_changes(use_shadow_swap=True):
                    mgr.conn.execute("INSERT INTO items VALUES (4, 'B', 40);")
                    raise RuntimeError("simulated user transaction failure")

            # Verify base table dirty mutation was rolled back
            base_items = mgr.conn.execute("SELECT id FROM items ORDER BY id;").fetchall()
            self.assertEqual(base_items, [(1,), (2,), (3,)])

            # Verify live backing table was preserved
            curr_backing = sorted(mgr.conn.execute(f"SELECT * FROM {backing_tbl};").fetchall())
            self.assertEqual(curr_backing, orig_backing_rows)

            # Verify live auxiliary tables were preserved (NOT deleted)
            for aux in aux_tables:
                curr_aux = sorted(mgr.conn.execute(f"SELECT * FROM {aux};").fetchall())
                self.assertEqual(curr_aux, orig_aux_rows[aux])

            # Verify view still returns original snapshot
            curr_view = sorted(mgr.conn.execute("SELECT * FROM v_cat_agg;").fetchall())
            self.assertEqual(curr_view, orig_view_rows)

            # Verify no dangling shadow or old tables remain
            master_tables = [
                r[0] for r in mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            ]
            for t in master_tables:
                self.assertFalse(t.endswith("__shadow"), f"Dangling shadow table found: {t}")
                self.assertFalse(t.endswith("__old"), f"Dangling old table found: {t}")

            # Verify triggers remain functional for subsequent normal DML
            mgr.conn.execute("INSERT INTO items VALUES (4, 'B', 40);")
            mgr.conn.commit()
            updated_view = sorted(mgr.conn.execute("SELECT * FROM v_cat_agg;").fetchall())
            self.assertEqual(len(updated_view), 2)
            # 'B' had 1 distinct val (30, sum=30), now has 2 (30, 40, sum=70)
            b_row = [r for r in updated_view if r[0] == "B"][0]
            self.assertEqual(b_row[1], 2)
            self.assertEqual(b_row[2], 70)
            mgr.conn.close()

    def test_shadow_swap_populate_failure_preserves_live_backing_and_aux_tables(self) -> None:
        """When an exception occurs DURING shadow table population, the live auxiliary

        tables must NOT have been pre-deleted, and live backing/aux tables remain intact.
        """
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            mgr = MaterializedViewManager(f"sqlite:///{tmp.name}")
            schema = "CREATE TABLE items (id INT PRIMARY KEY, cat TEXT, val INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO items VALUES (1, 'X', 10), (2, 'X', 20), (3, 'Y', 50);")
            mgr.conn.commit()

            v_sql = "CREATE VIEW v_distinct_items AS SELECT cat, COUNT(DISTINCT val) AS dv FROM items GROUP BY cat;"
            mgr.register_view(v_sql)

            catalog = mgr.get_catalog()
            aux_tables = catalog["v_distinct_items"]["aux_tables"]
            backing_tbl = catalog["v_distinct_items"]["backing_table"]

            orig_backing_rows = sorted(mgr.conn.execute(f"SELECT * FROM {backing_tbl};").fetchall())
            orig_aux_rows = {
                aux: sorted(mgr.conn.execute(f"SELECT * FROM {aux};").fetchall())
                for aux in aux_tables
            }
            self.assertTrue(all(len(rows) > 0 for rows in orig_aux_rows.values()))

            # Simulate an error while populating shadow tables (e.g. out of disk or query failure)
            real_get_pop = mgr.get_catalog()["v_distinct_items"]

            def failing_populate(incremental_backfill=False):
                raise RuntimeError("simulated disk full during shadow populate")

            with patch("matview_sync.MaterializedViewPlan.get_populate_statements", side_effect=failing_populate):
                with self.assertRaises(RuntimeError) as ctx:
                    with mgr.bulk_changes(use_shadow_swap=True):
                        mgr.conn.execute("INSERT INTO items VALUES (4, 'Y', 60);")
                self.assertIn("simulated disk full", str(ctx.exception))

            # Base table DML rolled back
            self.assertEqual(
                mgr.conn.execute("SELECT COUNT(*) FROM items WHERE id = 4;").fetchone()[0], 0
            )

            # LIVE auxiliary tables MUST still have all original rows!
            for aux in aux_tables:
                curr_aux = sorted(mgr.conn.execute(f"SELECT * FROM {aux};").fetchall())
                self.assertEqual(
                    curr_aux,
                    orig_aux_rows[aux],
                    f"Live auxiliary table {aux} was corrupted or pre-deleted before shadow populate!",
                )

            # Live backing table must also be intact
            curr_backing = sorted(mgr.conn.execute(f"SELECT * FROM {backing_tbl};").fetchall())
            self.assertEqual(curr_backing, orig_backing_rows)

            # No shadow tables left
            master_tables = [
                r[0] for r in mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            ]
            self.assertFalse(any(t.endswith("__shadow") for t in master_tables))

            # Triggers still functional
            mgr.conn.execute("INSERT INTO items VALUES (4, 'Y', 60);")
            mgr.conn.commit()
            y_dv = mgr.conn.execute("SELECT dv FROM v_distinct_items WHERE cat = 'Y';").fetchone()[0]
            self.assertEqual(y_dv, 2)
            mgr.conn.close()

    def test_shadow_swap_successful_rebuild_and_swap_with_aux_tables(self) -> None:
        """Successful bulk_changes(use_shadow_swap=True) rebuilds shadow aux tables,

        swaps them cleanly to live, updates view results, and leaves triggers fully active.
        """
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            mgr = MaterializedViewManager(f"sqlite:///{tmp.name}")
            schema = "CREATE TABLE products (pid INT PRIMARY KEY, brand TEXT, tag TEXT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO products VALUES (1, 'B1', 'T1'), (2, 'B1', 'T2'), (3, 'B2', 'T1');")
            mgr.conn.commit()

            v_sql = "CREATE VIEW v_brand_tags AS SELECT brand, COUNT(DISTINCT tag) AS dt FROM products GROUP BY brand;"
            mgr.register_view(v_sql)

            catalog = mgr.get_catalog()
            aux_tables = catalog["v_brand_tags"]["aux_tables"]
            self.assertTrue(len(aux_tables) > 0)

            with mgr.bulk_changes(use_shadow_swap=True):
                mgr.conn.execute("INSERT INTO products VALUES (4, 'B1', 'T3'), (5, 'B2', 'T2'), (6, 'B2', 'T3');")

            # Check view results after swap
            res = sorted(mgr.conn.execute("SELECT brand, dt FROM v_brand_tags;").fetchall())
            self.assertEqual(res, [("B1", 3), ("B2", 3)])

            # Check aux table contains rows for both brands
            for aux in aux_tables:
                rows = mgr.conn.execute(f"SELECT * FROM {aux};").fetchall()
                self.assertEqual(len(rows), 6)  # 3 distinct tags for B1 + 3 distinct tags for B2

            # Check no shadow tables remain
            master_tables = [
                r[0] for r in mgr.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            ]
            self.assertFalse(any(t.endswith("__shadow") for t in master_tables))

            # Normal DML updates the swapped view and aux table
            mgr.conn.execute("INSERT INTO products VALUES (7, 'B1', 'T4');")
            mgr.conn.commit()
            b1_dt = mgr.conn.execute("SELECT dt FROM v_brand_tags WHERE brand = 'B1';").fetchone()[0]
            self.assertEqual(b1_dt, 4)
            mgr.conn.close()

    # -------------------------------------------------------------------------
    # Finding #5: SQLite Concurrency, WAL Isolation, and Autocommit Trigger Safety
    # -------------------------------------------------------------------------

    def test_sqlite_wal_concurrent_reader_isolation_during_shadow_swap(self) -> None:
        """In SQLite WAL mode, a concurrent reader on an independent connection

        never observes missing views, missing tables, or uncommitted states during
        bulk_changes(use_shadow_swap=True).
        """
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            dbpath = tmp.name

        try:
            # Setup DB with WAL mode
            setup_conn = sqlite3.connect(dbpath)
            setup_conn.execute("PRAGMA journal_mode = WAL;")
            setup_conn.close()

            mgr = MaterializedViewManager(f"sqlite:///{dbpath}")
            schema = "CREATE TABLE metrics (id INT PRIMARY KEY, sensor TEXT, reading INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute(
                "INSERT INTO metrics VALUES (1, 's1', 10), (2, 's1', 20), (3, 's2', 15), (4, 's2', 25);"
            )
            mgr.conn.commit()

            v_sql = "CREATE VIEW v_sensor_summary AS SELECT sensor, COUNT(DISTINCT reading) AS dr, SUM(reading) AS total FROM metrics GROUP BY sensor;"
            mgr.register_view(v_sql)

            # Concurrent reader on independent connection and thread
            reader_errors = []
            observed_row_counts = []
            stop_event = threading.Event()

            def reader_thread_func():
                r_conn = sqlite3.connect(dbpath, isolation_level=None, timeout=1.0)
                try:
                    while not stop_event.is_set():
                        try:
                            rows = r_conn.execute("SELECT sensor, dr, total FROM v_sensor_summary ORDER BY sensor;").fetchall()
                            observed_row_counts.append(len(rows))
                        except Exception as e:
                            reader_errors.append(f"{type(e).__name__}: {e}")
                        time.sleep(0.001)
                finally:
                    r_conn.close()

            reader_thread = threading.Thread(target=reader_thread_func)
            reader_thread.start()

            # Execute bulk_changes with shadow swap
            with mgr.bulk_changes(use_shadow_swap=True):
                # Add a batch of new sensors and readings
                mgr.conn.execute(
                    "INSERT INTO metrics VALUES (5, 's3', 100), (6, 's3', 200), (7, 's4', 300);"
                )
                time.sleep(0.02)  # Give reader thread time to query while in-flight

            # Let reader observe post-commit state
            time.sleep(0.02)
            stop_event.set()
            reader_thread.join(timeout=3.0)

            # Assert reader encountered ZERO errors (never saw DROP VIEW or DROP TABLE)
            self.assertEqual(reader_errors, [], f"Concurrent WAL reader encountered errors: {reader_errors}")

            # Reader observed either 2 sensors (pre-commit) or 4 sensors (post-commit)
            self.assertTrue(len(observed_row_counts) > 0)
            self.assertTrue(set(observed_row_counts).issubset({2, 4}))

            # Final query confirms 4 sensors
            final_rows = mgr.conn.execute("SELECT sensor, dr, total FROM v_sensor_summary ORDER BY sensor;").fetchall()
            self.assertEqual(len(final_rows), 4)
            mgr.conn.close()

        finally:
            if os.path.exists(dbpath):
                try:
                    os.remove(dbpath)
                except OSError:
                    pass

    def test_sqlite_concurrent_writer_blocked_during_bulk_changes(self) -> None:
        """While Connection A is inside with manager.bulk_changes(), Connection B is

        safely locked out from performing concurrent writes, ensuring triggers are not
        bypassed by outside concurrent writers.
        """
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            dbpath = tmp.name

        try:
            setup_conn = sqlite3.connect(dbpath)
            setup_conn.execute("PRAGMA journal_mode = WAL;")
            setup_conn.close()

            mgr = MaterializedViewManager(f"sqlite:///{dbpath}")
            schema = "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO accounts VALUES (1, 100);")
            mgr.conn.commit()
            mgr.register_view("CREATE VIEW v_total_bal AS SELECT SUM(balance) AS total FROM accounts;")

            writer_blocked = False

            with mgr.bulk_changes():
                mgr.conn.execute("INSERT INTO accounts VALUES (2, 200);")

                # Try concurrent write from Connection B with a short timeout
                conn_b = sqlite3.connect(dbpath, timeout=0.05)
                try:
                    conn_b.execute("INSERT INTO accounts VALUES (3, 300);")
                    conn_b.commit()
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "busy" in str(e):
                        writer_blocked = True
                finally:
                    conn_b.close()

            self.assertTrue(
                writer_blocked,
                "Concurrent writer on Connection B should be locked out while Connection A holds write transaction in bulk_changes()",
            )

            # After Connection A exits bulk_changes(), Connection B can write normally and triggers fire
            conn_b = sqlite3.connect(dbpath, timeout=5.0)
            try:
                conn_b.execute("INSERT INTO accounts VALUES (3, 300);")
                conn_b.commit()
            finally:
                conn_b.close()

            # View should reflect accounts 1 (100), 2 (200), and 3 (300) = 600
            total = mgr.conn.execute("SELECT total FROM v_total_bal;").fetchone()[0]
            self.assertEqual(total, 600)
            mgr.conn.close()

        finally:
            if os.path.exists(dbpath):
                try:
                    os.remove(dbpath)
                except OSError:
                    pass

    def test_sqlite_autocommit_connection_trigger_restoration_on_abort(self) -> None:
        """When the manager's connection has isolation_level=None (autocommit),

        an exception inside bulk_changes() rolls back dirty DML and restores triggers cleanly.
        """
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            raw_conn = sqlite3.connect(tmp.name, isolation_level=None)
            mgr = MaterializedViewManager(raw_conn)
            schema = "CREATE TABLE audit_log (id INT PRIMARY KEY, msg TEXT, code INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO audit_log VALUES (1, 'init', 10);")

            mgr.register_view("CREATE VIEW v_audit_sum AS SELECT SUM(code) AS sum_code FROM audit_log;")
            self.assertEqual(mgr.conn.execute("SELECT sum_code FROM v_audit_sum;").fetchone()[0], 10)

            # Test A: In-place bulk_changes abort on autocommit connection
            with self.assertRaises(RuntimeError):
                with mgr.bulk_changes(use_shadow_swap=False, drop_triggers=False):
                    mgr.conn.execute("INSERT INTO audit_log VALUES (2, 'aborted_dml', 20);")
                    raise RuntimeError("simulated bulk failure on autocommit")

            # Mutation was rolled back
            self.assertEqual(
                mgr.conn.execute("SELECT id FROM audit_log ORDER BY id;").fetchall(), [(1,)]
            )
            self.assertEqual(
                mgr.conn.execute("SELECT sum_code FROM v_audit_sum;").fetchone()[0], 10
            )

            # Triggers were restored: subsequent normal DML works
            mgr.conn.execute("INSERT INTO audit_log VALUES (3, 'normal_dml', 30);")
            self.assertEqual(
                mgr.conn.execute("SELECT sum_code FROM v_audit_sum;").fetchone()[0], 40
            )

            # Test B: Shadow-swap bulk_changes abort on autocommit connection
            with self.assertRaises(RuntimeError):
                with mgr.bulk_changes(use_shadow_swap=True):
                    mgr.conn.execute("INSERT INTO audit_log VALUES (4, 'aborted_shadow', 40);")
                    raise RuntimeError("simulated shadow swap abort on autocommit")

            self.assertEqual(
                mgr.conn.execute("SELECT id FROM audit_log ORDER BY id;").fetchall(), [(1,), (3,)]
            )
            self.assertEqual(
                mgr.conn.execute("SELECT sum_code FROM v_audit_sum;").fetchone()[0], 40
            )

            # Triggers still functional after shadow abort
            mgr.conn.execute("INSERT INTO audit_log VALUES (5, 'after_shadow_abort', 50);")
            self.assertEqual(
                mgr.conn.execute("SELECT sum_code FROM v_audit_sum;").fetchone()[0], 90
            )
            raw_conn.close()

    # -------------------------------------------------------------------------
    # Dialect emission checks: PostgreSQL and MySQL atomic RENAME
    # -------------------------------------------------------------------------

    def test_postgres_atomic_shadow_swap_sql_emission(self) -> None:
        """PostgreSQL shadow swap must emit ALTER TABLE live RENAME TO old,

        ALTER TABLE shadow RENAME TO live, CREATE OR REPLACE VIEW (never DROP VIEW),
        and DROP TABLE IF EXISTS old for backing and auxiliary tables.
        """
        schema = "CREATE TABLE orders (id INT PRIMARY KEY, region TEXT, amount NUMERIC);"
        v_sql = "CREATE VIEW v_regional_orders AS SELECT region, COUNT(DISTINCT amount) AS da, SUM(amount) AS tot FROM orders GROUP BY region;"

        plan = generate_materialized_view(schema, v_sql, dialect=Dialect.POSTGRES, ignore_limitations=True)

        # Create manager instance with postgres dialect hint
        mock_conn = MagicMock()
        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = mock_conn
        mgr.dialect = Dialect.POSTGRES
        mgr.engine_version = parse_engine_version("postgres:16")
        mgr.object_prefix = "__matview_"
        mgr.catalog_table = "__matview_catalog"
        mgr.pg_use_transition_tables = False

        shadow_tbl = mgr._shadow_name(plan.backing_table_name)
        shadow_aux_map = {aux: mgr._shadow_name(aux) for aux in plan.auxiliary_table_names}

        swap_sqls = mgr._build_shadow_swap_sqls(plan, shadow_tbl, shadow_aux_map)

        # 1. Assert DROP VIEW is NEVER emitted for PostgreSQL
        for sql in swap_sqls:
            self.assertFalse(
                re.search(r"\bDROP\s+VIEW\b", sql, re.IGNORECASE),
                f"PostgreSQL shadow swap must never drop public view! Found: {sql}",
            )

        # 2. Assert CREATE OR REPLACE VIEW is emitted
        has_create_or_replace_view = any(
            re.search(r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW\s+v_regional_orders\b", s, re.IGNORECASE)
            for s in swap_sqls
        )
        self.assertTrue(
            has_create_or_replace_view,
            f"Expected CREATE OR REPLACE VIEW in swap_sqls: {swap_sqls}",
        )

        # 3. Assert backing table rename live -> old, shadow -> live, drop old
        old_backing = mgr._old_name(plan.backing_table_name)
        self.assertIn(f"ALTER TABLE {plan.backing_table_name} RENAME TO {old_backing};", swap_sqls)
        self.assertIn(f"ALTER TABLE {shadow_tbl} RENAME TO {plan.backing_table_name};", swap_sqls)
        self.assertIn(f"DROP TABLE IF EXISTS {old_backing};", swap_sqls)

        # 4. Assert auxiliary tables rename live -> old, shadow -> live, drop old
        for aux, s_aux in shadow_aux_map.items():
            old_aux = mgr._old_name(aux)
            self.assertIn(f"ALTER TABLE {aux} RENAME TO {old_aux};", swap_sqls)
            self.assertIn(f"ALTER TABLE {s_aux} RENAME TO {aux};", swap_sqls)
            self.assertIn(f"DROP TABLE IF EXISTS {old_aux};", swap_sqls)

        # 5. Verify ordering: RENAME live -> old, RENAME shadow -> live, CREATE OR REPLACE VIEW, DROP old
        idx_rename_old = swap_sqls.index(f"ALTER TABLE {plan.backing_table_name} RENAME TO {old_backing};")
        idx_rename_live = swap_sqls.index(f"ALTER TABLE {shadow_tbl} RENAME TO {plan.backing_table_name};")
        idx_replace_view = next(
            i for i, s in enumerate(swap_sqls) if re.search(r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW\b", s, re.IGNORECASE)
        )
        # DROP old must appear AFTER CREATE OR REPLACE VIEW (or at the cleanup step)
        idx_drop_old = [i for i, s in enumerate(swap_sqls) if s == f"DROP TABLE IF EXISTS {old_backing};"][-1]

        self.assertLess(idx_rename_old, idx_rename_live)
        self.assertLess(idx_rename_live, idx_replace_view)
        self.assertLess(idx_replace_view, idx_drop_old)

    def test_mysql_atomic_shadow_swap_sql_emission(self) -> None:
        """MySQL/MariaDB shadow swap must emit a single atomic RENAME TABLE statement

        swapping backing and auxiliary tables simultaneously, CREATE OR REPLACE VIEW
        (never DROP VIEW), and DROP TABLE IF EXISTS for the old tables.
        """
        schema = "CREATE TABLE sales (id INT PRIMARY KEY, category VARCHAR(64), customer_id INT);"
        v_sql = "CREATE VIEW v_cat_custs AS SELECT category, COUNT(DISTINCT customer_id) AS dc FROM sales GROUP BY category;"

        plan = generate_materialized_view(schema, v_sql, dialect=Dialect.MYSQL, ignore_limitations=True)

        mock_conn = MagicMock()
        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = mock_conn
        mgr.dialect = Dialect.MYSQL
        mgr.engine_version = parse_engine_version("mysql:8.4")
        mgr.object_prefix = "__matview_"
        mgr.catalog_table = "__matview_catalog"
        mgr.pg_use_transition_tables = False

        shadow_tbl = mgr._shadow_name(plan.backing_table_name)
        shadow_aux_map = {aux: mgr._shadow_name(aux) for aux in plan.auxiliary_table_names}

        swap_sqls = mgr._build_shadow_swap_sqls(plan, shadow_tbl, shadow_aux_map)

        # 1. Assert DROP VIEW is NEVER emitted for MySQL
        for sql in swap_sqls:
            self.assertFalse(
                re.search(r"\bDROP\s+VIEW\b", sql, re.IGNORECASE),
                f"MySQL shadow swap must never drop public view! Found: {sql}",
            )

        # 2. Assert single atomic RENAME TABLE is emitted
        rename_stmts = [s for s in swap_sqls if s.startswith("RENAME TABLE ")]
        self.assertEqual(
            len(rename_stmts),
            1,
            f"MySQL shadow swap must execute backing and aux swaps in exactly ONE atomic RENAME TABLE statement: {swap_sqls}",
        )
        atomic_rename = rename_stmts[0]

        # Verify all table pairs are in that single statement
        old_backing = mgr._old_name(plan.backing_table_name)
        self.assertIn(f"{plan.backing_table_name} TO {old_backing}", atomic_rename)
        self.assertIn(f"{shadow_tbl} TO {plan.backing_table_name}", atomic_rename)
        for aux, s_aux in shadow_aux_map.items():
            old_aux = mgr._old_name(aux)
            self.assertIn(f"{aux} TO {old_aux}", atomic_rename)
            self.assertIn(f"{s_aux} TO {aux}", atomic_rename)

        # 3. Assert CREATE OR REPLACE (or CREATE OR REPLACE ALGORITHM=TEMPTABLE) VIEW is emitted
        has_replace_view = any(
            re.search(r"^\s*CREATE\s+OR\s+REPLACE\s+(?:ALGORITHM\s*=\s*[A-Za-z0-9_]+\s+)?VIEW\b", s, re.IGNORECASE)
            for s in swap_sqls
        )
        self.assertTrue(
            has_replace_view,
            f"Expected CREATE OR REPLACE VIEW in MySQL swap_sqls: {swap_sqls}",
        )

        # 4. Assert DROP TABLE IF EXISTS for old tables
        drop_old_stmts = [s for s in swap_sqls if s.startswith("DROP TABLE IF EXISTS ")]
        self.assertTrue(len(drop_old_stmts) > 0)
        final_drop = drop_old_stmts[-1]
        self.assertIn(old_backing, final_drop)
        for aux in shadow_aux_map.keys():
            self.assertIn(mgr._old_name(aux), final_drop)

    def test_mysql_bulk_changes_execution_avoids_pre_delete_and_view_drop(self) -> None:
        """Executing bulk_changes(use_shadow_swap=True) on MySQL recording mock verifies

        that live auxiliary tables are never deleted prior to shadow populate, populate
        statements are redirected to shadow tables, and view is replaced atomically.
        """
        schema = "CREATE TABLE inventory (sku INT PRIMARY KEY, loc VARCHAR(32), qty INT);"
        v_sql = "CREATE VIEW v_loc_count AS SELECT loc, COUNT(DISTINCT qty) AS dq FROM inventory GROUP BY loc;"
        plan = generate_materialized_view(schema, v_sql, dialect=Dialect.MYSQL, ignore_limitations=True)

        executed_sqls = []

        class RecordingCursor:
            def execute(self, stmt, params=None):
                executed_sqls.append(str(stmt).strip())
                return self
            def fetchall(self):
                return []
            def fetchone(self):
                return None
            def close(self):
                pass

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()
            def commit(self):
                executed_sqls.append("COMMIT;")
            def rollback(self):
                executed_sqls.append("ROLLBACK;")

        rec_conn = RecordingConnection()
        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = rec_conn
        mgr.dialect = Dialect.MYSQL
        mgr.engine_version = parse_engine_version("mysql:8.4")
        mgr.object_prefix = "__matview_"
        mgr.catalog_table = "__matview_catalog"
        mgr.pg_use_transition_tables = False

        # Mock catalog
        mgr.get_catalog = lambda: {
            "v_loc_count": {
                "view_name": "v_loc_count",
                "view_sql": v_sql,
                "strategy": plan.strategy.value,
                "signature_hash": "sig123",
                "backing_table": plan.backing_table_name,
                "aux_tables": plan.auxiliary_table_names,
                "trigger_names": plan.trigger_names,
                "function_names": plan.function_names,
                "trigger_ddls": plan.trigger_ddls,
            }
        }
        mgr.fetch_schema = lambda: parse_table_schema(schema)

        with mgr.bulk_changes(use_shadow_swap=True):
            rec_conn.cursor().execute("INSERT INTO inventory VALUES (1, 'warehouse1', 50);")

        # 1. Assert DELETE FROM <live_aux> was NEVER executed before shadow populate!
        aux_table = plan.auxiliary_table_names[0]
        live_delete = f"DELETE FROM {aux_table};"
        self.assertNotIn(
            live_delete,
            executed_sqls,
            f"Live auxiliary table was deleted in bulk_changes(use_shadow_swap=True)! Executed: {executed_sqls}",
        )

        # 2. Assert shadow auxiliary table was created and populated
        s_aux = mgr._shadow_name(aux_table)
        self.assertTrue(any(s.startswith(f"CREATE TABLE {s_aux}") for s in executed_sqls))
        self.assertTrue(any(f"INSERT INTO {s_aux}" in s for s in executed_sqls))

        # 3. Assert atomic RENAME TABLE executed
        self.assertTrue(any(s.startswith("RENAME TABLE ") for s in executed_sqls))

        # 4. Assert DROP VIEW was NEVER called
        self.assertFalse(any(re.search(r"\bDROP\s+VIEW\b", s, re.IGNORECASE) for s in executed_sqls))

    def test_restoration_failure_is_reported_and_transactional_ddl_is_rolled_back(self) -> None:
        """A failed trigger reinstall must never be silently committed as a clean rollback."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            mgr = MaterializedViewManager(f"sqlite:///{tmp.name}")
            schema = "CREATE TABLE restore_items (id INT PRIMARY KEY, grp TEXT, amount INT);"
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO restore_items VALUES (1, 'A', 10);")
            mgr.conn.commit()
            mgr.register_view(
                "CREATE VIEW restore_totals AS SELECT grp, SUM(amount) AS total "
                "FROM restore_items GROUP BY grp;"
            )

            raw_conn = mgr.conn
            original_cursor = raw_conn.cursor
            injected = {"fail": False}

            class CursorProxy:
                def __init__(self, cursor):
                    self._cursor = cursor

                def execute(self, statement, params=None):
                    if injected["fail"] and str(statement).lstrip().upper().startswith("CREATE TRIGGER"):
                        injected["fail"] = False
                        raise sqlite3.OperationalError("injected trigger recreation failure")
                    if params is None:
                        return self._cursor.execute(statement)
                    return self._cursor.execute(statement, params)

                def __getattr__(self, name):
                    return getattr(self._cursor, name)

            class ConnectionProxy:
                def cursor(self, *args, **kwargs):
                    return CursorProxy(original_cursor(*args, **kwargs))

                def __getattr__(self, name):
                    return getattr(raw_conn, name)

            mgr.conn = ConnectionProxy()
            injected["fail"] = True
            with self.assertRaisesRegex(RuntimeError, "operation failed and maintenance-trigger restoration failed"):
                with mgr.bulk_changes():
                    mgr.conn.cursor().execute(
                        "INSERT INTO restore_items VALUES (2, 'A', 20);"
                    )
                    raise ValueError("injected user failure")

            # SQLite DDL is transactional: rollback of the restoration savepoint
            # must leave the original triggers installed and the user's DML undone.
            mgr.conn = raw_conn
            self.assertEqual(
                raw_conn.execute("SELECT id FROM restore_items ORDER BY id").fetchall(),
                [(1,)],
            )
            trigger_names = {
                row[0] for row in raw_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            }
            self.assertTrue(trigger_names, "original maintenance triggers must survive failed restoration")

            raw_conn.execute("INSERT INTO restore_items VALUES (3, 'B', 7);")
            raw_conn.commit()
            live = raw_conn.execute(
                "SELECT grp, SUM(amount) FROM restore_items GROUP BY grp ORDER BY grp"
            ).fetchall()
            reader = raw_conn.execute(
                "SELECT grp, total FROM restore_totals ORDER BY grp"
            ).fetchall()
            self.assertEqual(reader, live)

    def test_postgres_bulk_changes_execution_avoids_pre_delete_and_view_drop(self) -> None:
        """Executing bulk_changes(use_shadow_swap=True) on PostgreSQL recording mock verifies

        that live auxiliary tables are never deleted prior to shadow populate, populate
        statements are redirected to shadow tables, and view is replaced atomically via
        CREATE OR REPLACE VIEW without ever dropping the public view.
        """
        schema = "CREATE TABLE warehouse (id INT PRIMARY KEY, zone TEXT, val INT);"
        v_sql = "CREATE VIEW v_zone_distinct AS SELECT zone, COUNT(DISTINCT val) AS dv FROM warehouse GROUP BY zone;"
        plan = generate_materialized_view(schema, v_sql, dialect=Dialect.POSTGRES, ignore_limitations=True)

        executed_sqls = []

        class RecordingCursor:
            def execute(self, stmt, params=None):
                executed_sqls.append(str(stmt).strip())
                return self
            def fetchall(self):
                return []
            def fetchone(self):
                return None
            def close(self):
                pass

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()
            def commit(self):
                executed_sqls.append("COMMIT;")
            def rollback(self):
                executed_sqls.append("ROLLBACK;")

        rec_conn = RecordingConnection()
        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = rec_conn
        mgr.dialect = Dialect.POSTGRES
        mgr.engine_version = parse_engine_version("postgres:16")
        mgr.object_prefix = "__matview_"
        mgr.catalog_table = "__matview_catalog"
        mgr.pg_use_transition_tables = False

        mgr.get_catalog = lambda: {
            "v_zone_distinct": {
                "view_name": "v_zone_distinct",
                "view_sql": v_sql,
                "strategy": plan.strategy.value,
                "signature_hash": "sig456",
                "backing_table": plan.backing_table_name,
                "aux_tables": plan.auxiliary_table_names,
                "trigger_names": plan.trigger_names,
                "function_names": plan.function_names,
                "trigger_ddls": plan.trigger_ddls,
            }
        }
        mgr.fetch_schema = lambda: parse_table_schema(schema)

        with mgr.bulk_changes(use_shadow_swap=True):
            rec_conn.cursor().execute("INSERT INTO warehouse VALUES (1, 'east', 100);")

        # 1. Assert DELETE FROM <live_aux> was NEVER executed before shadow populate
        aux_table = plan.auxiliary_table_names[0]
        live_delete = f"DELETE FROM {aux_table};"
        self.assertNotIn(
            live_delete,
            executed_sqls,
            f"Live auxiliary table was deleted in PostgreSQL bulk_changes(use_shadow_swap=True)! Executed: {executed_sqls}",
        )

        # 2. Assert shadow auxiliary table was created and populated
        s_aux = mgr._shadow_name(aux_table)
        self.assertTrue(any(s.startswith(f"CREATE TABLE {s_aux}") for s in executed_sqls))
        self.assertTrue(any(f"INSERT INTO {s_aux}" in s for s in executed_sqls))

        # 3. Assert CREATE OR REPLACE VIEW was executed
        self.assertTrue(
            any(re.search(r"^\s*CREATE\s+OR\s+REPLACE\s+VIEW\s+v_zone_distinct\b", s, re.IGNORECASE) for s in executed_sqls),
            f"Expected CREATE OR REPLACE VIEW in executed_sqls: {executed_sqls}",
        )

        # 4. Assert DROP VIEW was NEVER called
        self.assertFalse(any(re.search(r"\bDROP\s+VIEW\b", s, re.IGNORECASE) for s in executed_sqls))



    def test_sqlite_begin_immediate_failure_propagates_before_trigger_suspension(self) -> None:
        """A failed SQLite writer reservation must not degrade to a deferred savepoint."""
        executed = []

        class RecordingCursor:
            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                if sql == "BEGIN IMMEDIATE;":
                    raise sqlite3.OperationalError("injected writer reservation failure")
                return self

        class RecordingConnection:
            in_transaction = False

            def cursor(self):
                return RecordingCursor()

            def rollback(self):
                executed.append("ROLLBACK;")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = RecordingConnection()
        manager.dialect = Dialect.SQLITE
        manager.engine_version = parse_engine_version("sqlite:3.45")
        manager.object_prefix = "__matview_"
        manager.catalog_table = "__matview_catalog"
        manager.pg_use_transition_tables = False
        manager.get_catalog = lambda: {}

        with self.assertRaisesRegex(sqlite3.OperationalError, "writer reservation failure"):
            with manager.bulk_changes():
                self.fail("bulk_changes must fail before entering the user block")

        self.assertEqual(executed[0], "BEGIN IMMEDIATE;")
        self.assertNotIn("SAVEPOINT __mv_bulk_sp;", executed)
        self.assertFalse(any(sql.upper().startswith("DROP TRIGGER") for sql in executed))


    def test_postgres_bypass_and_savepoint_failure_fails_closed(self) -> None:
        """Do not drop PostgreSQL maintenance functions if neither isolation mechanism works."""
        executed = []

        class RecordingCursor:
            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                if sql == "SAVEPOINT __mv_bulk_sp;":
                    raise RuntimeError("injected savepoint failure")
                if sql == "SET LOCAL matview.skip_triggers = 'on';":
                    raise RuntimeError("injected session bypass failure")
                return self

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()

            def rollback(self):
                executed.append("ROLLBACK;")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = RecordingConnection()
        manager.dialect = Dialect.POSTGRES
        manager.engine_version = parse_engine_version("postgres:16")
        manager.object_prefix = "__matview_"
        manager.catalog_table = "__matview_catalog"
        manager.pg_use_transition_tables = False
        manager.get_catalog = lambda: {
            "v_guard": {
                "view_name": "v_guard",
                "trigger_names": [],
                "function_names": ["mv_guard_insert_fn", "mv_guard_update_fn"],
                "view_sql": "CREATE VIEW v_guard AS SELECT id FROM guarded_rows;",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "could not establish the required transactional savepoint"):
            with manager.bulk_changes():
                self.fail("bulk_changes must fail before entering the user block")

        self.assertIn("SAVEPOINT __mv_bulk_sp;", executed)
        self.assertIn("SET LOCAL matview.skip_triggers = 'on';", executed)
        self.assertFalse(
            any(sql.upper().startswith("DROP FUNCTION") for sql in executed),
            executed,
        )
        self.assertEqual(executed[-1], "ROLLBACK;")


    def test_mysql_session_bypass_without_savepoint_fails_closed(self) -> None:
        """A session bypass alone is insufficient when MySQL cannot roll back bulk DML."""
        executed = []

        class RecordingCursor:
            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                if sql == "SAVEPOINT __mv_bulk_sp;":
                    raise RuntimeError("injected savepoint failure")
                return self

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()

            def rollback(self):
                executed.append("ROLLBACK;")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = RecordingConnection()
        manager.dialect = Dialect.MYSQL
        manager.engine_version = parse_engine_version("mysql:8.4")
        manager.object_prefix = "__matview_"
        manager.catalog_table = "__matview_catalog"
        manager.pg_use_transition_tables = False
        manager.get_catalog = lambda: {
            "v_guard": {
                "view_name": "v_guard",
                "trigger_names": ["mv_guard_insert"],
                "function_names": [],
                "view_sql": "CREATE VIEW v_guard AS SELECT id FROM guarded_rows;",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "required transactional savepoint"):
            with manager.bulk_changes():
                self.fail("bulk_changes must fail before entering the user block")

        self.assertIn("SAVEPOINT __mv_bulk_sp;", executed)
        self.assertIn("SET @__matview_skip_triggers = 1;", executed)
        self.assertFalse(
            any(sql.upper().startswith("DROP TRIGGER") for sql in executed),
            executed,
        )
        self.assertEqual(executed[-1], "ROLLBACK;")


    def test_mysql_session_bypass_failure_fails_closed_before_trigger_ddl(self) -> None:
        """A failed MySQL session bypass must not fall back to implicitly committed trigger DDL."""
        executed = []

        class RecordingCursor:
            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                if sql == "SET @__matview_skip_triggers = 1;":
                    raise RuntimeError("session variables unavailable")
                return self

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()

            def rollback(self):
                executed.append("ROLLBACK;")

            def commit(self):
                executed.append("COMMIT;")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = RecordingConnection()
        manager.dialect = Dialect.MYSQL
        manager.engine_version = parse_engine_version("mysql:8.4")
        manager.object_prefix = "__matview_"
        manager.catalog_table = "__matview_catalog"
        manager.pg_use_transition_tables = False
        manager.get_catalog = lambda: {
            "v_guard": {
                "view_name": "v_guard",
                "trigger_names": ["mv_guard_insert", "mv_guard_update"],
                "function_names": [],
                "view_sql": "CREATE VIEW v_guard AS SELECT id FROM guarded_rows;",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "refusing to drop maintenance triggers"):
            with manager.bulk_changes():
                self.fail("bulk_changes must fail before entering the user block")

        self.assertIn("SAVEPOINT __mv_bulk_sp;", executed)
        self.assertIn("SET @__matview_skip_triggers = 1;", executed)
        self.assertFalse(
            any(sql.upper().startswith("DROP TRIGGER") for sql in executed),
            executed,
        )


    def test_mysql_bulk_changes_rejects_implicit_commit_options_before_catalog_access(self) -> None:
        class Connection:
            def cursor(self):
                raise AssertionError("unsafe options must be rejected before SQL")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = Connection()
        manager.dialect = Dialect.MYSQL
        manager.get_catalog = lambda: self.fail("unsafe options must be rejected before catalog access")

        for options in ({"drop_triggers": True}, {"use_shadow_swap": True}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(RuntimeError, "implicitly committing DDL"):
                    with manager.bulk_changes(**options):
                        self.fail("unsafe MySQL DDL option must be rejected")



    def test_sqlite_savepoint_setup_failure_releases_new_writer_reservation(self) -> None:
        """If SAVEPOINT setup fails after BEGIN IMMEDIATE, release the writer lock."""
        executed = []

        class RecordingCursor:
            def execute(self, statement, params=None):
                sql = str(statement).strip()
                executed.append(sql)
                if sql == "SAVEPOINT __mv_bulk_sp;":
                    raise sqlite3.OperationalError("injected savepoint failure")
                return self

            def close(self):
                pass

        class RecordingConnection:
            in_transaction = False

            def cursor(self):
                return RecordingCursor()

            def rollback(self):
                executed.append("ROLLBACK;")

        manager = MaterializedViewManager.__new__(MaterializedViewManager)
        manager.conn = RecordingConnection()
        manager.dialect = Dialect.SQLITE
        manager.engine_version = parse_engine_version("sqlite:3.45")
        manager.object_prefix = "__matview_"
        manager.catalog_table = "__matview_catalog"
        manager.pg_use_transition_tables = False
        manager.get_catalog = lambda: {}

        with self.assertRaisesRegex(sqlite3.OperationalError, "savepoint failure"):
            with manager.bulk_changes():
                self.fail("bulk_changes must not enter the user block after savepoint setup fails")

        self.assertEqual(executed, ["BEGIN IMMEDIATE;", "SAVEPOINT __mv_bulk_sp;", "ROLLBACK;"])


    def test_sqlite_savepoint_setup_failure_preserves_caller_transaction(self) -> None:
        """A failed savepoint must not roll back a transaction owned by the caller."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            raw = sqlite3.connect(tmp.name)
            raw.execute("CREATE TABLE caller_rows (id INTEGER PRIMARY KEY);")
            raw.commit()
            raw.execute("BEGIN;")
            raw.execute("INSERT INTO caller_rows VALUES (7);")

            original_cursor = raw.cursor

            class CursorProxy:
                def __init__(self, cursor):
                    self._cursor = cursor

                def execute(self, statement, params=None):
                    if str(statement).strip() == "SAVEPOINT __mv_bulk_sp;":
                        raise sqlite3.OperationalError("injected caller savepoint failure")
                    if params is None:
                        return self._cursor.execute(statement)
                    return self._cursor.execute(statement, params)

                def __getattr__(self, name):
                    return getattr(self._cursor, name)

            class ConnectionProxy:
                def cursor(self, *args, **kwargs):
                    return CursorProxy(original_cursor(*args, **kwargs))

                def __getattr__(self, name):
                    return getattr(raw, name)

            manager = MaterializedViewManager.__new__(MaterializedViewManager)
            manager.conn = ConnectionProxy()
            manager.dialect = Dialect.SQLITE
            manager.engine_version = parse_engine_version("sqlite:3.45")
            manager.object_prefix = "__matview_"
            manager.catalog_table = "__matview_catalog"
            manager.pg_use_transition_tables = False
            manager.get_catalog = lambda: {}

            with self.assertRaisesRegex(sqlite3.OperationalError, "caller savepoint failure"):
                with manager.bulk_changes():
                    self.fail("bulk_changes must not enter the user block after savepoint setup fails")

            # The caller's active transaction and uncommitted write must survive.
            self.assertTrue(raw.in_transaction)
            self.assertEqual(raw.execute("SELECT id FROM caller_rows").fetchall(), [(7,)])
            raw.rollback()
            self.assertFalse(raw.in_transaction)
            self.assertEqual(raw.execute("SELECT id FROM caller_rows").fetchall(), [])
            raw.close()


    def test_sqlite_bulk_changes_fails_before_user_block_when_independent_writer_holds_lock(self) -> None:
        """A real competing SQLite writer must prevent bulk trigger suspension."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            manager = MaterializedViewManager(f"sqlite:///{tmp.name}")
            manager.apply_schema_ddl(
                "CREATE TABLE contention_rows (id INTEGER PRIMARY KEY, value INTEGER);",
                auto_sync_views=False,
            )
            manager.conn.execute("INSERT INTO contention_rows VALUES (1, 10);")
            manager.conn.commit()
            manager.register_view(
                "CREATE VIEW contention_view AS "
                "SELECT id, value FROM contention_rows;"
            )
            manager.conn.commit()

            catalog = manager.get_catalog()
            trigger_names = list(catalog["contention_view"]["trigger_names"])
            triggers_before = manager.conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'contention_rows' "
                "ORDER BY name"
            ).fetchall()
            rows_before = manager.conn.execute(
                "SELECT * FROM contention_view ORDER BY id"
            ).fetchall()

            # The independent connection reserves SQLite's single-writer slot.
            competing = sqlite3.connect(tmp.name, timeout=0)
            try:
                competing.execute("BEGIN IMMEDIATE;")
                competing.execute(
                    "INSERT INTO contention_rows VALUES (2, 20);"
                )
                manager.conn.execute("PRAGMA busy_timeout = 0;")

                with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                    with manager.bulk_changes():
                        self.fail(
                            "bulk_changes must not enter the user block while "
                            "another connection owns the SQLite writer reservation"
                        )

                self.assertFalse(manager.conn.in_transaction)
                self.assertTrue(competing.in_transaction)
                self.assertEqual(
                    competing.execute(
                        "SELECT id, value FROM contention_rows ORDER BY id"
                    ).fetchall(),
                    [(1, 10), (2, 20)],
                )
                self.assertEqual(
                    manager.conn.execute(
                        "SELECT * FROM contention_view ORDER BY id"
                    ).fetchall(),
                    rows_before,
                )
                triggers_after = manager.conn.execute(
                    "SELECT name, sql FROM sqlite_master "
                    "WHERE type = 'trigger' AND tbl_name = 'contention_rows' "
                    "ORDER BY name"
                ).fetchall()
                self.assertEqual(triggers_after, triggers_before)
                self.assertTrue(
                    set(trigger_names).issubset({row[0] for row in triggers_after})
                )
            finally:
                competing.rollback()
                competing.close()

            # Once the competing writer releases its reservation, normal DML
            # must still maintain the materialized view.
            manager.conn.execute(
                "INSERT INTO contention_rows VALUES (2, 20);"
            )
            manager.conn.commit()
            self.assertEqual(
                manager.conn.execute(
                    "SELECT * FROM contention_view ORDER BY id"
                ).fetchall(),
                [(1, 10), (2, 20)],
            )
            manager.conn.close()


if __name__ == "__main__":
    unittest.main()
