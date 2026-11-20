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

    def test_sqlite_writer_reservation_failure_fails_closed_before_yield(self) -> None:
        """A busy writer lock must abort before bulk_changes can yield or alter triggers."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            dbpath = tmp.name

        raw_conn = sqlite3.connect(dbpath, timeout=0.05)
        mgr = MaterializedViewManager(raw_conn)
        try:
            mgr.conn.execute("PRAGMA journal_mode = WAL;")
            mgr.apply_schema_ddl(
                "CREATE TABLE accounts (id INT PRIMARY KEY, balance INT);",
                auto_sync_views=False,
            )
            mgr.conn.execute("INSERT INTO accounts VALUES (1, 100);")
            mgr.conn.commit()
            mgr.register_view(
                "CREATE VIEW v_total_bal AS SELECT SUM(balance) AS total FROM accounts;"
            )
            trigger_names = list(mgr.get_catalog()["v_total_bal"]["trigger_names"])
            self.assertTrue(trigger_names)
            before = {
                row[0]
                for row in mgr.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger';"
                ).fetchall()
            }
            self.assertTrue(set(trigger_names).issubset(before))

            blocker = sqlite3.connect(dbpath, timeout=0.05, isolation_level=None)
            try:
                blocker.execute("BEGIN IMMEDIATE;")
                yielded = False
                with self.assertRaisesRegex(sqlite3.OperationalError, "locked|busy"):
                    with mgr.bulk_changes():
                        yielded = True
                self.assertFalse(yielded, "bulk_changes must not yield without its writer reservation")

                after = {
                    row[0]
                    for row in mgr.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='trigger';"
                    ).fetchall()
                }
                self.assertTrue(
                    set(trigger_names).issubset(after),
                    "trigger maintenance must remain installed after lock acquisition fails",
                )
            finally:
                blocker.rollback()
                blocker.close()

            mgr.conn.execute("INSERT INTO accounts VALUES (2, 200);")
            expected = mgr.conn.execute(
                "SELECT SUM(balance) FROM accounts;"
            ).fetchone()[0]
            actual = mgr.conn.execute(
                "SELECT total FROM v_total_bal;"
            ).fetchone()[0]
            self.assertEqual(actual, expected)
        finally:
            mgr.conn.close()

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

    def test_postgres_bulk_changes_suspending_modes_are_refused_before_any_sql(self) -> None:
        """PostgreSQL never suspends its statement-level maintenance, so the
        shadow swap and drop_triggers modes are refused before any SQL."""
        class NoSqlConnection:
            in_transaction = False

            def cursor(self):
                raise AssertionError("bulk_changes issued SQL for a refused PostgreSQL mode")

        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = NoSqlConnection()
        mgr.dialect = Dialect.POSTGRES
        for kwargs in ({"use_shadow_swap": True}, {"drop_triggers": True}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "statement-level triggers maintain the views"):
                    with mgr.bulk_changes(**kwargs):
                        self.fail("entered a refused PostgreSQL bulk mode")

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

    def test_mysql_bulk_changes_shadow_swap_is_refused_before_any_sql(self) -> None:
        """MySQL/MariaDB DDL commits implicitly, so bulk_changes(use_shadow_swap=True)
        (and drop_triggers=True) is refused before touching the connection; the
        swap SQL emission above stays covered for the generator only.
        """
        class NoSqlConnection:
            in_transaction = False

            def cursor(self):
                raise AssertionError("bulk_changes issued SQL for a refused MySQL DDL mode")

        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = NoSqlConnection()
        mgr.dialect = Dialect.MYSQL
        for kwargs in ({"use_shadow_swap": True}, {"drop_triggers": True}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "commits the transaction implicitly"):
                    with mgr.bulk_changes(**kwargs):
                        self.fail("entered a refused MySQL bulk mode")

    def test_postgres_bulk_changes_keeps_maintenance_and_only_owns_the_transaction(self) -> None:
        """On PostgreSQL the block runs in one manager-owned transaction with the
        statement-level triggers active: no session bypass, no trigger DDL, no
        repopulation, one COMMIT (recording connection, no server)."""
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
            # Reports its transaction state (bulk_changes refuses drivers that
            # do not): idle until a statement runs, idle again after COMMIT.
            in_transaction = False

            def cursor(self):
                return RecordingCursor()
            def commit(self):
                executed_sqls.append("COMMIT;")
                RecordingConnection.in_transaction = False
            def rollback(self):
                executed_sqls.append("ROLLBACK;")
                RecordingConnection.in_transaction = False

        class RecordingCursor:  # noqa: F811 - marks the implicit transaction
            def execute(self, stmt, params=None):
                executed_sqls.append(str(stmt).strip())
                RecordingConnection.in_transaction = True
                return self
            def fetchall(self):
                return []
            def fetchone(self):
                return None
            def close(self):
                pass

        rec_conn = RecordingConnection()
        mgr = MaterializedViewManager.__new__(MaterializedViewManager)
        mgr.conn = rec_conn
        mgr.dialect = Dialect.POSTGRES
        mgr.engine_version = parse_engine_version("postgres:16")
        mgr.object_prefix = "__matview_"
        mgr.catalog_table = "__matview_catalog"

        mgr._read_catalog = lambda *a, **k: {
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

        with mgr.bulk_changes():
            rec_conn.cursor().execute("INSERT INTO warehouse VALUES (1, 'east', 100);")

        joined = "\n".join(executed_sqls)
        self.assertNotIn("skip_triggers", joined)
        self.assertNotRegex(joined, r"(?i)\bDROP\s+(FUNCTION|TRIGGER|VIEW|TABLE)\b")
        self.assertNotRegex(joined, r"(?i)\bCREATE\b")
        for tbl in [plan.backing_table_name, *plan.auxiliary_table_names]:
            self.assertNotIn(f"DELETE FROM {tbl}", joined)
            self.assertNotIn(f"INSERT INTO {tbl}", joined)
        self.assertIn("INSERT INTO warehouse VALUES (1, 'east', 100);", executed_sqls)
        self.assertEqual(executed_sqls[-1], "COMMIT;")
        self.assertEqual(executed_sqls.count("COMMIT;"), 1)


if __name__ == "__main__":
    unittest.main()
