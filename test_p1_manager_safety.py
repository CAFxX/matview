#!/usr/bin/env python3
"""
Tests for P1 Bug #9 (Cursor leak prevention in MaterializedViewManager) and
P1 Bug #10 (Non-atomic registration rollback & partial DDL cleanup on MySQL/MariaDB).

Verifies:
1. Zero leaked cursors across all MaterializedViewManager operations (ensure_catalog,
   fetch_schema, register_view, sync_views, get_catalog, check_schema_drift,
   repair_view, apply_schema_ddl, drop_view) on BOTH success and injected failure paths.
2. Safe driver cursor handling (sqlite3, DB-API, mock cursors without close()).
3. MySQL/MariaDB partial DDL rollback cleanup: when registration fails mid-process,
   all created objects (triggers, views, backing tables) are dropped in reverse dependency
   order and partial catalog records deleted.
4. Pre-existing table preservation: if registration fails on an initial collision with
   a pre-existing table, cleanup does NOT drop the pre-existing table or destroy user data.
5. Live multi-engine integration when MYSQL_DSN or POSTGRES_DSN is set.
"""

from __future__ import annotations

import os
import re
import sqlite3
import unittest
from typing import Any, List, Optional, Set, Tuple
from unittest.mock import MagicMock

from test_matview_sync import FLOAT_OPT_IN
from matview_sync import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    MaintenanceStrategy,
    MaterializedViewManager,
    _extract_created_object,
    parse_table_schema,
)


class CursorTrackingWrapper:
    """
    Wraps a DB-API cursor, tracking its active/closed lifecycle.
    Intentionally does NOT implement __enter__/__exit__ to verify that code
    never assumes cursor is a Python context manager (matching sqlite3.Cursor).
    """

    def __init__(self, real_cursor: Any, connection: CursorTrackingConnection) -> None:
        self._real_cursor = real_cursor
        self._connection = connection
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self in self._connection.open_cursors:
                self._connection.open_cursors.remove(self)
            self._connection.closed_cursors.add(self)
            close_fn = getattr(self._real_cursor, "close", None)
            if callable(close_fn):
                close_fn()

    def __getattr__(self, name: str) -> Any:
        if name in ("__enter__", "__exit__"):
            raise AttributeError(f"Cursor object does not support context management: {name}")
        return getattr(self._real_cursor, name)


class CursorTrackingConnection:
    """
    Connection wrapper that tracks all created cursors and detects leaks.
    """

    def __init__(self, real_conn: Any, cursor_factory: Optional[Any] = None) -> None:
        self._real_conn = real_conn
        self.cursor_factory = cursor_factory
        self.open_cursors: Set[CursorTrackingWrapper] = set()
        self.closed_cursors: Set[CursorTrackingWrapper] = set()
        self.total_cursors_created: int = 0

    def cursor(self, *args: Any, **kwargs: Any) -> CursorTrackingWrapper:
        if self.cursor_factory:
            real_cur = self.cursor_factory(self._real_conn, *args, **kwargs)
        else:
            real_cur = self._real_conn.cursor(*args, **kwargs)
        if isinstance(real_cur, CursorTrackingWrapper):
            wrapped = real_cur
        else:
            wrapped = CursorTrackingWrapper(real_cur, self)
        self.open_cursors.add(wrapped)
        self.total_cursors_created += 1
        return wrapped

    def assert_zero_leaked(self, msg: str = "") -> None:
        leaked_count = len(self.open_cursors)
        if leaked_count > 0:
            extra = f" ({msg})" if msg else ""
            raise AssertionError(
                f"Leaked {leaked_count} cursor(s)! Total created: {self.total_cursors_created}, "
                f"closed: {len(self.closed_cursors)}{extra}"
            )

    def reset_tracking(self) -> None:
        self.open_cursors.clear()
        self.closed_cursors.clear()
        self.total_cursors_created = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_conn, name)


class FailingCursorWrapper(CursorTrackingWrapper):
    """
    Cursor wrapper that raises an exception when a matching statement is executed.
    """

    def __init__(
        self,
        real_cursor: Any,
        connection: CursorTrackingConnection,
        fail_predicate: Any,
        exception_factory: Any,
    ) -> None:
        super().__init__(real_cursor, connection)
        self._fail_predicate = fail_predicate
        self._exception_factory = exception_factory

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if self._fail_predicate(sql):
            raise self._exception_factory(sql)
        return self._real_cursor.execute(sql, *args, **kwargs)


class TestManagerCursorSafety(unittest.TestCase):
    """
    Verifies that MaterializedViewManager leaks zero cursors across all operations
    under normal execution and injected failure paths.
    """

    def setUp(self) -> None:
        raw_conn = sqlite3.connect(":memory:")
        self.tracked_conn = CursorTrackingConnection(raw_conn)
        # Create base test table
        with raw_conn:
            raw_conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, dept TEXT, amt REAL);")
            raw_conn.execute("INSERT INTO sales VALUES (1, 'eng', 100.0), (2, 'eng', 200.0), (3, 'sales', 50.0);")

    def tearDown(self) -> None:
        self.tracked_conn.assert_zero_leaked("tearDown verification")
        self.tracked_conn.close()

    def test_ensure_catalog_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        self.tracked_conn.assert_zero_leaked("Catalog ensure should close its cursor")
        self.assertGreater(len(self.tracked_conn.closed_cursors), 0)

    def test_fetch_schema_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        self.tracked_conn.reset_tracking()
        schemas = manager.fetch_schema()
        self.assertIn("sales", schemas)
        self.tracked_conn.assert_zero_leaked("fetch_schema should close all introspection cursors")
        self.assertGreater(len(self.tracked_conn.closed_cursors), 0)

    def test_register_view_and_idempotence_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        self.tracked_conn.reset_tracking()

        view_sql = "CREATE VIEW mv_dept_sales AS SELECT dept, SUM(amt) as tot FROM sales GROUP BY dept;"
        plan = manager.register_view(view_sql, populate=True)
        self.assertIsNotNone(plan)
        self.tracked_conn.assert_zero_leaked("register_view must close all cursors on success")

        # Idempotent re-registration (signature match, no rebuild)
        self.tracked_conn.reset_tracking()
        plan2 = manager.register_view(view_sql, force_rebuild=False)
        self.assertEqual(plan2.view_name, plan.view_name)
        self.tracked_conn.assert_zero_leaked("Idempotent register_view must close all cursors")

        # Forced rebuild
        self.tracked_conn.reset_tracking()
        plan3 = manager.register_view(view_sql, force_rebuild=True)
        self.assertEqual(plan3.view_name, plan.view_name)
        self.tracked_conn.assert_zero_leaked("Forced rebuild register_view must close all cursors")

    def test_get_catalog_and_fallback_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        view_sql = "CREATE VIEW mv_dept_sales AS SELECT dept, SUM(amt) as tot FROM sales GROUP BY dept;"
        manager.register_view(view_sql)

        self.tracked_conn.reset_tracking()
        catalog = manager.get_catalog()
        self.assertIn("mv_dept_sales", catalog)
        self.tracked_conn.assert_zero_leaked("get_catalog must close cursor")

        # Simulate legacy catalog without watermark columns by creating a legacy catalog table
        raw_conn2 = sqlite3.connect(":memory:")
        with raw_conn2:
            raw_conn2.execute(
                f"""
                CREATE TABLE {manager.catalog_table} (
                    view_name VARCHAR(255) PRIMARY KEY,
                    view_sql TEXT NOT NULL,
                    strategy VARCHAR(64) NOT NULL,
                    signature_hash VARCHAR(64) NOT NULL,
                    backing_table VARCHAR(255) NOT NULL,
                    aux_tables_json TEXT NOT NULL,
                    trigger_names_json TEXT NOT NULL,
                    function_names_json TEXT NOT NULL,
                    trigger_ddls_json TEXT NOT NULL
                );
                """
            )
            raw_conn2.execute(
                f"""
                INSERT INTO {manager.catalog_table} VALUES (
                    'mv_legacy', 'CREATE VIEW mv_legacy AS SELECT 1', 'algebraic', 'sig',
                    'store_mv_legacy', '[]', '[]', '[]', '[]'
                );
                """
            )
        tracked2 = CursorTrackingConnection(raw_conn2)
        # Avoid _ensure_catalog altering table by passing an already initialized mock
        manager2 = MaterializedViewManager.__new__(MaterializedViewManager)
        manager2.conn = tracked2
        manager2.dialect = Dialect.SQLITE
        manager2.catalog_table = manager.catalog_table
        manager2.object_prefix = DEFAULT_OBJECT_PREFIX

        legacy_cat = manager2.get_catalog()
        self.assertIn("mv_legacy", legacy_cat)
        self.assertIsNone(legacy_cat["mv_legacy"]["backfill_cursor_pk"])
        tracked2.assert_zero_leaked("Fallback get_catalog without watermark columns must close cursor")
        tracked2.close()

    def test_check_schema_drift_and_repair_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        view_sql = "CREATE VIEW mv_drift_test AS SELECT dept, SUM(amt) as tot FROM sales GROUP BY dept;"
        manager.register_view(view_sql)

        self.tracked_conn.reset_tracking()
        drift_report = manager.check_schema_drift()
        self.assertEqual(drift_report.get("mv_drift_test"), "ok")
        self.tracked_conn.assert_zero_leaked("check_schema_drift must close cursor on ok")

        # Cause drift by altering table
        with self.tracked_conn._real_conn:
            self.tracked_conn._real_conn.execute("ALTER TABLE sales ADD COLUMN note TEXT;")

        self.tracked_conn.reset_tracking()
        drift_report2 = manager.check_schema_drift()
        self.assertEqual(drift_report2.get("mv_drift_test"), "drifted")
        self.tracked_conn.assert_zero_leaked("check_schema_drift must close cursor on drift")

        # Repair without drift rebuild
        self.tracked_conn.reset_tracking()
        repair_res = manager.repair_view("mv_drift_test", rebuild_on_schema_drift=False)
        self.assertEqual(repair_res.get("mv_drift_test"), "repaired")
        self.tracked_conn.assert_zero_leaked("repair_view (repaired path) must close cursor")

        # Repair with drift rebuild
        self.tracked_conn.reset_tracking()
        repair_res2 = manager.repair_view("mv_drift_test", rebuild_on_schema_drift=True)
        self.assertEqual(repair_res2.get("mv_drift_test"), "rebuilt")
        self.tracked_conn.assert_zero_leaked("repair_view (rebuilt path) must close cursor")

    def test_apply_schema_ddl_and_drop_view_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        view_sql = "CREATE VIEW mv_ddl_test AS SELECT dept, SUM(amt) as tot FROM sales GROUP BY dept;"
        manager.register_view(view_sql)

        self.tracked_conn.reset_tracking()
        manager.apply_schema_ddl("ALTER TABLE sales ADD COLUMN tag TEXT;", auto_sync_views=True)
        self.tracked_conn.assert_zero_leaked("apply_schema_ddl must close cursor")

        # Drop existing view
        self.tracked_conn.reset_tracking()
        dropped = manager.drop_view("mv_ddl_test")
        self.assertTrue(dropped)
        self.tracked_conn.assert_zero_leaked("drop_view (existing) must close cursor")

        # Drop non-existing view
        self.tracked_conn.reset_tracking()
        dropped_nonexistent = manager.drop_view("nonexistent_view")
        self.assertFalse(dropped_nonexistent)
        self.tracked_conn.assert_zero_leaked("drop_view (non-existing) must close cursor")

    def test_injected_failure_during_register_zero_leaks(self) -> None:
        """Inject failure during statement execution in register_view to verify no cursor leak on error."""
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        self.tracked_conn.reset_tracking()

        def failing_cursor_fn(conn: Any, *args: Any, **kwargs: Any) -> Any:
            real_cur = conn.cursor(*args, **kwargs)
            return FailingCursorWrapper(
                real_cur,
                self.tracked_conn,
                fail_predicate=lambda sql: "INSERT INTO" in sql and manager.catalog_table in sql,
                exception_factory=lambda sql: sqlite3.OperationalError("Simulated catalog failure"),
            )

        self.tracked_conn.cursor_factory = failing_cursor_fn
        try:
            with self.assertRaises(sqlite3.OperationalError):
                manager.register_view(
                    "CREATE VIEW mv_fail_test AS SELECT dept, COUNT(*) as c FROM sales GROUP BY dept;"
                )
            self.tracked_conn.assert_zero_leaked("register_view must not leak cursor on failure")
        finally:
            self.tracked_conn.cursor_factory = None

    def test_injected_failure_during_apply_schema_ddl_zero_leaks(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        self.tracked_conn.reset_tracking()

        with self.assertRaises(sqlite3.OperationalError):
            manager.apply_schema_ddl("MALFORMED SQL STATEMENT HERE !!!;")
        self.tracked_conn.assert_zero_leaked("apply_schema_ddl must not leak cursor on syntax error")

    def test_driver_without_close_method_safely_handled(self) -> None:
        """Verify _cursor helper handles drivers or mock cursors where close is not callable or None."""
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)

        class NoCloseCursor:
            def execute(self, *args: Any, **kwargs: Any) -> None:
                pass

        class NoCloseConn:
            def cursor(self) -> Any:
                return NoCloseCursor()

        manager.conn = NoCloseConn()
        # Should execute without raising AttributeError: 'NoCloseCursor' object has no attribute 'close'
        with manager._cursor() as cur:
            self.assertIsInstance(cur, NoCloseCursor)

        # Non-callable close attribute
        class NonCallableCloseCursor:
            close = "not_a_function"

        class NonCallableCloseConn:
            def cursor(self) -> Any:
                return NonCallableCloseCursor()

        manager.conn = NonCallableCloseConn()
        with manager._cursor() as cur:
            self.assertIsInstance(cur, NonCallableCloseCursor)

    def test_diff_views_zero_leaks_and_dropped_object_detection(self) -> None:
        manager = MaterializedViewManager(self.tracked_conn, dialect=Dialect.SQLITE)
        view_sql = "CREATE VIEW mv_diff_test AS SELECT dept, SUM(amt) AS tot FROM sales GROUP BY dept;"
        plan = manager.register_view(view_sql)
        self.tracked_conn.reset_tracking()

        # 1. Clean state -> NOOP and zero leaked cursors
        rep_noop = manager.diff_views([view_sql])
        self.assertEqual(rep_noop["mv_diff_test"].status, "NOOP")
        self.tracked_conn.assert_zero_leaked("diff_views clean NOOP must not leak cursors")

        # 2. Drop all maintenance triggers out-of-band -> must NOT report false NOOP
        for t in plan.trigger_names:
            self.tracked_conn._real_conn.execute(f"DROP TRIGGER IF EXISTS {t};")
        self.tracked_conn._real_conn.commit()
        rep_alt = manager.diff_views([view_sql])
        self.assertEqual(rep_alt["mv_diff_test"].status, "ALTER")
        self.assertEqual(
            set(rep_alt["mv_diff_test"].pending_changes["triggers"]["to_create"]),
            set(plan.trigger_names),
        )
        self.tracked_conn.assert_zero_leaked("diff_views trigger ALTER must not leak cursors")

        # 3. Drop reader view out-of-band -> must report ALTER (recreate reader view without rebuild) and include reader view DDL
        self.tracked_conn._real_conn.execute("DROP VIEW IF EXISTS mv_diff_test;")
        self.tracked_conn._real_conn.commit()
        rep_rep = manager.diff_views([view_sql])
        self.assertEqual(rep_rep["mv_diff_test"].status, "ALTER")
        self.assertEqual(rep_rep["mv_diff_test"].pending_changes["reader_view"]["action"], "CREATE")
        self.assertTrue(
            any("CREATE VIEW" in s and "mv_diff_test" in s for s in rep_rep["mv_diff_test"].pending_ddls),
            "ALTER for dropped reader view must include reader view DDL",
        )
        self.tracked_conn.assert_zero_leaked("diff_views reader ALTER must not leak cursors")


class TestMySqlNonAtomicRegistrationRollback(unittest.TestCase):
    """
    Verifies P1 Bug #10 (Non-atomic registration rollback on MySQL/MariaDB).
    Tests that:
    1. _extract_created_object correctly parses trigger, view, and table DDLs across MySQL dialect forms.
    2. Partial DDL created during a failed registration attempt is completely cleaned up in reverse order.
    3. Pre-existing tables colliding with registration are NOT dropped on failure.
    """

    def test_extract_created_object_patterns(self) -> None:
        cases = [
            ("CREATE TABLE store_mv (id INT PRIMARY KEY);", ("table", "store_mv")),
            ("CREATE TABLE IF NOT EXISTS store_mv (id INT);", ("table", "store_mv")),
            ("CREATE TEMPORARY TABLE IF NOT EXISTS `store_mv` (id INT);", ("table", "store_mv")),
            ('CREATE TABLE "public"."store_mv" (id INT);', ("table", "store_mv")),
            ("CREATE VIEW mv AS SELECT 1;", ("view", "mv")),
            ("CREATE OR REPLACE VIEW mv AS SELECT 1;", ("view", "mv")),
            ("CREATE OR REPLACE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` VIEW `mv` AS SELECT 1;", ("view", "mv")),
            ('CREATE VIEW "mv" AS SELECT 1;', ("view", "mv")),
            ("CREATE TRIGGER tr_ins AFTER INSERT ON t FOR EACH ROW BEGIN END;", ("trigger", "tr_ins")),
            ("CREATE OR REPLACE TRIGGER tr_upd BEFORE UPDATE ON t ...", ("trigger", "tr_upd")),
            ("CREATE DEFINER=`admin`@`%` TRIGGER `tr_del` AFTER DELETE ON `t` ...", ("trigger", "tr_del")),
            ("INSERT INTO t VALUES (1);", None),
            ("SELECT * FROM t;", None),
            ("ALTER TABLE t ADD COLUMN x INT;", None),
            ("DROP TABLE IF EXISTS t;", None),
        ]
        for sql, expected in cases:
            res = _extract_created_object(sql)
            self.assertEqual(res, expected, f"Failed parsing: {sql}")

    def test_mysql_partial_ddl_cleanup_on_injected_trigger_failure(self) -> None:
        """
        Simulate MySQL non-transactional DDL failure:
        Backing table and view CREATE succeed, then trigger creation raises an error.
        Cleanup must drop the view, drop the backing table, and delete any catalog row.
        """
        raw_conn = sqlite3.connect(":memory:")
        tracked = CursorTrackingConnection(raw_conn)
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

        # Create base table in SQLite
        with raw_conn:
            raw_conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, score REAL);")

        view_sql = "CREATE VIEW mv_users AS SELECT status, SUM(score) as tot FROM users GROUP BY status;"
        schemas = parse_table_schema("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, score REAL);")
        manager.fetch_schema = lambda extra_schema_sql=None: schemas
        manager._fetch_plain_views = lambda **_kw: ({}, None)

        # Intercept cursor execution: let CREATE TABLE and CREATE VIEW succeed, but fail on CREATE TRIGGER
        executed_statements: List[str] = []

        def failing_cursor(conn: Any, *args: Any, **kwargs: Any) -> Any:
            real_cur = conn.cursor(*args, **kwargs)

            class InterceptingCursor(CursorTrackingWrapper):
                def execute(self, sql: str, *a: Any, **kw: Any) -> Any:
                    executed_statements.append(sql)
                    if "CREATE TRIGGER" in sql.upper():
                        raise sqlite3.OperationalError("Simulated MySQL trigger error (e.g. syntax or permission)")
                    # If simulating MySQL on SQLite, adjust MySQL-specific clauses so DDL succeeds in SQLite
                    if "CREATE ALGORITHM" in sql:
                        sql = sql.replace("ALGORITHM = TEMPTABLE ", "")
                    sql = sql.replace(" CHARACTER SET utf8mb4 COLLATE utf8mb4_bin", "")
                    return self._real_cursor.execute(sql, *a, **kw)

            return InterceptingCursor(real_cur, tracked)

        tracked.cursor_factory = failing_cursor

        with self.assertRaises(sqlite3.OperationalError) as exc_ctx:
            manager.register_view(view_sql)
        self.assertIn("Simulated MySQL trigger error", str(exc_ctx.exception))

        tracked.assert_zero_leaked("Cursor must not leak after MySQL rollback cleanup")

        # Verify that DROP statements were issued for the partially created objects
        drop_view_executed = any("DROP VIEW IF EXISTS mv_users" in s for s in executed_statements)
        drop_table_executed = any("DROP TABLE IF EXISTS" in s and "users" in s for s in executed_statements)
        self.assertTrue(drop_view_executed, "Cleanup must drop the created view")
        self.assertTrue(drop_table_executed, "Cleanup must drop the created backing table")

        # Verify the database has no orphaned view or backing table
        cur = raw_conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name LIKE '%mv_users%';")
        remaining = [row[0] for row in cur.fetchall()]
        cur.close()
        self.assertEqual(remaining, [], f"Orphaned objects left in database: {remaining}")

        # Verify catalog table does not have mv_users
        cur = raw_conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {manager.catalog_table} WHERE view_name = 'mv_users';")
        cat_count = cur.fetchone()[0]
        cur.close()
        self.assertEqual(cat_count, 0, "Catalog must not retain partially registered view")

    def test_mysql_preexisting_table_collision_is_preserved(self) -> None:
        """
        Verify that if registration fails because a pre-existing table collides
        on the first CREATE TABLE, cleanup does NOT drop the pre-existing table!
        """
        raw_conn = sqlite3.connect(":memory:")
        tracked = CursorTrackingConnection(raw_conn)
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

        # Create base table and pre-existing collision table with important user data
        store_tbl_name = f"{manager.object_prefix}store_mv_collision"
        with raw_conn:
            raw_conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL);")
            raw_conn.execute(f"CREATE TABLE {store_tbl_name} (id INTEGER PRIMARY KEY, note TEXT);")
            raw_conn.execute(f"INSERT INTO {store_tbl_name} VALUES (42, 'DO_NOT_DELETE_IMPORTANT_DATA');")

        view_sql = f"CREATE VIEW mv_collision AS SELECT id, SUM(balance) as tot FROM accounts GROUP BY id;"
        schemas = parse_table_schema("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL);")
        manager.fetch_schema = lambda extra_schema_sql=None: schemas

        def collision_cursor(conn: Any, *args: Any, **kwargs: Any) -> Any:
            real_cur = conn.cursor(*args, **kwargs)

            class CollisionWrapper(CursorTrackingWrapper):
                def execute(self, sql: str, *a: Any, **kw: Any) -> Any:
                    if "CREATE TABLE" in sql and store_tbl_name in sql:
                        # Fail with MySQL table already exists error
                        raise sqlite3.OperationalError(f"Table '{store_tbl_name}' already exists")
                    return self._real_cursor.execute(sql, *a, **kw)

            return CollisionWrapper(real_cur, tracked)

        tracked.cursor_factory = collision_cursor

        with self.assertRaises(sqlite3.OperationalError):
            manager.register_view(view_sql)

        # Verify that pre-existing collision table STILL EXISTS and its data is intact!
        cur = raw_conn.cursor()
        cur.execute(f"SELECT note FROM {store_tbl_name} WHERE id = 42;")
        row = cur.fetchone()
        cur.close()
        self.assertIsNotNone(row, "Pre-existing colliding table must NOT be dropped by cleanup!")
        self.assertEqual(row[0], "DO_NOT_DELETE_IMPORTANT_DATA")
        tracked.assert_zero_leaked("Zero leaks on collision failure")


class TestNativeEngineManagerSafety(unittest.TestCase):
    """
    Native database tests against live MySQL/MariaDB or PostgreSQL when environment
    variables (MYSQL_DSN, POSTGRES_DSN) are set in CI.
    """

    def test_native_mysql_zero_leaks_and_rollback_safety(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; skipping live MySQL/MariaDB safety test")

        import pymysql
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(dsn)
        raw_conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/"),
            autocommit=False,
        )
        tracked = CursorTrackingConnection(raw_conn)
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

        prefix = "t_p1_my_"
        base_tbl = f"{prefix}items"
        view_name = f"{prefix}mv_items"
        backing_tbl = f"{manager.object_prefix}store_{view_name}"

        cur = tracked.cursor()
        try:
            cur.execute(f"DROP VIEW IF EXISTS {view_name};")
            cur.execute(f"DROP TABLE IF EXISTS {backing_tbl};")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
            cur.execute(f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, grp VARCHAR(32), qty INT);")
            cur.execute(f"INSERT INTO {base_tbl} VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 5);")
            tracked.commit()
        finally:
            cur.close()
        tracked.reset_tracking()

        try:
            # 1. Successful registration and lifecycle with zero leaks
            view_sql = f"CREATE VIEW {view_name} AS SELECT grp, SUM(qty) AS total_qty FROM {base_tbl} GROUP BY grp;"
            plan = manager.register_view(view_sql, populate=True)
            self.assertEqual(plan.view_name, view_name)
            tracked.assert_zero_leaked("Native MySQL register_view must not leak cursors")

            cat = manager.get_catalog()
            self.assertIn(view_name, cat)
            tracked.assert_zero_leaked("Native MySQL get_catalog must not leak cursors")

            drift = manager.check_schema_drift()
            self.assertEqual(drift.get(view_name), "ok")
            tracked.assert_zero_leaked("Native MySQL check_schema_drift must not leak cursors")

            repair = manager.repair_view(view_name)
            tracked.assert_zero_leaked("Native MySQL repair_view must not leak cursors")

            manager.drop_view(view_name)
            tracked.assert_zero_leaked("Native MySQL drop_view must not leak cursors")

            # 2. Pre-existing table collision: verify pre-existing table is NOT dropped
            collide_tbl = f"{manager.object_prefix}store_{prefix}mv_col"
            col_view = f"{prefix}mv_col"
            cur = tracked.cursor()
            try:
                cur.execute(f"DROP TABLE IF EXISTS {collide_tbl};")
                cur.execute(f"CREATE TABLE {collide_tbl} (id INT PRIMARY KEY, data_str VARCHAR(64));")
                cur.execute(f"INSERT INTO {collide_tbl} VALUES (99, 'KEEP_SAFE');")
                tracked.commit()
            finally:
                cur.close()
            tracked.reset_tracking()

            col_view_sql = f"CREATE VIEW {col_view} AS SELECT grp, COUNT(*) AS c FROM {base_tbl} GROUP BY grp;"
            # Force backing table name to collide
            # Registration will attempt CREATE TABLE collide_tbl, which already exists!
            # Using an interceptor to simulate duplicate table error on MySQL:
            def col_fail_cursor(conn: Any, *a: Any, **kw: Any) -> Any:
                real_c = conn.cursor(*a, **kw)

                class Wrap(CursorTrackingWrapper):
                    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                        if "CREATE TABLE" in sql and collide_tbl in sql:
                            raise pymysql.err.OperationalError(1050, f"Table '{collide_tbl}' already exists")
                        return self._real_cursor.execute(sql, *args, **kwargs)

                return Wrap(real_c, tracked)

            tracked.cursor_factory = col_fail_cursor
            manager_col = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
            with self.assertRaises(pymysql.err.OperationalError):
                manager_col.register_view(
                    col_view_sql,
                    populate=False,
                    incremental_backfill=False,
                    batch_size=500,
                )
            tracked.cursor_factory = None

            cur = tracked.cursor()
            try:
                cur.execute(f"SELECT data_str FROM {collide_tbl} WHERE id = 99;")
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "KEEP_SAFE")
                cur.execute(f"DROP TABLE IF EXISTS {collide_tbl};")
                tracked.commit()
            finally:
                cur.close()
            tracked.assert_zero_leaked("Zero leaks after collision test")
        finally:
            cur = tracked.cursor()
            try:
                cur.execute(f"DROP VIEW IF EXISTS {view_name};")
                cur.execute(f"DROP TABLE IF EXISTS {backing_tbl};")
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
                tracked.commit()
            finally:
                cur.close()
            tracked.close()

    def test_native_postgres_zero_leaks(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; skipping live PostgreSQL safety test")

        import psycopg2
        raw_conn = psycopg2.connect(dsn)
        tracked = CursorTrackingConnection(raw_conn)
        manager = MaterializedViewManager(tracked, dialect=Dialect.POSTGRES, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

        prefix = "t_p1_pg_"
        base_tbl = f"{prefix}items"
        view_name = f"{prefix}mv_items"
        backing_tbl = f"{manager.object_prefix}store_{view_name}"

        cur = tracked.cursor()
        try:
            cur.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {backing_tbl} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
            cur.execute(f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, grp TEXT, val INT);")
            cur.execute(f"INSERT INTO {base_tbl} VALUES (1, 'x', 10), (2, 'x', 20);")
            tracked.commit()
        finally:
            cur.close()
        tracked.reset_tracking()

        try:
            view_sql = f"CREATE VIEW {view_name} AS SELECT grp, SUM(val) as tot FROM {base_tbl} GROUP BY grp;"
            plan = manager.register_view(view_sql, populate=True)
            self.assertEqual(plan.view_name, view_name)
            tracked.assert_zero_leaked("Native PostgreSQL register_view must not leak cursors")

            cat = manager.get_catalog()
            self.assertIn(view_name, cat)
            tracked.assert_zero_leaked("Native PostgreSQL get_catalog must not leak cursors")

            drift = manager.check_schema_drift()
            self.assertEqual(drift.get(view_name), "ok")
            tracked.assert_zero_leaked("Native PostgreSQL check_schema_drift must not leak cursors")

            repair = manager.repair_view(view_name)
            tracked.assert_zero_leaked("Native PostgreSQL repair_view must not leak cursors")

            manager.drop_view(view_name)
            tracked.assert_zero_leaked("Native PostgreSQL drop_view must not leak cursors")
        finally:
            cur = tracked.cursor()
            try:
                cur.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {backing_tbl} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
                tracked.commit()
            finally:
                cur.close()
            tracked.close()


class TestFkCascadeDriftAndFailClosedIntrospection(unittest.TestCase):
    """Verify fail-closed FK metadata introspection and FK-cascade drift detection."""

    def test_mysql_fk_introspection_fails_closed_on_metadata_error(self) -> None:
        from matview import introspect_database_schemas

        class FailingFkCursor:
            def execute(self, sql: str) -> None:
                if "referential_constraints" in sql.lower():
                    raise RuntimeError("SELECT command denied for referential_constraints")

            def fetchall(self) -> List[Tuple[Any, ...]]:
                return []

            def close(self) -> None:
                pass

        class FailingFkConn:
            def cursor(self) -> FailingFkCursor:
                return FailingFkCursor()

        with self.assertRaisesRegex(RuntimeError, "referential_constraints"):
            introspect_database_schemas(FailingFkConn(), dialect=Dialect.MYSQL)

    def test_postgres_fk_introspection_fails_closed_and_decodes_pg_constraint(self) -> None:
        from matview import introspect_database_schemas

        class FailingPgCursor:
            def __init__(self) -> None:
                self.sql = ""

            def execute(self, sql: str) -> None:
                self.sql = sql
                if "pg_constraint" in sql.lower():
                    raise RuntimeError("permission denied for pg_constraint")

            def fetchall(self) -> List[Tuple[Any, ...]]:
                return []

            def close(self) -> None:
                pass

        class FailingPgConn:
            def cursor(self) -> FailingPgCursor:
                return FailingPgCursor()

        with self.assertRaisesRegex(RuntimeError, "pg_constraint"):
            introspect_database_schemas(FailingPgConn(), dialect=Dialect.POSTGRES)

        class DecodingPgCursor:
            def __init__(self) -> None:
                self.sql = ""

            def execute(self, sql: str) -> None:
                self.sql = sql

            def fetchall(self) -> List[Tuple[Any, ...]]:
                low = self.sql.lower()
                if "table_constraints" in low and "key_column_usage" in low:
                    return [("child", "id")]
                if "pg_constraint" in low:
                    return [
                        ("child", b"c", b"n"),
                        ("child", "d", "a"),
                        ("child", "c", "r"),
                    ]
                if "information_schema.columns" in low:
                    return [
                        ("child", "id", "integer", "NO", None),
                        ("child", "pid", "integer", "YES", None),
                    ]
                return []

            def close(self) -> None:
                pass

        class DecodingPgConn:
            def cursor(self) -> DecodingPgCursor:
                return DecodingPgCursor()

        schemas = introspect_database_schemas(DecodingPgConn(), dialect=Dialect.POSTGRES)
        self.assertEqual(
            schemas["child"].fk_cascade_actions,
            ("ON DELETE CASCADE", "ON DELETE SET DEFAULT", "ON UPDATE SET NULL"),
        )

    def test_sqlite_fk_introspection_fails_closed_and_escapes_table_names(self) -> None:
        from matview import introspect_sqlite_schemas

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute('CREATE TABLE parent (pid INTEGER PRIMARY KEY);')
            conn.execute(
                'CREATE TABLE "child""tbl" ('
                'id INTEGER PRIMARY KEY, '
                'pid INTEGER REFERENCES parent(pid) ON DELETE CASCADE ON UPDATE SET NULL'
                ');'
            )
            schemas = introspect_sqlite_schemas(conn)
            self.assertIn('child"tbl', schemas)
            self.assertEqual(
                schemas['child"tbl'].fk_cascade_actions,
                ("ON DELETE CASCADE", "ON UPDATE SET NULL"),
            )

            class FailingPragmaCursor:
                def __init__(self, inner: sqlite3.Cursor) -> None:
                    self._inner = inner

                def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                    if "foreign_key_list" in sql.lower():
                        raise sqlite3.OperationalError("simulated foreign_key_list error")
                    return self._inner.execute(sql, *args, **kwargs)

                def fetchall(self) -> List[Tuple[Any, ...]]:
                    return self._inner.fetchall()

                def close(self) -> None:
                    self._inner.close()

            class FailingPragmaConn:
                def cursor(self) -> FailingPragmaCursor:
                    return FailingPragmaCursor(conn.cursor())

            with self.assertRaisesRegex(sqlite3.OperationalError, "foreign_key_list"):
                introspect_sqlite_schemas(FailingPragmaConn())

            # Also verify fail-closed through introspect_database_schemas(sqlite_conn)
            # (SQLAlchemy inspector path must not swallow introspect_sqlite_schemas errors).
            from matview import introspect_database_schemas

            def _deny_fk_pragma(action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
                if action == sqlite3.SQLITE_PRAGMA and str(arg1 or "").lower() == "foreign_key_list":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            conn.set_authorizer(_deny_fk_pragma)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    introspect_database_schemas(conn, dialect=Dialect.SQLITE)
            finally:
                conn.set_authorizer(None)
        finally:
            conn.close()

    def test_fk_cascade_drift_signature_and_mysql_limitation_reevaluation(self) -> None:
        from dataclasses import replace
        from matview import EngineLimitationError

        # 1. SQLite: out-of-band recreation adding ON DELETE CASCADE triggers drift
        sq_conn = sqlite3.connect(":memory:")
        try:
            sq_conn.execute("CREATE TABLE p (pid INTEGER PRIMARY KEY);")
            sq_conn.execute("CREATE TABLE c (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, pid INTEGER);")
            sq_conn.execute("INSERT INTO p VALUES (1), (2);")
            sq_conn.execute("INSERT INTO c VALUES (1, 10, 5, 1), (2, 10, 7, 2);")
            sq_conn.commit()
            mgr = MaterializedViewManager(sq_conn, dialect=Dialect.SQLITE)
            mgr.register_view("CREATE VIEW v_c AS SELECT g, COUNT(*) AS cnt, SUM(v) AS s FROM c GROUP BY g;")
            self.assertEqual(mgr.check_schema_drift(), {"v_c": "ok"})

            # Recreate c with ON DELETE CASCADE (triggers also recreated by repair_view)
            sq_conn.execute("DROP TABLE c;")
            sq_conn.execute(
                "CREATE TABLE c (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, "
                "pid INTEGER REFERENCES p(pid) ON DELETE CASCADE);"
            )
            sq_conn.execute("INSERT INTO c VALUES (1, 10, 5, 1), (2, 10, 7, 2);")
            sq_conn.commit()
            self.assertEqual(mgr.check_schema_drift(), {"v_c": "drifted"})
            self.assertEqual(mgr.repair_view(), {"v_c": "rebuilt"})
            self.assertEqual(mgr.check_schema_drift(), {"v_c": "ok"})
        finally:
            sq_conn.close()

        # 2. MySQL (transpiled): adding cascading FK after registration reports
        #    "invalid" when unaccepted (and refuses repair_view/sync_views) and
        #    "drifted" when MYSQL_FK_CASCADE_NO_TRIGGER was accepted.
        from test_matview_sync import TranspiledConnectionAdapter

        for accept_fk, expected_drift in ((False, "invalid"), (True, "drifted")):
            tconn = TranspiledConnectionAdapter(Dialect.MYSQL)
            try:
                cur = tconn.cursor()
                cur.execute("CREATE TABLE p (pid INT PRIMARY KEY, label VARCHAR(16));")
                cur.execute("CREATE TABLE c (id INT PRIMARY KEY, g INT, v INT, pid INT);")
                cur.execute("INSERT INTO p VALUES (1, 'a'), (2, 'b');")
                cur.execute("INSERT INTO c VALUES (1, 10, 5, 1), (2, 10, 7, 2);")
                tconn.commit()
                mgr = MaterializedViewManager(
                    tconn,
                    dialect=Dialect.MYSQL,
                    ignore_limitations=False,
                    accept_limitations=(
                        ("MYSQL_NO_TRUNCATE_TRIGGER", "MYSQL_FK_CASCADE_NO_TRIGGER")
                        if accept_fk
                        else ("MYSQL_NO_TRUNCATE_TRIGGER",)
                    ),
                )
                view_sql = "CREATE VIEW v_late AS SELECT g, COUNT(*) AS cnt, SUM(v) AS s FROM c GROUP BY g;"
                mgr.register_view(view_sql)
                self.assertEqual(mgr.check_schema_drift(), {"v_late": "ok"})
                tconn.commit()

                base_schemas = mgr.fetch_schema()
                mutated = dict(base_schemas)
                mutated["c"] = replace(base_schemas["c"], fk_cascade_actions=("ON DELETE CASCADE",))
                mgr.fetch_schema = lambda extra_schema_sql=None, _m=mutated: _m

                self.assertEqual(mgr.check_schema_drift(), {"v_late": expected_drift})
                tconn.commit()
                if not accept_fk:
                    with self.assertRaises(EngineLimitationError) as ctx:
                        mgr.repair_view()
                    self.assertIn("MYSQL_FK_CASCADE_NO_TRIGGER", str(ctx.exception))
                    tconn.rollback()
                    with self.assertRaises(EngineLimitationError) as ctx2:
                        mgr.sync_views(view_sql)
                    self.assertIn("MYSQL_FK_CASCADE_NO_TRIGGER", str(ctx2.exception))
                    tconn.rollback()
            finally:
                tconn.close()

    def test_native_postgres_fk_cascade_drift_and_tier2_strategy_rebuild(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; skipping native PostgreSQL FK drift test")

        import psycopg2
        from matview import MaintenanceStrategy, introspect_database_schemas

        conn = psycopg2.connect(dsn)
        prefix = "t_fkd_pg_"
        p_tbl = f"{prefix}p"
        c_tbl = f"{prefix}c"
        v_c = f"{prefix}v_c"
        v_lj = f"{prefix}v_lj"
        mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        cur = conn.cursor()
        try:
            for obj in (v_c, v_lj):
                try:
                    mgr.drop_view(obj)
                except Exception:
                    conn.rollback()
            cur.execute(f"DROP TABLE IF EXISTS {c_tbl} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {p_tbl} CASCADE;")
            cur.execute(f"CREATE TABLE {p_tbl} (pid INT PRIMARY KEY);")
            cur.execute(f"CREATE TABLE {c_tbl} (id INT PRIMARY KEY, g INT, v BIGINT, pid INT);")
            cur.execute(f"INSERT INTO {p_tbl} VALUES (1), (2), (3);")
            cur.execute(f"INSERT INTO {c_tbl} VALUES (1, 10, 5, 1), (2, 10, 7, 2), (3, 20, 9, 2);")
            conn.commit()

            p1 = mgr.register_view(f"CREATE VIEW {v_c} AS SELECT g, COUNT(*) AS cnt, SUM(v) AS s FROM {c_tbl} GROUP BY g;")
            p2 = mgr.register_view(
                f"CREATE VIEW {v_lj} AS SELECT p.pid, COUNT(c.id) AS cnt FROM {p_tbl} p "
                f"LEFT JOIN {c_tbl} c ON p.pid = c.pid GROUP BY p.pid;"
            )
            self.assertEqual(p1.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
            self.assertEqual(p2.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
            before_lj_ddls = list(mgr.get_catalog()[v_lj]["trigger_ddls"])
            self.assertEqual(mgr.check_schema_drift().get(v_c), "ok")
            self.assertEqual(mgr.check_schema_drift().get(v_lj), "ok")
            conn.commit()

            # Add cascading FK out-of-band (including PG 15+ column-scoped ON DELETE SET NULL (pid)
            # test afterwards to verify SQLAlchemy inspector preserves column-list FK actions).
            cur.execute(
                f"ALTER TABLE {c_tbl} ADD CONSTRAINT {prefix}fk FOREIGN KEY (pid) "
                f"REFERENCES {p_tbl} (pid) ON DELETE CASCADE ON UPDATE SET NULL;"
            )
            conn.commit()

            pulled = introspect_database_schemas(conn, dialect=Dialect.POSTGRES)
            self.assertEqual(
                pulled[c_tbl].fk_cascade_actions,
                ("ON DELETE CASCADE", "ON UPDATE SET NULL"),
            )
            drift = mgr.check_schema_drift()
            conn.commit()
            self.assertEqual(drift.get(v_c), "drifted")
            self.assertEqual(drift.get(v_lj), "drifted")

            self.assertEqual(mgr.repair_view(v_c), {v_c: "rebuilt"})
            self.assertEqual(mgr.repair_view(v_lj), {v_lj: "rebuilt"})
            self.assertEqual(mgr.check_schema_drift().get(v_c), "ok")
            self.assertEqual(mgr.check_schema_drift().get(v_lj), "ok")
            after_lj_ddls = list(mgr.get_catalog()[v_lj]["trigger_ddls"])
            self.assertNotEqual(before_lj_ddls, after_lj_ddls)
            conn.commit()

            # Verify live maintenance stays exact across DELETE CASCADE and UPDATE SET NULL
            cur.execute(f"DELETE FROM {p_tbl} WHERE pid = 2;")
            cur.execute(f"UPDATE {p_tbl} SET pid = 10 WHERE pid = 1;")
            conn.commit()
            cur.execute(f"SELECT g, cnt, s FROM {v_c} ORDER BY g;")
            got_c = cur.fetchall()
            cur.execute(f"SELECT g, COUNT(*), SUM(v) FROM {c_tbl} GROUP BY g ORDER BY g;")
            self.assertEqual(got_c, cur.fetchall())
            cur.execute(f"SELECT pid, cnt FROM {v_lj} ORDER BY pid;")
            got_lj = cur.fetchall()
            cur.execute(
                f"SELECT p.pid, COUNT(c.id) FROM {p_tbl} p LEFT JOIN {c_tbl} c ON p.pid = c.pid GROUP BY p.pid ORDER BY p.pid;"
            )
            self.assertEqual(got_lj, cur.fetchall())

            # PostgreSQL 15+ column-scoped ON DELETE SET NULL (pid) must also be introspected
            # through the SQLAlchemy inspector path (not dropped as an unrecognized action).
            if mgr.engine_version.version >= (15, 0):
                cur.execute(f"ALTER TABLE {c_tbl} DROP CONSTRAINT {prefix}fk;")
                cur.execute(
                    f"ALTER TABLE {c_tbl} ADD CONSTRAINT {prefix}fk_col FOREIGN KEY (pid) "
                    f"REFERENCES {p_tbl} (pid) ON DELETE SET NULL (pid);"
                )
                conn.commit()
                pulled_col = introspect_database_schemas(conn, dialect=Dialect.POSTGRES)
                self.assertEqual(pulled_col[c_tbl].fk_cascade_actions, ("ON DELETE SET NULL",))
                self.assertEqual(mgr.check_schema_drift().get(v_c), "drifted")
                conn.commit()
        finally:
            conn.rollback()
            try:
                mgr.drop_view(v_c)
            except Exception:
                conn.rollback()
            try:
                mgr.drop_view(v_lj)
            except Exception:
                conn.rollback()
            cur.execute(f"DROP TABLE IF EXISTS {c_tbl} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {p_tbl} CASCADE;")
            conn.commit()
            cur.close()
            conn.close()

    def test_native_mysql_fk_cascade_added_after_registration_variants(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; skipping native MySQL/MariaDB FK drift test")

        import pymysql
        from urllib.parse import urlparse
        from matview import EngineLimitationError

        parsed = urlparse(dsn)
        conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/"),
            autocommit=False,
        )
        prefix = "t_fkd_my_"
        p_tbl = f"{prefix}p"
        c_tbl = f"{prefix}c"
        v_def = f"{prefix}v_def"
        v_acc = f"{prefix}v_acc"
        mgr_def = MaterializedViewManager(
            conn,
            dialect=Dialect.MYSQL,
            ignore_limitations=False,
            accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
        )
        mgr_acc = MaterializedViewManager(
            conn,
            dialect=Dialect.MYSQL,
            ignore_limitations=False,
            accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER", "MYSQL_FK_CASCADE_NO_TRIGGER"),
        )
        cur = conn.cursor()
        try:
            for vn in (v_def, v_acc):
                try:
                    mgr_def.drop_view(vn)
                except Exception:
                    conn.rollback()
            cur.execute(f"DROP TABLE IF EXISTS {c_tbl};")
            cur.execute(f"DROP TABLE IF EXISTS {p_tbl};")
            cur.execute(f"CREATE TABLE {p_tbl} (pid INT PRIMARY KEY) ENGINE=InnoDB;")
            cur.execute(f"CREATE TABLE {c_tbl} (id INT PRIMARY KEY, g INT, v BIGINT, pid INT) ENGINE=InnoDB;")
            cur.execute(f"INSERT INTO {p_tbl} VALUES (1), (2);")
            cur.execute(f"INSERT INTO {c_tbl} VALUES (1, 10, 5, 1), (2, 10, 7, 2);")
            conn.commit()

            v_def_sql = f"CREATE VIEW {v_def} AS SELECT g, COUNT(*) AS cnt, SUM(v) AS s FROM {c_tbl} GROUP BY g;"
            mgr_def.register_view(v_def_sql)
            mgr_acc.register_view(f"CREATE VIEW {v_acc} AS SELECT g, COUNT(*) AS cnt FROM {c_tbl} GROUP BY g;")
            self.assertEqual(mgr_def.check_schema_drift().get(v_def), "ok")
            self.assertEqual(mgr_acc.check_schema_drift().get(v_acc), "ok")
            conn.commit()

            # Add ON UPDATE CASCADE + ON DELETE SET NULL out-of-band
            cur.execute(
                f"ALTER TABLE {c_tbl} ADD CONSTRAINT {prefix}fk FOREIGN KEY (pid) "
                f"REFERENCES {p_tbl} (pid) ON UPDATE CASCADE ON DELETE SET NULL;"
            )
            conn.commit()

            self.assertEqual(mgr_def.check_schema_drift().get(v_def), "invalid")
            self.assertEqual(mgr_acc.check_schema_drift().get(v_acc), "drifted")
            conn.commit()
            with self.assertRaisesRegex(EngineLimitationError, "MYSQL_FK_CASCADE_NO_TRIGGER"):
                mgr_def.repair_view(v_def)
            conn.rollback()
            with self.assertRaisesRegex(EngineLimitationError, "MYSQL_FK_CASCADE_NO_TRIGGER"):
                mgr_def.sync_views(v_def_sql)
            conn.rollback()

            cur.execute(f"ALTER TABLE {c_tbl} DROP FOREIGN KEY {prefix}fk;")
            conn.commit()
            self.assertEqual(mgr_def.check_schema_drift().get(v_def), "ok")
            self.assertEqual(mgr_acc.check_schema_drift().get(v_acc), "ok")
            conn.commit()
            self.assertEqual(mgr_def.repair_view(v_def), {v_def: "repaired"})
        finally:
            conn.rollback()
            for vn in (v_def, v_acc):
                try:
                    mgr_def.drop_view(vn)
                except Exception:
                    conn.rollback()
            cur.execute(f"DROP TABLE IF EXISTS {c_tbl};")
            cur.execute(f"DROP TABLE IF EXISTS {p_tbl};")
            conn.commit()
            cur.close()
            conn.close()


class TestPostgresTriggerAndFunctionDriftParsers(unittest.TestCase):
    """Offline tests for PostgreSQL function/trigger DDL and search_path parsers."""

    def test_parse_expected_functions_all_templates(self) -> None:
        ddls = [
            (
                "CREATE OR REPLACE FUNCTION __matview_fn_sync_v__ins()\n"
                "RETURNS TRIGGER AS $$\n"
                "BEGIN\n"
                "    -- literal mentioning SET search_path inside body\n"
                "    RETURN NULL;\n"
                "END;\n"
                "$$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
            ),
            (
                "CREATE OR REPLACE FUNCTION __matview_key_v(v anyelement) RETURNS text\n"
                "LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE SET search_path FROM CURRENT AS $mv_key$\n"
                "BEGIN\n"
                "    RETURN v::pg_catalog.text;\n"
                "END;\n"
                "$mv_key$;"
            ),
            (
                'CREATE OR REPLACE FUNCTION "__matview_fn_readonly_strict immutable ""View"""()\n'
                "RETURNS TRIGGER AS $$\n"
                "BEGIN\n"
                "    RAISE EXCEPTION 'SET search_path read-only';\n"
                "END;\n"
                "$$ LANGUAGE plpgsql;"
            ),
            (
                "CREATE OR REPLACE FUNCTION __matview_fn_s3_j_reconcile()\n"
                "RETURNS VOID AS $$\n"
                "BEGIN\n"
                "    RETURN;\n"
                "END;\n"
                "$$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
            ),
        ]
        parsed = MaterializedViewManager._pg_parse_expected_functions(ddls)
        self.assertEqual(len(parsed), 4)
        # (proname, arg_type, ret_type, prosrc, requires_search_path, is_strict, is_immutable)
        self.assertEqual(parsed[0][0], "__matview_fn_sync_v__ins")
        self.assertIsNone(parsed[0][1])
        self.assertEqual(parsed[0][2], "trigger")
        self.assertIn("RETURN NULL;", parsed[0][3])
        self.assertTrue(parsed[0][4])
        self.assertFalse(parsed[0][5])
        self.assertFalse(parsed[0][6])

        self.assertEqual(parsed[1][0], "__matview_key_v")
        self.assertEqual(parsed[1][1], "anyelement")
        self.assertEqual(parsed[1][2], "text")
        self.assertIn("RETURN v::pg_catalog.text;", parsed[1][3])
        self.assertTrue(parsed[1][4])
        self.assertTrue(parsed[1][5])
        self.assertTrue(parsed[1][6])

        # Quoted identifier preserves case/quotes; words like strict/immutable in
        # the function name and 'SET search_path' inside body must NOT trigger flags.
        self.assertEqual(parsed[2][0], '__matview_fn_readonly_strict immutable "View"')
        self.assertIsNone(parsed[2][1])
        self.assertEqual(parsed[2][2], "trigger")
        self.assertFalse(parsed[2][4])
        self.assertFalse(parsed[2][5])
        self.assertFalse(parsed[2][6])

        self.assertEqual(parsed[3][0], "__matview_fn_s3_j_reconcile")
        self.assertIsNone(parsed[3][1])
        self.assertEqual(parsed[3][2], "void")
        self.assertTrue(parsed[3][4])
        self.assertFalse(parsed[3][5])
        self.assertFalse(parsed[3][6])

    def test_parse_expected_triggers_all_kinds_and_tgtype(self) -> None:
        ddls = [
            (
                '  CREATE TRIGGER __matview_trg_v_ins\n'
                'AFTER INSERT ON "orders"\n'
                'REFERENCING NEW TABLE AS __mv_new\n'
                'FOR EACH STATEMENT EXECUTE FUNCTION __matview_fn_sync_v_ins();'
            ),
            (
                'CREATE TRIGGER __matview_trg_v_upd\n'
                'AFTER UPDATE ON "s1"."orders"\n'
                'REFERENCING OLD TABLE AS __mv_old NEW TABLE AS __mv_new\n'
                'FOR EACH STATEMENT EXECUTE FUNCTION __matview_fn_sync_v_upd();'
            ),
            (
                'CREATE TRIGGER __matview_trg_v_mixed1\n'
                'AFTER DELETE ON s1."My Table"\n'
                'REFERENCING OLD TABLE AS __mv_old\n'
                'FOR EACH STATEMENT EXECUTE FUNCTION __matview_fn_sync_v_del();'
            ),
            (
                'CREATE TRIGGER __matview_trg_v_mixed2\n'
                'AFTER TRUNCATE ON "My Schema".orders\n'
                'FOR EACH STATEMENT EXECUTE FUNCTION __matview_fn_sync_v_trunc();'
            ),
            (
                'CREATE TRIGGER __matview_trg_v_bkt\n'
                'BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ON "orders"\n'
                'FOR EACH STATEMENT EXECUTE FUNCTION __matview_fn_bkt_v();'
            ),
            (
                'CREATE TRIGGER "__matview_trg_My View__readonly"\n'
                'INSTEAD OF INSERT OR UPDATE OR DELETE ON "My View"\n'
                'FOR EACH ROW EXECUTE FUNCTION "__matview_fn_readonly_My View"();'
            ),
            # Function body containing literal CREATE TRIGGER text must NOT be parsed as a trigger.
            (
                "CREATE OR REPLACE FUNCTION __matview_fn_literal()\n"
                "RETURNS TRIGGER AS $$\n"
                "BEGIN\n"
                "PERFORM '\n"
                'CREATE TRIGGER fake_trg AFTER INSERT ON "orders" FOR EACH STATEMENT EXECUTE FUNCTION fake_fn();\n'
                "';\n"
                "RETURN NULL;\n"
                "END;\n"
                "$$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
            ),
        ]
        parsed = MaterializedViewManager._pg_parse_expected_triggers(ddls)
        self.assertEqual(
            parsed,
            [
                ("__matview_trg_v_ins", '"orders"', "__matview_fn_sync_v_ins", 4, "", "__mv_new"),
                ("__matview_trg_v_upd", '"s1"."orders"', "__matview_fn_sync_v_upd", 16, "__mv_old", "__mv_new"),
                ("__matview_trg_v_mixed1", 's1."My Table"', "__matview_fn_sync_v_del", 8, "__mv_old", ""),
                ("__matview_trg_v_mixed2", '"My Schema".orders', "__matview_fn_sync_v_trunc", 32, "", ""),
                ("__matview_trg_v_bkt", '"orders"', "__matview_fn_bkt_v", 62, "", ""),
                ("__matview_trg_My View__readonly", '"My View"', "__matview_fn_readonly_My View", 93, "", ""),
            ],
        )

    def test_search_path_setting_validation(self) -> None:
        check = MaterializedViewManager._pg_search_path_setting_is_valid
        self.assertTrue(check("mv_schema, public", "mv_schema", current_user="postgres"))
        self.assertTrue(check('"$user", public', "public", current_user="postgres"))
        # "$user" only resolves schema_name when schema_name == current_user.
        self.assertTrue(check('"$user", public', "alice", current_user="alice"))
        self.assertFalse(check('"$user", public', "mv_schema", current_user="postgres"))
        self.assertFalse(check('"$user", public', "alice"))
        # Embedded double quotes (both single " and consecutive "") must unescape exactly once.
        self.assertTrue(check('"My ""Schema""", public', 'My "Schema"', current_user="postgres"))
        self.assertTrue(check('"My """"Schema""""", public', 'My ""Schema""', current_user="postgres"))
        self.assertFalse(check('"My """"Schema""""", public', 'My "Schema"', current_user="postgres"))
        self.assertFalse(check(None, "public", current_user="postgres"))
        self.assertFalse(check("", "public", current_user="postgres"))
        self.assertFalse(check("   ", "public", current_user="postgres"))
        self.assertFalse(check("pg_catalog", "mv_schema", current_user="postgres"))
        self.assertFalse(check("other_schema, public", "mv_schema", current_user="postgres"))


class TestPlainViewAsSource(unittest.TestCase):
    """Tests for plain (non-materialized) SQL views used as materialized view sources."""

    def test_sqlite_plain_view_simple_unwraps_to_tier1_and_detects_drift(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, v INTEGER NOT NULL);")
            conn.executemany(
                "INSERT INTO t VALUES (?, ?, ?)",
                [(1, "A", 10), (2, "A", 20), (3, "B", 30), (4, None, 99)],
            )
            conn.execute("CREATE VIEW pv_src AS SELECT id, g, v FROM t WHERE g IS NOT NULL;")
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            mv_sql = "CREATE VIEW mv_over_pv AS SELECT g, COUNT(*) AS c, SUM(v) AS s FROM pv_src GROUP BY g;"
            plan = mgr.register_view(mv_sql)
            self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
            self.assertEqual(plan.view_spec.referenced_tables, ("t",))
            self.assertEqual(
                conn.execute("SELECT g, c, s FROM mv_over_pv ORDER BY g").fetchall(),
                [("A", 2, 30), ("B", 1, 30)],
            )

            # Re-registering same definition is a no-op; drift/diff/sync agree.
            mgr.register_view(mv_sql)
            self.assertEqual(mgr.check_schema_drift(), {"mv_over_pv": "ok"})
            diff = mgr.diff_views(mv_sql)
            self.assertEqual(diff["mv_over_pv"].status, "NOOP")
            self.assertEqual(mgr.sync_views(mv_sql), {"mv_over_pv": "unchanged"})

            # Base table mutations maintain exactness through the inlined plain view.
            conn.execute("INSERT INTO t VALUES (5, 'A', 5), (6, NULL, 100);")
            conn.execute("UPDATE t SET v = v + 10 WHERE id = 2;")
            conn.execute("DELETE FROM t WHERE id = 3;")
            conn.commit()
            self.assertEqual(
                conn.execute("SELECT g, c, s FROM mv_over_pv ORDER BY g").fetchall(),
                conn.execute("SELECT g, COUNT(*), SUM(v) FROM pv_src GROUP BY g ORDER BY g").fetchall(),
            )

            # Out-of-band plain-view change -> check_schema_drift() == "drifted", repair_view() rebuilds.
            conn.execute("DROP VIEW pv_src;")
            conn.execute("CREATE VIEW pv_src AS SELECT id, g, v * 2 AS v FROM t WHERE g IS NOT NULL AND v > 10;")
            conn.commit()
            self.assertEqual(mgr.check_schema_drift(), {"mv_over_pv": "drifted"})
            self.assertEqual(mgr.repair_view("mv_over_pv"), {"mv_over_pv": "rebuilt"})
            self.assertEqual(mgr.check_schema_drift(), {"mv_over_pv": "ok"})
            self.assertEqual(
                conn.execute("SELECT g, c, s FROM mv_over_pv ORDER BY g").fetchall(),
                conn.execute("SELECT g, COUNT(*), SUM(v) FROM pv_src GROUP BY g ORDER BY g").fetchall(),
            )

            # Out-of-band plain-view drop -> check_schema_drift() == "invalid".
            conn.execute("DROP VIEW pv_src;")
            conn.commit()
            self.assertEqual(mgr.check_schema_drift(), {"mv_over_pv": "invalid"})
        finally:
            conn.close()

    def test_sqlite_chained_diamond_joined_grouped_cte_and_explicit_columns(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, d_id INTEGER NOT NULL, g TEXT NOT NULL, v INTEGER NOT NULL);")
            conn.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
            conn.executemany("INSERT INTO d VALUES (?, ?)", [(1, "D1"), (2, "D2")])
            conn.executemany(
                "INSERT INTO t VALUES (?, ?, ?, ?)",
                [(1, 1, "A", 10), (2, 1, "A", 20), (3, 2, "B", 30)],
            )
            # Chained views v2 -> v1 -> t
            conn.execute("CREATE VIEW v1 AS SELECT id, d_id, g, v FROM t WHERE v >= 10;")
            conn.execute("CREATE VIEW v2 AS SELECT id, d_id, g, v * 3 AS v3 FROM v1;")
            # Explicit column list (projection, SELECT *, and UNION ALL)
            conn.execute("CREATE VIEW v_cols(k, amt) AS SELECT g, v FROM t;")
            conn.execute("CREATE VIEW v_star(x_id, x_did, x_g, x_v) AS SELECT * FROM t;")
            conn.execute("CREATE VIEW v_ucols(k, amt) AS SELECT g, v FROM t UNION ALL SELECT name, id FROM d;")
            # Diamond DAG: v_left + v_right both read v1
            conn.execute("CREATE VIEW v_left AS SELECT g, SUM(v) AS sl FROM v1 GROUP BY g;")
            conn.execute("CREATE VIEW v_right AS SELECT g, COUNT(*) AS cr FROM v1 GROUP BY g;")
            # Plain view with CTE and quoted/special-character plain view name
            conn.execute("CREATE VIEW v_cte AS WITH base AS (SELECT g, v FROM t WHERE v > 0) SELECT g, SUM(v) AS sv FROM base GROUP BY g;")
            conn.execute('CREATE VIEW "v\'s src" AS SELECT id, g, v FROM t WHERE v > 5;')
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            mvs = [
                ("mv_chain", "SELECT g, SUM(v3) AS s FROM v2 GROUP BY g", "SELECT g, SUM(v3) FROM v2 GROUP BY g ORDER BY g"),
                ("mv_cols", "SELECT k, SUM(amt) AS s FROM v_cols GROUP BY k", "SELECT k, SUM(amt) FROM v_cols GROUP BY k ORDER BY k"),
                ("mv_star", "SELECT x_g, SUM(x_v) AS s FROM v_star GROUP BY x_g", "SELECT x_g, SUM(x_v) FROM v_star GROUP BY x_g ORDER BY x_g"),
                ("mv_ucols", "SELECT k, SUM(amt) AS s FROM v_ucols GROUP BY k", "SELECT k, SUM(amt) FROM v_ucols GROUP BY k ORDER BY k"),
                ("mv_join_base", "SELECT d.name, SUM(v1.v) AS s FROM v1 JOIN d ON d.id = v1.d_id GROUP BY d.name", "SELECT d.name, SUM(v1.v) FROM v1 JOIN d ON d.id = v1.d_id GROUP BY d.name ORDER BY d.name"),
                ("mv_diamond", "SELECT l.g, l.sl, r.cr FROM v_left l JOIN v_right r ON r.g = l.g", "SELECT l.g, l.sl, r.cr FROM v_left l JOIN v_right r ON r.g = l.g ORDER BY l.g"),
                ("mv_cte", "SELECT g, sv * 2 AS sv2 FROM v_cte WHERE sv >= 20", "SELECT g, sv * 2 FROM v_cte WHERE sv >= 20 ORDER BY g"),
                ("mv_in_subq", "SELECT id, name FROM d WHERE id IN (SELECT d_id FROM v1 WHERE v >= 20)", "SELECT id, name FROM d WHERE id IN (SELECT d_id FROM v1 WHERE v >= 20) ORDER BY id"),
                ("mv_quoted_pv", 'SELECT g, COUNT(*) AS c FROM "v\'s src" GROUP BY g', 'SELECT g, COUNT(*) FROM "v\'s src" GROUP BY g ORDER BY g'),
            ]
            for vn, sel, _ in mvs:
                mgr.register_view(f"CREATE VIEW {vn} AS {sel};")

            def _assert_all() -> None:
                for vn, _, live in mvs:
                    actual = conn.execute(f"SELECT * FROM {vn} ORDER BY 1").fetchall()
                    expected = conn.execute(live).fetchall()
                    self.assertEqual(actual, expected, f"mismatch on {vn}")

            _assert_all()
            conn.execute("INSERT INTO d VALUES (3, 'D3');")
            conn.execute("INSERT INTO t VALUES (4, 3, 'C', 15), (5, 1, 'A', 5);")
            conn.execute("UPDATE d SET name = 'D1_NEW' WHERE id = 1;")
            conn.execute("UPDATE t SET v = 25 WHERE id = 5;")
            conn.execute("DELETE FROM t WHERE id = 2;")
            conn.commit()
            _assert_all()

            # Altering transitive upstream view v1 drifts mv_chain, mv_diamond, mv_join_base, mv_in_subq.
            conn.execute("DROP VIEW v1;")
            conn.execute("CREATE VIEW v1 AS SELECT id, d_id, g, v + 1 AS v FROM t WHERE v >= 10;")
            conn.commit()
            drift = mgr.check_schema_drift()
            for drifted_mv in ("mv_chain", "mv_diamond", "mv_join_base", "mv_in_subq"):
                self.assertEqual(drift[drifted_mv], "drifted", f"{drifted_mv}: {drift}")
            self.assertEqual(drift["mv_cols"], "ok")
            mgr.repair_view()
            _assert_all()
        finally:
            conn.close()

    def test_cte_shadowing_cycle_depth_and_registered_mv_refusal(self) -> None:
        from matview.sql_frontend import inline_plain_views

        # 1. Cycle detection (mutual and self-cycle) raises ValueError.
        with self.assertRaisesRegex(ValueError, "Cyclic plain SQL view reference"):
            inline_plain_views(
                "SELECT * FROM v1",
                {"v1": "CREATE VIEW v1 AS SELECT * FROM v2", "v2": "CREATE VIEW v2 AS SELECT * FROM v1"},
                dialect=Dialect.SQLITE,
            )
        with self.assertRaisesRegex(ValueError, "Cyclic plain SQL view reference"):
            inline_plain_views(
                "SELECT * FROM self_v",
                {"self_v": "CREATE VIEW self_v AS SELECT * FROM self_v"},
                dialect=Dialect.SQLITE,
            )

        # 2. Nesting depth limit raises ValueError.
        deep_chain = {f"v{i}": f"CREATE VIEW v{i} AS SELECT * FROM v{i + 1}" for i in range(20)}
        deep_chain["v20"] = "CREATE VIEW v20 AS SELECT id FROM t"
        with self.assertRaisesRegex(ValueError, "nesting/expansion limit"):
            inline_plain_views("SELECT * FROM v0", deep_chain, dialect=Dialect.SQLITE, known_tables={"t"})

        # 3. CTE shadowing: CTE of the same name shadows plain view; non-recursive CTE body
        #    can read the plain view of the same name while the outer query reads the CTE.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, v INTEGER NOT NULL);")
            conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, "A", 10), (2, "B", 20)])
            conn.execute("CREATE VIEW pv AS SELECT g, v FROM t WHERE g = 'A';")
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            mgr.register_view(
                "CREATE VIEW mv_shadow AS WITH pv AS (SELECT g, v FROM t WHERE g = 'B') SELECT g, SUM(v) AS s FROM pv GROUP BY g;"
            )
            self.assertEqual(conn.execute("SELECT g, s FROM mv_shadow").fetchall(), [("B", 20)])

            mgr.register_view(
                "CREATE VIEW mv_self_cte AS WITH pv AS (SELECT g, v * 10 AS v FROM pv) SELECT g, SUM(v) AS s FROM pv GROUP BY g;"
            )
            self.assertEqual(conn.execute("SELECT g, s FROM mv_self_cte").fetchall(), [("A", 100)])

            # 4. Referencing another registered materialized view (directly or via a plain view) is refused.
            with self.assertRaisesRegex(ValueError, "does not reference any known base tables"):
                mgr.register_view("CREATE VIEW mv_over_mv AS SELECT g, s FROM mv_shadow;")
            conn.execute("CREATE VIEW pv_over_mv AS SELECT g, s FROM mv_shadow;")
            conn.commit()
            with self.assertRaisesRegex(ValueError, "does not reference any known base tables"):
                mgr.register_view("CREATE VIEW mv_indirect_mv AS SELECT g, s FROM pv_over_mv;")
        finally:
            conn.close()

    def test_offline_generate_materialized_view_and_mysql_db_qualifier_stripping(self) -> None:
        from matview import generate_materialized_view, generate_materialized_views

        schema_with_view = """
        CREATE TABLE orders (id BIGINT PRIMARY KEY, region TEXT NOT NULL, amount BIGINT NOT NULL);
        CREATE VIEW settled_orders AS SELECT id, region, amount FROM orders WHERE amount > 0;
        """
        for d in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            plan = generate_materialized_view(
                schema_with_view,
                "CREATE VIEW mv_sales AS SELECT region, SUM(amount) AS total FROM settled_orders GROUP BY region;",
                dialect=d,
                ignore_limitations=(d == Dialect.MYSQL),
            )
            self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA, d)
            self.assertEqual(plan.view_spec.referenced_tables, ("orders",), d)
            self.assertEqual(len(plan.view_spec.referenced_plain_views), 1, d)

        plans = generate_materialized_views(
            schema_with_view,
            "CREATE VIEW mv_a AS SELECT region, COUNT(*) AS c FROM settled_orders GROUP BY region;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)

        # MySQL information_schema.VIEWS emits `current_db`.`table`.`col` qualifiers;
        # stripping current_schema unwraps cleanly to Tier 1 ALGEBRAIC_DELTA while
        # preserving foreign-database qualifiers (which fail closed).
        mysql_view_def = (
            "select `mydb`.`orders`.`id` AS `id`,`mydb`.`orders`.`region` AS `region`,"
            "`mydb`.`orders`.`amount` AS `amount` from `mydb`.`orders` "
            "where (`mydb`.`orders`.`amount` > 0)"
        )
        plan_my = generate_materialized_view(
            "CREATE TABLE orders (id BIGINT PRIMARY KEY, region VARCHAR(32) NOT NULL, amount BIGINT NOT NULL);",
            "CREATE VIEW mv_my AS SELECT region, SUM(amount) AS total FROM settled_orders GROUP BY region;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
            plain_views={"settled_orders": mysql_view_def},
            current_schema="mydb",
        )
        self.assertEqual(plan_my.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertEqual(plan_my.view_spec.referenced_tables, ("orders",))

        # Foreign-database table inside plain view is not stripped and fails closed,
        # even when a same-named local table (`orders`) exists in the target schema.
        foreign_view_def = "select `otherdb`.`orders`.`id` AS `id` from `otherdb`.`orders`"
        with self.assertRaisesRegex(ValueError, "outside the current schema/database"):
            generate_materialized_view(
                "CREATE TABLE orders (id BIGINT PRIMARY KEY, region VARCHAR(32) NOT NULL, amount BIGINT NOT NULL);",
                "CREATE VIEW mv_bad AS SELECT id FROM foreign_v;",
                dialect=Dialect.MYSQL,
                ignore_limitations=True,
                plain_views={"foreign_v": foreign_view_def},
                current_schema="mydb",
            )

        # Passing a ViewSpec object directly to generate_materialized_view() also inlines plain views.
        from matview.sql_frontend import parse_plain_views, parse_view_definition

        spec = parse_view_definition(
            "CREATE VIEW mv_from_spec AS SELECT region, SUM(amount) AS total FROM settled_orders GROUP BY region;",
            dialect=Dialect.POSTGRES,
        )
        plan_spec = generate_materialized_view(
            schema_with_view,
            spec,
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(plan_spec.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertEqual(plan_spec.view_spec.referenced_tables, ("orders",))
        self.assertEqual(len(plan_spec.view_spec.referenced_plain_views), 1)

        # parse_plain_views() ignores PostgreSQL CREATE MATERIALIZED VIEW statements.
        parsed = parse_plain_views(
            "CREATE MATERIALIZED VIEW pg_mv AS SELECT id FROM orders;"
            "CREATE VIEW pv_ok AS SELECT id FROM orders;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(list(parsed.keys()), ["pv_ok"])

        # FROM pv AS x(i, r, amt) column-alias list: simple view unwraps with alias
        # mapping to Tier 1; grouped view retains AS x(r, tot) intact.
        plan_alias_list = generate_materialized_view(
            schema_with_view,
            "CREATE VIEW mv_col_alias AS SELECT r, SUM(amt) AS s FROM settled_orders AS x(i, r, amt) GROUP BY r;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(plan_alias_list.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertEqual(plan_alias_list.view_spec.referenced_tables, ("orders",))
        schema_with_grouped = schema_with_view + (
            "\nCREATE VIEW grouped_orders AS SELECT region, SUM(amount) AS tot FROM orders GROUP BY region;\n"
        )
        plan_grp_alias = generate_materialized_view(
            schema_with_grouped,
            "CREATE VIEW mv_grp_alias AS SELECT r, tot + 1 AS tot1 FROM grouped_orders AS x(r, tot);",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(plan_grp_alias.view_spec.referenced_tables, ("orders",))
        self.assertIn("AS x(r, tot)", plan_grp_alias.view_spec.raw_select_sql)

        # Chained flat plain views (including PG 15 style view-qualified columns like
        # `pv_base.amount * 2`) recursively unwrap to Tier 1 ALGEBRAIC_DELTA so exact
        # column types resolve cleanly against the underlying base table.
        plan_pg15_chain = generate_materialized_view(
            "CREATE TABLE orders (id BIGINT PRIMARY KEY, region TEXT, amount BIGINT NOT NULL);",
            "CREATE VIEW mv_chain_pg15 AS SELECT region, SUM(amt2) AS s FROM pv_chain GROUP BY region;",
            dialect=Dialect.POSTGRES,
            plain_views={
                "pv_base": "SELECT orders.id, orders.region, orders.amount FROM orders WHERE (orders.region IS NOT NULL)",
                "pv_chain": "SELECT pv_base.id, pv_base.region, (pv_base.amount * 2) AS amt2 FROM pv_base WHERE (pv_base.amount >= 10)",
            },
            current_schema="public",
        )
        self.assertEqual(plan_pg15_chain.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertEqual(plan_pg15_chain.view_spec.referenced_tables, ("orders",))

    def test_adversarial_edge_cases_unwrap_aliases_scoping_and_table_switch(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, v INTEGER NOT NULL);")
            conn.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, g TEXT, v INTEGER NOT NULL);")
            conn.execute("CREATE TABLE d (g TEXT PRIMARY KEY);")
            conn.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, g TEXT, v INTEGER NOT NULL);")
            conn.executemany(
                "INSERT INTO t VALUES (?, ?, ?)",
                [(1, "t.A", 10), (2, "A", 20), (3, "B", 30), (4, "t.A", 5)],
            )
            conn.executemany("INSERT INTO t2 VALUES (?, ?, ?)", [(1, "X", 100), (2, "Y", 200)])
            conn.executemany("INSERT INTO d VALUES (?)", [("t.A",), ("B",)])
            conn.executemany("INSERT INTO s VALUES (?, ?, ?)", [(1, "Z", 999)])

            # 1. String literal containing 't.A' must never be rewritten by alias stripping.
            conn.execute("CREATE VIEW pv_lit AS SELECT t.id, t.g, t.v FROM t WHERE t.g = 't.A';")
            # 2. Plain view with WHERE subquery referencing another table `d`.
            conn.execute("CREATE VIEW pv_subq AS SELECT id, g, v FROM t WHERE g IN (SELECT g FROM d);")
            # 3. Plain view with explicit aliases matching SQL keywords + GROUP BY / HAVING.
            conn.execute('CREATE VIEW pv_kw AS SELECT id, g AS "select", v AS "from" FROM t WHERE v > 0;')
            conn.execute(
                'CREATE VIEW pv_grp("select", "order") AS '
                "SELECT g, SUM(v) AS tot FROM t GROUP BY g HAVING tot >= 15;"
            )
            # 4. Qualified SELECT t.* and chained SELECT * explicit-column views.
            conn.execute("CREATE VIEW pv_qstar(a, b, c) AS SELECT t.* FROM t;")
            conn.execute("CREATE VIEW pv_chain_star(x, y, z) AS SELECT * FROM pv_qstar;")
            # 5. Plain views for SELECT * / pv.* (including projected subset different from base table)
            #    and HAVING outer-alias collision.
            conn.execute("CREATE VIEW pv_double AS SELECT id, g, v * 2 AS v FROM t WHERE v > 0;")
            conn.execute("CREATE VIEW pv_2cols AS SELECT id, v * 3 AS v3 FROM t WHERE v > 0;")
            # 6. Plain view with nested CTE named `s` colliding with real base table `s`.
            conn.execute(
                "CREATE VIEW pv_cte_s AS "
                "WITH s AS (SELECT g, v FROM t WHERE v >= 10) SELECT g, SUM(v) AS sv FROM s GROUP BY g;"
            )
            # 7. Parenthesized join views + schema-qualified + TEMP shadowing main view.
            conn.execute("CREATE VIEW pv_p1 AS SELECT id, g, v FROM t WHERE v >= 10;")
            conn.execute("CREATE VIEW pv_p2 AS SELECT id, v AS v2 FROM t WHERE v <= 20;")
            conn.execute("CREATE VIEW tv AS SELECT id, g, v FROM t;")
            conn.execute("CREATE TEMP VIEW tv AS SELECT id, g, v * 10 AS v FROM t;")
            # 8. Plain view that will later switch base tables from `t` to `t2`.
            conn.execute("CREATE VIEW pv_switch AS SELECT id, g, v FROM t;")
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            views = [
                ("mv_lit", "SELECT g, SUM(v) AS s FROM pv_lit GROUP BY g", "SELECT g, SUM(v) FROM pv_lit GROUP BY g ORDER BY g"),
                ("mv_subq", "SELECT g, SUM(v) AS s FROM pv_subq GROUP BY g", "SELECT g, SUM(v) FROM pv_subq GROUP BY g ORDER BY g"),
                ("mv_kw", 'SELECT "select", SUM("from") AS s FROM pv_kw GROUP BY "select"', 'SELECT "select", SUM("from") FROM pv_kw GROUP BY "select" ORDER BY 1'),
                ("mv_grp", 'SELECT "select", "order" FROM pv_grp', 'SELECT "select", "order" FROM pv_grp ORDER BY 1'),
                ("mv_cstar", "SELECT y, SUM(z) AS s FROM pv_chain_star GROUP BY y", "SELECT y, SUM(z) FROM pv_chain_star GROUP BY y ORDER BY y"),
                ("mv_star_pv", "SELECT * FROM pv_double", "SELECT * FROM pv_double ORDER BY id"),
                ("mv_qstar_pv", "SELECT pv_double.* FROM pv_double", "SELECT pv_double.* FROM pv_double ORDER BY id"),
                ("mv_star_2cols", "SELECT * FROM pv_2cols", "SELECT * FROM pv_2cols ORDER BY id"),
                ("mv_qstar_2cols", "SELECT pv_2cols.* FROM pv_2cols", "SELECT pv_2cols.* FROM pv_2cols ORDER BY id"),
                ("mv_having", "SELECT g, SUM(v) AS v FROM pv_double GROUP BY g HAVING v >= 30", "SELECT g, SUM(v) AS v FROM pv_double GROUP BY g HAVING v >= 30 ORDER BY g"),
                ("mv_cte_s", "SELECT g, sv FROM pv_cte_s", "SELECT g, sv FROM pv_cte_s ORDER BY g"),
                ("mv_paren_j", "SELECT a.g, SUM(a.v + b.v2) AS s FROM (pv_p1 a JOIN pv_p2 b ON b.id = a.id) GROUP BY a.g", "SELECT a.g, SUM(a.v + b.v2) FROM (pv_p1 a JOIN pv_p2 b ON b.id = a.id) GROUP BY a.g ORDER BY a.g"),
                ("mv_main_qual", "SELECT main.pv_p1.g, SUM(main.pv_p1.v) AS s FROM main.pv_p1 GROUP BY main.pv_p1.g", "SELECT main.pv_p1.g, SUM(main.pv_p1.v) FROM main.pv_p1 GROUP BY main.pv_p1.g ORDER BY 1"),
                ("mv_temp_shadow", "SELECT g, SUM(v) AS s FROM tv GROUP BY g", "SELECT g, SUM(v) FROM tv GROUP BY g ORDER BY g"),
                ("mv_main_tv", "SELECT g, SUM(v) AS s FROM main.tv GROUP BY g", "SELECT g, SUM(v) FROM main.tv GROUP BY g ORDER BY g"),
                ("mv_switch", "SELECT g, SUM(v) AS s FROM pv_switch GROUP BY g", "SELECT g, SUM(v) FROM pv_switch GROUP BY g ORDER BY g"),
            ]
            plans = {}
            for vn, sel, _ in views:
                plans[vn] = mgr.register_view(f"CREATE VIEW {vn} AS {sel};")

            # Verify subquery plain view tracks BOTH `t` and `d`.
            self.assertEqual(set(plans["mv_subq"].view_spec.referenced_tables), {"t", "d"})
            # Verify CTE named `s` does NOT spuriously attach to unrelated base table `s`.
            self.assertEqual(plans["mv_cte_s"].view_spec.referenced_tables, ("t",))

            def _assert_all() -> None:
                for vn, _, live in views:
                    actual = conn.execute(f"SELECT * FROM {vn} ORDER BY 1").fetchall()
                    expected = conn.execute(live).fetchall()
                    self.assertEqual(actual, expected, f"mismatch on {vn}")

            _assert_all()
            # Mutate t, d, and unrelated table s; verify every view matches live SQL.
            conn.execute("INSERT INTO t VALUES (5, 't.A', 15), (6, 'C', 40);")
            conn.execute("INSERT INTO d VALUES ('C');")
            conn.execute("DELETE FROM d WHERE g = 'B';")
            conn.execute("UPDATE t SET v = v + 5 WHERE id = 1;")
            conn.execute("INSERT INTO s VALUES (2, 'W', 500);")
            conn.commit()
            _assert_all()

            # Switch pv_switch from base table `t` to `t2` -> drifted -> repair_view() rebuilds triggers onto `t2`.
            conn.execute("DROP VIEW pv_switch;")
            conn.execute("CREATE VIEW pv_switch AS SELECT id, g, v FROM t2;")
            conn.commit()
            self.assertEqual(mgr.check_schema_drift()["mv_switch"], "drifted")
            self.assertEqual(mgr.repair_view("mv_switch"), {"mv_switch": "rebuilt"})
            self.assertEqual(mgr.check_schema_drift()["mv_switch"], "ok")
            conn.execute("INSERT INTO t2 VALUES (3, 'X', 50);")
            conn.execute("INSERT INTO t VALUES (7, 'X', 999);")
            conn.commit()
            self.assertEqual(
                conn.execute("SELECT g, s FROM mv_switch ORDER BY g").fetchall(),
                conn.execute("SELECT g, SUM(v) FROM pv_switch GROUP BY g ORDER BY g").fetchall(),
            )
            # Confirm no triggers remain on `t` for mv_switch.
            t_trigs = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 't' AND name LIKE '%mv_switch%'"
                ).fetchall()
            ]
            self.assertEqual(t_trigs, [])
        finally:
            conn.close()

    def test_introspect_plain_views_fail_closed_and_no_idle_tx_leak(self) -> None:
        from matview.connectivity import introspect_plain_views

        # 1. Real DB/permission error must propagate (fail closed), never silently return {}.
        class _BrokenCursor:
            def execute(self, _sql: str) -> None:
                raise RuntimeError("permission denied for relation pg_class")

            def close(self) -> None:
                pass

        class _BrokenConn:
            def cursor(self) -> _BrokenCursor:
                return _BrokenCursor()

            def rollback(self) -> None:
                pass

        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            introspect_plain_views(_BrokenConn(), dialect=Dialect.POSTGRES)

        # 2. Standalone call on an idle sqlite3 connection must not leave a transaction open.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY);")
            conn.execute("CREATE VIEW pv AS SELECT id FROM t;")
            conn.commit()
            self.assertFalse(conn.in_transaction)
            res, cur_schema = introspect_plain_views(conn, dialect=Dialect.SQLITE)
            self.assertIn("pv", res)
            self.assertEqual(cur_schema, "main")
            self.assertFalse(conn.in_transaction)
        finally:
            conn.close()

        # 3. Standalone call on an idle SQLAlchemy Connection must not leave a transaction open.
        try:
            import sqlalchemy as sa
        except ImportError:
            return
        eng = sa.create_engine("sqlite:///:memory:")
        try:
            with eng.connect() as sa_conn:
                sa_conn.execute(sa.text("CREATE TABLE t (id INTEGER PRIMARY KEY)"))
                sa_conn.execute(sa.text("CREATE VIEW pv AS SELECT id FROM t"))
                sa_conn.commit()
                self.assertFalse(sa_conn.in_transaction())
                res, cur_schema = introspect_plain_views(sa_conn, dialect=Dialect.SQLITE)
                self.assertIn("pv", res)
                self.assertEqual(cur_schema, "main")
                self.assertFalse(sa_conn.in_transaction())
        finally:
            eng.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
