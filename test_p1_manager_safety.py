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

from matview_sync import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
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
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True)

        # Create base table in SQLite
        with raw_conn:
            raw_conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, score REAL);")

        view_sql = "CREATE VIEW mv_users AS SELECT status, SUM(score) as tot FROM users GROUP BY status;"
        schemas = parse_table_schema("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT, score REAL);")
        manager.fetch_schema = lambda extra_schema_sql=None: schemas

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
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True)

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
        manager = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True)

        prefix = "t_p1_my_"
        base_tbl = f"{prefix}items"
        view_name = f"{prefix}mv_items"
        backing_tbl = f"{manager.object_prefix}store_{view_name}"

        with tracked.cursor() as cur:
            cur.execute(f"DROP VIEW IF EXISTS {view_name};")
            cur.execute(f"DROP TABLE IF EXISTS {backing_tbl};")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
            cur.execute(f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, grp VARCHAR(32), qty INT);")
            cur.execute(f"INSERT INTO {base_tbl} VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 5);")
            tracked.commit()
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
            with tracked.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {collide_tbl};")
                cur.execute(f"CREATE TABLE {collide_tbl} (id INT PRIMARY KEY, data_str VARCHAR(64));")
                cur.execute(f"INSERT INTO {collide_tbl} VALUES (99, 'KEEP_SAFE');")
                tracked.commit()
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
            manager_col = MaterializedViewManager(tracked, dialect=Dialect.MYSQL, ignore_limitations=True)
            with self.assertRaises(pymysql.err.OperationalError):
                manager_col._register_single_view_spec(
                    v_spec=manager_col.fetch_schema()[base_tbl],  # placeholder spec
                    schemas=manager_col.fetch_schema(),
                    populate=False,
                    incremental_backfill=False,
                    batch_size=500,
                    on_progress=None,
                    force_rebuild=False,
                    effective_ignore=True,
                )
            tracked.cursor_factory = None

            with tracked.cursor() as cur:
                cur.execute(f"SELECT data_str FROM {collide_tbl} WHERE id = 99;")
                row = cur.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "KEEP_SAFE")
                cur.execute(f"DROP TABLE IF EXISTS {collide_tbl};")
                tracked.commit()
            tracked.assert_zero_leaked("Zero leaks after collision test")
        finally:
            with tracked.cursor() as cur:
                cur.execute(f"DROP VIEW IF EXISTS {view_name};")
                cur.execute(f"DROP TABLE IF EXISTS {backing_tbl};")
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
                tracked.commit()
            tracked.close()

    def test_native_postgres_zero_leaks(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; skipping live PostgreSQL safety test")

        import psycopg2
        raw_conn = psycopg2.connect(dsn)
        tracked = CursorTrackingConnection(raw_conn)
        manager = MaterializedViewManager(tracked, dialect=Dialect.POSTGRES, ignore_limitations=True)

        prefix = "t_p1_pg_"
        base_tbl = f"{prefix}items"
        view_name = f"{prefix}mv_items"
        backing_tbl = f"{manager.object_prefix}store_{view_name}"

        with tracked.cursor() as cur:
            cur.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {backing_tbl} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
            cur.execute(f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, grp TEXT, val INT);")
            cur.execute(f"INSERT INTO {base_tbl} VALUES (1, 'x', 10), (2, 'x', 20);")
            tracked.commit()
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
            with tracked.cursor() as cur:
                cur.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {backing_tbl} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
                tracked.commit()
            tracked.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
