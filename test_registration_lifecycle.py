#!/usr/bin/env python3
"""Transaction ownership of manager lifecycle operations.

Contract under test (see ``TransactionOwnershipError``): construction
(catalog initialization), ``register_view``, ``sync_views``, ``drop_view``,
``apply_schema_ddl``, ``repair_view`` and ``backfill_incrementally`` own the
transaction they run in. On a connection with an open caller transaction, or
whose driver does not report its transaction state, they refuse before
executing any SQL, so caller work is neither committed (explicitly, or
implicitly by MySQL/MariaDB DDL) nor rolled back. Read-only methods join a
caller transaction and never end it, and never leave a transaction of their
own open. A PostgreSQL connection in autocommit mode is run in one explicit
transaction per operation, so registration stays atomic and serialized
against writers, and autocommit is restored afterwards.

SQLite tests run offline on file-backed databases with independent observer
connections. Native PostgreSQL (POSTGRES_DSN) and MySQL/MariaDB (MYSQL_DSN)
tests use independent server sessions as observers and skip without a DSN.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse

from matview_sync import (
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    MaterializedViewPlan,
    TransactionOwnershipError,
)
from matview.connectivity import dbapi_transaction_state

VIEW_SQL = "CREATE VIEW owned_totals AS SELECT bucket, COUNT(*) AS n, SUM(amt) AS total FROM owned_rows GROUP BY bucket;"
LIVE_SQL = "SELECT bucket, COUNT(*), SUM(amt) FROM owned_rows GROUP BY bucket ORDER BY bucket"


class _SqliteFile:
    def __init__(self) -> None:
        fd, self.path = tempfile.mkstemp(prefix="matview-ownership-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE owned_rows (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);")
        conn.executemany("INSERT INTO owned_rows VALUES (?, ?, ?);", [(1, "a", 10), (2, "a", 20), (3, "b", 30)])
        conn.commit()
        conn.close()

    def connect(self, **kwargs: Any) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10, **kwargs)

    def cleanup(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.unlink(self.path + suffix)
            except FileNotFoundError:
                pass


class SqliteTransactionOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _SqliteFile()
        self.addCleanup(self.db.cleanup)
        self.observer = self.db.connect()
        self.addCleanup(self.observer.close)

    def _observed_count(self, row_id: int) -> int:
        return self.observer.execute("SELECT COUNT(*) FROM owned_rows WHERE id = ?;", (row_id,)).fetchone()[0]

    def _objects(self) -> List[tuple]:
        return self.observer.execute("SELECT type, name FROM sqlite_master ORDER BY type, name;").fetchall()

    def test_construction_refuses_caller_transaction_without_committing_it(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)
        objects_before = self._objects()
        conn.execute("INSERT INTO owned_rows VALUES (10, 'c', 1);")
        self.assertTrue(conn.in_transaction)
        with self.assertRaises(TransactionOwnershipError) as ctx:
            MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        self.assertIn("caller-owned transaction", str(ctx.exception))
        self.assertTrue(conn.in_transaction, "construction ended the caller's transaction")
        self.assertEqual(self._observed_count(10), 0, "construction committed caller-owned work")
        self.assertEqual(self._objects(), objects_before, "construction created the catalog")
        conn.rollback()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM owned_rows WHERE id = 10;").fetchone()[0], 0)

    def test_lifecycle_operations_refuse_caller_transaction_before_any_sql(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        plan = manager.register_view(VIEW_SQL)
        self.assertFalse(conn.in_transaction)
        changed_sql = VIEW_SQL.replace("GROUP BY", "WHERE amt > 0 GROUP BY")
        operations: List[tuple[str, Callable[[], Any]]] = [
            ("register_view", lambda: manager.register_view(VIEW_SQL)),
            ("register_view(force_rebuild)", lambda: manager.register_view(VIEW_SQL, force_rebuild=True)),
            ("sync_views", lambda: manager.sync_views([changed_sql])),
            ("drop_view", lambda: manager.drop_view(plan.view_name)),
            ("apply_schema_ddl", lambda: manager.apply_schema_ddl("CREATE TABLE other_rows (id INTEGER PRIMARY KEY);")),
            ("repair_view", lambda: manager.repair_view(plan.view_name)),
            ("backfill_incrementally", lambda: manager.backfill_incrementally(plan)),
        ]
        live_before = self.observer.execute(LIVE_SQL).fetchall()
        objects_before = self._objects()
        for label, operation in operations:
            with self.subTest(operation=label):
                conn.execute("INSERT INTO owned_rows VALUES (20, 'c', 7);")
                statements: List[str] = []
                conn.set_trace_callback(statements.append)
                try:
                    with self.assertRaises(TransactionOwnershipError):
                        operation()
                finally:
                    conn.set_trace_callback(None)
                self.assertEqual(statements, [], f"{label} executed SQL before refusing")
                self.assertTrue(conn.in_transaction, f"{label} ended the caller's transaction")
                self.assertEqual(self._observed_count(20), 0, f"{label} committed caller-owned work")
                self.assertEqual(self._objects(), objects_before)
                conn.rollback()
        # The published view is untouched and still maintained.
        self.assertEqual(conn.execute(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket;").fetchall(), live_before)
        conn.execute("INSERT INTO owned_rows VALUES (21, 'a', 5);")
        conn.commit()
        self.assertEqual(
            conn.execute(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket;").fetchall(),
            conn.execute(LIVE_SQL).fetchall(),
        )

    def test_read_operations_join_caller_transaction_without_ending_it(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        manager.register_view(VIEW_SQL)
        conn.execute("INSERT INTO owned_rows VALUES (30, 'a', 1);")
        for label, operation in (
            ("get_catalog", manager.get_catalog),
            ("fetch_schema", manager.fetch_schema),
            ("check_schema_drift", manager.check_schema_drift),
            ("diff_views", lambda: manager.diff_views([VIEW_SQL])),
        ):
            with self.subTest(operation=label):
                operation()
                self.assertTrue(conn.in_transaction, f"{label} ended the caller's transaction")
                self.assertEqual(self._observed_count(30), 0, f"{label} committed caller-owned work")
        conn.rollback()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM owned_rows WHERE id = 30;").fetchone()[0], 0)

    def test_unknown_transaction_state_is_refused(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)

        class HidesTransactionState:
            """A DB-API wrapper that does not report its transaction state."""

            def __init__(self, wrapped: sqlite3.Connection) -> None:
                self._wrapped = wrapped

            def cursor(self) -> Any:
                return self._wrapped.cursor()

            def commit(self) -> None:
                self._wrapped.commit()

            def rollback(self) -> None:
                self._wrapped.rollback()

        conn.execute("INSERT INTO owned_rows VALUES (40, 'c', 1);")
        with self.assertRaises(TransactionOwnershipError) as ctx:
            MaterializedViewManager(HidesTransactionState(conn), dialect=Dialect.SQLITE)
        self.assertIn("cannot determine", str(ctx.exception))
        # Actionable guidance for the drivers known not to report it.
        self.assertIn("pg8000", str(ctx.exception))
        self.assertIn("psycopg2", str(ctx.exception))
        self.assertTrue(conn.in_transaction)
        self.assertEqual(self._observed_count(40), 0)
        conn.rollback()

    def test_transaction_on_wrapped_driver_is_detected_behind_idle_wrapper(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)

        class IdleLookingProxy:
            """Pool-proxy shape: reports idle itself, exposes the physical driver."""

            in_transaction = False

            def __init__(self, wrapped: sqlite3.Connection) -> None:
                self.dbapi_connection = wrapped

            def cursor(self) -> Any:
                return self.dbapi_connection.cursor()

            def commit(self) -> None:
                self.dbapi_connection.commit()

            def rollback(self) -> None:
                self.dbapi_connection.rollback()

        conn.execute("INSERT INTO owned_rows VALUES (50, 'c', 1);")
        with self.assertRaises(TransactionOwnershipError):
            MaterializedViewManager(IdleLookingProxy(conn), dialect=Dialect.SQLITE)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(self._observed_count(50), 0)
        conn.rollback()

    @unittest.skipUnless(hasattr(sqlite3, "LEGACY_TRANSACTION_CONTROL"), "PEP 249 autocommit attribute needs Python 3.12+")
    def test_pep249_autocommit_modes_are_refused(self) -> None:
        for mode in (True, False):
            with self.subTest(autocommit=mode):
                conn = self.db.connect(autocommit=mode)
                try:
                    with self.assertRaises(TransactionOwnershipError) as ctx:
                        MaterializedViewManager(conn, dialect=Dialect.SQLITE)
                    self.assertIn("LEGACY_TRANSACTION_CONTROL", str(ctx.exception))
                finally:
                    conn.close()

    def test_isolation_level_none_connection_is_owned_and_left_idle(self) -> None:
        conn = self.db.connect(isolation_level=None)
        self.addCleanup(conn.close)
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        plan = manager.register_view(VIEW_SQL)
        self.assertFalse(conn.in_transaction)
        conn.execute("INSERT INTO owned_rows VALUES (60, 'b', 3);")
        self.assertEqual(
            self.observer.execute(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket;").fetchall(),
            self.observer.execute(LIVE_SQL).fetchall(),
        )
        self.assertTrue(manager.drop_view(plan.view_name))
        self.assertFalse(conn.in_transaction)


    def test_failed_catalog_read_keeps_caller_transaction(self) -> None:
        # No catalog exists yet (skip_ensure_catalog), so get_catalog's read
        # fails inside the caller's transaction; the caller's writes before
        # and after it must survive and commit together.
        conn = self.db.connect()
        self.addCleanup(conn.close)
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE, skip_ensure_catalog=True)
        conn.execute("INSERT INTO owned_rows VALUES (70, 'c', 1);")
        self.assertEqual(manager.get_catalog(), {})
        self.assertTrue(conn.in_transaction, "get_catalog ended the caller's transaction")
        conn.execute("INSERT INTO owned_rows VALUES (71, 'c', 1);")
        self.assertEqual(self._observed_count(70), 0)
        conn.commit()
        self.assertEqual(self._observed_count(70) + self._observed_count(71), 2)

    def test_success_requires_the_commit_to_end_the_transaction(self) -> None:
        conn = self.db.connect()
        self.addCleanup(conn.close)

        class CommitIsNoOp:
            def __init__(self, raw: sqlite3.Connection) -> None:
                self._raw = raw

            @property
            def in_transaction(self) -> bool:
                return self._raw.in_transaction

            def cursor(self) -> Any:
                return self._raw.cursor()

            def commit(self) -> None:
                return None

            def rollback(self) -> None:
                self._raw.rollback()

        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        manager.conn = CommitIsNoOp(conn)
        with self.assertRaisesRegex(RuntimeError, "committing did not end the transaction"):
            manager.register_view(VIEW_SQL)
        self.assertFalse(conn.in_transaction)
        self.assertNotIn("owned_totals", [n for _, n in self._objects()], "uncommitted registration persisted")

    def test_raw_apsw_connection_is_committed(self) -> None:
        try:
            import apsw  # type: ignore
        except ImportError:
            self.skipTest("apsw is not installed")
        raw = apsw.Connection(self.db.path)
        self.addCleanup(raw.close)
        manager = MaterializedViewManager(raw, dialect=Dialect.SQLITE)
        plan = manager.register_view(VIEW_SQL)
        self.assertFalse(raw.in_transaction)
        # Visible to (and not blocking) an independent connection.
        self.assertEqual(
            self.observer.execute(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket;").fetchall(),
            self.observer.execute(LIVE_SQL).fetchall(),
        )
        self.observer.execute("INSERT INTO owned_rows VALUES (90, 'a', 1);")
        self.observer.commit()
        self.assertTrue(manager.drop_view(plan.view_name))
        self.assertFalse(raw.in_transaction)


class _FailOnCatalogDelete:
    """Connection proxy raising (in Python) at drop_view's catalog DELETE."""

    def __init__(self, raw: Any, catalog_table: str) -> None:
        self._raw = raw
        self._needle = f"DELETE FROM {catalog_table}".upper()
        self.injected = False

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        outer = self
        raw_cursor = self._raw.cursor(*args, **kwargs)

        class _Cursor:
            def execute(self, sql: Any, params: Any = None) -> Any:
                if not outer.injected and " ".join(str(sql).split()).upper().startswith(outer._needle):
                    outer.injected = True
                    raise RuntimeError("injected failure before the catalog DELETE")
                return raw_cursor.execute(sql) if params is None else raw_cursor.execute(sql, params)

            def __getattr__(self, name: str) -> Any:
                return getattr(raw_cursor, name)

        return _Cursor()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class DriverTransactionStateTests(unittest.TestCase):
    """The documented transaction-state reports of the supported drivers."""

    def test_reports(self) -> None:
        class Obj:
            def __init__(self, **attrs: Any) -> None:
                self.__dict__.update(attrs)

        class Info:
            def __init__(self, status: int) -> None:
                self.transaction_status = status

        cases = [
            ("sqlite3/mysql-connector idle", Obj(in_transaction=False), False),
            ("sqlite3/mysql-connector open", Obj(in_transaction=True), True),
            ("psycopg2 IDLE", Obj(get_transaction_status=lambda: 0), False),
            ("psycopg2 INTRANS", Obj(get_transaction_status=lambda: 2), True),
            ("psycopg2 INERROR", Obj(get_transaction_status=lambda: 3), True),
            ("psycopg2 UNKNOWN", Obj(get_transaction_status=lambda: 4), True),
            ("psycopg3 IDLE", Obj(info=Info(0)), False),
            ("psycopg3 INTRANS", Obj(info=Info(2)), True),
            ("PyMySQL idle autocommit", Obj(server_status=0x0002), False),
            ("PyMySQL in trans", Obj(server_status=0x0003), True),
            ("nothing reported", Obj(), None),
            ("non-integer status", Obj(get_transaction_status=lambda: "idle"), None),
            ("wrapper idle, driver open", Obj(in_transaction=False, dbapi_connection=Obj(in_transaction=True)), True),
            ("wrapper unknown, driver idle", Obj(driver_connection=Obj(get_transaction_status=lambda: 0)), False),
        ]
        for label, conn, expected in cases:
            with self.subTest(case=label):
                self.assertIs(dbapi_transaction_state(conn), expected)


def _pg_connect(dsn: str):
    import psycopg2

    # A lock left held by a broken implementation fails the test instead of
    # hanging it.
    return psycopg2.connect(dsn, options="-c lock_timeout=20s")


def _mysql_connect(dsn: str, autocommit: bool = False):
    import pymysql

    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=autocommit,
        init_command="SET SESSION lock_wait_timeout = 20",
    )


class NativePostgresOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dsn = os.environ.get("POSTGRES_DSN")
        if not self.dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        from psycopg2 import extensions

        self.ext = extensions
        self.schema = "mv_own_" + uuid.uuid4().hex[:12]
        self.conn = _pg_connect(self.dsn)
        self.observer = _pg_connect(self.dsn)
        self.observer.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{self.schema}"')
            cur.execute(f'SET search_path TO "{self.schema}"')
            cur.execute("CREATE TABLE owned_rows (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL)")
            cur.execute("INSERT INTO owned_rows VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30)")
            cur.execute("SELECT version()")
            print(f"[native] {cur.fetchone()[0]}", file=sys.stderr)
        self.conn.commit()
        with self.observer.cursor() as cur:
            cur.execute(f'SET search_path TO "{self.schema}"')

    def tearDown(self) -> None:
        try:
            self.conn.rollback()
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        finally:
            self.conn.close()
            self.observer.close()

    def _status(self) -> int:
        return self.conn.get_transaction_status()

    def _observed(self, sql: str) -> Any:
        with self.observer.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def _manager(self) -> MaterializedViewManager:
        return MaterializedViewManager(self.conn, dialect=Dialect.POSTGRES, object_prefix=f"__mvo{self.schema[-6:]}_")

    def test_construction_and_lifecycle_refuse_caller_transaction(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO owned_rows VALUES (10, 'c', 1)")
        self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_INTRANS)
        with self.assertRaises(TransactionOwnershipError):
            self._manager()
        self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_INTRANS)
        self.assertEqual(self._observed("SELECT COUNT(*) FROM owned_rows WHERE id = 10"), [(0,)])
        self.conn.rollback()

        manager = self._manager()
        plan = manager.register_view(VIEW_SQL)
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO owned_rows VALUES (11, 'c', 1)")
        for label, operation in (
            ("register_view", lambda: manager.register_view(VIEW_SQL, force_rebuild=True)),
            ("sync_views", lambda: manager.sync_views([VIEW_SQL])),
            ("drop_view", lambda: manager.drop_view(plan.view_name)),
            ("repair_view", lambda: manager.repair_view(plan.view_name)),
            ("apply_schema_ddl", lambda: manager.apply_schema_ddl("CREATE TABLE other_rows (id INTEGER PRIMARY KEY)")),
        ):
            with self.subTest(operation=label):
                with self.assertRaises(TransactionOwnershipError):
                    operation()
                self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_INTRANS)
                self.assertEqual(self._observed("SELECT COUNT(*) FROM owned_rows WHERE id = 11"), [(0,)])
        self.conn.rollback()
        self.assertEqual(self._observed(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket"), self._observed(LIVE_SQL))

    def test_manager_never_leaves_its_own_transaction_open(self) -> None:
        manager = self._manager()
        self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_IDLE)
        for label, operation in (
            ("get_catalog", manager.get_catalog),
            ("fetch_schema", manager.fetch_schema),
            ("register_view", lambda: manager.register_view(VIEW_SQL)),
            ("register_view unchanged", lambda: manager.register_view(VIEW_SQL)),
            ("sync_views unchanged", lambda: manager.sync_views([VIEW_SQL])),
            ("check_schema_drift", manager.check_schema_drift),
            ("diff_views", lambda: manager.diff_views([VIEW_SQL])),
        ):
            with self.subTest(operation=label):
                operation()
                self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_IDLE, f"{label} left a transaction open")
        # ...so a following lifecycle operation is not mistaken for caller work.
        manager.get_catalog()
        manager.register_view(VIEW_SQL, force_rebuild=True)

    def test_autocommit_registration_is_atomic_and_serialized_with_writers(self) -> None:
        self.conn.autocommit = True
        manager = self._manager()
        writer = _pg_connect(self.dsn)
        self.addCleanup(writer.close)
        with writer.cursor() as cur:
            cur.execute(f'SET search_path TO "{self.schema}"')
        writer.commit()
        writer_pid = writer.get_backend_pid()
        observations: List[str] = []
        errors: List[BaseException] = []
        threads: List[threading.Thread] = []
        original = MaterializedViewPlan.get_populate_statements

        def write() -> None:
            try:
                with writer.cursor() as cur:
                    cur.execute("INSERT INTO owned_rows VALUES (100, 'b', 1000)")
                writer.commit()
            except BaseException as exc:  # inspected below
                errors.append(exc)

        def hooked(plan_self: Any, *args: Any, **kwargs: Any) -> Any:
            # Called after structure, triggers and catalog row, before the
            # initial population: a writer must not be able to commit now.
            if not threads:
                thread = threading.Thread(target=write, daemon=True)
                threads.append(thread)
                thread.start()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if not thread.is_alive():
                        observations.append("writer committed during registration")
                        break
                    rows = self._observed(f"SELECT wait_event_type FROM pg_stat_activity WHERE pid = {writer_pid}")
                    if rows and rows[0][0] == "Lock":
                        observations.append("writer waiting on lock")
                        break
                    time.sleep(0.01)
                else:
                    observations.append("timeout")
            return original(plan_self, *args, **kwargs)

        MaterializedViewPlan.get_populate_statements = hooked
        try:
            plan = manager.register_view(VIEW_SQL)
        finally:
            MaterializedViewPlan.get_populate_statements = original
        threads[0].join(10)
        self.assertFalse(threads[0].is_alive())
        if errors:
            raise errors[0]
        self.assertEqual(observations, ["writer waiting on lock"])
        self.assertIs(self.conn.autocommit, True, "autocommit mode was not restored")
        self.assertEqual(self._observed(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket"), self._observed(LIVE_SQL))

    def test_failed_catalog_read_keeps_caller_transaction(self) -> None:
        manager = MaterializedViewManager(
            self.conn, dialect=Dialect.POSTGRES, object_prefix=f"__mvo{self.schema[-6:]}_", skip_ensure_catalog=True
        )
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO owned_rows VALUES (70, 'c', 1)")
        self.assertEqual(manager.get_catalog(), {})
        self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_INTRANS, "transaction aborted or ended")
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO owned_rows VALUES (71, 'c', 1)")
        self.assertEqual(self._observed("SELECT COUNT(*) FROM owned_rows WHERE id IN (70, 71)"), [(0,)])
        self.conn.commit()
        self.assertEqual(self._observed("SELECT COUNT(*) FROM owned_rows WHERE id IN (70, 71)"), [(2,)])

    def test_python_failure_inside_drop_view_is_rolled_back(self) -> None:
        manager = self._manager()
        plan = manager.register_view(VIEW_SQL)
        proxy = _FailOnCatalogDelete(self.conn, manager.catalog_table)
        manager.conn = proxy
        try:
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                manager.drop_view(plan.view_name)
        finally:
            manager.conn = self.conn
        self.assertTrue(proxy.injected)
        self.assertEqual(self._status(), self.ext.TRANSACTION_STATUS_IDLE)
        self.assertIn(plan.view_name, manager.get_catalog())
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO owned_rows VALUES (80, 'a', 2)")
        self.conn.commit()
        self.assertEqual(self._observed(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket"), self._observed(LIVE_SQL))

    def test_sqlalchemy_pool_proxy_autocommit_is_set_on_the_driver_only(self) -> None:
        import sqlalchemy as sa

        engine = sa.create_engine(
            "postgresql+psycopg2://" + self.dsn.split("://", 1)[1],
            isolation_level="AUTOCOMMIT",
            connect_args={"options": f"-c search_path={self.schema} -c lock_timeout=20s"},
        )
        self.addCleanup(engine.dispose)
        manager = MaterializedViewManager(engine, dialect=Dialect.POSTGRES, object_prefix=f"__mvs{self.schema[-6:]}_")
        fairy = manager.conn
        plan = manager.register_view(VIEW_SQL)
        self.assertNotIn("autocommit", vars(fairy), "autocommit was written onto the pool proxy")
        self.assertIs(fairy.dbapi_connection.autocommit, True)
        self.assertEqual(self._observed(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket"), self._observed(LIVE_SQL))
        manager.drop_view(plan.view_name)

    def test_autocommit_registration_failure_leaves_nothing_installed(self) -> None:
        self.conn.autocommit = True
        manager = self._manager()
        original = MaterializedViewPlan.get_populate_statements

        def failing(plan_self: Any, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("injected population failure")

        MaterializedViewPlan.get_populate_statements = failing
        try:
            with self.assertRaisesRegex(RuntimeError, "injected population failure"):
                manager.register_view(VIEW_SQL)
        finally:
            MaterializedViewPlan.get_populate_statements = original
        self.assertIs(self.conn.autocommit, True)
        self.assertEqual(manager.get_catalog(), {})
        leftovers = self._observed(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE n.nspname = '{self.schema}' AND c.relname <> 'owned_rows' AND c.relname NOT LIKE '%_pkey' "
            f"AND c.relname <> '{manager.catalog_table}' AND c.relname NOT LIKE '{manager.catalog_table}%'"
        )
        self.assertEqual(leftovers, [], "failed autocommit registration left objects installed")
        triggers = self._observed(
            "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            f"JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{self.schema}' AND NOT t.tgisinternal"
        )
        self.assertEqual(triggers, [])


class NativeMysqlOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dsn = os.environ.get("MYSQL_DSN")
        if not self.dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        suffix = uuid.uuid4().hex[:10]
        self.table = f"owned_rows_{suffix}"
        self.prefix = f"__mvo{suffix}_"
        self.conn = _mysql_connect(self.dsn)
        self.observer = _mysql_connect(self.dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE TABLE {self.table} (id INT PRIMARY KEY, bucket VARCHAR(8) NOT NULL, amt INT NOT NULL) ENGINE=InnoDB")
            cur.execute(f"INSERT INTO {self.table} VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30)")
            cur.execute("SELECT VERSION()")
            print(f"[native] {cur.fetchone()[0]}", file=sys.stderr)
        self.conn.commit()

    def tearDown(self) -> None:
        try:
            self.conn.rollback()
            if getattr(self, "manager", None) is not None:
                for name in list(self.manager.get_catalog()):
                    self.manager.drop_view(name)
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
                cur.execute(f"DROP TABLE IF EXISTS {self.prefix}catalog")
            self.conn.commit()
        finally:
            self.conn.close()
            self.observer.close()

    def _observed(self, sql: str) -> Any:
        with self.observer.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def _catalog_exists(self) -> bool:
        return bool(self._observed(
            "SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_schema = DATABASE() AND table_name = '{self.prefix}catalog'"
        )[0][0])

    def _check_refusal(self, operation: Callable[[], Any], row_id: int, start_explicitly: bool) -> None:
        with self.conn.cursor() as cur:
            if start_explicitly:
                cur.execute("START TRANSACTION")
            cur.execute(f"INSERT INTO {self.table} VALUES ({row_id}, 'c', 1)")
        with self.assertRaises(TransactionOwnershipError):
            operation()
        # Not committed (DDL would have committed it implicitly) and still
        # owned by the caller: its rollback undoes it.
        self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id = {row_id}"), ((0,),))
        self.conn.rollback()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table} WHERE id = {row_id}")
            self.assertEqual(cur.fetchone(), (0,))

    def test_failed_catalog_read_keeps_caller_transaction(self) -> None:
        manager = MaterializedViewManager(
            self.conn, dialect=Dialect.MYSQL, object_prefix=self.prefix, skip_ensure_catalog=True
        )
        with self.conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table} VALUES (70, 'c', 1)")
        self.assertEqual(manager.get_catalog(), {})
        with self.conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table} VALUES (71, 'c', 1)")
        self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id IN (70, 71)"), ((0,),))
        self.conn.commit()
        self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id IN (70, 71)"), ((2,),))

    def test_construction_refuses_caller_transaction(self) -> None:
        self._check_refusal(
            lambda: MaterializedViewManager(self.conn, dialect=Dialect.MYSQL, object_prefix=self.prefix), 10, False
        )
        self.assertFalse(self._catalog_exists(), "construction ran catalog DDL")

    def test_construction_refuses_explicit_transaction_on_autocommit_connection(self) -> None:
        self.conn.autocommit(True)
        self._check_refusal(
            lambda: MaterializedViewManager(self.conn, dialect=Dialect.MYSQL, object_prefix=self.prefix), 11, True
        )
        self.assertFalse(self._catalog_exists())

    def test_lifecycle_operations_refuse_caller_transaction(self) -> None:
        # ignore_limitations: the unrelated MYSQL_NO_TRUNCATE_TRIGGER limitation
        # (TRUNCATE bypasses triggers) is not exercised here.
        self.manager = MaterializedViewManager(
            self.conn, dialect=Dialect.MYSQL, object_prefix=self.prefix, ignore_limitations=True
        )
        view_sql = VIEW_SQL.replace("owned_rows", self.table)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan = self.manager.register_view(view_sql)
        for row_id, (label, operation) in enumerate(
            (
                ("register_view", lambda: self.manager.register_view(view_sql)),
                ("sync_views", lambda: self.manager.sync_views([view_sql])),
                ("drop_view", lambda: self.manager.drop_view(plan.view_name)),
                ("repair_view", lambda: self.manager.repair_view(plan.view_name)),
                ("apply_schema_ddl", lambda: self.manager.apply_schema_ddl("CREATE TABLE IF NOT EXISTS t_unused (id INT PRIMARY KEY)")),
            ),
            start=20,
        ):
            with self.subTest(operation=label):
                self._check_refusal(operation, row_id, False)
        self.assertEqual(
            self._observed(f"SELECT bucket, n, total FROM {plan.view_name} ORDER BY bucket"),
            self._observed(LIVE_SQL.replace("owned_rows", self.table)),
        )


if __name__ == "__main__":
    unittest.main()
