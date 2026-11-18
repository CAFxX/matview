#!/usr/bin/env python3
"""Fail-closed, exception-safe and transaction-owning ``bulk_changes()``.

Contract under test (``MaterializedViewManager.bulk_changes``):

* The manager owns the transaction the context runs in, exactly like the
  other lifecycle operations: a connection with an open caller transaction,
  or whose driver does not report its transaction state, is refused with
  ``TransactionOwnershipError`` before any SQL, so caller work is neither
  committed nor rolled back. The body and the refresh run in one transaction
  that commits on success and rolls back on any exception (``BaseException``
  included). Autocommit connections are run in one explicit transaction.
* Maintenance is suspended only connection-locally and never globally
  without transactional protection: PostgreSQL uses ``SET LOCAL`` inside the
  owned transaction (or transactional trigger DDL that blocks other writers),
  SQLite drops triggers under ``BEGIN IMMEDIATE``, and MySQL/MariaDB uses
  only the session variable and DML. On MySQL/MariaDB, ``use_shadow_swap``
  and ``drop_triggers`` (DDL, which commits implicitly) are refused before
  any SQL. An unreadable catalog and unknown view names are refused; on
  PostgreSQL (session bypass) and MySQL/MariaDB a ``view_names`` subset that
  would leave other views unmaintained is refused.
* A failure anywhere (setup, body, refresh, restoration) rolls the whole
  transaction back, which restores the original maintenance objects; it is
  never swallowed. When the connection state cannot be restored (rollback
  fails, MySQL session flag cannot be cleared) the physical connection is
  closed, the manager refuses further use, and ``BulkChangesConnectionError``
  is raised.

Oracle: the original view's SELECT evaluated natively on the same database
state by an independent connection; equality is exact.

SQLite tests run offline on file-backed databases with independent
connections. Native PostgreSQL (POSTGRES_DSN) and MySQL/MariaDB (MYSQL_DSN)
tests use independent server sessions and skip without a DSN.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
from typing import Any, Callable, Dict, List, Optional
from unittest import mock
from urllib.parse import urlparse

import matview.manager as manager_module
from matview_sync import (
    BulkChangesConnectionError,
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    TransactionOwnershipError,
)

TABLE = "bulk_rows"
VIEW_SQL = (
    f"CREATE VIEW bulk_totals AS SELECT bucket, COUNT(*) AS n, SUM(amt) AS total "
    f"FROM {TABLE} GROUP BY bucket;"
)
OTHER_VIEW_SQL = (
    f"CREATE VIEW bulk_maxes AS SELECT bucket, COUNT(*) AS n, SUM(amt) AS total "
    f"FROM {TABLE} WHERE amt > 15 GROUP BY bucket;"
)
LIVE_SQL = f"SELECT bucket, COUNT(*), SUM(amt) FROM {TABLE} GROUP BY bucket ORDER BY bucket"
OTHER_LIVE_SQL = f"SELECT bucket, COUNT(*), SUM(amt) FROM {TABLE} WHERE amt > 15 GROUP BY bucket ORDER BY bucket"
INITIAL_ROWS = [(1, "a", 10), (2, "a", 20), (3, "b", 30)]
INJECTED = "matview_injected_restore_failure"


def _inject_failing_trigger_ddl(position: str = "middle") -> Any:
    """Patch the planner the manager uses so re-created trigger DDL fails.

    The failing statement sits in the middle of the trigger DDL list (after
    some maintenance objects were re-created) or at its end.
    """
    real = manager_module.generate_materialized_view

    def failing(*args: Any, **kwargs: Any) -> Any:
        plan = real(*args, **kwargs)
        ddls = list(plan.trigger_ddls)
        at = len(ddls) // 2 if position == "middle" else len(ddls)
        ddls.insert(at, f"SELECT {INJECTED}()")
        return dataclasses.replace(plan, trigger_ddls=ddls)

    return mock.patch.object(manager_module, "generate_materialized_view", failing)


def _quiet_register(manager: MaterializedViewManager, sql: str, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        return manager.register_view(sql, **kwargs)


# ---------------------------------------------------------------------------
# SQLite (offline, file-backed, independent connections)
# ---------------------------------------------------------------------------


class _SqliteDb:
    def __init__(self) -> None:
        fd, self.path = tempfile.mkstemp(prefix="matview-bulk-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(self.path)
        conn.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);")
        conn.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?);", INITIAL_ROWS)
        conn.commit()
        conn.close()

    def connect(self, **kwargs: Any) -> sqlite3.Connection:
        kwargs.setdefault("timeout", 10)
        return sqlite3.connect(self.path, **kwargs)

    def cleanup(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.unlink(self.path + suffix)
            except FileNotFoundError:
                pass


class _SqliteBase(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _SqliteDb()
        self.addCleanup(self.db.cleanup)
        self.observer = self.db.connect()
        self.addCleanup(self.observer.close)

    def _manager(self, **connect_kwargs: Any) -> MaterializedViewManager:
        conn = self.db.connect(**connect_kwargs)
        self.addCleanup(conn.close)
        return MaterializedViewManager(conn, dialect=Dialect.SQLITE)

    def _triggers(self) -> List[tuple]:
        return self.observer.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name;"
        ).fetchall()

    def _rows(self) -> List[tuple]:
        return self.observer.execute(f"SELECT id, bucket, amt FROM {TABLE} ORDER BY id;").fetchall()

    def _assert_exact(self, view: str = "bulk_totals", live: str = LIVE_SQL) -> None:
        self.assertEqual(
            self.observer.execute(f"SELECT bucket, n, total FROM {view} ORDER BY bucket;").fetchall(),
            self.observer.execute(live).fetchall(),
        )

    def _assert_maintained_by_other_session(self, row_id: int) -> None:
        """Triggers are installed and effective for an independent writer."""
        writer = self.db.connect()
        try:
            writer.execute(f"INSERT INTO {TABLE} VALUES (?, 'z', 5);", (row_id,))
            writer.commit()
        finally:
            writer.close()
        self._assert_exact()


class SqliteBulkChangesTests(_SqliteBase):
    def test_success_is_exact_committed_and_leaves_connection_idle(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        triggers = self._triggers()
        for step, kwargs in enumerate(({}, {"use_shadow_swap": True}, {"drop_triggers": True})):
            with self.subTest(**kwargs):
                base = 10 * (step + 1)
                with manager.bulk_changes(**kwargs):
                    manager.conn.execute(f"INSERT INTO {TABLE} VALUES ({base}, 'a', 1);")
                    manager.conn.execute(f"UPDATE {TABLE} SET bucket = 'c{step}' WHERE id = 3;")
                    manager.conn.execute(f"DELETE FROM {TABLE} WHERE id = {base + 1};")
                self.assertFalse(manager.conn.in_transaction)
                self.assertIn((base, "a", 1), self._rows())
                self.assertIn((3, f"c{step}", 30), self._rows())
                self._assert_exact()
                self.assertEqual(self._triggers(), triggers)
                # The reader stays read-only (the SQLite swap re-creates the view).
                with self.assertRaisesRegex(sqlite3.DatabaseError, "read-only"):
                    self.observer.execute("INSERT INTO bulk_totals VALUES ('x', 1, 1);")
                self.observer.rollback()
                self._assert_maintained_by_other_session(base + 1)

    def test_unreadable_catalog_is_refused(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        triggers = self._triggers()
        self.observer.execute(f"ALTER TABLE {manager.catalog_table} RENAME TO hidden_catalog;")
        self.observer.commit()
        with self.assertRaisesRegex(RuntimeError, "could not be read"):
            with manager.bulk_changes():
                self.fail("entered without knowing the registered views")
        self.assertFalse(manager.conn.in_transaction)
        self.assertEqual(self._triggers(), triggers)

    def test_caller_transaction_is_refused_before_any_sql(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        triggers = self._triggers()
        conn = manager.conn
        conn.execute(f"INSERT INTO {TABLE} VALUES (20, 'c', 7);")
        statements: List[str] = []
        conn.set_trace_callback(statements.append)
        entered = False
        try:
            with self.assertRaises(TransactionOwnershipError):
                with manager.bulk_changes():
                    entered = True
        finally:
            conn.set_trace_callback(None)
        self.assertFalse(entered)
        self.assertEqual(statements, [], "bulk_changes executed SQL on a caller-owned transaction")
        self.assertTrue(conn.in_transaction, "the caller's transaction was ended")
        self.assertNotIn((20, "c", 7), self._rows(), "caller-owned work was committed")
        self.assertEqual(self._triggers(), triggers)
        conn.rollback()
        self.assertNotIn((20, "c", 7), self._rows())

    def test_unknown_transaction_state_is_refused(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        real = manager.conn

        class HidesTransactionState:
            def cursor(self) -> Any:
                raise AssertionError("SQL issued on a connection of unknown transaction state")

            def commit(self) -> None:
                real.commit()

            def rollback(self) -> None:
                real.rollback()

        manager.conn = HidesTransactionState()
        with self.assertRaises(TransactionOwnershipError):
            with manager.bulk_changes():
                self.fail("entered")

    def test_body_exception_rolls_back_everything_and_restores_triggers(self) -> None:
        for kwargs in ({}, {"use_shadow_swap": True}, {"drop_triggers": True}):
            with self.subTest(**kwargs):
                manager = self._manager()
                if not manager.get_catalog():
                    _quiet_register(manager, VIEW_SQL)
                rows, triggers = self._rows(), self._triggers()
                with self.assertRaisesRegex(ValueError, "body failed"):
                    with manager.bulk_changes(**kwargs):
                        manager.conn.execute(f"INSERT INTO {TABLE} VALUES (30, 'a', 1);")
                        raise ValueError("body failed")
                self.assertFalse(manager.conn.in_transaction)
                self.assertEqual(self._rows(), rows)
                self.assertEqual(self._triggers(), triggers)
                self._assert_exact()

    def test_base_exception_in_body_rolls_back(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        rows, triggers = self._rows(), self._triggers()
        with self.assertRaises(KeyboardInterrupt):
            with manager.bulk_changes():
                manager.conn.execute(f"INSERT INTO {TABLE} VALUES (31, 'a', 1);")
                raise KeyboardInterrupt
        self.assertFalse(manager.conn.in_transaction)
        self.assertEqual(self._rows(), rows)
        self.assertEqual(self._triggers(), triggers)
        self._assert_maintained_by_other_session(32)

    def test_restoration_failure_is_raised_and_rolled_back(self) -> None:
        """A trigger re-creation failure after the body must not be swallowed
        or committed with maintenance missing (it used to be)."""
        for position in ("middle", "end"):
            for body_fails in (False, True):
                for kwargs in ({}, {"use_shadow_swap": True}):
                    with self.subTest(position=position, body_fails=body_fails, **kwargs):
                        manager = self._manager()
                        if not manager.get_catalog():
                            _quiet_register(manager, VIEW_SQL)
                        rows, triggers = self._rows(), self._triggers()
                        with self.assertRaises(Exception) as ctx:
                            with _inject_failing_trigger_ddl(position):
                                with manager.bulk_changes(**kwargs):
                                    manager.conn.execute(f"INSERT INTO {TABLE} VALUES (40, 'a', 1);")
                                    if body_fails:
                                        raise ValueError("body failed")
                        if body_fails:
                            # The body's exception propagates; nothing is re-created.
                            self.assertIsInstance(ctx.exception, ValueError)
                        else:
                            self.assertIn(INJECTED, str(ctx.exception))
                        self.assertFalse(manager.conn.in_transaction)
                        self.assertEqual(self._rows(), rows)
                        self.assertEqual(self._triggers(), triggers)
                        self._assert_exact()
                        self._assert_maintained_by_other_session(41)
                        self.observer.execute(f"DELETE FROM {TABLE} WHERE id = 41;")
                        self.observer.commit()

    def test_unknown_view_name_is_refused_before_any_change(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        triggers = self._triggers()
        with self.assertRaises(KeyError):
            with manager.bulk_changes(view_names=["bulk_totals", "no_such_view"]):
                self.fail("entered")
        self.assertFalse(manager.conn.in_transaction)
        self.assertEqual(self._triggers(), triggers)

    def test_view_subset_keeps_other_views_maintained(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        _quiet_register(manager, OTHER_VIEW_SQL)
        with manager.bulk_changes(view_names=["bulk_totals"]):
            manager.conn.execute(f"INSERT INTO {TABLE} VALUES (50, 'b', 99);")
        self._assert_exact()
        self._assert_exact("bulk_maxes", OTHER_LIVE_SQL)

    def test_failed_rollback_invalidates_connection_and_manager(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        real = manager.conn
        triggers = self._triggers()

        class RollbackFails:
            """A connection whose rollback fails (e.g. a lost server)."""

            closed = False

            def __init__(self) -> None:
                self.dbapi_connection = real

            def cursor(self) -> Any:
                return real.cursor()

            def commit(self) -> None:
                real.commit()

            def rollback(self) -> None:
                raise sqlite3.OperationalError("injected rollback failure")

            def close(self) -> None:
                self.closed = True
                real.close()

        wrapper = RollbackFails()
        manager.conn = wrapper
        with self.assertRaises(BulkChangesConnectionError) as ctx:
            with manager.bulk_changes():
                real.execute(f"INSERT INTO {TABLE} VALUES (70, 'a', 1);")
                raise ValueError("body failed")
        self.assertIsInstance(ctx.exception.__cause__ or ctx.exception.__context__, BaseException)
        self.assertTrue(wrapper.closed, "the connection with an unknown transaction state was not closed")
        # Closing rolled the transaction back: nothing was committed and the
        # triggers are intact for every other connection.
        self.assertNotIn(70, [r[0] for r in self._rows()])
        self.assertEqual(self._triggers(), triggers)
        self._assert_maintained_by_other_session(71)
        for operation in (manager.get_catalog, manager.check_schema_drift, lambda: manager.repair_view("bulk_totals")):
            with self.assertRaises(BulkChangesConnectionError):
                operation()
        with self.assertRaises(BulkChangesConnectionError):
            with manager.bulk_changes():
                self.fail("entered on an invalidated manager")

    def test_cursor_is_closed_on_every_exit(self) -> None:
        manager = self._manager()
        _quiet_register(manager, VIEW_SQL)
        real = manager.conn
        opened: List[Any] = []

        class CountingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur
                self.closed = False

            def close(self) -> None:
                self.closed = True
                self._cur.close()

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

        class Counting:
            def __init__(self) -> None:
                self.dbapi_connection = real

            def cursor(self) -> Any:
                cur = CountingCursor(real.cursor())
                opened.append(cur)
                return cur

            def commit(self) -> None:
                real.commit()

            def rollback(self) -> None:
                real.rollback()

        manager.conn = Counting()
        scenarios: Dict[str, Callable[[], None]] = {}

        def success() -> None:
            with manager.bulk_changes():
                real.execute(f"INSERT INTO {TABLE} VALUES (80, 'a', 1);")

        def body_failure() -> None:
            with self.assertRaises(ValueError):
                with manager.bulk_changes():
                    raise ValueError("body failed")

        def refresh_failure() -> None:
            with self.assertRaises(Exception):
                with _inject_failing_trigger_ddl():
                    with manager.bulk_changes():
                        pass

        def unknown_view() -> None:
            with self.assertRaises(KeyError):
                with manager.bulk_changes(view_names=["nope"]):
                    pass

        scenarios.update(success=success, body_failure=body_failure, refresh_failure=refresh_failure, unknown_view=unknown_view)
        for label, run in scenarios.items():
            with self.subTest(scenario=label):
                opened.clear()
                run()
                self.assertTrue(all(c.closed for c in opened), f"{label}: {sum(not c.closed for c in opened)} cursor(s) left open")
                self.assertFalse(real.in_transaction)



# ---------------------------------------------------------------------------
# Native PostgreSQL and MySQL/MariaDB (independent server sessions)
# ---------------------------------------------------------------------------


def _pg_connect(dsn: str):
    import psycopg2

    return psycopg2.connect(dsn, options="-c lock_timeout=20s")


def _mysql_connect(dsn: str, autocommit: bool = False, **kwargs: Any):
    import pymysql

    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=autocommit,
        init_command="SET SESSION lock_wait_timeout = 20, SESSION innodb_lock_wait_timeout = 20",
        **kwargs,
    )


class _NativeBase:
    """Shared native tests; mixed into one TestCase per engine."""

    engine: Dialect

    def setUp(self) -> None:
        env = "POSTGRES_DSN" if self.engine == Dialect.POSTGRES else "MYSQL_DSN"
        self.dsn = os.environ.get(env)
        if not self.dsn:
            self.skipTest(f"{env} is not set; native test only")
        suffix = uuid.uuid4().hex[:10]
        self.extra_conns: List[Any] = []
        # Registered first: tearDown does not run when setUp fails half-way.
        self.addCleanup(self._drop_everything)
        if self.engine == Dialect.POSTGRES:
            self.schema = f"mv_bulk_{suffix}"
            self.table = TABLE
            self.prefix = "__matview_"
            self.conn = self._connect()
            self.observer = self._connect(autocommit=True)
            with self.conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA "{self.schema}"')
            self.conn.commit()
        else:
            self.table = f"{TABLE}_{suffix}"
            self.prefix = f"__mvb{suffix}_"
            self.conn = self._connect()
            self.observer = self._connect(autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE TABLE {self.table} (id INT PRIMARY KEY, bucket VARCHAR(8) NOT NULL, amt INT NOT NULL)")
            cur.execute(f"INSERT INTO {self.table} VALUES " + ", ".join(f"({i}, '{b}', {a})" for i, b, a in INITIAL_ROWS))
            cur.execute("SELECT version()")
            print(f"[native] {cur.fetchone()[0]}", file=sys.stderr)
        self.conn.commit()
        self.manager = MaterializedViewManager(
            self.conn, dialect=self.engine, object_prefix=self.prefix, ignore_limitations=True
        )
        self.plan = _quiet_register(self.manager, self._sql(VIEW_SQL))

    def _connect(self, autocommit: bool = False) -> Any:
        if self.engine == Dialect.POSTGRES:
            conn = _pg_connect(self.dsn)
            with conn.cursor() as cur:
                cur.execute(f'SET search_path TO "{self.schema}"')
            conn.commit()
            conn.autocommit = autocommit
        else:
            conn = _mysql_connect(self.dsn, autocommit=autocommit)
        self.extra_conns.append(conn)
        return conn

    def _sql(self, sql: str) -> str:
        return sql.replace(TABLE, self.table)

    def _drop_everything(self) -> None:
        cleanup = self._connect(autocommit=True)
        try:
            with cleanup.cursor() as cur:
                if self.engine == Dialect.POSTGRES:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
                else:
                    cur.execute(
                        "SELECT trigger_name FROM information_schema.triggers "
                        "WHERE trigger_schema = DATABASE() AND event_object_table = %s",
                        (self.table,),
                    )
                    for (name,) in cur.fetchall():
                        cur.execute(f"DROP TRIGGER IF EXISTS `{name}`")
                    cur.execute(
                        "SELECT table_name, table_type FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() AND (table_name LIKE %s OR table_name = %s)",
                        (self.prefix + "%", self.table),
                    )
                    rows = cur.fetchall()
                    for name, kind in rows:
                        if kind == "VIEW":
                            cur.execute(f"DROP VIEW IF EXISTS `{name}`")
                    for view in ("bulk_totals", "bulk_maxes"):
                        cur.execute(f"DROP VIEW IF EXISTS {view}")
                    for name, kind in rows:
                        if kind != "VIEW":
                            cur.execute(f"DROP TABLE IF EXISTS `{name}`")
        finally:
            for conn in self.extra_conns:
                try:
                    conn.close()
                except Exception:
                    pass

    # -- observation -----------------------------------------------------
    def _observed(self, sql: str, params: Any = None) -> List[tuple]:
        with self.observer.cursor() as cur:
            cur.execute(sql, params)
            return [tuple(r) for r in cur.fetchall()]

    def _rows(self) -> List[tuple]:
        return self._observed(f"SELECT id, bucket, amt FROM {self.table} ORDER BY id")

    def _maintenance_objects(self) -> List[tuple]:
        """Maintenance triggers (and PostgreSQL functions) with identities that
        change when they are re-created."""
        if self.engine == Dialect.POSTGRES:
            return self._observed(
                "SELECT t.tgname, t.oid::bigint, p.proname, p.oid::bigint FROM pg_trigger t "
                "JOIN pg_class c ON c.oid = t.tgrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "JOIN pg_proc p ON p.oid = t.tgfoid WHERE n.nspname = %s AND NOT t.tgisinternal ORDER BY 1",
                (self.schema,),
            )
        return self._observed(
            "SELECT trigger_name, action_statement, created FROM information_schema.triggers "
            "WHERE trigger_schema = DATABASE() ORDER BY trigger_name"
        )

    def _assert_exact(self, view: str = "bulk_totals", live: str = LIVE_SQL) -> None:
        self.assertEqual(
            self._observed(f"SELECT bucket, n, total FROM {view} ORDER BY bucket"),
            self._observed(self._sql(live)),
        )

    def _assert_maintained_by_other_session(self, row_id: int) -> None:
        with self.observer.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table} VALUES ({row_id}, 'z', 5)")
        self._assert_exact()
        with self.observer.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE id = {row_id}")
        self._assert_exact()

    def _in_transaction(self) -> bool:
        if self.engine == Dialect.POSTGRES:
            return self.conn.get_transaction_status() != 0
        return bool(self.conn.server_status & 0x0001)

    def _write(self, sql: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(sql)

    # -- shared native tests -----------------------------------------------
    def test_success_is_exact_and_commits(self) -> None:
        objects = self._maintenance_objects()
        with self.manager.bulk_changes():
            self._write(f"INSERT INTO {self.table} VALUES (10, 'a', 1)")
            self._write(f"UPDATE {self.table} SET bucket = 'c' WHERE id = 3")
            self._write(f"DELETE FROM {self.table} WHERE id = 1")
        self.assertFalse(self._in_transaction())
        self.assertEqual(self._rows(), [(2, "a", 20), (3, "c", 30), (10, "a", 1)])
        self._assert_exact()
        self.assertEqual(self._maintenance_objects(), objects, "maintenance objects were dropped/re-created")
        self._assert_maintained_by_other_session(11)

    def test_caller_transaction_is_refused_and_left_intact(self) -> None:
        objects = self._maintenance_objects()
        self._write(f"INSERT INTO {self.table} VALUES (20, 'c', 7)")
        modes = [{}]
        if self.engine == Dialect.POSTGRES:
            modes += [{"use_shadow_swap": True}, {"drop_triggers": True}]
        for kwargs in modes:
            with self.subTest(**kwargs):
                with self.assertRaises(TransactionOwnershipError):
                    with self.manager.bulk_changes(**kwargs):
                        self.fail("entered inside a caller-owned transaction")
                self.assertTrue(self._in_transaction(), "the caller's transaction was ended")
                self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id = 20"), [(0,)])
        self.conn.rollback()
        self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id = 20"), [(0,)])
        self.assertEqual(self._maintenance_objects(), objects)
        self._assert_exact()

    def test_body_exception_rolls_back(self) -> None:
        objects, rows = self._maintenance_objects(), self._rows()
        for exc_type in (ValueError, KeyboardInterrupt):
            with self.subTest(exc=exc_type.__name__):
                with self.assertRaises(exc_type):
                    with self.manager.bulk_changes():
                        self._write(f"INSERT INTO {self.table} VALUES (30, 'a', 1)")
                        raise exc_type("body failed")
                self.assertFalse(self._in_transaction())
                self.assertEqual(self._rows(), rows)
                self.assertEqual(self._maintenance_objects(), objects, "maintenance objects were dropped/re-created")
                self._assert_exact()
                # The connection itself keeps maintaining after the context.
                self._write(f"INSERT INTO {self.table} VALUES (31, 'q', 2)")
                self.conn.commit()
                self._assert_exact()
                self._write(f"DELETE FROM {self.table} WHERE id = 31")
                self.conn.commit()

    def test_autocommit_connection_runs_atomically(self) -> None:
        conn = self._connect(autocommit=True)
        manager = MaterializedViewManager(conn, dialect=self.engine, object_prefix=self.prefix, ignore_limitations=True)
        rows = self._rows()
        with self.assertRaises(ValueError):
            with manager.bulk_changes():
                with conn.cursor() as cur:
                    cur.execute(f"INSERT INTO {self.table} VALUES (40, 'a', 1)")
                # An independent session must not see uncommitted bulk work.
                self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id = 40"), [(0,)])
                raise ValueError("body failed")
        self.assertEqual(self._rows(), rows, "work of a failed bulk_changes was autocommitted")
        self._assert_exact()
        with manager.bulk_changes():
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {self.table} VALUES (41, 'a', 1)")
        self.assertIn((41, "a", 1), self._rows())
        self._assert_exact()
        if self.engine == Dialect.POSTGRES:
            self.assertTrue(conn.autocommit, "autocommit was not restored")
        else:
            self.assertTrue(conn.get_autocommit(), "autocommit was not restored")
        # Back in autocommit: a write by that connection is committed and maintained.
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {self.table} VALUES (42, 'b', 2)")
        self._assert_exact()

    def test_view_subset_cannot_leave_other_views_unmaintained(self) -> None:
        """The session bypass suspends every view's triggers on this session;
        a subset that would leave another view stale must be refused (or the
        other view refreshed)."""
        _quiet_register(self.manager, self._sql(OTHER_VIEW_SQL))
        try:
            with self.manager.bulk_changes(view_names=["bulk_totals"]):
                self._write(f"INSERT INTO {self.table} VALUES (50, 'b', 99)")
        except ValueError as exc:
            self.assertIn("bulk_maxes", str(exc))
            self.assertFalse(self._in_transaction())
            self.assertNotIn(50, [r[0] for r in self._rows()])
        self._assert_exact()
        self._assert_exact("bulk_maxes", OTHER_LIVE_SQL)

    def test_unreadable_catalog_is_refused(self) -> None:
        """Without the catalog the bypass would suspend maintenance of views
        nobody refreshes: refuse (it used to proceed with no views)."""
        catalog = f"{self.prefix}catalog"
        with self.observer.cursor() as cur:
            cur.execute(f"ALTER TABLE {catalog} RENAME TO {self.prefix}hidden")
        try:
            rows = self._rows()
            with self.assertRaisesRegex(RuntimeError, "could not be read"):
                with self.manager.bulk_changes():
                    self.fail("entered without knowing the registered views")
            self.assertFalse(self._in_transaction())
            self.assertEqual(self._rows(), rows)
        finally:
            with self.observer.cursor() as cur:
                cur.execute(f"ALTER TABLE {self.prefix}hidden RENAME TO {catalog}")
        self._assert_maintained_by_other_session(130)


class NativePostgresBulkChangesTests(_NativeBase, unittest.TestCase):
    engine = Dialect.POSTGRES

    def test_drop_triggers_and_shadow_swap_block_other_writers_and_are_exact(self) -> None:
        from test_avg_concurrent_maintenance import _session_id, _wait_until_lock_waiting

        for row_id, kwargs in ((70, {"drop_triggers": True}), (80, {"use_shadow_swap": True})):
            with self.subTest(**kwargs):
                writer = self._connect()
                monitor = self._connect(autocommit=True)
                writer_sid = _session_id(self.engine, writer)
                bulk_sid = _session_id(self.engine, self.conn)
                done = threading.Event()
                errors: List[BaseException] = []

                def write() -> None:
                    try:
                        with writer.cursor() as cur:
                            cur.execute(f"INSERT INTO {self.table} VALUES ({row_id + 1}, 'w', 9)")
                        writer.commit()
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()

                with self.manager.bulk_changes(**kwargs):
                    self._write(f"INSERT INTO {self.table} VALUES ({row_id}, 'a', 1)")
                    if kwargs.get("drop_triggers"):
                        # The dropped triggers are invisible to others until
                        # commit: an independent writer must wait.
                        thread = threading.Thread(target=write, daemon=True)
                        thread.start()
                        _wait_until_lock_waiting(self.engine, monitor, writer_sid, done, blocker_sid=bulk_sid)
                if kwargs.get("drop_triggers"):
                    thread.join(60)
                    self.assertEqual(errors, [])
                    self.assertIn((row_id + 1, "w", 9), self._rows())
                self.assertFalse(self._in_transaction())
                self.assertIn((row_id, "a", 1), self._rows())
                self._assert_exact()
                self._assert_maintained_by_other_session(row_id + 5)

    def test_restoration_failure_is_raised_and_rolled_back(self) -> None:
        for position in ("middle", "end"):
            for kwargs in ({"drop_triggers": True}, {"use_shadow_swap": True}):
                with self.subTest(position=position, **kwargs):
                    rows = self._rows()
                    names = [r[0] for r in self._maintenance_objects()]
                    with self.assertRaises(Exception) as ctx:
                        with _inject_failing_trigger_ddl(position):
                            with self.manager.bulk_changes(**kwargs):
                                self._write(f"INSERT INTO {self.table} VALUES (90, 'a', 1)")
                    self.assertIn(INJECTED, str(ctx.exception))
                    self.assertFalse(self._in_transaction())
                    self.assertEqual(self._rows(), rows)
                    self.assertEqual([r[0] for r in self._maintenance_objects()], names)
                    self._assert_exact()
                    self._assert_maintained_by_other_session(91)


class NativeMysqlBulkChangesTests(_NativeBase, unittest.TestCase):
    engine = Dialect.MYSQL

    def test_ddl_modes_are_refused_before_any_sql(self) -> None:
        objects = self._maintenance_objects()
        for kwargs in ({"use_shadow_swap": True}, {"drop_triggers": True}):
            with self.subTest(**kwargs):
                # Also with a caller transaction open: refused, not committed.
                self._write(f"INSERT INTO {self.table} VALUES (100, 'c', 7)")
                with self.assertRaises((ValueError, TransactionOwnershipError)):
                    with self.manager.bulk_changes(**kwargs):
                        self.fail("entered a DDL bulk mode on MySQL/MariaDB")
                self.assertEqual(self._observed(f"SELECT COUNT(*) FROM {self.table} WHERE id = 100"), [(0,)])
                self.conn.rollback()
                with self.assertRaises(ValueError) as ctx:
                    with self.manager.bulk_changes(**kwargs):
                        self.fail("entered a DDL bulk mode on MySQL/MariaDB")
                self.assertIn("implicit", str(ctx.exception))
                self.assertFalse(self._in_transaction())
        self.assertEqual(self._maintenance_objects(), objects)
        self._assert_exact()

    def test_session_flag_reset_failure_discards_the_connection(self) -> None:
        conn = self._connect()
        manager = MaterializedViewManager(conn, dialect=self.engine, object_prefix=self.prefix, ignore_limitations=True)
        flag_seen: List[Any] = []

        class ResetFailingCursor:
            def __init__(self, cur: Any) -> None:
                self._cur = cur

            def execute(self, sql: Any, *args: Any) -> Any:
                compact = str(sql).replace(" ", "").rstrip(";").upper()
                if compact.startswith("SET@__MATVIEW_SKIP_TRIGGERS=") and not compact.endswith("=1"):
                    raise RuntimeError("injected lost reset response")
                return self._cur.execute(sql, *args)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cur, name)

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: Any) -> None:
                self._cur.close()

        class ResetFailing:
            def __init__(self) -> None:
                self.dbapi_connection = conn
                self.closed = False

            def cursor(self) -> Any:
                return ResetFailingCursor(conn.cursor())

            def commit(self) -> None:
                conn.commit()

            def rollback(self) -> None:
                conn.rollback()

            def close(self) -> None:
                with conn.cursor() as cur:
                    cur.execute("SELECT @__matview_skip_triggers")
                    flag_seen.append(cur.fetchone()[0])
                self.closed = True
                conn.close()

        wrapper = ResetFailing()
        manager.conn = wrapper
        rows = self._rows()
        with self.assertRaises(BulkChangesConnectionError):
            with manager.bulk_changes():
                with conn.cursor() as cur:
                    cur.execute(f"INSERT INTO {self.table} VALUES (110, 'a', 1)")
        self.assertTrue(wrapper.closed, "a session with the bypass flag set was left usable")
        self.assertFalse(conn.open)
        # The hazard was real: the flag was still set when the session was discarded.
        self.assertEqual(flag_seen, [1])
        self.assertEqual(self._rows(), rows, "work was committed although the session flag could not be cleared")
        self._assert_exact()
        with self.assertRaises(BulkChangesConnectionError):
            with manager.bulk_changes():
                self.fail("entered on an invalidated manager")

    def test_dml_path_issues_no_ddl(self) -> None:
        """MySQL/MariaDB DDL commits implicitly: neither success nor failure
        may drop/re-create triggers (it used to on failure)."""
        objects = self._maintenance_objects()
        with self.assertRaises(ValueError):
            with self.manager.bulk_changes():
                self._write(f"INSERT INTO {self.table} VALUES (120, 'a', 1)")
                raise ValueError("body failed")
        with self.manager.bulk_changes():
            self._write(f"INSERT INTO {self.table} VALUES (121, 'a', 1)")
        self.assertEqual(self._maintenance_objects(), objects)
        with self.conn.cursor() as cur:
            cur.execute("SELECT @__matview_skip_triggers")
            self.assertIn(cur.fetchone()[0], (None, 0), "the session bypass flag was left set")
        self.conn.commit()
        self._assert_exact()


if __name__ == "__main__":
    unittest.main(verbosity=2)
