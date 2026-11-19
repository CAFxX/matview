#!/usr/bin/env python3
"""Replacing a registered materialized view never loses the published one.

Contract under test:

* SQLite and PostgreSQL replace a view (``register_view(force_rebuild=True)``
  or a changed definition in ``sync_views``) in one transaction: the old
  generation is dropped and the new one built and populated before anything
  commits. A failure at any stage (teardown, structure, population) restores
  the old reader, backing/auxiliary tables, triggers and catalog row, and the
  connection stays usable. Independent readers never observe a missing,
  empty or partially built view.
* Replacement of a complete, published view is refused, before any SQL that
  changes state, where it cannot be atomic: on MySQL/MariaDB (DDL commits
  implicitly) and with ``incremental_backfill=True`` (batches commit
  separately). ``sync_views`` checks every requested view first, so a refused
  replacement leaves the whole request unapplied. Views whose backfill is
  incomplete (nothing correct is published) can still be rebuilt that way.

SQLite runs offline (file-backed, independent connections). Native
PostgreSQL (POSTGRES_DSN) and MySQL/MariaDB (MYSQL_DSN) tests skip without
a DSN.
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
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from matview_sync import (
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    MaterializedViewPlan,
    UnsafeReplacementError,
)

V1 = "CREATE VIEW replaced_totals AS SELECT bucket, COUNT(*) AS n, SUM(amt) AS total FROM replaced_rows GROUP BY bucket;"
V2 = "CREATE VIEW replaced_totals AS SELECT bucket, COUNT(*) AS n, SUM(amt) AS total FROM replaced_rows WHERE amt >= 20 GROUP BY bucket;"
LIVE_V1 = "SELECT bucket, COUNT(*), SUM(amt) FROM replaced_rows GROUP BY bucket ORDER BY bucket"
LIVE_V2 = "SELECT bucket, COUNT(*), SUM(amt) FROM replaced_rows WHERE amt >= 20 GROUP BY bucket ORDER BY bucket"
READ = "SELECT bucket, n, total FROM replaced_totals ORDER BY bucket"


class _PopulationFailure(RuntimeError):
    pass


class _inject_population_failure:
    """Fail when a registration reaches its initial population.

    At that point a replacement has dropped the old generation and installed
    the new structure, triggers and catalog row.
    """

    def __init__(self, hook: Optional[Callable[[], None]] = None, fail: bool = True) -> None:
        self.hook = hook
        self.fail = fail
        self.reached = False

    def __enter__(self) -> "_inject_population_failure":
        self.original = MaterializedViewPlan.get_populate_statements
        outer = self

        def patched(plan_self: Any, *args: Any, **kwargs: Any) -> Any:
            outer.reached = True
            if outer.hook is not None:
                outer.hook()
            if outer.fail:
                raise _PopulationFailure("injected replacement population failure")
            return outer.original(plan_self, *args, **kwargs)

        MaterializedViewPlan.get_populate_statements = patched
        return self

    def __exit__(self, *exc: Any) -> None:
        MaterializedViewPlan.get_populate_statements = self.original


class _FailingCatalogDeleteConnection:
    """Proxy whose cursors fail the catalog DELETE of a teardown.

    The failure happens after the reader view, backing and auxiliary tables
    (and on SQLite the triggers) of the old generation were dropped.
    """

    def __init__(self, raw: Any, catalog_table: str) -> None:
        self._raw = raw
        self._needle = f"DELETE FROM {catalog_table}".upper()
        self.dropped: List[str] = []
        self.injected = False

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        outer = self
        raw_cursor = self._raw.cursor(*args, **kwargs)

        class _Cursor:
            def execute(self, sql: Any, params: Any = None) -> Any:
                text = " ".join(str(sql).split()).upper()
                if text.startswith("DROP "):
                    outer.dropped.append(text)
                if not outer.injected and text.startswith(outer._needle):
                    outer.injected = True
                    raise RuntimeError("injected failure after reader/backing drop")
                return raw_cursor.execute(sql) if params is None else raw_cursor.execute(sql, params)

            def __getattr__(self, name: str) -> Any:
                return getattr(raw_cursor, name)

        return _Cursor()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


class _ReplacementScenarios:
    """Engine-independent scenarios; subclasses provide the fixture."""

    dialect: Dialect

    # -- fixture interface -------------------------------------------------
    def connect(self) -> Any:
        raise NotImplementedError

    def query(self, conn: Any, sql: str) -> List[tuple]:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            rows = [tuple(r) for r in cur.fetchall()]
        finally:
            cur.close()
        conn.commit()
        return rows

    def execute(self, conn: Any, sql: str) -> None:
        cur = conn.cursor()
        try:
            cur.execute(sql)
        finally:
            cur.close()
        conn.commit()

    def installed_objects(self) -> List[tuple]:
        raise NotImplementedError

    def manager(self, conn: Any) -> MaterializedViewManager:
        return MaterializedViewManager(conn, dialect=self.dialect)

    # -- helpers -------------------------------------------------------------
    def published_state(self, manager: MaterializedViewManager) -> Dict[str, Any]:
        observer = self.observer
        catalog = manager.get_catalog()
        return {
            "rows": self.query(observer, READ),
            "catalog": {k: (v["signature_hash"], v["backing_table"], v["trigger_names"]) for k, v in catalog.items()},
            "objects": self.installed_objects(),
        }

    def assert_old_generation_intact(self, manager: MaterializedViewManager, before: Dict[str, Any]) -> None:
        self.assertEqual(self.published_state(manager), before)
        # ...and still maintained by its triggers, through the same connection.
        self.execute(manager.conn, "INSERT INTO replaced_rows VALUES (100, 'b', 5)")
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V1))
        self.execute(manager.conn, "DELETE FROM replaced_rows WHERE id = 100")

    # -- scenarios -----------------------------------------------------------
    def run_population_failure(self, sync: bool) -> None:
        conn = self.conn
        manager = self.manager(conn)
        manager.register_view(V1)
        before = self.published_state(manager)
        with _inject_population_failure() as inj:
            with self.assertRaises(_PopulationFailure):
                if sync:
                    manager.sync_views([V2])
                else:
                    manager.register_view(V2, force_rebuild=True)
        self.assertTrue(inj.reached)
        self.assert_old_generation_intact(manager, before)
        # The connection is usable and a later replacement succeeds.
        status = manager.sync_views([V2])
        self.assertEqual(status, {"replaced_totals": "updated"})
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V2))

    def run_teardown_failure(self) -> None:
        manager = self.manager(self.conn)
        manager.register_view(V1)
        before = self.published_state(manager)
        proxy = _FailingCatalogDeleteConnection(self.conn, manager.catalog_table)
        manager.conn = proxy
        try:
            with self.assertRaisesRegex(RuntimeError, "injected failure after reader/backing drop"):
                manager.register_view(V2, force_rebuild=True)
        finally:
            manager.conn = self.conn
        self.assertTrue(proxy.injected, "fault injection did not reach the catalog cleanup")
        self.assertTrue(any(d.startswith("DROP VIEW") for d in proxy.dropped), proxy.dropped)
        self.assertTrue(any(d.startswith("DROP TABLE") for d in proxy.dropped), proxy.dropped)
        self.assert_old_generation_intact(manager, before)

    def run_independent_reader_never_sees_missing_view(self) -> None:
        manager = self.manager(self.conn)
        manager.register_view(V1)
        old_rows = self.query(self.observer, READ)
        outcome: List[Any] = []

        def read_now() -> None:
            # Runs while the replacement transaction has dropped the old
            # generation and built (unpopulated) the new one.
            outcome.append(self.read_during_replacement())

        with _inject_population_failure(hook=read_now, fail=False):
            manager.register_view(V2, force_rebuild=True)
        self.assertEqual(len(outcome), 1)
        self.check_read_during_replacement(outcome[0], old_rows)
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V2))

    def run_refusal(self, operation: Callable[[MaterializedViewManager], Any], manager: MaterializedViewManager) -> None:
        before = self.published_state(manager)
        with self.assertRaises(UnsafeReplacementError) as ctx:
            operation(manager)
        self.assertIn("drop_view()", str(ctx.exception))
        self.assertEqual(self.published_state(manager), before)


class SqliteReplacementTests(_ReplacementScenarios, unittest.TestCase):
    dialect = Dialect.SQLITE

    def setUp(self) -> None:
        fd, self.path = tempfile.mkstemp(prefix="matview-replace-", suffix=".sqlite")
        os.close(fd)
        setup = sqlite3.connect(self.path)
        setup.execute("PRAGMA journal_mode=WAL;")
        setup.execute("CREATE TABLE replaced_rows (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);")
        setup.executemany("INSERT INTO replaced_rows VALUES (?, ?, ?);", [(1, "a", 10), (2, "a", 20), (3, "b", 30), (4, "c", 5)])
        setup.commit()
        setup.close()
        self.conn = self.connect()
        self.observer = self.connect()

    def tearDown(self) -> None:
        self.conn.close()
        self.observer.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suffix)
            except FileNotFoundError:
                pass

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000;")
        return conn

    def installed_objects(self) -> List[tuple]:
        return self.query(self.observer, "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name")

    def read_during_replacement(self) -> Any:
        reader = self.connect()
        try:
            return ("rows", reader.execute(READ).fetchall())
        except sqlite3.Error as exc:
            return ("error", str(exc))
        finally:
            reader.close()

    def check_read_during_replacement(self, outcome: Any, old_rows: List[tuple]) -> None:
        # WAL: the reader sees the last committed state, i.e. the old view.
        self.assertEqual(outcome, ("rows", old_rows))

    def test_population_failure_restores_prior_view(self) -> None:
        self.run_population_failure(sync=False)

    def test_sync_views_population_failure_restores_prior_view(self) -> None:
        self.run_population_failure(sync=True)

    def test_teardown_failure_restores_prior_view(self) -> None:
        self.run_teardown_failure()

    def test_independent_reader_sees_prior_view_during_replacement(self) -> None:
        self.run_independent_reader_never_sees_missing_view()

    def test_detects_a_committing_teardown(self) -> None:
        """Sensitivity: the population-failure check catches a teardown that commits.

        The mutant commits right after dropping the old generation (the
        behavior before atomic replacement). The scenario must then fail.
        """
        original = MaterializedViewManager._drop_registered_objects

        def committing_teardown(manager_self: MaterializedViewManager, *args: Any, **kwargs: Any) -> Any:
            result = original(manager_self, *args, **kwargs)
            manager_self._commit()
            return result

        MaterializedViewManager._drop_registered_objects = committing_teardown
        try:
            with self.assertRaises(AssertionError):
                self.run_population_failure(sync=False)
        finally:
            MaterializedViewManager._drop_registered_objects = original

    def test_incremental_replacement_of_published_view_is_refused(self) -> None:
        manager = self.manager(self.conn)
        manager.register_view(V1)
        self.run_refusal(lambda m: m.register_view(V2, force_rebuild=True, incremental_backfill=True), manager)
        self.run_refusal(lambda m: m.sync_views([V2], incremental_backfill=True), manager)

    def test_sync_views_refuses_before_applying_any_view(self) -> None:
        manager = self.manager(self.conn)
        manager.register_view(V1)
        other = "CREATE VIEW other_totals AS SELECT bucket, COUNT(*) AS n FROM replaced_rows GROUP BY bucket;"
        statements: List[str] = []
        self.conn.set_trace_callback(statements.append)
        try:
            self.run_refusal(lambda m: m.sync_views([other, V2], incremental_backfill=True), manager)
        finally:
            self.conn.set_trace_callback(None)
        writes = [s for s in statements if s.lstrip().upper().startswith(("CREATE", "DROP", "INSERT", "DELETE", "UPDATE", "BEGIN"))]
        self.assertEqual(writes, [], "a refused sync_views changed the database")
        self.assertNotIn("other_totals", manager.get_catalog())

    def test_cli_reports_refusals_with_distinct_exit_codes(self) -> None:
        import contextlib
        import io

        from matview.cli import EXIT_TRANSACTION_OWNERSHIP, EXIT_UNSAFE_REPLACEMENT, main
        from matview_sync import TransactionOwnershipError

        manager = self.manager(self.conn)
        manager.register_view(V1)
        before = self.published_state(manager)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--db", self.path, "--view", V2, "--apply", "--incremental-backfill"])
        self.assertEqual(code, EXIT_UNSAFE_REPLACEMENT)
        self.assertIn("Refusing to replace materialized view", err.getvalue())
        self.assertIn("drop_view", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        self.assertEqual(self.published_state(manager), before)

        original = MaterializedViewManager.register_view

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise TransactionOwnershipError("injected ownership refusal")

        MaterializedViewManager.register_view = refuse  # type: ignore[assignment]
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main(["--db", self.path, "--view", V2, "--apply"])
        finally:
            MaterializedViewManager.register_view = original  # type: ignore[assignment]
        self.assertEqual(code, EXIT_TRANSACTION_OWNERSHIP)
        self.assertIn("injected ownership refusal", err.getvalue())
        self.assertNotEqual(EXIT_UNSAFE_REPLACEMENT, EXIT_TRANSACTION_OWNERSHIP)

    def test_incomplete_view_can_still_be_rebuilt_incrementally(self) -> None:
        manager = self.manager(self.conn)

        def interrupt(batch: int, done: int) -> None:
            if batch == 2:
                raise KeyboardInterrupt("injected interruption between batches")

        with self.assertRaises(KeyboardInterrupt):
            manager.register_view(V1, incremental_backfill=True, batch_size=1, on_progress=interrupt)
        entry = manager.get_catalog()["replaced_totals"]
        self.assertTrue(manager._is_backfill_incomplete(entry), entry)
        manager.register_view(V2, force_rebuild=True, incremental_backfill=True, batch_size=1)
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V2))


def _mysql_connect(dsn: str) -> Any:
    import pymysql

    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=False,
        # A metadata lock left held by a broken implementation fails the
        # test instead of hanging it (the server default is a year).
        init_command="SET SESSION lock_wait_timeout = 20",
    )


class NativePostgresReplacementTests(_ReplacementScenarios, unittest.TestCase):
    dialect = Dialect.POSTGRES

    def setUp(self) -> None:
        self.dsn = os.environ.get("POSTGRES_DSN")
        if not self.dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self.schema = "mv_repl_" + uuid.uuid4().hex[:12]
        self.conn = self.connect()
        cur = self.conn.cursor()
        cur.execute(f'CREATE SCHEMA "{self.schema}"')
        cur.execute(f'SET search_path TO "{self.schema}"')
        cur.execute("CREATE TABLE replaced_rows (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL)")
        cur.execute("INSERT INTO replaced_rows VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30), (4, 'c', 5)")
        cur.execute("SELECT version()")
        print(f"[native] {cur.fetchone()[0]}", file=sys.stderr)
        cur.close()
        self.conn.commit()
        self.observer = self.connect()

    def tearDown(self) -> None:
        try:
            self.conn.rollback()
            self.execute(self.conn, f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        finally:
            self.conn.close()
            self.observer.close()

    def connect(self) -> Any:
        import psycopg2

        # A lock left held by a broken implementation fails the test instead
        # of hanging it.
        conn = psycopg2.connect(self.dsn, options="-c lock_timeout=20s")
        if getattr(self, "schema", None):
            cur = conn.cursor()
            cur.execute(f'SET search_path TO "{self.schema}"')
            cur.close()
            conn.commit()
        return conn

    def manager(self, conn: Any) -> MaterializedViewManager:
        return MaterializedViewManager(conn, dialect=Dialect.POSTGRES, object_prefix=f"__mvr{self.schema[-6:]}_")

    def installed_objects(self) -> List[tuple]:
        return self.query(
            self.observer,
            "SELECT 'rel', c.relname, c.relkind::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE n.nspname = '{self.schema}' "
            "UNION ALL SELECT 'trigger', t.tgname, c.relname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            f"JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = '{self.schema}' AND NOT t.tgisinternal "
            "UNION ALL SELECT 'function', p.proname, md5(p.prosrc) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            f"WHERE n.nspname = '{self.schema}' ORDER BY 1, 2, 3",
        )

    def read_during_replacement(self) -> Any:
        # Start an independent reader and require it to wait on the
        # replacement's lock (the old reader view is being dropped).
        reader = self.connect()
        pid = reader.get_backend_pid()
        result: List[Any] = []

        def run() -> None:
            try:
                cur = reader.cursor()
                cur.execute(READ)
                result.append(("rows", [tuple(r) for r in cur.fetchall()]))
                reader.commit()
            except Exception as exc:  # inspected by the test
                result.append(("error", str(exc)))
                reader.rollback()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        state = "timeout"
        while time.monotonic() < deadline:
            if not thread.is_alive():
                state = "finished during replacement"
                break
            rows = self.query(self.observer, f"SELECT wait_event_type FROM pg_stat_activity WHERE pid = {pid}")
            if rows and rows[0][0] == "Lock":
                state = "waiting on lock"
                break
            time.sleep(0.01)
        self._reader = (thread, result, reader)
        return state

    def check_read_during_replacement(self, outcome: Any, old_rows: List[tuple]) -> None:
        thread, result, reader = self._reader
        thread.join(10)
        reader.close()
        self.assertEqual(outcome, "waiting on lock")
        # Once the replacement commits the reader gets the new view's rows.
        self.assertEqual(result, [("rows", self.query(self.observer, LIVE_V2))])

    def test_population_failure_restores_prior_view(self) -> None:
        self.run_population_failure(sync=False)

    def test_sync_views_population_failure_restores_prior_view(self) -> None:
        self.run_population_failure(sync=True)

    def test_engine_error_during_population_restores_prior_view(self) -> None:
        # The failure is raised by PostgreSQL itself inside the replacement
        # transaction (aborted-transaction state), not by Python code.
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE fail_switch (on_ BOOLEAN NOT NULL)")
        cur.execute("INSERT INTO fail_switch VALUES (FALSE)")
        cur.execute(
            "CREATE FUNCTION maybe_fail(v INTEGER) RETURNS INTEGER LANGUAGE plpgsql IMMUTABLE AS $$ "
            "BEGIN IF (SELECT on_ FROM fail_switch) THEN RAISE EXCEPTION 'injected engine failure'; END IF; RETURN v; END $$"
        )
        cur.close()
        self.conn.commit()
        manager = self.manager(self.conn)
        manager.register_view(V1)
        before = self.published_state(manager)
        self.execute(self.conn, "UPDATE fail_switch SET on_ = TRUE")
        with self.assertRaisesRegex(Exception, "injected engine failure"):
            # SUM over a function result is not provably exact: per-code opt-in.
            manager.register_view(
                V1.replace("SUM(amt)", "SUM(maybe_fail(amt))"), force_rebuild=True,
                accept_limitations=["ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"],
            )
        self.execute(self.conn, "UPDATE fail_switch SET on_ = FALSE")
        self.assert_old_generation_intact(manager, before)

    def test_teardown_failure_restores_prior_view(self) -> None:
        self.run_teardown_failure()

    def test_independent_reader_never_sees_missing_view(self) -> None:
        self.run_independent_reader_never_sees_missing_view()

    def test_autocommit_connection_replacement_is_atomic(self) -> None:
        self.conn.autocommit = True
        self.run_population_failure(sync=False)
        self.assertIs(self.conn.autocommit, True)

    def test_incremental_replacement_of_published_view_is_refused(self) -> None:
        manager = self.manager(self.conn)
        manager.register_view(V1)
        self.run_refusal(lambda m: m.register_view(V2, force_rebuild=True, incremental_backfill=True), manager)


class NativeMysqlReplacementTests(_ReplacementScenarios, unittest.TestCase):
    dialect = Dialect.MYSQL

    def setUp(self) -> None:
        self.dsn = os.environ.get("MYSQL_DSN")
        if not self.dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        suffix = uuid.uuid4().hex[:10]
        self.prefix = f"__mvr{suffix}_"
        self.conn = self.connect()
        self.observer = self.connect()
        # Unique base table per run; view SQL is rewritten to use it.
        self.table = f"replaced_rows_{suffix}"
        cur = self.conn.cursor()
        cur.execute(f"CREATE TABLE {self.table} (id INT PRIMARY KEY, bucket VARCHAR(8) NOT NULL, amt INT NOT NULL) ENGINE=InnoDB")
        cur.execute(f"INSERT INTO {self.table} VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30), (4, 'c', 5)")
        cur.execute("SELECT VERSION()")
        print(f"[native] {cur.fetchone()[0]}", file=sys.stderr)
        cur.close()
        self.conn.commit()

    def tearDown(self) -> None:
        try:
            self.conn.rollback()
            manager = MaterializedViewManager(self.conn, dialect=Dialect.MYSQL, object_prefix=self.prefix)
            for name in list(manager.get_catalog()):
                manager.drop_view(name)
            self.execute(self.conn, f"DROP TABLE IF EXISTS {self.table}")
            self.execute(self.conn, f"DROP TABLE IF EXISTS {self.prefix}catalog")
        finally:
            self.conn.close()
            self.observer.close()

    def connect(self) -> Any:
        return _mysql_connect(self.dsn)

    def sql(self, text: str) -> str:
        return text.replace("replaced_rows", self.table)

    def query(self, conn: Any, sql: str) -> List[tuple]:
        return super().query(conn, self.sql(sql))

    def execute(self, conn: Any, sql: str) -> None:
        super().execute(conn, self.sql(sql))

    def manager(self, conn: Any) -> MaterializedViewManager:
        # ignore_limitations: the unrelated MYSQL_NO_TRUNCATE_TRIGGER
        # limitation (TRUNCATE bypasses triggers) is not exercised here.
        return MaterializedViewManager(conn, dialect=Dialect.MYSQL, object_prefix=self.prefix, ignore_limitations=True)

    def installed_objects(self) -> List[tuple]:
        return self.query(
            self.observer,
            "SELECT 'table', table_name, table_type FROM information_schema.tables "
            f"WHERE table_schema = DATABASE() AND (table_name LIKE '{self.prefix}%' OR table_name = 'replaced_totals') "
            "UNION ALL SELECT 'trigger', trigger_name, action_statement FROM information_schema.triggers "
            f"WHERE trigger_schema = DATABASE() AND event_object_table = '{self.table}' ORDER BY 1, 2",
        )

    def test_replacement_of_published_view_is_refused_before_any_ddl(self) -> None:
        manager = self.manager(self.conn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            manager.register_view(self.sql(V1))
            for label, operation in (
                ("register_view(force_rebuild)", lambda m: m.register_view(self.sql(V1), force_rebuild=True)),
                ("register_view(changed)", lambda m: m.register_view(self.sql(V2))),
                ("sync_views(changed)", lambda m: m.sync_views([self.sql(V2)])),
                ("register_view(incremental)", lambda m: m.register_view(self.sql(V2), incremental_backfill=True)),
            ):
                with self.subTest(operation=label):
                    self.run_refusal(operation, manager)
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V1))

    def _columns(self) -> List[str]:
        # _ReplacementScenarios.query: no rewrite (the name is already the real table).
        return [r[0] for r in _ReplacementScenarios.query(
            self,
            self.observer,
            "SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() "
            f"AND table_name = '{self.table}' ORDER BY ordinal_position",
        )]

    def test_apply_schema_ddl_refuses_before_running_any_ddl(self) -> None:
        manager = self.manager(self.conn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            manager.register_view(self.sql(V1))
            before = self.published_state(manager)
            columns = self._columns()
            self.assertEqual(columns, ["id", "bucket", "amt"])
            with self.assertRaises(UnsafeReplacementError) as ctx:
                manager.apply_schema_ddl(f"ALTER TABLE {self.table} ADD COLUMN extra INT")
            self.assertIn("replaced_totals", str(ctx.exception))
            self.assertIn("Nothing was executed", str(ctx.exception))
            self.assertEqual(self._columns(), columns, "the DDL ran before the refusal")
            self.assertEqual(self.published_state(manager), before)
            # DDL on tables no registered view uses is applied as before.
            unrelated = f"{self.prefix}unrelated"
            manager.apply_schema_ddl(f"CREATE TABLE {unrelated} (id INT PRIMARY KEY)")
            self.execute(self.conn, f"DROP TABLE {unrelated}")
            self.assertEqual(self.published_state(manager), before)
            # The supported recipe: drop, apply the DDL, register again.
            manager.drop_view("replaced_totals")
            manager.apply_schema_ddl(f"ALTER TABLE {self.table} ADD COLUMN extra INT")
            manager.register_view(self.sql(V1))
        self.assertIn("extra", self._columns())
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V1))
        self.assertEqual(manager.check_schema_drift(), {"replaced_totals": "ok"})

    def _ddl_snapshot(self, manager: MaterializedViewManager) -> Any:
        q = lambda sql: _ReplacementScenarios.query(self, self.observer, sql)  # noqa: E731 (no table rewrite)
        return (
            self.published_state(manager),
            self._columns(),
            q(
                "SELECT index_name, column_name FROM information_schema.statistics WHERE table_schema = DATABASE() "
                f"AND table_name = '{self.table}' ORDER BY index_name, seq_in_index"
            ),
            q(f"SELECT COUNT(*) FROM {self.table}"),
            q(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() "
                f"AND (table_name LIKE '{self.table}%' OR table_name LIKE '{self.prefix}%') ORDER BY table_name"
            ),
        )

    def test_apply_schema_ddl_refuses_every_ddl_kind_on_a_view_base_table(self) -> None:
        """Pins the pre-check's classification: each statement is refused unexecuted."""
        manager = self.manager(self.conn)
        database = _ReplacementScenarios.query(self, self.observer, "SELECT DATABASE()")[0][0]
        other = f"{self.prefix}other"
        self.execute(self.conn, f"CREATE TABLE {other} (id INT PRIMARY KEY, x INT) ENGINE=InnoDB")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            manager.register_view(self.sql(V1))
        before = self._ddl_snapshot(manager)
        cases = [
            ("CREATE INDEX", f"CREATE INDEX ix_bucket ON {self.table} (bucket)"),
            ("ALTER DROP COLUMN", f"ALTER TABLE {self.table} DROP COLUMN amt"),
            ("ALTER ADD COLUMN, quoted", f"ALTER TABLE `{self.table}` ADD COLUMN z INT"),
            ("schema-qualified", f"ALTER TABLE `{database}`.{self.table} ADD COLUMN w INT"),
            ("TRUNCATE", f"TRUNCATE TABLE {self.table}"),
            # sqlglot cannot parse these (they fall back to an opaque command
            # with no table): refused without knowing the target (fail closed).
            ("RENAME TABLE (unparsed)", f"RENAME TABLE {self.table} TO {self.table}_renamed"),
            ("ALTER CONVERT (unparsed)", f"ALTER TABLE {self.table} CONVERT TO CHARACTER SET utf8mb4"),
            ("unparsed, other table", f"RENAME TABLE {other} TO {other}2"),
            ("multi-statement, view table second", f"ALTER TABLE {other} ADD COLUMN q INT; ALTER TABLE {self.table} ADD COLUMN q INT"),
        ]
        for label, ddl in cases:
            with self.subTest(ddl=label):
                with self.assertRaises(UnsafeReplacementError) as ctx:
                    manager.apply_schema_ddl(ddl)
                message = str(ctx.exception)
                self.assertIn("replaced_totals", message)
                self.assertIn("Nothing was executed", message)
                self.assertIn("drop_view()", message)
                self.assertIn("register_view()", message)
                self.assertEqual(self._ddl_snapshot(manager), before, f"{label}: DDL ran or state changed")
                other_cols = _ReplacementScenarios.query(
                    self, self.observer,
                    "SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() "
                    f"AND table_name = '{other}' ORDER BY ordinal_position",
                )
                self.assertEqual(other_cols, [("id",), ("x",)], f"{label}: statement on the other table ran")
        # Control: DDL that only touches a table no registered view uses runs.
        manager.apply_schema_ddl(f"ALTER TABLE {other} ADD COLUMN q INT")
        self.assertIn(("q",), _ReplacementScenarios.query(
            self, self.observer,
            f"SELECT column_name FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = '{other}'",
        ))
        self.assertEqual(self._ddl_snapshot(manager), before)
        self.assertEqual(manager.check_schema_drift(), {"replaced_totals": "ok"})
        self.execute(self.conn, f"DROP TABLE {other}")

    def test_explicit_drop_then_register_still_replaces(self) -> None:
        manager = self.manager(self.conn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan = manager.register_view(self.sql(V1))
            self.assertTrue(manager.drop_view(plan.view_name))
            manager.register_view(self.sql(V2))
        self.assertEqual(self.query(self.observer, READ), self.query(self.observer, LIVE_V2))


if __name__ == "__main__":
    unittest.main()
