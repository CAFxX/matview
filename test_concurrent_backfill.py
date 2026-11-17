#!/usr/bin/env python3
"""Independent-session tests for incremental materialized-view backfill."""

from __future__ import annotations

from decimal import Decimal

import os
import sqlite3
import tempfile
import threading
import time
import unittest
import warnings
import uuid
from urllib.parse import urlparse

from matview_sync import Dialect, EngineLimitationWarning, MaterializedViewManager
from test_matview_sync import _register_sqlite_compat_udfs

# Bound for replaying an application writer transaction chosen as an engine
# deadlock/serialization victim. Exhausting it fails the test.
TRANSIENT_RETRIES = 25


def _is_transient_lock_error(engine: Dialect, exc: BaseException) -> bool:
    """Match only engine-reported deadlock/serialization victims, independently
    of the implementation's own classifier."""
    if engine == Dialect.MYSQL:
        code = getattr(exc, "errno", None)
        if code is None and getattr(exc, "args", None):
            code = exc.args[0]
        return code in (1213, 1205)
    if engine == Dialect.POSTGRES:
        return (getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)) in ("40P01", "40001")
    return False


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
                    # The first batch ends at PK 3 and the captured upper
                    # watermark is PK 10. Exercise both inclusive boundaries:
                    # the cursor row must still receive trigger deltas, while
                    # the maximum row must remain owned by the pending backfill.
                    second.execute("UPDATE ledger SET amt = 777 WHERE id = 3;")
                    second.execute("UPDATE ledger SET amt = 888 WHERE id = 9;")
                    second.execute("UPDATE ledger SET amt = 666 WHERE id = 10;")
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
                "SELECT bucket, c, s FROM ledger_totals ORDER BY bucket;"
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


    def _pk_range_fault_fixture(self, fail_batches: dict[int, int]):
        """Register a pk-range-fold view whose chunk raises an injected victim
        error *after* folding, so a retry without rollback would double-count."""
        conn = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn)
        conn.execute("CREATE TABLE fr (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);")
        conn.executemany("INSERT INTO fr VALUES (?, ?, ?);", [(i, f"b{i % 3}", i * 7) for i in range(1, 13)])
        conn.commit()
        remaining = dict(fail_batches)  # lower PK bound -> injected failures left

        class InjectedVictim(Exception):
            pass

        def fault(lo: int, _hi: int) -> int:
            if remaining.get(lo, 0) > 0:
                remaining[lo] -= 1
                raise InjectedVictim(lo)
            return 0

        conn.create_function("__mv_test_fault", 2, fault)
        manager = MaterializedViewManager(conn)
        plan = manager.register_view(
            "CREATE VIEW fr_totals AS SELECT bucket, COUNT(*) AS c, SUM(amt) AS s FROM fr GROUP BY bucket;",
            populate=False,
        )
        spec = plan.incremental_backfill_spec
        self.assertEqual(spec.mode, "pk_range_fold")
        spec.chunk_sql_templates.append("SELECT __mv_test_fault(?, ?);")
        # Treat only the injected victim (surfaced by sqlite3 as an
        # OperationalError from the UDF) as transient, as an engine deadlock is.
        manager._is_transient_lock_error = (
            lambda exc: isinstance(exc, sqlite3.OperationalError)
            and "user-defined function raised exception" in str(exc)
        )
        return conn, manager, plan, remaining

    def _assert_equivalent(self, conn, plan) -> None:
        actual = conn.execute(f"SELECT bucket, c, s FROM {plan.view_name} ORDER BY bucket;").fetchall()
        expected = conn.execute(
            "SELECT bucket, COUNT(*), SUM(amt) FROM fr GROUP BY bucket ORDER BY bucket;"
        ).fetchall()
        self.assertEqual(actual, expected)

    def test_pk_range_batch_retries_transient_victim_exactly_once_per_fold(self) -> None:
        conn, manager, plan, remaining = self._pk_range_fault_fixture({4: 2, 12: 0})
        sleeps: list[float] = []
        manager._backfill_sleep = sleeps.append
        try:
            batches = manager.backfill_incrementally(plan, batch_size=4)
            self.assertEqual(batches, 3)
            self.assertEqual(remaining[4], 0, "injected victims were not exercised")
            self.assertEqual(len(sleeps), 2, "each retry must be preceded by a backoff")
            self._assert_equivalent(conn, plan)
            self.assertEqual(manager.get_catalog()[plan.view_name]["backfill_active"], 0)
        finally:
            conn.close()

    class _FixedRng:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self) -> float:
            return self.value

    def test_pk_range_batch_retry_is_bounded(self) -> None:
        limit = MaterializedViewManager._BACKFILL_TRANSIENT_RETRIES
        conn, manager, plan, remaining = self._pk_range_fault_fixture({4: limit + 1})
        sleeps: list[float] = []
        manager._backfill_sleep = sleeps.append
        manager._backfill_rng = self._FixedRng(1.0)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                manager.backfill_incrementally(plan, batch_size=4)
            self.assertEqual(remaining[4], 0)
            # Exactly one backoff between consecutive attempts, exponential and
            # capped (RNG pinned to the top of the jitter interval).
            base = MaterializedViewManager._BACKFILL_RETRY_BASE_DELAY_S
            cap = MaterializedViewManager._BACKFILL_RETRY_MAX_DELAY_S
            self.assertEqual(sleeps, [min(cap, base * 2 ** i) for i in range(limit)])
        finally:
            conn.close()

    def test_backoff_delay_is_jittered_within_bounds(self) -> None:
        manager = MaterializedViewManager(sqlite3.connect(":memory:"))
        base = MaterializedViewManager._BACKFILL_RETRY_BASE_DELAY_S
        cap = MaterializedViewManager._BACKFILL_RETRY_MAX_DELAY_S
        try:
            for attempt in range(1, 12):
                ceiling = min(cap, base * 2 ** (attempt - 1))
                manager._backfill_rng = self._FixedRng(0.0)
                self.assertEqual(manager._backfill_retry_delay(attempt), ceiling / 2)
                manager._backfill_rng = self._FixedRng(0.5)
                self.assertEqual(manager._backfill_retry_delay(attempt), ceiling * 0.75)
        finally:
            manager.conn.close()

    def test_pk_range_batch_non_transient_error_is_not_retried(self) -> None:
        conn, manager, plan, remaining = self._pk_range_fault_fixture({4: 1})
        manager._is_transient_lock_error = lambda exc: False
        sleeps: list[float] = []
        manager._backfill_sleep = sleeps.append
        try:
            with self.assertRaises(sqlite3.OperationalError):
                manager.backfill_incrementally(plan, batch_size=4)
            self.assertEqual(sleeps, [])
        finally:
            conn.close()

    def test_snapshot_isolated_non_pk_batches_block_independent_writes_until_commit(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="matview-snapshot-backfill-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE snapshot_rows (id INTEGER, bucket TEXT NOT NULL, value INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO snapshot_rows VALUES (?, ?, ?);",
            [(i, f"b{i % 3}", i * 10) for i in range(1, 10)],
        )
        conn.commit()
        manager = MaterializedViewManager(conn)
        cases = (
            (
                "CREATE VIEW snapshot_grouped AS "
                "SELECT bucket, COUNT(DISTINCT value) AS n "
                "FROM snapshot_rows GROUP BY bucket;",
                "group_key_batch",
                "SELECT bucket, COUNT(DISTINCT value) AS n FROM snapshot_rows GROUP BY bucket;",
                "INSERT INTO snapshot_rows VALUES (10, 'b1', 100);",
            ),
            (
                "CREATE VIEW snapshot_ranked AS "
                "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS row_num "
                "FROM snapshot_rows;",
                "offset_batch",
                "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS row_num FROM snapshot_rows;",
                "INSERT INTO snapshot_rows VALUES (11, 'b2', 110);",
            ),
        )
        try:
            for view_sql, expected_mode, live_sql, mutation_sql in cases:
                with self.subTest(mode=expected_mode):
                    writer_started = threading.Event()
                    statement_started = threading.Event()
                    # sqlite3 may invoke the trace callback repeatedly while the same
                    # top-level statement is replayed through trigger programs. Track
                    # distinct SQL statements, not callback invocations.
                    traced_statements: set[str] = set();
                    writer_finished = threading.Event()
                    writer_errors: list[BaseException] = []
                    writer_thread_ids: list[int] = []
                    writer_thread: threading.Thread | None = None

                    def write_from_independent_session() -> None:
                        second = sqlite3.connect(db_path, timeout=10)
                        try:
                            second.execute("PRAGMA busy_timeout=10000;")
                            _register_sqlite_compat_udfs(second)

                            # sqlite3's trace callback fires when the INSERT is
                            # actually handed to SQLite, before lock waiting. This
                            # distinguishes a blocked statement from a worker that
                            # was merely scheduled but had not reached execute().
                            def trace_statement(statement: str) -> None:
                                if statement.lstrip().upper().startswith(
                                    "INSERT INTO SNAPSHOT_ROWS"
                                ):
                                    traced_statements.add(statement)
                                    statement_started.set()

                            second.set_trace_callback(trace_statement)
                            writer_thread_ids.append(threading.get_ident())
                            writer_started.set()
                            second.execute(mutation_sql)
                            second.commit()
                        except BaseException as exc:
                            writer_errors.append(exc)
                        finally:
                            second.close()
                            writer_finished.set()

                    def on_progress(batch_idx: int, rows_done: int) -> None:
                        nonlocal writer_thread
                        self.assertTrue(
                            conn.in_transaction,
                            "snapshot transaction ended before non-PK backfill completed",
                        )
                        if batch_idx != 1:
                            return
                        writer_thread = threading.Thread(
                            target=write_from_independent_session, daemon=True
                        )
                        writer_thread.start()
                        if not writer_started.wait(timeout=10):
                            raise AssertionError("independent snapshot writer did not start")
                        if not statement_started.wait(timeout=10):
                            raise AssertionError(
                                "independent snapshot writer never entered its INSERT statement"
                            )
                        if writer_finished.wait(timeout=0.5):
                            raise AssertionError(
                                "independent writer committed before the snapshot backfill finished"
                            )

                    plan = manager.register_view(view_sql, populate=False)
                    self.assertIsNotNone(plan.incremental_backfill_spec)
                    self.assertEqual(plan.incremental_backfill_spec.mode, expected_mode)
                    batches = manager.backfill_incrementally(
                        plan,
                        batch_size=2,
                        on_progress=on_progress,
                        snapshot_isolation=True,
                    )
                    self.assertGreaterEqual(batches, 2)
                    self.assertFalse(conn.in_transaction)
                    self.assertIsNotNone(writer_thread)
                    writer_thread.join(timeout=10)
                    self.assertFalse(
                        writer_thread.is_alive(),
                        "independent writer did not resume after snapshot commit",
                    )
                    if writer_errors:
                        raise writer_errors[0]
                    self.assertTrue(writer_finished.is_set())
                    self.assertEqual(len(traced_statements), 1)
                    traced_statement = next(iter(traced_statements))
                    self.assertTrue(
                        traced_statement.lstrip().upper().startswith(
                            "INSERT INTO SNAPSHOT_ROWS"
                        )
                    )
                    self.assertTrue(writer_thread_ids)
                    self.assertNotEqual(writer_thread_ids[0], threading.get_ident())

                    actual = [
                        tuple(row)
                        for row in conn.execute(
                            f"SELECT * FROM {plan.view_name} ORDER BY 1;"
                        ).fetchall()
                    ]
                    expected = [
                        tuple(row)
                        for row in conn.execute(
                            live_sql.rstrip(";") + " ORDER BY 1;"
                        ).fetchall()
                    ]
                    self.assertEqual(actual, expected)
                    catalog = manager.get_catalog()[plan.view_name]
                    self.assertEqual(catalog["backfill_active"], 0)
                    self.assertIsNone(catalog["backfill_cursor_pk"])
                    self.assertIsNone(catalog["backfill_max_pk"])
                    manager.drop_view(plan.view_name)
        finally:
            conn.close()
            os.unlink(db_path)


    def test_snapshot_backfill_failure_rolls_back_partial_batches_and_allows_retry(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="matview-snapshot-rollback-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE rollback_rows (id INTEGER, bucket TEXT NOT NULL, value INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO rollback_rows VALUES (?, ?, ?);",
            [(i, f"b{i % 3}", i * 10) for i in range(1, 10)],
        )
        conn.commit()
        manager = MaterializedViewManager(conn)
        cases = (
            (
                "CREATE VIEW rollback_grouped AS "
                "SELECT bucket, COUNT(DISTINCT value) AS n "
                "FROM rollback_rows GROUP BY bucket;",
                "group_key_batch",
                "SELECT bucket, COUNT(DISTINCT value) AS n FROM rollback_rows GROUP BY bucket;",
            ),
            (
                "CREATE VIEW rollback_ranked AS "
                "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS row_num "
                "FROM rollback_rows;",
                "offset_batch",
                "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS row_num FROM rollback_rows;",
            ),
        )
        try:
            for view_sql, expected_mode, live_sql in cases:
                with self.subTest(mode=expected_mode):
                    plan = manager.register_view(view_sql, populate=False)
                    self.assertIsNotNone(plan.incremental_backfill_spec)
                    self.assertEqual(plan.incremental_backfill_spec.mode, expected_mode)

                    def fail_after_first_batch(batch_idx: int, rows_done: int) -> None:
                        if batch_idx == 1:
                            raise RuntimeError("injected snapshot backfill failure")

                    with self.assertRaisesRegex(
                        RuntimeError, "injected snapshot backfill failure"
                    ):
                        manager.backfill_incrementally(
                            plan,
                            batch_size=2,
                            on_progress=fail_after_first_batch,
                            snapshot_isolation=True,
                        )

                    self.assertFalse(
                        conn.in_transaction,
                        "failed snapshot backfill left its transaction/savepoint open",
                    )
                    backing_rows = conn.execute(
                        f"SELECT COUNT(*) FROM {plan.backing_table_name};"
                    ).fetchone()[0]
                    self.assertEqual(
                        backing_rows,
                        0,
                        "partial rows from the failed snapshot backfill were not rolled back",
                    )
                    catalog_entry = manager.get_catalog()[plan.view_name]
                    self.assertEqual(catalog_entry["backfill_active"], 0)
                    self.assertIsNone(catalog_entry["backfill_cursor_pk"])
                    self.assertIsNone(catalog_entry["backfill_max_pk"])
                    self.assertEqual(conn.execute("SELECT 1;").fetchone()[0], 1)

                    batches = manager.backfill_incrementally(
                        plan, batch_size=2, snapshot_isolation=True
                    )
                    self.assertGreaterEqual(batches, 2)
                    actual = conn.execute(
                        f"SELECT * FROM {plan.view_name} ORDER BY 1;"
                    ).fetchall()
                    expected = conn.execute(
                        live_sql.rstrip(";") + " ORDER BY 1;"
                    ).fetchall()
                    self.assertEqual(actual, expected)
                    manager.drop_view(plan.view_name)
        finally:
            conn.close()
            os.unlink(db_path)

    @staticmethod
    def _connect_native(engine: Dialect, dsn: str):
        if engine == Dialect.POSTGRES:
            import psycopg2
            return psycopg2.connect(dsn)
        parsed = urlparse(dsn)
        import pymysql
        return pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or None,
            autocommit=False,
        )

    def _run_native_watermark_interleaving(self, engine: Dialect, dsn: str) -> None:
        suffix = uuid.uuid4().hex[:10]
        schema_name = f"mv_backfill_{suffix}"
        table_name = f"ledger_{suffix}"
        view_name = f"ledger_totals_{suffix}"
        conn = self._connect_native(engine, dsn)
        manager = None
        plan = None
        worker_errors: list[BaseException] = []
        worker_thread_ids: list[int] = []
        progress_calls: list[tuple[int, int]] = []
        started = threading.Event()
        finished = threading.Event()

        try:
            cur = conn.cursor()
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema_name}";')
                cur.execute(f'SET search_path TO "{schema_name}";')
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(
                f"CREATE TABLE {table_name} ("
                "id BIGINT PRIMARY KEY, "
                "bucket VARCHAR(16) NOT NULL, "
                "amt BIGINT NOT NULL, "
                "approx_amt DOUBLE PRECISION NOT NULL"
                f"){engine_suffix};"
            )
            cur.executemany(
                f"INSERT INTO {table_name} (id, bucket, amt, approx_amt) VALUES (%s, %s, %s, %s)",
                [(i, "B1" if i % 2 else "B2", i * 10, i * 0.125 + 0.1) for i in range(1, 11)],
            )
            conn.commit()

            manager = MaterializedViewManager(conn, dialect=engine)

            def write_from_independent_session() -> None:
                second = None
                try:
                    second = self._connect_native(engine, dsn)
                    second_cur = second.cursor()
                    if engine == Dialect.POSTGRES:
                        second_cur.execute(f'SET search_path TO "{schema_name}";')
                        second.commit()
                    worker_thread_ids.append(threading.get_ident())
                    started.set()

                    # The first committed chunk has advanced the primary-key
                    # watermark. These writes straddle that boundary: inserts
                    # beyond max_pk, mutations below the cursor, and mutations
                    # in the not-yet-backfilled range.
                    second_cur.execute(
                        f"INSERT INTO {table_name} (id, bucket, amt, approx_amt) VALUES (%s, %s, %s, %s)",
                        (11, "B1", 110, 1.475),
                    )
                    second_cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (2,))
                    second_cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (8,))
                    second_cur.execute(
                        f"UPDATE {table_name} SET amt = %s, approx_amt = %s WHERE id = %s",
                        (999, 12.3456789, 1),
                    )
                    # batch_size=3 makes PK 3 the first committed cursor;
                    # PK 10 is the captured max watermark. Mutate both exact
                    # endpoints to catch off-by-one trigger-gating regressions.
                    second_cur.execute(
                        f"UPDATE {table_name} SET amt = %s, approx_amt = %s WHERE id = %s",
                        (777, 7.777, 3),
                    )
                    second_cur.execute(
                        f"UPDATE {table_name} SET amt = %s, approx_amt = %s WHERE id = %s",
                        (888, 9.87654321, 9),
                    )
                    second_cur.execute(
                        f"UPDATE {table_name} SET amt = %s, approx_amt = %s WHERE id = %s",
                        (666, 6.666, 10),
                    )
                    second.commit()

                    # Trigger effects must obey the same transaction as the base
                    # row: the insert below is intentionally rolled back.
                    second_cur.execute("BEGIN;")
                    second_cur.execute(
                        f"INSERT INTO {table_name} (id, bucket, amt, approx_amt) VALUES (%s, %s, %s, %s)",
                        (12, "B2", 120, 1.625),
                    )
                    second.rollback()
                except BaseException as exc:
                    worker_errors.append(exc)
                finally:
                    if second is not None:
                        second.close()
                    finished.set()

            def on_progress(batch_idx: int, rows_done: int) -> None:
                progress_calls.append((batch_idx, rows_done))
                if batch_idx != 1:
                    return
                worker = threading.Thread(
                    target=write_from_independent_session, daemon=True
                )
                worker.start()
                if not started.wait(timeout=20):
                    raise AssertionError(
                        f"{engine.value} independent writer did not start during backfill"
                    )
                if not finished.wait(timeout=20):
                    raise AssertionError(
                        f"{engine.value} independent writer did not finish before backfill resumed"
                    )
                worker.join(timeout=1)
                if worker.is_alive():
                    raise AssertionError(
                        f"{engine.value} independent writer thread did not terminate"
                    )
                if worker_errors:
                    raise worker_errors[0]

            view_sql = (
                f"CREATE VIEW {view_name} AS "
                f"SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, "
                f"AVG(amt) AS avg_amt, AVG(approx_amt) AS avg_approx_amt "
                f"FROM {table_name} GROUP BY bucket;"
            )
            # This interleaving test never uses TRUNCATE. Explicitly override
            # the unrelated MySQL limitation while keeping the warning and
            # limitation metadata observable to the test.
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always", EngineLimitationWarning)
                plan = manager.register_view(
                    view_sql,
                    incremental_backfill=True,
                    batch_size=3,
                    on_progress=on_progress,
                    ignore_limitations=True,
                )
            limitation_warnings = [
                warning for warning in emitted
                if issubclass(warning.category, EngineLimitationWarning)
            ]
            expected_limitation_codes = (
                {"MYSQL_NO_TRUNCATE_TRIGGER"} if engine == Dialect.MYSQL else set()
            )
            self.assertEqual(
                {lim.code for lim in plan.limitations},
                expected_limitation_codes,
            )
            self.assertEqual(len(limitation_warnings), len(expected_limitation_codes))
            if expected_limitation_codes:
                self.assertIn(
                    "MYSQL_NO_TRUNCATE_TRIGGER",
                    str(limitation_warnings[0].message),
                )

            self.assertTrue(
                progress_calls,
                f"{engine.value} backfill did not report chunk progress",
            )
            self.assertTrue(worker_thread_ids)
            self.assertNotEqual(worker_thread_ids[0], threading.get_ident())
            self.assertEqual(progress_calls[0][0], 1)

            cur.execute(
                f"SELECT bucket, c, s, avg_amt, avg_approx_amt "
                f"FROM {plan.view_name} ORDER BY bucket;"
            )
            actual = [tuple(row) for row in cur.fetchall()]
            cur.execute(
                f"SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, "
                f"AVG(amt) AS avg_amt, AVG(approx_amt) AS avg_approx_amt "
                f"FROM {table_name} GROUP BY bucket ORDER BY bucket;"
            )
            expected = [tuple(row) for row in cur.fetchall()]
            self.assertEqual(actual, expected)
            for row in actual:
                self.assertIsInstance(row[3], Decimal)
                self.assertIsInstance(row[4], float)
            cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE id = 12;")
            self.assertEqual(
                cur.fetchone()[0], 0,
                f"{engine.value} rolled-back base insert unexpectedly persisted",
            )
            catalog = manager.get_catalog()[plan.view_name]
            self.assertEqual(catalog["backfill_active"], 0)
            conn.commit()
        finally:
            self._cleanup_native_fixture(conn, engine, schema_name, table_name, manager, plan)

    def test_native_postgres_watermark_gates_independent_session_dml(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_watermark_interleaving(Dialect.POSTGRES, dsn)

    def test_native_mysql_family_watermark_gates_independent_session_dml(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_watermark_interleaving(Dialect.MYSQL, dsn)

    def test_sqlite_barrier_concurrent_multi_writer_backfill(self) -> None:
        """Exercise genuine multi-writer WAL overlap across chunked backfill without synchronous joins."""
        fd, db_path = tempfile.mkstemp(prefix="matview-barrier-backfill-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE ledger_concurrent ("
            "id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO ledger_concurrent VALUES (?, ?, ?);",
            [(i, f"B{(i % 4) + 1}", i * 10) for i in range(1, 61)],
        )
        conn.commit()

        start_barrier = threading.Barrier(3)
        mid_barrier = threading.Barrier(3)
        backfill_done = threading.Event()
        worker_errors: list[BaseException] = []
        overlap_commits: list[int] = []
        overlap_lock = threading.Lock()
        barrier_released = False
        mid_released = False

        def concurrent_writer(writer_id: int) -> None:
            second = sqlite3.connect(db_path, timeout=10, isolation_level=None)
            try:
                second.execute("PRAGMA busy_timeout=10000;")
                _register_sqlite_compat_udfs(second)
                start_barrier.wait(timeout=10)
                base_pk = 1000 * writer_id
                target_pk = 10 + writer_id * 5
                for step in range(12):
                    bucket = f"B{((writer_id + step) % 4) + 1}"
                    new_pk = base_pk + step
                    second.execute("BEGIN IMMEDIATE;")
                    second.execute(
                        "INSERT INTO ledger_concurrent VALUES (?, ?, ?);",
                        (new_pk, bucket, (step + 1) * 17),
                    )
                    second.execute(
                        "UPDATE ledger_concurrent SET amt = amt + ? WHERE id = ?;",
                        (writer_id + 1, target_pk + (step % 4)),
                    )
                    if step % 4 == 3:
                        second.execute(
                            "DELETE FROM ledger_concurrent WHERE id = ?;",
                            (base_pk + step - 1,),
                        )
                    second.execute("COMMIT;")
                    if not backfill_done.is_set():
                        with overlap_lock:
                            overlap_commits.append(writer_id)

                    # Exercise concurrent rolled-back DML while backfill runs.
                    second.execute("BEGIN IMMEDIATE;")
                    second.execute(
                        "INSERT INTO ledger_concurrent VALUES (?, ?, ?);",
                        (base_pk + 500 + step, bucket, 99999),
                    )
                    second.execute("ROLLBACK;")
                    if step == 0:
                        mid_barrier.wait(timeout=10)
            except BaseException as exc:
                worker_errors.append(exc)
            finally:
                second.close()

        def on_progress(batch_idx: int, _rows_done: int) -> None:
            nonlocal barrier_released, mid_released
            if batch_idx == 1 and not barrier_released:
                barrier_released = True
                start_barrier.wait(timeout=10)
            elif batch_idx == 3 and not mid_released:
                mid_released = True
                mid_barrier.wait(timeout=10)

        w1 = threading.Thread(target=concurrent_writer, args=(1,), daemon=True)
        w2 = threading.Thread(target=concurrent_writer, args=(2,), daemon=True)
        w1.start()
        w2.start()
        try:
            manager = MaterializedViewManager(conn)
            plan = manager.register_view(
                "CREATE VIEW ledger_concurrent_totals AS "
                "SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS avg_amt "
                "FROM ledger_concurrent GROUP BY bucket;",
                populate=False,
            )
            batches = manager.backfill_incrementally(
                plan,
                batch_size=4,
                on_progress=on_progress,
                snapshot_isolation=False,
            )
            backfill_done.set()
            w1.join(timeout=20)
            w2.join(timeout=20)
            self.assertFalse(w1.is_alive())
            self.assertFalse(w2.is_alive())
            if worker_errors:
                raise worker_errors[0]
            self.assertGreaterEqual(batches, 10)
            self.assertTrue(barrier_released)
            self.assertEqual(
                set(overlap_commits), {1, 2},
                "Expected both concurrent writers to commit overlapping chunks before backfill completion",
            )
            actual = conn.execute(
                f"SELECT bucket, c, s, ROUND(avg_amt, 8) FROM {plan.view_name} ORDER BY bucket;"
            ).fetchall()
            expected = conn.execute(
                "SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, ROUND(AVG(amt), 8) AS avg_amt "
                "FROM ledger_concurrent GROUP BY bucket ORDER BY bucket;"
            ).fetchall()
            self.assertEqual(actual, expected)
            self.assertEqual(manager.get_catalog()[plan.view_name]["backfill_active"], 0)
            manager.drop_view(plan.view_name)
        finally:
            backfill_done.set()
            start_barrier.abort()
            mid_barrier.abort()
            conn.close()
            os.unlink(db_path)

    def _run_native_barrier_concurrent_writers(self, engine: Dialect, dsn: str) -> None:
        suffix = uuid.uuid4().hex[:10]
        schema_name = f"mv_barrier_{suffix}"
        table_name = f"ledger_barrier_{suffix}"
        view_name = f"ledger_barrier_totals_{suffix}"
        conn = self._connect_native(engine, dsn)
        manager = None
        plan = None
        start_barrier = threading.Barrier(3)
        mid_barrier = threading.Barrier(3)
        backfill_done = threading.Event()
        worker_errors: list[BaseException] = []
        overlap_commits: list[int] = []
        overlap_lock = threading.Lock()
        writer_phases: dict[int, str] = {1: "not started", 2: "not started"}
        phase_lock = threading.Lock()
        transient_retries: dict[int, int] = {1: 0, 2: 0}
        retry_lock = threading.Lock()

        def set_writer_phase(writer_id: int, phase: str) -> None:
            with phase_lock:
                writer_phases[writer_id] = phase

        barrier_released = False
        mid_released = False

        try:
            cur = conn.cursor()
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema_name}";')
                cur.execute(f'SET search_path TO "{schema_name}";')
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(
                f"CREATE TABLE {table_name} ("
                "id BIGINT PRIMARY KEY, "
                "bucket VARCHAR(16) NOT NULL, "
                "amt BIGINT NOT NULL"
                f"){engine_suffix};"
            )
            cur.executemany(
                f"INSERT INTO {table_name} (id, bucket, amt) VALUES (%s, %s, %s)",
                [(i, f"B{(i % 4) + 1}", i * 10) for i in range(1, 61)],
            )
            conn.commit()

            def concurrent_writer(writer_id: int) -> None:
                second = None
                try:
                    set_writer_phase(writer_id, "connecting")
                    second = self._connect_native(engine, dsn)
                    second_cur = second.cursor()
                    if engine == Dialect.POSTGRES:
                        set_writer_phase(writer_id, "setting PostgreSQL search_path")
                        second_cur.execute(f'SET search_path TO "{schema_name}";')
                        second.commit()
                    set_writer_phase(writer_id, "waiting at start barrier")
                    start_barrier.wait(timeout=20)
                    set_writer_phase(writer_id, "started")
                    base_pk = 1000 * writer_id
                    target_pk = 10 + writer_id * 5
                    for step in range(10):
                        bucket = f"B{((writer_id + step) % 4) + 1}"
                        new_pk = base_pk + step
                        attempt = 0
                        while True:
                            try:
                                set_writer_phase(writer_id, f"executing insert at step {step} (attempt {attempt})")
                                second_cur.execute(
                                    f"INSERT INTO {table_name} (id, bucket, amt) VALUES (%s, %s, %s)",
                                    (new_pk, bucket, (step + 1) * 19),
                                )
                                set_writer_phase(writer_id, f"executing update at step {step} (attempt {attempt})")
                                second_cur.execute(
                                    f"UPDATE {table_name} SET amt = amt + %s WHERE id = %s",
                                    (writer_id + 1, target_pk + (step % 4)),
                                )
                                if step % 3 == 2:
                                    second_cur.execute(
                                        f"DELETE FROM {table_name} WHERE id = %s",
                                        (base_pk + step - 1,),
                                    )
                                set_writer_phase(writer_id, f"committing step {step} (attempt {attempt})")
                                second.commit()
                                break
                            except Exception as exc:
                                # Concurrent trigger maintenance of shared group
                                # rows can make an application transaction an
                                # engine deadlock/serialization victim. The engine
                                # rolls the transaction back atomically (base rows
                                # and trigger deltas together), so the application
                                # contract is to replay the whole transaction. The
                                # final original-view equivalence check is unchanged.
                                if attempt >= TRANSIENT_RETRIES or not _is_transient_lock_error(engine, exc):
                                    raise
                                second.rollback()
                                attempt += 1
                                with retry_lock:
                                    transient_retries[writer_id] += 1
                                time.sleep(0.01 * attempt)
                        set_writer_phase(writer_id, f"committed step {step}")
                        if not backfill_done.is_set():
                            with overlap_lock:
                                overlap_commits.append(writer_id)

                        set_writer_phase(writer_id, f"executing rollback probe at step {step}")
                        try:
                            second_cur.execute(
                                f"INSERT INTO {table_name} (id, bucket, amt) VALUES (%s, %s, %s)",
                                (base_pk + 500 + step, bucket, 99999),
                            )
                        except Exception as exc:
                            # The probe is rolled back either way; only an
                            # engine deadlock/serialization victim is tolerated.
                            if not _is_transient_lock_error(engine, exc):
                                raise
                        second.rollback()
                        set_writer_phase(writer_id, f"rolled back probe at step {step}")
                        if step == 0:
                            set_writer_phase(writer_id, "waiting at mid barrier")
                            mid_barrier.wait(timeout=20)
                            set_writer_phase(writer_id, "passed mid barrier")
                except BaseException as exc:
                    set_writer_phase(writer_id, f"failed: {type(exc).__name__}: {exc!r}")
                    worker_errors.append(exc)
                finally:
                    if second is not None:
                        second.close()
                    with phase_lock:
                        writer_phases[writer_id] = f"closed after: {writer_phases[writer_id]}"

            def on_progress(batch_idx: int, _rows_done: int) -> None:
                nonlocal barrier_released, mid_released
                if batch_idx == 1 and not barrier_released:
                    barrier_released = True
                    start_barrier.wait(timeout=20)
                elif batch_idx == 3 and not mid_released:
                    mid_released = True
                    try:
                        mid_barrier.wait(timeout=20)
                    except threading.BrokenBarrierError as exc:
                        with phase_lock:
                            phase_snapshot = dict(writer_phases)
                        error_snapshot = [repr(error) for error in worker_errors]
                        raise AssertionError(
                            f"{engine.value}: writers did not reach the mid-backfill barrier "
                            f"at batch {batch_idx}; writer_phases={phase_snapshot!r}; "
                            f"worker_errors={error_snapshot!r}; "
                            f"transient_retries={dict(transient_retries)!r}"
                        ) from exc

            w1 = threading.Thread(target=concurrent_writer, args=(1,), daemon=True)
            w2 = threading.Thread(target=concurrent_writer, args=(2,), daemon=True)
            w1.start()
            w2.start()

            manager = MaterializedViewManager(conn, dialect=engine)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always", EngineLimitationWarning)
                plan = manager.register_view(
                    f"CREATE VIEW {view_name} AS "
                    f"SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS avg_amt "
                    f"FROM {table_name} GROUP BY bucket;",
                    populate=False,
                    ignore_limitations=True,
                )
            limitation_warnings = [
                warning for warning in emitted
                if issubclass(warning.category, EngineLimitationWarning)
            ]
            expected_limitation_codes = (
                {"MYSQL_NO_TRUNCATE_TRIGGER"} if engine == Dialect.MYSQL else set()
            )
            self.assertEqual(
                {lim.code for lim in plan.limitations},
                expected_limitation_codes,
            )
            self.assertEqual(len(limitation_warnings), len(expected_limitation_codes))
            batches = manager.backfill_incrementally(
                plan,
                batch_size=4,
                on_progress=on_progress,
                snapshot_isolation=False,
            )
            backfill_done.set()
            w1.join(timeout=30)
            w2.join(timeout=30)
            self.assertFalse(w1.is_alive())
            self.assertFalse(w2.is_alive())
            if worker_errors:
                raise worker_errors[0]
            self.assertGreaterEqual(batches, 10)
            self.assertTrue(barrier_released)
            self.assertEqual(
                set(overlap_commits), {1, 2},
                f"{engine.value}: expected both concurrent writers to commit before backfill completion",
            )
            cur.execute(
                f"SELECT bucket, c, s, avg_amt FROM {plan.view_name} ORDER BY bucket;"
            )
            actual = [tuple(row) for row in cur.fetchall()]
            cur.execute(
                f"SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS avg_amt "
                f"FROM {table_name} GROUP BY bucket ORDER BY bucket;"
            )
            expected = [tuple(row) for row in cur.fetchall()]
            self.assertEqual(actual, expected)
            self.assertEqual(manager.get_catalog()[plan.view_name]["backfill_active"], 0)
            conn.commit()
        finally:
            backfill_done.set()
            start_barrier.abort()
            mid_barrier.abort()
            self._cleanup_native_fixture(conn, engine, schema_name, table_name, manager, plan)

    @staticmethod
    def _cleanup_native_fixture(
        conn, engine: Dialect, schema_name: str, table_name: str, manager, plan
    ) -> None:
        """Tear down temporary native schema, table, and materialized view cleanly."""
        try:
            conn.rollback()
            cur = conn.cursor()
            if engine == Dialect.POSTGRES:
                cur.execute("SET search_path TO public;")
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE;')
            else:
                if manager is not None and plan is not None:
                    try:
                        manager.drop_view(plan.view_name)
                    except Exception:
                        pass
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            conn.commit()
        except Exception:
            conn.rollback()
        conn.close()

    def test_native_postgres_barrier_concurrent_multi_writer_backfill(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_barrier_concurrent_writers(Dialect.POSTGRES, dsn)

    def test_native_mysql_family_barrier_concurrent_multi_writer_backfill(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_barrier_concurrent_writers(Dialect.MYSQL, dsn)


if __name__ == "__main__":
    unittest.main(verbosity=2)