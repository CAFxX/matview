#!/usr/bin/env python3
"""Independent-session tests for incremental materialized-view backfill."""

from __future__ import annotations

from decimal import Decimal

import os
import sqlite3
import tempfile
import threading
import time
import types
import unittest
import warnings
import uuid
from urllib.parse import urlparse

from matview_sync import BackfillInterruptedError, Dialect, EngineLimitationError, EngineLimitationWarning, MaterializedViewManager
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
            # Fail closed: chunk (0, 4] stays committed, the watermark keeps
            # gating (4, 12], and the error says so.
            entry = manager.get_catalog()[plan.view_name]
            self.assertEqual(
                (entry["backfill_active"], entry["backfill_cursor_pk"], entry["backfill_max_pk"]), (1, 4, 12)
            )
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

    def test_concurrent_independent_writers_keep_materialized_aggregates_consistent(self) -> None:
        """Stress trigger maintenance with concurrent committed and rolled-back DML."""
        fd, db_path = tempfile.mkstemp(prefix="matview-multiwriter-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE concurrent_ledger ("
            "id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amount INTEGER);"
        )
        conn.commit()

        manager = MaterializedViewManager(conn)
        plan = manager.register_view(
            "CREATE VIEW concurrent_totals AS "
            "SELECT bucket, COUNT(*) AS row_count, COUNT(amount) AS valued_count, "
            "SUM(amount) AS total_amount "
            "FROM concurrent_ledger GROUP BY bucket;"
        )
        worker_count = 4
        transactions_per_worker = 40
        start = threading.Barrier(worker_count)
        errors: list[BaseException] = []
        completed: list[int] = []
        error_lock = threading.Lock()

        def writer(worker_id: int) -> None:
            second = None
            try:
                second = sqlite3.connect(db_path, timeout=30)
                second.execute("PRAGMA busy_timeout=30000;")
                _register_sqlite_compat_udfs(second)
                start.wait(timeout=15)
                for sequence in range(transactions_per_worker):
                    row_id = worker_id * 10000 + sequence + 1
                    bucket = f"source_{worker_id % 2}"
                    amount = None if sequence % 9 == 0 else worker_id * 100 + sequence
                    second.execute("BEGIN IMMEDIATE;")
                    second.execute(
                        "INSERT INTO concurrent_ledger (id, bucket, amount) VALUES (?, ?, ?);",
                        (row_id, bucket, amount),
                    )
                    second.execute(
                        "UPDATE concurrent_ledger SET bucket = ?, "
                        "amount = CASE WHEN amount IS NULL THEN NULL ELSE amount + 7 END "
                        "WHERE id = ?;",
                        (f"target_{worker_id % 3}", row_id),
                    )
                    if sequence % 4 == 0:
                        second.execute(
                            "DELETE FROM concurrent_ledger WHERE id = ?;", (row_id,)
                        )
                    if sequence % 13 == 0:
                        second.rollback()
                    else:
                        second.commit()
                with error_lock:
                    completed.append(worker_id)
            except BaseException as exc:
                with error_lock:
                    errors.append(exc)
            finally:
                if second is not None:
                    second.close()

        threads = [
            threading.Thread(target=writer, args=(worker_id,), daemon=True)
            for worker_id in range(worker_count)
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=45)
            self.assertFalse(
                [thread.name for thread in threads if thread.is_alive()],
                "concurrent DML workers deadlocked or exceeded the timeout",
            )
            self.assertEqual(errors, [], f"concurrent writer errors: {errors!r}")
            self.assertEqual(sorted(completed), list(range(worker_count)))

            actual = conn.execute(
                f"SELECT bucket, row_count, valued_count, total_amount "
                f"FROM {plan.view_name} ORDER BY bucket;"
            ).fetchall()
            expected = conn.execute(
                "SELECT bucket, COUNT(*) AS row_count, COUNT(amount) AS valued_count, "
                "SUM(amount) AS total_amount FROM concurrent_ledger "
                "GROUP BY bucket ORDER BY bucket;"
            ).fetchall()
            self.assertEqual(actual, expected)

            expected_rows = sum(
                1
                for sequence in range(transactions_per_worker)
                for _worker_id in range(worker_count)
                if sequence % 13 != 0 and sequence % 4 != 0
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM concurrent_ledger;").fetchone()[0],
                expected_rows,
            )
            self.assertEqual(
                conn.execute(f"SELECT COUNT(*) FROM {plan.backing_table_name};").fetchone()[0],
                len(expected),
                "backing table must contain exactly one row per surviving group",
            )
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


    def _run_native_multiwriter_aggregate_stress(self, engine: Dialect, dsn: str) -> None:
        """Exercise trigger-maintained grouped aggregates from independent native sessions."""
        suffix = uuid.uuid4().hex[:10]
        schema_name = f"mv_writers_{suffix}"
        table_name = f"concurrent_ledger_{suffix}"
        view_name = f"concurrent_totals_{suffix}"
        conn = self._connect_native(engine, dsn)
        manager = None
        plan = None
        worker_errors: list[BaseException] = []
        completed: list[int] = []
        error_lock = threading.Lock()
        worker_count = 4
        transactions_per_worker = 40
        start = threading.Barrier(worker_count)

        try:
            cur = conn.cursor()
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema_name}";')
                cur.execute(f'SET search_path TO "{schema_name}";')
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(
                f"CREATE TABLE {table_name} ("
                "id BIGINT PRIMARY KEY, bucket VARCHAR(32) NOT NULL, amount BIGINT NULL"
                f"){engine_suffix};"
            )
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            # MySQL cannot intercept TRUNCATE, so the manager correctly
            # classifies general transparency as limited. This stress case
            # deliberately exercises only INSERT/UPDATE/DELETE and therefore
            # opts into that documented limitation without weakening the
            # production default-rejection contract.
            plan = manager.register_view(
                f"CREATE VIEW {view_name} AS "
                "SELECT bucket, COUNT(*) AS row_count, COUNT(amount) AS valued_count, "
                f"SUM(amount) AS total_amount FROM {table_name} GROUP BY bucket;",
                ignore_limitations=(engine == Dialect.MYSQL),
            )

            def writer(worker_id: int) -> None:
                second = None
                try:
                    second = self._connect_native(engine, dsn)
                    second_cur = second.cursor()
                    if engine == Dialect.POSTGRES:
                        second_cur.execute(f'SET search_path TO "{schema_name}";')
                        second.commit()
                    start.wait(timeout=30)
                    for sequence in range(transactions_per_worker):
                        row_id = worker_id * 10000 + sequence + 1
                        bucket = f"source_{worker_id % 2}"
                        amount = None if sequence % 9 == 0 else worker_id * 100 + sequence
                        # InnoDB and PostgreSQL may legitimately abort one
                        # transaction when concurrent trigger work contends on
                        # shared aggregate groups. Retry only recognized
                        # transaction-abort errors, from BEGIN through COMMIT,
                        # so the workload still completes every logical DML
                        # sequence exactly once.
                        for attempt in range(12):
                            try:
                                second_cur.execute("BEGIN;")
                                second_cur.execute(
                                    f"INSERT INTO {table_name} (id, bucket, amount) VALUES (%s, %s, %s)",
                                    (row_id, bucket, amount),
                                )
                                second_cur.execute(
                                    f"UPDATE {table_name} SET bucket = %s, "
                                    "amount = CASE WHEN amount IS NULL THEN NULL ELSE amount + 7 END "
                                    "WHERE id = %s",
                                    (f"target_{worker_id % 3}", row_id),
                                )
                                if sequence % 4 == 0:
                                    second_cur.execute(
                                        f"DELETE FROM {table_name} WHERE id = %s", (row_id,)
                                    )
                                if sequence % 13 == 0:
                                    second.rollback()
                                else:
                                    second.commit()
                                break
                            except Exception as exc:
                                second.rollback()
                                args = getattr(exc, "args", ())
                                mysql_code = args[0] if args and isinstance(args[0], int) else None
                                sqlstate = getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)
                                retryable = (
                                    engine == Dialect.MYSQL and mysql_code in {1205, 1213}
                                ) or (
                                    engine == Dialect.POSTGRES and sqlstate in {"40P01", "40001"}
                                )
                                if not retryable or attempt == 11:
                                    raise
                                time.sleep(min(0.25, 0.01 * (2 ** attempt)) + 0.003 * ((worker_id * 7 + sequence * 3 + attempt) % 5))
                    with error_lock:
                        completed.append(worker_id)
                except BaseException as exc:
                    with error_lock:
                        worker_errors.append(exc)
                finally:
                    if second is not None:
                        second.close()

            threads = [
                threading.Thread(target=writer, args=(worker_id,), daemon=True)
                for worker_id in range(worker_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=90)
            self.assertFalse(
                [thread.name for thread in threads if thread.is_alive()],
                f"{engine.value} concurrent writers deadlocked or exceeded timeout",
            )
            self.assertEqual(worker_errors, [], f"{engine.value} writer errors: {worker_errors!r}")
            self.assertEqual(sorted(completed), list(range(worker_count)))

            cur.execute(
                f"SELECT bucket, row_count, valued_count, total_amount "
                f"FROM {plan.view_name} ORDER BY bucket;"
            )
            actual = [tuple(row) for row in cur.fetchall()]
            cur.execute(
                f"SELECT bucket, COUNT(*) AS row_count, COUNT(amount) AS valued_count, "
                f"SUM(amount) AS total_amount FROM {table_name} GROUP BY bucket ORDER BY bucket;"
            )
            expected = [tuple(row) for row in cur.fetchall()]
            self.assertEqual(actual, expected)
            expected_rows = sum(
                1
                for sequence in range(transactions_per_worker)
                for _worker_id in range(worker_count)
                if sequence % 13 != 0 and sequence % 4 != 0
            )
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            self.assertEqual(cur.fetchone()[0], expected_rows)
            cur.execute(f"SELECT COUNT(*) FROM {plan.backing_table_name};")
            self.assertEqual(
                cur.fetchone()[0],
                len(expected),
                f"{engine.value} backing table must contain one row per surviving group",
            )
            conn.commit()
        finally:
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

    def test_native_postgres_concurrent_writers_keep_aggregates_consistent(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_multiwriter_aggregate_stress(Dialect.POSTGRES, dsn)

    def test_native_mysql_family_concurrent_writers_keep_aggregates_consistent(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_multiwriter_aggregate_stress(Dialect.MYSQL, dsn)

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

    def _run_native_transaction_visibility_invariant(self, engine: Dialect, dsn: str) -> None:
        """Verify MV trigger effects share the base transaction's visibility boundary."""
        suffix = uuid.uuid4().hex[:10]
        schema_name = f"mv_visibility_{suffix}"
        table_name = f"visibility_rows_{suffix}"
        view_name = f"visibility_totals_{suffix}"
        writer = self._connect_native(engine, dsn)
        reader = None
        manager = None
        schema_created = False

        def rows(connection, sql: str):
            cursor = connection.cursor()
            cursor.execute(sql)
            result = [tuple(row) for row in cursor.fetchall()]
            cursor.close()
            return result

        def assert_reader_matches_committed_state(expected):
            visible = rows(
                reader,
                f"SELECT grp, row_count, total FROM {view_name} ORDER BY grp",
            )
            live = rows(
                reader,
                f"SELECT grp, COUNT(*) AS row_count, SUM(amount) AS total "
                f"FROM {table_name} GROUP BY grp ORDER BY grp",
            )
            self.assertEqual(visible, expected, "reader saw uncommitted or stale MV state")
            self.assertEqual(live, expected, "reader's committed base-table state was unexpected")
            # End the read transaction so the next assertion obtains a fresh
            # READ COMMITTED statement snapshot (and a new InnoDB snapshot).
            reader.commit()

        try:
            writer_cursor = writer.cursor()
            if engine == Dialect.POSTGRES:
                writer_cursor.execute(f'CREATE SCHEMA "{schema_name}";')
                writer_cursor.execute(f'SET search_path TO "{schema_name}";')
                writer.commit()
                schema_created = True
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            writer_cursor.execute(
                f"CREATE TABLE {table_name} ("
                "id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amount BIGINT NOT NULL"
                f"){engine_suffix};"
            )
            writer_cursor.executemany(
                f"INSERT INTO {table_name} (id, grp, amount) VALUES (%s, %s, %s)",
                [(1, "A", 10), (2, "A", 20), (3, "B", 5)],
            )
            writer.commit()

            reader = self._connect_native(engine, dsn)
            reader_cursor = reader.cursor()
            if engine == Dialect.POSTGRES:
                reader_cursor.execute(f'SET search_path TO "{schema_name}";')
                reader.commit()

            manager = MaterializedViewManager(writer, dialect=engine)
            view_sql = (
                f"CREATE VIEW {view_name} AS "
                f"SELECT grp, COUNT(*) AS row_count, SUM(amount) AS total "
                f"FROM {table_name} GROUP BY grp;"
            )
            if engine == Dialect.MYSQL:
                with self.assertRaises(EngineLimitationError) as cm:
                    manager.register_view(view_sql)
                self.assertIn("MYSQL_NO_TRUNCATE_TRIGGER", str(cm.exception))

            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always", EngineLimitationWarning)
                plan = manager.register_view(
                    view_sql,
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
            writer.commit()
            self.assertEqual(plan.view_name, view_name)
            initial = [("A", 2, 30), ("B", 1, 5)]
            assert_reader_matches_committed_state(initial)

            # The writer must see its own trigger-maintained changes, while an
            # independent reader must continue to see the last committed state.
            writer_cursor.execute("BEGIN;")
            writer_cursor.execute(
                f"INSERT INTO {table_name} (id, grp, amount) VALUES (%s, %s, %s)",
                (4, "A", 40),
            )
            writer_cursor.execute(
                f"UPDATE {table_name} SET amount = %s WHERE id = %s",
                (7, 3),
            )
            writer_cursor.execute(
                f"DELETE FROM {table_name} WHERE id = %s", (2,)
            )
            uncommitted = [("A", 2, 50), ("B", 1, 7)]
            self.assertEqual(
                rows(writer, f"SELECT grp, row_count, total FROM {view_name} ORDER BY grp"),
                uncommitted,
                "writer did not see its own materialized trigger effects",
            )
            assert_reader_matches_committed_state(initial)

            writer.commit()
            committed = [("A", 2, 50), ("B", 1, 7)]
            assert_reader_matches_committed_state(committed)

            # A second transaction exercises both insertion and retraction,
            # then rolls back. Neither the base rows nor MV trigger effects may
            # leak to another connection.
            writer_cursor.execute("BEGIN;")
            writer_cursor.execute(
                f"INSERT INTO {table_name} (id, grp, amount) VALUES (%s, %s, %s)",
                (5, "C", 99),
            )
            writer_cursor.execute(
                f"UPDATE {table_name} SET amount = %s WHERE id = %s",
                (1000, 1),
            )
            self.assertEqual(
                rows(writer, f"SELECT grp, row_count, total FROM {view_name} ORDER BY grp"),
                [("A", 2, 1040), ("B", 1, 7), ("C", 1, 99)],
            )
            assert_reader_matches_committed_state(committed)
            writer.rollback()
            assert_reader_matches_committed_state(committed)
        finally:
            if reader is not None:
                try:
                    reader.rollback()
                finally:
                    reader.close()
            try:
                writer.rollback()
                cleanup = writer.cursor()
                if engine == Dialect.POSTGRES:
                    cleanup.execute("SET search_path TO public;")
                    if schema_created:
                        cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE;')
                else:
                    if manager is not None:
                        try:
                            manager.drop_view(view_name)
                        except Exception:
                            pass
                    cleanup.execute(f"DROP TABLE IF EXISTS {table_name};")
                writer.commit()
            except Exception:
                writer.rollback()
            writer.close()

    def test_native_postgres_materialized_trigger_transaction_visibility(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_transaction_visibility_invariant(Dialect.POSTGRES, dsn)

    def test_native_mysql_family_materialized_trigger_transaction_visibility(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_transaction_visibility_invariant(Dialect.MYSQL, dsn)



V_GROUPED = "CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;"
V_GLOBAL = "CREATE VIEW v AS SELECT COUNT(*) AS c, SUM(amt) AS s FROM t;"
NATIVE_OF = {
    V_GROUPED: "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
    V_GLOBAL: "SELECT COUNT(*), SUM(amt) FROM t",
}


class _Interrupt(Exception):
    pass


class InterruptedBackfillFailClosedTests(unittest.TestCase):
    """Contract after a non-snapshot pk-range backfill stops part-way.

    The committed chunks and the active watermark are kept (triggers keep
    excluding the unfolded range), and no later call may resume, re-fold or
    reset it: every backfill_incrementally() on the view is refused with
    BackfillInterruptedError and leaves state untouched. The supported repair
    is an explicit rebuild, after which the view equals the original view.
    """

    def _setup(self, view_sql: str = V_GROUPED, path: str = ":memory:"):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, g TEXT, amt INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(i, f"G{i % 3}", i) for i in range(1, 21)])
        conn.commit()
        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        plan = manager.register_view(view_sql, populate=False, ignore_limitations=True)
        return conn, manager, plan

    @staticmethod
    def _interrupt(manager, plan, at_progress_call: int = 2) -> None:
        calls = []

        def progress(_b: int, _d: int) -> None:
            calls.append(1)
            if len(calls) == at_progress_call:
                raise _Interrupt()

        try:
            manager.backfill_incrementally(plan, batch_size=5, on_progress=progress)
        except _Interrupt as exc:
            # The original exception propagates, annotated with the state.
            assert any("watermark is kept active" in n for n in getattr(exc, "__notes__", [])), exc
        else:
            raise AssertionError("backfill was not interrupted")

    @staticmethod
    def _wm(manager, name: str):
        e = manager.get_catalog()[name]
        return e["backfill_cursor_pk"], e["backfill_max_pk"], e["backfill_active"]

    def _rows(self, conn, sql: str):
        return conn.execute(sql).fetchall()

    def _assert_matches_native(self, conn, view_sql: str, name: str = "v") -> None:
        self.assertEqual(
            self._rows(conn, f"SELECT * FROM {name} ORDER BY 1"), self._rows(conn, NATIVE_OF[view_sql])
        )

    def _mutate(self, conn) -> None:
        # Independent DML on both sides of the cursor while interrupted.
        conn.execute("INSERT INTO t VALUES (100, 'G1', 1000)")
        conn.execute("DELETE FROM t WHERE id IN (3, 15)")
        conn.execute("UPDATE t SET amt = amt + 7, g = 'G2' WHERE id IN (2, 12)")
        conn.commit()

    def test_any_later_call_is_refused_without_touching_state_then_rebuild_repairs(self) -> None:
        # B1/B3: neither a failing call (batch_size=0), a snapshot re-run, nor a
        # plain re-run may reset, resume or re-fold the interrupted watermark.
        conn, manager, plan = self._setup()
        try:
            self._interrupt(manager, plan)
            wm = self._wm(manager, "v")
            self.assertEqual(wm, (10, 20, 1))
            self._mutate(conn)
            before = self._rows(conn, "SELECT * FROM __matview_store_v ORDER BY 1")
            for kwargs in ({"batch_size": 0}, {"batch_size": 5, "snapshot_isolation": True}, {"batch_size": 5}):
                with self.subTest(**kwargs):
                    with self.assertRaises(BackfillInterruptedError) as ctx:
                        manager.backfill_incrementally(plan, **kwargs)
                    self.assertIn("force_rebuild=True", str(ctx.exception))
                    self.assertEqual(self._wm(manager, "v"), wm)
                    self.assertEqual(self._rows(conn, "SELECT * FROM __matview_store_v ORDER BY 1"), before)
            # Re-registering the same definition must not adopt it as complete.
            with self.assertRaises(BackfillInterruptedError):
                manager.register_view(V_GROUPED, ignore_limitations=True)
            plan = manager.register_view(V_GROUPED, force_rebuild=True, ignore_limitations=True)
            self._assert_matches_native(conn, V_GROUPED)
            self._mutate_after_rebuild(conn)
            self._assert_matches_native(conn, V_GROUPED)
        finally:
            conn.close()

    @staticmethod
    def _mutate_after_rebuild(conn) -> None:
        conn.execute("INSERT INTO t VALUES (101, 'G2', 5)")
        conn.execute("DELETE FROM t WHERE id IN (18, 100)")
        conn.commit()

    def test_drop_and_register_repairs_interrupted_view(self) -> None:
        conn, manager, plan = self._setup()
        try:
            self._interrupt(manager, plan)
            self._mutate(conn)
            manager.drop_view(plan.view_name)
            plan = manager.register_view(V_GROUPED, populate=False, ignore_limitations=True)
            manager.backfill_incrementally(plan, batch_size=5)
            self._assert_matches_native(conn, V_GROUPED)
        finally:
            conn.close()

    def test_completed_backfill_is_not_folded_twice(self) -> None:
        conn, manager, plan = self._setup()
        try:
            manager.backfill_incrementally(plan, batch_size=5)
            self._assert_matches_native(conn, V_GROUPED)
            for snap in (False, True):
                with self.subTest(snapshot_isolation=snap):
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=5, snapshot_isolation=snap)
                    self._assert_matches_native(conn, V_GROUPED)
                    self.assertEqual(self._wm(manager, "v"), (20, 20, 0))
        finally:
            conn.close()

    def test_multi_view_plan_rerun_never_refolds_finished_sub_plans(self) -> None:
        # B2: va finished, vb interrupted. A re-run must not re-fold va.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, amt INTEGER)")
            conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(i, f"G{i % 3}", i) for i in range(1, 21)])
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            plan = manager.register_view(
                "CREATE VIEW va AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;\n"
                "CREATE VIEW vb AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;",
                populate=False, ignore_limitations=True,
            )
            self._interrupt(manager, plan, at_progress_call=6)  # va: 4 batches; vb: 2nd batch
            self.assertEqual(self._wm(manager, "va"), (20, 20, 0))
            self.assertEqual(self._wm(manager, "vb"), (10, 20, 1))
            va_before = self._rows(conn, "SELECT * FROM va ORDER BY g")
            vb_before = self._rows(conn, "SELECT * FROM vb ORDER BY g")
            self.assertEqual(va_before, self._rows(conn, "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g"))
            for snap in (False, True):
                with self.subTest(snapshot_isolation=snap):
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=5, snapshot_isolation=snap)
                    self.assertEqual(self._rows(conn, "SELECT * FROM va ORDER BY g"), va_before)
                    self.assertEqual(self._rows(conn, "SELECT * FROM vb ORDER BY g"), vb_before)
                    self.assertEqual(self._wm(manager, "vb"), (10, 20, 1))
            manager.drop_view("vb")
            manager.register_view(
                "CREATE VIEW vb AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;", ignore_limitations=True
            )
            self.assertEqual(self._rows(conn, "SELECT * FROM vb ORDER BY g"),
                             self._rows(conn, "SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g"))
            self.assertEqual(self._rows(conn, "SELECT * FROM va ORDER BY g"),
                             self._rows(conn, "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g"))
        finally:
            conn.close()

    def test_global_aggregate_interrupted_then_base_emptied(self) -> None:
        # B4: no resume path exists, so the former UNIQUE(__mv_global_id)
        # failure cannot occur; the re-run is refused and a rebuild is exact.
        for emptied in (True, False):
            with self.subTest(emptied=emptied):
                conn, manager, plan = self._setup(V_GLOBAL)
                try:
                    self._interrupt(manager, plan)
                    conn.execute("DELETE FROM t" if emptied else "DELETE FROM t WHERE id <= 12")
                    conn.commit()
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=5)
                    self.assertEqual(self._wm(manager, "v"), (10, 20, 1))
                    manager.register_view(V_GLOBAL, force_rebuild=True, ignore_limitations=True)
                    self._assert_matches_native(conn, V_GLOBAL)
                finally:
                    conn.close()

    def test_second_concurrent_backfill_from_independent_connection_is_refused(self) -> None:
        # B5: a second session sees the live watermark and must refuse, not
        # treat it as interrupted and re-fold or reset it.
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn, manager, plan = self._setup(path=path)
        conn2 = sqlite3.connect(path, timeout=5)
        try:
            manager2 = MaterializedViewManager(conn2, dialect=Dialect.SQLITE)
            plan2 = manager2.register_view(V_GROUPED, populate=False, ignore_limitations=True)
            outcomes = []

            def progress(batch: int, _d: int) -> None:
                if batch == 1:
                    with self.assertRaises(BackfillInterruptedError) as ctx:
                        manager2.backfill_incrementally(plan2, batch_size=5)
                    outcomes.append(str(ctx.exception))
                    self.assertEqual(self._wm(manager2, "v"), (5, 20, 1))

            manager.backfill_incrementally(plan, batch_size=5, on_progress=progress)
            self.assertEqual(len(outcomes), 1)
            self.assertIn("RUNNING in another session or was INTERRUPTED", outcomes[0])
            self._assert_matches_native(conn, V_GROUPED)
            self.assertEqual(self._wm(manager, "v"), (20, 20, 0))
        finally:
            conn2.close()
            conn.close()
            os.unlink(path)

    def test_claim_is_atomic_between_sessions_that_both_saw_it_unclaimed(self) -> None:
        # B5 race: both sessions pass the pre-check before either claims; the
        # conditional claim lets exactly one of them proceed.
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn, manager, plan = self._setup(path=path)
        conn2 = sqlite3.connect(path, timeout=5)
        try:
            manager2 = MaterializedViewManager(conn2, dialect=Dialect.SQLITE)
            plan2 = manager2.register_view(V_GROUPED, populate=False, ignore_limitations=True)
            original = manager._refuse_unless_backfill_unclaimed
            raced = []

            def precheck_then_let_other_claim(entry, name):
                original(entry, name)  # passes: unclaimed
                if not raced:
                    raced.append(1)
                    manager2.backfill_incrementally(plan2, batch_size=5)  # claims + completes

            manager._refuse_unless_backfill_unclaimed = precheck_then_let_other_claim
            with self.assertRaises(BackfillInterruptedError) as ctx:
                manager.backfill_incrementally(plan, batch_size=5)
            self.assertIn("claimed", str(ctx.exception))
            self._assert_matches_native(conn, V_GROUPED)
            self.assertEqual(self._wm(manager, "v"), (20, 20, 0))
        finally:
            conn2.close()
            conn.close()
            os.unlink(path)

    def test_failure_before_claim_never_resets_a_watermark_claimed_by_another_session(self) -> None:
        # A call that fails after its pre-check but before claiming owns no
        # watermark. Its best-effort reset must not clear the watermark another
        # session claimed in between (here: interrupted after one committed
        # chunk); clearing it would re-enable triggers for the unfolded range
        # and let a later backfill re-fold the committed prefix.
        for snap in (False, True):
            with self.subTest(snapshot_isolation=snap):
                fd, path = tempfile.mkstemp(suffix=".db")
                os.close(fd)
                setup = sqlite3.connect(path)
                setup.execute("PRAGMA journal_mode=WAL")  # readers never block the other writer
                setup.close()
                conn, manager, plan = self._setup(path=path)
                conn2 = sqlite3.connect(path, timeout=5)
                try:
                    manager2 = MaterializedViewManager(conn2, dialect=Dialect.SQLITE)
                    plan2 = manager2.register_view(V_GROUPED, populate=False, ignore_limitations=True)
                    original = manager._refuse_unless_backfill_unclaimed

                    class Injected(Exception):
                        pass

                    def precheck_then_other_claims_then_fail(entry, name):
                        original(entry, name)  # passes: unclaimed
                        self._interrupt(manager2, plan2)  # claims, commits one chunk, stops
                        raise Injected("failure after the pre-check, before this call claims")

                    manager._refuse_unless_backfill_unclaimed = precheck_then_other_claims_then_fail
                    with self.assertRaises(Injected):
                        manager.backfill_incrementally(plan, batch_size=5, snapshot_isolation=snap)
                    self.assertEqual(self._wm(manager2, "v"), (10, 20, 1))
                    with self.assertRaises(BackfillInterruptedError):
                        manager2.backfill_incrementally(plan2, batch_size=5)
                    self.assertEqual(self._wm(manager2, "v"), (10, 20, 1))
                    manager2.register_view(V_GROUPED, force_rebuild=True, ignore_limitations=True)
                    self._assert_matches_native(conn, V_GROUPED)
                finally:
                    conn2.close()
                    conn.close()
                    for suffix in ("", "-wal", "-shm"):
                        if os.path.exists(path + suffix):
                            os.unlink(path + suffix)

    def test_error_after_the_final_batch_does_not_report_an_incomplete_view(self) -> None:
        # on_progress raising after the last batch: that batch's
        # compare-and-set already recorded completion together with its fold.
        conn, manager, plan = self._setup()
        try:
            with self.assertRaises(_Interrupt) as ctx:
                manager.backfill_incrementally(
                    plan, batch_size=5,
                    on_progress=lambda b, _d: (_ for _ in ()).throw(_Interrupt()) if b == 4 else None,
                )
            notes = getattr(ctx.exception, "__notes__", [])
            self.assertTrue(any("had already completed" in n for n in notes), notes)
            self.assertFalse(any("incomplete" in n for n in notes), notes)
            self.assertEqual(self._wm(manager, "v"), (20, 20, 0))
            self.assertEqual(manager.check_schema_drift(), {"v": "ok"})
            self._assert_matches_native(conn, V_GROUPED)
        finally:
            conn.close()

    def test_repair_view_clears_watermark_so_later_dml_is_maintained(self) -> None:
        # Review item 1: repair_view() repopulated the store but left the
        # active watermark, so triggers ignored DML in (cursor, max].
        conn, manager, plan = self._setup()
        try:
            self._interrupt(manager, plan)
            self.assertEqual(self._wm(manager, "v"), (10, 20, 1))
            self.assertEqual(manager.check_schema_drift(), {"v": "backfill_incomplete"})
            self.assertEqual(manager.sync_views([V_GROUPED]), {"v": "backfill_incomplete"})
            self.assertEqual(manager.repair_view("v"), {"v": "repaired"})
            self.assertEqual(self._wm(manager, "v")[2], 0)
            self._assert_matches_native(conn, V_GROUPED)
            # DML inside the formerly gated range and outside it.
            conn.execute("UPDATE t SET amt = amt + 1000 WHERE id = 18")
            conn.execute("DELETE FROM t WHERE id IN (19, 4)")
            conn.execute("INSERT INTO t VALUES (21, 'G9', 3)")
            conn.commit()
            self._assert_matches_native(conn, V_GROUPED)
            self.assertEqual(manager.check_schema_drift(), {"v": "ok"})
            self.assertEqual(manager.sync_views([V_GROUPED]), {"v": "unchanged"})
            # Repaired == populated: a later backfill must not fold again.
            with self.assertRaises(BackfillInterruptedError) as ctx:
                manager.backfill_incrementally(plan, batch_size=5)
            self.assertIn("already populated", str(ctx.exception))
            self._assert_matches_native(conn, V_GROUPED)
        finally:
            conn.close()

    def test_repair_view_fences_out_a_running_backfill_in_another_session(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn, manager, plan = self._setup(path=path)
        conn2 = sqlite3.connect(path, timeout=5)
        try:
            manager2 = MaterializedViewManager(conn2, dialect=Dialect.SQLITE)
            repaired = []

            def progress(batch: int, _d: int) -> None:
                if batch == 1:
                    repaired.append(manager2.repair_view("v"))
                    conn2.execute("UPDATE t SET amt = amt + 500 WHERE id = 17")
                    conn2.commit()

            with self.assertRaises(BackfillInterruptedError) as ctx:
                manager.backfill_incrementally(plan, batch_size=5, on_progress=progress)
            self.assertIn("taken over", str(ctx.exception))
            self.assertEqual(repaired, [{"v": "repaired"}])
            self._assert_matches_native(conn, V_GROUPED)
            conn.execute("DELETE FROM t WHERE id = 16")
            conn.commit()
            self._assert_matches_native(conn, V_GROUPED)
        finally:
            conn2.close()
            conn.close()
            os.unlink(path)

    def test_empty_table_backfill_records_completion(self) -> None:
        # Review item 4: an empty-table backfill wrote (NULL, NULL, 0); rows
        # maintained by triggers afterwards were folded again by a second run.
        for view_sql in (V_GROUPED, V_GLOBAL):
            with self.subTest(view=view_sql):
                conn = sqlite3.connect(":memory:")
                try:
                    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, amt INTEGER)")
                    conn.commit()
                    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
                    plan = manager.register_view(view_sql, populate=False, ignore_limitations=True)
                    self.assertEqual(manager.backfill_incrementally(plan, batch_size=5), 0)
                    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(i, f"G{i % 3}", i) for i in range(1, 8)])
                    conn.commit()
                    self._assert_matches_native(conn, view_sql)
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=5)
                    self._assert_matches_native(conn, view_sql)
                    self.assertEqual(manager.check_schema_drift(), {"v": "ok"})
                finally:
                    conn.close()

    def test_lost_connection_during_failure_handling_keeps_original_exception(self) -> None:
        # Review item 2: the failure handler queried the catalog, so a closed
        # connection replaced the original error with ProgrammingError.
        conn, manager, plan = self._setup()

        class ConnLost(Exception):
            pass

        def progress(batch: int, _d: int) -> None:
            if batch == 2:
                conn.close()
                raise ConnLost("server went away")

        with self.assertRaises(ConnLost) as ctx:
            manager.backfill_incrementally(plan, batch_size=5, on_progress=progress)
        self.assertTrue(any("cursor_pk=10" in n for n in getattr(ctx.exception, "__notes__", [])))

    def test_native_failed_backfill_leaves_connection_idle_and_snapshot_claim_conflict_is_refused(self) -> None:
        """Native: after an interrupted backfill the connection is not left in
        an open transaction (a later snapshot backfill works), and a second
        claimer of a snapshot backfill waits for the owner and is refused."""
        engines = []
        if os.environ.get("POSTGRES_DSN"):
            engines.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
        if os.environ.get("MYSQL_DSN"):
            engines.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                self._run_native_idle_and_nowait(engine, dsn)

    def _run_native_idle_and_nowait(self, engine: Dialect, dsn: str) -> None:
        connect = IndependentSessionBackfillTests._connect_native
        suffix = uuid.uuid4().hex[:10]
        schema, table = f"mv_idle_{suffix}", f"idle_{suffix}"
        v1, v2 = f"idle_a_{suffix}", f"idle_b_{suffix}"
        conn = connect(engine, dsn)
        other = connect(engine, dsn)
        cur = conn.cursor()
        plans = []
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
                other.cursor().execute(f'SET search_path TO "{schema}"')
                other.commit()
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(f"INSERT INTO {table} VALUES (%s, %s, %s)", [(i, f"G{i % 3}", i) for i in range(1, 21)])
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                for name in (v1, v2):
                    plans.append(manager.register_view(
                        f"CREATE VIEW {name} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;",
                        populate=False, ignore_limitations=True,
                    ))
            with self.assertRaises(_Interrupt):
                manager.backfill_incrementally(
                    plans[0], batch_size=5,
                    on_progress=lambda b, _d: (_ for _ in ()).throw(_Interrupt()) if b == 2 else None,
                )
            if engine == Dialect.POSTGRES:
                import psycopg2.extensions as ext
                self.assertEqual(conn.get_transaction_status(), ext.TRANSACTION_STATUS_IDLE)
            # Snapshot backfill of the second view while another session
            # tries to claim it. The claim waits for the catalog row lock
            # (NOWAIT cannot tell a claim from a writer's shared lock) and is
            # refused once the first backfill commits; it never folds.
            other_manager = MaterializedViewManager(other, dialect=engine)
            outcome: list = []
            waited: list = []

            def claim_from_other_session() -> None:
                try:
                    other_manager.backfill_incrementally(plans[1], batch_size=5, snapshot_isolation=True)
                    outcome.append("folded")
                except BaseException as exc:  # inspected below
                    outcome.append(exc)

            th = threading.Thread(target=claim_from_other_session, daemon=True)

            def on_progress(batch: int, _d: int) -> None:
                if batch == 1:
                    th.start()
                    th.join(timeout=1.0)
                    waited.append(th.is_alive())

            manager.backfill_incrementally(plans[1], batch_size=5, snapshot_isolation=True, on_progress=on_progress)
            th.join(timeout=60)
            self.assertFalse(th.is_alive())
            self.assertEqual(waited, [True], "the claim did not wait for the owning snapshot backfill")
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], BackfillInterruptedError, repr(outcome))
            cur.execute(f"SELECT g, c, s FROM {v2} ORDER BY g")
            actual = [tuple(r) for r in cur.fetchall()]
            cur.execute(f"SELECT g, COUNT(*), SUM(amt) FROM {table} GROUP BY g ORDER BY g")
            self.assertEqual(actual, [tuple(r) for r in cur.fetchall()])
            conn.commit()
        finally:
            try:
                other.rollback()
                other.close()
            except Exception:
                pass
            if engine == Dialect.MYSQL and plans:
                try:
                    conn.rollback()
                    MaterializedViewManager(conn, dialect=engine).drop_view(plans[1].view_name)
                except Exception:
                    pass
            IndependentSessionBackfillTests._cleanup_native_fixture(
                conn, engine, schema, table,
                MaterializedViewManager(conn, dialect=engine) if plans else None,
                plans[0] if plans else None,
            )

    def test_native_failure_before_claim_never_resets_another_sessions_watermark(self) -> None:
        """Native, both isolation modes: a call that fails after its pre-check
        but before claiming must leave the watermark another session claimed
        in between (interrupted after one chunk) exactly as it was."""
        engines = []
        if os.environ.get("POSTGRES_DSN"):
            engines.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
        if os.environ.get("MYSQL_DSN"):
            engines.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for snap in (False, True):
                with self.subTest(engine=engine.value, snapshot_isolation=snap):
                    self._run_native_failure_before_claim(engine, dsn, snap)

    def _run_native_failure_before_claim(self, engine: Dialect, dsn: str, snap: bool) -> None:
        connect = IndependentSessionBackfillTests._connect_native
        suffix = uuid.uuid4().hex[:10]
        schema, table, view = f"mv_noreset_{suffix}", f"noreset_{suffix}", f"noreset_v_{suffix}"
        conn = connect(engine, dsn)
        cur = conn.cursor()
        managers, plans, sessions = [], [], []
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(f"INSERT INTO {table} VALUES (%s, %s, %s)", [(i, f"G{i % 3}", i) for i in range(1, 21)])
            conn.commit()
            view_sql = f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                for _ in range(2):
                    s = connect(engine, dsn)
                    if engine == Dialect.POSTGRES:
                        s.cursor().execute(f'SET search_path TO "{schema}"')
                        s.commit()
                    sessions.append(s)
                    m = MaterializedViewManager(s, dialect=engine)
                    managers.append(m)
                    plans.append(m.register_view(view_sql, populate=False, ignore_limitations=True))
            original = managers[0]._refuse_unless_backfill_unclaimed

            class Injected(Exception):
                pass

            def precheck_then_other_claims_then_fail(entry, name):
                original(entry, name)  # passes: unclaimed
                self._interrupt(managers[1], plans[1])  # other session: claims, one chunk, stops
                raise Injected("failure after the pre-check, before this call claims")

            managers[0]._refuse_unless_backfill_unclaimed = precheck_then_other_claims_then_fail
            with self.assertRaises(Injected):
                managers[0].backfill_incrementally(plans[0], batch_size=5, snapshot_isolation=snap)
            self.assertEqual(self._wm(managers[1], view), (10, 20, 1))
            with self.assertRaises(BackfillInterruptedError):
                managers[1].backfill_incrementally(plans[1], batch_size=5)
            self.assertEqual(self._wm(managers[1], view), (10, 20, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                managers[1].register_view(view_sql, force_rebuild=True, ignore_limitations=True)
            sessions[1].commit()
            conn.commit()
            cur.execute(f"SELECT g, c, s FROM {view} ORDER BY g")
            actual = [tuple(r) for r in cur.fetchall()]
            cur.execute(f"SELECT g, COUNT(*), SUM(amt) FROM {table} GROUP BY g ORDER BY g")
            self.assertEqual(actual, [tuple(r) for r in cur.fetchall()])
            conn.commit()
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            IndependentSessionBackfillTests._cleanup_native_fixture(
                conn, engine, schema, table,
                MaterializedViewManager(conn, dialect=engine) if plans else None,
                plans[0] if plans else None,
            )

    def test_native_concurrent_backfills_exactly_one_claims(self) -> None:
        """Independent native sessions start a backfill of the same view at
        once (synchronized by a barrier); exactly one may fold, the other is
        refused, and the result equals the original view."""
        engines = []
        if os.environ.get("POSTGRES_DSN"):
            engines.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
        if os.environ.get("MYSQL_DSN"):
            engines.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                self._run_native_concurrent_claim(engine, dsn)

    def _run_native_concurrent_claim(self, engine: Dialect, dsn: str) -> None:
        connect = IndependentSessionBackfillTests._connect_native
        suffix = uuid.uuid4().hex[:10]
        schema, table, view = f"mv_claim_{suffix}", f"claim_{suffix}", f"claim_v_{suffix}"
        conn = connect(engine, dsn)
        cur = conn.cursor()
        managers, plans = [], []
        sessions = []
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(f"INSERT INTO {table} VALUES (%s, %s, %s)", [(i, f"G{i % 3}", i) for i in range(1, 41)])
            conn.commit()
            view_sql = f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                for _ in range(2):
                    s = connect(engine, dsn)
                    if engine == Dialect.POSTGRES:
                        s.cursor().execute(f'SET search_path TO "{schema}"')
                        s.commit()
                    sessions.append(s)
                    m = MaterializedViewManager(s, dialect=engine)
                    managers.append(m)
                    plans.append(m.register_view(view_sql, populate=False, ignore_limitations=True))
            barrier = threading.Barrier(2)
            results: list = [None, None]

            def run(i: int) -> None:
                try:
                    barrier.wait(timeout=20)
                    results[i] = managers[i].backfill_incrementally(plans[i], batch_size=4)
                except BaseException as exc:  # inspected below
                    results[i] = exc

            threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(2)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=60)
            self.assertFalse(any(th.is_alive() for th in threads))
            refused = [r for r in results if isinstance(r, BackfillInterruptedError)]
            folded = [r for r in results if isinstance(r, int)]
            self.assertEqual((len(folded), len(refused)), (1, 1), repr(results))
            cur.execute(f"SELECT g, c, s FROM {view} ORDER BY g")
            actual = [tuple(r) for r in cur.fetchall()]
            cur.execute(f"SELECT g, COUNT(*), SUM(amt) FROM {table} GROUP BY g ORDER BY g")
            self.assertEqual(actual, [tuple(r) for r in cur.fetchall()])
            conn.commit()
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            IndependentSessionBackfillTests._cleanup_native_fixture(
                conn, engine, schema, table,
                MaterializedViewManager(conn, dialect=engine) if plans else None,
                plans[0] if plans else None,
            )


class BackfillTakeoverOwnershipTests(unittest.TestCase):
    """A backfill that was taken over (force_rebuild of the view from another
    session while it is paused between batches) must never write the
    watermark of, or fold into, the new owner's view.

    Both sessions use independent connections and threads synchronized with
    events at batch boundaries (on_progress). The new owner (B) runs
    register_view(force_rebuild=True, incremental_backfill=True) and pauses
    after one of its batches; then the stale owner (A) resumes. The final
    view must equal the native result of B's definition and B must finish.
    """

    ROWS = 20
    BATCH = 5

    def _engines(self):
        engines = [("sqlite", None)]
        if os.environ.get("POSTGRES_DSN"):
            engines.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
        if os.environ.get("MYSQL_DSN"):
            engines.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
        return engines

    def test_stale_owner_resuming_after_its_final_batch_does_not_overwrite_new_claim(self) -> None:
        # A pauses in on_progress after its LAST batch (its compare-and-set
        # already recorded completion). An unconditional completion write
        # after that would overwrite B's freshly claimed watermark, so B is
        # refused at its next batch and the view keeps one chunk only.
        for engine, dsn in self._engines():
            with self.subTest(engine=getattr(engine, "value", engine)):
                self._run(engine, dsn, a_pause_batch=self.ROWS // self.BATCH, b_pause_batch=1, redefine=False)

    def test_stale_owner_at_the_same_cursor_as_new_owner_is_fenced(self) -> None:
        # ABA: A (old definition) pauses after batch 2; B (new definition,
        # same view name, same pk span and batch size) pauses after batch 2,
        # so B's watermark equals the one A committed. A must not take B's
        # watermark for its own and fold the old definition's chunk into the
        # new store.
        for engine, dsn in self._engines():
            with self.subTest(engine=getattr(engine, "value", engine)):
                self._run(engine, dsn, a_pause_batch=2, b_pause_batch=2, redefine=True)

    def test_registration_whose_backfill_is_taken_over_by_repair_keeps_the_repaired_view(self) -> None:
        # register_view(incremental_backfill=True) is paused after batch 1;
        # another session runs repair_view(). The registration's backfill is
        # fenced (BackfillInterruptedError), and its failure handling must not
        # drop the view the repair just populated (MySQL/MariaDB cleanup).
        for engine, dsn in self._engines():
            with self.subTest(engine=getattr(engine, "value", engine)):
                self._run(engine, dsn, a_pause_batch=1, b_pause_batch=0, redefine=False, repair=True)

    def _run(self, engine, dsn, a_pause_batch: int, b_pause_batch: int, redefine: bool,
             repair: bool = False) -> None:
        suffix = uuid.uuid4().hex[:10]
        table, view, schema = f"own_{suffix}", f"own_v_{suffix}", f"mv_own_{suffix}"
        path = None
        sessions = []
        if engine == "sqlite":
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)

            def connect():
                c = sqlite3.connect(path, timeout=30, check_same_thread=False)
                sessions.append(c)
                return c
            dialect, ph = Dialect.SQLITE, "?"
        else:
            def connect():
                c = IndependentSessionBackfillTests._connect_native(engine, dsn)
                if engine == Dialect.POSTGRES:
                    c.cursor().execute(f'SET search_path TO "{schema}"')
                    c.commit()
                sessions.append(c)
                return c
            dialect, ph = engine, "%s"
        admin = (sqlite3.connect(path) if engine == "sqlite"
                 else IndependentSessionBackfillTests._connect_native(engine, dsn))
        cur = admin.cursor()
        old_sql = f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;"
        new_sql = (
            f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} "
            f"WHERE g <> 'G0' GROUP BY g;" if redefine else old_sql
        )
        expected_sql = (
            f"SELECT g, COUNT(*), SUM(amt) FROM {table} "
            + ("WHERE g <> 'G0' " if redefine else "") + "GROUP BY g ORDER BY g"
        )
        a_plan = b_plan = None
        registered_name = None
        res: dict = {}
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(
                f"INSERT INTO {table} VALUES ({ph}, {ph}, {ph})",
                [(i, f"G{i % 3}", i) for i in range(1, self.ROWS + 1)],
            )
            admin.commit()
            ma = MaterializedViewManager(connect(), dialect=dialect)
            mb = MaterializedViewManager(connect(), dialect=dialect)
            if not repair:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    a_plan = ma.register_view(old_sql, populate=False, ignore_limitations=True)
                registered_name = a_plan.view_name
            a_paused, a_go, b_paused, b_go = (threading.Event() for _ in range(4))

            def a_progress(batch: int, _d: int) -> None:
                if batch == a_pause_batch:
                    a_paused.set()
                    if not a_go.wait(60):
                        raise AssertionError("A was never released")

            def b_progress(batch: int, _d: int) -> None:
                if batch == b_pause_batch:
                    b_paused.set()
                    if not b_go.wait(60):
                        raise AssertionError("B was never released")

            def run_a() -> None:
                try:
                    if repair:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            res["a"] = ma.register_view(
                                old_sql, populate=True, incremental_backfill=True, batch_size=self.BATCH,
                                on_progress=a_progress, ignore_limitations=True,
                            )
                    else:
                        res["a"] = ma.backfill_incrementally(
                            a_plan, batch_size=self.BATCH, on_progress=a_progress
                        )
                except BaseException as exc:  # inspected below
                    res["a"] = exc
                a_paused.set()

            def run_b() -> None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        res["b"] = mb.register_view(
                            new_sql, populate=True, incremental_backfill=True, batch_size=self.BATCH,
                            force_rebuild=True, on_progress=b_progress, ignore_limitations=True,
                        )
                except BaseException as exc:  # inspected below
                    res["b"] = exc
                b_paused.set()

            ta = threading.Thread(target=run_a, daemon=True)
            tb = threading.Thread(target=run_b, daemon=True)
            ta.start()
            self.assertTrue(a_paused.wait(60), "A did not reach its pause point")
            self.assertNotIn("a", res, f"A ended before its pause point: {res.get('a')!r}")
            if repair:
                catalog = MaterializedViewManager(connect(), dialect=dialect).get_catalog()
                a_name = registered_name = next(n for n in catalog if n.endswith(view))
                self.assertEqual(mb.repair_view(a_name), {a_name: "repaired"})
                a_go.set()
                ta.join(60)
                self.assertFalse(ta.is_alive(), "A hung after resuming")
                self.assertIsInstance(res["a"], BackfillInterruptedError, repr(res["a"]))
                self.assertIn("taken over", str(res["a"]))
                self._assert_final(cur, admin, dialect, view, expected_sql, a_name, (0, 0, 0))
                return
            tb.start()
            self.assertTrue(b_paused.wait(60), "B did not reach its pause point")
            self.assertNotIn("b", res, f"B ended before its pause point: {res.get('b')!r}")
            a_go.set()
            ta.join(60)
            self.assertFalse(ta.is_alive(), "A hung after resuming")
            b_go.set()
            tb.join(60)
            self.assertFalse(tb.is_alive(), "B hung after resuming")

            b_plan = res["b"]
            self.assertNotIsInstance(b_plan, BaseException, f"new owner failed: {b_plan!r}")
            if redefine:
                # A resumed at a cursor equal to B's watermark: fenced, rolled back.
                self.assertIsInstance(res["a"], BackfillInterruptedError, repr(res["a"]))
                self.assertIn("taken over", str(res["a"]))
            else:
                # A had completed before it paused; resuming only returns.
                self.assertEqual(res["a"], self.ROWS // self.BATCH, repr(res["a"]))
            self._assert_final(
                cur, admin, dialect, view, expected_sql, b_plan.view_name, (self.ROWS, self.ROWS, 0)
            )
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            if engine == "sqlite":
                admin.close()
                os.unlink(path)
            else:
                IndependentSessionBackfillTests._cleanup_native_fixture(
                    admin, engine, schema, table,
                    MaterializedViewManager(admin, dialect=engine) if registered_name else None,
                    types.SimpleNamespace(view_name=registered_name) if registered_name else None,
                )

    def _assert_final(self, cur, admin, dialect, view, expected_sql, view_name, watermark) -> None:
        def rows():
            return [tuple(int(x) if not isinstance(x, str) else x for x in r) for r in cur.fetchall()]
        cur.execute(f"SELECT g, c, s FROM {view} ORDER BY g")
        actual = rows()
        cur.execute(expected_sql)
        expected = rows()
        admin.commit()
        self.assertEqual(actual, expected)
        checker = MaterializedViewManager(admin, dialect=dialect)
        entry = checker.get_catalog()[view_name]
        self.assertEqual(
            (entry["backfill_cursor_pk"], entry["backfill_max_pk"], entry["backfill_active"]), watermark
        )
        self.assertEqual(checker.check_schema_drift()[view_name], "ok")


class CatalogUpgradeTests(unittest.TestCase):
    """Catalogs created by older versions (no backfill_owner column; active
    watermarks written without an ownership token) and catalog setup on
    every manager construction."""

    LEGACY_CATALOG_DDL = (
        "CREATE TABLE {cat} (view_name VARCHAR(255) NOT NULL PRIMARY KEY, view_sql TEXT NOT NULL, "
        "strategy VARCHAR(64) NOT NULL, signature_hash VARCHAR(64) NOT NULL, "
        "backing_table VARCHAR(255) NOT NULL, aux_tables_json TEXT NOT NULL, "
        "trigger_names_json TEXT NOT NULL, function_names_json TEXT NOT NULL, "
        "trigger_ddls_json TEXT NOT NULL, backfill_cursor_pk BIGINT DEFAULT NULL, "
        "backfill_max_pk BIGINT DEFAULT NULL, backfill_active INTEGER NOT NULL DEFAULT 0)"
    )

    def _engines(self):
        return BackfillTakeoverOwnershipTests._engines(self)

    def _fixture(self, engine, dsn):
        """(connect, admin, dialect, ph, prefix, table, view, cleanup)."""
        suffix = uuid.uuid4().hex[:8]
        prefix, table, view, schema = f"__mvl{suffix}_", f"lg_{suffix}", f"lg_v_{suffix}", f"mv_lg_{suffix}"
        sessions: list = []
        if engine == "sqlite":
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)

            def connect():
                c = sqlite3.connect(path, timeout=30, check_same_thread=False)
                sessions.append(c)
                return c
            dialect, ph = Dialect.SQLITE, "?"
        else:
            path = None

            def connect():
                c = IndependentSessionBackfillTests._connect_native(engine, dsn)
                if engine == Dialect.POSTGRES:
                    c.cursor().execute(f'SET search_path TO "{schema}"')
                    c.commit()
                sessions.append(c)
                return c
            dialect, ph = engine, "%s"
        admin = sqlite3.connect(path) if engine == "sqlite" else IndependentSessionBackfillTests._connect_native(engine, dsn)
        cur = admin.cursor()
        if engine == Dialect.POSTGRES:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
        cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
        cur.executemany(f"INSERT INTO {table} VALUES ({ph}, {ph}, {ph})", [(i, f"G{i % 3}", i) for i in range(1, 21)])
        admin.commit()

        def cleanup(manager=None, view_name=None):
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            if engine == "sqlite":
                admin.close()
                os.unlink(path)
                return
            if manager is not None and view_name and engine != Dialect.POSTGRES:
                try:
                    manager.drop_view(view_name)
                except Exception:
                    pass
            try:
                admin.rollback()
                c = admin.cursor()
                if engine == Dialect.POSTGRES:
                    c.execute("SET search_path TO public")
                    c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                else:
                    c.execute(f"DROP TABLE IF EXISTS {prefix}catalog")
                    c.execute(f"DROP TABLE IF EXISTS {table}")
                admin.commit()
            except Exception:
                admin.rollback()
            admin.close()

        return connect, admin, dialect, ph, prefix, table, view, cleanup

    def _rows(self, cur, sql):
        cur.execute(sql)
        return [tuple(int(x) if not isinstance(x, str) else x for x in r) for r in cur.fetchall()]

    def test_legacy_catalog_is_upgraded_and_legacy_active_watermark_fails_closed(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=getattr(engine, "value", engine)):
                connect, admin, dialect, ph, prefix, table, view, cleanup = self._fixture(engine, dsn)
                manager = None
                try:
                    cur = admin.cursor()
                    cur.execute(self.LEGACY_CATALOG_DDL.format(cat=f"{prefix}catalog"))
                    admin.commit()
                    manager = MaterializedViewManager(connect(), dialect=dialect, object_prefix=prefix)
                    cur.execute(f"SELECT backfill_owner FROM {prefix}catalog WHERE 1 = 0")
                    cur.fetchall()
                    admin.commit()
                    view_sql = f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;"
                    native = f"SELECT g, COUNT(*), SUM(amt) FROM {table} GROUP BY g ORDER BY g"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = manager.register_view(view_sql, populate=False, ignore_limitations=True)
                    # An interrupted backfill recorded by older code: active,
                    # no ownership token. It must stay fail-closed.
                    cur.execute(
                        f"UPDATE {prefix}catalog SET backfill_cursor_pk = 5, backfill_max_pk = 20, "
                        f"backfill_active = 1, backfill_owner = NULL WHERE view_name = {ph}",
                        (plan.view_name,),
                    )
                    admin.commit()
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=5)
                    self.assertEqual(manager.check_schema_drift()[plan.view_name], "backfill_incomplete")
                    self.assertEqual(manager.repair_view(plan.view_name), {plan.view_name: "repaired"})
                    self.assertEqual(self._rows(cur, f"SELECT g, c, s FROM {view} ORDER BY g"), self._rows(cur, native))
                    admin.commit()
                    # A fresh backfill on the upgraded catalog works end to end.
                    manager.drop_view(plan.view_name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = manager.register_view(view_sql, populate=False, ignore_limitations=True)
                    self.assertEqual(manager.backfill_incrementally(plan, batch_size=5), 4)
                    self.assertEqual(self._rows(cur, f"SELECT g, c, s FROM {view} ORDER BY g"), self._rows(cur, native))
                    admin.commit()
                finally:
                    cleanup(manager, view)

    def test_manager_construction_does_not_wait_for_open_transactions_on_the_catalog(self) -> None:
        # On PostgreSQL an ALTER TABLE takes ACCESS EXCLUSIVE even when the
        # column already exists, so constructing a manager queued behind any
        # open transaction that read the catalog (every writer trigger does)
        # and then blocked all other writers. With an up-to-date catalog
        # construction must not need a conflicting lock.
        engines = [e for e in self._engines() if e[0] != "sqlite"]
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                connect, admin, dialect, ph, prefix, table, view, cleanup = self._fixture(engine, dsn)
                try:
                    MaterializedViewManager(connect(), dialect=dialect, object_prefix=prefix)
                    holder = connect()
                    hcur = holder.cursor()
                    hcur.execute(f"SELECT COUNT(*) FROM {prefix}catalog")
                    hcur.fetchall()  # transaction stays open, holding its shared lock
                    done: list = []

                    def construct() -> None:
                        try:
                            MaterializedViewManager(connect(), dialect=dialect, object_prefix=prefix)
                            done.append("ok")
                        except BaseException as exc:  # inspected below
                            done.append(exc)

                    th = threading.Thread(target=construct, daemon=True)
                    th.start()
                    th.join(timeout=10)
                    blocked = th.is_alive()
                    holder.rollback()
                    th.join(timeout=60)
                    self.assertFalse(blocked, "manager construction waited for an open transaction on the catalog")
                    self.assertEqual(done, ["ok"])
                finally:
                    cleanup()


class _HookedTemplates(list):
    """chunk_sql_templates that call `hook(batch_attempt)` when a batch starts
    folding, i.e. after the batch took its base-range row locks."""

    def __init__(self, items, hook):
        super().__init__(items)
        self.hook = hook
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        self.hook(self.calls)
        return super().__iter__()


class NativeBackfillLockingTests(unittest.TestCase):
    """Native lock-behaviour contract of a non-snapshot pk-range backfill with
    independent application writers (MySQL/MariaDB REPEATABLE READ triggers
    take shared locks on the catalog row; PostgreSQL triggers take none)."""

    def _engines(self):
        out = []
        if os.environ.get("POSTGRES_DSN"):
            out.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
        if os.environ.get("MYSQL_DSN"):
            out.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
        if not out:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        return out

    def _fixture(self, engine, dsn, rows=40):
        connect = IndependentSessionBackfillTests._connect_native
        suffix = uuid.uuid4().hex[:10]
        fx = {"schema": f"mv_lk_{suffix}", "table": f"lk_{suffix}", "view": f"lk_v_{suffix}", "sessions": []}
        conn = connect(engine, dsn)
        fx["conn"] = conn
        cur = conn.cursor()
        if engine == Dialect.POSTGRES:
            schema = fx["schema"]
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
        engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
        cur.execute(
            f"CREATE TABLE {fx['table']} (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)"
            f"{engine_suffix}"
        )
        cur.executemany(
            f"INSERT INTO {fx['table']} VALUES (%s, %s, %s)", [(i, f"G{i % 4}", i) for i in range(1, rows + 1)]
        )
        conn.commit()
        fx["manager"] = MaterializedViewManager(conn, dialect=engine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            fx["plan"] = fx["manager"].register_view(
                f"CREATE VIEW {fx['view']} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {fx['table']} GROUP BY g;",
                populate=False, ignore_limitations=True,
            )
        return fx

    def _session(self, engine, dsn, fx):
        s = IndependentSessionBackfillTests._connect_native(engine, dsn)
        if engine == Dialect.POSTGRES:
            s.cursor().execute(f'SET search_path TO "{fx["schema"]}"')
            s.commit()
        fx["sessions"].append(s)
        return s

    def _assert_equal_live(self, fx):
        conn = fx["conn"]
        cur = conn.cursor()
        conn.commit()
        cur.execute(f"SELECT g, c, s FROM {fx['view']} ORDER BY g")
        actual = [tuple(r) for r in cur.fetchall()]
        cur.execute(f"SELECT g, COUNT(*), SUM(amt) FROM {fx['table']} GROUP BY g ORDER BY g")
        expected = [tuple(r) for r in cur.fetchall()]
        conn.commit()
        self.assertEqual(actual, expected)

    def _cleanup(self, engine, fx):
        for s in fx["sessions"]:
            try:
                s.rollback()
                s.close()
            except Exception:
                pass
        IndependentSessionBackfillTests._cleanup_native_fixture(
            fx["conn"], engine, fx["schema"], fx["table"],
            MaterializedViewManager(fx["conn"], dialect=engine), fx["plan"],
        )

    def _for_each_engine(self, body):
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = self._fixture(engine, dsn)
                try:
                    body(engine, dsn, fx)
                finally:
                    self._cleanup(engine, fx)

    def test_writer_outside_the_batch_range_proceeds_during_a_batch(self) -> None:
        def body(engine, dsn, fx):
            writer = self._session(engine, dsn, fx)
            timings = []

            def write_outside() -> None:
                started = time.monotonic()
                c = writer.cursor()
                c.execute(f"UPDATE {fx['table']} SET amt = amt + 1 WHERE id = 40")
                writer.commit()
                timings.append(time.monotonic() - started)

            def hook(n: int) -> None:
                if n == 2:  # batch (10, 20] holds its range locks
                    th = threading.Thread(target=write_outside, daemon=True)
                    th.start()
                    th.join(timeout=3)
                    timings.append(("alive", th.is_alive()))
                    if th.is_alive():
                        th.join(timeout=60)

            spec = fx["plan"].incremental_backfill_spec
            object.__setattr__(spec, "chunk_sql_templates", _HookedTemplates(spec.chunk_sql_templates, hook))
            fx["manager"].backfill_incrementally(fx["plan"], batch_size=10)
            self.assertIn(("alive", False), timings, f"writer outside the batch was blocked: {timings!r}")
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_writer_holding_a_row_in_the_batch_range_does_not_deadlock(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _session_id

        def body(engine, dsn, fx):
            writer = self._session(engine, dsn, fx)
            monitor = self._session(engine, dsn, fx)
            backfill_sid = _session_id(engine, fx["conn"])
            wc = writer.cursor()
            wc.execute(f"SELECT id FROM {fx['table']} WHERE id = 3 FOR UPDATE")
            wc.fetchall()
            errors = []

            def writer_finishes() -> None:
                try:
                    deadline = time.monotonic() + 20
                    def blocked_on_range_lock() -> bool:
                        if _is_lock_waiting(engine, monitor, backfill_sid):
                            return True
                        if engine != Dialect.MYSQL:
                            return False
                        # MariaDB may report the waiting trx as RUNNING in
                        # innodb_trx; the range-lock statement cannot finish
                        # while row 3 is locked, so being stuck in it counts.
                        mc = monitor.cursor()
                        mc.execute(
                            "SELECT COUNT(*) FROM information_schema.processlist "
                            "WHERE id = %s AND info LIKE %s AND time >= 1",
                            (backfill_sid, "%FOR UPDATE%"),
                        )
                        stuck = int(mc.fetchone()[0]) > 0
                        monitor.commit()
                        return stuck

                    while not blocked_on_range_lock():
                        if time.monotonic() > deadline:
                            from test_avg_concurrent_maintenance import _lock_diagnostics
                            raise AssertionError(f"backfill never waited for the writer's row; sid={backfill_sid} {_lock_diagnostics(engine, monitor)!r}")
                        time.sleep(0.02)
                    # The backfill now waits for row 3; the writer's own
                    # trigger-maintained update must not deadlock with it.
                    wc.execute(f"UPDATE {fx['table']} SET amt = amt + 5 WHERE id = 3")
                    writer.commit()
                except BaseException as exc:
                    errors.append(exc)
                    writer.rollback()

            th = threading.Thread(target=writer_finishes, daemon=True)
            th.start()
            fx["manager"].backfill_incrementally(fx["plan"], batch_size=10)
            th.join(timeout=60)
            self.assertEqual(errors, [])
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_long_writer_transaction_does_not_break_a_healthy_backfill(self) -> None:
        def body(engine, dsn, fx):
            conn = fx["conn"]
            c = conn.cursor()
            if engine == Dialect.MYSQL:
                c.execute("SET SESSION innodb_lock_wait_timeout = 1")
            else:
                c.execute("SET lock_timeout = '1s'")
            conn.commit()
            writer = self._session(engine, dsn, fx)

            def progress(batch: int, _d: int) -> None:
                if batch == 1:
                    # Holds the store rows of already-folded groups for 2.5 s,
                    # longer than the backfill's lock timeout.
                    writer.cursor().execute(f"UPDATE {fx['table']} SET amt = amt + 100 WHERE id IN (2, 5, 8)")
                    threading.Timer(2.5, writer.commit).start()

            fx["manager"]._backfill_sleep = lambda d: time.sleep(max(d, 0.3))
            fx["manager"].backfill_incrementally(fx["plan"], batch_size=10, on_progress=progress)
            time.sleep(3)
            self.assertEqual(fx["manager"].get_catalog()[fx["plan"].view_name]["backfill_active"], 0)
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_open_writer_transaction_does_not_refuse_the_claim(self) -> None:
        # The open writer transaction inserts above the current MAX(pk): its
        # trigger runs (on MySQL/MariaDB it holds a shared lock on the catalog
        # row) but the row is outside the range the backfill folds, so the
        # check is not confounded by the pre-claim ungated window.
        def body(engine, dsn, fx):
            writer = self._session(engine, dsn, fx)
            writer.cursor().execute(f"INSERT INTO {fx['table']} VALUES (1000, 'G2', 7)")
            threading.Timer(1.0, writer.commit).start()
            fx["manager"].backfill_incrementally(fx["plan"], batch_size=10)
            time.sleep(1.2)
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_repair_view_racing_a_live_backfill(self) -> None:
        def body(engine, dsn, fx):
            other = self._session(engine, dsn, fx)
            other_manager = MaterializedViewManager(other, dialect=engine)
            done = []

            def repair_and_write() -> None:
                done.append(other_manager.repair_view(fx["plan"].view_name))
                c = other.cursor()
                c.execute(f"UPDATE {fx['table']} SET amt = amt + 1000 WHERE id IN (15, 35)")
                c.execute(f"DELETE FROM {fx['table']} WHERE id = 25")
                other.commit()

            def progress(batch: int, _d: int) -> None:
                if batch == 1:
                    th = threading.Thread(target=repair_and_write, daemon=True)
                    th.start()
                    th.join(timeout=30)

            with self.assertRaises(BackfillInterruptedError) as ctx:
                fx["manager"].backfill_incrementally(fx["plan"], batch_size=10, on_progress=progress)
            self.assertIn("taken over", str(ctx.exception))
            self.assertEqual(done, [{fx["plan"].view_name: "repaired"}])
            self._assert_equal_live(fx)
            c = fx["conn"].cursor()
            c.execute(f"INSERT INTO {fx['table']} VALUES (100, 'G1', 3)")
            fx["conn"].commit()
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_repair_view_racing_a_batch_in_flight(self) -> None:
        # repair_view starts while batch 2 has folded (holding its store-row
        # locks) and is about to take the catalog row lock for its ownership
        # check: the two transactions take the catalog row and the store rows
        # in opposite orders, so the engine may pick either as a deadlock
        # victim. Contract: neither error leaks; the repair completes, the
        # backfill either is fenced ("taken over") or finished before the
        # repair, and the view equals the live view and stays maintained.
        from test_avg_concurrent_maintenance import _is_lock_waiting, _lock_diagnostics, _session_id

        def body(engine, dsn, fx):
            other = self._session(engine, dsn, fx)
            monitor = self._session(engine, dsn, fx)
            repair_sid = _session_id(engine, other)
            other_manager = MaterializedViewManager(other, dialect=engine)
            outcome = {}

            def repair() -> None:
                try:
                    outcome["repair"] = other_manager.repair_view(fx["plan"].view_name)
                except BaseException as exc:
                    outcome["repair_error"] = exc

            manager = fx["manager"]
            original_advance = manager._advance_backfill_watermark
            started = []

            def repair_blocked() -> bool:
                if _is_lock_waiting(engine, monitor, repair_sid):
                    return True
                if engine != Dialect.MYSQL:
                    return False
                # MariaDB may report the waiting trx as RUNNING in innodb_trx;
                # a repair statement running for >= 1 s on this tiny fixture
                # is waiting for the batch's store-row locks.
                mc = monitor.cursor()
                mc.execute(
                    "SELECT COUNT(*) FROM information_schema.processlist "
                    "WHERE id = %s AND command = 'Query' AND time >= 1",
                    (repair_sid,),
                )
                stuck = int(mc.fetchone()[0]) > 0
                monitor.commit()
                return stuck

            def advance(cur, view_name, cursor_pk, upper_pk, max_pk, active):
                if upper_pk == 20 and not started:
                    th = threading.Thread(target=repair, daemon=True)
                    started.append(th)
                    th.start()
                    deadline = time.monotonic() + 20
                    while not repair_blocked() and "repair" not in outcome and "repair_error" not in outcome:
                        if time.monotonic() > deadline:
                            raise AssertionError(f"repair never blocked: {_lock_diagnostics(engine, monitor)!r}")
                        time.sleep(0.02)
                return original_advance(cur, view_name, cursor_pk, upper_pk, max_pk, active)

            manager._advance_backfill_watermark = advance
            manager._backfill_sleep = lambda d: time.sleep(min(d, 0.2))
            other_manager._backfill_sleep = lambda d: time.sleep(min(d, 0.2))
            try:
                manager.backfill_incrementally(fx["plan"], batch_size=10)
                backfill = "completed"
            except BackfillInterruptedError as exc:
                self.assertIn("taken over", str(exc))
                backfill = "fenced"
            self.assertTrue(started)
            started[0].join(timeout=60)
            self.assertNotIn("repair_error", outcome, repr(outcome.get("repair_error")))
            self.assertEqual(outcome.get("repair"), {fx["plan"].view_name: "repaired"}, backfill)
            self._assert_equal_live(fx)
            entry = manager.get_catalog()[fx["plan"].view_name]
            self.assertEqual(entry["backfill_active"], 0)
            c = fx["conn"].cursor()
            c.execute(f"UPDATE {fx['table']} SET amt = amt + 7 WHERE id IN (5, 30)")
            c.execute(f"INSERT INTO {fx['table']} VALUES (100, 'G1', 3)")
            fx["conn"].commit()
            self._assert_equal_live(fx)

        self._for_each_engine(body)

    def test_repair_view_gives_up_after_bounded_lock_retries_leaving_view_unchanged(self) -> None:
        # A session holding the catalog row lock past every retry: repair_view
        # re-runs its transaction a bounded number of times, then propagates
        # the engine's lock-timeout error with the watermark and contents
        # unchanged (all-or-nothing).
        def body(engine, dsn, fx):
            manager = fx["manager"]
            manager.backfill_incrementally(fx["plan"], batch_size=10)
            holder = self._session(engine, dsn, fx)
            repairer = self._session(engine, dsn, fx)
            rc = repairer.cursor()
            if engine == Dialect.MYSQL:
                rc.execute("SET SESSION innodb_lock_wait_timeout = 1")
            else:
                rc.execute("SET lock_timeout = '200ms'")
            repairer.commit()
            rm = MaterializedViewManager(repairer, dialect=engine)
            rm._BACKFILL_TRANSIENT_RETRIES = 2
            sleeps = []
            rm._backfill_sleep = sleeps.append
            catalog = rm.catalog_table
            before = manager.get_catalog()[fx["plan"].view_name]
            hc = holder.cursor()
            hc.execute(
                f"SELECT view_name FROM {catalog} WHERE view_name = %s FOR UPDATE", (fx["plan"].view_name,)
            )
            hc.fetchall()
            try:
                with self.assertRaises(Exception) as ctx:
                    rm.repair_view(fx["plan"].view_name)
                self.assertNotIsInstance(ctx.exception, BackfillInterruptedError)
                self.assertTrue(rm._is_transient_lock_error(ctx.exception) or rm._is_lock_timeout_error(ctx.exception),
                                repr(ctx.exception))
                self.assertEqual(len(sleeps), 2)
            finally:
                holder.rollback()
            after = manager.get_catalog()[fx["plan"].view_name]
            self.assertEqual(
                (after["backfill_cursor_pk"], after["backfill_max_pk"], after["backfill_active"]),
                (before["backfill_cursor_pk"], before["backfill_max_pk"], before["backfill_active"]),
            )
            self._assert_equal_live(fx)
            self.assertEqual(rm.repair_view(fx["plan"].view_name), {fx["plan"].view_name: "repaired"})
            self._assert_equal_live(fx)

        self._for_each_engine(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
