#!/usr/bin/env python3
"""Independent-session tests for incremental materialized-view backfill."""

from __future__ import annotations

from decimal import Decimal

import os
import re
import sqlite3
import tempfile
import threading
import time
import types
import unittest
import warnings
import uuid
from urllib.parse import urlparse

from matview_sync import (
    BackfillInterruptedError,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaterializedViewManager,
    TransactionOwnershipError,
    UnsafeReplacementError,
)
from test_matview_sync import FLOAT_OPT_IN, _register_sqlite_compat_udfs, pin_native_plan

# Bound for replaying an application writer transaction chosen as an engine
# deadlock/serialization victim. Exhausting it fails the test.
TRANSIENT_RETRIES = 25



def _key_collation(engine: Dialect) -> str:
    """MySQL/MariaDB text group-key columns of these concurrency fixtures declare a
    byte-exact collation: under a PAD SPACE UCA server default (CI runs
    utf8mb4_unicode_ci) MySQL groups text plan-dependently and matview maintains
    such views by an announced full refresh, which has no incremental backfill to
    race (see test_group_key_collation.py)."""
    return " COLLATE utf8mb4_bin" if engine == Dialect.MYSQL else ""

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
    def test_registration_blocks_independent_writer_until_initial_population_commits(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="matview-registration-race-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE registration_rows (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, value INTEGER NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO registration_rows VALUES (?, ?, ?);",
            [(1, "a", 10), (2, "a", 20), (3, "b", 30)],
        )
        conn.commit()

        writer_started = threading.Event()
        statement_started = threading.Event()
        writer_finished = threading.Event()
        writer_errors: list[BaseException] = []
        writer_thread: threading.Thread | None = None
        committed_during_install: list[bool] = []

        def write_from_independent_session() -> None:
            second = sqlite3.connect(db_path, timeout=10)
            try:
                second.execute("PRAGMA busy_timeout=10000;")
                _register_sqlite_compat_udfs(second)

                def trace_writer(statement: str) -> None:
                    if statement.lstrip().upper().startswith(
                        "INSERT INTO REGISTRATION_ROWS"
                    ):
                        statement_started.set()

                second.set_trace_callback(trace_writer)
                writer_started.set()
                second.execute("INSERT INTO registration_rows VALUES (4, 'b', 40);")
                second.commit()
            except BaseException as exc:
                writer_errors.append(exc)
            finally:
                second.close()
                writer_finished.set()

        def trace_registration(statement: str) -> None:
            nonlocal writer_thread
            if not statement.lstrip().upper().startswith(
                "INSERT INTO __MATVIEW_CATALOG"
            ):
                return
            if writer_thread is not None:
                return

            # All maintenance triggers now exist, but initial population has not
            # started. The independent INSERT must reach SQLite yet remain blocked
            # until registration commits, otherwise trigger and backfill effects
            # can overlap.
            writer_thread = threading.Thread(
                target=write_from_independent_session, daemon=True
            )
            writer_thread.start()
            if not writer_started.wait(timeout=10):
                committed_during_install.append(True)
                return
            if not statement_started.wait(timeout=10):
                committed_during_install.append(True)
                return
            if writer_finished.wait(timeout=0.5):
                committed_during_install.append(True)

        manager = MaterializedViewManager(conn)
        try:
            conn.set_trace_callback(trace_registration)
            plan = manager.register_view(
                "CREATE VIEW registration_totals AS "
                "SELECT bucket, COUNT(*) AS n, SUM(value) AS total "
                "FROM registration_rows GROUP BY bucket;"
            )
            conn.set_trace_callback(None)
            self.assertIsNotNone(writer_thread)
            writer_thread.join(timeout=10)
            self.assertFalse(
                writer_thread.is_alive(),
                "independent writer did not resume after registration committed",
            )
            if writer_errors:
                raise writer_errors[0]
            self.assertTrue(statement_started.is_set())
            self.assertFalse(
                committed_during_install,
                "independent writer committed between trigger installation and initial population",
            )

            actual = conn.execute(
                "SELECT bucket, n, total FROM registration_totals ORDER BY bucket;"
            ).fetchall()
            expected = conn.execute(
                "SELECT bucket, COUNT(*) AS n, SUM(value) AS total "
                "FROM registration_rows GROUP BY bucket ORDER BY bucket;"
            ).fetchall()
            self.assertEqual(actual, expected)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM registration_rows;").fetchone()[0],
                4,
            )
            self.assertEqual(manager.get_catalog()[plan.view_name]["backfill_active"], 0)
        finally:
            conn.set_trace_callback(None)
            if writer_thread is not None and writer_thread.is_alive():
                writer_thread.join(timeout=10)
            conn.close()
            os.unlink(db_path)

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
                    self.assertIsNotNone(catalog["backfill_cursor_pk"])
                    self.assertIsNotNone(catalog["backfill_max_pk"])
                    with self.assertRaises(BackfillInterruptedError):
                        manager.backfill_incrementally(plan, batch_size=2, snapshot_isolation=True)
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
                    self.assertEqual(catalog_entry["backfill_active"], 1)
                    self.assertIsNone(catalog_entry["backfill_cursor_pk"])
                    self.assertIsNone(catalog_entry["backfill_max_pk"])
                    self.assertEqual(manager.check_schema_drift()[plan.view_name], "backfill_incomplete")
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
                f"bucket VARCHAR(16){_key_collation(engine)} NOT NULL, "
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
                    # AVG over DOUBLE PRECISION is refused by default
                    # (plan-dependent accumulation order); per-code opt-in.
                    accept_limitations=FLOAT_OPT_IN,
                )
            limitation_warnings = [
                warning for warning in emitted
                if issubclass(warning.category, EngineLimitationWarning)
            ]
            expected_limitation_codes = (
                {
                    "MYSQL_NO_TRUNCATE_TRIGGER",
                    "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX",
                    "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE",
                }
                if engine == Dialect.MYSQL
                else {"ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"}
            )
            self.assertEqual(
                {lim.code for lim in plan.limitations},
                expected_limitation_codes,
            )
            self.assertEqual(len(limitation_warnings), 1)
            for code in expected_limitation_codes:
                self.assertIn(code, str(limitation_warnings[0].message))

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
            # Opted-in approximate AVG: exact only under the pinned plan (small
            # table, PostgreSQL serial seq scan + HashAggregate).
            pin_native_plan(cur, engine)
            cur.execute(
                f"SELECT bucket, COUNT(*) AS c, SUM(amt) AS s, "
                f"AVG(amt) AS avg_amt, AVG(approx_amt) AS avg_approx_amt "
                f"FROM {table_name} GROUP BY bucket;"
            )
            expected = sorted((tuple(row) for row in cur.fetchall()), key=lambda r: r[0])
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
                f"bucket VARCHAR(16){_key_collation(engine)} NOT NULL, "
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
                f"id BIGINT PRIMARY KEY, grp VARCHAR(16){_key_collation(engine)} NOT NULL, amount BIGINT NOT NULL"
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
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)")
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
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)")
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
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)")
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
                # A's completion is recorded, so the view is published and
                # complete: replacing it under an incremental backfill is
                # refused (UnsafeReplacementError, readers would see a partial
                # view), and the new owner takes over via drop_view() +
                # register_view() instead.
                self._run(
                    engine, dsn, a_pause_batch=self.ROWS // self.BATCH, b_pause_batch=1, redefine=False,
                    b_replaces_by_drop=True,
                )

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
             repair: bool = False, b_replaces_by_drop: bool = False) -> None:
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
            cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)")
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
                        if b_replaces_by_drop:
                            try:
                                mb.register_view(
                                    new_sql, populate=True, incremental_backfill=True, batch_size=self.BATCH,
                                    force_rebuild=True, ignore_limitations=True,
                                )
                            except UnsafeReplacementError:
                                res["b_refused"] = True
                            mb.drop_view(registered_name)
                        res["b"] = mb.register_view(
                            new_sql, populate=True, incremental_backfill=True, batch_size=self.BATCH,
                            force_rebuild=not b_replaces_by_drop, on_progress=b_progress, ignore_limitations=True,
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
            if b_replaces_by_drop:
                self.assertTrue(res.get("b_refused"), "incremental replacement of a complete view was not refused")
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
    """Active watermarks without an ownership token and catalog setup on
    every manager construction."""

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
        cur.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)")
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

    def test_ownerless_active_watermark_fails_closed(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=getattr(engine, "value", engine)):
                connect, admin, dialect, ph, prefix, table, view, cleanup = self._fixture(engine, dsn)
                manager = None
                try:
                    cur = admin.cursor()
                    manager = MaterializedViewManager(connect(), dialect=dialect, object_prefix=prefix)
                    view_sql = f"CREATE VIEW {view} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM {table} GROUP BY g;"
                    native = f"SELECT g, COUNT(*), SUM(amt) FROM {table} GROUP BY g ORDER BY g"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = manager.register_view(view_sql, populate=False, ignore_limitations=True)
                    # An interrupted backfill recorded without an ownership token
                    # (active, no owner). It must stay fail-closed.
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
                    # A fresh backfill afterwards works end to end.
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
            f"CREATE TABLE {fx['table']} (id BIGINT PRIMARY KEY, g VARCHAR(8){_key_collation(engine)} NOT NULL, amt BIGINT NOT NULL)"
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
            writer_sid = _session_id(engine, writer)
            wc = writer.cursor()
            wc.execute(f"SELECT id FROM {fx['table']} WHERE id = 3 FOR UPDATE")
            wc.fetchall()
            errors = []

            def writer_finishes() -> None:
                try:
                    deadline = time.monotonic() + 20
                    def blocked_on_range_lock() -> bool:
                        # Positive engine evidence only, attributed to the
                        # writer holding row 3 (see _lock_wait_evidence).
                        return _is_lock_waiting(engine, monitor, backfill_sid, blocker_sid=writer_sid)

                    while not blocked_on_range_lock():
                        if time.monotonic() > deadline:
                            from test_avg_concurrent_maintenance import _lock_diagnostics
                            raise AssertionError(
                                f"backfill never waited for the writer's row; sid={backfill_sid} "
                                f"{_lock_diagnostics(engine, monitor, backfill_sid)!r}"
                            )
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
            backfill_sid = _session_id(engine, fx["conn"])
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
                # Positive engine evidence only: the repair waits for a lock
                # held by the in-flight batch's session (see
                # _lock_wait_evidence).
                return _is_lock_waiting(engine, monitor, repair_sid, blocker_sid=backfill_sid)

            def advance(cur, view_name, cursor_pk, upper_pk, max_pk, active):
                if upper_pk == 20 and not started:
                    th = threading.Thread(target=repair, daemon=True)
                    started.append(th)
                    th.start()
                    deadline = time.monotonic() + 20
                    while not repair_blocked() and "repair" not in outcome and "repair_error" not in outcome:
                        if time.monotonic() > deadline:
                            raise AssertionError(
                                f"repair never blocked: {_lock_diagnostics(engine, monitor, repair_sid)!r}"
                            )
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


class PostgresRepairLockTargetsTests(unittest.TestCase):
    """Offline structural checks for PostgreSQL repair_view() table locking."""

    def test_lock_targets_deduplicates_sorts_and_preserves_postgres_quoting(self) -> None:
        from matview.manager import _postgres_base_table_lock_targets
        from matview_sync import generate_materialized_view, parse_table_schema

        schema = parse_table_schema(
            """
            CREATE TABLE z_tbl (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);
            CREATE TABLE a_tbl (id INT PRIMARY KEY, z_id INT NOT NULL, w INT NOT NULL);
            """,
            dialect=Dialect.POSTGRES,
        )

        # Reversed join order -> deterministic sorted order.
        p_join = generate_materialized_view(
            schema,
            "CREATE VIEW v_join AS SELECT z.g, SUM(a.w) AS s FROM z_tbl z JOIN a_tbl a ON a.z_id = z.id GROUP BY z.g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(_postgres_base_table_lock_targets(p_join), ["a_tbl", "z_tbl"])

        # Self-join -> deduplicated to one target.
        p_self = generate_materialized_view(
            schema,
            "CREATE VIEW v_self AS SELECT a.g, COUNT(*) AS c FROM z_tbl a JOIN Z_TBL b ON a.id = b.id GROUP BY a.g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(_postgres_base_table_lock_targets(p_self), ["z_tbl"])

        # UNION ALL -> both tables sorted.
        p_union = generate_materialized_view(
            schema,
            "CREATE VIEW v_union AS SELECT id, g FROM z_tbl UNION ALL SELECT id, z_id FROM a_tbl;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(_postgres_base_table_lock_targets(p_union), ["a_tbl", "z_tbl"])

        # CTE + EXISTS subquery -> CTE excluded, real tables sorted.
        p_cte = generate_materialized_view(
            schema,
            (
                "CREATE VIEW v_cte AS "
                "WITH filtered AS (SELECT id, g FROM z_tbl WHERE v > 0) "
                "SELECT f.g, COUNT(*) AS c FROM filtered f "
                "WHERE EXISTS (SELECT 1 FROM a_tbl a WHERE a.z_id = f.id) GROUP BY f.g;"
            ),
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(_postgres_base_table_lock_targets(p_cte), ["a_tbl", "z_tbl"])

        # Quoted reserved keyword, mixed-case, spaces, embedded quotes, schema qualifiers,
        # shadowed subquery CTE, and table-valued function (not lockable).
        fake_spec = types.SimpleNamespace(
            raw_select_sql=(
                'WITH "Shadow" AS (SELECT 1 AS x) '
                'SELECT * FROM "Shadow" '
                'JOIN s2."MixedCase" AS m ON true '
                'JOIN "order" AS o ON true '
                'JOIN "my table" AS mt ON true '
                'JOIN "a""b" AS ab ON true '
                'JOIN s1.plain AS p ON true '
                'LEFT JOIN generate_series(1, 3) AS gs ON true '
                'WHERE EXISTS (SELECT 1 FROM (WITH z AS (SELECT * FROM "order") SELECT * FROM z) sub)'
            ),
            referenced_tables=(),
        )
        fake_plan = types.SimpleNamespace(sub_plans=[], view_spec=fake_spec)
        self.assertEqual(
            _postgres_base_table_lock_targets(fake_plan),
            ['s2."MixedCase"', '"a""b"', '"my table"', '"order"', "s1.plain"],
        )

        # Sort order is invariant to optional schema qualification: public.a_tbl < m_tbl.
        qual_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql="SELECT * FROM m_tbl m JOIN public.a_tbl a ON m.id = a.id",
                referenced_tables=(),
            ),
        )
        self.assertEqual(_postgres_base_table_lock_targets(qual_plan), ["public.a_tbl", "m_tbl"])

        # Non-recursive self-shadowing CTE: inner `t` resolves to base table `t`.
        self_shadow_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql="WITH t AS (SELECT * FROM t WHERE v > 0) SELECT * FROM t JOIN u ON t.id = u.id",
                referenced_tables=(),
            ),
        )
        self.assertEqual(_postgres_base_table_lock_targets(self_shadow_plan), ["t", "u"])

        # Non-recursive forward-referenced CTE: `b` inside `a` resolves to base table `b`.
        fwd_cte_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql="WITH a AS (SELECT * FROM b), b AS (SELECT * FROM real_b) SELECT * FROM a",
                referenced_tables=(),
            ),
        )
        self.assertEqual(_postgres_base_table_lock_targets(fwd_cte_plan), ["b", "real_b"])

        # WITH RECURSIVE: all CTEs in the WITH clause are mutually visible.
        rec_cte_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql=(
                    "WITH RECURSIVE a AS (SELECT * FROM b), "
                    "b AS (SELECT * FROM base_t UNION ALL SELECT * FROM b) "
                    "SELECT * FROM a"
                ),
                referenced_tables=(),
            ),
        )
        self.assertEqual(_postgres_base_table_lock_targets(rec_cte_plan), ["base_t"])

        # ONLY and LATERAL references.
        only_lat_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql=(
                    "SELECT * FROM ONLY z_tbl, "
                    "LATERAL (SELECT * FROM a_tbl WHERE a_tbl.z_id = z_tbl.id) s"
                ),
                referenced_tables=(),
            ),
        )
        self.assertEqual(_postgres_base_table_lock_targets(only_lat_plan), ["a_tbl", "z_tbl"])

        # Fallback when raw_select_sql fails AST parsing (e.g. ONLY (t)) + custom object_prefix.
        fallback_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(
                raw_select_sql="SELECT * FROM ONLY (z_tbl)",
                referenced_tables=("public.z_tbl", "a_tbl", "__custom_store"),
            ),
        )
        self.assertEqual(
            _postgres_base_table_lock_targets(fallback_plan, object_prefix="__custom_"),
            ["a_tbl", "public.z_tbl"],
        )

        # Tableless constant query -> nothing to lock.
        const_plan = types.SimpleNamespace(
            sub_plans=[],
            view_spec=types.SimpleNamespace(raw_select_sql="SELECT 1 AS one", referenced_tables=()),
        )
        self.assertEqual(_postgres_base_table_lock_targets(const_plan), [])

    def test_repair_view_contents_locks_catalog_then_base_tables_and_reacquires_on_retry(self) -> None:
        from matview_sync import generate_materialized_view, parse_table_schema

        schema = parse_table_schema(
            """
            CREATE TABLE z_tbl (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);
            CREATE TABLE a_tbl (id INT PRIMARY KEY, z_id INT NOT NULL, w INT NOT NULL);
            """,
            dialect=Dialect.POSTGRES,
        )
        plan = generate_materialized_view(
            schema,
            "CREATE VIEW v_j AS SELECT z.g, SUM(a.w) AS s FROM z_tbl z JOIN a_tbl a ON a.z_id = z.id GROUP BY z.g;",
            dialect=Dialect.POSTGRES,
        )

        executed: List[str] = []
        fail_once = {"done": False}

        class _DeadlockError(Exception):
            pgcode = "40P01"

        class _FakeCursor:
            def __init__(self) -> None:
                self._rows: List[Tuple[Any, ...]] = []

            def execute(self, sql: str, params: Any = None) -> None:
                executed.append(sql.strip())
                if sql.strip().startswith("SELECT to_regclass"):
                    self._rows = [("__matview_catalog",)]
                else:
                    self._rows = []
                if sql.strip().startswith(f"DELETE FROM {plan.backing_table_name}") and not fail_once["done"]:
                    fail_once["done"] = True
                    raise _DeadlockError("deadlock detected")

            def fetchone(self) -> Any:
                return self._rows[0] if self._rows else None

            def close(self) -> None:
                pass

        class _FakePgConn:
            def cursor(self) -> _FakeCursor:
                return _FakeCursor()

            def commit(self) -> None:
                executed.append("COMMIT")

            def rollback(self) -> None:
                executed.append("ROLLBACK")

        mgr = MaterializedViewManager(
            _FakePgConn(),
            dialect=Dialect.POSTGRES,
            engine_version="postgresql 18",
            skip_ensure_catalog=True,
        )
        sleeps: List[float] = []
        mgr._backfill_sleep = sleeps.append
        mgr._repair_view_contents(plan, plan.view_name)

        self.assertEqual(len(sleeps), 1)
        rb_idx = executed.index("ROLLBACK")
        attempt_0 = executed[:rb_idx]
        attempt_1 = executed[rb_idx + 1 :]
        for attempt_sqls in (attempt_0, attempt_1):
            cat_lock = next(
                i for i, s in enumerate(attempt_sqls)
                if s == "LOCK TABLE __matview_catalog IN SHARE ROW EXCLUSIVE MODE"
            )
            base_lock = next(
                i for i, s in enumerate(attempt_sqls)
                if s == "LOCK TABLE a_tbl, z_tbl IN SHARE MODE;"
            )
            cat_upd = next(
                i for i, s in enumerate(attempt_sqls)
                if s.startswith("UPDATE __matview_catalog SET backfill_cursor_pk")
            )
            del_store = next(
                i for i, s in enumerate(attempt_sqls)
                if s.startswith(f"DELETE FROM {plan.backing_table_name}")
            )
            self.assertLess(cat_lock, base_lock)
            self.assertLess(base_lock, cat_upd)
            self.assertLess(cat_upd, del_store)
        self.assertEqual(attempt_1[-1], "COMMIT")


class NativePostgresRepairConcurrencyTests(unittest.TestCase):
    """Native PostgreSQL concurrency tests proving repair_view() is exact under
    concurrent READ COMMITTED writers without blocking readers."""

    def _dsn(self) -> str:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set; native PostgreSQL test only")
        return dsn

    def _connect_in_schema(self, dsn: str, schema: str):
        conn = IndependentSessionBackfillTests._connect_native(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        cur.execute(f'SET search_path TO "{schema}"')
        conn.commit()
        cur.close()
        return conn

    def _make_schema(self, dsn: str) -> Tuple[str, Any]:
        schema = f"mv_rep_{uuid.uuid4().hex[:10]}"
        conn = IndependentSessionBackfillTests._connect_native(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        cur.execute(f'CREATE SCHEMA "{schema}"')
        cur.execute(f'SET search_path TO "{schema}"')
        conn.commit()
        cur.close()
        return schema, conn

    def _drop_schema(self, dsn: str, schema: str, sessions: Sequence[Any]) -> None:
        for s in sessions:
            try:
                s.rollback()
                s.close()
            except Exception:
                pass
        cleaner = IndependentSessionBackfillTests._connect_native(Dialect.POSTGRES, dsn)
        try:
            cur = cleaner.cursor()
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            cleaner.commit()
            cur.close()
        finally:
            cleaner.close()

    def _assert_view_matches_live(self, conn: Any, view_sql: str, live_sql: str) -> None:
        conn.commit()
        cur = conn.cursor()
        try:
            cur.execute(view_sql)
            actual_rows = [tuple(r) for r in cur.fetchall()]
            cur.execute(live_sql)
            expected_rows = [tuple(r) for r in cur.fetchall()]
            conn.commit()
        finally:
            cur.close()

        def _typed(rows: Sequence[Tuple[Any, ...]]) -> List[Tuple[Tuple[type, Any], ...]]:
            return [
                tuple(
                    (type(v), v.as_tuple() if isinstance(v, Decimal) else v)
                    for v in row
                )
                for row in rows
            ]

        self.assertEqual(_typed(actual_rows), _typed(expected_rows), f"actual={actual_rows!r} expected={expected_rows!r}")

    def test_writer_in_flight_before_repair_blocks_repair_and_commits_cleanly(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _lock_diagnostics, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(
                "INSERT INTO t VALUES (%s, %s, %s)", [(i, f"G{i % 4}", i) for i in range(1, 21)]
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            mgr.register_view("CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;")

            writer = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            reader = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([writer, repairer, reader, monitor])

            writer_sid = _session_id(Dialect.POSTGRES, writer)
            repair_sid = _session_id(Dialect.POSTGRES, repairer)
            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)

            # Writer modifies base table with a NEW group (no lock on any
            # pre-existing backing row) and holds ROW EXCLUSIVE on `t`.
            wcur = writer.cursor()
            wcur.execute("INSERT INTO t VALUES (101, 'G_NEW', 1000)")

            outcome: Dict[str, Any] = {}

            def run_repair() -> None:
                try:
                    outcome["res"] = rmgr.repair_view("v")
                except BaseException as exc:
                    outcome["err"] = exc

            th = threading.Thread(target=run_repair, daemon=True)
            th.start()

            deadline = time.monotonic() + 20.0
            waited = False
            while time.monotonic() < deadline:
                if _is_lock_waiting(Dialect.POSTGRES, monitor, repair_sid, blocker_sid=writer_sid):
                    waited = True
                    break
                if "res" in outcome or "err" in outcome:
                    break
                time.sleep(0.02)
            self.assertTrue(
                waited,
                f"repair_view() did not wait for in-flight writer: outcome={outcome!r} "
                f"diag={_lock_diagnostics(Dialect.POSTGRES, monitor, repair_sid)!r}",
            )

            # Readers are never blocked while repair waits for the writer.
            rcur = reader.cursor()
            rcur.execute("SELECT g, c, s FROM v ORDER BY g")
            self.assertEqual(len(rcur.fetchall()), 4)
            reader.commit()
            rcur.close()

            # In-flight writer runs another DML statement and commits without deadlocking.
            wcur.execute("UPDATE t SET amt = amt + 500 WHERE id = 1")
            writer.commit()
            wcur.close()

            th.join(timeout=20.0)
            self.assertFalse(th.is_alive())
            self.assertNotIn("err", outcome, repr(outcome.get("err")))
            self.assertEqual(outcome.get("res"), {"v": "repaired"})

            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
            # Subsequent write continues to be maintained accurately.
            cur = admin.cursor()
            cur.execute("INSERT INTO t VALUES (102, 'G2', 77)")
            admin.commit()
            cur.close()
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_writer_during_repair_is_serialized_and_never_lost(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _lock_diagnostics, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(
                "INSERT INTO t VALUES (%s, %s, %s)", [(i, f"G{i % 4}", i) for i in range(1, 21)]
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(
                "CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;"
            )

            writer = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            reader = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([writer, repairer, reader, monitor])

            writer_sid = _session_id(Dialect.POSTGRES, writer)
            repair_sid = _session_id(Dialect.POSTGRES, repairer)

            entered_window = threading.Event()
            release_repair = threading.Event()
            backing_tbl = plan.backing_table_name

            class _PausingCursor:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                    res = self._inner.execute(sql, *args, **kwargs)
                    if sql.strip().startswith(f"DELETE FROM {backing_tbl}"):
                        entered_window.set()
                        release_repair.wait(timeout=20.0)
                    return res

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            class _PausingConn:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def cursor(self, *args: Any, **kwargs: Any) -> Any:
                    return _PausingCursor(self._inner.cursor(*args, **kwargs))

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            rmgr = MaterializedViewManager(_PausingConn(repairer), dialect=Dialect.POSTGRES)
            repair_out: Dict[str, Any] = {}
            writer_out: Dict[str, Any] = {}

            def run_repair() -> None:
                try:
                    repair_out["res"] = rmgr.repair_view("v")
                except BaseException as exc:
                    repair_out["err"] = exc

            def run_writer() -> None:
                try:
                    # Insert into a brand-new group so the writer never
                    # collides with rows deleted by DELETE FROM backing_table;
                    # only a SHARE lock on `t` serializes this writer.
                    wcur = writer.cursor()
                    wcur.execute("INSERT INTO t VALUES (201, 'G_NEW', 5000)")
                    writer.commit()
                    wcur.close()
                    writer_out["ok"] = True
                except BaseException as exc:
                    writer_out["err"] = exc

            r_th = threading.Thread(target=run_repair, daemon=True)
            r_th.start()
            self.assertTrue(entered_window.wait(timeout=20.0), f"repair never reached DELETE: {repair_out!r}")

            try:
                # Readers still see the committed pre-repair view without blocking.
                rcur = reader.cursor()
                rcur.execute("SELECT g, c, s FROM v ORDER BY g")
                self.assertEqual(len(rcur.fetchall()), 4)
                reader.commit()
                rcur.close()

                w_th = threading.Thread(target=run_writer, daemon=True)
                w_th.start()

                deadline = time.monotonic() + 20.0
                writer_waited = False
                while time.monotonic() < deadline:
                    if _is_lock_waiting(Dialect.POSTGRES, monitor, writer_sid, blocker_sid=repair_sid):
                        writer_waited = True
                        break
                    if "ok" in writer_out or "err" in writer_out:
                        break
                    time.sleep(0.02)
            finally:
                release_repair.set()

            r_th.join(timeout=20.0)
            w_th.join(timeout=20.0)
            self.assertFalse(r_th.is_alive())
            self.assertFalse(w_th.is_alive())
            self.assertNotIn("err", repair_out, repr(repair_out.get("err")))
            self.assertNotIn("err", writer_out, repr(writer_out.get("err")))
            self.assertEqual(repair_out.get("res"), {"v": "repaired"})
            self.assertTrue(
                writer_waited,
                "writer committed during repair instead of waiting for repair_view()'s SHARE lock "
                f"(diag={_lock_diagnostics(Dialect.POSTGRES, monitor, writer_sid)!r})",
            )

            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_interrupted_and_completed_views_repaired_under_concurrent_writers_stay_exact(self) -> None:
        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(
                "INSERT INTO t VALUES (%s, %s, %s)", [(i, f"G{i % 4}", i) for i in range(1, 41)]
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(
                "CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;",
                populate=False,
            )

            # Round 1: interrupted backfill (active watermark gating rows in (10, 40]).
            # Round 2: completed view repaired repeatedly under concurrent writers.
            for mode in ("interrupted", "completed"):
                with self.subTest(mode=mode):
                    if mode == "interrupted":
                        def boom(batch_idx: int, _done: int) -> None:
                            if batch_idx == 1:
                                raise RuntimeError("simulated crash after chunk 1")

                        with self.assertRaises(RuntimeError):
                            mgr.backfill_incrementally(plan, batch_size=10, on_progress=boom)
                        entry = mgr.get_catalog()["v"]
                        self.assertEqual(entry["backfill_active"], 1)

                    stop = threading.Event()
                    ready = threading.Barrier(4)
                    writer_errors: List[BaseException] = []
                    writes_done = [0, 0, 0]
                    worker_conns = [self._connect_in_schema(dsn, schema) for _ in range(3)]
                    sessions.extend(worker_conns)

                    def worker(wid: int) -> None:
                        wconn = worker_conns[wid]
                        base_id = (10_000 if mode == "interrupted" else 1_000_000) + wid * 200_000
                        ready.wait(timeout=20.0)
                        step = 0
                        wcur = wconn.cursor()
                        while not stop.is_set():
                            step += 1
                            try:
                                op = step % 3
                                if op == 0:
                                    # Insert into both existing and new groups.
                                    grp = f"N{wid}_{step % 5}" if step % 6 == 0 else f"G{(wid + step) % 4}"
                                    wcur.execute(
                                        "INSERT INTO t VALUES (%s, %s, %s)",
                                        (base_id + step, grp, step * 3),
                                    )
                                elif op == 1:
                                    # Update rows on both sides of the watermark without opposite-order group cycles.
                                    target_id = ((wid * 11 + step) % 40) + 1
                                    wcur.execute(
                                        "UPDATE t SET amt = amt + %s WHERE id = %s",
                                        (5, target_id),
                                    )
                                else:
                                    wcur.execute(
                                        "DELETE FROM t WHERE id = %s",
                                        (base_id + step - 2,),
                                    )
                                wconn.commit()
                                writes_done[wid] += 1
                            except BaseException as exc:
                                wconn.rollback()
                                writer_errors.append(exc)
                                break
                        wcur.close()

                    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(3)]
                    for th in threads:
                        th.start()
                    ready.wait(timeout=20.0)

                    repairer = self._connect_in_schema(dsn, schema)
                    sessions.append(repairer)
                    rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)
                    try:
                        for _ in range(4):
                            time.sleep(0.015)
                            self.assertEqual(rmgr.repair_view("v"), {"v": "repaired"})
                    finally:
                        stop.set()
                        for th in threads:
                            th.join(timeout=30.0)

                    self.assertEqual(writer_errors, [], repr(writer_errors))
                    self.assertTrue(all(c > 0 for c in writes_done), f"writes_done={writes_done!r}")
                    self.assertEqual(mgr.get_catalog()["v"]["backfill_active"], 0)
                    self._assert_view_matches_live(
                        admin,
                        "SELECT g, c, s FROM v ORDER BY g",
                        "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
                    )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_join_view_repair_serializes_writers_on_both_base_tables(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE customers (id BIGINT PRIMARY KEY, region VARCHAR(8) NOT NULL)")
            cur.execute("CREATE TABLE orders (id BIGINT PRIMARY KEY, cust_id BIGINT NOT NULL, amt BIGINT NOT NULL)")
            # Customers 4 ('W') and 5 ('X') have no initial orders, so neither
            # group exists in the backing table prior to repair: writing orders
            # for cust_id=4 or updating customer 5 will NOT collide with rows
            # deleted by DELETE FROM backing_table, proving both base tables
            # must be ShareLock'd explicitly.
            cur.executemany(
                "INSERT INTO customers VALUES (%s, %s)",
                [(1, "N"), (2, "S"), (3, "E"), (4, "W"), (5, "X")],
            )
            cur.executemany(
                "INSERT INTO orders VALUES (%s, %s, %s)",
                [(1, 1, 10), (2, 1, 20), (3, 2, 30), (4, 3, 40)],
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(
                "CREATE VIEW v_join AS "
                "SELECT c.region, COUNT(*) AS cnt, SUM(o.amt) AS total "
                "FROM orders o JOIN customers c ON o.cust_id = c.id GROUP BY c.region;"
            )

            w_orders = self._connect_in_schema(dsn, schema)
            w_customers = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([w_orders, w_customers, repairer, monitor])

            w_ord_sid = _session_id(Dialect.POSTGRES, w_orders)
            w_cust_sid = _session_id(Dialect.POSTGRES, w_customers)
            repair_sid = _session_id(Dialect.POSTGRES, repairer)

            entered = threading.Event()
            release = threading.Event()
            backing_tbl = plan.backing_table_name

            class _PausingCursor:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                    res = self._inner.execute(sql, *args, **kwargs)
                    if sql.strip().startswith(f"DELETE FROM {backing_tbl}"):
                        entered.set()
                        release.wait(timeout=20.0)
                    return res

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            class _PausingConn:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def cursor(self, *args: Any, **kwargs: Any) -> Any:
                    return _PausingCursor(self._inner.cursor(*args, **kwargs))

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            rmgr = MaterializedViewManager(_PausingConn(repairer), dialect=Dialect.POSTGRES)
            outcomes: Dict[str, Any] = {}

            def run_repair() -> None:
                try:
                    outcomes["repair"] = rmgr.repair_view("v_join")
                except BaseException as exc:
                    outcomes["repair_err"] = exc

            def run_ord_writer() -> None:
                try:
                    c = w_orders.cursor()
                    c.execute("INSERT INTO orders VALUES (10, 4, 100)")
                    w_orders.commit()
                    c.close()
                    outcomes["ord"] = True
                except BaseException as exc:
                    outcomes["ord_err"] = exc

            def run_cust_writer() -> None:
                try:
                    c = w_customers.cursor()
                    c.execute("UPDATE customers SET region = 'Y' WHERE id = 5")
                    c.execute("INSERT INTO customers VALUES (6, 'Z')")
                    w_customers.commit()
                    c.close()
                    outcomes["cust"] = True
                except BaseException as exc:
                    outcomes["cust_err"] = exc

            r_th = threading.Thread(target=run_repair, daemon=True)
            r_th.start()
            self.assertTrue(entered.wait(timeout=20.0), repr(outcomes))
            held_share_tables: List[str] = []
            try:
                mcur = monitor.cursor()
                mcur.execute(
                    "SELECT c.relname FROM pg_locks l "
                    "JOIN pg_class c ON c.oid = l.relation "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE l.pid = %s AND l.granted AND l.mode = 'ShareLock' AND n.nspname = %s "
                    "ORDER BY c.relname",
                    (repair_sid, schema),
                )
                held_share_tables = [r[0] for r in mcur.fetchall()]
                monitor.commit()
                mcur.close()

                o_th = threading.Thread(target=run_ord_writer, daemon=True)
                c_th = threading.Thread(target=run_cust_writer, daemon=True)
                o_th.start()
                c_th.start()

                deadline = time.monotonic() + 20.0
                both_waited = False
                while time.monotonic() < deadline:
                    if (
                        _is_lock_waiting(Dialect.POSTGRES, monitor, w_ord_sid, blocker_sid=repair_sid)
                        and _is_lock_waiting(Dialect.POSTGRES, monitor, w_cust_sid, blocker_sid=repair_sid)
                    ):
                        both_waited = True
                        break
                    if any(k in outcomes for k in ("ord", "cust", "ord_err", "cust_err")):
                        break
                    time.sleep(0.02)
            finally:
                release.set()

            for th in (r_th, o_th, c_th):
                th.join(timeout=20.0)
                self.assertFalse(th.is_alive())
            self.assertEqual(held_share_tables, ["customers", "orders"])
            self.assertNotIn("repair_err", outcomes, repr(outcomes.get("repair_err")))
            self.assertNotIn("ord_err", outcomes, repr(outcomes.get("ord_err")))
            self.assertNotIn("cust_err", outcomes, repr(outcomes.get("cust_err")))
            self.assertTrue(both_waited, f"expected both join-table writers to block on repair: {outcomes!r}")
            self._assert_view_matches_live(
                admin,
                "SELECT region, cnt, total FROM v_join ORDER BY region",
                "SELECT c.region, COUNT(*), SUM(o.amt) FROM orders o JOIN customers c ON o.cust_id = c.id GROUP BY c.region ORDER BY c.region",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_partitioned_table_repair_serializes_direct_partition_writer(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute(
                "CREATE TABLE p_parent (id BIGINT NOT NULL, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL, "
                "PRIMARY KEY (id)) PARTITION BY RANGE (id)"
            )
            cur.execute("CREATE TABLE p_part_0 PARTITION OF p_parent FOR VALUES FROM (0) TO (100)")
            cur.execute("CREATE TABLE p_part_1 PARTITION OF p_parent FOR VALUES FROM (100) TO (200)")
            cur.executemany(
                "INSERT INTO p_parent VALUES (%s, %s, %s)",
                [(1, "A", 10), (2, "B", 20), (101, "A", 30)],
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(
                "CREATE VIEW v_part AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM p_parent GROUP BY g;"
            )

            writer = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([writer, repairer, monitor])

            writer_sid = _session_id(Dialect.POSTGRES, writer)
            repair_sid = _session_id(Dialect.POSTGRES, repairer)
            entered = threading.Event()
            release = threading.Event()
            backing_tbl = plan.backing_table_name

            class _PausingCursor:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                    res = self._inner.execute(sql, *args, **kwargs)
                    if sql.strip().startswith(f"DELETE FROM {backing_tbl}"):
                        entered.set()
                        release.wait(timeout=20.0)
                    return res

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            class _PausingConn:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def cursor(self, *args: Any, **kwargs: Any) -> Any:
                    return _PausingCursor(self._inner.cursor(*args, **kwargs))

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            rmgr = MaterializedViewManager(_PausingConn(repairer), dialect=Dialect.POSTGRES)
            outcomes: Dict[str, Any] = {}

            def run_repair() -> None:
                try:
                    outcomes["repair"] = rmgr.repair_view("v_part")
                except BaseException as exc:
                    outcomes["repair_err"] = exc

            def run_direct_part_writer() -> None:
                try:
                    # Insert a new group so only the SHARE lock on p_parent
                    # (inherited by p_part_0) blocks the writer.
                    c = writer.cursor()
                    c.execute("INSERT INTO p_part_0 VALUES (50, 'Z_NEW', 999)")
                    writer.commit()
                    c.close()
                    outcomes["writer"] = True
                except BaseException as exc:
                    outcomes["writer_err"] = exc

            r_th = threading.Thread(target=run_repair, daemon=True)
            r_th.start()
            self.assertTrue(entered.wait(timeout=20.0), repr(outcomes))
            try:
                w_th = threading.Thread(target=run_direct_part_writer, daemon=True)
                w_th.start()

                deadline = time.monotonic() + 20.0
                waited = False
                while time.monotonic() < deadline:
                    if _is_lock_waiting(Dialect.POSTGRES, monitor, writer_sid, blocker_sid=repair_sid):
                        waited = True
                        break
                    if "writer" in outcomes or "writer_err" in outcomes:
                        break
                    time.sleep(0.02)
            finally:
                release.set()

            r_th.join(timeout=20.0)
            w_th.join(timeout=20.0)
            self.assertFalse(r_th.is_alive())
            self.assertFalse(w_th.is_alive())
            self.assertNotIn("repair_err", outcomes, repr(outcomes.get("repair_err")))
            self.assertNotIn("writer_err", outcomes, repr(outcomes.get("writer_err")))
            self.assertTrue(waited, f"direct child-partition write did not block on repair_view(): {outcomes!r}")
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v_part ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM p_parent GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_base_table_lock_timeout_retries_and_leaves_view_unchanged(self) -> None:
        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany("INSERT INTO t VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            mgr.register_view("CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;")

            holder = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            sessions.extend([holder, repairer])

            rc = repairer.cursor()
            rc.execute("SET lock_timeout = '200ms'")
            repairer.commit()
            rc.close()

            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)
            rmgr._BACKFILL_TRANSIENT_RETRIES = 2
            sleeps: List[float] = []
            rmgr._backfill_sleep = sleeps.append

            hc = holder.cursor()
            # Insert into a brand-new group so the holder only blocks
            # repair_view() via the base-table SHARE lock, not a backing row.
            hc.execute("INSERT INTO t VALUES (3, 'Z_NEW', 30)")
            try:
                with self.assertRaises(Exception) as ctx:
                    rmgr.repair_view("v")
                self.assertTrue(
                    rmgr._is_lock_timeout_error(ctx.exception)
                    or rmgr._is_transient_lock_error(ctx.exception),
                    repr(ctx.exception),
                )
                self.assertEqual(len(sleeps), 2)
            finally:
                holder.commit()
                hc.close()

            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
            self.assertEqual(rmgr.repair_view("v"), {"v": "repaired"})
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_autocommit_connection_repair_is_exact_and_preserves_autocommit(self) -> None:
        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany("INSERT INTO t VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            mgr.register_view("CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;")

            holder = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            sessions.extend([holder, repairer])
            repairer.autocommit = True

            rc = repairer.cursor()
            rc.execute("SET lock_timeout = '200ms'")
            rc.close()

            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)
            rmgr._BACKFILL_TRANSIENT_RETRIES = 1
            rmgr._backfill_sleep = lambda _: None

            # Failure path under autocommit=True must restore autocommit=True.
            hc = holder.cursor()
            hc.execute("INSERT INTO t VALUES (3, 'Z_NEW', 30)")
            try:
                with self.assertRaises(Exception) as ctx:
                    rmgr.repair_view("v")
                self.assertTrue(rmgr._is_lock_timeout_error(ctx.exception), repr(ctx.exception))
                self.assertTrue(repairer.autocommit)
            finally:
                holder.commit()
                hc.close()

            # Success path under autocommit=True must restore autocommit=True and match live.
            self.assertEqual(rmgr.repair_view("v"), {"v": "repaired"})
            self.assertTrue(repairer.autocommit)
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_multiview_repair_reacquires_catalog_lock_for_rebuilt_view_and_stays_exact(self) -> None:
        from test_avg_concurrent_maintenance import _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t1 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("CREATE TABLE t2 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany("INSERT INTO t1 VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20)])
            cur.executemany("INSERT INTO t2 VALUES (%s, %s, %s)", [(1, "X", 5), (2, "Y", 15)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            mgr.register_view("CREATE VIEW v1 AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t1 GROUP BY g;")
            p2 = mgr.register_view("CREATE VIEW v2 AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t2 GROUP BY g;")

            # Drop v2's triggers so repair_view(None) repairs v1 and rebuilds v2.
            cur = admin.cursor()
            for trig in p2.trigger_names:
                cur.execute(f"DROP TRIGGER IF EXISTS {trig} ON t2;")
            admin.commit()
            cur.close()

            repairer = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([repairer, monitor])
            repair_sid = _session_id(Dialect.POSTGRES, repairer)

            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)
            orig_register = rmgr.register_view
            catalog_locks_at_rebuild: List[str] = []

            def spy_register(*args: Any, **kwargs: Any) -> Any:
                mcur = monitor.cursor()
                mcur.execute(
                    "SELECT l.mode FROM pg_locks l "
                    "JOIN pg_class c ON c.oid = l.relation "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE l.pid = %s AND l.granted AND n.nspname = %s AND c.relname = %s",
                    (repair_sid, schema, rmgr.catalog_table),
                )
                catalog_locks_at_rebuild.extend(r[0] for r in mcur.fetchall())
                monitor.commit()
                mcur.close()
                return orig_register(*args, **kwargs)

            rmgr.register_view = spy_register  # type: ignore[method-assign]
            res = rmgr.repair_view()
            self.assertEqual(res, {"v1": "repaired", "v2": "rebuilt"})
            self.assertIn("ShareRowExclusiveLock", catalog_locks_at_rebuild)
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v1 ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t1 GROUP BY g ORDER BY g",
            )
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v2 ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t2 GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_union_all_view_repair_serializes_writers_on_both_branches(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE b1 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt INT NOT NULL)")
            cur.execute("CREATE TABLE b2 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt INT NOT NULL)")
            cur.executemany("INSERT INTO b1 VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20)])
            cur.executemany("INSERT INTO b2 VALUES (%s, %s, %s)", [(1, "A", 30), (2, "C", 40)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(
                "CREATE VIEW v_uall AS "
                "SELECT g, SUM(amt) AS total FROM ("
                "SELECT g, amt FROM b1 UNION ALL SELECT g, amt FROM b2"
                ") u GROUP BY g;"
            )

            w1 = self._connect_in_schema(dsn, schema)
            w2 = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([w1, w2, repairer, monitor])

            w1_sid = _session_id(Dialect.POSTGRES, w1)
            w2_sid = _session_id(Dialect.POSTGRES, w2)
            repair_sid = _session_id(Dialect.POSTGRES, repairer)
            entered = threading.Event()
            release = threading.Event()
            backing_tbl = plan.backing_table_name

            class _PausingCursor:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
                    res = self._inner.execute(sql, *args, **kwargs)
                    if sql.strip().startswith(f"DELETE FROM {backing_tbl}"):
                        entered.set()
                        release.wait(timeout=20.0)
                    return res

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            class _PausingConn:
                def __init__(self, inner: Any) -> None:
                    self._inner = inner

                def cursor(self, *args: Any, **kwargs: Any) -> Any:
                    return _PausingCursor(self._inner.cursor(*args, **kwargs))

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._inner, name)

            rmgr = MaterializedViewManager(_PausingConn(repairer), dialect=Dialect.POSTGRES)
            outcomes: Dict[str, Any] = {}

            def run_repair() -> None:
                try:
                    outcomes["repair"] = rmgr.repair_view("v_uall")
                except BaseException as exc:
                    outcomes["repair_err"] = exc

            def run_w1() -> None:
                try:
                    c = w1.cursor()
                    c.execute("INSERT INTO b1 VALUES (10, 'Z1', 100)")
                    w1.commit()
                    c.close()
                    outcomes["w1"] = True
                except BaseException as exc:
                    outcomes["w1_err"] = exc

            def run_w2() -> None:
                try:
                    c = w2.cursor()
                    c.execute("INSERT INTO b2 VALUES (10, 'Z2', 200)")
                    w2.commit()
                    c.close()
                    outcomes["w2"] = True
                except BaseException as exc:
                    outcomes["w2_err"] = exc

            r_th = threading.Thread(target=run_repair, daemon=True)
            r_th.start()
            self.assertTrue(entered.wait(timeout=20.0), repr(outcomes))
            try:
                t1 = threading.Thread(target=run_w1, daemon=True)
                t2 = threading.Thread(target=run_w2, daemon=True)
                t1.start()
                t2.start()

                deadline = time.monotonic() + 20.0
                both_waited = False
                while time.monotonic() < deadline:
                    if (
                        _is_lock_waiting(Dialect.POSTGRES, monitor, w1_sid, blocker_sid=repair_sid)
                        and _is_lock_waiting(Dialect.POSTGRES, monitor, w2_sid, blocker_sid=repair_sid)
                    ):
                        both_waited = True
                        break
                    if any(k in outcomes for k in ("w1", "w2", "w1_err", "w2_err")):
                        break
                    time.sleep(0.02)
            finally:
                release.set()

            for th in (r_th, t1, t2):
                th.join(timeout=20.0)
                self.assertFalse(th.is_alive())
            self.assertNotIn("repair_err", outcomes, repr(outcomes.get("repair_err")))
            self.assertNotIn("w1_err", outcomes, repr(outcomes.get("w1_err")))
            self.assertNotIn("w2_err", outcomes, repr(outcomes.get("w2_err")))
            self.assertTrue(both_waited, f"expected both UNION ALL branch writers to block: {outcomes!r}")
            self._assert_view_matches_live(
                admin,
                "SELECT g, total FROM v_uall ORDER BY g",
                "SELECT g, SUM(amt) FROM (SELECT g, amt FROM b1 UNION ALL SELECT g, amt FROM b2) u GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def _assert_serialization_failure(self, exc: BaseException) -> None:
        sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
        if sqlstate is None and getattr(exc, "orig", None) is not None:
            sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
        self.assertEqual(
            sqlstate,
            "40001",
            f"expected PostgreSQL SQLSTATE 40001 serialization_failure, got sqlstate={sqlstate!r} exc={exc!r}",
        )

    def test_pre_repair_repeatable_read_and_serializable_writers_fail_closed_with_40001(self) -> None:
        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t (id BIGINT PRIMARY KEY, d_id BIGINT NOT NULL, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("CREATE TABLE d (id BIGINT PRIMARY KEY, label VARCHAR(16) NOT NULL)")
            cur.executemany("INSERT INTO d VALUES (%s, %s)", [(1, "D1"), (2, "D2")])
            cur.executemany(
                "INSERT INTO t VALUES (%s, %s, %s, %s)",
                [(1, 1, "A", 10), (2, 1, "A", 20), (3, 2, "B", 30)],
            )
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            view_defs = [
                ("v_pop", "CREATE VIEW v_pop AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;", True),
                ("v_t2", "CREATE VIEW v_t2 AS SELECT g, STRING_AGG(id::text, ',' ORDER BY id) AS ids FROM t GROUP BY g;", True),
                ("v_t3_pop", "CREATE VIEW v_t3_pop AS SELECT id, g, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS rn FROM t;", True),
                ("v_join", "CREATE VIEW v_join AS SELECT d.label, COUNT(*) AS c, SUM(t.amt) AS s FROM t JOIN d ON d.id = t.d_id GROUP BY d.label;", True),
                ("v's repair", 'CREATE VIEW "v\'s repair" AS SELECT g, SUM(amt) AS s FROM t GROUP BY g;', True),
            ]
            for _, vsql, pop in view_defs:
                mgr.register_view(vsql, populate=pop)

            # Verify manager signature/drift parity with repair_fence_catalog_table enabled.
            drift = mgr.check_schema_drift()
            self.assertEqual(
                drift,
                {vn: "ok" for vn, _, _ in view_defs},
            )
            diff_rep = mgr.diff_views([vsql for _, vsql, _ in view_defs])
            self.assertTrue(
                all(d.signature_hash == d.current_signature_hash for d in diff_rep.values()),
                repr(diff_rep),
            )
            sync_rep = mgr.sync_views([vsql for _, vsql, _ in view_defs])
            self.assertEqual(sync_rep, {vn: "unchanged" for vn, _, _ in view_defs})

            writer = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            sessions.extend([writer, repairer])
            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)

            next_id = 100
            for iso_idx, iso in enumerate(("REPEATABLE READ", "SERIALIZABLE")):
                # Register fresh unpopulated (populate=False) Tier 1 and Tier 3 views
                # per isolation level so both REPEATABLE READ and SERIALIZABLE exercise
                # the empty-store + active-watermark pre-repair snapshot path.
                v_unpop = f"v_unpop_{iso_idx}"
                v_t3_unpop = f"v_t3_unpop_{iso_idx}"
                mgr.register_view(
                    f"CREATE VIEW {v_unpop} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t GROUP BY g;",
                    populate=False,
                )
                mgr.register_view(
                    f"CREATE VIEW {v_t3_unpop} AS SELECT id, g, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS rn FROM t;",
                    populate=False,
                )

                # Repair ONE view at a time so only that single view's catalog row is
                # updated: every earlier trigger's FOR SHARE succeeds, proving that
                # specifically the repaired view's own fence fires and raises 40001.
                per_view_dmls = [
                    (v_unpop, [("insert", f"INSERT INTO t VALUES ({next_id}, 1, 'A', 5)"), ("update", "UPDATE t SET amt = amt + 1 WHERE id = 1"), ("delete", "DELETE FROM t WHERE id = 2"), ("truncate", "TRUNCATE t")]),
                    ("v_pop", [("insert", f"INSERT INTO t VALUES ({next_id}, 1, 'A', 5)"), ("update", "UPDATE t SET amt = amt + 1 WHERE id = 1"), ("delete", "DELETE FROM t WHERE id = 2"), ("truncate", "TRUNCATE t")]),
                    ("v_t2", [("update", "UPDATE t SET amt = amt + 1 WHERE id = 1"), ("truncate", "TRUNCATE t")]),
                    (v_t3_unpop, [("delete", "DELETE FROM t WHERE id = 2"), ("truncate", "TRUNCATE t")]),
                    ("v_t3_pop", [("update", "UPDATE t SET amt = amt + 1 WHERE id = 1")]),
                    ("v's repair", [("insert", f"INSERT INTO t VALUES ({next_id}, 1, 'A', 5)"), ("truncate", "TRUNCATE t")]),
                    (
                        "v_join",
                        [
                            ("single_source_insert", f"INSERT INTO t VALUES ({next_id}, 1, 'A', 5)"),
                            (
                                "stage3_wcte",
                                f"WITH nd AS (INSERT INTO d VALUES ({next_id}, 'D_NEW') RETURNING id) "
                                f"INSERT INTO t SELECT {next_id}, id, 'Z', 99 FROM nd",
                            ),
                            ("stage3_truncate_d_first", "TRUNCATE d, t"),
                        ],
                    ),
                ]
                for target_view, dml_cases in per_view_dmls:
                    with self.subTest(isolation=iso, target_view=target_view):
                        wcur = writer.cursor()
                        wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                        wcur.execute("SELECT COUNT(*) FROM t")
                        wcur.fetchone()

                        self.assertEqual(rmgr.repair_view(target_view), {target_view: "repaired"})

                        for label, dml in dml_cases:
                            with self.subTest(isolation=iso, target_view=target_view, dml=label):
                                wcur.execute("SAVEPOINT sp_fence")
                                with self.assertRaises(Exception) as ctx:
                                    wcur.execute(dml)
                                self._assert_serialization_failure(ctx.exception)
                                wcur.execute("ROLLBACK TO SAVEPOINT sp_fence")
                        writer.rollback()
                        wcur.close()

                # A fresh transaction started AFTER repair_view() commits cleanly and stays exact.
                wcur = writer.cursor()
                wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                wcur.execute(
                    "WITH nd AS (INSERT INTO d VALUES (%s, %s) RETURNING id) "
                    "INSERT INTO t SELECT %s, id, 'C', 40 FROM nd",
                    (next_id, f"D_{next_id}", next_id),
                )
                writer.commit()
                wcur.close()
                next_id += 1

                for vn in (v_unpop, "v_pop"):
                    self._assert_view_matches_live(
                        admin,
                        f"SELECT g, c, s FROM {vn} ORDER BY g",
                        "SELECT g, COUNT(*), SUM(amt) FROM t GROUP BY g ORDER BY g",
                    )
                self._assert_view_matches_live(
                    admin,
                    "SELECT g, ids FROM v_t2 ORDER BY g",
                    "SELECT g, STRING_AGG(id::text, ',' ORDER BY id) FROM t GROUP BY g ORDER BY g",
                )
                for vn in (v_t3_unpop, "v_t3_pop"):
                    self._assert_view_matches_live(
                        admin,
                        f"SELECT id, g, rn FROM {vn} ORDER BY id",
                        "SELECT id, g, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id",
                    )
                self._assert_view_matches_live(
                    admin,
                    "SELECT label, c, s FROM v_join ORDER BY label",
                    "SELECT d.label, COUNT(*), SUM(t.amt) FROM t JOIN d ON d.id = t.d_id GROUP BY d.label ORDER BY d.label",
                )
                self._assert_view_matches_live(
                    admin,
                    'SELECT g, s FROM "v\'s repair" ORDER BY g',
                    "SELECT g, SUM(amt) FROM t GROUP BY g ORDER BY g",
                )

            # Pre-register_view() snapshot (NOT FOUND branch in the fence) must fail closed with 40001.
            cur = admin.cursor()
            cur.execute("CREATE TABLE t_fresh (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("INSERT INTO t_fresh VALUES (1, 'A', 10)")
            admin.commit()
            cur.close()
            for idx, iso in enumerate(("REPEATABLE READ", "SERIALIZABLE")):
                vname = f"v_pre_reg_{idx}"
                wcur = writer.cursor()
                wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                wcur.execute("SELECT COUNT(*) FROM t_fresh")
                wcur.fetchone()

                mgr.register_view(
                    f"CREATE VIEW {vname} AS SELECT g, SUM(amt) AS s FROM t_fresh GROUP BY g;",
                    populate=True,
                )
                with self.assertRaises(Exception) as ctx:
                    wcur.execute("INSERT INTO t_fresh VALUES (%s, 'A', 5)", (10 + idx,))
                self._assert_serialization_failure(ctx.exception)
                writer.rollback()
                wcur.close()
                self._assert_view_matches_live(
                    admin,
                    f"SELECT g, s FROM {vname} ORDER BY g",
                    "SELECT g, SUM(amt) FROM t_fresh GROUP BY g ORDER BY g",
                )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_pre_rebuild_0row_tier2_join_wcte_partition_and_backfill_fail_closed_with_40001(self) -> None:
        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE d2 (id BIGINT PRIMARY KEY, label VARCHAR(16) NOT NULL)")
            cur.execute("CREATE TABLE t2 (id BIGINT PRIMARY KEY, d_id BIGINT NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("INSERT INTO d2 VALUES (1, 'D1')")
            cur.execute("INSERT INTO t2 VALUES (1, 1, 10)")
            # Partitioned source table for direct-partition + parent write fencing.
            cur.execute("CREATE TABLE tp (id BIGINT NOT NULL, p INT NOT NULL, amt BIGINT NOT NULL, PRIMARY KEY (id, p)) PARTITION BY LIST (p)")
            cur.execute("CREATE TABLE tp_p1 PARTITION OF tp FOR VALUES IN (1)")
            cur.execute("CREATE TABLE tp_p2 PARTITION OF tp FOR VALUES IN (2)")
            cur.executemany("INSERT INTO tp VALUES (%s, %s, %s)", [(1, 1, 10), (2, 2, 20)])
            # Source table for backfill_incrementally() (chunked + snapshot_isolation).
            cur.execute("CREATE TABLE tb (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany("INSERT INTO tb VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20), (3, "A", 30)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            writer = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            sessions.extend([writer, repairer])
            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)

            # 1. Regression for unfenced overlap_close_sql: 0-row multi-source writable CTE
            #    on a Tier 2 multi-source join under a pre-register / pre-rebuild snapshot
            #    must fail closed with 40001 instead of inserting stale phantom rows.
            #    Freeze the writer snapshot via an unrelated table (tb) so DROP/CREATE
            #    TRIGGER DDL (ACCESS EXCLUSIVE on t2/d2) does not block on writer's
            #    ACCESS SHARE lock.
            wcur = writer.cursor()
            wcur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            wcur.execute("SELECT COUNT(*) FROM tb")
            wcur.fetchone()

            cur = admin.cursor()
            cur.execute("DELETE FROM t2")
            admin.commit()
            cur.close()
            mgr.register_view(
                "CREATE VIEW v_t2_join AS SELECT d2.label, STRING_AGG(t2.id::text, ',' ORDER BY t2.id) AS ids "
                "FROM d2 LEFT JOIN t2 ON t2.d_id = d2.id GROUP BY d2.label;",
                populate=True,
            )
            with self.assertRaises(Exception) as ctx:
                wcur.execute(
                    "WITH u AS (UPDATE d2 SET label = label WHERE id = -999 RETURNING id) "
                    "UPDATE t2 SET amt = amt WHERE id = -999"
                )
            self._assert_serialization_failure(ctx.exception)
            writer.rollback()
            wcur.close()
            self._assert_view_matches_live(
                admin,
                "SELECT label, ids FROM v_t2_join ORDER BY label",
                "SELECT d2.label, STRING_AGG(t2.id::text, ',' ORDER BY t2.id) FROM d2 LEFT JOIN t2 ON t2.d_id = d2.id GROUP BY d2.label ORDER BY d2.label",
            )

            # 2. repair_view() "rebuilt" path (dropped trigger triggers full rebuild) under
            #    pre-existing REPEATABLE READ and SERIALIZABLE snapshots fails closed with 40001.
            for iso in ("REPEATABLE READ", "SERIALIZABLE"):
                with self.subTest(rebuilt_iso=iso):
                    plan_j = mgr.get_catalog()["v_t2_join"]
                    trig_to_drop = next(tn for tn in plan_j["trigger_names"] if tn.endswith("_ai"))
                    # Determine which table owns trig_to_drop and drop it to force rebuild.
                    cur = admin.cursor()
                    cur.execute(
                        "SELECT c.relname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = current_schema() AND t.tgname = %s",
                        (trig_to_drop,),
                    )
                    owner_tbl = cur.fetchone()[0]
                    cur.execute(f'DROP TRIGGER "{trig_to_drop}" ON "{owner_tbl}"')
                    admin.commit()
                    cur.close()

                    wcur = writer.cursor()
                    wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                    wcur.execute("SELECT COUNT(*) FROM tb")
                    wcur.fetchone()

                    self.assertEqual(rmgr.repair_view("v_t2_join"), {"v_t2_join": "rebuilt"})

                    with self.assertRaises(Exception) as ctx:
                        wcur.execute(
                            "WITH u AS (UPDATE d2 SET label = label WHERE id = -999 RETURNING id) "
                            "UPDATE t2 SET amt = amt WHERE id = -999"
                        )
                    self._assert_serialization_failure(ctx.exception)
                    writer.rollback()
                    wcur.close()

            # 3. Partitioned source table: both parent write and direct child-partition write
            #    fail closed with 40001 under pre-repair REPEATABLE READ and SERIALIZABLE.
            mgr.register_view("CREATE VIEW v_part AS SELECT p, SUM(amt) AS s FROM tp GROUP BY p;", populate=True)
            for iso in ("REPEATABLE READ", "SERIALIZABLE"):
                with self.subTest(partition_iso=iso):
                    wcur = writer.cursor()
                    wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                    wcur.execute("SELECT COUNT(*) FROM tp")
                    wcur.fetchone()

                    self.assertEqual(rmgr.repair_view("v_part"), {"v_part": "repaired"})

                    for label, dml in (
                        ("parent_insert", "INSERT INTO tp VALUES (100, 1, 5)"),
                        ("child_partition_insert", "INSERT INTO tp_p2 VALUES (101, 2, 7)"),
                        ("child_partition_truncate", "TRUNCATE tp_p1"),
                    ):
                        with self.subTest(partition_iso=iso, dml=label):
                            wcur.execute("SAVEPOINT sp_part")
                            with self.assertRaises(Exception) as ctx:
                                wcur.execute(dml)
                            self._assert_serialization_failure(ctx.exception)
                            wcur.execute("ROLLBACK TO SAVEPOINT sp_part")
                    writer.rollback()
                    wcur.close()
            self._assert_view_matches_live(
                admin,
                "SELECT p, s FROM v_part ORDER BY p",
                "SELECT p, SUM(amt) FROM tp GROUP BY p ORDER BY p",
            )

            # 4. backfill_incrementally() (both chunked and snapshot_isolation=True) fences
            #    pre-backfill REPEATABLE READ and SERIALIZABLE snapshots with 40001.
            for idx, (snap_iso, iso) in enumerate((
                (False, "REPEATABLE READ"),
                (True, "REPEATABLE READ"),
                (True, "SERIALIZABLE"),
            )):
                vname = f"v_bf_{idx}"
                with self.subTest(backfill_snapshot_iso=snap_iso, writer_iso=iso):
                    plan_bf = mgr.register_view(
                        f"CREATE VIEW {vname} AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM tb GROUP BY g;",
                        populate=False,
                    )
                    wcur = writer.cursor()
                    wcur.execute(f"BEGIN ISOLATION LEVEL {iso}")
                    wcur.execute("SELECT COUNT(*) FROM tb")
                    wcur.fetchone()

                    rmgr.backfill_incrementally(plan_bf, batch_size=2, snapshot_isolation=snap_iso)

                    with self.assertRaises(Exception) as ctx:
                        wcur.execute("UPDATE tb SET amt = amt + 1 WHERE id = 1")
                    self._assert_serialization_failure(ctx.exception)
                    writer.rollback()
                    wcur.close()
                    self._assert_view_matches_live(
                        admin,
                        f"SELECT g, c, s FROM {vname} ORDER BY g",
                        "SELECT g, COUNT(*), SUM(amt) FROM tb GROUP BY g ORDER BY g",
                    )
        finally:
            self._drop_schema(dsn, schema, sessions)

    def test_repeatable_read_and_serializable_concurrent_writers_and_inflight_repair_do_not_deadlock(self) -> None:
        from test_avg_concurrent_maintenance import _is_lock_waiting, _lock_diagnostics, _session_id

        dsn = self._dsn()
        schema, admin = self._make_schema(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute("CREATE TABLE t1 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("CREATE TABLE t2 (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany("INSERT INTO t1 VALUES (%s, %s, %s)", [(1, "A", 10), (2, "B", 20)])
            cur.executemany("INSERT INTO t2 VALUES (%s, %s, %s)", [(1, "X", 100), (2, "Y", 200)])
            admin.commit()
            cur.close()

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            mgr.register_view("CREATE VIEW v1 AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t1 GROUP BY g;")
            mgr.register_view("CREATE VIEW v2 AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t2 GROUP BY g;")

            w1 = self._connect_in_schema(dsn, schema)
            w2 = self._connect_in_schema(dsn, schema)
            repairer = self._connect_in_schema(dsn, schema)
            monitor = self._connect_in_schema(dsn, schema)
            sessions.extend([w1, w2, repairer, monitor])
            rmgr = MaterializedViewManager(repairer, dialect=Dialect.POSTGRES)

            # 1. Two concurrent REPEATABLE READ / SERIALIZABLE writers on disjoint groups
            #    both commit cleanly (FOR SHARE is mutually compatible, no SSI false conflict).
            base_id = 10
            for iso in ("REPEATABLE READ", "SERIALIZABLE"):
                with self.subTest(concurrent_writers_iso=iso):
                    barrier = threading.Barrier(2)
                    errs: Dict[str, BaseException] = {}

                    def _run_writer(conn: Any, row_id: int, grp: str, key: str) -> None:
                        try:
                            c = conn.cursor()
                            c.execute(f"BEGIN ISOLATION LEVEL {iso}")
                            c.execute("INSERT INTO t1 VALUES (%s, %s, %s)", (row_id, grp, 5))
                            barrier.wait(timeout=10.0)
                            conn.commit()
                            c.close()
                        except BaseException as exc:
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            errs[key] = exc

                    th1 = threading.Thread(target=_run_writer, args=(w1, base_id, "A", "w1"), daemon=True)
                    th2 = threading.Thread(target=_run_writer, args=(w2, base_id + 1, "B", "w2"), daemon=True)
                    th1.start()
                    th2.start()
                    th1.join(timeout=20.0)
                    th2.join(timeout=20.0)
                    self.assertEqual(errs, {}, repr(errs))
                    base_id += 2

            # 2. In-flight REPEATABLE READ writer holding ROW EXCLUSIVE on t1 + FOR SHARE on
            #    v1's catalog row blocks repair_view("v1") on t1, then executes a SECOND write
            #    and commits without deadlocking.
            w1_sid = _session_id(Dialect.POSTGRES, w1)
            rep_sid = _session_id(Dialect.POSTGRES, repairer)
            w1_cur = w1.cursor()
            w1_cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            w1_cur.execute("INSERT INTO t1 VALUES (50, 'C', 500)")

            rep_outcome: Dict[str, Any] = {}

            def _run_rep() -> None:
                try:
                    rep_outcome["res"] = rmgr.repair_view("v1")
                except BaseException as exc:
                    rep_outcome["err"] = exc

            r_th = threading.Thread(target=_run_rep, daemon=True)
            r_th.start()
            deadline = time.monotonic() + 20.0
            waited = False
            while time.monotonic() < deadline:
                if _is_lock_waiting(Dialect.POSTGRES, monitor, rep_sid, blocker_sid=w1_sid):
                    waited = True
                    break
                if rep_outcome:
                    break
                time.sleep(0.02)
            self.assertTrue(
                waited,
                f"repair_view did not block on in-flight REPEATABLE READ writer: "
                f"outcome={rep_outcome!r} diag={_lock_diagnostics(Dialect.POSTGRES, monitor, rep_sid)!r}",
            )

            # Second DML inside the same in-flight REPEATABLE READ transaction must not deadlock.
            w1_cur.execute("UPDATE t1 SET amt = amt + 50 WHERE id = 50")
            w1.commit()
            w1_cur.close()
            r_th.join(timeout=20.0)
            self.assertFalse(r_th.is_alive())
            self.assertNotIn("err", rep_outcome, repr(rep_outcome.get("err")))
            self.assertEqual(rep_outcome.get("res"), {"v1": "repaired"})

            # 3. Repairing unrelated view v1 does not abort a pre-existing REPEATABLE READ writer on v2.
            w2_cur = w2.cursor()
            w2_cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            w2_cur.execute("SELECT COUNT(*) FROM t2")
            w2_cur.fetchone()
            self.assertEqual(rmgr.repair_view("v1"), {"v1": "repaired"})
            w2_cur.execute("INSERT INTO t2 VALUES (50, 'X', 50)")
            w2.commit()
            w2_cur.close()

            # 4. READ COMMITTED writers take zero locks on the catalog table.
            rc_cur = w1.cursor()
            rc_cur.execute("BEGIN ISOLATION LEVEL READ COMMITTED")
            rc_cur.execute("INSERT INTO t1 VALUES (60, 'A', 60)")
            rc_cur.execute(
                "SELECT COUNT(*) FROM pg_locks l "
                "JOIN pg_class c ON c.oid = l.relation "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE l.pid = pg_backend_pid() AND n.nspname = current_schema() AND c.relname = %s",
                (mgr.catalog_table,),
            )
            cat_locks = rc_cur.fetchone()[0]
            w1.commit()
            rc_cur.close()
            self.assertEqual(cat_locks, 0, f"expected 0 catalog locks under READ COMMITTED, got {cat_locks}")

            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v1 ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t1 GROUP BY g ORDER BY g",
            )
            self._assert_view_matches_live(
                admin,
                "SELECT g, c, s FROM v2 ORDER BY g",
                "SELECT g, COUNT(*), SUM(amt) FROM t2 GROUP BY g ORDER BY g",
            )
        finally:
            self._drop_schema(dsn, schema, sessions)


class FailClosedBackfillConcurrencyTests(unittest.TestCase):
    """Independent-session tests for fail-closed incremental backfill pre-claim gating
    and snapshot isolation under concurrent writers (SQLite + native PostgreSQL)."""

    def test_sqlite_concurrent_writer_in_pre_claim_window_does_not_double_count(self) -> None:
        fd, db_path = tempfile.mkstemp(prefix="matview-preclaim-", suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        _register_sqlite_compat_udfs(conn)
        conn.execute("CREATE TABLE pc_items (id INTEGER PRIMARY KEY, g TEXT NOT NULL, amt INTEGER NOT NULL);")
        conn.executemany(
            "INSERT INTO pc_items VALUES (?, ?, ?);",
            [(i, f"G{i % 3}", i * 10) for i in range(1, 11)],
        )
        conn.commit()
        manager = MaterializedViewManager(conn)
        orig_claim = manager._claim_non_snapshot_pk_range
        writer_ran = []
        writer_errors: list[BaseException] = []

        def claim_after_independent_writer(cur, view_name, base_t, pk_col):
            # Runs strictly AFTER register_view committed (NULL, NULL, 1) and
            # strictly BEFORE the watermark range is claimed.
            def run_writer() -> None:
                second = sqlite3.connect(db_path, timeout=15)
                try:
                    second.execute("PRAGMA busy_timeout=15000;")
                    _register_sqlite_compat_udfs(second)
                    second.execute("INSERT INTO pc_items VALUES (11, 'G1', 110);")
                    second.execute("INSERT INTO pc_items VALUES (12, 'G2', 120);")
                    second.execute("UPDATE pc_items SET amt = 999 WHERE id = 2;")
                    second.execute("DELETE FROM pc_items WHERE id = 5;")
                    second.commit()
                    writer_ran.append(True)
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    second.close()

            th = threading.Thread(target=run_writer, daemon=True)
            th.start()
            th.join(timeout=15)
            self.assertFalse(th.is_alive())
            if writer_errors:
                raise writer_errors[0]
            return orig_claim(cur, view_name, base_t, pk_col)

        manager._claim_non_snapshot_pk_range = claim_after_independent_writer  # type: ignore[method-assign]
        try:
            plan = manager.register_view(
                "CREATE VIEW v_pc AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM pc_items GROUP BY g;",
                populate=True,
                incremental_backfill=True,
                batch_size=3,
            )
            self.assertEqual(writer_ran, [True])
            self.assertEqual(manager.check_schema_drift()[plan.view_name], "ok")
            self.assertEqual(
                conn.execute("SELECT g, c, s FROM v_pc ORDER BY g;").fetchall(),
                conn.execute("SELECT g, COUNT(*), SUM(amt) FROM pc_items GROUP BY g ORDER BY g;").fetchall(),
            )
        finally:
            conn.close()
            os.unlink(db_path)

    @staticmethod
    def _pg_dsn() -> str:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            raise unittest.SkipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        return dsn

    @staticmethod
    def _pg_connect(dsn: str, schema: str):
        import psycopg2
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute(f'SET search_path TO "{schema}"')
        conn.commit()
        cur.close()
        return conn

    def test_pg_register_incremental_backfill_under_continuous_concurrent_writers(self) -> None:
        dsn = self._pg_dsn()
        import psycopg2
        schema = f"mv_fc_{uuid.uuid4().hex[:10]}"
        admin = psycopg2.connect(dsn)
        sessions = [admin]
        try:
            cur = admin.cursor()
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute("CREATE TABLE t_live (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.executemany(
                "INSERT INTO t_live VALUES (%s, %s, %s)",
                [(i, f"G{i % 4}", i * 10) for i in range(1, 31)],
            )
            admin.commit()
            cur.close()

            stop = threading.Event()
            writers_ready = threading.Barrier(4)
            in_gap_done = threading.Event()
            errors: list[BaseException] = []
            commits = [0, 0, 0]

            def writer(wid: int) -> None:
                c = self._pg_connect(dsn, schema)
                sessions.append(c)
                seq = 0
                try:
                    writers_ready.wait(timeout=20.0)
                    while (not stop.is_set() or commits[wid] < 5) and seq < 35:
                        seq += 1
                        row_id = 40 + wid * 40 + seq
                        for _attempt in range(TRANSIENT_RETRIES):
                            try:
                                wcur = c.cursor()
                                wcur.execute(
                                    "INSERT INTO t_live VALUES (%s, %s, %s)",
                                    (row_id, f"G{(row_id + wid) % 4}", seq * 3),
                                )
                                if seq % 3 == 0:
                                    wcur.execute(
                                        "UPDATE t_live SET amt = amt + 5 WHERE id = %s",
                                        ((seq % 20) + 1,),
                                    )
                                if seq % 5 == 0 and seq > 5:
                                    wcur.execute(
                                        "DELETE FROM t_live WHERE id = %s",
                                        (40 + wid * 40 + seq - 2,),
                                    )
                                c.commit()
                                wcur.close()
                                commits[wid] += 1
                                break
                            except Exception as exc:
                                c.rollback()
                                if not _is_transient_lock_error(Dialect.POSTGRES, exc):
                                    raise
                                time.sleep(0.01)
                        time.sleep(0.005)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(3)]
            for th in threads:
                th.start()
            writers_ready.wait(timeout=20.0)

            mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
            orig_claim = mgr._claim_non_snapshot_pk_range

            def claim_with_gap_writer(ccur, view_name, base_t, pk_col):
                # Force an explicit commit strictly inside the pre-claim gap.
                gap_conn = self._pg_connect(dsn, schema)
                try:
                    gcur = gap_conn.cursor()
                    gcur.execute("INSERT INTO t_live VALUES (200, 'G0', 777)")
                    gcur.execute("UPDATE t_live SET amt = 888 WHERE id = 1")
                    gap_conn.commit()
                    gcur.close()
                    in_gap_done.set()
                finally:
                    gap_conn.close()
                return orig_claim(ccur, view_name, base_t, pk_col)

            mgr._claim_non_snapshot_pk_range = claim_with_gap_writer  # type: ignore[method-assign]
            plan = mgr.register_view(
                "CREATE VIEW v_live AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t_live GROUP BY g;",
                populate=True,
                incremental_backfill=True,
                batch_size=10,
            )
            stop.set()
            for th in threads:
                th.join(timeout=30.0)
                self.assertFalse(th.is_alive())
            self.assertEqual(errors, [], repr(errors))
            self.assertTrue(in_gap_done.is_set())
            self.assertTrue(all(n >= 5 for n in commits), f"expected concurrent writes: {commits}")
            self.assertEqual(mgr.check_schema_drift()[plan.view_name], "ok")

            vcur = admin.cursor()
            vcur.execute("SELECT g, c, s FROM v_live ORDER BY g")
            actual = [tuple(r) for r in vcur.fetchall()]
            vcur.execute("SELECT g, COUNT(*), SUM(amt) FROM t_live GROUP BY g ORDER BY g")
            expected = [tuple(r) for r in vcur.fetchall()]
            admin.commit()
            vcur.close()
            self.assertEqual(actual, expected)
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            cleanup = psycopg2.connect(dsn)
            try:
                cleanup.autocommit = True
                cleanup.cursor().execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            finally:
                cleanup.close()

    def test_pg_snapshot_isolation_backfill_exact_under_concurrent_writers(self) -> None:
        dsn = self._pg_dsn()
        import psycopg2
        for label, view_sql, live_sql in (
            (
                "pk_range_fold_snapshot",
                "CREATE VIEW v_snap_pk AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t_snap GROUP BY g;",
                "SELECT g, COUNT(*), SUM(amt) FROM t_snap GROUP BY g ORDER BY g",
            ),
            (
                "group_key_batch_snapshot",
                "CREATE VIEW v_snap_gk AS SELECT g, COUNT(DISTINCT amt) AS cd FROM t_snap GROUP BY g;",
                "SELECT g, COUNT(DISTINCT amt) FROM t_snap GROUP BY g ORDER BY g",
            ),
        ):
            with self.subTest(case=label):
                schema = f"mv_sn_{uuid.uuid4().hex[:10]}"
                admin = psycopg2.connect(dsn)
                sessions = [admin]
                try:
                    cur = admin.cursor()
                    cur.execute(f'CREATE SCHEMA "{schema}"')
                    cur.execute(f'SET search_path TO "{schema}"')
                    cur.execute("CREATE TABLE t_snap (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
                    cur.executemany(
                        "INSERT INTO t_snap VALUES (%s, %s, %s)",
                        [(i, f"G{i % 4}", (i % 7) * 10) for i in range(1, 25)],
                    )
                    admin.commit()
                    cur.close()

                    mgr = MaterializedViewManager(admin, dialect=Dialect.POSTGRES)
                    plan = mgr.register_view(view_sql, populate=False)

                    stop = threading.Event()
                    ready = threading.Barrier(4)
                    errors: list[BaseException] = []
                    commits = [0, 0, 0]

                    def writer(wid: int) -> None:
                        c = self._pg_connect(dsn, schema)
                        sessions.append(c)
                        seq = 0
                        try:
                            ready.wait(timeout=20.0)
                            while (not stop.is_set() or commits[wid] < 4) and seq < 30:
                                seq += 1
                                row_id = 30 + wid * 40 + seq
                                for _ in range(TRANSIENT_RETRIES):
                                    try:
                                        wcur = c.cursor()
                                        wcur.execute(
                                            "INSERT INTO t_snap VALUES (%s, %s, %s)",
                                            (row_id, f"G{row_id % 4}", (seq % 11) * 10),
                                        )
                                        if seq % 2 == 0:
                                            wcur.execute(
                                                "UPDATE t_snap SET amt = amt + 10 WHERE id = %s",
                                                ((seq % 20) + 1,),
                                            )
                                        c.commit()
                                        wcur.close()
                                        commits[wid] += 1
                                        break
                                    except Exception as exc:
                                        c.rollback()
                                        if not _is_transient_lock_error(Dialect.POSTGRES, exc):
                                            raise
                                        time.sleep(0.01)
                                time.sleep(0.005)
                        except BaseException as exc:
                            errors.append(exc)

                    threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(3)]
                    for th in threads:
                        th.start()
                    ready.wait(timeout=20.0)

                    mgr.backfill_incrementally(plan, batch_size=8, snapshot_isolation=True)
                    stop.set()
                    for th in threads:
                        th.join(timeout=30.0)
                        self.assertFalse(th.is_alive())
                    self.assertEqual(errors, [], repr(errors))
                    self.assertTrue(all(n >= 4 for n in commits), repr(commits))

                    vcur = admin.cursor()
                    vcur.execute(f"SELECT * FROM {plan.view_name} ORDER BY g")
                    actual = [tuple(r) for r in vcur.fetchall()]
                    vcur.execute(live_sql)
                    expected = [tuple(r) for r in vcur.fetchall()]
                    admin.commit()
                    vcur.close()
                    self.assertEqual(actual, expected)
                finally:
                    for s in sessions:
                        try:
                            s.rollback()
                            s.close()
                        except Exception:
                            pass
                    cleanup = psycopg2.connect(dsn)
                    try:
                        cleanup.autocommit = True
                        cleanup.cursor().execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                    finally:
                        cleanup.close()

    def test_pg_same_definition_register_view_leaves_connection_idle(self) -> None:
        dsn = self._pg_dsn()
        import psycopg2
        import psycopg2.extensions as ext
        schema = f"mv_idle_{uuid.uuid4().hex[:10]}"
        conn = psycopg2.connect(dsn)
        try:
            cur = conn.cursor()
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute("CREATE TABLE t_idle (id BIGINT PRIMARY KEY, g VARCHAR(8) NOT NULL, amt BIGINT NOT NULL)")
            cur.execute("INSERT INTO t_idle VALUES (1, 'A', 10)")
            conn.commit()
            cur.close()

            mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
            view_sql = "CREATE VIEW v_idle AS SELECT g, COUNT(*) AS c, SUM(amt) AS s FROM t_idle GROUP BY g;"
            mgr.register_view(view_sql, populate=True)
            self.assertEqual(conn.get_transaction_status(), ext.TRANSACTION_STATUS_IDLE)

            # Same-definition early return must not leave a read transaction open.
            mgr.register_view(view_sql, populate=True)
            self.assertEqual(conn.get_transaction_status(), ext.TRANSACTION_STATUS_IDLE)

            # Unpopulated view: same-definition populate=True refuses and leaves connection idle.
            view_sql2 = "CREATE VIEW v_idle2 AS SELECT g, COUNT(*) AS c FROM t_idle GROUP BY g;"
            mgr.register_view(view_sql2, populate=False)
            self.assertEqual(conn.get_transaction_status(), ext.TRANSACTION_STATUS_IDLE)
            with self.assertRaises(BackfillInterruptedError):
                mgr.register_view(view_sql2, populate=True)
            self.assertEqual(conn.get_transaction_status(), ext.TRANSACTION_STATUS_IDLE)
        finally:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            cleanup = psycopg2.connect(dsn)
            try:
                cleanup.autocommit = True
                cleanup.cursor().execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            finally:
                cleanup.close()


class TestMySqlRepairViewLockingOffline(unittest.TestCase):
    """Offline structural checks for MySQL/MariaDB repair_view() isolation-level,
    clustered-index shared-lock ordering, and READ COMMITTED recompute locks."""

    def test_mysql_base_table_lock_stmts_sorts_dedups_and_handles_pk_and_keyless_tables(self) -> None:
        from matview.logical_ir import ColumnDef, TableSchema
        from matview.manager import _mysql_base_table_lock_stmts
        from matview_sync import generate_materialized_view, parse_table_schema

        schema = parse_table_schema(
            """
            CREATE TABLE z_tbl (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);
            CREATE TABLE a_tbl (id INT PRIMARY KEY, z_id INT NOT NULL, w INT NOT NULL);
            CREATE TABLE no_pk_tbl (g INT, x INT);
            """,
            dialect=Dialect.MYSQL,
        )
        # Simulate MySQL information_schema.columns COLUMN_KEY='PRI' on a UNIQUE NOT NULL
        # table that has no PRIMARY index (pk_columns is empty while c.primary_key is True).
        schema["uq_only_tbl"] = TableSchema(
            name="uq_only_tbl",
            columns={
                "id": ColumnDef(name="id", data_type="INT", not_null=True, primary_key=True),
                "g": ColumnDef(name="g", data_type="INT", not_null=False, primary_key=False),
            },
            pk_columns=[],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan_join = generate_materialized_view(
                schema,
                "CREATE VIEW v_mj AS "
                "WITH `MyCTE` AS (SELECT z.g, a.w FROM z_tbl z JOIN a_tbl a ON a.z_id = z.id) "
                "SELECT g, SUM(w) AS s FROM mycte GROUP BY g;",
                dialect=Dialect.MYSQL,
                ignore_limitations=True,
            )
            plan_keyless = generate_materialized_view(
                schema,
                "CREATE VIEW v_mk AS SELECT g, COUNT(*) AS cnt FROM no_pk_tbl GROUP BY g;",
                dialect=Dialect.MYSQL,
                ignore_limitations=True,
            )
            plan_uq_only = generate_materialized_view(
                schema,
                "CREATE VIEW v_muq AS SELECT g, COUNT(*) AS cnt FROM uq_only_tbl GROUP BY g;",
                dialect=Dialect.MYSQL,
                ignore_limitations=True,
            )

        stmts_join = _mysql_base_table_lock_stmts(plan_join, schemas=schema)
        self.assertEqual(
            stmts_join,
            [
                "SELECT COUNT(*) FROM a_tbl FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;",
                "SELECT COUNT(*) FROM z_tbl FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;",
            ],
        )

        stmts_keyless = _mysql_base_table_lock_stmts(plan_keyless, schemas=schema)
        self.assertEqual(
            stmts_keyless,
            [
                "SELECT SUM((CASE WHEN g IS NULL THEN 1 ELSE 0 END) + (CASE WHEN x IS NULL THEN 1 ELSE 0 END)) "
                "FROM no_pk_tbl LOCK IN SHARE MODE;"
            ],
        )

        # UNIQUE NOT NULL table without PRIMARY index must NOT emit FORCE INDEX (PRIMARY)
        # (which raises MySQL Error 1176).
        stmts_uq = _mysql_base_table_lock_stmts(plan_uq_only, schemas=schema)
        self.assertEqual(
            stmts_uq,
            [
                "SELECT SUM((CASE WHEN id IS NULL THEN 1 ELSE 0 END) + (CASE WHEN g IS NULL THEN 1 ELSE 0 END)) "
                "FROM uq_only_tbl LOCK IN SHARE MODE;"
            ],
        )

    def test_mysql_lock_planner_locks_a_table_that_is_only_a_cte_under_looser_server_rules(self) -> None:
        """MySQL with lower_case_table_names=0 compares CTE names case-sensitively and never resolves a
        forward sibling CTE; MariaDB does neither (verified natively: ``WITH Src AS (...) SELECT .. FROM src``
        reads the physical table on MySQL 8.4 and the CTE on MariaDB 11.8).  A name that is a CTE only under
        the looser rules but also names a real table must still be locked; a name that certainly is a CTE
        (exact match, earlier sibling) or has no physical table must not be."""
        from types import SimpleNamespace

        from matview.manager import _mysql_base_table_lock_stmts
        from matview_sync import parse_table_schema

        schema = parse_table_schema(
            "CREATE TABLE src (id INT PRIMARY KEY); CREATE TABLE later (id INT PRIMARY KEY);"
            "CREATE TABLE base (id INT PRIMARY KEY);",
            dialect=Dialect.MYSQL,
        )

        def lock_stmts(select_sql: str) -> List[str]:
            plan = SimpleNamespace(
                sub_plans=None, view_spec=SimpleNamespace(raw_select_sql=select_sql, referenced_tables=())
            )
            return _mysql_base_table_lock_stmts(plan, schemas=schema)

        def lock(tbl: str) -> str:
            return f"SELECT COUNT(*) FROM {tbl} FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;"

        # Case-only match with a physical table `src`: locked (the CTE `Src` does not shadow it on MySQL).
        self.assertEqual(
            lock_stmts("WITH Src AS (SELECT id FROM base) SELECT id FROM src"),
            [lock("base"), lock("src")],
        )
        # Exact-case reference to the CTE: never locked as a table.
        self.assertEqual(
            lock_stmts("WITH Src AS (SELECT id FROM base) SELECT id FROM Src"),
            [lock("base")],
        )
        # Forward sibling under WITH RECURSIVE: MySQL reads the physical table `later`.
        self.assertEqual(
            lock_stmts(
                "WITH RECURSIVE first_cte AS (SELECT id FROM later), later AS (SELECT id FROM base) "
                "SELECT id FROM first_cte"
            ),
            [lock("base"), lock("later")],
        )
        # A recursive self-reference and a later sibling's reference to an earlier one are CTEs.
        self.assertEqual(
            lock_stmts(
                "WITH RECURSIVE r AS (SELECT id FROM base UNION ALL SELECT id FROM r), s AS (SELECT id FROM r) "
                "SELECT id FROM s"
            ),
            [lock("base")],
        )

    def test_mysql_minmax_and_bitwise_recompute_subqueries_emit_lock_in_share_mode(self) -> None:
        from matview_sync import generate_materialized_view, parse_table_schema

        schema_my = parse_table_schema(
            "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, amt BIGINT, KEY idx_g (g));",
            dialect=Dialect.MYSQL,
        )
        schema_pg = parse_table_schema(
            "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, amt BIGINT);",
            dialect=Dialect.POSTGRES,
        )
        view_sql = (
            "CREATE VIEW v_rec AS "
            "SELECT g, MIN(amt) AS mn, MAX(amt) AS mx, BIT_AND(amt) AS ba, BIT_OR(amt) AS bo "
            "FROM t GROUP BY g;"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan_my = generate_materialized_view(
                schema_my, view_sql, dialect=Dialect.MYSQL, ignore_limitations=True
            )
            plan_pg = generate_materialized_view(
                schema_pg, view_sql, dialect=Dialect.POSTGRES, ignore_limitations=True
            )

        del_triggers_my = "\n".join(plan_my.trigger_ddls)
        for agg in ("MIN(amt)", "MAX(amt)", "BIT_AND(amt)", "BIT_OR(amt)"):
            self.assertRegex(
                del_triggers_my,
                rf"\(SELECT [^\n;]*{re.escape(agg)}[^\n;]*FROM t WHERE \(t\.g\) = \(OLD\.g\) LOCK IN SHARE MODE\)",
            )
        self.assertNotIn("LOCK IN SHARE MODE", "\n".join(plan_pg.trigger_ddls))

    def test_mysql_repair_view_contents_sets_repeatable_read_locks_base_tables_and_retries_on_deadlock(self) -> None:
        from matview_sync import generate_materialized_view, parse_table_schema

        schema = parse_table_schema(
            """
            CREATE TABLE z_tbl (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);
            CREATE TABLE a_tbl (id INT PRIMARY KEY, z_id INT NOT NULL, w INT NOT NULL);
            """,
            dialect=Dialect.MYSQL,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan = generate_materialized_view(
                schema,
                "CREATE VIEW v_mj AS SELECT z.g, SUM(a.w) AS s FROM z_tbl z JOIN a_tbl a ON a.z_id = z.id GROUP BY z.g;",
                dialect=Dialect.MYSQL,
                ignore_limitations=True,
            )

        executed: List[str] = []
        fail_count = {"n": 0}

        class _MySqlTransientError(Exception):
            pass

        class _FakeCursor:
            def execute(self, sql: str, params: Any = None) -> None:
                executed.append(sql.strip())
                if sql.strip().startswith(f"DELETE FROM {plan.backing_table_name}"):
                    if fail_count["n"] == 0:
                        fail_count["n"] += 1
                        raise _MySqlTransientError(1213, "Deadlock found when trying to get lock")
                    if fail_count["n"] == 1:
                        fail_count["n"] += 1
                        raise _MySqlTransientError(1020, "Record has changed since last read in table")

            def fetchall(self) -> List[Tuple[Any, ...]]:
                return [(0,)]

            def close(self) -> None:
                pass

        class _FakeMySqlConn:
            def cursor(self) -> _FakeCursor:
                return _FakeCursor()

            def commit(self) -> None:
                executed.append("COMMIT")

            def rollback(self) -> None:
                executed.append("ROLLBACK")

        mgr = MaterializedViewManager(
            _FakeMySqlConn(),
            dialect=Dialect.MYSQL,
            engine_version="mariadb 11.8.0",
            skip_ensure_catalog=True,
        )
        sleeps: List[float] = []
        mgr._backfill_sleep = sleeps.append
        mgr._repair_view_contents(plan, plan.view_name, schemas=schema)

        self.assertEqual(len(sleeps), 2)
        set_tx_indices = [
            i for i, s in enumerate(executed) if s == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;"
        ]
        self.assertEqual(len(set_tx_indices), 3, repr(executed))
        attempts = [
            executed[set_tx_indices[0] : set_tx_indices[1]],
            executed[set_tx_indices[1] : set_tx_indices[2]],
            executed[set_tx_indices[2] :],
        ]
        lock_a = "SELECT COUNT(*) FROM a_tbl FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;"
        lock_z = "SELECT COUNT(*) FROM z_tbl FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;"
        for att in attempts:
            self.assertIn("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;", att)
            self.assertIn("START TRANSACTION;", att)
            self.assertIn(lock_a, att, repr(att))
            self.assertIn(lock_z, att, repr(att))
            idx_set = att.index("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
            idx_start = att.index("START TRANSACTION;")
            idx_cat = next(i for i, s in enumerate(att) if s.startswith("UPDATE __matview_catalog SET backfill_cursor_pk"))
            idx_lock_a = att.index(lock_a)
            idx_lock_z = att.index(lock_z)
            idx_del = next(i for i, s in enumerate(att) if s.startswith(f"DELETE FROM {plan.backing_table_name}"))
            self.assertLess(idx_set, idx_start)
            self.assertLess(idx_start, idx_cat)
            self.assertLess(idx_cat, idx_lock_a)
            self.assertLess(idx_lock_a, idx_lock_z)
            self.assertLess(idx_lock_z, idx_del)
        self.assertEqual(attempts[2][-1], "COMMIT")


class TestMySqlIsolationLevelConcurrency(unittest.TestCase):
    """Native MySQL/MariaDB concurrency verification under non-default InnoDB
    isolation levels (READ COMMITTED and SERIALIZABLE): algebraic aggregates,
    recomputed aggregates (MIN/MAX/BIT_AND/BIT_OR with group index), and
    repair_view() racing concurrent writers across algebraic, recomputed,
    keyless Tier 3, UNIQUE NOT NULL (no PK), and join views."""

    @staticmethod
    def _mysql_dsn() -> str:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            raise unittest.SkipTest("MYSQL_DSN not set")
        return dsn

    @staticmethod
    def _connect(dsn: str) -> Any:
        return IndependentSessionBackfillTests._connect_native(Dialect.MYSQL, dsn)

    @staticmethod
    def _is_retryable_mysql_lock_err(exc: BaseException) -> bool:
        args = getattr(exc, "args", ())
        code = args[0] if args and isinstance(args[0], int) else None
        return code in {1020, 1205, 1213}

    @staticmethod
    def _iso_token(iso: str) -> str:
        return iso.upper().replace(" ", "-")

    @staticmethod
    def _read_session_isolation(cur: Any) -> str:
        """Read the MySQL/MariaDB session isolation level compatibly across
        MySQL 8.4/9.7 (`transaction_isolation`) and MariaDB 10.11 (`tx_isolation`)."""
        cur.execute(
            "SHOW SESSION VARIABLES WHERE Variable_name IN ('transaction_isolation', 'tx_isolation')"
        )
        rows = cur.fetchall()
        assert rows, "could not read MySQL/MariaDB session isolation level"
        return str(rows[0][1]).upper()

    def test_native_mysql_read_committed_minmax_bitwise_no_stale_snapshot(self) -> None:
        """Deterministic 2-session READ COMMITTED interleaving: Session B inserts
        a new row into group g0 and holds the X-lock on mv(g0); Session A deletes
        the old minimum in g0 and blocks on mv(g0) (verified via engine lock-wait
        evidence). Once Session B commits, Session A's recompute subqueries MUST
        use LOCK IN SHARE MODE (current read) so they see Session B's committed
        row instead of reading a stale pre-lock statement snapshot."""
        from test_avg_concurrent_maintenance import _session_id, _wait_until_lock_waiting

        dsn = self._mysql_dsn()
        tag = f"mrcm_{uuid.uuid4().hex[:8]}"
        tbl = f"t_{tag}"
        v_rec = f"v_rec_{tag}"
        admin = self._connect(dsn)
        sess_b = self._connect(dsn)
        sess_a = self._connect(dsn)
        monitor = self._connect(dsn)
        mgr: Optional[MaterializedViewManager] = None
        try:
            cur = admin.cursor()
            cur.execute(
                f"CREATE TABLE {tbl} ("
                "  id BIGINT PRIMARY KEY, "
                "  g VARCHAR(16) NOT NULL, "
                "  amt BIGINT NOT NULL, "
                "  KEY idx_g (g)"
                ") ENGINE=InnoDB"
            )
            cur.executemany(
                f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                [(1, "g0", 100), (10, "g0", 50)],
            )
            admin.commit()

            mgr = MaterializedViewManager(admin, dialect=Dialect.MYSQL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                mgr.register_view(
                    f"CREATE VIEW {v_rec} AS "
                    f"SELECT g, MIN(amt) AS mn, MAX(amt) AS mx, BIT_AND(amt) AS ba, BIT_OR(amt) AS bo "
                    f"FROM {tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
            admin.commit()

            sid_b = _session_id(Dialect.MYSQL, sess_b)
            sid_a = _session_id(Dialect.MYSQL, sess_a)

            cur_b = sess_b.cursor()
            cur_b.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
            self.assertEqual(self._read_session_isolation(cur_b), "READ-COMMITTED")
            cur_b.execute("START TRANSACTION")
            cur_b.execute(f"INSERT INTO {tbl} (id, g, amt) VALUES (11, 'g0', 75)")

            a_err: List[BaseException] = []
            a_done = threading.Event()

            def _run_a() -> None:
                cur_a = None
                try:
                    cur_a = sess_a.cursor()
                    cur_a.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    cur_a.execute("START TRANSACTION")
                    cur_a.execute(f"DELETE FROM {tbl} WHERE id = 10")
                    sess_a.commit()
                except BaseException as exc:
                    a_err.append(exc)
                    try:
                        sess_a.rollback()
                    except Exception:
                        pass
                finally:
                    if cur_a is not None:
                        try:
                            cur_a.close()
                        except Exception:
                            pass
                    a_done.set()

            th_a = threading.Thread(target=_run_a, daemon=True)
            th_a.start()
            _wait_until_lock_waiting(Dialect.MYSQL, monitor, sid_a, a_done, blocker_sid=sid_b)

            sess_b.commit()
            cur_b.close()
            th_a.join(timeout=30.0)
            self.assertFalse(th_a.is_alive(), "Session A hung after Session B committed")
            self.assertEqual(a_err, [], f"Session A failed: {a_err!r}")

            cur.execute(f"SELECT g, mn, mx, ba, bo FROM {v_rec} ORDER BY g")
            got = [tuple(r) for r in cur.fetchall()]
            cur.execute(
                f"SELECT g, MIN(amt), MAX(amt), BIT_AND(amt), BIT_OR(amt) FROM {tbl} GROUP BY g ORDER BY g"
            )
            exp = [tuple(r) for r in cur.fetchall()]
            self.assertEqual(got, exp, f"READ COMMITTED stale snapshot in MIN/MAX/BIT_AND/BIT_OR: {got!r} != {exp!r}")
            self.assertEqual(got, [("g0", 75, 100, 100 & 75, 100 | 75)])
            admin.commit()
        finally:
            for c_obj in (sess_b, sess_a, monitor):
                try:
                    c_obj.rollback()
                    c_obj.close()
                except Exception:
                    pass
            try:
                admin.rollback()
                if mgr is not None:
                    try:
                        mgr.drop_view(v_rec)
                    except Exception:
                        admin.rollback()
                c = admin.cursor()
                c.execute(f"DROP TABLE IF EXISTS {tbl}")
                admin.commit()
                c.close()
            except Exception:
                pass
            admin.close()

    def test_native_mysql_read_committed_and_serializable_algebraic_and_recomputed_writers(self) -> None:
        dsn = self._mysql_dsn()
        tag = f"miso_{uuid.uuid4().hex[:8]}"
        tbl = f"t_{tag}"
        v_alg = f"v_alg_{tag}"
        v_rec = f"v_rec_{tag}"
        admin = self._connect(dsn)
        mgr: Optional[MaterializedViewManager] = None
        try:
            cur = admin.cursor()
            cur.execute("SELECT VERSION()")
            ver_row = cur.fetchone()
            self.assertIsNotNone(ver_row)
            assert ver_row is not None
            self.assertTrue(len(str(ver_row[0])) > 0)

            cur.execute(
                f"CREATE TABLE {tbl} ("
                "  id BIGINT PRIMARY KEY, "
                "  g VARCHAR(16) NOT NULL, "
                "  amt BIGINT NULL, "
                "  KEY idx_g (g)"
                ") ENGINE=InnoDB"
            )
            cur.executemany(
                f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                [(1, "g0", 100), (2, "g1", 200), (3, "g2", 300)],
            )
            admin.commit()

            mgr = MaterializedViewManager(admin, dialect=Dialect.MYSQL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                mgr.register_view(
                    f"CREATE VIEW {v_alg} AS "
                    f"SELECT g, COUNT(*) AS cnt, COUNT(amt) AS vcnt, SUM(amt) AS tot, AVG(amt) AS avg_amt "
                    f"FROM {tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
                mgr.register_view(
                    f"CREATE VIEW {v_rec} AS "
                    f"SELECT g, MIN(amt) AS mn, MAX(amt) AS mx, BIT_AND(amt) AS ba, BIT_OR(amt) AS bo "
                    f"FROM {tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
            admin.commit()

            worker_count = 4
            tx_per_worker = 14
            for iso_idx, iso in enumerate(("READ COMMITTED", "SERIALIZABLE")):
                with self.subTest(isolation_level=iso):
                    start = threading.Barrier(worker_count)
                    errors: List[BaseException] = []
                    completed: List[int] = []
                    lock = threading.Lock()

                    def _worker(wid: int) -> None:
                        wconn = None
                        wcur = None
                        try:
                            wconn = self._connect(dsn)
                            wcur = wconn.cursor()
                            wcur.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                            got_iso = self._read_session_isolation(wcur)
                            self.assertEqual(got_iso, self._iso_token(iso))
                            wconn.commit()
                            start.wait(timeout=30.0)
                            for seq in range(tx_per_worker):
                                row_id = (iso_idx + 1) * 100000 + wid * 1000 + seq + 10
                                src_g = f"g{wid % 3}"
                                dst_g = f"g{(wid + seq) % 3}"
                                val = None if seq % 7 == 0 else ((wid + 1) * 50 + seq * 3)
                                for attempt in range(20):
                                    try:
                                        wcur.execute("START TRANSACTION")
                                        wcur.execute(
                                            f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                                            (row_id, src_g, val),
                                        )
                                        # Update both group and value (exercising MIN/MAX/BIT_AND/BIT_OR recomputes).
                                        new_val = None if seq % 11 == 0 else ((val or 17) ^ 5)
                                        wcur.execute(
                                            f"UPDATE {tbl} SET g = %s, amt = %s WHERE id = %s",
                                            (dst_g, new_val, row_id),
                                        )
                                        if seq % 4 == 0:
                                            wcur.execute(f"DELETE FROM {tbl} WHERE id = %s", (row_id,))
                                        if seq % 9 == 0:
                                            wconn.rollback()
                                        else:
                                            wconn.commit()
                                        break
                                    except BaseException as exc:
                                        try:
                                            wconn.rollback()
                                        except Exception:
                                            pass
                                        if not self._is_retryable_mysql_lock_err(exc) or attempt == 19:
                                            raise
                                        time.sleep(min(0.15, 0.005 * (attempt + 1)) + 0.002 * ((wid + seq) % 5))
                            with lock:
                                completed.append(wid)
                        except BaseException as exc:
                            start.abort()
                            with lock:
                                errors.append(exc)
                        finally:
                            if wcur is not None:
                                try:
                                    wcur.close()
                                except Exception:
                                    pass
                            if wconn is not None:
                                try:
                                    wconn.rollback()
                                    wconn.close()
                                except Exception:
                                    pass

                    threads = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(worker_count)]
                    for th in threads:
                        th.start()
                    for th in threads:
                        th.join(timeout=90.0)
                    self.assertFalse([th.name for th in threads if th.is_alive()], f"MySQL {iso} workers hung")
                    self.assertEqual(errors, [], f"MySQL {iso} worker errors: {errors!r}")
                    self.assertEqual(sorted(completed), list(range(worker_count)))

                    cur.execute(f"SELECT g, cnt, vcnt, tot, avg_amt FROM {v_alg} ORDER BY g")
                    got_alg = [tuple(r) for r in cur.fetchall()]
                    cur.execute(
                        f"SELECT g, COUNT(*), COUNT(amt), SUM(amt), AVG(amt) FROM {tbl} GROUP BY g ORDER BY g"
                    )
                    exp_alg = [tuple(r) for r in cur.fetchall()]
                    self.assertEqual(got_alg, exp_alg, f"MySQL {iso} algebraic mismatch")

                    cur.execute(f"SELECT g, mn, mx, ba, bo FROM {v_rec} ORDER BY g")
                    got_rec = [tuple(r) for r in cur.fetchall()]
                    cur.execute(
                        f"SELECT g, MIN(amt), MAX(amt), BIT_AND(amt), BIT_OR(amt) FROM {tbl} GROUP BY g ORDER BY g"
                    )
                    exp_rec = [tuple(r) for r in cur.fetchall()]
                    self.assertEqual(got_rec, exp_rec, f"MySQL {iso} recomputed aggregate mismatch")
                    admin.commit()
        finally:
            try:
                admin.rollback()
                if mgr is not None:
                    for vn in (v_alg, v_rec):
                        try:
                            mgr.drop_view(vn)
                        except Exception:
                            admin.rollback()
                c = admin.cursor()
                c.execute(f"DROP TABLE IF EXISTS {tbl}")
                admin.commit()
                c.close()
            except Exception:
                pass
            admin.close()

    def test_native_mysql_repair_view_under_read_committed_and_serializable_writers(self) -> None:
        from test_avg_concurrent_maintenance import _session_id, _wait_until_lock_waiting

        dsn = self._mysql_dsn()
        tag = f"mrep_{uuid.uuid4().hex[:8]}"
        tbl = f"t_{tag}"
        dim = f"d_{tag}"
        uq_tbl = f"uq_{tag}"
        v_alg = f"v_alg_{tag}"
        v_rec = f"v_rec_{tag}"
        v_t3 = f"v_t3_{tag}"
        v_join = f"v_jn_{tag}"
        v_uq = f"v_uq_{tag}"
        admin = self._connect(dsn)
        monitor = self._connect(dsn)
        mgr: Optional[MaterializedViewManager] = None
        try:
            cur = admin.cursor()
            cur.execute("SELECT VERSION()")
            ver_row = cur.fetchone()
            self.assertIsNotNone(ver_row)
            assert ver_row is not None
            self.assertTrue(len(str(ver_row[0])) > 0)

            cur.execute(
                f"CREATE TABLE {tbl} ("
                "  id BIGINT PRIMARY KEY, "
                "  g VARCHAR(16) NOT NULL, "
                "  amt BIGINT NOT NULL, "
                "  KEY idx_g (g)"
                ") ENGINE=InnoDB"
            )
            cur.execute(
                f"CREATE TABLE {dim} ("
                "  g VARCHAR(16) PRIMARY KEY, "
                "  label VARCHAR(16) NOT NULL"
                ") ENGINE=InnoDB"
            )
            cur.execute(
                f"CREATE TABLE {uq_tbl} ("
                "  id BIGINT NOT NULL, "
                "  g VARCHAR(16) NOT NULL, "
                "  amt BIGINT NOT NULL, "
                "  UNIQUE KEY uq_id (id)"
                ") ENGINE=InnoDB"
            )
            cur.executemany(
                f"INSERT INTO {dim} (g, label) VALUES (%s, %s)",
                [("g0", "L0"), ("g1", "L1"), ("g2", "L2"), ("new_rc", "LRC"), ("new_ser", "LSER")],
            )
            cur.executemany(
                f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                [(1, "g0", 10), (2, "g1", 20), (3, "g2", 30)],
            )
            cur.executemany(
                f"INSERT INTO {uq_tbl} (id, g, amt) VALUES (%s, %s, %s)",
                [(1, "g0", 11), (2, "g1", 22)],
            )
            admin.commit()

            mgr = MaterializedViewManager(admin, dialect=Dialect.MYSQL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                mgr.register_view(
                    f"CREATE VIEW {v_alg} AS SELECT g, COUNT(*) AS cnt, SUM(amt) AS tot FROM {tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
                mgr.register_view(
                    f"CREATE VIEW {v_rec} AS SELECT g, MIN(amt) AS mn, MAX(amt) AS mx FROM {tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
                # Keyless Tier 3 view (triggers do not read __matview_catalog):
                # exercises base-table LOCK IN SHARE MODE wait cleanly under both
                # READ COMMITTED and SERIALIZABLE, and prevents duplicate rows on repair.
                mgr.register_view(
                    f"CREATE VIEW {v_t3} AS SELECT g, COUNT(*) AS cnt, SUM(amt) AS tot FROM {tbl} GROUP BY g ORDER BY g LIMIT 100;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
                mgr.register_view(
                    f"CREATE VIEW {v_join} AS "
                    f"SELECT d.label, COUNT(*) AS cnt, SUM(t.amt) AS tot "
                    f"FROM {tbl} t JOIN {dim} d ON d.g = t.g GROUP BY d.label;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
                # UNIQUE NOT NULL table without PRIMARY KEY: repair_view() must not
                # emit FORCE INDEX (PRIMARY) (Error 1176).
                mgr.register_view(
                    f"CREATE VIEW {v_uq} AS SELECT g, COUNT(*) AS cnt, SUM(amt) AS tot FROM {uq_tbl} GROUP BY g;",
                    accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                )
            admin.commit()

            def _assert_all_exact(step: str) -> None:
                checks = [
                    (v_alg, f"SELECT g, cnt, tot FROM {v_alg} ORDER BY g",
                     f"SELECT g, COUNT(*), SUM(amt) FROM {tbl} GROUP BY g ORDER BY g"),
                    (v_rec, f"SELECT g, mn, mx FROM {v_rec} ORDER BY g",
                     f"SELECT g, MIN(amt), MAX(amt) FROM {tbl} GROUP BY g ORDER BY g"),
                    (v_t3, f"SELECT g, cnt, tot FROM {v_t3} ORDER BY g",
                     f"SELECT g, COUNT(*), SUM(amt) FROM {tbl} GROUP BY g ORDER BY g LIMIT 100"),
                    (v_join, f"SELECT label, cnt, tot FROM {v_join} ORDER BY label",
                     f"SELECT d.label, COUNT(*), SUM(t.amt) FROM {tbl} t JOIN {dim} d ON d.g = t.g GROUP BY d.label ORDER BY d.label"),
                    (v_uq, f"SELECT g, cnt, tot FROM {v_uq} ORDER BY g",
                     f"SELECT g, COUNT(*), SUM(amt) FROM {uq_tbl} GROUP BY g ORDER BY g"),
                ]
                for vn, mv_q, live_q in checks:
                    cur.execute(mv_q)
                    got = [tuple(r) for r in cur.fetchall()]
                    cur.execute(live_q)
                    exp = [tuple(r) for r in cur.fetchall()]
                    self.assertEqual(got, exp, f"MySQL {step} mismatch on {vn}: {got!r} != {exp!r}")
                admin.commit()

            for iso_idx, iso in enumerate(("READ COMMITTED", "SERIALIZABLE")):
                with self.subTest(isolation_level=iso):
                    # 1. Deterministic in-flight writer holding X-lock on a brand-new group
                    #    and updating a non-indexed column: repair_view(v_t3) must enter a
                    #    positive InnoDB lock wait attributed to w_hold on the base-table
                    #    LOCK IN SHARE MODE scan (not __matview_catalog), must preserve the
                    #    repairer session's own isolation level, and must reflect all writes.
                    w_hold = self._connect(dsn)
                    rep_conn = self._connect(dsn)
                    try:
                        wh_sid = _session_id(Dialect.MYSQL, w_hold)
                        rep_sid = _session_id(Dialect.MYSQL, rep_conn)

                        rcur = rep_conn.cursor()
                        rcur.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                        rep_conn.commit()

                        wh_cur = w_hold.cursor()
                        wh_cur.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                        wh_cur.execute("START TRANSACTION")
                        new_grp = "new_rc" if iso_idx == 0 else "new_ser"
                        hold_id = 500 + iso_idx
                        wh_cur.execute(
                            f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                            (hold_id, new_grp, 500),
                        )

                        # Also verifies _ensure_catalog() does not stall on MDL_EXCLUSIVE
                        # while w_hold holds MDL_SHARED_READ on __matview_catalog.
                        rmgr = MaterializedViewManager(rep_conn, dialect=Dialect.MYSQL)
                        rep_out: Dict[str, Any] = {}
                        rep_done = threading.Event()

                        def _do_repair() -> None:
                            try:
                                rep_out["res"] = rmgr.repair_view(v_t3)
                            except BaseException as exc:
                                rep_out["err"] = exc
                            finally:
                                rep_done.set()

                        rth = threading.Thread(target=_do_repair, daemon=True)
                        rth.start()
                        _wait_until_lock_waiting(
                            Dialect.MYSQL, monitor, rep_sid, rep_done, blocker_sid=wh_sid
                        )
                        # The wait must be on the base-table shared-lock scan that
                        # repair_view() issues before touching the backing table (taking the
                        # base locks first matches the writers' base-then-backing order), not
                        # on some later statement of the repair.
                        mcur = monitor.cursor()
                        mcur.execute("SELECT info FROM information_schema.processlist WHERE id = %s", (rep_sid,))
                        waiting_sql = str((mcur.fetchone() or ("",))[0] or "")
                        monitor.commit()
                        self.assertIn("LOCK IN SHARE MODE", waiting_sql.upper(), waiting_sql)
                        self.assertIn(tbl, waiting_sql, waiting_sql)

                        # Execute a second write on the held row (non-indexed column) and commit;
                        # must not deadlock with repair_view() waiting on base_t.
                        wh_cur.execute(f"UPDATE {tbl} SET amt = amt + 25 WHERE id = %s", (hold_id,))
                        w_hold.commit()
                        wh_cur.close()

                        rth.join(timeout=30.0)
                        self.assertFalse(rth.is_alive(), f"repair_view() hung under {iso}")
                        self.assertNotIn("err", rep_out, repr(rep_out.get("err")))
                        self.assertEqual(rep_out.get("res"), {v_t3: "repaired"})

                        # Also verify repair_view() on UNIQUE NOT NULL (no PK) table succeeds cleanly.
                        self.assertEqual(rmgr.repair_view(v_uq), {v_uq: "repaired"})

                        # Session isolation level must be preserved (SET TRANSACTION is tx-local).
                        self.assertEqual(self._read_session_isolation(rcur), self._iso_token(iso))
                        rep_conn.commit()
                        rcur.close()
                    finally:
                        for c_obj in (w_hold, rep_conn):
                            try:
                                c_obj.rollback()
                                c_obj.close()
                            except Exception:
                                pass
                    _assert_all_exact(f"{iso} in-flight writer")

                    if iso_idx == 0:
                        # repair_view() rolls back its own read snapshots on MySQL/MariaDB, so it
                        # must refuse (before any SQL) a connection holding caller-owned writes
                        # instead of discarding them.
                        own = self._connect(dsn)
                        try:
                            own_mgr = MaterializedViewManager(own, dialect=Dialect.MYSQL)
                            ocur = own.cursor()
                            ocur.execute(f"INSERT INTO {tbl} (id, g, amt) VALUES (900001, 'g0', 1)")
                            with self.assertRaises(TransactionOwnershipError):
                                own_mgr.repair_view(v_t3)
                            ocur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE id = 900001")
                            self.assertEqual(ocur.fetchone()[0], 1, "repair_view() discarded caller-owned writes")
                            own.rollback()
                        finally:
                            own.close()
                        _assert_all_exact(f"{iso} after refused repair")

                    # 2. Concurrent multi-writer stress racing repeated repair_view() across
                    #    algebraic, recomputed, keyless Tier 3, and join views.
                    stop_writers = threading.Event()
                    writer_ready = threading.Barrier(3)
                    errors: List[BaseException] = []
                    err_lock = threading.Lock()
                    repairs_done = {"count": 0}
                    # Handshake: the repairs must overlap live writer traffic, so the repair
                    # runner starts only after both writers committed a transaction.
                    writer_committed = [threading.Event(), threading.Event()]
                    commits = [0, 0]

                    def _stress_writer(wid: int) -> None:
                        wconn = None
                        wcur = None
                        try:
                            wconn = self._connect(dsn)
                            wcur = wconn.cursor()
                            wcur.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                            wconn.commit()
                            writer_ready.wait(timeout=20.0)
                            seq = 0
                            while not stop_writers.is_set() and seq < 20:
                                rid = (iso_idx + 1) * 50000 + wid * 500 + seq + 1
                                grp = f"g{(wid + seq) % 3}"
                                for attempt in range(20):
                                    try:
                                        wcur.execute("START TRANSACTION")
                                        wcur.execute(
                                            f"INSERT INTO {tbl} (id, g, amt) VALUES (%s, %s, %s)",
                                            (rid, grp, (wid + 1) * 10 + seq),
                                        )
                                        wcur.execute(
                                            f"UPDATE {tbl} SET amt = amt + 3 WHERE id = %s",
                                            (rid,),
                                        )
                                        if seq % 5 == 0:
                                            wcur.execute(f"DELETE FROM {tbl} WHERE id = %s", (rid,))
                                        if seq % 7 == 0:
                                            wconn.rollback()
                                        else:
                                            wconn.commit()
                                            commits[wid] += 1
                                            writer_committed[wid].set()
                                        break
                                    except BaseException as exc:
                                        try:
                                            wconn.rollback()
                                        except Exception:
                                            pass
                                        if not self._is_retryable_mysql_lock_err(exc) or attempt == 19:
                                            raise
                                        time.sleep(min(0.1, 0.005 * (attempt + 1)))
                                seq += 1
                        except BaseException as exc:
                            writer_ready.abort()
                            with err_lock:
                                errors.append(exc)
                        finally:
                            if wcur is not None:
                                try:
                                    wcur.close()
                                except Exception:
                                    pass
                            if wconn is not None:
                                try:
                                    wconn.rollback()
                                    wconn.close()
                                except Exception:
                                    pass

                    def _repair_runner() -> None:
                        rconn = None
                        rc = None
                        try:
                            rconn = self._connect(dsn)
                            rc = rconn.cursor()
                            rc.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                            rconn.commit()
                            rm = MaterializedViewManager(rconn, dialect=Dialect.MYSQL)
                            writer_ready.wait(timeout=20.0)
                            for committed in writer_committed:
                                if not committed.wait(timeout=30.0):
                                    raise AssertionError("a stress writer never committed a transaction")
                            for target_v in (v_alg, v_rec, v_t3, v_join, v_t3, v_rec):
                                res = rm.repair_view(target_v)
                                self.assertEqual(res, {target_v: "repaired"})
                                repairs_done["count"] += 1
                            self.assertEqual(self._read_session_isolation(rc), self._iso_token(iso))
                            rconn.commit()
                        except BaseException as exc:
                            writer_ready.abort()
                            with err_lock:
                                errors.append(exc)
                        finally:
                            stop_writers.set()
                            if rc is not None:
                                try:
                                    rc.close()
                                except Exception:
                                    pass
                            if rconn is not None:
                                try:
                                    rconn.rollback()
                                    rconn.close()
                                except Exception:
                                    pass

                    ths = [
                        threading.Thread(target=_stress_writer, args=(0,), daemon=True),
                        threading.Thread(target=_stress_writer, args=(1,), daemon=True),
                        threading.Thread(target=_repair_runner, daemon=True),
                    ]
                    for th in ths:
                        th.start()
                    for th in ths:
                        th.join(timeout=90.0)
                    self.assertFalse([th.name for th in ths if th.is_alive()], f"MySQL {iso} repair stress hung")
                    self.assertEqual(errors, [], f"MySQL {iso} repair stress errors: {errors!r}")
                    self.assertEqual(repairs_done["count"], 6)
                    self.assertTrue(all(c > 0 for c in commits), f"writers did not run: {commits!r}")
                    _assert_all_exact(f"{iso} repair stress")
        finally:
            try:
                monitor.rollback()
                monitor.close()
            except Exception:
                pass
            try:
                admin.rollback()
                if mgr is not None:
                    for vn in (v_alg, v_rec, v_t3, v_join, v_uq):
                        try:
                            mgr.drop_view(vn)
                        except Exception:
                            admin.rollback()
                c = admin.cursor()
                c.execute(f"DROP TABLE IF EXISTS {tbl}")
                c.execute(f"DROP TABLE IF EXISTS {dim}")
                c.execute(f"DROP TABLE IF EXISTS {uq_tbl}")
                admin.commit()
                c.close()
            except Exception:
                pass
            admin.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
