#!/usr/bin/env python3
"""Independent-session tests for incremental materialized-view backfill."""

from __future__ import annotations

from decimal import Decimal

import os
import sqlite3
import tempfile
import threading
import unittest
import warnings
import uuid
from urllib.parse import urlparse

from matview_sync import Dialect, EngineLimitationWarning, MaterializedViewManager
from test_matview_sync import _register_sqlite_compat_udfs


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
                    second.execute("UPDATE ledger SET amt = 888 WHERE id = 9;")
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
                    second_cur.execute(
                        f"UPDATE {table_name} SET amt = %s, approx_amt = %s WHERE id = %s",
                        (888, 9.87654321, 9),
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
            plan = manager.register_view(
                f"CREATE VIEW {view_name} AS "
                f"SELECT grp, COUNT(*) AS row_count, SUM(amount) AS total "
                f"FROM {table_name} GROUP BY grp;"
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


if __name__ == "__main__":
    unittest.main(verbosity=2)