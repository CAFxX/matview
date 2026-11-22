#!/usr/bin/env python3
"""Concurrency and exactness contract for materialized AVG maintenance.

Regression for a native PostgreSQL 18 failure: AVG was maintained by a
per-group recomputation subquery over the base table inside the same trigger
statement that locks the group's backing row. Under READ COMMITTED the
subquery reads the statement snapshot, which is taken *before* the statement
waits for the row lock held by a concurrent writer. After that writer commits,
the waiting statement folds COUNT/SUM correctly from the latest row version but
stores an AVG computed without the other writer's rows.

The native tests below force exactly that interleaving with two independent
connections and explicit synchronization (the second writer is observed in an
engine lock wait attributed to the first session before the first one commits), then compare the materialized
rows against the original view executed natively by the engine -- the only
oracle. Values are compared with their exact decimal scale.

The offline tests assert the maintenance contract that makes the interleaving
safe for argument types whose native AVG is exactly SUM/COUNT: the AVG column is
derived from the SUM/COUNT of the same locked backing row and never from a base
table read. They are structural guards; native semantics are only established by
the native tests.
"""

from __future__ import annotations

import os
import re
import threading
import time
import unittest
import uuid
import warnings
from decimal import Decimal
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from matview_sync import (
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    generate_materialized_view,
    parse_table_schema,
)
from test_matview_sync import FLOAT_OPT_IN, pin_native_plan

LOCK_WAIT_TIMEOUT_S = 20.0
# Once one lock-wait probe has timed out in this process, later probes wait at
# most this long, so a systematic probe failure costs one full timeout plus a
# bounded amount per sub-test instead of LOCK_WAIT_TIMEOUT_S each. The failure
# itself is still reported (with diagnostics) by every affected sub-test.
LOCK_WAIT_TIMEOUT_AFTER_FAILURE_S = 3.0
_lock_probe_timed_out = threading.Event()


# ---------------------------------------------------------------------------
# Offline maintenance-shape contract
# ---------------------------------------------------------------------------

_VIEW = (
    "CREATE VIEW t_avg AS SELECT grp, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS a, "
    "AVG(amt2) AS a2 FROM t GROUP BY grp;"
)


def _plan_statements(
    dialect: Dialect,
    version: str,
    amt_type: str,
    *,
    indexes: Optional[Sequence[str]] = None,
) -> Tuple[str, str, List[str]]:
    schema = (
        f"CREATE TABLE t (id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, "
        f"amt {amt_type}, amt2 {amt_type});"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        plan = generate_materialized_view(
            parse_table_schema(schema, dialect=dialect),
            _VIEW,
            dialect=dialect,
            engine_version=version,
            indexes=indexes,
            ignore_limitations=True,
            accept_limitations=FLOAT_OPT_IN,
            incremental_backfill=True,
        )
    stmts = list(plan.trigger_ddls)
    stmts.extend(plan.incremental_backfill_spec.chunk_sql_templates if plan.incremental_backfill_spec else [])
    stmts.extend(plan.incremental_populate_sqls or [])
    return plan.backing_table_name, plan.reader_view_ddl, stmts


def _avg_assignments(stmts: Sequence[str], col: str) -> List[str]:
    """Return the right-hand side of every `col = ...` SET assignment."""
    out: List[str] = []
    pat = re.compile(rf"(?<![\w.]){re.escape(col)}\s*=\s*")
    for sql in stmts:
        for m in pat.finditer(sql):
            depth, i = 0, m.end()
            while i < len(sql):
                ch = sql[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch in ",;" and depth == 0:
                    break
                i += 1
            out.append(sql[m.end():i])
    return out


class AvgMaintenanceShapeTests(unittest.TestCase):
    DERIVABLE = [
        (Dialect.POSTGRES, "postgresql 15", "SMALLINT"),
        (Dialect.POSTGRES, "postgresql 18", "INTEGER"),
        (Dialect.POSTGRES, "postgresql 18", "BIGINT"),
        (Dialect.MYSQL, "mysql 8.4", "BIGINT"),
        (Dialect.MYSQL, "mysql 9.7", "INT UNSIGNED"),
        (Dialect.MYSQL, "mysql 8.4", "DECIMAL(12,3)"),
        (Dialect.MYSQL, "mariadb 10.11", "BIGINT"),
        (Dialect.MYSQL, "mariadb 11.8", "DECIMAL(20,6)"),
    ]
    RECOMPUTED = [
        (Dialect.POSTGRES, "postgresql 18", "NUMERIC(12,3)"),
        (Dialect.POSTGRES, "postgresql 18", "DOUBLE PRECISION"),
        (Dialect.MYSQL, "mysql 8.4", "DOUBLE"),
    ]

    def test_exact_avg_is_derived_from_locked_row_without_base_reads(self) -> None:
        for dialect, version, amt_type in self.DERIVABLE:
            with self.subTest(dialect=dialect.value, version=version, type=amt_type):
                backing, reader_ddl, stmts = _plan_statements(dialect, version, amt_type)
                if dialect == Dialect.MYSQL:
                    # Stage 4 derives unindexed exact AVG in the reader view from
                    # the locked row's running sum/count columns (zero base reads
                    # and zero trigger assignments to a/a2), while indexed AVG
                    # still derives from the locked backing row without base reads.
                    for st in stmts:
                        self.assertNotRegex(st, r"(?i)\(\s*SELECT\s+AVG\(", st)
                    for col in ("a", "a2"):
                        self.assertIn(f") AS {col}", reader_ddl)
                        self.assertEqual(_avg_assignments(stmts, col), [])
                    idx_backing, _, idx_stmts = _plan_statements(
                        dialect,
                        version,
                        amt_type,
                        indexes=["CREATE INDEX idx_t_avg_a ON t_avg (a);", "CREATE INDEX idx_t_avg_a2 ON t_avg (a2);"],
                    )
                    for col in ("a", "a2"):
                        rhs = _avg_assignments(idx_stmts, col)
                        self.assertTrue(rhs, f"no maintenance assignment for indexed {col}")
                        for expr in rhs:
                            self.assertNotRegex(expr, r"(?i)\(\s*SELECT\b", expr)
                            self.assertIn(idx_backing, expr)
                else:
                    for col in ("a", "a2"):
                        rhs = _avg_assignments(stmts, col)
                        self.assertTrue(rhs, f"no maintenance assignment for {col}")
                        for expr in rhs:
                            # A subquery here would read the base table through
                            # a snapshot that predates the backing-row lock.
                            self.assertNotRegex(expr, r"(?i)\(\s*SELECT\b", expr)
                            self.assertIn(backing, expr)

    def test_inexact_avg_types_keep_native_recompute(self) -> None:
        # These keep native recomputation (exact scale / approximate
        # accumulation order), as a separate statement after the row lock
        # (see test_pgmy_exact_float_aggregates.py).
        for dialect, version, amt_type in self.RECOMPUTED:
            with self.subTest(dialect=dialect.value, type=amt_type):
                _, _, stmts = _plan_statements(dialect, version, amt_type)
                self.assertTrue(
                    any(re.search(r"(?i)\(SELECT AVG\(", e) for e in _avg_assignments(stmts, "a"))
                )


# ---------------------------------------------------------------------------
# Native deterministic interleavings
# ---------------------------------------------------------------------------


def _connect(engine: Dialect, dsn: str):
    if engine == Dialect.POSTGRES:
        import psycopg2

        return psycopg2.connect(dsn)
    import pymysql

    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=False,
    )


def _native_engines() -> List[Tuple[Dialect, str]]:
    out = []
    if os.environ.get("POSTGRES_DSN"):
        out.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
    if os.environ.get("MYSQL_DSN"):
        out.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
    return out


def _cleanup(engine: Dialect, conn, manager, plan, schema: str, table: str) -> None:
    try:
        conn.rollback()
        cur = conn.cursor()
        if engine == Dialect.POSTGRES:
            cur.execute("SET search_path TO public")
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        else:
            if manager is not None and plan is not None:
                try:
                    manager.drop_view(plan.view_name)
                except Exception:
                    pass
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        conn.close()


def _session_id(engine: Dialect, conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid()" if engine == Dialect.POSTGRES else "SELECT CONNECTION_ID()")
    sid = int(cur.fetchone()[0])
    conn.commit()
    return sid


# Lock-wait evidence.
#
# A session counts as blocked only on *positive* engine evidence that it is in
# a lock wait: never because a statement is merely slow or still running.
#
# * PostgreSQL: pg_stat_activity.wait_event_type = 'Lock', attributed to the
#   blocking backends by pg_blocking_pids().
# * MySQL 8.0+: performance_schema.data_lock_waits (live), attributed through
#   performance_schema.threads to the blocking connection.
# * MariaDB (no data_lock_waits) and MySQL: information_schema.innodb_trx
#   joined with innodb_lock_waits (MariaDB, MySQL 5.7), attributed to the
#   blocking connection. These INFORMATION_SCHEMA tables are served from a
#   cache that InnoDB refreshes only when nobody has read it for 100 ms
#   (trx0i_s.cc, CACHE_MIN_IDLE_TIME); a monitor polling faster than that
#   keeps reading one frozen snapshot forever (observed on MariaDB 10.11, 11.8,
#   12.3.3 and MySQL 8.4: the waiter never appears and long-gone transactions
#   keep being listed). Reads are therefore spaced by _IS_TRX_CACHE_IDLE_S.
# * SHOW ENGINE INNODB STATUS (live, both families): the waiter's transaction
#   block reports "LOCK WAIT" and "WAITING ... FOR THIS LOCK TO BE GRANTED".
#   It does not name the blocker, so it is evidence only when no specific
#   blocker is required.
_IS_TRX_CACHE_IDLE_S = 0.15
_is_trx_cache_guard = threading.Lock()
_is_trx_last_read = [float("-inf")]


def _is_trx_cache_due() -> bool:
    """True (and claims the read) when the I_S trx cache can have refreshed."""
    with _is_trx_cache_guard:
        now = time.monotonic()
        if now - _is_trx_last_read[0] < _IS_TRX_CACHE_IDLE_S:
            return False
        _is_trx_last_read[0] = now
        return True


def _mark_is_trx_read() -> None:
    with _is_trx_cache_guard:
        _is_trx_last_read[0] = time.monotonic()


def _innodb_status_wait_block(status: str, sid: int) -> Optional[str]:
    """The waiting transaction block for connection `sid`, if it is in a lock wait."""
    thread = re.compile(rf"\b(?:MySQL|MariaDB) thread id {int(sid)},")
    for block in re.split(r"\n---TRANSACTION ", status):
        if (
            thread.search(block)
            and re.search(r"(?m)^LOCK WAIT\b", block)
            and "FOR THIS LOCK TO BE GRANTED" in block
        ):
            return block
    return None


def _lock_wait_evidence(
    engine: Dialect, monitor, sid: int, *, force_is_trx: bool = False
) -> Tuple[List[Tuple[str, Optional[frozenset]]], dict]:
    """Return ([(source, blocker session ids or None)], {source: raw observation}).

    The first list holds only sources that positively show `sid` in a lock
    wait; the blocker set is None when the source cannot attribute it.
    """
    cur = monitor.cursor()
    evidence: List[Tuple[str, Optional[frozenset]]] = []
    seen: dict = {}

    def query(name: str, sql: str, params: Tuple = ()) -> Optional[list]:
        try:
            cur.execute(sql, params)
            rows = list(cur.fetchall())
        except Exception as exc:
            monitor.rollback()
            seen[name] = f"unavailable: {exc!r}"
            return None
        monitor.commit()
        seen[name] = rows
        return rows

    if engine == Dialect.POSTGRES:
        rows = query(
            "pg_stat_activity",
            "SELECT wait_event_type, wait_event, pg_blocking_pids(pid) FROM pg_stat_activity WHERE pid = %s",
            (sid,),
        )
        if rows and rows[0][0] == "Lock":
            evidence.append(("pg_stat_activity", frozenset(int(p) for p in (rows[0][2] or []))))
        return evidence, seen

    rows = query(
        "performance_schema.data_lock_waits",
        "SELECT bt.PROCESSLIST_ID FROM performance_schema.data_lock_waits w "
        "JOIN performance_schema.threads rt ON rt.THREAD_ID = w.REQUESTING_THREAD_ID "
        "LEFT JOIN performance_schema.threads bt ON bt.THREAD_ID = w.BLOCKING_THREAD_ID "
        "WHERE rt.PROCESSLIST_ID = %s",
        (sid,),
    )
    if rows:
        evidence.append(
            ("performance_schema.data_lock_waits", frozenset(int(r[0]) for r in rows if r[0] is not None))
        )

    if force_is_trx:
        time.sleep(_IS_TRX_CACHE_IDLE_S)
        _mark_is_trx_read()
        due = True
    else:
        due = _is_trx_cache_due()
    if due:
        name = "information_schema.innodb_trx+innodb_lock_waits"
        rows = query(
            name,
            "SELECT r.trx_state, b.trx_mysql_thread_id FROM information_schema.innodb_trx r "
            "LEFT JOIN information_schema.innodb_lock_waits w ON w.requesting_trx_id = r.trx_id "
            "LEFT JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_trx_id "
            "WHERE r.trx_mysql_thread_id = %s",
            (sid,),
        )
        if rows is None:  # MySQL 8.0+ has no innodb_lock_waits
            name = "information_schema.innodb_trx"
            rows = query(
                name,
                "SELECT trx_state, NULL FROM information_schema.innodb_trx WHERE trx_mysql_thread_id = %s",
                (sid,),
            )
        _mark_is_trx_read()
        if rows and any(r[0] == "LOCK WAIT" for r in rows):
            blockers = frozenset(int(r[1]) for r in rows if r[1] is not None)
            evidence.append((name, blockers or None))
    else:
        seen["information_schema.innodb_trx"] = "skipped: read within the cache idle interval"

    rows = query("SHOW ENGINE INNODB STATUS", "SHOW ENGINE INNODB STATUS")
    if rows:
        block = _innodb_status_wait_block(str(rows[0][2]), sid)
        seen["SHOW ENGINE INNODB STATUS"] = block[:600] if block else "no LOCK WAIT block for this connection"
        if block:
            evidence.append(("SHOW ENGINE INNODB STATUS", None))
    return evidence, seen


def _is_lock_waiting(engine: Dialect, monitor, sid: int, blocker_sid: Optional[int] = None) -> bool:
    """True only on positive evidence that `sid` waits for an engine lock.

    With `blocker_sid`, the evidence must also attribute the wait to a lock
    held by that session; sources that cannot name the blocker do not count.
    """
    evidence, _ = _lock_wait_evidence(engine, monitor, sid)
    return any(
        blocker_sid is None or (blockers is not None and blocker_sid in blockers)
        for _, blockers in evidence
    )


def _wait_until_lock_waiting(
    engine: Dialect, monitor, sid: int, done: threading.Event, blocker_sid: Optional[int] = None
) -> None:
    budget = LOCK_WAIT_TIMEOUT_AFTER_FAILURE_S if _lock_probe_timed_out.is_set() else LOCK_WAIT_TIMEOUT_S
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if _is_lock_waiting(engine, monitor, sid, blocker_sid):
            return
        if done.is_set():
            raise AssertionError(
                "second session finished without waiting for the first session's "
                "lock; the intended interleaving was not exercised"
            )
        time.sleep(0.02)
    _lock_probe_timed_out.set()
    raise AssertionError(
        f"session {sid} never entered an engine lock wait"
        + (f" attributed to session {blocker_sid}" if blocker_sid is not None else "")
        + f" within {budget}s; diagnostics={_lock_diagnostics(engine, monitor, sid)!r}"
    )


def _lock_diagnostics(engine: Dialect, monitor, sid: Optional[int] = None) -> Any:
    """What every catalog source showed (fresh I_S snapshot), for failure messages."""
    cur = monitor.cursor()
    out: dict = {}
    try:
        if engine == Dialect.POSTGRES:
            cur.execute(
                "SELECT pid, state, wait_event_type, wait_event, pg_blocking_pids(pid), left(query, 120) "
                "FROM pg_stat_activity"
            )
            out["pg_stat_activity"] = cur.fetchall()
        else:
            cur.execute("SELECT id, command, time, state, LEFT(info, 120) FROM information_schema.processlist")
            out["processlist"] = cur.fetchall()
            monitor.commit()
            time.sleep(_IS_TRX_CACHE_IDLE_S)  # let the I_S trx cache refresh
            cur.execute("SELECT trx_mysql_thread_id, trx_state, LEFT(trx_query, 120) FROM information_schema.innodb_trx")
            out["innodb_trx"] = cur.fetchall()
            _mark_is_trx_read()
    except Exception as exc:  # diagnostics only
        out["error"] = repr(exc)
    finally:
        monitor.commit()
    if sid is not None:
        try:
            out["evidence"], out["sources"] = _lock_wait_evidence(engine, monitor, sid, force_is_trx=True)
        except Exception as exc:  # diagnostics only
            out["evidence_error"] = repr(exc)
    return out


def _exact(v: Any) -> Any:
    """Represent numerics with their exact scale (Decimal('1.50') != Decimal('1.5'))."""
    if isinstance(v, Decimal):
        return ("decimal", v.as_tuple())
    if isinstance(v, int) and not isinstance(v, bool):
        return ("int", v)
    return (type(v).__name__, v)


Op = Callable[[Any, str], None]


def _ins(pk: int, grp: str, amt: Optional[int]) -> Op:
    return lambda cur, t: cur.execute(f"INSERT INTO {t} (id, grp, amt) VALUES (%s, %s, %s)", (pk, grp, amt))


def _upd_amt(pk: int, delta: int) -> Op:
    return lambda cur, t: cur.execute(f"UPDATE {t} SET amt = amt + %s WHERE id = %s", (delta, pk))


def _move(pk: int, grp: str) -> Op:
    return lambda cur, t: cur.execute(f"UPDATE {t} SET grp = %s WHERE id = %s", (grp, pk))


def _del(pk: int) -> Op:
    return lambda cur, t: cur.execute(f"DELETE FROM {t} WHERE id = %s", (pk,))


# (name, first session's uncommitted op, second session's op that must wait).
# Group G1 holds ids 1..6; G2 holds ids 7..8. Every pair touches G1 in both
# sessions so the second session waits for the first session's G1 row lock.
INTERLEAVINGS: List[Tuple[str, Op, Op]] = [
    ("insert/insert", _ins(101, "G1", 7), _ins(201, "G1", 1000)),
    ("insert/delete", _ins(101, "G1", 7), _del(2)),
    ("delete/insert", _del(1), _ins(201, "G1", 13)),
    ("delete/delete", _del(1), _del(2)),
    ("update/insert", _upd_amt(3, 5), _ins(201, "G1", 2)),
    ("insert/update", _ins(101, "G1", 999), _upd_amt(4, -11)),
    ("move-out/insert", _move(5, "G2"), _ins(201, "G1", 1)),
    ("insert/move-in", _ins(101, "G1", 3), _move(7, "G1")),
    ("null/insert", _ins(101, "G1", None), _ins(201, "G1", 4)),
    ("delete-all-but-one/insert", _del(1), _ins(201, "G1", 9)),
]

SEED = [(1, "G1", 10), (2, "G1", 21), (3, "G1", 33), (4, "G1", -4), (5, "G1", 5), (6, "G1", None),
        (7, "G2", 100), (8, "G2", 11)]


class NativeAvgInterleavingTests(unittest.TestCase):
    def _fixture(self, engine: Dialect, dsn: str, amt_type: str):
        suffix = uuid.uuid4().hex[:10]
        schema = f"mv_avgrace_{suffix}"
        table = f"avgrace_{suffix}"
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        if engine == Dialect.POSTGRES:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
        engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
        cur.execute(
            f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt {amt_type})"
            f"{engine_suffix}"
        )
        cur.executemany(f"INSERT INTO {table} (id, grp, amt) VALUES (%s, %s, %s)", SEED)
        conn.commit()
        manager = MaterializedViewManager(conn, dialect=engine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan = manager.register_view(
                f"CREATE VIEW v_{table} AS SELECT grp, COUNT(*) AS c, SUM(amt) AS s, "
                f"COUNT(amt) AS n, AVG(amt) AS a FROM {table} GROUP BY grp;",
                ignore_limitations=True,
                accept_limitations=FLOAT_OPT_IN,
            )
        conn.commit()
        return conn, manager, plan, schema, table

    def _assert_matches_native_view(self, engine: Dialect, conn, plan, table: str, label: str) -> None:
        cur = conn.cursor()
        conn.commit()
        cur.execute(f"SELECT grp, c, s, n, a FROM {plan.view_name} ORDER BY grp")
        actual = [tuple(_exact(v) for v in row) for row in cur.fetchall()]
        pin_native_plan(cur, engine)
        cur.execute(f"SELECT grp, COUNT(*), SUM(amt), COUNT(amt), AVG(amt) FROM {table} GROUP BY grp")
        expected = sorted((tuple(_exact(v) for v in row) for row in cur.fetchall()), key=lambda r: r[0])
        conn.commit()
        self.assertEqual(actual, expected, f"{engine.value} {label}: materialized view != native view")

    def _run_interleaving(
        self, engine: Dialect, dsn: str, amt_type: str, isolation: Optional[str],
        name: str, first: Op, second: Op,
    ) -> None:
        conn, manager, plan, schema, table = self._fixture(engine, dsn, amt_type)
        s1 = _connect(engine, dsn)
        s2 = _connect(engine, dsn)
        monitor = _connect(engine, dsn)
        done = threading.Event()
        errors: List[BaseException] = []
        try:
            for s in (s1, s2):
                c = s.cursor()
                if engine == Dialect.POSTGRES:
                    c.execute(f'SET search_path TO "{schema}"')
                if isolation:
                    c.execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {isolation}")
                s.commit()
            sid1 = _session_id(engine, s1)
            sid2 = _session_id(engine, s2)

            first(s1.cursor(), table)  # holds the G1 backing-row lock, uncommitted

            def run_second() -> None:
                try:
                    second(s2.cursor(), table)
                    s2.commit()
                except BaseException as exc:  # surfaced below
                    errors.append(exc)
                    s2.rollback()
                finally:
                    done.set()

            th = threading.Thread(target=run_second, daemon=True)
            th.start()
            try:
                _wait_until_lock_waiting(engine, monitor, sid2, done, blocker_sid=sid1)
            finally:
                s1.commit()
                # Never touch s2 (cleanup below) while the thread still uses it.
                th.join(timeout=LOCK_WAIT_TIMEOUT_S)
            self.assertFalse(th.is_alive(), "second session did not finish after the first committed")
            if errors:
                raise errors[0]
            self._assert_matches_native_view(engine, conn, plan, table, f"{amt_type} {isolation} {name}")
        finally:
            for s in (s1, s2, monitor):
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            _cleanup(engine, conn, manager, plan, schema, table)

    def _configs(self, engine: Dialect) -> List[Tuple[str, Optional[str]]]:
        if engine == Dialect.POSTGRES:
            # Default READ COMMITTED. NUMERIC/DOUBLE PRECISION AVG is
            # recomputed natively, in a statement after the group lock (a
            # fresh snapshot); it was stale before statement-level maintenance.
            return [(t, None) for t in ("BIGINT", "INTEGER", "SMALLINT", "NUMERIC(12,3)", "DOUBLE PRECISION")]
        # Default REPEATABLE READ and READ COMMITTED.
        return [(t, iso) for t in ("BIGINT", "DECIMAL(12,3)") for iso in (None, "READ COMMITTED")]

    def test_native_avg_serialized_with_concurrent_group_writers(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for amt_type, iso in self._configs(engine):
                for name, first, second in INTERLEAVINGS:
                    with self.subTest(engine=engine.value, type=amt_type, isolation=iso, case=name):
                        self._run_interleaving(engine, dsn, amt_type, iso, name, first, second)

    def test_native_backfill_chunk_serialized_with_concurrent_writer(self) -> None:
        """A backfill chunk folding a group must not store an AVG that misses a
        writer which committed while the chunk waited for the group row lock."""
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for amt_type in ("BIGINT",) if engine == Dialect.POSTGRES else ("BIGINT", "DECIMAL(12,3)"):
                with self.subTest(engine=engine.value, type=amt_type):
                    self._run_backfill_interleaving(engine, dsn, amt_type)

    def _run_backfill_interleaving(self, engine: Dialect, dsn: str, amt_type: str) -> None:
        suffix = uuid.uuid4().hex[:10]
        schema = f"mv_avgbf_{suffix}"
        table = f"avgbf_{suffix}"
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        manager = plan = None
        writer = _connect(engine, dsn)
        monitor = _connect(engine, dsn)
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
                writer.cursor().execute(f'SET search_path TO "{schema}"')
                writer.commit()
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(
                f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt {amt_type})"
                f"{engine_suffix}"
            )
            # One group so every chunk folds into the same backing row.
            cur.executemany(
                f"INSERT INTO {table} (id, grp, amt) VALUES (%s, %s, %s)",
                [(i, "G", (i * 37) % 101) for i in range(1, 13)],
            )
            conn.commit()
            backfill_sid = _session_id(engine, conn)
            writer_sid = _session_id(engine, writer)
            manager = MaterializedViewManager(conn, dialect=engine)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                plan = manager.register_view(
                    f"CREATE VIEW v_{table} AS SELECT grp, COUNT(*) AS c, SUM(amt) AS s, "
                    f"COUNT(amt) AS n, AVG(amt) AS a FROM {table} GROUP BY grp;",
                    populate=False,
                    ignore_limitations=True,
                    accept_limitations=FLOAT_OPT_IN,
                )
            interleaved = threading.Event()
            helper_errors: List[BaseException] = []

            def commit_when_chunk_waits() -> None:
                try:
                    _wait_until_lock_waiting(engine, monitor, backfill_sid, threading.Event(), blocker_sid=writer_sid)
                    interleaved.set()
                except BaseException as exc:
                    helper_errors.append(exc)
                finally:
                    writer.commit()

            def on_progress(batch_idx: int, _done: int) -> None:
                # 12 rows / batch_size 4 = 3 chunks. Interleave with the *last*
                # chunk: an earlier chunk's stale value would be overwritten by
                # a later chunk and hide the defect.
                if batch_idx != 2:
                    return
                # Chunks 1-2 committed and created the group row. A writer
                # above the backfill range updates it and stays uncommitted
                # while the final chunk folds the same group.
                writer.cursor().execute(
                    f"INSERT INTO {table} (id, grp, amt) VALUES (%s, %s, %s)", (10_000, "G", 5000)
                )
                threading.Thread(target=commit_when_chunk_waits, daemon=True).start()

            manager.backfill_incrementally(plan, batch_size=4, on_progress=on_progress, snapshot_isolation=False)
            if helper_errors:
                raise helper_errors[0]
            self.assertTrue(interleaved.is_set())
            self._assert_matches_native_view(engine, conn, plan, table, f"backfill {amt_type}")
        finally:
            for s in (writer, monitor):
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            _cleanup(engine, conn, manager, plan, schema, table)


class NativeAvgExactnessTests(unittest.TestCase):
    """Derived AVG must equal the native AVG value *and* scale at boundaries."""

    def test_native_avg_exact_boundaries(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            types = (
                ["SMALLINT", "INTEGER", "BIGINT"]
                if engine == Dialect.POSTGRES
                else ["TINYINT", "INT", "BIGINT", "BIGINT UNSIGNED", "DECIMAL(10,0)", "DECIMAL(30,7)", "DECIMAL(65,30)"]
            )
            for amt_type in types:
                with self.subTest(engine=engine.value, type=amt_type):
                    self._run_exactness(engine, dsn, amt_type)

    @staticmethod
    def _values(amt_type: str) -> List[Any]:
        t = amt_type.upper()
        if t == "SMALLINT":
            return [32767, 32767, -32768, 1, 2, -1, 0]
        if t == "TINYINT":
            return [127, 127, -128, 1, 2, -1, 0]
        if t in ("INTEGER", "INT"):
            return [2147483647, 2147483647, -2147483648, 1, 2, -1, 0]
        if t == "BIGINT":
            return [9223372036854775807, 9223372036854775807, -9223372036854775808, 1, 2, -1, 0]
        if t == "BIGINT UNSIGNED":
            return [18446744073709551615, 18446744073709551615, 0, 1, 2]
        if t == "DECIMAL(10,0)":
            return [Decimal("9999999999"), Decimal("-9999999999"), Decimal("1"), Decimal("2"), Decimal("5")]
        if t == "DECIMAL(30,7)":
            return [Decimal("12345678901234567890123.0000005"), Decimal("-0.0000005"), Decimal("1.0000001"),
                    Decimal("2"), Decimal("-3.3333333")]
        return [Decimal("1.000000000000000000000000000005"), Decimal("-0.000000000000000000000000000005"),
                Decimal("2.5"), Decimal("33333333333333333333333333333.333333333333333333333333333333")]

    def _run_exactness(self, engine: Dialect, dsn: str, amt_type: str) -> None:
        suffix = uuid.uuid4().hex[:10]
        schema = f"mv_avgx_{suffix}"
        table = f"avgx_{suffix}"
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        manager = plan = None
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            engine_suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(
                f"CREATE TABLE {table} (id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt {amt_type})"
                f"{engine_suffix}"
            )
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                plan = manager.register_view(
                    f"CREATE VIEW v_{table} AS SELECT grp, COUNT(*) AS c, COUNT(amt) AS n, AVG(amt) AS a "
                    f"FROM {table} GROUP BY grp;",
                    ignore_limitations=True,
                    accept_limitations=FLOAT_OPT_IN,
                )
            conn.commit()
            vals = self._values(amt_type)
            pk = 0

            def check(label: str) -> None:
                conn.commit()
                cur.execute(f"SELECT grp, c, n, a FROM {plan.view_name} ORDER BY grp")
                actual = [tuple(_exact(v) for v in r) for r in cur.fetchall()]
                pin_native_plan(cur, engine)
                cur.execute(f"SELECT grp, COUNT(*), COUNT(amt), AVG(amt) FROM {table} GROUP BY grp")
                expected = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=lambda r: r[0])
                conn.commit()
                self.assertEqual(actual, expected, f"{engine.value} {amt_type} {label}")

            # Single-row group, growth one row at a time (insert path, then
            # conflict path), a NULL, removal back to one row, to all-NULL and
            # to empty; then a multi-row statement and a group move.
            for v in vals:
                pk += 1
                cur.execute(f"INSERT INTO {table} (id, grp, amt) VALUES (%s, 'A', %s)", (pk, v))
                conn.commit()
                check(f"insert {v}")
            pk += 1
            cur.execute(f"INSERT INTO {table} (id, grp, amt) VALUES (%s, 'A', NULL)", (pk,))
            conn.commit()
            check("insert NULL")
            for victim in range(1, len(vals)):
                cur.execute(f"DELETE FROM {table} WHERE id = %s", (victim,))
                conn.commit()
                check(f"delete {victim}")
            cur.execute(f"UPDATE {table} SET amt = NULL WHERE id = %s", (len(vals),))
            conn.commit()
            check("all NULL")
            cur.executemany(
                f"INSERT INTO {table} (id, grp, amt) VALUES (%s, %s, %s)",
                [(1000 + i, "B", v) for i, v in enumerate(vals)],
            )
            conn.commit()
            check("bulk insert")
            cur.execute(f"UPDATE {table} SET grp = 'A' WHERE id IN (1000, 1002)")
            conn.commit()
            check("move")
            cur.execute(f"DELETE FROM {table} WHERE grp = 'B'")
            conn.commit()
            check("delete group")
        finally:
            _cleanup(engine, conn, manager, plan, schema, table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
