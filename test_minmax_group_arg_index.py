#!/usr/bin/env python3
"""MIN/MAX extreme deletes: the (group..., arg) index recommendation (lane L2).

Contract: deleting or updating a group's current MIN/MAX recomputes
``MIN/MAX(arg) ... WHERE <group key> = <value>`` from the base table. For a
single-table view whose group keys and MIN/MAX argument are plain columns,
``recommended_base_indexes`` names ``(group..., arg)`` (one index per distinct
argument; it also serves every group-key lookup, so the ``(group)`` index is
no longer recommended separately). With that index the recompute is one index
probe; with the group-key index alone it reads the whole group. When the
index is missing the plan diagnostics (and the CLI text) carry a
``minmax_extreme_recompute`` performance note with the CREATE INDEX statement.

Measured natively for this lane (40k rows, 2 groups, 100 deletes of the
current minimum in one transaction, net of the same DELETE without a
materialized view; method in the PR description): SQLite 3.46 no index 874 us, (g) 1967,
(g, v) 3; PostgreSQL 18 1268 / 1072 / 519; MySQL 8.4 7562 / 17060 / ~0;
MariaDB 11.8 7680 / 7765 / 311. The multiset auxiliary table (research doc
section 3.3 (b)) is not better on any engine for plain-column arguments; it
remains the follow-up for expression and join arguments (TODO.md).

The maintenance SQL is unchanged by this lane, so its exactness and locking
are those of the existing tier-1 MIN/MAX path; the native differential below
re-checks it on the extreme-delete sequences (ties, NULL values, NULL group,
group moves, empty groups, multi-row deletes of several extremes, rollback)
against the live view, exactly. The cost class is asserted through engine
row-read counters, not wall-clock time: PostgreSQL pg_stat_xact_user_tables,
MySQL/MariaDB session Handler_read_* counters, SQLite VM steps (progress
handler; PostgreSQL with enable_seqscan off inside the measured transaction,
because a 350-row table makes a sequential scan cheapest even for the DELETE's
own primary-key lookup); each is also shown to detect the group scan when only the group-key
index exists. ConcurrentExtremeDeleteTests covers two independent sessions
deleting successive extremes of one group (stale snapshot after a lock wait).
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import unittest
import uuid
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple

from matview_sync import EngineLimitationWarning, MaterializedViewManager, generate_materialized_view, parse_table_schema

GROUP_ROWS = 120


def _plan(dialect: str, schema: str, select: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        return generate_materialized_view(
            parse_table_schema(schema, dialect=dialect), f"CREATE VIEW mv AS {select}", dialect=dialect, ignore_limitations=True
        )


def _rec_cols(plan: Any) -> List[Tuple[str, ...]]:
    return [tuple(c.lower() for c in r["columns"]) for r in plan.base_index_recommendations]


SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, w INTEGER, s TEXT);"


class OfflineRecommendationTests(unittest.TestCase):
    def test_shapes(self) -> None:
        cases = [
            ("SELECT g, MIN(v) AS a FROM t GROUP BY g", [("g", "v")]),
            ("SELECT g, MIN(v) AS a, MAX(v) AS b, COUNT(*) AS n FROM t GROUP BY g", [("g", "v")]),
            ("SELECT g, MIN(v) AS a, MAX(w) AS b FROM t GROUP BY g", [("g", "v"), ("g", "w")]),
            ("SELECT g, h, MAX(w) AS b FROM t GROUP BY g, h", [("g", "h", "w")]),
            ("SELECT g, MIN(g) AS a FROM t GROUP BY g", [("g",)]),
            # Expression argument: no index can serve it; the group-key index stays.
            ("SELECT g, MIN(v + w) AS a FROM t GROUP BY g", [("g",)]),
            # Expression group key: no (group, arg) index.
            ("SELECT g + 1 AS k, MIN(v) AS a FROM t GROUP BY g + 1", []),
            # BIT_AND/BIT_OR recompute scans the group whatever the index.
            ("SELECT g, BIT_AND(v) AS a FROM t GROUP BY g", [("g",)]),
            # Ungrouped: no group lookup at all.
            ("SELECT MIN(v) AS a FROM t", []),
        ]
        for dialect in ("sqlite", "postgres", "mysql"):
            for select, expected in cases:
                plan = _plan(dialect, SCHEMA, select)
                self.assertEqual(_rec_cols(plan), expected, f"{dialect}: {select}")

    def test_mysql_text_argument_keeps_group_key_index(self) -> None:
        plan = _plan("mysql", SCHEMA, "SELECT g, MIN(s) AS a FROM t GROUP BY g")
        self.assertEqual(_rec_cols(plan), [("g",)])

    def test_note_names_the_missing_index_and_disappears_when_present(self) -> None:
        select = "SELECT g, MIN(v) AS a, MAX(v) AS b FROM t GROUP BY g"
        for dialect in ("sqlite", "postgres", "mysql"):
            for existing in ("", "CREATE INDEX t_g ON t (g);"):
                plan = _plan(dialect, SCHEMA + existing, select)
                notes = [n for n in plan.diagnostics.details["performance_notes"] if n.startswith("minmax_extreme_recompute")]
                self.assertEqual(len(notes), 1, (dialect, existing))
                self.assertRegex(notes[0], r"CREATE INDEX \S+ ON t \(g, v\);")
            plan = _plan(dialect, SCHEMA + "CREATE INDEX t_gv ON t (g, v);", select)
            self.assertEqual(plan.base_index_recommendations, [])
            self.assertFalse([n for n in plan.diagnostics.details["performance_notes"] if n.startswith("minmax_extreme_recompute")])
            used = plan.diagnostics.details["existing_base_indexes_used"]
            self.assertEqual([u["reason"] for u in used], ["minmax_extreme_recompute"])

    def test_join_views_are_unchanged(self) -> None:
        schema = SCHEMA + "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, x INTEGER);"
        plan = _plan("postgres", schema, "SELECT t.g, MIN(u.x) AS a FROM t JOIN u ON u.tid = t.id GROUP BY t.g")
        self.assertNotIn(("g", "x"), _rec_cols(plan))


# Argument type classes: (dialect, type, recommended?). PostgreSQL btree rows
# are limited to 2704 bytes (later writes of long values would fail);
# MySQL/MariaDB character arguments compare under a collation and keys are
# limited to 3072 bytes.
TYPE_CLASSES: List[Tuple[str, str, bool]] = [
    ("postgres", "INTEGER", True), ("postgres", "BIGINT", True), ("postgres", "SMALLINT", True),
    ("postgres", "DOUBLE PRECISION", True), ("postgres", "NUMERIC(38, 10)", True), ("postgres", "DATE", True),
    ("postgres", "TIMESTAMP", True), ("postgres", "TIMESTAMPTZ", True), ("postgres", "UUID", True),
    ("postgres", "VARCHAR(100)", True), ("postgres", "CHAR(10)", True), ("postgres", "BOOLEAN", True),
    ("postgres", "INTERVAL", True),
    ("postgres", "TEXT", False), ("postgres", "VARCHAR", False), ("postgres", "VARCHAR(1000)", False),
    ("postgres", "BYTEA", False), ("postgres", "JSONB", False), ("postgres", "NUMERIC", False),
    ("postgres", "INTEGER[]", False),
    ("mysql", "INT", True), ("mysql", "BIGINT UNSIGNED", True), ("mysql", "DECIMAL(65, 30)", True),
    ("mysql", "DOUBLE", True), ("mysql", "DATE", True), ("mysql", "DATETIME(6)", True),
    ("mysql", "TIMESTAMP", True), ("mysql", "VARBINARY(255)", True),
    ("mysql", "VARCHAR(10)", False), ("mysql", "VARCHAR(1000)", False), ("mysql", "TEXT", False),
    ("mysql", "JSON", False), ("mysql", "BLOB", False), ("mysql", "VARBINARY(3100)", False),
    ("sqlite", "TEXT", True), ("sqlite", "BLOB", True),
]


class ArgumentTypeClassTests(unittest.TestCase):
    def test_type_classes(self) -> None:
        for dialect, typ, recommended in TYPE_CLASSES:
            schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a {typ});"
            plan = _plan(dialect, schema, "SELECT g, MAX(a) AS m FROM t GROUP BY g")
            recs = _rec_cols(plan)
            notes = [n for n in plan.diagnostics.details["performance_notes"] if n.startswith("minmax_extreme_recompute")]
            if recommended:
                self.assertEqual(recs, [("g", "a")], f"{dialect} {typ}")
            else:
                self.assertEqual(recs, [("g",)], f"{dialect} {typ}")
                self.assertTrue(any("no (g, a) index is recommended" in n for n in notes), f"{dialect} {typ}: {notes}")

    def test_mysql_long_group_key_does_not_get_an_uncreatable_index(self) -> None:
        plan = _plan("mysql", "CREATE TABLE t (id INTEGER PRIMARY KEY, g VARCHAR(800), a INT);", "SELECT g, MIN(a) AS m FROM t GROUP BY g")
        self.assertNotIn(("g", "a"), _rec_cols(plan))


def _max_values(dialect: str) -> List[Tuple[str, str, Any]]:
    """(column, type, maximum-size value) for every recommended type class."""
    if dialect == "postgres":
        return [
            ("c_num", "NUMERIC(38, 10)", "9999999999999999999999999999.9999999999"),
            ("c_vc", "VARCHAR(100)", "\U0001F600" * 100),
            ("c_ch", "CHAR(10)", "\U0001F600" * 10),
            ("c_big", "BIGINT", 9223372036854775807),
            ("c_ts", "TIMESTAMPTZ", "294276-12-31 23:59:59+00"),
            ("c_iv", "INTERVAL", "178000000 years"),
            ("c_uuid", "UUID", "ffffffff-ffff-ffff-ffff-ffffffffffff"),
        ]
    return [
        ("c_num", "DECIMAL(65, 30)", "99999999999999999999999999999999999.999999999999999999999999999999"),
        ("c_vb", "VARBINARY(255)", b"\xff" * 255),
        ("c_big", "BIGINT UNSIGNED", 18446744073709551615),
        ("c_dt", "DATETIME(6)", "9999-12-31 23:59:59.999999"),
    ]


def _check_recommended_indexes_accept_max_values(test: unittest.TestCase, dialect: str, conn: Any, table: str) -> None:
    """Every recommended index can be created, and a maximum-size value of
    the declared type can be written (and its extreme deleted) afterwards."""
    cols = _max_values(dialect)
    cur = conn.cursor()
    col_sql = "id INTEGER PRIMARY KEY, g INTEGER, " + ", ".join(f"{c} {t}" for c, t, _ in cols)
    plan = _plan(dialect, f"CREATE TABLE {table} ({col_sql});", f"SELECT g, " + ", ".join(f"MAX({c}) AS m_{c}" for c, _, _ in cols) + f" FROM {table} GROUP BY g")
    test.assertEqual(sorted(_rec_cols(plan)), sorted(("g", c) for c, _, _ in cols), dialect)
    if dialect == "mysql":
        # The recommended key parts, created with the table (one DDL statement).
        inline = "".join(f", INDEX {n} {c}" for n, c in _index_parts(plan.recommended_base_indexes))
        cur.execute(f"CREATE TABLE {table} ({col_sql}{inline})")
    else:
        cur.execute(f"CREATE TABLE {table} ({col_sql})")
        for ddl in plan.recommended_base_indexes:
            cur.execute(ddl)
    ph = ", ".join(["%s"] * (len(cols) + 2))
    cur.execute(f"INSERT INTO {table} VALUES ({ph})", [1, 1, *[v for _, _, v in cols]])
    cur.execute(f"DELETE FROM {table} WHERE id = 1")
    conn.commit()


def _index_parts(ddls: Sequence[str]) -> List[Tuple[str, str]]:
    """(name, column list) of ``CREATE INDEX name ON table (cols);`` statements."""
    out = []
    for ddl in ddls:
        m = re.match(r"\s*CREATE INDEX (\S+) ON \S+ (\(.*\))\s*;?\s*$", ddl)
        assert m, ddl
        out.append((m.group(1), m.group(2)))
    return out


def _apply_index_changes(eng: "_Engine", add: Sequence[str] = (), drop: Sequence[str] = ()) -> None:
    """Run recommended CREATE INDEX statements and drop indexes. MySQL/MariaDB:
    one ALTER TABLE (each DDL statement commits and costs ~70 ms)."""
    if not add and not drop:
        return
    if eng.name == "mysql":
        parts = [f"DROP INDEX {n}" for n in drop] + [f"ADD INDEX {n} {cols}" for n, cols in _index_parts(add)]
        eng.run(f"ALTER TABLE {eng.table} " + ", ".join(parts))
        return
    for n in drop:
        eng.run(f"DROP INDEX {n}")
    for ddl in add:
        eng.run(ddl)


# ---------------------------------------------------------------- native
class _Engine:
    """Driver differences: placeholder, row-read counter, index DDL."""

    def __init__(self, name: str, conn: Any, table: str) -> None:
        self.name, self.conn, self.table = name, conn, table
        self.steps = 0

    def ph(self, sql: str) -> str:
        return sql.replace("%s", "?") if self.name == "sqlite" else sql

    def run(self, sql: str, params: Sequence[Any] = ()) -> List[Tuple[Any, ...]]:
        cur = self.conn.cursor()
        cur.execute(self.ph(sql), tuple(params))
        return list(cur.fetchall()) if cur.description else []

    def rows_read(self, stmt: str, params: Sequence[Any]) -> int:
        """Rows (SQLite: VM steps) read by ``stmt`` and its triggers; rolled back."""
        self.conn.commit()
        if self.name == "postgres":
            q = "SELECT COALESCE(seq_tup_read, 0) + COALESCE(idx_tup_fetch, 0) FROM pg_stat_xact_user_tables WHERE relid = %s::regclass"
            # Tiny test tables make a sequential scan the cheapest plan even
            # for the DELETE's own primary-key lookup, which would hide the
            # recompute's access path in the counter.
            self.run("SET LOCAL enable_seqscan = off")
            before = self.run(q, (self.table,))[0][0]
            self.run(stmt, params)
            after = self.run(q, (self.table,))[0][0]
        elif self.name == "sqlite":
            counter = [0]

            def tick() -> int:
                counter[0] += 1
                return 0

            self.conn.set_progress_handler(tick, 1)
            self.run(stmt, params)
            self.conn.set_progress_handler(None, 1)
            before, after = 0, counter[0]
        else:
            q = "SHOW SESSION STATUS LIKE 'Handler_read%%'"
            before = sum(int(r[1]) for r in self.run(q))
            self.run(stmt, params)
            after = sum(int(r[1]) for r in self.run(q))
        self.conn.rollback()
        return int(after) - int(before)


def _canon(v: Any) -> Tuple[str, str]:
    return ("float", v.hex()) if isinstance(v, float) else (type(v).__name__, repr(v))


def _scenario(test: unittest.TestCase, eng: _Engine, register: Callable[[str], Any]) -> None:
    t = eng.table
    rows: List[Tuple[Any, ...]] = []
    rid = 0
    for i in range(GROUP_ROWS):
        rid += 1
        rows.append((rid, 1, i % 3, None if i % 37 == 0 else i // 2, i % 11))  # ties: every v twice
    for i in range(40):
        rid += 1
        rows.append((rid, None, 0, i % 5, i))
    for i in range(6):
        rid += 1
        rows.append((rid, 2, 1, 100 + i, None))
    cur = eng.conn.cursor()
    cur.executemany(eng.ph(f"INSERT INTO {t} (id, g, h, v, w) VALUES (%s, %s, %s, %s, %s)"), rows)
    eng.conn.commit()
    views = {
        # One view (each registration costs several DDL commits on MySQL):
        # a two-column group key, two MIN/MAX arguments, a NULL group key.
        "mm1": f"SELECT g, h, MIN(v) AS mn, MAX(v) AS mx, COUNT(v) AS c, COUNT(*) AS n, MAX(w) AS mw, MIN(w) AS lw FROM {t} GROUP BY g, h",
    }
    plans = {}
    for name, sql in views.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plans[name] = register(f"CREATE VIEW {t}_{name} AS {sql}")
    p1 = plans["mm1"]
    test.assertEqual(_rec_cols(p1), [("g", "h", "v"), ("g", "h", "w")], eng.name)
    test.assertTrue(any(n.startswith("minmax_extreme_recompute") for n in p1.diagnostics.details["performance_notes"]))
    recommended = sorted({d for p in plans.values() for d in p.recommended_base_indexes})
    _apply_index_changes(eng, add=recommended)
    if eng.name == "postgres":
        eng.run(f"ANALYZE {t}")
    eng.conn.commit()

    def compare(context: str) -> None:
        for name, sql in views.items():
            mv = sorted(tuple(_canon(x) for x in r) for r in eng.run(f"SELECT * FROM {t}_{name}"))
            live = sorted(tuple(_canon(x) for x in r) for r in eng.run(sql))
            test.assertEqual(mv, live, f"[{eng.name}] {name} after {context}")
        eng.conn.commit()

    def min_id(g: Optional[int]) -> int:
        cond = "g IS NULL" if g is None else f"g = {g}"
        return eng.run(f"SELECT id FROM {t} WHERE {cond} AND v IS NOT NULL ORDER BY v, id LIMIT 1")[0][0]

    def max_id(g: int) -> int:
        return eng.run(f"SELECT id FROM {t} WHERE g = {g} AND v IS NOT NULL ORDER BY v DESC, id LIMIT 1")[0][0]

    compare("initial population")
    # Cost class: an extreme delete reads O(1) rows with the (g, v) index.
    reads_gv = eng.rows_read(f"DELETE FROM {t} WHERE id = %s", (min_id(1),))
    steps: List[Tuple[str, str, Sequence[Any]]] = [
        ("delete one of two tied minima", f"DELETE FROM {t} WHERE id = %s", (min_id(1),)),
        ("delete the other tied minimum", f"DELETE FROM {t} WHERE id = %s", (min_id(1),)),
        ("delete the current maximum", f"DELETE FROM {t} WHERE id = %s", (max_id(1),)),
        ("raise the minimum's value", f"UPDATE {t} SET v = v + 500 WHERE id = %s", (min_id(1),)),
        ("move the minimum to another group", f"UPDATE {t} SET g = 2 WHERE id = %s", (min_id(1),)),
        ("set the maximum to NULL", f"UPDATE {t} SET v = NULL, w = NULL WHERE id = %s", (max_id(1),)),
        ("delete the NULL group's minimum", f"DELETE FROM {t} WHERE id = %s", (min_id(None),)),
        ("delete several extremes at once", f"DELETE FROM {t} WHERE g = 1 AND v < %s", (8,)),
        ("insert a new minimum", f"INSERT INTO {t} (id, g, h, v, w) VALUES (%s, 1, 2, -5, -5)", (10_000,)),
        ("empty a group", f"DELETE FROM {t} WHERE g = %s", (2,)),
        ("move a whole group into another", f"UPDATE {t} SET g = 1 WHERE g IS NULL AND v <= %s", (1,)),
    ]
    for label, sql, params in steps:
        eng.run(sql, params)
        eng.conn.commit()
        compare(label)
    eng.run(f"DELETE FROM {t} WHERE id = %s", (min_id(1),))
    eng.conn.rollback()
    compare("rolled-back extreme delete")
    # Sensitivity of the cost counter: with the group-key index only, the
    # same extreme delete reads the whole group.
    _apply_index_changes(eng, add=[f"CREATE INDEX {t}_gonly ON {t} (g);"], drop=[n for n, _ in _index_parts(recommended)])
    if eng.name == "postgres":
        eng.run(f"ANALYZE {t}")
    eng.conn.commit()
    reads_g = eng.rows_read(f"DELETE FROM {t} WHERE id = %s", (min_id(1),))
    group_size = eng.run(f"SELECT COUNT(*) FROM {t} WHERE g = 1")[0][0]
    eng.conn.commit()
    if eng.name == "sqlite":
        test.assertLess(reads_gv * 3, reads_g, f"sqlite VM steps: (g, v) {reads_gv} vs (g) {reads_g}")
    else:
        test.assertLessEqual(reads_gv, 12, f"[{eng.name}] rows read with (g, v) index")
        test.assertGreaterEqual(reads_g, group_size // 2, f"[{eng.name}] rows read with (g) index only")


class SqliteNativeTests(unittest.TestCase):
    def test_extreme_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(os.path.join(d, "db.sqlite"))
            try:
                conn.execute("CREATE TABLE mmt (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, w INTEGER)")
                conn.commit()
                mgr = MaterializedViewManager(conn)
                eng = _Engine("sqlite", conn, "mmt")
                _scenario(self, eng, mgr.register_view)
            finally:
                conn.close()


@unittest.skipUnless(os.environ.get("POSTGRES_DSN"), "POSTGRES_DSN not set")
class PostgresNativeTests(unittest.TestCase):
    def test_extreme_deletes(self) -> None:
        import psycopg2

        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        schema = f"l2mm_{uuid.uuid4().hex[:8]}"
        try:
            cur = conn.cursor()
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute("CREATE TABLE mmt (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, w INTEGER)")
            conn.commit()
            mgr = MaterializedViewManager(conn)
            eng = _Engine("postgres", conn, "mmt")
            _scenario(self, eng, mgr.register_view)
            _check_recommended_indexes_accept_max_values(self, "postgres", conn, "mmtypes")
        finally:
            conn.rollback()
            cur = conn.cursor()
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
            conn.close()


@unittest.skipUnless(os.environ.get("MYSQL_DSN"), "MYSQL_DSN not set")
class MysqlNativeTests(unittest.TestCase):
    def test_extreme_deletes(self) -> None:
        from urllib.parse import urlparse

        import pymysql

        p = urlparse(_mysql_private_dsn())
        conn = pymysql.connect(
            host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
            password=p.password or "", database=p.path.lstrip("/") or None, autocommit=False,
        )
        table = "mmt"
        try:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, w INTEGER)")
            conn.commit()
            # ignore_limitations acknowledges MYSQL_NO_TRUNCATE_TRIGGER only.
            mgr = MaterializedViewManager(conn, ignore_limitations=True)
            eng = _Engine("mysql", conn, table)
            _scenario(self, eng, mgr.register_view)
            conn.commit()
            _check_recommended_indexes_accept_max_values(self, "mysql", conn, f"{table}_types")
        finally:
            conn.rollback()
            conn.close()


class ConcurrentExtremeDeleteTests(unittest.TestCase):
    """Two independent sessions delete successive extremes of one group.

    Session A deletes the current minimum and holds the group's backing row.
    Session B has already taken its snapshot (a consistent read of the base
    table), then deletes the next minimum and waits for A's lock (positive
    engine lock-wait evidence). A commits; B's recompute must not use its
    pre-wait snapshot, in which A's row is still present (PostgreSQL READ
    COMMITTED: the recompute is a separate statement after the lock;
    MySQL/MariaDB REPEATABLE READ: a consistent read would be stale). Final
    materialized view == live view, exactly.
    """

    def test_successive_minimum_deletes(self) -> None:
        import threading

        from matview_sync import Dialect
        from test_avg_concurrent_maintenance import _connect, _native_engines, _session_id, _wait_until_lock_waiting

        engines = _native_engines()
        if not engines:
            self.skipTest("no native engine DSN set")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                if engine != Dialect.POSTGRES:
                    dsn = _mysql_private_dsn()
                self._run(engine, dsn, Dialect, threading, _connect, _session_id, _wait_until_lock_waiting)

    def _run(self, engine: Any, dsn: str, Dialect: Any, threading: Any, _connect: Any, _session_id: Any, _wait: Any) -> None:
        is_pg = engine == Dialect.POSTGRES
        table = f"l2cc_{uuid.uuid4().hex[:6]}"
        setup = _connect(engine, dsn)
        a = _connect(engine, dsn)
        b = _connect(engine, dsn)
        monitor = _connect(engine, dsn)
        mgr: Optional[MaterializedViewManager] = None
        try:
            cur = setup.cursor()
            if is_pg:
                cur.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
                cur.execute(f"CREATE INDEX {table}_gv ON {table} (g, v)")
            else:  # one DDL statement
                cur.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, INDEX {table}_gv (g, v))")
            cur.executemany(f"INSERT INTO {table} VALUES (%s, %s, %s)", [(i, 1, i) for i in range(1, 41)] + [(100, 2, 7)])
            setup.commit()
            mgr = MaterializedViewManager(setup, ignore_limitations=not is_pg)
            mgr.register_view(f"CREATE VIEW {table}_v AS SELECT g, MIN(v) AS mn, MAX(v) AS mx, COUNT(*) AS n FROM {table} GROUP BY g")
            setup.commit()
            b_sid = _session_id(engine, b)
            a_sid = _session_id(engine, a)
            a.cursor().execute(f"DELETE FROM {table} WHERE id = 1")
            bc = b.cursor()
            bc.execute(f"SELECT COUNT(*) FROM {table}")  # B's snapshot: row 1 still present
            bc.fetchall()
            done = threading.Event()
            errors: List[BaseException] = []

            def run_b() -> None:
                try:
                    bc.execute(f"DELETE FROM {table} WHERE id = 2")
                    b.commit()
                except BaseException as e:  # surfaced below
                    errors.append(e)
                finally:
                    done.set()

            t = threading.Thread(target=run_b)
            t.start()
            _wait(engine, monitor, b_sid, done, blocker_sid=a_sid)
            a.commit()
            t.join(30)
            self.assertFalse(t.is_alive())
            snapshot_isolation = False
            if not is_pg:
                mc = monitor.cursor()
                mc.execute("SHOW VARIABLES LIKE 'innodb_snapshot_isolation'")
                row = mc.fetchone()
                snapshot_isolation = bool(row) and str(row[1]).upper() in ("ON", "1")
                monitor.commit()
            if snapshot_isolation:
                # MariaDB 11.6+ (innodb_snapshot_isolation=ON, the default):
                # B wrote the group row A changed after B's snapshot, so InnoDB
                # rejects B (ER_CHECKREAD 1020, retry the transaction) instead
                # of letting it act on a stale snapshot; nothing of B remains.
                self.assertEqual([getattr(e, "args", (None,))[0] for e in errors], [1020])
                b.rollback()
                expected_first = (1, 2, 40, 39)
            else:
                self.assertEqual(errors, [])
                expected_first = (1, 3, 40, 38)
            cur = setup.cursor()
            cur.execute(f"SELECT g, mn, mx, n FROM {table}_v ORDER BY g")
            mv = cur.fetchall()
            cur.execute(f"SELECT g, MIN(v), MAX(v), COUNT(*) FROM {table} GROUP BY g ORDER BY g")
            live = cur.fetchall()
            setup.commit()
            self.assertEqual([tuple(r) for r in mv], [tuple(r) for r in live])
            self.assertEqual(tuple(mv[0]), expected_first)
        finally:
            for c in (a, b, monitor):
                try:
                    c.rollback()
                    c.close()
                except Exception:
                    pass
            setup.rollback()
            if is_pg:  # MySQL/MariaDB: the module's private database is dropped once
                if mgr is not None:
                    try:
                        mgr.drop_view(f"{table}_v")
                    except Exception:
                        pass
                setup.cursor().execute(f"DROP TABLE IF EXISTS {table}")
                setup.commit()
            setup.close()


# MySQL/MariaDB: one private database for the whole module (each DDL commit
# costs ~70 ms; teardown is a single DROP DATABASE).
_MYSQL_PRIVATE_DB: Optional[str] = None


def _mysql_admin() -> Any:
    from urllib.parse import urlparse

    import pymysql

    p = urlparse(os.environ["MYSQL_DSN"])
    return pymysql.connect(
        host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
        password=p.password or "", autocommit=True,
    )


def _mysql_private_dsn() -> str:
    global _MYSQL_PRIVATE_DB
    from urllib.parse import urlparse, urlunparse

    if _MYSQL_PRIVATE_DB is None:
        _MYSQL_PRIVATE_DB = f"l2mm_{uuid.uuid4().hex[:8]}"
        admin = _mysql_admin()
        try:
            admin.cursor().execute(f"CREATE DATABASE {_MYSQL_PRIVATE_DB}")
        finally:
            admin.close()
    return urlunparse(urlparse(os.environ["MYSQL_DSN"])._replace(path=f"/{_MYSQL_PRIVATE_DB}"))


def tearDownModule() -> None:
    if _MYSQL_PRIVATE_DB is not None:
        admin = _mysql_admin()
        try:
            admin.cursor().execute(f"DROP DATABASE IF EXISTS {_MYSQL_PRIVATE_DB}")
        finally:
            admin.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
