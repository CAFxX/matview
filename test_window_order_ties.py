#!/usr/bin/env python3
"""Window functions over tied rows: exact routing.

A window ORDER BY that leaves rows tied (not a total order of the
partition's rows: the order keys and partition keys do not cover the primary
key) lets the engine number or pick tied rows in plan order. A
partition-scoped recompute reads the partition through another plan, so for
ROW_NUMBER, LAG/LEAD, FIRST_VALUE/LAST_VALUE/NTH_VALUE, NTILE, ROWS frames
and order-sensitive aggregates over such a window the only exact strategy is
the full refresh of the original query (announced; the per-code opt-in
AGGREGATE_ORDER_NOT_REPRODUCIBLE keeps the scoped recompute). RANK,
DENSE_RANK, PERCENT_RANK, CUME_DIST and order-insensitive aggregates over the
default frame give peers the same result and keep the partition-scoped
recompute; so does any window whose ORDER BY is total.

Native differential (SQLite in-process; PostgreSQL/MySQL/MariaDB with
POSTGRES_DSN / MYSQL_DSN): the oracle is the original SELECT, exact multiset
of (type, value) rows, after population and after DML that creates, moves
and removes ties (before this routing, LAG/ROW_NUMBER over ties mismatched
from the first DML on all three server engines).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
import uuid
import warnings
from typing import Any, Callable, List, Sequence, Tuple

from matview_sync import EngineLimitationError, MaintenanceStrategy, MaterializedViewManager, generate_materialized_view, parse_table_schema

T2 = MaintenanceStrategy.GROUP_SCOPED_DELTA
T3 = MaintenanceStrategy.ATOMIC_QUERY_SYNC

# (name, select, expected strategy). One view per strategy class keeps the
# number of registrations (DDL commits) low.
VIEWS: List[Tuple[str, str, MaintenanceStrategy]] = [
    (
        "w_ties",
        "SELECT id, g, s, LAG(x) OVER (PARTITION BY g ORDER BY s) AS lg, "
        "ROW_NUMBER() OVER (PARTITION BY g ORDER BY s) AS rn, "
        "FIRST_VALUE(x) OVER (PARTITION BY g ORDER BY s) AS fv, "
        "SUM(x) OVER (PARTITION BY g ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rs "
        "FROM t",
        T3,
    ),
    (
        "w_peers",
        "SELECT id, g, RANK() OVER (PARTITION BY g ORDER BY s) AS rk, "
        "DENSE_RANK() OVER (PARTITION BY g ORDER BY s) AS drk, "
        "SUM(x) OVER (PARTITION BY g ORDER BY s) AS rsum, "
        "ROW_NUMBER() OVER (PARTITION BY g ORDER BY s, id) AS rn_total, "
        "LAG(x) OVER (PARTITION BY g ORDER BY s, id) AS lg_total "
        "FROM t",
        T2,
    ),
]


def _canon(v: Any) -> Tuple[str, str]:
    return ("float", v.hex()) if isinstance(v, float) else (type(v).__name__, repr(v))


def _drive(
    test: unittest.TestCase, engine: str, conn: Any, register: Callable[..., Any], ph: str = "%s"
) -> None:
    cur = conn.cursor()
    # Many ties on s within each partition g.
    rows = [(i, i % 3, i * 7 % 11, "ab"[i % 2]) for i in range(1, 19)]
    cur.executemany(f"INSERT INTO t (id, g, x, s) VALUES ({ph}, {ph}, {ph}, {ph})", rows)
    conn.commit()
    ddl = ";\n".join(f"CREATE VIEW {n} AS {q}" for n, q, _ in VIEWS)
    if engine == "mariadb":
        # MariaDB numbers tied peers differently in the refresh's
        # INSERT ... SELECT and in a later SELECT: no strategy is exact, so
        # the default is a refusal naming the per-code opt-in...
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with test.assertRaisesRegex(EngineLimitationError, "AGGREGATE_ORDER_NOT_REPRODUCIBLE"):
                register(f"CREATE VIEW {VIEWS[0][0]} AS {VIEWS[0][1]}")
        conn.rollback()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ... which accepts a best-effort full refresh (relaxed, announced).
        accept = ["AGGREGATE_ORDER_NOT_REPRODUCIBLE"] if engine == "mariadb" else []
        plan = register(ddl, accept_limitations=accept)
    for (name, _q, expected), sub in zip(VIEWS, list(plan.sub_plans) or [plan]):
        test.assertEqual(sub.strategy, expected, f"[{engine}] {name}")
        if expected == T3:
            code = "ORDER_SENSITIVE_RELAXED_FULL_REFRESH" if engine == "mariadb" else "ORDER_SENSITIVE_AGGREGATE_FULL_REFRESH"
            test.assertEqual([d.code for d in sub.strategy_diagnostics], [code], f"[{engine}] {name}")
            if engine == "mariadb":
                test.assertTrue(sub.strategy_diagnostics[0].message.startswith("relaxed result allowed: using strategy full refresh"))
    conn.commit()

    def compare(context: str) -> None:
        c = conn.cursor()
        for name, q, _ in VIEWS:
            if name == "w_ties" and engine == "mariadb":
                # Named exclusion for the OPTED-IN case only
                # (AGGREGATE_ORDER_NOT_REPRODUCIBLE, best effort): which peer
                # gets which number is implementation-dependent; compare what
                # SQL defines, the numbers each peer group gets.
                c.execute(f"SELECT g, s, rn FROM {name}")
                mv_rn = sorted(tuple(_canon(x) for x in r) for r in c.fetchall())
                c.execute(f"SELECT g, s, rn FROM ({q}) AS live_q")
                live_rn = sorted(tuple(_canon(x) for x in r) for r in c.fetchall())
                test.assertEqual(mv_rn, live_rn, f"[{engine}] {name} peer-group numbers after {context}")
                continue
            c.execute(f"SELECT * FROM {name}")
            mv = sorted(tuple(_canon(x) for x in r) for r in c.fetchall())
            c.execute(q)
            live = sorted(tuple(_canon(x) for x in r) for r in c.fetchall())
            test.assertEqual(mv, live, f"[{engine}] {name} after {context}")
        conn.commit()

    compare("initial population")
    steps: Sequence[Tuple[str, Sequence[Any]]] = [
        (f"INSERT INTO t (id, g, x, s) VALUES ({ph}, {ph}, {ph}, {ph})", (100, 1, 50, "a")),  # a new tie
        (f"UPDATE t SET s = {ph} WHERE id = {ph}", ("a", 4)),  # joins a tie group
        (f"UPDATE t SET g = {ph} WHERE id = {ph}", (2, 5)),  # moves partition
        (f"DELETE FROM t WHERE id = {ph}", (1,)),
        (f"UPDATE t SET x = x + 1 WHERE g = {ph}", (0,)),
    ]
    for sql, params in steps:
        c = conn.cursor()
        c.execute(sql, tuple(params))
        conn.commit()
        compare(sql)


class SqliteTests(unittest.TestCase):
    def test_window_ties(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(os.path.join(d, "db.sqlite"))
            try:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, s TEXT)")
                conn.commit()
                _drive(self, "sqlite", conn, MaterializedViewManager(conn).register_view, ph="?")
            finally:
                conn.close()


@unittest.skipUnless(os.environ.get("POSTGRES_DSN"), "POSTGRES_DSN not set")
class PostgresTests(unittest.TestCase):
    def test_window_ties(self) -> None:
        import psycopg2

        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        schema = f"l1w_{uuid.uuid4().hex[:8]}"
        try:
            cur = conn.cursor()
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, s VARCHAR(4))")
            conn.commit()
            _drive(self, "postgres", conn, MaterializedViewManager(conn).register_view)
        finally:
            conn.rollback()
            cur = conn.cursor()
            cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
            conn.close()


@unittest.skipUnless(os.environ.get("MYSQL_DSN"), "MYSQL_DSN not set")
class MysqlTests(unittest.TestCase):
    def test_window_ties(self) -> None:
        from urllib.parse import urlparse

        import pymysql

        p = urlparse(os.environ["MYSQL_DSN"])
        conn = pymysql.connect(host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
                               password=p.password or "", autocommit=False)
        db = f"l1w_{uuid.uuid4().hex[:8]}"  # private: teardown is one DROP DATABASE
        try:
            cur = conn.cursor()
            cur.execute("SELECT VERSION()")
            engine = "mariadb" if "mariadb" in str(cur.fetchone()[0]).lower() else "mysql"
            cur.execute(f"CREATE DATABASE {db}")
            cur.execute(f"USE {db}")
            cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, s VARCHAR(4))")
            conn.commit()
            # ignore_limitations acknowledges MYSQL_NO_TRUNCATE_TRIGGER only.
            _drive(self, engine, conn, MaterializedViewManager(conn, ignore_limitations=True).register_view)
        finally:
            conn.rollback()
            conn.cursor().execute(f"DROP DATABASE IF EXISTS {db}")
            conn.commit()
            conn.close()


class OfflineTests(unittest.TestCase):
    def test_relaxed_opt_in_and_peer_functions(self) -> None:
        schema = parse_table_schema("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER, s VARCHAR(4));", dialect="postgres")
        q = "CREATE VIEW mv AS SELECT id, g, ROW_NUMBER() OVER (PARTITION BY g ORDER BY s) AS rn FROM t"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relaxed = generate_materialized_view(schema, q, dialect="postgres", accept_limitations=["AGGREGATE_ORDER_NOT_REPRODUCIBLE"])
        self.assertEqual(relaxed.strategy, T2)
        self.assertEqual([d.code for d in relaxed.strategy_diagnostics], ["ORDER_SENSITIVE_RELAXED_GROUP_RECOMPUTE"])
        for expr, expected in (
            ("NTILE(2) OVER (PARTITION BY g ORDER BY s)", T3),
            ("PERCENT_RANK() OVER (PARTITION BY g ORDER BY s)", T2),
            ("ROW_NUMBER() OVER (PARTITION BY id ORDER BY s)", T2),  # partition key is the primary key
        ):
            plan = generate_materialized_view(schema, f"CREATE VIEW mv AS SELECT id, g, {expr} AS a FROM t", dialect="postgres")
            self.assertEqual(plan.strategy, expected, expr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
