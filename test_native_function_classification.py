#!/usr/bin/env python3
"""Function names the SQL text cannot classify: engine-catalog classification.

Bug (pre-existing): ``SELECT my_agg(x) FROM t`` with a user-defined aggregate
parsed as a per-row scalar call and was maintained as a projection (SQLite:
one row per base row instead of one row; PostgreSQL: ``my_agg(g)`` next to
``COUNT(*)`` in a grouped view failed or was folded as a group key). Unknown
scalar functions were stored as TEXT (PostgreSQL returned '8' for 8).

Contract (matview.native_functions): at registration every call that sqlglot
does not model and that is not a known built-in aggregate is classified
through the engine catalog. Aggregates route like other aggregates without an
algebraic rule (full refresh, or group-scoped recompute with a total
aggregate-level ORDER BY); scalar functions keep their maintenance and get
their native result type (SQLite: no affinity). A name the engine cannot
classify is refused with a message that names it -- never guessed.

Oracle: the original SELECT on the same native state, exact multiset of
(type, repr) values plus the column type codes (PostgreSQL OIDs, MySQL
protocol codes), after population and after insert / delete / group move /
rolled-back DML.
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

from matview_sync import (
    EngineLimitationWarning,
    MaintenanceStrategy,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
)

T1 = MaintenanceStrategy.ALGEBRAIC_DELTA
T2 = MaintenanceStrategy.GROUP_SCOPED_DELTA
T3 = MaintenanceStrategy.ATOMIC_QUERY_SYNC


def _canon(v: Any) -> Tuple[str, str]:
    return ("float", v.hex()) if isinstance(v, float) else (type(v).__name__, repr(v))


def _snap(cur: Any, sql: str) -> Tuple[List[Any], List[Any]]:
    cur.execute(sql)
    types = [d[1] for d in cur.description]
    return sorted(tuple(_canon(x) for x in r) for r in cur.fetchall()), types


def _drive(
    test: unittest.TestCase,
    engine: str,
    conn: Any,
    register: Callable[[str], Any],
    views: Sequence[Tuple[str, str, MaintenanceStrategy]],
    ph: str = "%s",
) -> None:
    cur = conn.cursor()
    rows = [(1, 1, 1), (2, 1, 2), (3, 2, 5), (4, None, 3), (5, 2, None)]
    cur.executemany(f"INSERT INTO t (id, g, v) VALUES ({ph}, {ph}, {ph})", rows)
    conn.commit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = register(";\n".join(f"CREATE VIEW {n} AS {sql}" for n, sql, _ in views))
    for (name, _sql, expected), sub in zip(views, list(plan.sub_plans) or [plan]):
        test.assertEqual(sub.strategy, expected, f"[{engine}] {name}")
        if expected != T1:  # the routing fallback is announced (matview.diagnostics)
            test.assertTrue(
                any(d.message.startswith("exact result required: falling back to strategy") for d in sub.strategy_diagnostics),
                f"[{engine}] {name}: {sub.strategy_diagnostics!r}",
            )
    conn.commit()

    def compare(context: str) -> None:
        c = conn.cursor()
        for name, sql, _ in views:
            mv, mv_t = _snap(c, f"SELECT * FROM {name}")
            live, live_t = _snap(c, sql)
            test.assertEqual(mv, live, f"[{engine}] {name} after {context}")
            if engine != "sqlite":
                test.assertEqual(mv_t, live_t, f"[{engine}] {name} result types after {context}")
        conn.commit()

    compare("initial population")
    steps = [
        (f"INSERT INTO t (id, g, v) VALUES ({ph}, {ph}, {ph})", (6, 1, 10)),
        (f"DELETE FROM t WHERE id = {ph}", (3,)),
        (f"UPDATE t SET g = {ph} WHERE id = 1", (2,)),
        (f"UPDATE t SET v = NULL WHERE g IS NULL AND id = {ph}", (4,)),
    ]
    for sql, params in steps:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        compare(sql)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM t WHERE id = {ph}", (2,))
    conn.rollback()
    compare("rollback")


class SqliteTests(unittest.TestCase):
    def _conn(self, d: str) -> sqlite3.Connection:
        conn = sqlite3.connect(os.path.join(d, "db.sqlite"))

        class PySum:
            def __init__(self) -> None:
                self.total = 0

            def step(self, value: Any) -> None:
                self.total += value or 0

            def finalize(self) -> int:
                return self.total

        conn.create_aggregate("py_sum", 1, PySum)  # pyright: ignore[reportArgumentType]  # sqlite3 aggregate protocol
        conn.create_function("py_twice", 1, lambda x: None if x is None else 2 * x)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
        conn.commit()
        return conn

    def test_catalog_classified_aggregates_and_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            conn = self._conn(d)
            try:
                mgr = MaterializedViewManager(conn)
                _drive(
                    self, "sqlite", conn, mgr.register_view,
                    [
                        ("v_ungrouped", "SELECT py_sum(v) AS a FROM t", T3),
                        ("v_key_arg", "SELECT g, COUNT(*) AS n, py_sum(g) AS a FROM t GROUP BY g", T3),
                        ("v_scalar", "SELECT id, py_twice(v) AS a FROM t", T1),
                    ],
                    ph="?",
                )
                # The scalar result keeps its storage class (INTEGER, not TEXT).
                self.assertEqual(conn.execute("SELECT typeof(a) FROM v_scalar WHERE a IS NOT NULL LIMIT 1").fetchone(), ("integer",))
            finally:
                conn.close()

    def test_unclassifiable_name_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            conn = self._conn(d)
            try:
                mgr = MaterializedViewManager(conn)
                with self.assertRaisesRegex(UnsupportedMaterializedViewError, r"cannot tell whether NOPE is an aggregate"):
                    mgr.register_view("CREATE VIEW v_nope AS SELECT g, nope(v) AS a FROM t GROUP BY g")
                self.assertEqual(conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE '%v_nope%'").fetchone(), (0,))
            finally:
                conn.close()


class _NoFunctionListCursor(sqlite3.Cursor):
    def execute(self, sql: str, *args: Any) -> "sqlite3.Cursor":  # pyright: ignore[reportIncompatibleMethodOverride]  # test double
        if "function_list" in sql.lower():
            raise sqlite3.OperationalError("unknown pragma (simulated driver without PRAGMA function_list)")
        return super().execute(sql, *args)


class _NoFunctionListConnection(sqlite3.Connection):
    """A SQLite connection whose driver/build offers no PRAGMA function_list."""

    def cursor(self, factory: Any = None) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double
        return super().cursor(factory or _NoFunctionListCursor)


class SqliteWithoutFunctionListTests(unittest.TestCase):
    """Without a catalog to ask, classification falls back to the SQL text
    (the previous behaviour, as offline compilation does) and is announced
    with FUNCTION_CLASSIFICATION_FROM_TEXT instead of refusing the view."""

    def test_falls_back_to_text_classification_and_announces_it(self) -> None:
        from matview.diagnostics import MatviewDiagnosticWarning

        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(os.path.join(d, "db.sqlite"), factory=_NoFunctionListConnection)
            try:
                conn.create_function("py_twice", 1, lambda x: None if x is None else 2 * x)
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
                conn.execute("INSERT INTO t VALUES (1, 1, 1), (2, 1, 2), (3, 2, NULL)")
                conn.commit()
                mgr = MaterializedViewManager(conn)
                sql = "SELECT id, py_twice(v) AS a FROM t"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    plan = mgr.register_view(f"CREATE VIEW v_scalar AS {sql}")
                codes = [d.code for d in plan.strategy_diagnostics]
                self.assertIn("FUNCTION_CLASSIFICATION_FROM_TEXT", codes)
                warned = [w.message.diagnostic.code for w in caught if isinstance(w.message, MatviewDiagnosticWarning)]
                self.assertIn("FUNCTION_CLASSIFICATION_FROM_TEXT", warned)
                self.assertEqual(plan.strategy, T1)
                conn.execute("UPDATE t SET v = 5 WHERE id = 3")
                conn.commit()
                self.assertEqual(sorted(conn.execute("SELECT * FROM v_scalar").fetchall()), sorted(conn.execute(sql).fetchall()))
            finally:
                conn.close()


@unittest.skipUnless(os.environ.get("POSTGRES_DSN"), "POSTGRES_DSN not set")
class PostgresTests(unittest.TestCase):
    def setUp(self) -> None:
        import psycopg2

        self.conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        self.schema = f"l3udf_{uuid.uuid4().hex[:8]}"
        cur = self.conn.cursor()
        cur.execute(f'CREATE SCHEMA "{self.schema}"')
        cur.execute(f'SET search_path TO "{self.schema}"')
        cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
        cur.execute("CREATE AGGREGATE isum(integer) (SFUNC = int4pl, STYPE = integer)")
        cur.execute("CREATE FUNCTION iadd(integer, integer) RETURNS integer LANGUAGE sql IMMUTABLE AS 'SELECT $1 + $2'")
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        self.conn.commit()
        self.conn.close()

    def test_catalog_classified_aggregates_and_scalars(self) -> None:
        mgr = MaterializedViewManager(self.conn)
        _drive(
            self, "postgres", self.conn, mgr.register_view,
            [
                ("v_key_arg", "SELECT g, COUNT(*) AS n, isum(g) AS a FROM t GROUP BY g", T3),
                ("v_ordered", "SELECT g, isum(v ORDER BY id) AS a FROM t GROUP BY g", T2),
                ("v_ungrouped", "SELECT isum(v) AS a FROM t", T3),
                ("v_scalar", "SELECT id, iadd(v, id) AS a FROM t", T1),
            ],
        )

    def test_ambiguous_overload_is_refused(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE FUNCTION isum(bigint) RETURNS bigint LANGUAGE sql IMMUTABLE AS 'SELECT $1'")
        self.conn.commit()
        mgr = MaterializedViewManager(self.conn)
        with self.assertRaisesRegex(UnsupportedMaterializedViewError, r"ISUM/1: overloads of that arity are both aggregates and functions"):
            mgr.register_view("CREATE VIEW v_amb AS SELECT isum(v) AS a FROM t")
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM pg_class WHERE relname LIKE %s AND relnamespace = %s::regnamespace", ("%v_amb%", self.schema))
        self.assertEqual(cur.fetchone()[0], 0)


@unittest.skipUnless(os.environ.get("MYSQL_DSN"), "MYSQL_DSN not set")
class MysqlTests(unittest.TestCase):
    def setUp(self) -> None:
        from urllib.parse import urlparse

        import pymysql

        p = urlparse(os.environ["MYSQL_DSN"])
        self.conn = pymysql.connect(
            host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
            password=p.password or "", database=p.path.lstrip("/") or None, autocommit=False,
        )
        cur = self.conn.cursor()
        cur.execute("SELECT VERSION()")
        self.mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
        self.database = f"l3udf_{uuid.uuid4().hex[:8]}"
        cur.execute(f"CREATE DATABASE {self.database}")
        cur.execute(f"USE {self.database}")
        cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
        cur.execute("CREATE FUNCTION iadd(a INT, b INT) RETURNS INT DETERMINISTIC RETURN a + b")
        if self.mariadb:
            cur.execute(
                "CREATE AGGREGATE FUNCTION isum(x INT) RETURNS INT BEGIN "
                "DECLARE s INT DEFAULT 0; DECLARE CONTINUE HANDLER FOR NOT FOUND RETURN s; "
                "LOOP FETCH GROUP NEXT ROW; SET s = s + IFNULL(x, 0); END LOOP; END"
            )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS {self.database}")
        self.conn.commit()
        self.conn.close()

    def test_catalog_classified_functions(self) -> None:
        views = [("v_scalar", "SELECT id, iadd(v, id) AS a FROM t", T1)]
        if self.mariadb:  # MariaDB stored aggregate functions (CREATE AGGREGATE FUNCTION)
            views += [
                ("v_ungrouped", "SELECT isum(v) AS a FROM t", T3),
                ("v_key_arg", "SELECT g, COUNT(*) AS n, isum(g) AS a FROM t GROUP BY g", T3),
            ]
        # ignore_limitations acknowledges MYSQL_NO_TRUNCATE_TRIGGER only.
        mgr = MaterializedViewManager(self.conn, ignore_limitations=True)
        _drive(self, "mariadb" if self.mariadb else "mysql", self.conn, mgr.register_view, views)


if __name__ == "__main__":
    unittest.main(verbosity=2)
