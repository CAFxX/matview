#!/usr/bin/env python3
"""Aggregates without an algebraic rule, and grouping sets: exact routing.

Contract (AGENTS.md, "Strategy ladder"): an aggregate the algebraic compiler
does not decompose (JSON_AGG, JSONB_AGG, JSON_OBJECT_AGG, percentile_cont /
percentile_disc, mode(), hypothetical-set rank(), ANY_VALUE, regr_* / corr /
covar_*, XMLAGG, range_agg, JSON_ARRAYAGG / JSON_OBJECTAGG, user-defined
aggregates, ...) is maintained by re-running the view's ORIGINAL query:

* group-scoped recompute (GROUP_SCOPED_DELTA) when the view has a plain GROUP
  BY and the aggregate's result is a function of the group's multiset of rows,
  or the aggregate is fed its rows in a total order (an aggregate-level ORDER
  BY covering the primary key of every base table, not DISTINCT);
* full refresh (ATOMIC_QUERY_SYNC) for order-sensitive aggregates without such
  an ORDER BY (array_agg / json_agg / string_agg / GROUP_CONCAT / ANY_VALUE /
  float accumulators such as regr_slope), for opaque user-defined aggregates,
  for ungrouped views, and for GROUP BY ROLLUP / CUBE / GROUPING SETS (and
  MySQL WITH ROLLUP).

Each fallback is announced with the lead phrase ``exact result required:
falling back to strategy <X>``. A result type that cannot be determined on
PostgreSQL/MySQL is refused (with the CAST remedy) instead of being guessed.

Evidence: native differentials (set ``POSTGRES_DSN`` and/or ``MYSQL_DSN``;
SQLite always runs in-process). The oracle is the view's original SELECT on the
same database state, compared as an exact multiset: values by type and
``repr`` (floats bit for bit through ``float.hex``; JSON/JSONB as the server's
text, not decoded), and the per-column result type (PostgreSQL type OIDs,
MySQL/MariaDB protocol type codes). After initial population and after every
step of a seeded random DML sequence: single and multi-row INSERT, UPDATE of the
value, of the group (group move) and to NULL, DELETE of single rows and of whole
groups (empty groups), and rolled-back transactions. All views of one engine
share one fixture and one DML sequence (runtime budget: well under 3 s per
engine job).

Known exclusions (recorded in TODO.md): MySQL reports GROUP_CONCAT as
LONG_BLOB (251) while any TEXT backing column reports BLOB (252) -- the value
and its Python type are compared, the protocol code for GROUP_CONCAT columns is
not; values that compare equal under a nondeterministic collation are not
covered for percentile_disc/mode.
"""

from __future__ import annotations

import os
import random
import re
import sqlite3
import tempfile
import unittest
import uuid
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from matview_sync import (
    Dialect,
    EngineLimitationWarning,
    MaintenanceStrategy,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
    generate_materialized_view,
    parse_table_schema,
)
from matview.diagnostics import EXACT_LEAD, MatviewDiagnosticWarning, Strategy

T2 = MaintenanceStrategy.GROUP_SCOPED_DELTA
T3 = MaintenanceStrategy.ATOMIC_QUERY_SYNC
# Shared strategy vocabulary (matview.diagnostics) for each maintenance strategy.
ANNOUNCED = {T2: Strategy.GROUP_SCOPED_RECOMPUTE, T3: Strategy.FULL_REFRESH}
SEED = int(os.environ.get("MATVIEW_UNRECOGNIZED_AGG_SEED", "20261120"))
DML_STEPS = 12


class Case:
    def __init__(
        self,
        name: str,
        select_sql: str,
        expected: MaintenanceStrategy,
        *,
        min_version: Tuple[int, ...] = (),
        engines: Tuple[str, ...] = (),
        skip_type_code: Tuple[str, ...] = (),
        grouped_expr: Optional[str] = None,
        standalone: bool = False,
    ) -> None:
        self.standalone = standalone  # never merged into a wide view
        self.grouped_expr = grouped_expr  # set for "SELECT g, <expr> AS a ... GROUP BY g" cases
        self.name = name
        self.select_sql = select_sql
        self.expected = expected
        self.min_version = min_version
        self.engines = engines  # subset of ("mysql", "mariadb"); empty = all
        self.skip_type_code = skip_type_code


def _grouped(name: str, expr: str, expected: MaintenanceStrategy, **kw: Any) -> Case:
    return Case(name, f"SELECT g, {expr} AS a FROM {{t}} GROUP BY g", expected, grouped_expr=expr, **kw)


PG_CASES: List[Case] = [
    _grouped("json_agg_ordered", "JSON_AGG(v ORDER BY id)", T2),
    _grouped("json_agg_unordered", "JSON_AGG(v)", T3),
    _grouped("jsonb_agg_ordered", "JSONB_AGG(v ORDER BY v DESC, id)", T2),
    _grouped("jsonb_agg_coalesce", "COALESCE(JSONB_AGG(v ORDER BY id), '[]')", T2),
    _grouped("json_agg_case", "CASE WHEN COUNT(*) > 1 THEN JSON_AGG(v ORDER BY id) END", T2),
    _grouped("json_object_agg_ordered", "JSON_OBJECT_AGG(id, v ORDER BY id)", T2),
    _grouped("jsonb_object_agg_dup_keys", "JSONB_OBJECT_AGG(s, v)", T3),
    _grouped("percentile_cont", "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v)", T2),
    _grouped("percentile_cont_float", "PERCENTILE_CONT(0.3) WITHIN GROUP (ORDER BY x)", T2),
    _grouped("percentile_disc_array", "PERCENTILE_DISC(ARRAY[0.25, 0.75]) WITHIN GROUP (ORDER BY v)", T2),
    _grouped("mode", "MODE() WITHIN GROUP (ORDER BY v)", T2),
    # Polymorphic aggregates drop the argument's type modifier natively.
    _grouped("percentile_disc_numeric", "PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY d)", T2),
    _grouped("array_agg_numeric_ordered", "ARRAY_AGG(d ORDER BY id)", T2),
    _grouped("hypothetical_rank", "RANK(3) WITHIN GROUP (ORDER BY v)", T2),
    _grouped("any_value", "ANY_VALUE(v)", T3, min_version=(16,)),
    _grouped("regr_slope", "REGR_SLOPE(x, v)", T3),
    _grouped("corr_ordered", "CORR(x, v ORDER BY id)", T2),
    _grouped("regr_count", "REGR_COUNT(x, v)", T2),
    _grouped("covar_pop", "COVAR_POP(v, id)", T3),
    _grouped("array_agg_ordered", "ARRAY_AGG(v ORDER BY v, id)", T2),
    _grouped("array_agg_unordered", "ARRAY_AGG(v)", T2),  # recognized: keeps the group recompute
    _grouped("string_agg_ordered", "STRING_AGG(s, ',' ORDER BY id)", T2),
    _grouped("string_agg_unordered", "STRING_AGG(s, ',')", T2),  # recognized
    _grouped("xmlagg_ordered", "XMLAGG(XMLELEMENT(NAME e, v) ORDER BY id)", T2),
    _grouped("range_agg", "RANGE_AGG(INT4RANGE(v, v + 2))", T2),
    _grouped("json_arrayagg", "JSON_ARRAYAGG(v ORDER BY id)", T2, min_version=(16,)),
    _grouped("udf_cast", "CAST({udf}(v) AS INTEGER)", T3),
    Case(
        "having_percentile",
        "SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY v) > 3",
        T2,
    ),
    Case("ungrouped_percentile", "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v) AS a FROM {t}", T3),
    Case(
        "join_total_order",
        "SELECT t.g, JSONB_AGG(u.w ORDER BY t.id, u.id) AS a FROM {t} t JOIN {u} u ON u.tid = t.id GROUP BY t.g",
        T2,
    ),
    Case(
        "join_partial_order",
        "SELECT t.g, JSONB_AGG(u.w ORDER BY t.id) AS a FROM {t} t JOIN {u} u ON u.tid = t.id GROUP BY t.g",
        T3,
    ),
    Case(
        "rollup",
        "SELECT g, h, SUM(v) AS sv, COUNT(*) AS n, GROUPING(g, h) AS gr FROM {t} GROUP BY ROLLUP(g, h)",
        T3,
    ),
    Case("cube", "SELECT g, h, COUNT(v) AS n FROM {t} GROUP BY CUBE(g, h)", T3),
    Case("grouping_sets", "SELECT g, h, MAX(v) AS m FROM {t} GROUP BY GROUPING SETS ((g), (h), ())", T3),
]

MY_CASES: List[Case] = [
    _grouped("json_arrayagg", "JSON_ARRAYAGG(v)", T3),
    _grouped("json_objectagg", "JSON_OBJECTAGG(id, v)", T3),
    # MySQL's protocol type for ANY_VALUE(int) is plan-dependent (native 8.4:
    # BIGINT alone, INT next to JSON_ARRAYAGG or ungrouped), so it is
    # compared in its own view, where the live view's type is BIGINT.
    _grouped("any_value", "ANY_VALUE(v)", T3, engines=("mysql",), standalone=True),
    _grouped("group_concat_ordered", "GROUP_CONCAT(v ORDER BY id)", T2, skip_type_code=("a",)),
    # (Recognized GROUP_CONCAT without a total ORDER BY keeps the group
    # recompute, whose concatenation order can differ natively: TODO.md.)
    _grouped("group_concat_distinct", "GROUP_CONCAT(DISTINCT v ORDER BY v)", T2, skip_type_code=("a",)),  # recognized
    Case("with_rollup", "SELECT g, SUM(v) AS sv, COUNT(*) AS n FROM {t} GROUP BY g WITH ROLLUP", T3, engines=("mariadb",)),
    Case(
        "with_rollup_grouping",
        "SELECT g, SUM(v) AS sv, COUNT(*) AS n, GROUPING(g) AS gr FROM {t} GROUP BY g WITH ROLLUP",
        T3,
        engines=("mysql",),  # MariaDB has no GROUPING()
    ),
    Case("ungrouped_json_arrayagg", "SELECT JSON_ARRAYAGG(v) AS a FROM {t}", T3),
]

SQLITE_ORDERED_AGG = sqlite3.sqlite_version_info >= (3, 44, 0)
SQLITE_CASES: List[Case] = [
    _grouped("json_group_array", "json_group_array(v)", T2),  # recognized
    _grouped("json_group_object", "json_group_object(id, v)", T2),  # recognized
    _grouped("group_concat_unordered", "group_concat(v, ',')", T2),  # recognized
    _grouped("udf_aggregate", "py_sum(v)", T3),
    Case("ungrouped_json_group_array", "SELECT json_group_array(v) AS a FROM {t}", T3),
] + (
    [
        _grouped("group_concat_ordered", "group_concat(v, ',' ORDER BY id)", T2),
        _grouped("json_group_array_ordered", "json_group_array(v ORDER BY id)", T2),
    ]
    if SQLITE_ORDERED_AGG
    else []
)


# ----------------------------------------------------------------- oracle
def _canon(v: Any) -> Tuple[str, str]:
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(v).hex())
    return (type(v).__name__, repr(v))


def _snapshot(cur: Any, sql: str) -> Tuple[List[Any], List[Any]]:
    cur.execute(sql)
    types = [d[1] for d in cur.description]
    rows = sorted(tuple(_canon(v) for v in row) for row in cur.fetchall())
    return rows, types


class _Fixture:
    """One engine, one base schema, every case registered as its own view."""

    def __init__(self, engine: str, conn: Any, cases: Sequence[Case], *, version: Tuple[int, ...], prefix: str) -> None:
        self.engine = engine
        self.conn = conn
        self.version = version
        self.prefix = prefix
        self.t = f"{prefix}t"
        self.u = f"{prefix}u"
        self.udf = f"{prefix}isum"
        self.cases = [
            c for c in cases if (not c.min_version or version >= c.min_version) and (not c.engines or engine in c.engines)
        ]
        self.views: Dict[str, Tuple[Case, str, Any]] = {}
        self.rng = random.Random(SEED)
        self.next_id = 1000

    def sql(self, case: Case) -> str:
        return case.select_sql.format(t=self.t, u=self.u, udf=self.udf)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        cur = self.conn.cursor()
        if self.engine == "sqlite":
            cur.execute(sql.replace("%s", "?"), params)
        else:
            cur.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def compare_all(self, test: unittest.TestCase, context: str) -> None:
        cur = self.conn.cursor()
        for name, (case, view_name, _plan) in self.views.items():
            mv_rows, mv_types = _snapshot(cur, f"SELECT * FROM {view_name}")
            live_rows, live_types = _snapshot(cur, self.sql(case))
            test.assertEqual(mv_rows, live_rows, f"[{self.engine}] {name} ({context}): materialized != live view")
            if self.engine != "sqlite":
                cols = [d[0] for d in cur.description]
                for col, a, b in zip(cols, mv_types, live_types):
                    if col in case.skip_type_code:
                        continue
                    test.assertEqual(a, b, f"[{self.engine}] {name}.{col} ({context}): result type differs")
        self.conn.commit()

    # ------------------------------------------------------------ DML
    def _random_row(self) -> Tuple[Any, ...]:
        rng = self.rng
        self.next_id += 1
        g = rng.choice([1, 2, 3, None])
        h = rng.choice([1, 2])
        v = rng.choice([None, -2, 0, 1, 3, 3, 5, 8])
        x = rng.choice([0.1, -0.0, 2.5, 1e16, 3.0])
        s = rng.choice(["a", "b", "c"])
        d = rng.choice([None, "1.50", "-2.25", "1.50", "10.00"])
        return (self.next_id, g, h, v, x, s, d)

    def _ids(self) -> List[int]:
        cur = self.conn.cursor()
        cur.execute(f"SELECT id FROM {self.t} ORDER BY id")
        ids = [r[0] for r in cur.fetchall()]
        self.conn.commit()
        return ids

    def insert(self, n: int = 1) -> None:
        rows = [self._random_row() for _ in range(n)]
        ph = ", ".join(["(%s, %s, %s, %s, %s, %s, %s)"] * n)
        self.execute(f"INSERT INTO {self.t} (id, g, h, v, x, s, d) VALUES {ph}", [x for r in rows for x in r])
        for r in rows:
            if self.rng.random() < 0.7:
                self.execute(f"INSERT INTO {self.u} (id, tid, w) VALUES (%s, %s, %s)", (r[0] * 10, r[0], self.rng.choice([1, 2, None])))

    def step(self) -> str:
        rng = self.rng
        ids = self._ids()
        op = rng.choice(["insert", "insert_multi", "delete", "update_v", "update_g", "update_null", "delete_group", "update_multi", "update_u", "rollback"])
        if not ids:
            op = "insert_multi"
        target = rng.choice(ids) if ids else 0
        if op == "insert":
            self.insert(1)
        elif op == "insert_multi":
            self.insert(3)
        elif op == "delete":
            self.execute(f"DELETE FROM {self.u} WHERE tid = %s", (target,))
            self.execute(f"DELETE FROM {self.t} WHERE id = %s", (target,))
        elif op == "update_v":
            self.execute(f"UPDATE {self.t} SET v = %s, x = %s WHERE id = %s", (rng.choice([-1, 2, 3, 9]), rng.choice([0.7, -2.0, 1e-300]), target))
        elif op == "update_g":
            self.execute(f"UPDATE {self.t} SET g = %s, h = %s WHERE id = %s", (rng.choice([1, 2, 3, None]), rng.choice([1, 2]), target))
        elif op == "update_null":
            self.execute(f"UPDATE {self.t} SET v = NULL WHERE id = %s", (target,))
        elif op == "delete_group":
            g = rng.choice([1, 2, 3])
            self.execute(f"DELETE FROM {self.u} WHERE tid IN (SELECT id FROM (SELECT id FROM {self.t} WHERE g = %s) AS q)", (g,))
            self.execute(f"DELETE FROM {self.t} WHERE g = %s", (g,))
        elif op == "update_multi":
            self.execute(f"UPDATE {self.t} SET v = v + 1 WHERE g = %s", (rng.choice([1, 2, 3]),))
        elif op == "update_u":
            self.execute(f"UPDATE {self.u} SET w = %s WHERE tid = %s", (rng.choice([0, 4, None]), target))
        else:  # rollback
            self.conn.commit()
            self.execute(f"DELETE FROM {self.t} WHERE id = %s", (target,))
            self.insert(2)
            self.conn.rollback()
            return op
        self.commit()
        return op


def _native_views(fx: "_Fixture") -> List[Case]:
    """The fixture's views: every single-aggregate grouped case with the same
    expected strategy becomes one column of one wide view (one registration,
    one trigger set and one comparison per strategy instead of per aggregate;
    the per-aggregate routing is asserted by OfflineRoutingTests and by the
    strategy of the wide view, which any column of another class would
    change). Other shapes stay separate views."""
    out: List[Case] = []
    for strategy in (T2, T3):
        cols = [c for c in fx.cases if c.grouped_expr is not None and not c.standalone and c.expected == strategy]
        if cols:
            exprs = ", ".join(f"{c.grouped_expr} AS a_{c.name}" for c in cols)
            skip = tuple(f"a_{c.name}" for c in cols if c.skip_type_code)
            out.append(Case(f"wide_{strategy.value}", f"SELECT g, {exprs} FROM {{t}} GROUP BY g", strategy, skip_type_code=skip))
    out += [c for c in fx.cases if c.grouped_expr is None or c.standalone]
    return out


def _drive(test: unittest.TestCase, fx: _Fixture, register: Callable[[str], Any]) -> None:
    for i in range(12):
        fx.insert(1)
    fx.commit()
    views = _native_views(fx)
    ddl = ";\n".join(f"CREATE VIEW {fx.prefix}{case.name} AS {fx.sql(case)}" for case in views)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = register(ddl)
    announced = [w.message.diagnostic for w in caught if isinstance(w.message, MatviewDiagnosticWarning)]
    plans = list(plan.sub_plans) or [plan]
    test.assertEqual(len(plans), len(views))
    for case, sub in zip(views, plans):
        test.assertEqual(sub.strategy, case.expected, f"[{fx.engine}] {case.name}: strategy")
        lead = f"{EXACT_LEAD} {ANNOUNCED[case.expected].value}. Why: "
        reason = sub.diagnostics.fallback_reason if sub.diagnostics else ""
        test.assertTrue(
            (reason or "").startswith(lead),
            f"[{fx.engine}] {case.name}: fallback not announced with the lead phrase: {reason!r}",
        )
        # Announced at registration (a warning) and listed on the plan.
        mine = [d for d in sub.strategy_diagnostics if d.message.startswith(lead)]
        test.assertEqual(len(mine), 1, f"[{fx.engine}] {case.name}: {sub.strategy_diagnostics!r}")
        test.assertIn(mine[0], announced, f"[{fx.engine}] {case.name}: no registration warning")
        test.assertEqual(mine[0].strategy, ANNOUNCED[case.expected])
        fx.views[case.name] = (case, f"{fx.prefix}{case.name}", sub)
    fx.commit()
    fx.compare_all(test, "initial population")
    for i in range(DML_STEPS):
        op = fx.step()
        fx.compare_all(test, f"seed {SEED} step {i} {op}")


# ------------------------------------------------------------- engines
_BASE_DDL = "CREATE TABLE {t} (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, x {dbl}, s VARCHAR(8) NOT NULL, d NUMERIC(10, 2))"
_U_DDL = "CREATE TABLE {u} (id INTEGER PRIMARY KEY, tid INTEGER NOT NULL, w INTEGER)"


class SqliteUnrecognizedAggregateTests(unittest.TestCase):
    def test_native_differential(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            conn = sqlite3.connect(os.path.join(d, "db.sqlite"))

            class PySum:
                def __init__(self) -> None:
                    self.total: Optional[int] = None

                def step(self, value: Any) -> None:
                    if value is not None:
                        self.total = (self.total or 0) + value

                def finalize(self) -> Optional[int]:
                    return self.total

            conn.create_aggregate("py_sum", 1, PySum)  # pyright: ignore[reportArgumentType]  # sqlite3 aggregate protocol
            try:
                fx = _Fixture("sqlite", conn, SQLITE_CASES, version=sqlite3.sqlite_version_info, prefix="")
                conn.execute(_BASE_DDL.format(t=fx.t, dbl="REAL"))
                conn.execute(_U_DDL.format(u=fx.u))
                conn.commit()
                mgr = MaterializedViewManager(conn)
                _drive(self, fx, mgr.register_view)
            finally:
                conn.close()


def _native(env: str) -> Optional[str]:
    return os.environ.get(env) or None


@unittest.skipUnless(_native("POSTGRES_DSN"), "POSTGRES_DSN not set")
class PostgresUnrecognizedAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        import psycopg2
        import psycopg2.extras

        self.conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        # Compare JSON/JSONB as the server's text, not as decoded objects.
        psycopg2.extras.register_default_json(self.conn, loads=lambda s: s)
        psycopg2.extras.register_default_jsonb(self.conn, loads=lambda s: s)
        self.schema = f"l1agg_{uuid.uuid4().hex[:8]}"
        cur = self.conn.cursor()
        cur.execute(f'CREATE SCHEMA "{self.schema}"')
        cur.execute(f'SET search_path TO "{self.schema}"')
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        self.conn.commit()
        self.conn.close()

    def _version(self) -> Tuple[int, ...]:
        return (self.conn.server_version // 10000,)

    def test_native_differential(self) -> None:
        fx = _Fixture("postgres", self.conn, PG_CASES, version=self._version(), prefix="")
        cur = self.conn.cursor()
        cur.execute(_BASE_DDL.format(t=fx.t, dbl="DOUBLE PRECISION"))
        cur.execute(_U_DDL.format(u=fx.u))
        cur.execute(f"CREATE AGGREGATE {fx.udf}(integer) (SFUNC = int4pl, STYPE = integer)")
        self.conn.commit()
        mgr = MaterializedViewManager(self.conn)
        _drive(self, fx, mgr.register_view)

    def test_unknown_result_type_is_refused_with_cast_remedy(self) -> None:
        cur = self.conn.cursor()
        cur.execute(_BASE_DDL.format(t="t", dbl="DOUBLE PRECISION"))
        cur.execute("CREATE AGGREGATE isum(integer) (SFUNC = int4pl, STYPE = integer)")
        self.conn.commit()
        mgr = MaterializedViewManager(self.conn)
        with self.assertRaisesRegex(UnsupportedMaterializedViewError, r"CAST\(\.\.\. AS <type>\)"):
            mgr.register_view("CREATE VIEW isum_v AS SELECT g, isum(v) AS a FROM t GROUP BY g")
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM pg_class WHERE relname = 'isum_v' AND relnamespace = %s::regnamespace", (self.schema,))
        self.assertEqual(cur.fetchone()[0], 0)


@unittest.skipUnless(_native("MYSQL_DSN"), "MYSQL_DSN not set")
class MysqlUnrecognizedAggregateTests(unittest.TestCase):
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
        ver = str(cur.fetchone()[0])
        self.engine = "mariadb" if "mariadb" in ver.lower() else "mysql"
        self.version = tuple(int(x) for x in re.findall(r"\d+", ver)[:2])
        # A private database: teardown is one DROP DATABASE instead of a
        # drop_view per view (MySQL DDL commits are the dominant cost).
        self.database = f"l1agg_{uuid.uuid4().hex[:8]}"
        cur.execute(f"CREATE DATABASE {self.database}")
        cur.execute(f"USE {self.database}")
        self.conn.commit()
        self.prefix = ""

    def tearDown(self) -> None:
        self.conn.rollback()
        cur = self.conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS {self.database}")
        self.conn.commit()
        self.conn.close()

    def test_native_differential(self) -> None:
        fx = _Fixture(self.engine, self.conn, MY_CASES, version=self.version, prefix=self.prefix)
        cur = self.conn.cursor()
        cur.execute(_BASE_DDL.format(t=fx.t, dbl="DOUBLE"))
        cur.execute(_U_DDL.format(u=fx.u))
        self.conn.commit()
        # ignore_limitations acknowledges MYSQL_NO_TRUNCATE_TRIGGER only; it
        # never enables a weaker result guarantee.
        mgr = MaterializedViewManager(self.conn, ignore_limitations=True)
        _drive(self, fx, mgr.register_view)


class ConcurrentWritersTests(unittest.TestCase):
    """Two independent sessions write the same group of a routed view.

    A inserts into group 1 and keeps its transaction open; B has taken its
    snapshot, inserts into group 1 too and waits for A's lock (positive
    lock-wait evidence); A commits, then B. The group-scoped recompute (an
    ordered GROUP_CONCAT / JSONB_AGG) and the full refresh (an unordered one)
    must both include both rows: a recompute from B's pre-wait snapshot would
    miss A's row. MariaDB 11.6+ with innodb_snapshot_isolation may reject B
    (1020, retry); then only A's row is expected.
    """

    def test_two_writers_same_group(self) -> None:
        import threading

        from matview_sync import Dialect
        from test_avg_concurrent_maintenance import _connect, _native_engines, _session_id, _wait_until_lock_waiting

        engines = _native_engines()
        if not engines:
            self.skipTest("no native engine DSN set")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                is_pg = engine == Dialect.POSTGRES
                t = f"l1cc_{uuid.uuid4().hex[:6]}"
                ordered = "JSONB_AGG(v ORDER BY id)" if is_pg else "GROUP_CONCAT(v ORDER BY id)"
                unordered = "JSONB_AGG(v)" if is_pg else "JSON_ARRAYAGG(v)"
                views = {
                    f"{t}_t2": (f"SELECT g, {ordered} AS a FROM {t} GROUP BY g", T2),
                    f"{t}_t3": (f"SELECT g, {unordered} AS a FROM {t} GROUP BY g", T3),
                }
                setup, a, b, monitor = (_connect(engine, dsn) for _ in range(4))
                try:
                    cur = setup.cursor()
                    cur.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)")
                    cur.execute(f"CREATE INDEX {t}_g ON {t} (g)")
                    cur.executemany(f"INSERT INTO {t} VALUES (%s, %s, %s)", [(1, 1, 1), (2, 2, 2)])
                    setup.commit()
                    mgr = MaterializedViewManager(setup, ignore_limitations=not is_pg)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plan = mgr.register_view(";\n".join(f"CREATE VIEW {n} AS {q}" for n, (q, _) in views.items()))
                    self.assertEqual([p.strategy for p in plan.sub_plans], [e for _, e in views.values()])
                    setup.commit()
                    a_sid, b_sid = _session_id(engine, a), _session_id(engine, b)
                    a.cursor().execute(f"INSERT INTO {t} VALUES (10, 1, 10)")
                    bc = b.cursor()
                    bc.execute(f"SELECT COUNT(*) FROM {t}")
                    bc.fetchall()
                    done = threading.Event()
                    errors: List[BaseException] = []

                    def run_b() -> None:
                        try:
                            bc.execute(f"INSERT INTO {t} VALUES (11, 1, 11)")
                            b.commit()
                        except BaseException as e:  # surfaced below
                            errors.append(e)
                        finally:
                            done.set()

                    th = threading.Thread(target=run_b)
                    th.start()
                    _wait_until_lock_waiting(engine, monitor, b_sid, done, blocker_sid=a_sid)
                    a.commit()
                    th.join(30)
                    self.assertFalse(th.is_alive())
                    if errors:
                        self.assertEqual([getattr(e, "args", (None,))[0] for e in errors], [1020], errors)
                        b.rollback()
                    cur = setup.cursor()
                    if is_pg:
                        import psycopg2.extras

                        psycopg2.extras.register_default_jsonb(setup, loads=lambda x: x)
                    for name, (q, _) in views.items():
                        mv = _snapshot(cur, f"SELECT * FROM {name}")[0]
                        live = _snapshot(cur, q)[0]
                        self.assertEqual(mv, live, name)
                    cur.execute(f"SELECT COUNT(*) FROM {t} WHERE g = 1")
                    self.assertEqual(cur.fetchone()[0], 2 if errors else 3)
                    setup.commit()
                finally:
                    for c in (a, b, monitor):
                        try:
                            c.rollback()
                            c.close()
                        except Exception:
                            pass
                    setup.rollback()
                    cleanup = MaterializedViewManager(setup, ignore_limitations=True)
                    for n in views:
                        if n in cleanup.get_catalog():
                            setup.commit()
                            cleanup.drop_view(n)
                    setup.commit()
                    setup.cursor().execute(f"DROP TABLE IF EXISTS {t}")
                    setup.commit()
                    setup.close()


# ------------------------------------------------------------- offline
class OfflineRoutingTests(unittest.TestCase):
    """Structural guards for the routing and its diagnostics (no database)."""

    SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, h INTEGER NOT NULL, v INTEGER, x DOUBLE PRECISION, s VARCHAR(8) NOT NULL, d NUMERIC(10, 2));"

    def _plan(self, dialect: str, select: str) -> Any:
        schema = self.SCHEMA if dialect == "postgres" else self.SCHEMA.replace("DOUBLE PRECISION", "DOUBLE")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            return generate_materialized_view(
                parse_table_schema(schema, dialect=dialect), f"CREATE VIEW mv AS {select}", dialect=dialect, ignore_limitations=True
            )

    def test_reader_never_references_base_columns(self) -> None:
        # The original failure: an unrecognized aggregate was treated as a
        # decomposed compound expression, so the reader view selected the
        # aggregate over the backing table (PostgreSQL: UndefinedColumn).
        for case in PG_CASES:
            if "{u}" in case.select_sql or "{udf}" in case.select_sql:
                continue
            plan = self._plan("postgres", case.select_sql.format(t="t"))
            self.assertEqual(plan.strategy, case.expected, case.name)
            self.assertNotRegex(plan.reader_view_ddl, r"(?i)\b(?:PERCENTILE|JSONB?_AGG|MODE\(|REGR_|CORR\()", case.name)

    def test_total_order_requires_every_primary_key_column(self) -> None:
        schema = "CREATE TABLE t2 (a INTEGER NOT NULL, b INTEGER NOT NULL, g INTEGER, v INTEGER, PRIMARY KEY (a, b));"
        for order, expected in (("a, b", T2), ("b, a", T2), ("a", T3), ("v", T3), ("v, a, b", T2)):
            plan = generate_materialized_view(
                parse_table_schema(schema, dialect="postgres"),
                f"CREATE VIEW mv AS SELECT g, JSONB_AGG(v ORDER BY {order}) AS a FROM t2 GROUP BY g",
                dialect="postgres",
            )
            self.assertEqual(plan.strategy, expected, order)

    def test_recognized_holistic_aggregates_keep_the_group_recompute(self) -> None:
        # The routing only places aggregates nothing else handles: recognized
        # holistic aggregates keep group_scoped_delta whatever their ORDER BY
        # (including over a join), never a costlier strategy.
        cases = [
            ("postgres", "SELECT g, STRING_AGG(s, ',') AS a FROM t GROUP BY g"),
            ("postgres", "SELECT g, ARRAY_AGG(v ORDER BY v) AS a FROM t GROUP BY g"),
            ("mysql", "SELECT g, GROUP_CONCAT(DISTINCT v ORDER BY id) AS a FROM t GROUP BY g"),
            ("sqlite", "SELECT g, json_group_array(v) AS a FROM t GROUP BY g"),
        ]
        for dialect, select in cases:
            self.assertEqual(self._plan(dialect, select).strategy, T2, (dialect, select))
        schema = self.SCHEMA + "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, w VARCHAR(8));"
        plan = generate_materialized_view(
            parse_table_schema(schema, dialect="postgres"),
            "CREATE VIEW mv AS SELECT t.g, STRING_AGG(u.w, ',' ORDER BY u.w) AS a FROM t JOIN u ON u.tid = t.id GROUP BY t.g",
            dialect="postgres",
        )
        self.assertEqual(plan.strategy, T2)

    def test_distinct_ordered_aggregate_is_not_total(self) -> None:
        plan = self._plan("mysql", "SELECT g, JSON_ARRAYAGG(DISTINCT v) AS a FROM t GROUP BY g")
        self.assertEqual(plan.strategy, T3)

    def test_fallback_messages_name_cost_and_remedy(self) -> None:
        plan = self._plan("postgres", "SELECT g, JSON_AGG(v) AS a FROM t GROUP BY g")
        msg = plan.diagnostics.fallback_reason
        self.assertTrue(msg.startswith(f"{EXACT_LEAD} {Strategy.FULL_REFRESH.value}. Why: "), msg)
        self.assertIn("Cost: ", msg)
        self.assertIn("once per statement on PostgreSQL", msg)
        self.assertIn("To change it: add an ORDER BY", msg)
        self.assertEqual([d.code for d in plan.strategy_diagnostics], ["ORDER_SENSITIVE_AGGREGATE_FULL_REFRESH"])
        plan = self._plan("postgres", "SELECT g, MODE() WITHIN GROUP (ORDER BY v) AS a FROM t GROUP BY g")
        msg = plan.diagnostics.fallback_reason
        self.assertTrue(msg.startswith(f"{EXACT_LEAD} {Strategy.GROUP_SCOPED_RECOMPUTE.value}. Why: "), msg)
        self.assertIn("MODE", msg)
        plan = self._plan("postgres", "SELECT g, h, COUNT(*) AS n FROM t GROUP BY ROLLUP(g, h)")
        self.assertIn("UNION ALL", plan.diagnostics.fallback_reason)
        self.assertTrue(plan.diagnostics.to_dict()["fallback_reason"].startswith(EXACT_LEAD))
        self.assertEqual([d.code for d in plan.strategy_diagnostics], ["GROUPING_SETS_FULL_REFRESH"])

    def test_types_on_postgres(self) -> None:
        expect = {
            "JSON_AGG(v ORDER BY id)": "JSON",
            "JSONB_AGG(v ORDER BY id)": "JSONB",
            "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY v)": "DOUBLE PRECISION",
            "PERCENTILE_DISC(ARRAY[0.5]) WITHIN GROUP (ORDER BY v)": "INT[]",
            "MODE() WITHIN GROUP (ORDER BY s)": "VARCHAR",
            "REGR_COUNT(x, v)": "BIGINT",
            "ARRAY_AGG(v ORDER BY id)": "INT[]",
            "RANGE_AGG(INT4RANGE(v, v + 1))": "INT4MULTIRANGE",
        }
        for expr, typ in expect.items():
            plan = self._plan("postgres", f"SELECT g, {expr} AS a FROM t GROUP BY g")
            m = re.search(r"\ba\s+([A-Za-z0-9_ ()\[\],]+?)\s*(?:,\n|\n\))", plan.backing_table_ddl)
            self.assertIsNotNone(m, plan.backing_table_ddl)
            assert m is not None
            self.assertEqual(m.group(1).strip().upper(), typ, expr)

    def test_float_accumulators_default_to_exact_full_refresh(self) -> None:
        # regr_*/corr/covar_* over DOUBLE PRECISION accumulate in float8 in
        # input order. Under default settings (no ignore/accept) they are
        # maintained by re-running the original query (exact like a later
        # read of the live view), not refused: no limitation code is raised.
        for expr in ("REGR_SLOPE(x, v)", "CORR(x, v)", "COVAR_SAMP(x, v)"):
            plan = generate_materialized_view(
                parse_table_schema(self.SCHEMA, dialect="postgres"),
                f"CREATE VIEW mv AS SELECT g, {expr} AS a FROM t GROUP BY g",
                dialect="postgres",
            )
            self.assertEqual(plan.strategy, T3, expr)
            self.assertEqual([lim.code for lim in plan.limitations], [], expr)

    def test_compound_with_unrecognized_aggregate_on_mysql_is_refused_not_guessed(self) -> None:
        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "CAST"):
            self._plan("mysql", "SELECT g, COUNT(*) + JSON_LENGTH(JSON_ARRAYAGG(v)) AS a FROM t GROUP BY g")


if __name__ == "__main__":
    unittest.main(verbosity=2)
