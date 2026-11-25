#!/usr/bin/env python3
"""PostgreSQL / MySQL / MariaDB order-sensitive (floating-point) aggregates.

Default contract: REFUSED. SUM/AVG over approximate (REAL/FLOAT/DOUBLE or not
provably exact) arguments and VAR/STDDEV (MySQL/MariaDB: every argument type)
accumulate in binary floating point in the order the original view's plan
reads the rows. That order is not a function of the data: an index or range
scan for a WHERE clause, sort-based grouping, parallel aggregation and join
order change it (measured natively: PostgreSQL ungrouped ``SUM(x) WHERE y <
10`` with an index on y, MySQL 8.4 / MariaDB 10.11 grouped views with an index
on the WHERE column: original view 1.5, recompute 2.0). No maintained value
equals the original view under every plan, so these shapes raise
EngineLimitationError (ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE) and the
global ``ignore_limitations`` does not override them.

Opt-in contract (``accept_limitations`` per code): BEST EFFORT. Single-table
views (and PostgreSQL UNION ALL aggregates) recompute every affected group
natively, in a separate statement after the group row is locked, in a fixed
order (PostgreSQL ``AGG(x ORDER BY ctid)``, MySQL/MariaDB primary-key order
through a locking read):

* PostgreSQL READ COMMITTED takes a fresh snapshot per statement of the
  trigger function, i.e. after the row-lock wait. A recompute inside the
  locking statement would use the snapshot taken before the wait and miss the
  writer it waited for (the original AVG race).
* MySQL/MariaDB use a locking read (``LOCK IN SHARE MODE``), which reads the
  latest committed versions under REPEATABLE READ and READ COMMITTED alike.

The result equals the original view when the original view evaluates in that
order. The native tests below therefore claim exactness only under a pinned
plan: PostgreSQL ``PG_PINNED_PLAN_SETTINGS`` (serial sequential scan feeding
HashAggregate), MySQL/MariaDB small single-table views whose only secondary
index is the group-key index. Under that pin they compare floats bit for bit
(``float.hex``: sign of zero, NaN, infinities) with the Python result type --
no numeric tolerance. An add/retract fold would not be exact
(``1e16 + 1.5 - 1e16`` folds to 2.0 where the view returns 1.5).

Native tests (set ``POSTGRES_DSN`` and/or ``MYSQL_DSN``) are the evidence;
the offline SQL-shape and manager tests are structural guards only.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
import unittest
import uuid
import warnings
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from matview_sync import (
    PER_CODE_OPT_IN_LIMITATIONS,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
    generate_materialized_view,
    parse_table_schema,
)
from test_matview_sync import FLOAT_OPT_IN, PG_PINNED_PLAN_SETTINGS, pin_native_plan
from test_avg_concurrent_maintenance import (
    LOCK_WAIT_TIMEOUT_S,
    _connect,
    _native_engines,
    _session_id,
    _wait_until_lock_waiting,
)


def _exact(v: Any) -> Any:
    """Bit-exact, type-aware representation (-0.0 != 0.0, NaN == NaN)."""
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, Decimal):
        return ("decimal", v.as_tuple())
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    return (type(v).__name__, v)


def _rows(rows: Sequence[Sequence[Any]]) -> List[Tuple[Any, ...]]:
    return sorted((tuple(_exact(v) for v in r) for r in rows), key=repr)


def _pg_types() -> List[str]:
    return ["DOUBLE PRECISION", "REAL"]


def _my_types() -> List[str]:
    return ["DOUBLE", "FLOAT"]


class _Fixture:
    """An isolated schema (PostgreSQL) or uniquely named tables (MySQL)."""

    def __init__(self, engine: Dialect, dsn: str, tt: bool = False) -> None:
        self.engine, self.dsn = engine, dsn
        self.suffix = uuid.uuid4().hex[:10]
        self.schema = f"mv_pgmyf_{self.suffix}"
        self.conn = _connect(engine, dsn)
        self.tables: List[str] = []
        self.manager: Optional[MaterializedViewManager] = None
        self.plans: List[Any] = []
        cur = self.conn.cursor()
        if engine == Dialect.POSTGRES:
            cur.execute(f'CREATE SCHEMA "{self.schema}"')
            cur.execute(f'SET search_path TO "{self.schema}"')
        self.conn.commit()
        self.tt = tt

    def session(self) -> Any:
        s = _connect(self.engine, self.dsn)
        if self.engine == Dialect.POSTGRES:
            c = s.cursor()
            c.execute(f'SET search_path TO "{self.schema}"')
            s.commit()
        return s

    def table(self, name: str, cols: str, group_index: Optional[str] = None) -> str:
        t = f"{name}_{self.suffix}"
        engine_suffix = " ENGINE=InnoDB" if self.engine == Dialect.MYSQL else ""
        self.conn.cursor().execute(f"CREATE TABLE {t} ({cols}){engine_suffix}")
        if group_index and self.engine == Dialect.MYSQL:
            # MySQL/MariaDB refuse grouped order-sensitive aggregates without a
            # group-key index (MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX):
            # test the supported configuration.
            self.conn.cursor().execute(f"CREATE INDEX ix_{name}_{self.suffix} ON {t} ({group_index})")
        self.conn.commit()
        self.tables.append(t)
        return t

    def register(self, select_sql: str, **kw: Any) -> Any:
        if self.manager is None:
            self.manager = MaterializedViewManager(
                self.conn, dialect=self.engine
            )
        name = f"v{len(self.plans)}_{self.suffix}"
        # Every view here has an approximate order-sensitive aggregate: it is
        # refused by default and installed under the explicit per-code
        # opt-in, whose warning must state the best-effort contract.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", EngineLimitationWarning)
            plan = self.manager.register_view(
                f"CREATE VIEW {name} AS {select_sql};",
                ignore_limitations=True,
                accept_limitations=FLOAT_OPT_IN,
                **kw,
            )
        texts = [str(w.message) for w in caught if issubclass(w.category, EngineLimitationWarning)]
        plans = getattr(plan, "sub_plans", None) or [plan]
        opted_in = any(
            l.code == "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE" for p in plans for l in p.limitations
        )
        if opted_in and not any(
            "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE" in t and "BEST-EFFORT" in t for t in texts
        ):
            raise AssertionError(f"opt-in warning missing or without best-effort contract: {texts}")
        self.conn.commit()
        self.plans.append(plan)
        return plan

    def close(self) -> None:
        try:
            self.conn.rollback()
            cur = self.conn.cursor()
            if self.engine == Dialect.POSTGRES:
                cur.execute("SET search_path TO public")
                cur.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
            else:
                for plan in self.plans:
                    try:
                        self.manager.drop_view(plan.view_name)  # type: ignore[union-attr]
                    except Exception:
                        self.conn.rollback()
                for t in self.tables:
                    cur.execute(f"DROP TABLE IF EXISTS {t}")
            self.conn.commit()
        finally:
            self.conn.close()


class _ExactCase(unittest.TestCase):
    maxDiff = None

    def assertViewExact(self, conn: Any, plan: Any, select_sql: str, context: str) -> None:
        conn.commit()
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {plan.view_name}")
        mv = _rows(cur.fetchall())
        # Exactness is claimed only for the pinned plan (serial seq scan +
        # HashAggregate on PostgreSQL; small tables read in primary-key order
        # on MySQL/MariaDB), see test_matview_sync.PG_PINNED_PLAN_SETTINGS.
        pin_native_plan(cur, plan.dialect)
        cur.execute(select_sql)
        live = _rows(cur.fetchall())
        conn.commit()
        self.assertEqual(mv, live, f"{context}: materialized view != original view")


# ---------------------------------------------------------------------------
# Native: adversarial sequential histories
# ---------------------------------------------------------------------------


def _adversarial_values(engine: Dialect, col_type: str) -> Dict[str, Any]:
    single = col_type.upper() in ("REAL", "FLOAT")
    v: Dict[str, Any] = {
        # cancellation / absorption: big + small - big
        "big": 1e8 if single else 1e16,
        "small": 1.5,
        "tenths": (0.1, 0.2, 0.3),
        # subnormals (binary32 / binary64)
        "subnormals": (1e-45, 1.2e-38, -1e-45) if single else (5e-324, 2.2250738585072014e-308, -5e-324),
        # huge magnitudes and a tiny one (squares stay finite: VAR/STDDEV of
        # larger values overflow in the original view itself, see
        # test_postgres_overflowing_sum_rejects_the_write)
        "huge": (1e18, -1e18, 1e-38) if single else (1e150, -1e150, 1e-308),
        "neg_zero": -0.0,
    }
    return v


def _seq_ops(t: str, engine: Dialect, col_type: str) -> List[Tuple[str, Tuple[Any, ...]]]:
    """Deterministic history covering insert / multi-row insert / delete /
    value update / group moves / NULL values and keys / empty groups.

    On MySQL/MariaDB a group whose values are all -0.0 is avoided: the sign of
    its native SUM/AVG depends on the plan of the original view (see
    NativeSequentialExactness.test_mysql_negative_zero_group_is_plan_dependent),
    so no maintained value can equal it under every plan."""
    v = _adversarial_values(engine, col_type)
    zero_partner = 0.0 if engine == Dialect.POSTGRES else 2.5
    a, b, c = v["tenths"]
    s1, s2, s3 = v["subnormals"]
    h1, h2, h3 = v["huge"]
    ins = f"INSERT INTO {t} (id, g, x, k) VALUES "
    return [
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s),(%s,%s,%s,%s)", (1, 1, a, 1, 2, 1, b, 2, 3, 1, c, 3)),
        (f"DELETE FROM {t} WHERE id = %s", (2,)),
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s)", (4, 2, v["big"], 4, 5, 2, v["small"], 5)),
        (f"DELETE FROM {t} WHERE id = %s", (4,)),
        (ins + "(%s,%s,%s,%s)", (6, 2, -v["big"], 6)),
        (ins + "(%s,%s,%s,%s)", (7, 2, v["big"], 7)),
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s),(%s,%s,%s,%s)", (8, 3, s1, 8, 9, 3, s2, 9, 10, 3, s3, 10)),
        (f"DELETE FROM {t} WHERE id = %s", (8,)),
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s)", (11, 4, v["neg_zero"], 11, 20, 4, zero_partner, 20)),
        (ins + "(%s,%s,%s,%s)", (12, 4, 0.0, 12)),
        (f"DELETE FROM {t} WHERE id = %s", (12,)),
        (f"UPDATE {t} SET x = %s WHERE id = %s", (-0.0, 20)) if engine == Dialect.POSTGRES
        else (f"UPDATE {t} SET x = %s WHERE id = %s", (-1.5, 20)),
        (ins + "(%s,%s,%s,%s)", (13, 5, None, 13)),
        (ins + "(%s,%s,%s,%s)", (14, 5, 2.5, None)),
        (f"DELETE FROM {t} WHERE id = %s", (14,)),
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s)", (15, None, 0.7, 15, 16, None, 0.1, 16)),
        (f"DELETE FROM {t} WHERE id = %s", (15,)),
        (f"UPDATE {t} SET x = x + %s WHERE g = %s", (0.7, 1)),
        (f"UPDATE {t} SET g = %s WHERE id = %s", (2, 3)),
        (f"UPDATE {t} SET g = NULL WHERE id = %s", (1,)),
        (f"UPDATE {t} SET g = %s, x = %s WHERE id = %s", (3, 0.3, 16)),
        (f"DELETE FROM {t} WHERE g = %s", (3,)),
        (ins + "(%s,%s,%s,%s),(%s,%s,%s,%s),(%s,%s,%s,%s)", (17, 6, h1, 17, 18, 6, h2, 18, 19, 6, h3, 19)),
        (f"DELETE FROM {t} WHERE id = %s", (17,)),
        (f"UPDATE {t} SET x = x * %s", (1.0000001,)),
        (f"UPDATE {t} SET x = NULL WHERE g = %s", (2,)),
        (f"DELETE FROM {t} WHERE id > %s", (0,)),
    ]


def _views(t: str, engine: Dialect) -> List[str]:
    return [
        # Grouped by a nullable key: every order-sensitive aggregate, an
        # expression argument, and an exact (BIGINT) SUM/AVG that stays folded.
        f"SELECT g, COUNT(*) AS c, COUNT(x) AS n, SUM(x) AS s, AVG(x) AS a, VAR_POP(x) AS vp, "
        f"VAR_SAMP(x) AS vs, STDDEV_POP(x) AS sp, STDDEV_SAMP(x) AS ss, SUM(x * 3) AS s3, "
        f"SUM(k) AS sk, AVG(k) AS ak, MIN(x) AS mn, MAX(x) AS mx FROM {t} GROUP BY g",
        # Global aggregate (no GROUP BY).
        f"SELECT COUNT(*) AS c, SUM(x) AS s, AVG(x) AS a, STDDEV_SAMP(x) AS ss FROM {t}",
        # DISTINCT aggregates and a WHERE filter. (Mixing them with
        # non-DISTINCT order-sensitive aggregates in one grouped view is
        # rejected: see OfflineShapeGuards.test_grouped_distinct_mix_is_rejected.)
        f"SELECT g, COUNT(*) AS c, SUM(DISTINCT x) AS sd, AVG(DISTINCT x) AS ad "
        f"FROM {t} WHERE k IS NULL OR k <> 7 GROUP BY g",
        f"SELECT g, SUM(x) AS s, VAR_POP(x) AS vp FROM {t} WHERE k IS NULL OR k <> 7 GROUP BY g",
    ]


class NativeSequentialExactness(_ExactCase):
    def _configs(self, engine: Dialect) -> List[Tuple[str, bool]]:
        if engine == Dialect.POSTGRES:
            return [(ty, tt) for ty in _pg_types() for tt in (False,)]
        return [(ty, False) for ty in _my_types()]

    def test_adversarial_histories(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for col_type, tt in self._configs(engine):
                with self.subTest(engine=engine.value, type=col_type, tt=tt):
                    self._run_history(engine, dsn, col_type, tt)

    def _run_history(self, engine: Dialect, dsn: str, col_type: str, tt: bool) -> None:
        fx = _Fixture(engine, dsn, tt)
        try:
            t = fx.table("seq", f"id BIGINT PRIMARY KEY, g INT, x {col_type}, k BIGINT", group_index="g")
            plans = [(fx.register(sql), sql) for sql in _views(t, engine)]
            cur = fx.conn.cursor()
            for i, (sql, params) in enumerate(_seq_ops(t, engine, col_type)):
                cur.execute(sql, params)
                fx.conn.commit()
                for plan, live in plans:
                    self.assertViewExact(fx.conn, plan, live, f"{engine.value} {col_type} tt={tt} step {i}: {sql} {params}")
        finally:
            fx.close()

    def test_mysql_negative_zero_group_is_plan_dependent(self) -> None:
        """KNOWN EXCLUSION, demonstrated natively: MySQL's GROUP BY through a
        temporary table starts a group's SUM/AVG from its first value (-0.0),
        while index-ordered or filtered aggregation starts from 0.0
        (0.0 + -0.0 = 0.0). The original view's result for a group whose values
        are all -0.0 therefore depends on its plan (here: on an index), and the
        materialized value (computed like the filtered aggregate) can match
        only one of them. Every other value set is unaffected (x + 0.0 == x)."""
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        fx = _Fixture(Dialect.MYSQL, dsn)
        try:
            t = fx.table("nz", "id BIGINT PRIMARY KEY, g INT, x DOUBLE")
            live = f"SELECT g, SUM(x) AS s FROM {t} GROUP BY g"
            plan = fx.register(live)
            cur = fx.conn.cursor()
            cur.execute(f"INSERT INTO {t} VALUES (1, 1, %s)", (-0.0,))
            fx.conn.commit()
            cur.execute(f"SELECT SUM(x) FROM {t} WHERE g = 1")
            filtered = cur.fetchone()[0]
            cur.execute(f"SELECT s FROM {plan.view_name}")
            materialized = cur.fetchone()[0]
            self.assertEqual(_exact(materialized), _exact(filtered))
            cur.execute(f"SELECT COUNT(*) FROM information_schema.statistics WHERE table_name = %s AND column_name = 'g'", (t,))
            self.assertEqual(cur.fetchone()[0], 0)
            cur.execute(live)
            unindexed = cur.fetchone()[1]
            cur.execute(f"CREATE INDEX ix_nz_{fx.suffix} ON {t} (g)")
            cur.execute(live)
            indexed = cur.fetchone()[1]
            cur.execute("SELECT VERSION()")
            is_mariadb = "mariadb" in cur.fetchone()[0].lower()
            fx.conn.commit()
            self.assertEqual(math.copysign(1.0, materialized), 1.0)
            self.assertEqual(math.copysign(1.0, indexed), 1.0)
            # MySQL: the unindexed plan returns -0.0 (MariaDB: +0.0).
            self.assertEqual(math.copysign(1.0, unindexed), 1.0 if is_mariadb else -1.0)
        finally:
            fx.close()

    def test_postgres_row_relocation_without_referenced_change(self) -> None:
        """PostgreSQL accumulates a group in sequential-scan (ctid) order. An
        UPDATE that changes no column the view references still writes a new
        row version at a new ctid and can change the original view's float
        SUM; the materialized value must follow (the trigger cannot be
        restricted to UPDATE OF the referenced columns)."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        for tt in (False,):
            with self.subTest(tt=tt):
                fx = _Fixture(Dialect.POSTGRES, dsn, tt)
                try:
                    t = fx.table("mv", "id BIGINT PRIMARY KEY, g INT, x DOUBLE PRECISION, note TEXT", group_index="g")
                    live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s, VAR_POP(x) AS vp FROM {t} GROUP BY g"
                    plan = fx.register(live)
                    cur = fx.conn.cursor()
                    for row in ((1, 1, 1.5), (2, 1, 1e16), (3, 1, -1e16)):
                        cur.execute(f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)", row)
                    fx.conn.commit()
                    cur.execute(f"SELECT SUM(x) FROM {t}")
                    self.assertEqual(cur.fetchone()[0], 2.0)  # 1.5 + 1e16 - 1e16 in ctid order
                    self.assertViewExact(fx.conn, plan, live, f"tt={tt} initial")
                    cur.execute(f"UPDATE {t} SET note = 'moved' WHERE id = 1")
                    fx.conn.commit()
                    cur.execute(f"SELECT SUM(x) FROM {t}")
                    self.assertEqual(cur.fetchone()[0], 1.5)  # 1e16 - 1e16 + 1.5
                    fx.conn.commit()
                    self.assertViewExact(fx.conn, plan, live, f"tt={tt} after unreferenced-column update")
                    cur.execute(f"UPDATE {t} SET x = x WHERE id = 2")
                    fx.conn.commit()
                    self.assertViewExact(fx.conn, plan, live, f"tt={tt} after same-value update")
                finally:
                    fx.close()

    def test_postgres_table_rewrite_reorders_rows_known_exclusion(self) -> None:
        """KNOWN EXCLUSION, demonstrated natively: CLUSTER rewrites the table in
        index order without firing triggers, which can change the original
        view's float result while no row changed. repair_view() restores
        equality."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        fx = _Fixture(Dialect.POSTGRES, dsn)
        try:
            t = fx.table("cl", "id BIGINT PRIMARY KEY, g INT, x DOUBLE PRECISION", group_index="g")
            live = f"SELECT g, SUM(x) AS s FROM {t} GROUP BY g"
            plan = fx.register(live)
            cur = fx.conn.cursor()
            for row in ((3, 1, 1.5), (1, 1, 1e16), (2, 1, -1e16)):
                cur.execute(f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)", row)
            fx.conn.commit()
            self.assertViewExact(fx.conn, plan, live, "before CLUSTER")
            cur.execute(f"CLUSTER {t} USING {t}_pkey")
            fx.conn.commit()
            cur.execute(f"SELECT s FROM {plan.view_name}")
            materialized = cur.fetchone()[0]
            cur.execute(live)
            rewritten = cur.fetchone()[1]
            fx.conn.commit()
            self.assertEqual((materialized, rewritten), (2.0, 1.5))
            fx.manager.repair_view(plan.view_name)
            fx.conn.commit()
            self.assertViewExact(fx.conn, plan, live, "after repair_view")
        finally:
            fx.close()

    def test_postgres_aliased_base_table_ctid_order(self) -> None:
        """The ctid order pinning the float recompute is qualified like the
        recompute's FROM, which keeps the view's alias (`FROM t AS a`): a bare
        `t.ctid` is an invalid FROM-clause reference there and the view went
        stale after the first write."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        fx = _Fixture(Dialect.POSTGRES, dsn)
        try:
            t = fx.table("al", "id BIGINT PRIMARY KEY, g INT, f DOUBLE PRECISION", group_index="g")
            cur = fx.conn.cursor()
            for i in range(1, 9):
                cur.execute(f"INSERT INTO {t} (id, g, f) VALUES (%s, %s, %s)", (i, i % 3, i * 0.1))
            fx.conn.commit()
            live = f"SELECT a.g, SUM(a.f) AS s, COUNT(*) AS c FROM {t} AS a GROUP BY a.g"
            plan = fx.register(live)
            self.assertViewExact(fx.conn, plan, live, "initial")
            for step in (
                f"INSERT INTO {t} (id, g, f) VALUES (10, 1, 0.1), (11, 2, 0.2)",
                f"UPDATE {t} SET f = f * 3 WHERE id <= 5",
                f"DELETE FROM {t} WHERE id = 2",
                f"UPDATE {t} SET g = 2 WHERE id = 1",
            ):
                cur.execute(step)
                fx.conn.commit()
                self.assertViewExact(fx.conn, plan, live, step)
        finally:
            fx.close()

    def test_postgres_numeric_display_scale(self) -> None:
        """PostgreSQL SUM over values of different display scales returns the
        largest scale of the rows present; a fold would remember the scale of
        removed rows (1.50 + 2 - 1.50 -> 2.00 vs the view's 2). Unconstrained
        NUMERIC and division are recomputed; NUMERIC(p, s) and integers fold."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        for tt in (False,):
            with self.subTest(tt=tt):
                fx = _Fixture(Dialect.POSTGRES, dsn, tt)
                try:
                    t = fx.table("ns", "id BIGINT PRIMARY KEY, g INT, x NUMERIC, y NUMERIC(10,2), z INT", group_index="g")
                    live = (
                        f"SELECT g, SUM(x) AS sx, SUM(y) AS sy, SUM(z) AS sz, SUM(z / 3.0) AS sd, "
                        f"SUM(CASE WHEN z > 1 THEN 1.5 ELSE 2 END) AS sc FROM {t} GROUP BY g"
                    )
                    plan = fx.register(live)
                    cur = fx.conn.cursor()
                    steps = [
                        f"INSERT INTO {t} VALUES (1, 1, 1.50, 1.5, 1), (2, 1, 2, 2, 2), (3, 1, 0.125, 0.1, 3)",
                        f"DELETE FROM {t} WHERE id = 3",
                        f"DELETE FROM {t} WHERE id = 1",
                        f"UPDATE {t} SET x = 7.0000 WHERE id = 2",
                        f"UPDATE {t} SET x = 7 WHERE id = 2",
                    ]
                    for sql in steps:
                        cur.execute(sql)
                        fx.conn.commit()
                        self.assertViewExact(fx.conn, plan, live, f"tt={tt}: {sql}")
                finally:
                    fx.close()

    def test_postgres_infinities_and_nan(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set; PostgreSQL stores IEEE infinities/NaN natively")
        for col_type in _pg_types():
            for tt in (False,):
                with self.subTest(type=col_type, tt=tt):
                    fx = _Fixture(Dialect.POSTGRES, dsn, tt)
                    try:
                        t = fx.table("inf", f"id BIGINT PRIMARY KEY, g INT, x {col_type}, k BIGINT", group_index="g")
                        plans = [(fx.register(sql), sql) for sql in _views(t, Dialect.POSTGRES)]
                        cur = fx.conn.cursor()
                        steps = [
                            (f"INSERT INTO {t} VALUES (1, 1, 'Infinity', 1), (2, 1, '-Infinity', 2), (3, 1, 1.5, 3)", ()),
                            (f"DELETE FROM {t} WHERE id = 2", ()),  # NaN -> Infinity
                            (f"DELETE FROM {t} WHERE id = 1", ()),  # fold: inf - inf = NaN; view: 1.5
                            (f"INSERT INTO {t} VALUES (4, 1, 'NaN', 4)", ()),
                            (f"DELETE FROM {t} WHERE id = 4", ()),
                            (f"UPDATE {t} SET x = '-Infinity' WHERE id = 3", ()),
                            (f"UPDATE {t} SET x = 2.5 WHERE id = 3", ()),
                        ]
                        for sql, params in steps:
                            cur.execute(sql, params)
                            fx.conn.commit()
                            for plan, live in plans:
                                self.assertViewExact(fx.conn, plan, live, f"{col_type} tt={tt}: {sql}")
                    finally:
                        fx.close()

    def test_postgres_overflowing_sum_rejects_the_write(self) -> None:
        """PostgreSQL float8/float4 SUM raises on overflow; the original view
        then cannot be evaluated, so the write that causes it is rejected and
        the materialized view keeps matching the (unchanged) base rows."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        for col_type, big in (("DOUBLE PRECISION", 1.7976931348623157e308), ("REAL", 3.4e38)):
            for tt in (False,):
                with self.subTest(type=col_type, tt=tt):
                    fx = _Fixture(Dialect.POSTGRES, dsn, tt)
                    try:
                        t = fx.table("ovf", f"id BIGINT PRIMARY KEY, g INT, x {col_type}", group_index="g")
                        live = f"SELECT g, SUM(x) AS s FROM {t} GROUP BY g"
                        plan = fx.register(live)
                        cur = fx.conn.cursor()
                        cur.execute(f"INSERT INTO {t} VALUES (1, 1, %s)", (big,))
                        fx.conn.commit()
                        with self.assertRaisesRegex(Exception, "out of range"):
                            cur.execute(f"INSERT INTO {t} VALUES (2, 1, %s)", (big,))
                        fx.conn.rollback()
                        self.assertViewExact(fx.conn, plan, live, f"{col_type} tt={tt} after rejected write")
                    finally:
                        fx.close()


# ---------------------------------------------------------------------------
# Native: randomized differential (reproducible seeds)
# ---------------------------------------------------------------------------


_RANDOM_SEEDS = [int(s) for s in os.environ.get("MATVIEW_PGMY_FLOAT_SEEDS", "11,23,37,41,59,73").split(",") if s]


def _random_value(rng: random.Random, single: bool, allow_negative_zero: bool = True) -> Optional[float]:
    kind = rng.randrange(9)
    if kind == 0:
        return None
    if kind == 1:
        return rng.choice([1e8, -1e8] if single else [1e16, -1e16, 2.0 ** 53, -(2.0 ** 53)])
    if kind == 2:
        return rng.choice([0.1, 0.2, 0.3, 0.7, -0.1])
    if kind == 3:
        return rng.choice([1e-45, -1e-45] if single else [5e-324, -5e-324, 2.2250738585072014e-308])
    if kind == 4:
        # MySQL/MariaDB: -0.0 only together with other values (see _seq_ops).
        return rng.choice([-0.0, 0.0]) if allow_negative_zero else 0.0
    if kind == 5:
        return rng.choice([1e18, -1e18] if single else [1e150, -1e150])
    if kind == 6:
        return float(rng.randrange(-5, 6))
    return rng.uniform(-1000, 1000)


def _random_ops(rng: random.Random, t: str, single: bool, n_ops: int, allow_negative_zero: bool) -> List[Tuple[str, Tuple[Any, ...]]]:
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    next_id = 1
    groups = [None, 1, 2, 3]
    for _ in range(n_ops):
        kind = rng.randrange(6)
        if kind == 0 or next_id < 4:
            rows = [(next_id + i, rng.choice(groups), _random_value(rng, single, allow_negative_zero), rng.randrange(-3, 4)) for i in range(rng.randrange(1, 4))]
            next_id += len(rows)
            ops.append((f"INSERT INTO {t} (id, g, x, k) VALUES " + ", ".join("(%s,%s,%s,%s)" for _ in rows), tuple(v for r in rows for v in r)))
        elif kind == 1:
            ops.append((f"DELETE FROM {t} WHERE id = %s", (rng.randrange(1, next_id),)))
        elif kind == 2:
            ops.append((f"UPDATE {t} SET x = %s WHERE id = %s", (_random_value(rng, single, allow_negative_zero), rng.randrange(1, next_id))))
        elif kind == 3:
            g = rng.choice(groups)
            if g is None:
                ops.append((f"UPDATE {t} SET g = NULL WHERE id = %s", (rng.randrange(1, next_id),)))
            else:
                ops.append((f"UPDATE {t} SET g = %s WHERE id = %s", (g, rng.randrange(1, next_id))))
        elif kind == 4:
            lo = rng.randrange(1, next_id)
            ops.append((f"DELETE FROM {t} WHERE id BETWEEN %s AND %s", (lo, lo + rng.randrange(0, 3))))
        else:
            ops.append((f"UPDATE {t} SET x = x + %s WHERE k = %s", (rng.choice([0.1, 1e16, -0.3, 1.5]), rng.randrange(-3, 4))))
    return ops


class NativeRandomizedDifferential(_ExactCase):
    def test_random_histories(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            configs = (
                [(ty, tt) for ty in _pg_types() for tt in (False,)]
                if engine == Dialect.POSTGRES
                else [(ty, False) for ty in _my_types()]
            )
            for col_type, tt in configs:
                for seed in _RANDOM_SEEDS:
                    with self.subTest(engine=engine.value, type=col_type, tt=tt, seed=seed):
                        self._run(engine, dsn, col_type, tt, seed)

    def _run(self, engine: Dialect, dsn: str, col_type: str, tt: bool, seed: int) -> None:
        rng = random.Random(seed)
        single = col_type.upper() in ("REAL", "FLOAT")
        fx = _Fixture(engine, dsn, tt)
        try:
            t = fx.table("rnd", f"id BIGINT PRIMARY KEY, g INT, x {col_type}, k BIGINT", group_index="g")
            plans = [(fx.register(sql), sql) for sql in _views(t, engine)]
            cur = fx.conn.cursor()
            for i, (sql, params) in enumerate(_random_ops(rng, t, single, 40, engine == Dialect.POSTGRES)):
                try:
                    cur.execute(sql, params)
                    fx.conn.commit()
                except Exception as exc:
                    # Only an overflowing native aggregate may reject a write
                    # (the original view raises "out of range" as well).
                    fx.conn.rollback()
                    self.assertRegex(str(exc), "out of range", repr(exc))
                for plan, live in plans:
                    self.assertViewExact(
                        fx.conn, plan, live,
                        f"replay: MATVIEW_PGMY_FLOAT_SEEDS={seed} {engine.value} {col_type} tt={tt} step {i}: {sql} {params}",
                    )
        finally:
            fx.close()


# ---------------------------------------------------------------------------
# Native: two-session interleavings (second session observed waiting)
# ---------------------------------------------------------------------------


Op = Callable[[Any, str], None]


def _ins(pk: int, grp: str, amt: Optional[float]) -> Op:
    return lambda cur, t: cur.execute(f"INSERT INTO {t} (id, grp, amt) VALUES (%s, %s, %s)", (pk, grp, amt))


def _upd(pk: int, delta: float) -> Op:
    return lambda cur, t: cur.execute(f"UPDATE {t} SET amt = amt + %s WHERE id = %s", (delta, pk))


def _move(pk: int, grp: str) -> Op:
    return lambda cur, t: cur.execute(f"UPDATE {t} SET grp = %s WHERE id = %s", (grp, pk))


def _del(pk: int) -> Op:
    return lambda cur, t: cur.execute(f"DELETE FROM {t} WHERE id = %s", (pk,))


def _multi(*ops: Op) -> Op:
    def run(cur: Any, t: str) -> None:
        for op in ops:
            op(cur, t)
    return run


# Group G1 holds ids 1..6, G2 ids 7..8. Every pair touches G1 in both sessions.
FLOAT_SEED = [(1, "G1", 0.1), (2, "G1", 0.2), (3, "G1", 0.3), (4, "G1", -2.5), (5, "G1", 0.7), (6, "G1", None),
              (7, "G2", 1.5), (8, "G2", -0.1)]

INTERLEAVINGS: List[Tuple[str, Op, Op]] = [
    ("insert/insert", _ins(101, "G1", 0.7), _ins(201, "G1", 12.25)),
    ("insert/delete", _ins(101, "G1", 0.7), _del(2)),
    ("delete/insert", _del(1), _ins(201, "G1", 1.3)),
    ("delete/delete", _del(1), _del(2)),
    ("update/insert", _upd(3, 0.5), _ins(201, "G1", 2.2)),
    ("insert/update", _ins(101, "G1", 9.9), _upd(4, -1.1)),
    ("move-out/insert", _move(5, "G2"), _ins(201, "G1", 0.1)),
    ("insert/move-in", _ins(101, "G1", 0.3), _move(7, "G1")),
    ("null/insert", _ins(101, "G1", None), _ins(201, "G1", 4.5)),
    ("multi/delete", _multi(_ins(101, "G1", 0.1), _ins(102, "G2", 0.2)), _del(3)),
    # One statement whose first row (G2) reads before its second row (G1)
    # waits: under READ COMMITTED that statement's read view predates the
    # wait, so only a locking read sees the first session's G1 row.
    ("insert/multirow-other-group-first", _ins(101, "G1", 0.7),
     lambda cur, t: cur.execute(f"INSERT INTO {t} (id, grp, amt) VALUES (201, 'G2', 0.5), (202, 'G1', 2.5)")),
]


class NativeInterleavings(_ExactCase):
    VIEW = "SELECT grp, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS a, VAR_SAMP(amt) AS vs, STDDEV_POP(amt) AS sp FROM {t} GROUP BY grp"

    def _configs(self, engine: Dialect) -> List[Tuple[str, bool, Optional[str]]]:
        if engine == Dialect.POSTGRES:
            return [(ty, tt, None) for ty in _pg_types() for tt in (False,)]
        return [(ty, False, iso) for ty in _my_types() for iso in (None, "READ COMMITTED")]

    def test_second_writer_waits_and_sees_first(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for col_type, tt, iso in self._configs(engine):
                for name, first, second in INTERLEAVINGS:
                    with self.subTest(engine=engine.value, type=col_type, tt=tt, isolation=iso, case=name):
                        self._run(engine, dsn, col_type, tt, iso, first, second, name)

    def _run(self, engine: Dialect, dsn: str, col_type: str, tt: bool, iso: Optional[str],
             first: Op, second: Op, name: str) -> None:
        fx = _Fixture(engine, dsn, tt)
        sessions: List[Any] = []
        try:
            t = fx.table("race", f"id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt {col_type}", group_index="grp")
            fx.conn.cursor().executemany(f"INSERT INTO {t} (id, grp, amt) VALUES (%s, %s, %s)", FLOAT_SEED)
            fx.conn.commit()
            live = self.VIEW.format(t=t)
            plan = fx.register(live)
            s1, s2, monitor = fx.session(), fx.session(), fx.session()
            sessions = [s1, s2, monitor]
            for s in (s1, s2):
                if iso:
                    s.cursor().execute(f"SET SESSION TRANSACTION ISOLATION LEVEL {iso}")
                s.commit()
            sid1 = _session_id(engine, s1)
            sid2 = _session_id(engine, s2)
            first(s1.cursor(), t)  # holds the G1 backing-row lock, uncommitted
            done = threading.Event()
            errors: List[BaseException] = []

            def run_second() -> None:
                try:
                    second(s2.cursor(), t)
                    s2.commit()
                except BaseException as exc:
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
            th.join(timeout=LOCK_WAIT_TIMEOUT_S)
            self.assertFalse(th.is_alive(), "second session did not finish after the first committed")
            if errors:
                raise errors[0]
            self.assertViewExact(fx.conn, plan, live, f"{engine.value} {col_type} tt={tt} iso={iso} {name}")
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            fx.close()


class NativeMysqlPrelock(_ExactCase):
    # Measured on MySQL 9.7 with this workload, 3+3 runs: 2-6 deadlocks with
    # the BEFORE-trigger pre-lock, 30-64 without it.
    MAX_DEADLOCKS = 15

    def test_single_write_transactions_rarely_deadlock(self) -> None:
        """4 independent writers, 300 single-row transactions each, on 6
        indexed groups of a DOUBLE SUM/AVG view: the pre-lock keeps InnoDB
        deadlocks (1213, retried here) rare, and the result stays exact."""
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        fx = _Fixture(Dialect.MYSQL, dsn)
        try:
            t = fx.table("dl", "id BIGINT PRIMARY KEY, g INT NOT NULL, x DOUBLE", group_index="g")
            rng0 = random.Random(5)
            fx.conn.cursor().executemany(
                f"INSERT INTO {t} VALUES (%s, %s, %s)",
                [(i, rng0.randrange(6), rng0.choice([0.1, 0.2, 1e16, -1e16, 1.5])) for i in range(1, 200)],
            )
            fx.conn.commit()
            live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s, AVG(x) AS a FROM {t} GROUP BY g"
            plan = fx.register(live)
            deadlocks = [0]
            lock = threading.Lock()
            next_id = [10_000]
            errors: List[BaseException] = []

            def worker(wi: int) -> None:
                rng = random.Random(100 + wi)
                s = fx.session()
                try:
                    for _ in range(300):
                        kind = rng.random()
                        if kind < 0.4:
                            with lock:
                                next_id[0] += 1
                                nid = next_id[0]
                            sql, params = f"INSERT INTO {t} VALUES (%s, %s, %s)", (nid, rng.randrange(6), rng.choice([0.1, 0.3, 1e16, 2.5]))
                        elif kind < 0.6:
                            sql, params = f"DELETE FROM {t} WHERE id = %s", (rng.randrange(1, next_id[0]),)
                        elif kind < 0.8:
                            sql, params = f"UPDATE {t} SET x = %s WHERE id = %s", (rng.choice([0.7, -1e16]), rng.randrange(1, next_id[0]))
                        else:
                            sql, params = f"UPDATE {t} SET g = %s WHERE id = %s", (rng.randrange(6), rng.randrange(1, next_id[0]))
                        for _attempt in range(20):
                            try:
                                s.cursor().execute(sql, params)
                                s.commit()
                                break
                            except Exception as exc:
                                s.rollback()
                                if "1213" not in str(exc):
                                    raise
                                with lock:
                                    deadlocks[0] += 1
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    s.close()

            threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(4)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=300)
                self.assertFalse(th.is_alive())
            if errors:
                raise errors[0]
            self.assertViewExact(fx.conn, plan, live, f"after {deadlocks[0]} retried deadlocks")
            cur = fx.conn.cursor()
            cur.execute("SELECT VERSION()")
            is_mariadb = "mariadb" in cur.fetchone()[0].lower()
            fx.conn.commit()
            if is_mariadb:
                # MariaDB 10.11, same workload, 3 runs: 20-27 with the pre-lock,
                # 31-43 without: too close for a rate assertion; exactness is
                # what this test establishes there.
                return
            self.assertLessEqual(deadlocks[0], self.MAX_DEADLOCKS, "pre-lock missing or ineffective")
        finally:
            fx.close()

    def test_second_writer_queues_before_writing_its_base_row(self) -> None:
        """Session 1 writes group G1 and stays open; session 2's insert into G1
        waits (observed lock wait); session 1 then writes G1 again and both
        commit without a deadlock, with an exact result. (Here the next-key
        locks of session 1's locking recompute already block session 2 before
        its base row is written; the pre-lock covers the remaining window,
        measured by test_single_write_transactions_rarely_deadlock.)"""
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        fx = _Fixture(Dialect.MYSQL, dsn)
        sessions: List[Any] = []
        try:
            t = fx.table("pl", "id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt DOUBLE", group_index="grp")
            fx.conn.cursor().executemany(f"INSERT INTO {t} VALUES (%s, %s, %s)", [(1, "G1", 0.1), (2, "G2", 0.2)])
            fx.conn.commit()
            live = f"SELECT grp, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS a FROM {t} GROUP BY grp"
            plan = fx.register(live)
            s1, s2, monitor = fx.session(), fx.session(), fx.session()
            sessions = [s1, s2, monitor]
            sid1, sid2 = _session_id(Dialect.MYSQL, s1), _session_id(Dialect.MYSQL, s2)
            s1.cursor().execute(f"INSERT INTO {t} VALUES (101, 'G1', 0.7)")
            done = threading.Event()
            errors: List[BaseException] = []

            def second() -> None:
                try:
                    s2.cursor().execute(f"INSERT INTO {t} VALUES (201, 'G1', 1e16)")
                    s2.commit()
                except BaseException as exc:
                    errors.append(exc)
                    s2.rollback()
                finally:
                    done.set()

            th = threading.Thread(target=second, daemon=True)
            th.start()
            _wait_until_lock_waiting(Dialect.MYSQL, monitor, sid2, done, blocker_sid=sid1)
            # Session 2 waits on the group row and has not written row 201.
            s1.cursor().execute(f"INSERT INTO {t} VALUES (102, 'G1', -1e16)")
            s1.commit()
            th.join(timeout=LOCK_WAIT_TIMEOUT_S)
            self.assertFalse(th.is_alive())
            if errors:
                raise errors[0]
            self.assertViewExact(fx.conn, plan, live, "prelock")
        finally:
            for s in sessions:
                try:
                    s.rollback()
                    s.close()
                except Exception:
                    pass
            fx.close()


# ---------------------------------------------------------------------------
# Native: backfill chunk merge interleaved with a writer
# ---------------------------------------------------------------------------


class NativeBackfill(_ExactCase):
    def test_chunk_merges_are_exact(self) -> None:
        """Each chunk adds its rows to groups that earlier chunks created; the
        merged float sums must equal the native aggregate over all rows."""
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for col_type in (_pg_types() if engine == Dialect.POSTGRES else _my_types()):
                with self.subTest(engine=engine.value, type=col_type):
                    fx = _Fixture(engine, dsn)
                    try:
                        single = col_type.upper() in ("REAL", "FLOAT")
                        big = 1e8 if single else 1e16
                        t = fx.table("bf", f"id BIGINT PRIMARY KEY, g INT, x {col_type}", group_index="g")
                        vals = [0.1, big, 0.2, -big, 1.5, 0.3, None, -0.0, 0.7, big, -0.1, 2.5]
                        fx.conn.cursor().executemany(
                            f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)",
                            [(i + 1, i % 2, v) for i, v in enumerate(vals)],
                        )
                        fx.conn.commit()
                        live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s, AVG(x) AS a, VAR_POP(x) AS vp FROM {t} GROUP BY g"
                        plan = fx.register(live, populate=False)
                        seen: List[int] = []

                        def check(batch_idx: int, _done: int) -> None:
                            seen.append(batch_idx)

                        fx.manager.backfill_incrementally(plan, batch_size=3, on_progress=check, snapshot_isolation=False)
                        self.assertGreaterEqual(len(seen), 3)
                        self.assertViewExact(fx.conn, plan, live, f"{engine.value} {col_type} backfill")
                    finally:
                        fx.close()

    def test_chunk_waits_for_writer_and_includes_it(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            for col_type in (_pg_types() if engine == Dialect.POSTGRES else _my_types()):
                with self.subTest(engine=engine.value, type=col_type):
                    self._run(engine, dsn, col_type)

    def _run(self, engine: Dialect, dsn: str, col_type: str) -> None:
        fx = _Fixture(engine, dsn)
        writer = monitor = None
        try:
            t = fx.table("bfw", f"id BIGINT PRIMARY KEY, grp VARCHAR(16) NOT NULL, amt {col_type}", group_index="grp")
            fx.conn.cursor().executemany(
                f"INSERT INTO {t} (id, grp, amt) VALUES (%s, %s, %s)",
                [(i, "G", (0.1 * i) if i % 3 else 1e7 * (-1) ** i) for i in range(1, 13)],
            )
            fx.conn.commit()
            backfill_sid = _session_id(engine, fx.conn)
            live = f"SELECT grp, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS a, STDDEV_SAMP(amt) AS ss FROM {t} GROUP BY grp"
            plan = fx.register(live, populate=False)
            writer, monitor = fx.session(), fx.session()
            writer_sid = _session_id(engine, writer)
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
                if batch_idx != 2:
                    return
                # A writer above the backfill range updates the group row and
                # stays uncommitted while the final chunk merges the group.
                writer.cursor().execute(
                    f"INSERT INTO {t} (id, grp, amt) VALUES (%s, %s, %s)", (10_000, "G", 0.3)
                )
                threading.Thread(target=commit_when_chunk_waits, daemon=True).start()

            fx.manager.backfill_incrementally(plan, batch_size=4, on_progress=on_progress, snapshot_isolation=False)
            if helper_errors:
                raise helper_errors[0]
            self.assertTrue(interleaved.is_set())
            self.assertViewExact(fx.conn, plan, live, f"{engine.value} {col_type} backfill+writer")
        finally:
            for s in (writer, monitor):
                if s is not None:
                    try:
                        s.rollback()
                        s.close()
                    except Exception:
                        pass
            fx.close()


# ---------------------------------------------------------------------------
# Native: concurrent independent writers (+ a concurrent backfill)
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "1213" in s or "deadlock" in s or "could not serialize" in s or "1205" in s


class NativeConcurrentWriters(_ExactCase):
    THREADS = 4
    TXNS = 60

    def test_concurrent_writers_with_backfill(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            configs = (
                [("DOUBLE PRECISION", False), ("DOUBLE PRECISION", True), ("REAL", False)]
                if engine == Dialect.POSTGRES
                else [("DOUBLE", False), ("FLOAT", False)]
            )
            for col_type, tt in configs:
                with self.subTest(engine=engine.value, type=col_type, tt=tt):
                    self._run(engine, dsn, col_type, tt)

    def _run(self, engine: Dialect, dsn: str, col_type: str, tt: bool) -> None:
        fx = _Fixture(engine, dsn, tt)
        try:
            single = col_type.upper() in ("REAL", "FLOAT")
            t = fx.table("cw", f"id BIGINT PRIMARY KEY, g INT NOT NULL, x {col_type}")
            # Group-key index: the native recompute reads one group's rows
            # (see the README cost note for unindexed group keys).
            fx.conn.cursor().execute(f"CREATE INDEX ix_cw_{fx.suffix} ON {t} (g)")
            seed_rng = random.Random(7)
            fx.conn.cursor().executemany(
                f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)",
                [(i, seed_rng.randrange(5), _random_value(seed_rng, single, engine == Dialect.POSTGRES)) for i in range(1, 121)],
            )
            fx.conn.commit()
            live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s, AVG(x) AS a, VAR_SAMP(x) AS vs FROM {t} GROUP BY g"
            plan = fx.register(live, populate=False)
            stop_ids = [10_000]
            id_lock = threading.Lock()
            errors: List[BaseException] = []
            retries = [0]
            start = threading.Event()

            def worker(wi: int) -> None:
                rng = random.Random(1000 + wi)
                s = fx.session()
                try:
                    start.wait(30)
                    for _ in range(self.TXNS):
                        ops = []
                        for _k in range(rng.randrange(1, 3)):
                            kind = rng.randrange(4)
                            if kind == 0:
                                with id_lock:
                                    stop_ids[0] += 1
                                    nid = stop_ids[0]
                                ops.append((f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)", (nid, rng.randrange(5), _random_value(rng, single, engine == Dialect.POSTGRES))))
                            elif kind == 1:
                                ops.append((f"DELETE FROM {t} WHERE id = %s", (rng.randrange(1, 140),)))
                            elif kind == 2:
                                ops.append((f"UPDATE {t} SET x = %s WHERE id = %s", (_random_value(rng, single, engine == Dialect.POSTGRES), rng.randrange(1, 140))))
                            else:
                                ops.append((f"UPDATE {t} SET g = %s WHERE id = %s", (rng.randrange(5), rng.randrange(1, 140))))
                        for attempt in range(50):
                            try:
                                c = s.cursor()
                                for sql, params in ops:
                                    c.execute(sql, params)
                                s.commit()
                                break
                            except Exception as exc:
                                s.rollback()
                                if "out of range" in str(exc):
                                    break  # overflowing native aggregate: the write is rejected
                                if not _is_retryable(exc) or attempt == 49:
                                    raise
                                retries[0] += 1
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    s.close()

            threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(self.THREADS)]
            for th in threads:
                th.start()

            def on_progress(batch_idx: int, _done: int) -> None:
                # Writers start once the backfill has claimed its watermark.
                start.set()

            backfill_error: Optional[BaseException] = None
            try:
                fx.manager.backfill_incrementally(plan, batch_size=10, on_progress=on_progress, snapshot_isolation=False)
            except Exception as exc:
                # KNOWN LIMIT (MySQL/MariaDB): a chunk's locking recompute can
                # deadlock with a writer that inserted a row of the same group
                # and waits for the group row the chunk holds. InnoDB rolls the
                # chunk back; the backfill must then fail closed (watermark kept,
                # view reported incomplete), never leave a wrong value.
                fx.conn.rollback()
                if engine != Dialect.MYSQL or "1213" not in repr(exc) + repr(exc.__cause__) + repr(exc.__context__):
                    raise
                # Either the raw deadlock (no batch committed yet) or the
                # fail-closed "incomplete" error after committed batches; in
                # both cases repair_view() below must restore exactness.
                backfill_error = exc
            start.set()
            for th in threads:
                th.join(timeout=300)
                self.assertFalse(th.is_alive(), "writer did not finish")
            if errors:
                raise errors[0]
            if backfill_error is not None:
                print(f"note: {engine.value} {col_type}: backfill failed closed on a deadlock; repairing")
                fx.manager.repair_view(plan.view_name)
                fx.conn.commit()
            self.assertViewExact(fx.conn, plan, live, f"{engine.value} {col_type} tt={tt} concurrent (retries={retries[0]})")
        finally:
            fx.close()


# ---------------------------------------------------------------------------
# Offline SQL-shape guards (secondary; native tests above are the evidence)
# ---------------------------------------------------------------------------


def _plan(dialect: Dialect, version: str, view: str, schema: str, *, tt: bool = False, ignore: bool = True) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        return generate_materialized_view(
            parse_table_schema(schema, dialect=dialect),
            view,
            dialect=dialect,
            engine_version=version,
            ignore_limitations=ignore,
            accept_limitations=FLOAT_OPT_IN,
            incremental_backfill=True,
        )


_SCHEMA = "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, x {ty}, k BIGINT, d DECIMAL(12,3));"
_VIEW = (
    "CREATE VIEW v AS SELECT g, COUNT(*) AS c, SUM(x) AS sx, AVG(x) AS ax, VAR_SAMP(x) AS vx, "
    "SUM(k) AS sk, SUM(d) AS sd FROM t GROUP BY g;"
)


def _stmts_assigning(stmts: Sequence[str], col: str) -> List[str]:
    return [s for s in stmts if re.search(rf"(?<![\w.]){re.escape(col)}\s*=", s)]


class OfflineShapeGuards(unittest.TestCase):
    CASES = [
        (Dialect.POSTGRES, "postgresql 15", "DOUBLE PRECISION", False),
        (Dialect.POSTGRES, "postgresql 18", "REAL", False),
        (Dialect.POSTGRES, "postgresql 18", "DOUBLE PRECISION", True),
        (Dialect.MYSQL, "mysql 8.4", "DOUBLE", False),
        (Dialect.MYSQL, "mysql 9.7", "FLOAT", False),
        (Dialect.MYSQL, "mariadb 10.11", "DOUBLE", False),
        (Dialect.MYSQL, "mariadb 12.3", "DOUBLE", False),
    ]

    def test_approximate_columns_are_never_folded(self) -> None:
        for dialect, version, ty, tt in self.CASES:
            with self.subTest(dialect=dialect.value, version=version, ty=ty, tt=tt):
                plan = _plan(dialect, version, _VIEW, _SCHEMA.format(ty=ty), tt=tt)
                stmts = list(plan.trigger_ddls) + list(plan.incremental_backfill_spec.chunk_sql_templates)
                body = "\n".join(stmts)
                for col in ("sx", "ax", "vx"):
                    # No arithmetic fold of the stored value.
                    self.assertNotRegex(body, rf"\b{col}\s*=\s*\S+\.{col}\s*[-+]", col)
                    # Recomputed with the view's own aggregate, in an UPDATE
                    # that is not the fold (upsert) statement.
                    self.assertRegex(body, rf"\b{col} = \(SELECT (SUM|AVG|VAR_SAMP)\(")
                # Exact integer / fixed-point sums stay algebraic.
                self.assertRegex(body, r"\bsk\s*=\s*\(CASE WHEN .*sk \+")
                self.assertRegex(body, r"\bsd\s*=\s*\(CASE WHEN .*sd \+")
                if dialect == Dialect.MYSQL:
                    self.assertRegex(body, r"\(SELECT SUM\(x\) FROM t WHERE .* LOCK IN SHARE MODE\)")
                else:
                    # Accumulation pinned to sequential-scan (ctid) order.
                    for agg in ("SUM", "AVG", "VAR_SAMP"):
                        self.assertRegex(body, rf"\(SELECT {agg}\(x ORDER BY t\.ctid\) FROM t WHERE ")

    def test_recompute_is_a_separate_statement_after_the_fold(self) -> None:
        for dialect, version, ty, tt in self.CASES:
            if tt:
                continue
            with self.subTest(dialect=dialect.value, version=version, ty=ty):
                plan = _plan(dialect, version, _VIEW, _SCHEMA.format(ty=ty))
                body = "\n".join(plan.trigger_ddls)
                fold = [m.start() for m in re.finditer(r"(ON CONFLICT|ON DUPLICATE KEY UPDATE)", body)]
                # PostgreSQL: one scalar subquery per column; MySQL/MariaDB:
                # one locking scan computing every column (small trigger
                # programs, one read of the group).
                recompute = [
                    m.start() for m in re.finditer(
                        r"sx = \(SELECT SUM\(x[ )]|INNER JOIN \(SELECT SUM\(x\) AS sx, AVG\(x\) AS ax, VAR_SAMP\(x\) AS vx FROM t WHERE .*? LOCK IN SHARE MODE\) AS __mv_r ON TRUE",
                        body,
                    )
                ]
                self.assertTrue(fold and recompute)
                # The fold statement itself carries no base-table subquery for
                # these columns.
                for stmt in re.split(r";\s*\n", body):
                    if "ON CONFLICT" in stmt or "ON DUPLICATE KEY UPDATE" in stmt:
                        self.assertNotRegex(stmt, r"SELECT (SUM|AVG|VAR_SAMP)\(x[ )]|__mv_r")

    def test_mysql_grouped_recompute_requires_group_index(self) -> None:
        """Without an index on the GROUP BY columns the locking recompute
        share-locks the whole table and concurrent writers deadlock (measured
        135/600 on MySQL 9.7): a per-code limitation in addition to the
        order-sensitivity one."""
        code = "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX"
        base = "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, h INT NOT NULL, x DOUBLE, k BIGINT);"
        cases = [
            (base, "SELECT g, SUM(x) AS s FROM t GROUP BY g", True),
            (base + " CREATE INDEX ix_g ON t (g);", "SELECT g, SUM(x) AS s FROM t GROUP BY g", False),
            (base + " CREATE INDEX ix_hg ON t (h, g);", "SELECT g, h, SUM(x) AS s FROM t GROUP BY g, h", False),
            (base + " CREATE INDEX ix_hg ON t (h, g);", "SELECT g, SUM(x) AS s FROM t GROUP BY g", True),
            (base, "SELECT g, SUM(k) AS s FROM t GROUP BY g", False),  # exact: folded, no locking read
            (base, "SELECT SUM(x) AS s FROM t", False),  # global: the whole table is the group
        ]
        for schema, select, refused in cases:
            for version in ("mysql 8.4", "mariadb 10.11"):
                with self.subTest(select=select, schema=schema, version=version):
                    parsed = parse_table_schema(schema, dialect=Dialect.MYSQL)
                    view = f"CREATE VIEW v AS {select};"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = generate_materialized_view(
                            parsed, view, dialect=Dialect.MYSQL, engine_version=version,
                            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                        )
                    self.assertEqual(code in {l.code for l in plan.limitations}, refused)
                    if refused:
                        # Accepting only the order-sensitivity code is not enough.
                        with self.assertRaisesRegex(EngineLimitationError, code):
                            generate_materialized_view(
                                parsed, view, dialect=Dialect.MYSQL, engine_version=version,
                                ignore_limitations=True,
                                accept_limitations=["ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"],
                            )


# ---------------------------------------------------------------------------
# Native: why the default refuses, and the fail-closed upgrade
# ---------------------------------------------------------------------------


class NativeBestEffortConsequences(_ExactCase):
    """The opt-in is best effort: the original view's float result depends on
    its plan. With the adversarial rows 1.5, 1e16, -1e16 an index-order scan
    and the recompute's scan order disagree (1.5 vs 2.0); the pinned plan
    agrees. These are the measured counterexamples behind the default
    refusal; the default refusal itself is asserted natively too."""

    def _rows(self, engine: Dialect) -> Tuple[str, str, List[Tuple[Any, ...]]]:
        ty = "DOUBLE PRECISION" if engine == Dialect.POSTGRES else "DOUBLE"
        # Primary-key / insertion order 1.5, 1e16, -1e16 (sum 2.0); y order
        # 1e16, -1e16, 1.5 (sum 1.5).
        return ty, "id BIGINT PRIMARY KEY, g INT NOT NULL, y INT, x {ty}", [
            (1, 1, 3, 1.5), (2, 1, 1, 1e16), (3, 1, 2, -1e16)
        ]

    def test_where_index_scan_differs_from_recompute_order(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            for grouped in (False, True):
                with self.subTest(engine=engine.value, grouped=grouped):
                    fx = _Fixture(engine, dsn)
                    try:
                        ty, cols, seed = self._rows(engine)
                        t = fx.table("ce", cols.format(ty=ty), group_index="g")
                        cur = fx.conn.cursor()
                        cur.execute(f"CREATE INDEX ix_y_{fx.suffix} ON {t} (y)")
                        cur.executemany(f"INSERT INTO {t} (id, g, y, x) VALUES (%s, %s, %s, %s)", seed)
                        fx.conn.commit()
                        live = (
                            f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} WHERE y < 10 GROUP BY g"
                            if grouped else f"SELECT COUNT(*) AS c, SUM(x) AS s FROM {t} WHERE y < 10"
                        )
                        # Default (and the global override): refused.
                        mgr = MaterializedViewManager(fx.conn, dialect=engine)
                        with self.assertRaisesRegex(EngineLimitationError, _NR):
                            mgr.register_view(f"CREATE VIEW d_{fx.suffix} AS {live};", ignore_limitations=True)
                        fx.conn.rollback()
                        plan = fx.register(live)
                        cur.execute(f"INSERT INTO {t} (id, g, y, x) VALUES (4, 1, 5, 0.0)")
                        fx.conn.commit()
                        cur.execute(f"SELECT * FROM {plan.view_name}")
                        mv_sum = [r for r in cur.fetchall()][0][-1]
                        fx.conn.commit()
                        # The same original view through two access paths.
                        if engine == Dialect.POSTGRES:
                            pin_native_plan(cur, engine)
                            cur.execute(live)
                            scan_sum = cur.fetchall()[0][-1]
                            fx.conn.commit()
                            cur.execute("SET LOCAL enable_seqscan = off")
                            cur.execute("SET LOCAL enable_bitmapscan = off")
                            cur.execute(live)
                        else:
                            cur.execute(live.replace(f"FROM {t}", f"FROM {t} FORCE INDEX (PRIMARY)"))
                            scan_sum = cur.fetchall()[0][-1]
                            fx.conn.commit()
                            cur.execute(live.replace(f"FROM {t}", f"FROM {t} FORCE INDEX (ix_y_{fx.suffix})"))
                        index_sum = cur.fetchall()[0][-1]
                        fx.conn.commit()
                        self.assertEqual((scan_sum, index_sum), (2.0, 1.5))
                        if engine == Dialect.POSTGRES:
                            # Recompute in ctid order: equals the pinned plan.
                            self.assertEqual(mv_sum, 2.0)
                            self.assertViewExact(fx.conn, plan, live, "pinned plan")
                        else:
                            # MySQL/MariaDB recompute in the access-path order of
                            # their own locking read, which may itself use the
                            # WHERE index: equal to one of the two plans.
                            self.assertIn(mv_sum, (2.0, 1.5))
                    finally:
                        fx.close()


class NativeWindowExactTypes(_ExactCase):
    """Window SUM/AVG/COUNT of exact arguments keep the engine's result type
    (NUMERIC / DECIMAL), so they are compared exactly with no budget (stored
    as DOUBLE they returned 53.333333333333336 for PostgreSQL's
    53.3333333333333333, and lost 2**53 + 1)."""

    def test_window_aggregates_of_exact_arguments(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    t = fx.table("w", "id INT PRIMARY KEY, team VARCHAR(10), score INT, n NUMERIC(10,2), b BIGINT")
                    cur = fx.conn.cursor()
                    cur.execute(
                        f"INSERT INTO {t} VALUES (1,'a',50,1.25,9007199254740993),(2,'a',80,2.5,1),"
                        f"(4,'a',30,0.01,2),(3,'b',80,1,3)"
                    )
                    fx.conn.commit()
                    live = (
                        f"SELECT id, AVG(score) OVER (PARTITION BY team ORDER BY id) AS ra, "
                        f"SUM(score) OVER (PARTITION BY team) AS ss, AVG(n) OVER (PARTITION BY team) AS an, "
                        f"SUM(b) OVER () AS sb, COUNT(*) OVER (PARTITION BY team) AS cn FROM {t}"
                    )
                    plan = fx.register(live)
                    self.assertViewExact(fx.conn, plan, live, "initial")
                    for stmt in (
                        f"INSERT INTO {t} VALUES (5,'b',95,3.333,4)",
                        f"UPDATE {t} SET score = 31, b = b + 1 WHERE id = 4",
                        f"DELETE FROM {t} WHERE id = 2",
                    ):
                        cur.execute(stmt)
                        fx.conn.commit()
                        self.assertViewExact(fx.conn, plan, live, stmt)
                finally:
                    fx.close()


class NativeMysqlGroupIndexRecognition(_ExactCase):
    """MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX is decided from the
    introspected base-table indexes: an index created with CREATE INDEX or
    declared inline (KEY, composite with the group columns leading) satisfies
    it; none, or one whose leading column is not a group column, does not.
    (User-declared indexes= are on the backing table and never count.)"""

    def test_introspected_indexes(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        cases = [
            ("", None, True),
            ("", "CREATE INDEX ix_{s} ON {t} (g)", False),
            (", KEY k_g (g)", None, False),
            (", KEY k_gh (g, h)", None, False),
            (", KEY k_hg (h, g)", None, True),
        ]
        for inline, ddl, refused in cases:
            with self.subTest(inline=inline, ddl=ddl):
                fx = _Fixture(Dialect.MYSQL, dsn)
                try:
                    t = fx.table("gi", f"id BIGINT PRIMARY KEY, g INT NOT NULL, h INT NOT NULL, x DOUBLE{inline}")
                    if ddl:
                        fx.conn.cursor().execute(ddl.format(s=fx.suffix, t=t))
                        fx.conn.commit()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = generate_materialized_view(
                            fx.conn, f"CREATE VIEW v AS SELECT g, SUM(x) AS s FROM {t} GROUP BY g;",
                            dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                        )
                    self.assertEqual("MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX" in {l.code for l in plan.limitations}, refused)
                finally:
                    fx.close()


class NativeRepairNeverBypasses(_ExactCase):
    """repair_view() rebuilds a drifted view under exactly the limitation
    codes stored with it: when a base-table change makes the definition
    order-sensitive (an exact column becomes DOUBLE), the rebuild is refused
    instead of silently installing best-effort maintenance; with the code
    accepted explicitly it is rebuilt."""

    def test_acceptance_does_not_carry_over_to_a_new_definition(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    ty = "DOUBLE PRECISION" if engine == Dialect.POSTGRES else "DOUBLE"
                    t = fx.table("cd", f"id BIGINT PRIMARY KEY, g INT NOT NULL, k INT, x {ty}", group_index="g")
                    fx.conn.cursor().execute(f"INSERT INTO {t} VALUES (1, 1, 1, 1.5), (2, 1, 2, 2.5)")
                    fx.conn.commit()
                    live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} GROUP BY g"
                    plan = fx.register(live)
                    name = plan.view_name
                    mgr = MaterializedViewManager(fx.conn, dialect=engine, ignore_limitations=True)
                    other = f"CREATE VIEW {name} AS SELECT g, AVG(x) AS a, VAR_SAMP(x) AS vs FROM {t} WHERE k > 0 GROUP BY g;"
                    for call in (lambda: mgr.register_view(other), lambda: mgr.sync_views([other])):
                        with self.assertRaisesRegex(EngineLimitationError, _NR):
                            call()
                        fx.conn.rollback()
                    self.assertViewExact(fx.conn, plan, live, "installed view unchanged")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        mgr.register_view(f"CREATE VIEW {name} AS {live};")  # same definition: stored codes
                    fx.conn.commit()
                finally:
                    fx.close()

    def test_repair_after_type_change_refuses_without_acceptance(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    t = fx.table("rp", "id BIGINT PRIMARY KEY, g INT NOT NULL, x DECIMAL(12,3)", group_index="g")
                    cur = fx.conn.cursor()
                    cur.execute(f"INSERT INTO {t} VALUES (1, 1, 1.5), (2, 1, 2.25)")
                    fx.conn.commit()
                    live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} GROUP BY g"
                    mgr = MaterializedViewManager(fx.conn, dialect=engine, ignore_limitations=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = mgr.register_view(f"CREATE VIEW v_rp_{fx.suffix} AS {live};")
                    fx.manager, fx.plans = mgr, [plan]
                    fx.conn.commit()
                    self.assertNotIn(_NR, mgr.get_catalog()[plan.view_name]["accepted_limitations"])
                    if engine == Dialect.POSTGRES:
                        # ALTER COLUMN TYPE is refused while maintenance triggers
                        # depend on the column: re-create the table instead.
                        cur.execute(f"DROP TABLE {t}")
                        cur.execute(f"CREATE TABLE {t} (id BIGINT PRIMARY KEY, g INT NOT NULL, x DOUBLE PRECISION)")
                        cur.execute(f"INSERT INTO {t} VALUES (1, 1, 1.5), (2, 1, 2.25)")
                    else:
                        cur.execute(f"ALTER TABLE {t} MODIFY x DOUBLE")
                    fx.conn.commit()
                    self.assertEqual(mgr.check_schema_drift()[plan.view_name], "invalid")
                    with self.assertRaisesRegex(EngineLimitationError, _NR):
                        mgr.repair_view(plan.view_name)
                    fx.conn.rollback()
                    self.assertNotIn(_NR, mgr.get_catalog()[plan.view_name]["accepted_limitations"])
                finally:
                    fx.close()


# ---------------------------------------------------------------------------
# Offline: default refusal and the per-code opt-in
# ---------------------------------------------------------------------------

_NR = "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"
_NRF = "ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE"
_GI = "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX"

_SHAPE_SCHEMA = (
    "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, y INT, x {ty}, k BIGINT, d DECIMAL(12,3));"
    "CREATE TABLE u (id BIGINT PRIMARY KEY, tid BIGINT NOT NULL, z {ty}, n BIGINT);"
    "CREATE INDEX ix_t_g ON t (g);"
)

# (label, approximate shape, recomputed race-free under the opt-in)
_ORDER_SENSITIVE_SHAPES = [
    ("grouped", "SELECT g, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g", True),
    ("grouped_where", "SELECT g, COUNT(*) AS c, SUM(x) AS s FROM t WHERE y < 10 GROUP BY g", True),
    ("global", "SELECT COUNT(*) AS c, SUM(x) AS s FROM t", True),
    ("global_where", "SELECT COUNT(*) AS c, SUM(x) AS s FROM t WHERE y < 10", True),
    ("global_avg", "SELECT AVG(x) AS a FROM t", True),
    ("var", "SELECT g, VAR_POP(x) AS v, STDDEV_SAMP(x) AS sd FROM t GROUP BY g", True),
    ("expression", "SELECT g, SUM(x * 3 + k) AS s FROM t GROUP BY g", True),
    ("having", "SELECT g, SUM(x) AS s FROM t GROUP BY g HAVING COUNT(*) > 1", True),
    ("distinct_mix", "SELECT g, SUM(DISTINCT x) AS sd, SUM(x) AS s FROM t GROUP BY g", True),
    ("distinct_only", "SELECT g, SUM(DISTINCT x) AS sd FROM t GROUP BY g", True),
    ("window_sum", "SELECT id, g, SUM(x) OVER (PARTITION BY g) AS s FROM t", False),
    ("window_avg", "SELECT id, AVG(x) OVER (PARTITION BY g) AS a FROM t", False),
    ("join", "SELECT t.g, SUM(u.z) AS s FROM t JOIN u ON u.tid = t.id GROUP BY t.g", False),
    ("left_join", "SELECT t.g, COUNT(*) AS c, SUM(u.z) AS s FROM t LEFT JOIN u ON u.tid = t.id GROUP BY t.g", False),
]

# Exactly maintainable: no order-sensitive code on any engine.
_EXACT_SHAPES = [
    "SELECT g, COUNT(*) AS c, SUM(k) AS sk, SUM(d) AS sd FROM t GROUP BY g",
    "SELECT COUNT(*) AS c, SUM(k) AS sk, SUM(d) AS sd FROM t WHERE y < 10",
    "SELECT g, AVG(d) AS ad, MIN(x) AS mn, MAX(x) AS mx, COUNT(x) AS cx FROM t GROUP BY g",
    "SELECT id, SUM(k) OVER (PARTITION BY g) AS s FROM t",
    "SELECT t.g, SUM(u.n) AS s FROM t JOIN u ON u.tid = t.id GROUP BY t.g",
]

_ENGINES = [
    (Dialect.POSTGRES, "postgresql 15", ("DOUBLE PRECISION", "REAL")),
    (Dialect.POSTGRES, "postgresql 18", ("DOUBLE PRECISION",)),
    (Dialect.MYSQL, "mysql 8.4", ("DOUBLE", "FLOAT")),
    (Dialect.MYSQL, "mysql 9.7", ("DOUBLE",)),
    (Dialect.MYSQL, "mariadb 10.11", ("DOUBLE",)),
    (Dialect.MYSQL, "mariadb 11.8", ("DOUBLE",)),
    (Dialect.MYSQL, "mariadb 12.3", ("DOUBLE",)),
]


def _codes(err_or_plan: Any) -> set:
    return {lim.code for lim in err_or_plan.limitations}


class OfflineDefaultRefusal(unittest.TestCase):
    """Every approximate order-sensitive aggregate shape is refused by default
    on PostgreSQL/MySQL/MariaDB, the global override does not accept it, and
    the per-code opt-in installs it with a warning stating the best-effort
    contract (the original view's value depends on its plan: an index or
    range scan for WHERE, sort-based grouping, parallel aggregation)."""

    def test_order_sensitive_shapes_need_per_code_acceptance(self) -> None:
        for dialect, version, types in _ENGINES:
            for ty in types:
                parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty=ty), dialect=dialect)
                for label, select, recomputed in _ORDER_SENSITIVE_SHAPES:
                    view = f"CREATE VIEW v AS {select};"
                    with self.subTest(version=version, ty=ty, shape=label):
                        with self.assertRaises(EngineLimitationError) as default_ctx:
                            generate_materialized_view(parsed, view, dialect=dialect, engine_version=version)
                        self.assertIn(_NR, _codes(default_ctx.exception))
                        self.assertEqual(_NRF in _codes(default_ctx.exception), not recomputed)
                        # ignore_limitations alone never accepts per-code limitations.
                        with self.assertRaises(EngineLimitationError) as global_ctx:
                            generate_materialized_view(
                                parsed, view, dialect=dialect, engine_version=version, ignore_limitations=True
                            )
                        self.assertTrue(_codes(global_ctx.exception) <= set(PER_CODE_OPT_IN_LIMITATIONS))
                        self.assertIn(_NR, _codes(global_ctx.exception))
                        self.assertIn("accept_limitations", str(global_ctx.exception))
                        needed = sorted(_codes(global_ctx.exception))
                        # Accepting all but one needed code still refuses.
                        for missing in needed:
                            with self.assertRaisesRegex(EngineLimitationError, missing):
                                generate_materialized_view(
                                    parsed, view, dialect=dialect, engine_version=version,
                                    ignore_limitations=True,
                                    accept_limitations=[c for c in needed if c != missing],
                                )
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", EngineLimitationWarning)
                            plan = generate_materialized_view(
                                parsed, view, dialect=dialect, engine_version=version,
                                ignore_limitations=dialect == Dialect.MYSQL, accept_limitations=needed,
                            )
                        self.assertTrue(set(needed) <= _codes(plan))
                        text = "\n".join(str(w.message) for w in caught)
                        self.assertIn("BEST-EFFORT", text)
                        self.assertIn("accept_limitations", text)

    def test_exact_types_stay_supported(self) -> None:
        for dialect, version, types in _ENGINES:
            parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty=types[0]), dialect=dialect)
            for select in _EXACT_SHAPES:
                with self.subTest(version=version, select=select):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = generate_materialized_view(
                            parsed, f"CREATE VIEW v AS {select};", dialect=dialect, engine_version=version,
                            ignore_limitations=dialect == Dialect.MYSQL,
                        )
                    self.assertFalse(_codes(plan) & set(PER_CODE_OPT_IN_LIMITATIONS), _codes(plan))

    def test_sqlite_float_aggregates_need_no_opt_in(self) -> None:
        parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty="REAL"), dialect=Dialect.SQLITE)
        for label, select, _ in _ORDER_SENSITIVE_SHAPES:
            with self.subTest(shape=label):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", EngineLimitationWarning)
                    plan = generate_materialized_view(parsed, f"CREATE VIEW v AS {select};", dialect=Dialect.SQLITE)
                self.assertFalse(_codes(plan) & set(PER_CODE_OPT_IN_LIMITATIONS))

    def test_mysql_float_variance_over_integers_is_order_sensitive(self) -> None:
        # MySQL/MariaDB VAR/STDDEV accumulate in DOUBLE for every argument type.
        parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty="DOUBLE"), dialect=Dialect.MYSQL)
        with self.assertRaisesRegex(EngineLimitationError, _NR):
            generate_materialized_view(
                parsed, "CREATE VIEW v AS SELECT g, VAR_POP(k) AS v FROM t GROUP BY g;",
                dialect=Dialect.MYSQL, engine_version="mysql 8.4", ignore_limitations=True,
            )

    def test_cli_accept_limitation_flag(self) -> None:
        import contextlib
        import io

        from matview_sync import main as cli_main

        schema = _SHAPE_SCHEMA.format(ty="DOUBLE PRECISION")
        view = "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM t GROUP BY g;"
        base = ["--schema", schema, "--view", view, "--dialect", "postgresql:18"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertNotEqual(cli_main(base), 0)
            self.assertNotEqual(cli_main(base + ["--ignore-limitations"]), 0)
        self.assertIn("--accept-limitation", err.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            self.assertEqual(cli_main(base + ["--accept-limitation", _NR]), 0)
        self.assertIn("CREATE", out.getvalue())


class OfflineAggregateOrderBy(unittest.TestCase):
    def test_aggregate_order_by_is_refused_before_install(self) -> None:
        """SUM(x ORDER BY id) compiled into maintenance SQL that failed at
        install with a syntax error on every engine (also before this
        change); it is refused up front. Ordered string aggregates keep
        their ORDER BY."""
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            parsed = parse_table_schema("CREATE TABLE t (id BIGINT PRIMARY KEY, g INT, x BIGINT)", dialect=dialect)
            for agg in ("SUM(x ORDER BY id)", "COUNT(x ORDER BY id)", "AVG(x ORDER BY id DESC)", "MAX(x ORDER BY id)"):
                with self.subTest(dialect=dialect.value, agg=agg):
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "ORDER BY inside"):
                        generate_materialized_view(
                            parsed, f"CREATE VIEW v AS SELECT g, {agg} AS a FROM t GROUP BY g;",
                            dialect=dialect, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                        )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                generate_materialized_view(
                    parsed, "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM t GROUP BY g ORDER BY g;",
                    dialect=dialect, ignore_limitations=True,
                )
                generate_materialized_view(
                    parsed, "CREATE VIEW v AS SELECT g, GROUP_CONCAT(x ORDER BY id) AS s FROM t GROUP BY g;",
                    dialect=dialect, ignore_limitations=True,
                )


# ---------------------------------------------------------------------------
# Offline (transpiled engine emulation): stored acceptance, repair never
# bypasses limitations
# ---------------------------------------------------------------------------


class OfflineManagerAcceptance(unittest.TestCase):
    """Manager contract, exercised through the SQLite-backed PostgreSQL/MySQL
    emulation of test_matview_sync (structural: SQL semantics are SQLite's).
    The native variant is NativeRepairNeverBypasses."""

    VIEW = "CREATE VIEW v_up AS SELECT g, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g;"

    def _manager(self, dialect: Dialect, **kw: Any) -> Tuple[Any, Any]:
        from test_matview_sync import TranspiledConnectionAdapter

        conn = TranspiledConnectionAdapter(dialect)
        mgr = MaterializedViewManager(conn, dialect=dialect, **kw)
        ty = "DOUBLE PRECISION" if dialect == Dialect.POSTGRES else "DOUBLE"
        mgr.apply_schema_ddl(
            f"CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, x {ty});"
            "CREATE INDEX ix_t_g ON t (g);",
            auto_sync_views=False,
        )
        conn.execute("INSERT INTO t VALUES (1, 1, 1.5), (2, 1, 2.5), (3, 2, 4.0)")
        conn.commit()
        return conn, mgr

    def _install(self, mgr: Any, dialect: Dialect) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            return mgr.register_view(
                self.VIEW, ignore_limitations=dialect == Dialect.MYSQL, accept_limitations=FLOAT_OPT_IN
            )

    def _reader_exists(self, conn: Any) -> bool:
        try:
            conn.execute("SELECT * FROM v_up").fetchall()
            return True
        except Exception:
            conn.rollback()
            return False

    def test_global_override_does_not_accept(self) -> None:
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect, ignore_limitations=True)
                with self.assertRaisesRegex(EngineLimitationError, _NR):
                    mgr.register_view(self.VIEW)
                with self.assertRaisesRegex(EngineLimitationError, _NR):
                    mgr.sync_views([self.VIEW])
                self.assertEqual(mgr.get_catalog(), {})

    def test_acceptance_is_stored_and_reused(self) -> None:
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect)
                self._install(mgr, dialect)
                entry = mgr.get_catalog()["v_up"]
                self.assertIn(_NR, entry["accepted_limitations"])
                # A fresh manager without any override keeps the stored acceptance.
                mgr2 = MaterializedViewManager(conn, dialect=dialect)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    self.assertEqual(mgr2.sync_views([self.VIEW]), {"v_up": "unchanged"})
                    mgr2.register_view(self.VIEW)
                    self.assertEqual(mgr2.check_schema_drift(), {"v_up": "ok"})
                    self.assertEqual(mgr2.repair_view("v_up"), {"v_up": "repaired"})
                    # A rebuild re-installs under the stored codes (MySQL/MariaDB
                    # refuse non-atomic replacement of a published view).
                    if dialect == Dialect.POSTGRES:
                        conn.execute(f"UPDATE {mgr2.catalog_table} SET signature_hash = 'stale'")
                        conn.commit()
                        self.assertEqual(mgr2.repair_view("v_up"), {"v_up": "rebuilt"})
                self.assertIn(_NR, mgr2.get_catalog()["v_up"]["accepted_limitations"])
                # Explicitly declining the code refuses without dropping.
                with self.assertRaises(EngineLimitationError) as ctx:
                    mgr2.register_view(self.VIEW, force_rebuild=True, accept_limitations=[],
                                       ignore_limitations=True)
                self.assertIn("v_up", mgr2.get_catalog())

    def test_repair_uses_only_stored_codes(self) -> None:
        """repair_view() never overrides a per-code limitation the view was
        not installed with (structural: the stored codes are edited)."""
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            for rebuild in (False, True):
                with self.subTest(dialect=dialect.value, rebuild=rebuild):
                    conn, mgr = self._manager(dialect)
                    self._install(mgr, dialect)
                    entry = mgr.get_catalog()["v_up"]
                    stored = {
                        "definition": entry["accepted_definition"],
                        "codes": [c for c in entry["accepted_limitations"] if c != _NR],
                    }
                    conn.execute(
                        f"UPDATE {mgr.catalog_table} SET accepted_limitations_json = '{json.dumps(stored)}'"
                        + (", signature_hash = 'stale'" if rebuild else "")
                    )
                    conn.commit()
                    other = MaterializedViewManager(conn, dialect=dialect, ignore_limitations=True)
                    with self.assertRaisesRegex(EngineLimitationError, _NR):
                        other.repair_view("v_up")
                    with self.assertRaisesRegex(EngineLimitationError, _NR):
                        other.repair_view()
                    self.assertEqual(other.check_schema_drift(), {"v_up": "invalid"})

    def test_acceptance_is_bound_to_the_definition(self) -> None:
        """Codes accepted for one definition are not reused for another
        definition registered under the same name (register, force_rebuild,
        sync): the new definition needs its own explicit acceptance."""
        other_def = "CREATE VIEW v_up AS SELECT g, AVG(x) AS a, VAR_SAMP(x) AS vs FROM t WHERE id > 0 GROUP BY g;"
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            for op in ("register_view", "force_rebuild", "sync_views"):
                with self.subTest(dialect=dialect.value, op=op):
                    conn, mgr = self._manager(dialect, ignore_limitations=True)
                    self._install(mgr, dialect)
                    before = mgr.get_catalog()["v_up"]
                    fresh = MaterializedViewManager(conn, dialect=dialect, ignore_limitations=True)
                    with self.assertRaisesRegex(EngineLimitationError, _NR):
                        if op == "register_view":
                            fresh.register_view(other_def)
                        elif op == "force_rebuild":
                            fresh.register_view(other_def, force_rebuild=True)
                        else:
                            fresh.sync_views([other_def])
                    after = fresh.get_catalog()["v_up"]
                    self.assertEqual(after["signature_hash"], before["signature_hash"])
                    self.assertTrue(self._reader_exists(conn))
                    # Same definition still reuses the stored codes; the new
                    # one installs with its own explicit acceptance.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        fresh.register_view(self.VIEW)
                        if dialect == Dialect.MYSQL:
                            fresh.drop_view("v_up")  # no atomic replacement on MySQL
                        fresh.register_view(other_def, accept_limitations=FLOAT_OPT_IN)
                    self.assertIn(_NR, fresh.get_catalog()["v_up"]["accepted_limitations"])

    def test_acceptance_record_for_another_definition_fails_closed(self) -> None:
        """A catalog row whose accepted codes were recorded for a different
        definition accepts nothing: repair_view, bulk_changes (MySQL/MariaDB
        and SQLite, where it rebuilds) and check_schema_drift refuse / report
        invalid."""
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect)
                self._install(mgr, dialect)
                entry = mgr.get_catalog()["v_up"]
                forged = {"definition": "0" * 64, "codes": entry["accepted_limitations"]}
                conn.execute(f"UPDATE {mgr.catalog_table} SET accepted_limitations_json = '{json.dumps(forged)}'")
                conn.commit()
                other = MaterializedViewManager(conn, dialect=dialect, ignore_limitations=True)
                e2 = other.get_catalog()["v_up"]
                self.assertTrue(e2["accepted_limitations_mismatch"])
                self.assertEqual(e2["accepted_limitations"], [])
                self.assertEqual(other.check_schema_drift(), {"v_up": "invalid"})
                with self.assertRaisesRegex(EngineLimitationError, _NR):
                    other.repair_view("v_up")
                if dialect == Dialect.MYSQL:
                    # PostgreSQL bulk_changes() never suspends or refreshes the
                    # statement-level maintenance, so it compiles no plan.
                    with self.assertRaisesRegex(EngineLimitationError, _NR):
                        with other.bulk_changes():
                            pass

    def test_refused_new_definition_keeps_installed_view(self) -> None:
        """Replacing an installed (exact) view with a refused definition
        raises the ordinary refusal and leaves the installed view alone; only
        an INSTALLED definition that is now refused fails closed."""
        exact = "CREATE VIEW v_up AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;"
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            if True:
                for op in ("register_view", "sync_views"):
                    with self.subTest(dialect=dialect.value, op=op):
                        conn, mgr = self._manager(dialect, ignore_limitations=True)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            mgr.register_view(exact)
                        with self.assertRaises(EngineLimitationError) as ctx:
                            if op == "register_view":
                                mgr.register_view(self.VIEW, force_rebuild=True)
                            else:
                                mgr.sync_views([self.VIEW])
                        self.assertIn(_NR, str(ctx.exception))
                        self.assertIn("v_up", mgr.get_catalog())
                        self.assertTrue(self._reader_exists(conn))

if __name__ == "__main__":
    unittest.main(verbosity=2)
