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
EngineLimitationError (FLOAT_REASSOCIATION_ALLOWED) and the
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
    MaintenanceStrategy,
    generate_materialized_view,
    parse_table_schema,
)
from matview.diagnostics import (
    EXACT_LEAD,
    FLOAT_EXACT_REFRESH,
    FLOAT_RELAXED_MAINTENANCE,
    RELAXED_LEAD,
    MatviewDiagnosticWarning,
    Strategy,
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

    def grp_text(self) -> str:
        """A text group-key column type that incremental maintenance supports.

        MySQL's default utf8mb4_unicode_ci is a PAD SPACE UCA collation under
        which GROUP BY is plan-dependent, so such a key forces the exact full
        refresh (COLLATION_KEY_FALLBACK) whatever the float opt-in; the relaxed
        incremental paths are exercised with a byte-exact key instead."""
        return "VARCHAR(16) COLLATE utf8mb4_bin" if self.engine == Dialect.MYSQL else "VARCHAR(16)"

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

    def register(self, select_sql: str, relaxed: bool = True, expect_diagnostic: Optional[bool] = None,
                 **kw: Any) -> Any:
        """Register a view. ``relaxed`` (the historical default of this suite)
        opts in to FLOAT_REASSOCIATION_ALLOWED (incremental maintenance,
        asserted exact here only under the pinned plan); ``relaxed=False``
        keeps the exact default (full refresh of the original query)."""
        if self.manager is None:
            self.manager = MaterializedViewManager(
                self.conn, dialect=self.engine
            )
        name = f"v{len(self.plans)}_{self.suffix}"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = self.manager.register_view(
                f"CREATE VIEW {name} AS {select_sql};",
                ignore_limitations=True,
                accept_limitations=FLOAT_OPT_IN if relaxed else None,
                **kw,
            )
        float_codes = (FLOAT_EXACT_REFRESH, FLOAT_RELAXED_MAINTENANCE)
        codes = [d.code for d in _diag_warnings(caught) if d.code in float_codes]
        plan_codes = [d.code for d in plan.strategy_diagnostics if d.code in float_codes]
        expected = FLOAT_RELAXED_MAINTENANCE if relaxed else FLOAT_EXACT_REFRESH
        if expect_diagnostic is True and (codes != [expected] or plan_codes != [expected]):
            # A known floating-point view must announce its strategy in both
            # the plan and the warning: never silent.
            raise AssertionError(f"expected diagnostic {expected}: warnings {codes}, plan {plan_codes}")
        if expect_diagnostic is False and (codes or plan_codes):
            raise AssertionError(f"expected no strategy diagnostic: warnings {codes}, plan {plan_codes}")
        if plan_codes and codes != [expected]:
            raise AssertionError(f"expected diagnostic {expected}, got {codes}")
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
            t = fx.table("race", f"id BIGINT PRIMARY KEY, grp {fx.grp_text()} NOT NULL, amt {col_type}", group_index="grp")
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
        deadlocks (1213, retried here) rare, and the result stays exact.

        A ratio threshold on a timing-dependent count (failed once at 17 > 15
        under load): extended lane only (MATVIEW_TIMING_TESTS=1 in
        ci-extended.yml), not the per-change gate."""
        if os.environ.get("MATVIEW_TIMING_TESTS") != "1":
            self.skipTest("timing-sensitive rate test: set MATVIEW_TIMING_TESTS=1 (extended lane)")
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
            t = fx.table("pl", f"id BIGINT PRIMARY KEY, grp {fx.grp_text()} NOT NULL, amt DOUBLE", group_index="grp")
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
            t = fx.table("bfw", f"id BIGINT PRIMARY KEY, grp {fx.grp_text()} NOT NULL, amt {col_type}", group_index="grp")
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
                                accept_limitations=["FLOAT_REASSOCIATION_ALLOWED"],
                            )


# ---------------------------------------------------------------------------
# Native: why the default refuses, and the fail-closed upgrade
# ---------------------------------------------------------------------------


class NativeDefaultExactRefresh(_ExactCase):
    """Default mode, natively: a view with a floating-point order-sensitive
    aggregate is maintained by a full refresh of its original query and
    equals the live view exactly (no pinned plan) in cases where any
    incremental maintenance differs: cancellation (1e16 + 1.5 - 1e16,
    0.1 + 0.2 + 0.3 - 0.2), a WHERE served by an index (rows 1.5, 1e16,
    -1e16: live 1.5, scan order 2.0), windows and joins. The relaxed mode
    differs on the same histories (its weaker contract)."""

    def _engines(self) -> List[Tuple[Dialect, str]]:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        return engines

    def _ty(self, engine: Dialect) -> str:
        return "DOUBLE PRECISION" if engine == Dialect.POSTGRES else "DOUBLE"

    def _live_equals(self, fx: "_Fixture", plan: Any, live: str) -> bool:
        cur = fx.conn.cursor()
        fx.conn.commit()
        cur.execute(f"SELECT * FROM {plan.view_name}")
        mv = _rows(cur.fetchall())
        cur.execute(live)
        lv = _rows(cur.fetchall())
        fx.conn.commit()
        return mv == lv

    def test_cancellation_histories(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    t = fx.table("ca", f"id BIGINT PRIMARY KEY, g INT NOT NULL, x {self._ty(engine)}", group_index="g")
                    live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s, AVG(x) AS a FROM {t} GROUP BY g"
                    exact = fx.register(live, relaxed=False, expect_diagnostic=True)
                    self.assertEqual(exact.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    cur = fx.conn.cursor()
                    for sql, params in (
                        (f"INSERT INTO {t} VALUES (%s, %s, %s)", (1, 1, 1e16)),
                        (f"INSERT INTO {t} VALUES (%s, %s, %s)", (2, 1, 1.5)),
                        (f"DELETE FROM {t} WHERE id = %s", (1,)),
                        (f"INSERT INTO {t} VALUES (%s, %s, %s), (%s, %s, %s), (%s, %s, %s)",
                         (3, 2, 0.1, 4, 2, 0.2, 5, 2, 0.3)),
                        (f"DELETE FROM {t} WHERE id = %s", (4,)),
                        (f"UPDATE {t} SET g = 2 WHERE id = %s", (2,)),
                    ):
                        cur.execute(sql, params)
                        fx.conn.commit()
                        self.assertTrue(self._live_equals(fx, exact, live), f"{engine.value}: {sql} {params}")
                finally:
                    fx.close()

    def test_where_index_plan_is_followed_exactly(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    t = fx.table("wi", f"id BIGINT PRIMARY KEY, g INT NOT NULL, y INT, x {self._ty(engine)}", group_index="g")
                    cur = fx.conn.cursor()
                    cur.execute(f"CREATE INDEX ix_y_{fx.suffix} ON {t} (y)")
                    # Many non-matching rows make the planner use the y index.
                    cur.executemany(
                        f"INSERT INTO {t} VALUES (%s, %s, %s, %s)",
                        [(i, 1 + i % 3, 1000 + i, 0.0) for i in range(10, 2010)]
                        + [(1, 1, 3, 1.5), (2, 1, 1, 1e16), (3, 1, 2, -1e16)],
                    )
                    cur.execute(f"ANALYZE {t}" if engine == Dialect.POSTGRES else f"ANALYZE TABLE {t}")
                    if engine == Dialect.MYSQL:
                        cur.fetchall()
                    if engine == Dialect.POSTGRES:
                        # Precondition made explicit: the original query reads
                        # through the y index (the refresh, run in this
                        # session's writes, uses the same setting).
                        cur.execute("SET enable_seqscan = off")
                    fx.conn.commit()
                    live = f"SELECT COUNT(*) AS c, SUM(x) AS s FROM {t} WHERE y < 10"
                    exact = fx.register(live, relaxed=False, expect_diagnostic=True)
                    relaxed = fx.register(live, relaxed=True, expect_diagnostic=True)
                    cur.execute(f"INSERT INTO {t} VALUES (4, 1, 5, 0.0)")
                    fx.conn.commit()
                    cur.execute(live)
                    live_sum = cur.fetchall()[0][-1]
                    fx.conn.commit()
                    self.assertTrue(self._live_equals(fx, exact, live))
                    cur.execute(f"SELECT s FROM {relaxed.view_name}")
                    relaxed_sum = cur.fetchall()[0][0]
                    fx.conn.commit()
                    if engine == Dialect.POSTGRES:
                        # The plan reads through the y index (index order:
                        # 1.5, 1e16, -1e16 -> 1.5); the relaxed recompute
                        # (pinned to ctid order) gives 2.0. (MySQL/MariaDB's
                        # relaxed locking read may itself use the y index, so
                        # either value is allowed there.)
                        self.assertEqual(live_sum, 1.5, "precondition: index-ordered plan")
                        self.assertEqual(relaxed_sum, 2.0)
                finally:
                    fx.close()

    def test_windows_and_joins(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    ty = self._ty(engine)
                    a = fx.table("ja", f"id BIGINT PRIMARY KEY, g INT NOT NULL, x {ty}", group_index="g")
                    b = fx.table("jb", f"id BIGINT PRIMARY KEY, aid BIGINT NOT NULL, y {ty}")
                    join_view = (
                        f"SELECT {a}.g, COUNT(*) AS c, SUM({b}.y) AS s FROM {a} JOIN {b} ON {b}.aid = {a}.id GROUP BY {a}.g"
                    )
                    views = [f"SELECT id, g, SUM(x) OVER (PARTITION BY g) AS s FROM {a}"]
                    if engine == Dialect.MYSQL:
                        # MySQL/MariaDB: a later read can use another join order, so a
                        # float sum over a join is refused under the exact default.
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                            fx.register(join_view, relaxed=False)
                        fx.conn.rollback()
                    else:
                        views.append(join_view)
                    plans = [(fx.register(v, relaxed=False, expect_diagnostic=True), v) for v in views]
                    for p, _ in plans:
                        self.assertEqual(p.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    cur = fx.conn.cursor()
                    for sql in (
                        f"INSERT INTO {a} VALUES (1, 1, 1e16), (2, 1, 1.5), (3, 2, 0.1)",
                        f"INSERT INTO {b} VALUES (1, 1, 1e16), (2, 1, 1.5), (3, 3, 0.3), (4, 3, 0.2)",
                        f"DELETE FROM {a} WHERE id = 1",
                        f"DELETE FROM {b} WHERE id = 1",
                        f"UPDATE {b} SET aid = 2 WHERE id = 4",
                    ):
                        cur.execute(sql)
                        fx.conn.commit()
                        for p, v in plans:
                            self.assertTrue(self._live_equals(fx, p, v), f"{engine.value}: {sql}: {v}")
                finally:
                    fx.close()

    def test_nested_shapes_are_exact(self) -> None:
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    ty = self._ty(engine)
                    t = fx.table("na", f"id BIGINT PRIMARY KEY, g INT NOT NULL, x {ty}", group_index="g")
                    u = fx.table("nb", f"id BIGINT PRIMARY KEY, g INT NOT NULL, y {ty}", group_index="g")
                    views = [
                        f"WITH c AS (SELECT g, SUM(x) AS s FROM {t} GROUP BY g) SELECT g, s FROM c",
                        f"SELECT g, s FROM (SELECT g, AVG(x) AS s FROM {t} GROUP BY g) AS d",
                        f"SELECT id, (SELECT SUM(y) FROM {u} WHERE {u}.g = {t}.g) AS sy FROM {t}",
                        f"SELECT g, COUNT(*) AS c FROM {t} GROUP BY g HAVING SUM(x) > 1",
                        f"SELECT g, SUM(x) AS s FROM {t} GROUP BY g UNION ALL SELECT g, SUM(y) FROM {u} GROUP BY g",
                    ]
                    plans = [(fx.register(v, relaxed=False, expect_diagnostic=True), v) for v in views]
                    cur = fx.conn.cursor()
                    for sql in (
                        f"INSERT INTO {t} VALUES (1, 1, 1e16), (2, 1, 1.5), (3, 2, 0.1), (4, 2, 0.2), (5, 2, 0.3)",
                        f"INSERT INTO {u} VALUES (1, 1, 1e16), (2, 1, 1.5), (3, 2, 0.3)",
                        f"DELETE FROM {t} WHERE id IN (1, 4)",
                        f"DELETE FROM {u} WHERE id = 1",
                    ):
                        cur.execute(sql)
                        fx.conn.commit()
                        for p, v in plans:
                            self.assertTrue(self._live_equals(fx, p, v), f"{engine.value}: {sql}: {v}")
                finally:
                    fx.close()

    def test_two_writers_serialize(self) -> None:
        """Second writer waits for the first (lock on the view's state) and
        the full refresh then sees both: exact after both commit."""
        for engine, dsn in self._engines():
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    t = fx.table("cw", f"id BIGINT PRIMARY KEY, g INT NOT NULL, x {self._ty(engine)}", group_index="g")
                    live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} GROUP BY g"
                    plan = fx.register(live, relaxed=False, expect_diagnostic=True)
                    s1, s2 = fx.session(), fx.session()
                    try:
                        c1, c2 = s1.cursor(), s2.cursor()
                        sid1, sid2 = _session_id(engine, s1), _session_id(engine, s2)
                        c1.execute(f"INSERT INTO {t} VALUES (1, 1, 1e16)")
                        done = threading.Event()
                        err: List[BaseException] = []

                        def second() -> None:
                            try:
                                c2.execute(f"INSERT INTO {t} VALUES (2, 1, 1.5)")
                                s2.commit()
                            except BaseException as e:  # noqa: BLE001
                                err.append(e)
                            finally:
                                done.set()

                        th = threading.Thread(target=second)
                        th.start()
                        _wait_until_lock_waiting(engine, fx.conn, sid2, done, sid1)
                        self.assertFalse(done.is_set())
                        s1.commit()
                        th.join(LOCK_WAIT_TIMEOUT_S)
                        self.assertFalse(th.is_alive(), "second writer did not finish after the first committed")
                        self.assertFalse(err, err)
                    finally:
                        s1.close()
                        s2.close()
                    self.assertTrue(self._live_equals(fx, plan, live))
                finally:
                    fx.close()


class NativeMysqlStatisticsDefault(_ExactCase):
    """Default mode, natively, VAR/STDDEV on MySQL/MariaDB over every argument
    family: DOUBLE, FLOAT(M,D), DOUBLE(M,D), DECIMAL(12,5), DECIMAL(40,30),
    INT; grouped by a nullable key and global; NULL values and empty groups.

    MariaDB rounds VAR/STDDEV over exact or FLOAT(M,D)/DOUBLE(M,D) arguments
    to fixed decimals that a stored double cannot reproduce (measured: live
    7-8 decimals; DECIMAL(40,30) 1 ulp; global DECIMAL(12,5) tens of ulps
    after ROUND), so those views (and wrappers such as COALESCE) are refused,
    with or without the relaxed opt-in for wrappers; DOUBLE arguments and the
    documented remedy CAST(arg AS DOUBLE) refresh and equal the live view
    exactly. MySQL computes all of them in double precision: every shape
    refreshes and is exact. FLOAT(M,D) is declared on the live table, so the
    view's schema comes from live introspection (which must keep (M,D))."""

    _ARGS = ("d", "fm", "dm", "n5", "n30", "k")
    _FIXED = ("fm", "dm", "n5", "n30", "k")

    def _mysql(self) -> str:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("set MYSQL_DSN")
        return dsn

    def _shapes(self, t: str) -> List[Tuple[str, str]]:
        fns = ("VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP")
        out: List[Tuple[str, str]] = []
        for c in self._ARGS:
            cols = ", ".join(f"{fn}({c}) AS {fn.lower()}_{c}" for fn in fns)
            out.append((c, f"SELECT g, COUNT(*) AS n, {cols} FROM {t} GROUP BY g"))
            out.append((c, f"SELECT {cols} FROM {t}"))
        out.append(("fm", f"SELECT g, COALESCE(STDDEV_POP(fm), 0) AS s FROM {t} GROUP BY g"))
        out.append(("cast", f"SELECT g, VAR_SAMP(CAST(fm AS DOUBLE)) AS v, "
                            f"STDDEV_POP(CAST(n5 AS DOUBLE)) AS s FROM {t} GROUP BY g"))
        return out

    def test_statistics_refresh_exact_or_refused(self) -> None:
        dsn = self._mysql()
        fx = _Fixture(Dialect.MYSQL, dsn)
        try:
            cur = fx.conn.cursor()
            cur.execute("SELECT VERSION()")
            mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
            fx.conn.commit()
            t = fx.table(
                "st",
                "id BIGINT PRIMARY KEY, g INT, d DOUBLE, fm FLOAT(7,3), dm DOUBLE(12,4), "
                "n5 DECIMAL(12,5), n30 DECIMAL(40,30), k INT",
            )
            accepted: List[Tuple[Any, str]] = []
            for arg, sel in self._shapes(t):
                refused = mariadb and arg in self._FIXED
                with self.subTest(shape=sel[:60]):
                    if refused:
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "on MariaDB"):
                            fx.register(sel, relaxed=False)
                        fx.conn.rollback()
                        if sel.startswith(f"SELECT g, COALESCE"):
                            # A wrapper is refused under the relaxed opt-in too.
                            with self.assertRaisesRegex(UnsupportedMaterializedViewError, "on MariaDB"):
                                fx.register(sel, relaxed=True)
                            fx.conn.rollback()
                        continue
                    plan = fx.register(sel, relaxed=False)
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    accepted.append((plan, sel))
            self.assertTrue(accepted)
            rng = random.Random(255)
            vals = (None, 0, 1, -1, 1.5, 2.25, 123.456, 1e-3, 999.5, 7)
            ids: List[int] = []
            for step in range(40):
                op = rng.random()
                if op < 0.55 or not ids:
                    i = 1 + len(ids) + step * 7
                    ids.append(i)
                    v = [rng.choice(vals) for _ in range(5)]
                    cur.execute(
                        f"INSERT INTO {t} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        (i, rng.choice((1, 2, 3, None)), v[0], v[1], v[2], v[3], v[4],
                         None if v[0] is None else rng.randrange(-3, 5)),
                    )
                elif op < 0.8:
                    cur.execute(
                        f"UPDATE {t} SET n5 = %s, fm = %s, d = %s, g = %s WHERE id = %s",
                        (rng.choice(vals), rng.choice(vals), rng.choice(vals),
                         rng.choice((1, 2, 3, 4)), rng.choice(ids)),
                    )
                else:
                    cur.execute(f"DELETE FROM {t} WHERE id = %s", (ids.pop(rng.randrange(len(ids))),))
                fx.conn.commit()
                for plan, sel in accepted:
                    cur.execute(f"SELECT * FROM {plan.view_name}")
                    mv = _rows(cur.fetchall())
                    cur.execute(sel)
                    live = _rows(cur.fetchall())
                    fx.conn.commit()
                    self.assertEqual(mv, live, f"step {step}: {sel}")
            # Empty table: global statistics are NULL, groups disappear.
            cur.execute(f"DELETE FROM {t}")
            fx.conn.commit()
            for plan, sel in accepted:
                self.assertViewExact(fx.conn, plan, sel, f"empty: {sel}")
        finally:
            fx.close()


class NativeUnionAllMixedTypes(_ExactCase):
    """Default mode, natively: a set operation in a derived table or CTE
    whose branches mix integer/decimal and floating-point columns has the
    COMMON type of all branches (live: float8/real, DOUBLE). The stored
    columns must take that type, not the first branch's (regression: INT
    first branch stored SUM/AVG as BIGINT/NUMERIC or DECIMAL while the live
    view returned DOUBLE). Result column types and values (Python type and
    exact value) are compared with the live view after every write."""

    def _shapes(self, t: str, u: str) -> List[str]:
        return [
            f"SELECT g, SUM(x) AS s, AVG(x) AS a, COUNT(x) AS c FROM (SELECT g, k AS x FROM {t} UNION ALL SELECT g, f FROM {u}) q GROUP BY g",
            f"SELECT g, SUM(x) AS s, AVG(x) AS a FROM (SELECT g, n AS x FROM {t} UNION ALL SELECT g, r FROM {u}) q GROUP BY g",
            f"SELECT g, SUM(x) AS s, AVG(x) AS a FROM (SELECT g, f AS x FROM {t} UNION ALL SELECT g, k FROM {u}) q GROUP BY g",
            f"SELECT g, SUM(x) AS s FROM (SELECT g, k AS x FROM {t} UNION ALL SELECT g, k FROM {u} "
            f"UNION ALL SELECT g, f FROM {t}) q GROUP BY g",
            f"SELECT g, SUM(x) AS s, AVG(x) AS a FROM (SELECT g, k AS x FROM {t} UNION ALL SELECT g, NULL FROM {u} "
            f"UNION ALL SELECT g, f FROM {u}) q GROUP BY g",
            f"WITH q AS (SELECT g, k AS x FROM {t} UNION ALL SELECT g, f FROM {u}) "
            f"SELECT g, SUM(x) AS s, AVG(x) AS a FROM q GROUP BY g",
            f"SELECT g, x FROM (SELECT g, k AS x FROM {t} UNION ALL SELECT g, f FROM {u}) q",
        ]

    def _q(self, conn: Any, sql: str) -> Tuple[List[Any], List[Tuple[Any, ...]]]:
        cur = conn.cursor()
        cur.execute(sql)
        types = [d[1] for d in cur.description]
        rows = sorted(
            (tuple((type(v).__name__, _exact(v)) for v in r) for r in cur.fetchall()), key=repr
        )
        conn.commit()
        return types, rows

    def test_mixed_branch_types_match_live(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    pg = engine == Dialect.POSTGRES
                    f_t, r_t = ("DOUBLE PRECISION", "REAL") if pg else ("DOUBLE", "FLOAT")
                    n_t = "NUMERIC(10,2)" if pg else "DECIMAL(10,2)"
                    cols = f"id BIGINT PRIMARY KEY, g INT, k INT, n {n_t}, f {f_t}, r {r_t}"
                    t, u = fx.table("ua", cols), fx.table("ub", cols)
                    plans = [(fx.register(sel, relaxed=False), sel) for sel in self._shapes(t, u)]
                    rng = random.Random(280)
                    vals = (1e16, -1e16, 1.5, 0.1, 0.2, 0.3, 2.5, None)
                    cur = fx.conn.cursor()
                    applied = 0
                    for step in range(30):
                        tb = rng.choice((t, u))
                        i = rng.randrange(1, 25)
                        op = rng.random()
                        try:
                            if op < 0.55:
                                cur.execute(
                                    f"INSERT INTO {tb} VALUES (%s,%s,%s,%s,%s,%s)",
                                    (i, rng.randrange(3), rng.randrange(-5, 9), rng.choice((1.25, -3.5, None)),
                                     rng.choice(vals), rng.choice((1.5, 0.1, None))),
                                )
                            elif op < 0.8:
                                cur.execute(f"UPDATE {tb} SET f = %s, k = %s, g = %s WHERE id = %s",
                                            (rng.choice(vals), rng.randrange(-5, 9), rng.randrange(3), i))
                            else:
                                cur.execute(f"DELETE FROM {tb} WHERE id = %s", (i,))
                            applied += 1 if cur.rowcount and cur.rowcount > 0 else 0
                            fx.conn.commit()
                        except Exception as exc:
                            # Only a duplicate primary key is an expected
                            # rejection; anything else (a failing trigger)
                            # must fail the test.
                            if type(exc).__name__ not in ("UniqueViolation", "IntegrityError"):
                                raise
                            fx.conn.rollback()
                            cur = fx.conn.cursor()
                        for plan, sel in plans:
                            self.assertEqual(
                                self._q(fx.conn, f"SELECT * FROM {plan.view_name}"), self._q(fx.conn, sel),
                                f"{engine.value} step {step}: {sel}",
                            )
                    self.assertGreaterEqual(applied, 15, "too few writes changed rows")
                finally:
                    fx.close()


class NativeSubqueryScopes(_ExactCase):
    """Natively, default mode: a column inside a subquery resolves in the
    subquery's own scope first (inner u.x DOUBLE, not the outer t.x INT of
    the same name). Aggregates over the inner float are classified floating
    point (exact full refresh, FLOAT_EXACT_REFRESH) and projected scalar
    subqueries store the live type; correlated and exact shapes keep exact
    types. A projected subquery aggregating a derived table's column (type
    not provable) is refused. Types and values compared exactly with the
    live view after every write."""

    def test_subquery_scopes(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    f_t = "DOUBLE PRECISION" if engine == Dialect.POSTGRES else "DOUBLE"
                    t = fx.table("so", "id BIGINT PRIMARY KEY, g INT, x INT")
                    u = fx.table("si", f"id BIGINT PRIMARY KEY, g INT, x {f_t}, k INT")
                    float_views = [
                        f"SELECT g, COUNT(*) AS n, (SELECT SUM(x) FROM {u}) AS s FROM {t} GROUP BY g",
                        f"SELECT g, (SELECT AVG(x) FROM {u}) AS a FROM {t} GROUP BY g",
                        f"SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING (SELECT SUM(x) FROM {u}) > 0.25",
                        f"SELECT g, SUM(x) AS s FROM {t} WHERE x < (SELECT SUM(x) FROM {u}) GROUP BY g",
                        f"SELECT g, (SELECT SUM({u}.x) FROM {u} WHERE {u}.g = {t}.g) AS s FROM {t} GROUP BY g",
                        # alias shadowing: inner alias named like the outer table
                        f"SELECT g, (SELECT SUM(z.x) FROM {u} z) AS s FROM {t} z GROUP BY g",
                    ]
                    exact_views = [
                        f"SELECT g, COUNT(*) AS n, (SELECT SUM(k) FROM {u}) AS s FROM {t} GROUP BY g",
                        f"SELECT g, (SELECT SUM({u}.k) FROM {u} WHERE {u}.g = {t}.g) AS s FROM {t} GROUP BY g",
                    ]
                    plans = [(fx.register(v, relaxed=False, expect_diagnostic=True), v) for v in float_views]
                    plans += [(fx.register(v, relaxed=False), v) for v in exact_views]
                    # A subquery in the projection whose aggregate reads a derived
                    # table is typed by neither the typed-expression inference nor a
                    # provable declared column: refused (server-derived types lift it).
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "projected subquery"):
                        fx.register(
                            f"SELECT g, (SELECT SUM(x) FROM (SELECT x FROM {u}) d) AS s FROM {t} GROUP BY g",
                            relaxed=False,
                        )
                    fx.conn.rollback()
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "derived table"):
                        fx.register(
                            f"SELECT g, (SELECT SUM(x) FROM (SELECT x * 2 AS x FROM {u}) d) AS s FROM {t} GROUP BY g",
                            relaxed=False,
                        )
                    fx.conn.rollback()
                    cur = fx.conn.cursor()
                    rng = random.Random(255)
                    writes = [
                        (f"INSERT INTO {t} VALUES (%s, %s, %s)", (1, 1, 5)),
                        (f"INSERT INTO {t} VALUES (%s, %s, %s)", (2, 2, 7)),
                    ]
                    for i in range(1, 13):
                        writes.append((f"INSERT INTO {u} VALUES (%s, %s, %s, %s)",
                                       (i, rng.choice((1, 2)), rng.choice((1e16, -1e16, 0.1, 0.2, 0.3, 1.5)), i)))
                    writes += [(f"DELETE FROM {u} WHERE id = %s", (i,)) for i in (2, 5, 7)]
                    writes.append((f"UPDATE {t} SET x = %s WHERE id = %s", (1, 1)))
                    for sql, params in writes:
                        cur.execute(sql, params)
                        fx.conn.commit()
                        for plan, sel in plans:
                            mv_cur = fx.conn.cursor()
                            mv_cur.execute(f"SELECT * FROM {plan.view_name}")
                            mv = sorted((tuple((type(v).__name__, _exact(v)) for v in r) for r in mv_cur.fetchall()), key=repr)
                            mv_cur.execute(sel)
                            lv = sorted((tuple((type(v).__name__, _exact(v)) for v in r) for r in mv_cur.fetchall()), key=repr)
                            fx.conn.commit()
                            self.assertEqual(mv, lv, f"{engine.value}: {sql} {params}: {sel}")
                finally:
                    fx.close()


class NativeRecursiveCteSources(_ExactCase):
    """Natively, default mode: SUM/AVG over a recursive CTE's column is
    refused (the live view returns int on PostgreSQL and Decimal on MySQL/
    MariaDB for an INT anchor and a `v * 2` term, Decimal for `v * 1.5`; the
    refresh stored float). A recursive CTE joined to a table with only
    exact aggregates is accepted and equals the live view (types and values)."""

    def test_recursive_cte_sources(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                fx = _Fixture(engine, dsn)
                try:
                    int_t = "INT" if engine == Dialect.POSTGRES else "SIGNED"
                    t = fx.table("rc", "id BIGINT PRIMARY KEY, g INT, k INT")
                    for term in ("v * 2", "v * 1.5"):
                        sel = (
                            f"WITH RECURSIVE c(n, v) AS (SELECT 1, CAST(1 AS {int_t}) UNION ALL "
                            f"SELECT n + 1, {term} FROM c WHERE n < 3) "
                            f"SELECT {t}.g, SUM(c.v + {t}.k) AS s FROM {t} CROSS JOIN c GROUP BY {t}.g"
                        )
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "recursive CTE|table instances"):
                            fx.register(sel, relaxed=False)
                        fx.conn.rollback()
                    ok_sel = (
                        f"WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 3) "
                        f"SELECT {t}.g, COUNT(*) AS cnt, SUM({t}.k) AS sk FROM {t} CROSS JOIN c GROUP BY {t}.g"
                    )
                    plan = fx.register(ok_sel, relaxed=False, expect_diagnostic=False)
                    cur = fx.conn.cursor()
                    for sql, params in (
                        (f"INSERT INTO {t} VALUES (%s, %s, %s), (%s, %s, %s)", (1, 1, 5, 2, 1, 7)),
                        (f"INSERT INTO {t} VALUES (%s, %s, %s)", (3, 2, 2147483647)),
                        (f"UPDATE {t} SET g = %s WHERE id = %s", (2, 1)),
                        (f"DELETE FROM {t} WHERE id = %s", (2,)),
                    ):
                        cur.execute(sql, params)
                        fx.conn.commit()
                        mv_cur = fx.conn.cursor()
                        mv_cur.execute(f"SELECT * FROM {plan.view_name}")
                        mv = sorted(tuple((type(v).__name__, _exact(v)) for v in r) for r in mv_cur.fetchall())
                        mv_cur.execute(ok_sel)
                        lv = sorted(tuple((type(v).__name__, _exact(v)) for v in r) for r in mv_cur.fetchall())
                        fx.conn.commit()
                        self.assertEqual(mv, lv, f"{engine.value}: {sql}")
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


class NativeMysqlCollationFallbackDominatesRelaxed(_ExactCase):
    """A group key whose collation incremental maintenance cannot reproduce
    (MySQL's default utf8mb4_unicode_ci: PAD SPACE UCA, plan-dependent
    grouping) forces the exact full refresh of the original query
    (COLLATION_KEY_FALLBACK) even with the float opt-in: the opt-in only
    permits a cheaper strategy, it never overrides another exactness reason.
    The result stays equal to the live view after every transition."""

    def test_relaxed_opt_in_with_pad_space_uca_key_is_an_exact_full_refresh(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        fx = _Fixture(Dialect.MYSQL, dsn)
        try:
            cur = fx.conn.cursor()
            cur.execute("SELECT VERSION()")
            is_mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
            fx.conn.commit()
            if is_mariadb:
                # The fallback is MySQL's: MariaDB groups utf8mb4_unicode_ci keys
                # through a reproducible collation key, so the relaxed incremental
                # path applies there (covered by the other relaxed tests).
                self.skipTest("MySQL-only collation fallback (PAD SPACE UCA grouping is plan-dependent on MySQL)")
            t = fx.table(
                "cf", "id BIGINT PRIMARY KEY, grp VARCHAR(16) COLLATE utf8mb4_unicode_ci NOT NULL, amt DOUBLE",
                group_index="grp",
            )
            fx.conn.cursor().executemany(
                f"INSERT INTO {t} VALUES (%s, %s, %s)", [(1, "G1", 1e16), (2, "G1", 1.5), (3, "G2", 0.1)]
            )
            fx.conn.commit()
            live = f"SELECT grp, COUNT(*) AS c, SUM(amt) AS s, AVG(amt) AS a FROM {t} GROUP BY grp"
            fx.manager = MaterializedViewManager(fx.conn, dialect=Dialect.MYSQL)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plan = fx.manager.register_view(
                    f"CREATE VIEW vcf_{fx.suffix} AS {live};", ignore_limitations=True, accept_limitations=FLOAT_OPT_IN
                )
            fx.conn.commit()
            fx.plans.append(plan)
            self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
            codes = {d.code for d in plan.strategy_diagnostics}
            self.assertIn("COLLATION_KEY_FALLBACK", codes)
            self.assertNotIn(FLOAT_RELAXED_MAINTENANCE, codes)
            self.assertViewExact(fx.conn, plan, live, "populate")
            cur = fx.conn.cursor()
            for i, stmt in enumerate((
                f"INSERT INTO {t} VALUES (4, 'G1', -1e16)",
                f"UPDATE {t} SET amt = amt * 3 WHERE grp = 'G2'",
                f"DELETE FROM {t} WHERE id = 1",
                f"UPDATE {t} SET grp = 'G2' WHERE id = 2",
            )):
                cur.execute(stmt)
                fx.conn.commit()
                self.assertViewExact(fx.conn, plan, live, f"step {i}")
        finally:
            fx.close()


class NativeRepairNeverBypasses(_ExactCase):
    """repair_view() rebuilds under exactly the stored codes, and a stored
    relaxed acceptance never carries over to another definition: a schema
    change (DECIMAL -> DOUBLE) or a new definition gets the exact default
    (full refresh), not the relaxed mode."""

    def test_new_definition_gets_the_exact_default(self) -> None:
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
                    plan = fx.register(f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} GROUP BY g")
                    name = plan.view_name
                    mgr = MaterializedViewManager(fx.conn, dialect=engine, ignore_limitations=True)
                    other = f"SELECT g, AVG(x) AS a, VAR_SAMP(x) AS vs FROM {t} WHERE k > 0 GROUP BY g"
                    if engine == Dialect.MYSQL:
                        mgr.drop_view(name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        new_plan = mgr.register_view(f"CREATE VIEW {name} AS {other};")
                    fx.conn.commit()
                    self.assertEqual(new_plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    self.assertNotIn(_NR, mgr.get_catalog()[name]["accepted_limitations"])
                finally:
                    fx.close()

    def test_repair_after_type_change_uses_the_exact_default(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set (MySQL refuses non-atomic replacement)")
        fx = _Fixture(Dialect.POSTGRES, dsn)
        try:
            t = fx.table("rp", "id BIGINT PRIMARY KEY, g INT NOT NULL, x DECIMAL(12,3)")
            cur = fx.conn.cursor()
            cur.execute(f"INSERT INTO {t} VALUES (1, 1, 1.5), (2, 1, 2.25)")
            fx.conn.commit()
            live = f"SELECT g, COUNT(*) AS c, SUM(x) AS s FROM {t} GROUP BY g"
            mgr = MaterializedViewManager(fx.conn, dialect=Dialect.POSTGRES)
            plan = mgr.register_view(f"CREATE VIEW v_rp_{fx.suffix} AS {live};")
            fx.manager, fx.plans = mgr, [plan]
            fx.conn.commit()
            self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
            # ALTER COLUMN TYPE is refused while maintenance triggers depend
            # on the column: re-create the table instead.
            cur.execute(f"DROP TABLE {t}")
            cur.execute(f"CREATE TABLE {t} (id BIGINT PRIMARY KEY, g INT NOT NULL, x DOUBLE PRECISION)")
            cur.execute(f"INSERT INTO {t} VALUES (1, 1, 1e16), (2, 1, 1.5)")
            fx.conn.commit()
            self.assertEqual(mgr.check_schema_drift()[plan.view_name], "drifted")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(mgr.repair_view(plan.view_name), {plan.view_name: "rebuilt"})
            self.assertEqual(mgr.get_catalog()[plan.view_name]["strategy"], MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
            cur.execute(f"DELETE FROM {t} WHERE id = 1")
            fx.conn.commit()
            self.assertViewExact(fx.conn, plan, live, "after repair")
        finally:
            fx.close()


# ---------------------------------------------------------------------------
# Offline: default refusal and the per-code opt-in
# ---------------------------------------------------------------------------

_NR = "FLOAT_REASSOCIATION_ALLOWED"
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


def _diag_warnings(caught: Sequence[Any]) -> List[Any]:
    return [w.message.diagnostic for w in caught if issubclass(w.category, MatviewDiagnosticWarning)]


class OfflineDefaultExactRefresh(unittest.TestCase):
    """Default (exact) mode: every approximate order-sensitive aggregate
    shape on PostgreSQL/MySQL/MariaDB is maintained by a full refresh of the
    view's original query, with a FLOAT_EXACT_REFRESH diagnostic (lead phrase
    and strategy name from matview.diagnostics). The global override does not
    change that; only the per-code FLOAT_REASSOCIATION_ALLOWED selects the
    relaxed incremental mode, with a FLOAT_RELAXED_MAINTENANCE diagnostic."""

    def test_recursive_cte_sources_are_refused_and_never_recurse(self) -> None:
        """A recursive CTE column's type is the engine's union of its anchor
        and recursive terms, which the exact full refresh cannot prove from
        declared columns: an order-sensitive aggregate over it is refused
        (it would store float where the view returns int/Decimal). Typing
        self- and mutually-referencing CTEs terminates (no RecursionError),
        and aggregates that are not floating point keep working."""
        cases = (
            ("postgres:18", "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT, k INT, f DOUBLE PRECISION);", "INT"),
            ("mysql:8.4", "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT, k INT, f DOUBLE);", "SIGNED"),
            ("mariadb:11.8", "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT, k INT, f DOUBLE);", "SIGNED"),
        )
        for dialect, schema, int_t in cases:
            for term in ("v * 2", "v * 1.5", "v + t2.f"):
                body = (
                    f"WITH RECURSIVE c(n, v) AS (SELECT 1, CAST(1 AS {int_t}) UNION ALL "
                    f"SELECT n + 1, {term} FROM c, t t2 WHERE n < 3 AND t2.id = 1) "
                    f"SELECT t.g, SUM(c.v + t.k) AS s, AVG(c.v) AS a FROM t CROSS JOIN c GROUP BY t.g"
                )
                with self.subTest(dialect=dialect, term=term):
                    # MySQL/MariaDB refuse the 2+ table float sum first (plan dependence);
                    # PostgreSQL reports the unprovable recursive CTE column type.
                    reason = "table instances|recursive CTE" if not dialect.startswith("postgres") else "recursive CTE"
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, reason):
                        generate_materialized_view(schema, f"CREATE VIEW v AS {body};", dialect=dialect,
                                                   ignore_limitations=True)
            mutual = (
                "WITH RECURSIVE a(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM b WHERE n < 3), "
                "b(n) AS (SELECT n FROM a UNION ALL SELECT n FROM a) "
            )
            # Both CTE columns are provably INT (an integer literal anchor and a
            # `n + 1` step over the previous level, passed through b): the
            # resolver terminates, and SUM(INT * 1.5) is exact decimal arithmetic.
            for agg, refused in (("SUM(b.n * 1.5)", False), ("COUNT(*)", False)):
                body = f"{mutual}SELECT t.g, {agg} AS x FROM t CROSS JOIN b GROUP BY t.g"
                with self.subTest(dialect=dialect, mutual=agg):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            generate_materialized_view(schema, f"CREATE VIEW v AS {body};", dialect=dialect,
                                                       ignore_limitations=True)
                        outcome = "accepted"
                    except UnsupportedMaterializedViewError:
                        outcome = "refused"
                    if refused:
                        self.assertEqual(outcome, "refused")
            self_ref = (
                "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 3) "
                "SELECT t.g, COUNT(*) AS x, SUM(t.k) AS sk FROM t CROSS JOIN c GROUP BY t.g"
            )
            with self.subTest(dialect=dialect, self_ref=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plan = generate_materialized_view(schema, f"CREATE VIEW v AS {self_ref};", dialect=dialect,
                                                      ignore_limitations=True)
                # Exact (no float diagnostic); the full refresh itself is announced.
                self.assertEqual([d.code for d in plan.strategy_diagnostics], ["FULL_REFRESH_FALLBACK"])

    def test_default_is_full_refresh_with_diagnostic(self) -> None:
        for dialect, version, types in _ENGINES:
            for ty in types:
                parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty=ty), dialect=dialect)
                for label, select, _recomputed in _ORDER_SENSITIVE_SHAPES:
                    view = f"CREATE VIEW v AS {select};"
                    with self.subTest(version=version, ty=ty, shape=label):
                        if dialect == Dialect.MYSQL and label in ("join", "left_join"):
                            # A float sum over a join: another join order on a later read
                            # can differ in the last bits (MySQL/MariaDB): refused.
                            with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                                generate_materialized_view(
                                    parsed, view, dialect=dialect, engine_version=version, ignore_limitations=True,
                                )
                            continue
                        for ignore in (False, True):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                plan = generate_materialized_view(
                                    parsed, view, dialect=dialect, engine_version=version,
                                    ignore_limitations=ignore or dialect == Dialect.MYSQL,
                                )
                            self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                            self.assertFalse(_codes(plan) & set(PER_CODE_OPT_IN_LIMITATIONS))
                            diags = plan.strategy_diagnostics
                            self.assertEqual([d.code for d in diags], [FLOAT_EXACT_REFRESH])
                            d = diags[0]
                            self.assertEqual(d.strategy, Strategy.FULL_REFRESH)
                            self.assertTrue(d.message.startswith(f"{EXACT_LEAD} {Strategy.FULL_REFRESH.value}."))
                            self.assertIn("not associative", d.message)
                            self.assertIn("per written ROW" if dialect == Dialect.MYSQL else "per writing statement",
                                          d.message)
                            self.assertIn("DECIMAL/NUMERIC", d.message)
                            self.assertIn("FLOAT_REASSOCIATION_ALLOWED", d.message)
                            self.assertTrue(d.expressions)
                            self.assertEqual(_diag_warnings(caught), diags)

    def test_relaxed_mode_needs_per_code_acceptance(self) -> None:
        for dialect, version, types in _ENGINES:
            parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty=types[0]), dialect=dialect)
            for label, select, recomputed in _ORDER_SENSITIVE_SHAPES:
                view = f"CREATE VIEW v AS {select};"
                with self.subTest(version=version, shape=label):
                    needed = {_NR}
                    if not recomputed:
                        needed.add(_NRF)
                    # Accepting only the relaxed code where a second one is
                    # needed refuses.
                    if not recomputed:
                        with self.assertRaisesRegex(EngineLimitationError, _NRF):
                            generate_materialized_view(
                                parsed, view, dialect=dialect, engine_version=version,
                                ignore_limitations=True, accept_limitations=[_NR],
                            )
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        plan = generate_materialized_view(
                            parsed, view, dialect=dialect, engine_version=version,
                            ignore_limitations=dialect == Dialect.MYSQL,
                            accept_limitations=sorted(needed | {_GI}),
                        )
                    self.assertNotEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    self.assertIn(_NR, _codes(plan))
                    diags = plan.strategy_diagnostics
                    self.assertEqual([d.code for d in diags], [FLOAT_RELAXED_MAINTENANCE])
                    expected = (
                        Strategy.ALGEBRAIC_DELTA_NATIVE_RECOMPUTE if recomputed
                        else Strategy.GROUP_SCOPED_RECOMPUTE if plan.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA
                        else Strategy.ALGEBRAIC_DELTA
                    )
                    self.assertEqual(diags[0].strategy, expected)
                    self.assertTrue(diags[0].message.startswith(f"{RELAXED_LEAD} {expected.value}."))
                    self.assertIn("may differ from the live view with no fixed error bound", diags[0].message)
                    self.assertEqual(_diag_warnings(caught), diags)

    def test_exact_types_have_no_diagnostic(self) -> None:
        for dialect, version, types in _ENGINES:
            parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty=types[0]), dialect=dialect)
            for select in _EXACT_SHAPES:
                with self.subTest(version=version, select=select):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        plan = generate_materialized_view(
                            parsed, f"CREATE VIEW v AS {select};", dialect=dialect, engine_version=version,
                            ignore_limitations=dialect == Dialect.MYSQL,
                        )
                    self.assertEqual(plan.strategy_diagnostics, [])
                    self.assertEqual(_diag_warnings(caught), [])
                    self.assertFalse(_codes(plan) & set(PER_CODE_OPT_IN_LIMITATIONS))

    def test_mixed_view_names_only_the_float_columns(self) -> None:
        parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty="DOUBLE PRECISION"), dialect=Dialect.POSTGRES)
        plan = generate_materialized_view(
            parsed, "CREATE VIEW v AS SELECT g, SUM(k) AS sk, SUM(d) AS sd, SUM(x) AS sx, AVG(x) AS ax FROM t GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        (d,) = plan.strategy_diagnostics
        self.assertEqual(set(d.columns), {"sx", "ax"})
        self.assertEqual(set(d.expressions), {"SUM(x)", "AVG(x)"})

    def test_nested_float_aggregates_are_refreshed_and_announced(self) -> None:
        """A float order-sensitive aggregate anywhere in the query (CTE,
        derived table, scalar subquery, HAVING, set-operation branch) gets the
        full refresh and the diagnostic, whatever the relaxed acceptance."""
        schema = (
            "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, x {ty}, k BIGINT);"
            "CREATE TABLE u (id BIGINT PRIMARY KEY, g INT NOT NULL, y {ty});"
        )
        shapes = [
            ("cte", "WITH c AS (SELECT g, SUM(x) AS s FROM t GROUP BY g) SELECT g, s FROM c", "SUM(x)"),
            ("derived", "SELECT g, s FROM (SELECT g, AVG(x) AS s FROM t GROUP BY g) AS d", "AVG(x)"),
            ("scalar_subquery", "SELECT id, (SELECT SUM(y) FROM u WHERE u.g = t.g) AS sy FROM t", "SUM(y)"),
            ("having", "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING SUM(x) > 1", "SUM(x)"),
            ("union_branch", "SELECT g, SUM(x) AS s FROM t GROUP BY g UNION ALL SELECT g, SUM(y) FROM u GROUP BY g", "SUM(y)"),
        ]
        for dialect, version, types in _ENGINES[::2]:
            parsed = parse_table_schema(schema.format(ty=types[0]), dialect=dialect)
            for label, select, expr in shapes:
                for accept in ([], [_NR, _NRF, _GI]):
                    with self.subTest(version=version, shape=label, relaxed=bool(accept)):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            plan = generate_materialized_view(
                                parsed, f"CREATE VIEW v AS {select};", dialect=dialect, engine_version=version,
                                ignore_limitations=True, accept_limitations=accept,
                            )
                        (d,) = plan.strategy_diagnostics
                        self.assertEqual(_diag_warnings(caught), [d])
                        if not accept or plan.strategy == MaintenanceStrategy.ATOMIC_QUERY_SYNC:
                            # Default, or a shape only a refresh can maintain.
                            self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                            self.assertEqual((d.code, d.strategy), (FLOAT_EXACT_REFRESH, Strategy.FULL_REFRESH))
                            self.assertTrue(str(d).startswith(EXACT_LEAD))
                        else:
                            self.assertEqual(d.code, FLOAT_RELAXED_MAINTENANCE)
                            self.assertTrue(str(d).startswith(RELAXED_LEAD))
                        self.assertTrue(any(expr in e for e in d.expressions), d.expressions)
            # Exact nested aggregates: no floating-point diagnostic.
            plan = generate_materialized_view(
                parsed, "CREATE VIEW v AS WITH c AS (SELECT g, SUM(k) AS s FROM t GROUP BY g) SELECT g, s FROM c;",
                dialect=dialect, engine_version=version, ignore_limitations=True,
            )
            # Exact (no float diagnostic); the full refresh itself is announced.
            self.assertEqual([d.code for d in plan.strategy_diagnostics], ["FULL_REFRESH_FALLBACK"])

    def test_sqlite_has_no_diagnostic(self) -> None:
        parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty="REAL"), dialect=Dialect.SQLITE)
        for label, select, _ in _ORDER_SENSITIVE_SHAPES:
            with self.subTest(shape=label):
                plan = generate_materialized_view(parsed, f"CREATE VIEW v AS {select};", dialect=Dialect.SQLITE)
                self.assertEqual(plan.strategy_diagnostics, [])

    def test_silence_diagnostics_is_per_code(self) -> None:
        parsed = parse_table_schema(_SHAPE_SCHEMA.format(ty="DOUBLE PRECISION"), dialect=Dialect.POSTGRES)
        view = "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM t GROUP BY g;"
        for silence, emitted in (([FLOAT_EXACT_REFRESH], 0), (["OTHER_CODE"], 1), ([], 1)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plan = generate_materialized_view(
                    parsed, view, dialect=Dialect.POSTGRES, silence_diagnostics=silence, ignore_limitations=True
                )
            self.assertEqual(len(_diag_warnings(caught)), emitted)
            self.assertEqual(len(plan.strategy_diagnostics), 1)

    def test_cli_text_and_json(self) -> None:
        import contextlib
        import io
        import json as _json

        from matview_sync import main as cli_main

        schema = _SHAPE_SCHEMA.format(ty="DOUBLE PRECISION")
        view = "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM t GROUP BY g;"
        for extra, code, lead, strat in (
            ([], FLOAT_EXACT_REFRESH, EXACT_LEAD, Strategy.FULL_REFRESH),
            (["--accept-limitation", _NR], FLOAT_RELAXED_MAINTENANCE, RELAXED_LEAD, Strategy.ALGEBRAIC_DELTA_NATIVE_RECOMPUTE),
        ):
            with self.subTest(code=code):
                base = ["--schema", schema, "--view", view, "--dialect", "postgresql:18", *extra]

                def run(argv: List[str]) -> Tuple[str, str]:
                    import sys as _sys
                    with contextlib.redirect_stdout(io.StringIO()) as out, \
                            contextlib.redirect_stderr(io.StringIO()) as err, warnings.catch_warnings():
                        warnings.simplefilter("always")
                        old = warnings.showwarning
                        warnings.showwarning = lambda m, c, f, l, file=None, line=None: print(str(m), file=_sys.stderr)
                        try:
                            self.assertEqual(cli_main(argv), 0)
                        finally:
                            warnings.showwarning = old
                    return out.getvalue(), err.getvalue()

                text_out, text_err = run(base)
                self.assertIn(f"[{code}; view v]", text_err)
                self.assertIn(f"{lead} {strat.value}", text_err)
                # Text output: the diagnostic leads a stdout comment line.
                self.assertIn(f"-- {lead} {strat.value}", text_out)
                # --diff --json: stdout is exactly one JSON document.
                json_out, _ = run(base + ["--diff", "--json"])
                doc = _json.loads(json_out)
                self.assertEqual([d["code"] for d in doc["diagnostics"]], [code])
                self.assertEqual(doc["diagnostics"][0]["strategy"], strat.value)
                self.assertTrue(doc["diagnostics"][0]["message"].startswith(lead))


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

    def _strategy(self, mgr: Any) -> str:
        return mgr.get_catalog()["v_up"]["strategy"]

    def test_global_override_keeps_the_exact_default(self) -> None:
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect, ignore_limitations=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    mgr.register_view(self.VIEW)
                self.assertEqual(self._strategy(mgr), MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
                self.assertEqual([d.code for d in _diag_warnings(caught)], [FLOAT_EXACT_REFRESH])
                self.assertNotIn(_NR, mgr.get_catalog()["v_up"]["accepted_limitations"])
                (d,) = mgr.get_diagnostics("v_up")["v_up"]
                self.assertEqual((d.code, d.strategy), (FLOAT_EXACT_REFRESH, Strategy.FULL_REFRESH))

    def test_manager_defaults_never_widen_an_installed_contract(self) -> None:
        """A view installed under the exact default keeps it when a later
        manager has relaxed defaults: diagnostics, drift and repair use the
        stored contract, not the manager's current accept_limitations."""
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect, ignore_limitations=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mgr.register_view(self.VIEW)
                self.assertEqual(self._strategy(mgr), MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
                relaxed_mgr = MaterializedViewManager(
                    conn, dialect=dialect, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    (d,) = relaxed_mgr.get_diagnostics("v_up")["v_up"]
                    self.assertEqual((d.code, d.strategy), (FLOAT_EXACT_REFRESH, Strategy.FULL_REFRESH))
                    self.assertEqual(relaxed_mgr.check_schema_drift(), {"v_up": "ok"})
                    relaxed_mgr.repair_view("v_up")
                self.assertEqual(self._strategy(relaxed_mgr), MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
                self.assertNotIn(_NR, relaxed_mgr.get_catalog()["v_up"]["accepted_limitations"])

    def test_acceptance_is_stored_and_reused(self) -> None:
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect)
                self._install(mgr, dialect)
                self.assertIn(_NR, mgr.get_catalog()["v_up"]["accepted_limitations"])
                relaxed = self._strategy(mgr)
                self.assertEqual(relaxed, MaintenanceStrategy.ALGEBRAIC_DELTA.value)
                mgr2 = MaterializedViewManager(conn, dialect=dialect)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertEqual(mgr2.sync_views([self.VIEW]), {"v_up": "unchanged"})
                    mgr2.register_view(self.VIEW)
                    self.assertEqual(mgr2.check_schema_drift(), {"v_up": "ok"})
                    self.assertEqual(mgr2.repair_view("v_up"), {"v_up": "repaired"})
                    if dialect == Dialect.POSTGRES:
                        conn.execute(f"UPDATE {mgr2.catalog_table} SET signature_hash = 'stale'")
                        conn.commit()
                        self.assertEqual(mgr2.repair_view("v_up"), {"v_up": "rebuilt"})
                self.assertEqual(self._strategy(mgr2), relaxed)
                (d,) = mgr2.get_diagnostics("v_up")["v_up"]
                self.assertEqual(d.code, FLOAT_RELAXED_MAINTENANCE)

    def test_repair_never_adds_the_relaxed_mode(self) -> None:
        """repair_view() rebuilds under exactly the stored codes: a view whose
        stored acceptance lacks FLOAT_REASSOCIATION_ALLOWED is rebuilt with
        the exact full refresh, never with relaxed maintenance."""
        for dialect in (Dialect.POSTGRES,):  # MySQL refuses non-atomic replacement
            with self.subTest(dialect=dialect.value):
                conn, mgr = self._manager(dialect)
                self._install(mgr, dialect)
                entry = mgr.get_catalog()["v_up"]
                stored = {
                    "definition": entry["accepted_definition"],
                    "codes": [c for c in entry["accepted_limitations"] if c != _NR],
                }
                conn.execute(f"UPDATE {mgr.catalog_table} SET accepted_limitations_json = '{json.dumps(stored)}'")
                conn.commit()
                other = MaterializedViewManager(conn, dialect=dialect, ignore_limitations=True)
                self.assertEqual(other.check_schema_drift(), {"v_up": "drifted"})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertEqual(other.repair_view("v_up"), {"v_up": "rebuilt"})
                self.assertEqual(self._strategy(other), MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)

    def test_acceptance_is_bound_to_the_definition(self) -> None:
        """Codes accepted for one definition are not reused for another
        definition under the same name: the new definition gets the exact
        default unless it is accepted explicitly."""
        other_def = "CREATE VIEW v_up AS SELECT g, AVG(x) AS a, VAR_SAMP(x) AS vs FROM t WHERE id > 0 GROUP BY g;"
        conn, mgr = self._manager(Dialect.POSTGRES)
        self._install(mgr, Dialect.POSTGRES)
        fresh = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fresh.register_view(other_def)
        self.assertEqual(self._strategy(fresh), MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
        self.assertEqual([d.code for d in _diag_warnings(caught)], [FLOAT_EXACT_REFRESH])
        self.assertNotIn(_NR, fresh.get_catalog()["v_up"]["accepted_limitations"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh.register_view(other_def, accept_limitations=FLOAT_OPT_IN)
        self.assertEqual(self._strategy(fresh), MaintenanceStrategy.ALGEBRAIC_DELTA.value)

    def test_acceptance_record_for_another_definition_accepts_nothing(self) -> None:
        conn, mgr = self._manager(Dialect.POSTGRES)
        self._install(mgr, Dialect.POSTGRES)
        entry = mgr.get_catalog()["v_up"]
        forged = {"definition": "0" * 64, "codes": entry["accepted_limitations"]}
        conn.execute(f"UPDATE {mgr.catalog_table} SET accepted_limitations_json = '{json.dumps(forged)}'")
        conn.commit()
        other = MaterializedViewManager(conn, dialect=Dialect.POSTGRES, ignore_limitations=True)
        e2 = other.get_catalog()["v_up"]
        self.assertTrue(e2["accepted_limitations_mismatch"])
        self.assertEqual(e2["accepted_limitations"], [])
        # Recompiled without the relaxed acceptance: exact default, so drifted.
        self.assertEqual(other.check_schema_drift(), {"v_up": "drifted"})
        # The installed view is still the relaxed one: no diagnostic claims
        # an exact guarantee it does not have yet ...
        self.assertEqual(other.get_diagnostics("v_up")["v_up"], [])
        # ... until repair rebuilds it under the exact default.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other.repair_view("v_up")
        self.assertEqual(other.get_catalog()["v_up"]["strategy"], MaintenanceStrategy.ATOMIC_QUERY_SYNC.value)
        (d,) = other.get_diagnostics("v_up")["v_up"]
        self.assertEqual(d.code, FLOAT_EXACT_REFRESH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
