#!/usr/bin/env python3
"""Fail-closed provability of the exact floating-point refresh default.

Under the default (exact) mode a view with an order-sensitive floating-point
aggregate is maintained by a full refresh of its ORIGINAL query, so every stored
column type must equal the type the engine gives the original query. The types
are derived from declared base-table columns; matview.scope_resolver is the ONE
resolver deciding, for every nesting level and join form, whether an aggregate's
argument columns are provably declared base-table columns. Everything not
provable is refused (UnsupportedMaterializedViewError), never guessed.

Four independent adversarial reviews each found the same class in a different
shape (an aggregate argument that could not be proven to be a base column was
classified approximate but its stored type was derived wrongly). The cases below
are the minimal repros of those reviews plus the shapes a schema-aware
generator found (same-named columns of different types in different tables;
aggregates at every nesting level, windows, USING/NATURAL/ON joins, set
operations, scalar/correlated/EXISTS/IN subqueries; PostgreSQL 18, MySQL 8.4,
MariaDB 11.8).

* Offline (no database): the resolver verdicts and the plan decision of every
  case on PostgreSQL 18, MySQL 8.4 and MariaDB 11.8.
* Native (POSTGRES_DSN / MYSQL_DSN): every ACCEPTED shape is compared with the
  original view after population and after DML (values, Python types, column
  type family); every refused shape is refused through the manager with a
  clear message, never a raw error.
"""

from __future__ import annotations

import re
import unittest
import warnings
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp

from matview_sync import (
    Dialect,
    EngineLimitationWarning,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
    MaintenanceStrategy,
    generate_materialized_view,
    parse_table_schema,
)
from matview.diagnostics import EXACT_LEAD, FLOAT_EXACT_REFRESH, FLOAT_RELAXED_MAINTENANCE, RELAXED_LEAD
from matview.scope_resolver import BASE, CONFLICT, UNKNOWN, ScopeResolver
from test_avg_concurrent_maintenance import _connect, _native_engines

# ta.x INT, tb.x DOUBLE, tc.x DECIMAL: the same column NAME with different TYPES.
_SCHEMAS = {
    "pg": (Dialect.POSTGRES, "postgresql 18",
           "CREATE TABLE ta (id BIGINT PRIMARY KEY, g INT, x INTEGER, y DOUBLE PRECISION);"
           "CREATE TABLE tb (id BIGINT PRIMARY KEY, g INT, x DOUBLE PRECISION, y NUMERIC(12,3));"
           "CREATE TABLE tc (id BIGINT PRIMARY KEY, g INT, x NUMERIC(12,3), y INTEGER);"
           "CREATE TABLE te (id BIGINT PRIMARY KEY, g BIGINT, y DOUBLE PRECISION);"),
    "mysql": (Dialect.MYSQL, "mysql 8.4",
              "CREATE TABLE ta (id BIGINT PRIMARY KEY, g INT, x INT, y DOUBLE);"
              "CREATE TABLE tb (id BIGINT PRIMARY KEY, g INT, x DOUBLE, y DECIMAL(12,3));"
              "CREATE TABLE tc (id BIGINT PRIMARY KEY, g INT, x DECIMAL(12,3), y INT);"
              "CREATE TABLE te (id BIGINT PRIMARY KEY, g BIGINT, y DOUBLE);"),
    "mariadb": (Dialect.MYSQL, "mariadb 11.8",
                "CREATE TABLE ta (id BIGINT PRIMARY KEY, g INT, x INT, y DOUBLE);"
                "CREATE TABLE tb (id BIGINT PRIMARY KEY, g INT, x DOUBLE, y DECIMAL(12,3));"
                "CREATE TABLE tc (id BIGINT PRIMARY KEY, g INT, x DECIMAL(12,3), y INT);"
                "CREATE TABLE te (id BIGINT PRIMARY KEY, g BIGINT, y DOUBLE);"),
}


def _plan(kind: str, select: str) -> Any:
    dialect, version, ddl = _SCHEMAS[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_materialized_view(
            parse_table_schema(ddl, dialect=dialect), f"CREATE VIEW v AS {select};",
            dialect=dialect, engine_version=version, ignore_limitations=True,
        )


def _resolver(select: str, kind: str = "pg") -> Tuple[Any, ScopeResolver]:
    dialect, _, ddl = _SCHEMAS[kind]
    root = sqlglot.parse_one(select, read="postgres" if dialect == Dialect.POSTGRES else "mysql")
    return root, ScopeResolver(root, parse_table_schema(ddl, dialect=dialect), common_type=lambda ts, has_null: next((t for t in ts if "DOUBLE" in t), ts[0]), literal_type=lambda lit: "INT")


class ResolverVerdicts(unittest.TestCase):
    """The resolver alone: scope chain, shadowing, derived columns, USING."""

    def status(self, select: str, column: str, table: str = "") -> Tuple[str, str]:
        root, rs = _resolver(select)
        cols = [c for c in root.find_all(sqlglot_exp.Column) if c.name == column and (c.table or "") == table]
        r = rs.resolve(cols[0])
        return r.status, (f"{r.table}.{r.column.data_type}" if r.column is not None else r.reason)

    def test_table_driven(self) -> None:
        cases = [
            # (select, column, qualifier, expected status, expected type/table fragment)
            ("SELECT SUM(t.x) FROM ta t", "x", "t", BASE, "ta.INT"),
            ("SELECT SUM(x) FROM ta", "x", "", BASE, "ta.INT"),
            # An alias that is also an alias in a subquery is resolved in its OWN scope.
            ("SELECT t.g, (SELECT SUM(t.x) FROM tb t WHERE t.id = 1) FROM ta t", "x", "t", BASE, "tb.DOUBLE PRECISION"),
            ("WITH q AS (SELECT t.id AS id, AVG(t.x) OVER (PARTITION BY t.g) AS w FROM tb t) "
             "SELECT SUM(t.x) FROM ta t JOIN q ON q.id = t.id", "x", "t", BASE, "ta.INT"),
            # Derived-table / CTE columns: only a bare pass-through column is provable.
            ("SELECT SUM(d.x) FROM (SELECT g, x FROM tb) d", "x", "d", BASE, "tb.DOUBLE PRECISION"),
            ("WITH c AS (SELECT g, x FROM ta) SELECT SUM(c.x) FROM c", "x", "c", BASE, "ta.INT"),
            ("SELECT SUM(d.n) FROM (SELECT g, COUNT(*) AS n FROM ta GROUP BY g) d", "n", "d", UNKNOWN, "expression"),
            ("SELECT SUM(d.s) FROM (SELECT g, SUM(x) AS s FROM ta GROUP BY g) d", "s", "d", UNKNOWN, "expression"),
            ("SELECT SUM(d.w) FROM (SELECT AVG(x) OVER (PARTITION BY g) AS w FROM tb) d", "w", "d", UNKNOWN, "expression"),
            ("SELECT SUM(d.x) FROM (SELECT * FROM ta) d", "x", "d", UNKNOWN, ""),
            ("WITH RECURSIVE r(n, v) AS (SELECT 1, 1 UNION ALL SELECT n + 1, v * 2 FROM r WHERE n < 3) SELECT SUM(r.v) FROM r",
             "v", "r", UNKNOWN, "recursive CTE"),
            # USING / NATURAL / ambiguity: provable only when every source has the identical declared type.
            ("SELECT SUM(x) FROM ta JOIN tb USING (g, x)", "x", "", CONFLICT, "different types"),
            ("SELECT SUM(x) FROM tb RIGHT JOIN ta USING (g, x)", "x", "", CONFLICT, "different types"),
            ("SELECT SUM(x) FROM ta NATURAL JOIN tb", "x", "", CONFLICT, "different types"),
            ("SELECT SUM(g) FROM ta JOIN tb USING (g)", "g", "", BASE, "INT"),
            ("SELECT SUM(a.x) FROM ta a JOIN tb b USING (g)", "x", "a", BASE, "ta.INT"),
            # Correlated reference to the OUTER query.
            ("SELECT g FROM ta t WHERE EXISTS (SELECT 1 FROM tb u GROUP BY u.g HAVING SUM(t.x) > 0)", "x", "t", BASE, "ta.INT"),
            # Set-operation derived table: the common type of bare base columns.
            ("SELECT SUM(u.x) FROM (SELECT x FROM ta UNION ALL SELECT x FROM tb) u", "x", "u", BASE, "None.DOUBLE PRECISION"),
            ("SELECT SUM(u.x) FROM (SELECT x * 2 AS x FROM ta UNION ALL SELECT x FROM tb) u", "x", "u", UNKNOWN, "expression"),
            # Recursive CTE: an anchor/step pair of the same base type, a depth counter
            # (integer literal anchor, `previous + 1` step) and a pass-through are provable.
            ("WITH RECURSIVE r(id, root_id, score, depth) AS (SELECT id, id, x, 0 FROM ta UNION ALL "
             "SELECT n.id, t.root_id, n.x, t.depth + 1 FROM ta n JOIN r t ON n.g = t.id WHERE t.depth < 3) "
             "SELECT root_id, SUM(score), MAX(depth) FROM r GROUP BY root_id", "score", "", BASE, "None.INT"),
            ("WITH RECURSIVE r(id, root_id, score, depth) AS (SELECT id, id, x, 0 FROM ta UNION ALL "
             "SELECT n.id, t.root_id, n.x, t.depth + 1 FROM ta n JOIN r t ON n.g = t.id WHERE t.depth < 3) "
             "SELECT root_id, SUM(score), MAX(depth) FROM r GROUP BY root_id", "depth", "", BASE, "None.INT"),
            # ... a step that changes the type (`v * 1.5`) or reads another table's column is not.
            ("WITH RECURSIVE r(n, v) AS (SELECT 1, 1 UNION ALL SELECT n + 1, v * 1.5 FROM r WHERE n < 3) SELECT SUM(v) FROM r",
             "v", "", UNKNOWN, "recursive CTE"),
            ("WITH RECURSIVE r(n, v) AS (SELECT 1, x FROM ta UNION ALL SELECT n + 1, b.x FROM r JOIN tb b ON b.id = r.n) "
             "SELECT SUM(v) FROM r", "v", "", BASE, "None.DOUBLE PRECISION"),
            # Table function / unknown relation.
            ("SELECT SUM(s.v) FROM generate_series(1, 3) AS s(v)", "v", "s", UNKNOWN, ""),
            ("SELECT SUM(zz) FROM ta", "zz", "", UNKNOWN, ""),
        ]
        for select, column, qual, want_status, want_frag in cases:
            with self.subTest(select=select):
                status, detail = self.status(select, column, qual)
                self.assertEqual(status, want_status, detail)
                self.assertIn(want_frag, detail)


# (label, select, reason fragment). Refused on every engine, with a clear message.
_REFUSED: List[Tuple[str, str, str]] = [
    # Review 4 (HIGH 1): aggregate over a derived-table / CTE column of unknown type.
    ("derived_count", "SELECT d.g AS k, AVG(d.n) AS o FROM (SELECT g, COUNT(*) AS n FROM ta GROUP BY g) d GROUP BY d.g",
     "derived table or CTE column n is an expression"),
    ("derived_sum_of_float", "SELECT d.g AS k, AVG(d.s) AS o FROM (SELECT g, SUM(x) AS s FROM tb GROUP BY g) d GROUP BY d.g",
     "expression"),
    ("cte_stddev_of_int", "WITH c AS (SELECT g, STDDEV_POP(x) AS s FROM ta GROUP BY g) SELECT g, SUM(c.s) AS o FROM c GROUP BY g",
     "expression"),
    # Review 4 (HIGH 2): a shadowed alias resolved through the OUTER scope in a window / CTE.
    ("shadowed_alias_window",
     "WITH q AS (SELECT t.id AS id, AVG(t.x) OVER (PARTITION BY t.g) AS w FROM tb t) "
     "SELECT t.g AS g, SUM(q.w) AS s FROM ta t JOIN q ON q.id = t.id GROUP BY t.g", "expression"),
    # Review 4 (HIGH 3): a USING/NATURAL-merged column typed from the first table.
    ("using_conflict", "SELECT g, SUM(x) AS s FROM ta JOIN tb USING (g, x) GROUP BY g", "different types"),
    ("using_right_join", "SELECT g, SUM(x) AS s FROM tb RIGHT JOIN ta USING (g, x) GROUP BY g", "different types"),
    # MAX is not order-sensitive: only the strategy-independent conflict refusal catches it.
    ("using_conflict_minmax", "SELECT g, MAX(x) AS m FROM ta JOIN tb USING (g, x) GROUP BY g", "different types"),
    ("natural_conflict", "SELECT g, SUM(x) AS s FROM ta NATURAL JOIN tb GROUP BY g", "different types"),
    # Window function over an unprovable column.
    ("window_over_derived", "SELECT d.id AS id, SUM(d.w) OVER (PARTITION BY d.g) AS s "
     "FROM (SELECT id, g, SUM(x) OVER (PARTITION BY g) AS w FROM tb) d", "expression"),
    # Round 3: a recursive CTE column.
    ("recursive_cte", "WITH RECURSIVE r(n, v) AS (SELECT 1, 1.5 UNION ALL SELECT n + 1, v * 2 FROM r WHERE n < 3) "
     "SELECT a.g AS g, SUM(r.v) AS s FROM tb a CROSS JOIN r GROUP BY a.g", "recursive CTE"),
    # An aggregate nested in a derived table / CTE / subquery projection feeding a stored column.
    ("nested_aggregate_in_derived", "SELECT d.g AS g, d.s AS s, SUM(b.x) AS t FROM "
     "(SELECT g, MAX(x) AS s FROM ta GROUP BY g) d JOIN tb b ON b.g = d.g GROUP BY d.g, d.s", "derived table, CTE"),
    ("scalar_subquery_over_derived_expression", "SELECT g, SUM(x) AS s, "
     "(SELECT MAX(d.y) FROM (SELECT y * 2 AS y FROM tc) d) AS m FROM tb GROUP BY g", "expression"),
    # A compound item whose type is not provable (decimal scale of AVG * 2).
    ("compound_decimal", "SELECT g, AVG(x) * 2 AS a, SUM(y) AS s FROM ta GROUP BY g", "larger expression"),
    # A column the stored-type derivation cannot confirm (non-first set-operation branch of a derived union).
    ("derived_union_expression", "SELECT u.g AS g, SUM(u.x) AS s FROM "
     "(SELECT g, x * 2 AS x FROM ta UNION ALL SELECT g, x FROM tb) u GROUP BY u.g", "expression"),
    # Only the class of check each of these exercises can reject them (mutation-tested):
    ("setop_branch_types_differ", "SELECT g, SUM(x) AS s FROM ta GROUP BY g UNION ALL SELECT g, SUM(x) AS s FROM tb GROUP BY g",
     "identical types"),
    ("shadowed_scalar_subquery_derived_expression",
     "SELECT g, (SELECT SUM(x) FROM (SELECT x * 2 AS x FROM tb) d) AS q FROM ta GROUP BY g",
     "derived table or CTE column x is an expression"),
    ("key_typed_by_the_wrong_table", "SELECT d.k AS k, SUM(a.y) AS s FROM ta a JOIN (SELECT g AS k FROM te) d ON a.g = d.k GROUP BY d.k",
     "stored as"),
    ("minmax_decimal_integer_expression",
     "SELECT t.g AS g, MIN(c.x + c.y) AS m, SUM(t.y) AS s FROM ta t JOIN tc c ON t.g = c.g GROUP BY t.g",
     "MIN/MAX over an expression"),
    ("same_text_two_types", "SELECT t.g AS g, SUM(t.y) AS s, (SELECT MAX(t.y) FROM tb t) AS m FROM ta t GROUP BY t.g",
     "different types"),
]

# Accepted by the full refresh of the original query, announced on every engine.
_REFRESHED: List[Tuple[str, str]] = [
    ("plain_float", "SELECT g, SUM(y) AS s, AVG(y) AS a, COUNT(*) AS c FROM ta GROUP BY g"),
    ("qualified_other_table", "SELECT t.g AS g, SUM(u.x) AS s FROM ta t JOIN tb u ON t.g = u.g GROUP BY t.g"),
    ("same_name_exact_vs_float", "SELECT t.g AS g, SUM(t.x) AS si, SUM(u.x) AS sf FROM ta t JOIN tb u ON t.g = u.g GROUP BY t.g"),
    ("shadowed_alias_in_subquery", "SELECT t.g AS g, SUM(t.y) AS s FROM ta t "
     "WHERE t.g IN (SELECT t.g FROM tb t WHERE t.x > 0) GROUP BY t.g"),
    ("where_subquery_float", "SELECT g, COUNT(*) AS c FROM ta WHERE x > (SELECT AVG(w.x) FROM tb w) GROUP BY g"),
    ("using_same_type", "SELECT g, SUM(a.y) AS s, COUNT(*) AS c FROM ta a JOIN tc c USING (g) GROUP BY g"),
    ("derived_passthrough", "SELECT d.g AS k, SUM(d.x) AS s FROM (SELECT g, x FROM tb) d GROUP BY d.g"),
    ("derived_union_common_type", "SELECT u.g AS g, SUM(u.x) AS s FROM "
     "(SELECT g, x FROM ta UNION ALL SELECT g, x FROM tb) u GROUP BY u.g"),
    ("scalar_subquery_projection", "SELECT g, SUM(x) AS s, (SELECT MAX(w.y) FROM tc w) AS m FROM tb GROUP BY g"),
    ("setop_same_typed_shape", "SELECT g, SUM(y) AS s FROM ta GROUP BY g UNION ALL SELECT g, SUM(x) AS s FROM tb GROUP BY g"),
    ("recursive_passthrough", "WITH RECURSIVE r(id, g, y, d) AS (SELECT id, g, y, 0 FROM ta UNION ALL "
     "SELECT n.id, n.g, r.y, r.d + 1 FROM ta n JOIN r ON n.g = r.id WHERE r.d < 2) SELECT g, SUM(y) AS s FROM r GROUP BY g"),
    ("float_compound", "SELECT g, COALESCE(SUM(y), 0) - COALESCE(SUM(y * 2), 0) AS d FROM ta GROUP BY g"),
]


# Shapes whose float aggregate reads 2+ table instances: refused on MySQL/MariaDB
# (plan dependence), announced full refresh on PostgreSQL.
_MULTI_TABLE = {
    "qualified_other_table", "same_name_exact_vs_float", "shadowed_alias_in_subquery", "using_same_type",
    "scalar_subquery_projection", "recursive_passthrough",
}
# Refusal cases whose type-provability reason is reported on MySQL/MariaDB too
# (the aggregate reads one table instance, or the type check runs first).
_TYPE_REASON_ON_MYSQL = {"derived_count", "derived_sum_of_float", "compound_decimal",
                         "setop_branch_types_differ", "using_conflict", "using_right_join", "natural_conflict",
                         "using_conflict_minmax"}


class OfflinePlanDecisions(unittest.TestCase):
    def test_unprovable_types_are_refused_on_every_engine(self) -> None:
        # PostgreSQL states the exact reason (type provability); MySQL/MariaDB
        # refuse multi-table float sums earlier (plan dependence), so there the
        # case must be refused, whatever the reason.
        for kind in _SCHEMAS:
            for label, select, fragment in _REFUSED:
                with self.subTest(engine=kind, case=label):
                    with self.assertRaises(UnsupportedMaterializedViewError) as ctx:
                        _plan(kind, select)
                    if kind == "pg" or label in _TYPE_REASON_ON_MYSQL:
                        self.assertIn(fragment, str(ctx.exception))

    def test_alias_cycle_is_a_clear_refusal_not_a_recursion_error(self) -> None:
        # A derived-table alias equal to an inner table alias (u) used to recurse
        # without end in the name-based column resolution (an exact view, no float).
        select = ("SELECT u.k AS k, MAX(u.a0) AS a0 FROM ta d LEFT JOIN "
                  "(SELECT b.g AS k, MAX(u.x) AS a0 FROM tb b JOIN ta u ON b.g = u.g GROUP BY b.g) u "
                  "ON d.g = u.k GROUP BY u.k")
        for kind in _SCHEMAS:
            with self.subTest(engine=kind):
                with self.assertRaisesRegex(UnsupportedMaterializedViewError, "recursed without end"):
                    _plan(kind, select)

    def test_provable_shapes_use_the_announced_full_refresh(self) -> None:
        for kind in _SCHEMAS:
            for label, select in _REFRESHED:
                with self.subTest(engine=kind, case=label):
                    if kind != "pg" and label in _MULTI_TABLE:
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                            _plan(kind, select)
                        continue
                    plan = _plan(kind, select)
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], [FLOAT_EXACT_REFRESH])

    def test_mysql_multi_table_float_sum_is_not_exact_by_construction(self) -> None:
        # Measured on MySQL 8.4 (seed 7, 30 random writes): the refresh gave
        # ...4000000000001p+55, a later SELECT / CTAS / INSERT..SELECT ...002.
        select = "SELECT ta.g AS g, SUM(tb.x) AS s FROM ta JOIN tb USING (g, x) GROUP BY ta.g"
        for kind in ("mysql", "mariadb"):
            with self.subTest(engine=kind):
                with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                    _plan(kind, select)
                # The relaxed opt-in stays available and is announced.
                dialect, version, ddl = _SCHEMAS[kind]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plan = generate_materialized_view(
                        parse_table_schema(ddl, dialect=dialect), f"CREATE VIEW v AS {select};",
                        dialect=dialect, engine_version=version, ignore_limitations=True,
                        accept_limitations=(
                            "FLOAT_REASSOCIATION_ALLOWED", "ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE",
                            "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX",
                        ),
                    )
                self.assertTrue(plan.strategy_diagnostics)
        # PostgreSQL keeps the announced full refresh.
        self.assertEqual(_plan("pg", select).strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)

    def test_relaxed_full_refresh_over_joins_is_announced_relaxed_not_exact(self) -> None:
        # Under the opt-in these MySQL/MariaDB shapes still re-run the query, but
        # their float result depends on the join order (measured: MV ...0001p+55,
        # live ...0002p+55 at step 16): the announcement must be the relaxed one.
        shapes = [
            "SELECT ta.g AS g, SUM(tb.x) AS s FROM ta JOIN tb USING (g, x) GROUP BY ta.g",
            "SELECT d.g AS g, SUM(d.x) AS s FROM (SELECT ta.g AS g, tb.x AS x FROM ta JOIN tb ON ta.g = tb.g) d GROUP BY d.g",
            "SELECT ta.g AS g, SUM(tb.x) AS s FROM ta JOIN tb ON ta.g = tb.g GROUP BY ta.g "
            "UNION ALL SELECT g, SUM(y) AS s FROM ta GROUP BY g",
        ]
        opt_in = ("FLOAT_REASSOCIATION_ALLOWED", "ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE",
                  "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX")
        for kind in _SCHEMAS:
            dialect, version, ddl = _SCHEMAS[kind]
            for select in shapes:
                with self.subTest(engine=kind, select=select):
                    if kind == "pg":
                        plan = _plan(kind, select)
                        (d,) = plan.strategy_diagnostics
                        self.assertEqual(d.code, FLOAT_EXACT_REFRESH)
                        self.assertTrue(d.message.startswith(EXACT_LEAD), d.message)
                        continue
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                        _plan(kind, select)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plan = generate_materialized_view(
                            parse_table_schema(ddl, dialect=dialect), f"CREATE VIEW v AS {select};",
                            dialect=dialect, engine_version=version, ignore_limitations=True,
                            accept_limitations=opt_in,
                        )
                    codes = [d.code for d in plan.strategy_diagnostics]
                    self.assertEqual(codes, [FLOAT_RELAXED_MAINTENANCE], [str(d) for d in plan.strategy_diagnostics])
                    msg = plan.strategy_diagnostics[0].message
                    self.assertTrue(msg.startswith(RELAXED_LEAD), msg)
                    self.assertNotIn(EXACT_LEAD, msg)
                    self.assertIn("table instances", msg)
                    self.assertIn("FLOAT_REASSOCIATION_ALLOWED", {lim.code for lim in plan.limitations})

    def test_single_table_float_sums_stay_exact_by_full_refresh_on_every_engine(self) -> None:
        for kind in _SCHEMAS:
            for select in ("SELECT g, SUM(y) AS s, AVG(y) AS a FROM ta GROUP BY g",
                           "SELECT id, SUM(x) OVER (PARTITION BY g) AS w FROM tb"):
                with self.subTest(engine=kind, select=select):
                    plan = _plan(kind, select)
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], [FLOAT_EXACT_REFRESH])

    def test_conflicting_types_are_refused_even_for_exact_looking_columns(self) -> None:
        # The first table's INTEGER x used to classify this as exact and type
        # the sum NUMERIC/DECIMAL while the engine's merged column is DOUBLE.
        for kind in _SCHEMAS:
            with self.subTest(engine=kind):
                with self.assertRaisesRegex(UnsupportedMaterializedViewError, "different types"):
                    _plan(kind, "SELECT g, SUM(x) AS s FROM ta JOIN tb USING (g, x) GROUP BY g")

    def test_conflicting_types_are_refused_under_the_relaxed_opt_in_too(self) -> None:
        # The conflict refusal is independent of the strategy: the first
        # table's INT must not type the merged column under any mode.
        dialect, version, ddl = _SCHEMAS["pg"]
        for kind in _SCHEMAS:
            dialect, version, ddl = _SCHEMAS[kind]
            with self.subTest(engine=kind):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "different types"):
                        generate_materialized_view(
                            parse_table_schema(ddl, dialect=dialect),
                            "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM ta JOIN tc USING (g, x) GROUP BY g;",
                            dialect=dialect, engine_version=version, ignore_limitations=True,
                            accept_limitations=(
                                "FLOAT_REASSOCIATION_ALLOWED", "ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE",
                                "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX",
                            ),
                        )

    def test_exact_views_keep_their_incremental_strategy(self) -> None:
        for kind in _SCHEMAS:
            with self.subTest(engine=kind):
                plan = _plan(kind, "SELECT g, SUM(x) AS s, COUNT(*) AS c FROM ta GROUP BY g")
                self.assertEqual(plan.strategy_diagnostics, [])
                self.assertNotEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)

    def test_where_only_float_aggregates_keep_the_views_own_type_derivation(self) -> None:
        # A float aggregate that only filters (WHERE/HAVING/EXISTS) changes no
        # stored type: a view master accepted keeps being accepted.
        for kind in _SCHEMAS:
            with self.subTest(engine=kind):
                plan = _plan(kind, "SELECT t.g AS g, COUNT(*) AS c FROM ta t WHERE EXISTS "
                                   "(SELECT 1 FROM tb u WHERE u.g = t.g GROUP BY u.g HAVING SUM(u.x) > 1) GROUP BY t.g")
                self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC)


# ---------------------------------------------------------------------------
# Native differential
# ---------------------------------------------------------------------------

_NATIVE_TYPES = {
    Dialect.POSTGRES: {"int": "INTEGER", "dbl": "DOUBLE PRECISION", "dec": "NUMERIC(12,3)"},
    Dialect.MYSQL: {"int": "INT", "dbl": "DOUBLE", "dec": "DECIMAL(12,3)"},
}
_TABLES = {"ta": ("int", "dbl"), "tb": ("dbl", "dec"), "tc": ("dec", "int")}
_HOSTILE = {
    "int": [None, 1, -3, 7, 2147483000],
    "dbl": [None, 0.1, 0.2, 0.3, 1e16, -1e16, 1.5, 2.0 ** 55, -2.5],
    "dec": [None, Decimal("0.100"), Decimal("1.500"), Decimal("-2.250"), Decimal("99999.999")],
}
# (recursive CTE sources: NativeRecursiveCteSources; `te`: offline only.)
_NATIVE_REFUSED = [(l, s) for l, s, _ in _REFUSED if l != "recursive_cte" and not re.search(r"\bte\b", s)]
# MariaDB has no LATERAL; the shapes above use none.


def _exact(v: Any) -> Any:
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, Decimal):
        return ("decimal", v.as_tuple())
    return (type(v).__name__, v)


def _family(engine: Dialect, code: Any) -> str:
    if engine == Dialect.POSTGRES:
        return {20: "int", 21: "int", 23: "int", 700: "float", 701: "float", 1700: "decimal"}.get(code, str(code))
    if code in (0, 246):
        return "decimal"
    if code in (4, 5):
        return "float"
    return "int" if code in (1, 2, 3, 8, 9) else str(code)


class NativeDifferential(unittest.TestCase):
    maxDiff = None

    def _snapshot(self, engine: Dialect, conn: Any, sql: str) -> Tuple[List[Any], List[str]]:
        cur = conn.cursor()
        cur.execute(sql)
        rows = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
        fams = [_family(engine, d[1]) for d in cur.description]
        conn.commit()
        return rows, fams

    def test_accepted_shapes_equal_the_original_view_and_refusals_are_clear(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("POSTGRES_DSN / MYSQL_DSN not set; native engine test only")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                self._run(engine, dsn)

    def _run(self, engine: Dialect, dsn: str) -> None:
        import random
        import uuid

        conn = _connect(engine, dsn)
        sfx = uuid.uuid4().hex[:8]
        schema = f"mv_fc_{sfx}"
        names = {k: f"{k}_{sfx}" for k in _TABLES}
        cur = conn.cursor()
        if engine == Dialect.POSTGRES:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
        ty = _NATIVE_TYPES[engine]
        for k, (tx, tyy) in _TABLES.items():
            cur.execute(f"CREATE TABLE {names[k]} (id BIGINT PRIMARY KEY, g INT, x {ty[tx]}, y {ty[tyy]})")
        conn.commit()
        mgr = MaterializedViewManager(conn, dialect=engine)
        rng = random.Random(255)
        counters = {k: 0 for k in _TABLES}

        def insert(k: str) -> Tuple[str, Tuple[Any, ...]]:
            counters[k] += 1
            tx, tyy = _TABLES[k]
            return (f"INSERT INTO {names[k]} (id, g, x, y) VALUES (%s, %s, %s, %s)",
                    (counters[k], rng.choice([None, 1, 2, 2, 3]), rng.choice(_HOSTILE[tx]), rng.choice(_HOSTILE[tyy])))

        def render(sql: str) -> str:
            return re.sub(r"\b(ta|tb|tc)\b", lambda m: names[m.group(1)], sql)

        registered: List[Tuple[str, str, Any]] = []
        try:
            for _ in range(10):
                for k in _TABLES:
                    cur.execute(*insert(k))
            conn.commit()
            for i, (label, select) in enumerate(_REFRESHED):
                sql = render(select)
                if engine == Dialect.MYSQL and label in _MULTI_TABLE:
                    # Plan-dependent float order over a join: refused on MySQL/MariaDB.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                            mgr.register_view(f"CREATE VIEW v{i}_{sfx} AS {sql};", ignore_limitations=True)
                    conn.rollback()
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plan = mgr.register_view(f"CREATE VIEW v{i}_{sfx} AS {sql};", ignore_limitations=True)
                conn.commit()
                self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC, label)
                registered.append((label, sql, plan))
            # Refusals through the manager: clear message, not a raw error.
            for label, select in _NATIVE_REFUSED:
                with self.subTest(refused=label):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaises(UnsupportedMaterializedViewError):
                            mgr.register_view(f"CREATE VIEW r_{label}_{sfx} AS {render(select)};", ignore_limitations=True)
                    conn.rollback()
            if engine == Dialect.MYSQL:
                # The measured MySQL 8.4 repro (refresh ...4000000000001p+55, later read ...002).
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self.assertRaisesRegex(UnsupportedMaterializedViewError, "table instances"):
                        mgr.register_view(
                            f"CREATE VIEW r_plan_{sfx} AS " + render(
                                "SELECT ta.g AS g, SUM(tb.x) AS s FROM ta JOIN tb USING (g, x) GROUP BY ta.g;"),
                            ignore_limitations=True,
                        )
                conn.rollback()
            ops: List[Tuple[str, Tuple[Any, ...]]] = []
            for _ in range(12):
                ops.append(insert(rng.choice(list(_TABLES))))
            for k in _TABLES:
                ops.append((f"UPDATE {names[k]} SET x = %s WHERE id = %s", (rng.choice(_HOSTILE[_TABLES[k][0]]), 2)))
                ops.append((f"UPDATE {names[k]} SET g = %s WHERE id = %s", (3, 3)))
                ops.append((f"DELETE FROM {names[k]} WHERE id = %s", (1,)))
            for step in range(len(ops) + 1):
                if step:
                    cur.execute(*ops[step - 1])
                    conn.commit()
                for label, sql, plan in registered:
                    mv = self._snapshot(engine, conn, f"SELECT * FROM {plan.view_name}")
                    live = self._snapshot(engine, conn, sql)
                    self.assertEqual(mv, live, f"{engine.value} {label} step {step}: {ops[step - 1] if step else 'population'}")
        finally:
            conn.rollback()
            cur = conn.cursor()
            if engine == Dialect.POSTGRES:
                cur.execute("SET search_path TO public")
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            else:
                for _, _, plan in registered:
                    try:
                        mgr.drop_view(plan.view_name)
                    except Exception:
                        conn.rollback()
                for n in names.values():
                    cur.execute(f"DROP TABLE IF EXISTS {n}")
            conn.commit()
            conn.close()


if __name__ == "__main__":
    unittest.main()
