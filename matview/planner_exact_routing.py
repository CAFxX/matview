"""Exact routing for aggregates the algebraic compiler does not decompose.

Aggregates outside ``AggFunc`` (``JSON_AGG``, ``percentile_cont``,
``mode()``, ``ANY_VALUE``, ``regr_*``, user-defined aggregates, ...) and
``GROUP BY ROLLUP/CUBE/GROUPING SETS`` are maintained by re-running the view's
original query:

* group-scoped recompute (``GROUP_SCOPED_DELTA``) when the view has a plain
  ``GROUP BY`` and every aggregate's result is a function of its group's
  multiset of rows (or is fed its rows in a total order);
* full refresh (``ATOMIC_QUERY_SYNC``) otherwise: order-sensitive aggregates
  without a total aggregate-level ``ORDER BY`` (a group-scoped recompute reads
  the rows through a different plan than the live view), grouping sets, and
  non-aggregate expressions that cannot be classified (user-defined aggregates
  in a grouped view).

Each fallback is announced through ``matview.diagnostics``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp

from ._host import MixinHost
from .aggregate_semantics import (
    aggregate_calls,
    call_name,
    has_grouping_sets,
    iter_unordered_order_sensitive_calls,
)
from .diagnostics import Diagnostic, DiagnosticSeverity, Strategy, exact_fallback_message
from .engines import AggFunc, Dialect, MaintenanceStrategy, UnsupportedMaterializedViewError, _sqlglot_dialect
from .logical_ir import SelectItem
from .sql_frontend import parse_sql_ast

# Diagnostic codes (matview.diagnostics registry is open).
UNRECOGNIZED_AGGREGATE_GROUP_RECOMPUTE = "UNRECOGNIZED_AGGREGATE_GROUP_RECOMPUTE"
UNRECOGNIZED_AGGREGATE_FULL_REFRESH = "UNRECOGNIZED_AGGREGATE_FULL_REFRESH"
ORDER_SENSITIVE_AGGREGATE_FULL_REFRESH = "ORDER_SENSITIVE_AGGREGATE_FULL_REFRESH"
OPAQUE_AGGREGATE_FULL_REFRESH = "OPAQUE_AGGREGATE_FULL_REFRESH"
GROUPING_SETS_FULL_REFRESH = "GROUPING_SETS_FULL_REFRESH"


@dataclass(frozen=True)
class ExactRoutingFallback:
    """One routing fallback, worded through matview.diagnostics."""

    code: str
    strategy: Strategy
    reason: str
    cost: str
    remedy: str
    expressions: Tuple[str, ...]

    def message(self) -> str:
        return exact_fallback_message(self.strategy, self.reason, self.cost, self.remedy)


GROUP_RECOMPUTE_COST = (
    "every write recomputes each affected group from the base rows with the view's own "
    "aggregate: O(rows in the group) with an index on the GROUP BY columns "
    "(recommended_base_indexes), O(table) without, on every engine."
)
FULL_REFRESH_COST = (
    "every write re-runs the whole view query: O(table) once per statement on PostgreSQL, "
    "once per changed row on SQLite, MySQL and MariaDB."
)

# Aggregates with an algebraic delta rule (the decomposition targets).
_ALGEBRAIC_AGGREGATE_NAMES = frozenset(
    {
        "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX", "VAR_POP", "VARIANCE", "VAR_SAMP",
        "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP", "BIT_AND", "BIT_OR", "BIT_XOR",
        "BOOL_AND", "EVERY", "BOOL_OR",
    }
)
# Recognized (AggFunc) aggregates without an algebraic rule, as sqlglot
# renders them per dialect; their storage types are inferred by the dialect.
_RECOGNIZED_HOLISTIC_NAMES = frozenset({"GROUP_CONCAT", "STRING_AGG", "JSON_GROUP_ARRAY", "JSON_GROUP_OBJECT", "ARRAY_AGG"})
_PG_FLOAT8_AGGREGATES = frozenset(
    {
        "CORR", "COVAR_POP", "COVAR_SAMP", "REGR_AVGX", "REGR_AVGY", "REGR_INTERCEPT",
        "REGR_R2", "REGR_SLOPE", "REGR_SXX", "REGR_SXY", "REGR_SYY",
        "PERCENT_RANK", "CUME_DIST",
    }
)
_PG_RANGE_TO_MULTIRANGE = {
    "INT4RANGE": "INT4MULTIRANGE",
    "INT8RANGE": "INT8MULTIRANGE",
    "NUMRANGE": "NUMMULTIRANGE",
    "TSRANGE": "TSMULTIRANGE",
    "TSTZRANGE": "TSTZMULTIRANGE",
    "DATERANGE": "DATEMULTIRANGE",
}


class ExactAggregateRoutingMixin(MixinHost):
    """Planner hooks that route unrecognized aggregates to exact strategies."""

    # ------------------------------------------------------------------ AST
    def _exact_routing_select_ast(self) -> Optional[sqlglot_exp.Expr]:
        try:
            return parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
        except (sqlglot.errors.SqlglotError, ValueError):
            return None

    def _item_ast(self, raw_expr: str) -> Optional[sqlglot_exp.Expr]:
        try:
            return parse_sql_ast(raw_expr, dialect=self.dialect)
        except (sqlglot.errors.SqlglotError, ValueError):
            return None

    # ------------------------------------------------------- classification
    def _native_aggregate_names(self) -> frozenset[str]:
        """User-defined aggregates classified through the engine catalog."""
        return frozenset(n for n, _ in self.view.native_aggregates)

    def _view_has_grouping_sets(self) -> bool:
        q = self._exact_routing_select_ast()
        if q is None:
            return False
        selects = [q] if isinstance(q, sqlglot_exp.Select) else list(q.find_all(sqlglot_exp.Select))
        return any(has_grouping_sets(s) for s in selects)

    def _order_keys_cover_every_primary_key(self, keys: Sequence[sqlglot_exp.Expr]) -> bool:
        """Whether aggregate ORDER BY keys are a total order over the joined rows:
        they include every primary-key column of every base table (qualified
        by the table's alias when the view joins several tables)."""
        v = self.view
        if v.has_cte or v.has_subquery_in_from or v.has_compound_set_op or not v.joins:
            return False
        key_cols = [k for k in keys if isinstance(k, sqlglot_exp.Column)]
        single = len(v.joins) == 1
        for j in v.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts is None:
                return False
            pk_cols = ts.effective_pk_columns()
            if not pk_cols:
                return False
            names = {j.table_name.lower(), (j.alias or j.table_name).lower()}
            for pk in pk_cols:
                if not any(
                    kc.name.lower() == pk.lower()
                    and ((kc.table.lower() in names) if kc.table else single)
                    for kc in key_cols
                ):
                    return False
        return True

    def _unordered_order_sensitive_aggregates(self) -> List[str]:
        """SQL of order-sensitive aggregate calls without a total input order."""
        out: List[str] = []
        exprs = [it.raw_expr for it in self.view.select_items]
        if self.view.having_clause:
            exprs.append(self.view.having_clause)
        sg_d = _sqlglot_dialect(self.dialect)
        for raw in exprs:
            node = self._item_ast(raw)
            if node is None:
                continue
            for call in iter_unordered_order_sensitive_calls(
                node, self._order_keys_cover_every_primary_key, self._native_aggregate_names()
            ):
                if self._is_float_policy_aggregate(call):
                    continue
                out.append(call.sql(dialect=sg_d))
        return out

    @staticmethod
    def _is_float_policy_aggregate(call: sqlglot_exp.Expr) -> bool:
        """SUM/AVG/VAR/STDDEV are governed by the floating-point policy
        (ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE), not by this routing."""
        return isinstance(
            call,
            (
                sqlglot_exp.Sum, sqlglot_exp.Avg, sqlglot_exp.Stddev, sqlglot_exp.StddevPop,
                sqlglot_exp.StddevSamp, sqlglot_exp.Variance, sqlglot_exp.VariancePop,
            ),
        )

    def _opaque_grouped_items(self) -> List[SelectItem]:
        """Non-aggregate items of a grouped view that call a function sqlglot
        does not know over columns that are not GROUP BY keys: in valid SQL
        such a call is an aggregate (typically user-defined)."""
        v = self.view
        if not v.group_by_cols:
            return []
        group_exprs = {g.strip().lower() for g in v.group_by_cols}
        group_names = set()
        for g in v.group_by_cols:
            g_ast = self._item_ast(g)
            if isinstance(g_ast, sqlglot_exp.Column):
                group_names.add(g_ast.name.lower())
        out: List[SelectItem] = []
        for it in v.select_items:
            if it.is_aggregate or it.raw_expr.strip().lower() in group_exprs:
                continue
            node = self._item_ast(it.raw_expr)
            if node is None or node.find(sqlglot_exp.Window):
                continue
            calls = [a for a in node.find_all(sqlglot_exp.Anonymous)]
            if not calls:
                continue
            for call in calls:
                cols = [c for c in call.find_all(sqlglot_exp.Column)]
                if any(c.name.lower() not in group_names and c.sql().lower() not in group_exprs for c in cols):
                    out.append(it)
                    break
        return out

    def _has_unrecognized_aggregate(self, item: SelectItem) -> bool:
        if not item.is_aggregate or item.agg_func is not None:
            return False
        node = self._item_ast(item.raw_expr)
        if node is None:
            return False
        sg_d = _sqlglot_dialect(self.dialect)
        return any(
            isinstance(call, sqlglot_exp.WithinGroup)
            or call_name(call, sg_d) not in _ALGEBRAIC_AGGREGATE_NAMES | _RECOGNIZED_HOLISTIC_NAMES
            for call in aggregate_calls(node, self._native_aggregate_names())
        )

    # -------------------------------------------------------------- routing
    def _exact_full_refresh_fallback(self) -> Optional[ExactRoutingFallback]:
        """The full-refresh fallback this view needs for an exact result, if any."""
        if self._view_has_grouping_sets():
            return ExactRoutingFallback(
                GROUPING_SETS_FULL_REFRESH,
                Strategy.FULL_REFRESH,
                "GROUP BY ROLLUP/CUBE/GROUPING SETS (or WITH ROLLUP) is not decomposed into "
                "incrementally maintained grouping levels yet.",
                FULL_REFRESH_COST,
                "rewrite the view as a UNION ALL of plain GROUP BY queries, one per grouping level, "
                "so that each level is maintained incrementally.",
                (),
            )
        if self.view.has_cte or self.view.has_compound_set_op or self.view.has_window_func:
            return None  # these shapes are full-refresh (or windowed) already
        unordered = self._unordered_order_sensitive_aggregates()
        if unordered:
            return ExactRoutingFallback(
                ORDER_SENSITIVE_AGGREGATE_FULL_REFRESH,
                Strategy.FULL_REFRESH,
                "the result of " + ", ".join(unordered) + " depends on the order in which the "
                "group's rows are read, and only a full evaluation of the original query reads "
                "them as the live view does.",
                FULL_REFRESH_COST,
                "add an ORDER BY that covers the primary key of every base table inside the "
                "aggregate call (e.g. ARRAY_AGG(x ORDER BY <pk>)) to enable the group-scoped "
                "recompute.",
                tuple(unordered),
            )
        opaque = self._opaque_grouped_items()
        if opaque:
            return ExactRoutingFallback(
                OPAQUE_AGGREGATE_FULL_REFRESH,
                Strategy.FULL_REFRESH,
                "the expression(s) " + ", ".join(it.raw_expr for it in opaque) + " call a function "
                "that is not a known built-in over non-GROUP BY columns (a user-defined aggregate), "
                "whose order sensitivity is unknown.",
                FULL_REFRESH_COST,
                "no change is needed for an exact result.",
                tuple(it.raw_expr for it in opaque),
            )
        return None

    def _non_algebraic_aggregate_names(self) -> List[str]:
        """Names of the aggregate calls (select list and HAVING) that have no
        algebraic delta rule."""
        sg_d = _sqlglot_dialect(self.dialect)
        exprs = [it.raw_expr for it in self.view.select_items]
        if self.view.having_clause:
            exprs.append(self.view.having_clause)
        names = {
            name
            for raw in exprs
            for call in aggregate_calls(self._item_ast(raw) or sqlglot_exp.Null(), self._native_aggregate_names())
            for name in [call_name(call, sg_d)]
            if isinstance(call, sqlglot_exp.WithinGroup) or name not in _ALGEBRAIC_AGGREGATE_NAMES
        }
        return sorted(names)

    def _unrecognized_aggregate_group_fallback(self) -> Optional[ExactRoutingFallback]:
        """Announcement for a group-scoped recompute chosen because of
        aggregates the algebraic compiler does not decompose."""
        names = self._non_algebraic_aggregate_names()
        if not names:
            return None
        return ExactRoutingFallback(
            UNRECOGNIZED_AGGREGATE_GROUP_RECOMPUTE,
            Strategy.GROUP_SCOPED_RECOMPUTE,
            "aggregate(s) " + ", ".join(names) + " have no algebraic delta rule; each "
            "affected group is recomputed with the view's original aggregate.",
            GROUP_RECOMPUTE_COST,
            "create the indexes listed in recommended_base_indexes to keep the recompute "
            "O(rows in the group).",
            tuple(names),
        )

    def _unrecognized_aggregate_full_refresh_fallback(self) -> Optional[ExactRoutingFallback]:
        """Announcement for an ungrouped view (or a shape outside the group
        recompute) with aggregates that have no algebraic delta rule."""
        names = self._non_algebraic_aggregate_names()
        if not names:
            return None
        return ExactRoutingFallback(
            UNRECOGNIZED_AGGREGATE_FULL_REFRESH,
            Strategy.FULL_REFRESH,
            "aggregate(s) " + ", ".join(names) + " have no algebraic delta rule and the view "
            "has no plain GROUP BY whose groups could be recomputed separately.",
            FULL_REFRESH_COST,
            "no change is needed for an exact result.",
            tuple(names),
        )

    def _exact_routing_fallback_for(self, strategy: MaintenanceStrategy) -> Optional[ExactRoutingFallback]:
        """The routing fallback announced for the selected strategy, if any."""
        if strategy == MaintenanceStrategy.ATOMIC_QUERY_SYNC:
            fb = self._exact_full_refresh_fallback()
            if fb is None and not self.view.group_by_cols and not self.view.has_window_func:
                fb = self._unrecognized_aggregate_full_refresh_fallback()
            return fb
        if strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA and self._alg_mode != "window_partition":
            return self._unrecognized_aggregate_group_fallback()
        return None

    def _exact_routing_diagnostics(self, strategy: MaintenanceStrategy) -> Tuple[Diagnostic, ...]:
        """matview.diagnostics records for the routing fallback (emitted at
        registration and listed in plan.strategy_diagnostics)."""
        fb = self._exact_routing_fallback_for(strategy)
        if fb is None:
            return ()
        return (
            Diagnostic(
                code=fb.code,
                severity=DiagnosticSeverity.INFO,
                view=self.logical_view_name,
                strategy=fb.strategy,
                columns=(),
                expressions=fb.expressions,
                message=fb.message(),
                details={"maintenance_strategy": strategy.value},
            ),
        )

    # ---------------------------------------------------------------- types
    def _exact_routing_storage_type(self, item: SelectItem) -> Optional[str]:
        """Backing-column type for an item that contains an unrecognized
        aggregate (or an opaque call in a grouped view), or None when the
        regular inference applies. Raises when the native result type cannot
        be determined on PostgreSQL/MySQL: storing it with a guessed type
        would change the view's result type."""
        server_t = self._server_result_storage_type(item)
        if server_t is not None:
            return server_t
        scalar_t = self._native_scalar_storage_type(item)
        if scalar_t is not None:
            return scalar_t
        if self.dialect == Dialect.SQLITE:
            return None
        is_opaque = not item.is_aggregate and any(
            o.alias == item.alias and o.raw_expr == item.raw_expr for o in self._opaque_grouped_items()
        )
        pg_array_agg = self.dialect == Dialect.POSTGRES and item.agg_func == AggFunc.ARRAY_AGG
        if not is_opaque and not pg_array_agg and not self._has_unrecognized_aggregate(item):
            return None
        node = self._item_ast(item.raw_expr)
        resolved = self._routed_expr_storage_type(node, allow_aggregate=not is_opaque)
        if resolved is None:
            raise UnsupportedMaterializedViewError(
                f"Cannot determine the native result type of {item.raw_expr!r} (column {item.alias!r}) "
                f"on {self.dialect.value}; storing it with a guessed type would change the view's "
                f"result. Wrap the expression in an explicit CAST(... AS <type>)."
            )
        return resolved

    def _server_result_storage_type(self, item: SelectItem) -> Optional[str]:
        """The server-reported type of an item calling a function sqlglot does
        not model (matview.native_functions.probe_result_types)."""
        if self.dialect == Dialect.SQLITE or not self.view.native_result_types:
            return None
        native_t = dict(self.view.native_result_types).get(item.alias.lower())
        if native_t is None:
            return None
        node = self._item_ast(item.raw_expr)
        if node is None or node.find(sqlglot_exp.Anonymous) is None:
            return None
        # Verbatim: the server's own spelling is valid DDL (PostgreSQL
        # format_type, MySQL SHOW COLUMNS) and keeps modifiers such as UNSIGNED.
        return native_t.upper()

    def _native_scalar_storage_type(self, item: SelectItem) -> Optional[str]:
        """Native result type of a non-aggregate item that is one call of a
        scalar function classified through the engine catalog."""
        if item.is_aggregate or not self.view.native_scalar_types:
            return None
        node = self._item_ast(item.raw_expr)
        while isinstance(node, sqlglot_exp.Paren):
            node = node.this
        if not isinstance(node, sqlglot_exp.Anonymous):
            return None
        native_t = dict(self.view.native_scalar_types).get(f"{str(node.name).upper()}/{len(node.expressions)}")
        if native_t is None:
            return None
        return native_t if native_t == "BLOB" else self._normalize_dialect_type(native_t)

    def _routed_expr_storage_type(self, node: Optional[sqlglot_exp.Expr], *, allow_aggregate: bool) -> Optional[str]:
        """Native type of a routed select expression, or None when it is not
        modelled: an aggregate call, a CAST, PostgreSQL ``COALESCE(<expr>,
        '<literal>' | NULL)``, or a ``CASE`` whose non-NULL results all resolve
        to the same type (PostgreSQL/MySQL resolve a CASE of one type to it)."""
        while isinstance(node, sqlglot_exp.Paren):
            node = node.this
        if node is None:
            return None
        if isinstance(node, sqlglot_exp.Cast):
            to = node.args.get("to")
            if isinstance(to, sqlglot_exp.DataType):
                return self._normalize_dialect_type(to.sql(dialect=_sqlglot_dialect(self.dialect)))
            return None
        if (
            self.dialect == Dialect.POSTGRES
            and isinstance(node, sqlglot_exp.Coalesce)
            and isinstance(node.this, sqlglot_exp.Expr)
            and all(
                isinstance(e, sqlglot_exp.Null) or (isinstance(e, sqlglot_exp.Literal) and e.is_string)
                for e in node.expressions
            )
        ):
            # PostgreSQL: an untyped literal fallback adopts the first
            # argument's type. Other fallbacks resolve a common type that is
            # not modelled here.
            return self._routed_expr_storage_type(node.this, allow_aggregate=allow_aggregate)
        if isinstance(node, sqlglot_exp.Case):
            results = [i.args.get("true") for i in node.args.get("ifs") or ()] + [node.args.get("default")]
            types = {
                self._routed_expr_storage_type(r, allow_aggregate=allow_aggregate)
                for r in results
                if isinstance(r, sqlglot_exp.Expr) and not isinstance(r, sqlglot_exp.Null)
            }
            return types.pop() if len(types) == 1 and None not in types else None
        if allow_aggregate:
            calls = aggregate_calls(node, self._native_aggregate_names())
            if len(calls) == 1 and calls[0] is node:
                return self._aggregate_call_storage_type(node)
        return None

    def _arg_storage_type(self, arg: Optional[sqlglot_exp.Expr]) -> Optional[str]:
        if arg is None:
            return None
        while isinstance(arg, (sqlglot_exp.Paren, sqlglot_exp.Order, sqlglot_exp.Ordered, sqlglot_exp.Distinct)):
            exprs = arg.expressions if isinstance(arg, sqlglot_exp.Distinct) else [arg.this]
            nxt = exprs[0] if exprs else None
            if not isinstance(nxt, sqlglot_exp.Expr):
                return None
            arg = nxt
        sql = arg.sql(dialect=_sqlglot_dialect(self.dialect))
        if isinstance(arg, sqlglot_exp.Func):
            fn_up = call_name(arg, _sqlglot_dialect(self.dialect))
            if fn_up in _PG_RANGE_TO_MULTIRANGE and self.dialect == Dialect.POSTGRES:
                return fn_up
        c_def = self._lookup_col_def(sql)
        if c_def is not None:
            arg_t: Optional[str] = self._normalize_dialect_type(c_def.data_type)
        else:
            arg_t = self._infer_ast_expr_storage_type(sql)
        if arg_t is not None and self.dialect == Dialect.POSTGRES:
            arg_t = _pg_drop_typmod(arg_t)
        return arg_t

    def _aggregate_call_storage_type(self, call: sqlglot_exp.Expr) -> Optional[str]:
        name = call_name(call, _sqlglot_dialect(self.dialect))
        native = dict(self.view.native_aggregates)
        if isinstance(call, sqlglot_exp.Anonymous) and name in native:
            native_t = native[name]
            return self._normalize_dialect_type(native_t) if native_t else None
        raw_up = call.sql(dialect=_sqlglot_dialect(self.dialect)).upper()
        if " RETURNING " in f" {raw_up} ":
            return None
        if isinstance(call, sqlglot_exp.WithinGroup):
            order = call.args.get("expression")
            keys = order.expressions if isinstance(order, sqlglot_exp.Order) else []
            key_t = self._arg_storage_type(keys[0]) if len(keys) == 1 else None
            fraction = call.this.this if isinstance(call.this, sqlglot_exp.Func) else None
            is_array = isinstance(fraction, sqlglot_exp.Array)
            if self.dialect != Dialect.POSTGRES:
                return None
            if name in ("RANK", "DENSE_RANK"):
                return "BIGINT"
            if name in ("PERCENT_RANK", "CUME_DIST"):
                return "DOUBLE PRECISION"
            if name == "PERCENTILE_CONT":
                base = "INTERVAL" if key_t and key_t.upper() == "INTERVAL" else "DOUBLE PRECISION"
                return base + "[]" if is_array else base
            if name in ("PERCENTILE_DISC", "MODE") and key_t:
                return key_t + "[]" if is_array and name == "PERCENTILE_DISC" else key_t
            return None
        first = call.this if isinstance(call.this, sqlglot_exp.Expr) else (call.expressions[0] if call.expressions else None)
        if name == "ANY_VALUE":
            arg_t = self._arg_storage_type(first)
            if (
                self.dialect == Dialect.MYSQL
                and arg_t is not None
                and re.sub(r"\(\s*\d+\s*\)", "", arg_t).strip().upper()
                in ("TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER")
            ):
                return "BIGINT"  # native: MySQL 8.4/9.x ANY_VALUE(int column) is BIGINT
            return arg_t
        if self.dialect == Dialect.MYSQL:
            if name in ("JSON_ARRAYAGG", "JSON_OBJECTAGG"):
                return "JSON"
            if name == "GROUPING":
                return "BIGINT"  # native: MySQL 8.4/9.x GROUPING() is BIGINT
            return None
        if self.dialect != Dialect.POSTGRES:
            return None
        if name.startswith("JSONB_"):
            return "JSONB"
        if name.startswith("JSON_") or name in ("JSON_ARRAYAGG", "JSON_OBJECTAGG"):
            return "JSON"
        if name == "XMLAGG":
            return "XML"
        if name in _PG_FLOAT8_AGGREGATES:
            return "DOUBLE PRECISION"
        if name == "REGR_COUNT":
            return "BIGINT"
        if name == "GROUPING":
            return "INTEGER"
        if name in ("RANGE_AGG", "RANGE_INTERSECT_AGG"):
            arg_t = (self._arg_storage_type(first) or "").upper()
            if name == "RANGE_INTERSECT_AGG":
                return arg_t or None
            return _PG_RANGE_TO_MULTIRANGE.get(arg_t) or (arg_t if arg_t.endswith("MULTIRANGE") else None)
        if name == "ARRAY_AGG":
            arg_t = self._arg_storage_type(first)
            if not arg_t:
                return None
            return arg_t if arg_t.endswith("[]") else arg_t + "[]"
        return None


def _pg_drop_typmod(type_sql: str) -> str:
    """PostgreSQL polymorphic aggregates (ANY_VALUE, mode(), percentile_disc,
    ARRAY_AGG, ...) return their argument's type without its modifier
    (native: percentile_disc over numeric(10,2) is unconstrained numeric), so
    the backing column drops it too. Values keep their own scale/length.
    ``char(n)`` becomes ``bpchar`` (a bare ``char`` means ``char(1)``); a
    ``bit(n)`` keeps its modifier (a bare ``bit`` means ``bit(1)``)."""
    m = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_ ]*?)\s*\([^)]*\)\s*((?:\[\])*)\s*", type_sql)
    if not m:
        return type_sql
    base, dims = m.group(1).strip(), m.group(2)
    up = base.upper()
    if up in ("BIT", "BIT VARYING", "VARBIT"):
        return type_sql
    if up in ("CHAR", "CHARACTER", "BPCHAR"):
        base = "BPCHAR"
    return base + dims
