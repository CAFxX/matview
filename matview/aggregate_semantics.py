"""Classification of aggregate calls the algebraic compiler does not decompose.

The algebraic (Tier 1) compiler understands a fixed set of aggregates
(``AggFunc``). Every other aggregate -- ``JSON_AGG``, ``percentile_cont``,
``mode()``, ``ANY_VALUE``, ``regr_*``, user-defined aggregates, ... -- can
still be maintained exactly by re-running the view's ORIGINAL query: either
restricted to the affected groups (group-scoped recompute) or in full (full
refresh). This module answers the two questions the planner needs for that
routing, from the SQL text alone:

* which calls in an expression are aggregate calls (built-in names that
  sqlglot does not model are listed explicitly; see ``KNOWN_AGGREGATE_NAMES``);
* whether a call's result depends on the order in which the group's rows are
  fed to it. A group-scoped recompute reads the group's rows through a
  different plan than the live view (an extra ``WHERE`` on the group key), so
  it is only exact for aggregates whose result is a function of the group's
  multiset of rows. Order-sensitive calls are exact under a group-scoped
  recompute only with an aggregate-level ``ORDER BY`` that is a total order
  (it covers the primary key of every base table); otherwise the planner uses
  the full refresh of the original query.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Sequence

from sqlglot import exp

# Built-in aggregates (PostgreSQL 15-18, MySQL 8.4/9.x, MariaDB 10.11+,
# SQLite 3.x) that sqlglot parses as ``Anonymous`` instead of an ``AggFunc``.
KNOWN_AGGREGATE_NAMES = frozenset(
    {
        # PostgreSQL (pg_aggregate in pg_catalog, PG 18)
        "ARRAY_AGG",
        "JSON_AGG",
        "JSON_AGG_STRICT",
        "JSONB_AGG",
        "JSONB_AGG_STRICT",
        "JSON_OBJECT_AGG",
        "JSON_OBJECT_AGG_STRICT",
        "JSON_OBJECT_AGG_UNIQUE",
        "JSON_OBJECT_AGG_UNIQUE_STRICT",
        "JSONB_OBJECT_AGG",
        "JSONB_OBJECT_AGG_STRICT",
        "JSONB_OBJECT_AGG_UNIQUE",
        "JSONB_OBJECT_AGG_UNIQUE_STRICT",
        "XMLAGG",
        "RANGE_AGG",
        "RANGE_INTERSECT_AGG",
        # SQL/JSON standard spelling (PostgreSQL 16+, MySQL, MariaDB)
        "JSON_ARRAYAGG",
        "JSON_OBJECTAGG",
        # MySQL 8.0.24+ spatial aggregate
        "ST_COLLECT",
        # SQLite JSONB variants
        "JSONB_GROUP_ARRAY",
        "JSONB_GROUP_OBJECT",
    }
)

# Aggregate results that are a function of the group's multiset of rows
# (deterministic regardless of the input order). Anything not listed here,
# including user-defined aggregates, is treated as order-sensitive.
_ORDER_INSENSITIVE_CLASSES: tuple[type, ...] = (
    exp.Count,
    exp.Min,
    exp.Max,
    exp.Sum,  # float SUM/AVG/VAR are governed by the float policy (planner)
    exp.Avg,
    exp.Stddev,
    exp.StddevPop,
    exp.StddevSamp,
    exp.Variance,
    exp.VariancePop,
    exp.BitwiseAndAgg,
    exp.BitwiseOrAgg,
    exp.BitwiseXorAgg,
    exp.LogicalAnd,
    exp.LogicalOr,
    exp.RegrCount,
    exp.Grouping,
)
_ORDER_INSENSITIVE_NAMES = frozenset(
    {
        "COUNT", "MIN", "MAX", "SUM", "TOTAL", "AVG", "BIT_AND", "BIT_OR", "BIT_XOR",
        "BOOL_AND", "BOOL_OR", "EVERY", "REGR_COUNT", "RANGE_AGG", "RANGE_INTERSECT_AGG",
    }
)


def call_name(node: exp.Expr, dialect: str = "postgres") -> str:
    """Upper-case SQL name of an aggregate call node as ``dialect`` spells it
    (``WITHIN GROUP`` unwrapped)."""
    if isinstance(node, exp.WithinGroup):
        inner = node.this
        return call_name(inner, dialect) if isinstance(inner, exp.Expr) else "WITHIN GROUP"
    if isinstance(node, exp.Anonymous):
        return str(node.name).upper()
    rendered = node.sql(dialect=dialect)
    head = rendered.split("(", 1)[0].strip()
    return head.upper() if head else type(node).__name__.upper()


def is_aggregate_call(node: exp.Expr, extra_names: frozenset[str] = frozenset()) -> bool:
    """Whether ``node`` is an aggregate call (not a window function)."""
    if isinstance(node.parent, exp.Window):
        return False
    if isinstance(node, exp.WithinGroup):
        return True
    if isinstance(node, exp.AggFunc):
        return not isinstance(node.parent, exp.WithinGroup)
    if isinstance(node, exp.Anonymous):
        name = str(node.name).upper()
        return name in KNOWN_AGGREGATE_NAMES or name in extra_names
    return False


def aggregate_calls(root: exp.Expr, extra_names: frozenset[str] = frozenset()) -> List[exp.Expr]:
    """Outermost aggregate calls in ``root``, excluding subqueries and windows."""
    found: List[exp.Expr] = []

    def _walk(node: exp.Expr) -> None:
        if isinstance(node, (exp.Subquery, exp.Select)) and node is not root:
            return
        if isinstance(node, exp.Window):
            return
        if is_aggregate_call(node, extra_names):
            found.append(node)
            return
        for child in node.iter_expressions():
            _walk(child)

    _walk(root)
    return found


def is_order_sensitive(call: exp.Expr) -> bool:
    """Whether the call's result can depend on the order of its input rows."""
    if isinstance(call, exp.WithinGroup):
        # Ordered-set (percentile_cont/disc, mode) and hypothetical-set
        # aggregates sort their input by the WITHIN GROUP key; the result is a
        # function of the sorted multiset. (Values that compare equal but differ
        # -- a nondeterministic collation -- are a documented limitation.)
        return False
    if isinstance(call, _ORDER_INSENSITIVE_CLASSES):
        return False
    if isinstance(call, exp.Anonymous) and str(call.name).upper() in _ORDER_INSENSITIVE_NAMES:
        return False
    return True


def _is_distinct_call(call: exp.Expr) -> bool:
    if call.args.get("distinct"):
        return True
    return call.find(exp.Distinct) is not None


def aggregate_order_keys(call: exp.Expr) -> Optional[List[exp.Expr]]:
    """The aggregate-level ``ORDER BY`` keys of ``call``, or None without one."""

    def _find(node: exp.Expr) -> Optional[exp.Order]:
        for child in node.iter_expressions():
            if isinstance(child, (exp.Subquery, exp.Select, exp.Window)):
                continue
            if isinstance(child, exp.Order):
                return child
            if is_aggregate_call(child):
                continue
            nested = _find(child)
            if nested is not None:
                return nested
        return None

    order = _find(call)
    if order is None:
        return None
    return [o.this if isinstance(o, exp.Ordered) else o for o in order.expressions]


def has_total_input_order(
    call: exp.Expr,
    covers_every_primary_key: Callable[[Sequence[exp.Expr], exp.Expr], bool],
) -> bool:
    """Whether an order-sensitive call is fed its rows in one total order.

    True when the call has an aggregate-level ``ORDER BY`` whose keys cover
    the primary key of every base table (``covers_every_primary_key``) and the
    call is not ``DISTINCT`` (a DISTINCT call orders the surviving duplicates
    by an arbitrary representative row).
    """
    if _is_distinct_call(call):
        return False
    keys = aggregate_order_keys(call)
    if not keys:
        return False
    return covers_every_primary_key(keys, call)


def iter_unordered_order_sensitive_calls(
    root: exp.Expr,
    covers_every_primary_key: Callable[[Sequence[exp.Expr], exp.Expr], bool],
    extra_names: frozenset[str] = frozenset(),
) -> Iterator[exp.Expr]:
    """Order-sensitive aggregate calls in ``root`` without a total input order."""
    for call in aggregate_calls(root, extra_names):
        if is_order_sensitive(call) and not has_total_input_order(call, covers_every_primary_key):
            yield call


def has_grouping_sets(select: exp.Expr) -> bool:
    """Whether a SELECT groups by ROLLUP, CUBE or GROUPING SETS (any dialect)."""
    group = select.args.get("group") if isinstance(select, exp.Select) else None
    if not isinstance(group, exp.Group):
        return False
    if group.args.get("rollup") or group.args.get("cube") or group.args.get("grouping_sets") or group.args.get("totals"):
        return True
    return group.find(exp.Rollup, exp.Cube, exp.GroupingSets) is not None
