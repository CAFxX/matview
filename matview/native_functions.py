"""Classify function names the SQL text alone cannot classify.

``SELECT my_agg(x) FROM t`` is an aggregate when ``my_agg`` is a user-defined
aggregate and a per-row scalar call when it is a scalar function: the two
views have different results (one row versus one row per base row), and the
SQL text does not tell them apart. At registration the manager asks the
engine's catalog about every function name that is neither a known built-in
aggregate nor modelled by sqlglot, and:

* routes true aggregates through the exact routing of unrecognized aggregates
  (``ViewSpec.native_aggregates``; group-scoped recompute or full refresh);
* leaves scalar functions as they are;
* refuses the view when the engine cannot classify a name (missing catalog
  privileges, an introspection pragma compiled out, an overload set mixing
  aggregates and scalar functions of the same arity): it never guesses.

Catalogs: PostgreSQL ``pg_proc.prokind``; SQLite ``PRAGMA function_list``
(aggregate types ``a`` and ``w``, matched by arity); MySQL
``performance_schema.user_defined_functions`` (loadable UDFs) and
``information_schema.ROUTINES`` (stored functions, never aggregates);
MariaDB ``mysql.func`` (loadable UDFs) and ``mysql.proc.aggregate`` (stored
``CREATE AGGREGATE FUNCTION``). Offline compilation (no connection) cannot
consult a catalog and keeps the text-based classification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp

from .aggregate_semantics import KNOWN_AGGREGATE_NAMES
from .engines import Dialect, UnsupportedMaterializedViewError, _sqlglot_dialect
from .logical_ir import SelectItem, ViewSpec
from .sql_frontend import _FN_TO_AGG_ENUM, _STATISTICAL_ALIAS_NAMES

AGGREGATE = "aggregate"
SCALAR = "scalar"
UNKNOWN = "unknown"
# The engine offers no catalog to ask (a SQLite driver or build without
# PRAGMA function_list): the call is classified from the SQL text, as offline
# compilation does, and the registration announces it.
TEXT_ONLY = "text_only"

_POLYMORPHIC_PG_TYPES = frozenset(
    {"anyelement", "anyarray", "anynonarray", "anyenum", "anyrange", "anymultirange",
     "anycompatible", "anycompatiblearray", "anycompatiblenonarray", "anycompatiblerange",
     "anycompatiblemultirange", "internal", "record", "\"any\""}
)


@dataclass(frozen=True)
class FunctionCall:
    name: str  # upper case
    arity: int


@dataclass(frozen=True)
class FunctionClass:
    kind: str  # AGGREGATE, SCALAR or UNKNOWN
    return_type: Optional[str] = None
    reason: str = ""


def _parse(sql: str, dialect: Dialect) -> Optional[sqlglot_exp.Expr]:
    try:
        return sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except (sqlglot.errors.SqlglotError, ValueError):
        return None


def unclassified_calls(spec: ViewSpec, dialect: Dialect) -> List[FunctionCall]:
    """Calls in the select list and HAVING that sqlglot does not model and
    that are not known built-in aggregates (outside window functions)."""
    exprs = [it.raw_expr for it in spec.select_items]
    if spec.having_clause:
        exprs.append(spec.having_clause)
    seen: Set[Tuple[str, int]] = set()
    out: List[FunctionCall] = []
    for raw in exprs:
        node = _parse(raw, dialect)
        if node is None:
            continue
        for anon in node.find_all(sqlglot_exp.Anonymous):
            if anon.find_ancestor(sqlglot_exp.Window) is not None:
                continue
            name = str(anon.name).upper()
            if not name or name in KNOWN_AGGREGATE_NAMES or name in _FN_TO_AGG_ENUM or name in _STATISTICAL_ALIAS_NAMES or name == "COUNT":
                continue
            key = (name, len(anon.expressions))
            if key not in seen:
                seen.add(key)
                out.append(FunctionCall(name=name, arity=len(anon.expressions)))
    return out


def classify(conn: Any, dialect: Dialect, is_mariadb: bool, calls: Sequence[FunctionCall]) -> Dict[FunctionCall, FunctionClass]:
    if not calls:
        return {}
    if dialect == Dialect.POSTGRES:
        return _classify_postgres(conn, calls)
    if dialect == Dialect.SQLITE:
        return _classify_sqlite(conn, calls)
    return _classify_mysql(conn, calls, is_mariadb)


def _rows(conn: Any, sql: str, params: Sequence[Any] = ()) -> List[Tuple[Any, ...]]:
    cur = conn.cursor()
    try:
        cur.execute(sql, tuple(params)) if params else cur.execute(sql)
        return [tuple(r) for r in cur.fetchall()]
    finally:
        try:
            cur.close()
        except Exception:
            pass


def _classify_postgres(conn: Any, calls: Sequence[FunctionCall]) -> Dict[FunctionCall, FunctionClass]:
    names = sorted({c.name for c in calls})
    rows = _rows(
        conn,
        "SELECT upper(p.proname), p.prokind, p.pronargs, p.provariadic <> 0, "
        "pg_catalog.format_type(p.prorettype, NULL), p.proretset "
        "FROM pg_catalog.pg_proc p WHERE upper(p.proname) = ANY(%s) "
        "AND pg_catalog.pg_function_is_visible(p.oid)",
        (names,),
    )
    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        cands = [r for r in rows if r[0] == call.name and (r[2] == call.arity or (r[3] and call.arity >= r[2] - 1))]
        kinds = {r[1] for r in cands}
        if not cands:
            out[call] = FunctionClass(UNKNOWN, reason="no visible function of that name and arity in pg_proc")
        elif kinds == {"a"}:
            types = {r[4] for r in cands}
            rt = types.pop() if len(types) == 1 else None
            out[call] = FunctionClass(AGGREGATE, None if rt is None or rt in _POLYMORPHIC_PG_TYPES else rt)
        elif "a" in kinds:
            out[call] = FunctionClass(UNKNOWN, reason="overloads of that arity are both aggregates and functions")
        else:
            types = {r[4] for r in cands if not r[5]}
            rt = types.pop() if len(types) == 1 and len(cands) == 1 else None
            out[call] = FunctionClass(SCALAR, None if rt is None or rt in _POLYMORPHIC_PG_TYPES else rt)
    return out


def _classify_sqlite(conn: Any, calls: Sequence[FunctionCall]) -> Dict[FunctionCall, FunctionClass]:
    try:
        rows = _rows(conn, "PRAGMA function_list")
    except Exception as e:  # compiled out (SQLITE_OMIT_INTROSPECTION_PRAGMAS) or a driver without it
        return {c: FunctionClass(TEXT_ONLY, reason=f"PRAGMA function_list is unavailable ({e})") for c in calls}
    if not rows:
        return {c: FunctionClass(TEXT_ONLY, reason="PRAGMA function_list returned no functions") for c in calls}
    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        cands = [r for r in rows if str(r[0]).upper() == call.name and (int(r[4]) == call.arity or int(r[4]) < 0)]
        exact = [r for r in cands if int(r[4]) == call.arity]
        if exact:
            cands = exact  # SQLite prefers the exact-arity registration
        kinds = {"aggregate" if str(r[2]) in ("a", "w") else "scalar" for r in cands}
        if not cands:
            out[call] = FunctionClass(UNKNOWN, reason="no function of that name and arity in PRAGMA function_list")
        elif len(kinds) > 1:
            out[call] = FunctionClass(UNKNOWN, reason="registered both as an aggregate and as a scalar function")
        else:
            # A scalar result keeps its own storage class only in a column
            # without affinity (TEXT affinity would turn 3 into '3').
            out[call] = FunctionClass(AGGREGATE) if kinds == {"aggregate"} else FunctionClass(SCALAR, "BLOB")
    return out


def _classify_mysql(conn: Any, calls: Sequence[FunctionCall], is_mariadb: bool) -> Dict[FunctionCall, FunctionClass]:
    names = sorted({c.name for c in calls})
    ph = ", ".join(["%s"] * len(names))
    unreadable: List[str] = []

    def _try(label: str, sql: str) -> List[Tuple[Any, ...]]:
        try:
            return _rows(conn, sql, names)
        except Exception:
            unreadable.append(label)
            return []

    if is_mariadb:
        udfs = _try("mysql.func", f"SELECT UPPER(name), type, ret FROM mysql.func WHERE UPPER(name) IN ({ph})")
        stored = _try(
            "mysql.proc",
            f"SELECT UPPER(name), aggregate, returns FROM mysql.proc WHERE db = DATABASE() AND type = 'FUNCTION' AND UPPER(name) IN ({ph})",
        )
    else:
        udfs = _try(
            "performance_schema.user_defined_functions",
            f"SELECT UPPER(UDF_NAME), UDF_TYPE, UDF_RETURN_TYPE FROM performance_schema.user_defined_functions WHERE UPPER(UDF_NAME) IN ({ph})",
        )
        stored = _try(
            "information_schema.ROUTINES",
            f"SELECT UPPER(ROUTINE_NAME), 'NONE', DTD_IDENTIFIER FROM information_schema.ROUTINES "
            f"WHERE ROUTINE_SCHEMA = DATABASE() AND ROUTINE_TYPE = 'FUNCTION' AND UPPER(ROUTINE_NAME) IN ({ph})",
        )
    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        u = [r for r in udfs if r[0] == call.name]
        s = [r for r in stored if r[0] == call.name]
        if u:
            is_agg = str(u[0][1]).lower() == "aggregate"
            out[call] = FunctionClass(AGGREGATE if is_agg else SCALAR)
        elif s:
            agg_flag = s[0][1]
            if isinstance(agg_flag, (bytes, bytearray)):
                agg_flag = bytes(agg_flag).decode("utf-8", "replace")
            is_agg = str(agg_flag).upper() == "GROUP"
            ret = s[0][2]
            if isinstance(ret, (bytes, bytearray)):  # MariaDB mysql.proc.returns is a BLOB
                ret = bytes(ret).decode("utf-8", "replace")
            out[call] = FunctionClass(AGGREGATE if is_agg else SCALAR, str(ret).upper() if ret else None)
        elif unreadable:
            out[call] = FunctionClass(UNKNOWN, reason="cannot read " + " and ".join(unreadable))
        else:
            # Not a user-defined function: a built-in, and every built-in
            # aggregate of MySQL/MariaDB is modelled (KNOWN_AGGREGATE_NAMES).
            out[call] = FunctionClass(SCALAR)
    return out


def mark_native_aggregates(spec: ViewSpec, dialect: Dialect) -> ViewSpec:
    """Mark select items calling a catalog-classified aggregate as aggregates
    (``agg_func`` None: no algebraic rule), recursively for additional views."""
    names = frozenset(n for n, _ in spec.native_aggregates)
    if not names:
        return spec
    items: List[SelectItem] = []
    changed = False
    for it in spec.select_items:
        if not it.is_aggregate:
            node = _parse(it.raw_expr, dialect)
            if node is not None and any(
                str(a.name).upper() in names and a.find_ancestor(sqlglot_exp.Window) is None
                for a in node.find_all(sqlglot_exp.Anonymous)
            ):
                it = replace(it, is_aggregate=True, agg_func=None, arg_expr=it.raw_expr, source_col=None)
                changed = True
        items.append(it)
    return replace(spec, select_items=tuple(items)) if changed else spec


def apply_catalog_classification(
    spec: ViewSpec,
    dialect: Dialect,
    classify_calls: Callable[[Sequence[FunctionCall]], Dict[FunctionCall, FunctionClass]],
    on_text_only: Optional[Callable[[ViewSpec, Sequence[FunctionCall], str], None]] = None,
) -> ViewSpec:
    """Classify ``spec``'s unclassified calls and route its aggregates; raise
    UnsupportedMaterializedViewError for a name the engine cannot classify.
    Calls the engine has no catalog for (``TEXT_ONLY``) keep the text-based
    classification and are reported through ``on_text_only``."""
    additional = tuple(
        apply_catalog_classification(a, dialect, classify_calls, on_text_only) for a in spec.additional_views
    )
    if additional != spec.additional_views:
        spec = replace(spec, additional_views=additional)
    calls = unclassified_calls(spec, dialect)
    if not calls:
        return spec
    classes = classify_calls(calls)
    text_only = [c for c, k in classes.items() if k.kind == TEXT_ONLY]
    if text_only and on_text_only is not None:
        on_text_only(spec, text_only, classes[text_only[0]].reason)
    unknown = [(c, k) for c, k in classes.items() if k.kind == UNKNOWN]
    if unknown:
        listed = "; ".join(f"{c.name}/{c.arity}: {k.reason}" for c, k in unknown)
        raise UnsupportedMaterializedViewError(
            f"View {spec.view_name!r}: cannot tell whether {', '.join(sorted({c.name for c, _ in unknown}))} "
            f"is an aggregate or a scalar function ({listed}). An aggregate and a scalar call give "
            f"different results, so matview does not guess. Make the function visible in the engine's "
            f"catalog to this connection (or rename the overload) and register again."
        )
    aggs: Dict[str, Optional[str]] = {}
    scalar_types: Dict[str, str] = dict(spec.native_scalar_types)
    for c, k in classes.items():
        if k.kind == AGGREGATE:
            aggs[c.name] = k.return_type
        elif k.return_type:
            scalar_types[f"{c.name}/{c.arity}"] = k.return_type
    if scalar_types != dict(spec.native_scalar_types):
        spec = replace(spec, native_scalar_types=tuple(sorted(scalar_types.items())))
    if not aggs:
        return spec
    merged = dict(spec.native_aggregates)
    merged.update(aggs)
    return mark_native_aggregates(replace(spec, native_aggregates=tuple(sorted(merged.items()))), dialect)


def native_aggregate_names(spec: ViewSpec) -> frozenset[str]:
    return frozenset(n for n, _ in spec.native_aggregates)
