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

import re
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp

from .aggregate_semantics import KNOWN_AGGREGATE_NAMES
from .engines import Dialect, UnsupportedMaterializedViewError, _sqlglot_dialect
from .builtin_functions import MYSQL_BUILTIN_SCALAR_FUNCTIONS
from .logical_ir import SelectItem, ViewSpec
from .sql_frontend import _FN_TO_AGG_ENUM, _STATISTICAL_ALIAS_NAMES

AGGREGATE = "aggregate"
SCALAR = "scalar"
UNKNOWN = "unknown"
# The engine offers no catalog this connection can read (a SQLite driver or
# build without PRAGMA function_list, a MySQL/MariaDB user without access to
# the function catalogs): the call may be an aggregate or a scalar, so the
# view is maintained by the always-exact full refresh of the original query.
UNCLASSIFIABLE = "unclassifiable"

_POLYMORPHIC_PG_TYPES = frozenset(
    {"anyelement", "anyarray", "anynonarray", "anyenum", "anyrange", "anymultirange",
     "anycompatible", "anycompatiblearray", "anycompatiblenonarray", "anycompatiblerange",
     "anycompatiblemultirange", "internal", "record", "\"any\""}
)


@dataclass(frozen=True)
class FunctionCall:
    name: str  # upper case
    arity: int
    schema: Optional[str] = None  # qualifier as written (PostgreSQL schema, MySQL database)


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
    seen: Set[Tuple[str, str, int]] = set()
    out: List[FunctionCall] = []
    for raw in exprs:
        node = _parse(raw, dialect)
        if node is None:
            continue
        for anon in node.find_all(sqlglot_exp.Anonymous):
            if anon.find_ancestor(sqlglot_exp.Window) is not None:
                continue
            name = str(anon.name).upper()
            parent = anon.parent
            schema = (
                str(parent.this.name) if isinstance(parent, sqlglot_exp.Dot) and parent.expression is anon
                and isinstance(parent.this, (sqlglot_exp.Identifier, sqlglot_exp.Column)) else None
            )
            if not name or name in KNOWN_AGGREGATE_NAMES or name in _FN_TO_AGG_ENUM or name in _STATISTICAL_ALIAS_NAMES or name == "COUNT":
                continue
            if schema is None and dialect == Dialect.MYSQL and name in MYSQL_BUILTIN_SCALAR_FUNCTIONS:
                # MySQL/MariaDB resolve an unqualified built-in name before any
                # stored or loadable function (verified natively on 8.4 and
                # 11.8: a stored crc32() is reachable only qualified), so the
                # name cannot be shadowed: no catalog lookup needed. PostgreSQL
                # and SQLite let user functions shadow built-ins: always ask.
                continue
            key = (schema or "", name, len(anon.expressions))
            if key not in seen:
                seen.add(key)
                out.append(FunctionCall(name=name, arity=len(anon.expressions), schema=schema))
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
    # Candidates as PostgreSQL resolves them: a qualified call looks only in
    # its schema, an unqualified one only at visible functions; a call
    # matches when its argument count is between the required and the
    # declared parameters (DEFAULT values) or reaches a VARIADIC parameter.
    rows = _rows(
        conn,
        "SELECT upper(p.proname), p.prokind, p.pronargs, p.provariadic <> 0, "
        "pg_catalog.format_type(p.prorettype, NULL), p.proretset, p.pronargdefaults, n.nspname, "
        "pg_catalog.pg_function_is_visible(p.oid) "
        "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
        "WHERE upper(p.proname) = ANY(%s)",
        (names,),
    )

    def _matches(r: Tuple[Any, ...], call: FunctionCall) -> bool:
        if r[0] != call.name:
            return False
        if call.schema is not None:
            if r[7] != call.schema and r[7].lower() != call.schema.lower():
                return False
        elif not r[8]:
            return False
        required = r[2] - r[6]
        return required <= call.arity <= r[2] or (bool(r[3]) and call.arity >= r[2] - 1)

    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        cands = [r for r in rows if _matches(r, call)]
        kinds = {r[1] for r in cands}
        if not cands:
            where = f"schema {call.schema}" if call.schema else "the search_path"
            out[call] = FunctionClass(UNKNOWN, reason=f"no function of that name and arity in {where} (pg_proc)")
        elif kinds == {"a"}:
            types = {r[4] for r in cands}
            rt = types.pop() if len(types) == 1 else None
            out[call] = FunctionClass(AGGREGATE, None if rt is None or rt in _POLYMORPHIC_PG_TYPES else rt)
        elif "a" in kinds:
            # Which overload wins depends on argument types (e.g. a user
            # md5(int) aggregate next to pg_catalog.md5(text)); either may be
            # called, so only the full refresh is exact.
            out[call] = FunctionClass(UNCLASSIFIABLE, reason="overloads of that arity are both aggregates and functions")
        else:
            types = {r[4] for r in cands if not r[5]}
            rt = types.pop() if len(types) == 1 and len(cands) == 1 else None
            out[call] = FunctionClass(SCALAR, None if rt is None or rt in _POLYMORPHIC_PG_TYPES else rt)
    return out


def _classify_sqlite(conn: Any, calls: Sequence[FunctionCall]) -> Dict[FunctionCall, FunctionClass]:
    try:
        rows = _rows(conn, "PRAGMA function_list")
    except Exception as e:  # compiled out (SQLITE_OMIT_INTROSPECTION_PRAGMAS) or a driver without it
        return {c: FunctionClass(UNCLASSIFIABLE, reason=f"PRAGMA function_list is unavailable ({e})") for c in calls}
    if not rows:
        return {c: FunctionClass(UNCLASSIFIABLE, reason="PRAGMA function_list returned no functions") for c in calls}
    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        cands = [r for r in rows if str(r[0]).upper() == call.name and (int(r[4]) == call.arity or int(r[4]) < 0)]
        exact = [r for r in cands if int(r[4]) == call.arity]
        if exact:
            cands = exact  # SQLite prefers the exact-arity registration
        user = [r for r in cands if int(r[1]) == 0]
        if user:
            cands = user  # an application-defined function overrides the built-in
        kinds = {"aggregate" if str(r[2]) in ("a", "w") else "scalar" for r in cands}
        if not cands:
            out[call] = FunctionClass(UNKNOWN, reason="no function of that name and arity in PRAGMA function_list")
        elif len(kinds) > 1:
            out[call] = FunctionClass(UNCLASSIFIABLE, reason="registered both as an aggregate and as a scalar function")
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

    # Stored functions: the database of a qualified call, else the current one.
    if is_mariadb:
        udfs = _try("mysql.func", f"SELECT UPPER(name), type, ret FROM mysql.func WHERE UPPER(name) IN ({ph})")
        stored = _try(
            "mysql.proc",
            f"SELECT UPPER(name), aggregate, returns, db, db = DATABASE() FROM mysql.proc "
            f"WHERE type = 'FUNCTION' AND UPPER(name) IN ({ph})",
        )
    else:
        udfs = _try(
            "performance_schema.user_defined_functions",
            f"SELECT UPPER(UDF_NAME), UDF_TYPE, UDF_RETURN_TYPE FROM performance_schema.user_defined_functions WHERE UPPER(UDF_NAME) IN ({ph})",
        )
        stored = _try(
            "information_schema.ROUTINES",
            f"SELECT UPPER(ROUTINE_NAME), 'NONE', DTD_IDENTIFIER, ROUTINE_SCHEMA, ROUTINE_SCHEMA = DATABASE() "
            f"FROM information_schema.ROUTINES WHERE ROUTINE_TYPE = 'FUNCTION' AND UPPER(ROUTINE_NAME) IN ({ph})",
        )

    def _db_match(r: Tuple[Any, ...], call: FunctionCall) -> bool:
        db = r[3].decode() if isinstance(r[3], (bytes, bytearray)) else str(r[3])
        return db.lower() == call.schema.lower() if call.schema is not None else bool(r[4])

    out: Dict[FunctionCall, FunctionClass] = {}
    for call in calls:
        u = [r for r in udfs if r[0] == call.name] if call.schema is None else []
        s = [r for r in stored if r[0] == call.name and _db_match(r, call)]
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
            out[call] = FunctionClass(
                UNCLASSIFIABLE, reason="this connection cannot read " + " and ".join(unreadable)
            )
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
) -> ViewSpec:
    """Classify ``spec``'s unclassified calls and route its aggregates; raise
    UnsupportedMaterializedViewError for a name the catalog does not know.
    Calls the engine has no readable catalog for (``UNCLASSIFIABLE``) are
    recorded in ``unclassified_functions``: the planner maintains the view by
    full refresh (exact whatever the function is)."""
    additional = tuple(apply_catalog_classification(a, dialect, classify_calls) for a in spec.additional_views)
    if additional != spec.additional_views:
        spec = replace(spec, additional_views=additional)
    spec = replace(spec, functions_classified=True)
    calls = unclassified_calls(spec, dialect)
    if not calls:
        return spec
    classes = classify_calls(calls)
    kinds_by_name: Dict[str, Set[str]] = {}
    for c, k in classes.items():
        kinds_by_name.setdefault(c.name, set()).add(k.kind)
    for c, k in list(classes.items()):
        if AGGREGATE in kinds_by_name[c.name] and kinds_by_name[c.name] - {AGGREGATE}:
            classes[c] = FunctionClass(UNCLASSIFIABLE, reason="calls of that name resolve to both an aggregate and another function")
    unclassifiable = sorted({(c.name, k.reason) for c, k in classes.items() if k.kind == UNCLASSIFIABLE})
    if unclassifiable:
        merged_u = dict(spec.unclassified_functions)
        merged_u.update(dict(unclassifiable))
        spec = replace(spec, unclassified_functions=tuple(sorted(merged_u.items())))
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


def items_with_unmodelled_calls(spec: ViewSpec, dialect: Dialect) -> List[SelectItem]:
    """Select items calling a function sqlglot does not model (windows included)."""
    out: List[SelectItem] = []
    for it in spec.select_items:
        node = _parse(it.raw_expr, dialect)
        if node is not None and node.find(sqlglot_exp.Anonymous) is not None:
            out.append(it)
    return out


def probe_result_types(conn: Any, dialect: Dialect, select_sql: str) -> Dict[str, str]:
    """Output column -> native type of ``select_sql``, asked from the server
    without reading data; {} when the probe fails (the caller keeps the
    static inference). The probe leaves no object behind."""
    name = f"__matview_probe_{uuid.uuid4().hex[:12]}"
    body = select_sql.strip().rstrip(";")
    try:
        if dialect == Dialect.POSTGRES:
            savepoint = not getattr(conn, "autocommit", False)
            cur = conn.cursor()
            try:
                if savepoint:
                    cur.execute(f"SAVEPOINT {name}")
                try:
                    cur.execute(f"CREATE TEMPORARY VIEW {name} AS {body}")
                    cur.execute(
                        "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) FROM pg_catalog.pg_attribute a "
                        "WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
                        (f"pg_temp.{name}",),
                    )
                    rows = list(cur.fetchall())
                    cur.execute(f"DROP VIEW pg_temp.{name}")
                except Exception:
                    if savepoint:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    raise
                finally:
                    if savepoint:
                        cur.execute(f"RELEASE SAVEPOINT {name}")
            finally:
                cur.close()
            return {str(r[0]).lower(): str(r[1]) for r in rows}
        # MySQL/MariaDB: the result-set metadata of the query itself (a
        # CREATE TABLE ... AS SELECT widens INT results to BIGINT, so its
        # column types differ from what the live view returns).
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT * FROM ({body}) AS __mv_probe_q LIMIT 0")
            desc = list(cur.description or ())
            fields = list(getattr(getattr(cur, "_result", None), "fields", None) or ())
            cur.fetchall()
        finally:
            cur.close()
        out: Dict[str, str] = {}
        for i, d in enumerate(desc):
            f = fields[i] if i < len(fields) else None
            typ = _mysql_type_from_metadata(d, f)
            if typ is not None:
                out[str(d[0]).lower()] = typ
        return out
    except Exception:
        return {}


_MYSQL_UNSIGNED_FLAG = 32
_MYSQL_BINARY_CHARSET = 63
_MYSQL_INT_CODES = {1: "TINYINT", 2: "SMALLINT", 9: "MEDIUMINT", 3: "INT", 8: "BIGINT"}


def _mysql_type_from_metadata(desc: Sequence[Any], field: Any) -> Optional[str]:
    """DDL type reproducing a MySQL/MariaDB result column's protocol type, or
    None when not modelled (the static inference then applies)."""
    code = desc[1]
    unsigned = bool(field is not None and int(getattr(field, "flags", 0)) & _MYSQL_UNSIGNED_FLAG)
    if code in _MYSQL_INT_CODES:
        return _MYSQL_INT_CODES[code] + (" UNSIGNED" if unsigned else "")
    if code == 4:
        return "FLOAT"
    if code == 5:
        return "DOUBLE"
    if code in (0, 246):  # DECIMAL / NEWDECIMAL: precision and scale from the metadata
        precision, scale = desc[4], desc[5]
        if precision is None or scale is None:
            return None
        digits = max(int(precision) - (1 if int(scale) > 0 else 0) - (0 if unsigned else 1), 1)
        return f"DECIMAL({min(max(digits, int(scale)), 65)}, {int(scale)})"
    if code == 10:
        return "DATE"
    if code in (7, 12):
        fsp = int(desc[5] or 0)
        base = "TIMESTAMP" if code == 7 else "DATETIME"
        return f"{base}({fsp})" if 0 < fsp <= 6 else base
    if code == 11:
        return "TIME"
    if code == 13:
        return "YEAR"
    if code == 245:
        return "JSON"
    if code in (15, 253, 254):  # VARCHAR / VAR_STRING / STRING
        charsetnr = int(getattr(field, "charsetnr", 0) or 0) if field is not None else 0
        length = int(desc[3] or 0)
        if charsetnr == _MYSQL_BINARY_CHARSET:
            return f"VARBINARY({max(length, 1)})"
        # The metadata length is in bytes; using it as the character count
        # over-allocates but never truncates, whatever the character set.
        if not length:
            return None
        return f"VARCHAR({length})" if length <= 4096 else "LONGTEXT"
    return None


def _modelled_function_names(spec: ViewSpec, dialect: Dialect) -> Set[str]:
    """Names of the functions sqlglot models (Func nodes other than
    Anonymous) in the select list and HAVING, as the dialect spells them."""
    exprs = [it.raw_expr for it in spec.select_items]
    if spec.having_clause:
        exprs.append(spec.having_clause)
    sg_d = _sqlglot_dialect(dialect)
    out: Set[str] = set()
    for raw in exprs:
        node = _parse(raw, dialect)
        if node is None:
            continue
        for f in node.find_all(sqlglot_exp.Func):
            if isinstance(f, sqlglot_exp.Anonymous):
                continue
            head = f.sql(dialect=sg_d).split("(", 1)[0].strip().upper()
            if head and re.fullmatch(r"[A-Z_][A-Z0-9_]*", head):
                out.add(head)
            key = getattr(f, "key", "")
            if key:
                out.add(str(key).upper())
    return out


def overridden_modelled_functions(conn: Any, spec: ViewSpec, dialect: Dialect) -> List[Tuple[str, str]]:
    """Built-in functions that sqlglot models (min, max, abs, lower, md5, ...)
    for which the engine also has a user-defined function of the same name
    that an unqualified call can reach: the algebraic rules assume the
    built-in, so the view needs the full refresh. PostgreSQL: visible
    functions outside pg_catalog (overloads are chosen by argument type);
    SQLite: application-defined functions (PRAGMA function_list lists both).
    MySQL/MariaDB built-ins cannot be shadowed by unqualified calls."""
    names = _modelled_function_names(spec, dialect)
    if not names or dialect == Dialect.MYSQL:
        return []
    try:
        if dialect == Dialect.POSTGRES:
            rows = _rows(
                conn,
                "SELECT DISTINCT upper(p.proname) FROM pg_catalog.pg_proc p "
                "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
                "WHERE upper(p.proname) = ANY(%s) AND n.nspname <> 'pg_catalog' "
                "AND pg_catalog.pg_function_is_visible(p.oid)",
                (sorted(names),),
            )
            overridden = {str(r[0]) for r in rows}
        else:
            rows = _rows(conn, "PRAGMA function_list")
            builtin = {str(r[0]).upper() for r in rows if int(r[1]) == 1}
            overridden = {str(r[0]).upper() for r in rows if int(r[1]) == 0} & builtin & names
    except Exception:
        return []  # no readable catalog: unmodelled calls are unclassifiable anyway
    return [(n, "a user-defined function of that name can shadow the built-in") for n in sorted(overridden)]
