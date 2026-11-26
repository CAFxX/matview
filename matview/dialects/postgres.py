"""PostgreSQL-specific codegen helpers (CASE result type inference)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp

from .._host import MixinHost
from ..engines import (
    AggFunc,
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    UnsupportedMaterializedViewError,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
)
from ..logical_ir import ColumnDef, JoinSpec, SelectItem, TableSchema, ViewSpec
from ..sql_frontend import (
    _has_subquery_ast,
    _parse_single_agg_expr,
    _parse_single_create_view_stmt,
    _set_operation_arm_tree,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_sql_ast,
    quote_sql_identifier,
    tokenize_sql,
)
from .templates import render_procedural_sql

_PgRelCols = Optional[List[str]]
_PgScopeStack = List[Tuple[Dict[str, _PgRelCols], Dict[str, bool]]]


_PG_SESSION_TEXT_SCALARS = frozenset({"TIMESTAMPTZ", "TIMESTAMP", "DATE", "INTERVAL", "MONEY", "REG"})
_PG_SESSION_JSON_SCALARS = frozenset({"TIMESTAMPTZ", "INTERVAL", "MONEY"})
_PG_STABLE_STRING_PARSE_TARGETS = frozenset({
    "TIMESTAMPTZ", "TIMESTAMP", "DATE", "TIME", "TIMETZ", "INTERVAL", "MONEY", "XML", "REG",
})
_PG_REG_TYPES = frozenset({
    "REGCLASS", "REGTYPE", "REGPROC", "REGPROCEDURE", "REGOPER", "REGOPERATOR",
    "REGCONFIG", "REGDICTIONARY", "REGNAMESPACE", "REGROLE", "REGCOLLATION",
})
_PG_NUMERIC_TYPES = frozenset({
    "SMALLINT", "INT2", "INT", "INT4", "INTEGER", "BIGINT", "INT8",
    "NUMERIC", "DECIMAL", "REAL", "FLOAT4", "DOUBLE PRECISION", "FLOAT8",
})
_PG_STRING_TYPES = frozenset({
    "TEXT", "VARCHAR", "CHARACTER VARYING", "CHAR", "CHARACTER", "BPCHAR", "NAME", "CITEXT",
})
_PG_RANGE_ELEMS: Dict[str, str] = {
    "TSTZRANGE": "TIMESTAMPTZ",
    "TSRANGE": "TIMESTAMP",
    "DATERANGE": "DATE",
}
_PG_MULTIRANGE_ELEMS: Dict[str, str] = {
    "TSTZMULTIRANGE": "TIMESTAMPTZ",
    "TSMULTIRANGE": "TIMESTAMP",
    "DATEMULTIRANGE": "DATE",
}
_PG_JSON_SERIALIZE_FUNCS = frozenset({
    "TO_JSON", "TO_JSONB",
    "JSON_BUILD_OBJECT", "JSONB_BUILD_OBJECT",
    "JSON_BUILD_ARRAY", "JSONB_BUILD_ARRAY",
    "ROW_TO_JSON", "ARRAY_TO_JSON",
    "JSON_AGG", "JSONB_AGG",
    "JSON_OBJECT_AGG", "JSONB_OBJECT_AGG",
    "JSON_OBJECT", "JSON_ARRAY", "JSON_SCALAR",
    "JSON_OBJECTAGG", "JSON_ARRAYAGG",
})
_PG_ALWAYS_SESSION_OR_VOLATILE_FUNCS = frozenset({
    "STATEMENT_TIMESTAMP", "TRANSACTION_TIMESTAMP", "TIMEOFDAY",
    "CURRENT_SETTING", "CURRENT_DATABASE", "CURRENT_SCHEMA", "CURRENT_SCHEMAS",
    "CURRENT_USER", "SESSION_USER", "CURRENT_ROLE", "CURRENT_CATALOG",
    "LOCALTIME", "LOCALTIMESTAMP",
    "INET_CLIENT_ADDR", "INET_CLIENT_PORT", "INET_SERVER_ADDR", "INET_SERVER_PORT",
    "PG_BACKEND_PID", "TXID_CURRENT", "PG_CURRENT_XACT_ID",
    "PG_CURRENT_XACT_ID_IF_ASSIGNED", "PG_CURRENT_SNAPSHOT",
    "NEXTVAL", "CURRVAL", "LASTVAL", "SETVAL", "SETSEED",
    "UUIDV4", "UUIDV7", "PG_SLEEP", "PG_SLEEP_FOR", "PG_SLEEP_UNTIL",
})
_PG_RELATIVE_TEMPORAL_LITERALS = frozenset({"now", "today", "tomorrow", "yesterday", "allballs"})
_PG_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:$|[T\s])")
_PG_ISO_TIMESTAMPTZ_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T\s]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[+-]\d{2}(?::?\d{2}(?::?\d{2})?)?|[Zz])$"
)
_PG_TIME_PREFIX_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
_PG_ISO_TIMETZ_RE = re.compile(
    r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[+-]\d{2}(?::?\d{2}(?::?\d{2})?)?|[Zz])$"
)

_PgColTypes = Dict[str, Optional[str]]
_PgTypeScope = Dict[str, _PgColTypes]
_PgTypeScopeStack = List[_PgTypeScope]


def _pg_ident_key_from_node(node: Any) -> Optional[str]:
    """PostgreSQL identifier key: unquoted -> lowercase, double-quoted -> exact."""
    if isinstance(node, sqlglot_exp.Identifier):
        name = str(node.this or "")
        if not name:
            return None
        return name if node.quoted else name.lower()
    if isinstance(node, str) and node:
        return node.lower()
    return None


def _pg_normalize_type_str(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str:
        return None
    s = dt_str.strip()
    if not s:
        return None
    s = re.sub(r'\s+COLLATE\s+(?:"[^"]*"|\S+)\s*$', "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r'^(?:"pg_catalog"|pg_catalog)\.', "", s, flags=re.IGNORECASE).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].replace('""', '"').strip()
    arr_depth = 0
    while s.endswith("[]"):
        arr_depth += 1
        s = s[:-2].strip()
    while re.search(r"\bARRAY\s*$", s, flags=re.IGNORECASE):
        arr_depth += 1
        s = re.sub(r"\s*\bARRAY\s*$", "", s, flags=re.IGNORECASE).strip()
    up = re.sub(r"\s+", " ", s.upper()).strip()
    # Strip typmods like (3) or (10, 2) even before WITH/WITHOUT TIME ZONE.
    up_no_mod = re.sub(r"\(\s*\d+(?:\s*,\s*\d+)?\s*\)", "", up)
    up_no_mod = re.sub(r"\s+", " ", up_no_mod).strip()
    base: Optional[str] = None
    if up_no_mod in ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"):
        base = "TIMESTAMPTZ"
    elif up_no_mod in ("TIMESTAMP", "TIMESTAMP WITHOUT TIME ZONE"):
        base = "TIMESTAMP"
    elif up_no_mod == "DATE":
        base = "DATE"
    elif up_no_mod in ("TIMETZ", "TIME WITH TIME ZONE"):
        base = "TIMETZ"
    elif up_no_mod in ("TIME", "TIME WITHOUT TIME ZONE"):
        base = "TIME"
    elif up_no_mod == "INTERVAL" or up_no_mod.startswith("INTERVAL "):
        base = "INTERVAL"
    elif up_no_mod == "MONEY":
        base = "MONEY"
    elif up_no_mod in _PG_STRING_TYPES:
        base = "STRING"
    elif up_no_mod in _PG_NUMERIC_TYPES:
        base = "NUMERIC"
    elif up_no_mod in ("BOOL", "BOOLEAN"):
        base = "BOOL"
    elif up_no_mod == "BYTEA":
        base = "BYTEA"
    elif up_no_mod == "JSON":
        base = "JSON"
    elif up_no_mod == "JSONB":
        base = "JSONB"
    elif up_no_mod == "XML":
        base = "XML"
    elif up_no_mod in _PG_REG_TYPES:
        base = "REG"
    elif up_no_mod in _PG_RANGE_ELEMS:
        base = f"RANGE:{_PG_RANGE_ELEMS[up_no_mod]}"
    elif up_no_mod in _PG_MULTIRANGE_ELEMS:
        base = f"MULTIRANGE:{_PG_MULTIRANGE_ELEMS[up_no_mod]}"
    elif up_no_mod:
        base = "OTHER"
    if base is None:
        return None
    for _ in range(arr_depth):
        base = f"ARRAY:{base}"
    return base


def _pg_normalize_datatype_node(dt_node: Any) -> Optional[str]:
    if not isinstance(dt_node, sqlglot_exp.DataType):
        return None
    if dt_node.is_type(sqlglot_exp.DataType.Type.ARRAY) and dt_node.expressions:
        inner = _pg_normalize_datatype_node(dt_node.expressions[0])
        return f"ARRAY:{inner}" if inner else "ARRAY:OTHER"
    return _pg_normalize_type_str(dt_node.sql(dialect="postgres"))


def _pg_type_has_session_output(t: Optional[str], *, for_json: bool = False) -> bool:
    if not t:
        return False
    scalars = _PG_SESSION_JSON_SCALARS if for_json else _PG_SESSION_TEXT_SCALARS
    if t in scalars:
        return True
    for prefix in ("ARRAY:", "RANGE:", "MULTIRANGE:"):
        if t.startswith(prefix):
            return _pg_type_has_session_output(t[len(prefix):], for_json=for_json)
    if t.startswith("ROW:"):
        parts = t[4:].split(",") if t[4:] else []
        return any(_pg_type_has_session_output(p, for_json=for_json) for p in parts)
    return False


def _pg_is_safe_temporal_literal(lit_text: str, dst_type: str) -> bool:
    """Whether a string literal cast to a PostgreSQL temporal type is
    independent of session TimeZone / DateStyle."""
    s = lit_text.strip()
    low = s.lower()
    if low in ("infinity", "-infinity", "+infinity"):
        return True
    if low in _PG_RELATIVE_TEMPORAL_LITERALS:
        return False
    if dst_type == "DATE":
        return bool(_PG_ISO_DATE_PREFIX_RE.match(s))
    if dst_type == "TIMESTAMP":
        return bool(_PG_ISO_DATE_PREFIX_RE.match(s))
    if dst_type == "TIMESTAMPTZ":
        return bool(_PG_ISO_TIMESTAMPTZ_RE.match(s))
    if dst_type == "TIME":
        return bool(_PG_TIME_PREFIX_RE.match(s))
    if dst_type == "TIMETZ":
        return bool(_PG_ISO_TIMETZ_RE.match(s))
    if dst_type == "INTERVAL":
        return True
    return False


def _pg_common_expr_type(types: Sequence[Optional[str]]) -> Optional[str]:
    concrete = [t for t in types if t and t != "NULL"]
    if not concrete:
        return None
    normalized = ["STRING" if t == "STRING_LIT" else t for t in concrete]
    unique = set(normalized)
    if len(unique) == 1:
        return next(iter(unique))
    if "TIMESTAMPTZ" in unique:
        return "TIMESTAMPTZ"
    if "TIMESTAMP" in unique and unique <= {"TIMESTAMP", "DATE"}:
        return "TIMESTAMP"
    if "TIMETZ" in unique:
        return "TIMETZ"
    if unique <= {"NUMERIC"}:
        return "NUMERIC"
    return normalized[0]



class PostgresDialectMixin(MixinHost):
    def _postgres_case_result_storage_type(self, expr: str) -> Optional[str]:
        """Infer PostgreSQL CASE common types, including nested column branches."""
        if self.dialect != Dialect.POSTGRES:
            return None
        return self._infer_ast_expr_storage_type(expr)

    def _enforce_pg_no_pseudotype_outputs(self) -> None:
        """Refuse PostgreSQL views whose output select items evaluate to a
        pseudo-type (such as anonymous ``record`` from ``ROW(v, g)`` or
        ``(v, g)``, ``record[]`` from ``ARRAY[ROW(...)]``, or ``void``).

        PostgreSQL forbids pseudo-type columns in tables and views; silently
        falling back to a ``TEXT`` backing column would change the result type
        from ``record`` (OID 2249) to ``text`` (OID 25).
        """
        if self.dialect != Dialect.POSTGRES:
            return

        def _raise_for(alias: str, raw_expr: str, dt: str) -> None:
            raise UnsupportedMaterializedViewError(
                f"PostgreSQL materialized view column {alias!r} ({raw_expr!r}) has pseudo-type "
                f"{dt.lower()}; PostgreSQL does not allow pseudo-type columns in tables or views. "
                f"Cast the expression to a named composite type (for example CAST(... AS my_type)) "
                f"or to TEXT."
            )

        for item in self.view.select_items:
            inferred = self._infer_expr_storage_type_or_none(item)
            if self._is_pg_pseudotype(inferred):
                assert inferred is not None
                _raise_for(item.alias, item.raw_expr, inferred)

        raw_q = (self.view.raw_select_sql or "").strip()
        if not raw_q:
            return
        try:
            q_ast = parse_sql_ast(raw_q, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            q_ast = None
        if q_ast is None:
            return

        known_t = set(self._schemas_lower.keys())

        def _ctx_for_select(
            sel_node: sqlglot_exp.Select,
            with_prefix: str,
            fallback_ctx: Any,
            tag: str,
        ) -> Any:
            sel_sql = sel_node.sql(_sqlglot_dialect(self.dialect))
            if with_prefix and not re.match(r"^\s*WITH\b", sel_sql, flags=re.IGNORECASE):
                sel_sql = f"{with_prefix}{sel_sql}"
            try:
                arm_spec = _parse_single_create_view_stmt(
                    f"CREATE VIEW __pg_pt_{tag} AS {sel_sql}",
                    known_tables=known_t,
                    dialect=self.dialect,
                )
                arm_spec = replace(
                    arm_spec,
                    native_scalar_types=self.view.native_scalar_types,
                    native_result_types=self.view.native_result_types,
                    native_aggregates=self.view.native_aggregates,
                )
                return self.with_ir(
                    primary_schema=self._schemas_lower.get(arm_spec.base_table.lower()),
                    view=arm_spec,
                )
            except Exception:
                return fallback_ctx

        def _check_select_ast(
            sel_node: sqlglot_exp.Select,
            eval_ctx: Any,
            cte_map: Dict[str, sqlglot_exp.Expr],
            with_prefix: str,
            seen_ctes: Set[str],
            depth: int,
        ) -> None:
            if depth > 8:
                return
            for idx, proj in enumerate(sel_node.expressions or ()):
                if isinstance(proj, sqlglot_exp.Star):
                    from_node = sel_node.args.get("from_") or sel_node.args.get("from")
                    sources: List[Any] = []
                    if from_node is not None and from_node.this is not None:
                        sources.append(from_node.this)
                    for jn in sel_node.args.get("joins") or ():
                        if jn.this is not None:
                            sources.append(jn.this)
                    for src in sources:
                        if isinstance(src, sqlglot_exp.Subquery) and src.this is not None:
                            _check_query_ast(src.this, eval_ctx, cte_map, with_prefix, seen_ctes, depth + 1)
                        elif isinstance(src, sqlglot_exp.Table) and src.name:
                            cn = src.name.lower()
                            if cn in cte_map and cn not in seen_ctes:
                                _check_query_ast(cte_map[cn], eval_ctx, cte_map, with_prefix, seen_ctes | {cn}, depth + 1)
                    continue
                if isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Star):
                    tbl = (proj.table or "").lower()
                    if tbl and tbl in cte_map and tbl not in seen_ctes:
                        _check_query_ast(cte_map[tbl], eval_ctx, cte_map, with_prefix, seen_ctes | {tbl}, depth + 1)
                    else:
                        from_node = sel_node.args.get("from_") or sel_node.args.get("from")
                        srcs: List[Any] = []
                        if from_node is not None and from_node.this is not None:
                            srcs.append(from_node.this)
                        for jn in sel_node.args.get("joins") or ():
                            if jn.this is not None:
                                srcs.append(jn.this)
                        for src in srcs:
                            if isinstance(src, sqlglot_exp.Subquery) and src.this is not None:
                                if (src.alias or "").lower() == tbl:
                                    _check_query_ast(src.this, eval_ctx, cte_map, with_prefix, seen_ctes, depth + 1)
                            elif isinstance(src, sqlglot_exp.Table) and (src.alias_or_name or "").lower() == tbl:
                                cn = src.name.lower()
                                if cn in cte_map and cn not in seen_ctes:
                                    _check_query_ast(cte_map[cn], eval_ctx, cte_map, with_prefix, seen_ctes | {cn}, depth + 1)
                    continue
                alias = (
                    proj.alias
                    if isinstance(proj, sqlglot_exp.Alias)
                    else (
                        self.view.select_items[idx].alias
                        if idx < len(self.view.select_items)
                        else f"col_{idx}"
                    )
                )
                inner = proj.this if isinstance(proj, sqlglot_exp.Alias) else proj
                if not isinstance(inner, sqlglot_exp.Expr):
                    continue
                raw_expr = inner.sql(_sqlglot_dialect(self.dialect))
                agg_item = _parse_single_agg_expr(raw_expr, alias, dialect=self.dialect)
                item = agg_item if agg_item is not None else SelectItem(
                    raw_expr=raw_expr, alias=alias, is_aggregate=False,
                )
                dt = eval_ctx._infer_expr_storage_type_or_none(item)
                if self._is_pg_pseudotype(dt):
                    assert dt is not None
                    _raise_for(alias, raw_expr, dt)

        def _check_query_ast(
            q_node: sqlglot_exp.Expr,
            eval_ctx: Any,
            cte_map: Dict[str, sqlglot_exp.Expr],
            with_prefix: str,
            seen_ctes: Set[str],
            depth: int,
        ) -> None:
            if depth > 8:
                return
            local_ctes = dict(cte_map)
            local_prefix = with_prefix
            with_node = q_node.args.get("with_") or q_node.args.get("with")
            if isinstance(with_node, sqlglot_exp.With):
                local_prefix = with_node.sql(_sqlglot_dialect(self.dialect)) + " "
                for cte in with_node.expressions or ():
                    if isinstance(cte, sqlglot_exp.CTE) and cte.alias and cte.this is not None:
                        local_ctes[cte.alias.lower()] = cte.this
            if isinstance(q_node, sqlglot_exp.SetOperation):
                leaf_nodes, _ = _set_operation_arm_tree(q_node)
                for bi, leaf in enumerate(leaf_nodes):
                    if isinstance(leaf, sqlglot_exp.Select):
                        arm_ctx = _ctx_for_select(leaf, local_prefix, eval_ctx, f"b_{depth}_{bi}")
                        _check_select_ast(leaf, arm_ctx, local_ctes, local_prefix, seen_ctes, depth + 1)
            elif isinstance(q_node, sqlglot_exp.Select):
                sel_ctx = eval_ctx if depth == 0 else _ctx_for_select(q_node, local_prefix, eval_ctx, f"s_{depth}")
                _check_select_ast(q_node, sel_ctx, local_ctes, local_prefix, seen_ctes, depth + 1)
            elif isinstance(q_node, sqlglot_exp.Subquery) and q_node.this is not None:
                _check_query_ast(q_node.this, eval_ctx, local_ctes, local_prefix, seen_ctes, depth + 1)

        _check_query_ast(q_ast, self, {}, "", set(), 0)

    _PG_SYSTEM_COLUMNS: FrozenSet[str] = frozenset(
        {"ctid", "xmin", "xmax", "cmin", "cmax", "tableoid"}
    )

    def _enforce_pg_no_whole_row_or_system_columns(self) -> None:
        """Refuse PostgreSQL views that reference system columns (``ctid``,
        ``xmin``, ``xmax``, ``cmin``, ``cmax``, ``tableoid``) or whole-row
        table/relation variables (such as ``to_jsonb(t)``, ``to_jsonb(t.*)``,
        ``(t).col``, ``t::text``, or ``col(t)``).

        System columns have no stable value across maintenance statements or
        transition tables, and whole-row references read every column without
        being rewritten against transition tables (breaking base-table writes
        with ``UndefinedColumn``).
        """
        if self.dialect != Dialect.POSTGRES:
            return

        sys_cols = self._PG_SYSTEM_COLUMNS

        def _ident_key(ident: sqlglot_exp.Identifier) -> str:
            return ident.name if ident.quoted else ident.name.lower()

        def _is_sys_col_ident(ident: sqlglot_exp.Identifier) -> bool:
            if ident.quoted:
                return ident.name in sys_cols
            return ident.name.lower() in sys_cols

        def _has_user_col(col_list: _PgRelCols, ident: sqlglot_exp.Identifier) -> bool:
            if col_list is None:
                return False
            return _ident_key(ident) in col_list

        def _base_table_cols(table_name: str) -> Tuple[_PgRelCols, bool]:
            ts = self._schemas_lower.get(table_name.lower())
            if ts is None:
                return None, False
            # Preserve exact case for quoted mixed/uppercase column names (which
            # PostgreSQL never folds to lowercase) and exclude lowercase system
            # column names that a PostgreSQL base table cannot declare.
            cols = [
                c if c != c.lower() else c.lower()
                for c in ts.columns
                if c != c.lower() or c.lower() not in sys_cols
            ]
            return cols, True

        def _apply_alias_cols(
            base_cols: _PgRelCols,
            alias_col_nodes: Sequence[Any],
        ) -> _PgRelCols:
            if not alias_col_nodes:
                return base_cols
            renamed = [
                _ident_key(cn)
                for cn in alias_col_nodes
                if isinstance(cn, sqlglot_exp.Identifier)
            ]
            if not renamed:
                return base_cols
            if base_cols is not None and len(base_cols) > len(renamed):
                return renamed + base_cols[len(renamed):]
            return renamed

        def _extract_source_meta(
            src: sqlglot_exp.Expr,
            cte_cols: Dict[str, _PgRelCols],
            explicit_alias_node: Optional[Any] = None,
        ) -> Tuple[Optional[str], _PgRelCols, bool]:
            alias_node = explicit_alias_node if explicit_alias_node is not None else src.args.get("alias")
            rel_key: Optional[str] = None
            alias_col_nodes: Sequence[Any] = ()
            if isinstance(alias_node, sqlglot_exp.TableAlias):
                if isinstance(alias_node.this, sqlglot_exp.Identifier):
                    rel_key = _ident_key(alias_node.this)
                alias_col_nodes = tuple(alias_node.args.get("columns") or ())
            if isinstance(src, sqlglot_exp.Table):
                if isinstance(src.this, sqlglot_exp.Identifier):
                    if rel_key is None:
                        rel_key = _ident_key(src.this)
                    t_key = _ident_key(src.this)
                    if not src.db and t_key in cte_cols:
                        return rel_key, _apply_alias_cols(cte_cols[t_key], alias_col_nodes), False
                    cols, is_base = _base_table_cols(src.name or "")
                    return rel_key, _apply_alias_cols(cols, alias_col_nodes), is_base
                return rel_key, _apply_alias_cols(None, alias_col_nodes), False
            if isinstance(src, sqlglot_exp.Subquery):
                sub_cols = _projected_cols(src, cte_cols)
                return rel_key, _apply_alias_cols(sub_cols, alias_col_nodes), False
            return rel_key, _apply_alias_cols(None, alias_col_nodes), False

        def _collect_select_source_cols(
            sel_node: sqlglot_exp.Select,
            cte_cols: Dict[str, _PgRelCols],
        ) -> List[Tuple[Optional[str], _PgRelCols]]:
            out: List[Tuple[Optional[str], _PgRelCols]] = []

            def _add(src: Optional[sqlglot_exp.Expr]) -> None:
                if src is None:
                    return
                if isinstance(src, sqlglot_exp.Lateral):
                    if src.this is not None:
                        rk, rc, _ = _extract_source_meta(src.this, cte_cols, explicit_alias_node=src.args.get("alias"))
                        out.append((rk, rc))
                    return
                if isinstance(src, sqlglot_exp.Subquery) and not isinstance(
                    src.this, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Values)
                ):
                    if src.this is not None:
                        _add(src.this)
                        for jn in src.this.args.get("joins") or ():
                            if isinstance(jn, sqlglot_exp.Join):
                                _add(jn.this)
                    for jn in src.args.get("joins") or ():
                        if isinstance(jn, sqlglot_exp.Join):
                            _add(jn.this)
                    return
                rk, rc, _ = _extract_source_meta(src, cte_cols)
                out.append((rk, rc))

            from_node = sel_node.args.get("from_") or sel_node.args.get("from")
            if from_node is not None:
                _add(from_node.this)
            for jn in sel_node.args.get("joins") or ():
                if isinstance(jn, sqlglot_exp.Join):
                    _add(jn.this)
            return out

        def _projected_cols(
            query_node: Optional[sqlglot_exp.Expr],
            cte_cols: Dict[str, _PgRelCols],
        ) -> _PgRelCols:
            if query_node is None:
                return None
            if isinstance(query_node, sqlglot_exp.Subquery) and query_node.this is not None:
                return _projected_cols(query_node.this, cte_cols)
            if isinstance(query_node, sqlglot_exp.SetOperation):
                return _projected_cols(query_node.this, cte_cols)
            if not isinstance(query_node, sqlglot_exp.Select):
                return None
            src_list: Optional[List[Tuple[Optional[str], _PgRelCols]]] = None
            cols_out: List[str] = []
            for proj in query_node.expressions or ():
                if isinstance(proj, sqlglot_exp.Star):
                    if src_list is None:
                        src_list = _collect_select_source_cols(query_node, cte_cols)
                    if not src_list:
                        return None
                    for _, rc in src_list:
                        if rc is None:
                            return None
                        cols_out.extend(rc)
                    continue
                if isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Star):
                    tbl_id = proj.args.get("table")
                    if not isinstance(tbl_id, sqlglot_exp.Identifier):
                        return None
                    if src_list is None:
                        src_list = _collect_select_source_cols(query_node, cte_cols)
                    tbl_key = _ident_key(tbl_id)
                    matched = [rc for rk, rc in src_list if rk == tbl_key]
                    if not matched or matched[0] is None:
                        return None
                    cols_out.extend(matched[0])
                    continue
                if isinstance(proj, sqlglot_exp.Alias):
                    al_node = proj.args.get("alias")
                    if isinstance(al_node, sqlglot_exp.Identifier):
                        cols_out.append(_ident_key(al_node))
                    elif proj.alias:
                        cols_out.append(proj.alias.lower())
                    else:
                        cols_out.append("")
                elif isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Identifier):
                    cols_out.append(_ident_key(proj.this))
                elif isinstance(proj, sqlglot_exp.Func):
                    fn_name = (
                        proj.name.lower()
                        if isinstance(proj, sqlglot_exp.Anonymous)
                        else (proj.sql_name() or "").lower()
                    )
                    cols_out.append(fn_name)
                else:
                    cols_out.append("")
            return cols_out

        def _walk_expr(
            expr_node: Optional[sqlglot_exp.Expr],
            owner_select: Optional[sqlglot_exp.Select],
            scope_stack: _PgScopeStack,
            cte_cols: Dict[str, _PgRelCols],
            select_output_aliases: Set[str],
            *,
            allow_output_alias: bool = False,
        ) -> None:
            if expr_node is None:
                return
            if isinstance(expr_node, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Values)):
                _walk_query(expr_node, scope_stack, cte_cols)
                return
            if isinstance(expr_node, sqlglot_exp.Subquery) and isinstance(
                expr_node.this, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Subquery, sqlglot_exp.Values)
            ):
                _walk_query(expr_node, scope_stack, cte_cols)
                return

            if isinstance(expr_node, sqlglot_exp.Column):
                # Check A: qualified star (`t.*`) outside top-level SELECT projection is a whole-row reference.
                if isinstance(expr_node.this, sqlglot_exp.Star):
                    tbl_id = expr_node.args.get("table")
                    if (
                        isinstance(tbl_id, sqlglot_exp.Identifier)
                        and expr_node.parent is not owner_select
                    ):
                        tbl_key = _ident_key(tbl_id)
                        if any(tbl_key in rels for rels, _ in scope_stack):
                            raise UnsupportedMaterializedViewError(
                                f"PostgreSQL materialized view references whole-row value "
                                f"'{tbl_id.name}.*' inside an expression; whole-row references "
                                f"are not supported in incrementally maintained views. Reference "
                                f"explicit columns instead."
                            )
                    return

                col_ident = expr_node.this
                if not isinstance(col_ident, sqlglot_exp.Identifier):
                    return
                tbl_id = expr_node.args.get("table")
                if tbl_id is None:
                    col_key = _ident_key(col_ident)
                    parent_node = expr_node.parent
                    is_direct_group_or_order = allow_output_alias and (
                        isinstance(
                            parent_node,
                            (sqlglot_exp.Group, sqlglot_exp.Ordered, sqlglot_exp.Rollup, sqlglot_exp.Cube),
                        )
                        or (
                            isinstance(parent_node, sqlglot_exp.Paren)
                            and isinstance(parent_node.parent, sqlglot_exp.GroupingSets)
                        )
                    )
                    if is_direct_group_or_order and col_key in select_output_aliases:
                        return
                    any_user_col = any(
                        _has_user_col(cs, col_ident)
                        for rels, _ in scope_stack
                        for cs in rels.values()
                    )
                    # Check B1: bare whole-row reference (`to_jsonb(t)`, `(t).col`, `t::text`, `col(t)`).
                    if any(col_key in rels for rels, _ in scope_stack) and not any_user_col:
                        raise UnsupportedMaterializedViewError(
                            f"PostgreSQL materialized view references whole-row relation variable "
                            f"{col_ident.name!r}; whole-row references are not supported in "
                            f"materialized views. Reference explicit columns instead."
                        )
                    # Check B2: unqualified PostgreSQL system column (`ctid`, `xmin`, `xmax`, `cmin`, `cmax`, `tableoid`).
                    # Resolve from innermost to outermost scope so an inner base table's
                    # system column is never masked by an outer CTE/subquery column alias.
                    if _is_sys_col_ident(col_ident):
                        for rels, bases in reversed(scope_stack):
                            if any(_has_user_col(cs, col_ident) for cs in rels.values()):
                                break
                            if any(bases.values()):
                                raise UnsupportedMaterializedViewError(
                                    f"PostgreSQL materialized view references system column "
                                    f"{col_ident.name!r}; PostgreSQL system columns (ctid, xmin, xmax, "
                                    f"cmin, cmax, tableoid) have no stable value across maintenance "
                                    f"statements and are not supported."
                                )
                elif isinstance(tbl_id, sqlglot_exp.Identifier):
                    # Check C: qualified system column (`t.ctid`, `public.t.ctid`, `a.tableoid`).
                    if _is_sys_col_ident(col_ident):
                        tbl_key = _ident_key(tbl_id)
                        for rels, bases in reversed(scope_stack):
                            if tbl_key in rels:
                                if bases.get(tbl_key, False) and not _has_user_col(rels[tbl_key], col_ident):
                                    raise UnsupportedMaterializedViewError(
                                        f"PostgreSQL materialized view references system column "
                                        f"'{tbl_id.name}.{col_ident.name}'; PostgreSQL system columns "
                                        f"(ctid, xmin, xmax, cmin, cmax, tableoid) have no stable value "
                                        f"across maintenance statements and are not supported."
                                    )
                                break
                return

            # Recurse into child expressions of this node.
            for child in expr_node.iter_expressions():
                _walk_expr(
                    child,
                    owner_select,
                    scope_stack,
                    cte_cols,
                    select_output_aliases,
                    allow_output_alias=allow_output_alias,
                )

        def _walk_query_modifiers(
            query_node: sqlglot_exp.Expr,
            scope_stack: _PgScopeStack,
            cte_cols: Dict[str, _PgRelCols],
        ) -> None:
            for clause_key, allow_alias in (("order", True), ("limit", False), ("offset", False)):
                clause = query_node.args.get(clause_key)
                if isinstance(clause, sqlglot_exp.Expr):
                    _walk_expr(
                        clause,
                        None,
                        scope_stack,
                        cte_cols,
                        set(),
                        allow_output_alias=allow_alias,
                    )

        def _walk_query(
            query_node: Optional[sqlglot_exp.Expr],
            scope_stack: _PgScopeStack,
            cte_cols: Dict[str, _PgRelCols],
        ) -> None:
            if query_node is None:
                return
            local_ctes = dict(cte_cols)
            with_node = query_node.args.get("with_") or query_node.args.get("with")
            if isinstance(with_node, sqlglot_exp.With):
                is_recursive = bool(with_node.args.get("recursive"))
                for cte in with_node.expressions or ():
                    if not isinstance(cte, sqlglot_exp.CTE):
                        continue
                    cte_alias_node = cte.args.get("alias")
                    cte_key = ""
                    header_cols: _PgRelCols = None
                    if isinstance(cte_alias_node, sqlglot_exp.TableAlias):
                        if isinstance(cte_alias_node.this, sqlglot_exp.Identifier):
                            cte_key = _ident_key(cte_alias_node.this)
                        col_nodes: Sequence[Any] = tuple(cte_alias_node.args.get("columns") or ())
                        if col_nodes:
                            header_cols = [
                                _ident_key(cn)
                                for cn in col_nodes
                                if isinstance(cn, sqlglot_exp.Identifier)
                            ]
                    elif cte.alias:
                        cte_key = cte.alias.lower()
                    if is_recursive and cte_key:
                        local_ctes[cte_key] = header_cols or (
                            _projected_cols(cte.this, local_ctes) if cte.this is not None else None
                        )
                    if cte.this is not None:
                        _walk_query(cte.this, scope_stack, local_ctes)
                    if cte_key:
                        local_ctes[cte_key] = header_cols or (
                            _projected_cols(cte.this, local_ctes) if cte.this is not None else None
                        )

            if isinstance(query_node, sqlglot_exp.Values):
                for tup in query_node.expressions or ():
                    if isinstance(tup, sqlglot_exp.Expr):
                        _walk_expr(tup, None, scope_stack, local_ctes, set())
                _walk_query_modifiers(query_node, scope_stack, local_ctes)
                return

            if isinstance(query_node, sqlglot_exp.SetOperation):
                _walk_query(query_node.this, scope_stack, local_ctes)
                _walk_query(query_node.expression, scope_stack, local_ctes)
                _walk_query_modifiers(query_node, scope_stack, local_ctes)
                return

            if isinstance(query_node, sqlglot_exp.Subquery):
                if query_node.this is not None:
                    _walk_query(query_node.this, scope_stack, local_ctes)
                _walk_query_modifiers(query_node, scope_stack, local_ctes)
                return

            if not isinstance(query_node, sqlglot_exp.Select):
                return

            local_rels: Dict[str, _PgRelCols] = {}
            local_base: Dict[str, bool] = {}
            deferred_on_clauses: List[sqlglot_exp.Expr] = []

            def _register_source(
                src: Optional[sqlglot_exp.Expr],
                *,
                is_lateral: bool = False,
                explicit_alias_node: Optional[Any] = None,
            ) -> None:
                if src is None:
                    return
                if isinstance(src, sqlglot_exp.Lateral):
                    _register_source(
                        src.this,
                        is_lateral=True,
                        explicit_alias_node=src.args.get("alias"),
                    )
                    return
                cur_stack = [*scope_stack, (local_rels, local_base)]
                if isinstance(src, sqlglot_exp.Subquery):
                    if isinstance(src.this, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Values)):
                        _walk_query(src, cur_stack if is_lateral else scope_stack, local_ctes)
                        rk, rc, rb = _extract_source_meta(src, local_ctes, explicit_alias_node=explicit_alias_node)
                        if rk is not None:
                            local_rels[rk] = rc
                            local_base[rk] = rb
                        return
                    # Parenthesized table or join expression: `FROM (t JOIN t2 ON ...)` / `FROM (t AS a)`.
                    if src.this is not None:
                        _register_source(src.this, is_lateral=is_lateral, explicit_alias_node=explicit_alias_node)
                        for jn in src.this.args.get("joins") or ():
                            if isinstance(jn, sqlglot_exp.Join):
                                _register_source(jn.this, is_lateral=isinstance(jn.this, sqlglot_exp.Lateral))
                                if isinstance(jn.args.get("on"), sqlglot_exp.Expr):
                                    deferred_on_clauses.append(jn.args["on"])
                    for jn in src.args.get("joins") or ():
                        if isinstance(jn, sqlglot_exp.Join):
                            _register_source(jn.this, is_lateral=isinstance(jn.this, sqlglot_exp.Lateral))
                            if isinstance(jn.args.get("on"), sqlglot_exp.Expr):
                                deferred_on_clauses.append(jn.args["on"])
                    return
                if isinstance(src, sqlglot_exp.Values):
                    _walk_query(src, cur_stack if is_lateral else scope_stack, local_ctes)
                    rk, rc, rb = _extract_source_meta(src, local_ctes, explicit_alias_node=explicit_alias_node)
                    if rk is not None:
                        local_rels[rk] = rc
                        local_base[rk] = rb
                    return
                if isinstance(src, sqlglot_exp.Table):
                    rows_from = src.args.get("rows_from") or ()
                    for rf in rows_from:
                        if isinstance(rf, sqlglot_exp.Table) and isinstance(rf.this, sqlglot_exp.Expr):
                            _walk_expr(rf.this, query_node, cur_stack, local_ctes, set())
                        elif isinstance(rf, sqlglot_exp.Expr):
                            _walk_expr(rf, query_node, cur_stack, local_ctes, set())
                    if isinstance(src.this, sqlglot_exp.Expr) and not isinstance(src.this, sqlglot_exp.Identifier):
                        _walk_expr(src.this, query_node, cur_stack, local_ctes, set())
                    rk, rc, rb = _extract_source_meta(src, local_ctes, explicit_alias_node=explicit_alias_node)
                    if rk is not None:
                        local_rels[rk] = rc
                        local_base[rk] = rb
                    return
                # Bare table-valued expression (e.g. `UNNEST(...)`, `LATERAL jsonb_each(...)`).
                _walk_expr(src, query_node, cur_stack, local_ctes, set())
                rk, rc, rb = _extract_source_meta(src, local_ctes, explicit_alias_node=explicit_alias_node)
                if rk is not None:
                    local_rels[rk] = rc
                    local_base[rk] = rb

            from_node = query_node.args.get("from_") or query_node.args.get("from")
            if from_node is not None:
                _register_source(from_node.this)
            for jn in query_node.args.get("joins") or ():
                if isinstance(jn, sqlglot_exp.Join):
                    _register_source(jn.this, is_lateral=isinstance(jn.this, sqlglot_exp.Lateral))

            next_stack: _PgScopeStack = [*scope_stack, (local_rels, local_base)]

            select_output_aliases: Set[str] = set()
            for proj in query_node.expressions or ():
                if isinstance(proj, sqlglot_exp.Alias):
                    al_n = proj.args.get("alias")
                    if isinstance(al_n, sqlglot_exp.Identifier):
                        select_output_aliases.add(_ident_key(al_n))

            for on_clause in deferred_on_clauses:
                _walk_expr(on_clause, query_node, next_stack, local_ctes, select_output_aliases)
            for proj in query_node.expressions or ():
                _walk_expr(proj, query_node, next_stack, local_ctes, select_output_aliases)
            distinct_node = query_node.args.get("distinct")
            if isinstance(distinct_node, sqlglot_exp.Distinct):
                _walk_expr(distinct_node.args.get("on"), query_node, next_stack, local_ctes, select_output_aliases)
            for jn in query_node.args.get("joins") or ():
                if isinstance(jn, sqlglot_exp.Join):
                    _walk_expr(jn.args.get("on"), query_node, next_stack, local_ctes, select_output_aliases)
            for clause_key, allow_alias in (
                ("where", False),
                ("group", True),
                ("having", False),
                ("qualify", False),
                ("order", True),
                ("limit", False),
                ("offset", False),
            ):
                clause = query_node.args.get(clause_key)
                if isinstance(clause, sqlglot_exp.Expr):
                    _walk_expr(
                        clause,
                        query_node,
                        next_stack,
                        local_ctes,
                        select_output_aliases,
                        allow_output_alias=allow_alias,
                    )
            for win in query_node.args.get("windows") or ():
                if isinstance(win, sqlglot_exp.Expr):
                    _walk_expr(win, query_node, next_stack, local_ctes, select_output_aliases)

        sql_text = (self.raw_view.raw_select_sql or self.view.raw_select_sql or "").strip()
        if not sql_text:
            return
        try:
            ast_root = parse_sql_ast(sql_text, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return
        _walk_query(ast_root, [], {})

    def _enforce_pg_no_stable_or_session_dependent_exprs(self) -> None:
        """Refuse PostgreSQL views containing STABLE/VOLATILE or session-dependent
        expressions whose results depend on session settings (such as
        ``TimeZone``, ``DateStyle``, ``IntervalStyle``, ``lc_monetary``,
        ``xmloption``, or session identity) or execution state.

        A materialized view freezes expression values when base-table writes
        occur, whereas the live view evaluates them in each reader's session;
        freezing a session-dependent expression therefore diverges from the
        live view whenever writer and reader session settings differ.
        """
        if self.dialect != Dialect.POSTGRES:
            return
        raw_q = (self.raw_view.raw_select_sql or self.view.raw_select_sql or "").strip()
        if not raw_q:
            return
        try:
            q_ast = parse_sql_ast(raw_q, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return

        # Precompute base-table column types with PostgreSQL case-folding keys.
        base_table_cols: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        for t_low, t_schema in self._schemas_lower.items():
            cols_list: List[Tuple[str, Optional[str]]] = []
            for c_def in t_schema.columns.values():
                ckey = c_def.name if any(ch.isupper() for ch in c_def.name) else c_def.name.lower()
                cols_list.append((ckey, _pg_normalize_type_str(c_def.data_type)))
            base_table_cols[t_low] = cols_list

        def _refuse(expr_node: Any, reason: str, hint: str = "") -> None:
            sql_snippet = (
                expr_node.sql(dialect="postgres")
                if isinstance(expr_node, sqlglot_exp.Expr)
                else str(expr_node)
            )
            suffix = f" {hint}" if hint else ""
            raise UnsupportedMaterializedViewError(
                f"View {self.logical_view_name!r}: PostgreSQL expression {sql_snippet!r} is "
                f"STABLE/session-dependent ({reason}) and cannot be frozen in a materialized view.{suffix}"
            )

        def _unwrap_parens(node: Any) -> Any:
            cur = node
            while isinstance(cur, sqlglot_exp.Paren) and cur.this is not None:
                cur = cur.this
            return cur

        def _lookup_col(
            col_node: sqlglot_exp.Column,
            scope_stack: _PgTypeScopeStack,
            select_aliases: Optional[_PgColTypes] = None,
        ) -> Tuple[bool, Optional[str]]:
            col_key = _pg_ident_key_from_node(col_node.this)
            if col_key is None:
                return False, None
            tbl_part = col_node.args.get("table")
            if tbl_part is not None:
                rel_key = _pg_ident_key_from_node(tbl_part)
                if rel_key is None:
                    return False, None
                for scope in reversed(scope_stack):
                    if rel_key in scope:
                        cols_map = scope[rel_key]
                        if col_key in cols_map:
                            return True, cols_map[col_key]
                        return False, None
                return False, None
            for scope in reversed(scope_stack):
                matches = [cols_map[col_key] for cols_map in scope.values() if col_key in cols_map]
                if matches:
                    return True, matches[0]
            if select_aliases is not None and col_key in select_aliases:
                return True, select_aliases[col_key]
            return False, None

        def _check_cast(
            src_type: Optional[str],
            dst_type: Optional[str],
            inner_node: Any,
            cast_node: Any,
        ) -> None:
            if not dst_type or not src_type or src_type == "NULL":
                return
            # Unwrap matching array dimensions so element-wise array casts
            # (e.g. ts_arr::text[], ARRAY[ts]::date[], d_arr::timestamptz[],
            # s_arr::date[], v_arr::money[]) are checked identically to scalars.
            while src_type.startswith("ARRAY:") and dst_type.startswith("ARRAY:"):
                src_type = src_type[len("ARRAY:"):]
                dst_type = dst_type[len("ARRAY:"):]
            unwrapped = _unwrap_parens(inner_node)
            # 1. Explicit temporal timezone-converting casts (pg_cast provolatile='s').
            if src_type == "TIMESTAMPTZ" and dst_type in ("DATE", "TIMESTAMP", "TIME", "TIMETZ"):
                _refuse(
                    cast_node,
                    f"casting TIMESTAMPTZ to {dst_type} depends on session TimeZone",
                    "Convert with an explicit timezone first, e.g. CAST(ts AT TIME ZONE 'UTC' AS DATE).",
                )
            if src_type in ("DATE", "TIMESTAMP") and dst_type == "TIMESTAMPTZ":
                _refuse(
                    cast_node,
                    f"casting {src_type} to TIMESTAMPTZ depends on session TimeZone",
                    "Attach an explicit timezone with AT TIME ZONE 'UTC'.",
                )
            if src_type == "TIME" and dst_type == "TIMETZ":
                _refuse(
                    cast_node,
                    "casting TIME to TIMETZ depends on session TimeZone",
                    "Keep TIME without time zone or supply an explicit offset.",
                )
            # 2. Session-dependent output formatting (-> TEXT/VARCHAR/CHAR/NAME).
            if dst_type == "STRING" and _pg_type_has_session_output(src_type, for_json=False):
                _refuse(
                    cast_node,
                    f"formatting {src_type} as text depends on session TimeZone/DateStyle/IntervalStyle/lc_monetary/search_path",
                    "Project the typed column directly or format deterministically without session-dependent casts.",
                )
            # 3. Session-dependent input parsing (STRING -> temporal/money/xml/reg*).
            if src_type == "STRING_LIT" and dst_type in _PG_STABLE_STRING_PARSE_TARGETS:
                lit_val = str(unwrapped.this) if isinstance(unwrapped, sqlglot_exp.Literal) else ""
                if dst_type in ("MONEY", "XML", "REG") or not _pg_is_safe_temporal_literal(lit_val, dst_type):
                    _refuse(
                        cast_node,
                        f"parsing literal {lit_val!r} as {dst_type} depends on session settings",
                        "Use an unambiguous ISO literal with an explicit numeric timezone offset (for example '2024-03-05 20:00:00+00').",
                    )
            elif src_type == "STRING" and dst_type in _PG_STABLE_STRING_PARSE_TARGETS:
                _refuse(
                    cast_node,
                    f"parsing text as {dst_type} depends on session TimeZone/DateStyle/IntervalStyle/lc_monetary/search_path",
                    "Store the parsed value in a typed base-table column.",
                )
            # 4. Numeric <-> MONEY casts depend on lc_monetary.
            if (src_type == "NUMERIC" and dst_type == "MONEY") or (src_type == "MONEY" and dst_type == "NUMERIC"):
                _refuse(
                    cast_node,
                    f"casting between {src_type} and {dst_type} depends on session lc_monetary",
                    "Store monetary amounts as NUMERIC instead of MONEY.",
                )

        def _unwrap_container_scalar(t: Optional[str]) -> Optional[str]:
            cur = t
            while cur:
                for pfx in ("ARRAY:", "RANGE:", "MULTIRANGE:"):
                    if cur.startswith(pfx):
                        cur = cur[len(pfx):]
                        break
                else:
                    break
            return cur

        def _check_no_mixed_tz(types: Sequence[Optional[str]], expr_node: Any) -> None:
            present = {
                s for t in types
                for s in (_unwrap_container_scalar(t),)
                if s and s not in ("NULL", "STRING_LIT")
            }
            if "TIMESTAMPTZ" in present and (present & {"TIMESTAMP", "DATE"}):
                _refuse(
                    expr_node,
                    "implicitly coercing TIMESTAMP/DATE with TIMESTAMPTZ depends on session TimeZone",
                    "Convert explicitly with AT TIME ZONE 'UTC' so both sides share the same type.",
                )
            if "TIMETZ" in present and "TIME" in present:
                _refuse(
                    expr_node,
                    "implicitly coercing TIME with TIMETZ depends on session TimeZone",
                    "Align both sides to the same time type without implicit session-timezone casts.",
                )

        def _extract_call(node: Any) -> Optional[Tuple[Optional[str], str, List[Any]]]:
            if isinstance(node, sqlglot_exp.Anonymous):
                return None, str(node.name or "").upper(), list(node.expressions or ())
            if (
                isinstance(node, sqlglot_exp.Dot)
                and isinstance(node.this, (sqlglot_exp.Identifier, sqlglot_exp.Column))
                and isinstance(node.expression, sqlglot_exp.Anonymous)
            ):
                schema_name = (
                    str(node.this.this)
                    if isinstance(node.this, sqlglot_exp.Identifier)
                    else str(node.this.name)
                )
                return schema_name, str(node.expression.name or "").upper(), list(node.expression.expressions or ())
            return None

        def _eval_expr(
            node: Any,
            scope_stack: _PgTypeScopeStack,
            cte_types: Dict[str, List[Tuple[str, Optional[str]]]],
            select_aliases: Optional[_PgColTypes] = None,
            depth: int = 0,
        ) -> Optional[str]:
            if node is None or depth > 32:
                return None
            if isinstance(node, (sqlglot_exp.Paren, sqlglot_exp.Collate, sqlglot_exp.Alias, sqlglot_exp.Ordered)):
                return _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(node, sqlglot_exp.Order):
                for e in node.expressions or ():
                    _eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1)
                return _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(node, sqlglot_exp.Distinct):
                res_types = [
                    _eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1)
                    for e in (node.expressions or ())
                ]
                return res_types[0] if len(res_types) == 1 else None
            if isinstance(node, sqlglot_exp.Filter):
                _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                return _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(node, sqlglot_exp.Window):
                for p in node.args.get("partition_by") or ():
                    _eval_expr(p, scope_stack, cte_types, select_aliases, depth + 1)
                ord_node = node.args.get("order")
                ord_types: List[Optional[str]] = []
                if isinstance(ord_node, sqlglot_exp.Order):
                    for oe in ord_node.expressions or ():
                        ord_types.append(_eval_expr(oe, scope_stack, cte_types, select_aliases, depth + 1))
                elif ord_node is not None:
                    ord_types.append(_eval_expr(ord_node, scope_stack, cte_types, select_aliases, depth + 1))
                spec_node = node.args.get("spec")
                if isinstance(spec_node, sqlglot_exp.WindowSpec):
                    _eval_expr(spec_node, scope_stack, cte_types, select_aliases, depth + 1)
                    spec_kind = str(spec_node.args.get("kind") or "").upper()
                    if spec_kind == "RANGE" and "TIMESTAMPTZ" in ord_types:
                        for b_key in ("start", "end"):
                            b_node = spec_node.args.get(b_key)
                            if isinstance(b_node, sqlglot_exp.Interval) or (
                                isinstance(b_node, sqlglot_exp.Expr) and b_node.find(sqlglot_exp.Interval) is not None
                            ):
                                _refuse(
                                    node,
                                    "RANGE window frame with INTERVAL offset over TIMESTAMPTZ is STABLE (in_range depends on session TimeZone)",
                                    "Order by (ts AT TIME ZONE 'UTC') or use a ROWS frame.",
                                )
                elif spec_node is not None:
                    _eval_expr(spec_node, scope_stack, cte_types, select_aliases, depth + 1)
                return _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(node, sqlglot_exp.Subquery):
                sub_cols = _walk_query(node.this, scope_stack, cte_types, depth + 1)
                return sub_cols[0][1] if len(sub_cols) == 1 else None
            if isinstance(node, sqlglot_exp.Exists):
                _walk_query(node.this, scope_stack, cte_types, depth + 1)
                return "BOOL"
            if isinstance(node, sqlglot_exp.Literal):
                return "STRING_LIT" if node.is_string else "NUMERIC"
            if isinstance(node, sqlglot_exp.Null):
                return "NULL"
            if isinstance(node, sqlglot_exp.Boolean):
                return "BOOL"
            if isinstance(node, sqlglot_exp.Interval):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                return "INTERVAL"
            if isinstance(
                node,
                (
                    sqlglot_exp.CurrentUser,
                    sqlglot_exp.SessionUser,
                    sqlglot_exp.CurrentCatalog,
                    sqlglot_exp.CurrentSchema,
                    sqlglot_exp.CurrentDatabase,
                    sqlglot_exp.Localtimestamp,
                    sqlglot_exp.Localtime,
                ),
            ):
                _refuse(node, "depends on current session state or clock")
            if isinstance(node, sqlglot_exp.CurrentTimestamp):
                return "TIMESTAMPTZ"
            if isinstance(node, sqlglot_exp.CurrentDate):
                return "DATE"
            if isinstance(node, sqlglot_exp.CurrentTime):
                return "TIMETZ"
            if isinstance(node, sqlglot_exp.Column):
                if (
                    node.args.get("table") is None
                    and isinstance(node.this, sqlglot_exp.Identifier)
                    and not node.this.quoted
                    and str(node.this.this or "").upper() in ("CURRENT_ROLE", "USER")
                ):
                    _refuse(node, "SQL value function depends on the current session role")
                _found, col_t = _lookup_col(node, scope_stack, select_aliases)
                return col_t
            if isinstance(node, (sqlglot_exp.Cast, sqlglot_exp.TryCast)):
                src_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                dst_t = _pg_normalize_datatype_node(node.to)
                _check_cast(src_t, dst_t, node.this, node)
                return dst_t
            if isinstance(node, sqlglot_exp.Date):
                src_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _check_cast(src_t, "DATE", node.this, node)
                return "DATE"
            if isinstance(node, sqlglot_exp.Timestamp):
                src_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _check_cast(src_t, "TIMESTAMP", node.this, node)
                return "TIMESTAMP"
            if isinstance(node, sqlglot_exp.Time):
                src_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _check_cast(src_t, "TIME", node.this, node)
                return "TIME"
            if isinstance(node, sqlglot_exp.AtTimeZone):
                x_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                z_t = _eval_expr(node.args.get("zone"), scope_stack, cte_types, select_aliases, depth + 1)
                if x_t == "DATE":
                    _refuse(
                        node,
                        "DATE AT TIME ZONE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                        "Cast DATE to TIMESTAMP first, e.g. (d::timestamp AT TIME ZONE 'UTC').",
                    )
                if x_t == "TIME":
                    _refuse(
                        node,
                        "TIME AT TIME ZONE implicitly casts TIME to TIMETZ using session TimeZone",
                    )
                if x_t == "TIMETZ" and z_t != "INTERVAL":
                    _refuse(
                        node,
                        "TIME WITH TIME ZONE AT TIME ZONE <text> is STABLE (DST depends on the current date)",
                        "Use a fixed INTERVAL offset or TIMESTAMP WITH TIME ZONE AT TIME ZONE '...'.",
                    )
                if x_t == "TIMESTAMPTZ":
                    return "TIMESTAMP"
                if x_t == "TIMESTAMP":
                    return "TIMESTAMPTZ"
                if x_t == "TIMETZ":
                    return "TIMETZ"
                return None
            if isinstance(node, sqlglot_exp.TimestampTrunc):
                x_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                zone_arg = node.args.get("zone")
                if zone_arg is not None:
                    _eval_expr(zone_arg, scope_stack, cte_types, select_aliases, depth + 1)
                if x_t == "DATE":
                    _refuse(
                        node,
                        "DATE_TRUNC on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                        "Cast DATE to TIMESTAMP first, e.g. DATE_TRUNC('month', d::timestamp).",
                    )
                if x_t == "TIMESTAMP" and zone_arg is not None:
                    _refuse(
                        node,
                        "3-argument DATE_TRUNC on TIMESTAMP implicitly casts TIMESTAMP to TIMESTAMPTZ using session TimeZone",
                        "Use 2-argument DATE_TRUNC on TIMESTAMP or convert explicitly via AT TIME ZONE.",
                    )
                if x_t == "TIMESTAMPTZ" and zone_arg is None:
                    _refuse(
                        node,
                        "2-argument DATE_TRUNC on TIMESTAMPTZ depends on session TimeZone",
                        "Pass an explicit timezone as the 3rd argument, e.g. DATE_TRUNC('day', ts, 'UTC').",
                    )
                return x_t
            if isinstance(node, sqlglot_exp.Extract):
                x_t = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                if x_t == "TIMESTAMPTZ":
                    _refuse(
                        node,
                        "EXTRACT/DATE_PART on TIMESTAMPTZ is STABLE and depends on session TimeZone",
                        "Convert with an explicit timezone first, e.g. EXTRACT(... FROM ts AT TIME ZONE 'UTC').",
                    )
                return "NUMERIC"
            if isinstance(node, sqlglot_exp.TimeToStr):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _eval_expr(node.args.get("format"), scope_stack, cte_types, select_aliases, depth + 1)
                _refuse(
                    node,
                    "TO_CHAR is STABLE and depends on session TimeZone/lc_time/lc_numeric",
                    "Project the typed column directly instead of formatting it with TO_CHAR.",
                )
            if isinstance(node, sqlglot_exp.StrToDate):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _eval_expr(node.args.get("format"), scope_stack, cte_types, select_aliases, depth + 1)
                _refuse(node, "TO_DATE is STABLE and depends on session lc_time")
            if isinstance(node, sqlglot_exp.StrToTime):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _eval_expr(node.args.get("format"), scope_stack, cte_types, select_aliases, depth + 1)
                _refuse(node, "TO_TIMESTAMP(text, format) is STABLE and depends on session TimeZone/lc_time")
            if isinstance(node, sqlglot_exp.UnixToTime):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                return "TIMESTAMPTZ"
            if isinstance(node, sqlglot_exp.DateAdd):
                x_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                tz_arg = node.args.get("unit")
                if tz_arg is not None:
                    _eval_expr(tz_arg, scope_stack, cte_types, select_aliases, depth + 1)
                if x_t in ("DATE", "TIMESTAMP"):
                    _refuse(
                        node,
                        f"DATE_ADD on {x_t} implicitly casts {x_t} to TIMESTAMPTZ using session TimeZone",
                        "Use TIMESTAMP + INTERVAL or convert explicitly with AT TIME ZONE 'UTC'.",
                    )
                if x_t == "TIMESTAMPTZ" and tz_arg is None:
                    _refuse(
                        node,
                        "2-argument DATE_ADD on TIMESTAMPTZ is STABLE and depends on session TimeZone",
                        "Pass an explicit timezone as the 3rd argument, e.g. DATE_ADD(ts, INTERVAL '1 day', 'UTC').",
                    )
                return x_t
            if isinstance(node, (sqlglot_exp.Add, sqlglot_exp.Sub)):
                lt = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                rt = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                if (lt == "TIMESTAMPTZ" and rt == "INTERVAL") or (
                    isinstance(node, sqlglot_exp.Add) and lt == "INTERVAL" and rt == "TIMESTAMPTZ"
                ):
                    _refuse(
                        node,
                        "adding/subtracting INTERVAL on TIMESTAMPTZ is STABLE (DST depends on session TimeZone)",
                        "Use DATE_ADD(ts, iv, 'UTC') / DATE_SUBTRACT(ts, iv, 'UTC') or convert via AT TIME ZONE 'UTC'.",
                    )
                if isinstance(node, sqlglot_exp.Sub):
                    _check_no_mixed_tz([lt, rt], node)
                    if lt in ("TIMESTAMPTZ", "TIMESTAMP", "TIME") and rt == lt:
                        return "INTERVAL"
                    if lt == "DATE" and rt == "DATE":
                        return "NUMERIC"
                    if lt == "DATE" and rt == "NUMERIC":
                        return "DATE"
                    if lt in ("DATE", "TIMESTAMP") and rt == "INTERVAL":
                        return "TIMESTAMP"
                    if lt in ("TIME", "TIMETZ") and rt == "INTERVAL":
                        return lt
                    if lt == "INTERVAL" and rt == "INTERVAL":
                        return "INTERVAL"
                    return lt if lt == rt else None
                if (lt == "DATE" and rt == "NUMERIC") or (lt == "NUMERIC" and rt == "DATE"):
                    return "DATE"
                if (lt == "DATE" and rt == "TIME") or (lt == "TIME" and rt == "DATE"):
                    return "TIMESTAMP"
                if (lt == "DATE" and rt == "TIMETZ") or (lt == "TIMETZ" and rt == "DATE"):
                    return "TIMESTAMPTZ"
                if (lt in ("DATE", "TIMESTAMP") and rt == "INTERVAL") or (
                    lt == "INTERVAL" and rt in ("DATE", "TIMESTAMP")
                ):
                    return "TIMESTAMP"
                if (lt in ("TIME", "TIMETZ") and rt == "INTERVAL"):
                    return lt
                if (rt in ("TIME", "TIMETZ") and lt == "INTERVAL"):
                    return rt
                if lt == "INTERVAL" and rt == "INTERVAL":
                    return "INTERVAL"
                return lt if lt == rt else None
            if isinstance(node, sqlglot_exp.DPipe):
                lt = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                rt = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                is_array_op = (
                    (lt is not None and lt.startswith("ARRAY:") and rt not in ("STRING", "STRING_LIT"))
                    or (rt is not None and rt.startswith("ARRAY:") and lt not in ("STRING", "STRING_LIT"))
                )
                if is_array_op:
                    return lt if (lt and lt.startswith("ARRAY:")) else rt
                if _pg_type_has_session_output(lt, for_json=False) or _pg_type_has_session_output(rt, for_json=False):
                    _refuse(
                        node,
                        "concatenating temporal/money values with || implicitly formats them using session settings",
                    )
                return "STRING"
            if isinstance(node, (sqlglot_exp.Concat, sqlglot_exp.ConcatWs, sqlglot_exp.Format)):
                all_args: List[Any] = []
                if isinstance(node, sqlglot_exp.Format) and node.this is not None:
                    _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                all_args.extend(node.expressions or ())
                arg_types = [
                    _eval_expr(a, scope_stack, cte_types, select_aliases, depth + 1)
                    for a in all_args
                ]
                check_types = arg_types[1:] if isinstance(node, sqlglot_exp.ConcatWs) else arg_types
                if any(_pg_type_has_session_output(t, for_json=False) for t in check_types):
                    _refuse(
                        node,
                        "string formatting function formats temporal/money values using session settings",
                    )
                return "STRING"
            if isinstance(node, sqlglot_exp.Chr):
                chr_args = list(node.expressions or ())
                if node.this is not None:
                    chr_args.insert(0, node.this)
                chr_types = [
                    _eval_expr(a, scope_stack, cte_types, select_aliases, depth + 1)
                    for a in chr_args
                ]
                if len(chr_args) == 1 and _pg_type_has_session_output(chr_types[0], for_json=False):
                    _check_cast(chr_types[0], "STRING", chr_args[0], node)
                return "STRING"
            if isinstance(node, sqlglot_exp.ArrayToString):
                arr_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                if node.args.get("null") is not None:
                    _eval_expr(node.args.get("null"), scope_stack, cte_types, select_aliases, depth + 1)
                if _pg_type_has_session_output(arr_t, for_json=False):
                    _refuse(
                        node,
                        "ARRAY_TO_STRING formats temporal/money/reg elements using session settings",
                    )
                return "STRING"
            if isinstance(
                node,
                (
                    sqlglot_exp.Trim,
                    sqlglot_exp.Substring,
                    sqlglot_exp.Replace,
                    sqlglot_exp.Left,
                    sqlglot_exp.Right,
                    sqlglot_exp.SplitPart,
                    sqlglot_exp.Initcap,
                    sqlglot_exp.Pad,
                    sqlglot_exp.Repeat,
                    sqlglot_exp.Reverse,
                    sqlglot_exp.Translate,
                    sqlglot_exp.RegexpReplace,
                    sqlglot_exp.MD5,
                ),
            ):
                for child_val in node.args.values():
                    if isinstance(child_val, sqlglot_exp.Expr):
                        _eval_expr(child_val, scope_stack, cte_types, select_aliases, depth + 1)
                    elif isinstance(child_val, list):
                        for ch_item in child_val:
                            if isinstance(ch_item, sqlglot_exp.Expr):
                                _eval_expr(ch_item, scope_stack, cte_types, select_aliases, depth + 1)
                return "STRING"
            if isinstance(node, (sqlglot_exp.ArrayAppend, sqlglot_exp.ArrayPrepend)):
                arr_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                elem_t = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                _check_no_mixed_tz([arr_t, elem_t], node)
                if arr_t and arr_t.startswith("ARRAY:"):
                    return arr_t
                return f"ARRAY:{elem_t}" if elem_t else "ARRAY:OTHER"
            if isinstance(node, sqlglot_exp.ArrayConcat):
                parts_t: List[Optional[str]] = [
                    _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                ]
                for e in node.expressions or ():
                    parts_t.append(_eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1))
                _check_no_mixed_tz(parts_t, node)
                for pt in parts_t:
                    if pt and pt.startswith("ARRAY:"):
                        return pt
                return "ARRAY:OTHER"
            if isinstance(node, sqlglot_exp.JSONKeyValue):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                return _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(
                node,
                (
                    sqlglot_exp.JSONArrayAgg,
                    sqlglot_exp.JSONObjectAgg,
                    sqlglot_exp.JSONBObjectAgg,
                    sqlglot_exp.JSONObject,
                    sqlglot_exp.JSONArray,
                ),
            ):
                j_args: List[Any] = []
                if node.this is not None:
                    j_args.append(node.this)
                if node.expression is not None:
                    j_args.append(node.expression)
                j_args.extend(node.expressions or ())
                j_types = [
                    _eval_expr(a, scope_stack, cte_types, select_aliases, depth + 1)
                    for a in j_args
                ]
                if any(_pg_type_has_session_output(t, for_json=True) for t in j_types):
                    _refuse(
                        node,
                        "JSON serialization of TIMESTAMPTZ/INTERVAL/MONEY depends on session TimeZone/IntervalStyle/lc_monetary",
                    )
                return "JSONB" if isinstance(node, sqlglot_exp.JSONBObjectAgg) else "JSON"
            if isinstance(node, (sqlglot_exp.Any, sqlglot_exp.All)):
                if isinstance(node.this, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Values)):
                    sub_cols = _walk_query(node.this, scope_stack, cte_types, depth + 1)
                    return sub_cols[0][1] if len(sub_cols) == 1 else None
                inner_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                if inner_t and inner_t.startswith("ARRAY:"):
                    return inner_t[len("ARRAY:"):]
                return inner_t
            if isinstance(node, sqlglot_exp.DateBin):
                _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                src_t = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                orig_node = node.args.get("unit")
                if isinstance(orig_node, sqlglot_exp.Var):
                    _, orig_t = _lookup_col(
                        sqlglot_exp.Column(this=sqlglot_exp.Identifier(this=str(orig_node.this or ""), quoted=False)),
                        scope_stack,
                        select_aliases,
                    )
                else:
                    orig_t = _eval_expr(orig_node, scope_stack, cte_types, select_aliases, depth + 1)
                if src_t == "DATE" or orig_t == "DATE":
                    _refuse(
                        node,
                        "DATE_BIN on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                        "Cast DATE arguments to TIMESTAMP first.",
                    )
                _check_no_mixed_tz([src_t, orig_t], node)
                return src_t or orig_t
            if isinstance(node, sqlglot_exp.Overlaps):
                lhs_nodes = list(node.this.expressions or ()) if isinstance(node.this, sqlglot_exp.Tuple) else [node.this]
                rhs_nodes = list(node.expression.expressions or ()) if isinstance(node.expression, sqlglot_exp.Tuple) else [node.expression]
                lhs_t = [_eval_expr(x, scope_stack, cte_types, select_aliases, depth + 1) for x in lhs_nodes]
                rhs_t = [_eval_expr(x, scope_stack, cte_types, select_aliases, depth + 1) for x in rhs_nodes]
                all_t = [*lhs_t, *rhs_t]
                if "TIMESTAMPTZ" in all_t and "INTERVAL" in all_t:
                    _refuse(
                        node,
                        "OVERLAPS with TIMESTAMPTZ and INTERVAL is STABLE (DST depends on session TimeZone)",
                        "Convert endpoints explicitly with AT TIME ZONE 'UTC' or avoid INTERVAL spans on TIMESTAMPTZ.",
                    )
                _check_no_mixed_tz([t for t in all_t if t != "INTERVAL"], node)
                return "BOOL"
            if isinstance(
                node,
                (
                    sqlglot_exp.Lag,
                    sqlglot_exp.Lead,
                    sqlglot_exp.FirstValue,
                    sqlglot_exp.LastValue,
                    sqlglot_exp.NthValue,
                    sqlglot_exp.AnyValue,
                ),
            ):
                base_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                if node.args.get("offset") is not None:
                    _eval_expr(node.args.get("offset"), scope_stack, cte_types, select_aliases, depth + 1)
                def_node = node.args.get("default")
                if def_node is not None:
                    def_t = _eval_expr(def_node, scope_stack, cte_types, select_aliases, depth + 1)
                    _check_no_mixed_tz([base_t, def_t], node)
                    return _pg_common_expr_type([base_t, def_t])
                return base_t
            if isinstance(node, (sqlglot_exp.Coalesce, sqlglot_exp.Greatest, sqlglot_exp.Least)):
                branch_nodes: List[Any] = []
                if node.this is not None:
                    branch_nodes.append(node.this)
                branch_nodes.extend(node.expressions or ())
                b_types = [
                    _eval_expr(b, scope_stack, cte_types, select_aliases, depth + 1)
                    for b in branch_nodes
                ]
                _check_no_mixed_tz(b_types, node)
                return _pg_common_expr_type(b_types)
            if isinstance(node, sqlglot_exp.Nullif):
                lt = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                rt = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                _check_no_mixed_tz([lt, rt], node)
                return lt
            if isinstance(node, sqlglot_exp.Array):
                e_types = [
                    _eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1)
                    for e in (node.expressions or ())
                ]
                _check_no_mixed_tz(e_types, node)
                elem = _pg_common_expr_type(e_types)
                return f"ARRAY:{elem}" if elem else "ARRAY:OTHER"
            if isinstance(node, sqlglot_exp.Bracket):
                base_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                idx_nodes = list(node.expressions or ())
                has_slice = any(isinstance(ix, sqlglot_exp.Slice) for ix in idx_nodes)
                for ix in idx_nodes:
                    _eval_expr(ix, scope_stack, cte_types, select_aliases, depth + 1)
                if base_t and base_t.startswith("ARRAY:") and not has_slice:
                    return base_t[len("ARRAY:"):]
                return base_t
            if isinstance(node, sqlglot_exp.Case):
                case_val_t = (
                    _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                    if node.this is not None
                    else None
                )
                res_types: List[Optional[str]] = []
                for if_node in node.args.get("ifs") or ():
                    if isinstance(if_node, sqlglot_exp.If):
                        w_t = _eval_expr(if_node.this, scope_stack, cte_types, select_aliases, depth + 1)
                        if case_val_t is not None:
                            _check_no_mixed_tz([case_val_t, w_t], node)
                        res_types.append(_eval_expr(if_node.args.get("true"), scope_stack, cte_types, select_aliases, depth + 1))
                if node.args.get("default") is not None:
                    res_types.append(_eval_expr(node.args.get("default"), scope_stack, cte_types, select_aliases, depth + 1))
                _check_no_mixed_tz(res_types, node)
                return _pg_common_expr_type(res_types)
            if isinstance(
                node,
                (
                    sqlglot_exp.EQ,
                    sqlglot_exp.NEQ,
                    sqlglot_exp.LT,
                    sqlglot_exp.LTE,
                    sqlglot_exp.GT,
                    sqlglot_exp.GTE,
                    sqlglot_exp.NullSafeEQ,
                    sqlglot_exp.NullSafeNEQ,
                ),
            ):
                lt = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                rt = _eval_expr(node.expression, scope_stack, cte_types, select_aliases, depth + 1)
                _check_no_mixed_tz([lt, rt], node)
                if lt == "STRING_LIT" and rt in _PG_STABLE_STRING_PARSE_TARGETS:
                    _check_cast(lt, rt, node.this, node)
                elif rt == "STRING_LIT" and lt in _PG_STABLE_STRING_PARSE_TARGETS:
                    _check_cast(rt, lt, node.expression, node)
                return "BOOL"
            if isinstance(node, sqlglot_exp.Between):
                t_this = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                t_low = _eval_expr(node.args.get("low"), scope_stack, cte_types, select_aliases, depth + 1)
                t_high = _eval_expr(node.args.get("high"), scope_stack, cte_types, select_aliases, depth + 1)
                _check_no_mixed_tz([t_this, t_low, t_high], node)
                if t_this in _PG_STABLE_STRING_PARSE_TARGETS:
                    if t_low == "STRING_LIT":
                        _check_cast(t_low, t_this, node.args.get("low"), node)
                    if t_high == "STRING_LIT":
                        _check_cast(t_high, t_this, node.args.get("high"), node)
                return "BOOL"
            if isinstance(node, sqlglot_exp.In):
                t_this = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                in_exprs = list(node.expressions or ())
                rhs_types: List[Optional[str]] = [
                    _eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1)
                    for e in in_exprs
                ]
                q_arg = node.args.get("query")
                if q_arg is not None:
                    q_inner = q_arg.this if isinstance(q_arg, sqlglot_exp.Subquery) else q_arg
                    sub_cols = _walk_query(q_inner, scope_stack, cte_types, depth + 1)
                    if len(sub_cols) == 1:
                        rhs_types.append(sub_cols[0][1])
                _check_no_mixed_tz([t_this, *rhs_types], node)
                if t_this in _PG_STABLE_STRING_PARSE_TARGETS:
                    for e_node, e_t in zip(in_exprs, rhs_types):
                        if e_t == "STRING_LIT":
                            _check_cast(e_t, t_this, e_node, node)
                return "BOOL"
            if isinstance(node, (sqlglot_exp.Min, sqlglot_exp.Max)):
                return _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
            if isinstance(node, (sqlglot_exp.Lower, sqlglot_exp.Upper)):
                arg_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                if arg_t and arg_t.startswith("RANGE:"):
                    return arg_t[len("RANGE:"):]
                if arg_t and arg_t.startswith("MULTIRANGE:"):
                    return arg_t[len("MULTIRANGE:"):]
                return "STRING"
            if isinstance(node, sqlglot_exp.ArrayAgg):
                elem_t = _eval_expr(node.this, scope_stack, cte_types, select_aliases, depth + 1)
                return f"ARRAY:{elem_t}" if elem_t else "ARRAY:OTHER"
            if isinstance(node, (sqlglot_exp.ExplodingGenerateSeries, sqlglot_exp.GenerateSeries)):
                start_t = _eval_expr(node.args.get("start"), scope_stack, cte_types, select_aliases, depth + 1)
                end_t = _eval_expr(node.args.get("end"), scope_stack, cte_types, select_aliases, depth + 1)
                step_node = node.args.get("step")
                if step_node is not None:
                    _eval_expr(step_node, scope_stack, cte_types, select_aliases, depth + 1)
                tz_node = node.args.get("is_end_exclusive")
                if tz_node is not None:
                    _eval_expr(tz_node, scope_stack, cte_types, select_aliases, depth + 1)
                _check_no_mixed_tz([start_t, end_t], node)
                if tz_node is not None and (start_t in ("DATE", "TIMESTAMP") or end_t in ("DATE", "TIMESTAMP")):
                    _refuse(
                        node,
                        "4-argument GENERATE_SERIES on DATE/TIMESTAMP implicitly casts to TIMESTAMPTZ using session TimeZone",
                    )
                if tz_node is None and step_node is not None and (
                    start_t in ("TIMESTAMPTZ", "DATE") or end_t in ("TIMESTAMPTZ", "DATE")
                ):
                    _refuse(
                        node,
                        "3-argument GENERATE_SERIES on TIMESTAMPTZ/DATE is STABLE (depends on session TimeZone)",
                        "Cast DATE to TIMESTAMP or pass an explicit 4th timezone argument for TIMESTAMPTZ.",
                    )
                return start_t or end_t
            if isinstance(node, sqlglot_exp.Tuple):
                elem_types = [
                    _eval_expr(e, scope_stack, cte_types, select_aliases, depth + 1)
                    for e in (node.expressions or ())
                ]
                return "ROW:" + ",".join(t or "" for t in elem_types)

            call_info = _extract_call(node)
            if call_info is not None:
                schema_name, fn_up, arg_nodes = call_info
                arg_types = [
                    _eval_expr(a, scope_stack, cte_types, select_aliases, depth + 1)
                    for a in arg_nodes
                ]
                if fn_up == "ROW" and schema_name is None:
                    return "ROW:" + ",".join(t or "" for t in arg_types)
                is_pg_builtin = schema_name is None or schema_name.lower() == "pg_catalog"
                if is_pg_builtin:
                    if fn_up == "PG_TYPEOF":
                        return "PG_TYPEOF_REG"
                    if fn_up in _PG_ALWAYS_SESSION_OR_VOLATILE_FUNCS:
                        _refuse(node, f"{fn_up}() is STABLE/VOLATILE and depends on session or transaction state")
                    if fn_up == "TIMESTAMPTZ" and len(arg_nodes) == 2:
                        if arg_types[0] == "DATE" and arg_types[1] == "TIME":
                            _refuse(
                                node,
                                "timestamptz(date, time without time zone) is STABLE and depends on session TimeZone",
                            )
                        return "TIMESTAMPTZ"
                    if fn_up in ("DATE", "TIMESTAMP", "TIMESTAMPTZ", "TIME", "TIMETZ", "MONEY", "XML") and len(arg_nodes) == 1:
                        _check_cast(arg_types[0], fn_up, arg_nodes[0], node)
                        return fn_up
                    if fn_up in _PG_STRING_TYPES and len(arg_nodes) == 1:
                        _check_cast(arg_types[0], "STRING", arg_nodes[0], node)
                        return "STRING"
                    if fn_up in _PG_REG_TYPES and len(arg_nodes) == 1:
                        _check_cast(arg_types[0], "REG", arg_nodes[0], node)
                        return "REG"
                    if fn_up == "DATE_TRUNC":
                        if len(arg_nodes) >= 2 and arg_types[1] == "DATE":
                            _refuse(
                                node,
                                "DATE_TRUNC on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                                "Cast DATE to TIMESTAMP first, e.g. DATE_TRUNC('month', d::timestamp).",
                            )
                        if len(arg_nodes) == 3 and arg_types[1] == "TIMESTAMP":
                            _refuse(
                                node,
                                "3-argument DATE_TRUNC on TIMESTAMP implicitly casts TIMESTAMP to TIMESTAMPTZ using session TimeZone",
                            )
                        if len(arg_nodes) == 2 and arg_types[1] == "TIMESTAMPTZ":
                            _refuse(
                                node,
                                "2-argument DATE_TRUNC on TIMESTAMPTZ depends on session TimeZone",
                                "Pass an explicit timezone as the 3rd argument, e.g. DATE_TRUNC('day', ts, 'UTC').",
                            )
                        return arg_types[1] if len(arg_types) >= 2 else None
                    if fn_up in ("DATE_PART", "EXTRACT") and len(arg_nodes) == 2:
                        if arg_types[1] == "TIMESTAMPTZ":
                            _refuse(
                                node,
                                f"{fn_up} on TIMESTAMPTZ is STABLE and depends on session TimeZone",
                                "Convert with an explicit timezone first, e.g. ts AT TIME ZONE 'UTC'.",
                            )
                        return "NUMERIC"
                    if fn_up in ("TO_CHAR", "TO_DATE"):
                        _refuse(node, f"{fn_up} is STABLE and depends on session TimeZone/lc_time/lc_numeric")
                    if fn_up == "TO_TIMESTAMP":
                        if len(arg_nodes) == 2:
                            _refuse(node, "TO_TIMESTAMP(text, format) is STABLE and depends on session TimeZone/lc_time")
                        return "TIMESTAMPTZ"
                    if fn_up in ("DATE_ADD", "DATE_SUBTRACT"):
                        if arg_nodes and arg_types[0] in ("DATE", "TIMESTAMP"):
                            _refuse(
                                node,
                                f"{fn_up} on {arg_types[0]} implicitly casts {arg_types[0]} to TIMESTAMPTZ using session TimeZone",
                            )
                        if len(arg_nodes) == 2 and arg_types[0] == "TIMESTAMPTZ":
                            _refuse(
                                node,
                                f"2-argument {fn_up} on TIMESTAMPTZ is STABLE and depends on session TimeZone",
                                f"Pass an explicit timezone as the 3rd argument, e.g. {fn_up}(ts, iv, 'UTC').",
                            )
                        return arg_types[0] if arg_types else None
                    if fn_up == "DATE_BIN" and len(arg_types) >= 3:
                        if arg_types[1] == "DATE" or arg_types[2] == "DATE":
                            _refuse(
                                node,
                                "DATE_BIN on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                                "Cast DATE arguments to TIMESTAMP first.",
                            )
                        _check_no_mixed_tz([arg_types[1], arg_types[2]], node)
                        return arg_types[1]
                    if fn_up == "AGE":
                        if len(arg_nodes) == 1:
                            _refuse(node, "1-argument AGE() is STABLE and depends on current_date")
                        if len(arg_nodes) == 2:
                            _check_no_mixed_tz([arg_types[0], arg_types[1]], node)
                        return "INTERVAL"
                    if fn_up == "TIMEZONE":
                        if len(arg_nodes) == 1:
                            _refuse(node, "1-argument TIMEZONE() is STABLE and depends on session TimeZone")
                        if len(arg_nodes) == 2:
                            if arg_types[1] == "DATE":
                                _refuse(
                                    node,
                                    "timezone(tz, date) implicitly casts DATE to TIMESTAMPTZ using session TimeZone",
                                )
                            if arg_types[1] == "TIME":
                                _refuse(
                                    node,
                                    "timezone(tz, time) implicitly casts TIME to TIMETZ using session TimeZone",
                                )
                            if arg_types[1] == "TIMETZ" and arg_types[0] != "INTERVAL":
                                _refuse(
                                    node,
                                    "timezone(text, timetz) is STABLE (DST depends on current date)",
                                )
                            if arg_types[1] == "TIMESTAMPTZ":
                                return "TIMESTAMP"
                            if arg_types[1] == "TIMESTAMP":
                                return "TIMESTAMPTZ"
                            if arg_types[1] == "TIMETZ":
                                return "TIMETZ"
                        return None
                    if fn_up == "MAKE_TIMESTAMPTZ":
                        _refuse(node, "MAKE_TIMESTAMPTZ is STABLE in PostgreSQL")
                    if fn_up == "GENERATE_SERIES":
                        if len(arg_nodes) >= 2:
                            _check_no_mixed_tz([arg_types[0], arg_types[1]], node)
                        if len(arg_nodes) == 3 and (arg_types[0] in ("TIMESTAMPTZ", "DATE") or arg_types[1] in ("TIMESTAMPTZ", "DATE")):
                            _refuse(
                                node,
                                "3-argument GENERATE_SERIES on TIMESTAMPTZ/DATE is STABLE (depends on session TimeZone)",
                                "Cast DATE to TIMESTAMP or pass an explicit 4th timezone argument for TIMESTAMPTZ.",
                            )
                        if len(arg_nodes) == 4 and (arg_types[0] in ("DATE", "TIMESTAMP") or arg_types[1] in ("DATE", "TIMESTAMP")):
                            _refuse(
                                node,
                                "4-argument GENERATE_SERIES on DATE/TIMESTAMP implicitly casts to TIMESTAMPTZ using session TimeZone",
                            )
                        return arg_types[0] if arg_types else None
                    if fn_up in ("QUOTE_LITERAL", "QUOTE_NULLABLE"):
                        if any(_pg_type_has_session_output(t, for_json=False) for t in arg_types):
                            _refuse(
                                node,
                                f"{fn_up} formats temporal/money/reg values using session settings",
                            )
                        return "STRING"
                    if fn_up == "ARRAY_TO_STRING":
                        if arg_types and _pg_type_has_session_output(arg_types[0], for_json=False):
                            _refuse(
                                node,
                                "ARRAY_TO_STRING formats temporal/money/reg elements using session settings",
                            )
                        return "STRING"
                    if fn_up in ("BTRIM", "LTRIM", "RTRIM", "SUBSTR", "LPAD", "RPAD", "ENCODE", "CONVERT_FROM"):
                        return "STRING"
                    if fn_up in ("ARRAY_APPEND", "ARRAY_PREPEND", "ARRAY_CAT", "ARRAY_REMOVE", "ARRAY_REPLACE"):
                        _check_no_mixed_tz(arg_types, node)
                        for at in arg_types:
                            if at and at.startswith("ARRAY:"):
                                return at
                        return "ARRAY:OTHER"
                    if fn_up in ("CONCAT", "CONCAT_WS", "FORMAT"):
                        check_types = arg_types[1:] if fn_up in ("CONCAT_WS", "FORMAT") else arg_types
                        if any(_pg_type_has_session_output(t, for_json=False) for t in check_types):
                            _refuse(
                                node,
                                f"{fn_up} formats temporal/money values using session settings",
                            )
                        return "STRING"
                    if fn_up in _PG_JSON_SERIALIZE_FUNCS:
                        if any(_pg_type_has_session_output(t, for_json=True) for t in arg_types):
                            _refuse(
                                node,
                                f"{fn_up} serializes TIMESTAMPTZ/INTERVAL/MONEY using session settings",
                            )
                        return "JSONB" if "JSONB" in fn_up else "JSON"
                    if fn_up in _PG_RANGE_ELEMS:
                        if fn_up == "TSTZRANGE" and any(t in ("DATE", "TIMESTAMP") for t in arg_types[:2]):
                            _refuse(
                                node,
                                "tstzrange on DATE/TIMESTAMP implicitly casts bounds to TIMESTAMPTZ using session TimeZone",
                            )
                        return f"RANGE:{_PG_RANGE_ELEMS[fn_up]}"
                    if fn_up in _PG_MULTIRANGE_ELEMS:
                        return f"MULTIRANGE:{_PG_MULTIRANGE_ELEMS[fn_up]}"
                    if fn_up in ("LOWER", "UPPER") and len(arg_types) == 1 and arg_types[0]:
                        if arg_types[0].startswith("RANGE:"):
                            return arg_types[0][len("RANGE:"):]
                        if arg_types[0].startswith("MULTIRANGE:"):
                            return arg_types[0][len("MULTIRANGE:"):]
                return None

            # Generic fallback: walk all child expressions so nested violations
            # inside any unmodelled or composite AST node are always caught.
            if isinstance(node, sqlglot_exp.Expr):
                for child_val in node.args.values():
                    if isinstance(child_val, sqlglot_exp.Expr):
                        _eval_expr(child_val, scope_stack, cte_types, select_aliases, depth + 1)
                    elif isinstance(child_val, list):
                        for item in child_val:
                            if isinstance(item, sqlglot_exp.Expr):
                                _eval_expr(item, scope_stack, cte_types, select_aliases, depth + 1)
            return None

        def _apply_alias_cols(
            cols_list: List[Tuple[str, Optional[str]]],
            alias_node: Any,
        ) -> List[Tuple[str, Optional[str]]]:
            if not isinstance(alias_node, sqlglot_exp.TableAlias):
                return cols_list
            col_nodes: Sequence[Any] = alias_node.args.get("columns") or ()
            if not col_nodes:
                return cols_list
            renamed: List[Tuple[str, Optional[str]]] = []
            for idx, cn in enumerate(col_nodes):
                ckey = _pg_ident_key_from_node(cn) or f"col_{idx}"
                ctype = cols_list[idx][1] if idx < len(cols_list) else None
                renamed.append((ckey, ctype))
            if len(cols_list) > len(col_nodes):
                renamed.extend(cols_list[len(col_nodes):])
            return renamed

        def _register_from_source(
            src: Any,
            local_scope: _PgTypeScope,
            local_order: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
            outer_stack: _PgTypeScopeStack,
            cte_types: Dict[str, List[Tuple[str, Optional[str]]]],
            depth: int,
            *,
            explicit_alias_node: Optional[Any] = None,
        ) -> None:
            if src is None or depth > 32:
                return
            left_stack = [*outer_stack, local_scope] if local_scope else list(outer_stack)
            if isinstance(src, sqlglot_exp.Lateral):
                lat_alias = src.args.get("alias") or explicit_alias_node
                if isinstance(src.this, (sqlglot_exp.Subquery, sqlglot_exp.Values)):
                    _register_from_source(
                        src.this,
                        local_scope,
                        local_order,
                        left_stack,
                        cte_types,
                        depth + 1,
                        explicit_alias_node=lat_alias,
                    )
                else:
                    ret_t = _eval_expr(src.this, left_stack, cte_types, None, depth + 1)
                    alias_key = (
                        _pg_ident_key_from_node(lat_alias.this)
                        if isinstance(lat_alias, sqlglot_exp.TableAlias) and lat_alias.this is not None
                        else None
                    )
                    if alias_key:
                        cols = _apply_alias_cols([(alias_key, ret_t)], lat_alias)
                        local_scope[alias_key] = dict(cols)
                        local_order.append((alias_key, cols))
                return
            if isinstance(src, sqlglot_exp.Unnest):
                elem_types: List[Optional[str]] = []
                for e in src.expressions or ():
                    et = _eval_expr(e, left_stack, cte_types, None, depth + 1)
                    elem_types.append(et[len("ARRAY:"):] if (et and et.startswith("ARRAY:")) else None)
                alias_node = src.args.get("alias") or explicit_alias_node
                alias_key = (
                    _pg_ident_key_from_node(alias_node.this)
                    if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                    else "unnest"
                )
                if alias_key:
                    cols = _apply_alias_cols([(f"col_{i}", t) for i, t in enumerate(elem_types)], alias_node)
                    if len(cols) == 1 and not (
                        isinstance(alias_node, sqlglot_exp.TableAlias) and (alias_node.args.get("columns") or ())
                    ):
                        cols = [(alias_key, cols[0][1])]
                    local_scope[alias_key] = dict(cols)
                    local_order.append((alias_key, cols))
                return
            if isinstance(src, sqlglot_exp.Subquery):
                sub_inner = src.this
                # Parenthesized join tree: FROM (t JOIN u ON ...)
                if isinstance(sub_inner, (sqlglot_exp.Table, sqlglot_exp.Subquery)) or (
                    src.args.get("joins") and not isinstance(sub_inner, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Values))
                ):
                    _register_from_source(sub_inner, local_scope, local_order, outer_stack, cte_types, depth + 1)
                    for jn in src.args.get("joins") or ():
                        _register_from_source(jn.this, local_scope, local_order, outer_stack, cte_types, depth + 1)
                        if jn.args.get("on") is not None:
                            _eval_expr(jn.args.get("on"), [*outer_stack, local_scope], cte_types, None, depth + 1)
                    return
                sub_cols = _walk_query(sub_inner, left_stack, cte_types, depth + 1)
                for clause_key in ("order", "limit", "offset"):
                    c_node = src.args.get(clause_key)
                    if c_node is not None:
                        _eval_expr(c_node, [*outer_stack, dict(local_scope)], cte_types, dict(sub_cols), depth + 1)
                alias_node = src.args.get("alias") or explicit_alias_node
                alias_key = (
                    _pg_ident_key_from_node(alias_node.this)
                    if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                    else None
                )
                if alias_key:
                    cols = _apply_alias_cols(sub_cols, alias_node)
                    local_scope[alias_key] = dict(cols)
                    local_order.append((alias_key, cols))
                return
            if isinstance(src, sqlglot_exp.Table):
                # Table-valued function in FROM (e.g. generate_series(...), jsonb_each(...))
                if isinstance(src.this, (sqlglot_exp.Func, sqlglot_exp.Anonymous, sqlglot_exp.Dot)):
                    ret_t = _eval_expr(src.this, left_stack, cte_types, None, depth + 1)
                    alias_node = src.args.get("alias") or explicit_alias_node
                    alias_key = (
                        _pg_ident_key_from_node(alias_node.this)
                        if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                        else (
                            str(src.this.name).lower()
                            if isinstance(src.this, sqlglot_exp.Anonymous) and src.this.name
                            else None
                        )
                    )
                    if alias_key:
                        cols = _apply_alias_cols([(alias_key, ret_t)], alias_node)
                        local_scope[alias_key] = dict(cols)
                        local_order.append((alias_key, cols))
                    return
                tbl_key = _pg_ident_key_from_node(src.this)
                alias_node = src.args.get("alias") or explicit_alias_node
                alias_key = (
                    _pg_ident_key_from_node(alias_node.this)
                    if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                    else tbl_key
                )
                if tbl_key is not None and alias_key is not None:
                    if tbl_key in cte_types:
                        raw_cols = list(cte_types[tbl_key])
                    elif tbl_key.lower() in base_table_cols:
                        raw_cols = list(base_table_cols[tbl_key.lower()])
                    else:
                        raw_cols = []
                    cols = _apply_alias_cols(raw_cols, alias_node)
                    local_scope[alias_key] = dict(cols)
                    local_order.append((alias_key, cols))
                # Parenthesized join trees attach nested joins directly to Table.args["joins"].
                for jn in src.args.get("joins") or ():
                    _register_from_source(jn.this, local_scope, local_order, outer_stack, cte_types, depth + 1)
                    if jn.args.get("on") is not None:
                        _eval_expr(jn.args.get("on"), [*outer_stack, local_scope], cte_types, None, depth + 1)
                return
            if isinstance(src, sqlglot_exp.Values):
                val_cols = _walk_values(src, left_stack, cte_types, depth + 1)
                alias_node = src.args.get("alias") or explicit_alias_node
                alias_key = (
                    _pg_ident_key_from_node(alias_node.this)
                    if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                    else None
                )
                if alias_key:
                    cols = _apply_alias_cols(val_cols, alias_node)
                    local_scope[alias_key] = dict(cols)
                    local_order.append((alias_key, cols))
                return

        def _walk_values(
            val_node: sqlglot_exp.Values,
            scope_stack: _PgTypeScopeStack,
            cte_types: Dict[str, List[Tuple[str, Optional[str]]]],
            depth: int,
        ) -> List[Tuple[str, Optional[str]]]:
            col_buckets: List[List[Optional[str]]] = []
            for tup in val_node.expressions or ():
                if isinstance(tup, sqlglot_exp.Tuple):
                    for idx, elem in enumerate(tup.expressions or ()):
                        t = _eval_expr(elem, scope_stack, cte_types, None, depth + 1)
                        while len(col_buckets) <= idx:
                            col_buckets.append([])
                        col_buckets[idx].append(t)
                else:
                    _eval_expr(tup, scope_stack, cte_types, None, depth + 1)
            out: List[Tuple[str, Optional[str]]] = []
            for idx, bucket in enumerate(col_buckets):
                _check_no_mixed_tz(bucket, val_node)
                out.append((f"column{idx + 1}", _pg_common_expr_type(bucket)))
            return out

        def _walk_select(
            sel_node: sqlglot_exp.Select,
            outer_stack: _PgTypeScopeStack,
            cte_types: Dict[str, List[Tuple[str, Optional[str]]]],
            depth: int,
        ) -> List[Tuple[str, Optional[str]]]:
            local_scope: _PgTypeScope = {}
            local_order: List[Tuple[str, List[Tuple[str, Optional[str]]]]] = []
            from_node = sel_node.args.get("from_") or sel_node.args.get("from")
            if from_node is not None and from_node.this is not None:
                _register_from_source(from_node.this, local_scope, local_order, outer_stack, cte_types, depth + 1)
            for jn in sel_node.args.get("joins") or ():
                _register_from_source(jn.this, local_scope, local_order, outer_stack, cte_types, depth + 1)
                if jn.args.get("on") is not None:
                    _eval_expr(jn.args.get("on"), [*outer_stack, local_scope], cte_types, None, depth + 1)

            cur_stack: _PgTypeScopeStack = [*outer_stack, local_scope] if local_scope else list(outer_stack)

            # Evaluate SELECT list and record projected column names & types.
            projected: List[Tuple[str, Optional[str]]] = []
            select_aliases: _PgColTypes = {}
            for idx, proj in enumerate(sel_node.expressions or ()):
                if isinstance(proj, sqlglot_exp.Star):
                    for _, cols in local_order:
                        projected.extend(cols)
                    continue
                if isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Star):
                    rel_key = _pg_ident_key_from_node(proj.args.get("table"))
                    if rel_key is not None:
                        for r_k, cols in local_order:
                            if r_k == rel_key:
                                projected.extend(cols)
                                break
                    continue
                if isinstance(proj, sqlglot_exp.Alias):
                    t = _eval_expr(proj.this, cur_stack, cte_types, None, depth + 1)
                    al_key = _pg_ident_key_from_node(proj.args.get("alias")) or f"col_{idx}"
                    projected.append((al_key, t))
                    select_aliases[al_key] = t
                else:
                    t = _eval_expr(proj, cur_stack, cte_types, None, depth + 1)
                    if isinstance(proj, sqlglot_exp.Column):
                        ck = _pg_ident_key_from_node(proj.this) or f"col_{idx}"
                    else:
                        ck = f"col_{idx}"
                    projected.append((ck, t))
                    select_aliases[ck] = t

            # Walk DISTINCT ON, WHERE, GROUP BY, HAVING, WINDOW, ORDER BY, LIMIT, OFFSET.
            dist = sel_node.args.get("distinct")
            if isinstance(dist, sqlglot_exp.Distinct) and dist.args.get("on") is not None:
                _eval_expr(dist.args.get("on"), cur_stack, cte_types, select_aliases, depth + 1)
            for clause_key in ("where", "group", "having", "order", "limit", "offset"):
                c_node = sel_node.args.get(clause_key)
                if c_node is not None:
                    use_aliases = select_aliases if clause_key in ("group", "order") else None
                    _eval_expr(c_node, cur_stack, cte_types, use_aliases, depth + 1)
            for win_node in sel_node.args.get("windows") or ():
                _eval_expr(win_node, cur_stack, cte_types, select_aliases, depth + 1)
            return projected

        def _walk_query(
            query_node: Any,
            outer_stack: _PgTypeScopeStack,
            cte_types: Dict[str, List[Tuple[str, Optional[str]]]],
            depth: int,
        ) -> List[Tuple[str, Optional[str]]]:
            if query_node is None or depth > 32:
                return []
            local_ctes = dict(cte_types)
            with_node = query_node.args.get("with_") or query_node.args.get("with")
            if isinstance(with_node, sqlglot_exp.With):
                is_recursive = bool(with_node.args.get("recursive"))
                for cte in with_node.expressions or ():
                    if not isinstance(cte, sqlglot_exp.CTE) or cte.this is None:
                        continue
                    alias_node = cte.args.get("alias")
                    cte_key = (
                        _pg_ident_key_from_node(alias_node.this)
                        if isinstance(alias_node, sqlglot_exp.TableAlias) and alias_node.this is not None
                        else None
                    )
                    if is_recursive and isinstance(cte.this, sqlglot_exp.SetOperation) and cte_key:
                        # Seed recursive CTE column types from the anchor arm first.
                        anchor_cols = _walk_query(cte.this.this, outer_stack, local_ctes, depth + 1)
                        local_ctes[cte_key] = _apply_alias_cols(anchor_cols, alias_node)
                    cte_cols = _walk_query(cte.this, outer_stack, local_ctes, depth + 1)
                    if cte_key:
                        local_ctes[cte_key] = _apply_alias_cols(cte_cols, alias_node)

            if isinstance(query_node, sqlglot_exp.SetOperation):
                leaf_nodes, _ = _set_operation_arm_tree(query_node)
                arm_projections: List[List[Tuple[str, Optional[str]]]] = []
                for leaf in leaf_nodes:
                    arm_projections.append(_walk_query(leaf, outer_stack, local_ctes, depth + 1))
                out_cols: List[Tuple[str, Optional[str]]] = []
                if arm_projections:
                    n_cols = min(len(p) for p in arm_projections)
                    for idx in range(n_cols):
                        col_types = [p[idx][1] for p in arm_projections]
                        _check_no_mixed_tz(col_types, query_node)
                        out_cols.append((arm_projections[0][idx][0], _pg_common_expr_type(col_types)))
                out_aliases = dict(out_cols)
                for clause_key in ("order", "limit", "offset"):
                    c_node = query_node.args.get(clause_key)
                    if c_node is not None:
                        _eval_expr(c_node, outer_stack, local_ctes, out_aliases, depth + 1)
                return out_cols
            if isinstance(query_node, sqlglot_exp.Select):
                return _walk_select(query_node, outer_stack, local_ctes, depth + 1)
            if isinstance(query_node, sqlglot_exp.Values):
                return _walk_values(query_node, outer_stack, local_ctes, depth + 1)
            if isinstance(query_node, sqlglot_exp.Subquery):
                sub_cols = _walk_query(query_node.this, outer_stack, local_ctes, depth + 1)
                sub_aliases = dict(sub_cols)
                for clause_key in ("order", "limit", "offset"):
                    c_node = query_node.args.get(clause_key)
                    if c_node is not None:
                        _eval_expr(c_node, outer_stack, local_ctes, sub_aliases, depth + 1)
                return sub_cols
            return []

        _walk_query(q_ast, [], {}, 0)
