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

_PgColSet = Optional[Tuple[Set[str], Set[str]]]


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

        # Represent a relation's known user columns as _PgColSet:
        # (lower_cols, exact_cols), or None when unknown (e.g. unexpanded SELECT *).
        def _has_user_col(col_set: _PgColSet, ident: sqlglot_exp.Identifier) -> bool:
            if col_set is None:
                return False
            lower_cols, exact_cols = col_set
            if ident.quoted:
                if ident.name in exact_cols:
                    return True
                return ident.name == ident.name.lower() and ident.name in lower_cols
            return ident.name.lower() in lower_cols

        def _base_table_cols(table_name: str) -> Tuple[_PgColSet, bool]:
            ts = self._schemas_lower.get(table_name.lower())
            if ts is None:
                return None, False
            return ({c.lower() for c in ts.columns}, set(ts.columns)), True

        def _projected_cols(
            query_node: Optional[sqlglot_exp.Expr],
            cte_cols: Dict[str, _PgColSet],
        ) -> _PgColSet:
            if query_node is None:
                return None
            if isinstance(query_node, sqlglot_exp.Subquery) and query_node.this is not None:
                return _projected_cols(query_node.this, cte_cols)
            if isinstance(query_node, sqlglot_exp.SetOperation):
                return _projected_cols(query_node.this, cte_cols)
            if not isinstance(query_node, sqlglot_exp.Select):
                return None
            lower_c: Set[str] = set()
            exact_c: Set[str] = set()
            for proj in query_node.expressions or ():
                if isinstance(proj, sqlglot_exp.Star) or (
                    isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Star)
                ):
                    return None
                if isinstance(proj, sqlglot_exp.Alias):
                    al_node = proj.args.get("alias")
                    if isinstance(al_node, sqlglot_exp.Identifier):
                        k = _ident_key(al_node)
                        lower_c.add(k.lower())
                        exact_c.add(al_node.name)
                    elif proj.alias:
                        lower_c.add(proj.alias.lower())
                        exact_c.add(proj.alias)
                elif isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Identifier):
                    k = _ident_key(proj.this)
                    lower_c.add(k.lower())
                    exact_c.add(proj.this.name)
            return (lower_c, exact_c)

        def _table_alias_key_and_cols(
            src: sqlglot_exp.Expr,
            cte_cols: Dict[str, _PgColSet],
        ) -> Tuple[Optional[str], _PgColSet, bool]:
            alias_node = src.args.get("alias")
            rel_key: Optional[str] = None
            override_cols: _PgColSet = None
            if isinstance(alias_node, sqlglot_exp.TableAlias):
                if isinstance(alias_node.this, sqlglot_exp.Identifier):
                    rel_key = _ident_key(alias_node.this)
                alias_col_nodes = alias_node.args.get("columns") or ()
                if alias_col_nodes:
                    lc: Set[str] = set()
                    ec: Set[str] = set()
                    for cn in alias_col_nodes:
                        if isinstance(cn, sqlglot_exp.Identifier):
                            lc.add(_ident_key(cn).lower())
                            ec.add(cn.name)
                    override_cols = (lc, ec)
            if isinstance(src, sqlglot_exp.Table):
                if rel_key is None and isinstance(src.this, sqlglot_exp.Identifier):
                    rel_key = _ident_key(src.this)
                t_name = src.name or ""
                if not src.db and t_name.lower() in cte_cols:
                    return rel_key, override_cols or cte_cols[t_name.lower()], False
                cols, is_base = _base_table_cols(t_name)
                return rel_key, override_cols or cols, is_base
            if isinstance(src, sqlglot_exp.Subquery):
                sub_cols = _projected_cols(src, cte_cols)
                return rel_key, override_cols or sub_cols, False
            return rel_key, override_cols, False

        def _walk_expr(
            expr_node: Optional[sqlglot_exp.Expr],
            owner_select: Optional[sqlglot_exp.Select],
            in_scope_rels: Dict[str, _PgColSet],
            in_scope_base: Dict[str, bool],
            cte_cols: Dict[str, _PgColSet],
            select_output_aliases: Set[str],
            *,
            allow_output_alias: bool = False,
        ) -> None:
            if expr_node is None:
                return
            if isinstance(expr_node, (sqlglot_exp.Select, sqlglot_exp.SetOperation)):
                _walk_query(expr_node, in_scope_rels, in_scope_base, cte_cols)
                return
            if isinstance(expr_node, sqlglot_exp.Subquery) and isinstance(
                expr_node.this, (sqlglot_exp.Select, sqlglot_exp.SetOperation, sqlglot_exp.Subquery)
            ):
                _walk_query(expr_node.this, in_scope_rels, in_scope_base, cte_cols)
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
                        if tbl_key in in_scope_rels:
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
                    is_direct_group_or_order = allow_output_alias and (
                        isinstance(expr_node.parent, (sqlglot_exp.Group, sqlglot_exp.Ordered))
                    )
                    if is_direct_group_or_order and col_key in select_output_aliases:
                        return
                    any_user_col = any(_has_user_col(cs, col_ident) for cs in in_scope_rels.values())
                    # Check B1: bare whole-row reference (`to_jsonb(t)`, `(t).col`, `t::text`, `col(t)`).
                    if col_key in in_scope_rels and not any_user_col:
                        raise UnsupportedMaterializedViewError(
                            f"PostgreSQL materialized view references whole-row relation variable "
                            f"{col_ident.name!r}; whole-row references are not supported in "
                            f"materialized views. Reference explicit columns instead."
                        )
                    # Check B2: unqualified PostgreSQL system column (`ctid`, `xmin`, `xmax`, `cmin`, `cmax`, `tableoid`).
                    if _is_sys_col_ident(col_ident) and not any_user_col and any(in_scope_base.values()):
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
                        if tbl_key in in_scope_rels and in_scope_base.get(tbl_key, False):
                            if not _has_user_col(in_scope_rels[tbl_key], col_ident):
                                raise UnsupportedMaterializedViewError(
                                    f"PostgreSQL materialized view references system column "
                                    f"'{tbl_id.name}.{col_ident.name}'; PostgreSQL system columns "
                                    f"(ctid, xmin, xmax, cmin, cmax, tableoid) have no stable value "
                                    f"across maintenance statements and are not supported."
                                )
                return

            # Recurse into child expressions of this node.
            for child in expr_node.iter_expressions():
                _walk_expr(
                    child,
                    owner_select,
                    in_scope_rels,
                    in_scope_base,
                    cte_cols,
                    select_output_aliases,
                    allow_output_alias=allow_output_alias,
                )

        def _walk_query(
            query_node: Optional[sqlglot_exp.Expr],
            outer_rels: Dict[str, _PgColSet],
            outer_base: Dict[str, bool],
            cte_cols: Dict[str, _PgColSet],
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
                    cte_name = cte.alias.lower() if cte.alias else ""
                    header_cols: _PgColSet = None
                    if isinstance(cte_alias_node, sqlglot_exp.TableAlias):
                        col_nodes = cte_alias_node.args.get("columns") or ()
                        if col_nodes:
                            lc = {_ident_key(cn).lower() for cn in col_nodes if isinstance(cn, sqlglot_exp.Identifier)}
                            ec = {cn.name for cn in col_nodes if isinstance(cn, sqlglot_exp.Identifier)}
                            header_cols = (lc, ec)
                    if is_recursive and cte_name:
                        local_ctes[cte_name] = header_cols or (
                            _projected_cols(cte.this, local_ctes) if cte.this is not None else None
                        )
                    if cte.this is not None:
                        _walk_query(cte.this, outer_rels, outer_base, local_ctes)
                    if cte_name:
                        local_ctes[cte_name] = header_cols or (
                            _projected_cols(cte.this, local_ctes) if cte.this is not None else None
                        )

            if isinstance(query_node, sqlglot_exp.SetOperation):
                _walk_query(query_node.this, outer_rels, outer_base, local_ctes)
                _walk_query(query_node.expression, outer_rels, outer_base, local_ctes)
                return

            if isinstance(query_node, sqlglot_exp.Subquery) and query_node.this is not None:
                _walk_query(query_node.this, outer_rels, outer_base, local_ctes)
                return

            if not isinstance(query_node, sqlglot_exp.Select):
                return

            local_rels: Dict[str, _PgColSet] = {}
            local_base: Dict[str, bool] = {}

            def _register_source(src: Optional[sqlglot_exp.Expr], *, is_lateral: bool = False) -> None:
                if src is None:
                    return
                if isinstance(src, sqlglot_exp.Lateral):
                    _register_source(src.this, is_lateral=True)
                    return
                if isinstance(src, sqlglot_exp.Subquery) and src.this is not None:
                    if is_lateral:
                        _walk_query(src.this, {**outer_rels, **local_rels}, {**outer_base, **local_base}, local_ctes)
                    else:
                        _walk_query(src.this, outer_rels, outer_base, local_ctes)
                elif isinstance(src, sqlglot_exp.Table) and isinstance(src.this, sqlglot_exp.Func):
                    _walk_expr(
                        src.this,
                        query_node,
                        {**outer_rels, **local_rels},
                        {**outer_base, **local_base},
                        local_ctes,
                        set(),
                    )
                rk, rc, rb = _table_alias_key_and_cols(src, local_ctes)
                if rk is not None:
                    local_rels[rk] = rc
                    local_base[rk] = rb

            from_node = query_node.args.get("from_") or query_node.args.get("from")
            if from_node is not None:
                _register_source(from_node.this)
            for jn in query_node.args.get("joins") or ():
                if isinstance(jn, sqlglot_exp.Join):
                    _register_source(jn.this, is_lateral=isinstance(jn.this, sqlglot_exp.Lateral))

            in_scope_rels = {**outer_rels, **local_rels}
            in_scope_base = {**outer_base, **local_base}

            select_output_aliases: Set[str] = set()
            for proj in query_node.expressions or ():
                if isinstance(proj, sqlglot_exp.Alias):
                    al_n = proj.args.get("alias")
                    if isinstance(al_n, sqlglot_exp.Identifier):
                        select_output_aliases.add(_ident_key(al_n))

            for proj in query_node.expressions or ():
                _walk_expr(proj, query_node, in_scope_rels, in_scope_base, local_ctes, select_output_aliases)
            distinct_node = query_node.args.get("distinct")
            if isinstance(distinct_node, sqlglot_exp.Distinct):
                _walk_expr(distinct_node.args.get("on"), query_node, in_scope_rels, in_scope_base, local_ctes, select_output_aliases)
            for jn in query_node.args.get("joins") or ():
                if isinstance(jn, sqlglot_exp.Join):
                    _walk_expr(jn.args.get("on"), query_node, in_scope_rels, in_scope_base, local_ctes, select_output_aliases)
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
                        in_scope_rels,
                        in_scope_base,
                        local_ctes,
                        select_output_aliases,
                        allow_output_alias=allow_alias,
                    )
            for win in query_node.args.get("windows") or ():
                if isinstance(win, sqlglot_exp.Expr):
                    _walk_expr(win, query_node, in_scope_rels, in_scope_base, local_ctes, select_output_aliases)

        sqls_to_check: List[str] = []
        for raw in (self.raw_view.raw_select_sql, self.view.raw_select_sql):
            s = (raw or "").strip()
            if s and s not in sqls_to_check:
                sqls_to_check.append(s)
        for sql_text in sqls_to_check:
            try:
                ast_root = parse_sql_ast(sql_text, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            _walk_query(ast_root, {}, {}, {})



