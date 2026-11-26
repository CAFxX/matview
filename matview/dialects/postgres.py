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



