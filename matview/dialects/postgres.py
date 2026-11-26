"""PostgreSQL-specific codegen helpers (CASE result type inference)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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



