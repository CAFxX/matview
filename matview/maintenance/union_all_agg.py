"""Tier 1D: Z-set maintenance for GROUP BY over UNION ALL branches."""

from __future__ import annotations

import hashlib
import re
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .._host import MixinHost
from ..engines import (
    AggFunc,
    Dialect,
    EngineSubdialect,
    MaintenanceStrategy,
    UnsupportedMaterializedViewError,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
)
from ..logical_ir import (
    ColumnDef,
    IncrementalBackfillSpec,
    JoinSpec,
    MaterializedViewPlan,
    SelectItem,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
)
from ..sql_frontend import (
    _contains_aggregate_call,
    _extract_set_op_branches_from_ast,
    _extract_verbatim_cte_or_subquery_sql,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _parse_single_agg_expr,
    _sanitize_ident,
    _split_top_level,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_view_ast,
    quote_sql_identifier,
)


class Tier1UnionAllAggMixin(MixinHost):
    def _compile_tier1_union_all_agg(self) -> MaterializedViewPlan:
        """
        Compiles an aggregation query over a `UNION ALL` CTE or derived subquery into Tier 1 O(1)
        per-branch algebraic delta triggers.
        """
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = list(self._working_select_items)
        grp_exprs = list(self._effective_group_by_exprs)
        inner_u_sql = self._union_all_inner_sql
        u_alias = self._union_all_sub_alias
        branch_specs = self._union_all_branch_specs

        count_star_alias: Optional[str] = None
        for item in work_items:
            if (
                item.is_aggregate
                and item.agg_func == AggFunc.COUNT_STAR
                and not item.filter_where
            ):
                count_star_alias = item.alias
                break
        row_count_col = self._q(count_star_alias) if count_star_alias else "__mv_row_count"

        # Check if all group_by columns are NOT NULL across every UNION ALL branch
        uall_has_nullable = False
        if grp_exprs:
            first_b_aliases = [it.alias.lower() for it in branch_specs[0].select_items]
            seen_g_al: Set[str] = set()
            for g_expr in grp_exprs:
                g_id = _unquote_ident(g_expr.split(".")[-1], dialect=self.dialect).lower() if _is_simple_column_expr(g_expr.strip(), dialect=self.dialect) else ""
                if not g_id or g_id not in first_b_aliases:
                    uall_has_nullable = True
                    break
                col_pos = first_b_aliases.index(g_id)
                for bs in branch_specs:
                    b_it = bs.select_items[col_pos]
                    ts_b = self._schemas_lower[bs.base_table.lower()]
                    c_b = ts_b.columns.get(b_it.source_col or "") if b_it.source_col else None
                    if c_b is None or not c_b.not_null:
                        uall_has_nullable = True
                        break
                if uall_has_nullable:
                    break
                al_g = self._group_col_alias(g_expr).lower()
                if al_g in seen_g_al:
                    uall_has_nullable = True
                    break
                seen_g_al.add(al_g)
        scoped = self.with_ir(
            uall_has_nullable_grp=uall_has_nullable,
            uall_orig_grp_exprs=tuple(grp_exprs),
        )
        return scoped._compile_tier1_union_all_agg_with_context()

    def _compile_tier1_union_all_agg_with_context(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = list(self._working_select_items)
        grp_exprs = list(self._effective_group_by_exprs)
        inner_u_sql = self._union_all_inner_sql
        u_alias = self._union_all_sub_alias
        branch_specs = self._union_all_branch_specs

        count_star_alias: Optional[str] = None
        for item in work_items:
            if (
                item.is_aggregate
                and item.agg_func == AggFunc.COUNT_STAR
                and not item.filter_where
            ):
                count_star_alias = item.alias
                break
        row_count_col = self._q(count_star_alias) if count_star_alias else "__mv_row_count"

        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()

        col_defs_sql: List[str] = []
        user_columns: List[str] = [it.alias for it in self.view.select_items]
        pk_type = self._canonical_group_pk_type()

        if not grp_exprs:
            col_defs_sql.append("__mv_global_id INTEGER NOT NULL PRIMARY KEY")
        elif has_nullable_grp:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden_grp: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_grp:
                seen_hidden_grp.add(mv_col.lower())
                if self.dialect == Dialect.SQLITE:
                    dt = "BLOB"
                else:
                    fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                    dt = self._infer_expr_storage_type(fake_it)
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                        dt = "VARCHAR(255)"
                nn = " NOT NULL" if not has_nullable_grp else ""
                col_defs_sql.append(f"{self._q(mv_col)} {dt}{nn}")

        def _expr_key(item: SelectItem) -> str:
            eff = self._effective_agg_value_expr(item, record_prefix=None).strip().lower()
            return f"{'dist:' if item.is_distinct else ''}{eff}"

        col_nonnull_count_map: Dict[str, str] = {}
        col_sum_map: Dict[str, str] = {}
        col_sum2_map: Dict[str, str] = {}
        col_true_cnt_map: Dict[str, str] = {}

        for item in work_items:
            if not item.is_aggregate:
                continue
            ekey = _expr_key(item)
            if item.agg_func == AggFunc.COUNT_COL:
                col_nonnull_count_map[ekey] = self._q(item.alias)
            elif item.agg_func == AggFunc.SUM:
                has_var_for_ekey = any(
                    w.is_aggregate
                    and _expr_key(w) == ekey
                    and w.agg_func in (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP)
                    for w in work_items
                )
                if not has_var_for_ekey:
                    col_sum_map[ekey] = self._q(item.alias)

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        # Infer aggregate types from first branch
        first_b = branch_specs[0]
        first_b_map = {it.alias.lower(): it.raw_expr for it in first_b.select_items}

        for item in work_items:
            if not item.is_aggregate:
                if self.dialect == Dialect.SQLITE:
                    dt = "BLOB"
                else:
                    dt = self._infer_expr_storage_type(item)
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                        dt = "VARCHAR(255)"
                nn = " NOT NULL" if (grp_exprs and not has_nullable_grp) else ""
                col_defs_sql.append(f"{self._q(item.alias)} {dt}{nn}")
            else:
                # Check if arg_expr maps to an integer column across all branches
                storage_type = self._infer_expr_storage_type(item)
                if self.dialect != Dialect.SQLITE and item.agg_func in (AggFunc.SUM, AggFunc.MIN, AggFunc.MAX) and item.arg_expr:
                    arg_id = _unquote_ident(item.arg_expr, dialect=self.dialect).lower()
                    if arg_id in first_b_map:
                        b_types = []
                        for bs in branch_specs:
                            b_m = {bi.alias.lower(): bi for bi in bs.select_items}
                            if arg_id in b_m:
                                bi_item = b_m[arg_id]
                                ts_b = self._schemas_lower[bs.base_table.lower()]
                                c_b = ts_b.columns.get(bi_item.source_col or "") if bi_item.source_col else None
                                if c_b:
                                    b_types.append(self._normalize_dialect_type(c_b.data_type))
                                elif re.match(r"^\d+$", bi_item.raw_expr.strip()):
                                    b_types.append("INT")
                                else:
                                    b_types.append(self._float_type())
                        if b_types and all("INT" in bt.upper() for bt in b_types):
                            storage_type = self._int_type() if item.agg_func == AggFunc.SUM else b_types[0]
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    # SQLite TOTAL returns NULL when the floating-point sum is
                    # NaN (e.g. inf + -inf); NOT NULL would reject the write.
                    total_nn = "" if self.dialect == Dialect.SQLITE else " NOT NULL"
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type}{total_nn} DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type}")

        if count_star_alias is None:
            col_defs_sql.append(f"{row_count_col} {self._int_type()} NOT NULL DEFAULT 0")

        aux_idx = 0
        for item in work_items:
            if not item.is_aggregate or item.agg_func is None:
                continue
            ekey = _expr_key(item)
            safe_id = _sanitize_ident(item.alias)
            if item.agg_func in (
                AggFunc.SUM,
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BIT_XOR,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            ) or (item.agg_func == AggFunc.COUNT_STAR and item.filter_where):
                if item.agg_func == AggFunc.COUNT_STAR and item.filter_where:
                    col_nonnull_count_map[ekey] = self._q(item.alias)
                elif ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    cnt_col = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = cnt_col
                    col_defs_sql.append(f"{cnt_col} {self._int_type()} NOT NULL DEFAULT 0")

            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    sum_col = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = sum_col
                    col_defs_sql.append(f"{sum_col} {self._var_accum_type(item)}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    s2_prefix = "__mv_m2_" if self.dialect == Dialect.SQLITE else "__mv_sum2_"
                    sum2_col = f"{s2_prefix}{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._var_accum_type(item, squared=True)}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
            pk_cols=pk_cols if (grp_exprs and not has_nullable_grp) else None,
        )
        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col and "__mv_elem_" in expr_s
        }
        can_pg15_unique_nnd = (
            bool(grp_exprs)
            and has_nullable_grp
            and len(set(c.lower() for c in group_mv_cols)) == len(grp_exprs)
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
            unique_group_nulls_not_distinct=can_pg15_unique_nnd,
        )
        reader_select_override = [
            self._compound_reader_exprs.get(u_col, u_col) for u_col in user_columns
        ]
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                select_exprs_override=reader_select_override,
                extra_where_filter=self._rewritten_having_clause,
            )
        )

        target_cols: List[str] = []
        select_exprs: List[str] = []
        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(grp_exprs))

        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                target_cols.append(mv_col)
                select_exprs.append(g_expr)

        for item in work_items:
            target_cols.append(item.alias)
            if not item.is_aggregate:
                select_exprs.append(item.raw_expr)
            else:
                select_exprs.append(self._sql_backfill_agg_expr(item))

        if count_star_alias is None:
            target_cols.append(row_count_col)
            select_exprs.append("COUNT(*)")

        ekey_to_item: Dict[str, SelectItem] = {
            _expr_key(it): it for it in work_items if it.is_aggregate
        }
        q_target_cols = self._append_aux_stat_target_cols(
            target_cols=target_cols,
            select_exprs=select_exprs,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            ekey_to_item=ekey_to_item,
            qualify_join=False,
            include_distinct_keyword=False,
        )

        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        grp_for_q = self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=has_nullable_grp)
        g_sql = f"\nGROUP BY {', '.join(grp_for_q)}" if grp_for_q else ""

        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
            f"SELECT {', '.join(select_exprs)} FROM (\n{inner_u_sql}\n) AS {u_alias}{w_sql}{g_sql};"
        )

        # Group branches by base table and generate per-branch delta statements
        by_table: Dict[str, List[ViewSpec]] = {}
        for bs in branch_specs:
            by_table.setdefault(bs.base_table, []).append(bs)

        orig_view = self.view
        orig_work_items = list(self._working_select_items)
        orig_grp_exprs = tuple(self._effective_group_by_exprs)
        orig_grp_aliases = [self._group_col_alias(g) for g in orig_grp_exprs]

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for base_t, b_list in by_table.items():
            ts = self._schemas_lower[base_t.lower()]
            ins_all: List[str] = []
            del_all: List[str] = []
            upd_old_all: List[str] = []
            upd_new_all: List[str] = []
            cleanup_all: List[str] = []
            refresh_grp_all: List[str] = []

            for bs in b_list:
                b_alias = bs.joins[0].alias if bs.joins else base_t
                col_subst: Dict[str, str] = {}
                for col_i, b_it in enumerate(bs.select_items):
                    u_col_name = first_b.select_items[col_i].alias.lower()
                    cleaned = re.sub(
                        rf"\b{re.escape(b_alias)}\.",
                        "",
                        b_it.raw_expr.strip(),
                        flags=re.IGNORECASE,
                    )
                    col_subst[u_col_name] = f"({cleaned})"

                def _subst_u(frag: Optional[str]) -> Optional[str]:
                    if frag is None:
                        return None
                    toks = _tokenize_sql(frag, read=_sqlglot_dialect(self.dialect))
                    if not toks:
                        return frag
                    out = []
                    cursor = 0
                    i = 0
                    n = len(toks)
                    while i < n:
                        tk = toks[i]
                        if (
                            _is_word_or_ident_token(tk)
                            and i + 2 < n
                            and toks[i + 1].token_type == SQLGlotTokenType.DOT
                            and _is_word_or_ident_token(toks[i + 2])
                        ):
                            if tk.text.lower() == u_alias.lower():
                                c_col = toks[i + 2].text.lower()
                                if tk.start > cursor:
                                    out.append(frag[cursor:tk.start])
                                out.append(col_subst.get(c_col, toks[i + 2].text))
                                cursor = toks[i + 2].end + 1
                                i += 3
                                continue
                        if _is_word_or_ident_token(tk):
                            is_after_dot = (i > 0 and toks[i - 1].token_type == SQLGlotTokenType.DOT)
                            is_before_dot = (i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.DOT)
                            is_fn = (
                                tk.token_type != SQLGlotTokenType.IDENTIFIER
                                and i + 1 < n
                                and toks[i + 1].token_type == SQLGlotTokenType.L_PAREN
                            )
                            if not is_after_dot and not is_before_dot and not is_fn:
                                c_low = tk.text.lower()
                                if c_low in col_subst:
                                    if tk.start > cursor:
                                        out.append(frag[cursor:tk.start])
                                    out.append(col_subst[c_low])
                                    cursor = tk.end + 1
                                    i += 1
                                    continue
                        i += 1
                    if cursor < len(frag):
                        out.append(frag[cursor:])
                    return "".join(out)

                mapped_work_items: List[SelectItem] = []
                orig_item_map: Dict[int, SelectItem] = {}
                mapped_nonnull: Dict[str, str] = {}
                mapped_sum: Dict[str, str] = {}
                mapped_sum2: Dict[str, str] = {}
                mapped_tcnt: Dict[str, str] = {}

                for orig_it in orig_work_items:
                    if not orig_it.is_aggregate:
                        new_it = SelectItem(
                            raw_expr=_subst_u(orig_it.raw_expr) or orig_it.raw_expr,
                            alias=orig_it.alias,
                            is_aggregate=False,
                        )
                    else:
                        new_it = SelectItem(
                            raw_expr=_subst_u(orig_it.raw_expr) or orig_it.raw_expr,
                            alias=orig_it.alias,
                            is_aggregate=True,
                            agg_func=orig_it.agg_func,
                            arg_expr=_subst_u(orig_it.arg_expr) if orig_it.arg_expr else orig_it.arg_expr,
                            extra_args=orig_it.extra_args,
                            is_distinct=orig_it.is_distinct,
                            filter_where=_subst_u(orig_it.filter_where),
                        )
                        orig_ekey = _expr_key(orig_it)
                        new_ekey = _expr_key(new_it)
                        if orig_ekey in col_nonnull_count_map:
                            mapped_nonnull[new_ekey] = col_nonnull_count_map[orig_ekey]
                        if orig_ekey in col_sum_map:
                            mapped_sum[new_ekey] = col_sum_map[orig_ekey]
                        if orig_ekey in col_sum2_map:
                            mapped_sum2[new_ekey] = col_sum2_map[orig_ekey]
                        if orig_ekey in col_true_cnt_map:
                            mapped_tcnt[new_ekey] = col_true_cnt_map[orig_ekey]
                    orig_item_map[id(new_it)] = orig_it
                    mapped_work_items.append(new_it)

                mapped_grps = tuple(_subst_u(g) or g for g in orig_grp_exprs)
                b_w = (
                    re.sub(rf"\b{re.escape(b_alias)}\.", "", bs.where_clause, flags=re.IGNORECASE)
                    if bs.where_clause
                    else None
                )
                o_w = _subst_u(orig_view.where_clause)
                comb_w = f"({b_w}) AND ({o_w})" if (b_w and o_w) else (b_w or o_w)

                # Derive immutable per-branch IR context without mutating compiler state
                branch_view = ViewSpec(
                    view_name=orig_view.view_name,
                    raw_select_sql=orig_view.raw_select_sql,
                    base_table=base_t,
                    joins=(JoinSpec(join_type="FROM", table_name=base_t, alias=base_t),),
                    referenced_tables=(base_t,),
                    select_items=tuple(mapped_work_items),
                    where_clause=orig_view.where_clause,
                    group_by_cols=mapped_grps,
                )
                branch_ctx = self.with_ir(
                    primary_schema=ts,
                    working_select_items=tuple(mapped_work_items),
                    effective_group_by_exprs=mapped_grps,
                    uall_grp_alias_map=MappingProxyType(
                        {mg: oa for mg, oa in zip(mapped_grps, orig_grp_aliases)}
                    ),
                    uall_orig_item_map=MappingProxyType(orig_item_map),
                    view=branch_view,
                )

                plus_b = branch_ctx._build_plus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="NEW",
                )
                minus_del_b, _ = branch_ctx._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="OLD",
                    include_delete_zero_mv=True,
                )
                minus_upd_b, del_zero_b = branch_ctx._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="OLD",
                    include_delete_zero_mv=False,
                )

                w_new_b = branch_ctx._rewrite_expr_for_record(comb_w, "NEW") if comb_w else None
                w_old_b = branch_ctx._rewrite_expr_for_record(comb_w, "OLD") if comb_w else None
                refresh_grp_b = branch_ctx._sqlite_refresh_group_keys_stmts(
                    mv_table, record_prefix="OLD"
                )

                ins_all.extend(branch_ctx._apply_where_guard_to_stmt(s, w_new_b) for s in plus_b)
                del_all.extend(branch_ctx._apply_where_guard_to_stmt(s, w_old_b) for s in minus_del_b)
                upd_old_all.extend(branch_ctx._apply_where_guard_to_stmt(s, w_old_b) for s in minus_upd_b)
                upd_new_all.extend(branch_ctx._apply_where_guard_to_stmt(s, w_new_b) for s in plus_b)
                if del_zero_b:
                    cleanup_all.append(branch_ctx._apply_where_guard_to_stmt(del_zero_b, w_old_b))
                refresh_grp_all.extend(
                    branch_ctx._apply_where_guard_to_stmt(s, w_old_b) for s in refresh_grp_b
                )

            # Run SQLite group-key representative refreshes after ALL branches
            # for base_t have decremented/deleted empty rows so a multi-branch
            # table delete never refreshes an about-to-be-deleted row.
            del_all.extend(refresh_grp_all)
            cleanup_all.extend(refresh_grp_all)

            trunc_stmts = [f"DELETE FROM {mv_table};", initial_populate_sql]
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_all,
                del_stmts=del_all,
                upd_old_stmts=upd_old_all,
                upd_new_stmts=upd_new_all,
                upd_cleanup_old_stmts=cleanup_all,
                trunc_stmts=trunc_stmts,
            )
            trigger_ddls.extend(t_ddls)
            trig_names.extend(t_ns)
            fn_names.extend(f_ns)

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            chunk_sql_templates=[initial_populate_sql],
        )
        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=[],
            index_ddls=index_ddls,
            initial_populate_sqls=[initial_populate_sql],
            trigger_ddls=trigger_ddls,
            readonly_trigger_ddls=ro_trigs,
            user_columns=user_columns,
            reader_view_ddl=reader_view_ddl,
            incremental_backfill_spec=inc_spec,
            trigger_names=ro_trig_names + trig_names,
            function_names=ro_fn_names + fn_names,
            limitations=self.limitations,
            object_prefix=self.object_prefix,
            db_view_name=self.db_view_name,
            index_names=index_names,
            user_indexes=list(self.user_indexes),
        )

