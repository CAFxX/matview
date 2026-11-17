"""PostgreSQL-specific codegen helpers (CASE inference, vectorized transition triggers, MERGE)."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp

from ..engines import (
    AggFunc,
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
)
from ..logical_ir import ColumnDef, JoinSpec, SelectItem, TableSchema, ViewSpec
from ..sql_frontend import (
    _has_subquery_ast,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    quote_sql_identifier,
    tokenize_sql,
)
from .templates import render_procedural_sql


class PostgresDialectMixin:
    def _postgres_case_result_storage_type(self, expr: str) -> Optional[str]:
        """Infer PostgreSQL CASE common types, including nested column branches."""
        if self.dialect != Dialect.POSTGRES:
            return None
        return self._infer_ast_expr_storage_type(expr)


    def _build_tier1_vectorized_pg_bodies(
        self,
        mv_table: str,
        base_t: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        try:
            grp_exprs = self._effective_group_by_exprs
            has_nullable_grp = self._has_nullable_group_col()
            pk_cols = self._tier1_pk_cols()
            q_pk_cols = [self._q(c) for c in pk_cols]
            work_items = self._working_select_items

            # Native recomputation subqueries correlate against transition-row
            # columns. PostgreSQL rejects those outer references when GROUP BY
            # contains an expression (for example COALESCE(region, 'UNK')).
            # Keep this shape correct by using the existing row-loop fallback
            # until recomputation is joined to projected group-key values.
            def _requires_transition_group_recompute(item: SelectItem) -> bool:
                if not item.is_aggregate:
                    return False
                if item.agg_func == AggFunc.AVG:
                    return not item.is_distinct and self._requires_native_avg_recompute(item)
                return item.agg_func not in (
                    AggFunc.COUNT_STAR,
                    AggFunc.COUNT_COL,
                    AggFunc.SUM,
                    AggFunc.TOTAL,
                    AggFunc.MIN,
                    AggFunc.MAX,
                    AggFunc.BOOL_AND,
                    AggFunc.BOOL_OR,
                    AggFunc.BIT_AND,
                    AggFunc.BIT_OR,
                )

            if (
                grp_exprs
                and any(not self._is_simple_col_ref(g) for g in grp_exprs)
                and any(_requires_transition_group_recompute(it) for it in work_items)
            ):
                return None, None, None

            where_new_clause = self.view.where_clause
            where_old_clause = self.view.where_clause
            if self.watermark_catalog_table and self._alg_mode in ("single", "distinct"):
                wm_pk_col = None
                if self.primary_schema:
                    for c_def in self.primary_schema.columns.values():
                        if c_def.primary_key and "INT" in c_def.data_type.upper():
                            wm_pk_col = c_def.name
                            break
                if wm_pk_col:
                    q_pk = self._q(wm_pk_col)
                    wm_new = (
                        f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                        f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                        f"AND __mv_new_rows.{q_pk} > backfill_cursor_pk AND __mv_new_rows.{q_pk} <= backfill_max_pk)"
                    )
                    wm_old = (
                        f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                        f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                        f"AND __mv_old_rows.{q_pk} > backfill_cursor_pk AND __mv_old_rows.{q_pk} <= backfill_max_pk)"
                    )
                    where_new_clause = f"({where_new_clause}) AND ({wm_new})" if where_new_clause else wm_new
                    where_old_clause = f"({where_old_clause}) AND ({wm_old})" if where_old_clause else wm_old

            w_new_cond = self._rewrite_expr_for_record(where_new_clause, "__mv_new_rows") if where_new_clause else None
            w_old_cond = self._rewrite_expr_for_record(where_old_clause, "__mv_old_rows") if where_old_clause else None
            w_new_sql = f"\n    WHERE {w_new_cond}" if w_new_cond else ""
            w_old_sql = f"\n    WHERE {w_old_cond}" if w_old_cond else ""
            w_new_and = f" AND ({w_new_cond})" if w_new_cond else ""
            w_old_and = f" AND ({w_old_cond})" if w_old_cond else ""

            if not grp_exprs:
                rec_grp_new: List[str] = []
                rec_grp_old: List[str] = []
                grp_by_new_sql = ""
                grp_by_old_sql = ""
                grp_match_del = f"{mv_table}.__mv_global_id = __del.__mv_global_id"
            else:
                rec_grp_new = [self._rewrite_expr_for_record(g, "__mv_new_rows") for g in grp_exprs]
                rec_grp_old = [self._rewrite_expr_for_record(g, "__mv_old_rows") for g in grp_exprs]
                grp_by_new_sql = f"\n    GROUP BY {', '.join(rec_grp_new)}"
                grp_by_old_sql = f"\n    GROUP BY {', '.join(rec_grp_old)}"
                if has_nullable_grp:
                    pk_new_val = self._canonical_group_pk_expr(rec_grp_new)
                    pk_old_val = self._canonical_group_pk_expr(rec_grp_old)
                    grp_match_del = f"{mv_table}.__mv_pk = __del.__mv_pk"
                else:
                    grp_match_del = " AND ".join(
                        f"{mv_table}.{self._q(self._group_col_alias(g))} = __del.{self._q(self._group_col_alias(g))}"
                        for g in grp_exprs
                    )

            # Build plus (upsert) statements
            ins_cols: List[str] = []
            ins_vals: List[str] = []
            upd_assignments: List[str] = []
            updated_cols: Set[str] = set()

            if not grp_exprs:
                ins_cols.append("__mv_global_id")
                ins_vals.append("1 AS __mv_global_id")
            elif has_nullable_grp:
                ins_cols.append("__mv_pk")
                ins_vals.append(f"{pk_new_val} AS __mv_pk")

            for i, g_expr in enumerate(grp_exprs):
                mv_col = self._group_col_alias(g_expr)
                ins_cols.append(self._q(mv_col))
                ins_vals.append(f"{rec_grp_new[i]} AS {self._q(mv_col)}")

            ekey_to_item: Dict[str, SelectItem] = {
                expr_key_fn(it): it for it in work_items if it.is_aggregate
            }

            for item in work_items:
                if not item.is_aggregate:
                    continue
                col_alias = self._q(item.alias)
                updated_cols.add(col_alias)
                ekey = expr_key_fn(item)
                val_e_new = self._effective_agg_value_expr(item, record_prefix="__mv_new_rows")
                ins_cols.append(col_alias)

                if item.agg_func == AggFunc.COUNT_STAR:
                    if item.filter_where:
                        fw = self._rewrite_expr_for_record(item.filter_where, "__mv_new_rows")
                        ins_vals.append(f"COUNT(*) FILTER (WHERE {fw}) AS {col_alias}")
                    else:
                        ins_vals.append(f"COUNT(*) AS {col_alias}")
                    upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + EXCLUDED.{col_alias}")

                elif item.agg_func == AggFunc.COUNT_COL:
                    if item.is_distinct:
                        ins_vals.append(f"0 AS {col_alias}")
                        upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias}")
                    else:
                        ins_vals.append(f"COUNT({val_e_new}) AS {col_alias}")
                        upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + EXCLUDED.{col_alias}")

                elif item.agg_func == AggFunc.SUM:
                    if item.is_distinct:
                        ins_vals.append(f"0 AS {col_alias}")
                        upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias}")
                    else:
                        ins_vals.append(f"SUM({val_e_new}) AS {col_alias}")
                        upd_assignments.append(
                            f"{col_alias} = CASE WHEN {mv_table}.{col_alias} IS NULL THEN EXCLUDED.{col_alias} "
                            f"WHEN EXCLUDED.{col_alias} IS NULL THEN {mv_table}.{col_alias} "
                            f"ELSE {mv_table}.{col_alias} + EXCLUDED.{col_alias} END"
                        )

                elif item.agg_func == AggFunc.TOTAL:
                    if item.is_distinct:
                        ins_vals.append(f"0.0 AS {col_alias}")
                        upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias}")
                    else:
                        ins_vals.append(f"COALESCE(SUM(1.0 * ({val_e_new})), 0.0) AS {col_alias}")
                        upd_assignments.append(
                            f"{col_alias} = COALESCE({mv_table}.{col_alias}, 0.0) + COALESCE(EXCLUDED.{col_alias}, 0.0)"
                        )

                elif item.agg_func == AggFunc.MIN:
                    ins_vals.append(f"MIN({val_e_new}) AS {col_alias}")
                    upd_assignments.append(
                        f"{col_alias} = CASE WHEN {mv_table}.{col_alias} IS NULL THEN EXCLUDED.{col_alias} "
                        f"WHEN EXCLUDED.{col_alias} IS NULL THEN {mv_table}.{col_alias} "
                        f"WHEN EXCLUDED.{col_alias} < {mv_table}.{col_alias} THEN EXCLUDED.{col_alias} "
                        f"ELSE {mv_table}.{col_alias} END"
                    )

                elif item.agg_func == AggFunc.MAX:
                    ins_vals.append(f"MAX({val_e_new}) AS {col_alias}")
                    upd_assignments.append(
                        f"{col_alias} = CASE WHEN {mv_table}.{col_alias} IS NULL THEN EXCLUDED.{col_alias} "
                        f"WHEN EXCLUDED.{col_alias} IS NULL THEN {mv_table}.{col_alias} "
                        f"WHEN EXCLUDED.{col_alias} > {mv_table}.{col_alias} THEN EXCLUDED.{col_alias} "
                        f"ELSE {mv_table}.{col_alias} END"
                    )

                elif item.agg_func == AggFunc.AVG:
                    if item.is_distinct:
                        ins_vals.append(f"0 AS {col_alias}")
                        upd_assignments.append(f"{col_alias} = {mv_table}.{col_alias}")
                    elif self._requires_native_avg_recompute(item):
                        # A previously unseen group consists entirely of rows in
                        # this transition table. Aggregate those rows directly:
                        # a correlated subquery against raw transition-table
                        # columns is illegal in PostgreSQL when the outer GROUP BY
                        # key is an expression such as COALESCE(region, 'UNK').
                        ins_vals.append(f"AVG({val_e_new}) AS {col_alias}")
                        recomp = self._build_recompute_subquery_raw(item, mv_table)
                        upd_assignments.append(f"{col_alias} = {recomp}")
                    else:
                        cnt_col = col_nonnull_count_map[ekey]
                        sum_col = col_sum_map[ekey]
                        # A previously unseen group consists entirely of the
                        # transition rows; the native aggregate is exact.
                        ins_vals.append(f"AVG({val_e_new}) AS {col_alias}")
                        new_cnt = f"({mv_table}.{cnt_col} + EXCLUDED.{cnt_col})"
                        new_sum = (
                            f"(CASE WHEN {mv_table}.{sum_col} IS NULL THEN EXCLUDED.{sum_col} "
                            f"WHEN EXCLUDED.{sum_col} IS NULL THEN {mv_table}.{sum_col} "
                            f"ELSE {mv_table}.{sum_col} + EXCLUDED.{sum_col} END)"
                        )
                        upd_assignments.append(
                            f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                            f"ELSE {self._avg_from_sum_count_sql(item, new_sum, new_cnt)} END"
                        )

                elif item.agg_func == AggFunc.BOOL_AND:
                    cnt_col = col_nonnull_count_map[ekey]
                    tcnt_col = col_true_cnt_map[ekey]
                    ins_vals.append(
                        f"(CASE WHEN COUNT({val_e_new}) = 0 THEN NULL "
                        f"WHEN COUNT({val_e_new}) = COUNT(*) FILTER (WHERE ({val_e_new}) IS NOT NULL AND ({val_e_new})) THEN TRUE "
                        f"ELSE FALSE END) AS {col_alias}"
                    )
                    new_cnt = f"({mv_table}.{cnt_col} + EXCLUDED.{cnt_col})"
                    new_tcnt = f"({mv_table}.{tcnt_col} + EXCLUDED.{tcnt_col})"
                    upd_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL WHEN {new_cnt} = {new_tcnt} THEN TRUE ELSE FALSE END"
                    )

                elif item.agg_func == AggFunc.BOOL_OR:
                    cnt_col = col_nonnull_count_map[ekey]
                    ins_vals.append(
                        f"(CASE WHEN COUNT({val_e_new}) = 0 THEN NULL "
                        f"WHEN COUNT(*) FILTER (WHERE ({val_e_new}) IS NOT NULL AND ({val_e_new})) > 0 THEN TRUE "
                        f"ELSE FALSE END) AS {col_alias}"
                    )
                    new_cnt = f"({mv_table}.{cnt_col} + EXCLUDED.{cnt_col})"
                    upd_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                        f"WHEN ({mv_table}.{col_true_cnt_map[ekey]} + EXCLUDED.{col_true_cnt_map[ekey]}) > 0 THEN TRUE ELSE FALSE END"
                    )

                elif item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR):
                    fn = "BIT_AND" if item.agg_func == AggFunc.BIT_AND else "BIT_OR"
                    op = "&" if item.agg_func == AggFunc.BIT_AND else "|"
                    ins_vals.append(f"{fn}({val_e_new}) AS {col_alias}")
                    upd_assignments.append(
                        f"{col_alias} = CASE WHEN {mv_table}.{col_alias} IS NULL THEN EXCLUDED.{col_alias} "
                        f"WHEN EXCLUDED.{col_alias} IS NULL THEN {mv_table}.{col_alias} "
                        f"ELSE {mv_table}.{col_alias} {op} EXCLUDED.{col_alias} END"
                    )

                elif item.agg_func in (
                    AggFunc.VAR_POP,
                    AggFunc.VAR_SAMP,
                    AggFunc.STDDEV_POP,
                    AggFunc.STDDEV_SAMP,
                    AggFunc.BIT_XOR,
                ):
                    # As with AVG above, compute a new group's initial native
                    # aggregate over the grouped transition rows. Existing
                    # groups are recomputed from the base table on conflict.
                    ins_vals.append(f"{item.agg_func.value}({val_e_new}) AS {col_alias}")
                    recomp = self._build_recompute_subquery_raw(item, mv_table)
                    upd_assignments.append(f"{col_alias} = {recomp}")

                else:
                    ins_recomp = self._build_recompute_subquery_raw(item, "__mv_new_rows")
                    recomp = self._build_recompute_subquery_raw(item, mv_table)
                    ins_vals.append(f"{ins_recomp} AS {col_alias}")
                    upd_assignments.append(f"{col_alias} = {recomp}")

            for ekey, cnt_col in col_nonnull_count_map.items():
                if cnt_col not in updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_new_rows")
                    ins_cols.append(cnt_col)
                    ins_vals.append(f"COUNT({ve}) AS {cnt_col}")
                    upd_assignments.append(f"{cnt_col} = {mv_table}.{cnt_col} + EXCLUDED.{cnt_col}")
                    updated_cols.add(cnt_col)

            for ekey, sum_col in col_sum_map.items():
                if sum_col not in updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_new_rows")
                    ins_cols.append(sum_col)
                    ins_vals.append(f"SUM({ve}) AS {sum_col}")
                    upd_assignments.append(
                        f"{sum_col} = CASE WHEN {mv_table}.{sum_col} IS NULL THEN EXCLUDED.{sum_col} "
                        f"WHEN EXCLUDED.{sum_col} IS NULL THEN {mv_table}.{sum_col} "
                        f"ELSE {mv_table}.{sum_col} + EXCLUDED.{sum_col} END"
                    )
                    updated_cols.add(sum_col)

            for ekey, sum2_col in col_sum2_map.items():
                if sum2_col not in updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_new_rows")
                    ins_cols.append(sum2_col)
                    ins_vals.append(f"SUM(({ve}) * ({ve})) AS {sum2_col}")
                    upd_assignments.append(
                        f"{sum2_col} = CASE WHEN {mv_table}.{sum2_col} IS NULL THEN EXCLUDED.{sum2_col} "
                        f"WHEN EXCLUDED.{sum2_col} IS NULL THEN {mv_table}.{sum2_col} "
                        f"ELSE {mv_table}.{sum2_col} + EXCLUDED.{sum2_col} END"
                    )
                    updated_cols.add(sum2_col)

            for ekey, tcnt_col in col_true_cnt_map.items():
                if tcnt_col not in updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_new_rows")
                    ins_cols.append(tcnt_col)
                    ins_vals.append(f"COUNT(*) FILTER (WHERE ({ve}) IS NOT NULL AND ({ve})) AS {tcnt_col}")
                    upd_assignments.append(f"{tcnt_col} = {mv_table}.{tcnt_col} + EXCLUDED.{tcnt_col}")
                    updated_cols.add(tcnt_col)

            q_row_count = self._q(row_count_col)
            if q_row_count not in updated_cols:
                ins_cols.append(q_row_count)
                ins_vals.append(f"COUNT(*) AS {q_row_count}")
                upd_assignments.append(f"{q_row_count} = {mv_table}.{q_row_count} + EXCLUDED.{q_row_count}")
                updated_cols.add(q_row_count)

            having_clause = "\n    HAVING COUNT(*) > 0" if not grp_exprs else ""
            sep_upd = ",\n        "
            upsert_plus_sql = (
                f"INSERT INTO {mv_table} ({', '.join(ins_cols)})\n"
                f"    SELECT {', '.join(ins_vals)}\n"
                f"    FROM __mv_new_rows{w_new_sql}{grp_by_new_sql}{having_clause}\n"
                f"    ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
                f"        {sep_upd.join(upd_assignments)};"
            )

            # Build minus (delta update) statements for __mv_old_rows
            del_select_exprs: List[str] = []
            del_updates: List[str] = []
            del_updated_cols: Set[str] = set()

            if not grp_exprs:
                del_select_exprs.append("1 AS __mv_global_id")
            elif has_nullable_grp:
                del_select_exprs.append(f"{pk_old_val} AS __mv_pk")
            else:
                for i, g_expr in enumerate(grp_exprs):
                    mv_col = self._group_col_alias(g_expr)
                    del_select_exprs.append(f"{rec_grp_old[i]} AS {self._q(mv_col)}")

            for item in work_items:
                if not item.is_aggregate:
                    continue
                col_alias = self._q(item.alias)
                del_updated_cols.add(col_alias)
                ekey = expr_key_fn(item)
                val_e_old = self._effective_agg_value_expr(item, record_prefix="__mv_old_rows")

                if item.agg_func == AggFunc.COUNT_STAR:
                    if item.filter_where:
                        fw = self._rewrite_expr_for_record(item.filter_where, "__mv_old_rows")
                        del_select_exprs.append(f"COUNT(*) FILTER (WHERE {fw}) AS {col_alias}")
                    else:
                        del_select_exprs.append(f"COUNT(*) AS {col_alias}")
                    del_updates.append(f"{col_alias} = {mv_table}.{col_alias} - __del.{col_alias}")

                elif item.agg_func == AggFunc.COUNT_COL:
                    if not item.is_distinct:
                        del_select_exprs.append(f"COUNT({val_e_old}) AS {col_alias}")
                        del_updates.append(f"{col_alias} = {mv_table}.{col_alias} - __del.{col_alias}")

                elif item.agg_func == AggFunc.SUM:
                    if not item.is_distinct:
                        cnt_col = col_nonnull_count_map[ekey]
                        del_select_exprs.append(f"SUM({val_e_old}) AS {col_alias}")
                        del_updates.append(
                            f"{col_alias} = CASE WHEN ({mv_table}.{cnt_col} - __del.{cnt_col}) <= 0 THEN NULL "
                            f"ELSE {mv_table}.{col_alias} - COALESCE(__del.{col_alias}, 0) END"
                        )

                elif item.agg_func == AggFunc.TOTAL:
                    if not item.is_distinct:
                        del_select_exprs.append(f"COALESCE(SUM(1.0 * ({val_e_old})), 0.0) AS {col_alias}")
                        del_updates.append(f"{col_alias} = {mv_table}.{col_alias} - __del.{col_alias}")

                elif item.agg_func == AggFunc.MIN:
                    del_select_exprs.append(f"MIN({val_e_old}) AS {col_alias}")
                    recomp = self._build_recompute_subquery_raw(item, mv_table)
                    del_updates.append(
                        f"{col_alias} = CASE WHEN __del.{col_alias} IS NOT NULL AND {mv_table}.{col_alias} >= __del.{col_alias} "
                        f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                    )

                elif item.agg_func == AggFunc.MAX:
                    del_select_exprs.append(f"MAX({val_e_old}) AS {col_alias}")
                    recomp = self._build_recompute_subquery_raw(item, mv_table)
                    del_updates.append(
                        f"{col_alias} = CASE WHEN __del.{col_alias} IS NOT NULL AND {mv_table}.{col_alias} <= __del.{col_alias} "
                        f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                    )

                elif item.agg_func == AggFunc.AVG:
                    if not item.is_distinct:
                        if self._requires_native_avg_recompute(item):
                            recomp = self._build_recompute_subquery_raw(item, mv_table)
                            del_updates.append(f"{col_alias} = {recomp}")
                        else:
                            cnt_col = col_nonnull_count_map[ekey]
                            sum_col = col_sum_map[ekey]
                            new_cnt = f"({mv_table}.{cnt_col} - __del.{cnt_col})"
                            new_sum = f"({mv_table}.{sum_col} - COALESCE(__del.{sum_col}, 0))"
                            del_updates.append(
                                f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                                f"ELSE {self._avg_from_sum_count_sql(item, new_sum, new_cnt)} END"
                            )

                elif item.agg_func == AggFunc.BOOL_AND:
                    cnt_col = col_nonnull_count_map[ekey]
                    tcnt_col = col_true_cnt_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_col} - __del.{cnt_col})"
                    new_tcnt = f"({mv_table}.{tcnt_col} - __del.{tcnt_col})"
                    del_updates.append(
                        f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL WHEN {new_cnt} = {new_tcnt} THEN TRUE ELSE FALSE END"
                    )

                elif item.agg_func == AggFunc.BOOL_OR:
                    cnt_col = col_nonnull_count_map[ekey]
                    tcnt_col = col_true_cnt_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_col} - __del.{cnt_col})"
                    del_updates.append(
                        f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                        f"WHEN ({mv_table}.{tcnt_col} - __del.{tcnt_col}) > 0 THEN TRUE ELSE FALSE END"
                    )

                else:
                    recomp = self._build_recompute_subquery_raw(item, mv_table)
                    del_updates.append(f"{col_alias} = {recomp}")

            for ekey, cnt_col in col_nonnull_count_map.items():
                if cnt_col not in del_updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_old_rows")
                    del_select_exprs.append(f"COUNT({ve}) AS {cnt_col}")
                    del_updates.append(f"{cnt_col} = {mv_table}.{cnt_col} - __del.{cnt_col}")
                    del_updated_cols.add(cnt_col)

            for ekey, sum_col in col_sum_map.items():
                if sum_col not in del_updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_old_rows")
                    del_select_exprs.append(f"SUM({ve}) AS {sum_col}")
                    cnt_c = col_nonnull_count_map[ekey]
                    del_updates.append(
                        f"{sum_col} = CASE WHEN ({mv_table}.{cnt_c} - __del.{cnt_c}) <= 0 THEN NULL "
                        f"ELSE {mv_table}.{sum_col} - COALESCE(__del.{sum_col}, 0) END"
                    )
                    del_updated_cols.add(sum_col)

            for ekey, sum2_col in col_sum2_map.items():
                if sum2_col not in del_updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_old_rows")
                    del_select_exprs.append(f"SUM(({ve}) * ({ve})) AS {sum2_col}")
                    cnt_c = col_nonnull_count_map[ekey]
                    del_updates.append(
                        f"{sum2_col} = CASE WHEN ({mv_table}.{cnt_c} - __del.{cnt_c}) <= 0 THEN NULL "
                        f"ELSE {mv_table}.{sum2_col} - COALESCE(__del.{sum2_col}, 0) END"
                    )
                    del_updated_cols.add(sum2_col)

            for ekey, tcnt_col in col_true_cnt_map.items():
                if tcnt_col not in del_updated_cols:
                    it = ekey_to_item[ekey]
                    ve = self._effective_agg_value_expr(it, record_prefix="__mv_old_rows")
                    del_select_exprs.append(f"COUNT(*) FILTER (WHERE ({ve}) IS NOT NULL AND ({ve})) AS {tcnt_col}")
                    del_updates.append(f"{tcnt_col} = {mv_table}.{tcnt_col} - __del.{tcnt_col}")
                    del_updated_cols.add(tcnt_col)

            if q_row_count not in del_updated_cols:
                del_select_exprs.append(f"COUNT(*) AS {q_row_count}")
                del_updates.append(f"{q_row_count} = {mv_table}.{q_row_count} - __del.{q_row_count}")
                del_updated_cols.add(q_row_count)

            sep_del = ",\n        "
            minus_sql = (
                f"UPDATE {mv_table}\n"
                f"    SET {sep_del.join(del_updates)}\n"
                f"    FROM (\n"
                f"        SELECT {', '.join(del_select_exprs)}\n"
                f"        FROM __mv_old_rows{w_old_sql}{grp_by_old_sql}{having_clause}\n"
                f"    ) AS __del\n"
                f"    WHERE {grp_match_del};"
            )

            cleanup_sql = f"DELETE FROM {mv_table} WHERE {row_count_col} <= 0;" if grp_exprs else ""

            # Auxiliary tables (COUNT DISTINCT, etc.)
            distinct_aux_sqls_plus: List[str] = []
            distinct_aux_sqls_minus: List[str] = []
            distinct_recomp_mv_sqls: List[str] = []

            for ekey, (aux_tbl, rep_item, val_dt) in distinct_aux_tables.items():
                ve_new = self._effective_agg_value_expr(rep_item, record_prefix="__mv_new_rows")
                ve_old = self._effective_agg_value_expr(rep_item, record_prefix="__mv_old_rows")

                if not grp_exprs:
                    aux_cols = ["__mv_global_id", "__val", "__ref_cnt"]
                    aux_sel_new = ["1 AS __mv_global_id", f"({ve_new}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_new = ["1", f"({ve_new})"]
                    aux_sel_old = ["1 AS __mv_global_id", f"({ve_old}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_old = ["1", f"({ve_old})"]
                    aux_match = f"{aux_tbl}.__mv_global_id = __d.__mv_global_id AND {aux_tbl}.__val = __d.__val"
                    grp_match_aux = f"{aux_tbl}.__mv_global_id = {mv_table}.__mv_global_id"
                elif has_nullable_grp:
                    aux_cols = ["__mv_pk", "__val", "__ref_cnt"]
                    aux_sel_new = [f"{pk_new_val} AS __mv_pk", f"({ve_new}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_new = [pk_new_val, f"({ve_new})"]
                    aux_sel_old = [f"{pk_old_val} AS __mv_pk", f"({ve_old}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_old = [pk_old_val, f"({ve_old})"]
                    aux_match = f"{aux_tbl}.__mv_pk = __d.__mv_pk AND {aux_tbl}.__val = __d.__val"
                    grp_match_aux = f"{aux_tbl}.__mv_pk = {mv_table}.__mv_pk"
                else:
                    g_cols = [self._group_col_alias(g) for g in grp_exprs]
                    aux_cols = [*g_cols, "__val", "__ref_cnt"]
                    aux_sel_new = [*rec_grp_new, f"({ve_new}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_new = [*rec_grp_new, f"({ve_new})"]
                    aux_sel_old = [*rec_grp_old, f"({ve_old}) AS __val", "COUNT(*) AS __ref_cnt"]
                    aux_grp_old = [*rec_grp_old, f"({ve_old})"]
                    aux_match = " AND ".join(f"{aux_tbl}.{mc} = __d.{mc}" for mc in g_cols) + f" AND {aux_tbl}.__val = __d.__val"
                    grp_match_aux = " AND ".join(f"{aux_tbl}.{mc} = {mv_table}.{mc}" for mc in g_cols)

                aux_pk_cols = [self._q(c) for c in aux_cols if c != "__ref_cnt"]

                ins_aux_sql = (
                    f"INSERT INTO {aux_tbl} ({', '.join(aux_cols)})\n"
                    f"    SELECT {', '.join(aux_sel_new)}\n"
                    f"    FROM __mv_new_rows\n"
                    f"    WHERE ({ve_new}) IS NOT NULL{w_new_and}\n"
                    f"    GROUP BY {', '.join(aux_grp_new)}\n"
                    f"    ON CONFLICT ({', '.join(aux_pk_cols)}) DO UPDATE SET\n"
                    f"        __ref_cnt = {aux_tbl}.__ref_cnt + EXCLUDED.__ref_cnt;"
                )
                distinct_aux_sqls_plus.append(ins_aux_sql)

                upd_aux_sql = (
                    f"UPDATE {aux_tbl}\n"
                    f"    SET __ref_cnt = {aux_tbl}.__ref_cnt - __d.__ref_cnt\n"
                    f"    FROM (\n"
                    f"        SELECT {', '.join(aux_sel_old)}\n"
                    f"        FROM __mv_old_rows\n"
                    f"        WHERE ({ve_old}) IS NOT NULL{w_old_and}\n"
                    f"        GROUP BY {', '.join(aux_grp_old)}\n"
                    f"    ) AS __d\n"
                    f"    WHERE {aux_match};"
                )
                del_zero_aux_sql = f"DELETE FROM {aux_tbl} WHERE __ref_cnt <= 0;"
                distinct_aux_sqls_minus.extend([upd_aux_sql, del_zero_aux_sql])

                matching_items = [
                    it for it in work_items
                    if it.is_aggregate and it.is_distinct and expr_key_fn(it) == ekey
                ]
                if not matching_items:
                    matching_items = [rep_item]

                for d_item in matching_items:
                    alias_q = self._q(d_item.alias)
                    if d_item.agg_func == AggFunc.SUM:
                        recomp_distinct_sql = (
                            f"UPDATE {mv_table}\n"
                            f"    SET {alias_q} = (SELECT SUM(__val) FROM {aux_tbl} WHERE {grp_match_aux});"
                        )
                    elif d_item.agg_func == AggFunc.AVG:
                        recomp_distinct_sql = (
                            f"UPDATE {mv_table}\n"
                            f"    SET {alias_q} = (SELECT AVG(__val) FROM {aux_tbl} WHERE {grp_match_aux});"
                        )
                    elif d_item.agg_func == AggFunc.TOTAL:
                        recomp_distinct_sql = (
                            f"UPDATE {mv_table}\n"
                            f"    SET {alias_q} = COALESCE((SELECT 1.0 * SUM(__val) FROM {aux_tbl} WHERE {grp_match_aux}), 0.0);"
                        )
                    else:
                        recomp_distinct_sql = (
                            f"UPDATE {mv_table}\n"
                            f"    SET {alias_q} = COALESCE((SELECT COUNT(*) FROM {aux_tbl} WHERE {grp_match_aux}), 0);"
                        )
                    distinct_recomp_mv_sqls.append(recomp_distinct_sql)

            # Assemble AI body
            ai_parts: List[str] = []
            if distinct_aux_sqls_plus:
                ai_parts.extend(distinct_aux_sqls_plus)
            ai_parts.append(upsert_plus_sql)
            if distinct_recomp_mv_sqls:
                ai_parts.extend(distinct_recomp_mv_sqls)
            pg_ai_body = "\n".join(ai_parts)

            # Assemble AD body
            ad_parts: List[str] = []
            if distinct_aux_sqls_minus:
                ad_parts.extend(distinct_aux_sqls_minus)
            ad_parts.append(minus_sql)
            if distinct_recomp_mv_sqls:
                ad_parts.extend(distinct_recomp_mv_sqls)
            if cleanup_sql:
                ad_parts.append(cleanup_sql)
            pg_ad_body = "\n".join(ad_parts)

            # Assemble AU body: OLD deltas -> NEW upsert -> cleanup
            au_parts: List[str] = []
            if distinct_aux_sqls_minus:
                au_parts.extend(distinct_aux_sqls_minus)
            au_parts.append(minus_sql)
            if distinct_aux_sqls_plus:
                ai_parts_for_au = list(distinct_aux_sqls_plus)
                au_parts.extend(ai_parts_for_au)
            au_parts.append(upsert_plus_sql)
            if distinct_recomp_mv_sqls:
                au_parts.extend(distinct_recomp_mv_sqls)
            if cleanup_sql:
                au_parts.append(cleanup_sql)
            pg_au_body = "\n".join(au_parts)

            return pg_ai_body, pg_ad_body, pg_au_body
        except Exception:
            return None, None, None

