"""Tier 1A: Single-table O(1) algebraic delta maintenance rules."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp

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
    _parse_single_agg_expr,
    _sanitize_ident,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_view_ast,
    quote_sql_identifier,
)


class Tier1SingleMixin(MixinHost):
    def _effective_agg_value_expr(
        self, item: SelectItem, record_prefix: Optional[str] = None
    ) -> str:
        arg_e = item.arg_expr or "1"
        if arg_e.strip() == "*":
            arg_e = "1"
        rewritten_arg = (
            self._rewrite_expr_for_record(arg_e, record_prefix)
            if (record_prefix or self._alg_mode in ("single", "distinct"))
            else arg_e
        )
        if item.filter_where:
            rewritten_f = (
                self._rewrite_expr_for_record(item.filter_where, record_prefix)
                if (record_prefix or self._alg_mode in ("single", "distinct"))
                else item.filter_where
            )
            return f"CASE WHEN ({rewritten_f}) THEN ({rewritten_arg}) ELSE NULL END"
        return rewritten_arg


    def _sql_backfill_agg_expr(self, item: SelectItem) -> str:
        eff = self._effective_agg_value_expr(item, record_prefix=None)
        dist = "DISTINCT " if item.is_distinct else ""
        if item.agg_func == AggFunc.COUNT_STAR:
            if item.filter_where:
                return f"COUNT({eff})"
            return "COUNT(*)"
        if item.agg_func == AggFunc.COUNT_COL:
            return f"COUNT({dist}{eff})"
        if item.agg_func == AggFunc.SUM:
            return f"SUM({dist}{eff})"
        if item.agg_func == AggFunc.TOTAL:
            if self.dialect == Dialect.SQLITE:
                return f"TOTAL({dist}{eff})"
            return f"COALESCE(1.0 * SUM({dist}{eff}), 0.0)"
        if item.agg_func == AggFunc.AVG:
            # Preserve the argument's SQL type: MySQL AVG(exact) returns
            # DECIMAL, PostgreSQL AVG(integer/numeric) returns NUMERIC, while
            # approximate arguments remain floating-point. Multiplying by 1.0
            # would silently convert MySQL exact AVG into DOUBLE.
            return f"AVG({dist}{eff})"
        if item.agg_func == AggFunc.MIN:
            return f"MIN({eff})"
        if item.agg_func == AggFunc.MAX:
            return f"MAX({eff})"
        if item.agg_func in (
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
        ):
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                assert item.agg_func is not None
                fn_s = item.agg_func.value
                return f"{fn_s}({eff})"
            cnt_e = f"COUNT({eff})"
            sum_e = f"SUM(1.0 * ({eff}))"
            m2_e = self._sqlite_m2_agg_sql(item, eff_override=eff)
            return self._variance_formula_sql(
                new_cnt=cnt_e,
                new_sum=sum_e,
                new_sum2=m2_e,
                is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
            )
        if item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR):
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                assert item.agg_func is not None
                if self.dialect == Dialect.MYSQL:
                    return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL ELSE {item.agg_func.value}({eff}) END)"
                return f"{item.agg_func.value}({eff})"
            if item.agg_func == AggFunc.BIT_AND:
                return self._sqlite_bit_or_tree([f"MIN(({eff}) & {1 << b})" for b in range(63)])
            if item.agg_func == AggFunc.BIT_OR:
                return self._sqlite_bit_or_tree([f"MAX(({eff}) & {1 << b})" for b in range(63)])
            return self._sqlite_bit_or_tree([f"((SUM((({eff}) >> {b}) & 1) & 1) << {b})" for b in range(63)])
        if item.agg_func == AggFunc.BOOL_AND:
            if self.dialect == Dialect.POSTGRES:
                return f"BOOL_AND({eff})"
            return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL WHEN MIN(CASE WHEN ({eff}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
        if item.agg_func == AggFunc.BOOL_OR:
            if self.dialect == Dialect.POSTGRES:
                return f"BOOL_OR({eff})"
            return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL WHEN MAX(CASE WHEN ({eff}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
        return item.raw_expr

    # ------------------------------------------------------------------
    # SQLite exact SUM/TOTAL/AVG maintenance
    # ------------------------------------------------------------------
    # SQLite's SUM/TOTAL/AVG accumulate integer inputs exactly (int64) and
    # switch to binary floating point (Kahan-Babuska-Neumaier) as soon as one
    # non-integer value is seen. An add/retract fold over REAL values is not
    # equivalent: the maintained value depends on the history of values that
    # were added and later removed (e.g. 1e16 + 1.5 - 1e16 == 2.0, not 1.5).
    # Every SQLite column can hold REAL values regardless of its declared type.
    #
    # Each non-DISTINCT SUM/TOTAL/AVG argument therefore gets a hidden
    # "approximate count" column: the number of non-NULL values in the group
    # whose storage class is not INTEGER. While it is zero and the delta values
    # are integers, the fold is exact integer arithmetic (equal to the native
    # aggregate). Otherwise the affected group is recomputed from the base
    # table with the same native aggregate as the original view.
    _SQLITE_EXACT_FLOAT_LIMIT = "4503599627370496.0"  # 2**52
    _SQLITE_DIRTY_COL = "__mv_fold_dirty"

    def _sqlite_exact_sum_enabled(self) -> bool:
        return self.dialect == Dialect.SQLITE and self._alg_mode in ("single", "distinct")

    def _sqlite_distinct_sum_recompute(self, item: SelectItem, col_alias: str, record_prefix: str) -> str:
        """Native SUM(DISTINCT) for a visible SUM column; TOTAL(DISTINCT) for a
        hidden AVG sum (REAL; native AVG(DISTINCT) never raises on overflow)."""
        eff = self._effective_agg_value_expr(item, record_prefix=None)
        is_visible_sum = item.agg_func == AggFunc.SUM and self._q(col_alias) == self._q(item.alias)
        agg = "SUM" if is_visible_sum else "TOTAL"
        return self._build_native_group_recompute(
            f"{agg}(DISTINCT {eff})", record_prefix, processed_only=True
        )

    def _sqlite_union_native_sum(self, item: SelectItem) -> bool:
        """SQLite UNION ALL aggregates: SUM/TOTAL/AVG are always recomputed over
        the union (idempotent, so the OLD and NEW steps of an UPDATE agree)."""
        return (
            self.dialect == Dialect.SQLITE
            and self._alg_mode == "union_all_agg"
            and item.is_aggregate
            and item.agg_func in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG)
        )

    def _sqlite_acnt_map(self) -> Dict[str, str]:
        if not self._sqlite_exact_sum_enabled():
            return {}
        return getattr(self, "_sqlite_acnt_cols", {})

    @staticmethod
    def _sqlite_is_approx_value_sql(val_e: str) -> str:
        return f"(CASE WHEN ({val_e}) IS NOT NULL AND typeof({val_e}) <> 'integer' THEN 1 ELSE 0 END)"

    def _watermark_pk_column(self) -> Optional[str]:
        if not (self.watermark_catalog_table and self._alg_mode in ("single", "distinct")):
            return None
        wm_pk_col: Optional[str] = "rowid" if self.dialect == Dialect.SQLITE else None
        if self.primary_schema:
            for c_def in self.primary_schema.columns.values():
                if c_def.primary_key and "INT" in c_def.data_type.upper():
                    wm_pk_col = c_def.name
                    break
        return wm_pk_col

    def _build_native_group_recompute(
        self,
        agg_sql: str,
        record_prefix: str,
        *,
        processed_only: bool,
        rows_filter: Optional[str] = None,
    ) -> str:
        """Native aggregate over the record's group, optionally restricted to the
        rows the triggers have already folded (rows inside an active backfill
        watermark range are excluded, exactly like the trigger WHERE guards)."""
        q_base_t = self._q(self.view.base_table)
        conds: List[str] = []
        if self.view.where_clause:
            conds.append(f"({self.view.where_clause})")
        for g_expr in self._effective_group_by_exprs:
            if record_prefix == self._backing_table_name():
                rhs_g = f"{record_prefix}.{self._q(self._group_col_alias(g_expr))}"
            else:
                rhs_g = self._rewrite_expr_for_record(g_expr, record_prefix)
            lhs_g = self._rewrite_expr_for_record(g_expr, q_base_t)
            is_nn = self._is_simple_col_ref(g_expr) and self._col_def(g_expr).not_null
            conds.append(self._null_safe_eq(f"({lhs_g})", f"({rhs_g})", is_nullable=not is_nn))
        if rows_filter:
            conds.append(f"({rows_filter})")
        wm_pk = self._watermark_pk_column() if processed_only else None
        if wm_pk:
            q_pk = "rowid" if wm_pk == "rowid" else self._q(wm_pk)
            conds.append(
                f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                f"AND {q_base_t}.{q_pk} > backfill_cursor_pk AND {q_base_t}.{q_pk} <= backfill_max_pk)"
            )
        where_part = f" WHERE {' AND '.join(conds)}" if conds else ""
        return f"(SELECT {agg_sql} FROM {q_base_t}{where_part})"

    def _sqlite_guarded_fold(
        self,
        mv_table: str,
        acc_col: str,
        acnt_col: str,
        fold_expr: str,
        own_val: str,
        exact_conds: List[str],
    ) -> str:
        """Fold only when the fold is provably equal to the native aggregate.

        Otherwise keep the stored value and record the condition; the caller
        marks the row dirty and a final statement recomputes the group natively
        after every fold step of the trigger has run. Deciding per step would be
        wrong for UPDATE: a recompute in the OLD step already includes the NEW
        row, so folding NEW onto it would count NEW twice.
        """
        q_acc = self._q(acc_col)
        cond = (
            f"({mv_table}.{self._q(acnt_col)} = 0 "
            f"AND typeof({own_val}) IN ('integer', 'null') "
            # Integer accumulator (visible SUM, stored without affinity): the
            # fold is exact unless int64 arithmetic overflowed, which SQLite
            # turns into REAL; native SUM raises "integer overflow" instead, so
            # an overflowing fold is recomputed natively (and raises).
            f"AND ((typeof({mv_table}.{q_acc}) IN ('integer', 'null') "
            f"AND typeof({fold_expr}) IN ('integer', 'null')) OR "
            # REAL-affinity accumulators (TOTAL, hidden AVG sums): integer
            # arithmetic in binary64 is exact only below 2**52.
            f"(typeof({mv_table}.{q_acc}) = 'real' AND "
            f"ABS({mv_table}.{q_acc}) + ABS(1.0 * COALESCE({own_val}, 0)) "
            f"< {self._SQLITE_EXACT_FLOAT_LIMIT})))"
        )
        exact_conds.append(cond)
        return f"(CASE WHEN {cond} THEN {fold_expr} ELSE {mv_table}.{q_acc} END)"

    def _sqlite_dirty_fold_assignment(self, mv_table: str, exact_conds: Sequence[str]) -> str:
        q_dirty = self._q(self._SQLITE_DIRTY_COL)
        if not exact_conds:
            return f"{q_dirty} = {mv_table}.{q_dirty}"
        uniq = list(dict.fromkeys(exact_conds))
        return f"{q_dirty} = CASE WHEN {' AND '.join(uniq)} THEN {mv_table}.{q_dirty} ELSE 1 END"

    def _sqlite_recompute_dirty_stmt(
        self,
        mv_table: str,
        record_prefix: Optional[str],
        col_sum_map: Dict[str, str],
        expr_key_fn: Callable[[SelectItem], str],
        *,
        rows_filter: Optional[str] = None,
    ) -> Optional[str]:
        """Recompute SUM/TOTAL/AVG of dirty groups natively, then clear the flag.

        Trigger form (``record_prefix`` OLD/NEW): the record's group, over the
        rows the triggers have folded (active backfill range excluded).
        Backfill-chunk form (``record_prefix`` None): every dirty group, over
        the rows selected by ``rows_filter`` (the rows folded once the chunk
        is merged)."""
        processed_only = record_prefix is not None
        acnt_map = self._sqlite_acnt_map()
        if not acnt_map:
            return None
        assignments: List[str] = []
        seen: Set[str] = set()
        for item in self._working_select_items:
            if not item.is_aggregate or item.is_distinct:
                continue
            ekey = expr_key_fn(item)
            if ekey not in acnt_map or item.agg_func not in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG):
                continue
            q_col = self._q(item.alias)
            if q_col in seen:
                continue
            seen.add(q_col)
            assignments.append(
                f"{q_col} = "
                + self._build_native_group_recompute(
                    self._sql_backfill_agg_expr(item), mv_table,
                    processed_only=processed_only, rows_filter=rows_filter,
                )
            )
        ekey_to_item = {expr_key_fn(it): it for it in self._working_select_items if it.is_aggregate}
        for ekey, sum_col in col_sum_map.items():
            q_col = self._q(sum_col)
            if ekey not in acnt_map or q_col in seen:
                continue
            seen.add(q_col)
            eff = self._effective_agg_value_expr(ekey_to_item[ekey], record_prefix=None)
            # Hidden AVG sums have REAL affinity. TOTAL equals SUM converted to
            # REAL but never raises "integer overflow" (native AVG does not).
            assignments.append(
                f"{q_col} = "
                + self._build_native_group_recompute(
                    f"TOTAL({eff})", mv_table, processed_only=processed_only, rows_filter=rows_filter
                )
            )
        q_dirty = self._q(self._SQLITE_DIRTY_COL)
        assignments.append(f"{q_dirty} = 0")
        where = f"{mv_table}.{q_dirty} = 1"
        if record_prefix is not None:
            where = f"({self._group_match_predicate_tier1(mv_table, record_prefix)}) AND {where}"
        return (
            f"UPDATE {mv_table}\n"
            f"    SET {', '.join(assignments)}\n"
            f"    WHERE {where};"
        )

    def _sqlite_chunk_rows_filter(self, pk_col: str) -> str:
        """Rows folded once the current pk-range chunk is merged.

        The chunk statements run with (cursor_pk, upper_pk) bound; ``?2`` is the
        chunk's upper bound. Rows above the claimed backfill range are folded
        by the triggers (see the trigger watermark guard)."""
        q_base_t = self._q(self.view.base_table)
        q_pk = "rowid" if pk_col == "rowid" else self._q(pk_col)
        cond = f"{q_base_t}.{q_pk} <= ?2"
        if self.watermark_catalog_table:
            cond += (
                f" OR NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                f"AND {q_base_t}.{q_pk} <= backfill_max_pk)"
            )
        return cond

    def _compile_tier1_algebraic(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs
        has_nullable_grp = self._has_nullable_group_col()

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
                c_def = self._lookup_col_def(g_expr)
                if c_def:
                    dt = self._normalize_dialect_type(c_def.data_type)
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
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]] = {}

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
            if item.is_distinct and ekey not in distinct_aux_tables:
                p = self.object_prefix if self.object_prefix else "__matview_"
                aux_tbl = self._bounded_ident(f"{p}dist_", f"{self.logical_view_name}__{_sanitize_ident(item.alias)}")
                c_arg = self._lookup_col_def(item.arg_expr)
                if c_arg:
                    val_dt = self._normalize_dialect_type(c_arg.data_type)
                else:
                    fake_val = SelectItem(
                        raw_expr=item.arg_expr or "",
                        alias="__val",
                        is_aggregate=False,
                    )
                    val_dt = self._infer_expr_storage_type(fake_val)
                if self.dialect == Dialect.MYSQL and val_dt.upper() == "TEXT":
                    val_dt = "VARCHAR(255)"
                distinct_aux_tables[ekey] = (aux_tbl, item, val_dt)

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                c_def = self._lookup_col_def(item.raw_expr)
                dt = (
                    self._normalize_dialect_type(c_def.data_type)
                    if c_def
                    else self._infer_expr_storage_type(item)
                )
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if (grp_exprs and not has_nullable_grp) else ""
                col_defs_sql.append(f"{self._q(item.alias)} {dt}{nn}")
            else:
                storage_type = self._infer_expr_storage_type(item)
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

        self._sqlite_acnt_cols = {}
        if self._sqlite_exact_sum_enabled():
            for item in work_items:
                if (
                    item.is_aggregate
                    and not item.is_distinct
                    and item.agg_func in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG)
                ):
                    ekey = _expr_key(item)
                    if ekey not in self._sqlite_acnt_cols:
                        aux_idx += 1
                        acnt_col = f"__mv_acnt_{_sanitize_ident(item.alias)}_{aux_idx}"
                        self._sqlite_acnt_cols[ekey] = acnt_col
                        col_defs_sql.append(f"{acnt_col} {self._int_type()} NOT NULL DEFAULT 0")
            if self._sqlite_acnt_cols:
                col_defs_sql.append(f"{self._SQLITE_DIRTY_COL} {self._int_type()} NOT NULL DEFAULT 0")

        pk_cols = self._tier1_pk_cols()
        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
        )

        aux_ddls: List[str] = []
        aux_names: List[str] = []
        aux_index_specs: List[Tuple[str, str, List[str], bool]] = []
        for ekey, (aux_tbl, rep_item, val_dt) in distinct_aux_tables.items():
            aux_names.append(aux_tbl)
            aux_col_defs: List[str] = []
            if not grp_exprs:
                aux_col_defs.append("__mv_global_id INTEGER NOT NULL")
                aux_pk = ["__mv_global_id", "__val"]
            elif has_nullable_grp:
                aux_col_defs.append(f"__mv_pk {pk_type} NOT NULL")
                aux_pk = ["__mv_pk", "__val"]
            else:
                aux_pk = []
                for g_expr in grp_exprs:
                    mc = self._group_col_alias(g_expr)
                    c_def = self._lookup_col_def(g_expr)
                    if c_def:
                        dt = self._normalize_dialect_type(c_def.data_type)
                    else:
                        fake_grp = SelectItem(raw_expr=g_expr, alias=mc, is_aggregate=False)
                        dt = self._infer_expr_storage_type(fake_grp)
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                        dt = "VARCHAR(255)"
                    aux_col_defs.append(f"{self._q(mc)} {dt} NOT NULL")
                    aux_pk.append(self._q(mc))
                aux_pk.append(self._q("__val"))
            aux_col_defs.append(f"{self._q('__val')} {val_dt} NOT NULL")
            aux_col_defs.append(f"{self._q('__ref_cnt')} {self._int_type()} NOT NULL DEFAULT 0")
            aux_col_defs.append(f"PRIMARY KEY ({', '.join(aux_pk)})")
            aux_ddls.append(
                self._create_table_ddl(aux_tbl, aux_col_defs, is_backing_store=False, pk_cols=aux_pk)
            )
            aux_index_specs.append((aux_tbl, rep_item.alias, list(aux_pk), bool(grp_exprs and not has_nullable_grp)))

        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col and "__mv_elem_" in expr_s
        }
        can_pg15_unique_nnd = (
            bool(grp_exprs)
            and has_nullable_grp
            and self._alg_mode in ("single", "distinct")
            and len(set(c.lower() for c in group_mv_cols)) == len(grp_exprs)
            and all(
                (cd := self._lookup_col_def(g)) is not None
                and any(tk in cd.data_type.upper() for tk in ("INT", "CHAR", "TEXT"))
                for g in grp_exprs
            )
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
            unique_group_nulls_not_distinct=can_pg15_unique_nnd,
        )
        for aux_tbl, rep_alias, aux_pk_cols, has_nonnull_grp in aux_index_specs:
            a_ddls, a_names = self._build_aux_dist_index_ddls(
                aux_tbl=aux_tbl,
                item_alias=rep_alias,
                aux_pk_cols=aux_pk_cols,
                has_group_cols=has_nonnull_grp,
            )
            index_ddls.extend(a_ddls)
            index_names.extend(a_names)
        if self._sqlite_acnt_map():
            # Backfill chunks recompute every dirty group; a partial index keeps
            # that lookup proportional to the dirty groups, not the store size.
            dirty_idx = self._bounded_ident(
                f"{self.object_prefix or '__matview_'}idx_", self.logical_view_name, "__dirty"
            )
            index_ddls.append(
                f"CREATE INDEX {self._q(dirty_idx)} ON {mv_table} ({self._SQLITE_DIRTY_COL}) "
                f"WHERE {self._SQLITE_DIRTY_COL} = 1;"
            )
            index_names.append(dirty_idx)

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

        initial_populate_sqls, inc_spec = self._build_initial_populate_and_inc_spec_tier1(
            mv_table=mv_table,
            count_star_alias=count_star_alias,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            distinct_aux_tables=distinct_aux_tables,
            expr_key_fn=_expr_key,
        )

        trigger_ddls, trig_names, fn_names = self._build_tier1_triggers(
            view_name=view_name,
            mv_table=mv_table,
            base_t=self.view.base_table,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            distinct_aux_tables=distinct_aux_tables,
            expr_key_fn=_expr_key,
            initial_populate_sqls=initial_populate_sqls,
        )

        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=aux_ddls,
            index_ddls=index_ddls,
            initial_populate_sqls=initial_populate_sqls,
            trigger_ddls=trigger_ddls,
            readonly_trigger_ddls=ro_trigs,
            user_columns=user_columns,
            reader_view_ddl=reader_view_ddl,
            incremental_backfill_spec=inc_spec,
            auxiliary_table_names=aux_names,
            trigger_names=ro_trig_names + trig_names,
            function_names=ro_fn_names + fn_names,
            limitations=self.limitations,
            object_prefix=self.object_prefix,
            db_view_name=self.db_view_name,
            index_names=index_names,
            user_indexes=list(self.user_indexes),
        )

    def _append_aux_stat_target_cols(
        self,
        target_cols: List[str],
        select_exprs: List[str],
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        ekey_to_item: Dict[str, SelectItem],
        *,
        qualify_join: bool = False,
        include_distinct_keyword: bool = False,
    ) -> List[str]:
        """Append missing auxiliary stat columns and return quoted target column names."""
        # Compare quoted forms so visible columns stored pre-quoted in stat maps
        # (e.g. reserved keywords like "order") match unquoted entries in target_cols.
        already_in_target_q = {self._q(c) for c in target_cols}

        def _eff(it: SelectItem) -> str:
            base = self._effective_agg_value_expr(it, record_prefix=None)
            return self._qualify_join_expr(base) if qualify_join else base

        for ekey, cnt_col in col_nonnull_count_map.items():
            q_cnt = self._q(cnt_col)
            if q_cnt not in already_in_target_q:
                it = ekey_to_item[ekey]
                dist = "DISTINCT " if (include_distinct_keyword and it.is_distinct) else ""
                target_cols.append(cnt_col)
                select_exprs.append(f"COUNT({dist}{_eff(it)})")
                already_in_target_q.add(q_cnt)

        for ekey, sum_col in col_sum_map.items():
            q_sum = self._q(sum_col)
            if q_sum not in already_in_target_q:
                it = ekey_to_item[ekey]
                dist = "DISTINCT " if (include_distinct_keyword and it.is_distinct) else ""
                target_cols.append(sum_col)
                # Hidden sums (AVG/VAR helpers) are REAL on SQLite. TOTAL is the
                # same value as SUM but never raises "integer overflow", which
                # native AVG/VAR do not raise either.
                hidden_agg = "TOTAL" if self.dialect == Dialect.SQLITE else "SUM"
                select_exprs.append(f"{hidden_agg}({dist}{_eff(it)})")
                already_in_target_q.add(q_sum)

        for ekey, sum2_col in col_sum2_map.items():
            q_sum2 = self._q(sum2_col)
            if q_sum2 not in already_in_target_q:
                it = ekey_to_item[ekey]
                eff = _eff(it)
                target_cols.append(sum2_col)
                if self.dialect == Dialect.SQLITE:
                    select_exprs.append(self._sqlite_m2_agg_sql(it, eff_override=eff))
                else:
                    select_exprs.append(f"SUM({self._var_sq_sql(eff)})")
                already_in_target_q.add(q_sum2)

        for ekey, tcnt_col in col_true_cnt_map.items():
            q_tcnt = self._q(tcnt_col)
            if q_tcnt not in already_in_target_q:
                it = ekey_to_item[ekey]
                eff = _eff(it)
                target_cols.append(tcnt_col)
                select_exprs.append(
                    f"COALESCE(SUM(CASE WHEN ({eff}) IS NOT NULL AND ({eff}) THEN 1 ELSE 0 END), 0)"
                )
                already_in_target_q.add(q_tcnt)

        for ekey, acnt_col in self._sqlite_acnt_map().items():
            q_acnt = self._q(acnt_col)
            if q_acnt not in already_in_target_q:
                eff = _eff(ekey_to_item[ekey])
                target_cols.append(acnt_col)
                select_exprs.append(
                    f"COALESCE(SUM({self._sqlite_is_approx_value_sql(eff)}), 0)"
                )
                already_in_target_q.add(q_acnt)

        return [self._q(c) for c in target_cols]

    def _build_initial_populate_and_inc_spec_tier1(
        self,
        mv_table: str,
        count_star_alias: Optional[str],
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
    ) -> Tuple[List[str], IncrementalBackfillSpec]:
        stmts: List[str] = []
        has_nullable_grp = self._has_nullable_group_col()
        q_base_t = self._q(self.view.base_table)
        pk_cols = self._tier1_pk_cols()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            eff = self._effective_agg_value_expr(rep_item, record_prefix=None)
            dist_cols: List[str] = []
            dist_sel: List[str] = []
            dist_grp: List[str] = []
            if not grp_exprs:
                dist_cols.append("__mv_global_id")
                dist_sel.append("1")
            elif has_nullable_grp:
                dist_cols.append("__mv_pk")
                dist_sel.append(self._canonical_group_pk_expr(grp_exprs))
                dist_grp.extend(self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=True))
            else:
                for g_expr in grp_exprs:
                    dist_cols.append(self._group_col_alias(g_expr))
                    dist_sel.append(g_expr)
                    dist_grp.append(g_expr)
            dist_cols.extend(["__val", "__ref_cnt"])
            dist_sel.extend([eff, "COUNT(*)"])
            dist_grp.append(eff)

            w_parts: List[str] = []
            if self.view.where_clause:
                w_parts.append(f"({self.view.where_clause})")
            w_parts.append(f"({eff}) IS NOT NULL")
            where_sql = f"\nWHERE {' AND '.join(w_parts)}"
            group_sql = f"\nGROUP BY {', '.join(dist_grp)}"
            stmts.append(
                f"INSERT INTO {aux_tbl} ({', '.join(self._q(c) for c in dist_cols)})\n"
                f"SELECT {', '.join(dist_sel)}\n"
                f"FROM {q_base_t}{where_sql}{group_sql};"
            )

        target_cols: List[str] = []
        select_exprs: List[str] = []

        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(grp_exprs))

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
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
                select_exprs.append(self._rewrite_expr_for_record(item.raw_expr, None))
            else:
                select_exprs.append(self._sql_backfill_agg_expr(item))

        if count_star_alias is None:
            target_cols.append(row_count_col)
            select_exprs.append("COUNT(*)")

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
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
            include_distinct_keyword=True,
        )

        norm_where = self._rewrite_expr_for_record(self.view.where_clause, None) if self.view.where_clause else None
        where_sql = f"\nWHERE {norm_where}" if norm_where else ""
        if grp_exprs:
            norm_grps = [self._rewrite_expr_for_record(g, None) for g in grp_exprs]
            grp_for_q = self._group_by_exprs_for_query(norm_grps, include_canonical_pk=has_nullable_grp)
            group_sql = f"\nGROUP BY {', '.join(grp_for_q)}"
            having_sql = ""
        else:
            group_sql = ""
            having_sql = ""

        q_base_t = self._q(self.view.base_table)
        main_backfill_sql = (
            f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM {q_base_t}{where_sql}{group_sql}{having_sql};"
        )
        stmts.append(main_backfill_sql)

        holistic_funcs = {
            AggFunc.GROUP_CONCAT,
            AggFunc.STRING_AGG,
            AggFunc.JSON_GROUP_ARRAY,
            AggFunc.JSON_GROUP_OBJECT,
            AggFunc.ARRAY_AGG,
        }
        has_holistic_or_distinct = bool(distinct_aux_tables) or any(
            it.is_aggregate and it.agg_func in holistic_funcs for it in work_items
        )

        has_int_pk = bool(
            self.primary_schema
            and any(c.primary_key and "INT" in c.data_type.upper() for c in self.primary_schema.columns.values())
        )
        if not has_holistic_or_distinct and (self.dialect == Dialect.SQLITE or has_int_pk):
            pk_col = "rowid"
            if self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    if c_def.primary_key and "INT" in c_def.data_type.upper():
                        pk_col = c_def.name
                        break
            ph = "?" if self.dialect == Dialect.SQLITE else "%s"
            q_pk_col = "rowid" if pk_col == "rowid" else self._q(pk_col)
            chunk_where = (
                f"({self.view.where_clause}) AND ({q_pk_col} > {ph} AND {q_pk_col} <= {ph})"
                if self.view.where_clause
                else f"({q_pk_col} > {ph} AND {q_pk_col} <= {ph})"
            )
            is_pg_merge = (self.dialect == Dialect.POSTGRES and self.engine_version.supports_pg_merge)
            is_mysql_alias = (self.dialect == Dialect.MYSQL and self.engine_version.supports_mysql_row_alias_upsert)
            incoming_alias = "__mv_src" if (is_pg_merge or is_mysql_alias) else None
            merge_assignments = self._build_tier1_chunk_merge_assignments(
                mv_table=mv_table,
                row_count_col=row_count_col,
                col_nonnull_count_map=col_nonnull_count_map,
                col_sum_map=col_sum_map,
                col_sum2_map=col_sum2_map,
                col_true_cnt_map=col_true_cnt_map,
                expr_key_fn=expr_key_fn,
                use_mysql_values=(self.dialect == Dialect.MYSQL and not is_mysql_alias),
                incoming_alias=incoming_alias,
            )
            q_pk_cols = [self._q(c) for c in pk_cols]
            aliased_select_exprs = [
                f"{expr} AS {col}" for expr, col in zip(select_exprs, q_target_cols)
            ]
            if self.dialect == Dialect.MYSQL:
                if is_mysql_alias:
                    chunk_upsert_sql = (
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"SELECT * FROM (\n"
                        f"    SELECT {', '.join(aliased_select_exprs)}\n"
                        f"    FROM {q_base_t}\n"
                        f"    WHERE {chunk_where}{group_sql}{having_sql}\n"
                        f") AS __mv_src\n"
                        f"ON DUPLICATE KEY UPDATE\n"
                        f"    {', '.join(merge_assignments)};"
                    )
                else:
                    chunk_upsert_sql = (
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"SELECT {', '.join(select_exprs)}\n"
                        f"FROM {q_base_t}\n"
                        f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                        f"ON DUPLICATE KEY UPDATE\n"
                        f"    {', '.join(merge_assignments)};"
                    )
            elif is_pg_merge:
                chunk_upsert_sql = (
                    f"MERGE INTO {mv_table}\n"
                    f"USING (\n"
                    f"    SELECT {', '.join(aliased_select_exprs)}\n"
                    f"    FROM {q_base_t}\n"
                    f"    WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f") AS __mv_src\n"
                    f"ON ({' AND '.join(f'{mv_table}.{c} = __mv_src.{c}' for c in q_pk_cols)})\n"
                    f"WHEN MATCHED THEN\n"
                    f"    UPDATE SET\n"
                    f"        {', '.join(merge_assignments)}\n"
                    f"WHEN NOT MATCHED THEN\n"
                    f"    INSERT ({', '.join(q_target_cols)})\n"
                    f"    VALUES ({', '.join(f'__mv_src.{c}' for c in q_target_cols)});"
                )
            else:
                chunk_upsert_sql = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {q_base_t}\n"
                    f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f"ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
                    f"    {', '.join(merge_assignments)};"
                )
            chunk_templates_pk = [chunk_upsert_sql]
            dirty_chunk_stmt = self._sqlite_recompute_dirty_stmt(
                mv_table, None, col_sum_map, expr_key_fn,
                rows_filter=self._sqlite_chunk_rows_filter(pk_col),
            )
            if dirty_chunk_stmt:
                chunk_templates_pk.append(dirty_chunk_stmt)
            inc_spec = IncrementalBackfillSpec(
                mode="pk_range_fold",
                pk_column=pk_col,
                chunk_sql_templates=chunk_templates_pk,
            )
        elif grp_exprs:
            grp_expr = self._canonical_group_pk_expr(grp_exprs)
            w_grp = f" WHERE {self.view.where_clause}" if self.view.where_clause else ""
            dist_grp_query = (
                f"SELECT DISTINCT {grp_expr} AS __gkey FROM {q_base_t}{w_grp} ORDER BY __gkey"
            )
            chunk_templates: List[str] = []
            for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
                eff = self._effective_agg_value_expr(rep_item, record_prefix=None)
                dist_cols = [
                    *pk_cols,
                    "__val",
                    "__ref_cnt",
                ]
                dist_sel = []
                dist_grp_list = []
                if has_nullable_grp:
                    dist_sel.append(grp_expr)
                    dist_grp_list.extend(self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=True))
                else:
                    for g_expr in grp_exprs:
                        dist_sel.append(g_expr)
                        dist_grp_list.append(g_expr)
                dist_sel.extend([eff, "COUNT(*)"])
                dist_grp_list.append(eff)
                dist_cols_flat = [*pk_cols, "__val", "__ref_cnt"]
                q_dist_cols = [self._q(c) for c in dist_cols]
                q_pk_cols = [self._q(c) for c in pk_cols]
                q_dist_cols_flat = [self._q(c) for c in dist_cols_flat]
                dist_aliased_sel = [
                    f"{expr} AS {col}" for expr, col in zip(dist_sel, q_dist_cols_flat)
                ]
                w_p = [f"({eff}) IS NOT NULL", f"({grp_expr}) IN ({{placeholders}})"]
                if self.view.where_clause:
                    w_p.insert(0, f"({self.view.where_clause})")
                if self.dialect == Dialect.MYSQL:
                    if self.engine_version.supports_mysql_row_alias_upsert:
                        chunk_templates.append(
                            f"INSERT INTO {aux_tbl} ({', '.join(q_dist_cols)})\n"
                            f"SELECT * FROM (\n"
                            f"    SELECT {', '.join(dist_aliased_sel)}\n"
                            f"    FROM {q_base_t}\n"
                            f"    WHERE {' AND '.join(w_p)}\n"
                            f"    GROUP BY {', '.join(dist_grp_list)}\n"
                            f") AS __mv_src\n"
                            f"ON DUPLICATE KEY UPDATE {self._q('__ref_cnt')} = __mv_src.{self._q('__ref_cnt')};"
                        )
                    else:
                        chunk_templates.append(
                            f"INSERT INTO {aux_tbl} ({', '.join(q_dist_cols)})\n"
                            f"SELECT {', '.join(dist_sel)}\n"
                            f"FROM {q_base_t}\n"
                            f"WHERE {' AND '.join(w_p)}\n"
                            f"GROUP BY {', '.join(dist_grp_list)}\n"
                            f"ON DUPLICATE KEY UPDATE {self._q('__ref_cnt')} = VALUES({self._q('__ref_cnt')});"
                        )
                elif self.dialect == Dialect.POSTGRES and self.engine_version.supports_pg_merge:
                    chunk_templates.append(
                        f"MERGE INTO {aux_tbl}\n"
                        f"USING (\n"
                        f"    SELECT {', '.join(dist_aliased_sel)}\n"
                        f"    FROM {q_base_t}\n"
                        f"    WHERE {' AND '.join(w_p)}\n"
                        f"    GROUP BY {', '.join(dist_grp_list)}\n"
                        f") AS __mv_src\n"
                        f"ON ({' AND '.join(f'{aux_tbl}.{c} = __mv_src.{c}' for c in q_pk_cols)} AND {aux_tbl}.__val = __mv_src.__val)\n"
                        f"WHEN MATCHED THEN\n"
                        f"    UPDATE SET __ref_cnt = __mv_src.__ref_cnt\n"
                        f"WHEN NOT MATCHED THEN\n"
                        f"    INSERT ({', '.join(q_dist_cols)})\n"
                        f"    VALUES ({', '.join(f'__mv_src.{c}' for c in q_dist_cols_flat)});"
                    )
                else:
                    q_aux_conflict_cols = [self._q(c) for c in [*pk_cols, "__val"]]
                    aux_conflict = f"\nON CONFLICT ({', '.join(q_aux_conflict_cols)}) DO UPDATE SET {self._q('__ref_cnt')} = excluded.{self._q('__ref_cnt')}"
                    chunk_templates.append(
                        f"INSERT INTO {aux_tbl} ({', '.join(q_dist_cols)})\n"
                        f"SELECT {', '.join(dist_sel)}\n"
                        f"FROM {q_base_t}\n"
                        f"WHERE {' AND '.join(w_p)}\n"
                        f"GROUP BY {', '.join(dist_grp_list)}{aux_conflict};"
                    )
            w_main = [f"({grp_expr}) IN ({{placeholders}})"]
            if self.view.where_clause:
                w_main.insert(0, f"({self.view.where_clause})")
            q_pk_cols = [self._q(c) for c in pk_cols]
            q_pk_set = set(q_pk_cols)
            non_pk_target = [c for c in q_target_cols if c not in q_pk_set] or list(q_pk_cols)
            aliased_main_sel = [f"{expr} AS {col}" for expr, col in zip(select_exprs, q_target_cols)]
            if self.dialect == Dialect.MYSQL:
                if self.engine_version.supports_mysql_row_alias_upsert:
                    main_conflict = f"\nON DUPLICATE KEY UPDATE {', '.join(f'{c} = __mv_src.{c}' for c in non_pk_target)}"
                    chunk_templates.append(
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"SELECT * FROM (\n"
                        f"    SELECT {', '.join(aliased_main_sel)}\n"
                        f"    FROM {q_base_t}\n"
                        f"    WHERE {' AND '.join(w_main)}{group_sql}{having_sql}\n"
                        f") AS __mv_src{main_conflict};"
                    )
                else:
                    main_conflict = f"\nON DUPLICATE KEY UPDATE {', '.join(f'{c} = VALUES({c})' for c in non_pk_target)}"
                    chunk_templates.append(
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"SELECT {', '.join(select_exprs)}\n"
                        f"FROM {q_base_t}\n"
                        f"WHERE {' AND '.join(w_main)}{group_sql}{having_sql}{main_conflict};"
                    )
            elif self.dialect == Dialect.POSTGRES and self.engine_version.supports_pg_merge:
                chunk_templates.append(
                    f"MERGE INTO {mv_table}\n"
                    f"USING (\n"
                    f"    SELECT {', '.join(aliased_main_sel)}\n"
                    f"    FROM {q_base_t}\n"
                    f"    WHERE {' AND '.join(w_main)}{group_sql}{having_sql}\n"
                    f") AS __mv_src\n"
                    f"ON ({' AND '.join(f'{mv_table}.{c} = __mv_src.{c}' for c in q_pk_cols)})\n"
                    f"WHEN MATCHED THEN\n"
                    f"    UPDATE SET {', '.join(f'{c} = __mv_src.{c}' for c in non_pk_target)}\n"
                    f"WHEN NOT MATCHED THEN\n"
                    f"    INSERT ({', '.join(q_target_cols)})\n"
                    f"    VALUES ({', '.join(f'__mv_src.{c}' for c in q_target_cols)});"
                )
            else:
                main_conflict = f"\nON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in non_pk_target)}"
                chunk_templates.append(
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {q_base_t}\n"
                    f"WHERE {' AND '.join(w_main)}{group_sql}{having_sql}{main_conflict};"
                )
            inc_spec = IncrementalBackfillSpec(
                mode="group_key_batch",
                distinct_group_query=dist_grp_query,
                chunk_sql_templates=chunk_templates,
            )
        else:
            inc_spec = IncrementalBackfillSpec(
                mode="offset_batch",
                chunk_sql_templates=stmts,
            )

        return stmts, inc_spec

    def _build_tier1_chunk_merge_assignments(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        expr_key_fn: Callable[[SelectItem], str],
        use_mysql_values: bool = False,
        incoming_alias: Optional[str] = None,
    ) -> List[str]:
        assignments: List[str] = []
        updated: Set[str] = set()
        work_items = self._working_select_items
        ekey_to_item_m: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }

        def _incoming(col: str) -> str:
            qc = self._q(col)
            if incoming_alias:
                return f"{incoming_alias}.{qc}"
            return f"VALUES({qc})" if use_mysql_values else f"excluded.{qc}"

        acnt_map = self._sqlite_acnt_map()
        col_to_ekey = {self._q(c): k for k, c in col_sum_map.items()}
        merge_fold_conds: List[str] = []

        def _merged_sum(col: str, item: Optional[SelectItem] = None, *, total: bool = False) -> str:
            qc = self._q(col)
            inc = _incoming(qc)
            folded = (
                f"(CASE WHEN {inc} IS NULL THEN {mv_table}.{qc} "
                f"WHEN {mv_table}.{qc} IS NULL THEN {inc} "
                f"ELSE {mv_table}.{qc} + {inc} END)"
            )
            ekey = expr_key_fn(item) if item is not None else col_to_ekey.get(qc)
            if ekey is None or ekey not in acnt_map:
                return folded
            if item is None:
                item = ekey_to_item_m[ekey]
            # Exact only while both sides are integer-only and below 2**52 when
            # stored with REAL affinity; otherwise recompute the whole group
            # natively (idempotent, so concurrent trigger recomputes and later
            # chunks converge to the native aggregate when the backfill ends).
            q_acnt = self._q(acnt_map[ekey])
            cond = (
                f"({mv_table}.{q_acnt} = 0 AND {_incoming(q_acnt)} = 0 AND "
                f"((typeof({mv_table}.{qc}) IN ('integer', 'null') AND typeof({inc}) IN ('integer', 'null') "
                f"AND typeof({folded}) IN ('integer', 'null')) OR "
                f"(typeof({mv_table}.{qc}) = 'real' AND "
                f"ABS(1.0 * COALESCE({mv_table}.{qc}, 0)) + ABS(1.0 * COALESCE({inc}, 0)) < {self._SQLITE_EXACT_FLOAT_LIMIT})))"
            )
            # Fold only when provably exact; otherwise keep the row and mark it
            # dirty. The chunk's follow-up statement recomputes dirty groups
            # over exactly the rows folded so far (never rows of later chunks,
            # which would be folded again when their chunk is merged).
            merge_fold_conds.append(cond)
            return f"(CASE WHEN {cond} THEN {folded} ELSE {mv_table}.{qc} END)"

        def _merged_sum2(ekey: str, sum2_col: str) -> str:
            q_sum2 = self._q(sum2_col)
            if self.dialect != Dialect.SQLITE:
                return _merged_sum(q_sum2)
            cnt_c = self._q(col_nonnull_count_map[ekey])
            sum_c = self._q(col_sum_map[ekey])
            inc_cnt = _incoming(cnt_c)
            inc_sum = _incoming(sum_c)
            inc_m2 = _incoming(q_sum2)
            d_mu = f"(((1.0 * {inc_sum}) / {inc_cnt}) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_c}))"
            return (
                f"(CASE WHEN {inc_cnt} <= 0 OR {inc_m2} IS NULL THEN {mv_table}.{q_sum2} "
                f"WHEN {mv_table}.{cnt_c} <= 0 OR {mv_table}.{q_sum2} IS NULL THEN {inc_m2} "
                f"ELSE {mv_table}.{q_sum2} + {inc_m2} + ({d_mu} * {d_mu} * ((1.0 * {mv_table}.{cnt_c} * {inc_cnt}) / ({mv_table}.{cnt_c} + {inc_cnt}))) END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = self._q(item.alias)
            updated.add(c)
            ekey = expr_key_fn(item)
            inc_c = _incoming(c)
            if item.agg_func == AggFunc.TOTAL and ekey in acnt_map:
                assignments.append(f"{c} = {_merged_sum(c, item, total=True)}")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                assignments.append(f"{c} = {mv_table}.{c} + {inc_c}")
            elif item.agg_func == AggFunc.SUM:
                assignments.append(f"{c} = {_merged_sum(c, item)}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    # The native AVG result family is not reproduced by a
                    # SUM/COUNT fold for integer inputs (PostgreSQL integer
                    # division truncates) or for engine-specific DECIMAL scale.
                    recomputed = self._build_recompute_subquery_raw(item, mv_table)
                    assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = self._q(col_nonnull_count_map[ekey])
                    sum_c = self._q(col_sum_map[ekey])
                    new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                    new_sum = _merged_sum(sum_c, item)
                    derived = self._avg_from_sum_count_sql(item, new_sum, new_cnt)
                    assignments.append(
                        f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL ELSE {derived} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                native_stat_input = self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
                if native_stat_input or self._sqlite_exact_sum_enabled():
                    recomputed = self._build_recompute_subquery_raw(item, mv_table)
                    assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = self._q(col_nonnull_count_map[ekey])
                    sum_c = self._q(col_sum_map[ekey])
                    sum2_c = self._q(col_sum2_map[ekey])
                    new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                    new_sum = _merged_sum(sum_c)
                    new_sum2 = _merged_sum2(ekey, sum2_c)
                    formula = self._variance_formula_sql(
                        new_cnt=new_cnt,
                        new_sum=new_sum,
                        new_sum2=new_sum2,
                        is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                        is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                    )
                    assignments.append(f"{c} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} < {mv_table}.{c} "
                    f"THEN {inc_c} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} > {mv_table}.{c} "
                    f"THEN {inc_c} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.BIT_AND:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE ({mv_table}.{c} & {inc_c}) END"
                )
            elif item.agg_func == AggFunc.BIT_OR:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE ({mv_table}.{c} | {inc_c}) END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                xor_e = self._bitwise_xor_sql(f"{mv_table}.{c}", f"({inc_c})")
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE {xor_e} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_c = self._q(col_nonnull_count_map[ekey])
                tcnt_c = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_tcnt = f"({mv_table}.{tcnt_c} + {_incoming(tcnt_c)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                assignments.append(
                    f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_c = self._q(col_nonnull_count_map[ekey])
                tcnt_c = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_tcnt = f"({mv_table}.{tcnt_c} + {_incoming(tcnt_c)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                assignments.append(
                    f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )

        q_row_count_col = self._q(row_count_col)
        if q_row_count_col not in updated:
            assignments.append(f"{q_row_count_col} = {mv_table}.{q_row_count_col} + {_incoming(q_row_count_col)}")
            updated.add(q_row_count_col)

        for cnt_c in col_nonnull_count_map.values():
            q_cnt_c = self._q(cnt_c)
            if q_cnt_c not in updated:
                assignments.append(f"{q_cnt_c} = {mv_table}.{q_cnt_c} + {_incoming(q_cnt_c)}")
                updated.add(q_cnt_c)
        for sum_c in col_sum_map.values():
            q_sum_c = self._q(sum_c)
            if q_sum_c not in updated:
                assignments.append(f"{q_sum_c} = {_merged_sum(q_sum_c)}")
                updated.add(q_sum_c)
        for ekey, sum2_c in col_sum2_map.items():
            q_sum2_c = self._q(sum2_c)
            if q_sum2_c not in updated:
                assignments.append(f"{q_sum2_c} = {_merged_sum2(ekey, q_sum2_c)}")
                updated.add(q_sum2_c)
        for tcnt_c in col_true_cnt_map.values():
            q_tcnt_c = self._q(tcnt_c)
            if q_tcnt_c not in updated:
                assignments.append(f"{q_tcnt_c} = {mv_table}.{q_tcnt_c} + {_incoming(q_tcnt_c)}")
                updated.add(q_tcnt_c)
        for acnt_c in acnt_map.values():
            q_acnt_c = self._q(acnt_c)
            if q_acnt_c not in updated:
                assignments.append(f"{q_acnt_c} = {mv_table}.{q_acnt_c} + {_incoming(q_acnt_c)}")
                updated.add(q_acnt_c)
        if acnt_map:
            assignments.append(self._sqlite_dirty_fold_assignment(mv_table, merge_fold_conds))

        derived_aliases = {
            self._q(it.alias)
            for it in work_items
            if it.is_aggregate
            and it.agg_func
            in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            )
        }
        return [
            a for a in assignments if a.split("=")[0].strip() in derived_aliases
        ] + [
            a for a in assignments if a.split("=")[0].strip() not in derived_aliases
        ]

    def _build_recompute_subquery_raw(self, item: SelectItem, record_prefix: str) -> str:
        def _group_record_expr(group_expr: str) -> str:
            # Chunk-fold recomputation reads the current aggregate row from the
            # backing table. A projected group expression may have a different
            # stored alias (for example COALESCE(sub_cat, 'UNK') AS sc_norm),
            # so correlate against that stored column rather than the original
            # base-table column, which does not exist in the backing table.
            if record_prefix == self._backing_table_name():
                return f"{record_prefix}.{self._q(self._group_col_alias(group_expr))}"
            return self._rewrite_expr_for_record(group_expr, record_prefix)

        if getattr(self, "_alg_mode", None) == "union_all_agg":
            inner_u = getattr(self, "_union_all_inner_sql", "")
            u_alias = getattr(self, "_union_all_sub_alias", "__mv_u")
            orig_item = getattr(self, "_uall_orig_item_map", {}).get(id(item), item)
            orig_grps = getattr(self, "_uall_orig_grp_exprs", self._effective_group_by_exprs)
            u_conds: List[str] = []
            if self.view.where_clause:
                u_conds.append(f"({self.view.where_clause})")
            for orig_g, branch_g in zip(orig_grps, self._effective_group_by_exprs):
                rhs_g = _group_record_expr(branch_g)
                u_conds.append(self._null_safe_eq(f"({orig_g})", f"({rhs_g})", is_nullable=True))
            w_u = f" WHERE {' AND '.join(u_conds)}" if u_conds else ""
            u_agg_sql = self._sql_backfill_agg_expr(orig_item)
            return f"(SELECT {u_agg_sql} FROM ({inner_u}) AS {u_alias}{w_u})"
        base_t = self.view.base_table
        q_base_t = self._q(base_t)
        conds: List[str] = []
        if self.view.where_clause:
            conds.append(f"({self.view.where_clause})")
        for g_expr in self._effective_group_by_exprs:
            lhs_g = self._rewrite_expr_for_record(g_expr, q_base_t)
            rhs_g = _group_record_expr(g_expr)
            is_nn = self._is_simple_col_ref(g_expr) and self._col_def(g_expr).not_null
            conds.append(
                self._null_safe_eq(
                    f"({lhs_g})",
                    f"({rhs_g})",
                    is_nullable=not is_nn,
                )
            )
        where_part = f" WHERE {' AND '.join(conds)}" if conds else ""
        agg_sql = self._sql_backfill_agg_expr(item)
        return f"(SELECT {agg_sql} FROM {q_base_t}{where_part})"

    def _variance_formula_sql(
        self,
        new_cnt: str,
        new_sum: str,
        new_sum2: str,
        is_sample: bool,
        is_stddev: bool,
    ) -> str:
        denom = f"({new_cnt} - 1)" if is_sample else new_cnt
        min_n = 2 if is_sample else 1
        safe_cnt = f"NULLIF({new_cnt}, 0)"
        safe_denom = f"NULLIF({denom}, 0)"
        if self.dialect == Dialect.SQLITE:
            raw_var = f"((1.0 * ({new_sum2})) / {safe_denom})"
        elif self.dialect == Dialect.POSTGRES:
            raw_var = (
                f"(((1.0 * ({new_sum2})::NUMERIC) - ((1.0 * ({new_sum})::NUMERIC) * (1.0 * ({new_sum})::NUMERIC)) / {safe_cnt}) / {safe_denom})::DOUBLE PRECISION"
            )
        else:
            raw_var = f"((1.0 * {new_sum2}) - ((1.0 * {new_sum}) * (1.0 * {new_sum})) / {safe_cnt}) / {safe_denom}"
        clamped_var = f"(CASE WHEN ({raw_var}) < 0.0 THEN 0.0 ELSE ({raw_var}) END)"
        val_expr = f"SQRT({clamped_var})" if is_stddev else clamped_var
        return f"(CASE WHEN {new_cnt} < {min_n} THEN NULL ELSE {val_expr} END)"

    def _group_match_predicate_tier1(self, target_table: str, record_prefix: str) -> str:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return f"{target_table}.__mv_global_id = 1"
        if self._has_nullable_group_col():
            rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
            return f"{target_table}.__mv_pk = ({self._canonical_group_pk_expr(rec_exprs)})"
        preds: List[str] = []
        for g_expr in grp_exprs:
            mv_col = self._q(self._group_col_alias(g_expr))
            rec_e = self._rewrite_expr_for_record(g_expr, record_prefix)
            preds.append(f"{target_table}.{mv_col} = {rec_e}")
        return " AND ".join(preds)

    def _distinct_match_predicate(self, aux_tbl: str, val_expr: str, record_prefix: str) -> str:
        grp_pred = self._group_match_predicate_tier1(aux_tbl, record_prefix)
        return f"{grp_pred} AND {aux_tbl}.{self._q('__val')} = ({val_expr})"

    def _build_plus_delta_statements(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        record_prefix: str = "NEW",
    ) -> List[str]:
        stmts: List[str] = []
        acnt_map = self._sqlite_acnt_map()
        fold_conds: List[str] = []
        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            ins_cols: List[str] = []
            ins_vals: List[str] = []
            if not grp_exprs:
                ins_cols.append("__mv_global_id")
                ins_vals.append("1")
            elif has_nullable_grp:
                ins_cols.append("__mv_pk")
                rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
                ins_vals.append(self._canonical_group_pk_expr(rec_exprs))
            else:
                for g_expr in grp_exprs:
                    ins_cols.append(self._group_col_alias(g_expr))
                    ins_vals.append(self._rewrite_expr_for_record(g_expr, record_prefix))
            ins_cols.extend(["__val", "__ref_cnt"])
            ins_vals.extend([val_expr, "1"])
            if self.dialect == Dialect.MYSQL:
                stmts.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(self._q(c) for c in ins_cols)})\n"
                    f"    SELECT {', '.join(ins_vals)}{from_dual}\n"
                    f"    WHERE ({val_expr}) IS NOT NULL\n"
                    f"    ON DUPLICATE KEY UPDATE {self._q('__ref_cnt')} = {self._q('__ref_cnt')} + 1;"
                )
            else:
                q_aux_conflict = [self._q(c) for c in [*pk_cols, "__val"]]
                stmts.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(self._q(c) for c in ins_cols)})\n"
                    f"    SELECT {', '.join(ins_vals)}\n"
                    f"    WHERE ({val_expr}) IS NOT NULL\n"
                    f"    ON CONFLICT ({', '.join(q_aux_conflict)}) DO UPDATE SET\n"
                    f"        {self._q('__ref_cnt')} = {aux_tbl}.{self._q('__ref_cnt')} + 1;"
                )

        def _is_newly_distinct_cond(item: SelectItem) -> str:
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if not item.is_distinct:
                return f"({val_e}) IS NOT NULL"
            aux_tbl = distinct_aux_tables[ekey][0]
            match_p = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
            return f"({val_e}) IS NOT NULL AND (SELECT {aux_tbl}.{self._q('__ref_cnt')} FROM {aux_tbl} WHERE {match_p}) = 1"

        def _plus_inc_expr(item: SelectItem) -> str:
            return f"(CASE WHEN {_is_newly_distinct_cond(item)} THEN 1 ELSE 0 END)"

        def _plus_sum_expr(item: SelectItem, col_alias: str) -> str:
            raw = _plus_sum_expr_raw(item, col_alias)
            ekey = expr_key_fn(item)
            if ekey in acnt_map:
                return self._sqlite_guarded_fold(
                    mv_table, col_alias, acnt_map[ekey], raw,
                    self._effective_agg_value_expr(item, record_prefix=record_prefix), fold_conds,
                )
            if self._sqlite_exact_sum_enabled() and item.is_distinct:
                return self._sqlite_distinct_sum_recompute(item, col_alias, record_prefix)
            return raw

        def _plus_sum_expr_raw(item: SelectItem, col_alias: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            q_col = self._q(col_alias)
            if item.is_distinct:
                cond = _is_newly_distinct_cond(item)
                return (
                    f"(CASE WHEN NOT ({cond}) THEN {mv_table}.{q_col} "
                    f"WHEN {mv_table}.{q_col} IS NULL THEN ({val_e}) "
                    f"ELSE {mv_table}.{q_col} + ({val_e}) END)"
                )
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                f"WHEN {mv_table}.{q_col} IS NULL THEN ({val_e}) "
                f"ELSE {mv_table}.{q_col} + ({val_e}) END)"
            )

        def _plus_sum2_expr(item: SelectItem, col_alias: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            q_col = self._q(col_alias)
            if self.dialect == Dialect.SQLITE:
                ekey = expr_key_fn(item)
                cnt_c = self._q(col_nonnull_count_map[ekey])
                sum_c = self._q(col_sum_map[ekey])
                delta = f"((1.0 * ({val_e})) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_c}))"
                return (
                    f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                    f"WHEN {mv_table}.{cnt_c} <= 0 OR {mv_table}.{sum_c} IS NULL THEN 0.0 "
                    f"ELSE COALESCE({mv_table}.{q_col}, 0.0) + ({delta} * {delta} * ((1.0 * {mv_table}.{cnt_c}) / ({mv_table}.{cnt_c} + 1))) END)"
                )
            sq = self._var_sq_sql(val_e)
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                f"WHEN {mv_table}.{q_col} IS NULL THEN {sq} "
                f"ELSE {mv_table}.{q_col} + {sq} END)"
            )

        def _plus_tcnt_expr(item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if self.dialect == Dialect.POSTGRES:
                return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) != 0 THEN 1 ELSE 0 END)"

        insert_cols: List[str] = []
        insert_vals: List[str] = []
        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        if not grp_exprs:
            insert_cols.append("__mv_global_id")
            insert_vals.append("1")
        elif has_nullable_grp:
            insert_cols.append("__mv_pk")
            rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
            insert_vals.append(self._canonical_group_pk_expr(rec_exprs))

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                insert_cols.append(mv_col)
                insert_vals.append(self._rewrite_expr_for_record(g_expr, record_prefix))

        for item in work_items:
            col_alias = self._q(item.alias)
            insert_cols.append(col_alias)
            if not item.is_aggregate:
                insert_vals.append(self._rewrite_expr_for_record(item.raw_expr, record_prefix))
                continue

            updated_cols.add(col_alias)
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                insert_vals.append("1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
            elif self._sqlite_union_native_sum(item):
                insert_vals.append(self._build_recompute_subquery_raw(item, record_prefix))
            elif (
                self._sqlite_exact_sum_enabled()
                and item.is_distinct
                and item.agg_func in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG)
            ):
                # The new group holds exactly the rows the triggers have folded;
                # the native aggregate applies SQLite's numeric conversion of
                # TEXT/BLOB inputs that a raw copy of the value would skip.
                insert_vals.append(
                    self._build_native_group_recompute(
                        self._sql_backfill_agg_expr(item), record_prefix, processed_only=True
                    )
                )
            elif item.agg_func == AggFunc.SUM:
                insert_vals.append(f"({val_e})")
            elif item.agg_func == AggFunc.TOTAL:
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)")
            elif item.agg_func == AggFunc.AVG:
                if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL) and self._avg_native_sum_scale(item) is not None:
                    # A new group holds exactly this row: AVG = value / 1 with
                    # the native result scale (PostgreSQL numeric_div scale).
                    first = self._avg_from_sum_count_sql(item, f"({val_e})", "1")
                    insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE {first} END)")
                else:
                    insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 1.0 * ({val_e}) END)")
            elif item.agg_func in (AggFunc.VAR_POP, AggFunc.STDDEV_POP):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 0.0 END)")
            elif item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP):
                insert_vals.append("NULL")
            elif item.agg_func in (AggFunc.MIN, AggFunc.MAX, AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR):
                insert_vals.append(f"({val_e})")
            elif item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if self.dialect == Dialect.POSTGRES:
                    insert_vals.append(
                        f"(CASE WHEN ({val_e}) IS NULL THEN NULL WHEN ({val_e}) THEN TRUE ELSE FALSE END)"
                    )
                else:
                    insert_vals.append(
                        f"(CASE WHEN ({val_e}) IS NULL THEN NULL WHEN ({val_e}) != 0 THEN 1 ELSE 0 END)"
                    )
            else:
                insert_vals.append(f"({val_e})")

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + 1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                update_assignments.append(
                    f"{col_alias} = {mv_table}.{col_alias} + {_plus_inc_expr(item)}"
                )
            elif self._sqlite_union_native_sum(item):
                update_assignments.append(
                    f"{col_alias} = {self._build_recompute_subquery_raw(item, record_prefix)}"
                )
            elif item.agg_func == AggFunc.SUM:
                new_sum = _plus_sum_expr(item, col_alias)
                update_assignments.append(f"{col_alias} = {new_sum}")
            elif item.agg_func == AggFunc.TOTAL:
                if item.is_distinct:
                    cond = _is_newly_distinct_cond(item)
                    inc_val = f"(CASE WHEN ({cond}) THEN 1.0 * ({val_e}) ELSE 0.0 END)"
                else:
                    inc_val = f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                folded = f"{mv_table}.{col_alias} + {inc_val}"
                if ekey in acnt_map:
                    folded = self._sqlite_guarded_fold(
                        mv_table, item.alias, acnt_map[ekey], folded, val_e, fold_conds,
                    )
                elif self._sqlite_exact_sum_enabled() and item.is_distinct:
                    folded = self._build_native_group_recompute(
                        self._sql_backfill_agg_expr(item), record_prefix, processed_only=True
                    )
                update_assignments.append(f"{col_alias} = {folded}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                    update_assignments.append(f"{col_alias} = {recomputed}")
                elif self._sqlite_exact_sum_enabled() and item.is_distinct:
                    recomputed = self._build_native_group_recompute(
                        self._sql_backfill_agg_expr(item), record_prefix, processed_only=True
                    )
                    update_assignments.append(f"{col_alias} = {recomputed}")
                else:
                    cnt_col = self._q(col_nonnull_count_map[ekey])
                    sum_col = self._q(col_sum_map[ekey])
                    new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                    new_sum = _plus_sum_expr(item, sum_col)
                    derived = self._avg_from_sum_count_sql(item, new_sum, new_cnt)
                    update_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                        f"ELSE {derived} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                # Backfill uses the engine's native statistical aggregate.
                # A Welford/raw-moment delta update uses a different floating
                # operation order and is not bit-identical for approximate
                # inputs. Re-evaluate the affected group with the same native
                # aggregate so the materialized result matches the live view.
                recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomputed}")
            elif item.agg_func == AggFunc.MIN:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL OR ({val_e}) < {mv_table}.{col_alias} "
                    f"THEN ({val_e}) ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.MAX:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL OR ({val_e}) > {mv_table}.{col_alias} "
                    f"THEN ({val_e}) ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.BIT_AND:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE ({mv_table}.{col_alias} & ({val_e})) END"
                )
            elif item.agg_func == AggFunc.BIT_OR:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE ({mv_table}.{col_alias} | ({val_e})) END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                xor_expr = self._bitwise_xor_sql(f"{mv_table}.{col_alias}", f"({val_e})")
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE {xor_expr} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                tcnt_col = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} + {_plus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                tcnt_col = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} + {_plus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )
            else:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomp}")

        q_row_count_col = self._q(row_count_col)
        if q_row_count_col not in updated_cols:
            insert_cols.append(q_row_count_col)
            insert_vals.append("1")
            update_assignments.append(f"{q_row_count_col} = {mv_table}.{q_row_count_col} + 1")
            updated_cols.add(q_row_count_col)

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }
        already_in_ins = set(insert_cols)
        for ekey, cnt_col in col_nonnull_count_map.items():
            it = ekey_to_item[ekey]
            q_cnt_col = self._q(cnt_col)
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if q_cnt_col not in already_in_ins:
                insert_cols.append(q_cnt_col)
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
                already_in_ins.add(q_cnt_col)
            if q_cnt_col not in updated_cols:
                update_assignments.append(
                    f"{q_cnt_col} = {mv_table}.{q_cnt_col} + {_plus_inc_expr(it)}"
                )
                updated_cols.add(q_cnt_col)

        for ekey, sum_col in col_sum_map.items():
            it = ekey_to_item[ekey]
            q_sum_col = self._q(sum_col)
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if q_sum_col not in already_in_ins:
                insert_cols.append(q_sum_col)
                insert_vals.append(f"({val_e})")
                already_in_ins.add(q_sum_col)
            if q_sum_col not in updated_cols:
                update_assignments.append(f"{q_sum_col} = {_plus_sum_expr(it, q_sum_col)}")
                updated_cols.add(q_sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            it = ekey_to_item[ekey]
            q_sum2_col = self._q(sum2_col)
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if q_sum2_col not in already_in_ins:
                insert_cols.append(q_sum2_col)
                init_s2 = "0.0" if self.dialect == Dialect.SQLITE else self._var_sq_sql(val_e)
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE {init_s2} END)"
                )
                already_in_ins.add(q_sum2_col)
            if q_sum2_col not in updated_cols:
                update_assignments.append(f"{q_sum2_col} = {_plus_sum2_expr(it, q_sum2_col)}")
                updated_cols.add(q_sum2_col)

        for ekey, acnt_col in acnt_map.items():
            it = ekey_to_item[ekey]
            q_acnt = self._q(acnt_col)
            approx_e = self._sqlite_is_approx_value_sql(
                self._effective_agg_value_expr(it, record_prefix=record_prefix)
            )
            insert_cols.append(q_acnt)
            insert_vals.append(approx_e)
            already_in_ins.add(q_acnt)
            update_assignments.append(f"{q_acnt} = {mv_table}.{q_acnt} + {approx_e}")
            updated_cols.add(q_acnt)
        if acnt_map:
            # A new group row holds one value: exact for an integer (or NULL),
            # recomputed natively otherwise (e.g. SUM(-0.0) is 0.0 natively).
            q_dirty = self._q(self._SQLITE_DIRTY_COL)
            approx_any = " OR ".join(
                f"{self._sqlite_is_approx_value_sql(self._effective_agg_value_expr(ekey_to_item[k], record_prefix=record_prefix))} = 1"
                for k in acnt_map
            )
            insert_cols.append(q_dirty)
            insert_vals.append(f"(CASE WHEN {approx_any} THEN 1 ELSE 0 END)")
            already_in_ins.add(q_dirty)
            update_assignments.append(self._sqlite_dirty_fold_assignment(mv_table, fold_conds))
            updated_cols.add(q_dirty)

        for ekey, tcnt_col in col_true_cnt_map.items():
            it = ekey_to_item[ekey]
            q_tcnt_col = self._q(tcnt_col)
            if q_tcnt_col not in already_in_ins:
                insert_cols.append(q_tcnt_col)
                insert_vals.append(_plus_tcnt_expr(it))
                already_in_ins.add(q_tcnt_col)
            if q_tcnt_col not in updated_cols:
                update_assignments.append(
                    f"{q_tcnt_col} = {mv_table}.{q_tcnt_col} + {_plus_tcnt_expr(it)}"
                )
                updated_cols.add(q_tcnt_col)

        # In MySQL, ON DUPLICATE KEY UPDATE evaluates left-to-right, so derived columns must come first
        derived_aliases = {
            self._q(it.alias)
            for it in work_items
            if it.is_aggregate
            and it.agg_func
            in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            )
        }
        ordered_updates = [
            a for a in update_assignments if a.split("=")[0].strip() in derived_aliases
        ] + [
            a for a in update_assignments if a.split("=")[0].strip() not in derived_aliases
        ]

        q_insert_cols = [self._q(c) for c in insert_cols]
        q_pk_cols = [self._q(c) for c in pk_cols]
        if self.dialect == Dialect.MYSQL:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(q_insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE\n"
                f"        {', '.join(ordered_updates)};"
            )
        else:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(q_insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
                f"        {', '.join(ordered_updates)};"
            )
        return stmts

    def _build_minus_delta_statements(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        record_prefix: str = "OLD",
        include_delete_zero_mv: bool = True,
    ) -> Tuple[List[str], Optional[str]]:
        stmts: List[str] = []
        acnt_map = self._sqlite_acnt_map()
        fold_conds: List[str] = []
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            match_p = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"UPDATE {aux_tbl}\n"
                f"    SET {self._q('__ref_cnt')} = {aux_tbl}.{self._q('__ref_cnt')} - 1\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND {match_p};"
            )

        def _is_last_distinct_cond(item: SelectItem) -> str:
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if not item.is_distinct:
                return f"({val_e}) IS NOT NULL"
            aux_tbl = distinct_aux_tables[ekey][0]
            match_p = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
            return f"({val_e}) IS NOT NULL AND (SELECT {aux_tbl}.{self._q('__ref_cnt')} FROM {aux_tbl} WHERE {match_p}) <= 0"

        def _minus_dec_expr(item: SelectItem) -> str:
            return f"(CASE WHEN {_is_last_distinct_cond(item)} THEN 1 ELSE 0 END)"

        def _minus_sum_expr(item: SelectItem, col_alias: str, cnt_col: str) -> str:
            raw = _minus_sum_expr_raw(item, col_alias, cnt_col)
            ekey = expr_key_fn(item)
            if ekey in acnt_map:
                return self._sqlite_guarded_fold(
                    mv_table, col_alias, acnt_map[ekey], raw,
                    self._effective_agg_value_expr(item, record_prefix=record_prefix), fold_conds,
                )
            if self._sqlite_exact_sum_enabled() and item.is_distinct:
                return self._sqlite_distinct_sum_recompute(item, col_alias, record_prefix)
            return raw

        def _minus_sum_expr_raw(item: SelectItem, col_alias: str, cnt_col: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            q_col = self._q(col_alias)
            q_cnt = self._q(cnt_col)
            new_cnt = f"({mv_table}.{q_cnt} - {_minus_dec_expr(item)})"
            if item.is_distinct:
                cond = _is_last_distinct_cond(item)
                return (
                    f"(CASE WHEN NOT ({cond}) THEN {mv_table}.{q_col} "
                    f"WHEN {new_cnt} <= 0 THEN NULL "
                    f"ELSE {mv_table}.{q_col} - ({val_e}) END)"
                )
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                f"WHEN {new_cnt} <= 0 THEN NULL "
                f"ELSE {mv_table}.{q_col} - ({val_e}) END)"
            )

        def _minus_sum2_expr(item: SelectItem, col_alias: str, cnt_col: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            q_col = self._q(col_alias)
            q_cnt = self._q(cnt_col)
            new_cnt = f"({mv_table}.{q_cnt} - {_minus_dec_expr(item)})"
            if self.dialect == Dialect.SQLITE:
                ekey = expr_key_fn(item)
                sum_c = self._q(col_sum_map[ekey])
                delta = f"((1.0 * ({val_e})) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{q_cnt}))"
                return (
                    f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                    f"WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = 1 THEN 0.0 "
                    f"ELSE MAX(0.0, COALESCE({mv_table}.{q_col}, 0.0) - ({delta} * {delta} * ((1.0 * {mv_table}.{q_cnt}) / ({mv_table}.{q_cnt} - 1)))) END)"
                )
            sq = self._var_sq_sql(val_e)
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{q_col} "
                f"WHEN {new_cnt} <= 0 THEN NULL "
                f"ELSE {mv_table}.{q_col} - {sq} END)"
            )

        def _minus_tcnt_expr(item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if self.dialect == Dialect.POSTGRES:
                return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) != 0 THEN 1 ELSE 0 END)"

        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        for item in work_items:
            if not item.is_aggregate:
                continue
            col_alias = self._q(item.alias)
            updated_cols.add(col_alias)
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} - 1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                update_assignments.append(
                    f"{col_alias} = {mv_table}.{col_alias} - {_minus_dec_expr(item)}"
                )
            elif self._sqlite_union_native_sum(item):
                update_assignments.append(
                    f"{col_alias} = {self._build_recompute_subquery_raw(item, record_prefix)}"
                )
            elif item.agg_func == AggFunc.SUM:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                update_assignments.append(
                    f"{col_alias} = {_minus_sum_expr(item, col_alias, cnt_col)}"
                )
            elif item.agg_func == AggFunc.TOTAL:
                if item.is_distinct:
                    cond = _is_last_distinct_cond(item)
                    dec_val = f"(CASE WHEN ({cond}) THEN 1.0 * ({val_e}) ELSE 0.0 END)"
                else:
                    dec_val = f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                folded = f"{mv_table}.{col_alias} - {dec_val}"
                if ekey in acnt_map:
                    folded = self._sqlite_guarded_fold(
                        mv_table, item.alias, acnt_map[ekey], folded, val_e, fold_conds,
                    )
                elif self._sqlite_exact_sum_enabled() and item.is_distinct:
                    folded = self._build_native_group_recompute(
                        self._sql_backfill_agg_expr(item), record_prefix, processed_only=True
                    )
                update_assignments.append(f"{col_alias} = {folded}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                    update_assignments.append(f"{col_alias} = {recomputed}")
                elif self._sqlite_exact_sum_enabled() and item.is_distinct:
                    recomputed = self._build_native_group_recompute(
                        self._sql_backfill_agg_expr(item), record_prefix, processed_only=True
                    )
                    update_assignments.append(f"{col_alias} = {recomputed}")
                else:
                    cnt_col = self._q(col_nonnull_count_map[ekey])
                    sum_col = self._q(col_sum_map[ekey])
                    new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                    new_sum = _minus_sum_expr(item, sum_col, cnt_col)
                    derived = self._avg_from_sum_count_sql(item, new_sum, new_cnt)
                    update_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                        f"ELSE {derived} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomputed}")
            elif item.agg_func == AggFunc.MIN:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL AND {mv_table}.{col_alias} >= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL AND {mv_table}.{col_alias} <= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR):
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                xor_expr = self._bitwise_xor_sql(f"{mv_table}.{col_alias}", f"({val_e})")
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {new_cnt} <= 0 THEN NULL ELSE {xor_expr} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                tcnt_col = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_col = self._q(col_nonnull_count_map[ekey])
                tcnt_col = self._q(col_true_cnt_map[ekey])
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )
            else:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomp}")

        q_row_count_col = self._q(row_count_col)
        if q_row_count_col not in updated_cols:
            update_assignments.append(f"{q_row_count_col} = {mv_table}.{q_row_count_col} - 1")
            updated_cols.add(q_row_count_col)

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }
        for ekey, cnt_col in col_nonnull_count_map.items():
            q_cnt_col = self._q(cnt_col)
            if q_cnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{q_cnt_col} = {mv_table}.{q_cnt_col} - {_minus_dec_expr(it)}"
                )
                updated_cols.add(q_cnt_col)

        for ekey, sum_col in col_sum_map.items():
            q_sum_col = self._q(sum_col)
            if q_sum_col not in updated_cols:
                it = ekey_to_item[ekey]
                cnt_col = self._q(col_nonnull_count_map[ekey])
                update_assignments.append(
                    f"{q_sum_col} = {_minus_sum_expr(it, q_sum_col, cnt_col)}"
                )
                updated_cols.add(q_sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            q_sum2_col = self._q(sum2_col)
            if q_sum2_col not in updated_cols:
                it = ekey_to_item[ekey]
                cnt_col = self._q(col_nonnull_count_map[ekey])
                update_assignments.append(
                    f"{q_sum2_col} = {_minus_sum2_expr(it, q_sum2_col, cnt_col)}"
                )
                updated_cols.add(q_sum2_col)

        for ekey, acnt_col in acnt_map.items():
            it = ekey_to_item[ekey]
            q_acnt = self._q(acnt_col)
            approx_e = self._sqlite_is_approx_value_sql(
                self._effective_agg_value_expr(it, record_prefix=record_prefix)
            )
            update_assignments.append(f"{q_acnt} = {mv_table}.{q_acnt} - {approx_e}")
            updated_cols.add(q_acnt)
        if acnt_map:
            update_assignments.append(self._sqlite_dirty_fold_assignment(mv_table, fold_conds))
            updated_cols.add(self._q(self._SQLITE_DIRTY_COL))

        for ekey, tcnt_col in col_true_cnt_map.items():
            q_tcnt_col = self._q(tcnt_col)
            if q_tcnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{q_tcnt_col} = {mv_table}.{q_tcnt_col} - {_minus_tcnt_expr(it)}"
                )
                updated_cols.add(q_tcnt_col)

        # In MySQL, UPDATE evaluates SET assignments left-to-right!
        # Order: (1) derived aggs (AVG, VAR, STDDEV, BIT_XOR, BOOL_AND, BOOL_OR)
        #        (2) SUM / sum2 / true_cnt (which read col_nonnull_count_map)
        #        (3) counts & MIN/MAX
        high_priority_cols = (
            {
                self._q(it.alias)
                for it in work_items
                if it.is_aggregate
                and it.agg_func
                in (
                    AggFunc.AVG,
                    AggFunc.VAR_POP,
                    AggFunc.VAR_SAMP,
                    AggFunc.STDDEV_POP,
                    AggFunc.STDDEV_SAMP,
                    AggFunc.BIT_XOR,
                    AggFunc.BOOL_AND,
                    AggFunc.BOOL_OR,
                )
            }
        )
        mid_priority_cols = (
            {self._q(it.alias) for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
            | {self._q(c) for c in col_sum_map.values()}
            | {self._q(c) for c in col_sum2_map.values()}
            | {self._q(c) for c in col_true_cnt_map.values()}
        )
        tier_1 = [a for a in update_assignments if a.split("=")[0].strip() in high_priority_cols]
        tier_2 = [
            a
            for a in update_assignments
            if a.split("=")[0].strip() not in high_priority_cols
            and a.split("=")[0].strip() in mid_priority_cols
        ]
        tier_3 = [
            a
            for a in update_assignments
            if a.split("=")[0].strip() not in high_priority_cols
            and a.split("=")[0].strip() not in mid_priority_cols
        ]
        ordered_updates = tier_1 + tier_2 + tier_3

        grp_pred = self._group_match_predicate_tier1(mv_table, record_prefix)
        stmts.append(
            f"UPDATE {mv_table}\n"
            f"    SET {', '.join(ordered_updates)}\n"
            f"    WHERE {grp_pred};"
        )

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            match_p = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"DELETE FROM {aux_tbl}\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND ({match_p}) AND {aux_tbl}.{self._q('__ref_cnt')} <= 0;"
            )

        delete_zero_mv_stmt: Optional[str] = None
        if grp_exprs:
            delete_zero_mv_stmt = (
                f"DELETE FROM {mv_table}\n"
                f"    WHERE ({grp_pred}) AND {mv_table}.{self._q(row_count_col)} <= 0;"
            )
            if include_delete_zero_mv:
                stmts.append(delete_zero_mv_stmt)

        return stmts, delete_zero_mv_stmt

    def _apply_where_guard_to_stmt(self, stmt: str, where_cond: Optional[str]) -> str:
        if not where_cond:
            return stmt
        s = stmt.strip()
        if re.match(r"^INSERT\s+INTO\b", s, flags=re.IGNORECASE):
            if re.search(r"\n\s*WHERE\s+TRUE\b", s, flags=re.IGNORECASE):
                return re.sub(
                    r"\n\s*WHERE\s+TRUE\b",
                    f"\n    WHERE ({where_cond})",
                    s,
                    count=1,
                    flags=re.IGNORECASE,
                )
            return re.sub(
                r"\n\s*WHERE\s+",
                f"\n    WHERE ({where_cond}) AND ",
                s,
                count=1,
                flags=re.IGNORECASE,
            )
        if re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", s, flags=re.IGNORECASE):
            return re.sub(
                r"\n\s*WHERE\s+",
                f"\n    WHERE ({where_cond}) AND ",
                s,
                count=1,
                flags=re.IGNORECASE,
            )
        return stmt


    def _build_tier1_triggers(
        self,
        view_name: str,
        mv_table: str,
        base_t: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        initial_populate_sqls: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        plus_new_stmts = self._build_plus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="NEW",
        )
        minus_old_del_stmts, _ = self._build_minus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="OLD",
            include_delete_zero_mv=True,
        )
        minus_old_upd_stmts, delete_zero_mv_stmt = self._build_minus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="OLD",
            include_delete_zero_mv=False,
        )
        # SQLite SUM/TOTAL/AVG: fold steps only fold exactly or mark the group
        # dirty; dirty groups are recomputed natively once all fold steps of the
        # trigger have run (for UPDATE: after both the OLD and the NEW step).
        recompute_new = self._sqlite_recompute_dirty_stmt(mv_table, "NEW", col_sum_map, expr_key_fn)
        recompute_old = self._sqlite_recompute_dirty_stmt(mv_table, "OLD", col_sum_map, expr_key_fn)
        plus_upd_stmts = list(plus_new_stmts)
        upd_cleanup_stmts = [delete_zero_mv_stmt] if delete_zero_mv_stmt else []
        if recompute_new and recompute_old:
            plus_new_stmts = plus_new_stmts + [recompute_new]
            plus_upd_stmts = plus_upd_stmts + [recompute_new]
            minus_old_del_stmts = minus_old_del_stmts + [recompute_old]
            upd_cleanup_stmts = upd_cleanup_stmts + [recompute_old]

        where_new = (
            self._rewrite_expr_for_record(self.view.where_clause, "NEW")
            if self.view.where_clause
            else None
        )
        where_old = (
            self._rewrite_expr_for_record(self.view.where_clause, "OLD")
            if self.view.where_clause
            else None
        )
        if self.watermark_catalog_table and self._alg_mode in ("single", "distinct"):
            wm_pk_col: Optional[str] = "rowid" if self.dialect == Dialect.SQLITE else None
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
                    f"AND NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)"
                )
                wm_old = (
                    f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                    f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                    f"AND OLD.{q_pk} > backfill_cursor_pk AND OLD.{q_pk} <= backfill_max_pk)"
                )
                where_new = f"({where_new}) AND ({wm_new})" if where_new else wm_new
                where_old = f"({where_old}) AND ({wm_old})" if where_old else wm_old

        trunc_stmts = [
            f"DELETE FROM {aux_tbl};" for aux_tbl, _, _ in distinct_aux_tables.values()
        ] + [f"DELETE FROM {mv_table};"]
        if not self._effective_group_by_exprs and initial_populate_sqls:
            trunc_stmts.extend(initial_populate_sqls)

        pg_ai_body: Optional[str] = None
        pg_ad_body: Optional[str] = None
        pg_au_body: Optional[str] = None
        if self.dialect == Dialect.POSTGRES and self.pg_use_transition_tables and self.pg_vectorize_transition_tables:
            pg_ai_body, pg_ad_body, pg_au_body = self._build_tier1_vectorized_pg_bodies(
                mv_table=mv_table,
                base_t=base_t,
                row_count_col=row_count_col,
                col_nonnull_count_map=col_nonnull_count_map,
                col_sum_map=col_sum_map,
                col_sum2_map=col_sum2_map,
                col_true_cnt_map=col_true_cnt_map,
                distinct_aux_tables=distinct_aux_tables,
                expr_key_fn=expr_key_fn,
            )

        return self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=plus_new_stmts,
            del_stmts=minus_old_del_stmts,
            upd_old_stmts=minus_old_upd_stmts,
            upd_new_stmts=plus_upd_stmts,
            upd_cleanup_old_stmts=upd_cleanup_stmts,
            trunc_stmts=trunc_stmts,
            where_new=where_new,
            where_old=where_old,
            pg_ai_body=pg_ai_body,
            pg_ad_body=pg_ad_body,
            pg_au_body=pg_au_body,
        )

