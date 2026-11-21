"""Tier 1B: Algebraic incremental maintenance for INNER JOIN aggregate views."""

from __future__ import annotations

import hashlib
import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .._host import MixinHost
from ..dialects.pg_statement import (
    EVENTS as PG_EVENTS,
    SIGN,
    PgAggregateSpec,
    PgDeltaSource,
    PgStatementBracket,
)
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
    _sqlite_type_affinity,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_view_ast,
    quote_sql_identifier,
)


class Tier1JoinMixin(MixinHost):
    def _qualify_join_expr(self, expr: Optional[str]) -> str:
        if not expr:
            return ""
        col_to_aliases: Dict[str, List[Tuple[str, str]]] = {}
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts:
                for c_name in ts.columns:
                    col_to_aliases.setdefault(c_name.lower(), []).append((j.alias, c_name))

        alias_remap: Dict[str, Tuple[str, Optional[TableSchema]]] = {}
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            alias_remap[j.table_name.lower()] = (j.alias, ts)
            alias_remap[j.alias.lower()] = (j.alias, ts)

        try:
            toks = _tokenize_sql(expr, read=_sqlglot_dialect(self.dialect))
        except Exception:
            return expr
        out: List[str] = []
        i = 0
        n = len(toks)
        cursor = 0
        while i < n:
            tk = toks[i]
            if (
                _is_word_or_ident_token(tk)
                and i + 2 < n
                and toks[i + 1].token_type == SQLGlotTokenType.DOT
                and _is_word_or_ident_token(toks[i + 2])
            ):
                q_low = tk.text.lower()
                c_unq = toks[i + 2].text
                if q_low in alias_remap:
                    new_al, ts = alias_remap[q_low]
                    actual_c = c_unq
                    if ts:
                        for sc in ts.columns:
                            if sc.lower() == c_unq.lower():
                                actual_c = sc
                                break
                    if tk.start > cursor:
                        out.append(expr[cursor:tk.start])
                    out.append(f"{self._q(new_al)}.{self._q(actual_c)}")
                    cursor = toks[i + 2].end + 1
                    i += 3
                    continue
            if _is_word_or_ident_token(tk):
                is_after_dot = (i > 0 and toks[i - 1].token_type == SQLGlotTokenType.DOT)
                is_before_dot = (i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.DOT)
                is_fn_call = (
                    tk.token_type != SQLGlotTokenType.IDENTIFIER
                    and i + 1 < n
                    and toks[i + 1].token_type == SQLGlotTokenType.L_PAREN
                )
                if not is_after_dot and not is_before_dot and not is_fn_call:
                    c_low = tk.text.lower()
                    als = col_to_aliases.get(c_low)
                    if als and (
                        len(als) == 1
                        or any(
                            j.on_condition and c_low in {t.text.lower() for t in _tokenize_sql(j.on_condition, read=_sqlglot_dialect(self.dialect))}
                            for j in self.view.joins
                        )
                    ):
                        al, actual_c = als[0]
                        if tk.start > cursor:
                            out.append(expr[cursor:tk.start])
                        out.append(f"{self._q(al)}.{self._q(actual_c)}")
                        cursor = tk.end + 1
                        i += 1
                        continue
            i += 1
        if cursor < len(expr):
            out.append(expr[cursor:])
        return "".join(out)

    def _compile_tier1_join_algebraic(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

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

        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()
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
                    c_def = self._lookup_col_def(self._qualify_join_expr(g_expr)) or self._lookup_col_def(g_expr)
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
        extra_join_cols = self._auto_backing_extra_index_cols(
            excluded_leading_cols=group_mv_cols[:1]
        )
        indexed_cols_lower = self._indexed_column_names_lower(list(group_mv_cols) + list(extra_join_cols))

        for item in work_items:
            if not item.is_aggregate:
                if self.dialect == Dialect.SQLITE:
                    dt = "BLOB"
                else:
                    c_def = self._lookup_col_def(self._qualify_join_expr(item.raw_expr)) or self._lookup_col_def(item.raw_expr)
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

        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
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
            extra_auto_index_cols=extra_join_cols,
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

        qual_grp_exprs = [self._qualify_join_expr(g) for g in grp_exprs]
        target_cols: List[str] = []
        select_exprs: List[str] = []
        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(qual_grp_exprs))

        seen_hidden: Set[str] = set()
        for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
            mv_col = self._group_col_alias(g_raw)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                target_cols.append(mv_col)
                select_exprs.append(g_qual)

        for item in work_items:
            target_cols.append(item.alias)
            if not item.is_aggregate:
                select_exprs.append(self._qualify_join_expr(item.raw_expr))
            else:
                select_exprs.append(self._qualify_join_expr(self._sql_backfill_agg_expr(item)))

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
            qualify_join=True,
            include_distinct_keyword=False,
        )

        def _build_join_from_clause(replace_alias: Optional[str] = None, rec_prefix: Optional[str] = None) -> str:
            from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
            parts: List[str] = []
            for idx, j in enumerate(self.view.joins):
                if replace_alias and j.alias.lower() == replace_alias.lower() and rec_prefix:
                    if rec_prefix in ("__mv_new_rows", "__mv_old_rows"):
                        tbl_sql = f"{rec_prefix} AS {self._q(j.alias)}"
                    else:
                        ts = self._schemas_lower[j.table_name.lower()]
                        rec_proj = ", ".join(f"{rec_prefix}.{self._q(c)} AS {self._q(c)}" for c in ts.columns)
                        tbl_sql = f"(SELECT {rec_proj}{from_dual}) AS {self._q(j.alias)}"
                else:
                    tbl_sql = f"{self._q(j.table_name)} AS {self._q(j.alias)}"
                if idx == 0:
                    parts.append(tbl_sql)
                else:
                    on_sql = f" ON {self._qualify_join_expr(j.on_condition)}" if j.on_condition else ""
                    parts.append(f"{j.join_type} {tbl_sql}{on_sql}")
            return " ".join(parts)

        full_from_sql = _build_join_from_clause()
        qual_where = self._qualify_join_expr(self.view.where_clause) if self.view.where_clause else None
        where_sql = f"\nWHERE {qual_where}" if qual_where else ""
        qual_grp_for_q = self._group_by_exprs_for_query(qual_grp_exprs, include_canonical_pk=has_nullable_grp)
        group_sql = f"\nGROUP BY {', '.join(qual_grp_for_q)}" if qual_grp_for_q else ""

        q_target_cols = [self._q(c) for c in target_cols]
        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM {full_from_sql}{where_sql}{group_sql};"
        )
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                f"SELECT {', '.join(q_target_cols)}\n"
                f"FROM (\nSELECT {', '.join(f'{ex} AS {col}' for col, ex in zip(q_target_cols, select_exprs))}\n"
                f"FROM {full_from_sql}{where_sql}{group_sql}\n) AS __mv_init\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        def _build_join_recompute_subquery(item: SelectItem) -> str:
            alias_remap = {j.alias.lower(): f"__mv_j_{j.alias}" for j in self.view.joins}
            for j in self.view.joins:
                alias_remap[j.table_name.lower()] = f"__mv_j_{j.alias}"

            def _remap_to_j(frag: str) -> str:
                q_frag = self._qualify_join_expr(frag)
                tokens = re.split(r"('(?:''|[^'])*')", q_frag)
                out = []
                for idx_t, tok in enumerate(tokens):
                    if idx_t % 2 == 1:
                        out.append(tok)
                        continue
                    mod = tok
                    for old_a, new_a in sorted(alias_remap.items(), key=lambda x: len(x[0]), reverse=True):
                        mod = re.sub(rf"\b{re.escape(old_a)}\s*\.", f"{new_a}.", mod, flags=re.IGNORECASE)
                    out.append(mod)
                return "".join(out)

            j_parts: List[str] = []
            for idx, j in enumerate(self.view.joins):
                tbl_s = f"{self._q(j.table_name)} AS __mv_j_{j.alias}"
                if idx == 0:
                    j_parts.append(tbl_s)
                else:
                    on_s = f" ON {_remap_to_j(j.on_condition)}" if j.on_condition else ""
                    j_parts.append(f"{j.join_type} {tbl_s}{on_s}")
            conds: List[str] = []
            if self.view.where_clause:
                conds.append(f"({_remap_to_j(self.view.where_clause)})")
            for g_raw in grp_exprs:
                mv_col = self._q(self._group_col_alias(g_raw))
                lhs_j = _remap_to_j(g_raw)
                conds.append(
                    self._null_safe_eq(f"({lhs_j})", f"{mv_table}.{mv_col}", is_nullable=True)
                )
            w_sub = f" WHERE {' AND '.join(conds)}" if conds else ""
            agg_e = _remap_to_j(self._sql_backfill_agg_expr(item))
            return f"(SELECT {agg_e} FROM {' '.join(j_parts)}{w_sub})"

        plus_merge_assignments = self._build_tier1_chunk_merge_assignments(
            mv_table=mv_table,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            expr_key_fn=_expr_key,
            use_mysql_values=(self.dialect == Dialect.MYSQL),
        )
        # Native approximate aggregates must preserve the engine's operation
        # order; algebraically equivalent sum/count updates can differ by ULPs.
        # SQLite SUM/TOTAL/AVG switch to binary floating point as soon as one
        # input is not an INTEGER (any SQLite column can store REAL values), and
        # an add/retract fold then depends on the history of removed values
        # (1e16 + 1.5 - 1e16 == 2.0). Recompute the affected group natively.
        def _join_needs_native_recompute(item: SelectItem) -> bool:
            return item.is_aggregate and (
                item.agg_func in (
                    AggFunc.VAR_POP, AggFunc.VAR_SAMP,
                    AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP,
                )
                or self._requires_native_avg_recompute(item)
                or (
                    self.dialect == Dialect.SQLITE
                    and item.agg_func in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG)
                )
            )

        native_recompute_aliases = {
            self._q(item.alias)
            for item in work_items
            if _join_needs_native_recompute(item)
        }
        if native_recompute_aliases:
            plus_merge_assignments = [
                assignment for assignment in plus_merge_assignments
                if assignment.split("=", 1)[0].strip() not in native_recompute_aliases
            ]
            for item in work_items:
                if _join_needs_native_recompute(item):
                    plus_merge_assignments.append(
                        f"{self._q(item.alias)} = {_build_join_recompute_subquery(item)}"
                    )

        # Build minus assignments from `__mv_delta`
        minus_assignments: List[str] = []
        minus_updated: Set[str] = set()

        def _delta_sum_minus(col: str, cnt_col: str) -> str:
            qc = self._q(col)
            qcnt = self._q(cnt_col)
            new_c = f"({mv_table}.{qcnt} - __mv_delta.{qcnt})"
            return (
                f"(CASE WHEN __mv_delta.{qc} IS NULL THEN {mv_table}.{qc} "
                f"WHEN {new_c} <= 0 THEN NULL "
                f"ELSE {mv_table}.{qc} - __mv_delta.{qc} END)"
            )

        def _delta_sum2_minus(ekey: str, sum2_col: str, cnt_col: str) -> str:
            q_sum2 = self._q(sum2_col)
            q_cnt = self._q(cnt_col)
            if self.dialect != Dialect.SQLITE:
                return _delta_sum_minus(q_sum2, q_cnt)
            sum_c = self._q(col_sum_map[ekey])
            new_c = f"({mv_table}.{q_cnt} - __mv_delta.{q_cnt})"
            d_mu = f"(((1.0 * __mv_delta.{sum_c}) / __mv_delta.{q_cnt}) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_c}))"
            return (
                f"(CASE WHEN __mv_delta.{q_cnt} <= 0 OR __mv_delta.{q_sum2} IS NULL THEN {mv_table}.{q_sum2} "
                f"WHEN {new_c} <= 0 THEN NULL "
                f"WHEN {new_c} = 1 THEN 0.0 "
                f"ELSE MAX(0.0, COALESCE({mv_table}.{q_sum2}, 0.0) - COALESCE(__mv_delta.{q_sum2}, 0.0) - ({d_mu} * {d_mu} * ((1.0 * {mv_table}.{cnt_c} * __mv_delta.{q_cnt}) / {new_c}))) END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = self._q(item.alias)
            minus_updated.add(c)
            ekey = _expr_key(item)
            if self.dialect == Dialect.SQLITE and item.agg_func in (
                AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG
            ):
                minus_assignments.append(f"{c} = {_build_join_recompute_subquery(item)}")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                minus_assignments.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")
            elif item.agg_func == AggFunc.SUM:
                cnt_c = col_nonnull_count_map[ekey]
                minus_assignments.append(f"{c} = {_delta_sum_minus(c, cnt_c)}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    recomputed = _build_join_recompute_subquery(item)
                    minus_assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = col_nonnull_count_map[ekey]
                    sum_c = col_sum_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                    new_sum = _delta_sum_minus(sum_c, cnt_c)
                    exact_avg = (
                        self.dialect in (Dialect.MYSQL, Dialect.POSTGRES)
                        and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
                    )
                    dividend = new_sum if exact_avg else f"(1.0 * {new_sum})"
                    minus_assignments.append(
                        f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL ELSE {dividend} / {new_cnt} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                recomputed = _build_join_recompute_subquery(item)
                minus_assignments.append(f"{c} = {recomputed}")
            elif item.agg_func == AggFunc.MIN:
                recomp = _build_join_recompute_subquery(item)
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} >= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = _build_join_recompute_subquery(item)
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} <= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_c = col_nonnull_count_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                xor_e = self._bitwise_xor_sql(f"{mv_table}.{c}", f"(__mv_delta.{c})")
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {new_cnt} <= 0 THEN NULL ELSE {xor_e} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_assignments.append(
                    f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_assignments.append(
                    f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )

        q_row_count_col = self._q(row_count_col)
        if q_row_count_col not in minus_updated:
            minus_assignments.append(f"{q_row_count_col} = {mv_table}.{q_row_count_col} - __mv_delta.{q_row_count_col}")
            minus_updated.add(q_row_count_col)

        for ekey, cnt_c in col_nonnull_count_map.items():
            q_cnt_c = self._q(cnt_c)
            if q_cnt_c not in minus_updated:
                minus_assignments.append(f"{q_cnt_c} = {mv_table}.{q_cnt_c} - __mv_delta.{q_cnt_c}")
                minus_updated.add(q_cnt_c)
        for ekey, sum_c in col_sum_map.items():
            q_sum_c = self._q(sum_c)
            if q_sum_c not in minus_updated:
                cnt_c = self._q(col_nonnull_count_map[ekey])
                minus_assignments.append(f"{q_sum_c} = {_delta_sum_minus(q_sum_c, cnt_c)}")
                minus_updated.add(q_sum_c)
        for ekey, sum2_c in col_sum2_map.items():
            q_sum2_c = self._q(sum2_c)
            if q_sum2_c not in minus_updated:
                cnt_c = self._q(col_nonnull_count_map[ekey])
                minus_assignments.append(f"{q_sum2_c} = {_delta_sum2_minus(ekey, q_sum2_c, cnt_c)}")
                minus_updated.add(q_sum2_c)
        for ekey, tcnt_c in col_true_cnt_map.items():
            q_tcnt_c = self._q(tcnt_c)
            if q_tcnt_c not in minus_updated:
                minus_assignments.append(f"{q_tcnt_c} = {mv_table}.{q_tcnt_c} - __mv_delta.{q_tcnt_c}")
                minus_updated.add(q_tcnt_c)

        high_p = {
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
        mid_p = (
            {self._q(it.alias) for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
            | {self._q(c) for c in col_sum_map.values()}
            | {self._q(c) for c in col_sum2_map.values()}
            | {self._q(c) for c in col_true_cnt_map.values()}
        )
        ordered_minus = (
            [a for a in minus_assignments if a.split("=")[0].strip() in high_p]
            + [a for a in minus_assignments if a.split("=")[0].strip() not in high_p and a.split("=")[0].strip() in mid_p]
            + [a for a in minus_assignments if a.split("=")[0].strip() not in high_p and a.split("=")[0].strip() not in mid_p]
        )

        # For MySQL multi-table UPDATE, LHS column names must be qualified with `mv_table.`
        mysql_ordered_minus = [
            f"{mv_table}.{a}" for a in ordered_minus
        ]

        aliased_delta_select = ", ".join(
            f"{ex} AS {col}" for col, ex in zip(q_target_cols, select_exprs)
        )
        having_non_empty = "" if grp_exprs else "\nHAVING COUNT(*) > 0"

        # PostgreSQL statement-level maintenance: the signed delta of the
        # firing table joined with the other sources' current state.
        pg_group_aliases = {self._group_col_alias(g).lower() for g in grp_exprs}
        pg_spec = PgAggregateSpec(
            mv_table=mv_table,
            row_count_col=row_count_col,
            items=work_items,
            grp_exprs=grp_exprs,
            has_nullable_grp=has_nullable_grp,
            pk_cols=pk_cols,
            nonnull_map=col_nonnull_count_map,
            sum_map=col_sum_map,
            sum2_map=col_sum2_map,
            tcnt_map=col_true_cnt_map,
            distinct_aux={},
            value_key=_expr_key,
            recompute=_build_join_recompute_subquery,
            native_recompute=_join_needs_native_recompute,
        )

        def _pg_join_source(delta_alias: str) -> PgDeltaSource:
            def _from(signed: str) -> str:
                parts: List[str] = []
                for idx, jj in enumerate(self.view.joins):
                    if jj.alias.lower() == delta_alias.lower():
                        tbl_sql = f"(\n        {signed}\n        ) AS {self._q(jj.alias)}"
                    else:
                        tbl_sql = f"{self._q(jj.table_name)} AS {self._q(jj.alias)}"
                    if idx == 0:
                        parts.append(tbl_sql)
                    else:
                        on_sql = f" ON {self._qualify_join_expr(jj.on_condition)}" if jj.on_condition else ""
                        parts.append(f"{jj.join_type} {tbl_sql}{on_sql}")
                return "\n        ".join(parts)

            return PgDeltaSource(
                from_sql=_from,
                sign=f"{self._q(delta_alias)}.{SIGN}",
                where=qual_where,
                keys=qual_grp_exprs,
                values={
                    _expr_key(it): self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                    for it in work_items
                    if it.is_aggregate
                },
                extras={
                    it.alias.lower(): self._qualify_join_expr(it.raw_expr)
                    for it in work_items
                    if not it.is_aggregate and it.alias.lower() not in pg_group_aliases
                },
            )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        is_pg = self.dialect == Dialect.POSTGRES
        aux_table_ddls: List[str] = self._pg_view_lock_ddls() if is_pg else []
        aux_table_names: List[str] = [self._pg_view_lock_table()] if is_pg else []

        # PostgreSQL Stage 3: multi-source overlap bracket + inclusion-exclusion
        # reconciliation so writable CTEs, FK cascades, and user triggers modifying
        # several sources of an algebraic join in one statement remain exact.
        pg_brackets_by_idx: Dict[int, PgStatementBracket] = {}
        if is_pg and len(self.view.joins) >= 2:
            pg_view_lock = self._pg_view_lock_stmt()
            pg_lock_tbl = self._pg_view_lock_table()
            pg_full_refresh_sql = "\n".join(
                [pg_view_lock, f"DELETE FROM {mv_table};", initial_populate_sql]
            )
            if len(self.view.joins) <= 4:
                # Determine which columns of each source are referenced by the join view.
                needed_by_alias: Dict[str, List[str]] = {j.alias.lower(): [] for j in self.view.joins}
                seen_col_by_alias: Dict[str, Set[str]] = {j.alias.lower(): set() for j in self.view.joins}
                fallback_all_cols = False
                exprs_to_scan: List[str] = []
                for jj in self.view.joins:
                    if jj.on_condition:
                        exprs_to_scan.append(self._qualify_join_expr(jj.on_condition))
                if qual_where:
                    exprs_to_scan.append(qual_where)
                exprs_to_scan.extend(qual_grp_exprs)
                for it in work_items:
                    if it.is_aggregate:
                        exprs_to_scan.append(
                            self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                        )
                    else:
                        exprs_to_scan.append(self._qualify_join_expr(it.raw_expr))
                if self.view.having_clause:
                    exprs_to_scan.append(self._qualify_join_expr(self.view.having_clause))
                for ex in exprs_to_scan:
                    try:
                        toks = _tokenize_sql(ex, read=_sqlglot_dialect(self.dialect))
                    except Exception:
                        fallback_all_cols = True
                        break
                    ti = 0
                    tn = len(toks)
                    while ti < tn:
                        if (
                            _is_word_or_ident_token(toks[ti])
                            and ti + 2 < tn
                            and toks[ti + 1].token_type == SQLGlotTokenType.DOT
                            and _is_word_or_ident_token(toks[ti + 2])
                        ):
                            al_low = toks[ti].text.lower()
                            col_txt = toks[ti + 2].text
                            if al_low in seen_col_by_alias:
                                if col_txt.lower() not in seen_col_by_alias[al_low]:
                                    seen_col_by_alias[al_low].add(col_txt.lower())
                                    needed_by_alias[al_low].append(col_txt)
                                ti += 3
                                continue
                        ti += 1

                stg_table_by_idx: Dict[int, str] = {}
                staged_cols_by_idx: Dict[int, List[str]] = {}
                for idx, jj in enumerate(self.view.joins):
                    ts = self._schemas_lower.get(jj.table_name.lower())
                    if ts is None:
                        cols_for_stg: List[str] = []
                    elif fallback_all_cols:
                        cols_for_stg = list(ts.columns.keys())
                    else:
                        cols_by_low = {c.lower(): c for c in ts.columns}
                        cols_for_stg = [
                            cols_by_low[c.lower()]
                            for c in needed_by_alias[jj.alias.lower()]
                            if c.lower() in cols_by_low
                        ]
                    stg_tbl, stg_ddl, stg_idx, stg_idx_ddl = self._pg_join_stage_table_ddls(
                        jj.table_name, cols_for_stg
                    )
                    stg_table_by_idx[idx] = stg_tbl
                    staged_cols_by_idx[idx] = cols_for_stg
                    aux_table_ddls.append(stg_ddl)
                    aux_table_names.append(stg_tbl)
                    index_ddls.append(stg_idx_ddl)
                    index_names.append(stg_idx)

                # Build all 2^m - 1 non-empty inclusion-exclusion delta sources:
                # Δ(R_1 ⋈ ... ⋈ R_m) = Σ_{∅ ≠ T ⊆ {1..m}} (-1)^(|T|-1) (∏_{i∈T} ΔR_i) ⋈ (∏_{i∉T} R_i)
                ie_sources: List[PgDeltaSource] = []
                n_joins = len(self.view.joins)
                for subset_size in range(1, n_joins + 1):
                    parity = 1 if (subset_size % 2 == 1) else -1
                    for subset in itertools.combinations(range(n_joins), subset_size):
                        subset_set = set(subset)
                        sign_factors = [f"{self._q(self.view.joins[i].alias)}.{SIGN}" for i in subset]
                        sign_expr = (
                            " * ".join(sign_factors)
                            if parity == 1
                            else f"-({' * '.join(sign_factors)})"
                        )

                        def _make_ie_from(sub_s: Set[int]) -> Any:
                            def _from(_signed: str) -> str:
                                parts: List[str] = []
                                for j_idx, jj in enumerate(self.view.joins):
                                    if j_idx in sub_s:
                                        st = stg_table_by_idx[j_idx]
                                        tbl_sql = (
                                            f"(SELECT * FROM {st} "
                                            f"WHERE __mv_tx = pg_catalog.txid_current() AND __mv_seq = __mv_bseq) "
                                            f"AS {self._q(jj.alias)}"
                                        )
                                    else:
                                        tbl_sql = f"{self._q(jj.table_name)} AS {self._q(jj.alias)}"
                                    if j_idx == 0:
                                        parts.append(tbl_sql)
                                    else:
                                        on_sql = (
                                            f" ON {self._qualify_join_expr(jj.on_condition)}"
                                            if jj.on_condition
                                            else ""
                                        )
                                        parts.append(f"{jj.join_type} {tbl_sql}{on_sql}")
                                return "\n        ".join(parts)

                            return _from

                        ie_sources.append(
                            PgDeltaSource(
                                from_sql=_make_ie_from(subset_set),
                                sign=sign_expr,
                                where=qual_where,
                                keys=qual_grp_exprs,
                                values={
                                    _expr_key(it): self._qualify_join_expr(
                                        self._effective_agg_value_expr(it, record_prefix=None)
                                    )
                                    for it in work_items
                                    if it.is_aggregate
                                },
                                extras={
                                    it.alias.lower(): self._qualify_join_expr(it.raw_expr)
                                    for it in work_items
                                    if not it.is_aggregate and it.alias.lower() not in pg_group_aliases
                                },
                            )
                        )

                reconcile_body_sql = self._pg_aggregate_body(
                    pg_spec,
                    ie_sources,
                    "UPDATE",
                    pre_stmts=[pg_view_lock],
                    force_recompute_extremes=True,
                )
                reconcile_fn = self._pg_reconcile_fn_name()
                all_stg_tables = [stg_table_by_idx[i] for i in range(n_joins)]
                reconcile_ddl = self._pg_reconcile_fn_ddl(
                    reconcile_fn, pg_lock_tbl, all_stg_tables, reconcile_body_sql
                )
                trigger_ddls.append(reconcile_ddl)
                fn_names.append(reconcile_fn)

                trunc_cleanup_sql = "\n".join(
                    f"DELETE FROM {st} WHERE __mv_tx = pg_catalog.txid_current() AND __mv_seq = __mv_bseq;"
                    for st in all_stg_tables
                )
                for idx, jj in enumerate(self.view.joins):
                    st = stg_table_by_idx[idx]
                    q_cols = [self._q(c) for c in staged_cols_by_idx[idx]]
                    ins_cols = ["__mv_tx", "__mv_seq", SIGN, *q_cols]
                    sel_cols = [
                        "pg_catalog.txid_current()",
                        "__mv_bseq",
                        f"__mv_src.{SIGN}",
                        *[f"__mv_src.{c}" for c in q_cols],
                    ]
                    stage_sqls = {
                        ev: (
                            f"INSERT INTO {st} ({', '.join(ins_cols)})\n"
                            f"            SELECT {', '.join(sel_cols)}\n"
                            f"            FROM (\n"
                            f"            {self._pg_signed_rows_sql(ev, jj.table_name)}\n"
                            f"            ) AS __mv_src;"
                        )
                        for ev in PG_EVENTS
                    }
                    pg_brackets_by_idx[idx] = PgStatementBracket(
                        lock_table=pg_lock_tbl,
                        source_bit=1 << idx,
                        reconcile_fn=reconcile_fn,
                        stage_sqls=stage_sqls,
                        trunc_cleanup_sql=trunc_cleanup_sql,
                        overlap_lock_sql=pg_view_lock,
                    )
            else:
                for idx in range(len(self.view.joins)):
                    src_bit = (1 << idx) if idx < 60 else ((1 << 60) | (1 << 61))
                    pg_brackets_by_idx[idx] = PgStatementBracket(
                        lock_table=pg_lock_tbl,
                        source_bit=src_bit,
                        overlap_refresh_sql=pg_full_refresh_sql,
                        overlap_lock_sql=pg_view_lock,
                    )

        for j_idx, j in enumerate(self.view.joins):
            base_t = j.table_name
            j_alias = j.alias
            from_new = _build_join_from_clause(replace_alias=j_alias, rec_prefix="NEW")
            from_old = _build_join_from_clause(replace_alias=j_alias, rec_prefix="OLD")

            delta_pk_match = " AND ".join(
                f"{mv_table}.{self._q(c)} = __mv_delta.{self._q(c)}" for c in pk_cols
            )
            q_pk_cols_csv = ", ".join(self._q(c) for c in pk_cols)
            if self.dialect == Dialect.MYSQL:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {from_new}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ON DUPLICATE KEY UPDATE\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    INNER JOIN (\n"
                    f"        SELECT {aliased_delta_select}\n"
                    f"        FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta ON {delta_pk_match}\n"
                    f"    SET {', '.join(mysql_ordered_minus)};"
                )
            else:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {from_new}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ON CONFLICT ({q_pk_cols_csv}) DO UPDATE SET\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    SET {', '.join(ordered_minus)}\n"
                    f"    FROM (\n"
                    f"        SELECT {aliased_delta_select}\n"
                    f"        FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta\n"
                    f"    WHERE {delta_pk_match};"
                )

            del_zero_stmt = (
                f"DELETE FROM {mv_table} WHERE {mv_table}.{self._q(row_count_col)} <= 0;"
                if grp_exprs
                else None
            )
            refresh_grp_stmts: List[str] = []
            if self.dialect == Dialect.SQLITE and grp_exprs:
                seen_rcols: Set[str] = set()
                for item in work_items:
                    if item.is_aggregate:
                        continue
                    q_col = self._q(item.alias)
                    if q_col in seen_rcols:
                        continue
                    seen_rcols.add(q_col)
                    q_sel_e = self._qualify_join_expr(item.raw_expr)
                    c_def = self._lookup_col_def(q_sel_e) or self._lookup_col_def(item.raw_expr)
                    if c_def is not None:
                        aff = _sqlite_type_affinity(c_def.data_type)
                        if aff in ("TEXT", "REAL"):
                            continue
                        if aff in ("INTEGER", "NUMERIC"):
                            val_guard = (
                                f"__mv_delta.{q_col} = -9223372036854775808 "
                                f"AND typeof({mv_table}.{q_col}) = typeof(__mv_delta.{q_col})"
                            )
                        else:
                            val_guard = (
                                f"typeof(__mv_delta.{q_col}) IN ('integer', 'real') "
                                f"AND typeof({mv_table}.{q_col}) = typeof(__mv_delta.{q_col})"
                            )
                    else:
                        val_guard = (
                            f"typeof(__mv_delta.{q_col}) IN ('integer', 'real') "
                            f"AND typeof({mv_table}.{q_col}) = typeof(__mv_delta.{q_col})"
                        )
                    sub_conds: List[str] = []
                    if qual_where:
                        sub_conds.append(f"({qual_where})")
                    for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
                        mc = self._q(self._group_col_alias(g_raw))
                        sub_conds.append(f"({g_qual}) IS ({mv_table}.{mc})")
                    sub_w = f" WHERE {' AND '.join(sub_conds)}" if sub_conds else ""
                    refresh_grp_stmts.append(
                        f"UPDATE {mv_table}\n"
                        f"    SET {q_col} = COALESCE((SELECT {q_sel_e} FROM {full_from_sql}{sub_w} LIMIT 1), {mv_table}.{q_col})\n"
                        f"    WHERE EXISTS (\n"
                        f"        SELECT 1 FROM (\n"
                        f"            SELECT {aliased_delta_select}\n"
                        f"            FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                        f"        ) AS __mv_delta\n"
                        f"        WHERE {delta_pk_match} AND {val_guard}\n"
                        f"    );"
                    )
            del_stmts = [minus_stmt] + ([del_zero_stmt] if del_zero_stmt else []) + refresh_grp_stmts
            trunc_stmts = [f"DELETE FROM {mv_table};"] + (
                [initial_populate_sql] if not grp_exprs else []
            )

            pg_ai_body: Optional[str] = None
            pg_ad_body: Optional[str] = None
            pg_au_body: Optional[str] = None

            if self.dialect == Dialect.POSTGRES:
                # Serialize concurrent writers across different join sources via the
                # per-view lock row before joining the transition delta against the
                # other sources' tables. Under READ COMMITTED, the join statement runs
                # next with a fresh snapshot that includes every committed writer of
                # the other sources; under REPEATABLE READ / SERIALIZABLE, a concurrent
                # write to any source fails closed with 40001 on the lock row.
                pg_view_lock = self._pg_view_lock_stmt()
                trunc_stmts = [pg_view_lock, f"DELETE FROM {mv_table};", initial_populate_sql]
                pg_bodies = self._pg_aggregate_bodies(
                    pg_spec, [_pg_join_source(j_alias)], pre_stmts=[pg_view_lock], base_t=base_t
                )
                pg_ai_body, pg_au_body, pg_ad_body = pg_bodies["INSERT"], pg_bodies["UPDATE"], pg_bodies["DELETE"]

            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=[plus_stmt],
                del_stmts=del_stmts,
                upd_old_stmts=[minus_stmt],
                upd_new_stmts=[plus_stmt],
                upd_cleanup_old_stmts=([del_zero_stmt] if del_zero_stmt else []) + refresh_grp_stmts,
                trunc_stmts=trunc_stmts,
                pg_ai_body=pg_ai_body,
                pg_ad_body=pg_ad_body,
                pg_au_body=pg_au_body,
                pg_bracket=pg_brackets_by_idx.get(j_idx),
            )
            trigger_ddls.extend(t_ddls)
            trig_names.extend(t_ns)
            fn_names.extend(f_ns)

        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=aux_table_ddls,
            auxiliary_table_names=aux_table_names,
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

