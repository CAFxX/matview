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
    parse_sql_ast,
    parse_view_ast,
    quote_sql_identifier,
)


_JOIN_INTEGER_COL_TYPE_RE = re.compile(
    r"^(?:TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT|INT2|INT4|INT8)(?:\s*\(\s*\d+\s*\))?(?:\s+UNSIGNED)?(?:\s+ZEROFILL)?$",
    re.IGNORECASE,
)


class Tier1JoinMixin(MixinHost):
    def _substitute_join_alias_with_record(
        self, expr: str, target_alias: str, rec_prefix: str
    ) -> Optional[str]:
        """Replace qualified `<target_alias>.<col>` references with `<rec_prefix>.<col>`.

        Returns ``None`` if the expression cannot be safely tokenized or
        contains a nested ``SELECT`` subquery where alias shadowing could apply.
        """
        if not expr:
            return expr
        try:
            toks = _tokenize_sql(expr, read=_sqlglot_dialect(self.dialect))
        except Exception:
            return None
        if any(tk.token_type == SQLGlotTokenType.SELECT for tk in toks):
            return None
        target_low = target_alias.lower()
        out: List[str] = []
        i = 0
        n = len(toks)
        cursor = 0
        while i < n:
            tk = toks[i]
            if (
                _is_word_or_ident_token(tk)
                and tk.text.lower() == target_low
                and (i == 0 or toks[i - 1].token_type != SQLGlotTokenType.DOT)
                and i + 2 < n
                and toks[i + 1].token_type == SQLGlotTokenType.DOT
                and _is_word_or_ident_token(toks[i + 2])
            ):
                if tk.start > cursor:
                    out.append(expr[cursor:tk.start])
                out.append(f"{rec_prefix}.{self._q(toks[i + 2].text)}")
                cursor = toks[i + 2].end + 1
                i += 3
                continue
            i += 1
        if cursor < len(expr):
            out.append(expr[cursor:])
        return "".join(out)

    def _join_expr_referenced_aliases(self, qual_expr: str) -> Optional[Set[str]]:
        """Return lowercase join aliases referenced via `<alias>.<col>` in `qual_expr`."""
        if not qual_expr:
            return set()
        known_aliases = {j.alias.lower() for j in self.view.joins}
        try:
            toks = _tokenize_sql(qual_expr, read=_sqlglot_dialect(self.dialect))
        except Exception:
            return None
        if any(tk.token_type == SQLGlotTokenType.SELECT for tk in toks):
            return None
        found: Set[str] = set()
        n = len(toks)
        for i in range(n - 2):
            if (
                _is_word_or_ident_token(toks[i])
                and (i == 0 or toks[i - 1].token_type != SQLGlotTokenType.DOT)
                and toks[i + 1].token_type == SQLGlotTokenType.DOT
                and _is_word_or_ident_token(toks[i + 2])
            ):
                al = toks[i].text.lower()
                if al in known_aliases:
                    found.add(al)
        return found

    def _is_unique_pk_lookup_join(
        self,
        j_idx: int,
        other_j: JoinSpec,
        combined_qual_conds: Sequence[str],
    ) -> bool:
        """True iff a 2-table INNER JOIN constrains every PK column of `other_j`
        to equal an integer column of the firing table (guaranteeing 0 or 1
        matched rows per trigger invocation with no cross-type 2-to-1 collisions)."""
        if len(self.view.joins) != 2:
            return False
        firing_j = self.view.joins[j_idx]
        other_schema = self._schemas_lower.get(other_j.table_name.lower())
        firing_schema = self._schemas_lower.get(firing_j.table_name.lower())
        if other_schema is None or firing_schema is None:
            return False
        other_pk_cols = [c for c, cd in other_schema.columns.items() if cd.primary_key]
        if not other_pk_cols:
            return False
        other_cols_low = {c.lower(): cd for c, cd in other_schema.columns.items()}
        firing_cols_low = {c.lower(): cd for c, cd in firing_schema.columns.items()}
        other_al_low = other_j.alias.lower()
        firing_al_low = firing_j.alias.lower()
        if other_al_low == firing_al_low:
            return False

        matched_pks: Set[str] = set()

        def _collect_eqs(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, sqlglot_exp.Paren):
                _collect_eqs(node.this)
                return
            if isinstance(node, sqlglot_exp.And):
                _collect_eqs(node.this)
                _collect_eqs(node.expression)
                return
            if isinstance(node, sqlglot_exp.EQ):
                lhs, rhs = node.this, node.expression
                if isinstance(lhs, sqlglot_exp.Paren):
                    lhs = lhs.this
                if isinstance(rhs, sqlglot_exp.Paren):
                    rhs = rhs.this
                if isinstance(lhs, sqlglot_exp.Column) and isinstance(rhs, sqlglot_exp.Column):
                    l_tbl = (lhs.table or "").lower()
                    r_tbl = (rhs.table or "").lower()
                    l_col = (lhs.name or "").lower()
                    r_col = (rhs.name or "").lower()
                    if l_tbl == other_al_low and r_tbl == firing_al_low:
                        pk_cd = other_cols_low.get(l_col)
                        src_cd = firing_cols_low.get(r_col)
                        if (
                            pk_cd is not None
                            and pk_cd.primary_key
                            and src_cd is not None
                            and _JOIN_INTEGER_COL_TYPE_RE.fullmatch(pk_cd.data_type.strip())
                            and _JOIN_INTEGER_COL_TYPE_RE.fullmatch(src_cd.data_type.strip())
                        ):
                            matched_pks.add(l_col)
                    elif r_tbl == other_al_low and l_tbl == firing_al_low:
                        pk_cd = other_cols_low.get(r_col)
                        src_cd = firing_cols_low.get(l_col)
                        if (
                            pk_cd is not None
                            and pk_cd.primary_key
                            and src_cd is not None
                            and _JOIN_INTEGER_COL_TYPE_RE.fullmatch(pk_cd.data_type.strip())
                            and _JOIN_INTEGER_COL_TYPE_RE.fullmatch(src_cd.data_type.strip())
                        ):
                            matched_pks.add(r_col)

        for cond in combined_qual_conds:
            try:
                ast = parse_sql_ast(cond, dialect=self.dialect)
            except Exception:
                continue
            _collect_eqs(ast)
        return all(c.lower() in matched_pks for c in other_pk_cols)
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
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
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

        def _build_join_subquery(select_sql_fn: Any, *, limit_one: bool = False) -> str:
            alias_remap: Dict[str, str] = {}
            for idx_j, j in enumerate(self.view.joins):
                safe_a = f"__mv_j_{idx_j}"
                eff_a = (j.alias or j.table_name).lower()
                alias_remap[eff_a] = safe_a

            sg_d = _sqlglot_dialect(self.dialect)

            def _remap_to_j(frag: str) -> str:
                q_frag = self._qualify_join_expr(frag)
                try:
                    toks = _tokenize_sql(q_frag, read=sg_d)
                except Exception:
                    return q_frag
                if not toks:
                    return q_frag
                out: List[str] = []
                cursor = 0
                i_t = 0
                n_t = len(toks)
                while i_t < n_t:
                    tk = toks[i_t]
                    if (
                        _is_word_or_ident_token(tk)
                        and (i_t == 0 or toks[i_t - 1].token_type != SQLGlotTokenType.DOT)
                        and i_t + 2 < n_t
                        and toks[i_t + 1].token_type == SQLGlotTokenType.DOT
                        and _is_word_or_ident_token(toks[i_t + 2])
                    ):
                        key_low = _unquote_ident(tk.text, dialect=self.dialect).lower()
                        if key_low in alias_remap:
                            if tk.start > cursor:
                                out.append(q_frag[cursor : tk.start])
                            out.append(alias_remap[key_low])
                            cursor = tk.end + 1
                            i_t += 2
                            continue
                    i_t += 1
                if cursor < len(q_frag):
                    out.append(q_frag[cursor:])
                return "".join(out)

            j_parts: List[str] = []
            for idx, j in enumerate(self.view.joins):
                tbl_s = f"{self._q(j.table_name)} AS __mv_j_{idx}"
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
            lim = " LIMIT 1" if limit_one else ""
            sel_e = select_sql_fn(_remap_to_j)
            return f"SELECT {sel_e} FROM {' '.join(j_parts)}{w_sub}{lim}"

        def _build_join_recompute_subquery(item: SelectItem) -> str:
            return f"({_build_join_subquery(lambda r: r(self._sql_backfill_agg_expr(item)))})"

        def _build_join_group_keys_recompute(exprs: Sequence[str]) -> str:
            return _build_join_subquery(
                lambda r: ", ".join(f"{r(e)} AS __mv_l{idx}" for idx, e in enumerate(exprs)),
                limit_one=True,
            )

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
            recompute_group_keys=_build_join_group_keys_recompute,
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

        can_one_row_aggs = all(
            (not it.is_aggregate)
            or (
                not it.is_distinct
                and it.agg_func
                in (
                    AggFunc.COUNT_STAR,
                    AggFunc.COUNT_COL,
                    AggFunc.SUM,
                    AggFunc.TOTAL,
                    AggFunc.MIN,
                    AggFunc.MAX,
                )
            )
            for it in work_items
        )
        all_inner_joins = (
            len(self.view.joins) >= 2
            and all("." not in jj.table_name for jj in self.view.joins)
            and all(
                jj.join_type.upper() in ("INNER JOIN", "JOIN", "CROSS JOIN")
                for jj in self.view.joins[1:]
            )
        )

        def _build_one_row_select_exprs(
            target_alias: str, rec_prefix: str
        ) -> Optional[List[str]]:
            out_exprs: List[str] = []

            def _sub(e: str) -> Optional[str]:
                return self._substitute_join_alias_with_record(e, target_alias, rec_prefix)

            if not grp_exprs:
                out_exprs.append("1")
            elif has_nullable_grp:
                s_pk = _sub(self._canonical_group_pk_expr(qual_grp_exprs))
                if s_pk is None:
                    return None
                out_exprs.append(s_pk)

            seen_h: Set[str] = set()
            for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
                mv_col = self._group_col_alias(g_raw)
                if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_h:
                    seen_h.add(mv_col.lower())
                    sg = _sub(g_qual)
                    if sg is None:
                        return None
                    out_exprs.append(sg)

            for item in work_items:
                if not item.is_aggregate:
                    s_raw = _sub(self._qualify_join_expr(item.raw_expr))
                    if s_raw is None:
                        return None
                    out_exprs.append(s_raw)
                    continue
                eff_q = self._qualify_join_expr(
                    self._effective_agg_value_expr(item, record_prefix=None)
                )
                eff_s = _sub(eff_q)
                if eff_s is None:
                    return None
                if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                    out_exprs.append("1")
                elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    out_exprs.append(f"(CASE WHEN ({eff_s}) IS NOT NULL THEN 1 ELSE 0 END)")
                elif item.agg_func == AggFunc.SUM:
                    if self.dialect == Dialect.SQLITE:
                        # Preserve SQLite native SUM conversion on TEXT/BLOB/-0.0
                        # when creating a new group while staying O(1) for integers.
                        # Wrap in FROM (SELECT (...) AS __v) so referencing the
                        # joined outer table cannot turn the enclosing SELECT into
                        # an ungrouped aggregate when 0 rows match.
                        out_exprs.append(
                            f"(CASE WHEN typeof({eff_s}) IN ('integer', 'null') "
                            f"THEN ({eff_s}) ELSE (SELECT SUM(__v) FROM (SELECT ({eff_s}) AS __v)) END)"
                        )
                    else:
                        out_exprs.append(f"({eff_s})")
                elif item.agg_func == AggFunc.TOTAL:
                    if self.dialect == Dialect.SQLITE:
                        out_exprs.append(
                            f"(CASE WHEN typeof({eff_s}) = 'integer' THEN 1.0 * ({eff_s}) "
                            f"WHEN ({eff_s}) IS NULL THEN 0.0 "
                            f"ELSE (SELECT TOTAL(__v) FROM (SELECT ({eff_s}) AS __v)) END)"
                        )
                    else:
                        out_exprs.append(
                            f"(CASE WHEN ({eff_s}) IS NULL THEN 0.0 ELSE 1.0 * ({eff_s}) END)"
                        )
                elif item.agg_func in (AggFunc.MIN, AggFunc.MAX):
                    out_exprs.append(f"({eff_s})")
                else:
                    return None

            if count_star_alias is None:
                out_exprs.append("1")

            already_q = {self._q(c) for c in target_cols[: len(out_exprs)]}
            for ekey, cnt_col in col_nonnull_count_map.items():
                qc = self._q(cnt_col)
                if qc not in already_q:
                    it = ekey_to_item[ekey]
                    eff_s = _sub(
                        self._qualify_join_expr(
                            self._effective_agg_value_expr(it, record_prefix=None)
                        )
                    )
                    if eff_s is None:
                        return None
                    out_exprs.append(f"(CASE WHEN ({eff_s}) IS NOT NULL THEN 1 ELSE 0 END)")
                    already_q.add(qc)
            for ekey, sum_col in col_sum_map.items():
                qs = self._q(sum_col)
                if qs not in already_q:
                    it = ekey_to_item[ekey]
                    eff_s = _sub(
                        self._qualify_join_expr(
                            self._effective_agg_value_expr(it, record_prefix=None)
                        )
                    )
                    if eff_s is None:
                        return None
                    out_exprs.append(f"({eff_s})")
                    already_q.add(qs)
            if len(out_exprs) != len(q_target_cols):
                return None
            return out_exprs

        def _substitute_delta_refs(
            assignments: Sequence[str], col_expr_map: Dict[str, str]
        ) -> List[str]:
            sorted_cols = sorted(col_expr_map.keys(), key=len, reverse=True)
            out_a: List[str] = []
            for stmt_a in assignments:
                cur = stmt_a
                for qc in sorted_cols:
                    expr_rep = f"({col_expr_map[qc]})"
                    if qc and (qc[0] in ('"', "`", "[")):
                        cur = cur.replace(f"__mv_delta.{qc}", expr_rep)
                    else:
                        cur = re.sub(
                            rf"\b__mv_delta\.{re.escape(qc)}\b",
                            lambda _m, rep=expr_rep: rep,
                            cur,
                        )
                out_a.append(cur)
            return out_a

        for j_idx, j in enumerate(self.view.joins):
            base_t = j.table_name
            j_alias = j.alias

            delta_pk_match = " AND ".join(
                f"{mv_table}.{self._q(c)} = __mv_delta.{self._q(c)}" for c in pk_cols
            )
            q_pk_cols_csv = ", ".join(self._q(c) for c in pk_cols)

            # On SQLite/MySQL/MariaDB, eliminate the single-row derived table
            # `(SELECT NEW/OLD... [FROM DUAL]) AS <j_alias>` by inlining
            # `<j_alias>.<col>` as `NEW.<col>` / `OLD.<col>` and moving join
            # predicates referencing `<j_alias>` into WHERE.
            use_inline = False
            from_new_eff = ""
            from_old_eff = ""
            where_new_eff = where_sql
            where_old_eff = where_sql
            where_new_conds: List[str] = []
            where_old_conds: List[str] = []
            sel_new_eff = select_exprs
            sel_old_eff = select_exprs
            grp_new_sql = group_sql
            grp_old_sql = group_sql
            combined_qual_conds: List[str] = []

            if self.dialect in (Dialect.SQLITE, Dialect.MYSQL) and all_inner_joins:
                other_joins = [
                    (idx_o, jj) for idx_o, jj in enumerate(self.view.joins) if idx_o != j_idx
                ]
                raw_extra_conds: List[str] = []
                if j_idx > 0 and j.on_condition:
                    raw_extra_conds.append(self._qualify_join_expr(j.on_condition))
                if j_idx == 0 and other_joins[0][1].on_condition:
                    raw_extra_conds.append(self._qualify_join_expr(other_joins[0][1].on_condition))
                other_parts: List[str] = [
                    f"{self._q(other_joins[0][1].table_name)} AS {self._q(other_joins[0][1].alias)}"
                ]
                inline_ok = True
                for _, oj in other_joins[1:]:
                    if oj.on_condition:
                        q_on = self._qualify_join_expr(oj.on_condition)
                        refs = self._join_expr_referenced_aliases(q_on)
                        if refs is None:
                            inline_ok = False
                            break
                        if j_alias.lower() in refs:
                            raw_extra_conds.append(q_on)
                            # Use INNER JOIN ... ON 1 = 1 rather than CROSS JOIN
                            # so SQLite's planner remains free to reorder joins
                            # when an indexed predicate from NEW/OLD targets `oj`.
                            other_parts.append(
                                f"INNER JOIN {self._q(oj.table_name)} AS {self._q(oj.alias)} ON 1 = 1"
                            )
                        else:
                            other_parts.append(
                                f"{oj.join_type} {self._q(oj.table_name)} AS {self._q(oj.alias)} ON {q_on}"
                            )
                    else:
                        other_parts.append(
                            f"{oj.join_type} {self._q(oj.table_name)} AS {self._q(oj.alias)}"
                        )
                if qual_where:
                    raw_extra_conds.append(qual_where)
                combined_qual_conds = list(raw_extra_conds)

                if inline_ok:
                    sub_w_new = [
                        self._substitute_join_alias_with_record(c, j_alias, "NEW")
                        for c in raw_extra_conds
                    ]
                    sub_w_old = [
                        self._substitute_join_alias_with_record(c, j_alias, "OLD")
                        for c in raw_extra_conds
                    ]
                    sub_s_new = [
                        self._substitute_join_alias_with_record(ex, j_alias, "NEW")
                        for ex in select_exprs
                    ]
                    sub_s_old = [
                        self._substitute_join_alias_with_record(ex, j_alias, "OLD")
                        for ex in select_exprs
                    ]
                    sub_g_new = [
                        self._substitute_join_alias_with_record(g, j_alias, "NEW")
                        for g in qual_grp_for_q
                    ]
                    sub_g_old = [
                        self._substitute_join_alias_with_record(g, j_alias, "OLD")
                        for g in qual_grp_for_q
                    ]
                    if (
                        all(x is not None for x in sub_w_new)
                        and all(x is not None for x in sub_w_old)
                        and all(x is not None for x in sub_s_new)
                        and all(x is not None for x in sub_s_old)
                        and all(x is not None for x in sub_g_new)
                        and all(x is not None for x in sub_g_old)
                    ):
                        use_inline = True
                        from_new_eff = " ".join(other_parts)
                        from_old_eff = from_new_eff
                        where_new_conds = [x for x in sub_w_new if x is not None]
                        where_old_conds = [x for x in sub_w_old if x is not None]
                        where_new_eff = (
                            f"\nWHERE {' AND '.join(f'({c})' for c in where_new_conds)}"
                            if where_new_conds
                            else ""
                        )
                        where_old_eff = (
                            f"\nWHERE {' AND '.join(f'({c})' for c in where_old_conds)}"
                            if where_old_conds
                            else ""
                        )
                        sel_new_eff = [x for x in sub_s_new if x is not None]
                        sel_old_eff = [x for x in sub_s_old if x is not None]
                        grp_new_sql = (
                            f"\nGROUP BY {', '.join(x for x in sub_g_new if x is not None)}"
                            if sub_g_new
                            else ""
                        )
                        grp_old_sql = (
                            f"\nGROUP BY {', '.join(x for x in sub_g_old if x is not None)}"
                            if sub_g_old
                            else ""
                        )

            if not use_inline:
                from_new_eff = _build_join_from_clause(replace_alias=j_alias, rec_prefix="NEW")
                from_old_eff = _build_join_from_clause(replace_alias=j_alias, rec_prefix="OLD")

            # Further optimize 2-table unique-PK-lookup joins (0 or 1 matching
            # row per trigger invocation): drop GROUP BY and the __mv_delta
            # derived table completely.
            one_row_new: Optional[List[str]] = None
            one_row_old: Optional[List[str]] = None
            if (
                use_inline
                and can_one_row_aggs
                and len(self.view.joins) == 2
                and self._is_unique_pk_lookup_join(
                    j_idx,
                    self.view.joins[1 - j_idx],
                    combined_qual_conds,
                )
            ):
                one_row_new = _build_one_row_select_exprs(j_alias, "NEW")
                one_row_old = _build_one_row_select_exprs(j_alias, "OLD")

            aliased_delta_old = ", ".join(
                f"{ex} AS {col}" for col, ex in zip(q_target_cols, sel_old_eff)
            )

            if one_row_new is not None and one_row_old is not None:
                one_row_map_old = dict(zip(q_target_cols, one_row_old))
                direct_minus = _substitute_delta_refs(ordered_minus, one_row_map_old)
                direct_pk_match_old = " AND ".join(
                    f"{mv_table}.{self._q(c)} = ({one_row_map_old[self._q(c)]})"
                    for c in pk_cols
                )
                where_old_and = (
                    " AND ".join(f"({c})" for c in where_old_conds) + " AND "
                    if where_old_conds
                    else ""
                )
                if self.dialect == Dialect.MYSQL:
                    plus_stmt = (
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"    SELECT {', '.join(one_row_new)}\n"
                        f"    FROM {from_new_eff}{where_new_eff}\n"
                        f"    ON DUPLICATE KEY UPDATE\n"
                        f"        {', '.join(plus_merge_assignments)};"
                    )
                    minus_stmt = (
                        f"UPDATE {mv_table}\n"
                        f"    INNER JOIN {from_old_eff} ON {direct_pk_match_old}\n"
                        f"    SET {', '.join(f'{mv_table}.{a}' for a in direct_minus)}"
                        f"{where_old_eff};"
                    )
                else:
                    plus_stmt = (
                        f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                        f"    SELECT {', '.join(one_row_new)}\n"
                        f"    FROM {from_new_eff}{where_new_eff}\n"
                        f"    ON CONFLICT ({q_pk_cols_csv}) DO UPDATE SET\n"
                        f"        {', '.join(plus_merge_assignments)};"
                    )
                    minus_stmt = (
                        f"UPDATE {mv_table}\n"
                        f"    SET {', '.join(direct_minus)}\n"
                        f"    FROM {from_old_eff}\n"
                        f"    WHERE {where_old_and}{direct_pk_match_old};"
                    )
            elif self.dialect == Dialect.MYSQL:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"    SELECT {', '.join(sel_new_eff)}\n"
                    f"    FROM {from_new_eff}{where_new_eff}{grp_new_sql}{having_non_empty}\n"
                    f"    ON DUPLICATE KEY UPDATE\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    INNER JOIN (\n"
                    f"        SELECT {aliased_delta_old}\n"
                    f"        FROM {from_old_eff}{where_old_eff}{grp_old_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta ON {delta_pk_match}\n"
                    f"    SET {', '.join(mysql_ordered_minus)};"
                )
            else:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"    SELECT {', '.join(sel_new_eff)}\n"
                    f"    FROM {from_new_eff}{where_new_eff}{grp_new_sql}{having_non_empty}\n"
                    f"    ON CONFLICT ({q_pk_cols_csv}) DO UPDATE SET\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    SET {', '.join(ordered_minus)}\n"
                    f"    FROM (\n"
                    f"        SELECT {aliased_delta_old}\n"
                    f"        FROM {from_old_eff}{where_old_eff}{grp_old_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta\n"
                    f"    WHERE {delta_pk_match};"
                )

            # Restrict empty-group DELETE to the key(s) touched by OLD so row
            # triggers never full-scan `mv_table` on DELETE or UPDATE.
            del_zero_stmt: Optional[str] = None
            if grp_exprs:
                col_to_sel_old = dict(zip(q_target_cols, sel_old_eff))
                pk_sel_old = [col_to_sel_old[self._q(c)] for c in pk_cols]
                grp_refs = [self._join_expr_referenced_aliases(g) for g in qual_grp_exprs]
                grp_only_firing = all(
                    r is not None and r <= {j_alias.lower()} for r in grp_refs
                )
                direct_old_grp: Optional[List[str]] = None
                if grp_only_firing:
                    sub_grps = [
                        self._substitute_join_alias_with_record(g, j_alias, "OLD")
                        for g in qual_grp_exprs
                    ]
                    if all(x is not None for x in sub_grps):
                        direct_old_grp = [x for x in sub_grps if x is not None]

                if direct_old_grp is not None:
                    if has_nullable_grp:
                        direct_pk_pred = (
                            f"{mv_table}.__mv_pk = ({self._canonical_group_pk_expr(direct_old_grp)})"
                        )
                    else:
                        seen_pk_preds: Set[str] = set()
                        pk_pred_parts: List[str] = []
                        for g_raw, old_g in zip(grp_exprs, direct_old_grp):
                            mc = self._group_col_alias(g_raw)
                            if mc not in seen_pk_preds:
                                seen_pk_preds.add(mc)
                                pk_pred_parts.append(f"{mv_table}.{self._q(mc)} = ({old_g})")
                        direct_pk_pred = " AND ".join(pk_pred_parts)
                    del_zero_stmt = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({direct_pk_pred}) AND {mv_table}.{self._q(row_count_col)} <= 0;"
                    )
                elif self.dialect == Dialect.MYSQL:
                    if one_row_old is not None:
                        one_row_map_old = dict(zip(q_target_cols, one_row_old))
                        direct_pk_match_old = " AND ".join(
                            f"{mv_table}.{self._q(c)} = ({one_row_map_old[self._q(c)]})"
                            for c in pk_cols
                        )
                        where_old_and = (
                            " AND ".join(f"({c})" for c in where_old_conds) + " AND "
                            if where_old_conds
                            else ""
                        )
                        del_zero_stmt = (
                            f"DELETE {mv_table} FROM {mv_table}\n"
                            f"    INNER JOIN {from_old_eff} ON {direct_pk_match_old}\n"
                            f"    WHERE {where_old_and}{mv_table}.{self._q(row_count_col)} <= 0;"
                        )
                    else:
                        pk_delta_sel = ", ".join(
                            f"{ex} AS {self._q(c)}" for c, ex in zip(pk_cols, pk_sel_old)
                        )
                        del_zero_stmt = (
                            f"DELETE {mv_table} FROM {mv_table}\n"
                            f"    INNER JOIN (\n"
                            f"        SELECT DISTINCT {pk_delta_sel}\n"
                            f"        FROM {from_old_eff}{where_old_eff}\n"
                            f"    ) AS __mv_delta ON {delta_pk_match}\n"
                            f"    WHERE {mv_table}.{self._q(row_count_col)} <= 0;"
                        )
                else:
                    if len(pk_cols) == 1:
                        pk_in_lhs = f"{mv_table}.{self._q(pk_cols[0])}"
                    else:
                        pk_in_lhs = (
                            "("
                            + ", ".join(f"{mv_table}.{self._q(c)}" for c in pk_cols)
                            + ")"
                        )
                    del_zero_stmt = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE {pk_in_lhs} IN (\n"
                        f"        SELECT {', '.join(pk_sel_old)}\n"
                        f"        FROM {from_old_eff}{where_old_eff}\n"
                        f"    ) AND {mv_table}.{self._q(row_count_col)} <= 0;"
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
                        is_int_num_aff = aff in ("INTEGER", "NUMERIC")
                    else:
                        is_int_num_aff = False
                    sub_conds: List[str] = []
                    if qual_where:
                        sub_conds.append(f"({qual_where})")
                    for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
                        mc = self._q(self._group_col_alias(g_raw))
                        sub_conds.append(f"({g_qual}) IS ({mv_table}.{mc})")
                    sub_w = f" WHERE {' AND '.join(sub_conds)}" if sub_conds else ""
                    if one_row_old is not None:
                        one_row_map_old = dict(zip(q_target_cols, one_row_old))
                        old_val_expr = one_row_map_old[q_col]
                        direct_pk_match_old = " AND ".join(
                            f"{mv_table}.{self._q(c)} = ({one_row_map_old[self._q(c)]})"
                            for c in pk_cols
                        )
                        if is_int_num_aff:
                            val_guard_direct = (
                                f"({old_val_expr}) = -9223372036854775808 "
                                f"AND typeof({mv_table}.{q_col}) = typeof({old_val_expr})"
                            )
                        else:
                            val_guard_direct = (
                                f"typeof({old_val_expr}) IN ('integer', 'real') "
                                f"AND typeof({mv_table}.{q_col}) = typeof({old_val_expr})"
                            )
                        where_old_and = (
                            " AND ".join(f"({c})" for c in where_old_conds) + " AND "
                            if where_old_conds
                            else ""
                        )
                        refresh_grp_stmts.append(
                            f"UPDATE {mv_table}\n"
                            f"    SET {q_col} = COALESCE((SELECT {q_sel_e} FROM {full_from_sql}{sub_w} LIMIT 1), {mv_table}.{q_col})\n"
                            f"    FROM {from_old_eff}\n"
                            f"    WHERE ({val_guard_direct}) AND {where_old_and}{direct_pk_match_old};"
                        )
                    else:
                        if is_int_num_aff:
                            val_guard = (
                                f"__mv_delta.{q_col} = -9223372036854775808 "
                                f"AND typeof({mv_table}.{q_col}) = typeof(__mv_delta.{q_col})"
                            )
                        else:
                            val_guard = (
                                f"typeof(__mv_delta.{q_col}) IN ('integer', 'real') "
                                f"AND typeof({mv_table}.{q_col}) = typeof(__mv_delta.{q_col})"
                            )
                        refresh_grp_stmts.append(
                            f"UPDATE {mv_table}\n"
                            f"    SET {q_col} = COALESCE((SELECT {q_sel_e} FROM {full_from_sql}{sub_w} LIMIT 1), {mv_table}.{q_col})\n"
                            f"    FROM (\n"
                            f"        SELECT {aliased_delta_old}\n"
                            f"        FROM {from_old_eff}{where_old_eff}{grp_old_sql}{having_non_empty}\n"
                            f"    ) AS __mv_delta\n"
                            f"    WHERE {delta_pk_match} AND {val_guard};"
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

