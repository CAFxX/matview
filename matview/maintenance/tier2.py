"""Tier 2: Group-scoped delta recomputation rules."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .._host import MixinHost
from ..dialects.pg_statement import signed_rows_sql as pg_signed_rows_sql
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
    _find_top_level_token,
    _has_subquery_ast,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _parse_single_agg_expr,
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


class Tier2Mixin(MixinHost):
    def _can_push_down_group_filter_into_where(self) -> bool:
        if not self.view.where_clause:
            return True
        return not _has_subquery_ast(self.view.where_clause, dialect=self.dialect)

    def _inject_where_into_source_query(self, source_query_sql: str, extra_predicate: str) -> str:
        s = source_query_sql.strip().rstrip(";")
        tok_where = _find_top_level_token(s, (SQLGlotTokenType.WHERE,), dialect=self.dialect)
        tok_group = _find_top_level_token(s, (SQLGlotTokenType.GROUP_BY,), dialect=self.dialect)
        if tok_where is not None:
            insert_pos = tok_where.end + 1
            before = s[: tok_where.start].rstrip()
            if tok_group is not None:
                existing_w = s[insert_pos : tok_group.start].strip()
                after = s[tok_group.start :]
                return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})\n{after}"
            existing_w = s[insert_pos:].strip()
            return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})"
        if tok_group is not None:
            before = s[: tok_group.start].rstrip()
            after = s[tok_group.start :]
            return f"{before}\nWHERE ({extra_predicate})\n{after}"
        return f"{s}\nWHERE ({extra_predicate})"

    def _group_scope_predicate_for_table(
        self,
        base_t: str,
        group_mv_cols: List[Tuple[str, str]],
        lhs_table_alias: Optional[str],
        record_prefix: str,
    ) -> Optional[str]:
        if not group_mv_cols:
            return None

        if len(self.view.joins) <= 1 and base_t.lower() != self.view.base_table.lower():
            return None
        if self.view.where_clause and re.search(r"\b(?:EXISTS|IN)\s*\(", self.view.where_clause, flags=re.IGNORECASE):
            if base_t.lower() != self.view.base_table.lower():
                return None

        def _lhs_expr(g_expr: str, mv_col: str) -> str:
            if lhs_table_alias:
                return f"{lhs_table_alias}.{self._q(mv_col)}"
            if record_prefix in ("__mv_new_rows", "__mv_old_rows") and self.view.joins:
                outer_alias = self._q(self.view.joins[0].alias or base_t)
                return f"({self._rewrite_expr_for_record(g_expr, outer_alias)})"
            return f"({g_expr})"

        if len(self.view.joins) == 1:
            conds: List[str] = []
            for g_expr, mv_col in group_mv_cols:
                rhs_e = self._rewrite_expr_for_record(g_expr, record_prefix)
                conds.append(self._null_safe_eq(_lhs_expr(g_expr, mv_col), f"({rhs_e})", is_nullable=True))
            cond_str = " AND ".join(conds)
            if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                return f"EXISTS (SELECT 1 FROM {record_prefix} WHERE {cond_str})"
            return cond_str

        target_join: Optional[JoinSpec] = None
        for j in self.view.joins:
            if j.table_name.lower() == base_t.lower():
                target_join = j
                break
        if target_join is None:
            return None

        # If base_t is the table that owns all group columns directly, compare directly
        target_alias = target_join.alias
        all_owned = True
        direct_conds: List[str] = []
        ts_target = self._schemas_lower[base_t.lower()]
        for g_expr, mv_col in group_mv_cols:
            gs = g_expr.strip()
            if "." in gs:
                p_al, c_part = gs.split(".", 1)
                if p_al.strip().lower() in (target_alias.lower(), base_t.lower()):
                    direct_conds.append(
                        self._null_safe_eq(
                            _lhs_expr(g_expr, mv_col),
                            f"({record_prefix}.{self._q(_unquote_ident(c_part, dialect=self.dialect))})",
                            is_nullable=True,
                        )
                    )
                else:
                    all_owned = False
                    break
            elif _unquote_ident(gs, dialect=self.dialect) in ts_target.columns:
                direct_conds.append(
                    self._null_safe_eq(
                        _lhs_expr(g_expr, mv_col),
                        f"({record_prefix}.{self._q(_unquote_ident(gs, dialect=self.dialect))})",
                        is_nullable=True,
                    )
                )
            else:
                all_owned = False
                break

        if all_owned and direct_conds:
            cond_str = " AND ".join(direct_conds)
            if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                return f"EXISTS (SELECT 1 FROM {record_prefix} WHERE {cond_str})"
            return cond_str

        # Otherwise build an EXISTS subquery over the join graph with `target_join` replaced by `(SELECT NEW/OLD.*)`
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        rec_proj = ", ".join(f"{record_prefix}.{self._q(c)} AS {self._q(c)}" for c in ts_target.columns)
        alias_remap: Dict[str, str] = {}
        for j in self.view.joins:
            alias_remap[j.alias.lower()] = f"__mv_j_{j.alias}"
            alias_remap[j.table_name.lower()] = f"__mv_j_{j.alias}"

        def _remap_aliases_in_sql(frag: str) -> str:
            toks = _tokenize_sql(frag, read=_sqlglot_dialect(self.dialect))
            if not toks:
                return frag
            out: List[str] = []
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
                    tbl_low = tk.text.lower()
                    if tbl_low in alias_remap:
                        if tk.start > cursor:
                            out.append(frag[cursor:tk.start])
                        out.append(f"{alias_remap[tbl_low]}.{toks[i + 2].text}")
                        cursor = toks[i + 2].end + 1
                        i += 3
                        continue
                i += 1
            if cursor < len(frag):
                out.append(frag[cursor:])
            return "".join(out)

        if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
            inline_row_table = f"{record_prefix} AS __mv_j_{target_alias}"
        else:
            inline_row_table = f"(SELECT {rec_proj}{from_dual}) AS __mv_j_{target_alias}"
        join_parts: List[str] = []
        for idx, j in enumerate(self.view.joins):
            remapped_alias = f"__mv_j_{j.alias}"
            tbl_expr = (
                inline_row_table
                if j.alias.lower() == target_alias.lower()
                else f"{self._q(j.table_name)} AS {remapped_alias}"
            )
            if idx == 0:
                join_parts.append(tbl_expr)
            else:
                on_sql = (
                    f" ON {_remap_aliases_in_sql(j.on_condition)}"
                    if j.on_condition
                    else ""
                )
                eff_join_type = (
                    "INNER JOIN"
                    if j.alias.lower() == target_alias.lower() and j.on_condition
                    else j.join_type
                )
                join_parts.append(f"{eff_join_type} {tbl_expr}{on_sql}")

        from_join_sql = " ".join(join_parts)
        left_join_remapped = {
            f"__mv_j_{j.alias}".lower()
            for j in self.view.joins
            if j.join_type == "LEFT JOIN"
        }
        match_conds: List[str] = []
        for g_expr, mv_col in group_mv_cols:
            remapped_g = _remap_aliases_in_sql(g_expr)
            lhs_e = _lhs_expr(g_expr, mv_col)
            eq_cond = self._null_safe_eq(lhs_e, f"({remapped_g})", is_nullable=True)
            if any(
                re.search(rf"\b{re.escape(la)}\.", remapped_g, flags=re.IGNORECASE)
                for la in left_join_remapped
            ):
                match_conds.append(f"({eq_cond} OR {lhs_e} IS NULL)")
            else:
                match_conds.append(eq_cond)

        exists_pred = (
            f"EXISTS (\n"
            f"        SELECT 1 FROM {from_join_sql}\n"
            f"        WHERE {' AND '.join(match_conds)}\n"
            f"    )"
        )
        if any(j.join_type == "LEFT JOIN" for j in self.view.joins) and target_join != self.view.joins[0]:  # pyright: ignore[reportGeneralTypeIssues]  # pyright's len() narrowing wrongly infers empty joins here; any() guards it
            null_preds = " AND ".join(
                f"{_lhs_expr(g_expr, mv_col)} IS NULL" for g_expr, mv_col in group_mv_cols
            )
            return f"({exists_pred} OR ({null_preds}))"
        return exists_pred

    def _build_tier2_triggers_for_table(
        self,
        view_name: str,
        mv_table: str,
        base_t: str,
        target_cols: List[str],
        group_mv_cols: List[Tuple[str, str]],
        source_query_sql: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        q_target_cols = [self._q(c) for c in target_cols]
        cols_csv = ", ".join(q_target_cols)
        src_cols_csv = ", ".join(f"__mv_src.{self._q(c)}" for c in target_cols)

        pred_mv_old = self._group_scope_predicate_for_table(
            base_t, group_mv_cols, mv_table, "OLD"
        )
        pred_mv_new = self._group_scope_predicate_for_table(
            base_t, group_mv_cols, mv_table, "NEW"
        )
        can_push_down = self._can_push_down_group_filter_into_where()

        full_refresh = [
            f"DELETE FROM {mv_table};",
            f"INSERT INTO {mv_table} ({cols_csv})\n"
            f"    SELECT {src_cols_csv} FROM (\n{source_query_sql}\n    ) AS __mv_src;",
        ]

        if pred_mv_old and pred_mv_new:
            if can_push_down:
                pred_inner_old = self._group_scope_predicate_for_table(
                    base_t, group_mv_cols, None, "OLD"
                )
                pred_inner_new = self._group_scope_predicate_for_table(
                    base_t, group_mv_cols, None, "NEW"
                )
                assert pred_inner_old and pred_inner_new
                q_ins = self._inject_where_into_source_query(source_query_sql, pred_inner_new)
                q_del = self._inject_where_into_source_query(source_query_sql, pred_inner_old)
                q_upd = self._inject_where_into_source_query(
                    source_query_sql, f"({pred_inner_old}) OR ({pred_inner_new})"
                )
                ins_stmts = [
                    f"DELETE FROM {mv_table} WHERE {pred_mv_new};",
                    f"INSERT INTO {mv_table} ({cols_csv})\n    {q_ins};",
                ]
                del_stmts = [
                    f"DELETE FROM {mv_table} WHERE {pred_mv_old};",
                    f"INSERT INTO {mv_table} ({cols_csv})\n    {q_del};",
                ]
                upd_coalesced_stmts = [
                    f"DELETE FROM {mv_table} WHERE ({pred_mv_old}) OR ({pred_mv_new});",
                    f"INSERT INTO {mv_table} ({cols_csv})\n    {q_upd};",
                ]
            else:
                pred_src_old = self._group_scope_predicate_for_table(
                    base_t, group_mv_cols, "__mv_src", "OLD"
                )
                pred_src_new = self._group_scope_predicate_for_table(
                    base_t, group_mv_cols, "__mv_src", "NEW"
                )
                assert pred_src_old and pred_src_new
                ins_stmts = [
                    f"DELETE FROM {mv_table} WHERE {pred_mv_new};",
                    f"INSERT INTO {mv_table} ({cols_csv})\n"
                    f"    SELECT {src_cols_csv} FROM (\n{source_query_sql}\n    ) AS __mv_src\n"
                    f"    WHERE {pred_src_new};",
                ]
                del_stmts = [
                    f"DELETE FROM {mv_table} WHERE {pred_mv_old};",
                    f"INSERT INTO {mv_table} ({cols_csv})\n"
                    f"    SELECT {src_cols_csv} FROM (\n{source_query_sql}\n    ) AS __mv_src\n"
                    f"    WHERE {pred_src_old};",
                ]
                upd_coalesced_stmts = [
                    f"DELETE FROM {mv_table} WHERE ({pred_mv_old}) OR ({pred_mv_new});",
                    f"INSERT INTO {mv_table} ({cols_csv})\n"
                    f"    SELECT {src_cols_csv} FROM (\n{source_query_sql}\n    ) AS __mv_src\n"
                    f"    WHERE ({pred_src_old}) OR ({pred_src_new});",
                ]
            upd_old_stmts = []
            upd_new_stmts = upd_coalesced_stmts
        else:
            ins_stmts = full_refresh
            del_stmts = full_refresh
            upd_old_stmts = []
            upd_new_stmts = full_refresh

        pg_ai_body: Optional[str] = None
        pg_ad_body: Optional[str] = None
        pg_au_body: Optional[str] = None
        pg_trunc_stmts = full_refresh
        if self.dialect == Dialect.POSTGRES:
            pg_bodies = self._pg_group_scoped_bodies(
                base_t, mv_table, cols_csv, src_cols_csv, group_mv_cols, source_query_sql,
                can_push_down, full_refresh,
            )
            pg_ai_body, pg_au_body, pg_ad_body = pg_bodies["INSERT"], pg_bodies["UPDATE"], pg_bodies["DELETE"]
            # TRUNCATE recomputes everything: the whole-view lock key.
            pg_trunc_stmts = [self._pg_view_lock_stmt(), *full_refresh]

        return self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=ins_stmts,
            del_stmts=del_stmts,
            upd_old_stmts=upd_old_stmts,
            upd_new_stmts=upd_new_stmts,
            trunc_stmts=pg_trunc_stmts,
            pg_ai_body=pg_ai_body,
            pg_ad_body=pg_ad_body,
            pg_au_body=pg_au_body,
        )


    def _pg_group_scope_keys(
        self, base_t: str, group_mv_cols: List[Tuple[str, str]], signed_sql: str
    ) -> Optional[Tuple[str, List[str]]]:
        """``(FROM clause, key expressions)`` listing the group keys a statement
        on ``base_t`` affects (PostgreSQL), or ``None`` when they cannot be
        derived from the delta (the caller recomputes the whole view).

        Mirrors ``_group_scope_predicate_for_table``: single-source views and
        keys owned by ``base_t`` read them from the signed rows; otherwise the
        signed rows replace ``base_t`` in the view's join graph. A LEFT JOIN
        whose nullable side is ``base_t`` also affects the NULL-key group.
        """
        if not group_mv_cols:
            return None
        if len(self.view.joins) <= 1 and base_t.lower() != self.view.base_table.lower():
            return None
        if self.view.where_clause and re.search(r"\b(?:EXISTS|IN)\s*\(", self.view.where_clause, flags=re.IGNORECASE):
            if base_t.lower() != self.view.base_table.lower():
                return None
        rec = "__mv_d"
        signed_from = f"(\n        {signed_sql}\n        ) AS {rec}"
        if len(self.view.joins) <= 1:
            return signed_from, [self._rewrite_expr_for_record(g, rec) for g, _ in group_mv_cols]
        target_join: Optional[JoinSpec] = None
        for j in self.view.joins:
            if j.table_name.lower() == base_t.lower():
                target_join = j
                break
        if target_join is None:
            return None
        ts_target = self._schemas_lower[base_t.lower()]
        owned: List[str] = []
        for g_expr, _mv_col in group_mv_cols:
            gs = g_expr.strip()
            if "." in gs:
                p_al, c_part = gs.split(".", 1)
                if p_al.strip().lower() in (target_join.alias.lower(), base_t.lower()):
                    owned.append(f"{rec}.{self._q(_unquote_ident(c_part, dialect=self.dialect))}")
                    continue
                break
            if _unquote_ident(gs, dialect=self.dialect) in ts_target.columns:
                owned.append(f"{rec}.{self._q(_unquote_ident(gs, dialect=self.dialect))}")
                continue
            break
        if len(owned) == len(group_mv_cols):
            return signed_from, owned
        parts: List[str] = []
        for idx, j in enumerate(self.view.joins):
            if j.alias.lower() == target_join.alias.lower():
                tbl = f"(\n        {signed_sql}\n        ) AS {self._q(j.alias)}"
            else:
                tbl = f"{self._q(j.table_name)} AS {self._q(j.alias)}"
            if idx == 0:
                parts.append(tbl)
            else:
                join_type = "INNER JOIN" if (j.alias.lower() == target_join.alias.lower() and j.on_condition) else j.join_type
                on_sql = f" ON {j.on_condition}" if j.on_condition else ""
                parts.append(f"{join_type} {tbl}{on_sql}")
        from_sql = "\n        ".join(parts)
        keys = [g for g, _ in group_mv_cols]
        if any(j.join_type == "LEFT JOIN" for j in self.view.joins) and target_join != self.view.joins[0]:
            nulls = ", ".join("NULL" for _ in keys)
            sel = ", ".join(f"{k} AS __mv_k{i}" for i, k in enumerate(keys))
            from_sql = (
                f"(SELECT {sel} FROM {from_sql}\n        UNION ALL SELECT {nulls}) AS __mv_kk"
            )
            keys = [f"__mv_kk.__mv_k{i}" for i in range(len(keys))]
        return from_sql, keys

    def _pg_group_scoped_bodies(
        self,
        base_t: str,
        mv_table: str,
        cols_csv: str,
        src_cols_csv: str,
        group_mv_cols: List[Tuple[str, str]],
        source_query_sql: str,
        can_push_down: bool,
        full_refresh: List[str],
    ) -> Dict[str, str]:
        """PostgreSQL statement-level bodies for group-scoped recompute.

        1. Take the view-wide lock row (id 0) in SHARE mode, so a whole-view
           recompute (UPDATE of that row, e.g. TRUNCATE or a source whose
           affected groups cannot be derived) excludes group writers, then
           lock the affected groups only, in key order: one row per group key
           hash (odd ids) in the view's lock table, upserted in hash order (READ
           COMMITTED: later statements see every committed writer of those
           groups; REPEATABLE READ/SERIALIZABLE: a concurrent change of the
           group fails with a serialization error). Writers of other groups
           neither wait nor deadlock on it.
        2. Delete and re-insert the affected groups in later statements,
           matching keys with index-usable predicates: ``col = ANY(ARRAY(...))``
           for keys without NULLs, and a separate statement for keys with a
           NULL part, guarded so it does no work when there is none.
        """
        bodies: Dict[str, str] = {}
        refresh = "\n".join([self._pg_view_lock_stmt(), *full_refresh])
        lock_t = self._pg_view_lock_table()
        for event in ("INSERT", "UPDATE", "DELETE"):
            got = self._pg_group_scope_keys(base_t, group_mv_cols, pg_signed_rows_sql(event))
            if got is None:
                bodies[event] = refresh
                continue
            from_sql, keys = got
            n = len(keys)
            sel = ", ".join(f"{k} AS __mv_k{i}" for i, k in enumerate(keys))
            kset = f"(SELECT DISTINCT {sel} FROM {from_sql}) AS __mv_kset"
            kcols = [f"__mv_kset.__mv_k{i}" for i in range(n)]
            canon = self._canonical_group_pk_expr(kcols)
            no_null = " AND ".join(f"{k} IS NOT NULL" for k in kcols)
            any_null = " OR ".join(f"{k} IS NULL" for k in kcols)
            stmts = [
                *self._pg_view_shared_lock_stmts(),
                f"INSERT INTO {lock_t} (__mv_id, __mv_n)\n"
                f"    SELECT DISTINCT hashtextextended({canon}, 0) | 1, 1 FROM {kset}\n"
                f"    ORDER BY 1\n"
                f"    ON CONFLICT (__mv_id) DO UPDATE SET __mv_n = {lock_t}.__mv_n + 1;"
            ]

            def _match_nonnull(lhs: List[str]) -> str:
                conds = [
                    f"{l} = ANY (ARRAY(SELECT __mv_kset.__mv_k{i} FROM {kset} WHERE {no_null}))"
                    for i, l in enumerate(lhs)
                ]
                if n > 1:
                    conds.append(f"({', '.join(lhs)}) IN (SELECT {', '.join(kcols)} FROM {kset} WHERE {no_null})")
                return " AND ".join(conds)

            def _match_null(lhs: List[str]) -> str:
                eq = " AND ".join(
                    f"({l} = {k} OR ({l} IS NULL AND {k} IS NULL))" for l, k in zip(lhs, kcols)
                )
                return (
                    f"EXISTS (SELECT 1 FROM {kset} WHERE {any_null})\n"
                    f"    AND EXISTS (SELECT 1 FROM {kset} WHERE ({any_null}) AND {eq})"
                )

            mv_lhs = [f"{mv_table}.{self._q(mv_c)}" for _g, mv_c in group_mv_cols]
            src_lhs_push = [g for g, _ in group_mv_cols]
            src_lhs_out = [f"__mv_src.{self._q(mv_c)}" for _g, mv_c in group_mv_cols]
            for match in (_match_nonnull, _match_null):
                stmts.append(f"DELETE FROM {mv_table}\n    WHERE {match(mv_lhs)};")
            for match in (_match_nonnull, _match_null):
                if can_push_down:
                    q = self._inject_where_into_source_query(source_query_sql, match(src_lhs_push))
                    stmts.append(f"INSERT INTO {mv_table} ({cols_csv})\n    {q};")
                else:
                    stmts.append(
                        f"INSERT INTO {mv_table} ({cols_csv})\n"
                        f"    SELECT {src_cols_csv} FROM (\n{source_query_sql}\n    ) AS __mv_src\n"
                        f"    WHERE {match(src_lhs_out)};"
                    )
            bodies[event] = "\n".join(stmts)
        return bodies

    def _compile_tier2_group_scoped(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in self.view.select_items]
        grp_exprs = self._tier2_scope_exprs()

        target_cols: List[str] = list(user_columns)
        group_mv_cols: List[Tuple[str, str]] = []
        hidden_grp_defs: List[Tuple[str, str, str]] = []

        non_agg_map: Dict[str, str] = {}
        for it in self.view.select_items:
            if not it.is_aggregate:
                non_agg_map[it.raw_expr.strip().lower()] = it.alias
                if it.source_col:
                    non_agg_map[it.source_col.lower()] = it.alias

        for idx_g, g_expr in enumerate(grp_exprs):
            g_low = g_expr.strip().lower()
            g_unq = _unquote_ident(g_expr, dialect=self.dialect).lower() if self._is_simple_col_ref(g_expr) else ""
            if g_low in non_agg_map:
                mv_c = non_agg_map[g_low]
            elif g_unq and g_unq in non_agg_map:
                mv_c = non_agg_map[g_unq]
            else:
                mv_c = f"__mv_grp_{idx_g}"
                target_cols.append(mv_c)
                if self.dialect == Dialect.SQLITE:
                    dt = "BLOB"
                else:
                    c_def = self._lookup_col_def(g_expr)
                    dt = (
                        self._normalize_dialect_type(c_def.data_type)
                        if c_def
                        else ("VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT")
                    )
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                        dt = "VARCHAR(255)"
                hidden_grp_defs.append((mv_c, g_expr, dt))
            group_mv_cols.append((g_expr, mv_c))

        auto_grp_col_names = [mc for _, mc in group_mv_cols]
        indexed_cols_lower = self._indexed_column_names_lower(auto_grp_col_names)
        grp_mv_cols_lower = {mc.lower() for _, mc in group_mv_cols}

        col_defs_sql: List[str] = []
        for it in self.view.select_items:
            if self.dialect == Dialect.SQLITE and (
                (
                    not it.is_aggregate
                    and (
                        it.alias.lower() in grp_mv_cols_lower
                        or self.view.has_cte
                        or self.view.has_compound_set_op
                        or self._lookup_col_def(it.raw_expr) is None
                    )
                )
                or (it.is_aggregate and it.agg_func is None)
            ):
                dt = "BLOB"
            else:
                dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{self._q(it.alias)} {dt}")
        for mv_c, _, dt in hidden_grp_defs:
            col_defs_sql.append(f"{self._q(mv_c)} {dt}")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=auto_grp_col_names,
            has_pk_on_group_cols=False,
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        raw_q = self._rewrite_sqlite_total_in_sql(self.view.raw_select_sql.strip().rstrip(";"))
        if hidden_grp_defs:
            tok_from = _find_top_level_token(raw_q, (SQLGlotTokenType.FROM,), dialect=self.dialect)
            assert tok_from is not None
            extra_sel = ", ".join(f"{g_e} AS {self._q(mv_c)}" for mv_c, g_e, _ in hidden_grp_defs)
            source_query_sql = f"{raw_q[:tok_from.start].rstrip()}, {extra_sel} {raw_q[tok_from.start:]}"
        else:
            source_query_sql = raw_q

        q_target_cols = [self._q(c) for c in target_cols]
        cols_csv = ", ".join(q_target_cols)
        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({cols_csv})\n"
            f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init;"
        )
        ord_t2 = ", ".join(str(k) for k in range(1, len(target_cols) + 1))
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init\n"
                f"ORDER BY {ord_t2}\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        for ref_table in self.view.referenced_tables:
            t_ddls, t_ns, f_ns = self._build_tier2_triggers_for_table(
                view_name=view_name,
                mv_table=mv_table,
                base_t=ref_table,
                target_cols=target_cols,
                group_mv_cols=group_mv_cols,
                source_query_sql=source_query_sql,
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
            auxiliary_table_ddls=self._pg_view_lock_ddls() if self.dialect == Dialect.POSTGRES else [],
            auxiliary_table_names=[self._pg_view_lock_table()] if self.dialect == Dialect.POSTGRES else [],
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

