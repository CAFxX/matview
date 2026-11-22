"""Tier 1C: Z-set / multiplicity maintenance for set operations, SELECT DISTINCT, and projections."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp

from .._host import MixinHost
from ..dialects.pg_statement import SIGN, PgDeltaSource
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
    SQLToken,
    SQLTokenType,
    _contains_aggregate_call,
    _extract_set_op_branches_from_ast,
    _extract_verbatim_cte_or_subquery_sql,
    _has_subquery_ast,
    _is_simple_column_expr,
    _parse_single_agg_expr,
    _parse_single_create_view_stmt,
    _sanitize_ident,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_sql_ast,
    parse_view_ast,
    quote_sql_identifier,
    tokenize_sql,
)


class Tier1ZSetMixin(MixinHost):
    def _parse_setop_branches(self) -> Optional[List[Tuple[str, ViewSpec]]]:
        raw = self.view.raw_select_sql.strip()
        try:
            q_ast = parse_sql_ast(raw, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(q_ast, sqlglot_exp.SetOperation):
            return None

        ops, branch_sqls = _extract_set_op_branches_from_ast(q_ast, raw, dialect=self.dialect)
        if not ops or len(set(ops)) != 1:
            return None
        op_type = ops[0]

        if op_type in ("INTERSECT", "EXCEPT") and len(branch_sqls) != 2:
            return None

        branches: List[Tuple[str, ViewSpec]] = []
        col_count: Optional[int] = None
        for idx, b_sql in enumerate(branch_sqls):
            try:
                bs = _parse_single_create_view_stmt(
                    f"CREATE VIEW __set_b_{idx} AS {b_sql}",
                    known_tables=set(self._schemas_lower.keys()),
                    dialect=self.dialect,
                )
            except Exception:
                return None
            if (
                bs.has_cte
                or bs.has_window_func
                or bs.has_compound_set_op
                or bs.has_top_level_distinct
                or bs.has_limit_or_order
                or bs.has_subquery_in_from
                or len(bs.joins) != 1
                or bs.group_by_cols
                or any(it.is_aggregate or it.raw_expr.strip() == "*" for it in bs.select_items)
                or bs.base_table.lower() not in self._schemas_lower
            ):
                return None
            if bs.where_clause and _has_subquery_ast(bs.where_clause, dialect=self.dialect):
                return None
            if col_count is None:
                col_count = len(bs.select_items)
            elif len(bs.select_items) != col_count:
                return None
            if op_type == "UNION ALL":
                ts = self._schemas_lower[bs.base_table.lower()]
                if not any(c.primary_key for c in ts.columns.values()):
                    return None
            branches.append((op_type, bs))
        return branches

    def _rewrite_branch_expr(self, expr: str, ts: TableSchema, alias: str, record_prefix: Optional[str]) -> str:
        col_map = {c.lower(): c for c in ts.columns}
        quals_lower = {q.lower() for q in (ts.name, alias) if q}
        toks = tokenize_sql(expr, dialect=self.dialect)
        out: List[str] = []
        i = 0
        n = len(toks)

        def _next_non_ws(idx: int) -> Optional[SQLToken]:
            k = idx
            while k < n:
                if toks[k].kind != SQLTokenType.WHITESPACE:
                    return toks[k]
                k += 1
            return None

        def _prev_non_ws(idx: int) -> Optional[SQLToken]:
            k = idx
            while k >= 0:
                if toks[k].kind != SQLTokenType.WHITESPACE:
                    return toks[k]
                k -= 1
            return None

        while i < n:
            t = toks[i]
            if (
                t.kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT)
                and i + 2 < n
                and toks[i + 1].kind == SQLTokenType.PUNCT
                and toks[i + 1].text == "."
                and toks[i + 2].kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT)
            ):
                q_low = t.unquoted.lower()
                c_low = toks[i + 2].unquoted.lower()
                actual_c = col_map.get(c_low, toks[i + 2].unquoted)
                if q_low in quals_lower:
                    if record_prefix:
                        out.append(f"{record_prefix}.{self._q(actual_c)}")
                    else:
                        out.append(f"{self._q(t.unquoted)}.{self._q(actual_c)}")
                    i += 3
                    continue
            if t.kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT):
                prev_t = _prev_non_ws(i - 1)
                next_t = _next_non_ws(i + 1)
                is_after_dot = prev_t is not None and prev_t.kind == SQLTokenType.PUNCT and prev_t.text == "."
                is_before_dot = next_t is not None and next_t.kind == SQLTokenType.PUNCT and next_t.text == "."
                is_fn_call = (
                    t.kind == SQLTokenType.IDENT
                    and next_t is not None
                    and next_t.kind == SQLTokenType.PUNCT
                    and next_t.text == "("
                )
                if not is_after_dot and not is_before_dot and not is_fn_call:
                    w_low = t.unquoted.lower()
                    if w_low in col_map:
                        actual_c = col_map[w_low]
                        if record_prefix:
                            out.append(f"{record_prefix}.{self._q(actual_c)}")
                        else:
                            out.append(self._q(actual_c))
                        i += 1
                        continue
            out.append(t.text)
            i += 1
        return "".join(out)

    def _table_pk_is_nonnull(self, pk_defs: Sequence[ColumnDef]) -> bool:
        """Return True iff the table has a non-empty PK whose columns cannot be NULL."""
        if not pk_defs:
            return False
        for c in pk_defs:
            if c.not_null or self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                continue
            if (
                self.dialect == Dialect.SQLITE
                and len(pk_defs) == 1
                and re.sub(r"\s+", " ", c.data_type.strip().upper()) == "INTEGER"
            ):
                continue
            return False
        return True

    def _projection_row_pk_expr(self, exprs: Sequence[str]) -> str:
        """Exact row-identity canonical key for bag projections and nullable-PK UNION ALL.

        Unlike ``_canonical_group_pk_expr`` (which folds SQLite int64-valued REALs
        like ``1.0`` onto ``INTEGER 1`` and normalizes PostgreSQL ``NUMERIC 1.0`` /
        ``1.00`` and ``FLOAT8 0.0`` / ``-0.0`` for GROUP BY / DISTINCT / UNION
        equivalence), projections preserve each row's exact representation so
        distinct storage classes, decimal scales, and signed zeros stay distinct.
        """
        def _clean(e: str) -> str:
            s = e.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
                return self._q(s)
            m_dot = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", s)
            if m_dot:
                return f"{m_dot.group(1)}.{self._q(m_dot.group(2))}"
            return s

        if self.dialect == Dialect.SQLITE:
            terms = [
                f"(typeof({s}) || ':' || length(quote({s})) || ':' || quote({s}))"
                for s in (_clean(e) for e in exprs)
            ]
            return " || '|' || ".join(terms)
        if self.dialect == Dialect.POSTGRES:
            pg_terms: List[str] = []
            for e in exprs:
                s = _clean(e)
                if s.upper() == "NULL":
                    pg_terms.append("'N'")
                elif s.startswith("'") and s.endswith("'"):
                    cast_s = f"CAST({s} AS pg_catalog.text)"
                    pg_terms.append(f"('V:' || pg_catalog.octet_length({cast_s}) || ':' || {cast_s})")
                else:
                    pg_terms.append(
                        f"(CASE WHEN ({s}) IS NOT DISTINCT FROM NULL THEN 'N' "
                        f"ELSE 'V:' || pg_catalog.octet_length(({s})::pg_catalog.text) || ':' || ({s})::pg_catalog.text END)"
                    )
            return f"ROW({', '.join(pg_terms)})::pg_catalog.text"
        return self._canonical_group_pk_expr(exprs)

    def _resolve_setop_pk_info(
        self, branches: Optional[List[Tuple[str, ViewSpec]]] = None
    ) -> Tuple[bool, Tuple[str, ...], List[str], List[List[ColumnDef]]]:
        if branches is None:
            branches = self._parse_setop_branches()
        assert branches is not None
        op_type = branches[0][0]
        branch_pk_defs_list: List[List[ColumnDef]] = []
        for _, bs in branches:
            ts_b = self._schemas_lower[bs.base_table.lower()]
            branch_pk_defs_list.append([c for c in ts_b.columns.values() if c.primary_key])

        use_native_uall_pk = False
        uall_pk_cols: Tuple[str, ...] = ("__mv_pk",)
        uall_pk_col_types: List[str] = []
        if op_type == "UNION ALL" and branch_pk_defs_list and branch_pk_defs_list[0]:
            k_pks = len(branch_pk_defs_list[0])
            if (
                all(len(pks) == k_pks for pks in branch_pk_defs_list)
                and all(self._table_pk_is_nonnull(pks) for pks in branch_pk_defs_list)
            ):
                cand_types: List[str] = []
                types_match = True
                for i in range(k_pks):
                    dt_i = self._normalize_dialect_type(branch_pk_defs_list[0][i].data_type)
                    if self.dialect == Dialect.MYSQL and dt_i.upper() == "TEXT":
                        dt_i = "VARCHAR(255)"
                    for other_pks in branch_pk_defs_list[1:]:
                        other_dt = self._normalize_dialect_type(other_pks[i].data_type)
                        if self.dialect == Dialect.MYSQL and other_dt.upper() == "TEXT":
                            other_dt = "VARCHAR(255)"
                        if other_dt.upper() != dt_i.upper():
                            types_match = False
                            break
                    if not types_match:
                        break
                    cand_types.append(dt_i)
                if types_match:
                    use_native_uall_pk = True
                    uall_pk_cols = ("__mv_branch_id", *(f"__mv_pk_{i}" for i in range(k_pks)))
                    uall_pk_col_types = cand_types
        return use_native_uall_pk, uall_pk_cols, uall_pk_col_types, branch_pk_defs_list

    def _compile_tier1_zset_setop(self) -> MaterializedViewPlan:
        branches = self._parse_setop_branches()
        assert branches is not None
        op_type = branches[0][0]
        first_bs = branches[0][1]
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in first_bs.select_items]
        pk_type = self._canonical_group_pk_type()
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""

        use_native_uall_pk, uall_pk_cols_t, uall_pk_col_types, branch_pk_defs_list = (
            self._resolve_setop_pk_info(branches)
        )
        uall_pk_cols = list(uall_pk_cols_t)

        col_defs_sql: List[str] = []
        if use_native_uall_pk:
            col_defs_sql.append("__mv_branch_id INTEGER NOT NULL")
            for i, dt_i in enumerate(uall_pk_col_types):
                col_defs_sql.append(f"__mv_pk_{i} {dt_i} NOT NULL")
        else:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        indexed_cols_lower = self._indexed_column_names_lower()
        for it in first_bs.select_items:
            if self.dialect == Dialect.SQLITE:
                # SQLite compound SELECT columns have no affinity; storing
                # without affinity (BLOB) preserves exact storage classes across
                # branches with differing affinities.
                dt = "BLOB"
            else:
                dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{self._q(it.alias)} {dt}")

        if op_type in ("UNION", "UNION ALL"):
            col_defs_sql.append(f"__mv_row_count {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = None
        elif op_type == "INTERSECT":
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = "__mv_cnt_0 > 0 AND __mv_cnt_1 > 0"
        else:  # EXCEPT
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = "__mv_cnt_0 > 0 AND __mv_cnt_1 = 0"

        if use_native_uall_pk:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in uall_pk_cols)})")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
            pk_cols=uall_pk_cols if use_native_uall_pk else None,
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
        )
        any_nullable_branch_pk = op_type == "UNION ALL" and any(
            not self._table_pk_is_nonnull(pks) for pks in branch_pk_defs_list
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                extra_where_filter=extra_filter,
                repeat_count_col="__mv_row_count" if any_nullable_branch_pk else None,
            )
        )

        initial_populate_sqls: List[str] = []
        by_table_branches_lower: Dict[str, Tuple[str, List[Tuple[int, ViewSpec]]]] = {}
        for b_idx, (_, bs) in enumerate(branches):
            t_low = bs.base_table.lower()
            if t_low not in by_table_branches_lower:
                canon_name = self._schemas_lower[t_low].name if t_low in self._schemas_lower else bs.base_table
                by_table_branches_lower[t_low] = (canon_name, [])
            by_table_branches_lower[t_low][1].append((b_idx, bs))
        by_table_branches: Dict[str, List[Tuple[int, ViewSpec]]] = {
            canon_name: b_items for canon_name, b_items in by_table_branches_lower.values()
        }

        q_uall_pk_cols = [self._q(c) for c in uall_pk_cols]
        q_user_columns = [self._q(c) for c in user_columns]
        u_parts: List[str] = []

        if op_type == "UNION ALL":
            for b_idx, (_, bs) in enumerate(branches):
                ts = self._schemas_lower[bs.base_table.lower()]
                pk_defs = [c for c in ts.columns.values() if c.primary_key]
                pk_names = [c.name for c in pk_defs]
                sel_exprs = [it.raw_expr for it in bs.select_items]
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                b_alias = bs.joins[0].alias if bs.joins else bs.base_table
                is_aliased = b_alias.lower() != bs.base_table.lower()
                from_tbl = (
                    f"{self._q(bs.base_table)} AS {self._q(b_alias)}"
                    if is_aliased
                    else self._q(bs.base_table)
                )
                q_alias_prefix = f"{self._q(b_alias)}." if is_aliased else ""
                q_pk_names = [f"{q_alias_prefix}{self._q(c)}" for c in pk_names]
                if use_native_uall_pk:
                    initial_populate_sqls.append(
                        f"INSERT INTO {mv_table} ({', '.join(q_uall_pk_cols)}, {', '.join(q_user_columns)}, __mv_row_count)\n"
                        f"SELECT {b_idx}, {', '.join(q_pk_names)}, {', '.join(sel_exprs)}, 1 FROM {from_tbl}{w_sql};"
                    )
                else:
                    branch_nonnull_pk = self._table_pk_is_nonnull(pk_defs)
                    b_hash_cols = pk_names if branch_nonnull_pk else list(ts.columns.keys())
                    pk_expr = self._projection_row_pk_expr(
                        [f"'B{b_idx}'", *[f"{q_alias_prefix}{self._q(c)}" for c in b_hash_cols]]
                    )
                    if branch_nonnull_pk:
                        initial_populate_sqls.append(
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"SELECT {pk_expr}, {', '.join(sel_exprs)}, 1 FROM {from_tbl}{w_sql};"
                        )
                    else:
                        inner_cols = [
                            f"{pk_expr} AS __mv_k0",
                            *[f"{expr} AS __mv_c{j}" for j, expr in enumerate(sel_exprs)],
                        ]
                        outer_cols = ["__mv_p.__mv_k0", *[f"__mv_p.__mv_c{j}" for j in range(len(sel_exprs))]]
                        initial_populate_sqls.append(
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"SELECT {', '.join(outer_cols)}, COUNT(*)\n"
                            f"FROM (\n"
                            f"    SELECT {', '.join(inner_cols)} FROM {from_tbl}{w_sql}\n"
                            f") AS __mv_p\n"
                            f"GROUP BY {', '.join(outer_cols)};"
                        )
        elif op_type == "UNION":
            u_parts = []
            for _, bs in branches:
                sel_aliased = ", ".join(
                    f"{it.raw_expr} AS {self._q(col)}" for it, col in zip(bs.select_items, user_columns)
                )
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                b_alias = bs.joins[0].alias if bs.joins else bs.base_table
                from_tbl = (
                    f"{self._q(bs.base_table)} AS {self._q(b_alias)}"
                    if b_alias.lower() != bs.base_table.lower()
                    else self._q(bs.base_table)
                )
                u_parts.append(f"    SELECT {sel_aliased} FROM {from_tbl}{w_sql}")
            pk_u = self._canonical_group_pk_expr(user_columns)
            grp_u = ", ".join(self._group_by_exprs_for_query(user_columns, include_canonical_pk=True))
            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                f"SELECT {pk_u}, {', '.join(q_user_columns)}, COUNT(*) FROM (\n"
                + "\n    UNION ALL\n".join(u_parts)
                + f"\n) AS __mv_u\nGROUP BY {grp_u};"
            )
        else:  # INTERSECT or EXCEPT
            u_parts = []
            for b_idx, (_, bs) in enumerate(branches):
                sel_aliased = ", ".join(
                    f"{it.raw_expr} AS {self._q(col)}" for it, col in zip(bs.select_items, user_columns)
                )
                b0 = "1" if b_idx == 0 else "0"
                b1 = "1" if b_idx == 1 else "0"
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                b_alias = bs.joins[0].alias if bs.joins else bs.base_table
                from_tbl = (
                    f"{self._q(bs.base_table)} AS {self._q(b_alias)}"
                    if b_alias.lower() != bs.base_table.lower()
                    else self._q(bs.base_table)
                )
                u_parts.append(
                    f"    SELECT {sel_aliased}, {b0} AS __mv_b0, {b1} AS __mv_b1 FROM {from_tbl}{w_sql}"
                )
            pk_u = self._canonical_group_pk_expr(user_columns)
            grp_u = ", ".join(self._group_by_exprs_for_query(user_columns, include_canonical_pk=True))
            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                f"SELECT {pk_u}, {', '.join(q_user_columns)}, SUM(__mv_b0), SUM(__mv_b1) FROM (\n"
                + "\n    UNION ALL\n".join(u_parts)
                + f"\n) AS __mv_u\nGROUP BY {grp_u};"
            )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for base_t, b_list in by_table_branches.items():
            ts = self._schemas_lower[base_t.lower()]
            ins_stmts: List[str] = []
            del_stmts: List[str] = []
            upd_old_stmts: List[str] = []
            upd_new_stmts: List[str] = []
            cleanup_stmts: List[str] = []

            for b_idx, bs in b_list:
                b_alias = bs.joins[0].alias if bs.joins else base_t
                w_new = (
                    self._rewrite_branch_expr(bs.where_clause, ts, b_alias, "NEW")
                    if bs.where_clause
                    else "TRUE"
                )
                w_old = (
                    self._rewrite_branch_expr(bs.where_clause, ts, b_alias, "OLD")
                    if bs.where_clause
                    else "TRUE"
                )
                new_vals = [
                    self._rewrite_branch_expr(it.raw_expr, ts, b_alias, "NEW")
                    for it in bs.select_items
                ]
                old_vals = [
                    self._rewrite_branch_expr(it.raw_expr, ts, b_alias, "OLD")
                    for it in bs.select_items
                ]
                pk_old = ""

                if op_type == "UNION ALL":
                    pk_defs = [c for c in ts.columns.values() if c.primary_key]
                    pk_names = [c.name for c in pk_defs]
                    if use_native_uall_pk:
                        new_pk_vals = [str(b_idx), *[f"NEW.{self._q(c)}" for c in pk_names]]
                        old_pk_match = f"{mv_table}.__mv_branch_id = {b_idx} AND " + " AND ".join(
                            f"{mv_table}.__mv_pk_{i} = OLD.{self._q(c)}" for i, c in enumerate(pk_names)
                        )
                        if self.dialect == Dialect.MYSQL:
                            if self.engine_version.supports_mysql_row_alias_upsert:
                                upd_set = ", ".join(f"{self._q(c)} = __mv_new.{self._q(c)}" for c in user_columns)
                                aliased_uall_sel = [
                                    *[f"{expr} AS {col}" for expr, col in zip(new_pk_vals, q_uall_pk_cols)],
                                    *[f"{expr} AS {col}" for expr, col in zip(new_vals, q_user_columns)],
                                    "1 AS __mv_row_count",
                                ]
                                plus_s = (
                                    f"INSERT INTO {mv_table} ({', '.join(q_uall_pk_cols)}, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                    f"    SELECT * FROM (\n"
                                    f"        SELECT {', '.join(aliased_uall_sel)}{from_dual}\n"
                                    f"        WHERE {w_new}\n"
                                    f"    ) AS __mv_new\n"
                                    f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                                )
                            else:
                                upd_set = ", ".join(f"{self._q(c)} = VALUES({self._q(c)})" for c in user_columns)
                                plus_s = (
                                    f"INSERT INTO {mv_table} ({', '.join(q_uall_pk_cols)}, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                    f"    SELECT {', '.join(new_pk_vals)}, {', '.join(new_vals)}, 1{from_dual}\n"
                                    f"    WHERE {w_new}\n"
                                    f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                                )
                        else:
                            upd_set = ", ".join(f"{self._q(c)} = excluded.{self._q(c)}" for c in user_columns)
                            plus_s = (
                                f"INSERT INTO {mv_table} ({', '.join(q_uall_pk_cols)}, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                f"    SELECT {', '.join(new_pk_vals)}, {', '.join(new_vals)}, 1\n"
                                f"    WHERE {w_new}\n"
                                f"    ON CONFLICT ({', '.join(q_uall_pk_cols)}) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                            )
                        minus_s = (
                            f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                            f"    WHERE ({w_old}) AND ({old_pk_match});"
                        )
                        clean_s = (
                            f"DELETE FROM {mv_table}\n"
                            f"    WHERE ({w_old}) AND ({old_pk_match}) AND {mv_table}.__mv_row_count <= 0;"
                        )
                    else:
                        b_hash_cols = pk_names if self._table_pk_is_nonnull(pk_defs) else list(ts.columns.keys())
                        pk_new = self._projection_row_pk_expr([f"'B{b_idx}'", *[f"NEW.{self._q(c)}" for c in b_hash_cols]])
                        pk_old = self._projection_row_pk_expr([f"'B{b_idx}'", *[f"OLD.{self._q(c)}" for c in b_hash_cols]])
                        if self.dialect == Dialect.MYSQL:
                            if self.engine_version.supports_mysql_row_alias_upsert:
                                upd_set = ", ".join(f"{self._q(c)} = __mv_new.{self._q(c)}" for c in user_columns)
                                aliased_uall_sel = [
                                    f"{pk_new} AS __mv_pk",
                                    *[f"{expr} AS {col}" for expr, col in zip(new_vals, q_user_columns)],
                                    "1 AS __mv_row_count",
                                ]
                                plus_s = (
                                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                    f"    SELECT * FROM (\n"
                                    f"        SELECT {', '.join(aliased_uall_sel)}{from_dual}\n"
                                    f"        WHERE {w_new}\n"
                                    f"    ) AS __mv_new\n"
                                    f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                                )
                            else:
                                upd_set = ", ".join(f"{self._q(c)} = VALUES({self._q(c)})" for c in user_columns)
                                plus_s = (
                                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                    f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                                    f"    WHERE {w_new}\n"
                                    f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                                )
                        else:
                            upd_set = ", ".join(f"{self._q(c)} = excluded.{self._q(c)}" for c in user_columns)
                            plus_s = (
                                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                                f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                                f"    WHERE {w_new}\n"
                                f"    ON CONFLICT (__mv_pk) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                            )
                        minus_s = (
                            f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                            f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                        )
                        clean_s = (
                            f"DELETE FROM {mv_table}\n"
                            f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
                        )
                elif op_type == "UNION":
                    pk_new = self._canonical_group_pk_expr(new_vals)
                    pk_old = self._canonical_group_pk_expr(old_vals)
                    if self.dialect == Dialect.MYSQL:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON DUPLICATE KEY UPDATE __mv_row_count = {mv_table}.__mv_row_count + 1;"
                        )
                    else:
                        if self.dialect == Dialect.SQLITE:
                            rep_sets = ", ".join(
                                f"{qc} = CASE WHEN {mv_table}.__mv_row_count <= 0 THEN excluded.{qc} ELSE {mv_table}.{qc} END"
                                for qc in q_user_columns
                            )
                            conflict_set = f"{rep_sets}, __mv_row_count = {mv_table}.__mv_row_count + 1"
                        else:
                            conflict_set = f"__mv_row_count = {mv_table}.__mv_row_count + 1"
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                            f"    WHERE {w_new}\n"
                            f"    ON CONFLICT (__mv_pk) DO UPDATE SET {conflict_set};"
                        )
                    minus_s = (
                        f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                        f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                    )
                    clean_s = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
                    )
                else:  # INTERSECT or EXCEPT
                    pk_new = self._canonical_group_pk_expr(new_vals)
                    pk_old = self._canonical_group_pk_expr(old_vals)
                    cnt_col = f"__mv_cnt_{b_idx}"
                    init_c0 = "1" if b_idx == 0 else "0"
                    init_c1 = "1" if b_idx == 1 else "0"
                    if self.dialect == Dialect.MYSQL:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, {init_c0}, {init_c1}{from_dual}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON DUPLICATE KEY UPDATE {cnt_col} = {mv_table}.{cnt_col} + 1;"
                        )
                    else:
                        if self.dialect == Dialect.SQLITE and b_idx == 0:
                            # Branch 0 defines the output representative for INTERSECT/EXCEPT.
                            # Whenever branch 0 transitions from 0 -> 1 rows, adopt excluded values
                            # so a prior branch-1 row (or departed branch-0 row) never leaves a
                            # stale storage class.
                            rep_sets = ", ".join(
                                f"{qc} = CASE WHEN {mv_table}.__mv_cnt_0 <= 0 THEN excluded.{qc} ELSE {mv_table}.{qc} END"
                                for qc in q_user_columns
                            )
                            conflict_set = f"{rep_sets}, {cnt_col} = {mv_table}.{cnt_col} + 1"
                        else:
                            conflict_set = f"{cnt_col} = {mv_table}.{cnt_col} + 1"
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, {init_c0}, {init_c1}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON CONFLICT (__mv_pk) DO UPDATE SET {conflict_set};"
                        )
                    minus_s = (
                        f"UPDATE {mv_table} SET {cnt_col} = {mv_table}.{cnt_col} - 1\n"
                        f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                    )
                    clean_s = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_cnt_0 <= 0 AND {mv_table}.__mv_cnt_1 <= 0;"
                    )

                refresh_stmts: List[str] = []
                if self.dialect == Dialect.SQLITE and op_type in ("UNION", "INTERSECT", "EXCEPT"):
                    surv_src = "\n    UNION ALL\n".join(u_parts) if op_type == "UNION" else u_parts[0]
                    surv_cond = (
                        f"{mv_table}.__mv_row_count > 0"
                        if op_type == "UNION"
                        else f"{mv_table}.__mv_cnt_0 > 0"
                    )
                    match_all_cols = " AND ".join(
                        f"(__mv_u.{self._q(c)}) IS ({mv_table}.{self._q(c)})" for c in user_columns
                    )
                    for col, old_v in zip(user_columns, old_vals):
                        q_col = self._q(col)
                        refresh_stmts.append(
                            f"UPDATE {mv_table}\n"
                            f"    SET {q_col} = COALESCE((SELECT __mv_u.{q_col} FROM (\n{surv_src}\n    ) AS __mv_u WHERE {match_all_cols} LIMIT 1), {mv_table}.{q_col})\n"
                            f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {surv_cond} "
                            f"AND typeof({old_v}) IN ('integer', 'real') AND typeof({mv_table}.{q_col}) = typeof({old_v});"
                        )

                ins_stmts.append(plus_s)
                del_stmts.extend([minus_s, clean_s, *refresh_stmts])
                upd_old_stmts.append(minus_s)
                upd_new_stmts.append(plus_s)
                cleanup_stmts.extend([clean_s, *refresh_stmts])

            trunc_stmts = [f"DELETE FROM {mv_table};", *initial_populate_sqls]

            pg_ai_body: Optional[str] = None
            pg_ad_body: Optional[str] = None
            pg_au_body: Optional[str] = None

            if self.dialect == Dialect.POSTGRES:
                pg_bodies = self._pg_setop_bodies(
                    op_type, b_list, mv_table, user_columns, use_native_uall_pk, uall_pk_cols
                )
                pg_ai_body, pg_au_body, pg_ad_body = pg_bodies["INSERT"], pg_bodies["UPDATE"], pg_bodies["DELETE"]

            uall_upd_keys: Optional[List[str]] = None
            if op_type == "UNION ALL":
                ts_pks = [c for c in ts.columns.values() if c.primary_key]
                uall_upd_keys = (
                    [c.name for c in ts_pks]
                    if self._table_pk_is_nonnull(ts_pks)
                    else list(ts.columns.keys())
                )

            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_stmts,
                del_stmts=del_stmts,
                upd_old_stmts=upd_old_stmts,
                upd_new_stmts=upd_new_stmts,
                upd_cleanup_old_stmts=cleanup_stmts,
                trunc_stmts=trunc_stmts,
                update_key_columns=uall_upd_keys,
                pg_ai_body=pg_ai_body,
                pg_ad_body=pg_ad_body,
                pg_au_body=pg_au_body,
            )
            trigger_ddls.extend(t_ddls)
            trig_names.extend(t_ns)
            fn_names.extend(f_ns)

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            chunk_sql_templates=initial_populate_sqls,
        )
        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=[],
            index_ddls=index_ddls,
            initial_populate_sqls=initial_populate_sqls,
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

    def _pg_setop_bodies(
        self,
        op_type: str,
        b_list: Sequence[Tuple[int, ViewSpec]],
        mv_table: str,
        user_columns: Sequence[str],
        use_native_uall_pk: bool,
        uall_pk_cols: Sequence[str],
    ) -> Dict[str, str]:
        """PostgreSQL statement-level bodies for the branches of one source table.

        UNION ALL rows are keyed by (branch, source primary key); UNION,
        INTERSECT and EXCEPT rows are counted per distinct value tuple.
        """
        rec = "__mv_d"

        def _pg_from(signed: str) -> str:
            return f"(\n        {signed}\n        ) AS {rec}"

        sources: List[Tuple[PgDeltaSource, Dict[str, str]]] = []
        for b_idx, bs in b_list:
            ts = self._schemas_lower[bs.base_table.lower()]
            b_alias = bs.joins[0].alias if bs.joins else bs.base_table
            where = self._rewrite_branch_expr(bs.where_clause, ts, b_alias, rec) if bs.where_clause else None
            vals = [self._rewrite_branch_expr(it.raw_expr, ts, b_alias, rec) for it in bs.select_items]
            values = dict(zip(user_columns, vals))
            pk_defs = [c for c in ts.columns.values() if c.primary_key]
            pk_names = [c.name for c in pk_defs]
            if op_type == "UNION ALL":
                if use_native_uall_pk:
                    keys = [str(b_idx), *[f"{rec}.{self._q(c)}" for c in pk_names]]
                else:
                    b_hash_cols = pk_names if self._table_pk_is_nonnull(pk_defs) else list(ts.columns.keys())
                    keys = [self._projection_row_pk_expr([f"'B{b_idx}'", *[f"{rec}.{self._q(c)}" for c in b_hash_cols]])]
                contrib: Dict[str, str] = {}
            elif op_type == "UNION":
                keys = [self._canonical_group_pk_expr(vals)]
                contrib = {"__mv_row_count": "1"}
            else:
                keys = [self._canonical_group_pk_expr(vals)]
                contrib = {"__mv_cnt_0": "1" if b_idx == 0 else "0", "__mv_cnt_1": "1" if b_idx == 1 else "0"}
            sources.append((
                PgDeltaSource(from_sql=_pg_from, sign=f"{rec}.{SIGN}", where=where, keys=keys, values=values),
                contrib,
            ))

        setop_base_t = b_list[0][1].base_table if b_list else None
        if op_type == "UNION ALL":
            key_cols = list(uall_pk_cols) if use_native_uall_pk else ["__mv_pk"]
            return self._pg_keyed_rows_bodies(
                mv_table,
                key_cols,
                list(user_columns),
                [src for src, _ in sources],
                count_col="__mv_row_count",
                base_t=setop_base_t,
            )
        if op_type == "UNION":
            return self._pg_counted_rows_bodies(
                mv_table,
                "__mv_pk",
                user_columns,
                ["__mv_row_count"],
                sources,
                f"{mv_table}.__mv_row_count <= 0",
                base_t=setop_base_t,
            )
        return self._pg_counted_rows_bodies(
            mv_table,
            "__mv_pk",
            user_columns,
            ["__mv_cnt_0", "__mv_cnt_1"],
            sources,
            f"{mv_table}.__mv_cnt_0 <= 0 AND {mv_table}.__mv_cnt_1 <= 0",
            base_t=setop_base_t,
        )

    def _compile_tier1_zset_distinct(self) -> MaterializedViewPlan:
        # `SELECT DISTINCT c1, c2 FROM t WHERE ...` is isomorphic to `GROUP BY c1, c2` with hidden `__mv_row_count`
        distinct_compiler = self.with_ir(
            effective_group_by_exprs=tuple(it.raw_expr for it in self.view.select_items)
        )
        return distinct_compiler._compile_tier1_algebraic()

    def _projection_pk_mapping(self) -> Tuple[List[str], List[Tuple[str, str, str]], bool]:
        """
        Returns `(mv_pk_cols, hidden_pk_defs, uses_native_pk)` for `_compile_tier1_zset_projection`.
        Preserves the base table's actual primary/composite key columns in the backing table
        whenever all base PK columns are non-nullable.
        """
        assert self.primary_schema is not None
        pk_defs = [c for c in self.primary_schema.columns.values() if c.primary_key]
        can_native = self._table_pk_is_nonnull(pk_defs)
        if not can_native:
            return ["__mv_pk"], [], False

        base_alias = self.view.joins[0].alias.lower() if self.view.joins else self.view.base_table.lower()
        base_name = self.view.base_table.lower()
        mv_pk_cols: List[str] = []
        hidden_pk_defs: List[Tuple[str, str, str]] = []
        used_user_aliases: Set[str] = set()

        for c_def in pk_defs:
            c_low = c_def.name.lower()
            matched_alias: Optional[str] = None
            for it in self.view.select_items:
                raw_s = _unquote_ident(it.raw_expr.strip(), dialect=self.dialect).lower() if re.match(r'^[A-Za-z0-9_."`\[\]]+$', it.raw_expr.strip()) else ""
                raw_full = it.raw_expr.strip().lower()
                raw_parts = [p.strip().strip('"').strip('`').strip('[').strip(']').lower() for p in raw_full.split('.')]
                col_match = (
                    raw_full == c_low
                    or raw_full == f"{base_alias}.{c_low}"
                    or raw_full == f"{base_name}.{c_low}"
                    or (raw_s == c_low and "." not in raw_full)
                    or (len(raw_parts) == 2 and raw_parts[0] in (base_alias, base_name) and raw_parts[1] == c_low)
                    or (len(raw_parts) == 1 and raw_parts[0] == c_low)
                )
                if col_match and it.alias.lower() not in used_user_aliases:
                    matched_alias = it.alias
                    used_user_aliases.add(it.alias.lower())
                    break
            if matched_alias is not None:
                mv_pk_cols.append(matched_alias)
            else:
                hid_col = f"__mv_pk_{_sanitize_ident(c_def.name)}"
                dt = self._normalize_dialect_type(c_def.data_type)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                hidden_pk_defs.append((hid_col, c_def.name, dt))
                mv_pk_cols.append(hid_col)

        return mv_pk_cols, hidden_pk_defs, True

    def _projection_pk_cols(self) -> List[str]:
        mv_pk_cols, _, _ = self._projection_pk_mapping()
        return mv_pk_cols

    def _compile_tier1_zset_projection(self) -> MaterializedViewPlan:
        assert self.primary_schema is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        base_t = self.view.base_table
        q_base_t = self._q(base_t)
        user_columns = [it.alias for it in self.view.select_items]
        pk_type = self._canonical_group_pk_type()
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""

        pk_names = [c.name for c in self.primary_schema.columns.values() if c.primary_key]
        mv_pk_cols, hidden_pk_defs, uses_native_pk = self._projection_pk_mapping()
        pk_col_set_lower = {c.lower() for c in mv_pk_cols}

        col_defs_sql: List[str] = []
        if not uses_native_pk:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        extra_auto_cols = self._auto_backing_extra_index_cols(
            excluded_leading_cols=(mv_pk_cols[0],) if (uses_native_pk and mv_pk_cols) else ("__mv_pk",)
        )
        indexed_cols_lower = self._indexed_column_names_lower(
            list(mv_pk_cols if uses_native_pk else ()) + list(extra_auto_cols)
        )
        for it in self.view.select_items:
            if self.dialect == Dialect.SQLITE and self._lookup_col_def(it.raw_expr) is None:
                dt = "BLOB"
            else:
                dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            nn = " NOT NULL" if (uses_native_pk and it.alias.lower() in pk_col_set_lower) else ""
            col_defs_sql.append(f"{self._q(it.alias)} {dt}{nn}")

        for hid_col, _, hid_dt in hidden_pk_defs:
            col_defs_sql.append(f"{self._q(hid_col)} {hid_dt} NOT NULL")

        q_mv_pk_cols = [self._q(c) for c in mv_pk_cols]
        if uses_native_pk:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(q_mv_pk_cols)})")
        else:
            col_defs_sql.append(f"__mv_row_count {self._int_type()} NOT NULL DEFAULT 0")

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
            pk_cols=mv_pk_cols if uses_native_pk else None,
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
            extra_auto_index_cols=extra_auto_cols,
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                repeat_count_col="__mv_row_count" if not uses_native_pk else None,
            )
        )

        sel_exprs = [it.raw_expr for it in self.view.select_items]
        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        new_vals = [self._rewrite_expr_for_record(it.raw_expr, "NEW") for it in self.view.select_items]
        base_alias = self.view.joins[0].alias if self.view.joins else base_t
        is_base_aliased = base_alias.lower() != base_t.lower()
        from_base_sql = (
            f"{q_base_t} AS {self._q(base_alias)}"
            if is_base_aliased
            else q_base_t
        )
        q_base_prefix = f"{self._q(base_alias)}." if is_base_aliased else ""

        wm_pk_col = self._resolve_pk_range_column(
            allow_sqlite_rowid=not is_base_aliased,
            require_non_composite_for_rowid=True,
        )

        if uses_native_pk:
            all_backing_cols = [*user_columns, *[h[0] for h in hidden_pk_defs]]
            q_all_backing_cols = [self._q(c) for c in all_backing_cols]
            all_select_exprs = [*sel_exprs, *[f"{q_base_prefix}{self._q(h[1])}" for h in hidden_pk_defs]]
            all_new_vals = [*new_vals, *[f"NEW.{self._q(h[1])}" for h in hidden_pk_defs]]
            initial_populate_sql = (
                f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                f"SELECT {', '.join(all_select_exprs)} FROM {from_base_sql}{w_sql};"
            )
            if self.dialect == Dialect.MYSQL:
                if self.engine_version.supports_mysql_row_alias_upsert:
                    upd_set = ", ".join(f"{self._q(c)} = __mv_new.{self._q(c)}" for c in all_backing_cols)
                    plus_stmt = (
                        f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                        f"    VALUES ({', '.join(all_new_vals)}) AS __mv_new\n"
                        f"    ON DUPLICATE KEY UPDATE {upd_set};"
                    )
                else:
                    upd_set = ", ".join(f"{self._q(c)} = VALUES({self._q(c)})" for c in all_backing_cols)
                    plus_stmt = (
                        f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                        f"    SELECT {', '.join(all_new_vals)}{from_dual}\n"
                        f"    WHERE TRUE\n"
                        f"    ON DUPLICATE KEY UPDATE {upd_set};"
                    )
            else:
                upd_set = ", ".join(f"{self._q(c)} = excluded.{self._q(c)}" for c in all_backing_cols)
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                    f"    SELECT {', '.join(all_new_vals)}\n"
                    f"    WHERE TRUE\n"
                    f"    ON CONFLICT ({', '.join(q_mv_pk_cols)}) DO UPDATE SET {upd_set};"
                )
            if wm_pk_col:
                ph = "?" if self.dialect == Dialect.SQLITE else "%s"
                q_wm_pk = f"{q_base_prefix}{self._quote_pk_range_col(wm_pk_col)}"
                chunk_w = (
                    f"({self.view.where_clause}) AND ({q_wm_pk} > {ph} AND {q_wm_pk} <= {ph})"
                    if self.view.where_clause
                    else f"({q_wm_pk} > {ph} AND {q_wm_pk} <= {ph})"
                )
                if self.dialect == Dialect.MYSQL:
                    if self.engine_version.supports_mysql_row_alias_upsert:
                        aliased_chunk_sel = [
                            f"{expr} AS {col}" for expr, col in zip(all_select_exprs, q_all_backing_cols)
                        ]
                        chunk_upd = ", ".join(f"{c} = __mv_src.{c}" for c in q_all_backing_cols)
                        chunk_upsert_sql: Optional[str] = (
                            f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                            f"SELECT * FROM (\n"
                            f"    SELECT {', '.join(aliased_chunk_sel)}\n"
                            f"    FROM {from_base_sql}\n"
                            f"    WHERE {chunk_w}\n"
                            f") AS __mv_src\n"
                            f"ON DUPLICATE KEY UPDATE {chunk_upd};"
                        )
                    else:
                        chunk_upsert_sql = (
                            f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                            f"SELECT {', '.join(all_select_exprs)}\nFROM {from_base_sql}\n"
                            f"WHERE {chunk_w}\n"
                            f"ON DUPLICATE KEY UPDATE {upd_set};"
                        )
                else:
                    chunk_upsert_sql = (
                        f"INSERT INTO {mv_table} ({', '.join(q_all_backing_cols)})\n"
                        f"SELECT {', '.join(all_select_exprs)}\nFROM {from_base_sql}\n"
                        f"WHERE {chunk_w}\n"
                        f"ON CONFLICT ({', '.join(q_mv_pk_cols)}) DO UPDATE SET {upd_set};"
                    )
            else:
                chunk_upsert_sql = None
            pk_old_preds = [
                f"{mv_table}.{self._q(mv_c)} = OLD.{self._q(src_c)}"
                for mv_c, src_c in zip(mv_pk_cols, pk_names)
            ]
            del_where = " AND ".join(pk_old_preds)
            del_stmt = f"DELETE FROM {mv_table}\n    WHERE {del_where};"
            del_stmts_list = [del_stmt]

            base_pk_eq_old = " AND ".join(
                f"{q_base_t}.{self._q(src_c)} = OLD.{self._q(src_c)}"
                for src_c in pk_names
            )
            base_where_parts: List[str] = []
            if self.view.where_clause:
                base_where_parts.append(f"({self._rewrite_expr_for_record(self.view.where_clause, q_base_t)})")
            if self.watermark_catalog_table and wm_pk_col:
                q_wm = self._quote_pk_range_col(wm_pk_col)
                view_lit = self.db_view_name.replace("'", "''")
                base_where_parts.append(
                    f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                    f"WHERE view_name = '{view_lit}' AND backfill_active = 1 "
                    f"AND (backfill_cursor_pk IS NULL OR ({q_base_t}.{q_wm} > backfill_cursor_pk AND {q_base_t}.{q_wm} <= backfill_max_pk)))"
                )
            and_base_where = f" AND {' AND '.join(base_where_parts)}" if base_where_parts else ""
            upd_del_stmt = (
                f"DELETE FROM {mv_table}\n"
                f"    WHERE {del_where}\n"
                f"    AND NOT EXISTS (SELECT 1 FROM {q_base_t} WHERE {base_pk_eq_old}{and_base_where});"
            )
            upd_old_list = [upd_del_stmt]
            upd_clean_list: List[str] = []
        else:
            all_base_cols = list(self.primary_schema.columns.keys())
            pk_defs = [c for c in self.primary_schema.columns.values() if c.primary_key]
            has_nullable_pk = (not pk_defs) or any(not c.not_null for c in pk_defs)
            pk_hash_cols = all_base_cols if (has_nullable_pk and all_base_cols) else (pk_names or sel_exprs)
            q_pk_hash_cols = (
                [f"{q_base_prefix}{self._q(c)}" for c in pk_hash_cols]
                if (has_nullable_pk and all_base_cols) or pk_names
                else sel_exprs
            )
            pk_expr_base = self._projection_row_pk_expr(q_pk_hash_cols)
            q_user_columns = [self._q(c) for c in user_columns]
            inner_pop_cols = [
                f"{pk_expr_base} AS __mv_k0",
                *[f"{expr} AS __mv_c{j}" for j, expr in enumerate(sel_exprs)],
            ]
            outer_pop_cols = ["__mv_p.__mv_k0", *[f"__mv_p.__mv_c{j}" for j in range(len(sel_exprs))]]
            initial_populate_sql = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                f"SELECT {', '.join(outer_pop_cols)}, COUNT(*)\n"
                f"FROM (\n"
                f"    SELECT {', '.join(inner_pop_cols)} FROM {from_base_sql}{w_sql}\n"
                f") AS __mv_p\n"
                f"GROUP BY {', '.join(outer_pop_cols)};"
            )
            pk_new = self._projection_row_pk_expr([f"NEW.{self._q(c)}" for c in pk_hash_cols])
            pk_old = self._projection_row_pk_expr([f"OLD.{self._q(c)}" for c in pk_hash_cols])
            if self.dialect == Dialect.MYSQL:
                if self.engine_version.supports_mysql_row_alias_upsert:
                    upd_set = ", ".join(f"{self._q(c)} = __mv_new.{self._q(c)}" for c in user_columns)
                    plus_stmt = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                        f"    VALUES ({pk_new}, {', '.join(new_vals)}, 1) AS __mv_new\n"
                        f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                    )
                else:
                    upd_set = ", ".join(f"{self._q(c)} = VALUES({self._q(c)})" for c in user_columns)
                    plus_stmt = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                        f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                        f"    WHERE TRUE\n"
                        f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                    )
            else:
                upd_set = ", ".join(f"{self._q(c)} = excluded.{self._q(c)}" for c in user_columns)
                plus_stmt = (
                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                    f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                    f"    WHERE TRUE\n"
                    f"    ON CONFLICT (__mv_pk) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                )
            if wm_pk_col:
                ph = "?" if self.dialect == Dialect.SQLITE else "%s"
                q_wm_pk = f"{q_base_prefix}{self._quote_pk_range_col(wm_pk_col)}"
                chunk_w = (
                    f"({self.view.where_clause}) AND ({q_wm_pk} > {ph} AND {q_wm_pk} <= {ph})"
                    if self.view.where_clause
                    else f"({q_wm_pk} > {ph} AND {q_wm_pk} <= {ph})"
                )
                grouped_chunk_sel = (
                    f"SELECT {', '.join(f'{oc} AS {tc}' for oc, tc in zip(outer_pop_cols, ['__mv_pk', *q_user_columns]))}, "
                    f"COUNT(*) AS __mv_row_count\n"
                    f"FROM (\n"
                    f"SELECT {', '.join(inner_pop_cols)}\n"
                    f"FROM {from_base_sql}\n"
                    f"WHERE {chunk_w}\n"
                    f") AS __mv_p\n"
                    f"GROUP BY {', '.join(outer_pop_cols)}"
                )
                if self.dialect == Dialect.MYSQL:
                    if self.engine_version.supports_mysql_row_alias_upsert:
                        chunk_upd = ", ".join(f"{c} = __mv_src.{c}" for c in q_user_columns)
                        chunk_upsert_sql = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"SELECT * FROM (\n"
                            f"{grouped_chunk_sel}\n"
                            f") AS __mv_src\n"
                            f"ON DUPLICATE KEY UPDATE {chunk_upd}, __mv_row_count = {mv_table}.__mv_row_count + __mv_src.__mv_row_count;"
                        )
                    else:
                        chunk_upsert_sql = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                            f"{grouped_chunk_sel}\n"
                            f"ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + VALUES(__mv_row_count);"
                        )
                else:
                    chunk_upsert_sql = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\n"
                        f"{grouped_chunk_sel}\n"
                        f"ON CONFLICT (__mv_pk) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + excluded.__mv_row_count;"
                    )
            else:
                chunk_upsert_sql = None
            minus_stmt = (
                f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                f"    WHERE {mv_table}.__mv_pk = ({pk_old});"
            )
            clean_stmt = (
                f"DELETE FROM {mv_table}\n"
                f"    WHERE ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
            )
            del_stmts_list = [minus_stmt, clean_stmt]
            upd_old_list = [minus_stmt]
            upd_clean_list = [clean_stmt]

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

        if self.watermark_catalog_table and wm_pk_col:
            q_pk = self._quote_pk_range_col(wm_pk_col)
            view_lit = self.db_view_name.replace("'", "''")
            wm_new = (
                f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                f"WHERE view_name = '{view_lit}' AND backfill_active = 1 "
                f"AND (backfill_cursor_pk IS NULL OR (NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)))"
            )
            wm_old = (
                f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                f"WHERE view_name = '{view_lit}' AND backfill_active = 1 "
                f"AND (backfill_cursor_pk IS NULL OR (OLD.{q_pk} > backfill_cursor_pk AND OLD.{q_pk} <= backfill_max_pk)))"
            )
            where_new = f"({where_new}) AND ({wm_new})" if where_new else wm_new
            where_old = f"({where_old}) AND ({wm_old})" if where_old else wm_old

        pg_ai_body: Optional[str] = None
        pg_ad_body: Optional[str] = None
        pg_au_body: Optional[str] = None

        trunc_stmts = [f"DELETE FROM {mv_table};"]
        if self.dialect == Dialect.POSTGRES:
            rec = "__mv_d"
            pg_where = self._rewrite_expr_for_record(self.view.where_clause, rec) if self.view.where_clause else None
            pg_wm = self._pg_watermark_predicate(rec, wm_pk_col) if wm_pk_col else None
            if pg_wm:
                pg_where = f"({pg_where}) AND ({pg_wm})" if pg_where else pg_wm
            pg_values = {
                it.alias: self._rewrite_expr_for_record(it.raw_expr, rec) for it in self.view.select_items
            }

            def _pg_from(signed: str) -> str:
                return f"(\n        {signed}\n        ) AS {rec}"

            if uses_native_pk:
                pk_col_lower = {c.lower() for c in mv_pk_cols}
                for hid_col, src_col, _dt in hidden_pk_defs:
                    pg_values[hid_col] = f"{rec}.{self._q(src_col)}"
                value_cols = [c for c in [*user_columns, *[h[0] for h in hidden_pk_defs]] if c.lower() not in pk_col_lower]
                pg_bodies = self._pg_keyed_rows_bodies(
                    mv_table,
                    mv_pk_cols,
                    value_cols,
                    [PgDeltaSource(
                        from_sql=_pg_from,
                        sign=f"{rec}.{SIGN}",
                        where=pg_where,
                        keys=[f"{rec}.{self._q(src_c)}" for src_c in pk_names],
                        values=pg_values,
                    )],
                    base_t=self.view.base_table,
                )
            else:
                pg_bodies = self._pg_counted_rows_bodies(
                    mv_table,
                    "__mv_pk",
                    user_columns,
                    ["__mv_row_count"],
                    [(
                        PgDeltaSource(
                            from_sql=_pg_from,
                            sign=f"{rec}.{SIGN}",
                            where=pg_where,
                            keys=[self._projection_row_pk_expr([f"{rec}.{self._q(c)}" for c in pk_hash_cols])],  # pyright: ignore[reportPossiblyUnboundVariable]  # bound in the non-native-PK branch at the same uses_native_pk guard
                            values=pg_values,
                        ),
                        {"__mv_row_count": "1"},
                    )],
                    f"{mv_table}.__mv_row_count <= 0",
                    base_t=self.view.base_table,
                )
            pg_ai_body, pg_au_body, pg_ad_body = pg_bodies["INSERT"], pg_bodies["UPDATE"], pg_bodies["DELETE"]
            trunc_stmts = [f"DELETE FROM {mv_table};", initial_populate_sql]

        upd_key_cols = list(pk_names if uses_native_pk else pk_hash_cols)  # pyright: ignore[reportPossiblyUnboundVariable]  # bound in the non-native-PK branch at the same uses_native_pk guard
        if (
            self.watermark_catalog_table
            and wm_pk_col
            and wm_pk_col.lower() not in {c.lower() for c in upd_key_cols}
        ):
            upd_key_cols.append(wm_pk_col)
        trigger_ddls, trig_names, fn_names = self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=[plus_stmt],
            del_stmts=del_stmts_list,
            upd_old_stmts=upd_old_list,
            upd_new_stmts=[plus_stmt],
            upd_cleanup_old_stmts=upd_clean_list,
            trunc_stmts=trunc_stmts,
            where_new=where_new,
            where_old=where_old,
            update_key_columns=upd_key_cols,
            pg_ai_body=pg_ai_body,
            pg_ad_body=pg_ad_body,
            pg_au_body=pg_au_body,
        )

        if wm_pk_col and chunk_upsert_sql:
            inc_spec = IncrementalBackfillSpec(
                mode="pk_range_fold",
                pk_column=wm_pk_col,
                chunk_sql_templates=[chunk_upsert_sql],
            )
        else:
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

