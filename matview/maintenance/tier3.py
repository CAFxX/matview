"""Tier 3: Universal atomic query refresh rules."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    _parse_single_create_view_stmt,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    parse_sql_ast,
    parse_view_ast,
    quote_sql_identifier,
)


class Tier3Mixin(MixinHost):
    def _compile_tier3_atomic_sync(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        col_defs_sql: List[str] = []
        user_columns: List[str] = []

        extra_auto_cols = self._auto_backing_extra_index_cols()
        indexed_cols_lower = self._indexed_column_names_lower(extra_auto_cols)
        if len(self.view.select_items) == 1 and self.view.select_items[0].raw_expr.strip() == "*":
            for j in self.view.joins or []:
                t_schema = self._schemas_lower.get(j.table_name.lower())
                if t_schema:
                    for c_def in t_schema.columns.values():
                        if c_def.name not in user_columns:
                            user_columns.append(c_def.name)
                            dt = "BLOB" if self.dialect == Dialect.SQLITE else self._normalize_dialect_type(c_def.data_type)
                            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and c_def.name.lower() in indexed_cols_lower:
                                dt = "VARCHAR(255)"
                            col_defs_sql.append(f"{self._q(c_def.name)} {dt}")
            if not user_columns and self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    user_columns.append(c_def.name)
                    dt = "BLOB" if self.dialect == Dialect.SQLITE else self._normalize_dialect_type(c_def.data_type)
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and c_def.name.lower() in indexed_cols_lower:
                        dt = "VARCHAR(255)"
                    col_defs_sql.append(f"{self._q(c_def.name)} {dt}")
        else:
            # For top-level compound set operations (e.g. UNION ALL ... ORDER BY / LIMIT,
            # keyless UNION ALL, or multi-branch INTERSECT/EXCEPT), widen each column's
            # storage type across all branches rather than using only branch 0.
            setop_col_types: Optional[List[str]] = None
            if self.dialect != Dialect.SQLITE and self.view.has_compound_set_op:
                try:
                    q_ast = parse_sql_ast(self.view.raw_select_sql.strip(), dialect=self.dialect)
                    if isinstance(q_ast, sqlglot_exp.SetOperation):
                        _, b_sqls = _extract_set_op_branches_from_ast(
                            q_ast, self.view.raw_select_sql.strip(), dialect=self.dialect
                        )
                        if len(b_sqls) >= 2:
                            parsed_bs: List[ViewSpec] = []
                            known_t = set(self._schemas_lower.keys())
                            for bi, b_sql in enumerate(b_sqls):
                                bs = _parse_single_create_view_stmt(
                                    f"CREATE VIEW __t3_set_b_{bi} AS {b_sql}",
                                    known_tables=known_t,
                                    dialect=self.dialect,
                                )
                                parsed_bs.append(bs)
                            if (
                                parsed_bs
                                and all(len(bs.select_items) == len(self.view.select_items) for bs in parsed_bs)
                                and all(not any(it.raw_expr.strip() == "*" for it in bs.select_items) for bs in parsed_bs)
                            ):
                                m_ctx = self._union_all_common_ctx(parsed_bs)
                                if m_ctx.primary_schema:
                                    setop_col_types = [
                                        m_ctx.primary_schema.columns[fb_it.alias].data_type
                                        for fb_it in parsed_bs[0].select_items
                                    ]
                except Exception:
                    setop_col_types = None

            for col_i, item in enumerate(self.view.select_items):
                user_columns.append(item.alias)
                if self.dialect == Dialect.SQLITE:
                    # Tier 3 handles CTEs, compound set operations, ORDER BY,
                    # and compound aggregate expressions where any declared
                    # SQLite affinity could coerce runtime storage classes.
                    col_type = "BLOB"
                elif setop_col_types is not None and col_i < len(setop_col_types):
                    col_type = setop_col_types[col_i]
                else:
                    col_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and col_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    col_type = "VARCHAR(255)"
                col_defs_sql.append(f"{self._q(item.alias)} {col_type}")

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
            extra_auto_index_cols=extra_auto_cols,
        )

        backing_table_ddl = self._create_table_ddl(
            mv_table,
            col_defs_sql,
            is_backing_store=True,
        )

        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        raw_query = self._rewrite_sqlite_total_in_sql(self.view.raw_select_sql.strip().rstrip(";"))
        q_user_columns = [self._q(c) for c in user_columns]
        cols_csv = ", ".join(q_user_columns)
        initial_populate_sql = f"INSERT INTO {mv_table} ({cols_csv})\n{raw_query};"

        ord_t3 = ", ".join(str(k) for k in range(1, len(user_columns) + 1))
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{raw_query}\n) AS __mv_init\n"
                f"ORDER BY {ord_t3}\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        refresh_stmts = [
            f"DELETE FROM {mv_table};",
            f"INSERT INTO {mv_table} ({cols_csv})\n    {raw_query};",
        ]

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        is_pg = self.dialect == Dialect.POSTGRES
        # PostgreSQL: one per-view lock row serializes the full recompute;
        # the DELETE/INSERT run after it (fresh READ COMMITTED snapshot).
        pg_refresh = "\n".join([self._pg_view_lock_stmt(), *refresh_stmts]) if is_pg else None
        for ref_table in self.view.referenced_tables:
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=ref_table,
                ins_stmts=refresh_stmts,
                del_stmts=refresh_stmts,
                upd_old_stmts=[],
                upd_new_stmts=refresh_stmts,
                trunc_stmts=[self._pg_view_lock_stmt(), *refresh_stmts] if is_pg else refresh_stmts,
                pg_ai_body=pg_refresh,
                pg_ad_body=pg_refresh,
                pg_au_body=pg_refresh,
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
            auxiliary_table_ddls=self._pg_view_lock_ddls() if is_pg else [],
            auxiliary_table_names=[self._pg_view_lock_table()] if is_pg else [],
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

