"""Unified dialect trigger, readonly view, and incremental backfill emitters."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .._host import MixinHost
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
from ..logical_ir import ColumnDef, JoinSpec, MaterializedViewPlan, SelectItem, TableSchema, ViewSpec
from ..sql_frontend import (
    _ast_table_expr,
    _find_top_level_token,
    _has_subquery_ast,
    _split_top_level,
    _tokenize_sql,
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


class DialectEmitMixin(MixinHost):
    def _build_readonly_view_and_triggers(
        self,
        view_name: str,
        mv_table: str,
        user_columns: Sequence[str],
        select_exprs_override: Optional[Sequence[str]] = None,
        extra_where_filter: Optional[str] = None,
        repeat_count_col: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        db_view = self.db_view_name
        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        sg_d = _sqlglot_dialect(self.dialect)
        q_db_view = _ast_table_expr(db_view, self.dialect, allow_qualified=False).sql(dialect=sg_d)
        user_cols_lower = {c.lower() for c in user_columns}
        proj_exprs = (
            [self._q(ex) if ex.lower() in user_cols_lower else ex for ex in select_exprs_override]
            if select_exprs_override
            else [self._q(c) for c in user_columns]
        )

        if repeat_count_col:
            q_mv_table = _ast_table_expr(mv_table, self.dialect).sql(dialect=sg_d)
            q_rep = self._q(repeat_count_col)
            qual_proj = (
                [
                    f"{q_mv_table}.{self._q(ex)}" if ex.lower() in user_cols_lower else ex
                    for ex in select_exprs_override
                ]
                if select_exprs_override
                else [f"{q_mv_table}.{self._q(c)}" for c in user_columns]
            )
            proj_csv = ", ".join(qual_proj)
            base_where = (
                f"({extra_where_filter}) AND {q_mv_table}.{q_rep} > 0"
                if extra_where_filter
                else f"{q_mv_table}.{q_rep} > 0"
            )
            if self.dialect == Dialect.POSTGRES:
                reader_view_ddl = (
                    f"CREATE VIEW {q_db_view} AS\n"
                    f"SELECT {proj_csv}\n"
                    f"FROM {q_mv_table}\n"
                    f"CROSS JOIN LATERAL generate_series(1, {q_mv_table}.{q_rep}) AS __mv_rep(__mv_n)\n"
                    f"WHERE {base_where};"
                )
            elif self.dialect == Dialect.MYSQL:
                reader_view_ddl = (
                    f"CREATE ALGORITHM = TEMPTABLE VIEW {q_db_view} AS\n"
                    f"WITH RECURSIVE __mv_bits(__mv_b) AS (\n"
                    f"    SELECT 0 UNION ALL SELECT 1\n"
                    f"), __mv_seq(__mv_n, __mv_max) AS (\n"
                    f"    SELECT CAST(1 AS SIGNED), COALESCE(MAX({q_rep}), 0) FROM {q_mv_table}\n"
                    f"    UNION ALL\n"
                    f"    SELECT __mv_seq.__mv_n * 2 + __mv_bits.__mv_b, __mv_seq.__mv_max\n"
                    f"    FROM __mv_seq\n"
                    f"    CROSS JOIN __mv_bits\n"
                    f"    WHERE __mv_seq.__mv_n * 2 + __mv_bits.__mv_b <= __mv_seq.__mv_max\n"
                    f")\n"
                    f"SELECT {proj_csv}\n"
                    f"FROM {q_mv_table}\n"
                    f"JOIN __mv_seq ON __mv_seq.__mv_n <= {q_mv_table}.{q_rep}\n"
                    f"WHERE {base_where} AND EXISTS (SELECT 1);"
                )
                return reader_view_ddl, [], [], []
            else:
                reader_view_ddl = (
                    f"CREATE VIEW {q_db_view} AS\n"
                    f"WITH RECURSIVE __mv_bits(__mv_b) AS (\n"
                    f"    SELECT 0 UNION ALL SELECT 1\n"
                    f"), __mv_seq(__mv_n, __mv_max) AS (\n"
                    f"    SELECT 1, COALESCE(MAX({q_rep}), 0) FROM {q_mv_table}\n"
                    f"    UNION ALL\n"
                    f"    SELECT __mv_seq.__mv_n * 2 + __mv_bits.__mv_b, __mv_seq.__mv_max\n"
                    f"    FROM __mv_seq\n"
                    f"    CROSS JOIN __mv_bits\n"
                    f"    WHERE __mv_seq.__mv_n * 2 + __mv_bits.__mv_b <= __mv_seq.__mv_max\n"
                    f")\n"
                    f"SELECT {proj_csv}\n"
                    f"FROM {q_mv_table}\n"
                    f"JOIN __mv_seq ON __mv_seq.__mv_n <= {q_mv_table}.{q_rep}\n"
                    f"WHERE {base_where};"
                )
        elif self.dialect == Dialect.MYSQL:
            where_cond = (
                f"({extra_where_filter}) AND EXISTS (SELECT 1)"
                if extra_where_filter
                else "EXISTS (SELECT 1)"
            )
            reader_view_ddl = build_create_view_sql(
                db_view,
                mv_table,
                proj_exprs,
                where_sql=where_cond,
                dialect=self.dialect,
                algorithm="TEMPTABLE",
                allow_qualified_view=False,
            )
            return reader_view_ddl, [], [], []
        else:
            reader_view_ddl = build_create_view_sql(
                db_view,
                mv_table,
                proj_exprs,
                where_sql=extra_where_filter,
                dialect=self.dialect,
                allow_qualified_view=False,
            )

        if self.dialect == Dialect.SQLITE:
            ro_trigs: List[str] = []
            ro_names: List[str] = []
            for op, suf in (("INSERT", "ins"), ("UPDATE", "upd"), ("DELETE", "del")):
                t_name = self._bounded_ident(f"{p}trg_", log_vname, f"__ro_{suf}")
                ro_names.append(t_name)
                ro_trigs.append(
                    render_procedural_sql(
                        "ro_trigger_sqlite",
                        trg_name=self._q(t_name),
                        op=op,
                        db_view=q_db_view,
                        raw_db_view=db_view.replace("'", "''"),
                    )
                )
            return reader_view_ddl, ro_trigs, ro_names, []

        # PostgreSQL: INSTEAD OF trigger on view
        fn_ro = self._bounded_ident(f"{p}fn_readonly_", log_vname)
        t_ro = self._bounded_ident(f"{p}trg_", log_vname, "__readonly")
        fn_ddl = render_procedural_sql("ro_function_pg", fn_name=self._q(fn_ro))
        t_ddl = render_procedural_sql(
            "ro_trigger_pg",
            trg_name=self._q(t_ro),
            db_view=q_db_view,
            fn_name=self._q(fn_ro),
        )
        return reader_view_ddl, [fn_ddl, t_ddl], [t_ro], [fn_ro]

    def _build_dialect_incremental_populate_sqls(self, raw_plan: MaterializedViewPlan) -> List[str]:
        spec = raw_plan.incremental_backfill_spec
        b_size = self.batch_size
        if spec is None:
            return list(raw_plan.initial_populate_sqls)

        # Case 1: PK / rowid range folding (`pk_range_fold`)
        if spec.mode == "pk_range_fold" and spec.pk_column and spec.chunk_sql_templates:
            base_t = raw_plan.view_spec.base_table
            base_alias = raw_plan.view_spec.joins[0].alias if raw_plan.view_spec.joins else base_t
            is_base_aliased = base_alias.lower() != base_t.lower()
            pk_col = spec.pk_column
            tmpl = spec.chunk_sql_templates[0].strip().rstrip(";")
            ph = "?" if self.dialect == Dialect.SQLITE else "%s"
            is_global_agg = (
                any(it.is_aggregate for it in raw_plan.view_spec.select_items)
                and not raw_plan.view_spec.group_by_cols
                and not raw_plan.view_spec.has_top_level_distinct
            )
            q_base_t = self._q(base_t)
            q_pk_col = self._quote_pk_range_col(pk_col)
            q_table_ref = self._q(base_alias) if is_base_aliased else q_base_t
            q_cond_pk = f"{q_table_ref}.{q_pk_col}" if is_base_aliased else q_pk_col
            from_join_target = f"{q_base_t} AS {self._q(base_alias)}" if is_base_aliased else q_base_t
            from_base_pat = (
                rf"\nFROM\s+{re.escape(q_base_t)}(?:\s+AS\s+{re.escape(self._q(base_alias))})?\b"
                if is_base_aliased
                else rf"\nFROM\s+{re.escape(q_base_t)}\b"
            )

            def _bound_follow_up_templates() -> List[str]:
                # Follow-up chunk statements (PostgreSQL/MySQL native
                # recompute of order-sensitive aggregates) are bound with
                # (cursor_pk, upper_pk) in that order, like the chunk itself.
                out: List[str] = []
                for extra in spec.chunk_sql_templates[1:]:
                    bound = extra.strip().rstrip(";")
                    # The procedural loop has the bounds in variables: drop the
                    # one-row bounds table (MariaDB rejects references to it
                    # from the EXISTS subquery of a multi-table UPDATE inside
                    # a stored procedure: 1054 Unknown column '__mv_p.lo').
                    for params_sql in (
                        "\n    FROM (SELECT CAST(%s AS BIGINT) AS lo, CAST(%s AS BIGINT) AS hi) AS __mv_p",
                        " INNER JOIN (SELECT CAST(%s AS SIGNED) AS lo, CAST(%s AS SIGNED) AS hi) AS __mv_p ON TRUE",
                    ):
                        if params_sql in bound:
                            bound = bound.replace(params_sql, "")
                            bound = bound.replace("__mv_p.lo", "__mv_cur_pk").replace("__mv_p.hi", "__mv_top_pk")
                            # Single-table UPDATE now: unqualified SET targets.
                            bound = re.sub(
                                r"(\bSET\s+|,\n\s+)[`\w]+\.([`\w]+ = \(SELECT\b)", r"\1\2", bound
                            )
                    for var in ("__mv_cur_pk", "__mv_top_pk"):
                        bound = bound.replace("%s", var, 1)
                    out.append(bound)
                return out

            if self.dialect == Dialect.SQLITE:
                # Replace `(pk_col > ? AND pk_col <= ?)` (or aliased `s.pk_col`) with recursive CTE batch join
                cond_pat = f"({q_cond_pk} > {ph} AND {q_cond_pk} <= {ph})"
                joined_tmpl = tmpl.replace(
                    cond_pat,
                    f"({q_table_ref}.{q_pk_col} > __mv_b.lo AND {q_table_ref}.{q_pk_col} <= __mv_b.hi)",
                )
                if joined_tmpl == tmpl and q_cond_pk != q_pk_col:
                    joined_tmpl = tmpl.replace(
                        f"({q_pk_col} > {ph} AND {q_pk_col} <= {ph})",
                        f"({q_table_ref}.{q_pk_col} > __mv_b.lo AND {q_table_ref}.{q_pk_col} <= __mv_b.hi)",
                    )
                joined_tmpl = re.sub(
                    from_base_pat,
                    f"\nFROM __mv_batches AS __mv_b JOIN {from_join_target} ON ({q_table_ref}.{q_pk_col} > __mv_b.lo AND {q_table_ref}.{q_pk_col} <= __mv_b.hi)",
                    joined_tmpl,
                    count=1,
                    flags=re.IGNORECASE,
                )
                return [
                    render_procedural_sql(
                        "backfill_pk_sqlite",
                        q_pk_col=q_pk_col,
                        q_base_t=q_base_t,
                        batch_size=b_size,
                        joined_tmpl=joined_tmpl,
                    )
                ]

            elif self.dialect == Dialect.POSTGRES:
                pg_chunk = tmpl.replace(
                    f"({q_cond_pk} > %s AND {q_cond_pk} <= %s)",
                    f"({q_cond_pk} > __mv_cur_pk AND {q_cond_pk} <= __mv_top_pk)",
                )
                if pg_chunk == tmpl and q_cond_pk != q_pk_col:
                    pg_chunk = tmpl.replace(
                        f"({q_pk_col} > %s AND {q_pk_col} <= %s)",
                        f"({q_pk_col} > __mv_cur_pk AND {q_pk_col} <= __mv_top_pk)",
                    )
                if is_global_agg:
                    pg_chunk = re.sub(
                        r"\nON CONFLICT\b",
                        "\nHAVING COUNT(*) > 0\nON CONFLICT",
                        pg_chunk,
                        count=1,
                    )
                pg_chunk = ";\n            ".join([pg_chunk, *_bound_follow_up_templates()])
                fallback_populate_sqls = (
                    [pop_s.strip().rstrip(";") for pop_s in raw_plan.initial_populate_sqls]
                    if is_global_agg
                    else []
                )
                return [
                    render_procedural_sql(
                        "backfill_pk_pg",
                        batch_size=b_size,
                        q_pk_col=q_pk_col,
                        q_base_t=q_base_t,
                        pg_chunk=pg_chunk,
                        fallback_populate_sqls=fallback_populate_sqls,
                        q_backing_table=self._q(raw_plan.backing_table_name),
                    )
                ]

            elif self.dialect == Dialect.MYSQL:
                my_chunk = tmpl.replace(
                    f"({q_cond_pk} > %s AND {q_cond_pk} <= %s)",
                    f"({q_cond_pk} > __mv_cur_pk AND {q_cond_pk} <= __mv_top_pk)",
                )
                if my_chunk == tmpl and q_cond_pk != q_pk_col:
                    my_chunk = tmpl.replace(
                        f"({q_pk_col} > %s AND {q_pk_col} <= %s)",
                        f"({q_pk_col} > __mv_cur_pk AND {q_pk_col} <= __mv_top_pk)",
                    )
                if is_global_agg:
                    if "\n) AS __mv_src\nON DUPLICATE KEY UPDATE" in my_chunk:
                        my_chunk = my_chunk.replace(
                            "\n) AS __mv_src\nON DUPLICATE KEY UPDATE",
                            "\n    HAVING COUNT(*) > 0\n) AS __mv_src\nON DUPLICATE KEY UPDATE",
                            1,
                        )
                    else:
                        my_chunk = re.sub(
                            r"\nON DUPLICATE KEY UPDATE\b",
                            "\nHAVING COUNT(*) > 0\nON DUPLICATE KEY UPDATE",
                            my_chunk,
                            count=1,
                        )
                my_chunk = ";\n            ".join([my_chunk, *_bound_follow_up_templates()])
                fallback_populate_sqls = (
                    [pop_s.strip().rstrip(";") for pop_s in raw_plan.initial_populate_sqls]
                    if is_global_agg
                    else []
                )
                proc_name = self._bounded_ident(f"{self.object_prefix}proc_backfill_", self.logical_view_name)
                return [
                    render_procedural_sql("drop_procedure_mysql", proc_name=proc_name),
                    render_procedural_sql(
                        "backfill_pk_mysql_proc",
                        proc_name=proc_name,
                        batch_size=b_size,
                        q_pk_col=q_pk_col,
                        q_base_t=q_base_t,
                        my_chunk=my_chunk,
                        fallback_populate_sqls=fallback_populate_sqls,
                        q_backing_table=self._q(raw_plan.backing_table_name),
                    ),
                    render_procedural_sql("call_procedure_mysql", proc_name=proc_name),
                    render_procedural_sql("drop_procedure_mysql", proc_name=proc_name),
                ]

        # Case 2: Group-batch or Offset-batch over `raw_plan.initial_populate_sqls`
        inc_stmts: List[str] = []
        main_pk_cols: List[str] = []
        if raw_plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA:
            if self._alg_mode in ("single", "join", "union_all_agg"):
                main_pk_cols = self._tier1_pk_cols()
            elif self._alg_mode == "setop":
                setop_pks = getattr(self, "_setop_pk_cols", ())
                main_pk_cols = list(setop_pks) if setop_pks else ["__mv_pk"]
            elif self._alg_mode == "projection":
                main_pk_cols = self._projection_pk_cols()

        sg_dial = _sqlglot_dialect(self.dialect)
        for idx_s, pop_sql in enumerate(raw_plan.initial_populate_sqls):
            clean_pop = pop_sql.strip().rstrip(";")
            try:
                pop_ast = parse_sql_ast(clean_pop, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                pop_ast = None

            if not isinstance(pop_ast, sqlglot_exp.Insert) or not pop_ast.this:
                inc_stmts.append(pop_sql)
                continue

            target_tbl = (
                pop_ast.this.this.name
                if isinstance(pop_ast.this, sqlglot_exp.Schema)
                else (pop_ast.this.name or str(pop_ast.this))
            )
            cols = [c.name for c in pop_ast.this.expressions] if isinstance(pop_ast.this, sqlglot_exp.Schema) else []
            sel_node = pop_ast.expression
            if not isinstance(sel_node, sqlglot_exp.Select):
                inc_stmts.append(pop_sql)
                continue

            pop_toks = _tokenize_sql(clean_pop, read=sg_dial)
            sel_tok_pos: Optional[int] = None
            sel_end_pos: int = len(clean_pop)
            p_depth = 0
            for idx_tk, ptk in enumerate(pop_toks):
                if ptk.token_type == SQLGlotTokenType.L_PAREN:
                    p_depth += 1
                elif ptk.token_type == SQLGlotTokenType.R_PAREN and p_depth > 0:
                    p_depth -= 1
                elif p_depth == 0:
                    if sel_tok_pos is None and ptk.token_type in (SQLGlotTokenType.SELECT, getattr(SQLGlotTokenType, "WITH", SQLGlotTokenType.SELECT)):
                        sel_tok_pos = ptk.start
                    elif sel_tok_pos is not None and ptk.token_type == SQLGlotTokenType.ON and idx_tk + 1 < len(pop_toks):
                        nxt_up = pop_toks[idx_tk + 1].text.upper()
                        if nxt_up in ("CONFLICT", "DUPLICATE"):
                            sel_end_pos = ptk.start
                            break
            select_body = clean_pop[sel_tok_pos:sel_end_pos].strip() if sel_tok_pos is not None else sel_node.sql(sg_dial).strip()

            aliased_inner = select_body
            sb_toks = _tokenize_sql(select_body, read=sg_dial)
            if (
                sb_toks
                and sb_toks[0].token_type == SQLGlotTokenType.SELECT
                and not sel_node.args.get("distinct")
                and not (sel_node.args.get("with_") or sel_node.args.get("with"))
                and (sel_node.args.get("from_") or sel_node.args.get("from")) is not None
            ):
                from_tok_sb = _find_top_level_token(select_body, (SQLGlotTokenType.FROM,), dialect=self.dialect)
                if from_tok_sb is not None:
                    proj_slice = select_body[sb_toks[0].end + 1 : from_tok_sb.start].strip()
                    from_and_rest = select_body[from_tok_sb.start :].strip()
                    raw_projs = [p.strip() for p in _split_top_level(proj_slice, ",", dialect=self.dialect) if p.strip()]
                    if len(raw_projs) == len(cols):
                        aliased_list = []
                        for c_name, r_proj in zip(cols, raw_projs):
                            rp_toks = _tokenize_sql(r_proj, read=sg_dial)
                            if (
                                len(rp_toks) >= 2
                                and rp_toks[-2].token_type == SQLGlotTokenType.ALIAS
                                and _unquote_ident(rp_toks[-1].text, dialect=self.dialect).lower() == _unquote_ident(c_name, dialect=self.dialect).lower()
                            ):
                                aliased_list.append(r_proj)
                            else:
                                aliased_list.append(f"{r_proj} AS {self._q(c_name)}")
                        aliased_inner = f"SELECT {', '.join(aliased_list)}\n{from_and_rest}"

            is_aux_dist = "__val" in cols and "__ref_cnt" in cols
            tbl_pks = (
                ([*main_pk_cols, "__val"] if is_aux_dist else main_pk_cols)
                if (is_aux_dist or target_tbl == raw_plan.backing_table_name)
                else []
            )
            tbl_pks = [c for c in tbl_pks if c in cols]
            q_cols = [self._q(c) for c in cols]
            q_tbl_pks = [self._q(c) for c in tbl_pks]

            if self.dialect == Dialect.SQLITE:
                upsert_tail = ""
                if tbl_pks:
                    upd_cols = [f"{self._q(c)} = excluded.{self._q(c)}" for c in cols if c not in tbl_pks] or [
                        f"{self._q(tbl_pks[0])} = excluded.{self._q(tbl_pks[0])}"
                    ]
                    upsert_tail = f"\nON CONFLICT ({', '.join(q_tbl_pks)}) DO UPDATE SET {', '.join(upd_cols)}"
                inc_stmts.append(
                    render_procedural_sql(
                        "backfill_offset_sqlite",
                        batch_size=b_size,
                        aliased_inner=aliased_inner,
                        q_target_tbl=self._q(target_tbl),
                        q_cols_csv=", ".join(q_cols),
                        upsert_tail=upsert_tail,
                    )
                )
            elif self.dialect == Dialect.POSTGRES:
                ord_cols = ", ".join(str(k) for k in range(1, len(cols) + 1))
                ord_tail = f"\n        ORDER BY {ord_cols}" if not self.view.has_limit_or_order else f"\n        ORDER BY {ord_cols}"
                if tbl_pks and self.engine_version.supports_pg_merge:
                    non_pk_cols = [c for c in cols if c not in tbl_pks]
                    if non_pk_cols:
                        matched_clause = f"WHEN MATCHED THEN\n            UPDATE SET {', '.join(f'{self._q(c)} = __mv_src.{self._q(c)}' for c in non_pk_cols)}"
                    else:
                        matched_clause = "WHEN MATCHED THEN DO NOTHING"
                    inc_stmts.append(
                        render_procedural_sql(
                            "backfill_offset_pg",
                            use_merge=True,
                            batch_size=b_size,
                            q_target_tbl=self._q(target_tbl),
                            q_cols_csv=", ".join(q_cols),
                            aliased_inner=aliased_inner,
                            ord_tail=ord_tail,
                            merge_on_csv=" AND ".join(f"__mv_dst.{self._q(c)} = __mv_src.{self._q(c)}" for c in tbl_pks),
                            matched_clause=matched_clause,
                            merge_vals_csv=", ".join(f"__mv_src.{self._q(c)}" for c in cols),
                        )
                    )
                else:
                    upsert_tail = ""
                    if tbl_pks:
                        upd_cols = [f"{self._q(c)} = excluded.{self._q(c)}" for c in cols if c not in tbl_pks] or [
                            f"{self._q(tbl_pks[0])} = excluded.{self._q(tbl_pks[0])}"
                        ]
                        upsert_tail = f"\n        ON CONFLICT ({', '.join(q_tbl_pks)}) DO UPDATE SET {', '.join(upd_cols)}"
                    inc_stmts.append(
                        render_procedural_sql(
                            "backfill_offset_pg",
                            use_merge=False,
                            batch_size=b_size,
                            q_target_tbl=self._q(target_tbl),
                            q_cols_csv=", ".join(q_cols),
                            aliased_inner=aliased_inner,
                            ord_tail=ord_tail,
                            upsert_tail=upsert_tail,
                        )
                    )
            elif self.dialect == Dialect.MYSQL:
                upsert_tail = ""
                if tbl_pks:
                    if self.engine_version.supports_mysql_row_alias_upsert:
                        upd_cols = [f"{self._q(c)} = __mv_init.{self._q(c)}" for c in cols if c not in tbl_pks] or [
                            f"{self._q(tbl_pks[0])} = __mv_init.{self._q(tbl_pks[0])}"
                        ]
                    else:
                        upd_cols = [f"{self._q(c)} = VALUES({self._q(c)})" for c in cols if c not in tbl_pks] or [
                            f"{self._q(tbl_pks[0])} = VALUES({self._q(tbl_pks[0])})"
                        ]
                    upsert_tail = f"\n        ON DUPLICATE KEY UPDATE {', '.join(upd_cols)}"
                ord_cols = ", ".join(str(k) for k in range(1, len(cols) + 1))
                ord_tail = f"\n        ORDER BY {ord_cols}"
                proc_name = self._bounded_ident(f"{self.object_prefix}proc_backfill_", self.logical_view_name, f"_{idx_s}")
                inc_stmts.extend([
                    render_procedural_sql("drop_procedure_mysql", proc_name=proc_name),
                    render_procedural_sql(
                        "backfill_offset_mysql_proc",
                        proc_name=proc_name,
                        batch_size=b_size,
                        q_target_tbl=self._q(target_tbl),
                        q_cols_csv=", ".join(q_cols),
                        aliased_inner=aliased_inner,
                        ord_tail=ord_tail,
                        upsert_tail=upsert_tail,
                    ),
                    render_procedural_sql("call_procedure_mysql", proc_name=proc_name),
                    render_procedural_sql("drop_procedure_mysql", proc_name=proc_name),
                ])
        return inc_stmts


    @staticmethod
    def _mariadb_row_key(t_schema: Any) -> Optional[Tuple[List[str], bool]]:
        """Columns that identify one row of the table (for the MariaDB
        ignored-update guard) and whether any of them is nullable: the
        primary key, else a UNIQUE key on NOT NULL columns, else a nullable
        UNIQUE key; None without any (no duplicate-key skip is possible)."""
        pk = list(t_schema.effective_pk_columns())
        if pk:
            return pk, False
        uniques = [
            list(idx.columns) for idx in t_schema.indexes
            if idx.unique and not idx.where_clause and idx.columns
            and all(c in t_schema.columns for c in idx.columns)
        ]
        for cols in uniques:
            if all(t_schema.columns[c].not_null for c in cols):
                return cols, False
        if uniques:
            return uniques[0], True
        return None

    def _identical_row_values_sql(self, old: str, new: str, is_nullable: bool) -> str:
        """True only when the OLD and NEW values are identical, not merely equal.

        Equality is not identity: under a case-insensitive or PAD SPACE
        collation 'ab' = 'AB ' holds, yet the view returns the new value.
        A guard that treats such an UPDATE as a no-op leaves the old value
        stored, so the comparison must be by value representation.
        """
        if self.dialect == Dialect.POSTGRES:
            # PostgreSQL row triggers are being replaced by statement-level
            # maintenance (bulk stage 2); keep their existing comparison.
            return self._null_safe_eq(old, new, is_nullable=is_nullable)
        if self.dialect == Dialect.MYSQL:
            # CAST(... AS BINARY) prints a FLOAT with about 6 significant
            # digits, so two different FLOAT values (0.8 and 0.8 * 1.0000001)
            # print alike: also require the values themselves to be equal
            # (measured natively, MySQL 9.7 / MariaDB 11.8: the UPDATE was
            # skipped and the view kept the old value).
            return f"(CAST({old} AS BINARY) <=> CAST({new} AS BINARY) AND {old} <=> {new})"
        # SQLite: IS compares with the column's collation (NOCASE, RTRIM), so
        # force BINARY. 3 IS 3.0 is true, but a column without affinity (BLOB
        # or no declared type) can change INTEGER 3 to REAL 3.0, which changes
        # the storage class of SUM and of every value the view returns; only a
        # change of neither value nor storage class may skip maintenance.
        # REAL -0.0 IS 0.0 is true with matching typeof('real') yet the stored
        # sign differs, so a REAL zero is never treated as unchanged.
        eq = f"{old} IS {new} COLLATE BINARY" if is_nullable else f"{old} = {new} COLLATE BINARY"
        return f"({eq} AND typeof({old}) = typeof({new}) AND NOT (typeof({old}) = 'real' AND {old} = 0.0))"

    def _emit_dialect_triggers_for_table(
        self,
        view_name: str,
        base_t: str,
        ins_stmts: List[str],
        del_stmts: List[str],
        upd_old_stmts: List[str],
        upd_new_stmts: List[str],
        upd_cleanup_old_stmts: Optional[List[str]] = None,
        trunc_stmts: Optional[List[str]] = None,
        where_new: Optional[str] = None,
        where_old: Optional[str] = None,
        update_key_columns: Optional[Sequence[str]] = None,
        pg_ai_body: Optional[str] = None,
        pg_ad_body: Optional[str] = None,
        pg_au_body: Optional[str] = None,
        pg_bracket: Optional[Any] = None,
        mysql_prelock: Optional[Dict[str, str]] = None,
        mysql_same_group_cond: Optional[str] = None,
        mysql_same_group_stmts: Optional[List[str]] = None,
        sqlite_same_group_cond: Optional[str] = None,
        sqlite_same_group_stmts: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        if self.dialect == Dialect.POSTGRES:
            # Statement-level transition-table triggers are the only
            # PostgreSQL mechanism (matview/dialects/pg_statement.py); every
            # strategy supplies set-based bodies, there is no row fallback.
            if pg_ai_body is None or pg_ad_body is None or pg_au_body is None:
                raise AssertionError(
                    f"no set-based PostgreSQL statement-level bodies for view {self.logical_view_name!r} on {base_t!r}"
                )
            trunc_body = "\n".join(trunc_stmts or [f"DELETE FROM {self._backing_table_name()};"])
            return self._pg_emit_statement_triggers(
                base_t,
                {"INSERT": pg_ai_body, "UPDATE": pg_au_body, "DELETE": pg_ad_body},
                trunc_body,
                bracket=pg_bracket,
            )

        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        stem = f"{log_vname}__{base_t}"
        t_ai_name = self._bounded_ident(f"{p}trg_", stem, "_ai")
        t_ad_name = self._bounded_ident(f"{p}trg_", stem, "_ad")
        t_au_name = self._bounded_ident(f"{p}trg_", stem, "_au")
        q_base_t = self._q(base_t)
        ref_cols = self._referenced_columns_for_table(base_t)
        t_schema = self._schemas_lower.get(base_t.lower())
        schema_cols_by_lower = (
            {c.lower(): c_def for c, c_def in t_schema.columns.items()}
            if t_schema
            else {}
        )
        # Projection backing rows (and watermark-gated backfill triggers) depend
        # on key columns even when the public SELECT omits them. A key-only UPDATE
        # must therefore fire the maintenance trigger and must not be suppressed
        # by the dialect's unchanged-column guard (SQLite and MySQL/MariaDB).
        implicit_rowid_aliases: List[str] = []
        seen_ref_cols = {c.lower() for c in ref_cols}
        for key_col in update_key_columns or ():
            k_low = key_col.lower()
            if k_low not in seen_ref_cols:
                if (
                    self.dialect == Dialect.SQLITE
                    and k_low in ("rowid", "_rowid_", "oid")
                    and k_low not in schema_cols_by_lower
                ):
                    # SQLite forbids `AFTER UPDATE OF rowid` (or `_rowid_`/`oid`)
                    # when it is an implicit pseudo-column; omit the OF list and
                    # guard via `OLD.<alias> = NEW.<alias>` in the WHEN clause.
                    if k_low not in implicit_rowid_aliases:
                        implicit_rowid_aliases.append(k_low)
                else:
                    ref_cols.append(key_col)
                    seen_ref_cols.add(k_low)
        # SQLite fires "AFTER UPDATE OF cols" only when a listed column is in
        # the SET list; a referenced generated column changes when the columns
        # it is computed from do, so it disables the column filter (the WHEN
        # guard below still skips rows whose referenced values are unchanged).
        refs_generated = t_schema is not None and any(
            (cd := schema_cols_by_lower.get(c.lower())) is not None and cd.generated for c in ref_cols
        )
        upd_of_clause = (
            f" OF {', '.join(self._q(c) for c in ref_cols)}"
            if ref_cols and self.dialect == Dialect.SQLITE and not refs_generated and not implicit_rowid_aliases
            else ""
        )

        unchanged_conds: List[str] = []
        if t_schema and ref_cols:
            for c_name in ref_cols:
                c_def = schema_cols_by_lower.get(c_name.lower())
                is_null = not (c_def.not_null if c_def else False)
                qc = self._q(c_name)
                unchanged_conds.append(self._identical_row_values_sql(f"OLD.{qc}", f"NEW.{qc}", is_null))
            for rowid_alias in implicit_rowid_aliases:
                unchanged_conds.append(f"OLD.{rowid_alias} = NEW.{rowid_alias}")
        changed_guard = f"NOT ({' AND '.join(unchanged_conds)})" if unchanged_conds else None
        if (
            self.dialect == Dialect.POSTGRES
            and getattr(self, "_alg_mode", None) == "single"
            and self._pgmy_native_recompute_items()
        ):
            # PostgreSQL order-sensitive aggregates follow the group's
            # sequential-scan (ctid) order. Every UPDATE writes a new row
            # version at a new ctid -- also one that changes no referenced
            # column -- and can thereby change the original view's float
            # result. Maintain (recompute) on every UPDATE.
            upd_of_clause = ""
            changed_guard = None

        cleanup_stmts = upd_cleanup_old_stmts or []

        if self.dialect == Dialect.SQLITE:
            def _to_sqlite_values_upsert(stmt: str) -> str:
                # Replace unconditional `SELECT <vals>\n    WHERE TRUE\n    ON CONFLICT`
                # with direct `VALUES (<vals>)\n    ON CONFLICT` to avoid an extra
                # SELECT subquery coroutine per row.
                return re.sub(
                    r"\n(\s*)SELECT\s+(.+?)\n\s*WHERE\s+TRUE\n(\s*ON\s+CONFLICT\b)",
                    r"\n\1VALUES (\2)\n\3",
                    stmt,
                    count=1,
                    flags=re.IGNORECASE | re.DOTALL,
                )

            lean_ins_stmts = [_to_sqlite_values_upsert(s) for s in ins_stmts]
            trig_ins = render_procedural_sql(
                "trigger_sqlite_after_insert",
                trg_name=t_ai_name,
                q_base_t=q_base_t,
                when_clause=where_new,
                body="\n    ".join(lean_ins_stmts),
            )
            trig_del = render_procedural_sql(
                "trigger_sqlite_after_delete",
                trg_name=t_ad_name,
                q_base_t=q_base_t,
                when_clause=where_old,
                body="\n    ".join(del_stmts),
            )
            upd_new_guarded = (
                [self._apply_where_guard_to_stmt(s, where_new) for s in upd_new_stmts]
                if where_new
                else [_to_sqlite_values_upsert(s) for s in upd_new_stmts]
            )
            upd_stmts = (
                [self._apply_where_guard_to_stmt(s, where_old) for s in upd_old_stmts]
                + upd_new_guarded
                + [self._apply_where_guard_to_stmt(s, where_old) for s in cleanup_stmts]
            )
            if sqlite_same_group_cond and sqlite_same_group_stmts:
                move_when = (
                    f"({changed_guard}) AND NOT ({sqlite_same_group_cond})"
                    if changed_guard
                    else f"NOT ({sqlite_same_group_cond})"
                )
                same_when_parts: List[str] = []
                if changed_guard:
                    same_when_parts.append(f"({changed_guard})")
                same_when_parts.append(f"({sqlite_same_group_cond})")
                if where_new:
                    same_when_parts.append(f"({where_new})")
                same_when = " AND ".join(same_when_parts)
                trig_upd = render_procedural_sql(
                    "trigger_sqlite_after_update",
                    trg_name=t_au_name,
                    upd_of_clause=upd_of_clause,
                    q_base_t=q_base_t,
                    when_clause=move_when,
                    body="\n    ".join(upd_stmts),
                )
                t_aus_name = self._bounded_ident(f"{p}trg_", stem, "_aus")
                trig_upd_same = render_procedural_sql(
                    "trigger_sqlite_after_update",
                    trg_name=t_aus_name,
                    upd_of_clause=upd_of_clause,
                    q_base_t=q_base_t,
                    when_clause=same_when,
                    body="\n    ".join(sqlite_same_group_stmts),
                )
                t_names = [t_ai_name, t_ad_name, t_au_name, t_aus_name]
                return [trig_ins, trig_del, trig_upd, trig_upd_same], t_names, []
            trig_upd = render_procedural_sql(
                "trigger_sqlite_after_update",
                trg_name=t_au_name,
                upd_of_clause=upd_of_clause,
                q_base_t=q_base_t,
                when_clause=changed_guard,
                body="\n    ".join(upd_stmts),
            )
            t_names = [t_ai_name, t_ad_name, t_au_name]
            return [trig_ins, trig_del, trig_upd], t_names, []

        elif self.dialect == Dialect.MYSQL:
            ins_block = "\n        ".join(ins_stmts)
            del_block = "\n        ".join(del_stmts)
            minus_block = "\n        ".join(upd_old_stmts)
            plus_block = "\n        ".join(upd_new_stmts)
            cleanup_block = "\n        ".join(cleanup_stmts)

            ins_section = (
                f"IF ({where_new}) THEN\n        {ins_block}\n    END IF;"
                if where_new and ins_block
                else ins_block
            )
            del_section = (
                f"IF ({where_old}) THEN\n        {del_block}\n    END IF;"
                if where_old and del_block
                else del_block
            )
            old_section = (
                f"IF ({where_old}) THEN\n        {minus_block}\n    END IF;"
                if where_old and minus_block
                else minus_block
            )
            new_section = (
                f"IF ({where_new}) THEN\n        {plus_block}\n    END IF;"
                if where_new and plus_block
                else plus_block
            )
            clean_section = (
                f"\n    IF ({where_old}) THEN\n        {cleanup_block}\n    END IF;"
                if where_old and cleanup_block
                else (f"\n    {cleanup_block}" if cleanup_block else "")
            )

            # MariaDB fires AFTER UPDATE for a row whose UPDATE IGNORE was
            # skipped on a duplicate key (the row keeps its OLD values; MySQL
            # does not fire). Apply the change only when the row is no longer
            # in its OLD state: compared with OLD, which is read from storage,
            # so the comparison is exact for every type.
            # (Without the changed-column guard, the update maintenance
            # recomputes from the table and is idempotent; no guard is added.)
            # Only a table with a primary or unique key can skip a row on a
            # duplicate key, and only such a key identifies the row: without
            # one, an identical twin row would look "still OLD" and a real
            # update would be skipped.
            row_key = self._mariadb_row_key(t_schema) if t_schema is not None else None
            if self.engine_version.is_mariadb and row_key is not None and ref_cols and changed_guard:
                key_cols, nullable_key = row_key
                cmp_cols = list(dict.fromkeys([*key_cols, *ref_cols]))
                # Key lookup by equality; then identity, as in the changed
                # guard (a case-only change under a case-insensitive
                # collation is a change).
                still_old = " AND ".join(
                    [f"__mv_cur.{self._q(c)} <=> OLD.{self._q(c)}" for c in key_cols]
                    + [self._identical_row_values_sql(f"__mv_cur.{self._q(c)}", f"OLD.{self._q(c)}", True)
                       for c in cmp_cols]
                )
                # A locking read: a consistent (snapshot) read here would open
                # the statement's read view before the maintenance statements
                # lock the backing rows, and MariaDB's snapshot isolation
                # (innodb_snapshot_isolation) then fails them with 1020 when a
                # concurrent writer committed in between (CI, MariaDB 11.8/12.3:
                # test_avg_concurrent_maintenance). The row is the one this
                # statement just wrote and already holds locked; a trigger may
                # not take FOR UPDATE on its own table (1442), LOCK IN SHARE
                # MODE reads the latest version without a read view.
                applied = f"NOT EXISTS (SELECT 1 FROM {q_base_t} AS __mv_cur WHERE {still_old} LOCK IN SHARE MODE)"
                if nullable_key:
                    # A NULL in a UNIQUE key is never a duplicate, so rows
                    # with a NULL key part can be twins: apply those.
                    applied = "(" + " OR ".join(f"OLD.{self._q(c)} IS NULL" for c in key_cols) + f" OR {applied})"
            else:
                applied = None
            upd_guard_open = f"IF ({changed_guard}) THEN\n    " if changed_guard else ""
            upd_guard_close = "\n    END IF;" if changed_guard else ""
            # The ignored-row check nests inside the changed-row guard (it runs
            # only for rows whose referenced values changed).
            after_guard_open = upd_guard_open + (f"IF {applied} THEN\n    " if applied else "")
            after_guard_close = ("\n    END IF;" if applied else "") + upd_guard_close
            upd_inner = (
                f"{old_section}\n    {new_section}{clean_section}"
                if old_section.strip()
                else f"{new_section}{clean_section}"
            )
            if mysql_same_group_cond and mysql_same_group_stmts:
                same_cond_parts: List[str] = []
                if where_old:
                    same_cond_parts.append(f"({where_old})")
                if where_new:
                    same_cond_parts.append(f"({where_new})")
                same_cond_parts.append(f"({mysql_same_group_cond})")
                full_same_cond = " AND ".join(same_cond_parts)
                same_block = "\n        ".join(mysql_same_group_stmts)
                upd_inner = (
                    f"IF ({full_same_cond}) THEN\n"
                    f"        {same_block}\n"
                    f"    ELSE\n"
                    f"        {upd_inner}\n"
                    f"    END IF;"
                )

            trig_ins = render_procedural_sql(
                "mysql_row_trigger",
                trg_name=t_ai_name,
                event="INSERT",
                q_base_t=q_base_t,
                body=ins_section,
            )
            trig_del = render_procedural_sql(
                "mysql_row_trigger",
                trg_name=t_ad_name,
                event="DELETE",
                q_base_t=q_base_t,
                body=del_section,
            )
            trig_upd = render_procedural_sql(
                "mysql_row_trigger",
                trg_name=t_au_name,
                event="UPDATE",
                q_base_t=q_base_t,
                body=f"{after_guard_open}{upd_inner}{after_guard_close}",
            )
            t_names = [t_ai_name, t_ad_name, t_au_name]
            if not mysql_prelock:
                return [trig_ins, trig_del, trig_upd], t_names, []

            # Lock the affected existing group rows before the base row is
            # written (see _mysql_group_prelock).
            def _guarded(cond: Optional[str], stmt: str) -> str:
                return f"IF ({cond}) THEN\n        {stmt}\n    END IF;" if cond else stmt

            lock_new = mysql_prelock["lock_new"]
            lock_old = mysql_prelock["lock_old"]
            g_new = _guarded(where_new, lock_new)
            g_old = _guarded(where_old, lock_old)
            bu_body = (
                f"IF ({mysql_prelock['new_first']}) THEN\n    {g_new}\n    {g_old}\n"
                f"    ELSE\n    {g_old}\n    {g_new}\n    END IF;"
            )
            t_bi_name = self._bounded_ident(f"{p}trg_", stem, "_bi")
            t_bd_name = self._bounded_ident(f"{p}trg_", stem, "_bd")
            t_bu_name = self._bounded_ident(f"{p}trg_", stem, "_bu")
            trig_bi = render_procedural_sql(
                "mysql_row_trigger", trg_name=t_bi_name, timing="BEFORE", event="INSERT",
                q_base_t=q_base_t, body=g_new,
            )
            trig_bd = render_procedural_sql(
                "mysql_row_trigger", trg_name=t_bd_name, timing="BEFORE", event="DELETE",
                q_base_t=q_base_t, body=g_old,
            )
            trig_bu = render_procedural_sql(
                "mysql_row_trigger", trg_name=t_bu_name, timing="BEFORE", event="UPDATE",
                q_base_t=q_base_t, body=f"{upd_guard_open}{bu_body}{upd_guard_close}",
            )
            return (
                [trig_bi, trig_bd, trig_bu, trig_ins, trig_del, trig_upd],
                [t_bi_name, t_bd_name, t_bu_name, *t_names],
                [],
            )

        return [], [], []

