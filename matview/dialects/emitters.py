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
    ) -> Tuple[str, List[str], List[str], List[str]]:
        db_view = self.db_view_name
        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        proj_exprs = (
            [self._q(ex) if ex in user_columns else ex for ex in select_exprs_override]
            if select_exprs_override
            else [self._q(c) for c in user_columns]
        )

        if self.dialect == Dialect.MYSQL:
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
            )
            return reader_view_ddl, [], [], []

        reader_view_ddl = build_create_view_sql(
            db_view,
            mv_table,
            proj_exprs,
            where_sql=extra_where_filter,
            dialect=self.dialect,
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
                        db_view=self._q(db_view),
                        raw_db_view=db_view,
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
            db_view=self._q(db_view),
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
            pk_col = spec.pk_column
            tmpl = spec.chunk_sql_templates[0].strip().rstrip(";")
            ph = "?" if self.dialect == Dialect.SQLITE else "%s"
            is_global_agg = (
                any(it.is_aggregate for it in raw_plan.view_spec.select_items)
                and not raw_plan.view_spec.group_by_cols
                and not raw_plan.view_spec.has_top_level_distinct
            )
            q_base_t = self._q(base_t)
            q_pk_col = self._q(pk_col)

            if self.dialect == Dialect.SQLITE:
                # Replace `(pk_col > ? AND pk_col <= ?)` with recursive CTE batch join
                cond_pat = f"({q_pk_col} > {ph} AND {q_pk_col} <= {ph})"
                joined_tmpl = tmpl.replace(
                    cond_pat,
                    f"({q_base_t}.{q_pk_col} > __mv_b.lo AND {q_base_t}.{q_pk_col} <= __mv_b.hi)",
                )
                joined_tmpl = re.sub(
                    rf"\nFROM\s+{re.escape(q_base_t)}\b",
                    f"\nFROM __mv_batches AS __mv_b JOIN {q_base_t} ON ({q_base_t}.{q_pk_col} > __mv_b.lo AND {q_base_t}.{q_pk_col} <= __mv_b.hi)",
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
            p_depth = 0
            for ptk in pop_toks:
                if ptk.token_type == SQLGlotTokenType.L_PAREN:
                    p_depth += 1
                elif ptk.token_type == SQLGlotTokenType.R_PAREN and p_depth > 0:
                    p_depth -= 1
                elif p_depth == 0 and ptk.token_type in (SQLGlotTokenType.SELECT, getattr(SQLGlotTokenType, "WITH", SQLGlotTokenType.SELECT)):
                    sel_tok_pos = ptk.start
                    break
            select_body = clean_pop[sel_tok_pos:].strip() if sel_tok_pos is not None else sel_node.sql(sg_dial).strip()

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
            return f"(CAST({old} AS BINARY) <=> CAST({new} AS BINARY))"
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
    ) -> Tuple[List[str], List[str], List[str]]:
        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        stem = f"{log_vname}__{base_t}"
        t_ai_name = self._bounded_ident(f"{p}trg_", stem, "_ai")
        t_ad_name = self._bounded_ident(f"{p}trg_", stem, "_ad")
        t_au_name = self._bounded_ident(f"{p}trg_", stem, "_au")
        t_trunc_name = self._bounded_ident(f"{p}trg_", stem, "_trunc")
        t_sync_name = self._bounded_ident(f"{p}trg_", stem, "_sync")
        q_base_t = self._q(base_t)
        ref_cols = self._referenced_columns_for_table(base_t)
        # Projection backing rows are keyed by the source PK even when the
        # public SELECT omits it. A PK-only UPDATE must therefore fire the
        # maintenance trigger and must not be suppressed by the dialect's
        # unchanged-column guard (SQLite and MySQL/MariaDB).
        seen_ref_cols = {c.lower() for c in ref_cols}
        for key_col in update_key_columns or ():
            if key_col.lower() not in seen_ref_cols:
                ref_cols.append(key_col)
                seen_ref_cols.add(key_col.lower())
        upd_of_clause = (
            f" OF {', '.join(self._q(c) for c in ref_cols)}"
            if ref_cols and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES)
            else ""
        )

        t_schema = self._schemas_lower.get(base_t.lower())
        unchanged_conds: List[str] = []
        if t_schema and ref_cols:
            for c_name in ref_cols:
                c_def = t_schema.columns.get(c_name)
                is_null = not (c_def.not_null if c_def else False)
                qc = self._q(c_name)
                unchanged_conds.append(self._identical_row_values_sql(f"OLD.{qc}", f"NEW.{qc}", is_null))
        changed_guard = f"NOT ({' AND '.join(unchanged_conds)})" if unchanged_conds else None

        cleanup_stmts = upd_cleanup_old_stmts or []

        if self.dialect == Dialect.SQLITE:
            trig_ins = render_procedural_sql(
                "trigger_sqlite_after_insert",
                trg_name=t_ai_name,
                q_base_t=q_base_t,
                when_clause=where_new,
                body="\n    ".join(ins_stmts),
            )
            trig_del = render_procedural_sql(
                "trigger_sqlite_after_delete",
                trg_name=t_ad_name,
                q_base_t=q_base_t,
                when_clause=where_old,
                body="\n    ".join(del_stmts),
            )
            upd_stmts = (
                [self._apply_where_guard_to_stmt(s, where_old) for s in upd_old_stmts]
                + [self._apply_where_guard_to_stmt(s, where_new) for s in upd_new_stmts]
                + [self._apply_where_guard_to_stmt(s, where_old) for s in cleanup_stmts]
            )
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

        elif self.dialect == Dialect.POSTGRES:
            ins_block = "\n            ".join(ins_stmts)
            minus_block = "\n            ".join(upd_old_stmts)
            plus_block = "\n            ".join(upd_new_stmts)
            cleanup_block = "\n            ".join(cleanup_stmts)
            trunc_block = "\n        ".join(trunc_stmts or [f"DELETE FROM {self._backing_table_name()};"])

            ins_section = (
                f"IF ({where_new}) THEN\n            {ins_block}\n        END IF;"
                if where_new and ins_block
                else ins_block
            )
            old_section = (
                f"IF ({where_old}) THEN\n            {minus_block}\n        END IF;"
                if where_old and minus_block
                else minus_block
            )
            new_section = (
                f"IF ({where_new}) THEN\n            {plus_block}\n        END IF;"
                if where_new and plus_block
                else plus_block
            )
            clean_section = (
                f"\n        IF ({where_old}) THEN\n            {cleanup_block}\n        END IF;"
                if where_old and cleanup_block
                else (f"\n        {cleanup_block}" if cleanup_block else "")
            )
            del_full_block = "\n            ".join(del_stmts)
            del_section = (
                f"IF ({where_old}) THEN\n            {del_full_block}\n        END IF;"
                if where_old
                else del_full_block
            )

            fn_trunc = self._bounded_ident(f"{p}fn_sync_", stem, "_trunc")
            fn_trunc_ddl = render_procedural_sql(
                "pg_truncate_function",
                fn_name=fn_trunc,
                trunc_block=trunc_block,
            )
            t_trunc = render_procedural_sql(
                "pg_truncate_trigger",
                trg_name=t_trunc_name,
                q_base_t=q_base_t,
                fn_name=fn_trunc,
            )

            if self.pg_use_transition_tables:
                fn_ai = self._bounded_ident(f"{p}fn_sync_", stem, "_ai")
                fn_ad = self._bounded_ident(f"{p}fn_sync_", stem, "_ad")
                fn_au = self._bounded_ident(f"{p}fn_sync_", stem, "_au")

                use_vec = bool(self.pg_vectorize_transition_tables)

                if use_vec and pg_ai_body is not None:
                    fn_ai_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_ai,
                        body_sql=pg_ai_body.strip(),
                    )
                elif use_vec and ("NEW." not in ins_section and ":NEW" not in ins_section):
                    fn_ai_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_ai,
                        body_sql=ins_section.strip(),
                    )
                else:
                    fn_ai_ddl = render_procedural_sql(
                        "pg_transition_loop_function",
                        fn_name=fn_ai,
                        row_var="NEW",
                        transition_table="__mv_new_rows",
                        section_sql=ins_section,
                    )

                if use_vec and pg_ad_body is not None:
                    fn_ad_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_ad,
                        body_sql=pg_ad_body.strip(),
                    )
                elif use_vec and ("OLD." not in del_section and ":OLD" not in del_section):
                    fn_ad_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_ad,
                        body_sql=del_section.strip(),
                    )
                else:
                    fn_ad_ddl = render_procedural_sql(
                        "pg_transition_loop_function",
                        fn_name=fn_ad,
                        row_var="OLD",
                        transition_table="__mv_old_rows",
                        section_sql=del_section,
                    )

                if use_vec and pg_au_body is not None:
                    fn_au_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_au,
                        body_sql=pg_au_body.strip(),
                    )
                elif use_vec and (
                    "OLD." not in old_section
                    and ":OLD" not in old_section
                    and "NEW." not in new_section
                    and ":NEW" not in new_section
                    and "OLD." not in clean_section
                    and ":OLD" not in clean_section
                ):
                    au_parts = [s.strip() for s in (old_section, new_section, clean_section) if s.strip()]
                    fn_au_ddl = render_procedural_sql(
                        "pg_transition_simple_function",
                        fn_name=fn_au,
                        body_sql="\n    ".join(au_parts),
                    )
                else:
                    au_loops: List[Dict[str, str]] = []
                    if old_section.strip():
                        au_loops.append({
                            "row_var": "OLD",
                            "transition_table": "__mv_old_rows",
                            "section_sql": old_section,
                        })
                    if new_section.strip():
                        au_loops.append({
                            "row_var": "NEW",
                            "transition_table": "__mv_new_rows",
                            "section_sql": new_section,
                        })
                    if clean_section.strip():
                        au_loops.append({
                            "row_var": "OLD",
                            "transition_table": "__mv_old_rows",
                            "section_sql": clean_section.strip(),
                        })
                    fn_au_ddl = render_procedural_sql(
                        "pg_transition_au_loop_function",
                        fn_name=fn_au,
                        loops=au_loops,
                    )

                t_ai = render_procedural_sql(
                    "pg_transition_trigger",
                    trg_name=t_ai_name,
                    event="INSERT",
                    q_base_t=q_base_t,
                    referencing="NEW TABLE AS __mv_new_rows",
                    fn_name=fn_ai,
                )
                t_ad = render_procedural_sql(
                    "pg_transition_trigger",
                    trg_name=t_ad_name,
                    event="DELETE",
                    q_base_t=q_base_t,
                    referencing="OLD TABLE AS __mv_old_rows",
                    fn_name=fn_ad,
                )
                t_au = render_procedural_sql(
                    "pg_transition_trigger",
                    trg_name=t_au_name,
                    event="UPDATE",
                    q_base_t=q_base_t,
                    referencing="OLD TABLE AS __mv_old_rows NEW TABLE AS __mv_new_rows",
                    fn_name=fn_au,
                )
                return (
                    [fn_ai_ddl, t_ai, fn_ad_ddl, t_ad, fn_au_ddl, t_au, fn_trunc_ddl, t_trunc],
                    [t_ai_name, t_ad_name, t_au_name, t_trunc_name],
                    [fn_ai, fn_ad, fn_au, fn_trunc],
                )

            fn_name = self._bounded_ident(f"{p}fn_sync_", stem)
            upd_guard_open = f"IF ({changed_guard}) THEN\n        " if changed_guard else ""
            upd_guard_close = "\n        END IF;" if changed_guard else ""
            upd_inner = (
                f"{old_section}\n        {new_section}{clean_section}"
                if old_section.strip()
                else f"{new_section}{clean_section}"
            )
            fn_ddl = render_procedural_sql(
                "pg_row_sync_function",
                fn_name=fn_name,
                del_section=del_section,
                ins_section=ins_section,
                upd_body=f"{upd_guard_open}{upd_inner}{upd_guard_close}",
            )
            t_sync = render_procedural_sql(
                "pg_row_sync_trigger",
                trg_name=t_sync_name,
                upd_of_clause=upd_of_clause,
                q_base_t=q_base_t,
                fn_name=fn_name,
            )
            return (
                [fn_ddl, t_sync, fn_trunc_ddl, t_trunc],
                [t_sync_name, t_trunc_name],
                [fn_name, fn_trunc],
            )

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

            upd_guard_open = f"IF ({changed_guard}) THEN\n    " if changed_guard else ""
            upd_guard_close = "\n    END IF;" if changed_guard else ""
            upd_inner = (
                f"{old_section}\n    {new_section}{clean_section}"
                if old_section.strip()
                else f"{new_section}{clean_section}"
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
                body=f"{upd_guard_open}{upd_inner}{upd_guard_close}",
            )
            t_names = [t_ai_name, t_ad_name, t_au_name]
            return [trig_ins, trig_del, trig_upd], t_names, []

        return [], [], []

