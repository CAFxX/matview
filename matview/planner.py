"""Compilation planning, strategy selection, limitation analysis, index validation, and diagnostics."""

from __future__ import annotations

from dataclasses import replace
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from ._host import MixinHost
from .engines import (
    AggFunc,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    EngineSubdialect,
    EngineVersion,
    LimitationSeverity,
    MaintenanceStrategy,
    StrictCompilationError,
    TransparencyLimitation,
    UnsupportedMaterializedViewError,
    _sqlglot_dialect,
    PER_CODE_OPT_IN_LIMITATIONS,
)
from .logical_ir import (
    ColumnDef,
    CompilationContext,
    JoinSpec,
    LogicalIR,
    MaterializedViewPlan,
    PlanDiagnostics,
    SelectItem,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
)
from .sql_frontend import (
    _SQL_EXPR_FUNCTIONS,
    _contains_aggregate_call,
    _extract_cte_names_and_strip_with,
    _extract_set_op_branches_from_ast,
    _extract_verbatim_cte_or_subquery_sql,
    _find_top_level_token,
    _has_subquery_ast,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _parse_index_col_ast,
    _parse_select_item,
    _parse_single_agg_expr,
    _parse_single_create_view_stmt,
    _sanitize_ident,
    _split_top_level,
    _strip_sql_comments,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_index_sql,
    parse_sql_ast,
    parse_view_ast,
)


class PlannerMixin(MixinHost):
    """Planner methods for view unwrapping, decomposition, validation, strategy selection, and indexing."""

    def _enforce_or_warn(self) -> None:
        self._enforce_limitations()

    def _detect_limitations(self) -> List[TransparencyLimitation]:
        lims: List[TransparencyLimitation] = []
        ref_tables = list(self.view.referenced_tables)
        m_nd = re.search(
            r"\b(?:RANDOM|RAND|UUID|GEN_RANDOM_UUID|NOW|CLOCK_TIMESTAMP|SYSDATE)\s*\(|\bCURRENT_(?:TIMESTAMP|DATE|TIME)\b",
            self.view.raw_select_sql,
            flags=re.IGNORECASE,
        )
        if m_nd:
            lims.append(
                TransparencyLimitation(
                    code="NONDETERMINISTIC_EXPR",
                    severity=LimitationSeverity.BLOCKER,
                    dialect=self.dialect,
                    affected_object=self.logical_view_name,
                    description=(
                        f"View query contains non-deterministic expression '{m_nd.group(0)}', which cannot "
                        f"be maintained deterministically by incremental triggers."
                    ),
                    mitigation="Pass timestamps or random values as explicit columns on the base table.",
                )
            )
        if self.dialect == Dialect.MYSQL and ref_tables:
            lims.append(
                TransparencyLimitation(
                    code="MYSQL_NO_TRUNCATE_TRIGGER",
                    severity=LimitationSeverity.WARNING,
                    dialect=Dialect.MYSQL,
                    affected_object=", ".join(ref_tables),
                    description=(
                        f"MySQL treats TRUNCATE TABLE on base table(s) ({', '.join(ref_tables)}) as a DDL "
                        f"operation that bypasses AFTER DELETE triggers, and MySQL does not support AFTER TRUNCATE triggers."
                    ),
                    mitigation="Use DELETE FROM <table>; instead of TRUNCATE TABLE, or run within manager.bulk_changes().",
                )
            )
            for t_name in ref_tables:
                ts = self._schemas_lower.get(t_name.lower())
                if ts and ts.fk_cascade_actions:
                    lims.append(
                        TransparencyLimitation(
                            code="MYSQL_FK_CASCADE_NO_TRIGGER",
                            severity=LimitationSeverity.BLOCKER,
                            dialect=Dialect.MYSQL,
                            affected_object=ts.name,
                            description=(
                                f"Base table '{ts.name}' defines foreign-key cascading action(s) "
                                f"({', '.join(ts.fk_cascade_actions)}), which do not activate AFTER DELETE "
                                f"or AFTER UPDATE triggers in MySQL InnoDB."
                            ),
                            mitigation="Perform cascading deletes/updates explicitly in application SQL or use PostgreSQL/SQLite.",
                        )
                    )
        if self.dialect == Dialect.POSTGRES and not self.engine_version.supports_pg_statement_maintenance:
            lims.append(
                TransparencyLimitation(
                    code="POSTGRES_VERSION_NO_STATEMENT_TRIGGERS",
                    severity=LimitationSeverity.BLOCKER,
                    dialect=Dialect.POSTGRES,
                    affected_object=self.logical_view_name,
                    description=(
                        "PostgreSQL maintenance uses statement-level triggers with transition tables "
                        f"created with EXECUTE FUNCTION, which require PostgreSQL >= 11 (target engine is "
                        f"{self.engine_version.label})."
                    ),
                    mitigation="Upgrade PostgreSQL to 11 or later.",
                )
            )
        if self.view.has_window_func:
            if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_window_functions:
                lims.append(
                    TransparencyLimitation(
                        code="SQLITE_VERSION_NO_WINDOW_FUNC",
                        severity=LimitationSeverity.BLOCKER,
                        dialect=Dialect.SQLITE,
                        affected_object=self.logical_view_name,
                        description=(
                            f"View '{self.logical_view_name}' uses window functions, which require "
                            f"SQLite >= 3.25.0 (target engine is {self.engine_version.label})."
                        ),
                        mitigation="Upgrade SQLite to >= 3.25.0 or rewrite the query without OVER (...).",
                    )
                )
            elif self.dialect == Dialect.MYSQL and not self.engine_version.supports_mysql_window_functions:
                lims.append(
                    TransparencyLimitation(
                        code="MYSQL_VERSION_NO_WINDOW_FUNC",
                        severity=LimitationSeverity.BLOCKER,
                        dialect=Dialect.MYSQL,
                        affected_object=self.logical_view_name,
                        description=(
                            f"View '{self.logical_view_name}' uses window functions, which require "
                            f"MySQL >= 8.0.0 or MariaDB >= 10.2.0 (target engine is {self.engine_version.label})."
                        ),
                        mitigation="Upgrade MySQL to >= 8.0.0 / MariaDB to >= 10.2.0 or rewrite the query.",
                    )
                )
        return lims

    def _order_sensitive_aggregate_limitations(
        self, strategy: MaintenanceStrategy, alg_mode: str
    ) -> List[TransparencyLimitation]:
        return (
            self._order_sensitive_shape_limitations(strategy, alg_mode)
            + self._mysql_group_index_limitations(strategy, alg_mode)
        )

    def _window_order_sensitive_exprs(self) -> List[str]:
        """Window SUM/AVG/VAR/STDDEV over arguments that are not provably
        exact (their running value also accumulates in plan order)."""
        if not self.view.has_window_func:
            return []
        try:
            root = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return ["<window aggregate>"]
        names = {"SUM", "AVG", "VARIANCE", "VARIANCE_POP", "VAR_POP", "VAR_SAMP", "STDDEV",
                 "STDDEV_POP", "STDDEV_SAMP", "STD"}
        out: List[str] = []
        sg = _sqlglot_dialect(self.dialect)
        for w in root.find_all(sqlglot_exp.Window):
            fn = w.this
            if not isinstance(fn, sqlglot_exp.Func):
                continue
            fname = fn.sql_name().upper()
            if fname not in names and str(getattr(fn, "name", "")).upper() not in names:
                continue
            arg = fn.this
            if isinstance(arg, sqlglot_exp.Distinct):
                arg = arg.expressions[0] if arg.expressions else None
            arg_sql = arg.sql(dialect=sg) if arg is not None else ""
            if self._exact_numeric_precision_scale(arg_sql) is None or fname not in ("SUM", "AVG"):
                out.append(w.sql(dialect=sg))
        return out

    def _order_sensitive_shape_limitations(
        self, strategy: MaintenanceStrategy, alg_mode: str
    ) -> List[TransparencyLimitation]:
        """PostgreSQL/MySQL/MariaDB refuse floating-point order-sensitive
        aggregates by default, whatever the shape.

        SUM/AVG over approximate (or not provably exact) arguments accumulate
        in binary floating point in the order the engine reads the rows, and
        VAR/STDDEV use the engine's streaming algorithm (MySQL/MariaDB in
        DOUBLE for every argument type). That order is chosen by the plan of
        the original view and is not a function of the data: an index or range
        scan for a WHERE clause, sort-based grouping, parallel aggregation and
        the join order all change it (measured natively: PostgreSQL ungrouped
        `SUM(x) WHERE y < 10` with an index on y and MySQL 8.4 / MariaDB 10.11
        grouped views with an index on the WHERE column: original view 1.5,
        any fixed recompute order 2.0; PostgreSQL parallel aggregation makes
        the original view itself nondeterministic). No maintained value can
        equal the original view under every plan, so these views are only
        installed with an explicit per-code opt-in (``accept_limitations``),
        which gives best-effort maintenance: race-free, and equal to the
        original view when it evaluates in the recompute's order.
        """
        if self.dialect not in (Dialect.POSTGRES, Dialect.MYSQL):
            return []
        items = [
            it.raw_expr for it in self._working_select_items
            if it.is_aggregate and self._is_order_sensitive_aggregate(it)
        ]
        items += self._window_order_sensitive_exprs()
        if not items:
            return []
        order = (
            "sequential-scan (ctid) order" if self.dialect == Dialect.POSTGRES
            else "primary-key order"
        )
        listed = ", ".join(dict.fromkeys(items))
        lims: List[TransparencyLimitation] = []
        recomputed = strategy == MaintenanceStrategy.ALGEBRAIC_DELTA and (
            alg_mode == "single" or (alg_mode == "union_all_agg" and self.dialect == Dialect.POSTGRES)
        )
        if not recomputed:
            # Only single-table algebraic views (and PostgreSQL UNION ALL
            # aggregates) recompute the order-sensitive aggregate race-free
            # after locking; join folds, MySQL/MariaDB UNION ALL folds and
            # group-scoped / atomic re-evaluation (joins, window functions)
            # are neither drift-free nor verified race-free.
            lims.append(
                TransparencyLimitation(
                    code="ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE",
                    severity=LimitationSeverity.BLOCKER,
                    dialect=self.dialect,
                    affected_object=self.logical_view_name,
                    description=(
                        f"Aggregate(s) {listed} in a {strategy.value}/{alg_mode} view are maintained by "
                        f"folding deltas or by re-evaluating the group without the lock-then-recompute "
                        f"protocol of single-table views: rounding can accumulate across writes and "
                        f"concurrent writers are not verified to be serialized. With accept_limitations=['ORDER_SENSITIVE_AGGREGATE_"
                        f"NO_RACE_FREE_RECOMPUTE'] values can drift from the original view until "
                        f"repair_view()."
                    ),
                    mitigation=(
                        "Aggregate an exact type (integer/DECIMAL/NUMERIC), materialize the aggregate over "
                        "a single table, or accept the code explicitly."
                    ),
                )
            )
        return lims + [
            TransparencyLimitation(
                code="ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE",
                severity=LimitationSeverity.BLOCKER,
                dialect=self.dialect,
                affected_object=self.logical_view_name,
                description=(
                    f"Aggregate(s) {listed} accumulate in floating point in the order the original "
                    f"view's plan reads the rows (index and range scans, sort-based grouping, "
                    f"parallel aggregation, join order), so no maintained value equals the original "
                    f"view under every plan. With accept_limitations=['ORDER_SENSITIVE_AGGREGATE_"
                    f"NOT_REPRODUCIBLE'] the view is maintained on a BEST-EFFORT basis: race-free "
                    f"native recompute in {order}, equal to the original view only when it "
                    f"evaluates in that order."
                ),
                mitigation=(
                    "Aggregate an exact type (integer/DECIMAL/NUMERIC), or accept the code explicitly "
                    "(accept_limitations / --accept-limitation)."
                ),
            )
        ]

    def _mysql_group_index_limitations(
        self, strategy: MaintenanceStrategy, alg_mode: str
    ) -> List[TransparencyLimitation]:
        """MySQL/MariaDB: the race-free recompute of order-sensitive aggregates
        is a locking read of the group's base rows. Without an index whose
        leading columns are the group-key columns InnoDB scans and share-locks
        the whole table under REPEATABLE READ, and concurrent writers of
        different groups deadlock constantly (measured, MySQL 9.7 / MariaDB
        10.11, 4 writers on 6 groups: 135 / 28 of 600 single-statement
        transactions rolled back with 1213, vs 1 / 0 with the index). Refuse
        such views until the index exists."""
        if self.dialect != Dialect.MYSQL or strategy != MaintenanceStrategy.ALGEBRAIC_DELTA or alg_mode != "single":
            return []
        grp_exprs = list(self._effective_group_by_exprs)
        if not grp_exprs:
            return []
        try:
            items = self._pgmy_native_recompute_items()
        except Exception:
            items = []
        if not items:
            return []
        schema = self.primary_schema
        group_cols: List[str] = []
        for g in grp_exprs:
            if not self._is_simple_col_ref(g):
                group_cols = []
                break
            group_cols.append(self._col_def(g).name.lower())
        covered = False
        if schema is not None and group_cols:
            want = set(group_cols)
            candidates = [list(schema.effective_pk_columns())] + [list(idx.columns) for idx in schema.indexes if not idx.where_clause]
            for cols in candidates:
                lead = [str(c).strip().strip('`"').lower() for c in cols[: len(want)]]
                if len(lead) == len(want) and set(lead) == want:
                    covered = True
                    break
        if covered:
            return []
        listed = ", ".join(it.raw_expr for it in items)
        return [
            TransparencyLimitation(
                code="MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX",
                severity=LimitationSeverity.BLOCKER,
                dialect=self.dialect,
                affected_object=self.logical_view_name,
                description=(
                    f"Aggregate(s) {listed} are recomputed per group with a locking read; without an index "
                    f"whose leading columns are the GROUP BY columns ({', '.join(grp_exprs)}) InnoDB "
                    f"share-locks the whole table and concurrent writers deadlock."
                ),
                mitigation=(
                    "Create an index on the base table's GROUP BY columns (see "
                    "plan.recommended_base_indexes), group by plain columns, or aggregate an exact type."
                ),
            )
        ]

    def _enforce_limitations(self, limitations: Optional[Sequence[TransparencyLimitation]] = None) -> None:
        """Refuse unless every limitation is overridden.

        Codes in PER_CODE_OPT_IN_LIMITATIONS are only overridden by naming
        them in ``accept_limitations``; the global ``ignore_limitations``
        does not cover them (on MySQL/MariaDB every view already needs it for
        MYSQL_NO_TRUNCATE_TRIGGER, so it cannot express consent to a weaker
        result contract). Other codes are overridden by either."""
        lims = list(self.limitations if limitations is None else limitations)
        if not lims:
            return
        accepted = set(self.accepted_limitations)
        unaccepted = [
            l for l in lims
            if l.code not in accepted
            and (l.code in PER_CODE_OPT_IN_LIMITATIONS or not self.ignore_limitations)
        ]
        if unaccepted:
            raise EngineLimitationError(self.logical_view_name, self.dialect, unaccepted)
        summary = "; ".join(f"[{l.code}] {l.description}" for l in lims)
        how = []
        if any(l.code in accepted for l in lims):
            how.append("accept_limitations")
        if any(l.code not in accepted for l in lims):
            how.append("--ignore-limitations")
        warnings.warn(
            f"Proceeding with materialized view '{self.logical_view_name}' on '{self.dialect.value}' "
            f"with {' and '.join(how)}: {summary}",
            EngineLimitationWarning,
            stacklevel=2,
        )

    def _try_expand_select_star(self, v: ViewSpec) -> ViewSpec:
        """
        Expands `SELECT *` or `SELECT alias.*` on single-table views into explicit schema columns
        so `SELECT * FROM t [WHERE ...]` and `SELECT DISTINCT * FROM t [WHERE ...]` compile with
        Tier 1 O(1) ALGEBRAIC_DELTA maintenance instead of falling back to Tier 3.
        """
        if (
            self.primary_schema is None
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
            or v.has_cte
            or v.has_compound_set_op
            or v.has_subquery_in_from
            or v.has_window_func
            or v.has_limit_or_order
        ):
            return v
        base_alias = v.joins[0].alias.lower() if v.joins else v.base_table.lower()
        base_name = v.base_table.lower()
        has_star = any(
            it.raw_expr.strip() == "*"
            or it.raw_expr.strip().lower() in (f"{base_alias}.*", f"{base_name}.*")
            for it in v.select_items
        )
        if not has_star:
            return v
        expanded: List[SelectItem] = []
        for it in v.select_items:
            raw_s = it.raw_expr.strip()
            if raw_s == "*" or raw_s.lower() in (f"{base_alias}.*", f"{base_name}.*"):
                for c_def in self.primary_schema.columns.values():
                    expanded.append(
                        SelectItem(
                            raw_expr=c_def.name,
                            alias=c_def.name,
                            is_aggregate=False,
                            source_col=c_def.name,
                        )
                    )
            else:
                expanded.append(it)
        sel_str = ", ".join(f"{it.raw_expr} AS {it.alias}" for it in expanded)
        dist_kw = "DISTINCT " if v.has_top_level_distinct else ""
        from_j = v.joins[0]
        from_str = (
            f"{from_j.table_name} AS {from_j.alias}"
            if from_j.alias and from_j.alias.lower() != from_j.table_name.lower()
            else from_j.table_name
        )
        rebuilt_sql = f"SELECT {dist_kw}{sel_str} FROM {from_str}"
        if v.where_clause:
            rebuilt_sql += f" WHERE {v.where_clause}"
        if v.group_by_cols:
            rebuilt_sql += f" GROUP BY {', '.join(v.group_by_cols)}"
        if v.having_clause:
            rebuilt_sql += f" HAVING {v.having_clause}"
        return ViewSpec(
            view_name=v.view_name,
            raw_select_sql=rebuilt_sql,
            base_table=v.base_table,
            joins=v.joins,
            referenced_tables=v.referenced_tables,
            select_items=tuple(expanded),
            where_clause=v.where_clause,
            group_by_cols=v.group_by_cols,
            having_clause=v.having_clause,
            has_cte=v.has_cte,
            has_window_func=v.has_window_func,
            has_compound_set_op=v.has_compound_set_op,
            has_top_level_distinct=v.has_top_level_distinct,
            has_limit_or_order=v.has_limit_or_order,
            has_subquery_in_from=v.has_subquery_in_from,
            user_indexes=v.user_indexes,
            additional_views=v.additional_views,
        )

    def _try_unwrap_simple_cte_or_derived_view(self, v: ViewSpec) -> ViewSpec:
        """
        Unwraps a non-recursive single-CTE or derived subquery over a single base table
        when the inner query is a flat projection/filter (no inner GROUP BY, aggregates,
        DISTINCT, window functions, set ops, or LIMIT) so the outer query can use Tier 1 O(1)
        algebraic maintenance.
        """
        if v.has_window_func or v.has_compound_set_op or v.has_limit_or_order:
            return v

        inner_sql: Optional[str] = None
        cte_or_sub_alias: Optional[str] = None
        outer_sql: Optional[str] = None

        try:
            q_ast = parse_sql_ast(v.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(q_ast, sqlglot_exp.Select):
            return v

        sg_d = _sqlglot_dialect(self.dialect)
        if v.has_cte:
            with_node = q_ast.args.get("with_") or q_ast.args.get("with")
            if not isinstance(with_node, sqlglot_exp.With) or bool(with_node.args.get("recursive")) or len(with_node.expressions) != 1:
                return v
            cte_node = with_node.expressions[0]
            if not isinstance(cte_node, sqlglot_exp.CTE) or not isinstance(cte_node.this, sqlglot_exp.Expr):
                return v
            cte_alias_node = cte_node.args.get("alias")
            if isinstance(cte_alias_node, sqlglot_exp.TableAlias) and cte_alias_node.columns:
                return v
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, True, dialect=self.dialect) or cte_node.this.sql(sg_d).strip()
            cte_or_sub_alias = cte_node.alias
            _, outer_sql, _ = _extract_cte_names_and_strip_with(v.raw_select_sql, dialect=self.dialect)
        elif v.has_subquery_in_from and len(v.joins) == 1:
            from_node = q_ast.args.get("from_") or q_ast.args.get("from")
            if not isinstance(from_node, sqlglot_exp.From) or not isinstance(from_node.this, sqlglot_exp.Subquery):
                return v
            sub_node = from_node.this
            if not isinstance(sub_node.this, sqlglot_exp.Expr):
                return v
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, False, dialect=self.dialect) or sub_node.this.sql(sg_d).strip()
            cte_or_sub_alias = sub_node.alias or "__mv_sub"
            outer_sql = v.raw_select_sql
        else:
            return v

        if not inner_sql or not outer_sql:
            return v
        try:
            inner_ast_check = parse_sql_ast(inner_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(inner_ast_check, sqlglot_exp.Select) or any(
            inner_ast_check.args.get(k) for k in ("group", "having", "order", "limit", "offset")
        ):
            return v

        try:
            inner_spec = _parse_single_create_view_stmt(
                f"CREATE VIEW __inner_tmp AS {inner_sql}",
                known_tables=set(self._schemas_lower.keys()),
                dialect=self.dialect,
            )
        except Exception:
            return v

        if (
            inner_spec.has_cte
            or inner_spec.has_window_func
            or inner_spec.has_compound_set_op
            or inner_spec.has_top_level_distinct
            or inner_spec.has_limit_or_order
            or inner_spec.has_subquery_in_from
            or len(inner_spec.joins) != 1
            or inner_spec.group_by_cols
            or any(it.is_aggregate for it in inner_spec.select_items)
        ):
            return v

        if len(v.joins) != 1:
            return v

        inner_base = inner_spec.joins[0].table_name
        if inner_base.lower() not in self._schemas_lower:
            return v

        # Build column substitution map: alias -> raw_expr (with inner table alias stripped)
        inner_alias = inner_spec.joins[0].alias
        col_map: Dict[str, str] = {}
        for it in inner_spec.select_items:
            if it.raw_expr.strip() == "*":
                ts = self._schemas_lower[inner_base.lower()]
                for c_name in ts.columns:
                    col_map[c_name.lower()] = c_name
            else:
                cleaned_e = re.sub(
                    rf"\b{re.escape(inner_alias)}\.",
                    "",
                    it.raw_expr.strip(),
                    flags=re.IGNORECASE,
                )
                col_map[it.alias.lower()] = (
                    cleaned_e
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", cleaned_e)
                    else f"({cleaned_e})"
                )

        outer_alias = v.joins[0].alias

        def _subst(frag: Optional[str]) -> Optional[str]:
            if frag is None:
                return None
            toks = _tokenize_sql(frag, read=sg_d)
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
                    tbl_u = tk.text.lower()
                    col_u = toks[i + 2].text.lower()
                    if tbl_u in {outer_alias.lower(), (cte_or_sub_alias or "").lower()}:
                        if tk.start > cursor:
                            out.append(frag[cursor:tk.start])
                        rep = col_map[col_u] if col_u in col_map else toks[i + 2].text
                        out.append(rep)
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
                        col_u = tk.text.lower()
                        if col_u in col_map:
                            if tk.start > cursor:
                                out.append(frag[cursor:tk.start])
                            out.append(col_map[col_u])
                            cursor = tk.end + 1
                            i += 1
                            continue
                i += 1
            if cursor < len(frag):
                out.append(frag[cursor:])
            return "".join(out)

        new_items: List[SelectItem] = []
        for it in v.select_items:
            sub_raw = _subst(it.raw_expr) or it.raw_expr
            parsed = _parse_select_item(f"{sub_raw} AS {it.alias}", 0, dialect=self.dialect)
            new_items.append(parsed)

        inner_w = None
        if inner_spec.where_clause:
            inner_w = re.sub(
                rf"\b{re.escape(inner_alias)}\.",
                "",
                inner_spec.where_clause,
                flags=re.IGNORECASE,
            )
        outer_w = _subst(v.where_clause)
        if inner_w and outer_w:
            comb_w = f"({inner_w}) AND ({outer_w})"
        else:
            comb_w = inner_w or outer_w

        new_grps = tuple(_subst(g) or g for g in v.group_by_cols)
        new_hav = _subst(v.having_clause)

        sel_str = ", ".join(f"{it.raw_expr} AS {it.alias}" for it in new_items)
        Dist = "DISTINCT " if v.has_top_level_distinct else ""
        rebuilt_sql = f"SELECT {Dist}{sel_str} FROM {inner_base}"
        if comb_w:
            rebuilt_sql += f" WHERE {comb_w}"
        if new_grps:
            rebuilt_sql += f" GROUP BY {', '.join(new_grps)}"
        if new_hav:
            rebuilt_sql += f" HAVING {new_hav}"

        return ViewSpec(
            view_name=v.view_name,
            raw_select_sql=rebuilt_sql,
            base_table=inner_base,
            joins=(JoinSpec(join_type="FROM", table_name=inner_base, alias=inner_base),),
            referenced_tables=(inner_base,),
            select_items=tuple(new_items),
            where_clause=comb_w,
            group_by_cols=new_grps,
            having_clause=new_hav,
            has_cte=False,
            has_window_func=False,
            has_compound_set_op=False,
            has_top_level_distinct=v.has_top_level_distinct,
            has_limit_or_order=False,
            has_subquery_in_from=False,
            user_indexes=v.user_indexes,
        )

    def _decompose_compound_aggregates_and_having(
        self,
    ) -> Tuple[List[SelectItem], Dict[str, str], List[SelectItem], Optional[str], bool]:
        """
        Decomposes compound aggregate expressions in SELECT (e.g. `COALESCE(SUM(a), 0) - COALESCE(SUM(b), 0) AS net`)
        and aggregate calls in `HAVING` into elementary `SelectItem`s stored in the backing table,
        plus reader-view projection expressions and a backing-table `WHERE` filter for `HAVING`.
        """
        v = self.view
        work_items: List[SelectItem] = []
        reader_exprs: Dict[str, str] = {}
        extra_having_items: List[SelectItem] = []
        elem_counter = 0

        # Map canonical aggregate call string -> backing column alias
        sg_d = _sqlglot_dialect(self.dialect)
        agg_call_to_alias: Dict[str, str] = {}
        for it in v.select_items:
            if it.is_aggregate and it.agg_func is not None:
                agg_call_to_alias[" ".join(t.text.upper() for t in _tokenize_sql(it.raw_expr.strip(), read=sg_d))] = it.alias

        def _extract_agg_calls_from_expr(expr_str: str) -> Tuple[str, bool]:
            nonlocal elem_counter
            toks = _tokenize_sql(expr_str, read=sg_d)
            if not toks:
                return expr_str, True
            out_parts: List[str] = []
            idx = 0
            n = len(toks)
            cursor = 0
            agg_targets = {
                "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX",
                "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
            }
            while idx < n:
                tk = toks[idx]
                fn_up = tk.text.upper()
                if fn_up in agg_targets and idx + 1 < n and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN:
                    depth = 0
                    rp_idx = -1
                    for j in range(idx + 1, n):
                        if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                            depth += 1
                        elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                            depth -= 1
                            if depth == 0:
                                rp_idx = j
                                break
                    if rp_idx != -1:
                        end_tok_idx = rp_idx
                        if (
                            rp_idx + 4 < n
                            and toks[rp_idx + 1].token_type == SQLGlotTokenType.FILTER
                            and toks[rp_idx + 2].token_type == SQLGlotTokenType.L_PAREN
                            and toks[rp_idx + 3].token_type == SQLGlotTokenType.WHERE
                        ):
                            f_depth = 0
                            f_rp = -1
                            for k in range(rp_idx + 2, n):
                                if toks[k].token_type == SQLGlotTokenType.L_PAREN:
                                    f_depth += 1
                                elif toks[k].token_type == SQLGlotTokenType.R_PAREN:
                                    f_depth -= 1
                                    if f_depth == 0:
                                        f_rp = k
                                        break
                            if f_rp != -1:
                                end_tok_idx = f_rp
                        full_call = expr_str[tk.start : toks[end_tok_idx].end + 1]
                        norm_key = " ".join(t.text.upper() for t in _tokenize_sql(full_call, read=sg_d))
                        if norm_key in agg_call_to_alias:
                            col_al = agg_call_to_alias[norm_key]
                        else:
                            elem_counter += 1
                            col_al = f"__mv_elem_{elem_counter}"
                            parsed_el = _parse_single_agg_expr(full_call, col_al, dialect=self.dialect)
                            if parsed_el is None:
                                return expr_str, False
                            agg_call_to_alias[norm_key] = col_al
                            work_items.append(parsed_el)
                            extra_having_items.append(parsed_el)
                        if tk.start > cursor:
                            out_parts.append(expr_str[cursor:tk.start])
                        out_parts.append(col_al)
                        cursor = toks[end_tok_idx].end + 1
                        idx = end_tok_idx + 1
                        continue
                idx += 1
            if cursor < len(expr_str):
                out_parts.append(expr_str[cursor:])
            return "".join(out_parts), True

        # An aggregate inside a scalar subquery belongs to that subquery, not
        # to the view's groups; the decomposition below would replace it with
        # the outer group's stored value (wrong values, or "Subquery returns
        # more than 1 row"). Refuse such projections and HAVING clauses.
        def _aggregate_in_subquery(sql_text: str) -> bool:
            try:
                node = parse_sql_ast(sql_text, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return False
            for sub in node.find_all(sqlglot_exp.Subquery, sqlglot_exp.Select):
                if sub is node:
                    continue
                if any(True for _ in sub.find_all(sqlglot_exp.AggFunc)):
                    return True
            return False

        for text in [it.raw_expr for it in v.select_items if it.is_aggregate] + (
            [v.having_clause] if v.having_clause else []
        ):
            if _aggregate_in_subquery(text):
                raise UnsupportedMaterializedViewError(
                    f"View '{self.logical_view_name}' uses an aggregate inside a scalar subquery in an "
                    f"aggregated projection or HAVING ({text.strip()[:120]}), which is not maintained."
                )

        all_ok = True
        for it in v.select_items:
            if not it.is_aggregate or it.agg_func is not None:
                work_items.append(it)
                if (
                    self.subdialect == EngineSubdialect.MARIADB
                    and it.agg_func in (
                        AggFunc.VAR_POP,
                        AggFunc.VAR_SAMP,
                        AggFunc.STDDEV_POP,
                        AggFunc.STDDEV_SAMP,
                    )
                ):
                    result_scale = self._mariadb_statistical_result_scale(it)
                    if 0 <= result_scale < self.MARIADB_NOT_FIXED_DEC:
                        reader_exprs[it.alias] = f"ROUND({it.alias}, {result_scale}) AS {it.alias}"
                    else:
                        reader_exprs[it.alias] = it.alias
                elif self.dialect == Dialect.MYSQL and it.agg_func in (AggFunc.BIT_OR, AggFunc.BIT_XOR):
                    # Keep COALESCE's result type unsigned, matching MySQL's
                    # BIT_* aggregate contract (unsigned 64-bit integer).
                    reader_exprs[it.alias] = f"COALESCE({it.alias}, CAST(0 AS UNSIGNED)) AS {it.alias}"
                elif self.dialect == Dialect.MYSQL and it.agg_func == AggFunc.BIT_AND:
                    reader_exprs[it.alias] = f"COALESCE({it.alias}, CAST(18446744073709551615 AS UNSIGNED)) AS {it.alias}"
                else:
                    reader_exprs[it.alias] = it.alias
            else:
                rewritten_e, ok = _extract_agg_calls_from_expr(it.raw_expr)
                if not ok:
                    all_ok = False
                    work_items.append(it)
                    reader_exprs[it.alias] = it.alias
                else:
                    reader_exprs[it.alias] = f"({rewritten_e}) AS {it.alias}"

        rewritten_having: Optional[str] = None
        if v.having_clause:
            rh, ok_h = _extract_agg_calls_from_expr(v.having_clause)
            if ok_h:
                # Also strip table qualifiers on group columns in HAVING if they match group aliases
                for g_it in v.select_items:
                    if not g_it.is_aggregate and "." in g_it.raw_expr:
                        rh = re.sub(
                            rf"\b{re.escape(g_it.raw_expr.strip())}\b",
                            g_it.alias,
                            rh,
                            flags=re.IGNORECASE,
                        )
                rewritten_having = rh
            else:
                all_ok = False
                rewritten_having = v.having_clause

        return work_items, reader_exprs, extra_having_items, rewritten_having, all_ok

    def _resolve_effective_group_by_exprs(self) -> Tuple[str, ...]:
        if not self.view.group_by_cols:
            return ()
        resolved: List[str] = []
        alias_to_raw = {
            it.alias.lower(): it.raw_expr.strip()
            for it in self.view.select_items
            if not it.is_aggregate
        }
        for g in self.view.group_by_cols:
            gs = g.strip()
            if re.match(r"^\d+$", gs):
                pos = int(gs) - 1
                if 0 <= pos < len(self.view.select_items):
                    resolved.append(self.view.select_items[pos].raw_expr.strip())
                    continue
            if gs.lower() in alias_to_raw:
                # Check if gs is an alias for an expression rather than a base column
                raw_e = alias_to_raw[gs.lower()]
                if self._lookup_col_def(gs) is None and raw_e.lower() != gs.lower():
                    resolved.append(raw_e)
                    continue
            resolved.append(gs)
        return tuple(resolved)

    def _can_use_single_table_algebraic(self) -> bool:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            not has_any_agg
            or not self._decomposed_ok
            or v.has_cte
            or v.has_window_func
            or v.has_compound_set_op
            or v.has_top_level_distinct
            or v.has_limit_or_order
            or v.has_subquery_in_from
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
        ):
            return False
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return False

        allowed_aggs = {
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
            AggFunc.SUM,
            AggFunc.TOTAL,
            AggFunc.AVG,
            AggFunc.MIN,
            AggFunc.MAX,
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
            AggFunc.BIT_AND,
            AggFunc.BIT_OR,
            AggFunc.BIT_XOR,
            AggFunc.BOOL_AND,
            AggFunc.BOOL_OR,
        }
        for it in self._working_select_items:
            if _has_subquery_ast(it.raw_expr, dialect=self.dialect):
                return False
            if it.is_aggregate:
                if it.agg_func not in allowed_aggs:
                    return False
                if it.is_distinct:
                    if it.agg_func not in (AggFunc.COUNT_COL, AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG):
                        return False
                    if it.filter_where:
                        return False
        return True

    def _can_use_join_algebraic(self) -> bool:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            not has_any_agg
            or not self._decomposed_ok
            or v.has_cte
            or v.has_window_func
            or v.has_compound_set_op
            or v.has_top_level_distinct
            or v.has_limit_or_order
            or v.has_subquery_in_from
            or len(v.joins) < 2
        ):
            return False
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return False

        # Only INNER JOIN and CROSS JOIN are algebraically linear
        seen_tables: Set[str] = set()
        for j in v.joins:
            if j.join_type not in ("FROM", "INNER JOIN", "CROSS JOIN"):
                return False
            t_low = j.table_name.lower()
            if t_low not in self._schemas_lower or t_low in seen_tables:
                return False
            seen_tables.add(t_low)

        allowed_aggs = {
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
            AggFunc.SUM,
            AggFunc.TOTAL,
            AggFunc.AVG,
            AggFunc.MIN,
            AggFunc.MAX,
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
            AggFunc.BIT_XOR,
            AggFunc.BOOL_AND,
            AggFunc.BOOL_OR,
        }
        for it in self._working_select_items:
            if _has_subquery_ast(it.raw_expr, dialect=self.dialect):
                return False
            if it.is_aggregate:
                if it.is_distinct or it.agg_func not in allowed_aggs:
                    return False
        return True

    def _can_use_zset_setop(self) -> bool:
        v = self.view
        if (
            not v.has_compound_set_op
            or v.has_cte
            or v.has_window_func
            or v.has_limit_or_order
        ):
            return False
        branches = self._parse_setop_branches()
        return branches is not None and len(branches) >= 2

    def _can_use_zset_distinct(self) -> bool:
        v = self.view
        if (
            not v.has_top_level_distinct
            or any(it.is_aggregate for it in v.select_items)
            or v.group_by_cols
            or v.having_clause
            or v.has_cte
            or v.has_window_func
            or v.has_compound_set_op
            or v.has_limit_or_order
            or v.has_subquery_in_from
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
        ):
            return False
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return False
        if any(it.raw_expr.strip() == "*" or _has_subquery_ast(it.raw_expr, dialect=self.dialect) for it in v.select_items):
            return False
        return True

    def _can_use_zset_projection(self) -> bool:
        v = self.view
        if (
            any(it.is_aggregate for it in v.select_items)
            or v.group_by_cols
            or v.having_clause
            or v.has_top_level_distinct
            or v.has_cte
            or v.has_window_func
            or v.has_compound_set_op
            or v.has_limit_or_order
            or v.has_subquery_in_from
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
            or self.primary_schema is None
        ):
            return False
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return False
        if len(v.select_items) == 1 and v.select_items[0].raw_expr.strip() == "*":
            return False
        if not any(c.primary_key for c in self.primary_schema.columns.values()):
            return False
        for it in v.select_items:
            if _has_subquery_ast(it.raw_expr, dialect=self.dialect):
                return False
        return True

    def _extract_union_all_agg_metadata(
        self,
    ) -> Optional[Tuple[str, str, Tuple[ViewSpec, ...]]]:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            not has_any_agg
            or not self._decomposed_ok
            or v.has_compound_set_op
            or v.has_window_func
            or v.has_limit_or_order
            or v.has_top_level_distinct
        ):
            return None
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return None

        allowed_aggs = {
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
            AggFunc.SUM,
            AggFunc.TOTAL,
            AggFunc.AVG,
            AggFunc.MIN,
            AggFunc.MAX,
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
            AggFunc.BIT_XOR,
            AggFunc.BOOL_AND,
            AggFunc.BOOL_OR,
        }
        for it in self._working_select_items:
            if _has_subquery_ast(it.raw_expr, dialect=self.dialect):
                return None
            if it.is_aggregate and (it.is_distinct or it.agg_func not in allowed_aggs):
                return None

        inner_union_sql: Optional[str] = None
        sub_alias: str = "__mv_u"
        try:
            q_ast = parse_sql_ast(v.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(q_ast, sqlglot_exp.Select):
            return None

        sg_d = _sqlglot_dialect(self.dialect)
        if v.has_cte:
            with_node = q_ast.args.get("with_") or q_ast.args.get("with")
            if not isinstance(with_node, sqlglot_exp.With) or bool(with_node.args.get("recursive")) or len(with_node.expressions) != 1:
                return None
            cte_node = with_node.expressions[0]
            cte_name = cte_node.alias
            if len(v.joins) != 1 or v.joins[0].table_name.lower() != cte_name.lower():
                return None
            inner_union_ast = cte_node.this
            inner_union_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, True, dialect=self.dialect) or (inner_union_ast.sql(sg_d).strip() if isinstance(inner_union_ast, sqlglot_exp.Expr) else None)
            sub_alias = v.joins[0].alias or cte_name
        else:
            from_node = q_ast.args.get("from_") or q_ast.args.get("from")
            if not isinstance(from_node, sqlglot_exp.From) or not isinstance(from_node.this, sqlglot_exp.Subquery):
                return None
            sub_node = from_node.this
            inner_union_ast = sub_node.this
            inner_union_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, False, dialect=self.dialect) or (inner_union_ast.sql(sg_d).strip() if isinstance(inner_union_ast, sqlglot_exp.Expr) else None)
            if sub_node.alias:
                sub_alias = sub_node.alias

        if not inner_union_sql or not isinstance(inner_union_ast, sqlglot_exp.SetOperation):
            return None

        ops, branch_sqls = _extract_set_op_branches_from_ast(inner_union_ast, inner_union_sql, dialect=self.dialect)
        if not ops or any(op != "UNION ALL" for op in ops):
            return None

        branch_specs: List[ViewSpec] = []
        for idx_b, b_sql in enumerate(branch_sqls):
            try:
                bs = _parse_single_create_view_stmt(
                    f"CREATE VIEW __uall_b_{idx_b} AS {b_sql}",
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
                or any(it.is_aggregate for it in bs.select_items)
                or bs.base_table.lower() not in self._schemas_lower
            ):
                return None
            if bs.where_clause and _has_subquery_ast(bs.where_clause, dialect=self.dialect):
                return None
            branch_specs.append(bs)

        return (inner_union_sql, sub_alias, tuple(branch_specs))

    def _can_use_union_all_agg_algebraic(self) -> bool:
        return self._extract_union_all_agg_metadata() is not None

    def _extract_window_partition_keys(self) -> Optional[Tuple[str, ...]]:
        """Return common `PARTITION BY` expressions when a window query can use Tier 2 partition scoping."""
        v = self.view
        if (
            not v.has_window_func
            or v.has_cte
            or v.has_compound_set_op
            or v.has_limit_or_order
            or v.has_subquery_in_from
            or v.has_top_level_distinct
            or v.having_clause
            or v.group_by_cols
            or any(it.is_aggregate for it in v.select_items)
        ):
            return None
        if any(j.join_type not in ("FROM", "INNER JOIN") for j in v.joins):
            return None
        if v.where_clause and _has_subquery_ast(v.where_clause, dialect=self.dialect):
            return None

        sg_d = _sqlglot_dialect(self.dialect)
        # Disallow named WINDOW definitions (e.g. `OVER w`) unless all partitions are inline.
        try:
            q_ast = parse_sql_ast(v.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(q_ast, sqlglot_exp.Select) or q_ast.args.get("windows"):
            return None

        window_partitions: List[List[Tuple[str, str]]] = []
        for it in v.select_items:
            if _has_subquery_ast(it.raw_expr, dialect=self.dialect):
                return None
            try:
                it_ast = parse_sql_ast(it.raw_expr, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return None
            for win in it_ast.find_all(sqlglot_exp.Window):
                part_exprs = list(win.args.get("partition_by") or ())
                if not part_exprs:
                    return None
                cur_parts: List[Tuple[str, str]] = []
                for pe in part_exprs:
                    pe_sql = pe.sql(dialect=sg_d)
                    if _has_subquery_ast(pe_sql, dialect=self.dialect):
                        return None
                    # Normalize canonical key so `t.col` and `col` or case differences align
                    if isinstance(pe, sqlglot_exp.Column):
                        res = self._resolve_ast_col_to_table(pe, self._alias_to_table_map())
                        key = f"{res[0].lower()}.{res[1].lower()}" if res else pe_sql.strip().lower()
                    else:
                        key = pe_sql.strip().lower()
                    cur_parts.append((key, pe_sql))
                window_partitions.append(cur_parts)

        if not window_partitions:
            return None

        common_keys = set(k for k, _ in window_partitions[0])
        for wp in window_partitions[1:]:
            common_keys &= {k for k, _ in wp}
        if not common_keys:
            return None

        result: List[str] = []
        seen: Set[str] = set()
        for key, pe_sql in window_partitions[0]:
            if key in common_keys and key not in seen:
                seen.add(key)
                result.append(pe_sql)
        return tuple(result) if result else None

    def _tier2_scope_exprs(self) -> Tuple[str, ...]:
        if self._alg_mode == "window_partition":
            return self._extract_window_partition_keys() or ()
        return self._effective_group_by_exprs

    def _select_strategy_and_mode(self) -> Tuple[MaintenanceStrategy, str]:
        win_part_keys = self._extract_window_partition_keys()
        if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_upsert:
            v = self.view
            has_any_agg = any(it.is_aggregate for it in v.select_items)
            if (
                has_any_agg
                and v.group_by_cols
                and not v.has_cte
                and not v.has_window_func
                and not v.has_compound_set_op
                and not v.has_limit_or_order
                and not v.has_subquery_in_from
            ):
                return MaintenanceStrategy.GROUP_SCOPED_DELTA, "single"
            if win_part_keys:
                return MaintenanceStrategy.GROUP_SCOPED_DELTA, "window_partition"
            return MaintenanceStrategy.ATOMIC_QUERY_SYNC, "single"
        if self._can_use_single_table_algebraic():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "single"
        if self._can_use_join_algebraic():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "join"
        if self._can_use_zset_setop():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "setop"
        if self._can_use_zset_distinct():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "distinct"
        if self._can_use_zset_projection():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "projection"
        if self._can_use_union_all_agg_algebraic():
            return MaintenanceStrategy.ALGEBRAIC_DELTA, "union_all_agg"

        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            has_any_agg
            and v.group_by_cols
            and not v.has_cte
            and not v.has_window_func
            and not v.has_compound_set_op
            and not v.has_limit_or_order
            and not v.has_subquery_in_from
        ):
            return MaintenanceStrategy.GROUP_SCOPED_DELTA, "single"
        if win_part_keys:
            return MaintenanceStrategy.GROUP_SCOPED_DELTA, "window_partition"

        return MaintenanceStrategy.ATOMIC_QUERY_SYNC, "single"

    def _choose_strategy(self) -> MaintenanceStrategy:
        strategy, _ = self._select_strategy_and_mode()
        return strategy

    def _validate_schema_and_view(self) -> None:
        v = self.view
        if not v.referenced_tables:
            raise ValueError(f"View '{self.logical_view_name}' does not reference any known base tables.")
        for t_name in v.referenced_tables:
            if t_name.lower() not in self._schemas_lower:
                raise ValueError(
                    f"Base table {t_name!r} referenced by view {self.logical_view_name!r} was not found in schema."
                )

        if (
            len(v.joins) == 1
            and not v.has_cte
            and not v.has_subquery_in_from
            and not v.has_compound_set_op
            and self.primary_schema is not None
        ):
            assert self.primary_schema is not None
            col_map = {k.lower(): v for k, v in self.primary_schema.columns.items()}
            for g_expr in self._effective_group_by_exprs:
                if _is_simple_column_expr(g_expr.strip(), dialect=self.dialect):
                    g_col = _unquote_ident(g_expr, dialect=self.dialect)
                    if g_col.lower() not in col_map:
                        raise ValueError(
                            f"GROUP BY column {g_col!r} does not exist on table {self.primary_schema.name!r}"
                        )
            grp_exprs_lower = {g.strip().lower() for g in self._effective_group_by_exprs}
            grp_cols_lower = {_unquote_ident(g, dialect=self.dialect).lower() for g in self._effective_group_by_exprs}
            has_any_agg = any(it.is_aggregate for it in self._working_select_items)
            for item in self._working_select_items:
                if not item.is_aggregate:
                    if item.source_col is not None and _is_simple_column_expr(item.raw_expr.strip(), dialect=self.dialect):
                        if item.source_col.lower() not in col_map:
                            raise ValueError(
                                f"Selected column {item.source_col!r} does not exist on table {self.primary_schema.name!r}"
                            )
                        if (
                            has_any_agg
                            and item.source_col.lower() not in grp_cols_lower
                            and item.raw_expr.strip().lower() not in grp_exprs_lower
                        ):
                            raise ValueError(
                                f"Non-aggregated column {item.source_col!r} must appear in GROUP BY clause"
                            )
                elif item.arg_expr and _is_simple_column_expr(item.arg_expr.strip(), dialect=self.dialect):
                    col_ident = _unquote_ident(item.arg_expr, dialect=self.dialect)
                    if col_ident.lower() not in col_map:
                        raise ValueError(
                            f"Aggregate argument column {col_ident!r} does not exist on table {self.primary_schema.name!r}"
                        )

        self._validate_user_indexes()

    def _exposed_view_columns(self) -> List[str]:
        if len(self.view.select_items) == 1 and self.view.select_items[0].raw_expr.strip() == "*":
            cols: List[str] = []
            for j in self.view.joins or []:
                t_schema = self._schemas_lower.get(j.table_name.lower())
                if t_schema:
                    for c_def in t_schema.columns.values():
                        if c_def.name not in cols:
                            cols.append(c_def.name)
            if not cols and self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    cols.append(c_def.name)
            return cols
        return [item.alias for item in self.view.select_items]

    def _extract_index_col_base_ident(self, col_spec: str) -> str:
        cleaned = re.sub(
            r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|COLLATE\s+[A-Za-z0-9_.\"'`]+)\b",
            "",
            col_spec,
            flags=re.IGNORECASE,
        ).strip()
        return _unquote_ident(cleaned, dialect=self.dialect)

    def _validate_user_indexes(self) -> None:
        if not self.user_indexes:
            return
        exposed = self._exposed_view_columns()
        exposed_lower = {c.lower() for c in exposed}
        sql_kw = {
            "asc", "desc", "nulls", "first", "last", "collate", "nocase", "binary",
            "rtrim", "and", "or", "not", "is", "null", "true", "false", "in",
            "between", "like", "case", "when", "then", "else", "end", "cast", "as",
            "int", "integer", "text", "real", "double", "precision", "varchar", "char",
        } | {f.lower() for f in _SQL_EXPR_FUNCTIONS}

        for idx_spec in self.user_indexes:
            if idx_spec.target_view is not None:
                t_low = idx_spec.target_view.lower()
                valid_targets = {
                    self.logical_view_name.lower(),
                    self.db_view_name.lower(),
                    self._backing_table_name().lower(),
                    f"__matview_{self.logical_view_name.lower()}",
                    f"__matview_store_{self.logical_view_name.lower()}",
                }
                if t_low not in valid_targets:
                    raise ValueError(
                        f"Index target {idx_spec.target_view!r} does not match view {self.logical_view_name!r}"
                    )
            if not idx_spec.columns:
                raise ValueError(f"Index on view '{self.logical_view_name}' must specify at least one column.")
            for col_spec in idx_spec.columns:
                base_expr = re.sub(
                    r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|COLLATE\s+[A-Za-z0-9_.\"'`]+)\b",
                    "",
                    col_spec,
                    flags=re.IGNORECASE,
                ).strip()
                if re.match(r"^[A-Za-z0-9_\"`\[\]]+$", base_expr):
                    c_ident = _unquote_ident(base_expr, dialect=self.dialect)
                    if c_ident.lower() not in exposed_lower:
                        raise ValueError(
                            f"Index on view '{self.logical_view_name}' references unknown column "
                            f"{c_ident!r}. Available view columns: {exposed}"
                        )
                else:
                    no_str = re.sub(r"'(?:''|[^'])*'", "''", base_expr)
                    referenced_cols: List[str] = []
                    for m_id in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", no_str):
                        tok = m_id.group(1)
                        if tok.lower() in sql_kw:
                            continue
                        if tok.lower() not in exposed_lower:
                            raise ValueError(
                                f"Index expression {col_spec!r} on view '{self.logical_view_name}' "
                                f"references unknown column {tok!r}. Available view columns: {exposed}"
                            )
                        referenced_cols.append(tok)
                    if not referenced_cols:
                        raise ValueError(
                            f"Index expression {col_spec!r} on view '{self.logical_view_name}' "
                            f"does not reference any view column. Available view columns: {exposed}"
                        )

    def _indexed_column_names_lower(self, auto_group_cols: Sequence[str] = ()) -> Set[str]:
        out: Set[str] = {c.lower() for c in auto_group_cols}
        for idx_spec in self.user_indexes:
            for col_spec in idx_spec.columns:
                base_c = self._extract_index_col_base_ident(col_spec)
                out.add(base_c.lower())
        return out

    def _build_aux_dist_index_ddls(
        self,
        aux_tbl: str,
        item_alias: str,
        aux_pk_cols: Sequence[str],
        has_group_cols: bool = True,
        val_col: str = "__val",
    ) -> Tuple[List[str], List[str]]:
        """Create optimal secondary indexes on `__matview_dist_*` auxiliary tables."""
        ddls: List[str] = []
        names: List[str] = []
        p = self.object_prefix if self.object_prefix else "__matview_"
        log_vname = self.logical_view_name
        alias_tag = _sanitize_ident(item_alias)

        # Normalize PK columns (may be quoted); strip leading group col and trailing __val
        clean_pk = [_unquote_ident(c, dialect=self.dialect) for c in aux_pk_cols]
        grp_cols = [c for c in clean_pk if c.lower() not in (val_col.lower(), "__mv_pk", "__mv_global_id")]

        # Index trailing group columns for multi-column GROUP BY auxiliary tables.
        if has_group_cols and len(grp_cols) > 1:
            for idx_pos, g_col in enumerate(grp_cols[1:], start=2):
                g_idx_name = self._bounded_ident(
                    f"{p}idx_",
                    f"{log_vname}__dist_{alias_tag}_{_sanitize_ident(g_col)}",
                    f"__g{idx_pos}",
                )
                if g_idx_name not in names:
                    ddls.append(
                        build_create_index_sql(
                            g_idx_name,
                            aux_tbl,
                            [self._q(g_col)],
                            dialect=self.dialect,
                        )
                    )
                    names.append(g_idx_name)

        # Index __val so global value lookups/pruning avoid scanning every group partition.
        if val_col.lower() not in {c.lower() for c in clean_pk[:1]}:
            v_idx_name = self._bounded_ident(
                f"{p}idx_",
                f"{log_vname}__dist_{alias_tag}",
                "__val",
            )
            if v_idx_name not in names:
                ddls.append(
                    build_create_index_sql(
                        v_idx_name,
                        aux_tbl,
                        [self._q(val_col)],
                        dialect=self.dialect,
                    )
                )
                names.append(v_idx_name)
        return ddls, names

    def _alias_to_table_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if self.view.base_table:
            mapping[self.view.base_table.lower()] = self.view.base_table
        for j in self.view.joins:
            if j.table_name:
                mapping[j.table_name.lower()] = j.table_name
                if j.alias:
                    mapping[j.alias.lower()] = j.table_name
        return mapping

    def _resolve_ast_col_to_table(
        self,
        col_node: sqlglot_exp.Column,
        alias_map: Dict[str, str],
    ) -> Optional[Tuple[str, str]]:
        c_name = col_node.name
        if not c_name:
            return None
        tbl_hint = col_node.table.lower() if col_node.table else ""
        if tbl_hint:
            actual = alias_map.get(tbl_hint)
            if actual and actual.lower() in self._schemas_lower:
                col_def = self._schemas_lower[actual.lower()].columns.get(c_name)
                if col_def is None:
                    for k, v in self._schemas_lower[actual.lower()].columns.items():
                        if k.lower() == c_name.lower():
                            col_def = v
                            break
                if col_def is not None:
                    return (self._schemas_lower[actual.lower()].name, col_def.name)
            return None
        for tbl_name in self.view.referenced_tables:
            ts = self._schemas_lower.get(tbl_name.lower())
            if ts is None:
                continue
            for col_def in ts.columns.values():
                if col_def.name.lower() == c_name.lower():
                    return (ts.name, col_def.name)
        return None

    def _auto_backing_extra_index_cols(
        self,
        excluded_leading_cols: Sequence[str] = (),
    ) -> List[str]:
        """Identify projected group, join-key, or partition columns for auto-indexing on backing table."""
        excluded_low = {c.lower() for c in excluded_leading_cols if c}
        # Map lowercase base column references (tbl.col and unqualified col) -> user alias
        expr_to_alias: Dict[str, str] = {}
        for it in self.view.select_items:
            if it.is_aggregate:
                continue
            try:
                node = parse_sql_ast(it.raw_expr, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            if isinstance(node, sqlglot_exp.Column) and node.name:
                if node.table:
                    expr_to_alias[f"{node.table.lower()}.{node.name.lower()}"] = it.alias
                expr_to_alias.setdefault(node.name.lower(), it.alias)

        candidates: List[str] = []
        seen_low: Set[str] = set()

        def _add_candidate(raw_expr: str) -> None:
            try:
                node = parse_sql_ast(raw_expr, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return
            if isinstance(node, sqlglot_exp.Column) and node.name:
                key = f"{node.table.lower()}.{node.name.lower()}" if node.table else node.name.lower()
                alias = expr_to_alias.get(key) or expr_to_alias.get(node.name.lower())
                if alias and alias.lower() not in excluded_low and alias.lower() not in seen_low:
                    seen_low.add(alias.lower())
                    candidates.append(alias)

        for g in self.view.group_by_cols:
            _add_candidate(g)
        for j in self.view.joins:
            if not j.on_condition:
                continue
            try:
                on_ast = parse_sql_ast(j.on_condition, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            for eq in on_ast.find_all(sqlglot_exp.EQ):
                if isinstance(eq.left, sqlglot_exp.Column):
                    _add_candidate(eq.left.sql())
                if isinstance(eq.right, sqlglot_exp.Column):
                    _add_candidate(eq.right.sql())
        if self.view.has_window_func:
            for it in self.view.select_items:
                try:
                    ast_it = parse_sql_ast(it.raw_expr, dialect=self.dialect)
                except sqlglot.errors.SqlglotError:
                    continue
                for win in ast_it.find_all(sqlglot_exp.Window):
                    for part_e in (win.args.get("partition_by") or ()):
                        if isinstance(part_e, sqlglot_exp.Column):
                            _add_candidate(part_e.sql())
        return candidates

    def _build_all_index_ddls(
        self,
        mv_table: str,
        auto_group_cols: Sequence[str] = (),
        has_pk_on_group_cols: bool = False,
        backing_col_map: Optional[Dict[str, str]] = None,
        unique_group_nulls_not_distinct: bool = False,
        extra_auto_index_cols: Sequence[str] = (),
    ) -> Tuple[List[str], List[str]]:
        ddls: List[str] = []
        names: List[str] = []
        p = self.object_prefix if self.object_prefix else "__matview_"
        log_vname = self.logical_view_name
        covered_leading_low: Set[str] = set()
        if auto_group_cols and has_pk_on_group_cols:
            covered_leading_low.add(auto_group_cols[0].lower())
        elif auto_group_cols:
            covered_leading_low.add(auto_group_cols[0].lower())
        for idx_spec in self.user_indexes:
            if idx_spec.columns:
                covered_leading_low.add(
                    self._extract_index_col_base_ident(idx_spec.columns[0]).lower()
                )

        q_auto_group_cols = [self._q(c) for c in auto_group_cols]
        if q_auto_group_cols and not has_pk_on_group_cols:
            idx_name = self._bounded_ident(f"{p}idx_", log_vname, "__grp")
            use_nnd = bool(unique_group_nulls_not_distinct and self.engine_version.supports_pg_nulls_not_distinct)
            ddls.append(
                build_create_index_sql(
                    idx_name,
                    mv_table,
                    q_auto_group_cols,
                    dialect=self.dialect,
                    unique=use_nnd,
                    nulls_not_distinct=use_nnd,
                )
            )
            names.append(idx_name)

        if len(auto_group_cols) > 1:
            for g_idx, g_col in enumerate(auto_group_cols[1:], start=2):
                sub_idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{_sanitize_ident(g_col)}", f"__g{g_idx}")
                if sub_idx_name not in names:
                    ddls.append(
                        build_create_index_sql(sub_idx_name, mv_table, [self._q(g_col)], dialect=self.dialect)
                    )
                    names.append(sub_idx_name)
                    covered_leading_low.add(g_col.lower())

        for extra_col in extra_auto_index_cols:
            if not extra_col or extra_col.lower() in covered_leading_low:
                continue
            covered_leading_low.add(extra_col.lower())
            extra_idx_name = self._bounded_ident(
                f"{p}idx_",
                f"{log_vname}__{_sanitize_ident(extra_col)}",
                "__auto",
            )
            if extra_idx_name not in names:
                ddls.append(
                    build_create_index_sql(
                        extra_idx_name,
                        mv_table,
                        [self._q(extra_col)],
                        dialect=self.dialect,
                    )
                )
                names.append(extra_idx_name)

        for i, idx_spec in enumerate(self.user_indexes, 1):
            if idx_spec.name:
                raw_n = idx_spec.name
                if self.object_prefix and not raw_n.startswith(self.object_prefix):
                    idx_name = f"{self.object_prefix}{raw_n}"
                else:
                    idx_name = raw_n
            else:
                col_tag = "_".join(_sanitize_ident(self._extract_index_col_base_ident(c)) for c in idx_spec.columns)
                idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{col_tag}", f"_{i}")

            mapped_cols: List[str] = []
            for col_spec in idx_spec.columns:
                c_mod = col_spec.strip()
                if backing_col_map:
                    for u_col, b_expr in backing_col_map.items():
                        c_mod = re.sub(rf"\b{re.escape(u_col)}\b", b_expr, c_mod, flags=re.IGNORECASE)
                m_ident = re.match(
                    r"^((?:\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*))"
                    r"(\s+COLLATE\s+\S+)?"
                    r"(\s+(?:ASC|DESC))?"
                    r"(\s+NULLS\s+(?:FIRST|LAST))?$",
                    c_mod,
                    flags=re.IGNORECASE,
                )
                if m_ident:
                    q_col = self._q(m_ident.group(1))
                    collate_s = m_ident.group(2) or ""
                    dir_s = m_ident.group(3) or ""
                    nulls_s = m_ident.group(4) or ""
                    if self.dialect == Dialect.MYSQL:
                        c_mod = f"{q_col}{collate_s}{dir_s}"
                    else:
                        c_mod = f"{q_col}{collate_s}{dir_s}{nulls_s}"
                elif self.dialect == Dialect.MYSQL:
                    c_mod = re.sub(r"\bNULLS\s+(?:FIRST|LAST)\b", "", c_mod, flags=re.IGNORECASE).strip()
                    m_dir = re.search(r"\s+(ASC|DESC)\s*$", c_mod, flags=re.IGNORECASE)
                    dir_s = f" {m_dir.group(1).upper()}" if m_dir else ""
                    expr_part = c_mod[: m_dir.start()].strip() if m_dir else c_mod
                    if not (expr_part.startswith("((") and expr_part.endswith("))")):
                        c_mod = f"(({expr_part})){dir_s}"
                elif self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                    m_tail = re.search(
                        r"(\s+(?:ASC|DESC))?(\s+NULLS\s+(?:FIRST|LAST))?\s*$",
                        c_mod,
                        flags=re.IGNORECASE,
                    )
                    tail_s = m_tail.group(0) if m_tail else ""
                    expr_part = c_mod[: len(c_mod) - len(tail_s)].strip()
                    if not (expr_part.startswith("(") and expr_part.endswith(")")):
                        c_mod = f"({expr_part}){tail_s}"
                mapped_cols.append(c_mod)

            using_method = (
                idx_spec.using_method
                if idx_spec.using_method and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
                else None
            )
            where_clause = None
            if idx_spec.where_clause and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                w_mod = idx_spec.where_clause
                if backing_col_map:
                    for u_col, b_expr in backing_col_map.items():
                        w_mod = re.sub(rf"\b{re.escape(u_col)}\b", b_expr, w_mod, flags=re.IGNORECASE)
                where_clause = w_mod

            ddls.append(
                build_create_index_sql(
                    idx_name,
                    mv_table,
                    mapped_cols,
                    dialect=self.dialect,
                    unique=idx_spec.unique,
                    using_method=using_method,
                    where_clause=where_clause,
                )
            )
            names.append(idx_name)

        return ddls, names


    def _diagnose_tier1_selection(self) -> Tuple[str, Optional[str]]:
        mode_descriptions = {
            "single": "single-table O(1) incremental aggregates",
            "join": "multi-table INNER/CROSS JOIN O(1) delta maintenance",
            "setop": "Z-set compound set operation (UNION/INTERSECT/EXCEPT)",
            "distinct": "Z-set SELECT DISTINCT with reference counting",
            "projection": "PK-lineage projection maintenance",
            "union_all_agg": "algebraic aggregation over UNION ALL",
        }
        desc = mode_descriptions.get(self._alg_mode, "algebraic delta maintenance")
        return (f"Tier 1 algebraic delta maintenance: {desc}.", None)

    def _diagnose_tier2_fallback(self) -> Tuple[str, str]:
        if self._alg_mode == "window_partition":
            return (
                "Tier 2 partition-scoped window maintenance: recomputes only the affected PARTITION BY partition(s).",
                "Window functions require recomputing sibling rows within the affected PARTITION BY partition rather than single-row O(1) deltas.",
            )
        sel = "Tier 2 group-scoped delta maintenance: recomputes only the affected GROUP BY groups."
        if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_upsert:
            return (sel, "SQLite version lacks UPSERT (INSERT ... ON CONFLICT) required for Tier 1 algebraic delta maintenance.")

        allowed_aggs = {
            AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG,
            AggFunc.MIN, AggFunc.MAX, AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP, AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR,
            AggFunc.BOOL_AND, AggFunc.BOOL_OR,
        }
        for it in self._working_select_items:
            if it.is_aggregate and it.agg_func not in allowed_aggs:
                fn_name = it.agg_func.value if it.agg_func is not None else str(it.raw_expr)
                return (sel, f"Holistic aggregate function '{fn_name}' cannot be maintained algebraically with O(1) deltas.")
            if it.is_aggregate and it.is_distinct:
                fn_name = it.agg_func.value if it.agg_func is not None else str(it.raw_expr)
                return (sel, f"DISTINCT aggregate expression '{fn_name}' requires group-scoped recomputation.")

        for j in self.view.joins:
            if j.join_type not in ("FROM", "INNER JOIN", "CROSS JOIN"):
                return (sel, f"Non-inner join '{j.join_type}' cannot be maintained algebraically in Tier 1.")

        if not self._decomposed_ok:
            return (sel, "Compound aggregate expression could not be decomposed into algebraic components.")

        return (sel, "Query structure requires group-scoped delta recomputation rather than O(1) algebraic maintenance.")

    def _diagnose_tier3_fallback(self) -> Tuple[str, str]:
        sel = "Tier 3 atomic query sync: full atomic refresh executed on base-table mutations."
        if self.view.has_window_func:
            return (sel, "Window function (e.g. ROW_NUMBER, RANK, LAG, LEAD) requires universal atomic refresh.")
        if self.view.has_cte:
            return (sel, "Common Table Expression (WITH / CTE) requires universal atomic query refresh.")
        if self.view.has_limit_or_order:
            return (sel, "LIMIT or ORDER BY clause requires full query evaluation.")
        if self.view.has_compound_set_op:
            return (sel, "Compound set operation does not satisfy Tier 1 algebraic delta requirements.")
        if self.view.has_subquery_in_from:
            return (sel, "Subquery in FROM clause cannot be maintained incrementally.")
        if self.view.where_clause and _has_subquery_ast(self.view.where_clause):
            return (sel, "Correlated or scalar subquery in WHERE clause requires atomic refresh.")
        if any(_has_subquery_ast(it.raw_expr) for it in self.view.select_items):
            return (sel, "Subquery in SELECT list requires atomic refresh.")
        if not any(it.is_aggregate for it in self.view.select_items):
            return (sel, "Non-aggregate projection lacks a primary key on base table for incremental delta tracking.")
        if not self.view.group_by_cols:
            return (sel, "Global aggregation without GROUP BY could not be maintained algebraically.")
        if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_upsert:
            return (sel, "SQLite version lacks UPSERT support and query does not qualify for Tier 2 group-scoped maintenance.")
        return (sel, "Query structure prevents incremental maintenance; falls back to universal atomic query sync.")

    def _collect_base_table_index_needs(
        self,
        raw_plan: MaterializedViewPlan,
    ) -> List[Tuple[str, Tuple[str, ...], str]]:
        """Collect `(table_name, columns, reason)` lookup patterns needed by maintenance triggers/backfill."""
        alias_map = self._alias_to_table_map()
        needs: List[Tuple[str, Tuple[str, ...], str]] = []
        seen_keys: Set[Tuple[str, Tuple[str, ...]]] = set()

        def _record(tbl_name: str, cols: Sequence[str], reason: str) -> None:
            clean_cols = tuple(c for c in cols if c)
            if not tbl_name or not clean_cols:
                return
            key = (tbl_name.lower(), tuple(c.lower() for c in clean_cols))
            if key in seen_keys:
                return
            seen_keys.add(key)
            needs.append((tbl_name, clean_cols, reason))

        # 1. Join ON equality lookup columns per table (crucial for Tier 1 JOIN triggers & Tier 2/3 join probes).
        for j in self.view.joins:
            if not j.on_condition:
                continue
            try:
                on_ast = parse_sql_ast(j.on_condition, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            per_tbl: Dict[str, List[str]] = {}
            for eq in on_ast.find_all(sqlglot_exp.EQ):
                for side in (eq.left, eq.right):
                    if isinstance(side, sqlglot_exp.Column):
                        res = self._resolve_ast_col_to_table(side, alias_map)
                        if res is not None:
                            t_n, c_n = res
                            lst = per_tbl.setdefault(t_n, [])
                            if c_n.lower() not in {x.lower() for x in lst}:
                                lst.append(c_n)
            for t_n, j_cols in per_tbl.items():
                _record(t_n, j_cols, f"join_on_key ({j.join_type})")

        # 2. Group-by lookup columns for MIN/MAX/BIT_AND/BIT_OR delete recomputation and Tier 2 group-scoped refresh.
        # SQLite SUM/TOTAL/AVG (any column can hold REAL values) and
        # VAR/STDDEV are recomputed per affected group from the base table
        # whenever a group holds a non-INTEGER value (always for join, UNION
        # ALL and DISTINCT forms); without a group-key index each such write
        # scans the whole base table.
        sqlite_native_group_recompute = self.dialect == Dialect.SQLITE and any(
            it.is_aggregate
            and it.agg_func in (
                AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG,
                AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP,
            )
            for it in self.view.select_items
        )
        # PostgreSQL/MySQL/MariaDB order-sensitive aggregates (approximate
        # SUM/AVG, VAR/STDDEV, natively recomputed AVG) recompute every
        # affected group on every write; on MySQL/MariaDB that is a locking
        # read, which without a group-key index share-locks the whole table.
        pgmy_native_group_recompute = (
            raw_plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA
            and self._pgmy_native_recompute_items_for_plan()
        )
        needs_base_group_scan = (
            raw_plan.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA
            or sqlite_native_group_recompute
            or pgmy_native_group_recompute
            or any(
                it.is_aggregate
                and it.agg_func in (AggFunc.MIN, AggFunc.MAX, AggFunc.BIT_AND, AggFunc.BIT_OR)
                for it in self.view.select_items
            )
        )
        if needs_base_group_scan and self.view.group_by_cols and self.view.base_table:
            by_tbl: Dict[str, List[str]] = {}
            for g in self.view.group_by_cols:
                try:
                    g_ast = parse_sql_ast(g, dialect=self.dialect)
                except sqlglot.errors.SqlglotError:
                    continue
                if isinstance(g_ast, sqlglot_exp.Column):
                    res = self._resolve_ast_col_to_table(g_ast, alias_map)
                    if res is not None:
                        t_n, c_n = res
                        lst = by_tbl.setdefault(t_n, [])
                        if c_n.lower() not in {x.lower() for x in lst}:
                            lst.append(c_n)
            if raw_plan.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA:
                reason = "tier2_group_scoped_recompute"
            elif sqlite_native_group_recompute:
                reason = "sqlite_native_sum_avg_group_recompute"
            elif pgmy_native_group_recompute:
                reason = "native_order_sensitive_group_recompute"
            else:
                reason = "holistic_minmax_delete_recompute"
            for t_n, g_cols in by_tbl.items():
                _record(t_n, g_cols, reason)

        # 3. Window PARTITION BY columns (future Tier 2 window partition scoping / partitioning reads).
        if self.view.has_window_func:
            part_cols: List[str] = []
            part_tbl = self.view.base_table
            for it in self.view.select_items:
                try:
                    ast_it = parse_sql_ast(it.raw_expr, dialect=self.dialect)
                except sqlglot.errors.SqlglotError:
                    continue
                for win in ast_it.find_all(sqlglot_exp.Window):
                    for part_e in (win.args.get("partition_by") or ()):
                        if isinstance(part_e, sqlglot_exp.Column):
                            res = self._resolve_ast_col_to_table(part_e, alias_map)
                            if res is not None and res[0].lower() == part_tbl.lower():
                                if res[1].lower() not in {x.lower() for x in part_cols}:
                                    part_cols.append(res[1])
            if part_cols and part_tbl:
                _record(part_tbl, part_cols, "window_partition_by")

        # 4. Incremental backfill PK / watermark cursor column.
        if (
            self.incremental_backfill
            and raw_plan.incremental_backfill_spec is not None
            and raw_plan.incremental_backfill_spec.pk_column
            and raw_plan.incremental_backfill_spec.pk_column.lower() != "rowid"
            and self.view.base_table
        ):
            _record(
                self.view.base_table,
                [raw_plan.incremental_backfill_spec.pk_column],
                "incremental_backfill_cursor",
            )

        return needs

    def _analyze_base_table_indexes(
        self,
        raw_plan: MaterializedViewPlan,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Evaluate base-table index coverage and propose missing `recommended_base_indexes`."""
        rec_ddls: List[str] = []
        rec_meta: List[Dict[str, Any]] = []
        used_existing: List[Dict[str, Any]] = []
        p = self.object_prefix if self.object_prefix else "__matview_"

        existing_inventory: Dict[str, List[Dict[str, Any]]] = {}
        for t_name in self.view.referenced_tables:
            ts = self._schemas_lower.get(t_name.lower())
            if ts is None:
                continue
            entries: List[Dict[str, Any]] = []
            eff_pk = ts.effective_pk_columns()
            if eff_pk:
                entries.append({"name": "PRIMARY KEY", "columns": list(eff_pk), "unique": True})
            for ix in ts.indexes:
                entries.append({"name": ix.name, "columns": list(ix.columns), "unique": ix.unique})
            if entries:
                existing_inventory[ts.name] = entries

        seen_rec_names: Set[str] = set()
        for tbl_name, cols, reason in self._collect_base_table_index_needs(raw_plan):
            ts = self._schemas_lower.get(tbl_name.lower())
            if ts is None:
                continue
            matched = ts.covering_indexes_for(cols)
            if matched:
                used_existing.append(
                    {
                        "table": ts.name,
                        "columns": list(cols),
                        "indexes": list(matched),
                        "reason": reason,
                    }
                )
                continue

            # MySQL rejects unprefixed indexes on TEXT/BLOB columns.
            if self.dialect == Dialect.MYSQL:
                has_unindexable = any(
                    (cd := ts.columns.get(c)) is not None
                    and any(tk in cd.data_type.upper() for tk in ("TEXT", "BLOB"))
                    for c in cols
                )
                if has_unindexable:
                    continue

            col_tag = "_".join(_sanitize_ident(c) for c in cols)
            idx_name = self._bounded_ident(
                f"{p}base_idx_",
                f"{_sanitize_ident(ts.name)}__{col_tag}",
                "",
            )
            if idx_name in seen_rec_names:
                continue
            seen_rec_names.add(idx_name)
            ddl = build_create_index_sql(
                idx_name,
                ts.name,
                [self._q(c) for c in cols],
                dialect=self.dialect,
            )
            rec_ddls.append(ddl)
            rec_meta.append(
                {
                    "index_name": idx_name,
                    "table": ts.name,
                    "columns": list(cols),
                    "reason": reason,
                    "ddl": ddl,
                }
            )
        return rec_ddls, rec_meta, used_existing, existing_inventory

    def _build_compilation_diagnostics(
        self,
        raw_plan: MaterializedViewPlan,
        *,
        recommended_base_indexes: Sequence[str] = (),
        base_index_recommendations: Sequence[Dict[str, Any]] = (),
        existing_base_indexes_used: Sequence[Dict[str, Any]] = (),
        existing_base_indexes: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> PlanDiagnostics:
        tier = (
            1 if raw_plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA
            else 2 if raw_plan.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA
            else 3
        )
        if tier == 1:
            sel_reason, fb_reason = self._diagnose_tier1_selection()
            alg_mode = self._alg_mode
        elif tier == 2:
            sel_reason, fb_reason = self._diagnose_tier2_fallback()
            alg_mode = self._alg_mode if self._alg_mode == "window_partition" else None
        else:
            sel_reason, fb_reason = self._diagnose_tier3_fallback()
            alg_mode = None

        return PlanDiagnostics(
            strategy=raw_plan.strategy.value,
            tier=tier,
            is_incremental=(tier in (1, 2)),
            selection_reason=sel_reason,
            fallback_reason=fb_reason,
            algebraic_mode=alg_mode,
            engine_limitations=[
                {
                    "code": lim.code,
                    "description": lim.description,
                    "severity": lim.severity.value,
                }
                for lim in raw_plan.limitations
            ],
            backing_table=raw_plan.backing_table_name,
            auxiliary_tables=list(raw_plan.auxiliary_table_names),
            triggers=list(raw_plan.trigger_names),
            indexes=list(raw_plan.index_names),
            reader_view=raw_plan.view_name,
            details={
                "dialect": self.dialect.value,
                "engine_version": self.engine_version.label if self.engine_version else None,
                "base_tables": list(self.view.referenced_tables),
                "existing_base_indexes": dict(existing_base_indexes or {}),
                "existing_base_indexes_used": list(existing_base_indexes_used),
                "recommended_base_indexes": list(recommended_base_indexes),
                "base_index_recommendations": list(base_index_recommendations),
                "performance_notes": self._performance_notes(),
            },
        )

    def _pgmy_native_recompute_items_for_plan(self) -> bool:
        try:
            return bool(self._pgmy_native_recompute_items())
        except Exception:
            return False

    def _performance_notes(self) -> List[str]:
        """Documented maintenance cost classes that are not O(1) per write."""
        notes: List[str] = []
        if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL) and self._alg_mode in ("single", "union_all_agg"):
            if self._pgmy_native_recompute_items_for_plan():
                notes.append(
                    "native_order_sensitive_group_recompute: approximate SUM/AVG, VAR/STDDEV and "
                    "natively recomputed AVG columns are recomputed from the base table for every "
                    "affected group on every write, in a separate statement after the group row is "
                    "locked (O(group rows) with a group-key index, O(table) without)."
                )
                if self.dialect == Dialect.MYSQL:
                    notes.append(
                        "mysql_locking_group_recompute: on MySQL/MariaDB that recompute is a locking "
                        "read (LOCK IN SHARE MODE). Without an index on the group-key columns InnoDB "
                        "share-locks every scanned base row until commit, so concurrent writers to "
                        "different groups deadlock frequently (InnoDB rolls one back with error 1213; "
                        "retry it). Create the indexes in recommended_base_indexes."
                    )
            return notes
        if self.dialect != Dialect.SQLITE:
            return notes
        exact_sum_items = [
            it for it in self.view.select_items
            if it.is_aggregate and it.agg_func in (AggFunc.SUM, AggFunc.TOTAL, AggFunc.AVG)
        ]
        if not exact_sum_items:
            return notes
        if self.view.joins and len(self.view.joins) > 1 or any(it.is_distinct for it in exact_sum_items) \
                or self._alg_mode in ("union_all_agg", "join"):
            notes.append(
                "sqlite_sum_total_avg_native_recompute_always: join, UNION ALL and DISTINCT "
                "SUM/TOTAL/AVG recompute each affected group from the base table(s) on every "
                "write (O(group rows) with a group/join-key index, O(table) without)."
            )
        else:
            notes.append(
                "sqlite_sum_total_avg_native_recompute_on_real: a group holding any non-INTEGER "
                "value (any value of a REAL-affinity column) is recomputed from the base table on "
                "every write (O(group rows) with a group-key index, O(table) without); "
                "integer-only groups keep the O(1) exact fold."
            )
        return notes

    def _pg_plan_prologue(
        self, raw_plan: MaterializedViewPlan, extra_sqls: Sequence[str]
    ) -> Tuple[List[str], List[str]]:
        """PostgreSQL objects every plan statement may depend on, run first in
        the trigger DDLs (after the structure, before population): the
        collation guard, and the canonical-key-text function when any
        statement calls it. Returns ``(trigger_ddls, function_names)``."""
        trigger_ddls = list(raw_plan.trigger_ddls)
        function_names = list(raw_plan.function_names)
        if self.dialect != Dialect.POSTGRES:
            return trigger_ddls, function_names
        prologue: List[str] = []
        guard = self._pg_collation_guard_ddl()
        if guard:
            prologue.append(guard)
        fn = self._pg_key_text_fn()
        statements = [
            raw_plan.backing_table_ddl, raw_plan.reader_view_ddl or "",
            *raw_plan.auxiliary_table_ddls, *raw_plan.index_ddls, *raw_plan.readonly_trigger_ddls,
            *trigger_ddls, *raw_plan.initial_populate_sqls, *extra_sqls,
        ]
        fn_re = re.compile(rf"(?<![A-Za-z0-9_$]){re.escape(fn)}\(")
        # The flag covers statements outside the plan's SQL lists (e.g. the
        # incremental backfill spec's queries).
        if getattr(self, "_pg_key_text_fn_used", False) or any(fn_re.search(s) for s in statements):
            prologue.append(self._pg_key_text_fn_ddl())
            function_names.append(f"{fn}(anyelement)")
        return prologue + trigger_ddls, function_names

    def _finalize_plan_backfill(self, raw_plan: MaterializedViewPlan) -> MaterializedViewPlan:
        full_sqls = list(raw_plan.initial_populate_sqls)
        inc_sqls = self._build_dialect_incremental_populate_sqls(raw_plan)
        pg_trigger_ddls, pg_function_names = self._pg_plan_prologue(raw_plan, inc_sqls)
        active_sqls = inc_sqls if self.incremental_backfill else full_sqls
        rec_ddls, rec_meta, used_existing, existing_inv = self._analyze_base_table_indexes(raw_plan)
        diagnostics = self._build_compilation_diagnostics(
            raw_plan,
            recommended_base_indexes=rec_ddls,
            base_index_recommendations=rec_meta,
            existing_base_indexes_used=used_existing,
            existing_base_indexes=existing_inv,
        )
        return MaterializedViewPlan(
            dialect=raw_plan.dialect,
            strategy=raw_plan.strategy,
            view_spec=raw_plan.view_spec,
            backing_table_name=raw_plan.backing_table_name,
            backing_table_ddl=raw_plan.backing_table_ddl,
            auxiliary_table_ddls=raw_plan.auxiliary_table_ddls,
            index_ddls=raw_plan.index_ddls,
            initial_populate_sqls=active_sqls,
            trigger_ddls=pg_trigger_ddls,
            readonly_trigger_ddls=raw_plan.readonly_trigger_ddls,
            user_columns=raw_plan.user_columns,
            reader_view_ddl=raw_plan.reader_view_ddl,
            incremental_backfill_spec=raw_plan.incremental_backfill_spec,
            auxiliary_table_names=raw_plan.auxiliary_table_names,
            trigger_names=raw_plan.trigger_names,
            function_names=pg_function_names,
            limitations=raw_plan.limitations,
            object_prefix=raw_plan.object_prefix,
            db_view_name=raw_plan.db_view_name,
            index_names=raw_plan.index_names,
            user_indexes=raw_plan.user_indexes,
            sub_plans=raw_plan.sub_plans,
            incremental_backfill=self.incremental_backfill,
            batch_size=self.batch_size,
            full_populate_sqls=full_sqls,
            incremental_populate_sqls=inc_sqls,
            engine_version=self.engine_version,
            diagnostics=diagnostics,
            recommended_base_indexes=rec_ddls,
            base_index_recommendations=rec_meta,
        )

    def _compile_single_plan(self) -> MaterializedViewPlan:
        if self.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA:
            if self._alg_mode == "join":
                raw = self._compile_tier1_join_algebraic()
            elif self._alg_mode == "setop":
                raw = self._compile_tier1_zset_setop()
            elif self._alg_mode == "distinct":
                raw = self._compile_tier1_zset_distinct()
            elif self._alg_mode == "projection":
                raw = self._compile_tier1_zset_projection()
            elif self._alg_mode == "union_all_agg":
                raw = self._compile_tier1_union_all_agg()
            else:
                raw = self._compile_tier1_algebraic()
        elif self.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA:
            raw = self._compile_tier2_group_scoped()
        else:
            raw = self._compile_tier3_atomic_sync()
        if self.dialect == Dialect.MYSQL:
            self._enforce_mariadb_statistical_exactness(raw.reader_view_ddl or "")
            self._enforce_mysql_minmax_typing()
        return self._finalize_plan_backfill(raw)

    def compile(self) -> MaterializedViewPlan:
        first_plan = self._compile_single_plan()
        if not self.raw_view.additional_views:
            return first_plan

        sub_plans: List[MaterializedViewPlan] = [first_plan]
        for extra_v in self.raw_view.additional_views:
            sub_comp = self.__class__(
                schemas=self.schemas,
                view_spec=extra_v,
                dialect=self.dialect,
                ignore_limitations=self.ignore_limitations,
                accept_limitations=self.accepted_limitations,
                object_prefix=self.object_prefix,
                incremental_backfill=self.incremental_backfill,
                batch_size=self.batch_size,
                strict=self.strict,
                engine_version=self.engine_version,
                pg_fillfactor=getattr(self, "pg_fillfactor", 80),
            )
            sub_plans.append(sub_comp._compile_single_plan())

        return MaterializedViewPlan(
            dialect=first_plan.dialect,
            strategy=first_plan.strategy,
            view_spec=first_plan.view_spec,
            backing_table_name=first_plan.backing_table_name,
            backing_table_ddl=first_plan.backing_table_ddl,
            auxiliary_table_ddls=[d for sp in sub_plans for d in sp.auxiliary_table_ddls],
            index_ddls=[d for sp in sub_plans for d in sp.index_ddls],
            initial_populate_sqls=[d for sp in sub_plans for d in sp.initial_populate_sqls],
            trigger_ddls=[d for sp in sub_plans for d in sp.trigger_ddls],
            readonly_trigger_ddls=[d for sp in sub_plans for d in sp.readonly_trigger_ddls],
            user_columns=first_plan.user_columns,
            reader_view_ddl=first_plan.reader_view_ddl,
            incremental_backfill_spec=first_plan.incremental_backfill_spec,
            auxiliary_table_names=[n for sp in sub_plans for n in sp.auxiliary_table_names],
            trigger_names=[n for sp in sub_plans for n in sp.trigger_names],
            function_names=[n for sp in sub_plans for n in sp.function_names],
            limitations=[lim for sp in sub_plans for lim in sp.limitations],
            object_prefix=first_plan.object_prefix,
            db_view_name=first_plan.db_view_name,
            index_names=[n for sp in sub_plans for n in sp.index_names],
            user_indexes=first_plan.user_indexes,
            sub_plans=sub_plans,
            incremental_backfill=self.incremental_backfill,
            batch_size=self.batch_size,
            full_populate_sqls=[d for sp in sub_plans for d in sp.full_populate_sqls],
            incremental_populate_sqls=[d for sp in sub_plans for d in sp.incremental_populate_sqls],
            engine_version=self.engine_version,
            diagnostics=first_plan.diagnostics,
        )

