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
from .diagnostics import Diagnostic, DiagnosticSeverity, Strategy, emit as emit_diagnostics, exact_fallback_message
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
    _FN_TO_AGG_ENUM,
    _SQL_EXPR_FUNCTIONS,
    _STATISTICAL_ALIAS_NAMES,
    _contains_aggregate_call,
    _extract_cte_names_and_strip_with,
    _extract_set_op_branches_from_ast,
    _extract_verbatim_cte_or_subquery_sql,
    _find_top_level_token,
    _has_subquery_ast,
    _is_pg_catalog_qualified,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _parse_index_col_ast,
    _parse_select_item,
    _parse_single_agg_expr,
    _parse_single_create_view_stmt,
    _sanitize_ident,
    _split_top_level,
    _strip_outer_ident_quotes,
    _strip_sql_comments,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
    _unwrap_pg_catalog_agg_node,
    build_create_index_sql,
    parse_sql_ast,
    parse_view_ast,
    quote_sql_identifier,
)



# Diagnostic code of the HAVING scalar-subquery-aggregate fallback.
HAVING_SUBQUERY_AGGREGATE_FALLBACK = "HAVING_SUBQUERY_AGGREGATE_FALLBACK"

def _iter_agg_ast_nodes(root: sqlglot_exp.Expr, dialect: Dialect) -> List[sqlglot_exp.Expr]:
    """Return every aggregate call node in `root`, including PostgreSQL
    `pg_catalog.<agg>(...)` (with or without `FILTER`), while skipping
    non-`pg_catalog` schema-qualified UDFs (e.g. `myschema.sum(x)`)."""
    out: List[sqlglot_exp.Expr] = []
    seen: Set[int] = set()
    known_names = set(_FN_TO_AGG_ENUM) | set(_STATISTICAL_ALIAS_NAMES) | {"COUNT"}
    for n in root.walk():
        if isinstance(n, (sqlglot_exp.Dot, sqlglot_exp.Column)):
            unwrapped = _unwrap_pg_catalog_agg_node(n, dialect)
            if unwrapped is not None:
                fname = (
                    unwrapped.name.upper()
                    if isinstance(unwrapped, sqlglot_exp.Anonymous)
                    else unwrapped.sql_name().upper()
                )
                if isinstance(unwrapped, sqlglot_exp.AggFunc) or fname in known_names:
                    if id(unwrapped) not in seen:
                        seen.add(id(unwrapped))
                        out.append(n)
                    continue
        if isinstance(n, sqlglot_exp.AggFunc) and id(n) not in seen:
            parent = n.parent
            if isinstance(parent, sqlglot_exp.Dot) and parent.expression is n:
                if _unwrap_pg_catalog_agg_node(parent, dialect) is None:
                    continue
            seen.add(id(n))
            out.append(n)
    return out


def _has_subquery_aggregate(sql_text: str, dialect: Dialect) -> bool:
    """Whether an aggregate appears inside a (scalar) subquery of `sql_text`:
    it belongs to that subquery, not to the view's groups."""
    try:
        node = parse_sql_ast(sql_text, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    for sub in node.find_all(sqlglot_exp.Subquery, sqlglot_exp.Select):
        if sub is node:
            continue
        if _iter_agg_ast_nodes(sub, dialect):
            return True
    return False


def _mixes_outer_and_subquery_aggregates(sql_text: str, dialect: Dialect) -> bool:
    """Whether `sql_text` combines an aggregate of the view's groups with an
    aggregate inside a scalar subquery (e.g. `SUM(x) + (SELECT MAX(v) FROM o)`)."""
    if not _has_subquery_aggregate(sql_text, dialect):
        return False
    try:
        node = parse_sql_ast(sql_text, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    for agg in _iter_agg_ast_nodes(node, dialect):
        parent = agg.parent
        while parent is not None and parent is not node and not isinstance(parent, (sqlglot_exp.Subquery, sqlglot_exp.Select)):
            parent = parent.parent
        if parent is None or parent is node:
            return True
    return False

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
            if fn is not None:
                unwrapped_pg = _unwrap_pg_catalog_agg_node(fn, self.dialect)
                if unwrapped_pg is not None:
                    fn = unwrapped_pg
            if not isinstance(fn, sqlglot_exp.Func):
                continue
            fname = (
                fn.name.upper()
                if isinstance(fn, sqlglot_exp.Anonymous)
                else fn.sql_name().upper()
            )
            if fname not in names and str(getattr(fn, "name", "")).upper() not in names:
                continue
            arg = fn.this
            if isinstance(arg, sqlglot_exp.Distinct):
                arg = arg.expressions[0] if arg.expressions else None
            elif isinstance(fn, sqlglot_exp.Anonymous) and getattr(fn, "expressions", None):
                arg = fn.expressions[0]
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
        uall_branches = self._union_all_branch_specs
        if alg_mode == "union_all_agg" and not uall_branches:
            meta = self._extract_union_all_agg_metadata()
            if meta is not None:
                uall_branches = tuple(meta[2])
        check_ctx = self._union_all_common_ctx(uall_branches) if uall_branches else self
        items = [
            it.raw_expr for it in self._working_select_items
            if it.is_aggregate and check_ctx._is_order_sensitive_aggregate(it)
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
                lead = [_strip_outer_ident_quotes(str(c)).lower() for c in cols[: len(want)]]
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

        def _is_base_star(raw_expr: str) -> bool:
            rs = raw_expr.strip()
            if rs == "*":
                return True
            if rs.endswith(".*"):
                prefix = _strip_outer_ident_quotes(rs[:-2].strip()).lower()
                return prefix in (base_alias, base_name)
            return False

        has_star = any(_is_base_star(it.raw_expr) for it in v.select_items)
        if not has_star:
            return v
        expanded: List[SelectItem] = []
        for it in v.select_items:
            if _is_base_star(it.raw_expr):
                for c_def in self.primary_schema.columns.values():
                    expanded.append(
                        SelectItem(
                            raw_expr=self._q(c_def.name),
                            alias=c_def.name,
                            is_aggregate=False,
                            source_col=c_def.name,
                        )
                    )
            else:
                expanded.append(it)
        sel_str = ", ".join(f"{it.raw_expr} AS {self._q(it.alias)}" for it in expanded)
        dist_kw = "DISTINCT " if v.has_top_level_distinct else ""
        from_j = v.joins[0]
        from_str = (
            f"{self._q(from_j.table_name)} AS {self._q(from_j.alias)}"
            if from_j.alias and from_j.alias.lower() != from_j.table_name.lower()
            else self._q(from_j.table_name)
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
            original_select_sql=v.original_select_sql,
            referenced_plain_views=v.referenced_plain_views,
        )

    def _try_unwrap_simple_cte_or_derived_view(self, v: ViewSpec, _depth: int = 0) -> ViewSpec:
        """
        Unwraps a non-recursive single-CTE or derived subquery over a single base table
        (including linearly chained flat plain views) when the inner query is a flat
        projection/filter (no inner GROUP BY, aggregates, DISTINCT, window functions,
        set ops, or LIMIT) so the outer query can use Tier 1 O(1) algebraic maintenance.
        """
        if _depth >= 16 or v.has_window_func or v.has_compound_set_op or v.has_limit_or_order:
            return v

        inner_sql: Optional[str] = None
        cte_or_sub_alias: Optional[str] = None
        outer_sql: Optional[str] = None
        inner_ast_node: Optional[sqlglot_exp.Expr] = None

        try:
            q_ast = parse_sql_ast(v.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(q_ast, sqlglot_exp.Select):
            return v

        sg_d = _sqlglot_dialect(self.dialect)
        alias_col_names: Optional[List[str]] = None
        if v.has_cte:
            with_node = q_ast.args.get("with_") or q_ast.args.get("with")
            if not isinstance(with_node, sqlglot_exp.With) or bool(with_node.args.get("recursive")) or len(with_node.expressions) != 1:
                return v
            cte_node = with_node.expressions[0]
            if not isinstance(cte_node, sqlglot_exp.CTE) or not isinstance(cte_node.this, sqlglot_exp.Expr):
                return v
            cte_alias_node = cte_node.args.get("alias")
            if isinstance(cte_alias_node, sqlglot_exp.TableAlias) and cte_alias_node.columns:
                alias_col_names = [
                    c.name for c in cte_alias_node.columns if isinstance(c, sqlglot_exp.Identifier) and c.name
                ]
                if len(alias_col_names) != len(cte_alias_node.columns):
                    return v
            inner_ast_node = cte_node.this
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, True, dialect=self.dialect) or cte_node.this.sql(sg_d).strip()
            cte_or_sub_alias = cte_node.alias
            _, outer_sql, _ = _extract_cte_names_and_strip_with(v.raw_select_sql, dialect=self.dialect)
        elif v.has_subquery_in_from and len(v.joins) == 1:
            from_node = q_ast.args.get("from_") or q_ast.args.get("from")
            if not isinstance(from_node, sqlglot_exp.From) or not isinstance(from_node.this, sqlglot_exp.Subquery):
                return v
            sub_node = from_node.this
            sub_alias_node = sub_node.args.get("alias")
            if isinstance(sub_alias_node, sqlglot_exp.TableAlias) and sub_alias_node.columns:
                alias_col_names = [
                    c.name for c in sub_alias_node.columns if isinstance(c, sqlglot_exp.Identifier) and c.name
                ]
                if len(alias_col_names) != len(sub_alias_node.columns):
                    return v
            if not isinstance(sub_node.this, sqlglot_exp.Expr):
                return v
            inner_ast_node = sub_node.this
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, False, dialect=self.dialect) or sub_node.this.sql(sg_d).strip()
            cte_or_sub_alias = sub_node.alias or "__mv_sub"
            outer_sql = v.raw_select_sql
        else:
            return v

        if not inner_sql or not outer_sql or inner_ast_node is None:
            return v
        # Bail out if outer query references additional tables or contains any nested
        # subquery outside the single CTE / FROM derived table being unwrapped.
        if len(v.referenced_tables) != 1:
            return v

        def _within_inner(node: sqlglot_exp.Expr) -> bool:
            cur: Optional[sqlglot_exp.Expr] = node
            while cur is not None:
                if cur is inner_ast_node:
                    return True
                cur = cur.parent
            return False

        for sel_node in q_ast.find_all(sqlglot_exp.Select):
            if sel_node is not q_ast and not _within_inner(sel_node):
                return v

        try:
            inner_spec = _parse_single_create_view_stmt(
                f"CREATE VIEW __inner_tmp AS {inner_sql}",
                known_tables=set(self._schemas_lower.keys()),
                dialect=self.dialect,
            )
        except Exception:
            return v

        # Recursively unwrap linearly chained flat derived views (e.g. pv2 -> pv1 -> base).
        if inner_spec.has_cte or inner_spec.has_subquery_in_from:
            inner_spec = self._try_unwrap_simple_cte_or_derived_view(inner_spec, _depth=_depth + 1)

        try:
            inner_ast_check = parse_sql_ast(inner_spec.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(inner_ast_check, sqlglot_exp.Select) or any(
            inner_ast_check.args.get(k) for k in ("group", "having", "order", "limit", "offset")
        ):
            return v
        if len(list(inner_ast_check.find_all(sqlglot_exp.Select))) != 1:
            return v
        inner_from = inner_ast_check.args.get("from_") or inner_ast_check.args.get("from")
        if isinstance(inner_from, sqlglot_exp.From) and isinstance(inner_from.this, sqlglot_exp.Table):
            if inner_from.this.db or inner_from.this.catalog:
                return v

        if (
            inner_spec.has_cte
            or inner_spec.has_window_func
            or inner_spec.has_compound_set_op
            or inner_spec.has_top_level_distinct
            or inner_spec.has_limit_or_order
            or inner_spec.has_subquery_in_from
            or len(inner_spec.joins) != 1
            or len(inner_spec.referenced_tables) != 1
            or inner_spec.group_by_cols
            or any(it.is_aggregate for it in inner_spec.select_items)
        ):
            return v

        if len(v.joins) != 1:
            return v

        inner_base = inner_spec.joins[0].table_name
        if inner_base.lower() not in self._schemas_lower:
            return v

        # Build column substitution map: alias -> raw_expr (with inner table alias stripped
        # via SQL tokenization so string literals like `'t.A'` are never modified).
        inner_alias = inner_spec.joins[0].alias

        def _strip_inner_alias(frag: str) -> str:
            toks_f = _tokenize_sql(frag, read=sg_d)
            if not toks_f:
                return frag
            pieces: List[str] = []
            cur_pos = 0
            idx_t = 0
            n_t = len(toks_f)
            al_low = inner_alias.lower()
            while idx_t < n_t:
                tk_t = toks_f[idx_t]
                if (
                    tk_t.token_type in (SQLGlotTokenType.VAR, SQLGlotTokenType.IDENTIFIER)
                    and tk_t.text.lower() == al_low
                    and (idx_t == 0 or toks_f[idx_t - 1].token_type != SQLGlotTokenType.DOT)
                    and idx_t + 2 < n_t
                    and toks_f[idx_t + 1].token_type == SQLGlotTokenType.DOT
                    and _is_word_or_ident_token(toks_f[idx_t + 2])
                ):
                    if tk_t.start > cur_pos:
                        pieces.append(frag[cur_pos : tk_t.start])
                    cur_pos = toks_f[idx_t + 1].end + 1
                    idx_t += 2
                    continue
                idx_t += 1
            if cur_pos < len(frag):
                pieces.append(frag[cur_pos:])
            return "".join(pieces)

        inner_expanded: List[Tuple[str, str]] = []
        inner_base_low = inner_base.lower()
        inner_alias_low = inner_alias.lower()
        for it in inner_spec.select_items:
            raw_in = it.raw_expr.strip()
            is_inner_star = raw_in == "*" or (
                raw_in.endswith(".*")
                and _unquote_ident(raw_in[:-2].strip(), dialect=self.dialect).lower() in (inner_base_low, inner_alias_low)
            )
            if is_inner_star:
                ts = self._schemas_lower[inner_base_low]
                for c_name in ts.columns:
                    inner_expanded.append((c_name, quote_sql_identifier(c_name, self.dialect, allow_qualified=False)))
            elif raw_in.endswith(".*"):
                return v
            else:
                cleaned_e = _strip_inner_alias(raw_in).strip()
                wrapped_e = (
                    cleaned_e
                    if re.match(r'^(?:[A-Za-z_][A-Za-z0-9_]*|"[^"]+"|`[^`]+`)$', cleaned_e)
                    else f"({cleaned_e})"
                )
                inner_expanded.append((it.alias, wrapped_e))

        if alias_col_names:
            if len(alias_col_names) > len(inner_expanded):
                return v
            inner_expanded = [
                (alias_col_names[idx] if idx < len(alias_col_names) else cname, expr)
                for idx, (cname, expr) in enumerate(inner_expanded)
            ]

        col_map: Dict[str, str] = {cname.lower(): expr for cname, expr in inner_expanded}

        # If HAVING references an outer SELECT alias that collides with an inner
        # column name (e.g. `SUM(v) AS v ... HAVING v >= 10`), bail out rather than
        # substituting the inner expression into HAVING.
        if v.having_clause:
            outer_agg_aliases = {
                it.alias.lower()
                for it in v.select_items
                if it.alias and (it.is_aggregate or it.alias.lower() != it.raw_expr.strip().lower())
            }
            if outer_agg_aliases & set(col_map.keys()):
                return v

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
                    tk.token_type in (SQLGlotTokenType.VAR, SQLGlotTokenType.IDENTIFIER)
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
                if tk.token_type in (SQLGlotTokenType.VAR, SQLGlotTokenType.IDENTIFIER):
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

        outer_alias_set = {outer_alias.lower(), (cte_or_sub_alias or "").lower()}
        new_items: List[SelectItem] = []
        for it in v.select_items:
            raw_out = it.raw_expr.strip()
            is_outer_star = raw_out == "*" or (
                raw_out.endswith(".*")
                and _unquote_ident(raw_out[:-2].strip(), dialect=self.dialect).lower() in outer_alias_set
            )
            if is_outer_star:
                for cname, cexpr in inner_expanded:
                    q_al = quote_sql_identifier(cname, self.dialect, allow_qualified=False)
                    new_items.append(_parse_select_item(f"{cexpr} AS {q_al}", 0, dialect=self.dialect))
            elif raw_out.endswith(".*"):
                return v
            else:
                sub_raw = _subst(it.raw_expr) or it.raw_expr
                q_al = quote_sql_identifier(it.alias, self.dialect, allow_qualified=False)
                parsed = _parse_select_item(f"{sub_raw} AS {q_al}", 0, dialect=self.dialect)
                new_items.append(parsed)

        inner_w = None
        if inner_spec.where_clause:
            inner_w = _strip_inner_alias(inner_spec.where_clause).strip()
        outer_w = _subst(v.where_clause)
        if inner_w and outer_w:
            comb_w = f"({inner_w}) AND ({outer_w})"
        else:
            comb_w = inner_w or outer_w

        new_grps = tuple(_subst(g) or g for g in v.group_by_cols)
        new_hav = _subst(v.having_clause)

        sel_str = ", ".join(
            f"{it.raw_expr} AS {quote_sql_identifier(it.alias, self.dialect, allow_qualified=False)}"
            for it in new_items
        )
        Dist = "DISTINCT " if v.has_top_level_distinct else ""
        q_inner_base = quote_sql_identifier(inner_base, self.dialect, allow_qualified=False)
        rebuilt_sql = f"SELECT {Dist}{sel_str} FROM {q_inner_base}"
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
            original_select_sql=v.original_select_sql,
            referenced_plain_views=v.referenced_plain_views,
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
                unwrapped_it = _unwrap_outer_parens(it.raw_expr.strip(), dialect=self.dialect)
                it_toks = _tokenize_sql(unwrapped_it, read=sg_d)
                if len(it_toks) >= 4 and _is_pg_catalog_qualified(it_toks, 2, self.dialect):
                    it_toks = it_toks[2:]
                agg_call_to_alias[" ".join(t.text.upper() for t in it_toks)] = it.alias

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
                has_dot_prefix = idx >= 1 and toks[idx - 1].token_type == SQLGlotTokenType.DOT
                is_pg_cat = _is_pg_catalog_qualified(toks, idx, self.dialect)
                if has_dot_prefix and not is_pg_cat:
                    idx += 1
                    continue
                fn_up = _unquote_ident(tk.text, dialect=self.dialect).upper() if is_pg_cat else tk.text.upper()
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
                        call_start = toks[idx - 2].start if is_pg_cat else tk.start
                        full_call = expr_str[call_start : toks[end_tok_idx].end + 1]
                        unqual_call = expr_str[tk.start : toks[end_tok_idx].end + 1]
                        norm_key = " ".join(t.text.upper() for t in _tokenize_sql(unqual_call, read=sg_d))
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
                        if call_start > cursor:
                            out_parts.append(expr_str[cursor:call_start])
                        out_parts.append(self._q(col_al))
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
        # more than 1 row"). A HAVING clause holding one is not decomposed
        # (all_ok=False): the strategy selection falls back to a mode that
        # evaluates the original clause (announced as a fallback diagnostic).
        # A projected column holding one is refused (its type is not derived).
        subquery_agg_texts = {
            text
            for text in [it.raw_expr for it in v.select_items if it.is_aggregate] + (
                [v.having_clause] if v.having_clause else []
            )
            if _has_subquery_aggregate(text, self.dialect)
        }

        all_ok = True
        for it in v.select_items:
            if (it.is_aggregate and it.raw_expr in subquery_agg_texts) or (
                not it.is_aggregate and _mixes_outer_and_subquery_aggregates(it.raw_expr, self.dialect)
            ):
                # A projected column holding a subquery aggregate would be
                # stored: its result type is not derived (a fallback would
                # store e.g. SUM(x) + (SELECT MAX(v) ..) as DOUBLE where the
                # engine returns BIGINT/DECIMAL), so it stays refused.
                raise UnsupportedMaterializedViewError(
                    f"View '{self.logical_view_name}' uses an aggregate inside a scalar subquery in an "
                    f"aggregated projection ({it.raw_expr.strip()[:120]}), which is not maintained."
                )
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
                        q_al = self._q(it.alias)
                        reader_exprs[it.alias] = f"ROUND({q_al}, {result_scale}) AS {q_al}"
                    else:
                        reader_exprs[it.alias] = it.alias
                elif self.dialect == Dialect.MYSQL and it.agg_func in (AggFunc.BIT_OR, AggFunc.BIT_XOR):
                    # Keep COALESCE's result type unsigned, matching MySQL's
                    # BIT_* aggregate contract (unsigned 64-bit integer).
                    q_al = self._q(it.alias)
                    reader_exprs[it.alias] = f"COALESCE({q_al}, CAST(0 AS UNSIGNED)) AS {q_al}"
                elif self.dialect == Dialect.MYSQL and it.agg_func == AggFunc.BIT_AND:
                    q_al = self._q(it.alias)
                    reader_exprs[it.alias] = f"COALESCE({q_al}, CAST(18446744073709551615 AS UNSIGNED)) AS {q_al}"
                else:
                    reader_exprs[it.alias] = it.alias
            else:
                rewritten_e, ok = _extract_agg_calls_from_expr(it.raw_expr)
                if not ok:
                    all_ok = False
                    work_items.append(it)
                    reader_exprs[it.alias] = it.alias
                else:
                    reader_exprs[it.alias] = f"({rewritten_e}) AS {self._q(it.alias)}"

        rewritten_having: Optional[str] = None
        if v.having_clause:
            rh, ok_h = (
                (v.having_clause, False)
                if v.having_clause in subquery_agg_texts
                else _extract_agg_calls_from_expr(v.having_clause)
            )
            if ok_h:
                # Also strip table qualifiers on group columns in HAVING if they match group aliases
                qual_group_map: Dict[Tuple[str, str], str] = {}
                for g_it in v.select_items:
                    if not g_it.is_aggregate and "." in g_it.raw_expr:
                        g_parts = _split_top_level(g_it.raw_expr.strip(), ".", dialect=self.dialect)
                        if len(g_parts) == 2:
                            q_k = _strip_outer_ident_quotes(g_parts[0]).lower()
                            c_k = _strip_outer_ident_quotes(g_parts[1]).lower()
                            qual_group_map[(q_k, c_k)] = self._q(g_it.alias)
                if qual_group_map:
                    try:
                        rh_toks = _tokenize_sql(rh, read=sg_d)
                    except sqlglot.errors.SqlglotError:
                        rh_toks = []
                    if rh_toks:
                        rh_parts: List[str] = []
                        rh_cur = 0
                        rh_i = 0
                        while rh_i < len(rh_toks):
                            if (
                                rh_i + 2 < len(rh_toks)
                                and _is_word_or_ident_token(rh_toks[rh_i])
                                and rh_toks[rh_i + 1].token_type == SQLGlotTokenType.DOT
                                and _is_word_or_ident_token(rh_toks[rh_i + 2])
                            ):
                                pair_k = (rh_toks[rh_i].text.lower(), rh_toks[rh_i + 2].text.lower())
                                repl_al = qual_group_map.get(pair_k)
                                if repl_al is not None:
                                    if rh_toks[rh_i].start > rh_cur:
                                        rh_parts.append(rh[rh_cur : rh_toks[rh_i].start])
                                    rh_parts.append(repl_al)
                                    rh_cur = rh_toks[rh_i + 2].end + 1
                                    rh_i += 3
                                    continue
                            rh_i += 1
                        if rh_cur < len(rh):
                            rh_parts.append(rh[rh_cur:])
                        rh = "".join(rh_parts)
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
            gs_unq_low = _strip_outer_ident_quotes(gs).lower()
            if gs_unq_low in alias_to_raw:
                # Check if gs is an alias for an expression rather than a base column
                raw_e = alias_to_raw[gs_unq_low]
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

    def _parse_user_index_col_info(
        self, col_spec: str, exposed_lower: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], List[str], Optional[sqlglot_exp.Expr]]:
        """Parse a user index key spec via sqlglot AST.

        Returns ``(simple_col_name, referenced_col_names, parsed_ast)`` where
        ``simple_col_name`` is non-None when the key targets a single column
        (with optional COLLATE / ASC / DESC / NULLS FIRST|LAST) and None for
        expression keys.
        """
        from .sql_frontend import _parse_index_col_ast

        s = col_spec.strip()
        if not s:
            return None, [], None
        if exposed_lower is not None and s.lower() in exposed_lower:
            return s, [s], sqlglot_exp.Column(this=sqlglot_exp.to_identifier(s, quoted=True))
        try:
            parsed = _parse_index_col_ast(s, self.dialect)
        except sqlglot.errors.SqlglotError:
            return None, [], None
        inner = parsed
        while isinstance(inner, (sqlglot_exp.Ordered, sqlglot_exp.Collate, sqlglot_exp.Paren)) and inner.this is not None:
            inner = inner.this
        if isinstance(inner, sqlglot_exp.Column) and inner.name and not inner.table:
            return inner.name, [inner.name], parsed
        refs = [c.name for c in inner.find_all(sqlglot_exp.Column) if c.name and not c.table]
        return None, refs, parsed

    def _extract_index_col_base_ident(self, col_spec: str) -> str:
        exposed_lower = {c.lower() for c in self._exposed_view_columns()}
        simple_col, refs, _ = self._parse_user_index_col_info(col_spec, exposed_lower)
        if simple_col is not None:
            return simple_col
        if refs:
            return refs[0]
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
                simple_col, referenced_cols, parsed = self._parse_user_index_col_info(col_spec, exposed_lower)
                if simple_col is not None:
                    if simple_col.lower() not in exposed_lower:
                        raise ValueError(
                            f"Index on view '{self.logical_view_name}' references unknown column "
                            f"{simple_col!r}. Available view columns: {exposed}"
                        )
                    continue
                if parsed is None:
                    raise ValueError(
                        f"Index expression {col_spec!r} on view '{self.logical_view_name}' "
                        f"could not be parsed. Available view columns: {exposed}"
                    )
                for col_ref in referenced_cols:
                    if col_ref.lower() not in exposed_lower:
                        raise ValueError(
                            f"Index expression {col_spec!r} on view '{self.logical_view_name}' "
                            f"references unknown column {col_ref!r}. Available view columns: {exposed}"
                        )
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
                            allow_qualified=False,
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
                        allow_qualified=False,
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
                    allow_qualified=False,
                )
            )
            names.append(idx_name)

        if len(auto_group_cols) > 1:
            for g_idx, g_col in enumerate(auto_group_cols[1:], start=2):
                sub_idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{_sanitize_ident(g_col)}", f"__g{g_idx}")
                if sub_idx_name not in names:
                    ddls.append(
                        build_create_index_sql(
                            sub_idx_name,
                            mv_table,
                            [self._q(g_col)],
                            dialect=self.dialect,
                            allow_qualified=False,
                        )
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
                        allow_qualified=False,
                    )
                )
                names.append(extra_idx_name)

        exposed_lower = {c.lower() for c in self._exposed_view_columns()}
        backing_map_lower = {k.lower(): v for k, v in (backing_col_map or {}).items()}
        sg_d = _sqlglot_dialect(self.dialect)

        def _rewrite_index_ast_cols(node: sqlglot_exp.Expr) -> sqlglot_exp.Expr:
            if not backing_map_lower:
                return node
            copied = node.copy()
            if isinstance(copied, sqlglot_exp.Column) and copied.name and not copied.table:
                mapped_root = backing_map_lower.get(copied.name.lower())
                if mapped_root is not None:
                    return parse_sql_ast(mapped_root, dialect=self.dialect)
            for col_n in list(copied.find_all(sqlglot_exp.Column)):
                if col_n is copied:
                    continue
                if col_n.name and not col_n.table and col_n.name.lower() in backing_map_lower:
                    repl_expr = parse_sql_ast(backing_map_lower[col_n.name.lower()], dialect=self.dialect)
                    col_n.replace(repl_expr)
            return copied

        def _explicit_nulls_clause(sql_frag: str) -> str:
            if self.dialect == Dialect.MYSQL:
                return ""
            try:
                toks = _tokenize_sql(sql_frag, read=sg_d)
            except sqlglot.errors.SqlglotError:
                return ""
            depth = 0
            for idx_t, tk in enumerate(toks):
                if tk.token_type == SQLGlotTokenType.L_PAREN:
                    depth += 1
                elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                    depth -= 1
                elif (
                    depth == 0
                    and tk.token_type != SQLGlotTokenType.IDENTIFIER
                    and tk.text.upper() == "NULLS"
                    and idx_t + 1 < len(toks)
                    and toks[idx_t + 1].token_type != SQLGlotTokenType.IDENTIFIER
                ):
                    nxt = toks[idx_t + 1].text.upper()
                    if nxt in ("FIRST", "LAST"):
                        return f" NULLS {nxt}"
            return ""

        for i, idx_spec in enumerate(self.user_indexes, 1):
            if idx_spec.name:
                raw_n = idx_spec.name
                if self.object_prefix and not raw_n.startswith(self.object_prefix):
                    idx_name = f"{self.object_prefix}{raw_n}"
                else:
                    idx_name = raw_n
                idx_name = self._bound_user_index_name(idx_name)
            else:
                col_tag = "_".join(_sanitize_ident(self._extract_index_col_base_ident(c)) for c in idx_spec.columns)
                idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{col_tag}", f"_{i}")

            mapped_cols: List[str] = []
            for col_spec in idx_spec.columns:
                c_mod = col_spec.strip()
                if c_mod.lower() in exposed_lower:
                    c_mod = quote_sql_identifier(c_mod, self.dialect, allow_qualified=False)
                explicit_nulls = _explicit_nulls_clause(c_mod)
                simple_col, _, parsed_col = self._parse_user_index_col_info(c_mod, exposed_lower)
                if parsed_col is not None and backing_map_lower:
                    parsed_col = _rewrite_index_ast_cols(parsed_col)
                    c_mod = parsed_col.sql(dialect=sg_d)
                    simple_col, _, parsed_col = self._parse_user_index_col_info(c_mod, None)
                if parsed_col is not None and simple_col is not None:
                    ord_node = parsed_col if isinstance(parsed_col, sqlglot_exp.Ordered) else None
                    inner_node = ord_node.this if ord_node is not None else parsed_col
                    while isinstance(inner_node, sqlglot_exp.Paren) and inner_node.this is not None:
                        inner_node = inner_node.this
                    collate_s = ""
                    if isinstance(inner_node, sqlglot_exp.Collate) and inner_node.expression is not None:
                        collate_s = f" COLLATE {inner_node.expression.sql(dialect=sg_d)}"
                    dir_s = ""
                    nulls_s = ""
                    if ord_node is not None:
                        desc_val = ord_node.args.get("desc")
                        if desc_val is True:
                            dir_s = " DESC"
                        elif desc_val is False:
                            dir_s = " ASC"
                        nulls_s = explicit_nulls
                    q_col = quote_sql_identifier(simple_col, self.dialect, allow_qualified=False)
                    c_mod = f"{q_col}{collate_s}{dir_s}{nulls_s}"
                elif self.dialect == Dialect.MYSQL:
                    ord_node = parsed_col if isinstance(parsed_col, sqlglot_exp.Ordered) else None
                    if ord_node is not None and ord_node.this is not None:
                        desc_val = ord_node.args.get("desc")
                        dir_s = " DESC" if desc_val is True else (" ASC" if desc_val is False else "")
                        expr_part = ord_node.this.sql(dialect=sg_d).strip()
                    else:
                        dir_s = ""
                        expr_part = c_mod
                    inner1 = _unwrap_outer_parens(expr_part)
                    has_double = inner1 != expr_part and _unwrap_outer_parens(inner1) != inner1
                    if not has_double:
                        c_mod = f"(({inner1})){dir_s}"
                    else:
                        c_mod = f"{expr_part}{dir_s}"
                elif self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                    ord_node = parsed_col if isinstance(parsed_col, sqlglot_exp.Ordered) else None
                    if ord_node is not None and ord_node.this is not None:
                        desc_val = ord_node.args.get("desc")
                        dir_s = " DESC" if desc_val is True else (" ASC" if desc_val is False else "")
                        tail_s = f"{dir_s}{explicit_nulls}"
                        expr_part = ord_node.this.sql(dialect=sg_d).strip()
                    else:
                        tail_s = ""
                        expr_part = c_mod
                    if _unwrap_outer_parens(expr_part) == expr_part:
                        c_mod = f"({expr_part}){tail_s}"
                    else:
                        c_mod = f"{expr_part}{tail_s}"
                mapped_cols.append(c_mod)

            using_method = (
                idx_spec.using_method
                if idx_spec.using_method and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
                else None
            )
            where_clause = None
            if idx_spec.where_clause and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                w_mod = idx_spec.where_clause
                if backing_map_lower:
                    try:
                        w_ast = _rewrite_index_ast_cols(parse_sql_ast(w_mod, dialect=self.dialect))
                        w_mod = w_ast.sql(dialect=sg_d)
                    except sqlglot.errors.SqlglotError:
                        for u_col, b_expr in backing_col_map.items() if backing_col_map else ():
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
                    allow_qualified=False,
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
            minmax_needs = self._minmax_group_arg_index_needs(raw_plan, by_tbl, alias_map)
            for t_n, cols in minmax_needs:
                _record(t_n, cols, "minmax_extreme_recompute")
            for t_n, g_cols in by_tbl.items():
                if any(t_n == m_t for m_t, _ in minmax_needs):
                    # (group..., arg) also serves every group-key lookup.
                    continue
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

    def _minmax_group_arg_index_needs(
        self,
        raw_plan: MaterializedViewPlan,
        group_cols_by_table: Dict[str, List[str]],
        alias_map: Dict[str, str],
    ) -> List[Tuple[str, List[str]]]:
        """`(table, group columns + argument)` indexes for single-table MIN/MAX.

        Deleting or updating a group's current extreme recomputes
        ``MIN/MAX(arg) ... WHERE <group key> = ...`` from the base table. An
        index on the group columns followed by the argument turns that into
        one index probe; an index on the group columns alone still reads the
        whole group (lane measurement, 40k rows, 100 extreme deletes in one
        transaction, net of the bare DELETE, (g) vs (g, v): SQLite 1967 -> 3
        us, PostgreSQL 18 1072 -> 519, MySQL 8.4 17060 -> ~0, MariaDB 11.8
        7765 -> 311). Only plain-column arguments and plain-column group keys
        of the single base table qualify, and only argument types whose index
        is safe to create and to keep writing to (``_minmax_index_arg_blocker``);
        other shapes keep the group-key index.
        """
        v = self.view
        if (
            raw_plan.strategy != MaintenanceStrategy.ALGEBRAIC_DELTA
            or self._alg_mode != "single"
            or len(v.joins) != 1
            or len(group_cols_by_table) != 1
        ):
            return []
        (t_n, g_cols), = group_cols_by_table.items()
        if len(g_cols) != len(v.group_by_cols):
            return []
        out: List[Tuple[str, List[str]]] = []
        for it in self._working_select_items:
            if not it.is_aggregate or it.agg_func not in (AggFunc.MIN, AggFunc.MAX) or not it.arg_expr:
                continue
            try:
                a_ast = parse_sql_ast(it.arg_expr, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            if not isinstance(a_ast, sqlglot_exp.Column):
                continue
            res = self._resolve_ast_col_to_table(a_ast, alias_map)
            if res is None or res[0] != t_n:
                continue
            if self._minmax_index_arg_blocker(t_n, g_cols, res[1]) is not None:
                continue
            cols = list(g_cols) + ([res[1]] if res[1].lower() not in {c.lower() for c in g_cols} else [])
            if cols not in [c for _, c in out]:
                out.append((t_n, cols))
        return out

    def _minmax_index_arg_blocker(self, table: str, group_cols: Sequence[str], arg_col: str) -> Optional[str]:
        """Why no ``(group..., arg)`` index is recommended for this argument,
        or None when it is safe to create and to keep writing to."""
        ts = self._schemas_lower.get(table.lower())
        if ts is None:
            return "the column types are unknown"
        defs = {k.lower(): v for k, v in ts.columns.items()}
        arg_def = defs.get(arg_col.lower())
        if arg_def is None:
            return "the column type is unknown"
        return _minmax_index_type_blocker(
            self.dialect,
            arg_def.data_type,
            [defs[g.lower()].data_type if g.lower() in defs else "" for g in group_cols],
        )

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
                allow_qualified=False,
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
                "performance_notes": self._performance_notes() + self._minmax_index_notes(base_index_recommendations) + self._minmax_unindexable_notes(),
            },
        )

    def _minmax_unindexable_notes(self) -> List[str]:
        """MIN/MAX arguments for which no (group, arg) index is recommended."""
        v = self.view
        if self._alg_mode != "single" or len(v.joins) != 1 or not v.group_by_cols:
            return []
        alias_map = self._alias_to_table_map()
        g_cols: List[str] = []
        for g in v.group_by_cols:
            try:
                g_ast = parse_sql_ast(g, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return []
            res_g = self._resolve_ast_col_to_table(g_ast, alias_map) if isinstance(g_ast, sqlglot_exp.Column) else None
            if res_g is None:
                return []
            g_cols.append(res_g[1])
        notes: List[str] = []
        for it in self._working_select_items:
            if not it.is_aggregate or it.agg_func not in (AggFunc.MIN, AggFunc.MAX) or not it.arg_expr:
                continue
            try:
                a_ast = parse_sql_ast(it.arg_expr, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                continue
            res = self._resolve_ast_col_to_table(a_ast, alias_map) if isinstance(a_ast, sqlglot_exp.Column) else None
            if res is None:
                continue
            why = self._minmax_index_arg_blocker(res[0], g_cols, res[1])
            if why is not None:
                notes.append(
                    f"minmax_extreme_recompute: no ({', '.join(g_cols + [res[1]])}) index is recommended for "
                    f"{it.raw_expr}: {why}. Deleting or updating a group's current extreme reads every row of "
                    f"the group (O(group rows)) through the group-key index; keep the group small, or "
                    f"maintain the extreme in an indexable column of the table."
                )
        return notes

    @staticmethod
    def _minmax_index_notes(base_index_recommendations: Sequence[Dict[str, Any]]) -> List[str]:
        """What to do when a MIN/MAX extreme recompute has no (group, arg) index."""
        notes: List[str] = []
        for rec in base_index_recommendations:
            if rec.get("reason") != "minmax_extreme_recompute":
                continue
            cols = ", ".join(rec.get("columns") or ())
            notes.append(
                f"minmax_extreme_recompute: deleting or updating the current MIN/MAX of a group "
                f"recomputes it from {rec.get('table')}; without an index on ({cols}) that reads "
                f"every row of the group (O(group rows)), with it it is one index probe "
                f"(O(log n)). Create it with: {rec.get('ddl')}"
            )
        return notes

    def _pgmy_native_recompute_items_for_plan(self) -> bool:
        try:
            return bool(self._pgmy_native_recompute_items())
        except Exception:
            return False

    def _fallback_diagnostics(self, strategy: MaintenanceStrategy) -> Tuple[Diagnostic, ...]:
        """Announced fallbacks of this plan (see matview.diagnostics)."""
        having = self.view.having_clause
        if not having or strategy == MaintenanceStrategy.ALGEBRAIC_DELTA or not _has_subquery_aggregate(having, self.dialect):
            return ()
        cost = (
            "a write to a table the subquery reads recomputes every group of the view; other writes "
            "recompute the touched groups."
        )
        if self.dialect == Dialect.MYSQL:
            cost = cost[:-1] + (
                "; concurrent writers to the subquery's table deadlock (error 1213) and roll back "
                "(about 45% of writes with 4 concurrent writers, measured on MySQL 8.4 and MariaDB 10.11)."
            )
        return (
            Diagnostic(
                code=HAVING_SUBQUERY_AGGREGATE_FALLBACK,
                severity=DiagnosticSeverity.INFO,
                view=self.logical_view_name,
                strategy=Strategy.GROUP_SCOPED_RECOMPUTE,
                columns=(),
                expressions=(having.strip(),),
                message=exact_fallback_message(
                    Strategy.GROUP_SCOPED_RECOMPUTE,
                    "HAVING compares with an aggregate inside a scalar subquery, which the algebraic "
                    "decomposition cannot maintain.",
                    cost,
                    "move the scalar-subquery condition out of HAVING (e.g. filter when reading the view).",
                ),
                details={"maintenance_strategy": strategy.value},
            ),
        )

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
        strategy_diagnostics = list(self._fallback_diagnostics(raw_plan.strategy))
        emit_diagnostics(strategy_diagnostics)
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
            strategy_diagnostics=strategy_diagnostics,
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
                watermark_catalog_table=self.watermark_catalog_table,
                pg_fillfactor=getattr(self, "pg_fillfactor", 80),
                repair_fence_catalog_table=self.repair_fence_catalog_table,
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
            strategy_diagnostics=[d for sp in sub_plans for d in sp.strategy_diagnostics],
        )


_PG_SAFE_INDEX_TYPE = re.compile(
    r"(?:SMALLINT|INT2|INTEGER|INT|INT4|BIGINT|INT8|SMALLSERIAL|SERIAL|BIGSERIAL|SERIAL4|SERIAL8|"
    r"REAL|FLOAT4|DOUBLE PRECISION|FLOAT8|FLOAT(?:\(\d+\))?|BOOLEAN|BOOL|DATE|"
    r"TIME(?:\(\d+\))?(?: WITH(?:OUT)? TIME ZONE)?|TIMETZ|"
    r"TIMESTAMP(?:\(\d+\))?(?: WITH(?:OUT)? TIME ZONE)?|TIMESTAMPTZ|INTERVAL(?:\(\d+\))?|"
    r"UUID|MONEY|INET|CIDR|MACADDR|MACADDR8|OID)"
)
_PG_KEY_LIMIT_BYTES = 2600  # 2704-byte btree tuple limit minus header/alignment margin


def _pg_key_bytes(data_type: str) -> Optional[int]:
    """Upper bound of a PostgreSQL btree key column in bytes, or None when the
    type's values are unbounded or long (TEXT, BYTEA, JSONB, arrays, ...).
    Character types count 4 bytes per character (UTF-8) plus the varlena
    header; NUMERIC(p) counts its digit groups."""
    t = re.sub(r"\s+", " ", data_type.strip().upper())
    if not t:
        return None
    if _PG_SAFE_INDEX_TYPE.fullmatch(t):
        if re.match(r"(?:SMALLINT|INT2|SMALLSERIAL|BOOL)", t):
            return 2
        if re.match(r"(?:INTERVAL|UUID|MACADDR8)", t):
            return 16
        if re.match(r"(?:INET|CIDR)", t):
            return 24
        return 8
    m = re.fullmatch(r"(?:NUMERIC|DECIMAL)\s*\(\s*(\d+)(?:\s*,\s*-?\d+)?\s*\)", t)
    if m and int(m.group(1)) <= 300:
        return (int(m.group(1)) + 3) // 4 * 2 + 8
    m = re.fullmatch(r"(?:CHARACTER VARYING|VARCHAR|CHARACTER|CHAR|BPCHAR)\s*\(\s*(\d+)\s*\)", t)
    if m and int(m.group(1)) <= 100:
        return int(m.group(1)) * 4 + 4
    return None


_MYSQL_INT_BYTES = {"TINYINT": 1, "SMALLINT": 2, "MEDIUMINT": 3, "INT": 4, "INTEGER": 4, "BIGINT": 8}
_MYSQL_KEY_LIMIT_BYTES = 3072  # InnoDB DYNAMIC/COMPRESSED row formats


def _mysql_key_bytes(data_type: str) -> Optional[int]:
    """Upper bound of a column's InnoDB key part in bytes (utf8mb4 assumed for
    character columns: 4 bytes per character), or None when not modelled."""
    t = data_type.strip().upper()
    base = re.sub(r"\s+(?:UNSIGNED|SIGNED|ZEROFILL)\b", "", t).strip()
    m = re.fullmatch(r"([A-Z ]+?)\s*(?:\((\d+)(?:\s*,\s*(\d+))?\))?", base)
    if not m:
        return None
    name, n = m.group(1).strip(), int(m.group(2)) if m.group(2) else None
    if name in _MYSQL_INT_BYTES:
        return _MYSQL_INT_BYTES[name]
    if name in ("DECIMAL", "NUMERIC", "DEC", "FIXED"):
        return ((n or 10) + 1) // 2 + 1
    if name in ("FLOAT", "DOUBLE", "DOUBLE PRECISION", "REAL"):
        return 8
    if name in ("DATE", "YEAR", "TIME", "DATETIME", "TIMESTAMP", "BOOL", "BOOLEAN", "BIT"):
        return 8
    if name in ("CHAR", "VARCHAR", "CHARACTER", "CHARACTER VARYING", "NCHAR", "NVARCHAR"):
        return (n if n is not None else 1) * 4 + 2
    if name in ("BINARY", "VARBINARY"):
        return (n if n is not None else 1) + 2
    return None


def _minmax_index_type_blocker(dialect: Dialect, arg_type: str, group_types: Sequence[str]) -> Optional[str]:
    """Why a ``(group..., arg)`` btree is not recommended for these types.

    * PostgreSQL: a btree row is limited to 2704 bytes; an index on a column
      whose values can be longer (TEXT, unbounded or long VARCHAR, BYTEA,
      JSONB, unconstrained NUMERIC, arrays, ...) makes later writes of long
      values fail. Only fixed-size types, NUMERIC(p <= 300) and
      CHAR/VARCHAR(n <= 100) qualify.
    * MySQL/MariaDB: character arguments compare under their collation
      (case- and, for PAD SPACE collations, trailing-space-insensitive by
      default), so an index changes which of several equal-comparing values
      the live query returns; and every key must fit InnoDB's 3072 bytes.
      Numeric, temporal and BINARY/VARBINARY arguments qualify within that
      limit.
    * SQLite: no row-size limit; every type qualifies.
    """
    t = re.sub(r"\s+", " ", arg_type.strip().upper())
    if dialect == Dialect.SQLITE:
        return None
    if dialect == Dialect.POSTGRES:
        if _pg_key_bytes(arg_type) is None:
            return (
                f"values of type {arg_type} can exceed PostgreSQL's 2704-byte btree row limit, so writing a "
                f"long value after creating the index would fail"
            )
        total = 0
        for ty in [arg_type, *group_types]:
            b = _pg_key_bytes(ty)
            if b is None:
                return (
                    f"group key values of type {ty or 'unknown'} can make the index row exceed PostgreSQL's "
                    f"2704-byte btree limit"
                )
            total += b
        if total > _PG_KEY_LIMIT_BYTES:
            return (
                f"an index row would take up to {total} bytes, more than PostgreSQL's btree limit "
                f"(2704 bytes including the row header)"
            )
        return None
    if re.match(r"(?:CHAR|VARCHAR|CHARACTER|NCHAR|NVARCHAR|TINYTEXT|TEXT|MEDIUMTEXT|LONGTEXT|ENUM|SET|JSON)\b", t):
        return (
            f"{arg_type} values compare under the column's collation (case- and trailing-space-insensitive "
            f"by default), so an index would change which of several equal values the live query returns"
        )
    if re.match(r"(?:TINYBLOB|BLOB|MEDIUMBLOB|LONGBLOB|GEOMETRY|POINT)\b", t):
        return f"{arg_type} columns cannot be indexed without a prefix length"
    total = 0
    for ty in [arg_type, *group_types]:
        b = _mysql_key_bytes(ty)
        if b is None:
            return f"the key length of type {ty or 'unknown'} is not modelled"
        total += b
    if total > _MYSQL_KEY_LIMIT_BYTES:
        return f"the key would take up to {total} bytes, more than InnoDB's {_MYSQL_KEY_LIMIT_BYTES}-byte limit"
    return None
