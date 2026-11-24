"""Tier 2: Group-scoped delta recomputation rules."""

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
    parse_sql_ast,
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

    def _tier2_subquery_tables(self) -> Set[str]:
        """Return lowercased table names referenced inside any subquery in
        WHERE, HAVING, SELECT expressions, or JOIN ON conditions."""
        clauses: List[str] = []
        if self.view.where_clause:
            clauses.append(self.view.where_clause)
        if self.view.having_clause:
            clauses.append(self.view.having_clause)
        for it in self.view.select_items:
            if it.raw_expr:
                clauses.append(it.raw_expr)
        for j in self.view.joins:
            if j.on_condition:
                clauses.append(j.on_condition)
        all_tables = {j.table_name.lower() for j in self.view.joins} | {self.view.base_table.lower()}
        subq_tables: Set[str] = set()
        for clause in clauses:
            try:
                ast = parse_sql_ast(clause, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return all_tables
            for subq in ast.find_all(sqlglot_exp.Select, sqlglot_exp.Subquery):
                for tbl in subq.find_all(sqlglot_exp.Table):
                    if tbl.name:
                        subq_tables.add(tbl.name.lower())
        return subq_tables

    def _tier2_ident_expr(
        self,
        name: str,
        original_ident: Optional[sqlglot_exp.Identifier] = None,
    ) -> sqlglot_exp.Identifier:
        if isinstance(original_ident, sqlglot_exp.Identifier) and original_ident.quoted:
            return sqlglot_exp.to_identifier(original_ident.name, quoted=True)
        needs_q = (
            name.upper() in _SQL_RESERVED_KEYWORDS
            or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)
        )
        return sqlglot_exp.to_identifier(name, quoted=needs_q)

    def _tier2_raw_join_metadata(
        self,
    ) -> Tuple[List[str], List[sqlglot_exp.Identifier]]:
        """Extract raw table SQL (preserving schema qualification and quoting)
        and exact outer alias Identifiers for each join from ``raw_select_sql``."""
        sg_d = _sqlglot_dialect(self.dialect)
        fallback_tables = [self._q(j.table_name) for j in self.view.joins]
        fallback_aliases = [self._tier2_ident_expr(j.alias) for j in self.view.joins]
        try:
            q_ast = parse_sql_ast(self.view.raw_select_sql.strip().rstrip(";"), dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return fallback_tables, fallback_aliases
        if not isinstance(q_ast, sqlglot_exp.Select):
            return fallback_tables, fallback_aliases
        from_node = q_ast.args.get("from_") or q_ast.args.get("from")
        if from_node is None or not isinstance(from_node.this, sqlglot_exp.Table):
            return fallback_tables, fallback_aliases
        tbl_nodes: List[sqlglot_exp.Table] = [from_node.this]
        for j_node in q_ast.args.get("joins") or ():
            if isinstance(j_node.this, sqlglot_exp.Table):
                tbl_nodes.append(j_node.this)
            else:
                return fallback_tables, fallback_aliases
        if len(tbl_nodes) != len(self.view.joins):
            return fallback_tables, fallback_aliases
        raw_tables: List[str] = []
        outer_aliases: List[sqlglot_exp.Identifier] = []
        for idx, tbl_node in enumerate(tbl_nodes):
            bare_tbl = tbl_node.copy()
            bare_tbl.set("alias", None)
            raw_tables.append(bare_tbl.sql(dialect=sg_d))
            alias_arg = tbl_node.args.get("alias")
            if isinstance(alias_arg, sqlglot_exp.TableAlias) and isinstance(alias_arg.this, sqlglot_exp.Identifier):
                outer_aliases.append(alias_arg.this.copy())
            elif isinstance(tbl_node.this, sqlglot_exp.Identifier):
                outer_aliases.append(tbl_node.this.copy())
            else:
                outer_aliases.append(fallback_aliases[idx])
        return raw_tables, outer_aliases

    def _tier2_typed_null_node(
        self,
        col_def: Optional[ColumnDef],
        *,
        allow_untyped_null: bool = True,
    ) -> Optional[sqlglot_exp.Expr]:
        if self.dialect != Dialect.POSTGRES or col_def is None:
            return sqlglot_exp.Null()
        dt = self._normalize_dialect_type(col_def.data_type).strip()
        m_coll = re.search(
            r"(\s+COLLATE\s+(?:(?:\"[^\"]+\"|\S+)\.)?(?:\"[^\"]+\"|\S+))\s*$",
            dt,
            re.IGNORECASE,
        )
        if m_coll:
            base_dt = dt[: m_coll.start()].strip()
            coll_tail = m_coll.group(1)
        else:
            base_dt = dt
            coll_tail = ""
        if not base_dt or base_dt.upper() in ("ARRAY", "USER-DEFINED"):
            return sqlglot_exp.Null() if allow_untyped_null else None
        try:
            return parse_sql_ast(f"CAST(NULL AS {base_dt}){coll_tail}", dialect=Dialect.POSTGRES)
        except sqlglot.errors.SqlglotError:
            return sqlglot_exp.Null() if allow_untyped_null else None

    def _tier2_join_graph_info(
        self,
    ) -> Tuple[
        Dict[str, int],
        Dict[str, int],
        List[Optional[TableSchema]],
        Set[int],
        bool,
        bool,
    ]:
        """Build join-graph lookup maps and outer-join nullability metadata."""
        tbl_counts: Dict[str, int] = {}
        for j in self.view.joins:
            t_low = j.table_name.lower()
            tbl_counts[t_low] = tbl_counts.get(t_low, 0) + 1

        join_quals: Dict[str, int] = {}
        schemas_by_idx: List[Optional[TableSchema]] = []
        col_owners: Dict[str, List[int]] = {}
        nullable_join_indices: Set[int] = set()
        has_fk_cascade = False
        has_natural_join = False

        for idx, j in enumerate(self.view.joins):
            join_quals[j.alias.lower()] = idx
            if tbl_counts.get(j.table_name.lower(), 0) == 1:
                join_quals.setdefault(j.table_name.lower(), idx)
            ts_j = self._schemas_lower.get(j.table_name.lower())
            schemas_by_idx.append(ts_j)
            if ts_j:
                if ts_j.fk_cascade_actions:
                    has_fk_cascade = True
                for c_name in ts_j.columns:
                    col_owners.setdefault(c_name.lower(), []).append(idx)
            jt_up = j.join_type.upper()
            if "NATURAL" in jt_up:
                has_natural_join = True
            if "LEFT" in jt_up:
                nullable_join_indices.add(idx)
            elif "RIGHT" in jt_up:
                nullable_join_indices.update(range(idx))
            elif "FULL" in jt_up:
                nullable_join_indices.update(range(idx + 1))

        unqual_col_to_join = {
            c_low: idxs[0] for c_low, idxs in col_owners.items() if len(idxs) == 1
        }
        return (
            join_quals,
            unqual_col_to_join,
            schemas_by_idx,
            nullable_join_indices,
            has_fk_cascade,
            has_natural_join,
        )

    def _tier2_resolve_col(
        self,
        col: sqlglot_exp.Column,
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Optional[Tuple[int, ColumnDef]]:
        col_low = col.name.lower()
        if not col_low:
            return None
        if col.table:
            j_idx = join_quals.get(col.table.lower())
            if j_idx is None:
                return None
        else:
            j_idx = unqual_col_to_join.get(col_low)
            if j_idx is None:
                return None
        ts_j = schemas_by_idx[j_idx]
        if ts_j is None:
            return None
        for c_def in ts_j.columns.values():
            if c_def.name.lower() == col_low:
                return j_idx, c_def
        return None

    def _tier2_on_strict_null_rejecting(
        self,
        on_sql: str,
        nullified: Set[int],
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Tuple[Optional[Set[int]], bool]:
        """Return ``(referenced_join_indices, is_strict_null_rejecting)`` for an ON clause."""
        try:
            ast = parse_sql_ast(on_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None, False
        if ast.find(sqlglot_exp.Select) is not None or ast.find(sqlglot_exp.Subquery) is not None:
            return None, False

        refs: Set[int] = set()
        for col in ast.find_all(sqlglot_exp.Column):
            resolved = self._tier2_resolve_col(col, join_quals, unqual_col_to_join, schemas_by_idx)
            if resolved is None:
                return None, False
            refs.add(resolved[0])

        def _strict_null_prop(node: Optional[sqlglot_exp.Expr]) -> bool:
            if node is None:
                return False
            if isinstance(node, sqlglot_exp.Column):
                res = self._tier2_resolve_col(node, join_quals, unqual_col_to_join, schemas_by_idx)
                return res is not None and res[0] in nullified
            if isinstance(node, (sqlglot_exp.Paren, sqlglot_exp.Neg, sqlglot_exp.Cast)):
                return _strict_null_prop(node.this)
            if isinstance(
                node,
                (
                    sqlglot_exp.Add,
                    sqlglot_exp.Sub,
                    sqlglot_exp.Mul,
                    sqlglot_exp.Div,
                    sqlglot_exp.Mod,
                ),
            ):
                return _strict_null_prop(node.this) or _strict_null_prop(node.expression)
            return False

        conjuncts: List[sqlglot_exp.Expr] = []
        stack: List[sqlglot_exp.Expr] = [ast]
        while stack:
            cur = stack.pop()
            if isinstance(cur, sqlglot_exp.Paren) and cur.this is not None:
                stack.append(cur.this)
            elif isinstance(cur, sqlglot_exp.And):
                if cur.this is not None:
                    stack.append(cur.this)
                if cur.expression is not None:
                    stack.append(cur.expression)
            else:
                conjuncts.append(cur)

        strict_cmp_types = (
            sqlglot_exp.EQ,
            sqlglot_exp.NEQ,
            sqlglot_exp.LT,
            sqlglot_exp.LTE,
            sqlglot_exp.GT,
            sqlglot_exp.GTE,
        )
        is_strict = any(
            isinstance(c, strict_cmp_types)
            and (_strict_null_prop(c.this) or _strict_null_prop(c.expression))
            for c in conjuncts
        )
        return refs, is_strict

    def _tier2_nullified_join_indices(
        self,
        target_idx: int,
        nullable_join_indices: Set[int],
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Optional[Set[int]]:
        """Compute the exact set of join indices that become NULL-extended when
        ``self.view.joins[target_idx]`` stops matching, or ``None`` if scoping
        must fail closed to a full view refresh."""
        # Fail closed on 3+-table FULL OUTER JOINs (two-sided nullification).
        if len(self.view.joins) > 2 and any("FULL" in j.join_type.upper() for j in self.view.joins):
            return None
        own_on = self.view.joins[target_idx].on_condition
        if own_on:
            try:
                own_ast = parse_sql_ast(own_on, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                return None
            if own_ast.find(sqlglot_exp.Select) is not None or own_ast.find(sqlglot_exp.Subquery) is not None:
                return None
        if target_idx not in nullable_join_indices:
            return set()
        target_jt = self.view.joins[target_idx].join_type.upper()
        # A LEFT or FULL join at target_idx is immediately nullified at target_idx;
        # a FROM/INNER/CROSS/RIGHT join is only nullified once a subsequent RIGHT or
        # FULL join null-extends its left subtree (0..k-1).
        immediately_nullified = "LEFT" in target_jt or "FULL" in target_jt
        nullified: Set[int] = {target_idx} if immediately_nullified else set()
        eliminates_left_subtree = "LEFT" not in target_jt
        for k in range(target_idx + 1, len(self.view.joins)):
            j_k = self.view.joins[k]
            jt_k = j_k.join_type.upper()
            on_refs: Optional[Set[int]] = set()
            is_strict = False
            if j_k.on_condition:
                check_nulls = nullified | ({target_idx} if "FULL" in jt_k and len(self.view.joins) == 2 else set())
                on_refs, is_strict = self._tier2_on_strict_null_rejecting(
                    j_k.on_condition, check_nulls, join_quals, unqual_col_to_join, schemas_by_idx
                )
                if on_refs is None:
                    return None
            touches_nullified = bool(on_refs & nullified)
            if "FULL" in jt_k:
                # 2-table `o FULL OUTER JOIN p`: modifying o nullifies only o.
                if len(self.view.joins) == 2 and target_idx == 0 and k == 1 and is_strict:
                    nullified.add(0)
                    continue
                return None
            if "RIGHT" in jt_k:
                if touches_nullified and not is_strict:
                    return None
                if touches_nullified or eliminates_left_subtree:
                    nullified.update(range(k))
                eliminates_left_subtree = False
                continue
            if not touches_nullified:
                continue
            if not is_strict:
                return None
            if "LEFT" in jt_k:
                nullified.add(k)
            else:
                # INNER JOIN depending on a nullified table can change match set.
                return None
        return nullified

    def _tier2_ast_qualify_outer(
        self,
        expr: str,
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
        outer_aliases: Sequence[sqlglot_exp.Identifier],
    ) -> Optional[str]:
        """Qualify all column references in ``expr`` against the outer join aliases."""
        sg_d = _sqlglot_dialect(self.dialect)
        try:
            ast = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        failed = False

        def _tx(node: sqlglot_exp.Expr) -> sqlglot_exp.Expr:
            nonlocal failed
            if isinstance(node, sqlglot_exp.Column):
                resolved = self._tier2_resolve_col(node, join_quals, unqual_col_to_join, schemas_by_idx)
                if resolved is None:
                    failed = True
                    return node
                j_idx, c_def = resolved
                orig_col = node.this if isinstance(node.this, sqlglot_exp.Identifier) else None
                return sqlglot_exp.Column(
                    this=self._tier2_ident_expr(c_def.name, orig_col),
                    table=outer_aliases[j_idx].copy(),
                )
            return node

        out_ast = ast.transform(_tx)
        if failed:
            return None
        return out_ast.sql(dialect=sg_d)

    def _tier2_ast_rewrite_single_join(
        self,
        expr: str,
        target_idx: int,
        rec_prefix: str,
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Optional[str]:
        """Rewrite ``expr`` against ``rec_prefix`` if every column reference
        resolves to ``self.view.joins[target_idx]``."""
        sg_d = _sqlglot_dialect(self.dialect)
        try:
            ast = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        saw_col = False
        failed = False

        def _tx(node: sqlglot_exp.Expr) -> sqlglot_exp.Expr:
            nonlocal saw_col, failed
            if isinstance(node, sqlglot_exp.Column):
                resolved = self._tier2_resolve_col(node, join_quals, unqual_col_to_join, schemas_by_idx)
                if resolved is None or resolved[0] != target_idx:
                    failed = True
                    return node
                saw_col = True
                _, c_def = resolved
                orig_col = node.this if isinstance(node.this, sqlglot_exp.Identifier) else None
                return sqlglot_exp.Column(
                    this=self._tier2_ident_expr(c_def.name, orig_col),
                    table=sqlglot_exp.to_identifier(rec_prefix, quoted=False),
                )
            return node

        out_ast = ast.transform(_tx)
        if failed or not saw_col:
            return None
        return out_ast.sql(dialect=sg_d)

    def _tier2_ast_remap_and_nullify(
        self,
        expr: str,
        alias_by_idx: Optional[Dict[int, str]],
        nullified_indices: Set[int],
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Optional[str]:
        """AST-remap column references to ``alias_by_idx`` and replace columns
        from ``nullified_indices`` with (typed) NULL."""
        sg_d = _sqlglot_dialect(self.dialect)
        try:
            ast = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        is_bare_col = isinstance(ast, sqlglot_exp.Column)
        failed = False

        def _tx(node: sqlglot_exp.Expr) -> sqlglot_exp.Expr:
            nonlocal failed
            if isinstance(node, sqlglot_exp.Column):
                resolved = self._tier2_resolve_col(node, join_quals, unqual_col_to_join, schemas_by_idx)
                if resolved is None:
                    failed = True
                    return node
                j_idx, c_def = resolved
                if j_idx in nullified_indices:
                    tn = self._tier2_typed_null_node(c_def, allow_untyped_null=is_bare_col)
                    if tn is None:
                        failed = True
                        return node
                    return tn
                if alias_by_idx is not None:
                    orig_col = node.this if isinstance(node.this, sqlglot_exp.Identifier) else None
                    return sqlglot_exp.Column(
                        this=self._tier2_ident_expr(c_def.name, orig_col),
                        table=sqlglot_exp.to_identifier(alias_by_idx[j_idx], quoted=False),
                    )
            return node

        out_ast = ast.transform(_tx)
        if failed:
            return None
        return out_ast.sql(dialect=sg_d)

    def _tier2_build_inner_from_sql(
        self,
        target_idx: int,
        inline_row_table: str,
        alias_by_idx: Dict[int, str],
        raw_tables: Sequence[str],
        join_quals: Dict[str, int],
        unqual_col_to_join: Dict[str, int],
        schemas_by_idx: List[Optional[TableSchema]],
    ) -> Optional[str]:
        """Build the join graph with ``self.view.joins[target_idx]`` replaced by
        ``inline_row_table`` and all ON clauses remapped to ``alias_by_idx``."""
        is_two_table_full = (
            len(self.view.joins) == 2
            and "FULL" in self.view.joins[1].join_type.upper()
        )
        join_parts: List[str] = []
        for idx, j in enumerate(self.view.joins):
            remapped_alias = alias_by_idx[idx]
            tbl_expr = (
                inline_row_table
                if idx == target_idx
                else f"{raw_tables[idx]} AS {remapped_alias}"
            )
            if idx == 0:
                join_parts.append(tbl_expr)
            else:
                on_sql = ""
                if j.on_condition:
                    remapped_on = self._tier2_ast_remap_and_nullify(
                        j.on_condition,
                        alias_by_idx,
                        set(),
                        join_quals,
                        unqual_col_to_join,
                        schemas_by_idx,
                    )
                    if remapped_on is None:
                        return None
                    on_sql = f" ON {remapped_on}"
                jt_up = j.join_type.upper()
                if is_two_table_full and idx == 1:
                    eff_join_type = "LEFT JOIN" if target_idx == 0 else "RIGHT JOIN"
                elif idx == target_idx and j.on_condition and "RIGHT" not in jt_up and "FULL" not in jt_up:
                    eff_join_type = "INNER JOIN"
                else:
                    eff_join_type = j.join_type
                join_parts.append(f"{eff_join_type} {tbl_expr}{on_sql}")
        return " ".join(join_parts)

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
        if base_t.lower() in self._tier2_subquery_tables():
            return None

        if len(self.view.joins) == 1:
            def _single_lhs_expr(g_expr: str, mv_col: str) -> str:
                if lhs_table_alias:
                    return f"{lhs_table_alias}.{self._q(mv_col)}"
                if record_prefix in ("__mv_new_rows", "__mv_old_rows") and self.view.joins:
                    outer_alias = self._q(self.view.joins[0].alias or base_t)
                    return f"({self._rewrite_expr_for_record(g_expr, outer_alias)})"
                return f"({g_expr})"

            conds: List[str] = []
            for g_expr, mv_col in group_mv_cols:
                rhs_e = self._rewrite_expr_for_record(g_expr, record_prefix)
                conds.append(self._null_safe_eq(_single_lhs_expr(g_expr, mv_col), f"({rhs_e})", is_nullable=True))
            cond_str = " AND ".join(conds)
            if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                return f"EXISTS (SELECT 1 FROM {record_prefix} WHERE {cond_str})"
            return cond_str

        matching_joins = [
            (idx, j)
            for idx, j in enumerate(self.view.joins)
            if j.table_name.lower() == base_t.lower()
        ]
        if not matching_joins:
            return None

        (
            join_quals,
            unqual_col_to_join,
            schemas_by_idx,
            nullable_join_indices,
            has_fk_cascade,
            has_natural_join,
        ) = self._tier2_join_graph_info()
        if has_natural_join:
            return None

        raw_tables, outer_aliases = self._tier2_raw_join_metadata()

        lhs_exprs: List[str] = []
        for g_expr, mv_col in group_mv_cols:
            if lhs_table_alias:
                lhs_exprs.append(f"{lhs_table_alias}.{self._q(mv_col)}")
            else:
                qual_g = self._tier2_ast_qualify_outer(
                    g_expr, join_quals, unqual_col_to_join, schemas_by_idx, outer_aliases
                )
                if qual_g is None:
                    return None
                lhs_exprs.append(f"({qual_g})")

        ts_target = self._schemas_lower[base_t.lower()]
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        rec_proj = ", ".join(f"{record_prefix}.{self._q(c)} AS {self._q(c)}" for c in ts_target.columns)
        alias_by_idx = {idx: f"__mv_j_{idx}" for idx in range(len(self.view.joins))}

        occurrence_preds: List[str] = []
        for target_idx, _target_join in matching_joins:
            nullified_indices = self._tier2_nullified_join_indices(
                target_idx, nullable_join_indices, join_quals, unqual_col_to_join, schemas_by_idx
            )
            if nullified_indices is None:
                return None

            # If this join occurrence owns all group expressions directly, compare
            # directly against OLD/NEW (plus the NULL-extended group key when
            # this join is on the nullable side of an outer join).
            direct_conds: List[str] = []
            all_owned = True
            for (g_expr, _mv_col), lhs_e in zip(group_mv_cols, lhs_exprs):
                rewritten = self._tier2_ast_rewrite_single_join(
                    g_expr, target_idx, record_prefix, join_quals, unqual_col_to_join, schemas_by_idx
                )
                if rewritten is None:
                    all_owned = False
                    break
                direct_conds.append(self._null_safe_eq(lhs_e, f"({rewritten})", is_nullable=True))

            if all_owned and direct_conds:
                cond_str = " AND ".join(direct_conds)
                if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                    matched_pred = f"EXISTS (SELECT 1 FROM {record_prefix} WHERE {cond_str})"
                else:
                    matched_pred = cond_str
                if nullified_indices:
                    unmatched_conds: List[str] = []
                    for (g_expr, _mv_col), lhs_e in zip(group_mv_cols, lhs_exprs):
                        unmatched_e = self._tier2_ast_remap_and_nullify(
                            g_expr, None, nullified_indices, join_quals, unqual_col_to_join, schemas_by_idx
                        )
                        if unmatched_e is None:
                            return None
                        unmatched_conds.append(
                            self._null_safe_eq(lhs_e, f"({unmatched_e})", is_nullable=True)
                        )
                    unmatched_pred = " AND ".join(unmatched_conds)
                    if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                        unmatched_pred = f"(EXISTS (SELECT 1 FROM {record_prefix}) AND {unmatched_pred})"
                    occurrence_preds.append(f"(({matched_pred}) OR ({unmatched_pred}))")
                else:
                    occurrence_preds.append(matched_pred)
                continue

            # When FK cascades can mutate joined tables before triggers see the
            # pre-cascade join pair, or when self-joins require correlating two
            # aliases against OLD/NEW simultaneously, fall back safely to full
            # refresh (all_owned keys do not join other tables and stay scoped).
            if has_fk_cascade or len(matching_joins) > 1:
                return None

            if record_prefix in ("__mv_new_rows", "__mv_old_rows"):
                inline_row_table = f"{record_prefix} AS {alias_by_idx[target_idx]}"
            else:
                inline_row_table = f"(SELECT {rec_proj}{from_dual}) AS {alias_by_idx[target_idx]}"

            from_join_sql = self._tier2_build_inner_from_sql(
                target_idx,
                inline_row_table,
                alias_by_idx,
                raw_tables,
                join_quals,
                unqual_col_to_join,
                schemas_by_idx,
            )
            if from_join_sql is None:
                return None

            match_conds: List[str] = []
            for (g_expr, _mv_col), lhs_e in zip(group_mv_cols, lhs_exprs):
                remapped_g = self._tier2_ast_remap_and_nullify(
                    g_expr, alias_by_idx, set(), join_quals, unqual_col_to_join, schemas_by_idx
                )
                if remapped_g is None:
                    return None
                eq_cond = self._null_safe_eq(lhs_e, f"({remapped_g})", is_nullable=True)
                if nullified_indices:
                    unmatched_g = self._tier2_ast_remap_and_nullify(
                        g_expr, alias_by_idx, nullified_indices, join_quals, unqual_col_to_join, schemas_by_idx
                    )
                    if unmatched_g is None:
                        return None
                    if unmatched_g != remapped_g:
                        unmatched_eq = self._null_safe_eq(lhs_e, f"({unmatched_g})", is_nullable=True)
                        match_conds.append(f"({eq_cond} OR {unmatched_eq})")
                    else:
                        match_conds.append(eq_cond)
                else:
                    match_conds.append(eq_cond)

            occurrence_preds.append(
                f"EXISTS (\n"
                f"        SELECT 1 FROM {from_join_sql}\n"
                f"        WHERE {' AND '.join(match_conds)}\n"
                f"    )"
            )

        if len(occurrence_preds) == 1:
            return occurrence_preds[0]
        return "(" + " OR ".join(f"({p})" for p in occurrence_preds) + ")"

    def _build_tier2_triggers_for_table(
        self,
        view_name: str,
        mv_table: str,
        base_t: str,
        target_cols: List[str],
        group_mv_cols: List[Tuple[str, str]],
        source_query_sql: str,
        group_col_types: Optional[Dict[str, str]] = None,
        uses_group_buckets: bool = False,
        pg_bracket_source_bit: Optional[int] = None,
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
        pg_bracket: Optional[Any] = None
        pg_trunc_stmts = full_refresh
        if self.dialect == Dialect.POSTGRES:
            from ..dialects.pg_statement import PgStatementBracket

            pg_bodies = self._pg_group_scoped_bodies(
                base_t, mv_table, cols_csv, src_cols_csv, group_mv_cols, source_query_sql,
                can_push_down, full_refresh, group_col_types or {},
                uses_group_buckets=uses_group_buckets,
            )
            pg_ai_body, pg_au_body, pg_ad_body = pg_bodies["INSERT"], pg_bodies["UPDATE"], pg_bodies["DELETE"]
            # TRUNCATE recomputes everything: lock all buckets when group-scoped
            # writers exist, otherwise only row 0.
            pg_trunc_stmts = [self._pg_view_lock_stmt(all_buckets=uses_group_buckets), *full_refresh]
            if pg_bracket_source_bit is not None:
                pg_bracket = PgStatementBracket(
                    lock_table=self._pg_view_lock_table(),
                    source_bit=pg_bracket_source_bit,
                    overlap_refresh_sql="\n".join(pg_trunc_stmts),
                    overlap_lock_sql=self._pg_view_lock_stmt(all_buckets=uses_group_buckets),
                )

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
            pg_bracket=pg_bracket,
        )


    def _pg_group_scope_keys(
        self, base_t: str, group_mv_cols: List[Tuple[str, str]], signed_sql: str
    ) -> Optional[Tuple[str, List[str]]]:
        """``(FROM clause, key expressions)`` listing the group keys a statement
        on ``base_t`` affects (PostgreSQL), or ``None`` when they cannot be
        derived from the delta (the caller recomputes the whole view).

        Mirrors ``_group_scope_predicate_for_table``: single-source views and
        keys owned by ``base_t`` read them from the signed rows; otherwise the
        signed rows replace ``base_t`` in the view's join graph. An outer join
        whose nullable side includes ``base_t`` also emits the NULL-extended
        group keys (including ``COALESCE``/expression keys and chained outer
        joins).
        """
        if not group_mv_cols:
            return None
        if len(self.view.joins) <= 1 and base_t.lower() != self.view.base_table.lower():
            return None
        if base_t.lower() in self._tier2_subquery_tables():
            return None
        rec = "__mv_d"
        signed_from = f"(\n        {signed_sql}\n        ) AS {rec}"
        if len(self.view.joins) <= 1:
            return signed_from, [self._rewrite_expr_for_record(g, rec) for g, _ in group_mv_cols]

        matching_joins = [
            (idx, j)
            for idx, j in enumerate(self.view.joins)
            if j.table_name.lower() == base_t.lower()
        ]
        if not matching_joins:
            return None

        (
            join_quals,
            unqual_col_to_join,
            schemas_by_idx,
            nullable_join_indices,
            has_fk_cascade,
            has_natural_join,
        ) = self._tier2_join_graph_info()
        if has_natural_join:
            return None

        alias_by_idx = {idx: f"__mv_j_{idx}" for idx in range(len(self.view.joins))}
        raw_tables, _outer_aliases = self._tier2_raw_join_metadata()
        key_branches: List[str] = []

        for target_idx, _target_join in matching_joins:
            nullified_indices = self._tier2_nullified_join_indices(
                target_idx, nullable_join_indices, join_quals, unqual_col_to_join, schemas_by_idx
            )
            if nullified_indices is None:
                return None

            owned_exprs: List[str] = []
            all_owned = True
            for g_expr, _mv_col in group_mv_cols:
                rewritten = self._tier2_ast_rewrite_single_join(
                    g_expr, target_idx, rec, join_quals, unqual_col_to_join, schemas_by_idx
                )
                if rewritten is None:
                    all_owned = False
                    break
                owned_exprs.append(rewritten)

            if all_owned and owned_exprs:
                sel_owned = ", ".join(f"({k}) AS __mv_k{i}" for i, k in enumerate(owned_exprs))
                key_branches.append(f"SELECT {sel_owned} FROM {signed_from}")
                if nullified_indices:
                    unmatched_exprs: List[str] = []
                    for g_expr, _mv_col in group_mv_cols:
                        u_e = self._tier2_ast_remap_and_nullify(
                            g_expr, None, nullified_indices, join_quals, unqual_col_to_join, schemas_by_idx
                        )
                        if u_e is None:
                            return None
                        unmatched_exprs.append(u_e)
                    sel_unmatched = ", ".join(
                        f"({k}) AS __mv_k{i}" for i, k in enumerate(unmatched_exprs)
                    )
                    key_branches.append(
                        f"SELECT {sel_unmatched} WHERE EXISTS (SELECT 1 FROM {signed_from})"
                    )
                continue

            if has_fk_cascade or len(matching_joins) > 1:
                return None

            inline_row_table = f"(\n        {signed_sql}\n        ) AS {alias_by_idx[target_idx]}"
            from_join_sql = self._tier2_build_inner_from_sql(
                target_idx,
                inline_row_table,
                alias_by_idx,
                raw_tables,
                join_quals,
                unqual_col_to_join,
                schemas_by_idx,
            )
            if from_join_sql is None:
                return None

            matched_exprs: List[str] = []
            unmatched_exprs = []
            any_diff = False
            for g_expr, _mv_col in group_mv_cols:
                remapped_g = self._tier2_ast_remap_and_nullify(
                    g_expr, alias_by_idx, set(), join_quals, unqual_col_to_join, schemas_by_idx
                )
                if remapped_g is None:
                    return None
                matched_exprs.append(remapped_g)
                if nullified_indices:
                    u_g = self._tier2_ast_remap_and_nullify(
                        g_expr, alias_by_idx, nullified_indices, join_quals, unqual_col_to_join, schemas_by_idx
                    )
                    if u_g is None:
                        return None
                    unmatched_exprs.append(u_g)
                    if u_g != remapped_g:
                        any_diff = True

            sel_matched = ", ".join(f"({k}) AS __mv_k{i}" for i, k in enumerate(matched_exprs))
            key_branches.append(f"SELECT {sel_matched} FROM {from_join_sql}")
            if nullified_indices and any_diff:
                sel_unmatched = ", ".join(
                    f"({k}) AS __mv_k{i}" for i, k in enumerate(unmatched_exprs)
                )
                key_branches.append(f"SELECT {sel_unmatched} FROM {from_join_sql}")

        if not key_branches:
            return None
        from_sql = "(\n        " + "\n        UNION ALL\n        ".join(key_branches) + "\n        ) AS __mv_kk"
        return from_sql, [f"__mv_kk.__mv_k{i}" for i in range(len(group_mv_cols))]

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
        group_col_types: Optional[Dict[str, str]] = None,
        uses_group_buckets: bool = False,
    ) -> Dict[str, str]:
        """PostgreSQL statement-level bodies for group-scoped recompute.

        1. Single-source group-scoped writers lock only the bounded hash
           buckets (``1..PG_GROUP_LOCK_BUCKETS``) of their affected groups in
           ascending bucket-id order, with no ``FOR SHARE`` lock on row 0
           (eliminating MultiXactId allocations and bounding the lock table at
           ``PG_GROUP_LOCK_BUCKETS + 1`` rows). Whole-view recomputes on such a
           view (``TRUNCATE`` or a source whose affected groups cannot be
           derived) lock all rows ``0..PG_GROUP_LOCK_BUCKETS`` in ascending
           order, symmetrically excluding group writers under READ COMMITTED,
           REPEATABLE READ, and SERIALIZABLE.
        2. Delete and re-insert the affected groups in later statements,
           matching keys with index-usable predicates: ``col = ANY(ARRAY(...))``
           for scalar keys without NULLs (``IN (SELECT ...)`` for array /
           user-defined types that PostgreSQL forbids in ``ARRAY(...)``), and a
           separate statement for keys with a NULL part, guarded so it does no
           work when there is none.
        """
        bodies: Dict[str, str] = {}
        refresh = "\n".join([self._pg_view_lock_stmt(all_buckets=uses_group_buckets), *full_refresh])
        lock_t = self._pg_view_lock_table()
        col_types_map = group_col_types or {}
        src_lhs_push: Optional[List[str]] = None
        if len(self.view.joins) > 1:
            jq_pg, uq_pg, sb_pg, _, _, _ = self._tier2_join_graph_info()
            _, outer_aliases_pg = self._tier2_raw_join_metadata()
            qual_list = [
                self._tier2_ast_qualify_outer(g, jq_pg, uq_pg, sb_pg, outer_aliases_pg)
                for g, _ in group_mv_cols
            ]
            if all(q is not None for q in qual_list):
                src_lhs_push = [f"({q})" for q in qual_list if q is not None]
        else:
            src_lhs_push = [f"({g})" for g, _ in group_mv_cols]
        if can_push_down and src_lhs_push is None:
            return {event: refresh for event in ("INSERT", "UPDATE", "DELETE")}

        key_uses_in = [
            ("[]" in dt or dt.strip().upper() in ("ARRAY", "USER-DEFINED"))
            for dt in (col_types_map.get(mv_c.lower(), "") for _g, mv_c in group_mv_cols)
        ]

        for event in ("INSERT", "UPDATE", "DELETE"):
            got = self._pg_group_scope_keys(base_t, group_mv_cols, self._pg_signed_rows_sql(event, base_t))
            if got is None:
                bodies[event] = refresh
                continue
            from_sql, keys = got
            n = len(keys)
            sel = ", ".join(f"{k} AS __mv_k{i}" for i, k in enumerate(keys))
            kset = f"(SELECT DISTINCT {sel} FROM {from_sql}) AS __mv_kset"
            kcols = [f"__mv_kset.__mv_k{i}" for i in range(n)]
            no_null = " AND ".join(f"{k} IS DISTINCT FROM NULL" for k in kcols)
            any_null = " OR ".join(f"{k} IS NOT DISTINCT FROM NULL" for k in kcols)
            if len(self.view.joins) > 1:
                # Multi-source join view: acquire the per-view lock row exclusively
                # before evaluating kset against other sources, so concurrent writers
                # to different sources serialize and never compute an empty/stale kset
                # before the other source commits (while still keeping DELETE/INSERT
                # scoped to the affected groups in kset).
                stmts = [self._pg_view_lock_stmt()]
            else:
                canon = self._canonical_group_pk_expr(kcols)
                stmts = [
                    f"INSERT INTO {lock_t} (__mv_id, __mv_n)\n"
                    f"    SELECT DISTINCT {self._pg_group_lock_id_sql(canon)}, 1 FROM {kset}\n"
                    f"    ORDER BY 1\n"
                    f"    ON CONFLICT (__mv_id) DO UPDATE SET __mv_n = {lock_t}.__mv_n + 1;"
                ]

            def _match_nonnull(lhs: List[str]) -> str:
                conds = [
                    (
                        f"{l} IN (SELECT __mv_kset.__mv_k{i} FROM {kset} WHERE {no_null})"
                        if key_uses_in[i]
                        else f"{l} = ANY (ARRAY(SELECT __mv_kset.__mv_k{i} FROM {kset} WHERE {no_null}))"
                    )
                    for i, l in enumerate(lhs)
                ]
                if n > 1:
                    conds.append(f"({', '.join(lhs)}) IN (SELECT {', '.join(kcols)} FROM {kset} WHERE {no_null})")
                return " AND ".join(conds)

            def _match_null(lhs: List[str]) -> str:
                eq = " AND ".join(
                    f"({l} = {k} OR ({l} IS NOT DISTINCT FROM NULL AND {k} IS NOT DISTINCT FROM NULL))"
                    for l, k in zip(lhs, kcols)
                )
                return (
                    f"EXISTS (SELECT 1 FROM {kset} WHERE {any_null})\n"
                    f"    AND EXISTS (SELECT 1 FROM {kset} WHERE ({any_null}) AND {eq})"
                )

            mv_lhs = [f"{mv_table}.{self._q(mv_c)}" for _g, mv_c in group_mv_cols]
            src_lhs_out = [f"__mv_src.{self._q(mv_c)}" for _g, mv_c in group_mv_cols]
            for match in (_match_nonnull, _match_null):
                stmts.append(f"DELETE FROM {mv_table}\n    WHERE {match(mv_lhs)};")
            for match in (_match_nonnull, _match_null):
                if can_push_down and src_lhs_push is not None:
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

    def _tier2_norm_expr(self, expr: str) -> str:
        """Canonicalize an expression via sqlglot AST for matching GROUP BY keys
        to projected SELECT items regardless of whitespace/formatting."""
        try:
            return parse_sql_ast(expr, dialect=self.dialect).sql(dialect=_sqlglot_dialect(self.dialect)).lower()
        except sqlglot.errors.SqlglotError:
            return expr.strip().lower()

    def _compile_tier2_group_scoped(self) -> MaterializedViewPlan:
        if self.dialect == Dialect.MYSQL and any(
            "FULL" in j.join_type.upper() for j in self.view.joins
        ):
            raise UnsupportedMaterializedViewError(
                "MySQL/MariaDB do not support FULL OUTER JOIN with GROUP BY or aggregation; "
                "rewrite as a UNION ALL of LEFT JOIN and RIGHT JOIN."
            )
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in self.view.select_items]
        grp_exprs = self._tier2_scope_exprs()

        target_cols: List[str] = list(user_columns)
        group_mv_cols: List[Tuple[str, str]] = []
        hidden_grp_defs: List[Tuple[str, str, str]] = []

        non_agg_map: Dict[str, str] = {}
        is_multi_join = len(self.view.joins) > 1
        for it in self.view.select_items:
            if not it.is_aggregate:
                non_agg_map[it.raw_expr.strip().lower()] = it.alias
                non_agg_map[self._tier2_norm_expr(it.raw_expr)] = it.alias
                if it.source_col and not is_multi_join:
                    non_agg_map[it.source_col.lower()] = it.alias

        for idx_g, g_expr in enumerate(grp_exprs):
            g_low = g_expr.strip().lower()
            g_norm = self._tier2_norm_expr(g_expr)
            g_unq = (
                _unquote_ident(g_expr, dialect=self.dialect).lower()
                if (not is_multi_join and self._is_simple_col_ref(g_expr))
                else ""
            )
            if g_low in non_agg_map:
                mv_c = non_agg_map[g_low]
            elif g_norm in non_agg_map:
                mv_c = non_agg_map[g_norm]
            elif g_unq and g_unq in non_agg_map:
                mv_c = non_agg_map[g_unq]
            else:
                mv_c = f"__mv_grp_{idx_g}"
                target_cols.append(mv_c)
                if self.dialect == Dialect.SQLITE:
                    dt = "BLOB" + self._sqlite_key_collate_suffix(g_expr)
                else:
                    dt = self._infer_expr_storage_type(
                        SelectItem(raw_expr=g_expr, alias=mv_c, is_aggregate=False)
                    )
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                        dt = "VARCHAR(255)"
                if self.dialect == Dialect.MYSQL:
                    dt = dt + self._key_col_collate_suffix(g_expr, dt)
                hidden_grp_defs.append((mv_c, g_expr, dt))
            group_mv_cols.append((g_expr, mv_c))

        auto_grp_col_names = [mc for _, mc in group_mv_cols]
        indexed_cols_lower = self._indexed_column_names_lower(auto_grp_col_names)
        grp_mv_cols_lower = {mc.lower() for _, mc in group_mv_cols}

        col_defs_sql: List[str] = []
        col_types_by_lower: Dict[str, str] = {}
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
                dt = "BLOB" + self._sqlite_key_collate_suffix(it.raw_expr)
            else:
                dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            if self.dialect == Dialect.MYSQL:
                dt = dt + self._key_col_collate_suffix(it.raw_expr, dt)
            col_types_by_lower[it.alias.lower()] = dt
            col_defs_sql.append(f"{self._q(it.alias)} {dt}")
        for mv_c, _, dt in hidden_grp_defs:
            col_types_by_lower[mv_c.lower()] = dt
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

        uses_group_buckets = (
            self.dialect == Dialect.POSTGRES
            and len(self.view.joins) <= 1
            and bool(group_mv_cols)
            and self.view.base_table.lower() not in self._tier2_subquery_tables()
        )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        pg_multi_source = self.dialect == Dialect.POSTGRES and len(self.view.referenced_tables) >= 2
        for ref_idx, ref_table in enumerate(self.view.referenced_tables):
            src_bit: Optional[int] = None
            if pg_multi_source:
                src_bit = (1 << ref_idx) if ref_idx < 60 else ((1 << 60) | (1 << 61))
            t_ddls, t_ns, f_ns = self._build_tier2_triggers_for_table(
                view_name=view_name,
                mv_table=mv_table,
                base_t=ref_table,
                target_cols=target_cols,
                group_mv_cols=group_mv_cols,
                source_query_sql=source_query_sql,
                group_col_types=col_types_by_lower,
                uses_group_buckets=uses_group_buckets,
                pg_bracket_source_bit=src_bit,
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
            auxiliary_table_ddls=(
                self._pg_view_lock_ddls(with_group_buckets=uses_group_buckets)
                if self.dialect == Dialect.POSTGRES
                else []
            ),
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

