"""SQLite-specific codegen helpers."""

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
    UnsupportedMaterializedViewError,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
)
from ..logical_ir import ColumnDef, JoinSpec, SelectItem, TableSchema, ViewSpec
from ..sql_frontend import (
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


class SQLiteDialectMixin(MixinHost):
    def _enforce_nonstandard_statistics_refusal(self) -> None:
        """Refuse VAR_POP/VAR_SAMP/STDDEV_POP/STDDEV_SAMP when the connection
        defines that function and its results differ from the exact definition.

        SQLite has no built-in statistical aggregates, so such a function was
        supplied by the connection (sqlean's `stats` extension, a loaded
        extension, an application-defined function) and is what the original
        view would call. matview maintains the exact definition (a sample
        statistic of one value is NULL, a population one is 0; correctly
        accumulated), which that function does not reproduce (probed
        behaviourally, see probe_sqlite_nonstandard_statistics), so the
        materialized view could not equal the view. Not bypassable by
        ignore_limitations (no relaxed guarantee covers it).
        """
        flagged = self.engine_version.sqlite_nonstandard_statistics
        unprobed = self.engine_version.sqlite_statistics_unprobed
        if self.dialect != Dialect.SQLITE or not (flagged or unprobed):
            return
        from ..engines import SQLITE_STATISTICAL_FUNCTIONS, UnsupportedMaterializedViewError

        try:
            root = sqlglot.parse_one(self.raw_view.raw_select_sql, read="sqlite")
        except sqlglot.errors.SqlglotError:
            return
        canonical = {
            sqlglot_exp.Variance: "var_samp", sqlglot_exp.VariancePop: "var_pop",
            sqlglot_exp.StddevPop: "stddev_pop", sqlglot_exp.StddevSamp: "stddev_samp",
        }
        used: Set[str] = set()
        for node in root.walk():
            name = canonical.get(type(node))
            if name is None and isinstance(node, sqlglot_exp.Anonymous):
                name = node.name.lower()
            if name is not None and name in (flagged if flagged else SQLITE_STATISTICAL_FUNCTIONS):
                used.add(name)
        if not used:
            return
        # The user's own spelling and arguments, from the source text.
        source = self.raw_view.raw_select_sql
        calls: List[str] = []
        for m in re.finditer(r"\b(%s)\s*\(" % "|".join(sorted(used)), source, flags=re.IGNORECASE):
            depth, end = 1, m.end()
            while end < len(source) and depth:
                depth += {"(": 1, ")": -1}.get(source[end], 0)
                end += 1
            calls.append(source[m.start():end])
        shown = ", ".join(dict.fromkeys(calls)) or ", ".join(sorted(used))
        if not flagged:
            raise UnsupportedMaterializedViewError(
                f"View '{self.logical_view_name}' uses {shown}, but the connection could not be probed "
                "(no usable cursor, e.g. another thread, or a stub), so it is unknown whether its "
                "VAR_*/STDDEV_* functions (SQLite has none built in: sqlean's stats extension, a loaded "
                "extension or an application-defined function) match the exact definition matview "
                "maintains. Register the view from a usable connection of the thread that owns it."
            )
        raise UnsupportedMaterializedViewError(
            f"View '{self.logical_view_name}' uses {shown}, but this SQLite connection defines "
            f"{', '.join(sorted(used))} (SQLite has none built in: sqlean's stats extension, a loaded "
            "extension or an application-defined function) with results that differ from the exact "
            "definition matview maintains (e.g. a sample statistic of one value is NULL there, 0.0 in "
            "sqlean), so the materialized view could not equal the original view. Use the connection's "
            "own differently named functions (sqlean: stats_var_pop, stats_var_samp, stats_stddev_pop, "
            "stats_stddev_samp), or a connection that does not define "
            f"{', '.join(sorted(used))}, on which matview maintains the exact aggregate itself."
        )

    def _sqlite_m2_agg_sql(
        self,
        item: SelectItem,
        eff_override: Optional[str] = None,
    ) -> str:
        """
        Computes Welford's centered sum of squared deviations `M2 = SUM((x - mean)^2)`
        in SQLite using an anchor-shifted two-pass identity `SUM((x - K)^2) - SUM(x - K)^2 / N`
        so that large offsets (e.g. `1e9 + delta` or `1e12 + delta`) cancel before squaring.
        """
        eff = eff_override if eff_override is not None else self._effective_agg_value_expr(item, record_prefix=None)
        anchor_k = "0.0"
        mode = getattr(self, "_alg_mode", "single")
        if mode in ("single", "distinct") and self.view.base_table:
            k_eff = self._effective_agg_value_expr(item, record_prefix="__mv_k")
            outer_t = self._q(self.view.base_table)
            base_alias = (
                self.view.joins[0].alias
                if len(self.view.joins) == 1
                and self.view.joins[0].alias
                and self.view.joins[0].alias.lower() != self.view.base_table.lower()
                else None
            )
            outer_ref = self._q(base_alias) if base_alias else outer_t
            k_conds = [f"({k_eff}) IS NOT NULL"]
            for g_expr in getattr(self, "_effective_group_by_exprs", ()):
                k_g = self._rewrite_expr_for_record(g_expr, "__mv_k")
                o_g = self._rewrite_expr_for_record(g_expr, outer_ref)
                k_conds.append(f"(({k_g}) IS ({o_g}))")
            anchor_k = (
                f"COALESCE((SELECT 1.0 * ({k_eff}) FROM {outer_t} AS __mv_k "
                f"WHERE {' AND '.join(k_conds)} LIMIT 1), 0.0)"
            )
        elif mode == "join" and self.view.joins:
            for j in self.view.joins:
                if re.search(rf"\b{re.escape(j.alias)}\.", eff, flags=re.IGNORECASE):
                    k_eff = re.sub(
                        rf"\b{re.escape(j.alias)}\.",
                        "__mv_k.",
                        eff,
                        flags=re.IGNORECASE,
                    )
                    if not any(
                        re.search(rf"\b{re.escape(oj.alias)}\.", k_eff, flags=re.IGNORECASE)
                        for oj in self.view.joins
                        if oj.alias.lower() != j.alias.lower()
                    ):
                        anchor_k = (
                            f"COALESCE((SELECT 1.0 * ({k_eff}) FROM {self._q(j.table_name)} AS __mv_k "
                            f"WHERE ({k_eff}) IS NOT NULL LIMIT 1), 0.0)"
                        )
                    break
        elif mode == "union_all_agg" and getattr(self, "_union_all_inner_sql", None):
            inner_u = getattr(self, "_union_all_inner_sql")
            u_al = getattr(self, "_union_all_sub_alias", "__mv_u")
            k_eff = re.sub(rf"\b{re.escape(u_al)}\.", "__mv_k.", eff, flags=re.IGNORECASE)
            anchor_k = (
                f"COALESCE((SELECT 1.0 * ({k_eff}) FROM ({inner_u}) AS __mv_k "
                f"WHERE ({k_eff}) IS NOT NULL LIMIT 1), 0.0)"
            )
        d = f"((1.0 * ({eff})) - {anchor_k})"
        raw_m2 = f"(SUM({d} * {d}) - (SUM({d}) * SUM({d})) / COUNT({eff}))"
        return (
            f"(CASE WHEN COUNT({eff}) < 1 THEN NULL "
            f"WHEN COUNT({eff}) = 1 OR MIN(1.0 * ({eff})) = MAX(1.0 * ({eff})) THEN 0.0 "
            f"WHEN ({raw_m2}) < 0.0 THEN 0.0 "
            f"ELSE ({raw_m2}) END)"
        )


    @staticmethod
    def _sqlite_bit_or_tree(terms: Sequence[str]) -> str:
        chunks = [
            "(" + " | ".join(terms[i : i + 8]) + ")"
            for i in range(0, len(terms), 8)
        ]
        return "(" + " | ".join(chunks) + ")"

    def _sqlite_enclosing_select_context(
        self,
        sql: str,
        toks: Sequence[Any],
        fn_tok_idx: int,
    ) -> Optional[Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]]:
        """Extract ``(from_joins_sql, where_sql, resolved_group_by_items)`` for
        the ``SELECT`` branch directly enclosing token ``fn_tok_idx``.

        Each item in ``resolved_group_by_items`` is ``(expr_sql, collate_suffix)``
        where ``collate_suffix`` is e.g. ``" COLLATE NOCASE"`` or ``""``.
        ``from_joins_sql`` and ``where_sql`` are sliced verbatim from ``sql``
        (avoiding sqlglot dialect rewrites).
        Returns ``None`` if the enclosing branch cannot be parsed as a ``SELECT``.
        Returns ``(None, None, [])`` for a ``FROM``-less constant ``SELECT``.
        """
        # Locate the innermost enclosing paren scope [scope_lo, scope_hi) around
        # fn_tok_idx that actually starts a SELECT (or WITH ... SELECT) query,
        # skipping expression parens like COALESCE(VAR_POP(DISTINCT x), -1.0)
        # or (1 + VAR_POP(DISTINCT x)).
        scope_lo = 0
        scope_hi = len(toks)
        depth = 0
        for i in range(fn_tok_idx - 1, -1, -1):
            tt = toks[i].token_type
            if tt == SQLGlotTokenType.R_PAREN:
                depth += 1
            elif tt == SQLGlotTokenType.L_PAREN:
                if depth == 0:
                    cand_lo = i + 1
                    if cand_lo < len(toks) and toks[cand_lo].token_type in (
                        SQLGlotTokenType.SELECT,
                        SQLGlotTokenType.WITH,
                    ):
                        scope_lo = cand_lo
                        # Find the matching R_PAREN for this L_PAREN.
                        r_depth = 0
                        for j in range(i + 1, len(toks)):
                            jtt = toks[j].token_type
                            if jtt == SQLGlotTokenType.L_PAREN:
                                r_depth += 1
                            elif jtt == SQLGlotTokenType.R_PAREN:
                                if r_depth == 0:
                                    scope_hi = j
                                    break
                                r_depth -= 1
                        break
                else:
                    depth -= 1

        # Narrow to the specific SELECT branch within [scope_lo, scope_hi)
        # (handling leading WITH ... CTE headers and UNION/INTERSECT/EXCEPT).
        setop_types = (
            SQLGlotTokenType.UNION,
            SQLGlotTokenType.INTERSECT,
            SQLGlotTokenType.EXCEPT,
        )
        branch_lo = scope_lo
        branch_hi = scope_hi
        cur_depth = 0
        last_branch_start = scope_lo
        for i in range(scope_lo, scope_hi):
            tt = toks[i].token_type
            if tt == SQLGlotTokenType.L_PAREN:
                cur_depth += 1
                continue
            if tt == SQLGlotTokenType.R_PAREN:
                cur_depth = max(0, cur_depth - 1)
                continue
            if cur_depth != 0:
                continue
            if tt == SQLGlotTokenType.SELECT and i <= fn_tok_idx:
                last_branch_start = i
            elif tt in setop_types:
                if i < fn_tok_idx:
                    last_branch_start = i + 1
                elif i > fn_tok_idx:
                    branch_hi = i
                    break
        branch_lo = last_branch_start
        # Advance past ALL/DISTINCT right after UNION/INTERSECT/EXCEPT if needed.
        while branch_lo < fn_tok_idx and toks[branch_lo].token_type != SQLGlotTokenType.SELECT:
            branch_lo += 1
        if branch_lo >= branch_hi or toks[branch_lo].token_type != SQLGlotTokenType.SELECT:
            return None

        # Slice FROM ... JOIN ... and WHERE ... verbatim from sql at top-level
        # depth within [branch_lo, branch_hi) to avoid sqlglot dialect rewrites.
        from_idx: Optional[int] = None
        where_idx: Optional[int] = None
        post_where_idx: Optional[int] = None
        clause_stop_types = {
            SQLGlotTokenType.GROUP_BY,
            SQLGlotTokenType.HAVING,
            SQLGlotTokenType.WINDOW,
            SQLGlotTokenType.ORDER_BY,
            SQLGlotTokenType.LIMIT,
            SQLGlotTokenType.OFFSET,
            SQLGlotTokenType.FETCH,
            SQLGlotTokenType.VALUES,
            SQLGlotTokenType.UNION,
            SQLGlotTokenType.INTERSECT,
            SQLGlotTokenType.EXCEPT,
            SQLGlotTokenType.SEMICOLON,
        }
        c_depth = 0
        for i in range(branch_lo, branch_hi):
            tt = toks[i].token_type
            if tt == SQLGlotTokenType.L_PAREN:
                c_depth += 1
                continue
            if tt == SQLGlotTokenType.R_PAREN:
                c_depth = max(0, c_depth - 1)
                continue
            if c_depth != 0:
                continue
            if from_idx is None:
                if tt == SQLGlotTokenType.FROM:
                    from_idx = i
            elif where_idx is None:
                if tt == SQLGlotTokenType.WHERE:
                    where_idx = i
                elif tt in clause_stop_types or toks[i].text.upper() in ("GROUP", "ORDER", "WINDOW"):
                    post_where_idx = i
                    break
            else:
                if tt in clause_stop_types or toks[i].text.upper() in ("GROUP", "ORDER", "WINDOW"):
                    post_where_idx = i
                    break

        if from_idx is None:
            if any(
                toks[i].token_type == SQLGlotTokenType.WHERE for i in range(branch_lo, branch_hi)
            ):
                # ``SELECT BIT_XOR(DISTINCT c) WHERE <cond>``: the aggregate must see the
                # implicit input row only when <cond> holds (one NULL row otherwise).
                raise UnsupportedMaterializedViewError(
                    "A FROM-less SELECT with a WHERE clause and a DISTINCT/FILTER VAR_*/STDDEV_*/BIT_*/"
                    "BOOL_* aggregate is not supported on SQLite; add a FROM source."
                )
            return (None, None, [])

        branch_end_char = toks[branch_hi - 1].end + 1
        from_end_char = (
            toks[where_idx].start
            if where_idx is not None
            else (toks[post_where_idx].start if post_where_idx is not None else branch_end_char)
        )
        from_joins_sql = sql[toks[from_idx].end + 1 : from_end_char].strip()
        if not from_joins_sql:
            return (None, None, [])
        where_sql: Optional[str] = None
        if where_idx is not None:
            where_end_char = (
                toks[post_where_idx].start if post_where_idx is not None else branch_end_char
            )
            ws = sql[toks[where_idx].end + 1 : where_end_char].strip()
            where_sql = ws if ws else None

        branch_sql = sql[toks[branch_lo].start : branch_end_char].strip()
        try:
            ast_node = parse_sql_ast(branch_sql, dialect=Dialect.SQLITE)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(ast_node, sqlglot_exp.Select):
            return None

        # Collect CTE projected column names visible in the enclosing scope (if any)
        # so CTE sources report only their own columns, not shadowed base columns.
        cte_cols_map: Dict[str, Set[str]] = {}
        try:
            scope_sql = sql[toks[scope_lo].start : toks[scope_hi - 1].end + 1].strip()
            scope_ast: Any = parse_sql_ast(scope_sql, dialect=Dialect.SQLITE)
            with_node = scope_ast.args.get("with_") if hasattr(scope_ast, "args") else None
            if with_node is not None:
                for cte in list(with_node.expressions or ()):
                    cte_alias = cte.alias
                    if not cte_alias:
                        continue
                    cols_set: Set[str] = set()
                    t_al: Any = cte.args.get("alias")
                    if isinstance(t_al, sqlglot_exp.TableAlias) and t_al.columns:
                        cols_set.update(c.name.lower() for c in list(t_al.columns) if c.name)
                    elif isinstance(cte.this, sqlglot_exp.Select):
                        for ce in list(cte.this.expressions or ()):
                            cname = ce.alias or (ce.this.name if isinstance(ce, sqlglot_exp.Alias) else ce.name)
                            if cname:
                                cols_set.add(cname.lower())
                    elif isinstance(cte.this, sqlglot_exp.SetOperation) and isinstance(cte.this.this, sqlglot_exp.Select):
                        for ce in list(cte.this.this.expressions or ()):
                            cname = ce.alias or (ce.this.name if isinstance(ce, sqlglot_exp.Alias) else ce.name)
                            if cname:
                                cols_set.add(cname.lower())
                    cte_cols_map[cte_alias.lower()] = cols_set
        except sqlglot.errors.SqlglotError:
            pass

        # Collect column names exposed ONLY by tables/CTEs/derived tables in this
        # SELECT's own FROM/JOIN clause so a SELECT alias sharing an unreferenced
        # base table's column name still resolves in GROUP BY.
        base_cols_lower: Set[str] = set()
        # A derived-table ``*`` whose columns cannot be determined: a SELECT alias
        # might collide with a hidden source column, so no alias is substituted.
        unknown_source_cols: List[bool] = []

        def _collect_from_source_cols(src: Any) -> None:
            if src is None:
                return
            if isinstance(src, sqlglot_exp.Table):
                t_low = src.name.lower()
                if t_low in cte_cols_map:
                    base_cols_lower.update(cte_cols_map[t_low])
                    return
                ts = self._schemas_lower.get(t_low)
                if ts is None and src.db:
                    ts = self._schemas_lower.get(f"{src.db.lower()}.{t_low}")
                if ts is not None:
                    base_cols_lower.update(c.lower() for c in ts.columns)
                return
            if isinstance(src, sqlglot_exp.Subquery):
                t_al: Any = src.args.get("alias")
                if isinstance(t_al, sqlglot_exp.TableAlias) and t_al.columns:
                    base_cols_lower.update(c.name.lower() for c in list(t_al.columns) if c.name)
                    return
                sub_q = src.this
                if isinstance(sub_q, sqlglot_exp.SetOperation):
                    sub_q = sub_q.this
                if isinstance(sub_q, sqlglot_exp.Select):
                    for se in list(sub_q.expressions or ()):
                        if isinstance(se, sqlglot_exp.Star) or (
                            isinstance(se, sqlglot_exp.Column) and isinstance(se.this, sqlglot_exp.Star)
                        ):
                            # ``SELECT *`` / ``t.*``: the column set is the sources' columns.
                            star_before = len(base_cols_lower)
                            inner_from = sub_q.args.get("from_") or sub_q.args.get("from")
                            if inner_from is not None:
                                _collect_from_source_cols(inner_from.this)
                            for ij in list(sub_q.args.get("joins") or ()):
                                _collect_from_source_cols(ij)
                            if len(base_cols_lower) == star_before:
                                unknown_source_cols.append(True)
                            continue
                        cname = se.alias or (se.this.name if isinstance(se, sqlglot_exp.Alias) else se.name)
                        if cname:
                            base_cols_lower.add(cname.lower())
                return
            if isinstance(src, sqlglot_exp.Join):
                _collect_from_source_cols(src.this)
                return
            if isinstance(src, sqlglot_exp.Paren):
                _collect_from_source_cols(src.this)
                for sj in list(src.args.get("joins") or ()):
                    _collect_from_source_cols(sj)

        from_node = ast_node.args.get("from_") or ast_node.args.get("from")
        if from_node is not None:
            _collect_from_source_cols(from_node.this)
        for j_node in list(ast_node.args.get("joins") or ()):
            _collect_from_source_cols(j_node)

        sel_exprs = list(ast_node.expressions or ())
        select_pos_list: List[str] = []
        select_alias_map: Dict[str, str] = {}
        for se in sel_exprs:
            if isinstance(se, sqlglot_exp.Alias):
                raw_s = se.this.sql("sqlite")
                select_pos_list.append(raw_s)
                al_name = se.alias
                if al_name and al_name.lower() not in base_cols_lower and not unknown_source_cols:
                    select_alias_map[al_name.lower()] = raw_s
            else:
                select_pos_list.append(se.sql("sqlite"))

        def _subst_select_aliases(expr_node: Any) -> Any:
            if not select_alias_map:
                return expr_node

            def _tx(n: Any) -> Any:
                if (
                    isinstance(n, sqlglot_exp.Column)
                    and not n.table
                    and n.name.lower() in select_alias_map
                ):
                    try:
                        rep_ast = parse_sql_ast(select_alias_map[n.name.lower()], dialect=Dialect.SQLITE)
                        return sqlglot_exp.Paren(this=rep_ast)
                    except sqlglot.errors.SqlglotError:
                        return n
                return n

            cloned = expr_node.copy()
            # Avoid descending into nested subqueries.
            def _walk(cur: Any) -> Any:
                if isinstance(cur, (sqlglot_exp.Subquery, sqlglot_exp.Select)):
                    return cur
                if isinstance(cur, sqlglot_exp.Column):
                    return _tx(cur)
                if not hasattr(cur, "args"):
                    return cur
                for arg_k, arg_v in list(cur.args.items()):
                    if hasattr(arg_v, "args"):
                        new_v = _walk(arg_v)
                        if new_v is not arg_v:
                            cur.set(arg_k, new_v)
                    elif isinstance(arg_v, list):
                        new_list = [
                            _walk(item) if hasattr(item, "args") else item
                            for item in arg_v
                        ]
                        cur.set(arg_k, new_list)
                return cur

            return _walk(cloned)

        grp_items: List[Tuple[str, str]] = []
        group_node = ast_node.args.get("group")
        if group_node is not None:
            for ge in list(group_node.expressions or ()):
                coll_suffix = ""
                inner_ge = ge
                if isinstance(ge, sqlglot_exp.Collate) and ge.this is not None and ge.expression is not None:
                    coll_suffix = f" COLLATE {ge.expression.sql('sqlite')}"
                    inner_ge = ge.this
                if (
                    isinstance(inner_ge, sqlglot_exp.Literal)
                    and not inner_ge.args.get("is_string")
                    and str(inner_ge.this).isdigit()
                ):
                    pos_i = int(str(inner_ge.this))
                    if 1 <= pos_i <= len(select_pos_list):
                        resolved = select_pos_list[pos_i - 1]
                        if coll_suffix:
                            grp_items.append((f"({resolved}){coll_suffix}", coll_suffix))
                        else:
                            grp_items.append((resolved, ""))
                        continue
                resolved_node = _subst_select_aliases(inner_ge)
                resolved_sql = resolved_node.sql("sqlite")
                if coll_suffix:
                    grp_items.append((f"({resolved_sql}){coll_suffix}", coll_suffix))
                else:
                    grp_items.append((resolved_sql, ""))
        return (from_joins_sql, where_sql, grp_items)

    def _rewrite_sqlite_total_in_sql(self, sql: str) -> str:
        if self.dialect == Dialect.SQLITE:
            # Prefix of every generated column/alias name; extended until it cannot
            # occur in the user's SQL, so a user column named ``__mv_g0`` (or an
            # alias ``__mv_u``) can never be shadowed by or collide with ours.
            mv_tag = ""
            sql_lower = sql.lower()
            while f"__mv{mv_tag}_" in sql_lower:
                mv_tag += "x"
            mv_ = f"__mv{mv_tag}_"

            def _ctx_checked(
                cur_sql: str, cur_toks: Sequence[Any], fn_idx: int
            ) -> Optional[Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]]:
                """Enclosing-SELECT context for a lowering that builds a subquery."""
                return self._sqlite_enclosing_select_context(cur_sql, cur_toks, fn_idx)

            def _sq_var_expr(
                eff_arg: str,
                ctx: Optional[Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]],
                is_samp: bool,
                is_std: bool,
            ) -> str:
                anchor_k = "0.0"
                if ctx is not None and ctx[0] is not None:
                    from_joins_sql, where_sql, grp_items = ctx
                    inner_where_parts: List[str] = []
                    if where_sql:
                        inner_where_parts.append(f"({where_sql})")
                    inner_where_parts.append(f"({eff_arg}) IS NOT NULL")
                    inner_where = " AND ".join(inner_where_parts)
                    if grp_items:
                        g_sel = ", ".join(
                            f"({g_expr}) AS {mv_}g{i}" for i, (g_expr, _) in enumerate(grp_items)
                        )
                        g_grp = ", ".join(f"({g_expr})" for g_expr, _ in grp_items)
                        corr_where = " AND ".join(
                            f"({mv_}a.{mv_}g{i} IS ({g_expr}){g_coll})"
                            for i, (g_expr, g_coll) in enumerate(grp_items)
                        )
                        anchor_k = (
                            f"COALESCE((SELECT {mv_}a.{mv_}k "
                            f"FROM (SELECT {g_sel}, MIN(1.0 * ({eff_arg})) AS {mv_}k "
                            f"FROM {from_joins_sql} WHERE {inner_where} "
                            f"GROUP BY {g_grp}) AS {mv_}a "
                            f"WHERE {corr_where}), 0.0)"
                        )
                    else:
                        anchor_k = (
                            f"COALESCE((SELECT MIN(1.0 * ({eff_arg})) "
                            f"FROM {from_joins_sql} WHERE {inner_where}), 0.0)"
                        )
                d = f"((1.0 * ({eff_arg})) - {anchor_k})"
                raw_m2 = f"(SUM({d} * {d}) - (SUM({d}) * SUM({d})) / COUNT({eff_arg}))"
                m2_s = (
                    f"(CASE WHEN COUNT({eff_arg}) < 1 THEN NULL "
                    f"WHEN COUNT({eff_arg}) = 1 OR MIN(1.0 * ({eff_arg})) = MAX(1.0 * ({eff_arg})) THEN 0.0 "
                    f"WHEN ({raw_m2}) < 0.0 THEN 0.0 "
                    f"ELSE ({raw_m2}) END)"
                )
                return self._variance_formula_sql(
                    new_cnt=f"COUNT({eff_arg})",
                    new_sum=f"SUM(1.0 * ({eff_arg}))",
                    new_sum2=m2_s,
                    is_sample=is_samp,
                    is_stddev=is_std,
                )

            def _sq_distinct_agg_expr(
                fn_up: str,
                eff_arg: str,
                ctx: Optional[Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]],
            ) -> str:
                is_samp = fn_up in ("VAR_SAMP", "VARIANCE_SAMP", "STDDEV_SAMP")
                is_std = fn_up in ("STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP")
                if ctx is None or ctx[0] is None:
                    if fn_up == "BIT_XOR":
                        return f"({eff_arg})"
                    if is_samp:
                        return "NULL"
                    return f"(CASE WHEN ({eff_arg}) IS NULL THEN NULL ELSE 0.0 END)"
                from_joins_sql, where_sql, grp_items = ctx
                inner_where_parts: List[str] = []
                if where_sql:
                    inner_where_parts.append(f"({where_sql})")
                inner_where_parts.append(f"({eff_arg}) IS NOT NULL")
                inner_where = " AND ".join(inner_where_parts)

                if fn_up == "BIT_XOR":
                    xor_tree = self._sqlite_bit_or_tree(
                        [f"((SUM((({mv_}u.{mv_}dv) >> {b}) & 1) & 1) << {b})" for b in range(64)]
                    )
                    agg_body = f"(CASE WHEN COUNT({mv_}u.{mv_}dv) = 0 THEN NULL ELSE {xor_tree} END)"
                    if grp_items:
                        g_sel = ", ".join(
                            f"({g_expr}) AS {mv_}g{i}" for i, (g_expr, _) in enumerate(grp_items)
                        )
                        g_grp = ", ".join(f"({g_expr})" for g_expr, _ in grp_items)
                        corr_where = " AND ".join(
                            f"({mv_}u.{mv_}g{i} IS ({g_expr}){g_coll})"
                            for i, (g_expr, g_coll) in enumerate(grp_items)
                        )
                        return (
                            f"(SELECT {agg_body} "
                            f"FROM (SELECT {g_sel}, ({eff_arg}) AS {mv_}dv "
                            f"FROM {from_joins_sql} WHERE {inner_where} "
                            f"GROUP BY {g_grp}, ({eff_arg})) AS {mv_}u "
                            f"WHERE {corr_where})"
                        )
                    # Wrap global scalar subquery in CASE WHEN COUNT(*) >= 0 so
                    # COUNT(*) always binds to the immediately enclosing SELECT
                    # (never an outer query) and still guarantees 1-row global
                    # aggregate semantics on an empty table.
                    return (
                        f"(CASE WHEN COUNT(*) >= 0 THEN (SELECT {agg_body} "
                        f"FROM (SELECT DISTINCT ({eff_arg}) AS {mv_}dv "
                        f"FROM {from_joins_sql} WHERE {inner_where}) AS {mv_}u) END)"
                    )

                d_expr = f"((1.0 * ({mv_}u.{mv_}dv)) - {mv_}a.{mv_}k)"
                raw_m2 = f"(SUM({d_expr} * {d_expr}) - (SUM({d_expr}) * SUM({d_expr})) / COUNT({mv_}u.{mv_}dv))"
                m2_s = (
                    f"(CASE WHEN COUNT({mv_}u.{mv_}dv) < 1 THEN NULL "
                    f"WHEN COUNT({mv_}u.{mv_}dv) = 1 OR MIN(1.0 * ({mv_}u.{mv_}dv)) = MAX(1.0 * ({mv_}u.{mv_}dv)) THEN 0.0 "
                    f"WHEN ({raw_m2}) < 0.0 THEN 0.0 "
                    f"ELSE ({raw_m2}) END)"
                )
                var_body = self._variance_formula_sql(
                    new_cnt=f"COUNT({mv_}u.{mv_}dv)",
                    new_sum=f"SUM(1.0 * ({mv_}u.{mv_}dv))",
                    new_sum2=m2_s,
                    is_sample=is_samp,
                    is_stddev=is_std,
                )
                if grp_items:
                    g_sel = ", ".join(
                        f"({g_expr}) AS {mv_}g{i}" for i, (g_expr, _) in enumerate(grp_items)
                    )
                    g_grp = ", ".join(f"({g_expr})" for g_expr, _ in grp_items)
                    on_cond = " AND ".join(
                        f"({mv_}u.{mv_}g{i} IS {mv_}a.{mv_}g{i}{g_coll})"
                        for i, (_, g_coll) in enumerate(grp_items)
                    )
                    corr_where = " AND ".join(
                        f"({mv_}u.{mv_}g{i} IS ({g_expr}){g_coll})"
                        for i, (g_expr, g_coll) in enumerate(grp_items)
                    )
                    return (
                        f"(SELECT {var_body} "
                        f"FROM (SELECT {g_sel}, ({eff_arg}) AS {mv_}dv "
                        f"FROM {from_joins_sql} WHERE {inner_where} "
                        f"GROUP BY {g_grp}, ({eff_arg})) AS {mv_}u "
                        f"JOIN (SELECT {g_sel}, MIN(1.0 * ({eff_arg})) AS {mv_}k "
                        f"FROM {from_joins_sql} WHERE {inner_where} "
                        f"GROUP BY {g_grp}) AS {mv_}a "
                        f"ON {on_cond} "
                        f"WHERE {corr_where})"
                    )
                return (
                    f"(CASE WHEN COUNT(*) >= 0 THEN (SELECT {var_body} "
                    f"FROM (SELECT DISTINCT ({eff_arg}) AS {mv_}dv "
                    f"FROM {from_joins_sql} WHERE {inner_where}) AS {mv_}u "
                    f"JOIN (SELECT MIN(1.0 * ({eff_arg})) AS {mv_}k "
                    f"FROM {from_joins_sql} WHERE {inner_where}) AS {mv_}a "
                    f"ON 1 = 1) END)"
                )

            s_sq = sql
            sq_fn_targets = {
                "VAR_POP", "VARIANCE", "VARIANCE_POP", "VAR_SAMP", "VARIANCE_SAMP",
                "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
            }
            while True:
                toks = _tokenize_sql(s_sq, read=_sqlglot_dialect(self.dialect))
                # Pick the deepest-nested (innermost-first, rightmost tie-break)
                # target call so inner FROM/WHERE subqueries are lowered before
                # outer calls slice and copy them.
                best: Optional[Tuple[int, int, str, int, int, Optional[str]]] = None
                p_depth = 0
                for idx in range(len(toks)):
                    tk = toks[idx]
                    tt = tk.token_type
                    if tt == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                        continue
                    if tt == SQLGlotTokenType.R_PAREN:
                        p_depth = max(0, p_depth - 1)
                        continue
                    if idx >= len(toks) - 2:
                        continue
                    fn_up = tk.text.upper()
                    if not (
                        fn_up in sq_fn_targets
                        and (idx == 0 or toks[idx - 1].token_type != SQLGlotTokenType.DOT)
                        and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN
                    ):
                        continue
                    # Skip CTE column-list headers like `WITH every(a, b) AS (...)`.
                    if idx > 0 and toks[idx - 1].token_type in (
                        SQLGlotTokenType.WITH,
                        SQLGlotTokenType.RECURSIVE,
                    ):
                        continue
                    depth = 0
                    rp_idx = -1
                    for j in range(idx + 1, len(toks)):
                        if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                            depth += 1
                        elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                            depth -= 1
                            if depth == 0:
                                rp_idx = j
                                break
                    if rp_idx == -1:
                        continue
                    if (
                        rp_idx + 2 < len(toks)
                        and toks[rp_idx + 1].token_type == SQLGlotTokenType.ALIAS
                        and (
                            toks[rp_idx + 2].token_type == SQLGlotTokenType.L_PAREN
                            or toks[rp_idx + 2].text.upper() in ("MATERIALIZED", "NOT")
                        )
                    ):
                        continue

                    end_idx = rp_idx
                    filter_cond: Optional[str] = None
                    if (
                        rp_idx + 3 < len(toks)
                        and toks[rp_idx + 1].text.upper() == "FILTER"
                        and toks[rp_idx + 2].token_type == SQLGlotTokenType.L_PAREN
                        and toks[rp_idx + 3].token_type == SQLGlotTokenType.WHERE
                    ):
                        f_depth = 0
                        f_rp = -1
                        for fj in range(rp_idx + 2, len(toks)):
                            if toks[fj].token_type == SQLGlotTokenType.L_PAREN:
                                f_depth += 1
                            elif toks[fj].token_type == SQLGlotTokenType.R_PAREN:
                                f_depth -= 1
                                if f_depth == 0:
                                    f_rp = fj
                                    break
                        if f_rp != -1:
                            filter_cond = s_sq[toks[rp_idx + 3].end + 1 : toks[f_rp].start].strip()
                            end_idx = f_rp

                    # Leave window functions (OVER (...)) untouched.
                    # (inline ``OVER (...)`` and named ``OVER w``)
                    if end_idx + 1 < len(toks) and toks[end_idx + 1].text.upper() == "OVER":
                        raise UnsupportedMaterializedViewError(
                            f"{fn_up}(...) OVER (...) is not supported on SQLite: it has no {fn_up} "
                            "aggregate (matview emulates only the plain grouped/global aggregate forms), "
                            "so it cannot be used as a window function."
                        )

                    cand = (p_depth, idx, fn_up, rp_idx, end_idx, filter_cond)
                    if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                        best = cand

                if best is None:
                    break
                _, idx, fn_up, rp_idx, end_idx, filter_cond = best
                tk = toks[idx]
                is_distinct = (
                    idx + 2 < rp_idx
                    and toks[idx + 2].token_type == SQLGlotTokenType.DISTINCT
                )
                is_explicit_all = (
                    idx + 2 < rp_idx
                    and toks[idx + 2].token_type == SQLGlotTokenType.ALL
                )
                if is_distinct or is_explicit_all:
                    inner_arg = s_sq[toks[idx + 2].end + 1 : toks[rp_idx].start].strip()
                else:
                    inner_arg = s_sq[toks[idx + 1].end + 1 : toks[rp_idx].start].strip()
                eff_arg = (
                    f"(CASE WHEN ({filter_cond}) THEN ({inner_arg}) ELSE NULL END)"
                    if filter_cond
                    else inner_arg
                )

                if is_distinct and fn_up in (
                    "VAR_POP", "VARIANCE", "VARIANCE_POP", "VAR_SAMP", "VARIANCE_SAMP",
                    "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                    "BIT_XOR",
                ):
                    ctx = _ctx_checked(s_sq, toks, idx)
                    rep = _sq_distinct_agg_expr(fn_up, eff_arg, ctx)
                elif fn_up in ("VAR_POP", "VARIANCE", "VARIANCE_POP"):
                    ctx = _ctx_checked(s_sq, toks, idx)
                    rep = _sq_var_expr(eff_arg, ctx, False, False)
                elif fn_up in ("VAR_SAMP", "VARIANCE_SAMP"):
                    ctx = _ctx_checked(s_sq, toks, idx)
                    rep = _sq_var_expr(eff_arg, ctx, True, False)
                elif fn_up in ("STDDEV_POP", "STDDEV", "STD"):
                    ctx = _ctx_checked(s_sq, toks, idx)
                    rep = _sq_var_expr(eff_arg, ctx, False, True)
                elif fn_up == "STDDEV_SAMP":
                    ctx = _ctx_checked(s_sq, toks, idx)
                    rep = _sq_var_expr(eff_arg, ctx, True, True)
                elif fn_up == "BIT_AND":
                    rep = self._sqlite_bit_or_tree([f"(MIN((({eff_arg}) >> {b}) & 1) << {b})" for b in range(64)])
                elif fn_up == "BIT_OR":
                    rep = self._sqlite_bit_or_tree([f"(MAX((({eff_arg}) >> {b}) & 1) << {b})" for b in range(64)])
                elif fn_up == "BIT_XOR":
                    rep = self._sqlite_bit_or_tree([f"((SUM((({eff_arg}) >> {b}) & 1) & 1) << {b})" for b in range(64)])
                elif fn_up in ("BOOL_AND", "EVERY"):
                    rep = f"(CASE WHEN COUNT({eff_arg}) = 0 THEN NULL WHEN MIN(CASE WHEN ({eff_arg}) IS NULL THEN NULL WHEN ({eff_arg}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
                else:
                    rep = f"(CASE WHEN COUNT({eff_arg}) = 0 THEN NULL WHEN MAX(CASE WHEN ({eff_arg}) IS NULL THEN NULL WHEN ({eff_arg}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
                s_sq = s_sq[: tk.start] + rep + s_sq[toks[end_idx].end + 1 :]
            return s_sq

        # Non-SQLite dialects
        s = sql
        toks = _tokenize_sql(s, read=_sqlglot_dialect(self.dialect))
        idx = 0
        while idx < len(toks) - 2:
            tk = toks[idx]
            if tk.text.upper() == "TOTAL" and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN:
                depth = 0
                rp_idx = -1
                for j in range(idx + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        depth -= 1
                        if depth == 0:
                            rp_idx = j
                            break
                if rp_idx != -1:
                    inner_arg = s[toks[idx + 1].end + 1 : toks[rp_idx].start].strip()
                    rep = f"COALESCE(1.0 * SUM({inner_arg}), 0.0)"
                    s = s[: tk.start] + rep + s[toks[rp_idx].end + 1 :]
                    toks = _tokenize_sql(s, read=_sqlglot_dialect(self.dialect))
                    idx = 0
                    continue
            idx += 1

        if self.dialect == Dialect.MYSQL:
            s = self._rewrite_mysql_not_in_subquery(s)
            # Full outer join lowering via sqlglot AST
            try:
                sel_ast = parse_sql_ast(s.strip(), dialect=Dialect.MYSQL)
            except sqlglot.errors.SqlglotError:
                sel_ast = None
            if (
                isinstance(sel_ast, sqlglot_exp.Select)
                and sel_ast.args.get("joins")
                and len(sel_ast.args["joins"]) == 1
                and not sel_ast.args.get("group")
                and not sel_ast.args.get("having")
                and not sel_ast.args.get("where")
            ):
                j0 = sel_ast.args["joins"][0]
                from_node = sel_ast.args.get("from_") or sel_ast.args.get("from")
                if j0.side.upper() == "FULL" and from_node and j0.args.get("on"):
                    from_t = from_node.this.sql("mysql")
                    j0_t = j0.this.sql("mysql")
                    on_s = j0.args["on"].sql("mysql")
                    sel_s = ", ".join(e.sql("mysql") for e in sel_ast.expressions)
                    lhs_col = j0.args["on"].expressions[0].sql("mysql") if j0.args["on"].expressions else j0.args["on"].sql("mysql").split("=")[0].strip()
                    s = (
                        f"SELECT {sel_s} FROM {from_t} LEFT JOIN {j0_t} ON {on_s} "
                        f"UNION ALL "
                        f"SELECT {sel_s} FROM {from_t} RIGHT JOIN {j0_t} ON {on_s} "
                        f"WHERE {lhs_col} IS NULL"
                    )
        return s

