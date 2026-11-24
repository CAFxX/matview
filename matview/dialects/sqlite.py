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
            k_conds = [f"({k_eff}) IS NOT NULL"]
            for g_expr in getattr(self, "_effective_group_by_exprs", ()):
                k_g = self._rewrite_expr_for_record(g_expr, "__mv_k")
                o_g = self._rewrite_expr_for_record(g_expr, outer_t)
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

    def _rewrite_sqlite_total_in_sql(self, sql: str) -> str:
        if self.dialect == Dialect.SQLITE:
            def _sq_var_expr(arg_s: str, is_samp: bool, is_std: bool) -> str:
                fake_it = SelectItem(raw_expr=f"VAR_POP({arg_s})", alias="__v", is_aggregate=True, agg_func=AggFunc.VAR_POP, arg_expr=arg_s)
                m2_s = self._sqlite_m2_agg_sql(fake_it, eff_override=arg_s)
                return self._variance_formula_sql(
                    new_cnt=f"COUNT({arg_s})",
                    new_sum=f"SUM(1.0 * ({arg_s}))",
                    new_sum2=m2_s,
                    is_sample=is_samp,
                    is_stddev=is_std,
                )

            s_sq = sql
            toks = _tokenize_sql(s_sq, read=_sqlglot_dialect(self.dialect))
            idx = 0
            sq_fn_targets = {
                "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
            }
            while idx < len(toks) - 2:
                tk = toks[idx]
                fn_up = tk.text.upper()
                if fn_up in sq_fn_targets and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN:
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
                        arg_s = s_sq[toks[idx + 1].end + 1 : toks[rp_idx].start].strip()
                        if fn_up in ("VAR_POP", "VARIANCE"):
                            rep = _sq_var_expr(arg_s, False, False)
                        elif fn_up == "VAR_SAMP":
                            rep = _sq_var_expr(arg_s, True, False)
                        elif fn_up in ("STDDEV_POP", "STDDEV", "STD"):
                            rep = _sq_var_expr(arg_s, False, True)
                        elif fn_up == "STDDEV_SAMP":
                            rep = _sq_var_expr(arg_s, True, True)
                        elif fn_up == "BIT_AND":
                            rep = self._sqlite_bit_or_tree([f"MIN(({arg_s}) & {1 << b})" for b in range(63)])
                        elif fn_up == "BIT_OR":
                            rep = self._sqlite_bit_or_tree([f"MAX(({arg_s}) & {1 << b})" for b in range(63)])
                        elif fn_up == "BIT_XOR":
                            rep = self._sqlite_bit_or_tree([f"((SUM((({arg_s}) >> {b}) & 1) & 1) << {b})" for b in range(63)])
                        elif fn_up in ("BOOL_AND", "EVERY"):
                            rep = f"(CASE WHEN COUNT({arg_s}) = 0 THEN NULL WHEN MIN(CASE WHEN ({arg_s}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
                        else:
                            rep = f"(CASE WHEN COUNT({arg_s}) = 0 THEN NULL WHEN MAX(CASE WHEN ({arg_s}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
                        s_sq = s_sq[: tk.start] + rep + s_sq[toks[rp_idx].end + 1 :]
                        toks = _tokenize_sql(s_sq, read=_sqlglot_dialect(self.dialect))
                        idx = 0
                        continue
                idx += 1
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

