"""MySQL, MariaDB, and NewSQL-specific codegen helpers."""

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


class MySQLDialectMixin(MixinHost):
    # MariaDB's NOT_FIXED_DEC: a DOUBLE result without a fixed number of
    # decimals. Rounding such a value to 31 decimals is *not* an identity in
    # MariaDB (ROUND(3.3323552986941074, 31) = 3.332355298694107).
    MARIADB_NOT_FIXED_DEC = 31
    # The result decimals cannot be derived from the argument.
    MARIADB_UNKNOWN_DEC = -1

    def _mariadb_statistical_result_scale(self, item: SelectItem) -> int:
        """Return the fixed decimals of MariaDB's native VAR/STDDEV result for `item`.

        MariaDB derives exact-input scale from argument scale plus the session's
        div_precision_increment (capped at 30) and fixes the DOUBLE result to
        that many decimals; the text protocol rounds the value to them.
        Capture that setting during engine detection: CREATE VIEW rejects
        system-variable references. `FLOAT(M,D)`/`DOUBLE(M,D)` inputs fix
        D + 4 decimals. Plain approximate inputs return `MARIADB_NOT_FIXED_DEC`
        (no rounding), and arguments whose family cannot be derived return
        `MARIADB_UNKNOWN_DEC` (the planner reports a limitation instead of
        guessing).
        """
        source = self._lookup_source_col_def(item.arg_expr) if item.arg_expr else None
        source_type = source.data_type.strip().upper() if source else ""

        if re.search(r"\b(?:DECIMAL|NUMERIC|DEC|FIXED)\b", source_type):
            precision_scale = self._numeric_precision_scale_from_type(source_type)
            scale = precision_scale[1] if precision_scale else 0
            return min(30, scale + self.engine_version.mariadb_div_precision_increment)

        if re.search(r"\b(?:TINYINT|SMALLINT|MEDIUMINT|INTEGER|INT|BIGINT|SERIAL)\b", source_type):
            return min(30, self.engine_version.mariadb_div_precision_increment)

        float_match = re.search(
            r"\b(?:FLOAT|DOUBLE|REAL)\s*\(\s*\d+\s*,\s*(\d+)\s*\)",
            source_type,
        )
        if float_match:
            return min(int(float_match.group(1)) + 4, self.MARIADB_NOT_FIXED_DEC)
        if any(token in source_type for token in ("FLOAT", "DOUBLE", "REAL")):
            return self.MARIADB_NOT_FIXED_DEC

        exact_type = self._exact_numeric_precision_scale(item.arg_expr or "")
        if exact_type is not None:
            return min(30, exact_type[1] + self.engine_version.mariadb_div_precision_increment)
        inferred = self._infer_ast_expr_storage_type(item.arg_expr) if item.arg_expr else None
        if inferred is not None and re.fullmatch(r"(?:DOUBLE(?: PRECISION)?|FLOAT|REAL)", inferred.strip().upper()):
            return self.MARIADB_NOT_FIXED_DEC
        return self.MARIADB_UNKNOWN_DEC

    def _mariadb_unreproducible_statistics(self, reader_view_ddl: str) -> List[str]:
        """VAR/STDDEV calls whose MariaDB result the plan cannot reproduce exactly.

        MariaDB fixes the decimals of VAR/STDDEV over exact (integer/DECIMAL) or
        FLOAT(M,D) arguments and rounds the result to them on output; those
        decimals then propagate through any enclosing expression, CTE, derived
        table, UNION branch or join, and window VAR/STDDEV accumulate in a
        row-order-dependent way. Every call anywhere in the view is therefore
        checked:
        - window calls are never reproducible;
        - calls over approximate arguments (no fixed decimals) are exact
          wherever they appear (native recomputation);
        - a fixed-decimal call is reproducible only as a top-level projected
          aggregate whose reader column is ROUND(col, <native decimals>) --
          verified on the compiled reader DDL, whatever the strategy -- or
          when it is used only in HAVING (compared unrounded on both sides).
        Returns the offending call texts (empty when the plan is exact).
        """
        stat_funcs = {"VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "VARIANCE", "STDDEV", "STD"}
        from ..sql_frontend import _parse_single_agg_expr

        sql = self.raw_view.raw_select_sql
        toks = sqlglot.tokenize(sql, read="mysql")
        compact = lambda text: re.sub(r"\s+", "", text).upper()
        top_level = {
            compact(it.raw_expr): it
            for it in self.raw_view.select_items
            if it.is_aggregate and it.agg_func is not None
        }
        simple_query = not (
            self.raw_view.has_cte
            or self.raw_view.has_subquery_in_from
            or self.raw_view.has_compound_set_op
            or self.raw_view.has_window_func
        )
        projected = " ".join(compact(it.raw_expr) for it in self.raw_view.select_items)
        reader_compact = compact(reader_view_ddl)
        issues: List[str] = []
        for idx in range(len(toks) - 1):
            if (
                toks[idx].text.upper() not in stat_funcs
                or toks[idx + 1].token_type != SQLGlotTokenType.L_PAREN
                or (idx > 0 and toks[idx - 1].token_type == SQLGlotTokenType.DOT)
            ):
                continue
            depth, close = 0, -1
            for j in range(idx + 1, len(toks)):
                if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                    depth += 1
                elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                    depth -= 1
                    if depth == 0:
                        close = j
                        break
            if close == -1:
                issues.append(toks[idx].text)
                continue
            call = sql[toks[idx].start : toks[close].end + 1]
            if close + 1 < len(toks) and toks[close + 1].text.upper() == "OVER":
                issues.append(call)
                continue
            parsed = _parse_single_agg_expr(call, "__v", dialect=Dialect.MYSQL)
            scale = self._mariadb_statistical_result_scale(parsed) if parsed is not None else self.MARIADB_UNKNOWN_DEC
            if scale == self.MARIADB_NOT_FIXED_DEC:
                continue
            if scale != self.MARIADB_UNKNOWN_DEC and simple_query:
                item = top_level.get(compact(call))
                if item is not None and compact(f"ROUND({item.alias},{scale})AS{item.alias}") in reader_compact:
                    continue
                if item is None and compact(call) not in projected and self.raw_view.having_clause and compact(call) in compact(self.raw_view.having_clause):
                    continue
            issues.append(call)
        return issues

    def _enforce_mariadb_statistical_exactness(self, reader_view_ddl: str) -> None:
        if self.subdialect != EngineSubdialect.MARIADB:
            return
        issues = self._mariadb_unreproducible_statistics(reader_view_ddl)
        if issues:
            from ..engines import UnsupportedMaterializedViewError

            raise UnsupportedMaterializedViewError(
                f"View '{self.logical_view_name}' uses {', '.join(dict.fromkeys(issues))} on MariaDB, "
                "whose result the materialized view cannot reproduce exactly: MariaDB fixes the "
                "decimals of VAR/STDDEV over exact or FLOAT(M,D) arguments and rounds results to them, "
                "and accumulates window VAR/STDDEV in row order. Project such a statistic as its own "
                "top-level column of a single-query view, or CAST its argument to DOUBLE."
            )


    def _rewrite_mysql_not_in_subquery(self, sql: str) -> str:
        toks = sqlglot.tokenize(sql, read="mysql")
        n = len(toks)
        outer_tbl = ""
        for i, t in enumerate(toks):
            if t.token_type == SQLGlotTokenType.FROM and i + 1 < n:
                outer_tbl = toks[i + 1].text
                if i + 2 < n and toks[i + 2].token_type == SQLGlotTokenType.VAR and toks[i + 2].text.upper() != "WHERE":
                    outer_tbl = toks[i + 2].text
                break
        i = 0
        while i < n - 4:
            if (
                toks[i + 1].token_type == SQLGlotTokenType.NOT
                and toks[i + 2].token_type == SQLGlotTokenType.IN
                and toks[i + 3].token_type == SQLGlotTokenType.L_PAREN
                and toks[i + 4].token_type == SQLGlotTokenType.SELECT
            ):
                start_i = i
                if i >= 2 and toks[i - 1].token_type == SQLGlotTokenType.DOT:
                    start_i = i - 2
                    lhs = f"{toks[i - 2].text}.{toks[i].text}"
                else:
                    lhs = toks[i].text
                depth = 0
                rp_idx = -1
                for j in range(i + 3, n):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        depth -= 1
                        if depth == 0:
                            rp_idx = j
                            break
                if rp_idx != -1:
                    inner_sql = sql[toks[i + 3].end + 1 : toks[rp_idx].start].strip()
                    try:
                        sub_ast = parse_sql_ast(inner_sql, dialect=Dialect.MYSQL)
                    except sqlglot.errors.SqlglotError:
                        sub_ast = None
                    from_node = (
                        (sub_ast.args.get("from_") or sub_ast.args.get("from"))
                        if isinstance(sub_ast, sqlglot_exp.Select)
                        else None
                    )
                    if isinstance(sub_ast, sqlglot_exp.Select) and from_node:
                        in_from = from_node.this
                        in_tbl = in_from.name
                        in_alias = in_from.alias or "__mv_not_in"
                        rhs_col = sub_ast.expressions[0].name if sub_ast.expressions else "id"
                        w_node = sub_ast.args.get("where")
                        in_w = w_node.this.sql("mysql").strip() if w_node and w_node.this else ""
                        lhs_q = lhs if "." in lhs else f"{outer_tbl}.{lhs}"
                        rhs_q = f"{in_alias}.{rhs_col}"
                        w_empty = f" WHERE ({in_w})" if in_w else ""
                        w_corr = f" WHERE ({in_w}) AND ({rhs_q} IS NULL OR {rhs_q} = {lhs_q})" if in_w else f" WHERE ({rhs_q} IS NULL OR {rhs_q} = {lhs_q})"
                        replacement = (
                            f"(NOT EXISTS (SELECT 1 FROM {in_tbl} AS {in_alias}{w_empty}) "
                            f"OR ({lhs_q} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM {in_tbl} AS {in_alias}{w_corr})))"
                        )
                        sql = sql[: toks[start_i].start] + replacement + sql[toks[rp_idx].end + 1 :]
                        toks = sqlglot.tokenize(sql, read="mysql")
                        n = len(toks)
                        i = 0
                        continue
            i += 1
        return sql
