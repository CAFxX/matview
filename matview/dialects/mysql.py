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

    def _mariadb_statistical_decimals_issues(self) -> List[str]:
        """Projections whose MariaDB VAR/STDDEV result decimals cannot be preserved.

        A top-level VAR/STDDEV aggregate keeps its native decimals through a
        reader-side ROUND. Nested in a larger projection expression (arithmetic,
        COALESCE, window, scalar subquery), the native result's fixed decimals
        propagate through MariaDB's expression typing and determine the
        rounding clients observe; the materialized expression over a stored
        DOUBLE does not carry them. Only non-fixed (approximate-input) results
        are safe there, and not as window functions: MariaDB's window
        VAR/STDDEV accumulation is row-order dependent.
        """
        stat_funcs = {"VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "VARIANCE", "STDDEV", "STD"}
        stat_aggs = (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP)
        from ..sql_frontend import _parse_single_agg_expr

        issues: List[str] = []
        for it in self.view.select_items:
            if it.is_aggregate and it.agg_func in stat_aggs:
                if self._mariadb_statistical_result_scale(it) == self.MARIADB_UNKNOWN_DEC:
                    issues.append(it.alias)
                continue
            toks = sqlglot.tokenize(it.raw_expr, read="mysql")
            for idx in range(len(toks) - 1):
                if (
                    toks[idx].text.upper() not in stat_funcs
                    or toks[idx + 1].token_type != SQLGlotTokenType.L_PAREN
                ):
                    continue
                depth = 0
                close = -1
                for j in range(idx + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        depth -= 1
                        if depth == 0:
                            close = j
                            break
                call = it.raw_expr[toks[idx].start : toks[close].end + 1] if close != -1 else ""
                if close != -1 and close + 1 < len(toks) and toks[close + 1].text.upper() == "OVER":
                    # MariaDB evaluates window VAR/STDDEV with a different,
                    # row-order-dependent accumulation than the materializing
                    # statement can reproduce (1-ULP differences observed over
                    # DOUBLE), whatever the argument type.
                    issues.append(it.alias)
                    break
                parsed = _parse_single_agg_expr(call, "__v", dialect=Dialect.MYSQL) if call else None
                if (
                    parsed is None
                    or self._mariadb_statistical_result_scale(parsed) != self.MARIADB_NOT_FIXED_DEC
                ):
                    issues.append(it.alias)
                    break
        return issues


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
