"""MySQL, MariaDB, and NewSQL-specific codegen helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot.tokens import TokenType as SQLGlotTokenType

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


class MySQLDialectMixin:
    def _mariadb_statistical_result_scale(self, item: SelectItem) -> str:
        """Return MariaDB's native VAR/STDDEV result scale as a view-safe literal.

        MariaDB derives exact-input scale from argument scale plus the session's
        div_precision_increment (capped at 30). Capture that setting during engine
        detection: CREATE VIEW rejects system-variable references. Approximate
        inputs retain their native DOUBLE display precision.
        """
        source = self._lookup_source_col_def(item.arg_expr) if item.arg_expr else None
        source_type = source.data_type.strip().upper() if source else ""

        if re.search(r"\b(?:DECIMAL|NUMERIC|DEC|FIXED)\b", source_type):
            precision_scale = self._numeric_precision_scale_from_type(source_type)
            scale = precision_scale[1] if precision_scale else 0
            return str(min(30, scale + self.engine_version.mariadb_div_precision_increment))

        if re.search(r"\b(?:TINYINT|SMALLINT|MEDIUMINT|INTEGER|INT|BIGINT|SERIAL)\b", source_type):
            return str(min(30, self.engine_version.mariadb_div_precision_increment))

        float_match = re.search(
            r"\b(?:FLOAT|DOUBLE|REAL)\s*\(\s*\d+\s*,\s*(\d+)\s*\)",
            source_type,
        )
        if float_match:
            return str(min(int(float_match.group(1)) + 4, 31))
        if any(token in source_type for token in ("FLOAT", "DOUBLE", "REAL")):
            return "31"

        exact_type = self._exact_numeric_precision_scale(item.arg_expr or "")
        if exact_type is not None:
            return str(min(30, exact_type[1] + self.engine_version.mariadb_div_precision_increment))
        return "31"


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
                    if isinstance(sub_ast, sqlglot_exp.Select) and (sub_ast.args.get("from_") or sub_ast.args.get("from")):
                        in_from = (sub_ast.args.get("from_") or sub_ast.args.get("from")).this
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
