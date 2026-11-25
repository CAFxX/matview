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
        row-order-dependent way. Every call node in the view's AST is checked
        by its position, never by its text:
        - window calls are never reproducible;
        - calls over approximate arguments (no fixed decimals) are exact
          wherever they appear (native recomputation);
        - a fixed-decimal call is reproducible only when that very node is a
          whole top-level select item of a single query whose compiled reader
          column is ROUND(col, <native decimals>). Any other occurrence is
          refused, including another copy of the same call text (e.g.
          `VARIANCE(x) + 0` next to a projected `VARIANCE(x)`) and any use in
          HAVING: string conversions there (CAST(VARIANCE(x) AS CHAR),
          CONCAT(STD(x), '')) observe the fixed decimals too.
        Returns the offending call texts (empty when the plan is exact).
        """
        from ..sql_frontend import _parse_single_agg_expr

        stat_classes = (
            sqlglot_exp.Variance, sqlglot_exp.VariancePop, sqlglot_exp.Stddev,
            sqlglot_exp.StddevPop, sqlglot_exp.StddevSamp,
        )
        stat_names = {"VAR_POP", "VAR_SAMP", "STDDEV_POP", "STDDEV_SAMP", "VARIANCE", "STDDEV", "STD"}
        try:
            root = sqlglot.parse_one(self.raw_view.raw_select_sql, read="mysql")
        except sqlglot.errors.SqlglotError:
            return ["<unparseable view>"]
        calls = [
            n for n in root.walk()
            if isinstance(n, stat_classes)
            or (isinstance(n, sqlglot_exp.Anonymous) and n.name.upper() in stat_names)
        ]
        if not calls:
            return []
        simple_query = isinstance(root, sqlglot_exp.Select) and not (
            self.raw_view.has_cte
            or self.raw_view.has_subquery_in_from
            or self.raw_view.has_compound_set_op
            or self.raw_view.has_window_func
        )
        compact = lambda text: re.sub(r"\s+", "", text).upper()
        reader_compact = compact(reader_view_ddl)
        top_items = list(root.expressions) if isinstance(root, sqlglot_exp.Select) else []
        issues: List[str] = []
        fixed_aliases: Set[str] = set()
        for node in calls:
            text = node.sql(dialect="mysql")
            if isinstance(node.parent, sqlglot_exp.Window):
                issues.append(text)
                continue
            parsed = _parse_single_agg_expr(text, "__v", dialect=Dialect.MYSQL)
            scale = (
                self._mariadb_statistical_result_scale(parsed)
                if parsed is not None and parsed.agg_func is not None
                else self.MARIADB_UNKNOWN_DEC
            )
            if scale == self.MARIADB_NOT_FIXED_DEC:
                continue
            if scale != self.MARIADB_UNKNOWN_DEC and simple_query:
                position = next(
                    (
                        i for i, item in enumerate(top_items)
                        if item is node or (isinstance(item, sqlglot_exp.Alias) and item.this is node)
                    ),
                    None,
                )
                if position is not None and position < len(self.raw_view.select_items):
                    alias = self.raw_view.select_items[position].alias
                    if compact(f"ROUND({alias},{scale})AS{alias}") in reader_compact:
                        fixed_aliases.add(alias.lower())
                        continue
            issues.append(text)
        # HAVING may also reach an accepted statistic through its select alias
        # (`HAVING CONCAT(v, '') = ...`), which MariaDB evaluates on the
        # fixed-decimal value: refused as well.
        having = root.args.get("having") if isinstance(root, sqlglot_exp.Select) else None
        if having is not None and fixed_aliases:
            for col in having.find_all(sqlglot_exp.Column):
                if not col.table and col.name.lower() in fixed_aliases:
                    issues.append(f"HAVING reference to {col.name}")
        return issues

    def _mysql_untyped_minmax_uses(self) -> List[str]:
        """MIN/MAX uses whose server-chosen result type the planner cannot reproduce.

        Measured natively (each argument type x context, through DML):
        - MySQL 8.4/9.7: MIN/MAX of TINYINT/SMALLINT/MEDIUMINT (any sign),
          YEAR, BIT or BOOL is typed by context, visibly on the wire, when it
          is nested in a projected expression (COALESCE(MAX(ti), 0) is
          BIGINT, MAX(ti) OVER () + 1 is INT) or is a window function over a
          CTE/derived table. INT, INT UNSIGNED and BIGINT are exact there.
        - MariaDB 10.11: grouped and derived-table uses are exact; a window
          MIN/MAX nested in an expression is not for the narrow types above
          and INT UNSIGNED (wrong values for UNSIGNED, YEAR, BIT).
        Those uses are refused. A whole projected MIN/MAX and HAVING (not
        projected) stay supported; an argument of unknown type is refused in
        the refused contexts.
        """
        try:
            root = sqlglot.parse_one(self.raw_view.raw_select_sql, read="mysql")
        except sqlglot.errors.SqlglotError:
            return []
        mariadb = self.subdialect == EngineSubdialect.MARIADB
        derived = self.raw_view.has_cte or self.raw_view.has_subquery_in_from
        top_items = list(root.expressions) if isinstance(root, sqlglot_exp.Select) else []
        having = root.args.get("having") if isinstance(root, sqlglot_exp.Select) else None
        issues: List[str] = []
        for node in root.find_all(sqlglot_exp.Min, sqlglot_exp.Max):
            arg = node.this
            if arg is None:
                continue
            col = self._lookup_source_col_def(arg.sql(dialect="mysql"))
            arg_type = (col.data_type if col is not None else (self._infer_ast_expr_storage_type(arg.sql(dialect="mysql")) or "")).upper()
            unsigned = "UNSIGNED" in arg_type
            base = re.sub(r"\s*\(\d+\)", "", arg_type).replace(" UNSIGNED", "").replace(" ZEROFILL", "").strip()
            narrow = base in ("TINYINT", "SMALLINT", "MEDIUMINT", "YEAR", "BOOL", "BOOLEAN") or base.startswith("BIT")
            unknown = not base
            if mariadb:
                affected = narrow or unknown or (base in ("INT", "INTEGER") and unsigned)
            else:
                affected = narrow or unknown
            if not affected:
                continue
            windowed = isinstance(node.parent, sqlglot_exp.Window)
            target = node.parent if windowed and node.parent is not None else node
            if windowed and derived and not mariadb:
                issues.append(target.sql(dialect="mysql"))
                continue
            if mariadb and not windowed:
                continue
            in_having = False
            cur = target.parent
            while cur is not None:
                if cur is having:
                    in_having = True
                    break
                cur = cur.parent
            if in_having:
                continue
            whole_item = any(
                item is target or (isinstance(item, sqlglot_exp.Alias) and item.this is target)
                for item in top_items
            )
            if not whole_item and target.find_ancestor(sqlglot_exp.Select) is root:
                issues.append(target.sql(dialect="mysql"))
        return issues

    def _enforce_mysql_minmax_typing(self) -> None:
        issues = self._mysql_untyped_minmax_uses()
        if issues:
            from ..engines import UnsupportedMaterializedViewError

            raise UnsupportedMaterializedViewError(
                f"View '{self.logical_view_name}' uses {', '.join(dict.fromkeys(issues))} on "
                f"{self.subdialect.value}, whose result type the server derives by context for this "
                "argument type; project MIN/MAX as its own column, or CAST the argument to SIGNED."
            )

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
                "and accumulates window VAR/STDDEV in row order. For a non-window statistic, CAST its "
                "argument to DOUBLE (kept exact by the full refresh), or, under the "
                "FLOAT_REASSOCIATION_ALLOWED opt-in, project the statistic of an exact argument as its "
                "own top-level column of a single-query view. Window VAR/STDDEV are not supported on "
                "MariaDB, whatever the argument type."
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
