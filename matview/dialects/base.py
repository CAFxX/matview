"""Shared dialect helpers, identifier quoting, storage type inference, and DDL creation."""

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
    _parse_single_agg_expr,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _sanitize_ident,
    _split_top_level,
    _sqlite_type_affinity,
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


class DialectBaseMixin(MixinHost):
    def _bounded_ident(self, prefix: str, stem: str, suffix: str = "", max_len: int = 62) -> str:
        full = f"{prefix}{stem}{suffix}"
        if len(full) <= max_len:
            return full
        h = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        avail = max(4, max_len - len(prefix) - len(suffix) - 9)
        return f"{prefix}{stem[:avail]}_{h}{suffix}"

    def _backing_table_name(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}store_", self.logical_view_name)

    def _create_table_ddl(
        self,
        table_name: str,
        col_defs_sql: List[str],
        is_backing_store: bool = False,
        pk_cols: Optional[Sequence[str]] = None,
    ) -> str:
        fillfactor_tail = ""
        if is_backing_store and self.dialect == Dialect.POSTGRES and self.pg_fillfactor is not None:
            fillfactor_tail = f" WITH (fillfactor = {self.pg_fillfactor})"

        without_rowid_tail = ""
        if self.dialect == Dialect.SQLITE:
            effective_pks: List[str] = list(pk_cols) if pk_cols else []
            if not effective_pks:
                for line in col_defs_sql:
                    m = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", line, flags=re.IGNORECASE)
                    if m:
                        effective_pks = [
                            c.strip().strip('"').strip("`")
                            for c in m.group(1).split(",")
                            if c.strip()
                        ]
                        break
            if len(effective_pks) >= 2:
                pk_set = {p.lower() for p in effective_pks}
                found_nn: Set[str] = set()
                has_autoincrement = False
                for line in col_defs_sql:
                    line_s = line.strip()
                    if re.match(r"^PRIMARY\s+KEY\b", line_s, flags=re.IGNORECASE):
                        continue
                    if "AUTOINCREMENT" in line_s.upper():
                        has_autoincrement = True
                    col_match = re.match(r'^("([^"]+)"|`([^`]+)`|([A-Za-z0-9_]+))\s+(.*)$', line_s)
                    if col_match:
                        col_ident = (col_match.group(2) or col_match.group(3) or col_match.group(4)).lower()
                        if col_ident in pk_set:
                            rest = col_match.group(5).upper()
                            if "NOT NULL" in rest:
                                found_nn.add(col_ident)
                if len(found_nn) == len(pk_set) and not has_autoincrement:
                    without_rowid_tail = " WITHOUT ROWID"

        cols_joined = ",\n    ".join(col_defs_sql)
        return f"CREATE TABLE {table_name} (\n    {cols_joined}\n){fillfactor_tail}{without_rowid_tail};"


    def _q(self, ident: str) -> str:
        return quote_sql_identifier(ident, self.dialect)

    def _lookup_col_def(self, expr: Optional[str]) -> Optional[ColumnDef]:
        if not expr:
            return None
        s = expr.strip()
        if not _is_simple_column_expr(s, dialect=self.dialect):
            return None
        parts = _split_top_level(s, ".", dialect=self.dialect)
        if len(parts) == 2:
            prefix, col_part = parts[0], parts[1]
            prefix_clean = _unquote_ident(prefix, dialect=self.dialect).lower()
            col_clean = _unquote_ident(col_part, dialect=self.dialect).lower()
            for j in self.view.joins:
                if j.alias.lower() == prefix_clean or j.table_name.lower() == prefix_clean:
                    ts = self._schemas_lower.get(j.table_name.lower())
                    if ts:
                        for c_def in ts.columns.values():
                            if c_def.name.lower() == col_clean:
                                return c_def
            ts = self._schemas_lower.get(prefix_clean)
            if ts:
                for c_def in ts.columns.values():
                    if c_def.name.lower() == col_clean:
                        return c_def
            return None

        col_clean = _unquote_ident(s, dialect=self.dialect).lower()
        if self.primary_schema:
            for c_def in self.primary_schema.columns.values():
                if c_def.name.lower() == col_clean:
                    return c_def
        for t_name in self.view.referenced_tables:
            ts = self._schemas_lower.get(t_name.lower())
            if ts:
                for c_def in ts.columns.values():
                    if c_def.name.lower() == col_clean:
                        return c_def
        return None

    def _is_simple_col_ref(self, expr: str) -> bool:
        return self._lookup_col_def(expr) is not None

    def _col_def(self, expr: str) -> ColumnDef:
        c = self._lookup_col_def(expr)
        assert c is not None
        return c

    def _int_type(self) -> str:
        return "BIGINT"

    def _float_type(self) -> str:
        if self.dialect == Dialect.POSTGRES:
            return "DOUBLE PRECISION"
        elif self.dialect == Dialect.MYSQL:
            return "DOUBLE"
        return "REAL"

    def _lookup_source_col_def(self, expr: Optional[str]) -> Optional[ColumnDef]:
        """Resolve a column expression directly or through CTE/subquery projections."""
        col = self._lookup_col_def(expr)
        if col is not None or not expr:
            return col
        try:
            node = parse_sql_ast(expr, dialect=self.dialect)
        except (sqlglot.errors.SqlglotError, AttributeError, TypeError, ValueError):
            return None
        if isinstance(node, sqlglot_exp.Column):
            resolved = self._resolve_cte_column_expression(node)
            if isinstance(resolved, sqlglot_exp.Column):
                return self._lookup_col_def(resolved.sql())
        return None

    def _sum_result_storage_type(self, item: SelectItem) -> str:
        """Preserve the engine's native SUM result type for direct numeric columns."""
        source = self._lookup_source_col_def(item.arg_expr) if item.arg_expr else None
        source_type = source.data_type.strip().upper() if source else ""
        if self.dialect == Dialect.SQLITE:
            # SQLite SUM returns INTEGER when every input is an integer and REAL
            # as soon as one input is not, whatever the declared column type
            # (an INTEGER/NUMERIC column can store REAL values). A column with
            # INTEGER/NUMERIC affinity would turn an integral REAL sum such as
            # 10.0 into INTEGER 10, so only a REAL-affinity source (whose values
            # are always REAL) keeps REAL; everything else is stored without
            # affinity (declared BLOB), preserving the native storage class.
            if source is not None and _sqlite_type_affinity(source_type) == "REAL":
                return "REAL"
            return "BLOB"
        if not source_type and self.dialect == Dialect.POSTGRES and item.arg_expr:
            # An expression argument (SUM(a * b), SUM(CASE ... THEN 1 ELSE 0
            # END)) has the native type of the expression; classify it like a
            # column of that type.
            source_type = (self._infer_ast_expr_storage_type(item.arg_expr) or "").upper()
        if not source_type:
            exact_type = self._exact_numeric_precision_scale(item.arg_expr or "")
            if exact_type is not None:
                if self.dialect == Dialect.MYSQL:
                    return f"DECIMAL(65, {exact_type[1]})"
                if self.dialect == Dialect.POSTGRES:
                    return "NUMERIC"
                return "NUMERIC"
            return self._float_type()

        is_approximate = any(token in source_type for token in ("FLOAT", "DOUBLE", "REAL"))
        is_integer = any(
            token in source_type
            for token in ("TINYINT", "SMALLINT", "MEDIUMINT", "INTEGER", "INT", "BIGINT", "SERIAL")
        )
        is_exact_decimal = any(token in source_type for token in ("DECIMAL", "NUMERIC"))

        if self.dialect == Dialect.MYSQL:
            # MySQL SUM returns DECIMAL for exact-value arguments (integers
            # included) and DOUBLE for approximate-value arguments.
            if is_approximate:
                return "DOUBLE"
            if is_integer or is_exact_decimal:
                match = re.search(
                    r"\b(?:DECIMAL|NUMERIC)\s*\(\s*\d+\s*,\s*(\d+)\s*\)",
                    source_type,
                )
                scale = int(match.group(1)) if match else 0
                return f"DECIMAL(65, {scale})"
        elif self.dialect == Dialect.POSTGRES:
            # PostgreSQL widens smallint/integer SUM to bigint and bigint /
            # numeric SUM to numeric; approximate SUM retains its input family.
            if is_exact_decimal or "BIGINT" in source_type:
                return "NUMERIC"
            if is_integer:
                return "BIGINT"
            if "REAL" in source_type:
                return "REAL"
            if is_approximate:
                return "DOUBLE PRECISION"
        else:
            if is_integer:
                return "INTEGER"
            if is_exact_decimal:
                return "NUMERIC"
            if is_approximate:
                return "REAL"
        return self._float_type()

    def _statistical_result_storage_type(self, item: SelectItem) -> str:
        """Match native VAR/STDDEV result types for exact versus approximate inputs."""
        source = self._lookup_source_col_def(item.arg_expr) if item.arg_expr else None
        source_type = source.data_type.strip().upper() if source else ""
        is_exact_numeric = bool(
            (
                source_type
                and any(token in source_type for token in ("INT", "DECIMAL", "NUMERIC", "SERIAL"))
                and not any(token in source_type for token in ("FLOAT", "DOUBLE", "REAL"))
            )
            or (
                not source_type
                and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
            )
        )
        if self.dialect == Dialect.POSTGRES:
            # PostgreSQL statistical aggregates over integer/numeric inputs
            # return NUMERIC; floating inputs return DOUBLE PRECISION.
            return "NUMERIC" if is_exact_numeric else "DOUBLE PRECISION"
        if self.dialect == Dialect.MYSQL:
            # MySQL VAR_POP / STDDEV_* return DOUBLE for both exact and
            # approximate numeric argument families.
            return "DOUBLE"
        return "REAL"


    def _max_decimal_scale(self) -> int:
        if (
            self.subdialect == EngineSubdialect.MARIADB
            and self.engine_version.version >= (10, 2, 1)
        ):
            return 38
        return 30

    def _numeric_precision_scale_from_type(self, data_type: str) -> Optional[Tuple[int, int]]:
        upper = data_type.strip().upper()
        decimal_match = re.search(
            r"\b(?:DECIMAL|NUMERIC|DEC|FIXED)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?",
            upper,
        )
        if decimal_match:
            return int(decimal_match.group(1) or 10), int(decimal_match.group(2) or 0)
        if re.search(r"\bTINYINT\b", upper):
            return 3, 0
        if re.search(r"\bSMALLINT\b", upper):
            return 5, 0
        if re.search(r"\bMEDIUMINT\b", upper):
            return 7, 0
        if re.search(r"\b(?:BIGINT|BIGSERIAL)\b", upper):
            return (20 if "UNSIGNED" in upper or "SERIAL" in upper else 19), 0
        if re.search(r"\b(?:INTEGER|INT)\b", upper):
            return 10, 0
        if re.search(r"\bSERIAL\b", upper):
            return (20 if self.dialect == Dialect.MYSQL else 10), 0
        if re.search(r"\bSMALLSERIAL\b", upper):
            return 5, 0
        return None

    def _resolve_cte_column_expression(self, column: Any) -> Optional[Any]:
        """Resolve a projected CTE column through chained CTEs to its source expression."""
        if not isinstance(column, sqlglot_exp.Column):
            return None
        try:
            query_ast = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None

        cte_queries: Dict[str, Any] = {}
        for cte in query_ast.find_all(sqlglot_exp.CTE):
            cte_name = cte.alias
            if cte_name:
                cte_queries[cte_name.lower()] = cte.this
        def unwrap_select(query: Any) -> Any:
            while isinstance(query, sqlglot_exp.SetOperation):
                query = query.this
            return query

        def select_table_aliases(select: Any) -> Dict[str, str]:
            aliases: Dict[str, str] = {}
            from_clause = select.args.get("from_") if hasattr(select, "args") else None
            source = getattr(from_clause, "this", None)
            if isinstance(source, sqlglot_exp.Table) and source.name:
                sname = source.name.lower()
                aliases[sname] = sname
                if source.alias:
                    aliases[source.alias.lower()] = sname
            for j in (select.args.get("joins") or ()):
                j_tbl = getattr(j, "this", None)
                if isinstance(j_tbl, sqlglot_exp.Table) and j_tbl.name:
                    jname = j_tbl.name.lower()
                    aliases[jname] = jname
                    if j_tbl.alias:
                        aliases[j_tbl.alias.lower()] = jname
            return aliases

        def source_cte_name(select: Any) -> Optional[str]:
            from_clause = select.args.get("from_") if hasattr(select, "args") else None
            source = getattr(from_clause, "this", None)
            if isinstance(source, sqlglot_exp.Table):
                return source.name.lower()
            return None

        def bind_select_sources(node: Any, inner_aliases: Dict[str, str], default_src: Optional[str]) -> Any:
            if isinstance(node, sqlglot_exp.Column):
                raw_up = node.table.lower() if node.table else default_src
                actual_up = inner_aliases.get(raw_up, raw_up) if raw_up else None
                if actual_up:
                    return sqlglot_exp.column(node.name, table=actual_up)
                return node
            copied = node.copy()
            for col_n in list(copied.find_all(sqlglot_exp.Column)):
                raw_up = col_n.table.lower() if col_n.table else default_src
                actual_up = inner_aliases.get(raw_up, raw_up) if raw_up else None
                if actual_up:
                    col_n.replace(sqlglot_exp.column(col_n.name, table=actual_up))
            return copied

        root_select = unwrap_select(query_ast)
        if not isinstance(root_select, sqlglot_exp.Select):
            return None

        def resolve(relation: Optional[str], name: str, seen: set[Tuple[str, str]]) -> Optional[Any]:
            if not relation:
                return None
            relation = relation.lower()
            key = (relation, name.lower())
            if key in seen:
                return None
            cte_query = cte_queries.get(relation)
            if cte_query is None:
                return None
            select = unwrap_select(cte_query)
            if not isinstance(select, sqlglot_exp.Select):
                return None
            inner_aliases = select_table_aliases(select)
            default_src = source_cte_name(select)
            next_seen = set(seen)
            next_seen.add(key)
            for projection in select.expressions:
                output_name = projection.alias
                if not output_name and isinstance(projection, sqlglot_exp.Column):
                    output_name = projection.name
                if output_name.lower() != name.lower():
                    continue
                expression = projection.this if isinstance(projection, sqlglot_exp.Alias) else projection
                if isinstance(expression, sqlglot_exp.Column):
                    raw_up = expression.table.lower() if expression.table else default_src
                    actual_up = inner_aliases.get(raw_up, raw_up) if raw_up else None
                    resolved = resolve(actual_up, expression.name, next_seen)
                    if resolved is not None:
                        return resolved
                return bind_select_sources(expression, inner_aliases, default_src)
            return None

        root_aliases = select_table_aliases(root_select)
        raw_root_rel = column.table.lower() if column.table else source_cte_name(root_select)
        root_relation = root_aliases.get(raw_root_rel, raw_root_rel) if raw_root_rel else None
        resolved_cte = resolve(root_relation, column.name, set())
        if resolved_cte is not None:
            return resolved_cte

        # Resolve projected columns of derived tables as well as named CTEs.
        # In particular, UNION/UNION ALL must merge the numeric types of every
        # arm; looking only at the left-most SELECT can incorrectly turn an
        # exact MySQL AVG argument into an approximate DOUBLE.
        from_clause = root_select.args.get("from_")
        derived = getattr(from_clause, "this", None)
        if isinstance(derived, sqlglot_exp.Subquery):
            derived_alias = derived.alias
            if not column.table or (derived_alias and derived_alias.lower() == column.table.lower()):
                query = derived.this

                def set_operation_arms(node: Any) -> List[Any]:
                    if isinstance(node, sqlglot_exp.SetOperation):
                        return set_operation_arms(node.this) + set_operation_arms(node.expression)
                    return [node]

                arms = set_operation_arms(query)
                expressions: List[Any] = []
                for arm in arms:
                    if not isinstance(arm, sqlglot_exp.Select):
                        continue
                    for projection in arm.expressions:
                        output_name = projection.alias
                        if not output_name and isinstance(projection, sqlglot_exp.Column):
                            output_name = projection.name
                        if output_name.lower() == column.name.lower():
                            expressions.append(
                                projection.this if isinstance(projection, sqlglot_exp.Alias) else projection
                            )
                            break
                if expressions:
                    if len(expressions) == 1:
                        return expressions[0]
                    return sqlglot_exp.Coalesce(this=expressions[0], expressions=expressions[1:])
        return None

    def _exact_numeric_precision_scale(self, expr: str) -> Optional[Tuple[int, int]]:
        """Infer exact numeric precision/scale for AVG arguments conservatively."""
        if not expr or not expr.strip():
            return None

        def merge_types(values: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
            if not values:
                return None
            scale = min(self._max_decimal_scale(), max(value[1] for value in values))
            integer_digits = max(value[0] - value[1] for value in values)
            return min(65, max(scale, integer_digits + scale)), scale

        def infer(node: Any) -> Optional[Tuple[int, int]]:
            kind = node.__class__.__name__
            if kind == "Column":
                column = self._lookup_col_def(node.sql())
                if column:
                    return self._numeric_precision_scale_from_type(column.data_type)
                cte_expression = self._resolve_cte_column_expression(node)
                return infer(cte_expression) if cte_expression is not None else None
            if kind == "Literal":
                if node.args.get("is_string"):
                    return None
                value = str(node.this)
                if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", value):
                    return None
                whole, dot, fraction = value.partition(".")
                scale = len(fraction) if dot else 0
                precision = max(1, len((whole + fraction).lstrip("0")), scale)
                return precision, min(self._max_decimal_scale(), scale)
            if kind in ("Sum", "Min", "Max") and node.this is not None and not node.find(sqlglot_exp.Window):
                # MySQL SUM over an exact argument is DECIMAL with the
                # argument's scale (precision p + 22); MIN/MAX keep the type.
                arg = node.this.this if isinstance(node.this, sqlglot_exp.Distinct) else node.this
                if isinstance(node.this, sqlglot_exp.Distinct):
                    exprs = node.this.expressions
                    arg = exprs[0] if exprs else None
                inner = infer(arg) if arg is not None else None
                if inner is None:
                    return None
                if kind == "Sum":
                    return min(65, inner[0] + 22), inner[1]
                return inner
            if kind in ("Paren", "Neg", "Abs", "Floor", "Ceil", "Ceiling", "Sign", "Trunc"):
                return infer(node.this)
            if kind == "Cast":
                target = node.args.get("to")
                return self._numeric_precision_scale_from_type(target.sql()) if target is not None else None
            if kind in ("Add", "Sub", "Mul", "Div", "Mod"):
                left, right = infer(node.this), infer(node.expression)
                if left is None or right is None:
                    return None
                p1, s1 = left
                p2, s2 = right
                if kind == "Mod":
                    # MySQL/MariaDB MOD keeps the larger scale and the larger
                    # integer part (DECIMAL(12,3) % 2 is DECIMAL(12,3)).
                    scale = max(s1, s2)
                    precision = max(p1 - s1, p2 - s2) + scale
                elif kind in ("Add", "Sub"):
                    scale = max(s1, s2)
                    precision = max(p1 - s1, p2 - s2) + scale + 1
                elif kind == "Mul":
                    scale, precision = s1 + s2, p1 + p2
                else:
                    scale = s1 + 4
                    precision = (p1 - s1) + s2 + scale
                scale = min(self._max_decimal_scale(), scale)
                return min(65, max(scale, precision)), scale
            if kind in ("Coalesce", "IfNull", "Nullif", "Greatest", "Least"):
                values = [infer(child) for child in node.iter_expressions()]
                if not values or any(value is None for value in values):
                    return None
                return merge_types([value for value in values if value is not None])
            if kind == "Case":
                branches = []
                for branch in node.args.get("ifs") or ():
                    value = branch.args.get("true")
                    if value is not None:
                        branches.append(infer(value))
                default = node.args.get("default")
                if default is not None:
                    branches.append(infer(default))
                if not branches or any(value is None for value in branches):
                    return None
                return merge_types([value for value in branches if value is not None])
            if kind == "Round":
                value = infer(node.this)
                scale_node = node.expression
                if value is None or scale_node is None or scale_node.__class__.__name__ != "Literal":
                    return value
                try:
                    requested_scale = max(0, int(scale_node.this))
                except (TypeError, ValueError):
                    return value
                requested_scale = min(self._max_decimal_scale(), requested_scale)
                return min(65, max(requested_scale, value[0] - value[1] + requested_scale + 1)), requested_scale
            return None

        try:
            return infer(parse_sql_ast(expr, dialect=self.dialect))
        except (sqlglot.errors.SqlglotError, AttributeError, TypeError, ValueError):
            return None

    def _avg_result_storage_type_for_expr(self, expr: str) -> str:
        exact_type = self._exact_numeric_precision_scale(expr)
        if exact_type is None:
            return self._float_type()
        if self.dialect == Dialect.POSTGRES:
            return "NUMERIC"
        if self.dialect == Dialect.MYSQL:
            precision, scale = exact_type
            result_scale = min(self._max_decimal_scale(), max(4, scale + 4))
            result_precision = min(65, precision + 4)
            return f"DECIMAL({max(result_precision, result_scale)}, {result_scale})"
        return self._float_type()

    def _avg_result_storage_type(self, item: SelectItem) -> str:
        """Match the native AVG result family and fixed-point precision/scale."""
        return self._avg_result_storage_type_for_expr(item.arg_expr or "")

    _PG_INTEGER_TYPE_RE = re.compile(
        r"(?:SMALLINT|INTEGER|INT|INT2|INT4|INT8|BIGINT|SMALLSERIAL|SERIAL|SERIAL2|SERIAL4|SERIAL8|BIGSERIAL)"
    )
    _MYSQL_INTEGER_TYPE_RE = re.compile(
        r"(?:TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT)(?:\s*\(\s*\d+\s*\))?(?:\s+UNSIGNED)?(?:\s+ZEROFILL)?"
    )
    _MYSQL_DECIMAL_TYPE_RE = re.compile(
        r"(?:DECIMAL|NUMERIC|DEC|FIXED)(?:\s*\(\s*\d+\s*(?:,\s*(\d+)\s*)?\))?(?:\s+UNSIGNED)?(?:\s+ZEROFILL)?"
    )

    def _avg_native_sum_scale(self, item: SelectItem) -> Optional[int]:
        """Return the native SUM scale when AVG(x) == SUM(x)/COUNT(x) exactly.

        The engine's AVG for these argument types is, by implementation, the
        engine's exact division of the exact running sum by the non-NULL count:

        * PostgreSQL ``int2``/``int4``/``int8``: ``int8_avg``/``numeric_poly_avg``
          return ``numeric_div(sum::numeric, count::numeric)``; an integer sum
          always has display scale 0.
        * MySQL/MariaDB integer and fixed-point ``DECIMAL(p, s)``:
          ``Item_sum_avg`` returns ``my_decimal_div(sum, count,
          div_precision_increment)`` over an exact DECIMAL sum of scale ``s``,
          which is what the ``/`` operator computes for the same operands.

        For these types the materialized AVG can therefore be derived from the
        SUM and non-NULL COUNT maintained in the same backing row, inside the
        same row-locked statement, with a result identical to the native AVG.
        ``None`` means no such guarantee exists (PostgreSQL ``numeric`` has a
        data-dependent display scale and NaN, approximate types depend on the
        native accumulation order, DISTINCT and UNION-derived projections have
        different maintenance, and expression result types are inferred only
        approximately); those keep the native per-group recomputation.
        """
        if item.agg_func != AggFunc.AVG or item.is_distinct or not item.arg_expr:
            return None
        if getattr(self, "_alg_mode", "single") != "single":
            return None
        source = self._lookup_source_col_def(item.arg_expr)
        if source is None:
            return None
        source_type = re.sub(r"\s+", " ", source.data_type.strip().upper())
        if self.dialect == Dialect.POSTGRES:
            return 0 if self._PG_INTEGER_TYPE_RE.fullmatch(source_type) else None
        if self.dialect == Dialect.MYSQL:
            if self._MYSQL_INTEGER_TYPE_RE.fullmatch(source_type):
                return 0
            match = self._MYSQL_DECIMAL_TYPE_RE.fullmatch(source_type)
            if match:
                return int(match.group(1) or 0)
        return None

    def _avg_from_sum_count_sql(self, item: SelectItem, sum_sql: str, cnt_sql: str) -> str:
        """Division expression deriving AVG from a maintained SUM and COUNT.

        Callers must guard ``cnt_sql = 0`` (the result is NULL for an empty
        group). On PostgreSQL/MySQL this is only used for arguments accepted by
        `_avg_native_sum_scale`, and reproduces the native AVG value and scale.
        """
        if self.dialect == Dialect.POSTGRES:
            return f"(({sum_sql})::NUMERIC / ({cnt_sql})::NUMERIC)"
        if self.dialect == Dialect.MYSQL:
            scale = self._avg_native_sum_scale(item)
            assert scale is not None, "MySQL AVG derivation requires an exact native sum scale"
            # Normalize the stored sum to the native SUM scale so the decimal
            # division sees the same operands as Item_sum_avg.
            return f"(CAST(({sum_sql}) AS DECIMAL(65, {scale})) / ({cnt_sql}))"
        return f"(1.0 * {sum_sql}) / {cnt_sql}"

    def _requires_native_avg_recompute(self, item: SelectItem) -> bool:
        """Use the engine aggregate for AVG on engines with native exact semantics.

        A SUM/COUNT maintenance fold is not equivalent for every exact input:
        PostgreSQL integer division can truncate, numeric display scale is
        data-dependent, and derived UNION projections can have a native result
        type that is lost by the fold. Recomputing the affected group preserves
        the source view's exact precision/scale and approximate rounding
        semantics for those inputs.

        Where `_avg_native_sum_scale` proves the fold exact, AVG is derived from
        the SUM/COUNT columns of the same backing row instead. This is required
        for concurrency correctness, not only speed: a recomputation subquery
        reads the base table through the statement snapshot, which under READ
        COMMITTED predates acquiring the group row lock, so it can miss a
        concurrently committed writer's rows and store a stale AVG next to a
        correct SUM/COUNT. The derived value is computed from the locked row
        itself and is therefore serialized with every other group writer.
        """
        return (
            self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
            and item.agg_func == AggFunc.AVG
            and self._avg_native_sum_scale(item) is None
        )

    def _var_accum_type(
        self, item: Optional[SelectItem] = None, *, squared: bool = False
    ) -> str:
        if self.dialect == Dialect.POSTGRES:
            return "NUMERIC"
        elif self.dialect == Dialect.MYSQL:
            scale = 12
            if item is not None and item.arg_expr:
                exact_type = self._exact_numeric_precision_scale(item.arg_expr)
                if exact_type is not None:
                    scale = max(scale, exact_type[1])
            if squared:
                scale = min(self._max_decimal_scale(), max(12, scale * 2))
            return f"DECIMAL(65, {scale})"
        return "REAL"

    def _var_sq_sql(self, expr: str) -> str:
        if self.dialect == Dialect.POSTGRES:
            return f"((1.0 * ({expr})::NUMERIC) * ({expr})::NUMERIC)"
        return f"(1.0 * ({expr}) * ({expr}))"

    def _normalize_dialect_type(self, dt: str) -> str:
        u = dt.strip().upper()
        if self.dialect == Dialect.POSTGRES:
            if u == "DOUBLE":
                return "DOUBLE PRECISION"
            if u == "CHARACTER VARYING":
                return "VARCHAR(255)"
            if u in ("SMALLSERIAL", "SERIAL2"):
                return "SMALLINT"
            if u in ("SERIAL", "SERIAL4"):
                return "INTEGER"
            if u in ("BIGSERIAL", "SERIAL8"):
                return "BIGINT"
        if self.dialect == Dialect.MYSQL:
            if u == "DOUBLE PRECISION":
                return "DOUBLE"
            if u in ("VARCHAR", "CHAR", "CHARACTER VARYING", "CHARACTER", "TEXT"):
                return "VARCHAR(191)"
        return dt

    def _infer_cte_or_subquery_col_type(self, col_name: str) -> Optional[str]:
        ident = _unquote_ident(col_name, dialect=self.dialect).strip()
        if not ident:
            return None
        try:
            q_ast = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None

        for alias_node in q_ast.find_all(sqlglot_exp.Alias):
            if alias_node.alias.lower() == ident.lower():
                inner = alias_node.this
                if isinstance(inner, (sqlglot_exp.Count, sqlglot_exp.Window)):
                    return self._int_type()
                if isinstance(inner, sqlglot_exp.Sum):
                    arg_expr = inner.this.sql() if inner.this is not None else ""
                    sum_item = SelectItem(
                        raw_expr=inner.sql(),
                        alias=ident,
                        is_aggregate=True,
                        agg_func=AggFunc.SUM,
                        arg_expr=arg_expr,
                    )
                    return self._sum_result_storage_type(sum_item)
                if isinstance(inner, (sqlglot_exp.Min, sqlglot_exp.Max)):
                    if self.dialect == Dialect.SQLITE:
                        return "BLOB"
                    c_arg = self._lookup_col_def(inner.this.sql().strip() if isinstance(inner.this, sqlglot_exp.Expr) else None)
                    if c_arg:
                        return self._normalize_dialect_type(c_arg.data_type)
                    return self._float_type()
                if isinstance(inner, sqlglot_exp.Avg):
                    arg_expr = inner.this.sql() if inner.this is not None else ""
                    return self._avg_result_storage_type_for_expr(arg_expr)
                if isinstance(inner, (sqlglot_exp.AggFunc, sqlglot_exp.Anonymous)) and not isinstance(inner, sqlglot_exp.Count):
                    # Any other aggregate: its verified projection result type
                    # (e.g. PostgreSQL VAR_SAMP(int) is NUMERIC, not DOUBLE).
                    agg_item = _parse_single_agg_expr(
                        inner.sql(_sqlglot_dialect(self.dialect)), ident, dialect=self.dialect
                    )
                    if agg_item is not None and agg_item.agg_func is not None:
                        return self._infer_expr_storage_type(agg_item)
                if isinstance(inner, (sqlglot_exp.AggFunc, sqlglot_exp.Func, sqlglot_exp.Anonymous)):
                    fn_up = (inner.sql_name().upper() if hasattr(inner, "sql_name") else "") or inner.name.upper()
                    if fn_up in ("COUNT", "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "BIT_AND", "BIT_OR", "BIT_XOR", "BITWISEANDAGG", "BITWISEORAGG", "BITWISEXORAGG", "BOOL_AND", "BOOL_OR", "LOGICALAND", "LOGICALOR"):
                        return self._int_type()
                    if fn_up in ("SUM", "TOTAL", "AVG", "VAR_POP", "VARIANCE", "VARIANCE_POP", "VARIANCEPOP", "VAR_SAMP", "VARSAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP", "STDDEVSAMP"):
                        return self._float_type()
                if isinstance(inner, sqlglot_exp.Column) and not isinstance(inner.this, sqlglot_exp.Star):
                    c_def = self._lookup_col_def(inner.name)
                    if c_def:
                        return self._normalize_dialect_type(c_def.data_type)

        for cte in q_ast.find_all(sqlglot_exp.CTE):
            inner_query = cte.this
            while isinstance(inner_query, sqlglot_exp.SetOperation):
                inner_query = inner_query.this
            if not isinstance(inner_query, sqlglot_exp.Select):
                continue

            table_alias = cte.args.get("alias")
            if isinstance(table_alias, sqlglot_exp.TableAlias) and table_alias.columns:
                hdr_cols = [c.name.lower() for c in table_alias.columns]
                if ident.lower() in hdr_cols:
                    idx_h = hdr_cols.index(ident.lower())
                    if idx_h < len(inner_query.expressions):
                        proj_e = inner_query.expressions[idx_h]
                        src_e = proj_e.this if isinstance(proj_e, sqlglot_exp.Alias) else proj_e
                        if isinstance(src_e, sqlglot_exp.Literal):
                            return self._literal_storage_type(src_e)
                        c_src = self._lookup_col_def(src_e.sql().strip())
                        if c_src:
                            return self._normalize_dialect_type(c_src.data_type)
            for proj in inner_query.expressions:
                c_name = proj.alias or (proj.this.name if isinstance(proj, sqlglot_exp.Alias) else proj.name)
                if c_name.lower() == ident.lower():
                    src_node = proj.this if isinstance(proj, sqlglot_exp.Alias) else proj
                    if isinstance(src_node, sqlglot_exp.Literal):
                        return self._literal_storage_type(src_node)
                    src_col = src_node.name or src_node.sql().strip()
                    c_src = self._lookup_col_def(src_col)
                    if c_src:
                        return self._normalize_dialect_type(c_src.data_type)

        if "." in col_name:
            c_base = self._lookup_col_def(ident)
            if c_base:
                return self._normalize_dialect_type(c_base.data_type)

        return None

    # MariaDB integer column display lengths (sign included when signed); the
    # declared display width (INT(5)) does not change arithmetic lengths.
    _MARIADB_INT_DIGITS = {"TINYINT": (3, 3), "SMALLINT": (5, 5), "MEDIUMINT": (8, 8),
                           "INT": (10, 10), "INTEGER": (10, 10), "BIGINT": (19, 20)}
    _MARIADB_INT_COLUMN_RE = re.compile(
        r"(TINYINT|SMALLINT|MEDIUMINT|INTEGER|INT|BIGINT)(?:\s*\(\s*\d+\s*\))?(\s+UNSIGNED)?(?:\s+ZEROFILL)?"
    )

    def _mariadb_int_arithmetic_length(self, root: Any) -> Optional[int]:
        """MariaDB's display length of an integer arithmetic expression.

        MariaDB types an integer expression result as INT when its display
        length is below 10 characters and as BIGINT otherwise, so `i % 7` over
        INT is BIGINT while `s % 2` over SMALLINT is INT. The length follows
        the operators' rules (digits, sign): column of the integer type;
        literal (digits, unsigned); -x (d + 1 if x is signed else d, signed);
        x + y, x - y (max(d) + 1, either signed); x * y (sum(d), either
        signed); x % y (max(d), signedness of x); top-level x DIV y (length
        d(x)). Verified natively on MariaDB 10.11 and 12.3 (test_native_bulk_
        statements.py::test_integer_expression_group_key_result_type).
        Returns None when the top-level node is not integer arithmetic or an
        operand is outside these rules, leaving the caller's default.
        """
        def shape(node: Any) -> Optional[Tuple[int, bool]]:
            kind = node.__class__.__name__
            if kind == "Paren":
                return shape(node.this)
            if kind == "Column":
                c_def = self._lookup_col_def(node.sql()) or self._lookup_col_def(node.name)
                if c_def is None:
                    return None
                m = self._MARIADB_INT_COLUMN_RE.fullmatch(re.sub(r"\s+", " ", c_def.data_type.strip().upper()))
                if m is None:
                    return None
                signed_d, unsigned_d = self._MARIADB_INT_DIGITS[m.group(1)]
                return (unsigned_d, False) if m.group(2) else (signed_d, True)
            if kind == "Literal":
                text = str(node.this)
                return (len(text), False) if not node.args.get("is_string") and text.isdigit() else None
            if kind == "Neg":
                inner = node.this
                while inner.__class__.__name__ == "Paren":
                    inner = inner.this
                if inner.__class__.__name__ == "Literal":
                    lit = shape(inner)
                    return (lit[0], True) if lit else None
                x = shape(inner)
                return (x[0] + 1, True) if x and x[1] else ((x[0], True) if x else None)
            if kind in ("Add", "Sub", "Mul", "Mod"):
                a, b = shape(node.this), shape(node.expression)
                if a is None or b is None:
                    return None
                if kind == "Mul":
                    return (a[0] + b[0], a[1] or b[1])
                if kind == "Mod":
                    return (max(a[0], b[0]), a[1])
                return (max(a[0], b[0]) + 1, a[1] or b[1])
            return None

        top = root
        while top.__class__.__name__ == "Paren":
            top = top.this
        kind = top.__class__.__name__
        if kind == "IntDiv":
            left = shape(top.this)
            return left[0] if left and shape(top.expression) else None
        if kind not in ("Add", "Sub", "Mul", "Mod", "Neg"):
            return None
        s = shape(top)
        return s[0] + (1 if s[1] else 0) if s else None

    def _infer_ast_expr_storage_type(self, expr: str) -> Optional[str]:
        """Infer storage type across dialects from AST nodes (CASE, COALESCE, arithmetic, scalars)."""
        try:
            root = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None

        ranks = {
            "SMALLINT": 1,
            "INTEGER": 2,
            "BIGINT": 3,
            "NUMERIC": 4,
            "REAL": 5,
            "DOUBLE PRECISION": 6,
        }
        str_fn_names = {"UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM", "SUBSTR", "SUBSTRING", "CONCAT", "CONCAT_WS", "REPLACE"}
        int_fn_names = {"LENGTH", "CHAR_LENGTH", "CHARACTER_LENGTH", "INSTR", "STRPOS"}
        agg_fn_names = {
            "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX", "VAR_POP", "VAR_SAMP", "STDDEV_POP",
            "STDDEV_SAMP", "VARIANCE", "STDDEV", "STD", "BIT_AND", "BIT_OR", "BIT_XOR",
            "BOOL_AND", "BOOL_OR", "EVERY",
        }

        def canonical_type(data_type: str) -> Optional[str]:
            upper = re.sub(r"\s*\(.*\)\s*$", "", data_type.upper()).strip()
            if upper in ("SMALLINT", "INT2"):
                return "SMALLINT"
            if upper in ("INTEGER", "INT", "INT4", "MEDIUMINT", "TINYINT"):
                return "INTEGER"
            if upper in ("BIGINT", "INT8", "BIGINT UNSIGNED", "INT UNSIGNED"):
                return "BIGINT"
            if upper in ("NUMERIC", "DECIMAL", "DEC", "FIXED"):
                return "NUMERIC"
            if upper in ("REAL", "FLOAT4"):
                return "REAL"
            if upper in ("DOUBLE PRECISION", "DOUBLE", "FLOAT8", "FLOAT"):
                return "DOUBLE PRECISION"
            if upper in ("TEXT", "VARCHAR", "CHAR", "CHARACTER VARYING", "CHARACTER", "NVARCHAR", "NCHAR", "CLOB"):
                return "TEXT"
            if upper in ("BOOLEAN", "BOOL"):
                return "BOOLEAN"
            if upper in ("DATE", "DATETIME", "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "TIMESTAMP WITHOUT TIME ZONE"):
                return upper
            return None

        def merge(types: Sequence[Optional[str]]) -> Optional[str]:
            known = [value for value in types if value is not None and value != "UNKNOWN"]
            if not known:
                return None
            if any(value not in ranks for value in known):
                if all(value == known[0] for value in known):
                    return known[0]
                if all(value in ("TEXT", "VARCHAR") for value in known):
                    return "TEXT"
                return None
            return max(known, key=lambda value: ranks[value])

        def infer(node: Any) -> Optional[str]:
            if node is None:
                return "UNKNOWN"
            kind = node.__class__.__name__
            if kind == "Paren":
                return infer(node.this)
            if kind == "Null":
                return "UNKNOWN"
            if kind == "Boolean":
                return "BOOLEAN"
            if kind == "Or" and self.dialect == Dialect.MYSQL and "||" in expr:
                # Under MySQL/MariaDB PIPES_AS_CONCAT, `||` is string concatenation
                # even though sqlglot parses it as logical Or.
                return None
            if kind in (
                "EQ",
                "NEQ",
                "LT",
                "LTE",
                "GT",
                "GTE",
                "Is",
                "In",
                "Between",
                "Like",
                "ILike",
                "And",
                "Or",
                "Not",
                "Exists",
            ):
                return "BOOLEAN"
            if kind == "Abs":
                return infer(node.this)
            if kind == "Column":
                column = self._lookup_col_def(node.sql())
                if column is None:
                    column = self._lookup_col_def(node.name)
                if column is not None:
                    return canonical_type(column.data_type)
                cte_t = self._infer_cte_or_subquery_col_type(node.sql()) or self._infer_cte_or_subquery_col_type(node.name)
                return canonical_type(cte_t) if cte_t else None
            if kind == "Literal":
                if node.args.get("is_string"):
                    return "TEXT"
                value = str(node.this)
                if re.fullmatch(r"[+-]?\d+", value):
                    number = int(value)
                    if -(2 ** 31) <= number < 2 ** 31:
                        return "INTEGER"
                    if -(2 ** 63) <= number < 2 ** 63:
                        return "BIGINT"
                    return "NUMERIC"
                if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", value):
                    return "DOUBLE PRECISION" if ("e" in value.lower() or self.dialect == Dialect.SQLITE) else "NUMERIC"
                return None
            if kind in ("Neg", "Positive"):
                return infer(node.this)
            if kind == "Window":
                inner = node.this
                inner_kind = inner.__class__.__name__ if inner is not None else ""
                if inner_kind in ("RowNumber", "Rank", "DenseRank", "Ntile", "Count"):
                    return "BIGINT"
                if inner_kind in ("PercentRank", "CumeDist"):
                    return "DOUBLE PRECISION"
                if inner is not None and inner_kind in ("Lag", "Lead", "FirstValue", "LastValue", "NthValue"):
                    return infer(inner.this)
                win_t = self._window_aggregate_storage_type(node.sql(_sqlglot_dialect(self.dialect)))
                return canonical_type(win_t) if win_t else None
            if kind == "Cast":
                target = node.args.get("to")
                return canonical_type(target.sql()) if target is not None else None
            if kind in ("Coalesce", "IfNull"):
                vals = [infer(ch) for ch in node.iter_expressions()]
                return merge(vals)
            if kind == "Nullif":
                return infer(node.this)
            if kind == "If":
                t_br = infer(node.args.get("true"))
                f_br = infer(node.args.get("false"))
                return merge((t_br, f_br))
            if kind == "Case":
                branches: List[Optional[str]] = []
                for branch in node.args.get("ifs") or ():
                    value = branch.args.get("true")
                    if value is not None:
                        inferred = infer(value)
                        if inferred is None:
                            return None
                        branches.append(inferred)
                default = node.args.get("default")
                if default is not None:
                    inferred = infer(default)
                    if inferred is None:
                        return None
                    branches.append(inferred)
                return merge(branches)
            if kind in ("Add", "Sub", "Mul", "Div", "Mod", "IntDiv"):
                left, right = infer(node.this), infer(node.expression)
                if left is None or right is None:
                    return None
                merged = merge((left, right))
                if kind == "Div" and merged in ("SMALLINT", "INTEGER", "BIGINT") and self.dialect != Dialect.POSTGRES:
                    return "DOUBLE PRECISION"
                if kind == "IntDiv":
                    # MySQL/MariaDB DIV: integer result (BIGINT on MySQL;
                    # MariaDB's width rule is applied below).
                    return "BIGINT" if merged in ("SMALLINT", "INTEGER", "BIGINT") else None
                if (
                    kind in ("Add", "Sub", "Mul")
                    and merged in ("SMALLINT", "INTEGER")
                    and self.dialect != Dialect.POSTGRES
                ):
                    # PostgreSQL integer operators keep the wider operand type
                    # (int4 + int4 is int4, overflow raises); other engines'
                    # integer arithmetic is 64-bit (MariaDB's width rule is
                    # applied below).
                    return "BIGINT"
                if merged == "REAL":
                    return "DOUBLE PRECISION"
                return merged
            if kind == "Window" and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                # Window SUM/AVG/COUNT keep the engine's native result type
                # (stored as DOUBLE they lost the exact NUMERIC/DECIMAL value:
                # PostgreSQL AVG(int) 53.3333333333333333 vs 53.333333333333336).
                fn = node.this
                if fn is None:
                    return None
                fn_kind = fn.__class__.__name__
                if fn_kind == "Count":
                    return "BIGINT"
                if fn_kind in ("Sum", "Avg"):
                    arg = fn.this
                    if isinstance(arg, sqlglot_exp.Distinct):
                        arg = arg.expressions[0] if arg.expressions else None
                    arg_t = infer(arg) if arg is not None else None
                    if arg_t in ("REAL", "DOUBLE PRECISION"):
                        if self.dialect == Dialect.POSTGRES and fn_kind == "Sum":
                            return arg_t
                        return "DOUBLE PRECISION"
                    if arg_t in ("SMALLINT", "INTEGER", "BIGINT", "NUMERIC"):
                        if self.dialect == Dialect.MYSQL and arg is not None:
                            ps = self._exact_numeric_precision_scale(
                                arg.sql(dialect=_sqlglot_dialect(self.dialect))
                            )
                            if ps is None:
                                return None
                            scale = int(ps[1])
                            if fn_kind == "Avg":
                                scale += self.engine_version.mariadb_div_precision_increment
                            return f"MYSQL_WINDOW_DECIMAL:{min(30, scale)}"
                        if fn_kind == "Sum" and arg_t in ("SMALLINT", "INTEGER"):
                            return "BIGINT"
                        return "NUMERIC"
                    return None
            if isinstance(node, sqlglot_exp.AggFunc) or (
                kind == "Anonymous" and node.name.upper() in agg_fn_names
            ):
                # An aggregate inside a larger expression has the same result
                # type as the projected aggregate (e.g. PostgreSQL
                # 1.0 * SUM(int) is NUMERIC, not DOUBLE PRECISION).
                agg_item = _parse_single_agg_expr(
                    node.sql(_sqlglot_dialect(self.dialect)), "__agg", dialect=self.dialect
                )
                if agg_item is None or agg_item.agg_func is None:
                    return None
                return canonical_type(self._infer_expr_storage_type(agg_item))
            if isinstance(node, sqlglot_exp.Func):
                fn_name = (node.sql_name() or kind).upper()
                if fn_name in str_fn_names:
                    return "TEXT"
                if fn_name in int_fn_names:
                    return "BIGINT"
                if fn_name in ("LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE"):
                    return infer(node.this)
            return None

        inferred = infer(root)
        if inferred is None or inferred == "UNKNOWN":
            return None
        if inferred.startswith("MYSQL_WINDOW_DECIMAL:"):
            # MySQL/MariaDB window SUM/AVG of exact arguments return DECIMAL
            # with the argument's scale (AVG: plus div_precision_increment);
            # 65 digits hold every such value.
            return f"DECIMAL(65, {inferred.split(':', 1)[1]})"
        if inferred == "TEXT" and self.dialect == Dialect.POSTGRES:
            return self._pg_string_result_type(root) or "TEXT"
        if inferred == "NUMERIC":
            if self.dialect == Dialect.POSTGRES:
                return "NUMERIC"
            if self.dialect == Dialect.MYSQL:
                ps = self._exact_numeric_precision_scale(expr)
                if ps is not None:
                    return f"DECIMAL({ps[0]}, {ps[1]})"
                return "DECIMAL(38, 10)"
            return "NUMERIC"
        if inferred == "BOOLEAN":
            return "BOOLEAN" if self.dialect == Dialect.POSTGRES else "INTEGER"
        if inferred in ("SMALLINT", "INTEGER", "BIGINT") and self.dialect != Dialect.POSTGRES:
            if self.dialect == Dialect.MYSQL and self.subdialect == EngineSubdialect.MARIADB:
                arith_len = self._mariadb_int_arithmetic_length(root)
                if arith_len is not None:
                    return "BIGINT" if arith_len >= 10 else "INT"
                if inferred == "INTEGER":
                    return "INT"
            return self._int_type()
        if inferred in ("REAL", "DOUBLE PRECISION") and self.dialect != Dialect.POSTGRES:
            return self._float_type()
        return self._normalize_dialect_type(inferred)


    def _literal_storage_type(self, literal: Any) -> str:
        """Storage type of a CTE/derived-table column defined by a bare literal.

        PostgreSQL types an integer literal as integer (bigint/numeric only
        when out of range) and a decimal literal as numeric; a string literal
        resolves to text. Other engines keep the previous mapping.
        """
        text = str(literal.this)
        if self.dialect == Dialect.POSTGRES:
            if literal.args.get("is_string"):
                return "TEXT"
            if re.fullmatch(r"[+-]?\d+", text):
                n = int(text)
                if -(2 ** 31) <= n < 2 ** 31:
                    return "INTEGER"
                return "BIGINT" if -(2 ** 63) <= n < 2 ** 63 else "NUMERIC"
            return "DOUBLE PRECISION" if "e" in text.lower() else "NUMERIC"
        if (
            self.subdialect == EngineSubdialect.MARIADB
            and re.fullmatch(r"[+-]?\d+", text)
            and -(2 ** 31) <= int(text) < 2 ** 31
        ):
            # MariaDB types a small integer literal column as INT (a recursive
            # CTE's `0` seed column is INT; checked natively).
            return "INT"
        return self._float_type() if "." in text else self._int_type()

    def _pg_string_result_type(self, node: Any) -> Optional[str]:
        """PostgreSQL type of a string-valued COALESCE/CASE/NULLIF expression.

        Untyped string literals adopt the type of the first typed input in
        PostgreSQL's resolution order (COALESCE: argument order; CASE: ELSE
        first) with its modifier dropped (COALESCE(varchar(32), 'x') is character varying,
        COALESCE(char(3), 'x') is bpchar); string functions return text; all
        literal branches resolve to text.
        """
        kind = node.__class__.__name__
        if kind == "Paren":
            return self._pg_string_result_type(node.this)
        if kind == "Literal":
            return None
        if kind == "Null":
            return None
        if kind == "Column":
            col = self._lookup_col_def(node.sql()) or self._lookup_col_def(node.name)
            if col is None:
                return "TEXT"
            base = re.sub(r"\s*\(.*\)\s*$", "", col.data_type.strip().upper())
            if base in ("VARCHAR", "CHARACTER VARYING"):
                return "VARCHAR"
            if base in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
                return "BPCHAR"
            return "TEXT"
        if kind == "Cast":
            target = node.args.get("to")
            return self._pg_string_result_type_from_name(target.sql() if target is not None else "")
        branches: List[Any] = []
        if kind in ("Coalesce", "IfNull"):
            branches = list(node.iter_expressions())
        elif kind == "Nullif":
            branches = [node.this]
        elif kind == "Case":
            # PostgreSQL resolves a CASE with its ELSE result first, then the
            # THEN results in order (transformCaseExpr), and keeps the first
            # typed string input: varchar, bpchar and text convert implicitly
            # into each other (checked natively: CASE ... THEN varchar ELSE
            # char(3) END is bpchar, the reverse is varchar).
            branches = [node.args.get("default")]
            branches += [b.args.get("true") for b in (node.args.get("ifs") or ())]
        elif kind == "If":
            branches = [node.args.get("true"), node.args.get("false")]
        else:
            return "TEXT"
        for branch in branches:
            if branch is None:
                continue
            t = self._pg_string_result_type(branch)
            if t is not None:
                return t
        return None

    @staticmethod
    def _pg_string_result_type_from_name(type_name: str) -> str:
        base = re.sub(r"\s*\(.*\)\s*$", "", type_name.strip().upper())
        if base in ("VARCHAR", "CHARACTER VARYING"):
            return "VARCHAR"
        if base in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
            return "BPCHAR"
        return "TEXT"

    def _is_case_over_int_literals(self, expr: str) -> bool:
        try:
            root = parse_sql_ast(_unwrap_outer_parens(expr, dialect=self.dialect), dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return False
        if not isinstance(root, sqlglot_exp.Case):
            return False
        values = [b.args.get("true") for b in (root.args.get("ifs") or ())] + [root.args.get("default")]

        def small_int(v: Any) -> bool:
            if isinstance(v, sqlglot_exp.Neg):
                v = v.this
            return (
                isinstance(v, sqlglot_exp.Literal)
                and not v.args.get("is_string")
                and re.fullmatch(r"\d+", str(v.this)) is not None
                and int(str(v.this)) < 2 ** 31
            )

        return all(small_int(v) for v in values if v is not None)

    def _window_aggregate_storage_type(self, expr: str) -> Optional[str]:
        """Storage type of `<aggregate>(...) OVER (...)`: the aggregate's own result type.

        An aggregate used as a window function returns the same type as the
        plain aggregate (SUM(int) is BIGINT/NUMERIC/DECIMAL, VAR_SAMP(int) is
        NUMERIC on PostgreSQL, MIN/MAX keep the argument type), so the
        verified aggregate inference is reused. Ranking/offset window
        functions are not aggregates and return None.
        """
        try:
            node = parse_sql_ast(_unwrap_outer_parens(expr, dialect=self.dialect), dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(node, sqlglot_exp.Window) or node.this is None:
            return None
        inner_sql = node.this.sql(_sqlglot_dialect(self.dialect))
        agg_item = _parse_single_agg_expr(inner_sql, "__w", dialect=self.dialect)
        if agg_item is None or agg_item.agg_func is None:
            return None
        result = self._infer_expr_storage_type(agg_item)
        if (
            self.dialect == Dialect.MYSQL
            and self.subdialect != EngineSubdialect.MARIADB
            and agg_item.agg_func in (AggFunc.MIN, AggFunc.MAX)
        ):
            # MySQL widens window MIN/MAX of integers (checked natively on 8.4:
            # TINYINT/SMALLINT/MEDIUMINT -> INT, INT -> BIGINT, INT UNSIGNED ->
            # BIGINT UNSIGNED); MariaDB keeps the argument type.
            u = re.sub(r"\s*\(\d+\)", "", result.strip().upper())
            if u in ("TINYINT", "SMALLINT", "MEDIUMINT"):
                return "INT"
            if u in ("INT", "INTEGER"):
                return "BIGINT"
            if u in ("INT UNSIGNED", "INTEGER UNSIGNED"):
                return "BIGINT UNSIGNED"
        return result

    def _infer_expr_storage_type(self, item: SelectItem) -> str:
        if not item.is_aggregate:
            c_def = self._lookup_col_def(item.raw_expr)
            if c_def:
                return self._normalize_dialect_type(c_def.data_type)
            cte_t = self._infer_cte_or_subquery_col_type(item.raw_expr)
            if cte_t:
                return cte_t
            window_t = self._window_aggregate_storage_type(item.raw_expr)
            if window_t is not None:
                return window_t
            if (
                self.dialect == Dialect.MYSQL
                and self.subdialect != EngineSubdialect.MARIADB
                and self.view.group_by_cols
                and self._is_case_over_int_literals(item.raw_expr)
            ):
                # MySQL sends a CASE over small integer literals as BIGINT when
                # projected directly but as INT once grouped through a
                # temporary table (both checked natively).
                return "INT"
            # Remove only balanced outer parentheses before applying the simple
            # expression-type heuristics below. CASE expressions emitted by the
            # AST generators are parenthesized, and leaving those wrappers in
            # place incorrectly falls through to TEXT affinity on SQLite.
            raw_expr = item.raw_expr.strip()
            while raw_expr.startswith("(") and raw_expr.endswith(")"):
                try:
                    tokens = _tokenize_sql(
                        raw_expr, read=_sqlglot_dialect(self.dialect)
                    )
                except sqlglot.errors.SqlglotError:
                    break
                if not tokens or tokens[0].text != "(" or tokens[-1].text != ")":
                    break
                depth = 0
                closes_before_end = False
                for token_idx, token in enumerate(tokens):
                    if token.text == "(":
                        depth += 1
                    elif token.text == ")":
                        depth -= 1
                        if depth == 0 and token_idx != len(tokens) - 1:
                            closes_before_end = True
                            break
                if closes_before_end or depth != 0:
                    break
                raw_expr = raw_expr[tokens[0].end + 1:tokens[-1].start].strip()
            raw_u = raw_expr.upper()
            ast_inferred = self._infer_ast_expr_storage_type(raw_expr)
            if ast_inferred is not None:
                return ast_inferred
            m_coal = re.match(
                r'^(?:COALESCE|IFNULL)\s*\(\s*([A-Za-z0-9_."`\[\]]+)\s*,',
                item.raw_expr.strip(),
                re.IGNORECASE,
            )
            if m_coal:
                c_inner = self._lookup_col_def(m_coal.group(1))
                if c_inner:
                    return self._normalize_dialect_type(c_inner.data_type)
            m_win_col = re.match(
                r'^(?:LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE)\s*\(\s*([A-Za-z0-9_."`\[\]]+)',
                item.raw_expr.strip(),
                re.IGNORECASE,
            )
            if m_win_col:
                c_win = self._lookup_col_def(m_win_col.group(1))
                if c_win:
                    return self._normalize_dialect_type(c_win.data_type)
            if raw_u.startswith("CASE ") and self.dialect == Dialect.POSTGRES:
                native_case_type = self._postgres_case_result_storage_type(raw_expr)
                if native_case_type is not None:
                    return native_case_type
            if raw_u.startswith("CASE ") and not re.search(r"\b(?:THEN|ELSE)\s*'", raw_u):
                num_branches = re.findall(r"\b(?:THEN|ELSE)\s+(-?\d+(?:\.\d+)?)\b", raw_u)
                if num_branches:
                    return self._float_type() if any("." in nb for nb in num_branches) else self._int_type()
            if re.match(r"^(?:ROW_NUMBER|RANK|DENSE_RANK|NTILE|COUNT|LENGTH|CHAR_LENGTH|INSTR)\s*\(", raw_u):
                return self._int_type()
            if re.match(r"^(?:PERCENT_RANK|CUME_DIST|AVG|SUM|TOTAL|ROUND|ABS)\s*\(", raw_u):
                return self._float_type()
            if raw_u.startswith("(") and "SELECT" in raw_u:
                if "COUNT(" in raw_u:
                    return self._int_type()
                if any(fn in raw_u for fn in ("SUM(", "AVG(", "TOTAL(", "MIN(", "MAX(")):
                    return self._float_type()
            if "%" in raw_u:
                return self._int_type()
            if any(op in raw_u for op in ("*", "/", "+", "-")):
                if "/" not in raw_u and not re.search(r"\d+\.\d+", raw_u):
                    cols_found = [
                        self._lookup_col_def(m_id.group(1))
                        for m_id in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_.]*)\b", item.raw_expr)
                    ]
                    valid_cols = [c for c in cols_found if c is not None]
                    if valid_cols and all("INT" in c.data_type.upper() for c in valid_cols):
                        return self._int_type()
                return self._float_type()
            return "VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT"
        if item.agg_func is None and item.raw_expr and self.dialect == Dialect.SQLITE:
            # A compound aggregate expression's SQLite storage class is chosen
            # per value at run time (COALESCE(SUM(int), 0) is INTEGER, with a
            # REAL input it is REAL); a column without affinity keeps it.
            return "BLOB"
        if item.agg_func is None and item.raw_expr and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            # A compound expression over aggregates (COALESCE(SUM(x), 0),
            # 1.0 * SUM(x), ...): type it from its AST, aggregates included.
            # (Validated natively on PostgreSQL; MySQL/SQLite keep the
            # previous inference until their native result types are modelled.)
            compound_t = self._infer_ast_expr_storage_type(_unwrap_outer_parens(item.raw_expr, dialect=self.dialect))
            if compound_t is not None:
                return compound_t
        if item.agg_func in (
            AggFunc.BIT_AND,
            AggFunc.BIT_OR,
            AggFunc.BIT_XOR,
        ):
            if self.dialect == Dialect.POSTGRES and item.arg_expr:
                # PostgreSQL's bit aggregates return their argument's integer
                # type (bit_and(int4) is int4).
                c_def = self._lookup_col_def(item.arg_expr)
                arg_t = (
                    c_def.data_type if c_def else (self._infer_ast_expr_storage_type(item.arg_expr) or "")
                ).strip().upper()
                if arg_t in ("SMALLINT", "INT2"):
                    return "SMALLINT"
                if arg_t in ("INTEGER", "INT", "INT4", "SERIAL"):
                    return "INTEGER"
            return "BIGINT UNSIGNED" if self.dialect == Dialect.MYSQL else self._int_type()
        if item.agg_func in (
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
        ):
            return self._int_type()
        if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
            return "BOOLEAN" if self.dialect == Dialect.POSTGRES else "INTEGER"
        if item.agg_func == AggFunc.AVG:
            return self._avg_result_storage_type(item)
        if item.agg_func == AggFunc.TOTAL:
            return self._float_type()
        if item.agg_func in (
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
        ):
            return self._statistical_result_storage_type(item)
        if item.agg_func in (
            AggFunc.GROUP_CONCAT,
            AggFunc.STRING_AGG,
            AggFunc.JSON_GROUP_ARRAY,
            AggFunc.JSON_GROUP_OBJECT,
            AggFunc.ARRAY_AGG,
        ):
            return "TEXT"
        if item.agg_func == AggFunc.SUM:
            return self._sum_result_storage_type(item)
        if item.agg_func in (AggFunc.MIN, AggFunc.MAX) and item.arg_expr:
            if self.dialect == Dialect.SQLITE:
                # SQLite MIN/MAX return the winning value with its exact
                # runtime storage class; declaring any non-BLOB affinity would
                # coerce values (e.g. REAL 4000.0 -> INTEGER 4000 or REAL -> TEXT).
                return "BLOB"
            c_def = self._lookup_col_def(item.arg_expr)
            if c_def:
                return self._minmax_result_type(self._normalize_dialect_type(c_def.data_type))
            arg_inferred = self._infer_expr_storage_type(
                SelectItem(raw_expr=item.arg_expr, alias=item.alias, is_aggregate=False)
            )
            if self.dialect == Dialect.MYSQL and arg_inferred and re.fullmatch(
                r"(?:TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT)", arg_inferred.strip().upper()
            ):
                # MySQL/MariaDB MIN/MAX of an integer expression (not a bare
                # column) is BIGINT even where the expression itself is INT.
                return "BIGINT"
            if arg_inferred:
                return self._minmax_result_type(arg_inferred)
        if self.dialect == Dialect.SQLITE and item.agg_func is None:
            # Compound aggregate expressions (e.g. COUNT(*) + 1, COALESCE(MAX(x), 0))
            # have no affinity in SQLite live views; falling through to REAL would
            # coerce integer results to REAL.
            return "BLOB"
        return self._float_type()

    @staticmethod
    def _sqlite_canonical_pk_term(expr: str) -> str:
        s = expr.strip()
        if re.fullmatch(r"'(?:''|[^'])*'", s):
            return f"quote({s})"
        # SQLite GROUP BY / DISTINCT equate INTEGER and REAL when numerically
        # equal (e.g. 1 and 1.0, 0 and -0.0, 2^53 and 2^53.0, -2^63 and -2^63.0),
        # whereas quote(1) != quote(1.0). Canonicalize exact int64-valued REALs
        # to INTEGER before quote().
        e = f"({expr})"
        return (
            f"quote(CASE WHEN typeof({e}) = 'real' "
            f"AND {e} >= -9223372036854775808.0 AND {e} < 9223372036854775808.0 "
            f"AND CAST({e} AS INTEGER) = {e} THEN CAST({e} AS INTEGER) ELSE {e} END)"
        )

    def _minmax_result_type(self, arg_type: str) -> str:
        """MIN/MAX result type for an argument type.

        PostgreSQL's MIN/MAX are declared over unmodified base types, so the
        result drops the argument's type modifier and character types resolve
        to their aggregate's type: numeric(12,3) -> numeric, varchar(10) ->
        text, char(5) -> bpchar, timestamp(0) -> timestamp. MySQL/MariaDB and
        SQLite keep the argument type.
        """
        if (
            self.dialect == Dialect.MYSQL
            and self.subdialect != EngineSubdialect.MARIADB
            and (self.raw_view.has_cte or self.raw_view.has_subquery_in_from)
            and re.match(r"^\s*(TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER)\b", arg_type, re.I)
            and "UNSIGNED" not in arg_type.upper()
        ):
            # MySQL widens MIN/MAX of a signed integer column read through a
            # CTE or derived table to BIGINT (checked natively).
            return "BIGINT"
        if self.dialect != Dialect.POSTGRES:
            return arg_type
        u = re.sub(r"\s+", " ", arg_type.strip().upper())
        base = re.sub(r"\s*\([^)]*\)", "", u).strip()
        if base in ("NUMERIC", "DECIMAL", "DEC"):
            return "NUMERIC"
        if base in ("VARCHAR", "CHARACTER VARYING", "TEXT"):
            return "TEXT"
        if base in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
            return "BPCHAR"
        if base in ("TIMESTAMP", "TIMESTAMP WITHOUT TIME ZONE"):
            return "TIMESTAMP"
        if base in ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"):
            return "TIMESTAMPTZ"
        if base in ("TIME", "TIME WITHOUT TIME ZONE"):
            return "TIME"
        if base in ("TIMETZ", "TIME WITH TIME ZONE"):
            return "TIMETZ"
        if base.startswith("INTERVAL"):
            return "INTERVAL"
        return arg_type

    def _null_safe_eq(self, left_expr: str, right_expr: str, is_nullable: bool = True) -> str:
        if not is_nullable:
            return f"{left_expr} = {right_expr}"
        if self.dialect == Dialect.SQLITE:
            return f"{left_expr} IS {right_expr}"
        elif self.dialect == Dialect.POSTGRES:
            return f"{left_expr} IS NOT DISTINCT FROM {right_expr}"
        elif self.dialect == Dialect.MYSQL:
            return f"{left_expr} <=> {right_expr}"
        return f"{left_expr} IS {right_expr}"

    def _bitwise_xor_sql(self, a: str, b: str) -> str:
        if self.dialect == Dialect.POSTGRES:
            return f"({a} # {b})"
        elif self.dialect == Dialect.MYSQL:
            return f"({a} ^ {b})"
        return f"(({a} | {b}) - ({a} & {b}))"

    def _canonical_group_pk_type(self) -> str:
        if self.dialect == Dialect.MYSQL:
            return "VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
        return "TEXT"

    def _canonical_group_pk_expr(self, group_exprs: Sequence[str]) -> str:
        def _clean_pk_arg(e: str) -> str:
            s = e.strip()
            if not s:
                return s
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
                return self._q(s)
            m_dot = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", s)
            if m_dot:
                return f"{m_dot.group(1)}.{self._q(m_dot.group(2))}"
            return s

        cleaned_exprs = [_clean_pk_arg(e) for e in group_exprs]
        if self.dialect == Dialect.SQLITE:
            return " || '|' || ".join(self._sqlite_canonical_pk_term(e) for e in cleaned_exprs)
        elif self.dialect == Dialect.POSTGRES:
            # Each member goes through the view's canonical-key-text function
            # (_pg_key_text_fn_ddl), so keys equal under their type's equality
            # (NUMERIC 1.0 and 1.00, FLOAT8 0 and -0, ...) get the same text in
            # any session. The planner creates the function with the plan.
            fn = self._pg_key_text_fn()
            self._pg_key_text_fn_used = True
            pg_parts = []
            for e in cleaned_exprs:
                s = e.strip()
                if s.upper() == "NULL":
                    pg_parts.append("NULL::TEXT")
                elif s.startswith("'") and s.endswith("'"):
                    pg_parts.append(f"{fn}(CAST({e} AS TEXT))")
                else:
                    pg_parts.append(f"{fn}({e})")
            return f"ROW({', '.join(pg_parts)})::TEXT"
        elif self.dialect == Dialect.MYSQL:
            parts = [
                f"IF({e} IS NULL, 'N', CONCAT('V', OCTET_LENGTH(CAST({e} AS CHAR CHARACTER SET utf8mb4)), ':', CAST({e} AS CHAR CHARACTER SET utf8mb4)))"
                for e in cleaned_exprs
            ]
            return f"CONCAT_WS('|', {', '.join(parts)}) COLLATE utf8mb4_bin"
        return " || '|' || ".join(self._sqlite_canonical_pk_term(e) for e in cleaned_exprs)

    def _group_by_exprs_for_query(
        self, group_exprs: Sequence[str], include_canonical_pk: bool = True
    ) -> List[str]:
        def _clean_grp(g: str) -> str:
            s = g.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
                return self._q(s)
            m_dot = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$", s)
            if m_dot:
                return f"{self._q(m_dot.group(1))}.{self._q(m_dot.group(2))}"
            return s

        cleaned = [_clean_grp(g) for g in group_exprs]
        if (
            include_canonical_pk
            and self.dialect == Dialect.MYSQL
            and cleaned
            and any(
                not re.match(r"^(?:[A-Za-z_`\"][A-Za-z0-9_`\".]*|'[^']*')$", g.strip())
                for g in cleaned
            )
        ):
            return [self._canonical_group_pk_expr(cleaned), *cleaned]
        return list(cleaned)

    def _group_col_alias(self, g_expr: str) -> str:
        uall_map = getattr(self, "_uall_grp_alias_map", None)
        if uall_map and g_expr in uall_map:
            return uall_map[g_expr]
        for it in self.view.select_items:
            if not it.is_aggregate and (
                it.raw_expr.strip().lower() == g_expr.strip().lower()
                or (
                    it.source_col
                    and self._is_simple_col_ref(g_expr)
                    and it.source_col.lower() == _unquote_ident(g_expr, dialect=self.dialect).lower()
                )
            ):
                return it.alias
        if self._is_simple_col_ref(g_expr):
            if "." in g_expr.strip():
                return f"__mv_grp_{_sanitize_ident(g_expr)}"
            return f"__mv_grp_{_unquote_ident(g_expr, dialect=self.dialect)}"
        return f"__mv_grp_{_sanitize_ident(g_expr)}"

    def _has_nullable_group_col(self) -> bool:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return False
        mode = getattr(self, "_alg_mode", None)
        if mode == "union_all_agg":
            uall_flag = getattr(self, "_uall_has_nullable_grp", None)
            if uall_flag is not None:
                return bool(uall_flag)
            return True
        seen_aliases: Set[str] = set()
        for g in grp_exprs:
            c_def = (
                (self._lookup_col_def(self._qualify_join_expr(g)) or self._lookup_col_def(g))
                if mode == "join"
                else self._lookup_col_def(g)
            )
            if c_def is None or not c_def.not_null:
                return True
            al = self._group_col_alias(g).lower()
            if al in seen_aliases:
                return True
            seen_aliases.add(al)
        return False

    def _tier1_pk_cols(self) -> List[str]:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return ["__mv_global_id"]
        if self._has_nullable_group_col():
            return ["__mv_pk"]
        return list(dict.fromkeys(self._group_col_alias(g) for g in grp_exprs))

    def _rewrite_expr_for_record(self, expr: str, record_prefix: Optional[str]) -> str:
        base_t = self.view.base_table
        base_alias = self.view.joins[0].alias if self.view.joins else base_t
        quals_lower = {q.lower() for q in (base_t, base_alias) if q}
        col_names_lower = (
            {c.lower(): c for c in self.primary_schema.columns}
            if self.primary_schema
            else {}
        )
        try:
            toks = _tokenize_sql(expr, read=_sqlglot_dialect(self.dialect))
        except Exception:
            return expr
        out: List[str] = []
        i = 0
        n = len(toks)
        cursor = 0
        while i < n:
            tk = toks[i]
            if (
                _is_word_or_ident_token(tk)
                and i + 2 < n
                and toks[i + 1].token_type == SQLGlotTokenType.DOT
                and _is_word_or_ident_token(toks[i + 2])
            ):
                q_unq = tk.text.lower()
                c_unq = toks[i + 2].text.lower()
                actual_c = col_names_lower[c_unq] if c_unq in col_names_lower else toks[i + 2].text
                if q_unq in quals_lower:
                    if tk.start > cursor:
                        out.append(expr[cursor:tk.start])
                    if record_prefix:
                        out.append(f"{record_prefix}.{self._q(actual_c)}")
                    else:
                        out.append(f"{self._q(tk.text)}.{self._q(actual_c)}")
                    cursor = toks[i + 2].end + 1
                    i += 3
                    continue
            if _is_word_or_ident_token(tk):
                is_after_dot = (i > 0 and toks[i - 1].token_type == SQLGlotTokenType.DOT)
                is_before_dot = (i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.DOT)
                is_fn_call = (
                    tk.token_type != SQLGlotTokenType.IDENTIFIER
                    and i + 1 < n
                    and toks[i + 1].token_type == SQLGlotTokenType.L_PAREN
                )
                if not is_after_dot and not is_before_dot and not is_fn_call:
                    w_low = tk.text.lower()
                    if w_low in col_names_lower:
                        actual_c = col_names_lower[w_low]
                        if tk.start > cursor:
                            out.append(expr[cursor:tk.start])
                        if record_prefix:
                            out.append(f"{record_prefix}.{self._q(actual_c)}")
                        else:
                            out.append(self._q(actual_c))
                        cursor = tk.end + 1
                        i += 1
                        continue
                    elif tk.token_type == SQLGlotTokenType.IDENTIFIER:
                        if tk.start > cursor:
                            out.append(expr[cursor:tk.start])
                        out.append(self._q(tk.text))
                        cursor = tk.end + 1
                        i += 1
                        continue
            i += 1
        if cursor < len(expr):
            out.append(expr[cursor:])
        return "".join(out)

    def _referenced_columns_for_table(self, table_name: str) -> List[str]:
        ts = self._schemas_lower.get(table_name.lower())
        if not ts:
            return []
        try:
            q_ast = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
            if q_ast.find(sqlglot_exp.Star):
                return list(ts.columns.keys())
            cols_in_ast = {c.name.lower() for c in q_ast.find_all(sqlglot_exp.Column)}
            found = [c for c in ts.columns if c.lower() in cols_in_ast]
            return found if found else list(ts.columns.keys())
        except Exception:
            return list(ts.columns.keys())

