"""Shared dialect helpers, identifier quoting, storage type inference, and DDL creation."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .. import key_collation
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
    _is_simple_column_expr,
    _is_untyped_null_expr,
    _is_word_or_ident_token,
    _parse_single_agg_expr,
    _sanitize_ident,
    _split_top_level,
    _sqlite_type_affinity,
    _strip_outer_ident_quotes,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
    _unwrap_pg_catalog_agg_node,
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
        needs_sanitize = not bool(re.fullmatch(r"[A-Za-z0-9_]+", stem))
        safe_stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower() if needs_sanitize else stem
        if needs_sanitize and not safe_stem:
            safe_stem = "v"
        full = f"{prefix}{safe_stem}{suffix}"
        if not needs_sanitize and len(full) <= max_len:
            return full
        # Hash the raw `stem` so sanitized identifiers (e.g. "sales summary" vs
        # "sales_summary" vs "Sales Summary") and truncated names never collide.
        h = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        avail = max_len - len(prefix) - len(suffix) - 9
        if avail >= 1:
            return f"{prefix}{safe_stem[:avail]}_{h}{suffix}"
        if len(prefix) + 8 + len(suffix) <= max_len:
            return f"{prefix}{h}{suffix}"
        full_h = hashlib.sha1(f"{prefix}{stem}{suffix}".encode("utf-8")).hexdigest()[:8]
        p_avail = max(1, max_len - len(suffix) - 9)
        return f"{prefix[:p_avail]}_{full_h}{suffix}"[:max_len]

    def _backing_table_name(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}store_", self.logical_view_name)

    def _bound_user_index_name(self, idx_name: str) -> str:
        """Keep a user-supplied index name verbatim when it fits within the
        target engine's identifier limit; otherwise hash-truncate deterministically."""
        if self.dialect == Dialect.POSTGRES:
            max_bytes = 63
            if len(idx_name.encode("utf-8")) <= max_bytes:
                return idx_name
            h = hashlib.sha1(idx_name.encode("utf-8")).hexdigest()[:8]
            suffix = f"_{h}"
            budget = max_bytes - len(suffix.encode("utf-8"))
            prefix_chars: List[str] = []
            used = 0
            for ch in idx_name:
                ch_b = len(ch.encode("utf-8"))
                if used + ch_b > budget:
                    break
                prefix_chars.append(ch)
                used += ch_b
            return "".join(prefix_chars) + suffix
        if self.dialect == Dialect.MYSQL:
            max_chars = 64
            if len(idx_name) <= max_chars:
                return idx_name
            h = hashlib.sha1(idx_name.encode("utf-8")).hexdigest()[:8]
            suffix = f"_{h}"
            return idx_name[: max_chars - len(suffix)] + suffix
        return idx_name

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
                            _strip_outer_ident_quotes(c)
                            for c in _split_top_level(m.group(1), ",", dialect=self.dialect)
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
                    col_match = re.match(r'^("((?:""|[^"])+)"|`((?:``|[^`])+)`|([A-Za-z0-9_]+))\s+(.*)$', line_s)
                    if col_match:
                        if col_match.group(2) is not None:
                            col_ident = col_match.group(2).replace('""', '"').lower()
                        elif col_match.group(3) is not None:
                            col_ident = col_match.group(3).replace("``", "`").lower()
                        else:
                            col_ident = col_match.group(4).lower()
                        if col_ident in pk_set:
                            rest = col_match.group(5).upper()
                            if "NOT NULL" in rest:
                                found_nn.add(col_ident)
                if len(found_nn) == len(pk_set) and not has_autoincrement:
                    without_rowid_tail = " WITHOUT ROWID"

        cols_joined = ",\n    ".join(col_defs_sql)
        return f"CREATE TABLE {table_name} (\n    {cols_joined}\n){fillfactor_tail}{without_rowid_tail};"


    def _is_known_single_ident(self, raw: str) -> bool:
        low = raw.lower()
        if low in self._schemas_lower:
            return True
        for ts in self._schemas_lower.values():
            for c_name in ts.columns:
                if c_name.lower() == low:
                    return True
        for it in self.view.select_items:
            if it.alias and it.alias.lower() == low:
                return True
        if low in (
            (self.logical_view_name or "").lower(),
            (self.db_view_name or "").lower(),
        ):
            return True
        return False

    def _q(self, ident: str) -> str:
        s = ident.strip()
        if not s:
            return s
        # Preserve literal dots inside already-quoted single identifiers (e.g.
        # `"my.orders"`, `` `a.b` ``) and known unquoted table/column/view names
        # from the schema instead of splitting them as `schema.table`.
        is_quoted = (
            (s.startswith('"') and s.endswith('"') and len(s) >= 2)
            or (s.startswith("`") and s.endswith("`") and len(s) >= 2)
            or (s.startswith("[") and s.endswith("]") and len(s) >= 2)
        )
        if is_quoted:
            try:
                if len(_split_top_level(s, ".", dialect=self.dialect)) == 1:
                    return quote_sql_identifier(s, self.dialect, allow_qualified=False)
            except sqlglot.errors.SqlglotError:
                pass
        if "." in s and self._is_known_single_ident(s):
            return quote_sql_identifier(s, self.dialect, allow_qualified=False)
        return quote_sql_identifier(s, self.dialect)

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
            known_qual = False
            for j in self.view.joins:
                if j.alias.lower() == prefix_clean or j.table_name.lower() == prefix_clean:
                    known_qual = True
                    ts = self._schemas_lower.get(j.table_name.lower())
                    if ts:
                        for c_def in ts.columns.values():
                            if c_def.name.lower() == col_clean:
                                return c_def
            ts = self._schemas_lower.get(prefix_clean)
            if ts:
                known_qual = True
                for c_def in ts.columns.values():
                    if c_def.name.lower() == col_clean:
                        return c_def
            if known_qual:
                return None

        col_clean = (
            s.lower()
            if ("." in s and not (s.startswith('"') or s.startswith("`") or s.startswith("[")))
            else _unquote_ident(s, dialect=self.dialect).lower()
        )
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

        is_container_or_interval = any(
            token in source_type for token in ("[", "]", "ARRAY", "RANGE", "INTERVAL")
        )
        is_approximate = (not is_container_or_interval) and bool(
            re.search(r"\b(?:FLOAT|FLOAT4|FLOAT8|DOUBLE|REAL)\b", source_type)
        )
        is_integer = (not is_container_or_interval) and (not is_approximate) and bool(
            re.search(
                r"\b(?:TINYINT|UTINYINT|SMALLINT|USMALLINT|MEDIUMINT|UMEDIUMINT|"
                r"INTEGER|INT|UINT|INT2|INT4|INT8|BIGINT|UBIGINT|"
                r"SMALLSERIAL|SERIAL|SERIAL2|SERIAL4|SERIAL8|BIGSERIAL)\b",
                source_type,
            )
        )
        is_exact_decimal = (not is_container_or_interval) and bool(
            re.search(r"\b(?:DECIMAL|NUMERIC|DEC|FIXED)\b", source_type)
        )

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
            if is_exact_decimal or bool(
                re.search(r"\b(?:BIGINT|INT8|BIGSERIAL|SERIAL8)\b", source_type)
            ):
                return "NUMERIC"
            if is_integer:
                return "BIGINT"
            if bool(re.search(r"\b(?:REAL|FLOAT4)\b", source_type)):
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
        upper = re.sub(r"\s+", " ", data_type.strip().upper())
        if any(token in upper for token in ("[", "]", "ARRAY", "RANGE", "INTERVAL")):
            return None
        if re.search(r"\b(?:FLOAT|FLOAT4|FLOAT8|DOUBLE|REAL)\b", upper):
            return None
        decimal_match = re.search(
            r"\b(?:DECIMAL|NUMERIC|DEC|FIXED)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?",
            upper,
        )
        if decimal_match:
            return int(decimal_match.group(1) or 10), int(decimal_match.group(2) or 0)
        if re.search(r"\b(?:TINYINT|UTINYINT)\b", upper):
            return 3, 0
        if re.search(r"\b(?:SMALLINT|USMALLINT|INT2|SMALLSERIAL|SERIAL2)\b", upper):
            return 5, 0
        if re.search(r"\b(?:MEDIUMINT|UMEDIUMINT)\b", upper):
            return 7, 0
        if re.search(r"\b(?:BIGINT|UBIGINT|INT8|BIGSERIAL|SERIAL8)\b", upper):
            return (20 if "UNSIGNED" in upper or "UBIGINT" in upper or "SERIAL" in upper else 19), 0
        if re.search(r"\b(?:INTEGER|INT|UINT|INT4|SERIAL4)\b", upper):
            return 10, 0
        if re.search(r"\bSERIAL\b", upper):
            return (20 if self.dialect == Dialect.MYSQL else 10), 0
        if re.fullmatch(r"(?:SIGNED|UNSIGNED)(?:\s+INTEGER)?", upper):
            return (20 if "UNSIGNED" in upper else 19), 0
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
            if kind in ("Paren", "Neg", "Abs"):
                return infer(node.this)
            if kind in ("Floor", "Ceil", "Ceiling", "Sign"):
                if node.this is None:
                    return None
                value = infer(node.this)
                if value is None:
                    return None
                if self.dialect == Dialect.POSTGRES:
                    # PostgreSQL FLOOR/CEIL/SIGN only stay exact NUMERIC for
                    # NUMERIC/DECIMAL inputs; integer inputs coerce to DOUBLE PRECISION.
                    arg_t = self._infer_ast_expr_storage_type(node.this.sql(_sqlglot_dialect(self.dialect)))
                    if arg_t != "NUMERIC":
                        return None
                    return min(65, max(1, value[0] - value[1] + 1)), 0
                if self.dialect == Dialect.MYSQL:
                    return min(65, max(1, value[0] - value[1] + 1)), 0
                return value
            if kind == "Cast":
                target = node.args.get("to")
                return self._numeric_precision_scale_from_type(target.sql()) if target is not None else None
            if kind in ("Add", "Sub", "Mul", "Div", "Mod", "IntDiv"):
                left, right = infer(node.this), infer(node.expression)
                if left is None or right is None:
                    return None
                p1, s1 = left
                p2, s2 = right
                if kind == "IntDiv":
                    # MySQL/MariaDB integer division (DIV) produces an exact
                    # integer (scale 0) even for DECIMAL operands.
                    return min(65, max(19, p1 - s1)), 0
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
            if kind == "Nullif":
                if self.dialect == Dialect.POSTGRES:
                    # PostgreSQL NULLIF(a, b) resolves via the `=` operator:
                    # int = int keeps a's exact type; int = numeric promotes to
                    # numeric; exact = float promotes to double precision (None here);
                    # NULLIF(NULL, b) takes b's type.
                    if isinstance(node.this, sqlglot_exp.Null):
                        if node.expression is None or isinstance(node.expression, sqlglot_exp.Null):
                            return None
                        return infer(node.expression)
                    left = infer(node.this)
                    if left is None:
                        return None
                    if node.expression is None or isinstance(node.expression, sqlglot_exp.Null):
                        return left
                    right = infer(node.expression)
                    if right is None:
                        return None
                    return merge_types([left, right])
                if isinstance(node.this, sqlglot_exp.Null):
                    return None
                return infer(node.this)
            if kind in ("Coalesce", "IfNull", "Greatest", "Least"):
                values = []
                for child in node.iter_expressions():
                    if isinstance(child, sqlglot_exp.Null):
                        continue
                    inferred = infer(child)
                    if inferred is None:
                        return None
                    values.append(inferred)
                return merge_types(values) if values else None
            if kind == "If":
                branches = []
                for branch_key in ("true", "false"):
                    branch_node = node.args.get(branch_key)
                    if branch_node is None or isinstance(branch_node, sqlglot_exp.Null):
                        continue
                    inferred = infer(branch_node)
                    if inferred is None:
                        return None
                    branches.append(inferred)
                return merge_types(branches) if branches else None
            if kind == "Case":
                branches = []
                for branch in node.args.get("ifs") or ():
                    value = branch.args.get("true")
                    if value is not None and not isinstance(value, sqlglot_exp.Null):
                        inferred = infer(value)
                        if inferred is None:
                            return None
                        branches.append(inferred)
                default = node.args.get("default")
                if default is not None and not isinstance(default, sqlglot_exp.Null):
                    inferred = infer(default)
                    if inferred is None:
                        return None
                    branches.append(inferred)
                return merge_types(branches) if branches else None
            if kind in ("Round", "Trunc"):
                if node.this is None:
                    return None
                value = infer(node.this)
                if value is None:
                    return None
                scale_node = node.args.get("decimals") or node.expression
                if self.dialect == Dialect.POSTGRES:
                    # PostgreSQL 1-arg ROUND/TRUNC(int) coerces to DOUBLE PRECISION;
                    # 2-arg ROUND/TRUNC(x, n) requires NUMERIC and returns NUMERIC.
                    if scale_node is None:
                        arg_t = self._infer_ast_expr_storage_type(node.this.sql(_sqlglot_dialect(self.dialect)))
                        if arg_t != "NUMERIC":
                            return None
                        return min(65, max(1, value[0] - value[1] + 1)), 0
                if scale_node is None:
                    return min(65, max(1, value[0] - value[1] + 1)), 0
                if scale_node.__class__.__name__ != "Literal":
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
        s = dt.strip()
        u = s.upper()
        if self.dialect == Dialect.POSTGRES:
            m_generic_arr = re.match(r"^ARRAY\s*<\s*(.+)\s*>$", s, flags=re.IGNORECASE)
            if m_generic_arr:
                inner = m_generic_arr.group(1).strip()
                norm_inner = self._normalize_dialect_type(inner)
                if inner.strip().upper() == "CHARACTER VARYING":
                    norm_inner = "CHARACTER VARYING"
                return f"{norm_inner}[]"
            m_pg_arr = re.match(r"^(.*?)((?:\[\])+)$", s)
            if m_pg_arr:
                elem_raw, brackets = m_pg_arr.group(1).strip(), m_pg_arr.group(2)
                if elem_raw.upper() in ("CHARACTER VARYING", "VARCHAR"):
                    return f"{elem_raw}{brackets}"
                return f"{self._normalize_dialect_type(elem_raw)}{brackets}"
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
                    agg_item = _parse_single_agg_expr(
                        inner.sql(_sqlglot_dialect(self.dialect)), ident, dialect=self.dialect
                    )
                    if agg_item is not None and agg_item.agg_func is not None:
                        return self._infer_expr_storage_type(agg_item)
                    c_arg = self._lookup_col_def(inner.this.sql().strip() if isinstance(inner.this, sqlglot_exp.Expr) else None)
                    if c_arg:
                        return self._minmax_result_type(
                            self._normalize_dialect_type(c_arg.data_type),
                            inner.this.sql(_sqlglot_dialect(self.dialect)) if isinstance(inner.this, sqlglot_exp.Expr) else None,
                        )
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
            cte_branches: List[sqlglot_exp.Select] = []
            stack: List[Any] = [cte.this]
            while stack:
                cur_q = stack.pop()
                if isinstance(cur_q, sqlglot_exp.Subquery):
                    stack.append(cur_q.this)
                elif isinstance(cur_q, sqlglot_exp.SetOperation):
                    stack.append(cur_q.expression)
                    stack.append(cur_q.this)
                elif isinstance(cur_q, sqlglot_exp.Select):
                    cte_branches.append(cur_q)
            if not cte_branches:
                continue
            inner_query = cte_branches[0]

            col_pos: Optional[int] = None
            table_alias = cte.args.get("alias")
            if isinstance(table_alias, sqlglot_exp.TableAlias) and table_alias.columns:
                hdr_cols = [c.name.lower() for c in table_alias.columns]
                if ident.lower() in hdr_cols:
                    col_pos = hdr_cols.index(ident.lower())
            if col_pos is None:
                for p_i, proj in enumerate(inner_query.expressions):
                    c_name = proj.alias or (proj.this.name if isinstance(proj, sqlglot_exp.Alias) else proj.name)
                    if c_name.lower() == ident.lower():
                        col_pos = p_i
                        break
            if col_pos is not None:
                b_types: List[str] = []
                has_null_b = False
                has_str_lit_b = False
                for br in cte_branches:
                    if col_pos >= len(br.expressions):
                        continue
                    proj_e = br.expressions[col_pos]
                    src_e = proj_e.this if isinstance(proj_e, sqlglot_exp.Alias) else proj_e
                    while isinstance(src_e, sqlglot_exp.Paren):
                        src_e = src_e.this
                    if isinstance(src_e, sqlglot_exp.Null):
                        has_null_b = True
                        continue
                    if isinstance(src_e, sqlglot_exp.Literal):
                        if self.dialect == Dialect.POSTGRES and src_e.args.get("is_string"):
                            has_str_lit_b = True
                        else:
                            b_types.append(self._literal_storage_type(src_e))
                        continue
                    src_col = src_e.name or src_e.sql().strip()
                    c_src = self._lookup_col_def(src_e.sql().strip()) or self._lookup_col_def(src_col)
                    if c_src:
                        b_types.append(c_src.data_type)
                if not b_types and has_str_lit_b:
                    b_types.append("TEXT")
                if b_types:
                    if len(cte_branches) > 1:
                        return self._common_storage_type(b_types, has_null=has_null_b)
                    return self._normalize_dialect_type(b_types[0])

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

    def _mysql_group_key_narrow_integer_type(self, expr: str) -> Optional[str]:
        """MySQL's grouped narrow-integer arithmetic key has INT result metadata.

        MySQL 8.4/9.7 reports a plain projection such as s % 2 or s + s as
        BIGINT, but the same expression selected as a GROUP BY key is INT.
        This is distinct from MariaDB's display-length rule below. Keep the
        inference deliberately narrow: only a direct group-key ADD of two
        signed TINYINT/SMALLINT/MEDIUMINT columns, or MOD with a narrow signed
        column on the left and a narrow column/small integer literal on the
        right, is covered. Unsigned, multiplying, casted and unknown shapes
        retain the ordinary expression inference.
        """
        if (
            self.dialect != Dialect.MYSQL
            or self.subdialect == EngineSubdialect.MARIADB
            or tuple(self.engine_version.version[:2]) not in ((8, 4), (9, 7))
            or not self.view.group_by_cols
        ):
            return None

        try:
            root = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None

        def strip_parens(node: Any) -> Any:
            while node.__class__.__name__ == "Paren":
                node = node.this
            return node

        root = strip_parens(root)
        kind = root.__class__.__name__
        if kind not in ("Add", "Mod"):
            return None

        sql_dialect = _sqlglot_dialect(self.dialect)
        root_sql = root.sql(dialect=sql_dialect)
        for group_expr in self.view.group_by_cols:
            try:
                group_root = strip_parens(parse_sql_ast(group_expr, dialect=self.dialect))
            except sqlglot.errors.SqlglotError:
                continue
            if group_root.sql(dialect=sql_dialect) != root_sql:
                continue

            def narrow_signed_column(node: Any) -> bool:
                node = strip_parens(node)
                if node.__class__.__name__ != "Column":
                    return False
                col = self._lookup_col_def(node.sql(dialect=sql_dialect)) or self._lookup_col_def(node.name)
                if col is None:
                    return False
                declared = re.sub(r"\s+", " ", col.data_type.strip().upper())
                return re.fullmatch(
                    r"(?:TINYINT|SMALLINT|MEDIUMINT)(?:\s*\(\s*\d+\s*\))?",
                    declared,
                ) is not None

            def small_integer_literal(node: Any) -> bool:
                node = strip_parens(node)
                if node.__class__.__name__ == "Neg":
                    node = strip_parens(node.this)
                if node.__class__.__name__ != "Literal" or node.args.get("is_string"):
                    return False
                value = str(node.this)
                return re.fullmatch(r"\d+", value) is not None and int(value) < 2 ** 31

            left = root.this
            right = root.expression
            if kind == "Add" and narrow_signed_column(left) and narrow_signed_column(right):
                return "INT"
            if kind == "Mod" and narrow_signed_column(left) and (
                narrow_signed_column(right) or small_integer_literal(right)
            ):
                return "INT"
            return None
        return None

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

        def _pg_canonical_arr_elem(elem_raw: str) -> str:
            elem_s = elem_raw.strip()
            if '"' in elem_s:
                return elem_s
            elem_up = re.sub(r"\s+", " ", elem_s.upper())
            elem_canon = {
                "DECIMAL": "NUMERIC",
                "DEC": "NUMERIC",
                "INT": "INTEGER",
                "INT4": "INTEGER",
                "INT2": "SMALLINT",
                "INT8": "BIGINT",
                "FLOAT4": "REAL",
                "FLOAT8": "DOUBLE PRECISION",
                "BOOL": "BOOLEAN",
                "CHARACTER VARYING": "VARCHAR",
                "CHARACTER": "BPCHAR",
                "TIMESTAMPTZ": "TIMESTAMP WITH TIME ZONE",
                "TIMETZ": "TIME WITH TIME ZONE",
            }.get(elem_up)
            if elem_canon is not None:
                return elem_canon
            return elem_up

        def _pg_strip_arr_elem_mod(elem: str) -> str:
            if '"' in elem:
                return elem
            base = re.sub(r"\s*\([^)]*\)", "", elem).strip().upper()
            return {
                "DECIMAL": "NUMERIC",
                "DEC": "NUMERIC",
                "INT": "INTEGER",
                "INT4": "INTEGER",
                "INT2": "SMALLINT",
                "INT8": "BIGINT",
                "FLOAT4": "REAL",
                "FLOAT8": "DOUBLE PRECISION",
                "BOOL": "BOOLEAN",
                "CHARACTER VARYING": "VARCHAR",
                "CHARACTER": "BPCHAR",
                "CHAR": "BPCHAR",
                "NCHAR": "BPCHAR",
                "TIMESTAMPTZ": "TIMESTAMP WITH TIME ZONE",
                "TIMETZ": "TIME WITH TIME ZONE",
            }.get(base, base)

        def canonical_type(data_type: str) -> Optional[str]:
            if self.dialect == Dialect.POSTGRES:
                stripped_dt = data_type.strip()
                norm_dt = self._normalize_dialect_type(stripped_dt)
                # Strip trailing COLLATE clause before parsing array brackets or
                # checking for quoted custom type names (e.g. `VARCHAR(20) COLLATE "C"`).
                no_coll = self._PG_COLLATE_TAIL_RE.sub("", norm_dt.strip()).strip()
                no_coll_raw = self._PG_COLLATE_TAIL_RE.sub("", stripped_dt).strip()
                m_arr = re.match(r"^(.*?)((?:\[\])+)$", no_coll)
                if m_arr:
                    elem_raw = m_arr.group(1).strip()
                    # PostgreSQL array columns/values all share a single 1D
                    # array type OID regardless of declared bracket count.
                    return f"{_pg_canonical_arr_elem(elem_raw)}[]"
                if '"' in no_coll_raw:
                    return no_coll_raw
                dt_up = re.sub(r"\s+", " ", no_coll_raw.upper())
                if dt_up == "BIT" or re.match(r"^BIT\s*\(\s*\d+\s*\)$", dt_up):
                    return re.sub(r"\s+", "", dt_up)
                if dt_up in ("VARBIT", "BIT VARYING") or re.match(
                    r"^(?:VARBIT|BIT VARYING)\s*\(\s*\d+\s*\)$", dt_up
                ):
                    return "BIT VARYING"
                data_type = no_coll_raw
            # Introspected MySQL/MariaDB string types carry their collation
            # ("CHAR(5) COLLATE UTF8MB4_UNICODE_CI" under a non-default
            # server collation): the clause does not change the family.
            bare = re.sub(r"\s+(?:COLLATE|CHARACTER\s+SET|CHARSET)\s+\S+", "", data_type.upper())
            bare = re.sub(r"\s+ZEROFILL\b", "", bare)
            upper = re.sub(r"\s*\(.*\)\s*$", "", bare).strip()
            if upper in ("SMALLINT", "INT2", "SMALLSERIAL", "SERIAL2"):
                return "SMALLINT"
            if upper in (
                "INTEGER", "INT", "INT4", "MEDIUMINT", "TINYINT",
                "SERIAL", "SERIAL4",
            ):
                return "INTEGER"
            if upper in (
                "BIGINT", "INT8", "BIGSERIAL", "SERIAL8", "SIGNED", "SIGNED INTEGER",
            ):
                return "BIGINT"
            if upper in ("NUMERIC", "DECIMAL", "DEC", "FIXED"):
                return "NUMERIC"
            if upper in ("REAL", "FLOAT4"):
                return "REAL"
            if upper in ("DOUBLE PRECISION", "DOUBLE", "FLOAT8", "FLOAT"):
                return "DOUBLE PRECISION"
            if upper in ("TEXT", "VARCHAR", "CHAR", "BPCHAR", "CHARACTER VARYING", "CHARACTER", "NVARCHAR", "NCHAR", "CLOB") or (
                self.dialect == Dialect.POSTGRES and upper == "NAME"
            ):
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
                if all(value in ("TEXT", "VARCHAR", "BPCHAR") for value in known):
                    if all(value == "VARCHAR" for value in known):
                        return "VARCHAR"
                    if all(value == "BPCHAR" for value in known):
                        return "BPCHAR"
                    return "TEXT"
                if self.dialect == Dialect.POSTGRES and all(value.endswith("[]") for value in known):
                    elems = [re.sub(r"(?:\[\])+$", "", v).strip() for v in known]
                    if all(e == elems[0] for e in elems):
                        return f"{elems[0]}[]"
                    stripped = [_pg_strip_arr_elem_mod(e) for e in elems]
                    if all(s in ranks for s in stripped):
                        return f"{max(stripped, key=lambda v: ranks[v])}[]"
                    if all(s in ("TEXT", "VARCHAR", "BPCHAR") for s in stripped):
                        if all(s == "VARCHAR" for s in stripped):
                            return "VARCHAR[]"
                        if all(s == "BPCHAR" for s in stripped):
                            return "BPCHAR[]"
                        return "TEXT[]"
                    if all(s == stripped[0] for s in stripped):
                        return f"{stripped[0]}[]"
                if self.dialect == Dialect.POSTGRES and all(
                    v == "BIT" or v == "BIT VARYING" or v.startswith("BIT(") for v in known
                ):
                    return "BIT VARYING"
                return None
            return max(known, key=lambda value: ranks[value])

        def infer(node: Any) -> Optional[str]:
            if node is None:
                return "UNKNOWN"
            kind = node.__class__.__name__
            if kind in ("Paren", "Collate"):
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
            if kind in ("Floor", "Ceil", "Ceiling", "Sign") and self.dialect == Dialect.POSTGRES:
                arg_t = infer(node.this)
                if arg_t == "NUMERIC":
                    return "NUMERIC"
                if arg_t in ranks:
                    return "DOUBLE PRECISION"
                return None
            if kind in ("Round", "Trunc") and self.dialect == Dialect.POSTGRES:
                arg_t = infer(node.this)
                scale_node = node.args.get("decimals") or node.expression
                if scale_node is not None:
                    return "NUMERIC" if arg_t in ranks else None
                if arg_t == "NUMERIC":
                    return "NUMERIC"
                if arg_t in ranks:
                    return "DOUBLE PRECISION"
                return None
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
                if target is None:
                    return None
                target_sql = (
                    target.sql(_sqlglot_dialect(self.dialect))
                    if self.dialect == Dialect.POSTGRES
                    else target.sql()
                )
                return canonical_type(target_sql)
            if kind == "Array" and self.dialect == Dialect.POSTGRES:
                ch_nodes = list(node.expressions or ())
                if not ch_nodes:
                    return None
                def _infer_arr_elem(ch: Any) -> Optional[str]:
                    unw = ch
                    while isinstance(unw, (sqlglot_exp.Paren, sqlglot_exp.Collate)) and unw.this is not None:
                        unw = unw.this
                    if isinstance(unw, sqlglot_exp.Column):
                        col = self._lookup_col_def(unw.sql()) or self._lookup_col_def(unw.name)
                        dt_raw = (
                            col.data_type
                            if col is not None
                            else (
                                self._infer_cte_or_subquery_col_type(unw.sql())
                                or self._infer_cte_or_subquery_col_type(unw.name)
                            )
                        )
                        if dt_raw:
                            no_coll = self._PG_COLLATE_TAIL_RE.sub("", dt_raw.strip()).strip()
                            if not no_coll.endswith("[]"):
                                base_str = _pg_strip_arr_elem_mod(no_coll)
                                if base_str in ("VARCHAR", "BPCHAR"):
                                    return base_str
                    return infer(ch)
                elem_types = [_infer_arr_elem(ch) for ch in ch_nodes]
                if all(t == "UNKNOWN" for t in elem_types):
                    return "TEXT[]"
                # In PostgreSQL, bare string literals have type `unknown` and
                # coerce to any explicitly typed element (e.g. ARRAY[1, '2'] -> INTEGER[]).
                non_str_lit_types = [
                    t
                    for ch, t in zip(ch_nodes, elem_types)
                    if not (isinstance(ch, sqlglot_exp.Literal) and ch.args.get("is_string"))
                    and t is not None
                    and t != "UNKNOWN"
                ]
                merged_elem = merge(non_str_lit_types) if non_str_lit_types else merge(elem_types)
                if merged_elem is not None:
                    return f"{merged_elem}[]"
                return None
            if kind == "Bracket" and self.dialect == Dialect.POSTGRES:
                # Unwrap nested brackets (e.g. a_int2d[1][1] or a_int2d[1:2][1:2]):
                # on PostgreSQL, any slice returns the array type; pure subscripts
                # always return the scalar element type (preserving VARCHAR/BPCHAR
                # OIDs rather than collapsing to TEXT).
                has_slice = False
                cur_n: Any = node
                while cur_n is not None and cur_n.__class__.__name__ == "Bracket":
                    if any(isinstance(e, sqlglot_exp.Slice) for e in (cur_n.expressions or ())):
                        has_slice = True
                    cur_n = cur_n.this
                base_t = infer(cur_n)
                if base_t is not None and base_t.endswith("[]"):
                    if has_slice:
                        return base_t
                    elem_raw = re.sub(r"(?:\[\])+$", "", base_t).strip()
                    stripped_elem = _pg_strip_arr_elem_mod(elem_raw)
                    if stripped_elem in ("VARCHAR", "BPCHAR"):
                        return stripped_elem
                    return canonical_type(stripped_elem) or stripped_elem
                return None
            if kind in ("Coalesce", "IfNull"):
                vals = [infer(ch) for ch in node.iter_expressions()]
                return merge(vals)
            if kind in ("Greatest", "Least"):
                vals = [infer(ch) for ch in node.iter_expressions()]
                if any(v is None for v in vals):
                    return None
                return merge(vals)
            if kind == "Nullif":
                if self.dialect == Dialect.POSTGRES:
                    # PostgreSQL NULLIF(a, b) resolves via the `=` operator and
                    # returns the coerced left operand (oprleft):
                    # - int* = int* keeps a's type (no widening)
                    # - float* = float* keeps a's type
                    # - exact = float* promotes to DOUBLE PRECISION
                    # - int* = numeric promotes to NUMERIC
                    # - NULLIF(NULL, b) resolves to b
                    a_t = infer(node.this)
                    b_t = infer(node.expression) if node.expression is not None else "UNKNOWN"
                    if a_t is None or b_t is None:
                        return None
                    if a_t == "UNKNOWN":
                        return b_t
                    if b_t == "UNKNOWN":
                        return a_t
                    int_types = ("SMALLINT", "INTEGER", "BIGINT")
                    float_types = ("REAL", "DOUBLE PRECISION")
                    if a_t in int_types and b_t in int_types:
                        return a_t
                    if a_t in float_types and b_t in float_types:
                        return a_t
                    if (a_t in (*int_types, "NUMERIC") and b_t in float_types) or (
                        b_t in (*int_types, "NUMERIC") and a_t in float_types
                    ):
                        return "DOUBLE PRECISION"
                    if (a_t in int_types and b_t == "NUMERIC") or (b_t in int_types and a_t == "NUMERIC"):
                        return "NUMERIC"
                    return merge((a_t, b_t))
                return infer(node.this)
            if kind == "If":
                t_br = infer(node.args.get("true"))
                f_br = infer(node.args.get("false"))
                if t_br is None or f_br is None:
                    return None
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
            if kind == "BitString" and self.dialect == Dialect.POSTGRES:
                bits = str(node.this or "")
                return f"BIT({len(bits)})" if bits else "BIT VARYING"
            if kind in ("BitwiseAnd", "BitwiseOr", "BitwiseXor") and self.dialect == Dialect.POSTGRES:
                left, right = infer(node.this), infer(node.expression)
                if left is None or right is None:
                    return None
                return merge((left, right))
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
                if merged == "REAL" and not (
                    self.dialect == Dialect.POSTGRES
                    and left == "REAL"
                    and right == "REAL"
                    and kind in ("Add", "Sub", "Mul", "Div")
                ):
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
            if isinstance(node, (sqlglot_exp.Dot, sqlglot_exp.Column, sqlglot_exp.Filter)):
                pg_cat_fn = _unwrap_pg_catalog_agg_node(node, self.dialect)
                if pg_cat_fn is not None:
                    pg_fn_up = (
                        pg_cat_fn.name.upper()
                        if isinstance(pg_cat_fn, sqlglot_exp.Anonymous)
                        else pg_cat_fn.sql_name().upper()
                    )
                    if isinstance(pg_cat_fn, sqlglot_exp.AggFunc) or pg_fn_up in agg_fn_names:
                        agg_item = _parse_single_agg_expr(
                            node.sql(_sqlglot_dialect(self.dialect)), "__agg", dialect=self.dialect
                        )
                        if agg_item is not None and agg_item.agg_func is not None:
                            return canonical_type(self._infer_expr_storage_type(agg_item))
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
        if kind in ("Paren", "Collate"):
            return self._pg_string_result_type(node.this)
        if kind == "Literal":
            return None
        if kind == "Null":
            return None
        if kind == "Column":
            col = self._lookup_col_def(node.sql()) or self._lookup_col_def(node.name)
            if col is not None:
                dt_raw = re.sub(r"\s+COLLATE\s+.*$", "", col.data_type.strip(), flags=re.IGNORECASE)
                return self._pg_string_result_type_from_name(dt_raw)
            cte_t = self._infer_cte_or_subquery_col_type(node.name)
            if cte_t is not None:
                dt_raw = re.sub(r"\s+COLLATE\s+.*$", "", cte_t.strip(), flags=re.IGNORECASE)
                return self._pg_string_result_type_from_name(dt_raw)
            return "TEXT"
        if kind == "Cast":
            target = node.args.get("to")
            return self._pg_string_result_type_from_name(target.sql() if target is not None else "")
        if kind == "Nullif":
            # Verified natively across the full 6x6 matrix on PostgreSQL 18:
            # NULLIF(a, b) returns the left-operand type of PostgreSQL's chosen `=` operator.
            ta = self._pg_string_result_type(node.this) if node.this is not None else None
            tb = self._pg_string_result_type(node.expression) if node.expression is not None else None
            if ta == "NAME":
                return "NAME"
            if ta is None:
                if tb in ("NAME", "BPCHAR"):
                    return tb
                return "TEXT" if tb is not None else None
            if ta == "BPCHAR":
                return "BPCHAR" if tb in (None, "BPCHAR", "VARCHAR") else "TEXT"
            if ta == "VARCHAR":
                return "BPCHAR" if tb == "BPCHAR" else "TEXT"
            return "TEXT"
        branches: List[Any] = []
        if kind in ("Coalesce", "IfNull", "Greatest", "Least"):
            branches = list(node.iter_expressions())
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
        resolved_branches: List[str] = []
        for branch in branches:
            if branch is None:
                continue
            t = self._pg_string_result_type(branch)
            if t is not None:
                resolved_branches.append(t)
        if not resolved_branches:
            return None
        # Verified natively on PostgreSQL 18: TEXT first stays TEXT (even when
        # mixed with NAME); otherwise NAME beats VARCHAR/BPCHAR in either order
        # (COALESCE/GREATEST/LEAST over name + char/varchar resolve to name);
        # VARCHAR/BPCHAR/TEXT keep the first typed input.
        if resolved_branches[0] == "TEXT":
            return "TEXT"
        if "NAME" in resolved_branches:
            return "NAME"
        return resolved_branches[0]

    @staticmethod
    def _pg_string_result_type_from_name(type_name: str) -> str:
        clean = re.sub(r"\s+COLLATE\s+.*$", "", type_name.strip(), flags=re.IGNORECASE)
        base = re.sub(r"\s*\(.*\)\s*$", "", clean.upper()).strip()
        if base in ("VARCHAR", "CHARACTER VARYING"):
            return "VARCHAR"
        if base in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
            return "BPCHAR"
        if base == "NAME":
            return "NAME"
        return "TEXT"

    def _common_storage_type(self, types: Sequence[str], *, has_null: bool = False) -> str:
        """Merge column storage types across branches of a set operation
        (UNION ALL / UNION / INTERSECT / EXCEPT) using target dialect widening rules."""
        raw_nonempty = [t.strip() for t in types if t and t.strip()]
        if not raw_nonempty:
            return "VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT"

        if self.dialect == Dialect.POSTGRES:
            cleaned = [self._normalize_dialect_type(t).strip() for t in raw_nonempty]
            no_coll = [self._PG_COLLATE_TAIL_RE.sub("", t).strip() for t in cleaned]
            colls = [
                (m.group(0).rstrip() if (m := self._PG_COLLATE_TAIL_RE.search(t)) else "")
                for t in cleaned
            ]
            # Preserve a consistent explicit/column collation across all non-NULL branches.
            shared_coll = ""
            if colls and all(colls):
                norm_colls = {re.sub(r"\s+", " ", c.strip()).lower() for c in colls}
                if len(norm_colls) == 1:
                    shared_coll = colls[0]
            # Recursive array widening (e.g. SMALLINT[] + BIGINT[] -> BIGINT[], VARCHAR(5)[] + VARCHAR(50)[] -> VARCHAR[]).
            arr_matches = [re.match(r"^(.*?)((?:\[\])+)$", t) for t in no_coll]
            if all(m is not None for m in arr_matches):
                brs = {m.group(2) for m in arr_matches if m is not None}
                if len(brs) == 1:
                    br = next(iter(brs))
                    elems = [m.group(1).strip() for m in arr_matches if m is not None]
                    merged_elem = self._common_storage_type(elems, has_null=has_null)
                    if merged_elem.upper() == "VARCHAR(255)":
                        merged_elem = "VARCHAR"
                    res_arr = f"{merged_elem}{br}"
                    return f"{res_arr}{shared_coll}" if shared_coll and self._pg_type_supports_collation(res_arr) else res_arr
            if len(cleaned) == 1:
                if has_null:
                    b0 = re.sub(r"\s*\([^)]*\)\s*$", "", no_coll[0].upper()).strip()
                    if b0 in ("NUMERIC", "DECIMAL", "DEC"):
                        return "NUMERIC"
                    if b0 in ("VARCHAR", "CHARACTER VARYING"):
                        return f"VARCHAR{shared_coll}"
                    if b0 in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
                        return f"BPCHAR{shared_coll}"
                return cleaned[0]
            if all(t.upper() == cleaned[0].upper() for t in cleaned):
                if has_null:
                    b0 = re.sub(r"\s*\([^)]*\)\s*$", "", no_coll[0].upper()).strip()
                    if b0 in ("NUMERIC", "DECIMAL", "DEC"):
                        return "NUMERIC"
                    if b0 in ("VARCHAR", "CHARACTER VARYING"):
                        return f"VARCHAR{shared_coll}"
                    if b0 in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
                        return f"BPCHAR{shared_coll}"
                return cleaned[0]

            pg_num_rank = {
                "SMALLINT": (1, "SMALLINT"),
                "INT2": (1, "SMALLINT"),
                "INTEGER": (2, "INTEGER"),
                "INT": (2, "INTEGER"),
                "INT4": (2, "INTEGER"),
                "BIGINT": (3, "BIGINT"),
                "INT8": (3, "BIGINT"),
                "NUMERIC": (4, "NUMERIC"),
                "DECIMAL": (4, "NUMERIC"),
                "DEC": (4, "NUMERIC"),
                "REAL": (5, "REAL"),
                "FLOAT4": (5, "REAL"),
                "DOUBLE PRECISION": (6, "DOUBLE PRECISION"),
                "FLOAT8": (6, "DOUBLE PRECISION"),
            }
            pg_str_map = {
                "TEXT": "TEXT",
                "VARCHAR": "VARCHAR",
                "CHARACTER VARYING": "VARCHAR",
                "CHAR": "BPCHAR",
                "CHARACTER": "BPCHAR",
                "BPCHAR": "BPCHAR",
                "NCHAR": "BPCHAR",
                "NAME": "NAME",
            }
            bases = [re.sub(r"\s*\([^)]*\)\s*$", "", t.upper()).strip() for t in no_coll]
            if all(b in pg_num_rank for b in bases):
                top_rank = max(pg_num_rank[b][0] for b in bases)
                return next(name for r, name in pg_num_rank.values() if r == top_rank)
            if all(b in pg_str_map for b in bases):
                str_kinds = [pg_str_map[b] for b in bases]
                if str_kinds[0] == "TEXT":
                    res_s = "TEXT"
                elif "NAME" in str_kinds:
                    res_s = "NAME"
                else:
                    res_s = str_kinds[0]
                return f"{res_s}{shared_coll}" if shared_coll and res_s != "NAME" else res_s
            pg_time_rank = {
                "DATE": (1, "DATE"),
                "TIMESTAMP": (2, "TIMESTAMP"),
                "TIMESTAMP WITHOUT TIME ZONE": (2, "TIMESTAMP"),
                "TIMESTAMPTZ": (3, "TIMESTAMPTZ"),
                "TIMESTAMP WITH TIME ZONE": (3, "TIMESTAMPTZ"),
            }
            if all(b in pg_time_rank for b in bases):
                top_t = max(pg_time_rank[b][0] for b in bases)
                return next(name for r, name in pg_time_rank.values() if r == top_t)
            pg_clock_rank = {
                "TIME": (1, "TIME"),
                "TIME WITHOUT TIME ZONE": (1, "TIME"),
                "TIMETZ": (2, "TIMETZ"),
                "TIME WITH TIME ZONE": (2, "TIMETZ"),
            }
            if all(b in pg_clock_rank for b in bases):
                top_c = max(pg_clock_rank[b][0] for b in bases)
                return next(name for r, name in pg_clock_rank.values() if r == top_c)
            if all(b in ("BIT", "VARBIT", "BIT VARYING") for b in bases):
                return "BIT VARYING"
            if all(b == bases[0] for b in bases):
                return bases[0]
            return cleaned[0]

        if self.dialect == Dialect.MYSQL:
            if len(raw_nonempty) == 1:
                return self._normalize_dialect_type(raw_nonempty[0])
            if all(t.upper() == raw_nonempty[0].upper() for t in raw_nonempty):
                return self._normalize_dialect_type(raw_nonempty[0])

            # Keep raw TEXT/TINYTEXT/... distinct from VARCHAR(191) while stripping
            # charset/collation and promoting ZEROFILL -> UNSIGNED (MySQL semantics).
            stripped: List[str] = []
            for t in raw_nonempty:
                s = re.sub(r"\s+(?:COLLATE|CHARACTER\s+SET|CHARSET)\s+\S+", "", t.upper())
                if re.search(r"\bZEROFILL\b", s):
                    s = re.sub(r"\s*\bZEROFILL\b", "", s).strip()
                    if "UNSIGNED" not in s:
                        s = f"{s} UNSIGNED"
                stripped.append(re.sub(r"\s+", " ", s).strip())

            # Integer, DECIMAL, and floating-point numeric widening.
            int_specs: Dict[str, Tuple[int, bool, int]] = {
                # base_type: (width_rank, is_unsigned, max_decimal_digits)
                "BOOL": (1, False, 3),
                "BOOLEAN": (1, False, 3),
                "TINYINT": (1, False, 3),
                "TINYINT UNSIGNED": (1, True, 3),
                "SMALLINT": (2, False, 5),
                "SMALLINT UNSIGNED": (2, True, 5),
                "MEDIUMINT": (3, False, 7),
                "MEDIUMINT UNSIGNED": (3, True, 8),
                "INT": (4, False, 10),
                "INTEGER": (4, False, 10),
                "INT UNSIGNED": (4, True, 10),
                "INTEGER UNSIGNED": (4, True, 10),
                "BIGINT": (5, False, 19),
                "BIGINT UNSIGNED": (5, True, 20),
            }
            signed_by_rank = {1: "TINYINT", 2: "SMALLINT", 3: "MEDIUMINT", 4: "INT", 5: "BIGINT"}
            unsigned_by_rank = {
                1: "TINYINT UNSIGNED",
                2: "SMALLINT UNSIGNED",
                3: "MEDIUMINT UNSIGNED",
                4: "INT UNSIGNED",
                5: "BIGINT UNSIGNED",
            }

            def _mysql_num_kind(s: str) -> Optional[Tuple[str, Any]]:
                no_w = re.sub(r"\(\s*\d+\s*\)", "", s).strip()
                no_w = re.sub(r"\s+", " ", no_w)
                if no_w in int_specs:
                    return ("int", int_specs[no_w])
                m_dec = re.match(r"^(?:DECIMAL|NUMERIC|DEC|FIXED)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?(?:\s+UNSIGNED)?$", s)
                if m_dec:
                    p = int(m_dec.group(1)) if m_dec.group(1) else 10
                    sc = int(m_dec.group(2)) if m_dec.group(2) else 0
                    return ("dec", (p, sc))
                if re.match(r"^(?:FLOAT)(?:\s*\([^)]*\))?(?:\s+UNSIGNED)?$", s):
                    return ("float", None)
                if re.match(r"^(?:DOUBLE|DOUBLE\s+PRECISION|REAL)(?:\s*\([^)]*\))?(?:\s+UNSIGNED)?$", s):
                    return ("double", None)
                return None

            def _mysql_str_info(s: str) -> Optional[Tuple[str, int]]:
                m_c = re.match(r"^(?:N?CHAR|CHARACTER|N?VARCHAR|CHARACTER\s+VARYING)\s*\(\s*(\d+)\s*\)$", s)
                if m_c:
                    is_fixed = s.startswith(("CHAR(", "CHAR ", "NCHAR(", "NCHAR ", "CHARACTER(", "CHARACTER ")) and "VARYING" not in s
                    return ("char" if is_fixed else "varchar", int(m_c.group(1)))
                if s in ("CHAR", "CHARACTER", "NCHAR"):
                    return ("char", 1)
                if s in ("VARCHAR", "NVARCHAR", "CHARACTER VARYING"):
                    return ("varchar", 191)
                if s in ("TINYTEXT", "TEXT", "MEDIUMTEXT", "LONGTEXT"):
                    rank_map = {"TINYTEXT": 1, "TEXT": 2, "MEDIUMTEXT": 3, "LONGTEXT": 4}
                    return ("text", rank_map[s])
                if s.startswith(("ENUM(", "SET(")):
                    return ("varchar", 255)
                return None

            num_kinds = [_mysql_num_kind(s) for s in stripped]
            if all(k is not None for k in num_kinds):
                kinds = [k for k in num_kinds if k is not None]
                tag_set = {k[0] for k in kinds}
                if "double" in tag_set:
                    return "DOUBLE"
                if "float" in tag_set:
                    # Mixing FLOAT with BIGINT or DECIMAL widens to DOUBLE in MySQL/MariaDB.
                    if "dec" in tag_set or any(k[0] == "int" and k[1][0] >= 4 for k in kinds):
                        return "DOUBLE"
                    return "FLOAT"
                if tag_set == {"int"}:
                    s_ranks = [k[1][0] for k in kinds if not k[1][1]]
                    u_ranks = [k[1][0] for k in kinds if k[1][1]]
                    if not u_ranks:
                        return signed_by_rank[max(s_ranks)]
                    if not s_ranks:
                        return unsigned_by_rank[max(u_ranks)]
                    max_u = max(u_ranks)
                    max_s = max(s_ranks)
                    if max_u >= 5:
                        return "DECIMAL(20, 0)"
                    return signed_by_rank[max(max_s, max_u + 1)]
                # Exact mix of integers and/or DECIMAL(p, s)
                max_int_digits = 0
                max_scale = 0
                for tag, info in kinds:
                    if tag == "int":
                        max_int_digits = max(max_int_digits, info[2])
                    else:
                        p, sc = info
                        max_int_digits = max(max_int_digits, max(0, p - sc))
                        max_scale = max(max_scale, sc)
                tot_p = min(65, max(1, max_int_digits + max_scale))
                return f"DECIMAL({tot_p}, {max_scale})"

            # BIT(n) widening
            bit_matches = [re.match(r"^BIT(?:\s*\(\s*(\d+)\s*\))?$", s) for s in stripped]
            if all(m is not None for m in bit_matches):
                max_bits = max(int(m.group(1)) if (m is not None and m.group(1)) else 1 for m in bit_matches)
                return f"BIT({max_bits})"

            # Binary / BLOB widening
            def _mysql_bin_info(s: str) -> Optional[Tuple[str, int]]:
                m_b = re.match(r"^(BINARY|VARBINARY)\s*\(\s*(\d+)\s*\)$", s)
                if m_b:
                    return (m_b.group(1).lower(), int(m_b.group(2)))
                if s in ("TINYBLOB", "BLOB", "MEDIUMBLOB", "LONGBLOB"):
                    b_rank = {"TINYBLOB": 1, "BLOB": 2, "MEDIUMBLOB": 3, "LONGBLOB": 4}
                    return ("blob", b_rank[s])
                return None

            bin_infos = [_mysql_bin_info(s) for s in stripped]
            if all(bi is not None for bi in bin_infos):
                b_list = [bi for bi in bin_infos if bi is not None]
                if any(tag == "blob" for tag, _ in b_list):
                    max_var = max((n for tag, n in b_list if tag != "blob"), default=0)
                    top_blob = max((lvl for tag, lvl in b_list if tag == "blob"), default=2)
                    if max_var > 255 and top_blob < 2:
                        top_blob = 2
                    return {1: "TINYBLOB", 2: "BLOB", 3: "MEDIUMBLOB", 4: "LONGBLOB"}[top_blob]
                max_len = max(n for _, n in b_list)
                if all(tag == "binary" for tag, _ in b_list) and len({n for _, n in b_list}) == 1:
                    return f"BINARY({max_len})"
                return f"VARBINARY({max_len})"

            # String widening (CHAR/VARCHAR/TEXT) and MySQL numeric + string widening
            str_infos = [_mysql_str_info(s) for s in stripped]
            if all(si is not None or nk is not None for si, nk in zip(str_infos, num_kinds)) and any(si is not None for si in str_infos):
                s_list = [si for si in str_infos if si is not None]
                has_num_mix = any(si is None for si in str_infos)
                if any(tag == "text" for tag, _ in s_list):
                    # TINYTEXT max is 255 bytes (63 utf8mb4 chars); promote to TEXT if any VARCHAR exceeds 63.
                    max_vc = max((n for tag, n in s_list if tag != "text"), default=0)
                    top_text = max((lvl for tag, lvl in s_list if tag == "text"), default=2)
                    if max_vc > 63 and top_text < 2:
                        top_text = 2
                    return {1: "TINYTEXT", 2: "TEXT", 3: "MEDIUMTEXT", 4: "LONGTEXT"}[top_text]
                max_len = max(n for _, n in s_list)
                if has_num_mix:
                    return f"VARCHAR({max(max_len, 24)})"
                if all(tag == "char" for tag, _ in s_list) and len({n for _, n in s_list}) == 1:
                    return f"CHAR({max_len})"
                return f"VARCHAR({max_len})"

            # Temporal widening preserving maximum fractional-seconds precision (fsp)
            def _fsp(s: str) -> int:
                m_f = re.search(r"\(\s*(\d+)\s*\)", s)
                return int(m_f.group(1)) if m_f else 0

            base_temps = [re.sub(r"\s*\([^)]*\)\s*$", "", s).strip() for s in stripped]
            if all(bt in ("DATE", "DATETIME", "TIMESTAMP") for bt in base_temps):
                max_fsp = max(_fsp(s) for s in stripped)
                return f"DATETIME({max_fsp})" if max_fsp > 0 else "DATETIME"
            if all(bt == "TIME" for bt in base_temps):
                max_fsp = max(_fsp(s) for s in stripped)
                return f"TIME({max_fsp})" if max_fsp > 0 else "TIME"

            return self._normalize_dialect_type(raw_nonempty[0])

        return self._normalize_dialect_type(raw_nonempty[0])

    def _union_all_common_ctx(self, branch_specs: Sequence[ViewSpec]) -> Any:
        """Build a compilation context whose `primary_schema` holds the widened
        common column types across all `UNION ALL` branches (positional alignment)."""
        if not branch_specs:
            return self
        first_b = branch_specs[0]
        branch_ctxs = [
            self.with_ir(
                primary_schema=self._schemas_lower.get(bs.base_table.lower()),
                view=bs,
            )
            for bs in branch_specs
        ]
        merged_cols: Dict[str, ColumnDef] = {}
        for col_i, fb_it in enumerate(first_b.select_items):
            col_b_types: List[str] = []
            has_null = False
            has_str_lit = False
            all_not_null = True
            for b_ctx, bs in zip(branch_ctxs, branch_specs):
                if col_i >= len(bs.select_items):
                    all_not_null = False
                    continue
                bi_item = bs.select_items[col_i]
                if _is_untyped_null_expr(bi_item.raw_expr, dialect=self.dialect):
                    has_null = True
                    all_not_null = False
                    continue
                ts_b = self._schemas_lower.get(bs.base_table.lower())
                c_b: Optional[ColumnDef] = None
                if ts_b is not None and bi_item.source_col and _is_simple_column_expr(bi_item.raw_expr.strip(), dialect=self.dialect):
                    src_low = _unquote_ident(bi_item.source_col, dialect=self.dialect).lower()
                    c_b = next((c for k, c in ts_b.columns.items() if k.lower() == src_low), None)
                if c_b is not None:
                    col_b_types.append(c_b.data_type)
                    if not c_b.not_null:
                        all_not_null = False
                    continue
                all_not_null = False
                if self.dialect == Dialect.POSTGRES:
                    try:
                        lit_ast = parse_sql_ast(_unwrap_outer_parens(bi_item.raw_expr, dialect=self.dialect), dialect=self.dialect)
                        if isinstance(lit_ast, sqlglot_exp.Literal) and lit_ast.args.get("is_string"):
                            has_str_lit = True
                            continue
                    except sqlglot.errors.SqlglotError:
                        pass
                col_b_types.append(b_ctx._infer_expr_storage_type(bi_item))
            if not col_b_types and has_str_lit:
                col_b_types.append("TEXT")
            if self.dialect == Dialect.SQLITE:
                common_dt = "BLOB"
            elif col_b_types:
                common_dt = self._common_storage_type(col_b_types, has_null=has_null)
            else:
                common_dt = "VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT"
            merged_cols[fb_it.alias] = ColumnDef(
                name=fb_it.alias,
                data_type=common_dt,
                not_null=all_not_null,
            )
        merged_schema = TableSchema(
            name=self.view.base_table or "__mv_union_all",
            columns=merged_cols,
        )
        return self.with_ir(primary_schema=merged_schema)

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
        if agg_item.agg_func in (AggFunc.MIN, AggFunc.MAX) and self.dialect == Dialect.MYSQL and result:
            string_type = self._mysql_window_minmax_string_type(result)
            if string_type is not None:
                return string_type
        if (
            agg_item.agg_func in (AggFunc.MIN, AggFunc.MAX)
            and not (self.raw_view.has_cte or self.raw_view.has_subquery_in_from)
        ):
            # Window MIN/MAX: MySQL widens integer types (see
            # _mysql_widened_minmax_type). Through a CTE/derived table the
            # MIN/MAX type is already BIGINT (_mysql_derived_minmax_type).
            return self._mysql_widened_minmax_type(result)
        return result

    def _infer_expr_storage_type(self, item: SelectItem) -> str:
        routed_type = self._exact_routing_storage_type(item)
        if routed_type is not None:
            return routed_type
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
            mysql_group_key_type = self._mysql_group_key_narrow_integer_type(item.raw_expr)
            if mysql_group_key_type is not None:
                return mysql_group_key_type
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
                if self.dialect == Dialect.POSTGRES and self._pg_type_supports_collation(ast_inferred):
                    inferred_coll = self._infer_pg_expr_collation(raw_expr)
                    if inferred_coll:
                        base_no_coll = self._PG_COLLATE_TAIL_RE.sub("", ast_inferred).strip()
                        return f"{base_no_coll}{inferred_coll}"
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
            unwrapped_raw = _unwrap_outer_parens(item.raw_expr, dialect=self.dialect)
            compound_t = self._infer_ast_expr_storage_type(unwrapped_raw)
            if compound_t is not None:
                if self.dialect == Dialect.POSTGRES and self._pg_type_supports_collation(compound_t):
                    comp_coll = self._infer_pg_expr_collation(unwrapped_raw)
                    if comp_coll:
                        base_no_coll = self._PG_COLLATE_TAIL_RE.sub("", compound_t).strip()
                        return f"{base_no_coll}{comp_coll}"
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
                if arg_t in ("SMALLINT", "INT2", "SMALLSERIAL", "SERIAL2"):
                    return "SMALLINT"
                if arg_t in ("INTEGER", "INT", "INT4", "SERIAL", "SERIAL4"):
                    return "INTEGER"
                if arg_t == "BIT" or re.match(r"^BIT\s*\(\s*\d+\s*\)$", arg_t):
                    return arg_t
                if arg_t in ("VARBIT", "BIT VARYING") or re.match(r"^(?:VARBIT|BIT\s+VARYING)\s*\(\s*\d+\s*\)$", arg_t):
                    return "BIT VARYING"
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
                return self._minmax_result_type(self._normalize_dialect_type(c_def.data_type), item.arg_expr)
            arg_inferred = self._infer_expr_storage_type(
                SelectItem(raw_expr=item.arg_expr, alias=item.alias, is_aggregate=False)
            )
            if (
                self.dialect == Dialect.MYSQL
                and arg_inferred
                and re.fullmatch(r"(?:TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT)", arg_inferred.strip().upper())
                and not re.fullmatch(r"(?:`?\w+`?\.)?`?\w+`?", item.arg_expr.strip())
            ):
                # MySQL/MariaDB MIN/MAX of an integer expression is BIGINT even
                # where the expression itself is INT. A bare column of a CTE or
                # derived table is not an expression: MariaDB keeps its type
                # (MAX(depth) of a recursive CTE's INT column is INT), and
                # MySQL's derived-table rule applies in _minmax_result_type.
                return "BIGINT"
            if arg_inferred:
                return self._minmax_result_type(arg_inferred, item.arg_expr)
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

    _EXPLICIT_COLLATE_RE = re.compile(
        r"^(?P<inner>.*\S)\s+COLLATE\s+(?P<coll>\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)\s*$",
        re.IGNORECASE | re.DOTALL,
    )

    # --- group-key collation (see matview/key_collation.py) -------------------

    def _key_tables(self) -> List[TableSchema]:
        tables: List[TableSchema] = []
        seen: Set[str] = set()
        if self.primary_schema is not None:
            tables.append(self.primary_schema)
            seen.add(self.primary_schema.name.lower())
        for t_name in self.view.referenced_tables:
            ts = self._schemas_lower.get(t_name.lower())
            if ts is not None and ts.name.lower() not in seen:
                tables.append(ts)
                seen.add(ts.name.lower())
        return tables

    def _key_col_signature(self, c: ColumnDef) -> Tuple[Any, ...]:
        if self.dialect == Dialect.MYSQL:
            return (key_collation.mysql_column_is_character(c), (c.effective_collation or "").upper())
        return ((c.effective_collation or "BINARY").upper(),)

    def _key_resolver_by_name(self, tables: Optional[Sequence[TableSchema]] = None) -> key_collation.ColumnResolver:
        """Resolve a column by its bare name over the view's tables. This is what the
        key text sees (trigger bodies carry ``NEW.col``/``OLD.col``/table-qualified
        references), so a name declared differently by several tables is ambiguous."""
        tbls = list(tables) if tables is not None else self._key_tables()

        def resolve(col: sqlglot_exp.Column) -> Optional[ColumnDef]:
            name = (col.name or "").lower()
            if (
                tables is None
                and getattr(self, "_alg_mode", "single") == "union_all_agg"
                and not getattr(self, "_uall_orig_item_map", None)
            ):
                # The outer query of a UNION ALL aggregate names subquery columns.
                return self._uall_alias_column(name)
            found = [c for ts in tbls for c in ts.columns.values() if c.name.lower() == name]
            if not found or len({self._key_col_signature(c) for c in found}) != 1:
                return None
            return found[0]

        return resolve

    def _uall_alias_column(self, name: str) -> Optional[ColumnDef]:
        """The declared column behind a UNION ALL subquery column ``name`` (the outer
        GROUP BY of a UNION ALL aggregate names it): the first branch's plain-column
        projection of that name; None when a branch projects an expression."""
        cache: Dict[str, Optional[ColumnDef]] = self.__dict__.setdefault("_uall_alias_cache", {})
        if name in cache:
            return cache[name]
        found: Optional[ColumnDef] = None
        try:
            meta = self._extract_union_all_agg_metadata()
        except Exception:
            meta = None
        if meta is not None and meta[2]:
            bs = meta[2][0]
            src = next((it for it in bs.select_items if it.alias.lower() == name), None)
            if src is not None:
                try:
                    node = parse_sql_ast(src.raw_expr, dialect=self.dialect)
                except sqlglot.errors.SqlglotError:
                    node = None
                if isinstance(node, sqlglot_exp.Column):
                    found = self._key_resolver_by_alias(bs)(node)
        cache[name] = found
        return found

    def _key_resolver_by_alias(self, spec: Optional[ViewSpec] = None) -> key_collation.ColumnResolver:
        """Resolve a column the way the original view does: through its table alias."""
        v = spec or self.view
        by_alias: Dict[str, TableSchema] = {}
        for j in v.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts is not None:
                by_alias[j.alias.lower()] = ts
                by_alias.setdefault(j.table_name.lower(), ts)
        all_tables = list({id(ts): ts for ts in by_alias.values()}.values())
        by_name = self._key_resolver_by_name(all_tables)

        def resolve(col: sqlglot_exp.Column) -> Optional[ColumnDef]:
            qual = (col.table or "").lower()
            if qual:
                ts = by_alias.get(qual)
                if ts is None:
                    return None
                for c in ts.columns.values():
                    if c.name.lower() == (col.name or "").lower():
                        return c
                return None
            return by_name(col)

        return resolve

    def _classify_key(self, expr: str, resolve: Optional[key_collation.ColumnResolver] = None) -> key_collation.KeyClass:
        res = resolve if resolve is not None else self._key_resolver_by_name()
        if self.dialect == Dialect.SQLITE:
            return key_collation.classify_sqlite_key(expr, res)
        return key_collation.classify_mysql_key(expr, res, self.dialect)

    def _key_class_for_key(self, expr: str) -> key_collation.KeyClass:
        """Classification used while emitting key text. An expression the planner
        did not route to a full refresh must be derivable; anything else is refused
        here rather than keyed byte-exactly (which would silently split groups)."""
        cls = self._classify_key(expr)
        if cls.kind == key_collation.KeyKind.UNKNOWN:
            if cls.missing_column:
                # Internal (hidden/aux) column: never a user-collated value.
                return key_collation.KeyClass(
                    key_collation.KeyKind.TEXT if self.dialect == Dialect.SQLITE else key_collation.KeyKind.PLAIN,
                    "BINARY" if self.dialect == Dialect.SQLITE else None,
                )
            raise UnsupportedMaterializedViewError(
                f"cannot derive the comparison collation of key expression {expr.strip()}: {cls.reason}"
            )
        return cls

    def _group_key_collation(self, expr: str, strict: bool = False) -> Tuple[str, Optional[str]]:
        """``(expression without a trailing COLLATE, collation)`` of a group-key
        expression: the collation the engine compares it under (SQLite: derived;
        MySQL/MariaDB: the declared collation of the character column it derives
        from, a hint); ``None`` when no collation applies. With ``strict`` an
        expression whose collation cannot be derived is refused (key text); without,
        it has no collation (storage declarations of non-key columns)."""
        if strict:
            cls = self._key_class_for_key(expr)
        else:
            cls = self._classify_key(expr)
            if cls.kind == key_collation.KeyKind.UNKNOWN:
                return expr, None
        coll = cls.collation if cls.kind == key_collation.KeyKind.TEXT else None
        inner = expr
        m = self._EXPLICIT_COLLATE_RE.match(expr.strip())
        if m:
            try:
                root = parse_sql_ast(expr.strip(), dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                root = None
            if isinstance(root, sqlglot_exp.Collate):
                inner = m.group("inner")
        return inner, coll

    def _key_collation_refusal(self, cls: key_collation.KeyClass) -> Optional[str]:
        """Why a classified key cannot be keyed exactly, or None when it can."""
        if cls.kind == key_collation.KeyKind.UNKNOWN:
            return cls.reason or "its collation cannot be derived"
        if cls.kind != key_collation.KeyKind.TEXT:
            return None
        name = (cls.collation or "").upper()
        if self.dialect == Dialect.SQLITE:
            if name not in ("BINARY", "NOCASE", "RTRIM"):
                return (
                    f"it compares under collation {name}, which matview cannot reproduce in its group-key text "
                    "(supported: BINARY, NOCASE, RTRIM)"
                )
            return None
        if self.dialect == Dialect.MYSQL and self.subdialect != EngineSubdialect.MARIADB:
            # A character value that derives from no column compares under the
            # session's collation_connection (known when a connection was probed).
            effective = name or (self.engine_version.mysql_connection_collation or "")
            if key_collation.mysql_collation_is_plan_dependent(effective):
                return (
                    f"it compares under collation {effective.upper()}, a MySQL PAD SPACE collation under which "
                    "GROUP BY / DISTINCT treat characters weighing like a space (NBSP, ideographic space, ...) as "
                    "trailing padding in some execution plans (index scan, COUNT(DISTINCT)) and not in others, so "
                    "the original view has no single result for such values and no group key can reproduce it "
                    "(MySQL groups PAD SPACE UCA text plan-dependently)"
                )
        if self.dialect == Dialect.MYSQL and key_collation.mysql_collation_has_unsound_weights(name):
            return (
                f"it compares under collation {name}, a multi-level collation whose equality is not a function of "
                "its weight string (or whose pad weights cannot be normalised)"
            )
        return None

    def _sqlite_key_collate_suffix(self, expr: str) -> str:
        """`` COLLATE NOCASE`` / `` COLLATE RTRIM`` for a backing group-key column
        (so comparisons of the stored key use the grouped expression's collation,
        also in the reader view); empty for BINARY, other engines or no collation."""
        if self.dialect != Dialect.SQLITE:
            return ""
        _, coll = self._group_key_collation(expr)
        return f" COLLATE {coll}" if coll in ("NOCASE", "RTRIM") else ""

    def _key_col_collate_suffix(self, expr: str, data_type: str = "") -> str:
        """`` COLLATE x`` for a MySQL/MariaDB character backing key column whose
        grouped expression derives from a character column with a known collation
        (so the stored key compares like the base expression); SQLite: see
        ``_sqlite_key_collate_suffix``."""
        if self.dialect == Dialect.SQLITE:
            return self._sqlite_key_collate_suffix(expr)
        if self.dialect != Dialect.MYSQL or "COLLATE" in data_type.upper():
            return ""
        _, coll = self._group_key_collation(expr)
        dt = data_type.upper()
        if not coll or coll == "BINARY" or not ("CHAR" in dt or "TEXT" in dt) or "CHARACTER SET" in dt:
            return ""
        charset = coll.lower().split("_", 1)[0]
        return f" CHARACTER SET {charset} COLLATE {coll.lower()}"

    def _mysql_text_key_part(self, expr: str) -> Optional[str]:
        """Key text of a MySQL/MariaDB character-valued expression: a digest of the
        server's own ``WEIGHT_STRING`` of the expression, which is what its grouping
        comparison uses (any case-/accent-insensitive collation merges variants
        into one weight string). Under a PAD SPACE collation trailing pad weights are
        insignificant, so trailing units equal to the weight of a space (under the
        expression's own collation; NBSP and other characters weighing like a space
        included) are removed first. Whether the collation pads is asked of the
        server: ``' ' = ''`` holds under PAD SPACE only. None for expressions that
        are not character-valued."""
        cls = self._key_class_for_key(expr)
        if cls.kind != key_collation.KeyKind.TEXT:
            return None
        why = self._key_collation_refusal(cls)
        if why:
            raise UnsupportedMaterializedViewError(f"cannot key {expr.strip()}: {why}")
        spc = f"RIGHT(CONCAT({expr}, ' '), 1)"
        w = f"HEX(WEIGHT_STRING({expr}))"
        if self.subdialect != EngineSubdialect.MARIADB:
            # MySQL's GROUP BY / DISTINCT / UNION treat only trailing U+0020 as
            # insignificant padding (NBSP and other characters weighing like a space
            # are significant there, although ``=`` pads by weight).
            padded = f"HEX(WEIGHT_STRING(RTRIM({expr})))"
        else:
            # MariaDB pads by weight (``'a'`` = ``'a'`` + NBSP, ideographic space, ...
            # under the UCA collations): drop the trailing weight units equal to the
            # weight of a space. The unit is the weight string of one space under the
            # expression's own collation, trimmed from the end (for collations with
            # uniform weight units this is the aligned prefix before the trailing run;
            # unlike the anchored regular expression used before, it never maps a
            # weight string that is not a whole number of units to '').
            # ``TRIM(TRAILING unit ...)`` under PAD SPACE, and a unit that never
            # occurs in a hex string ('G') under NO PAD, keeps the expression to one
            # weight string of the value and one of the space with a single
            # comparison: every extra item in a MariaDB trigger statement costs
            # memory and time on each trigger's first execution, and the previous
            # form (a row-dependent REGEXP_SUBSTR pattern plus a second weight string
            # under an IF) made the MariaDB CI jobs about 1.3-1.45x slower. An empty
            # unit trims nothing, so the weight string is then kept whole.
            unit = f"IF({spc} = '', HEX(WEIGHT_STRING({spc})), 'G')"
            return f"IF({expr} IS NULL, 'N', CONCAT('W', SHA2(TRIM(TRAILING {unit} FROM {w}), 256)))"
        return f"IF({expr} IS NULL, 'N', CONCAT('W', SHA2(IF({spc} = '', {padded}, {w}), 256)))"

    def _sqlite_collation_folded(self, expr: str) -> str:
        """``expr`` rewritten so that values equal under the group key's
        collation have the same canonical key text: NOCASE folds ASCII case (what
        the collation itself folds), RTRIM drops trailing spaces; BINARY keeps
        the value. Other collations cannot be reproduced in a key text on SQLite
        (the planner routes those views to a full refresh instead)."""
        inner, coll = self._group_key_collation(expr, strict=True)
        if coll in (None, "BINARY"):
            return inner
        e = f"({inner})"
        if coll == "NOCASE":
            return f"(CASE WHEN typeof({e}) = 'text' THEN lower({e}) ELSE {e} END)"
        if coll == "RTRIM":
            return f"(CASE WHEN typeof({e}) = 'text' THEN rtrim({e}, ' ') ELSE {e} END)"
        raise UnsupportedMaterializedViewError(
            f"GROUP BY / DISTINCT key {inner.strip()} uses collation {coll}, which matview cannot reproduce in "
            "its group-key text on SQLite (supported: BINARY, NOCASE, RTRIM)."
        )

    def _key_spelling_sensitive(self, cls: key_collation.KeyClass) -> bool:
        """True when distinct spellings can compare equal under the key's collation."""
        if cls.kind != key_collation.KeyKind.TEXT:
            return False
        if self.dialect == Dialect.SQLITE:
            return (cls.collation or "BINARY") in ("NOCASE", "RTRIM")
        return key_collation.mysql_collation_has_spelling_variants(cls.collation)

    def _key_spelling_refresh_needed(self) -> bool:
        """True when a displayed group key is spelling-sensitive (its spelling is
        refreshed from the group's base rows when the supplying row leaves)."""
        if self.dialect not in (Dialect.SQLITE, Dialect.MYSQL) or not self._effective_group_by_exprs:
            return False
        try:
            return any(
                self._key_spelling_sensitive(self._key_class_for_key(it.raw_expr))
                for it in self._working_select_items
                if not it.is_aggregate
            )
        except UnsupportedMaterializedViewError:
            return False

    def _key_collation_fallback_reason(self, strategy: MaintenanceStrategy, mode: str) -> Optional[str]:
        """Why the view cannot keep its collation-sensitive keys exact under the
        maintenance ``mode`` the planner picked, or None when it can. A reason sends
        the view to the full refresh of the original query (exact by construction)
        with an announced diagnostic instead of a key text that could split or
        merge groups differently from the engine.

        Collation-sensitive expressions: GROUP BY / window PARTITION BY keys, the
        columns of DISTINCT and of de-duplicating set operations (per output column
        across all branches), and DISTINCT aggregate arguments. Each must derive an
        exactly known collation (``matview.key_collation``), the by-name resolution
        used while the key text is emitted must agree with the alias-aware one of the
        original view, and the branches of a set operation / UNION ALL aggregate
        must agree per column."""
        if self.dialect not in (Dialect.SQLITE, Dialect.MYSQL) or strategy == MaintenanceStrategy.ATOMIC_QUERY_SYNC:
            return None
        v = self.view
        # (label, [(expr, alias-aware resolver, by-name resolver) per branch])
        positions: List[Tuple[str, List[Tuple[str, key_collation.ColumnResolver, key_collation.ColumnResolver]]]] = []
        # DISTINCT aggregate arguments: their value store carries the argument's
        # collation, so no displayed spelling is involved.
        argument_labels: Set[str] = set()
        proper = self._key_resolver_by_alias()
        byname = self._key_resolver_by_name()
        if mode in ("single", "join", "window_partition"):
            scope = self._tier2_scope_exprs() if mode == "window_partition" else tuple(self._effective_group_by_exprs)
            for g in scope:
                positions.append((f"group key {g.strip()}", [(g, proper, byname)]))
            for it in self._working_select_items:
                if it.is_aggregate and it.is_distinct and it.arg_expr:
                    label = f"DISTINCT aggregate argument {it.arg_expr.strip()}"
                    argument_labels.add(label)
                    positions.append((label, [(it.arg_expr, proper, byname)]))
        elif mode == "distinct":
            for it in v.select_items:
                if not it.is_aggregate:
                    positions.append((f"DISTINCT column {it.raw_expr.strip()}", [(it.raw_expr, proper, byname)]))
        elif mode == "setop":
            branches = self._parse_setop_branches()
            if branches and branches[0][0].upper() != "UNION ALL":
                width = len(branches[0][1].select_items)
                for i in range(width):
                    items = []
                    for _, bs in branches:
                        if len(bs.select_items) != width:
                            return "the branches of the set operation differ in column count"
                        items.append((bs.select_items[i].raw_expr, self._key_resolver_by_alias(bs), byname))
                    positions.append((f"set-operation column {i + 1} ({items[0][0].strip()})", items))
        elif mode == "union_all_agg":
            meta = self._extract_union_all_agg_metadata()
            if meta is None:
                return None
            for g in self._effective_group_by_exprs:
                try:
                    g_ast = parse_sql_ast(g, dialect=self.dialect)
                except sqlglot.errors.SqlglotError:
                    return f"group key {g.strip()} cannot be parsed"
                if not isinstance(g_ast, sqlglot_exp.Column):
                    return f"group key {g.strip()} is an expression over the UNION ALL subquery"
                items = []
                for bs in meta[2]:
                    src = next((it for it in bs.select_items if it.alias.lower() == (g_ast.name or "").lower()), None)
                    ts = self._schemas_lower.get(bs.base_table.lower())
                    if src is None or ts is None:
                        return f"group key {g.strip()} is not a plain column of every UNION ALL branch"
                    try:
                        src_node = parse_sql_ast(src.raw_expr, dialect=self.dialect)
                    except sqlglot.errors.SqlglotError:
                        src_node = None
                    if not isinstance(src_node, sqlglot_exp.Column):
                        return f"group key {g.strip()} is an expression ({src.raw_expr.strip()}) in a UNION ALL branch"
                    items.append((src.raw_expr, self._key_resolver_by_alias(bs), self._key_resolver_by_name([ts])))
                positions.append((f"group key {g.strip()}", items))
        for label, items in positions:
            classes: List[key_collation.KeyClass] = []
            for expr, res_proper, res_name in items:
                cp = self._classify_key(expr, res_proper)
                cn = self._classify_key(expr, res_name)
                why = self._key_collation_refusal(cp)
                if why:
                    return f"{label}: {why}"
                if self.dialect == Dialect.MYSQL and label not in argument_labels:
                    try:
                        key_node = parse_sql_ast(expr, dialect=self.dialect)
                    except sqlglot.errors.SqlglotError:
                        key_node = None
                    key_col = res_proper(key_node) if isinstance(key_node, sqlglot_exp.Column) else None
                    if key_col is not None and re.match(r"\s*(?:MEDIUM|LONG)(?:TEXT|BLOB)\b", key_col.data_type or "", re.I):
                        return (
                            f"{label}: MySQL/MariaDB cannot index a {key_col.data_type.strip()} column "
                            "(error 1170), and the group key is stored in an indexed backing column"
                        )
                if (cp.kind, cp.collation) != (cn.kind, cn.collation):
                    return f"{label}: a column name is declared differently by several tables of the view"
                if label not in argument_labels and self._key_spelling_sensitive(cp) and not (
                    strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA
                    or (strategy == MaintenanceStrategy.ALGEBRAIC_DELTA and mode in ("single", "join", "union_all_agg", "distinct"))
                ):
                    return (
                        f"{label}: values equal under its collation can be spelled differently, and the "
                        f"displayed spelling must stay one some member row has, which this maintenance mode "
                        f"({mode}) does not keep up when the row supplying it leaves"
                    )
                classes.append(cp)
            if len(classes) > 1:
                first = classes[0]
                if self.dialect == Dialect.SQLITE:
                    same = all(c.collation == first.collation for c in classes)
                    if not (same or first.collation == "BINARY"):
                        return f"{label}: its branches compare under different collations"
                else:
                    same = all(c.kind == first.kind and (c.kind != key_collation.KeyKind.TEXT or (
                        c.collation == first.collation)) for c in classes)
                    if not same:
                        return f"{label}: its branches compare under different collations"
        return None

    def _mysql_widened_minmax_type(self, arg_type: str) -> str:
        """MySQL's window MIN/MAX result type.

        Measured natively on MySQL 8.4 and 9.7 (wire type and catalog):
        TINYINT/SMALLINT/MEDIUMINT (incl. BOOL, TINYINT(1)) -> INT, INT ->
        BIGINT, each keeping UNSIGNED; YEAR -> INT UNSIGNED; BIGINT unchanged;
        BIT(1..9) -> INT UNSIGNED and BIT(10..64) -> BIGINT UNSIGNED on 8.4,
        while 9.x keeps BIT. MariaDB and other
        dialects keep the argument type.
        """
        if self.dialect != Dialect.MYSQL or self.subdialect == EngineSubdialect.MARIADB:
            return arg_type
        u = re.sub(r"\s+", " ", arg_type.strip().upper())
        unsigned = " UNSIGNED" if "UNSIGNED" in u else ""
        base = re.sub(r"\s*\(\d+\)", "", u).replace(" UNSIGNED", "").replace(" ZEROFILL", "").strip()
        if base in ("TINYINT", "SMALLINT", "MEDIUMINT", "BOOL", "BOOLEAN"):
            return "INT" + unsigned
        if base in ("INT", "INTEGER"):
            return "BIGINT" + unsigned
        if base == "YEAR":
            return "INT UNSIGNED"
        bit = re.match(r"^BIT(?:\s*\((\d+)\))?", u)
        if bit:
            # Window MIN/MAX over BIT changed between the measured releases:
            # MySQL 8.4 widens it to an integer, 9.7 keeps BIT (bytes vs int
            # values). Only those release series are known; any other version
            # (8.0, 9.0-9.6, later, or an unspecified version) is refused.
            major_minor = tuple(self.engine_version.version[:2])
            if major_minor == (9, 7):
                return arg_type
            if major_minor == (8, 4):
                width = int(bit.group(1) or 1)
                return "INT UNSIGNED" if width <= 9 else "BIGINT UNSIGNED"
            raise UnsupportedMaterializedViewError(
                f"Window MIN/MAX over {arg_type} on MySQL {self.engine_version.label}: its result type "
                "is only known for MySQL 8.4 (integer) and 9.7 (BIT); this version was not measured."
            )
        return arg_type

    @staticmethod
    def _mysql_window_minmax_string_type(arg_type: str) -> Optional[str]:
        """MySQL window MIN/MAX over CHAR(n), ENUM or SET returns VARCHAR
        (measured on MySQL 8.4/9.7: type 253 with the argument's character
        length; ENUM/SET with the longest member), while the grouped
        aggregate keeps CHAR. Returns None for other types."""
        u = arg_type.strip()
        m = re.match(r"(?i)^(?:CHAR|CHARACTER|NCHAR)\s*\(\s*(\d+)\s*\)(.*)$", u)
        if m:
            return f"VARCHAR({int(m.group(1))}){m.group(2)}"
        if re.match(r"(?i)^CHAR(?:ACTER)?\s*$", u):
            return "VARCHAR(1)"
        m = re.match(r"(?i)^(ENUM|SET)\s*\((.*)\)\s*$", u, re.S)
        if m:
            members = [x.replace("''", "'") for x in re.findall(r"'((?:''|[^'])*)'", m.group(2))]
            if not members:
                return None
            if m.group(1).upper() == "ENUM":
                width = max(len(x) for x in members)
            else:
                width = sum(len(x) for x in members) + len(members) - 1
            return f"VARCHAR({max(width, 1)})"
        return None

    def _mysql_derived_minmax_type(self, arg_type: str) -> str:
        """MySQL's MIN/MAX of an integer-like column read through a CTE or
        derived table: always sent as BIGINT (LONGLONG) on MySQL 8.4/9.7,
        whatever the argument's integer type (TINYINT .. BIGINT, UNSIGNED,
        YEAR, BIT, BOOL; measured natively), although the catalog keeps the
        argument type. Unsigned and BIT values keep an unsigned column."""
        if self.dialect != Dialect.MYSQL or self.subdialect == EngineSubdialect.MARIADB:
            return arg_type
        u = re.sub(r"\s+", " ", arg_type.strip().upper())
        base = re.sub(r"\s*\(\d+\)", "", u).replace(" UNSIGNED", "").replace(" ZEROFILL", "").strip()
        if base in ("TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "BIGINT", "BOOL", "BOOLEAN", "YEAR"):
            return "BIGINT UNSIGNED" if "UNSIGNED" in u else "BIGINT"
        if base.startswith("BIT"):
            return "BIGINT UNSIGNED"
        return arg_type

    def _reads_recursive_cte(self, arg_expr: Optional[str]) -> bool:
        """Whether a bare column argument is read from a recursive CTE.

        MySQL 8.4/9.7 keep the column type for MIN/MAX over a recursive CTE's
        column (TINYINT stays TINYINT), while a non-recursive CTE or derived
        table sends BIGINT, even under the RECURSIVE keyword (measured
        natively). The source is the column's qualifier, or the single FROM
        source of the outer query when unqualified; a CTE is recursive when its
        body references its own name.
        """
        if not arg_expr:
            return False
        m = re.fullmatch(r"\s*(?:`?(\w+)`?\.)?`?(\w+)`?\s*", arg_expr)
        if m is None:
            return False
        try:
            q = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return False
        with_ = q.args.get("with") or q.args.get("with_")
        if not isinstance(with_, sqlglot_exp.With) or not with_.args.get("recursive"):
            return False
        recursive = {
            cte.alias_or_name.lower()
            for cte in with_.expressions
            if any(t.name.lower() == cte.alias_or_name.lower() for t in cte.this.find_all(sqlglot_exp.Table))
        }
        if not recursive:
            return False
        sources = {}
        from_ = q.args.get("from") or q.args.get("from_")
        tables = [from_.this] if from_ is not None else []
        tables += [j.this for j in q.args.get("joins") or []]
        for t in tables:
            if isinstance(t, sqlglot_exp.Table):
                sources[(t.alias_or_name or t.name).lower()] = t.name.lower()
        if m.group(1):
            src = sources.get(m.group(1).lower())
        elif len(sources) == 1:
            src = next(iter(sources.values()))
        else:
            return False
        return src in recursive

    def _reads_setop_cte_or_derived(self, arg_expr: Optional[str]) -> bool:
        """Whether a bare column argument is read from a set-operation CTE or
        derived table (`UNION [ALL]` / `INTERSECT` / `EXCEPT`).

        Like recursive CTEs, MySQL 8.4/9.7 materialize set-operation CTEs and
        derived tables into a temporary table holding the widened common column
        type, so `MIN`/`MAX` over an integer column keeps that type (e.g. `INT`
        stays `INT`, `SMALLINT + INT` stays `INT`) rather than widening to
        `BIGINT` as single-SELECT non-recursive CTEs/derived tables do.
        """
        if not arg_expr:
            return False
        m = re.fullmatch(r"\s*(?:`?(\w+)`?\.)?`?(\w+)`?\s*", arg_expr)
        if m is None:
            return False
        try:
            q = parse_sql_ast(self.raw_view.raw_select_sql, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return False
        setop_ctes: Set[str] = set()
        with_ = q.args.get("with") or q.args.get("with_")
        if isinstance(with_, sqlglot_exp.With):
            for cte in with_.expressions:
                body = cte.this
                while isinstance(body, sqlglot_exp.Subquery):
                    body = body.this
                if isinstance(body, sqlglot_exp.SetOperation):
                    setop_ctes.add(cte.alias_or_name.lower())
        sources: Dict[str, bool] = {}
        from_ = q.args.get("from") or q.args.get("from_")
        tables = [from_.this] if from_ is not None else []
        tables += [j.this for j in q.args.get("joins") or []]
        for t in tables:
            if isinstance(t, sqlglot_exp.Table):
                sources[(t.alias_or_name or t.name).lower()] = t.name.lower() in setop_ctes
            elif isinstance(t, sqlglot_exp.Subquery):
                body = t.this
                while isinstance(body, sqlglot_exp.Subquery):
                    body = body.this
                alias = (t.alias or "").lower()
                sources[alias] = isinstance(body, sqlglot_exp.SetOperation)
        if m.group(1):
            return bool(sources.get(m.group(1).lower(), False))
        if len(sources) == 1:
            return bool(next(iter(sources.values())))
        return False

    _PG_COLLATE_TAIL_RE = re.compile(
        r'\s+COLLATE\s+(?:(?:"(?:""|[^"])*"|[A-Za-z_][A-Za-z0-9_$]*)\.)?'
        r'(?:"(?:""|[^"])*"|[A-Za-z_][A-Za-z0-9_$]*)\s*$',
        re.IGNORECASE,
    )

    def _extract_pg_explicit_collation(self, expr: Optional[str]) -> Optional[str]:
        """Return ``' COLLATE <coll>'`` when ``expr`` has a top-level PostgreSQL COLLATE clause."""
        if self.dialect != Dialect.POSTGRES or not expr or not expr.strip():
            return None
        try:
            node = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None
        while isinstance(node, sqlglot_exp.Paren) and node.this is not None:
            node = node.this
        if isinstance(node, sqlglot_exp.Collate) and node.expression is not None:
            coll_sql = node.expression.sql(dialect="postgres").strip()
            if coll_sql:
                return f" COLLATE {coll_sql}"
        return None

    def _infer_pg_expr_collation(self, expr: Optional[str]) -> Optional[str]:
        """Infer the PostgreSQL result collation clause (``' COLLATE <coll>'``)
        of an expression when it carries an explicit ``COLLATE`` or a consistent
        non-default implicit column collation (propagated through string
        functions, ``||``, ``CASE``, ``COALESCE``, ``NULLIF``, ``MIN``/``MAX``).
        Returns ``None`` when the expression resolves to the database default
        collation or when multiple non-default collations conflict."""
        if self.dialect != Dialect.POSTGRES or not expr or not expr.strip():
            return None
        try:
            root = parse_sql_ast(expr, dialect=self.dialect)
        except sqlglot.errors.SqlglotError:
            return None

        def _norm_coll_key(coll_clause: str) -> str:
            # Normalize pg_catalog."C" vs "C" for consistency comparison.
            body = re.sub(r"^\s*COLLATE\s+", "", coll_clause.strip(), flags=re.IGNORECASE)
            body = re.sub(r"^(?:pg_catalog\.)", "", body, flags=re.IGNORECASE)
            return body.strip()

        def _extract_from_dt(dt: Optional[str]) -> Optional[str]:
            if not dt:
                return None
            m = self._PG_COLLATE_TAIL_RE.search(dt.strip())
            if m:
                return m.group(0).rstrip()
            base_u = re.sub(r"\s*\([^)]*\)", "", dt.strip().upper()).strip()
            if base_u == "NAME":
                return ' COLLATE "C"'
            return None

        def _merge_children(children: Sequence[Any]) -> Tuple[Optional[str], bool]:
            # Returns (coll_clause, is_explicit). Explicit COLLATE overrides implicit.
            explicits: List[str] = []
            implicits: List[str] = []
            for ch in children:
                if ch is None:
                    continue
                c_val, c_exp = _walk(ch)
                if c_val is None:
                    continue
                if c_exp:
                    explicits.append(c_val)
                else:
                    implicits.append(c_val)
            if explicits:
                keys = {_norm_coll_key(c) for c in explicits}
                return (explicits[0], True) if len(keys) == 1 else (None, False)
            if implicits:
                keys = {_norm_coll_key(c) for c in implicits}
                return (implicits[0], False) if len(keys) == 1 else (None, False)
            return None, False

        def _walk(node: Any) -> Tuple[Optional[str], bool]:
            if node is None:
                return None, False
            if isinstance(node, sqlglot_exp.Paren):
                return _walk(node.this)
            if isinstance(node, sqlglot_exp.Collate) and node.expression is not None:
                coll_sql = node.expression.sql(dialect="postgres").strip()
                if coll_sql:
                    return f" COLLATE {coll_sql}", True
                return None, False
            if isinstance(node, sqlglot_exp.Column):
                c_def = self._lookup_col_def(node.sql()) or self._lookup_col_def(node.name)
                if c_def is not None:
                    return _extract_from_dt(c_def.data_type), False
                cte_t = self._infer_cte_or_subquery_col_type(node.sql()) or self._infer_cte_or_subquery_col_type(node.name)
                return _extract_from_dt(cte_t), False
            if isinstance(node, (sqlglot_exp.Min, sqlglot_exp.Max)):
                return _walk(node.this)
            if isinstance(node, sqlglot_exp.Distinct):
                return _merge_children(list(node.iter_expressions()))
            if isinstance(node, sqlglot_exp.Window):
                return _walk(node.this)
            if isinstance(node, sqlglot_exp.Case):
                branches = [node.args.get("default")] + [
                    b.args.get("true") for b in (node.args.get("ifs") or ())
                ]
                return _merge_children(branches)
            if isinstance(node, sqlglot_exp.If):
                return _merge_children([node.args.get("true"), node.args.get("false")])
            if isinstance(node, (sqlglot_exp.Coalesce, sqlglot_exp.Greatest, sqlglot_exp.Least, sqlglot_exp.Array)):
                return _merge_children(list(node.iter_expressions()))
            if isinstance(node, sqlglot_exp.Nullif):
                return _merge_children([node.this, node.expression])
            if isinstance(node, sqlglot_exp.DPipe):
                return _merge_children([node.this, node.expression])
            if isinstance(node, sqlglot_exp.Func):
                fn_name = (node.sql_name() or node.__class__.__name__).upper()
                if fn_name in (
                    "UPPER",
                    "LOWER",
                    "TRIM",
                    "LTRIM",
                    "RTRIM",
                    "BTRIM",
                    "SUBSTR",
                    "SUBSTRING",
                    "REPLACE",
                    "LEFT",
                    "RIGHT",
                    "LPAD",
                    "RPAD",
                    "REVERSE",
                    "INITCAP",
                    "CONCAT",
                    "CONCAT_WS",
                    "IFNULL",
                ):
                    return _merge_children(list(node.iter_expressions()))
            return None, False

        coll_val, _ = _walk(root)
        return coll_val

    @classmethod
    def _pg_type_supports_collation(cls, data_type: str) -> bool:
        """Whether a PostgreSQL storage type accepts a column-level COLLATE clause."""
        clean = cls._PG_COLLATE_TAIL_RE.sub("", data_type.strip()).strip()
        m_arr = re.match(r"^(.*?)((?:\[\])+)$", clean)
        elem = m_arr.group(1).strip() if m_arr else clean
        base = re.sub(r"\s*\([^)]*\)", "", re.sub(r"\s+", " ", elem.upper())).strip()
        return base in (
            "TEXT",
            "VARCHAR",
            "CHARACTER VARYING",
            "CHAR",
            "CHARACTER",
            "BPCHAR",
            "NCHAR",
            "NAME",
            "CITEXT",
        )

    @classmethod
    def _pg_type_has_variable_numeric_scale(cls, data_type: str) -> bool:
        """Whether a PostgreSQL type string can store equal values with
        different display scales (unconstrained ``NUMERIC``, multi-dimensional
        ``NUMERIC[]``/``NUMERIC[][]``, ``NUMRANGE``/``NUMRANGE[]``,
        ``NUMMULTIRANGE``/``NUMMULTIRANGE[]``). Fixed-scale ``NUMERIC(p, s)``
        returns False."""
        if not data_type or not data_type.strip():
            return False
        clean = cls._PG_COLLATE_TAIL_RE.sub("", data_type.strip()).strip()
        elem = re.sub(r"(?:\[\])+$", "", clean).strip()
        dt_u = re.sub(r"\s+", " ", elem.upper())
        return dt_u in ("NUMERIC", "DECIMAL", "DEC", "NUMRANGE", "NUMMULTIRANGE")

    def _pg_expr_has_variable_numeric_scale(self, expr: str) -> bool:
        """Whether a PostgreSQL expression's type can store equal values with
        different display scales (e.g. unconstrained ``NUMERIC`` 1.0 vs 1.00,
        domains over unconstrained ``NUMERIC``, ``NUMERIC[]``/``NUMERIC[][]``,
        ``NUMRANGE``/``NUMRANGE[]``, ``NUMMULTIRANGE``). Fixed-scale
        ``NUMERIC(p, s)`` returns False."""
        if self.dialect != Dialect.POSTGRES or not expr or not expr.strip():
            return False
        c_def = self._lookup_col_def(expr)
        if c_def is not None:
            dt = self._normalize_dialect_type(c_def.data_type)
        else:
            dt = self._infer_expr_storage_type(
                SelectItem(raw_expr=expr, alias="__mv_g", is_aggregate=False)
            )
        return self._pg_type_has_variable_numeric_scale(dt)

    def _minmax_result_type(self, arg_type: str, arg_expr: Optional[str] = None) -> str:
        """MIN/MAX result type for an argument type.

        PostgreSQL's MIN/MAX are declared over unmodified base types, so the
        result drops the argument's type modifier and character types resolve
        to their aggregate's type while preserving collation: numeric(12,3) ->
        numeric, varchar(10) [COLLATE c] -> text [COLLATE c], char(5)
        [COLLATE c] -> bpchar [COLLATE c], name -> text COLLATE "C",
        timestamp(0) -> timestamp. MySQL/MariaDB and SQLite keep the argument
        type.
        """
        if (self.raw_view.has_cte or self.raw_view.has_subquery_in_from) and not (
            self.dialect == Dialect.MYSQL
            and (
                self._reads_recursive_cte(arg_expr)
                or self._reads_setop_cte_or_derived(arg_expr)
            )
        ):
            arg_type = self._mysql_derived_minmax_type(arg_type)
        if self.dialect != Dialect.POSTGRES:
            return arg_type
        s_raw = arg_type.strip()
        m_coll = self._PG_COLLATE_TAIL_RE.search(s_raw)
        if m_coll:
            type_part = s_raw[: m_coll.start()].strip()
            coll_tail = s_raw[m_coll.start() :].rstrip()
        else:
            type_part = s_raw
            coll_tail = ""
        explicit_coll = self._extract_pg_explicit_collation(arg_expr)
        if explicit_coll:
            coll_tail = explicit_coll
        elif not coll_tail and arg_expr:
            inferred_coll = self._infer_pg_expr_collation(arg_expr)
            if inferred_coll:
                coll_tail = inferred_coll
        m_arr = re.match(r"^(.*?)((?:\[\])+)$", type_part)
        if m_arr:
            elem_raw, brackets = m_arr.group(1).strip(), m_arr.group(2)
            if '"' in elem_raw:
                return f"{elem_raw}{brackets}{coll_tail}"
            elem_u = re.sub(r"\s+", " ", elem_raw.upper())
            elem_base = re.sub(r"\s*\([^)]*\)", "", elem_u).strip()
            # PostgreSQL MIN/MAX(anyarray) is polymorphic over the array type
            # and drops element typmods (e.g. VARCHAR(20)[] -> VARCHAR[],
            # CHAR(3)[] -> BPCHAR[], NUMERIC(10,2)[] -> NUMERIC[]).
            arr_map = {
                "NUMERIC": "NUMERIC",
                "DECIMAL": "NUMERIC",
                "DEC": "NUMERIC",
                "VARCHAR": "VARCHAR",
                "CHARACTER VARYING": "VARCHAR",
                "CHAR": "BPCHAR",
                "CHARACTER": "BPCHAR",
                "BPCHAR": "BPCHAR",
                "NCHAR": "BPCHAR",
                "TIMESTAMP": "TIMESTAMP",
                "TIMESTAMP WITHOUT TIME ZONE": "TIMESTAMP",
                "TIMESTAMPTZ": "TIMESTAMPTZ",
                "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
                "TIME": "TIME",
                "TIME WITHOUT TIME ZONE": "TIME",
                "TIMETZ": "TIMETZ",
                "TIME WITH TIME ZONE": "TIMETZ",
            }
            if elem_base in arr_map:
                res_arr = f"{arr_map[elem_base]}{brackets}"
                return f"{res_arr}{coll_tail}" if coll_tail and self._pg_type_supports_collation(res_arr) else res_arr
            if elem_base.startswith("INTERVAL"):
                return f"INTERVAL{brackets}"
            return f"{type_part}{coll_tail}" if coll_tail and self._pg_type_supports_collation(type_part) else type_part
        u = re.sub(r"\s+", " ", type_part.upper())
        base = re.sub(r"\s*\([^)]*\)", "", u).strip()
        if base in ("NUMERIC", "DECIMAL", "DEC"):
            return "NUMERIC"
        if base == "NAME":
            # PostgreSQL NAME has intrinsic collation "C", which MIN/MAX(name)
            # preserves on its TEXT result unless overridden by an explicit COLLATE.
            return f"TEXT{coll_tail}" if coll_tail else 'TEXT COLLATE "C"'
        if base in ("VARCHAR", "CHARACTER VARYING", "TEXT"):
            return f"TEXT{coll_tail}"
        if base in ("CHAR", "CHARACTER", "BPCHAR", "NCHAR"):
            return f"BPCHAR{coll_tail}"
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
        return f"{type_part}{coll_tail}" if coll_tail and self._pg_type_supports_collation(type_part) else arg_type

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

    def _canonical_group_pk_expr(self, group_exprs: Sequence[str], *, collation_aware: bool = True) -> str:
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
            return " || '|' || ".join(
                self._sqlite_canonical_pk_term(self._sqlite_collation_folded(e) if collation_aware else e)
                for e in cleaned_exprs
            )
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
            parts = []
            for e in cleaned_exprs:
                # Character key: values that compare equal under the expression's
                # collation must share one key text (see _mysql_text_key_part).
                text_part = self._mysql_text_key_part(e) if collation_aware else None
                if text_part is not None:
                    parts.append(text_part)
                else:
                    parts.append(
                        f"IF({e} IS NULL, 'N', CONCAT('V', OCTET_LENGTH(CAST({e} AS CHAR CHARACTER SET utf8mb4)), ':', CAST({e} AS CHAR CHARACTER SET utf8mb4)))"
                    )
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
                        out.append(
                            quote_sql_identifier(
                                tk.text,
                                self.dialect,
                                force=(self.dialect == Dialect.POSTGRES),
                                allow_qualified=False,
                            )
                        )
                        cursor = tk.end + 1
                        i += 1
                        continue
            i += 1
        if cursor < len(expr):
            out.append(expr[cursor:])
        return "".join(out)

    _STRICT_INT_BASE_TYPES: frozenset[str] = frozenset(
        {
            "TINYINT",
            "SMALLINT",
            "MEDIUMINT",
            "INT",
            "INTEGER",
            "BIGINT",
            "INT1",
            "INT2",
            "INT3",
            "INT4",
            "INT8",
            "SMALLSERIAL",
            "SERIAL",
            "BIGSERIAL",
            "SERIAL2",
            "SERIAL4",
            "SERIAL8",
        }
    )

    @classmethod
    def _is_strict_integer_type(cls, data_type: str) -> bool:
        """True only for exact integer column types (never INTERVAL, POINT, arrays, or ranges)."""
        dt = re.sub(r"\s+", " ", (data_type or "").strip().upper())
        if not dt:
            return False
        m = re.match(
            r"^([A-Z0-9_]+)(?:\s*\(\s*\d+\s*\))?(?:\s+(?:UNSIGNED|SIGNED|ZEROFILL))*$",
            dt,
        )
        if not m:
            return False
        return m.group(1) in cls._STRICT_INT_BASE_TYPES

    def _is_implicit_sqlite_rowid(self, col_name: Optional[str]) -> bool:
        if self.dialect != Dialect.SQLITE or not col_name:
            return False
        low = col_name.lower()
        if low not in ("rowid", "_rowid_", "oid"):
            return False
        ts = self.primary_schema
        return ts is None or all(c.lower() != low for c in ts.columns)

    def _quote_pk_range_col(self, col_name: str) -> str:
        if self._is_implicit_sqlite_rowid(col_name):
            return col_name.lower()
        return self._q(col_name)

    def _resolve_pk_range_column(
        self,
        *,
        allow_sqlite_rowid: bool = True,
        require_non_composite_for_rowid: bool = False,
    ) -> Optional[str]:
        """Return the single non-null exact-integer column suitable for pk_range_fold
        backfill and watermark trigger gating, or None when pk_range_fold is unsafe."""
        ts = self.primary_schema
        if ts is None:
            return "rowid" if (allow_sqlite_rowid and self.dialect == Dialect.SQLITE) else None
        pk_defs = [c for c in ts.columns.values() if c.primary_key]
        if len(pk_defs) == 1:
            pk_c = pk_defs[0]
            if self.dialect == Dialect.SQLITE:
                # On SQLite, non-STRICT INT/BIGINT/WITHOUT ROWID PKs only have
                # integer affinity and accept REAL (2.5) or TEXT ('abc'), which
                # would truncate or raise ValueError during pk_range_fold.
                # Only a true INTEGER PRIMARY KEY rowid alias, or a NOT NULL
                # integer PK on a STRICT table, is guaranteed 64-bit integer.
                if pk_c.is_rowid_alias or (
                    ts.strict and pk_c.not_null and self._is_strict_integer_type(pk_c.data_type)
                ):
                    return pk_c.name
            else:
                # PostgreSQL and MySQL/MariaDB enforce NOT NULL and strict
                # integer typing on single-column integer PRIMARY KEY columns.
                if self._is_strict_integer_type(pk_c.data_type):
                    return pk_c.name
        if (
            allow_sqlite_rowid
            and self.dialect == Dialect.SQLITE
            and not ts.without_rowid
            and (not require_non_composite_for_rowid or len(pk_defs) <= 1)
        ):
            user_cols_lower = {c.lower() for c in ts.columns}
            for alias in ("rowid", "_rowid_", "oid"):
                if alias not in user_cols_lower:
                    return alias
        return None

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

