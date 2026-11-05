#!/usr/bin/env python3
"""
matview_sync.py — Zero-Runtime-Dependency Incremental Materialized View Compiler & Manager
==========================================================================================

Compiles standard SQL `CREATE VIEW` definitions (including multi-view `.sql` bundles and
user-defined `CREATE [UNIQUE] INDEX` statements on views) into native database objects:
  1. Backing storage tables (`__matview_store_<view>`) & reference-counting auxiliary tables
  2. Automatic group/lookup indexes + user-defined view indexes mapped onto backing tables
  3. Read-only user-facing views (`__matview_<view>`) with write-protection triggers /
     `ALGORITHM = TEMPTABLE` enforcement
  4. Atomic `AFTER INSERT`, `AFTER UPDATE`, `AFTER DELETE`, and `AFTER TRUNCATE` triggers
     (installed BEFORE initial backfill so concurrent base-table writes are never missed)
  5. Full single-shot or chunked incremental backfill statements (`--incremental-backfill`)

Supported Dialects:
  - SQLite 3 (`sqlite`, plus APSW, SQLite3 Multiple Ciphers, SQLCipher, sqlean, Turso libSQL)
  - PostgreSQL (`postgres`, plus Percona PG, TimescaleDB, PostGIS, pgvector, Row & Transition Tables)
  - MySQL / MariaDB / Percona Server (`mysql`)
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import sqlite3
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

DEFAULT_OBJECT_PREFIX = "__matview_"


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"


class MaintenanceStrategy(str, Enum):
    ALGEBRAIC_DELTA = "algebraic_delta"
    GROUP_SCOPED_DELTA = "group_scoped_delta"
    ATOMIC_QUERY_SYNC = "atomic_query_sync"


class AggFunc(str, Enum):
    COUNT_STAR = "COUNT(*)"
    COUNT_COL = "COUNT"
    SUM = "SUM"
    TOTAL = "TOTAL"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    GROUP_CONCAT = "GROUP_CONCAT"
    STRING_AGG = "STRING_AGG"
    JSON_GROUP_ARRAY = "JSON_GROUP_ARRAY"
    JSON_GROUP_OBJECT = "JSON_GROUP_OBJECT"
    ARRAY_AGG = "ARRAY_AGG"
    VAR_POP = "VAR_POP"
    VAR_SAMP = "VAR_SAMP"
    STDDEV_POP = "STDDEV_POP"
    STDDEV_SAMP = "STDDEV_SAMP"
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BOOL_AND = "BOOL_AND"
    BOOL_OR = "BOOL_OR"


class LimitationSeverity(str, Enum):
    WARNING = "warning"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class TransparencyLimitation:
    code: str
    severity: LimitationSeverity
    dialect: Dialect
    affected_object: str
    description: str
    mitigation: str


class EngineLimitationError(RuntimeError):
    """Raised when a materialized view encounters engine transparency limitations and ignore_limitations=False."""

    def __init__(self, view_name: str, dialect: Dialect, limitations: Sequence[TransparencyLimitation]) -> None:
        self.view_name = view_name
        self.dialect = dialect
        self.limitations = list(limitations)
        bullet_lines = "\n".join(
            f"  - [{lim.code}] {lim.description} Mitigation: {lim.mitigation}"
            for lim in self.limitations
        )
        msg = (
            f"Refusing to create materialized view '{view_name}' on engine '{dialect.value}' "
            f"because full transparency cannot be guaranteed for all cases:\n{bullet_lines}\n"
            f"Pass `--ignore-limitations` (CLI) or `ignore_limitations=True` (API) to override "
            f"this check and proceed anyway."
        )
        super().__init__(msg)


class EngineLimitationWarning(UserWarning):
    """Emitted when proceeding with ignore_limitations=True despite engine transparency limitations."""


@dataclass(frozen=True)
class ColumnDef:
    name: str
    data_type: str
    not_null: bool = False
    primary_key: bool = False


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Dict[str, ColumnDef]
    fk_cascade_actions: Tuple[str, ...] = ()
    has_on_conflict_replace: bool = False


@dataclass(frozen=True)
class SelectItem:
    raw_expr: str
    alias: str
    is_aggregate: bool
    agg_func: Optional[AggFunc] = None
    arg_expr: Optional[str] = None
    extra_args: Optional[str] = None
    is_distinct: bool = False
    filter_where: Optional[str] = None
    source_col: Optional[str] = None


@dataclass(frozen=True)
class JoinSpec:
    join_type: str  # "FROM", "INNER JOIN", "LEFT JOIN", "CROSS JOIN"
    table_name: str
    alias: str
    on_condition: Optional[str] = None


@dataclass(frozen=True)
class ViewIndexSpec:
    columns: Tuple[str, ...]
    name: Optional[str] = None
    unique: bool = False
    where_clause: Optional[str] = None
    using_method: Optional[str] = None
    target_view: Optional[str] = None


@dataclass(frozen=True)
class ViewSpec:
    view_name: str
    raw_select_sql: str
    base_table: str
    joins: Tuple[JoinSpec, ...]
    referenced_tables: Tuple[str, ...]
    select_items: Tuple[SelectItem, ...]
    where_clause: Optional[str] = None
    group_by_cols: Tuple[str, ...] = ()
    having_clause: Optional[str] = None
    has_cte: bool = False
    has_window_func: bool = False
    has_compound_set_op: bool = False
    has_top_level_distinct: bool = False
    has_limit_or_order: bool = False
    has_subquery_in_from: bool = False
    user_indexes: Tuple[ViewIndexSpec, ...] = ()
    additional_views: Tuple["ViewSpec", ...] = ()


@dataclass(frozen=True)
class IncrementalBackfillSpec:
    mode: str  # "pk_range_fold", "group_key_batch", "offset_batch"
    pk_column: Optional[str] = None
    chunk_sql_templates: List[str] = field(default_factory=list)
    distinct_group_query: Optional[str] = None
    group_insert_template: Optional[str] = None
    offset_insert_template: Optional[str] = None


@dataclass(frozen=True)
class MaterializedViewPlan:
    dialect: Dialect
    strategy: MaintenanceStrategy
    view_spec: ViewSpec
    backing_table_name: str
    backing_table_ddl: str
    auxiliary_table_ddls: List[str]
    index_ddls: List[str]
    initial_populate_sqls: List[str]
    trigger_ddls: List[str]
    readonly_trigger_ddls: List[str] = field(default_factory=list)
    user_columns: List[str] = field(default_factory=list)
    reader_view_ddl: Optional[str] = None
    incremental_backfill_spec: Optional[IncrementalBackfillSpec] = None
    auxiliary_table_names: List[str] = field(default_factory=list)
    trigger_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    limitations: List[TransparencyLimitation] = field(default_factory=list)
    object_prefix: str = DEFAULT_OBJECT_PREFIX
    db_view_name: str = ""
    index_names: List[str] = field(default_factory=list)
    user_indexes: List[ViewIndexSpec] = field(default_factory=list)
    sub_plans: List["MaterializedViewPlan"] = field(default_factory=list)
    incremental_backfill: bool = False
    batch_size: int = 1000
    full_populate_sqls: List[str] = field(default_factory=list)
    incremental_populate_sqls: List[str] = field(default_factory=list)

    @property
    def view_name(self) -> str:
        if self.db_view_name:
            return self.db_view_name
        raw = self.view_spec.view_name
        if self.object_prefix and not raw.startswith(self.object_prefix):
            return f"{self.object_prefix}{raw}"
        return raw

    @property
    def logical_view_name(self) -> str:
        raw = self.view_spec.view_name
        if self.object_prefix and raw.startswith(self.object_prefix):
            return raw[len(self.object_prefix) :]
        return raw

    @property
    def initial_populate_sql(self) -> str:
        return "\n\n".join(self.initial_populate_sqls)

    def get_populate_statements(
        self,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        use_inc = self.incremental_backfill if incremental_backfill is None else incremental_backfill
        eff_batch = self.batch_size if batch_size is None else batch_size
        if use_inc and self.incremental_populate_sqls:
            if eff_batch != self.batch_size and self.batch_size > 0:
                old_b = str(self.batch_size)
                new_b = str(eff_batch)
                return [
                    re.sub(rf"\b{re.escape(old_b)}\b", new_b, s)
                    for s in self.incremental_populate_sqls
                ]
            return list(self.incremental_populate_sqls)
        if not use_inc and self.full_populate_sqls:
            return list(self.full_populate_sqls)
        return list(self.initial_populate_sqls)

    def create_structure_statements(self) -> List[str]:
        """Returns DDLs to create auxiliary tables, backing table, indexes, and read-only view."""
        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(sp.create_structure_statements())
            return stmts
        stmts = []
        stmts.extend(self.auxiliary_table_ddls)
        stmts.append(self.backing_table_ddl)
        stmts.extend(self.index_ddls)
        if self.reader_view_ddl:
            stmts.append(self.reader_view_ddl)
        stmts.extend(self.readonly_trigger_ddls)
        return stmts

    def all_statements(
        self,
        include_populate: bool = True,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Returns all setup statements in concurrency-safe order:
          1. Backing & auxiliary tables, indexes, read-only user view & write-protection triggers
          2. Atomic maintenance triggers on base tables (installed BEFORE backfill so no concurrent
             base-table DML can be missed while backfill is running)
          3. Initial backfill (single-shot or chunked incremental backfill)
        """
        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(
                    sp.all_statements(
                        include_populate=include_populate,
                        incremental_backfill=incremental_backfill,
                        batch_size=batch_size,
                    )
                )
            return stmts
        stmts = self.create_structure_statements()
        stmts.extend(self.trigger_ddls)
        if include_populate:
            stmts.extend(
                self.get_populate_statements(
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                )
            )
        return stmts

    def drop_statements(self) -> List[str]:
        """Returns idempotent DROP statements for triggers, functions, view, and backing tables."""
        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(sp.drop_statements())
            return stmts
        stmts = []
        if self.dialect != Dialect.POSTGRES:
            for t_name in self.trigger_names:
                stmts.append(f"DROP TRIGGER IF EXISTS {t_name};")
        for fn_name in self.function_names:
            stmts.append(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
        stmts.append(f"DROP VIEW IF EXISTS {self.view_name};")
        stmts.append(f"DROP TABLE IF EXISTS {self.backing_table_name};")
        for aux_t in self.auxiliary_table_names:
            stmts.append(f"DROP TABLE IF EXISTS {aux_t};")
        return stmts

    def render_sql_script(
        self,
        include_populate: bool = True,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> str:
        if self.sub_plans:
            return "\n".join(
                sp.render_sql_script(
                    include_populate=include_populate,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                )
                for sp in self.sub_plans
            )
        use_inc = self.incremental_backfill if incremental_backfill is None else incremental_backfill
        eff_batch = self.batch_size if batch_size is None else batch_size
        pop_stmts = self.get_populate_statements(
            incremental_backfill=use_inc,
            batch_size=eff_batch,
        )
        bf_label = f"incremental (batch_size={eff_batch})" if use_inc else "full"
        lines: List[str] = [
            "-- =============================================================================",
            f"-- Incremental Materialized View: {self.view_name} (logical: {self.logical_view_name})",
            f"-- Backing Table: {self.backing_table_name}",
            f"-- Base Tables: {', '.join(self.view_spec.referenced_tables)} | Dialect: {self.dialect.value} | Strategy: {self.strategy.value} | Backfill: {bf_label}",
            "-- =============================================================================",
        ]
        if self.limitations:
            lines.append("-- WARNING (Ignored Engine/View Limitations):")
            for lim in self.limitations:
                lines.append(f"--   * [{lim.code}] {lim.description}")
        lines.append("")
        lines.append("-- 1. Internal Backing Table")
        for aux in self.auxiliary_table_ddls:
            lines.append(aux)
        lines.append(self.backing_table_ddl)
        lines.append("")
        if self.index_ddls:
            lines.append("-- 2. Indexes")
            for idx_ddl in self.index_ddls:
                lines.append(idx_ddl)
            lines.append("")
        if self.reader_view_ddl or self.readonly_trigger_ddls:
            lines.append("-- 3. Read-Only User View & Write-Protection Triggers")
            if self.reader_view_ddl:
                lines.append(self.reader_view_ddl)
            for rt in self.readonly_trigger_ddls:
                lines.append(rt)
            lines.append("")
        lines.append("-- 4. Atomic Maintenance Triggers (Installed Before Backfill)")
        for trig in self.trigger_ddls:
            lines.append(trig)
            lines.append("")
        if include_populate and pop_stmts:
            lines.append(
                f"-- 5. Initial Backfill ({'Incremental Batched' if use_inc else 'Full'})"
            )
            for pop in pop_stmts:
                lines.append(pop)
            lines.append("")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# SQL Lexing, Splitting, Schema Parsing & View Definition Parsing
# -----------------------------------------------------------------------------


def _strip_sql_comments(sql: str) -> str:
    """Strips `-- ...` line comments and `/* ... */` block comments outside string literals."""
    out: List[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    while i < n:
        ch = sql[i]
        if ch == "'" and not in_double:
            out.append(ch)
            if in_single and i + 1 < n and sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            in_single = not in_single
            i += 1
        elif ch == '"' and not in_single:
            out.append(ch)
            in_double = not in_double
            i += 1
        elif not in_single and not in_double and sql[i : i + 2] == "--":
            nl = sql.find("\n", i + 2)
            if nl == -1:
                break
            out.append("\n")
            i = nl + 1
        elif not in_single and not in_double and sql[i : i + 2] == "/*":
            end = sql.find("*/", i + 2)
            if end == -1:
                break
            out.append(" ")
            i = end + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _unquote_ident(ident: str) -> str:
    s = ident.strip()
    if "." in s:
        s = s.split(".")[-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("`") and s.endswith("`")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        return s[1:-1]
    return s


def _sanitize_ident(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw.strip())
    return cleaned.strip("_") or "col"


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    n = len(text)
    sep_len = len(sep)
    while i < n:
        ch = text[i]
        if ch == "'" and not in_double:
            buf.append(ch)
            if in_single and i + 1 < n and text[i + 1] == "'":
                buf.append("'")
                i += 2
                continue
            in_single = not in_single
            i += 1
        elif ch == '"' and not in_single:
            buf.append(ch)
            in_double = not in_double
            i += 1
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
                buf.append(ch)
                i += 1
            elif ch == ")":
                depth = max(0, depth - 1)
                buf.append(ch)
                i += 1
            elif depth == 0 and text[i : i + sep_len] == sep:
                parts.append("".join(buf))
                buf = []
                i += sep_len
            else:
                buf.append(ch)
                i += 1
        else:
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def _find_top_level_keyword(sql: str, keyword_regex: str) -> Optional[re.Match[str]]:
    depth = 0
    in_single = False
    in_double = False
    mask = list(sql)
    for i, ch in enumerate(sql):
        if ch == "'" and not in_double:
            in_single = not in_single
            mask[i] = " "
        elif ch == '"' and not in_single:
            in_double = not in_double
            mask[i] = " "
        elif in_single or in_double:
            mask[i] = " "
        else:
            if ch == "(":
                depth += 1
                mask[i] = " "
            elif ch == ")":
                depth = max(0, depth - 1)
                mask[i] = " "
            elif depth > 0:
                mask[i] = " "
    masked_str = "".join(mask)
    return re.search(keyword_regex, masked_str, flags=re.IGNORECASE | re.DOTALL)


def _find_all_top_level_keywords(sql: str, keyword_regex: str) -> List[re.Match[str]]:
    depth = 0
    in_single = False
    in_double = False
    mask = list(sql)
    for i, ch in enumerate(sql):
        if ch == "'" and not in_double:
            in_single = not in_single
            mask[i] = " "
        elif ch == '"' and not in_single:
            in_double = not in_double
            mask[i] = " "
        elif in_single or in_double:
            mask[i] = " "
        else:
            if ch == "(":
                depth += 1
                mask[i] = " "
            elif ch == ")":
                depth = max(0, depth - 1)
                mask[i] = " "
            elif depth > 0:
                mask[i] = " "
    masked_str = "".join(mask)
    return list(re.finditer(keyword_regex, masked_str, flags=re.IGNORECASE | re.DOTALL))


def parse_table_schema(schema_sql: str) -> Dict[str, TableSchema]:
    """Parses one or more CREATE TABLE statements into TableSchema objects."""
    clean = _strip_sql_comments(schema_sql)
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"`\[\]]+)\s*\((.*)\)\s*;?$",
        re.IGNORECASE | re.DOTALL,
    )
    schemas: Dict[str, TableSchema] = {}
    for stmt in _split_top_level(clean, ";"):
        m = pattern.match(stmt.strip())
        if not m:
            continue
        table_name = _unquote_ident(m.group(1))
        if table_name.startswith(("__matview_", "__mv_")):
            continue
        body = m.group(2)
        items = _split_top_level(body, ",")
        cols: Dict[str, ColumnDef] = {}
        table_pk_cols: Set[str] = set()
        fk_cascades: List[str] = []

        for m_fk in re.finditer(
            r"\bON\s+(DELETE|UPDATE)\s+(CASCADE|SET\s+NULL|SET\s+DEFAULT)\b",
            body,
            flags=re.IGNORECASE,
        ):
            action = f"ON {m_fk.group(1).upper()} {' '.join(m_fk.group(2).upper().split())}"
            if action not in fk_cascades:
                fk_cascades.append(action)

        has_on_conflict_replace = bool(
            re.search(r"\bON\s+CONFLICT\s+REPLACE\b", body, flags=re.IGNORECASE)
        )

        for item in items:
            upper_item = item.upper().strip()
            pk_match = re.match(
                r"^(?:CONSTRAINT\s+\S+\s+)?PRIMARY\s+KEY\s*\(([^)]+)\)", upper_item
            )
            if pk_match:
                for pk_col in pk_match.group(1).split(","):
                    table_pk_cols.add(_unquote_ident(pk_col).lower())
                continue
            if re.match(
                r"^(?:CONSTRAINT|UNIQUE|FOREIGN\s+KEY|CHECK|INDEX|KEY)\b", upper_item
            ):
                continue

            tokens = item.strip().split()
            if not tokens:
                continue
            col_name = _unquote_ident(tokens[0])
            rest = item.strip()[len(tokens[0]) :].strip()
            type_match = re.match(
                r"^((?:DOUBLE\s+PRECISION|CHARACTER\s+VARYING(?:\s*\([^)]+\))?|TIMESTAMP(?:\s+(?:WITH|WITHOUT)\s+TIME\s+ZONE)?|TIME(?:\s+(?:WITH|WITHOUT)\s+TIME\s+ZONE)?|[A-Za-z0-9_]+(?:\s*\([^)]+\))?)(?:\s+UNSIGNED)?)",
                rest,
                re.IGNORECASE,
            )
            data_type = type_match.group(1).upper() if type_match else "TEXT"
            upper_rest = rest.upper()
            is_pk = "PRIMARY KEY" in upper_rest
            not_null = ("NOT NULL" in upper_rest) or is_pk
            cols[col_name] = ColumnDef(
                name=col_name,
                data_type=data_type,
                not_null=not_null,
                primary_key=is_pk,
            )

        if table_pk_cols:
            for c_name, c_def in list(cols.items()):
                if c_name.lower() in table_pk_cols:
                    cols[c_name] = ColumnDef(
                        name=c_def.name,
                        data_type=c_def.data_type,
                        not_null=True,
                        primary_key=True,
                    )

        schemas[table_name] = TableSchema(
            name=table_name,
            columns=cols,
            fk_cascade_actions=tuple(fk_cascades),
            has_on_conflict_replace=has_on_conflict_replace,
        )
    return schemas


def introspect_sqlite_schemas(
    conn: sqlite3.Connection,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> Dict[str, TableSchema]:
    """Introspects all non-internal base tables from a live SQLite connection."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    rows = cur.fetchall()
    ddl_parts: List[str] = []
    for name, sql in rows:
        if not sql:
            continue
        if name.startswith(("__matview_", "__mv_")) or (
            object_prefix and name.startswith(object_prefix)
        ):
            continue
        ddl_parts.append(sql.rstrip(";") + ";")
    return parse_table_schema("\n".join(ddl_parts))


_SQL_EXPR_FUNCTIONS = {
    "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX",
    "GROUP_CONCAT", "STRING_AGG", "JSON_GROUP_ARRAY", "JSON_GROUP_OBJECT", "ARRAY_AGG",
    "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
    "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
    "COALESCE", "IFNULL", "NULLIF", "IIF", "IF",
    "LOWER", "UPPER", "TRIM", "LTRIM", "RTRIM", "LENGTH", "CHAR_LENGTH",
    "SUBSTR", "SUBSTRING", "REPLACE", "INSTR", "PRINTF", "FORMAT", "CONCAT", "CONCAT_WS",
    "ABS", "ROUND", "CEIL", "CEILING", "FLOOR", "MOD", "POWER", "SQRT", "SIGN",
    "CAST", "TYPEOF", "QUOTE", "HEX", "ZEROBLOB",
    "DATE", "TIME", "DATETIME", "JULIANDAY", "STRFTIME", "UNIXEPOCH",
    "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD",
    "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "PERCENT_RANK", "CUME_DIST",
    "JSON", "JSON_ARRAY", "JSON_OBJECT", "JSON_EXTRACT", "JSON_TYPE", "JSON_VALID",
}


def _parse_single_agg_expr(expr: str, alias: str) -> Optional[SelectItem]:
    """Attempts to parse a pure single aggregate function call (with optional FILTER)."""
    s = expr.strip()
    while s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        depth = 0
        in_str = False
        balanced = True
        for ch in inner:
            if ch == "'":
                in_str = not in_str
            elif not in_str:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        balanced = False
                        break
        if balanced and depth == 0:
            s = inner
        else:
            break

    if re.search(r"\bOVER\s*\(", s, flags=re.IGNORECASE):
        return None

    m = re.match(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)(?:\s*FILTER\s*\(\s*WHERE\s+(.*)\))?$",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    fn_raw = m.group(1).upper()
    inside = m.group(2).strip()
    filter_where = m.group(3).strip() if m.group(3) else None

    # Ensure the closing paren of fn(...) was not prematurely matched across a compound expression like SUM(a) + SUM(b)
    depth = 0
    in_single = False
    in_double = False
    for ch in inside:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return None
    if depth != 0:
        return None

    fn_map: Dict[str, AggFunc] = {
        "SUM": AggFunc.SUM,
        "TOTAL": AggFunc.TOTAL,
        "AVG": AggFunc.AVG,
        "MIN": AggFunc.MIN,
        "MAX": AggFunc.MAX,
        "GROUP_CONCAT": AggFunc.GROUP_CONCAT,
        "STRING_AGG": AggFunc.STRING_AGG,
        "JSON_GROUP_ARRAY": AggFunc.JSON_GROUP_ARRAY,
        "JSON_GROUP_OBJECT": AggFunc.JSON_GROUP_OBJECT,
        "ARRAY_AGG": AggFunc.ARRAY_AGG,
        "VAR_POP": AggFunc.VAR_POP,
        "VARIANCE": AggFunc.VAR_POP,
        "VAR_SAMP": AggFunc.VAR_SAMP,
        "STDDEV_POP": AggFunc.STDDEV_POP,
        "STDDEV": AggFunc.STDDEV_POP,
        "STD": AggFunc.STDDEV_POP,
        "STDDEV_SAMP": AggFunc.STDDEV_SAMP,
        "BIT_AND": AggFunc.BIT_AND,
        "BIT_OR": AggFunc.BIT_OR,
        "BIT_XOR": AggFunc.BIT_XOR,
        "BOOL_AND": AggFunc.BOOL_AND,
        "EVERY": AggFunc.BOOL_AND,
        "BOOL_OR": AggFunc.BOOL_OR,
    }

    if fn_raw == "COUNT":
        if inside == "*" or inside == "":
            return SelectItem(
                raw_expr=expr.strip(),
                alias=alias,
                is_aggregate=True,
                agg_func=AggFunc.COUNT_STAR,
                arg_expr="*",
                is_distinct=False,
                filter_where=filter_where,
            )
        is_dist = False
        arg_body = inside
        m_dist = re.match(r"^DISTINCT\s+(.*)$", inside, flags=re.IGNORECASE | re.DOTALL)
        if m_dist:
            is_dist = True
            arg_body = m_dist.group(1).strip()
        elif re.match(r"^ALL\s+(.*)$", inside, flags=re.IGNORECASE | re.DOTALL):
            arg_body = re.sub(r"^ALL\s+", "", inside, flags=re.IGNORECASE).strip()
        return SelectItem(
            raw_expr=expr.strip(),
            alias=alias,
            is_aggregate=True,
            agg_func=AggFunc.COUNT_COL,
            arg_expr=arg_body,
            is_distinct=is_dist,
            filter_where=filter_where,
            source_col=_unquote_ident(arg_body) if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", arg_body) else None,
        )

    if fn_raw not in fn_map:
        return None

    agg_fn = fn_map[fn_raw]
    is_dist = False
    arg_body = inside
    m_dist = re.match(r"^DISTINCT\s+(.*)$", inside, flags=re.IGNORECASE | re.DOTALL)
    if m_dist:
        is_dist = True
        arg_body = m_dist.group(1).strip()
    elif re.match(r"^ALL\s+(.*)$", inside, flags=re.IGNORECASE | re.DOTALL):
        arg_body = re.sub(r"^ALL\s+", "", inside, flags=re.IGNORECASE).strip()

    top_args = _split_top_level(arg_body, ",")
    first_arg = top_args[0].strip() if top_args else arg_body
    extra_args = ", ".join(a.strip() for a in top_args[1:]) if len(top_args) > 1 else None

    return SelectItem(
        raw_expr=expr.strip(),
        alias=alias,
        is_aggregate=True,
        agg_func=agg_fn,
        arg_expr=first_arg,
        extra_args=extra_args,
        is_distinct=is_dist,
        filter_where=filter_where,
        source_col=_unquote_ident(first_arg) if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", first_arg) else None,
    )


_AGG_FN_PATTERN = re.compile(
    r"\b(COUNT|SUM|TOTAL|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG|JSON_GROUP_ARRAY|JSON_GROUP_OBJECT|ARRAY_AGG|VAR_POP|VARIANCE|VAR_SAMP|STDDEV_POP|STDDEV|STD|STDDEV_SAMP|BIT_AND|BIT_OR|BIT_XOR|BOOL_AND|EVERY|BOOL_OR)\s*\(",
    re.IGNORECASE,
)


def _contains_aggregate_call(expr: str) -> bool:
    if re.search(r"\bOVER\s*\(", expr, flags=re.IGNORECASE):
        return False
    no_str = re.sub(r"'(?:''|[^'])*'", "''", expr)
    return bool(_AGG_FN_PATTERN.search(no_str))


def _parse_select_item(raw_item: str, idx: int) -> SelectItem:
    s = raw_item.strip()
    alias: Optional[str] = None
    expr = s

    # Match top-level `AS alias` or trailing `alias`
    depth = 0
    in_single = False
    in_double = False
    masked = list(s)
    for i, ch in enumerate(s):
        if ch == "'" and not in_double:
            in_single = not in_single
            masked[i] = " "
        elif ch == '"' and not in_single:
            in_double = not in_double
            masked[i] = " "
        elif in_single or in_double:
            masked[i] = " "
        else:
            if ch == "(":
                depth += 1
                masked[i] = " "
            elif ch == ")":
                depth = max(0, depth - 1)
                masked[i] = " "
            elif depth > 0:
                masked[i] = " "
    masked_s = "".join(masked)

    m_as = list(re.finditer(r"\bAS\s+([A-Za-z0-9_\"`\[\]]+)\s*$", masked_s, flags=re.IGNORECASE))
    if m_as:
        last_m = m_as[-1]
        expr = s[: last_m.start()].strip()
        alias_raw = s[last_m.start(1) : last_m.end(1)].strip()
        alias = _unquote_ident(alias_raw)
    else:
        # Check for implicit alias: `<expr> <ident>` at top level
        m_imp = re.search(r"\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\"|`[^`]+`)\s*$", masked_s)
        if m_imp:
            cand = s[m_imp.start(1) : m_imp.end(1)].strip()
            prefix_s = s[: m_imp.start()].strip()
            if prefix_s and cand.upper() not in {
                "ASC", "DESC", "NULLS", "FIRST", "LAST", "END", "ELSE", "THEN", "AND", "OR", "NOT", "NULL", "TRUE", "FALSE"
            }:
                if not re.search(r"(?:[+\-*/%|&^=<>!]|\bIS|\bNOT|\bIN|\bLIKE|\bBETWEEN)\s*$", prefix_s, flags=re.IGNORECASE):
                    expr = prefix_s
                    alias = _unquote_ident(cand)

    if not alias:
        if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", expr):
            alias = _unquote_ident(expr)
        else:
            alias = f"col_{idx}"

    parsed_agg = _parse_single_agg_expr(expr, alias)
    if parsed_agg is not None:
        return parsed_agg

    if _contains_aggregate_call(expr):
        # Compound aggregate expression (e.g. COALESCE(SUM(a), 0) - COALESCE(SUM(b), 0))
        return SelectItem(
            raw_expr=expr,
            alias=alias,
            is_aggregate=True,
            agg_func=None,
            arg_expr=expr,
        )

    src_col = _unquote_ident(expr) if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", expr) else None
    return SelectItem(
        raw_expr=expr,
        alias=alias,
        is_aggregate=False,
        source_col=src_col,
    )


def _extract_cte_names_and_strip_with(select_sql: str) -> Tuple[Set[str], str, bool]:
    s = select_sql.strip()
    if not re.match(r"^WITH\b", s, flags=re.IGNORECASE):
        return set(), s, False

    rest = re.sub(r"^WITH(?:\s+RECURSIVE)?\s+", "", s, flags=re.IGNORECASE)
    cte_names: Set[str] = set()
    pos = 0
    n = len(rest)
    while pos < n:
        m_head = re.match(
            r"^\s*([A-Za-z0-9_\"`\[\]]+)(?:\s*\([^)]*\))?\s+AS\s*(?:(?:MATERIALIZED|NOT\s+MATERIALIZED)\s*)?\(",
            rest[pos:],
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m_head:
            break
        cte_names.add(_unquote_ident(m_head.group(1)))
        open_paren = pos + m_head.end() - 1
        depth = 1
        i = open_paren + 1
        in_single = False
        in_double = False
        while i < n and depth > 0:
            ch = rest[i]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            i += 1
        pos = i
        while pos < n and rest[pos].isspace():
            pos += 1
        if pos < n and rest[pos] == ",":
            pos += 1
            continue
        break

    main_select = rest[pos:].strip()
    return cte_names, (main_select if main_select else s), True


def _parse_from_and_joins(from_clause: str, cte_names: Set[str]) -> Tuple[List[JoinSpec], bool]:
    s = from_clause.strip()
    if not s:
        return [], False

    has_subquery = s.startswith("(")
    # Tokenize top-level JOIN keywords
    join_kw_re = r"\b(?:(?:INNER|LEFT(?:\s+OUTER)?|RIGHT(?:\s+OUTER)?|FULL(?:\s+OUTER)?|CROSS|NATURAL)\s+)?JOIN\b|,"
    matches = _find_all_top_level_keywords(s, join_kw_re)

    if not matches:
        # Single FROM target
        if s.startswith("("):
            m_sub = re.match(r"^\((.*)\)(?:\s+(?:AS\s+)?([A-Za-z0-9_\"`\[\]]+))?$", s, flags=re.IGNORECASE | re.DOTALL)
            alias = _unquote_ident(m_sub.group(2)) if (m_sub and m_sub.group(2)) else "__mv_sub"
            return [JoinSpec(join_type="FROM", table_name=alias, alias=alias)], True
        tokens = s.split()
        t_name = _unquote_ident(tokens[0])
        alias = t_name
        if len(tokens) >= 2:
            if tokens[1].upper() == "AS" and len(tokens) >= 3:
                alias = _unquote_ident(tokens[2])
            elif tokens[1].upper() not in {"ON", "USING", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"}:
                alias = _unquote_ident(tokens[1])
        return [JoinSpec(join_type="FROM", table_name=t_name, alias=alias)], False

    joins: List[JoinSpec] = []
    first_seg = s[: matches[0].start()].strip()
    if first_seg.startswith("("):
        has_subquery = True
        m_sub = re.match(r"^\((.*)\)(?:\s+(?:AS\s+)?([A-Za-z0-9_\"`\[\]]+))?$", first_seg, flags=re.IGNORECASE | re.DOTALL)
        alias = _unquote_ident(m_sub.group(2)) if (m_sub and m_sub.group(2)) else "__mv_sub"
        joins.append(JoinSpec(join_type="FROM", table_name=alias, alias=alias))
    else:
        toks = first_seg.split()
        t_name = _unquote_ident(toks[0])
        alias = t_name
        if len(toks) >= 2:
            if toks[1].upper() == "AS" and len(toks) >= 3:
                alias = _unquote_ident(toks[2])
            else:
                alias = _unquote_ident(toks[1])
        joins.append(JoinSpec(join_type="FROM", table_name=t_name, alias=alias))

    for idx, m in enumerate(matches):
        kw_raw = m.group(0).strip().upper()
        if kw_raw == ",":
            j_type = "CROSS JOIN"
        elif "LEFT" in kw_raw:
            j_type = "LEFT JOIN"
        elif "RIGHT" in kw_raw:
            j_type = "RIGHT JOIN"
        elif "FULL" in kw_raw:
            j_type = "FULL JOIN"
        elif "CROSS" in kw_raw:
            j_type = "CROSS JOIN"
        elif "NATURAL" in kw_raw:
            j_type = "NATURAL JOIN"
        else:
            j_type = "INNER JOIN"

        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(s)
        seg = s[m.end() : seg_end].strip()
        if seg.startswith("("):
            has_subquery = True

        m_on = _find_top_level_keyword(seg, r"\bON\b")
        on_cond: Optional[str] = None
        if m_on:
            target_part = seg[: m_on.start()].strip()
            on_cond = seg[m_on.end() :].strip()
        else:
            m_using = _find_top_level_keyword(seg, r"\bUSING\b")
            if m_using:
                target_part = seg[: m_using.start()].strip()
                has_subquery = True
            else:
                target_part = seg

        if target_part.startswith("("):
            has_subquery = True
            t_name = f"__mv_sub_{idx}"
            alias = t_name
        else:
            toks = target_part.split()
            t_name = _unquote_ident(toks[0]) if toks else f"t_{idx}"
            alias = t_name
            if len(toks) >= 2:
                if toks[1].upper() == "AS" and len(toks) >= 3:
                    alias = _unquote_ident(toks[2])
                else:
                    alias = _unquote_ident(toks[1])

        joins.append(
            JoinSpec(
                join_type=j_type,
                table_name=t_name,
                alias=alias,
                on_condition=on_cond,
            )
        )

    return joins, has_subquery


def _extract_all_referenced_tables(
    select_sql: str,
    cte_names: Set[str],
    known_tables: Optional[Set[str]] = None,
) -> List[str]:
    no_str = re.sub(r"'(?:''|[^'])*'", "''", select_sql)
    cte_lower = {c.lower() for c in cte_names}
    found: List[str] = []
    seen_lower: Set[str] = set()
    sql_reserved = {
        "select", "from", "where", "group", "by", "having", "order", "limit", "offset",
        "union", "all", "intersect", "except", "join", "inner", "left", "right", "full",
        "outer", "cross", "natural", "on", "using", "as", "distinct", "case", "when",
        "then", "else", "end", "and", "or", "not", "in", "exists", "is", "null", "true",
        "false", "dual", "lateral", "values",
    }

    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])",
        no_str,
        flags=re.IGNORECASE,
    ):
        cand = _unquote_ident(m.group(1))
        c_low = cand.lower()
        if c_low in sql_reserved or c_low in cte_lower or c_low.startswith(("__matview_", "__mv_")):
            continue
        if known_tables is not None and known_tables and c_low not in known_tables:
            continue
        if c_low not in seen_lower:
            seen_lower.add(c_low)
            found.append(cand)

    # Also check comma joins in FROM clauses: `FROM t1 a, t2 b WHERE ...`
    for m_from in re.finditer(
        r"\bFROM\s+([^;()]+?)(?=\bWHERE\b|\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|$)",
        no_str,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        frag = m_from.group(1)
        if "," in frag and not re.search(r"\bJOIN\b", frag, flags=re.IGNORECASE):
            for part in frag.split(","):
                toks = part.strip().split()
                if toks:
                    cand = _unquote_ident(toks[0])
                    c_low = cand.lower()
                    if (
                        re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", cand)
                        and c_low not in sql_reserved
                        and c_low not in cte_lower
                        and not c_low.startswith(("__matview_", "__mv_"))
                    ):
                        if known_tables is None or not known_tables or c_low in known_tables:
                            if c_low not in seen_lower:
                                seen_lower.add(c_low)
                                found.append(cand)

    return found


def parse_view_index_spec(
    spec: ViewIndexSpec | str | Sequence[str] | Dict[str, Any],
    default_view_name: Optional[str] = None,
) -> ViewIndexSpec:
    """Parses a user-defined index specification (SQL `CREATE [UNIQUE] INDEX ...`, tuple/list, dict, or ViewIndexSpec)."""
    if isinstance(spec, ViewIndexSpec):
        if spec.target_view is None and default_view_name:
            return ViewIndexSpec(
                columns=spec.columns,
                name=spec.name,
                unique=spec.unique,
                where_clause=spec.where_clause,
                using_method=spec.using_method,
                target_view=default_view_name,
            )
        return spec

    if isinstance(spec, dict):
        cols = tuple(str(c).strip() for c in spec.get("columns", ()) if str(c).strip())
        return ViewIndexSpec(
            columns=cols,
            name=spec.get("name"),
            unique=bool(spec.get("unique", False)),
            where_clause=spec.get("where_clause") or spec.get("where"),
            using_method=spec.get("using_method") or spec.get("using"),
            target_view=spec.get("target_view") or spec.get("view") or default_view_name,
        )

    if isinstance(spec, (list, tuple)):
        cols = tuple(str(c).strip() for c in spec if str(c).strip())
        return ViewIndexSpec(columns=cols, target_view=default_view_name)

    if not isinstance(spec, str):
        raise ValueError(f"Unsupported index specification: {spec!r}")

    s = _strip_sql_comments(spec).strip().rstrip(";").strip()
    if not s:
        raise ValueError("Empty index specification")

    if re.match(r"^CREATE\s+(?:UNIQUE\s+)?INDEX\b", s, flags=re.IGNORECASE):
        m = re.match(
            r"^CREATE\s+(UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:([A-Za-z0-9_\"`\[\]]+)\s+)?ON\s+([A-Za-z0-9_.\"`\[\]]+)(?:\s+USING\s+([A-Za-z0-9_]+))?\s*\((.*)\)(?:\s+WHERE\s+(.*))?$",
            s,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            raise ValueError(f"Invalid CREATE INDEX syntax: {spec!r}")
        is_unique = bool(m.group(1))
        idx_name = _unquote_ident(m.group(2)) if m.group(2) else None
        target_v = _unquote_ident(m.group(3))
        using_m = m.group(4).strip() if m.group(4) else None
        cols_body = m.group(5).strip()
        where_c = m.group(6).strip() if m.group(6) else None

        # Handle case where WHERE was captured inside cols_body if nested parens existed
        if where_c is None:
            # Find the matching closing paren of the column list
            raw_after_on = s[m.start(5) - 1 :]
            depth = 0
            in_single = False
            in_double = False
            close_idx = -1
            for i_ch, ch in enumerate(raw_after_on):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            close_idx = i_ch
                            break
            if close_idx != -1:
                cols_body = raw_after_on[1:close_idx].strip()
                tail = raw_after_on[close_idx + 1 :].strip()
                m_w = re.match(r"^WHERE\s+(.*)$", tail, flags=re.IGNORECASE | re.DOTALL)
                if m_w:
                    where_c = m_w.group(1).strip()

        cols = tuple(c.strip() for c in _split_top_level(cols_body, ",") if c.strip())
        if not cols:
            raise ValueError(f"CREATE INDEX must specify at least one column: {spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=idx_name,
            unique=is_unique,
            where_clause=where_c,
            using_method=using_m,
            target_view=target_v,
        )

    # Shorthand string format: e.g. "region, category DESC" or "idx_name:region,category"
    idx_name = None
    cols_part = s
    if ":" in s and not re.search(r"'.*:.*'", s):
        maybe_name, rest = s.split(":", 1)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", maybe_name.strip()):
            idx_name = maybe_name.strip()
            cols_part = rest.strip()
    cols = tuple(c.strip() for c in _split_top_level(cols_part, ",") if c.strip())
    if not cols:
        raise ValueError(f"Index specification must contain at least one column: {spec!r}")
    return ViewIndexSpec(columns=cols, name=idx_name, target_view=default_view_name)


def _parse_single_create_view_stmt(
    stmt: str,
    known_tables: Optional[Set[str]] = None,
    extra_indexes: Sequence[ViewIndexSpec] = (),
) -> ViewSpec:
    clean = _strip_sql_comments(stmt).strip().rstrip(";").strip()
    m_head = re.match(
        r"^CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:MATERIALIZED\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"`\[\]]+)(?:\s*\(([^)]+)\))?\s+AS\s+(.*)$",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_head:
        raise ValueError(f"Invalid CREATE VIEW syntax: {stmt!r}")

    view_name = _unquote_ident(m_head.group(1))
    explicit_col_names: Optional[List[str]] = None
    if m_head.group(2):
        explicit_col_names = [
            _unquote_ident(c.strip())
            for c in _split_top_level(m_head.group(2), ",")
            if c.strip()
        ]
    raw_select = m_head.group(3).strip()

    cte_names, main_select, has_cte = _extract_cte_names_and_strip_with(raw_select)
    has_window = bool(re.search(r"\bOVER\s*\(", raw_select, flags=re.IGNORECASE))
    has_compound = bool(
        _find_top_level_keyword(
            main_select,
            r"\b(?:UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b",
        )
    )
    has_limit_or_order = bool(
        _find_top_level_keyword(main_select, r"\b(?:ORDER\s+BY|LIMIT|OFFSET)\b")
    )

    # Extract first branch if compound set operation for select_items
    first_branch = main_select
    if has_compound:
        m_set = _find_top_level_keyword(
            main_select,
            r"\b(?:UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b",
        )
        if m_set:
            first_branch = main_select[: m_set.start()].strip()

    m_sel = re.match(r"^SELECT\s+(DISTINCT\s+|ALL\s+)?(.*)$", first_branch, flags=re.IGNORECASE | re.DOTALL)
    if not m_sel:
        raise ValueError(f"CREATE VIEW body must start with SELECT or WITH ... SELECT: {stmt!r}")

    has_top_distinct = bool(m_sel.group(1) and m_sel.group(1).strip().upper() == "DISTINCT")
    after_select = m_sel.group(2).strip()

    m_from = _find_top_level_keyword(after_select, r"\bFROM\b")
    if m_from:
        select_list_str = after_select[: m_from.start()].strip()
        from_and_rest = after_select[m_from.end() :].strip()
    else:
        select_list_str = after_select
        from_and_rest = ""

    raw_items = [it.strip() for it in _split_top_level(select_list_str, ",") if it.strip()]
    select_items = [_parse_select_item(it, idx) for idx, it in enumerate(raw_items)]
    if explicit_col_names and len(explicit_col_names) == len(select_items):
        select_items = [
            SelectItem(
                raw_expr=it.raw_expr,
                alias=col_alias,
                is_aggregate=it.is_aggregate,
                agg_func=it.agg_func,
                arg_expr=it.arg_expr,
                extra_args=it.extra_args,
                is_distinct=it.is_distinct,
                filter_where=it.filter_where,
                source_col=it.source_col,
            )
            for it, col_alias in zip(select_items, explicit_col_names)
        ]

    # Carve out FROM, WHERE, GROUP BY, HAVING from `from_and_rest` (on `first_branch` if not compound)
    where_clause: Optional[str] = None
    group_by_cols: List[str] = []
    having_clause: Optional[str] = None
    from_clause = from_and_rest

    if not has_compound and from_and_rest:
        # Strip trailing ORDER BY / LIMIT
        m_ord_lim = _find_top_level_keyword(from_and_rest, r"\b(?:ORDER\s+BY|LIMIT|OFFSET)\b")
        core_from = from_and_rest[: m_ord_lim.start()].strip() if m_ord_lim else from_and_rest

        m_where = _find_top_level_keyword(core_from, r"\bWHERE\b")
        m_group = _find_top_level_keyword(core_from, r"\bGROUP\s+BY\b")
        m_having = _find_top_level_keyword(core_from, r"\bHAVING\b")

        clause_positions = [p for p in [
            m_where.start() if m_where else None,
            m_group.start() if m_group else None,
            m_having.start() if m_having else None,
        ] if p is not None]

        from_clause = core_from[: min(clause_positions)].strip() if clause_positions else core_from.strip()

        if m_where:
            w_end_candidates = [p for p in [
                m_group.start() if m_group else None,
                m_having.start() if m_having else None,
            ] if p is not None and p > m_where.start()]
            w_end = min(w_end_candidates) if w_end_candidates else len(core_from)
            where_clause = core_from[m_where.end() : w_end].strip()

        if m_group:
            g_end = m_having.start() if (m_having and m_having.start() > m_group.start()) else len(core_from)
            grp_raw = core_from[m_group.end() : g_end].strip()
            group_by_cols = [g.strip() for g in _split_top_level(grp_raw, ",") if g.strip()]

        if m_having:
            having_clause = core_from[m_having.end() :].strip()

    joins, has_subquery_in_from = _parse_from_and_joins(from_clause, cte_names)
    ref_tables = _extract_all_referenced_tables(raw_select, cte_names, known_tables=known_tables)
    if not ref_tables and joins and joins[0].table_name not in cte_names and not joins[0].table_name.startswith("__mv_"):
        ref_tables = [joins[0].table_name]
    base_table = ref_tables[0] if ref_tables else (joins[0].table_name if joins else "unknown_table")

    return ViewSpec(
        view_name=view_name,
        raw_select_sql=raw_select,
        base_table=base_table,
        joins=tuple(joins),
        referenced_tables=tuple(ref_tables),
        select_items=tuple(select_items),
        where_clause=where_clause,
        group_by_cols=tuple(group_by_cols),
        having_clause=having_clause,
        has_cte=has_cte,
        has_window_func=has_window,
        has_compound_set_op=has_compound,
        has_top_level_distinct=has_top_distinct,
        has_limit_or_order=has_limit_or_order,
        has_subquery_in_from=has_subquery_in_from,
        user_indexes=tuple(extra_indexes),
    )


def parse_view_definitions(
    view_sql_or_list: Sequence[ViewSpec | str] | ViewSpec | str,
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
) -> List[ViewSpec]:
    """
    Parses one or more CREATE VIEW statements (and optional inline `CREATE [UNIQUE] INDEX`
    statements or programmatic `indexes`) into a list of `ViewSpec` objects.
    """
    raw_inputs: List[ViewSpec | str]
    if isinstance(view_sql_or_list, (str, ViewSpec)):
        raw_inputs = [view_sql_or_list]
    else:
        raw_inputs = list(view_sql_or_list)

    initial_specs: List[ViewSpec] = []
    pending_index_specs: List[ViewIndexSpec] = []

    for item in raw_inputs:
        if isinstance(item, ViewSpec):
            initial_specs.append(item)
            for av in item.additional_views:
                initial_specs.append(av)
            continue
        clean = _strip_sql_comments(item).strip()
        stmts = [st.strip() for st in _split_top_level(clean, ";") if st.strip()]
        if not stmts:
            raise ValueError(f"Invalid CREATE VIEW syntax: {item!r}")
        for st in stmts:
            if re.match(
                r"^CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:MATERIALIZED\s+)?VIEW\b",
                st,
                flags=re.IGNORECASE,
            ):
                initial_specs.append(_parse_single_create_view_stmt(st, known_tables=known_tables))
            elif re.match(r"^CREATE\s+(?:UNIQUE\s+)?INDEX\b", st, flags=re.IGNORECASE):
                pending_index_specs.append(parse_view_index_spec(st))
            else:
                raise ValueError(f"Unsupported statement in view definition script: {st!r}")

    if not initial_specs:
        raise ValueError(f"No CREATE VIEW statement found in: {view_sql_or_list!r}")

    if indexes:
        default_v = initial_specs[0].view_name if len(initial_specs) == 1 else None
        for raw_idx in indexes:
            pending_index_specs.append(parse_view_index_spec(raw_idx, default_view_name=default_v))

    if not pending_index_specs:
        return initial_specs

    by_view_idx: Dict[int, List[ViewIndexSpec]] = {i: list(vs.user_indexes) for i, vs in enumerate(initial_specs)}
    for idx_spec in pending_index_specs:
        if idx_spec.target_view is None and len(initial_specs) == 1:
            by_view_idx[0].append(idx_spec)
            continue
        matched = False
        if idx_spec.target_view is not None:
            t_low = idx_spec.target_view.lower()
            for i, vs in enumerate(initial_specs):
                v_low = vs.view_name.lower()
                v_unpref = v_low[len(DEFAULT_OBJECT_PREFIX) :] if v_low.startswith(DEFAULT_OBJECT_PREFIX) else v_low
                if t_low in {
                    v_low,
                    v_unpref,
                    f"{DEFAULT_OBJECT_PREFIX}{v_unpref}",
                    f"{DEFAULT_OBJECT_PREFIX}store_{v_unpref}",
                    f"__mv_store_{v_unpref}",
                }:
                    by_view_idx[i].append(idx_spec)
                    matched = True
                    break
            if not matched:
                by_view_idx[0].append(idx_spec)
        else:
            for i, vs in enumerate(initial_specs):
                exposed_low = {it.alias.lower() for it in vs.select_items}
                cols_low = {
                    _unquote_ident(re.sub(r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST)\b", "", c, flags=re.IGNORECASE).strip()).lower()
                    for c in idx_spec.columns
                }
                if cols_low and cols_low.issubset(exposed_low):
                    by_view_idx[i].append(idx_spec)
                    matched = True
            if not matched:
                by_view_idx[0].append(idx_spec)

    out_specs: List[ViewSpec] = []
    for i, vs in enumerate(initial_specs):
        out_specs.append(
            ViewSpec(
                view_name=vs.view_name,
                raw_select_sql=vs.raw_select_sql,
                base_table=vs.base_table,
                joins=vs.joins,
                referenced_tables=vs.referenced_tables,
                select_items=vs.select_items,
                where_clause=vs.where_clause,
                group_by_cols=vs.group_by_cols,
                having_clause=vs.having_clause,
                has_cte=vs.has_cte,
                has_window_func=vs.has_window_func,
                has_compound_set_op=vs.has_compound_set_op,
                has_top_level_distinct=vs.has_top_level_distinct,
                has_limit_or_order=vs.has_limit_or_order,
                has_subquery_in_from=vs.has_subquery_in_from,
                user_indexes=tuple(by_view_idx[i]),
            )
        )
    return out_specs


def parse_view_definition(
    view_sql: str | ViewSpec,
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
) -> ViewSpec:
    """
    Parses a single-view or multi-view SQL script into a `ViewSpec`.
    If the script contains multiple `CREATE VIEW` statements, the first `ViewSpec`
    carries the remaining views in `additional_views` so both `parse_view_definition`
    and `generate_materialized_view` handle multi-view `.sql` files seamlessly.
    """
    specs = parse_view_definitions(view_sql, known_tables=known_tables, indexes=indexes)
    if len(specs) == 1:
        return specs[0]
    first = specs[0]
    return ViewSpec(
        view_name=first.view_name,
        raw_select_sql=first.raw_select_sql,
        base_table=first.base_table,
        joins=first.joins,
        referenced_tables=first.referenced_tables,
        select_items=first.select_items,
        where_clause=first.where_clause,
        group_by_cols=first.group_by_cols,
        having_clause=first.having_clause,
        has_cte=first.has_cte,
        has_window_func=first.has_window_func,
        has_compound_set_op=first.has_compound_set_op,
        has_top_level_distinct=first.has_top_level_distinct,
        has_limit_or_order=first.has_limit_or_order,
        has_subquery_in_from=first.has_subquery_in_from,
        user_indexes=first.user_indexes,
        additional_views=tuple(specs[1:]),
    )


# -----------------------------------------------------------------------------
# Incremental Materialized View Compiler (`IncrementalViewCompiler`)
# -----------------------------------------------------------------------------


class IncrementalViewCompiler:
    """
    Compiles a `ViewSpec` + `Dict[str, TableSchema]` into a `MaterializedViewPlan`.
    Selects the highest-granularity maintenance strategy supported by the query:
      - Tier 1 (`ALGEBRAIC_DELTA`):
          * Single-table O(1) algebraic aggregates (including compound aggregate arithmetic,
            expression GROUP BY, hidden GROUP BY columns, HAVING, FILTER, and DISTINCT aggs)
          * Multi-table INNER JOIN / CROSS JOIN O(1) algebraic aggregates
          * Z-set set operations (`UNION ALL`, `UNION`, `INTERSECT`, `EXCEPT`)
          * Z-set `SELECT DISTINCT` & PK-lineage non-aggregate projections
          * Aggregations over `UNION ALL` CTEs / derived tables
          * Automatic single-table CTE / subquery unwrapping
      - Tier 2 (`GROUP_SCOPED_DELTA`):
          * Group-scoped recomputation for holistic aggregates (`GROUP_CONCAT`, `STRING_AGG`,
            `JSON_GROUP_ARRAY`, `ARRAY_AGG`), `LEFT JOIN` grouped queries, etc.
      - Tier 3 (`ATOMIC_QUERY_SYNC`):
          * Universal atomic refresh for window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`,
            `LAG`, `LEAD`), recursive CTEs, `ORDER BY ... LIMIT`, correlated subqueries, etc.
    """

    def __init__(
        self,
        schemas: Dict[str, TableSchema],
        view_spec: ViewSpec,
        dialect: Dialect = Dialect.SQLITE,
        pg_use_transition_tables: bool = False,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        incremental_backfill: bool = False,
        batch_size: int = 1000,
    ) -> None:
        self.schemas = dict(schemas)
        self._schemas_lower: Dict[str, TableSchema] = {
            k.lower(): v for k, v in self.schemas.items()
        }
        self.raw_view = view_spec
        self.dialect = dialect
        self.pg_use_transition_tables = pg_use_transition_tables
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.incremental_backfill = incremental_backfill
        self.batch_size = max(1, int(batch_size)) if batch_size else 1000

        raw_vname = view_spec.view_name
        if self.object_prefix and raw_vname.startswith(self.object_prefix):
            self.logical_view_name = raw_vname[len(self.object_prefix) :]
            self.db_view_name = raw_vname
        elif self.object_prefix:
            self.logical_view_name = raw_vname
            self.db_view_name = f"{self.object_prefix}{raw_vname}"
        else:
            self.logical_view_name = raw_vname
            self.db_view_name = raw_vname

        self.user_indexes: List[ViewIndexSpec] = list(view_spec.user_indexes)
        self.view = self._try_unwrap_simple_cte_or_derived_view(view_spec)
        self.primary_schema: Optional[TableSchema] = self._schemas_lower.get(
            self.view.base_table.lower()
        )

        (
            self._working_select_items,
            self._compound_reader_exprs,
            self._extra_having_select_items,
            self._rewritten_having_clause,
            self._decomposed_ok,
        ) = self._decompose_compound_aggregates_and_having()

        self._effective_group_by_exprs: Tuple[str, ...] = self._resolve_effective_group_by_exprs()
        self.limitations: List[TransparencyLimitation] = self._detect_limitations()
        self._enforce_limitations()
        self._validate_schema_and_view()
        self._alg_mode: str = "single"
        self.strategy: MaintenanceStrategy = self._choose_strategy()

    def _backing_table_name(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return f"{p}store_{self.logical_view_name}"

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
        return lims

    def _enforce_limitations(self) -> None:
        if not self.limitations:
            return
        if not self.ignore_limitations:
            raise EngineLimitationError(self.logical_view_name, self.dialect, self.limitations)
        summary = "; ".join(f"[{l.code}] {l.description}" for l in self.limitations)
        warnings.warn(
            f"Proceeding with materialized view '{self.logical_view_name}' on '{self.dialect.value}' "
            f"with --ignore-limitations: {summary}",
            EngineLimitationWarning,
            stacklevel=2,
        )

    def _try_unwrap_simple_cte_or_derived_view(self, v: ViewSpec) -> ViewSpec:
        """
        Unwraps a non-recursive single-CTE or derived subquery over a single base table
        when the inner query is a flat projection/filter (no inner GROUP BY, aggregates,
        DISTINCT, window functions, set ops, or LIMIT) so the outer query can use Tier 1 O(1)
        algebraic maintenance.
        """
        if v.has_window_func or v.has_compound_set_op or v.has_limit_or_order:
            return v

        inner_sql: Optional[str] = None
        cte_or_sub_alias: Optional[str] = None
        outer_sql: Optional[str] = None

        if v.has_cte:
            if re.match(r"^WITH\s+RECURSIVE\b", v.raw_select_sql.strip(), flags=re.IGNORECASE):
                return v
            cte_names, main_sel, _ = _extract_cte_names_and_strip_with(v.raw_select_sql)
            if len(cte_names) != 1:
                return v
            cte_name = next(iter(cte_names))
            m_cte = re.match(
                r"^WITH\s+([A-Za-z0-9_\"`\[\]]+)\s*AS\s*(?:(?:MATERIALIZED|NOT\s+MATERIALIZED)\s*)?\((.*)\)\s*"
                + re.escape(main_sel)
                + r"\s*$",
                v.raw_select_sql.strip(),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m_cte:
                return v
            inner_sql = m_cte.group(2).strip()
            cte_or_sub_alias = cte_name
            outer_sql = main_sel
        elif v.has_subquery_in_from and len(v.joins) == 1:
            m_from = _find_top_level_keyword(v.raw_select_sql, r"\bFROM\b")
            if not m_from:
                return v
            from_rest = v.raw_select_sql[m_from.end() :].strip()
            kw_pos = []
            for kw_p in [r"\bWHERE\b", r"\bGROUP\s+BY\b", r"\bHAVING\b", r"\bORDER\s+BY\b", r"\bLIMIT\b"]:
                mk = _find_top_level_keyword(from_rest, kw_p)
                if mk:
                    kw_pos.append(mk.start())
            from_part = from_rest[: min(kw_pos)].strip() if kw_pos else from_rest
            m_sub = re.match(
                r"^\((.*)\)(?:\s+(?:AS\s+)?([A-Za-z0-9_\"`\[\]]+))?$",
                from_part,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m_sub:
                return v
            inner_sql = m_sub.group(1).strip()
            cte_or_sub_alias = _unquote_ident(m_sub.group(2)) if m_sub.group(2) else "__mv_sub"
            outer_sql = v.raw_select_sql
        else:
            return v

        if not inner_sql or not outer_sql:
            return v
        if _find_top_level_keyword(inner_sql, r"\b(?:UNION|INTERSECT|EXCEPT|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT)\b"):
            return v

        try:
            inner_spec = _parse_single_create_view_stmt(
                f"CREATE VIEW __inner_tmp AS {inner_sql}",
                known_tables=set(self._schemas_lower.keys()),
            )
        except Exception:
            return v

        if (
            inner_spec.has_cte
            or inner_spec.has_window_func
            or inner_spec.has_compound_set_op
            or inner_spec.has_top_level_distinct
            or inner_spec.has_limit_or_order
            or inner_spec.has_subquery_in_from
            or len(inner_spec.joins) != 1
            or inner_spec.group_by_cols
            or any(it.is_aggregate for it in inner_spec.select_items)
        ):
            return v

        if len(v.joins) != 1:
            return v

        inner_base = inner_spec.joins[0].table_name
        if inner_base.lower() not in self._schemas_lower:
            return v

        # Build column substitution map: alias -> raw_expr (with inner table alias stripped)
        inner_alias = inner_spec.joins[0].alias
        col_map: Dict[str, str] = {}
        for it in inner_spec.select_items:
            if it.raw_expr.strip() == "*":
                ts = self._schemas_lower[inner_base.lower()]
                for c_name in ts.columns:
                    col_map[c_name.lower()] = c_name
            else:
                cleaned_e = re.sub(
                    rf"\b{re.escape(inner_alias)}\.",
                    "",
                    it.raw_expr.strip(),
                    flags=re.IGNORECASE,
                )
                col_map[it.alias.lower()] = (
                    cleaned_e
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", cleaned_e)
                    else f"({cleaned_e})"
                )

        outer_alias = v.joins[0].alias

        def _subst(frag: Optional[str]) -> Optional[str]:
            if frag is None:
                return None
            tokens = re.split(r"('(?:''|[^'])*')", frag)
            out: List[str] = []
            for idx_t, tok in enumerate(tokens):
                if idx_t % 2 == 1:
                    out.append(tok)
                    continue
                mod = tok
                if outer_alias:
                    mod = re.sub(rf"\b{re.escape(outer_alias)}\.", "", mod, flags=re.IGNORECASE)
                if cte_or_sub_alias:
                    mod = re.sub(rf"\b{re.escape(cte_or_sub_alias)}\.", "", mod, flags=re.IGNORECASE)

                def _repl_id(m_id: re.Match[str]) -> str:
                    w = m_id.group(1)
                    return col_map.get(w.lower(), w)

                mod = re.sub(r"(?<![.\w])\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", _repl_id, mod)
                out.append(mod)
            return "".join(out)

        new_items: List[SelectItem] = []
        for it in v.select_items:
            sub_raw = _subst(it.raw_expr) or it.raw_expr
            parsed = _parse_select_item(f"{sub_raw} AS {it.alias}", 0)
            new_items.append(parsed)

        inner_w = None
        if inner_spec.where_clause:
            inner_w = re.sub(
                rf"\b{re.escape(inner_alias)}\.",
                "",
                inner_spec.where_clause,
                flags=re.IGNORECASE,
            )
        outer_w = _subst(v.where_clause)
        if inner_w and outer_w:
            comb_w = f"({inner_w}) AND ({outer_w})"
        else:
            comb_w = inner_w or outer_w

        new_grps = tuple(_subst(g) or g for g in v.group_by_cols)
        new_hav = _subst(v.having_clause)

        sel_str = ", ".join(f"{it.raw_expr} AS {it.alias}" for it in new_items)
        Dist = "DISTINCT " if v.has_top_level_distinct else ""
        rebuilt_sql = f"SELECT {Dist}{sel_str} FROM {inner_base}"
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
        agg_call_to_alias: Dict[str, str] = {}
        for it in v.select_items:
            if it.is_aggregate and it.agg_func is not None:
                agg_call_to_alias[re.sub(r"\s+", " ", it.raw_expr.strip().upper())] = it.alias

        def _extract_agg_calls_from_expr(expr_str: str) -> Tuple[str, bool]:
            nonlocal elem_counter
            s = expr_str
            out_chars: List[str] = []
            i = 0
            n = len(s)
            in_single = False
            in_double = False
            while i < n:
                ch = s[i]
                if ch == "'" and not in_double:
                    in_single = not in_single
                    out_chars.append(ch)
                    i += 1
                    continue
                if ch == '"' and not in_single:
                    in_double = not in_double
                    out_chars.append(ch)
                    i += 1
                    continue
                if not in_single and not in_double:
                    m_fn = re.match(
                        r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                        s[i:],
                        flags=re.IGNORECASE,
                    )
                    if m_fn and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] in "_.")):
                        fn_name = m_fn.group(1).upper()
                        if fn_name in {
                            "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX",
                            "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                            "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
                        }:
                            open_p = i + m_fn.end() - 1
                            depth = 1
                            j = open_p + 1
                            in_s2 = False
                            in_d2 = False
                            while j < n and depth > 0:
                                c2 = s[j]
                                if c2 == "'" and not in_d2:
                                    in_s2 = not in_s2
                                elif c2 == '"' and not in_s2:
                                    in_d2 = not in_d2
                                elif not in_s2 and not in_d2:
                                    if c2 == "(":
                                        depth += 1
                                    elif c2 == ")":
                                        depth -= 1
                                j += 1
                            if depth != 0:
                                return expr_str, False
                            # Check for optional FILTER (WHERE ...)
                            m_filt = re.match(r"^\s*FILTER\s*\(", s[j:], flags=re.IGNORECASE)
                            if m_filt:
                                fp = j + m_filt.end() - 1
                                d_f = 1
                                k = fp + 1
                                while k < n and d_f > 0:
                                    if s[k] == "(":
                                        d_f += 1
                                    elif s[k] == ")":
                                        d_f -= 1
                                    k += 1
                                j = k
                            full_call = s[i:j].strip()
                            norm_key = re.sub(r"\s+", " ", full_call.upper())
                            if norm_key in agg_call_to_alias:
                                col_al = agg_call_to_alias[norm_key]
                            else:
                                elem_counter += 1
                                col_al = f"__mv_elem_{elem_counter}"
                                parsed_el = _parse_single_agg_expr(full_call, col_al)
                                if parsed_el is None:
                                    return expr_str, False
                                agg_call_to_alias[norm_key] = col_al
                                work_items.append(parsed_el)
                                extra_having_items.append(parsed_el)
                            out_chars.append(col_al)
                            i = j
                            continue
                out_chars.append(ch)
                i += 1
            return "".join(out_chars), True

        all_ok = True
        for it in v.select_items:
            if not it.is_aggregate or it.agg_func is not None:
                work_items.append(it)
                reader_exprs[it.alias] = it.alias
            else:
                rewritten_e, ok = _extract_agg_calls_from_expr(it.raw_expr)
                if not ok:
                    all_ok = False
                    work_items.append(it)
                    reader_exprs[it.alias] = it.alias
                else:
                    reader_exprs[it.alias] = f"({rewritten_e}) AS {it.alias}"

        rewritten_having: Optional[str] = None
        if v.having_clause:
            rh, ok_h = _extract_agg_calls_from_expr(v.having_clause)
            if ok_h:
                # Also strip table qualifiers on group columns in HAVING if they match group aliases
                for g_it in v.select_items:
                    if not g_it.is_aggregate and "." in g_it.raw_expr:
                        rh = re.sub(
                            rf"\b{re.escape(g_it.raw_expr.strip())}\b",
                            g_it.alias,
                            rh,
                            flags=re.IGNORECASE,
                        )
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
            if gs.lower() in alias_to_raw:
                # Check if gs is an alias for an expression rather than a base column
                raw_e = alias_to_raw[gs.lower()]
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
        if v.where_clause and re.search(r"\(\s*SELECT\b", v.where_clause, flags=re.IGNORECASE):
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
            if re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE):
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
        if v.where_clause and re.search(r"\(\s*SELECT\b", v.where_clause, flags=re.IGNORECASE):
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
            if re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE):
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
        if v.where_clause and re.search(r"\(\s*SELECT\b", v.where_clause, flags=re.IGNORECASE):
            return False
        if any(it.raw_expr.strip() == "*" or re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE) for it in v.select_items):
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
        if v.where_clause and re.search(r"\(\s*SELECT\b", v.where_clause, flags=re.IGNORECASE):
            return False
        if len(v.select_items) == 1 and v.select_items[0].raw_expr.strip() == "*":
            return False
        if not any(c.primary_key for c in self.primary_schema.columns.values()):
            return False
        for it in v.select_items:
            if re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE):
                return False
        return True

    def _can_use_union_all_agg_algebraic(self) -> bool:
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
            return False
        if v.where_clause and re.search(r"\(\s*SELECT\b", v.where_clause, flags=re.IGNORECASE):
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
            AggFunc.BIT_XOR,
            AggFunc.BOOL_AND,
            AggFunc.BOOL_OR,
        }
        for it in self._working_select_items:
            if re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE):
                return False
            if it.is_aggregate and (it.is_distinct or it.agg_func not in allowed_aggs):
                return False

        inner_union_sql: Optional[str] = None
        sub_alias: str = "__mv_u"
        cte_names, main_select, has_cte = _extract_cte_names_and_strip_with(v.raw_select_sql)
        if has_cte:
            if len(cte_names) != 1 or re.match(r"^WITH\s+RECURSIVE\b", v.raw_select_sql.strip(), flags=re.IGNORECASE):
                return False
            cte_name = next(iter(cte_names))
            m_cte = re.match(
                r"^WITH\s+([A-Za-z0-9_\"`\[\]]+)\s*AS\s*(?:(?:MATERIALIZED|NOT\s+MATERIALIZED)\s*)?\((.*)\)\s*"
                + re.escape(main_select)
                + r"\s*$",
                v.raw_select_sql.strip(),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m_cte or len(v.joins) != 1 or v.joins[0].table_name.lower() != cte_name.lower():
                return False
            inner_union_sql = m_cte.group(2).strip()
            sub_alias = v.joins[0].alias or cte_name
        else:
            m_from = _find_top_level_keyword(main_select, r"\bFROM\b")
            if not m_from:
                return False
            from_rest = main_select[m_from.end() :].strip()
            kw_pos = []
            for kw_p in [r"\bWHERE\b", r"\bGROUP\s+BY\b", r"\bHAVING\b"]:
                mk = _find_top_level_keyword(from_rest, kw_p)
                if mk:
                    kw_pos.append(mk.start())
            from_clause = from_rest[: min(kw_pos)].strip() if kw_pos else from_rest.strip()
            m_sub = re.match(
                r"^\((.*)\)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?$",
                from_clause,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m_sub:
                return False
            inner_union_sql = m_sub.group(1).strip()
            if m_sub.group(2):
                sub_alias = m_sub.group(2).strip()

        if not inner_union_sql:
            return False

        op_matches = _find_all_top_level_keywords(
            inner_union_sql,
            r"\b(?:UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b",
        )
        if not op_matches:
            return False
        for om in op_matches:
            if not re.match(r"^UNION\s+ALL$", om.group(0).strip(), flags=re.IGNORECASE):
                return False

        branch_sqls: List[str] = []
        last_pos = 0
        for om in op_matches:
            branch_sqls.append(inner_union_sql[last_pos : om.start()].strip())
            last_pos = om.end()
        branch_sqls.append(inner_union_sql[last_pos:].strip())

        branch_specs: List[ViewSpec] = []
        for idx_b, b_sql in enumerate(branch_sqls):
            try:
                bs = _parse_single_create_view_stmt(
                    f"CREATE VIEW __uall_b_{idx_b} AS {b_sql}",
                    known_tables=set(self._schemas_lower.keys()),
                )
            except Exception:
                return False
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
                return False
            if bs.where_clause and re.search(r"\(\s*SELECT\b", bs.where_clause, flags=re.IGNORECASE):
                return False
            branch_specs.append(bs)

        self._union_all_inner_sql = inner_union_sql
        self._union_all_sub_alias = sub_alias
        self._union_all_branch_specs = branch_specs
        return True

    def _choose_strategy(self) -> MaintenanceStrategy:
        if self._can_use_single_table_algebraic():
            self._alg_mode = "single"
            return MaintenanceStrategy.ALGEBRAIC_DELTA
        if self._can_use_join_algebraic():
            self._alg_mode = "join"
            return MaintenanceStrategy.ALGEBRAIC_DELTA
        if self._can_use_zset_setop():
            self._alg_mode = "setop"
            return MaintenanceStrategy.ALGEBRAIC_DELTA
        if self._can_use_zset_distinct():
            self._alg_mode = "distinct"
            return MaintenanceStrategy.ALGEBRAIC_DELTA
        if self._can_use_zset_projection():
            self._alg_mode = "projection"
            return MaintenanceStrategy.ALGEBRAIC_DELTA
        if self._can_use_union_all_agg_algebraic():
            self._alg_mode = "union_all_agg"
            return MaintenanceStrategy.ALGEBRAIC_DELTA

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
            return MaintenanceStrategy.GROUP_SCOPED_DELTA

        return MaintenanceStrategy.ATOMIC_QUERY_SYNC

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
                if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", g_expr.strip()):
                    g_col = _unquote_ident(g_expr)
                    if g_col.lower() not in col_map:
                        raise ValueError(
                            f"GROUP BY column {g_col!r} does not exist on table {self.primary_schema.name!r}"
                        )
            grp_exprs_lower = {g.strip().lower() for g in self._effective_group_by_exprs}
            grp_cols_lower = {_unquote_ident(g).lower() for g in self._effective_group_by_exprs}
            has_any_agg = any(it.is_aggregate for it in self._working_select_items)
            for item in self._working_select_items:
                if not item.is_aggregate:
                    if item.source_col is not None and re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", item.raw_expr.strip()):
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
                elif item.arg_expr and re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", item.arg_expr.strip()):
                    col_ident = _unquote_ident(item.arg_expr)
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

    @staticmethod
    def _extract_index_col_base_ident(col_spec: str) -> str:
        cleaned = re.sub(
            r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|COLLATE\s+[A-Za-z0-9_.\"'`]+)\b",
            "",
            col_spec,
            flags=re.IGNORECASE,
        ).strip()
        return _unquote_ident(cleaned)

    def _validate_user_indexes(self) -> None:
        if not self.user_indexes:
            return
        exposed = self._exposed_view_columns()
        exposed_lower = {c.lower() for c in exposed}
        sql_kw = {
            "asc", "desc", "nulls", "first", "last", "collate", "nocase", "binary",
            "rtrim", "and", "or", "not", "is", "null", "true", "false", "in",
            "between", "like", "case", "when", "then", "else", "end", "cast", "as",
            "int", "integer", "text", "real", "double", "precision", "varchar", "char",
        } | {f.lower() for f in _SQL_EXPR_FUNCTIONS}

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
                base_expr = re.sub(
                    r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST|COLLATE\s+[A-Za-z0-9_.\"'`]+)\b",
                    "",
                    col_spec,
                    flags=re.IGNORECASE,
                ).strip()
                if re.match(r"^[A-Za-z0-9_\"`\[\]]+$", base_expr):
                    c_ident = _unquote_ident(base_expr)
                    if c_ident.lower() not in exposed_lower:
                        raise ValueError(
                            f"Index on view '{self.logical_view_name}' references unknown column "
                            f"{c_ident!r}. Available view columns: {exposed}"
                        )
                else:
                    no_str = re.sub(r"'(?:''|[^'])*'", "''", base_expr)
                    referenced_cols: List[str] = []
                    for m_id in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", no_str):
                        tok = m_id.group(1)
                        if tok.lower() in sql_kw:
                            continue
                        if tok.lower() not in exposed_lower:
                            raise ValueError(
                                f"Index expression {col_spec!r} on view '{self.logical_view_name}' "
                                f"references unknown column {tok!r}. Available view columns: {exposed}"
                            )
                        referenced_cols.append(tok)
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

    def _build_all_index_ddls(
        self,
        mv_table: str,
        auto_group_cols: Sequence[str] = (),
        has_pk_on_group_cols: bool = False,
        backing_col_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], List[str]]:
        ddls: List[str] = []
        names: List[str] = []
        p = self.object_prefix if self.object_prefix else "__matview_"
        log_vname = self.logical_view_name

        if auto_group_cols and not has_pk_on_group_cols:
            idx_name = f"{p}idx_{log_vname}__grp"
            ddls.append(f"CREATE INDEX {idx_name} ON {mv_table} ({', '.join(auto_group_cols)});")
            names.append(idx_name)

        for i, idx_spec in enumerate(self.user_indexes, 1):
            if idx_spec.name:
                raw_n = idx_spec.name
                if self.object_prefix and not raw_n.startswith(self.object_prefix):
                    idx_name = f"{self.object_prefix}{raw_n}"
                else:
                    idx_name = raw_n
            else:
                col_tag = "_".join(_sanitize_ident(self._extract_index_col_base_ident(c)) for c in idx_spec.columns)
                idx_name = f"{p}idx_{log_vname}__{col_tag}_{i}"

            mapped_cols: List[str] = []
            for col_spec in idx_spec.columns:
                c_mod = col_spec.strip()
                if backing_col_map:
                    for u_col, b_expr in backing_col_map.items():
                        c_mod = re.sub(rf"\b{re.escape(u_col)}\b", b_expr, c_mod, flags=re.IGNORECASE)
                if self.dialect == Dialect.MYSQL:
                    c_mod = re.sub(r"\bNULLS\s+(?:FIRST|LAST)\b", "", c_mod, flags=re.IGNORECASE).strip()
                    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s+(?:ASC|DESC))?$", c_mod, flags=re.IGNORECASE):
                        m_dir = re.search(r"\s+(ASC|DESC)\s*$", c_mod, flags=re.IGNORECASE)
                        dir_s = f" {m_dir.group(1).upper()}" if m_dir else ""
                        expr_part = c_mod[: m_dir.start()].strip() if m_dir else c_mod
                        if not (expr_part.startswith("((") and expr_part.endswith("))")):
                            c_mod = f"(({expr_part})){dir_s}"
                elif self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                    if not re.match(
                        r"^[A-Za-z_][A-Za-z0-9_]*(\s+COLLATE\s+\S+)?(\s+(?:ASC|DESC))?(\s+NULLS\s+(?:FIRST|LAST))?$",
                        c_mod,
                        flags=re.IGNORECASE,
                    ):
                        m_tail = re.search(
                            r"(\s+(?:ASC|DESC))?(\s+NULLS\s+(?:FIRST|LAST))?\s*$",
                            c_mod,
                            flags=re.IGNORECASE,
                        )
                        tail_s = m_tail.group(0) if m_tail else ""
                        expr_part = c_mod[: len(c_mod) - len(tail_s)].strip()
                        if not (expr_part.startswith("(") and expr_part.endswith(")")):
                            c_mod = f"({expr_part}){tail_s}"
                mapped_cols.append(c_mod)

            u_kw = "UNIQUE " if idx_spec.unique else ""
            using_str = (
                f" USING {idx_spec.using_method}"
                if idx_spec.using_method and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
                else ""
            )
            where_str = ""
            if idx_spec.where_clause and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                w_mod = idx_spec.where_clause
                if backing_col_map:
                    for u_col, b_expr in backing_col_map.items():
                        w_mod = re.sub(rf"\b{re.escape(u_col)}\b", b_expr, w_mod, flags=re.IGNORECASE)
                where_str = f" WHERE {w_mod}"

            ddls.append(
                f"CREATE {u_kw}INDEX {idx_name} ON {mv_table}{using_str} ({', '.join(mapped_cols)}){where_str};"
            )
            names.append(idx_name)

        return ddls, names

    def _lookup_col_def(self, expr: Optional[str]) -> Optional[ColumnDef]:
        if not expr:
            return None
        s = expr.strip()
        if not re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", s):
            return None
        if "." in s:
            prefix, col_part = s.split(".", 1)
            prefix_clean = _unquote_ident(prefix).lower()
            col_clean = _unquote_ident(col_part).lower()
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

        col_clean = _unquote_ident(s).lower()
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

    def _normalize_dialect_type(self, dt: str) -> str:
        u = dt.strip().upper()
        if self.dialect == Dialect.POSTGRES:
            if u == "DOUBLE":
                return "DOUBLE PRECISION"
            if u == "CHARACTER VARYING":
                return "VARCHAR(255)"
        if self.dialect == Dialect.MYSQL:
            if u == "DOUBLE PRECISION":
                return "DOUBLE"
            if u in ("VARCHAR", "CHAR", "CHARACTER VARYING", "CHARACTER"):
                return "VARCHAR(255)"
        return dt

    def _infer_cte_or_subquery_col_type(self, col_name: str) -> Optional[str]:
        ident = _unquote_ident(col_name.split(".")[-1]).strip()
        if not ident:
            return None
        pat = re.compile(
            rf'((?:COUNT|SUM|AVG|TOTAL|MIN|MAX|VAR_POP|VARIANCE|VAR_SAMP|STDDEV_POP|STDDEV|STD|STDDEV_SAMP|ROW_NUMBER|RANK|DENSE_RANK|NTILE)\s*\(([^)]*)\)(?:\s+FILTER\s*\([^)]*\))?(?:\s+OVER\s*\([^)]*\))?)\s+AS\s+["`\[]?{re.escape(ident)}["`\]]?\b',
            re.IGNORECASE,
        )
        m = pat.search(self.view.raw_select_sql)
        if m:
            fn_expr = m.group(1).strip().upper()
            inner_arg = m.group(2).strip()
            if fn_expr.startswith(("COUNT", "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE")):
                return self._int_type()
            if fn_expr.startswith("SUM"):
                c_arg = self._lookup_col_def(inner_arg)
                if c_arg and "INT" in c_arg.data_type.upper():
                    return self._int_type()
                return self._float_type()
            if fn_expr.startswith(("MIN", "MAX")):
                c_arg = self._lookup_col_def(inner_arg)
                if c_arg:
                    return self._normalize_dialect_type(c_arg.data_type)
                return self._float_type()
            if fn_expr.startswith(("AVG", "TOTAL", "VAR_", "VARIANCE", "STDDEV", "STD")):
                return self._float_type()
        pat_col = re.compile(
            rf'\b([A-Za-z_][A-Za-z0-9_.]*)\s+AS\s+["`\[]?{re.escape(ident)}["`\]]?\b',
            re.IGNORECASE,
        )
        for m_col in pat_col.finditer(self.view.raw_select_sql):
            src_ident = m_col.group(1)
            if src_ident.lower() != ident.lower():
                c_def = self._lookup_col_def(src_ident)
                if c_def:
                    return self._normalize_dialect_type(c_def.data_type)
        pat_arith = re.compile(
            rf'([A-Za-z0-9_."`\[\]]+\s*[+\-*/%]\s*[A-Za-z0-9_."`\[\]]+)\s+AS\s+["`\[]?{re.escape(ident)}["`\]]?\b',
            re.IGNORECASE,
        )
        m_arith = pat_arith.search(self.view.raw_select_sql)
        if m_arith:
            a_expr = m_arith.group(1)
            if "/" not in a_expr and not re.search(r"\d+\.\d+", a_expr):
                cols_found = [
                    self._lookup_col_def(m_id.group(1))
                    for m_id in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_.]*)\b", a_expr)
                ]
                valid_cols = [c for c in cols_found if c is not None]
                if valid_cols and all("INT" in c.data_type.upper() for c in valid_cols):
                    return self._int_type()
            return self._float_type()
        for m_hdr in re.finditer(
            r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]+)\)\s+AS\s*\(\s*SELECT\s+(.*?)\bFROM\b",
            self.view.raw_select_sql,
            re.IGNORECASE | re.DOTALL,
        ):
            hdr_cols = [_unquote_ident(c.strip()).lower() for c in _split_top_level(m_hdr.group(1), ",")]
            sel_exprs = [e.strip() for e in _split_top_level(m_hdr.group(2), ",")]
            if ident.lower() in hdr_cols:
                idx_h = hdr_cols.index(ident.lower())
                if idx_h < len(sel_exprs):
                    src_e = re.sub(r'\s+AS\s+[A-Za-z0-9_"`\[\]]+$', "", sel_exprs[idx_h], flags=re.IGNORECASE).strip()
                    if re.match(r"^-?\d+$", src_e):
                        return self._int_type()
                    if re.match(r"^-?\d+\.\d+$", src_e):
                        return self._float_type()
                    c_src = self._lookup_col_def(src_e)
                    if c_src:
                        return self._normalize_dialect_type(c_src.data_type)
        if "." in col_name:
            c_base = self._lookup_col_def(ident)
            if c_base:
                return self._normalize_dialect_type(c_base.data_type)
        return None

    def _infer_expr_storage_type(self, item: SelectItem) -> str:
        if not item.is_aggregate:
            c_def = self._lookup_col_def(item.raw_expr)
            if c_def:
                return self._normalize_dialect_type(c_def.data_type)
            cte_t = self._infer_cte_or_subquery_col_type(item.raw_expr)
            if cte_t:
                return cte_t
            raw_u = item.raw_expr.strip().upper()
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
        if item.agg_func in (
            AggFunc.BIT_AND,
            AggFunc.BIT_OR,
            AggFunc.BIT_XOR,
        ):
            return "BIGINT UNSIGNED" if self.dialect == Dialect.MYSQL else self._int_type()
        if item.agg_func in (
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
        ):
            return self._int_type()
        if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
            return "BOOLEAN" if self.dialect == Dialect.POSTGRES else "INTEGER"
        if item.agg_func in (
            AggFunc.AVG,
            AggFunc.TOTAL,
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
        ):
            return self._float_type()
        if item.agg_func in (
            AggFunc.GROUP_CONCAT,
            AggFunc.STRING_AGG,
            AggFunc.JSON_GROUP_ARRAY,
            AggFunc.JSON_GROUP_OBJECT,
            AggFunc.ARRAY_AGG,
        ):
            return "TEXT"
        if item.agg_func == AggFunc.SUM:
            if item.arg_expr:
                c_def = self._lookup_col_def(item.arg_expr)
                if c_def and any(k in c_def.data_type.upper() for k in ("INT", "SERIAL")):
                    if not any(op in item.arg_expr for op in (".", "/")):
                        return self._int_type()
            return self._float_type()
        if item.agg_func in (AggFunc.MIN, AggFunc.MAX) and item.arg_expr:
            c_def = self._lookup_col_def(item.arg_expr)
            if c_def:
                return self._normalize_dialect_type(c_def.data_type)
        return self._float_type()

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

    def _canonical_group_pk_expr(self, group_exprs: Sequence[str]) -> str:
        if self.dialect == Dialect.SQLITE:
            return " || '|' || ".join(f"quote({e})" for e in group_exprs)
        elif self.dialect == Dialect.POSTGRES:
            pg_parts = [
                f"CAST({e} AS TEXT)" if e.strip().startswith("'") and e.strip().endswith("'") else e
                for e in group_exprs
            ]
            return f"ROW({', '.join(pg_parts)})::TEXT"
        elif self.dialect == Dialect.MYSQL:
            parts = [
                f"IF({e} IS NULL, 'N', CONCAT('V', LENGTH(CAST({e} AS CHAR)), ':', CAST({e} AS CHAR)))"
                for e in group_exprs
            ]
            return f"CONCAT_WS('|', {', '.join(parts)})"
        return " || '|' || ".join(f"quote({e})" for e in group_exprs)

    def _group_by_exprs_for_query(
        self, group_exprs: Sequence[str], include_canonical_pk: bool = True
    ) -> List[str]:
        if (
            include_canonical_pk
            and self.dialect == Dialect.MYSQL
            and group_exprs
            and any(
                not re.match(r"^(?:[A-Za-z_][A-Za-z0-9_.]*|'[^']*')$", g.strip())
                for g in group_exprs
            )
        ):
            return [self._canonical_group_pk_expr(group_exprs), *group_exprs]
        return list(group_exprs)

    def _group_col_alias(self, g_expr: str) -> str:
        for it in self.view.select_items:
            if not it.is_aggregate and (
                it.raw_expr.strip().lower() == g_expr.strip().lower()
                or (
                    it.source_col
                    and self._is_simple_col_ref(g_expr)
                    and it.source_col.lower() == _unquote_ident(g_expr).lower()
                )
            ):
                return it.alias
        if self._is_simple_col_ref(g_expr):
            if "." in g_expr.strip():
                return f"__mv_grp_{_sanitize_ident(g_expr)}"
            return f"__mv_grp_{_unquote_ident(g_expr)}"
        return f"__mv_grp_{_sanitize_ident(g_expr)}"

    def _has_nullable_group_col(self) -> bool:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return False
        if len(grp_exprs) > 1 and self.dialect == Dialect.MYSQL:
            return True
        for g in grp_exprs:
            c_def = self._lookup_col_def(g)
            if c_def is None or not c_def.not_null:
                return True
        return False

    def _tier1_pk_cols(self) -> List[str]:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return ["__mv_global_id"]
        if self._has_nullable_group_col():
            return ["__mv_pk"]
        return [self._group_col_alias(g) for g in grp_exprs]

    def _rewrite_expr_for_record(self, expr: str, record_prefix: Optional[str]) -> str:
        if not record_prefix:
            return expr
        base_t = self.view.base_table
        base_alias = self.view.joins[0].alias if self.view.joins else base_t
        col_names_lower = (
            {c.lower(): c for c in self.primary_schema.columns}
            if self.primary_schema
            else {}
        )
        tokens = re.split(r"('(?:''|[^'])*')", expr)
        out: List[str] = []
        for idx_t, tok in enumerate(tokens):
            if idx_t % 2 == 1:
                out.append(tok)
                continue
            mod = tok
            # Replace qualified references `base_t.col` or `base_alias.col`
            for qual in {base_t, base_alias}:
                if qual:
                    mod = re.sub(
                        rf"\b{re.escape(qual)}\.([A-Za-z_][A-Za-z0-9_]*)\b",
                        rf"{record_prefix}.\1",
                        mod,
                        flags=re.IGNORECASE,
                    )

            def _repl_unqual(m_id: re.Match[str]) -> str:
                word = m_id.group(1)
                w_low = word.lower()
                if w_low in col_names_lower:
                    return f"{record_prefix}.{col_names_lower[w_low]}"
                return word

            mod = re.sub(
                r"(?<![.\w])\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*[.(])",
                _repl_unqual,
                mod,
            )
            out.append(mod)
        return "".join(out)

    def _referenced_columns_for_table(self, table_name: str) -> List[str]:
        ts = self._schemas_lower.get(table_name.lower())
        if not ts:
            return []
        no_str = re.sub(r"'(?:''|[^'])*'", "''", self.view.raw_select_sql)
        if re.search(r"(?:SELECT\s+\*|,\s*\*|\.\*)", no_str, flags=re.IGNORECASE):
            return list(ts.columns.keys())
        found: List[str] = []
        for c_name in ts.columns:
            if re.search(rf"\b{re.escape(c_name)}\b", no_str, flags=re.IGNORECASE):
                found.append(c_name)
        return found if found else list(ts.columns.keys())

    def _build_readonly_view_and_triggers(
        self,
        view_name: str,
        mv_table: str,
        user_columns: Sequence[str],
        select_exprs_override: Optional[Sequence[str]] = None,
        extra_where_filter: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        db_view = self.db_view_name
        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        proj_csv = (
            ", ".join(select_exprs_override)
            if select_exprs_override
            else ", ".join(user_columns)
        )

        if self.dialect == Dialect.MYSQL:
            where_cond = (
                f"({extra_where_filter}) AND EXISTS (SELECT 1)"
                if extra_where_filter
                else "EXISTS (SELECT 1)"
            )
            reader_view_ddl = (
                f"CREATE ALGORITHM = TEMPTABLE VIEW {db_view} AS\n"
                f"SELECT {proj_csv}\n"
                f"FROM {mv_table}\n"
                f"WHERE {where_cond};"
            )
            return reader_view_ddl, [], [], []

        where_sql = f"\nWHERE {extra_where_filter}" if extra_where_filter else ""
        reader_view_ddl = (
            f"CREATE VIEW {db_view} AS\n"
            f"SELECT {proj_csv}\n"
            f"FROM {mv_table}{where_sql};"
        )

        if self.dialect == Dialect.SQLITE:
            ro_trigs: List[str] = []
            ro_names: List[str] = []
            for op, suf in (("INSERT", "ins"), ("UPDATE", "upd"), ("DELETE", "del")):
                t_name = f"{p}trg_{log_vname}__ro_{suf}"
                ro_names.append(t_name)
                ro_trigs.append(
                    f"CREATE TRIGGER {t_name}\n"
                    f"INSTEAD OF {op} ON {db_view}\n"
                    f"BEGIN\n"
                    f"    SELECT RAISE(ABORT, 'Materialized view {db_view} is read-only and maintained automatically');\n"
                    f"END;"
                )
            return reader_view_ddl, ro_trigs, ro_names, []

        # PostgreSQL: INSTEAD OF trigger on view
        fn_ro = f"{p}fn_readonly_{log_vname}"
        t_ro = f"{p}trg_{log_vname}__readonly"
        fn_ddl = (
            f"CREATE OR REPLACE FUNCTION {fn_ro}()\n"
            f"RETURNS TRIGGER AS $$\n"
            f"BEGIN\n"
            f"    RAISE EXCEPTION 'Materialized view % is read-only and maintained automatically', TG_TABLE_NAME;\n"
            f"END;\n"
            f"$$ LANGUAGE plpgsql;"
        )
        t_ddl = (
            f"CREATE TRIGGER {t_ro}\n"
            f"INSTEAD OF INSERT OR UPDATE OR DELETE ON {db_view}\n"
            f"FOR EACH ROW EXECUTE FUNCTION {fn_ro}();"
        )
        return reader_view_ddl, [fn_ddl, t_ddl], [t_ro], [fn_ro]

    def _build_dialect_incremental_populate_sqls(self, raw_plan: MaterializedViewPlan) -> List[str]:
        spec = raw_plan.incremental_backfill_spec
        b_size = self.batch_size
        if spec is None:
            return list(raw_plan.initial_populate_sqls)

        # Case 1: PK / rowid range folding (`pk_range_fold`)
        if spec.mode == "pk_range_fold" and spec.pk_column and spec.chunk_sql_templates:
            base_t = raw_plan.view_spec.base_table
            pk_col = spec.pk_column
            tmpl = spec.chunk_sql_templates[0].strip().rstrip(";")
            ph = "?" if self.dialect == Dialect.SQLITE else "%s"

            if self.dialect == Dialect.SQLITE:
                # Replace `(pk_col > ? AND pk_col <= ?)` with recursive CTE batch join
                cond_pat = f"({pk_col} > {ph} AND {pk_col} <= {ph})"
                joined_tmpl = tmpl.replace(
                    cond_pat,
                    f"({base_t}.{pk_col} > __mv_b.lo AND {base_t}.{pk_col} <= __mv_b.hi)",
                )
                joined_tmpl = re.sub(
                    rf"\bFROM\s+{re.escape(base_t)}\b",
                    f"FROM __mv_batches AS __mv_b JOIN {base_t} ON ({base_t}.{pk_col} > __mv_b.lo AND {base_t}.{pk_col} <= __mv_b.hi)",
                    joined_tmpl,
                    count=1,
                    flags=re.IGNORECASE,
                )
                out_stmts = [
                    f"WITH RECURSIVE __mv_bounds AS (\n"
                    f"    SELECT MIN({pk_col}) AS min_id, MAX({pk_col}) AS max_id FROM {base_t}\n"
                    f"),\n"
                    f"__mv_batches(lo, hi, max_id) AS (\n"
                    f"    SELECT min_id - 1, MIN(max_id, min_id - 1 + {b_size}), max_id\n"
                    f"    FROM __mv_bounds\n"
                    f"    WHERE min_id IS NOT NULL AND max_id IS NOT NULL\n"
                    f"    UNION ALL\n"
                    f"    SELECT hi, MIN(max_id, hi + {b_size}), max_id\n"
                    f"    FROM __mv_batches\n"
                    f"    WHERE hi < max_id\n"
                    f")\n"
                    f"{joined_tmpl};"
                ]
                if not raw_plan.view_spec.group_by_cols and not raw_plan.view_spec.has_top_level_distinct:
                    # Global aggregate: ensure empty base table or all-filtered table still has its 1 global row
                    for pop_s in raw_plan.initial_populate_sqls:
                        p_clean = pop_s.strip().rstrip(";")
                        out_stmts.append(
                            f"{p_clean}\nWHERE NOT EXISTS (SELECT 1 FROM {raw_plan.backing_table_name});"
                        )
                return out_stmts

            elif self.dialect == Dialect.POSTGRES:
                pg_chunk = tmpl.replace(
                    f"({pk_col} > %s AND {pk_col} <= %s)",
                    f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                )
                if not raw_plan.view_spec.group_by_cols and not raw_plan.view_spec.has_top_level_distinct:
                    pg_chunk = re.sub(
                        r"\nON CONFLICT\b",
                        "\nHAVING COUNT(*) > 0\nON CONFLICT",
                        pg_chunk,
                        count=1,
                    )
                fallback_global = ""
                if not raw_plan.view_spec.group_by_cols and not raw_plan.view_spec.has_top_level_distinct:
                    for pop_s in raw_plan.initial_populate_sqls:
                        p_clean = pop_s.strip().rstrip(";")
                        fallback_global += (
                            f"\n    IF NOT EXISTS (SELECT 1 FROM {raw_plan.backing_table_name}) THEN\n"
                            f"        {p_clean};\n"
                            f"    END IF;"
                        )
                return [
                    f"DO $$\n"
                    f"DECLARE\n"
                    f"    __mv_min_pk BIGINT;\n"
                    f"    __mv_max_pk BIGINT;\n"
                    f"    __mv_cur_pk BIGINT;\n"
                    f"    __mv_top_pk BIGINT;\n"
                    f"    __mv_batch_size BIGINT := {b_size};\n"
                    f"BEGIN\n"
                    f"    SELECT MIN({pk_col}), MAX({pk_col}) INTO __mv_min_pk, __mv_max_pk FROM {base_t};\n"
                    f"    IF __mv_min_pk IS NOT NULL AND __mv_max_pk IS NOT NULL THEN\n"
                    f"        __mv_cur_pk := __mv_min_pk - 1;\n"
                    f"        WHILE __mv_cur_pk < __mv_max_pk LOOP\n"
                    f"            __mv_top_pk := LEAST(__mv_max_pk, __mv_cur_pk + __mv_batch_size);\n"
                    f"            {pg_chunk};\n"
                    f"            __mv_cur_pk := __mv_top_pk;\n"
                    f"        END LOOP;\n"
                    f"    END IF;{fallback_global}\n"
                    f"END $$;"
                ]

            elif self.dialect == Dialect.MYSQL:
                my_chunk = tmpl.replace(
                    f"({pk_col} > %s AND {pk_col} <= %s)",
                    f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                )
                if not raw_plan.view_spec.group_by_cols and not raw_plan.view_spec.has_top_level_distinct:
                    my_chunk = re.sub(
                        r"\nON DUPLICATE KEY UPDATE\b",
                        "\nHAVING COUNT(*) > 0\nON DUPLICATE KEY UPDATE",
                        my_chunk,
                        count=1,
                    )
                fallback_global = ""
                if not raw_plan.view_spec.group_by_cols and not raw_plan.view_spec.has_top_level_distinct:
                    for pop_s in raw_plan.initial_populate_sqls:
                        p_clean = pop_s.strip().rstrip(";")
                        fallback_global += (
                            f"\n    IF NOT EXISTS (SELECT 1 FROM {raw_plan.backing_table_name}) THEN\n"
                            f"        {p_clean};\n"
                            f"    END IF;"
                        )
                proc_name = f"{self.object_prefix}proc_backfill_{self.logical_view_name}"
                return [
                    f"DROP PROCEDURE IF EXISTS {proc_name};",
                    (
                        f"CREATE PROCEDURE {proc_name}()\n"
                        f"BEGIN\n"
                        f"    DECLARE __mv_min_pk BIGINT;\n"
                        f"    DECLARE __mv_max_pk BIGINT;\n"
                        f"    DECLARE __mv_cur_pk BIGINT;\n"
                        f"    DECLARE __mv_top_pk BIGINT;\n"
                        f"    DECLARE __mv_batch_size BIGINT DEFAULT {b_size};\n"
                        f"    SELECT MIN({pk_col}), MAX({pk_col}) INTO __mv_min_pk, __mv_max_pk FROM {base_t};\n"
                        f"    IF __mv_min_pk IS NOT NULL AND __mv_max_pk IS NOT NULL THEN\n"
                        f"        SET __mv_cur_pk = __mv_min_pk - 1;\n"
                        f"        WHILE __mv_cur_pk < __mv_max_pk DO\n"
                        f"            SET __mv_top_pk = LEAST(__mv_max_pk, __mv_cur_pk + __mv_batch_size);\n"
                        f"            {my_chunk};\n"
                        f"            SET __mv_cur_pk = __mv_top_pk;\n"
                        f"        END WHILE;\n"
                        f"    END IF;{fallback_global}\n"
                        f"END;"
                    ),
                    f"CALL {proc_name}();",
                    f"DROP PROCEDURE IF EXISTS {proc_name};",
                ]

        # Case 2: Group-batch or Offset-batch over `raw_plan.initial_populate_sqls`
        inc_stmts: List[str] = []
        main_pk_cols: List[str] = []
        if raw_plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA:
            if self._alg_mode in ("single", "join", "union_all_agg"):
                main_pk_cols = self._tier1_pk_cols()
            elif self._alg_mode in ("setop", "projection"):
                main_pk_cols = ["__mv_pk"]

        for idx_s, pop_sql in enumerate(raw_plan.initial_populate_sqls):
            clean_pop = pop_sql.strip().rstrip(";")
            m_ins = re.match(
                r"^INSERT\s+INTO\s+([A-Za-z0-9_.\"`\[\]]+)\s*\(([^)]+)\)\s*(.*)$",
                clean_pop,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m_ins:
                inc_stmts.append(pop_sql)
                continue
            target_tbl = m_ins.group(1).strip()
            cols = [c.strip() for c in m_ins.group(2).split(",") if c.strip()]
            select_body = m_ins.group(3).strip()

            # Ensure top-level SELECT expressions in `select_body` are aliased to `cols`
            m_sel_top = re.match(r"^SELECT\s+(.*)$", select_body, flags=re.IGNORECASE | re.DOTALL)
            if m_sel_top and not _find_top_level_keyword(select_body, r"\b(?:UNION|INTERSECT|EXCEPT)\b"):
                after_sel = m_sel_top.group(1)
                m_from_top = _find_top_level_keyword(after_sel, r"\bFROM\b")
                if m_from_top:
                    sel_exprs_raw = _split_top_level(after_sel[: m_from_top.start()], ",")
                    from_tail = after_sel[m_from_top.start() :]
                    if len(sel_exprs_raw) == len(cols):
                        aliased_list: List[str] = []
                        for c_name, raw_e in zip(cols, sel_exprs_raw):
                            re_s = raw_e.strip()
                            if re.search(rf"\bAS\s+{re.escape(c_name)}\s*$", re_s, flags=re.IGNORECASE):
                                aliased_list.append(re_s)
                            else:
                                aliased_list.append(f"{re_s} AS {c_name}")
                        aliased_inner = f"SELECT {', '.join(aliased_list)} {from_tail}"
                    else:
                        aliased_inner = select_body
                else:
                    aliased_inner = select_body
            else:
                aliased_inner = select_body

            is_aux_dist = "__val" in cols and "__ref_cnt" in cols
            tbl_pks = (
                ([*main_pk_cols, "__val"] if is_aux_dist else main_pk_cols)
                if (is_aux_dist or target_tbl == raw_plan.backing_table_name)
                else []
            )
            tbl_pks = [c for c in tbl_pks if c in cols]

            if self.dialect == Dialect.SQLITE:
                upsert_tail = ""
                if tbl_pks:
                    upd_cols = [f"{c} = excluded.{c}" for c in cols if c not in tbl_pks] or [
                        f"{tbl_pks[0]} = excluded.{tbl_pks[0]}"
                    ]
                    upsert_tail = f"\nON CONFLICT ({', '.join(tbl_pks)}) DO UPDATE SET {', '.join(upd_cols)}"
                inc_stmts.append(
                    f"WITH __mv_batched_src AS (\n"
                    f"    SELECT\n"
                    f"        ((ROW_NUMBER() OVER () - 1) / {b_size}) AS __mv_batch_id,\n"
                    f"        __mv_raw_src.*\n"
                    f"    FROM (\n"
                    f"        {aliased_inner}\n"
                    f"    ) AS __mv_raw_src\n"
                    f"    ORDER BY 1\n"
                    f")\n"
                    f"INSERT INTO {target_tbl} ({', '.join(cols)})\n"
                    f"SELECT {', '.join(cols)}\n"
                    f"FROM __mv_batched_src\n"
                    f"WHERE TRUE{upsert_tail};"
                )
            elif self.dialect == Dialect.POSTGRES:
                upsert_tail = ""
                if tbl_pks:
                    upd_cols = [f"{c} = excluded.{c}" for c in cols if c not in tbl_pks] or [
                        f"{tbl_pks[0]} = excluded.{tbl_pks[0]}"
                    ]
                    upsert_tail = f"\n        ON CONFLICT ({', '.join(tbl_pks)}) DO UPDATE SET {', '.join(upd_cols)}"
                inc_stmts.append(
                    f"DO $$\n"
                    f"DECLARE\n"
                    f"    __mv_offset BIGINT := 0;\n"
                    f"    __mv_batch_size BIGINT := {b_size};\n"
                    f"    __mv_rows BIGINT := 0;\n"
                    f"BEGIN\n"
                    f"    LOOP\n"
                    f"        INSERT INTO {target_tbl} ({', '.join(cols)})\n"
                    f"        SELECT {', '.join(cols)}\n"
                    f"        FROM (\n"
                    f"            {aliased_inner}\n"
                    f"        ) AS __mv_init\n"
                    f"        LIMIT __mv_batch_size OFFSET __mv_offset{upsert_tail};\n"
                    f"        GET DIAGNOSTICS __mv_rows = ROW_COUNT;\n"
                    f"        EXIT WHEN __mv_rows < __mv_batch_size;\n"
                    f"        __mv_offset := __mv_offset + __mv_rows;\n"
                    f"    END LOOP;\n"
                    f"END $$;"
                )
            elif self.dialect == Dialect.MYSQL:
                upsert_tail = ""
                if tbl_pks:
                    upd_cols = [f"{c} = VALUES({c})" for c in cols if c not in tbl_pks] or [
                        f"{tbl_pks[0]} = VALUES({tbl_pks[0]})"
                    ]
                    upsert_tail = f"\n        ON DUPLICATE KEY UPDATE {', '.join(upd_cols)}"
                proc_name = f"{self.object_prefix}proc_backfill_{self.logical_view_name}_{idx_s}"
                inc_stmts.extend([
                    f"DROP PROCEDURE IF EXISTS {proc_name};",
                    (
                        f"CREATE PROCEDURE {proc_name}()\n"
                        f"BEGIN\n"
                        f"    DECLARE __mv_offset BIGINT DEFAULT 0;\n"
                        f"    DECLARE __mv_batch_size BIGINT DEFAULT {b_size};\n"
                        f"    DECLARE __mv_rows BIGINT DEFAULT 0;\n"
                        f"    REPEAT\n"
                        f"        INSERT INTO {target_tbl} ({', '.join(cols)})\n"
                        f"        SELECT {', '.join(cols)}\n"
                        f"        FROM (\n"
                        f"            {aliased_inner}\n"
                        f"        ) AS __mv_init\n"
                        f"        LIMIT __mv_batch_size OFFSET __mv_offset{upsert_tail};\n"
                        f"        SET __mv_rows = ROW_COUNT();\n"
                        f"        SET __mv_offset = __mv_offset + __mv_batch_size;\n"
                        f"    UNTIL __mv_rows <= 0 END REPEAT;\n"
                        f"END;"
                    ),
                    f"CALL {proc_name}();",
                    f"DROP PROCEDURE IF EXISTS {proc_name};",
                ])
        return inc_stmts

    def _finalize_plan_backfill(self, raw_plan: MaterializedViewPlan) -> MaterializedViewPlan:
        full_sqls = list(raw_plan.initial_populate_sqls)
        inc_sqls = self._build_dialect_incremental_populate_sqls(raw_plan)
        active_sqls = inc_sqls if self.incremental_backfill else full_sqls
        return MaterializedViewPlan(
            dialect=raw_plan.dialect,
            strategy=raw_plan.strategy,
            view_spec=raw_plan.view_spec,
            backing_table_name=raw_plan.backing_table_name,
            backing_table_ddl=raw_plan.backing_table_ddl,
            auxiliary_table_ddls=raw_plan.auxiliary_table_ddls,
            index_ddls=raw_plan.index_ddls,
            initial_populate_sqls=active_sqls,
            trigger_ddls=raw_plan.trigger_ddls,
            readonly_trigger_ddls=raw_plan.readonly_trigger_ddls,
            user_columns=raw_plan.user_columns,
            reader_view_ddl=raw_plan.reader_view_ddl,
            incremental_backfill_spec=raw_plan.incremental_backfill_spec,
            auxiliary_table_names=raw_plan.auxiliary_table_names,
            trigger_names=raw_plan.trigger_names,
            function_names=raw_plan.function_names,
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
        return self._finalize_plan_backfill(raw)

    def compile(self) -> MaterializedViewPlan:
        first_plan = self._compile_single_plan()
        if not self.raw_view.additional_views:
            return first_plan

        sub_plans: List[MaterializedViewPlan] = [first_plan]
        for extra_v in self.raw_view.additional_views:
            sub_comp = IncrementalViewCompiler(
                schemas=self.schemas,
                view_spec=extra_v,
                dialect=self.dialect,
                pg_use_transition_tables=self.pg_use_transition_tables,
                ignore_limitations=self.ignore_limitations,
                object_prefix=self.object_prefix,
                incremental_backfill=self.incremental_backfill,
                batch_size=self.batch_size,
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
        )


    # =========================================================================
    # TIER 1A: Single-Table O(1) Algebraic Delta Maintenance
    # =========================================================================

    def _effective_agg_value_expr(
        self, item: SelectItem, record_prefix: Optional[str] = None
    ) -> str:
        arg_e = item.arg_expr or "1"
        if arg_e.strip() == "*":
            arg_e = "1"
        rewritten_arg = (
            self._rewrite_expr_for_record(arg_e, record_prefix)
            if record_prefix
            else arg_e
        )
        if item.filter_where:
            rewritten_f = (
                self._rewrite_expr_for_record(item.filter_where, record_prefix)
                if record_prefix
                else item.filter_where
            )
            return f"CASE WHEN ({rewritten_f}) THEN ({rewritten_arg}) ELSE NULL END"
        return rewritten_arg

    def _sql_backfill_agg_expr(self, item: SelectItem) -> str:
        eff = self._effective_agg_value_expr(item, record_prefix=None)
        dist = "DISTINCT " if item.is_distinct else ""
        if item.agg_func == AggFunc.COUNT_STAR:
            if item.filter_where:
                return f"COUNT({eff})"
            return "COUNT(*)"
        if item.agg_func == AggFunc.COUNT_COL:
            return f"COUNT({dist}{eff})"
        if item.agg_func == AggFunc.SUM:
            return f"SUM({dist}{eff})"
        if item.agg_func == AggFunc.TOTAL:
            if self.dialect == Dialect.SQLITE:
                return f"TOTAL({dist}{eff})"
            return f"COALESCE(SUM(1.0 * ({dist}{eff})), 0.0)"
        if item.agg_func == AggFunc.AVG:
            if self.dialect == Dialect.SQLITE:
                return f"AVG({dist}{eff})"
            return f"AVG(1.0 * ({dist}{eff}))" if not item.is_distinct else f"AVG({dist}{eff})"
        if item.agg_func == AggFunc.MIN:
            return f"MIN({eff})"
        if item.agg_func == AggFunc.MAX:
            return f"MAX({eff})"
        if item.agg_func in (
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
        ):
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                fn_s = item.agg_func.value
                return f"{fn_s}({eff})"
            cnt_e = f"COUNT({eff})"
            sum_e = f"SUM(1.0 * ({eff}))"
            sum2_e = f"SUM(1.0 * ({eff}) * ({eff}))"
            return self._variance_formula_sql(
                new_cnt=cnt_e,
                new_sum=sum_e,
                new_sum2=sum2_e,
                is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
            )
        if item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR):
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                if self.dialect == Dialect.MYSQL:
                    return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL ELSE {item.agg_func.value}({eff}) END)"
                return f"{item.agg_func.value}({eff})"
            return f"SUM({eff})"
        if item.agg_func == AggFunc.BOOL_AND:
            if self.dialect == Dialect.POSTGRES:
                return f"BOOL_AND({eff})"
            return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL WHEN MIN(CASE WHEN ({eff}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
        if item.agg_func == AggFunc.BOOL_OR:
            if self.dialect == Dialect.POSTGRES:
                return f"BOOL_OR({eff})"
            return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL WHEN MAX(CASE WHEN ({eff}) THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END)"
        return item.raw_expr

    def _compile_tier1_algebraic(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs
        has_nullable_grp = self._has_nullable_group_col()

        count_star_alias: Optional[str] = None
        for item in work_items:
            if (
                item.is_aggregate
                and item.agg_func == AggFunc.COUNT_STAR
                and not item.filter_where
            ):
                count_star_alias = item.alias
                break
        row_count_col = count_star_alias or "__mv_row_count"

        col_defs_sql: List[str] = []
        user_columns: List[str] = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"

        if not grp_exprs:
            col_defs_sql.append("__mv_global_id INTEGER NOT NULL PRIMARY KEY")
        elif has_nullable_grp:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden_grp: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_grp:
                seen_hidden_grp.add(mv_col.lower())
                c_def = self._lookup_col_def(g_expr)
                if c_def:
                    dt = self._normalize_dialect_type(c_def.data_type)
                else:
                    fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                    dt = self._infer_expr_storage_type(fake_it)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if not has_nullable_grp else ""
                col_defs_sql.append(f"{mv_col} {dt}{nn}")

        def _expr_key(item: SelectItem) -> str:
            eff = self._effective_agg_value_expr(item, record_prefix=None).strip().lower()
            return f"{'dist:' if item.is_distinct else ''}{eff}"

        col_nonnull_count_map: Dict[str, str] = {}
        col_sum_map: Dict[str, str] = {}
        col_sum2_map: Dict[str, str] = {}
        col_true_cnt_map: Dict[str, str] = {}
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]] = {}

        for item in work_items:
            if not item.is_aggregate:
                continue
            ekey = _expr_key(item)
            if item.agg_func == AggFunc.COUNT_COL:
                col_nonnull_count_map[ekey] = item.alias
            elif item.agg_func == AggFunc.SUM:
                col_sum_map[ekey] = item.alias
            if item.is_distinct and ekey not in distinct_aux_tables:
                p = self.object_prefix if self.object_prefix else "__matview_"
                aux_tbl = f"{p}dist_{self.logical_view_name}__{_sanitize_ident(item.alias)}"
                c_arg = self._lookup_col_def(item.arg_expr)
                val_dt = (
                    self._normalize_dialect_type(c_arg.data_type)
                    if c_arg
                    else ("VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT")
                )
                if self.dialect == Dialect.MYSQL and val_dt.upper() == "TEXT":
                    val_dt = "VARCHAR(255)"
                distinct_aux_tables[ekey] = (aux_tbl, item, val_dt)

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                c_def = self._lookup_col_def(item.raw_expr)
                dt = (
                    self._normalize_dialect_type(c_def.data_type)
                    if c_def
                    else self._infer_expr_storage_type(item)
                )
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if (grp_exprs and not has_nullable_grp) else ""
                col_defs_sql.append(f"{item.alias} {dt}{nn}")
            else:
                storage_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{item.alias} {storage_type}")

        if count_star_alias is None:
            col_defs_sql.append(f"{row_count_col} {self._int_type()} NOT NULL DEFAULT 0")

        aux_idx = 0
        for item in work_items:
            if not item.is_aggregate or item.agg_func is None:
                continue
            ekey = _expr_key(item)
            safe_id = _sanitize_ident(item.alias)
            if item.agg_func in (
                AggFunc.SUM,
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BIT_XOR,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            ) or (item.agg_func == AggFunc.COUNT_STAR and item.filter_where):
                if item.agg_func == AggFunc.COUNT_STAR and item.filter_where:
                    col_nonnull_count_map[ekey] = item.alias
                elif ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    cnt_col = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = cnt_col
                    col_defs_sql.append(f"{cnt_col} {self._int_type()} NOT NULL DEFAULT 0")

            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    sum_col = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = sum_col
                    col_defs_sql.append(f"{sum_col} {self._float_type()}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    sum2_col = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._float_type()}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        pk_cols = self._tier1_pk_cols()
        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(pk_cols)})")

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        aux_ddls: List[str] = []
        aux_names: List[str] = []
        for ekey, (aux_tbl, rep_item, val_dt) in distinct_aux_tables.items():
            aux_names.append(aux_tbl)
            aux_col_defs: List[str] = []
            if not grp_exprs:
                aux_col_defs.append("__mv_global_id INTEGER NOT NULL")
                aux_pk = ["__mv_global_id", "__val"]
            elif has_nullable_grp:
                aux_col_defs.append(f"__mv_pk {pk_type} NOT NULL")
                aux_pk = ["__mv_pk", "__val"]
            else:
                aux_pk = []
                for g_expr in grp_exprs:
                    mc = self._group_col_alias(g_expr)
                    c_def = self._lookup_col_def(g_expr)
                    dt = (
                        self._normalize_dialect_type(c_def.data_type)
                        if c_def
                        else ("VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT")
                    )
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                        dt = "VARCHAR(255)"
                    aux_col_defs.append(f"{mc} {dt} NOT NULL")
                    aux_pk.append(mc)
                aux_pk.append("__val")
            aux_col_defs.append(f"__val {val_dt} NOT NULL")
            aux_col_defs.append(f"__ref_cnt {self._int_type()} NOT NULL DEFAULT 0")
            aux_col_defs.append(f"PRIMARY KEY ({', '.join(aux_pk)})")
            aux_ddls.append(
                f"CREATE TABLE {aux_tbl} (\n    " + ",\n    ".join(aux_col_defs) + "\n);"
            )

        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col
        }
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
        )

        reader_select_override = [
            self._compound_reader_exprs.get(u_col, u_col) for u_col in user_columns
        ]
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                select_exprs_override=reader_select_override,
                extra_where_filter=self._rewritten_having_clause,
            )
        )

        initial_populate_sqls, inc_spec = self._build_initial_populate_and_inc_spec_tier1(
            mv_table=mv_table,
            count_star_alias=count_star_alias,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            distinct_aux_tables=distinct_aux_tables,
            expr_key_fn=_expr_key,
        )

        trigger_ddls, trig_names, fn_names = self._build_tier1_triggers(
            view_name=view_name,
            mv_table=mv_table,
            base_t=self.view.base_table,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            distinct_aux_tables=distinct_aux_tables,
            expr_key_fn=_expr_key,
            initial_populate_sqls=initial_populate_sqls,
        )

        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=aux_ddls,
            index_ddls=index_ddls,
            initial_populate_sqls=initial_populate_sqls,
            trigger_ddls=trigger_ddls,
            readonly_trigger_ddls=ro_trigs,
            user_columns=user_columns,
            reader_view_ddl=reader_view_ddl,
            incremental_backfill_spec=inc_spec,
            auxiliary_table_names=aux_names,
            trigger_names=ro_trig_names + trig_names,
            function_names=ro_fn_names + fn_names,
            limitations=self.limitations,
            object_prefix=self.object_prefix,
            db_view_name=self.db_view_name,
            index_names=index_names,
            user_indexes=list(self.user_indexes),
        )

    def _build_initial_populate_and_inc_spec_tier1(
        self,
        mv_table: str,
        count_star_alias: Optional[str],
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
    ) -> Tuple[List[str], IncrementalBackfillSpec]:
        stmts: List[str] = []
        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            eff = self._effective_agg_value_expr(rep_item, record_prefix=None)
            dist_cols: List[str] = []
            dist_sel: List[str] = []
            dist_grp: List[str] = []
            if not grp_exprs:
                dist_cols.append("__mv_global_id")
                dist_sel.append("1")
            elif has_nullable_grp:
                dist_cols.append("__mv_pk")
                dist_sel.append(self._canonical_group_pk_expr(grp_exprs))
                dist_grp.extend(self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=True))
            else:
                for g_expr in grp_exprs:
                    dist_cols.append(self._group_col_alias(g_expr))
                    dist_sel.append(g_expr)
                    dist_grp.append(g_expr)
            dist_cols.extend(["__val", "__ref_cnt"])
            dist_sel.extend([eff, "COUNT(*)"])
            dist_grp.append(eff)

            w_parts: List[str] = []
            if self.view.where_clause:
                w_parts.append(f"({self.view.where_clause})")
            w_parts.append(f"({eff}) IS NOT NULL")
            where_sql = f"\nWHERE {' AND '.join(w_parts)}"
            group_sql = f"\nGROUP BY {', '.join(dist_grp)}"
            stmts.append(
                f"INSERT INTO {aux_tbl} ({', '.join(dist_cols)})\n"
                f"SELECT {', '.join(dist_sel)}\n"
                f"FROM {self.view.base_table}{where_sql}{group_sql};"
            )

        target_cols: List[str] = []
        select_exprs: List[str] = []

        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(grp_exprs))

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                target_cols.append(mv_col)
                select_exprs.append(g_expr)

        for item in work_items:
            target_cols.append(item.alias)
            if not item.is_aggregate:
                select_exprs.append(item.raw_expr)
            else:
                select_exprs.append(self._sql_backfill_agg_expr(item))

        if count_star_alias is None:
            target_cols.append(row_count_col)
            select_exprs.append("COUNT(*)")

        already_in_target = set(target_cols)
        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }

        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                dist = "DISTINCT " if it.is_distinct else ""
                target_cols.append(cnt_col)
                select_exprs.append(f"COUNT({dist}{eff})")
                already_in_target.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            if sum_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                dist = "DISTINCT " if it.is_distinct else ""
                target_cols.append(sum_col)
                select_exprs.append(f"SUM({dist}{eff})")
                already_in_target.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(sum2_col)
                select_exprs.append(f"SUM(1.0 * ({eff}) * ({eff}))")
                already_in_target.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(tcnt_col)
                select_exprs.append(
                    f"COALESCE(SUM(CASE WHEN ({eff}) IS NOT NULL AND ({eff}) THEN 1 ELSE 0 END), 0)"
                )
                already_in_target.add(tcnt_col)

        where_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        if grp_exprs:
            grp_for_q = self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=has_nullable_grp)
            group_sql = f"\nGROUP BY {', '.join(grp_for_q)}"
            having_sql = ""
        else:
            group_sql = ""
            having_sql = ""

        main_backfill_sql = (
            f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM {self.view.base_table}{where_sql}{group_sql}{having_sql};"
        )
        stmts.append(main_backfill_sql)

        holistic_funcs = {
            AggFunc.GROUP_CONCAT,
            AggFunc.STRING_AGG,
            AggFunc.JSON_GROUP_ARRAY,
            AggFunc.JSON_GROUP_OBJECT,
            AggFunc.ARRAY_AGG,
        }
        has_holistic_or_distinct = bool(distinct_aux_tables) or any(
            it.is_aggregate and it.agg_func in holistic_funcs for it in work_items
        )

        has_int_pk = bool(
            self.primary_schema
            and any(c.primary_key and "INT" in c.data_type.upper() for c in self.primary_schema.columns.values())
        )
        if not has_holistic_or_distinct and (self.dialect == Dialect.SQLITE or has_int_pk):
            pk_col = "rowid"
            if self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    if c_def.primary_key and "INT" in c_def.data_type.upper():
                        pk_col = c_def.name
                        break
            ph = "?" if self.dialect == Dialect.SQLITE else "%s"
            chunk_where = (
                f"({self.view.where_clause}) AND ({pk_col} > {ph} AND {pk_col} <= {ph})"
                if self.view.where_clause
                else f"({pk_col} > {ph} AND {pk_col} <= {ph})"
            )
            merge_assignments = self._build_tier1_chunk_merge_assignments(
                mv_table=mv_table,
                row_count_col=row_count_col,
                col_nonnull_count_map=col_nonnull_count_map,
                col_sum_map=col_sum_map,
                col_sum2_map=col_sum2_map,
                col_true_cnt_map=col_true_cnt_map,
                expr_key_fn=expr_key_fn,
                use_mysql_values=(self.dialect == Dialect.MYSQL),
            )
            if self.dialect == Dialect.MYSQL:
                chunk_upsert_sql = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {self.view.base_table}\n"
                    f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f"ON DUPLICATE KEY UPDATE\n"
                    f"    {', '.join(merge_assignments)};"
                )
            else:
                chunk_upsert_sql = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {self.view.base_table}\n"
                    f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET\n"
                    f"    {', '.join(merge_assignments)};"
                )
            inc_spec = IncrementalBackfillSpec(
                mode="pk_range_fold",
                pk_column=pk_col,
                chunk_sql_templates=[chunk_upsert_sql],
            )
        elif grp_exprs:
            grp_expr = self._canonical_group_pk_expr(grp_exprs)
            w_grp = f" WHERE {self.view.where_clause}" if self.view.where_clause else ""
            dist_grp_query = (
                f"SELECT DISTINCT {grp_expr} AS __gkey FROM {self.view.base_table}{w_grp} ORDER BY __gkey"
            )
            chunk_templates: List[str] = []
            for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
                eff = self._effective_agg_value_expr(rep_item, record_prefix=None)
                dist_cols = [
                    pk_cols[0] if len(pk_cols) == 1 else ", ".join(pk_cols),
                    "__val",
                    "__ref_cnt",
                ]
                dist_sel = []
                dist_grp_list = []
                if has_nullable_grp:
                    dist_sel.append(grp_expr)
                    dist_grp_list.extend(self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=True))
                else:
                    for g_expr in grp_exprs:
                        dist_sel.append(g_expr)
                        dist_grp_list.append(g_expr)
                dist_sel.extend([eff, "COUNT(*)"])
                dist_grp_list.append(eff)
                w_p = [f"({eff}) IS NOT NULL", f"({grp_expr}) IN ({{placeholders}})"]
                if self.view.where_clause:
                    w_p.insert(0, f"({self.view.where_clause})")
                aux_conflict = (
                    "\nON DUPLICATE KEY UPDATE __ref_cnt = VALUES(__ref_cnt)"
                    if self.dialect == Dialect.MYSQL
                    else f"\nON CONFLICT ({', '.join(pk_cols)}, __val) DO UPDATE SET __ref_cnt = excluded.__ref_cnt"
                )
                chunk_templates.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(dist_cols)})\n"
                    f"SELECT {', '.join(dist_sel)}\n"
                    f"FROM {self.view.base_table}\n"
                    f"WHERE {' AND '.join(w_p)}\n"
                    f"GROUP BY {', '.join(dist_grp_list)}{aux_conflict};"
                )
            w_main = [f"({grp_expr}) IN ({{placeholders}})"]
            if self.view.where_clause:
                w_main.insert(0, f"({self.view.where_clause})")
            non_pk_target = [c for c in target_cols if c not in pk_cols] or list(pk_cols)
            main_conflict = (
                f"\nON DUPLICATE KEY UPDATE {', '.join(f'{c} = VALUES({c})' for c in non_pk_target)}"
                if self.dialect == Dialect.MYSQL
                else f"\nON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in non_pk_target)}"
            )
            chunk_templates.append(
                f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                f"SELECT {', '.join(select_exprs)}\n"
                f"FROM {self.view.base_table}\n"
                f"WHERE {' AND '.join(w_main)}{group_sql}{having_sql}{main_conflict};"
            )
            inc_spec = IncrementalBackfillSpec(
                mode="group_key_batch",
                distinct_group_query=dist_grp_query,
                chunk_sql_templates=chunk_templates,
            )
        else:
            inc_spec = IncrementalBackfillSpec(
                mode="offset_batch",
                chunk_sql_templates=stmts,
            )

        return stmts, inc_spec

    def _build_tier1_chunk_merge_assignments(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        expr_key_fn: Callable[[SelectItem], str],
        use_mysql_values: bool = False,
    ) -> List[str]:
        assignments: List[str] = []
        updated: Set[str] = set()
        work_items = self._working_select_items

        def _incoming(col: str) -> str:
            return f"VALUES({col})" if use_mysql_values else f"excluded.{col}"

        def _merged_sum(col: str) -> str:
            inc = _incoming(col)
            return (
                f"(CASE WHEN {inc} IS NULL THEN {mv_table}.{col} "
                f"WHEN {mv_table}.{col} IS NULL THEN {inc} "
                f"ELSE {mv_table}.{col} + {inc} END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = item.alias
            updated.add(c)
            ekey = expr_key_fn(item)
            inc_c = _incoming(c)
            if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                assignments.append(f"{c} = {mv_table}.{c} + {inc_c}")
            elif item.agg_func == AggFunc.SUM:
                assignments.append(f"{c} = {_merged_sum(c)}")
            elif item.agg_func == AggFunc.AVG:
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_sum = _merged_sum(sum_c)
                assignments.append(
                    f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL ELSE (1.0 * {new_sum}) / {new_cnt} END"
                )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                sum2_c = col_sum2_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_sum = _merged_sum(sum_c)
                new_sum2 = _merged_sum(sum2_c)
                formula = self._variance_formula_sql(
                    new_cnt=new_cnt,
                    new_sum=new_sum,
                    new_sum2=new_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                assignments.append(f"{c} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} < {mv_table}.{c} "
                    f"THEN {inc_c} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} > {mv_table}.{c} "
                    f"THEN {inc_c} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.BIT_AND:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE ({mv_table}.{c} & {inc_c}) END"
                )
            elif item.agg_func == AggFunc.BIT_OR:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE ({mv_table}.{c} | {inc_c}) END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                xor_e = self._bitwise_xor_sql(f"{mv_table}.{c}", f"({inc_c})")
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL THEN {inc_c} "
                    f"ELSE {xor_e} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_tcnt = f"({mv_table}.{tcnt_c} + {_incoming(tcnt_c)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                assignments.append(
                    f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                new_tcnt = f"({mv_table}.{tcnt_c} + {_incoming(tcnt_c)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                assignments.append(
                    f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )

        if row_count_col not in updated:
            assignments.append(f"{row_count_col} = {mv_table}.{row_count_col} + {_incoming(row_count_col)}")
            updated.add(row_count_col)

        for cnt_c in col_nonnull_count_map.values():
            if cnt_c not in updated:
                assignments.append(f"{cnt_c} = {mv_table}.{cnt_c} + {_incoming(cnt_c)}")
                updated.add(cnt_c)
        for sum_c in col_sum_map.values():
            if sum_c not in updated:
                assignments.append(f"{sum_c} = {_merged_sum(sum_c)}")
                updated.add(sum_c)
        for sum2_c in col_sum2_map.values():
            if sum2_c not in updated:
                assignments.append(f"{sum2_c} = {_merged_sum(sum2_c)}")
                updated.add(sum2_c)
        for tcnt_c in col_true_cnt_map.values():
            if tcnt_c not in updated:
                assignments.append(f"{tcnt_c} = {mv_table}.{tcnt_c} + {_incoming(tcnt_c)}")
                updated.add(tcnt_c)

        derived_aliases = {
            it.alias
            for it in work_items
            if it.is_aggregate
            and it.agg_func
            in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            )
        }
        return [
            a for a in assignments if a.split("=")[0].strip() in derived_aliases
        ] + [
            a for a in assignments if a.split("=")[0].strip() not in derived_aliases
        ]

    def _build_recompute_subquery_raw(self, item: SelectItem, record_prefix: str) -> str:
        if getattr(self, "_alg_mode", None) == "union_all_agg":
            inner_u = getattr(self, "_union_all_inner_sql", "")
            u_alias = getattr(self, "_union_all_sub_alias", "__mv_u")
            orig_item = getattr(self, "_uall_orig_item_map", {}).get(id(item), item)
            orig_grps = getattr(self, "_uall_orig_grp_exprs", self._effective_group_by_exprs)
            u_conds: List[str] = []
            if self.view.where_clause:
                u_conds.append(f"({self.view.where_clause})")
            for orig_g, branch_g in zip(orig_grps, self._effective_group_by_exprs):
                rhs_g = self._rewrite_expr_for_record(branch_g, record_prefix)
                u_conds.append(self._null_safe_eq(f"({orig_g})", f"({rhs_g})", is_nullable=True))
            w_u = f" WHERE {' AND '.join(u_conds)}" if u_conds else ""
            u_agg_sql = self._sql_backfill_agg_expr(orig_item)
            return f"(SELECT {u_agg_sql} FROM ({inner_u}) AS {u_alias}{w_u})"
        base_t = self.view.base_table
        conds: List[str] = []
        if self.view.where_clause:
            conds.append(f"({self.view.where_clause})")
        for g_expr in self._effective_group_by_exprs:
            lhs_g = self._rewrite_expr_for_record(g_expr, base_t)
            rhs_g = self._rewrite_expr_for_record(g_expr, record_prefix)
            is_nn = self._is_simple_col_ref(g_expr) and self._col_def(g_expr).not_null
            conds.append(
                self._null_safe_eq(
                    f"({lhs_g})",
                    f"({rhs_g})",
                    is_nullable=not is_nn,
                )
            )
        where_part = f" WHERE {' AND '.join(conds)}" if conds else ""
        agg_sql = self._sql_backfill_agg_expr(item)
        return f"(SELECT {agg_sql} FROM {base_t}{where_part})"

    def _variance_formula_sql(
        self,
        new_cnt: str,
        new_sum: str,
        new_sum2: str,
        is_sample: bool,
        is_stddev: bool,
    ) -> str:
        denom = f"({new_cnt} - 1)" if is_sample else new_cnt
        min_n = 2 if is_sample else 1
        raw_var = f"((1.0 * {new_sum2}) - ((1.0 * {new_sum}) * (1.0 * {new_sum})) / {new_cnt}) / {denom}"
        clamped_var = f"(CASE WHEN ({raw_var}) < 0.0 THEN 0.0 ELSE ({raw_var}) END)"
        val_expr = f"SQRT({clamped_var})" if is_stddev else clamped_var
        return f"(CASE WHEN {new_cnt} < {min_n} THEN NULL ELSE {val_expr} END)"

    def _group_match_predicate_tier1(self, target_table: str, record_prefix: str) -> str:
        grp_exprs = self._effective_group_by_exprs
        if not grp_exprs:
            return f"{target_table}.__mv_global_id = 1"
        if self._has_nullable_group_col():
            rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
            return f"{target_table}.__mv_pk = ({self._canonical_group_pk_expr(rec_exprs)})"
        preds: List[str] = []
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            rec_e = self._rewrite_expr_for_record(g_expr, record_prefix)
            preds.append(f"{target_table}.{mv_col} = {rec_e}")
        return " AND ".join(preds)

    def _distinct_match_predicate(self, aux_tbl: str, val_expr: str, record_prefix: str) -> str:
        grp_pred = self._group_match_predicate_tier1(aux_tbl, record_prefix)
        return f"{grp_pred} AND {aux_tbl}.__val = ({val_expr})"

    def _build_plus_delta_statements(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        record_prefix: str = "NEW",
    ) -> List[str]:
        stmts: List[str] = []
        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            ins_cols: List[str] = []
            ins_vals: List[str] = []
            if not grp_exprs:
                ins_cols.append("__mv_global_id")
                ins_vals.append("1")
            elif has_nullable_grp:
                ins_cols.append("__mv_pk")
                rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
                ins_vals.append(self._canonical_group_pk_expr(rec_exprs))
            else:
                for g_expr in grp_exprs:
                    ins_cols.append(self._group_col_alias(g_expr))
                    ins_vals.append(self._rewrite_expr_for_record(g_expr, record_prefix))
            ins_cols.extend(["__val", "__ref_cnt"])
            ins_vals.extend([val_expr, "1"])
            if self.dialect == Dialect.MYSQL:
                stmts.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(ins_cols)})\n"
                    f"    SELECT {', '.join(ins_vals)}{from_dual}\n"
                    f"    WHERE ({val_expr}) IS NOT NULL\n"
                    f"    ON DUPLICATE KEY UPDATE __ref_cnt = __ref_cnt + 1;"
                )
            else:
                stmts.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(ins_cols)})\n"
                    f"    SELECT {', '.join(ins_vals)}\n"
                    f"    WHERE ({val_expr}) IS NOT NULL\n"
                    f"    ON CONFLICT ({', '.join(pk_cols)}, __val) DO UPDATE SET\n"
                    f"        __ref_cnt = {aux_tbl}.__ref_cnt + 1;"
                )

        def _is_newly_distinct_cond(item: SelectItem) -> str:
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if not item.is_distinct:
                return f"({val_e}) IS NOT NULL"
            aux_tbl = distinct_aux_tables[ekey][0]
            match_p = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
            return f"({val_e}) IS NOT NULL AND (SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {match_p}) = 1"

        def _plus_inc_expr(item: SelectItem) -> str:
            return f"(CASE WHEN {_is_newly_distinct_cond(item)} THEN 1 ELSE 0 END)"

        def _plus_sum_expr(item: SelectItem, col_alias: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if item.is_distinct:
                cond = _is_newly_distinct_cond(item)
                return (
                    f"(CASE WHEN NOT ({cond}) THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE {mv_table}.{col_alias} + ({val_e}) END)"
                )
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                f"ELSE {mv_table}.{col_alias} + ({val_e}) END)"
            )

        def _plus_sum2_expr(item: SelectItem, col_alias: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            sq = f"(1.0 * ({val_e}) * ({val_e}))"
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                f"WHEN {mv_table}.{col_alias} IS NULL THEN {sq} "
                f"ELSE {mv_table}.{col_alias} + {sq} END)"
            )

        def _plus_tcnt_expr(item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if self.dialect == Dialect.POSTGRES:
                return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) != 0 THEN 1 ELSE 0 END)"

        insert_cols: List[str] = []
        insert_vals: List[str] = []
        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        if not grp_exprs:
            insert_cols.append("__mv_global_id")
            insert_vals.append("1")
        elif has_nullable_grp:
            insert_cols.append("__mv_pk")
            rec_exprs = [self._rewrite_expr_for_record(g, record_prefix) for g in grp_exprs]
            insert_vals.append(self._canonical_group_pk_expr(rec_exprs))

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                insert_cols.append(mv_col)
                insert_vals.append(self._rewrite_expr_for_record(g_expr, record_prefix))

        for item in work_items:
            col_alias = item.alias
            insert_cols.append(col_alias)
            if not item.is_aggregate:
                insert_vals.append(self._rewrite_expr_for_record(item.raw_expr, record_prefix))
                continue

            updated_cols.add(col_alias)
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                insert_vals.append("1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
            elif item.agg_func == AggFunc.SUM:
                insert_vals.append(f"({val_e})")
            elif item.agg_func == AggFunc.TOTAL:
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)")
            elif item.agg_func == AggFunc.AVG:
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 1.0 * ({val_e}) END)")
            elif item.agg_func in (AggFunc.VAR_POP, AggFunc.STDDEV_POP):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 0.0 END)")
            elif item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP):
                insert_vals.append("NULL")
            elif item.agg_func in (AggFunc.MIN, AggFunc.MAX, AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR):
                insert_vals.append(f"({val_e})")
            elif item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if self.dialect == Dialect.POSTGRES:
                    insert_vals.append(
                        f"(CASE WHEN ({val_e}) IS NULL THEN NULL WHEN ({val_e}) THEN TRUE ELSE FALSE END)"
                    )
                else:
                    insert_vals.append(
                        f"(CASE WHEN ({val_e}) IS NULL THEN NULL WHEN ({val_e}) != 0 THEN 1 ELSE 0 END)"
                    )
            else:
                insert_vals.append(f"({val_e})")

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + 1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                update_assignments.append(
                    f"{col_alias} = {mv_table}.{col_alias} + {_plus_inc_expr(item)}"
                )
            elif item.agg_func == AggFunc.SUM:
                new_sum = _plus_sum_expr(item, col_alias)
                update_assignments.append(f"{col_alias} = {new_sum}")
            elif item.agg_func == AggFunc.TOTAL:
                if item.is_distinct:
                    cond = _is_newly_distinct_cond(item)
                    inc_val = f"(CASE WHEN ({cond}) THEN 1.0 * ({val_e}) ELSE 0.0 END)"
                else:
                    inc_val = f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + {inc_val}")
            elif item.agg_func == AggFunc.AVG:
                cnt_col = col_nonnull_count_map[ekey]
                sum_col = col_sum_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_sum = _plus_sum_expr(item, sum_col)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"ELSE (1.0 * {new_sum}) / {new_cnt} END"
                )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                cnt_col = col_nonnull_count_map[ekey]
                sum_col = col_sum_map[ekey]
                sum2_col = col_sum2_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_sum = _plus_sum_expr(item, sum_col)
                new_sum2 = _plus_sum2_expr(item, sum2_col)
                formula = self._variance_formula_sql(
                    new_cnt=new_cnt,
                    new_sum=new_sum,
                    new_sum2=new_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                update_assignments.append(f"{col_alias} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL OR ({val_e}) < {mv_table}.{col_alias} "
                    f"THEN ({val_e}) ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.MAX:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL OR ({val_e}) > {mv_table}.{col_alias} "
                    f"THEN ({val_e}) ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.BIT_AND:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE ({mv_table}.{col_alias} & ({val_e})) END"
                )
            elif item.agg_func == AggFunc.BIT_OR:
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE ({mv_table}.{col_alias} | ({val_e})) END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                xor_expr = self._bitwise_xor_sql(f"{mv_table}.{col_alias}", f"({val_e})")
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{col_alias} IS NULL THEN ({val_e}) "
                    f"ELSE {xor_expr} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} + {_plus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} + {_plus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )
            else:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomp}")

        if row_count_col not in updated_cols:
            insert_cols.append(row_count_col)
            insert_vals.append("1")
            update_assignments.append(f"{row_count_col} = {mv_table}.{row_count_col} + 1")
            updated_cols.add(row_count_col)

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }
        already_in_ins = set(insert_cols)
        for ekey, cnt_col in col_nonnull_count_map.items():
            it = ekey_to_item[ekey]
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if cnt_col not in already_in_ins:
                insert_cols.append(cnt_col)
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
                already_in_ins.add(cnt_col)
            if cnt_col not in updated_cols:
                update_assignments.append(
                    f"{cnt_col} = {mv_table}.{cnt_col} + {_plus_inc_expr(it)}"
                )
                updated_cols.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            it = ekey_to_item[ekey]
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if sum_col not in already_in_ins:
                insert_cols.append(sum_col)
                insert_vals.append(f"({val_e})")
                already_in_ins.add(sum_col)
            if sum_col not in updated_cols:
                update_assignments.append(f"{sum_col} = {_plus_sum_expr(it, sum_col)}")
                updated_cols.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            it = ekey_to_item[ekey]
            val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
            if sum2_col not in already_in_ins:
                insert_cols.append(sum2_col)
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 1.0 * ({val_e}) * ({val_e}) END)"
                )
                already_in_ins.add(sum2_col)
            if sum2_col not in updated_cols:
                update_assignments.append(f"{sum2_col} = {_plus_sum2_expr(it, sum2_col)}")
                updated_cols.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            it = ekey_to_item[ekey]
            if tcnt_col not in already_in_ins:
                insert_cols.append(tcnt_col)
                insert_vals.append(_plus_tcnt_expr(it))
                already_in_ins.add(tcnt_col)
            if tcnt_col not in updated_cols:
                update_assignments.append(
                    f"{tcnt_col} = {mv_table}.{tcnt_col} + {_plus_tcnt_expr(it)}"
                )
                updated_cols.add(tcnt_col)

        # In MySQL, ON DUPLICATE KEY UPDATE evaluates left-to-right, so derived columns must come first
        derived_aliases = {
            it.alias
            for it in work_items
            if it.is_aggregate
            and it.agg_func
            in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            )
        }
        ordered_updates = [
            a for a in update_assignments if a.split("=")[0].strip() in derived_aliases
        ] + [
            a for a in update_assignments if a.split("=")[0].strip() not in derived_aliases
        ]

        if self.dialect == Dialect.MYSQL:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE\n"
                f"        {', '.join(ordered_updates)};"
            )
        else:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET\n"
                f"        {', '.join(ordered_updates)};"
            )
        return stmts

    def _build_minus_delta_statements(
        self,
        mv_table: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        record_prefix: str = "OLD",
        include_delete_zero_mv: bool = True,
    ) -> Tuple[List[str], Optional[str]]:
        stmts: List[str] = []
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            match_p = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"UPDATE {aux_tbl}\n"
                f"    SET __ref_cnt = {aux_tbl}.__ref_cnt - 1\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND {match_p};"
            )

        def _is_last_distinct_cond(item: SelectItem) -> str:
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if not item.is_distinct:
                return f"({val_e}) IS NOT NULL"
            aux_tbl = distinct_aux_tables[ekey][0]
            match_p = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
            return f"({val_e}) IS NOT NULL AND (SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {match_p}) <= 0"

        def _minus_dec_expr(item: SelectItem) -> str:
            return f"(CASE WHEN {_is_last_distinct_cond(item)} THEN 1 ELSE 0 END)"

        def _minus_sum_expr(item: SelectItem, col_alias: str, cnt_col: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
            if item.is_distinct:
                cond = _is_last_distinct_cond(item)
                return (
                    f"(CASE WHEN NOT ({cond}) THEN {mv_table}.{col_alias} "
                    f"WHEN {new_cnt} <= 0 THEN NULL "
                    f"ELSE {mv_table}.{col_alias} - ({val_e}) END)"
                )
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                f"WHEN {new_cnt} <= 0 THEN NULL "
                f"ELSE {mv_table}.{col_alias} - ({val_e}) END)"
            )

        def _minus_sum2_expr(item: SelectItem, col_alias: str, cnt_col: str) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
            sq = f"(1.0 * ({val_e}) * ({val_e}))"
            return (
                f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                f"WHEN {new_cnt} <= 0 THEN NULL "
                f"ELSE {mv_table}.{col_alias} - {sq} END)"
            )

        def _minus_tcnt_expr(item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if self.dialect == Dialect.POSTGRES:
                return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) != 0 THEN 1 ELSE 0 END)"

        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        for item in work_items:
            if not item.is_aggregate:
                continue
            col_alias = item.alias
            updated_cols.add(col_alias)
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} - 1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                update_assignments.append(
                    f"{col_alias} = {mv_table}.{col_alias} - {_minus_dec_expr(item)}"
                )
            elif item.agg_func == AggFunc.SUM:
                cnt_col = col_nonnull_count_map[ekey]
                update_assignments.append(
                    f"{col_alias} = {_minus_sum_expr(item, col_alias, cnt_col)}"
                )
            elif item.agg_func == AggFunc.TOTAL:
                if item.is_distinct:
                    cond = _is_last_distinct_cond(item)
                    dec_val = f"(CASE WHEN ({cond}) THEN 1.0 * ({val_e}) ELSE 0.0 END)"
                else:
                    dec_val = f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} - {dec_val}")
            elif item.agg_func == AggFunc.AVG:
                cnt_col = col_nonnull_count_map[ekey]
                sum_col = col_sum_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_sum = _minus_sum_expr(item, sum_col, cnt_col)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"ELSE (1.0 * {new_sum}) / {new_cnt} END"
                )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                cnt_col = col_nonnull_count_map[ekey]
                sum_col = col_sum_map[ekey]
                sum2_col = col_sum2_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_sum = _minus_sum_expr(item, sum_col, cnt_col)
                new_sum2 = _minus_sum2_expr(item, sum2_col, cnt_col)
                formula = self._variance_formula_sql(
                    new_cnt=new_cnt,
                    new_sum=new_sum,
                    new_sum2=new_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                update_assignments.append(f"{col_alias} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL AND {mv_table}.{col_alias} >= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL AND {mv_table}.{col_alias} <= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR):
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NOT NULL "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_col = col_nonnull_count_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                xor_expr = self._bitwise_xor_sql(f"{mv_table}.{col_alias}", f"({val_e})")
                update_assignments.append(
                    f"{col_alias} = CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {new_cnt} <= 0 THEN NULL ELSE {xor_expr} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                new_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )
            else:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomp}")

        if row_count_col not in updated_cols:
            update_assignments.append(f"{row_count_col} = {mv_table}.{row_count_col} - 1")
            updated_cols.add(row_count_col)

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }
        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{cnt_col} = {mv_table}.{cnt_col} - {_minus_dec_expr(it)}"
                )
                updated_cols.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            if sum_col not in updated_cols:
                it = ekey_to_item[ekey]
                cnt_col = col_nonnull_count_map[ekey]
                update_assignments.append(
                    f"{sum_col} = {_minus_sum_expr(it, sum_col, cnt_col)}"
                )
                updated_cols.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in updated_cols:
                it = ekey_to_item[ekey]
                cnt_col = col_nonnull_count_map[ekey]
                update_assignments.append(
                    f"{sum2_col} = {_minus_sum2_expr(it, sum2_col, cnt_col)}"
                )
                updated_cols.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{tcnt_col} = {mv_table}.{tcnt_col} - {_minus_tcnt_expr(it)}"
                )
                updated_cols.add(tcnt_col)

        # In MySQL, UPDATE evaluates SET assignments left-to-right!
        # Order: (1) derived aggs (AVG, VAR, STDDEV, BIT_XOR, BOOL_AND, BOOL_OR)
        #        (2) SUM / sum2 / true_cnt (which read col_nonnull_count_map)
        #        (3) counts & MIN/MAX
        high_priority_cols = (
            {
                it.alias
                for it in work_items
                if it.is_aggregate
                and it.agg_func
                in (
                    AggFunc.AVG,
                    AggFunc.VAR_POP,
                    AggFunc.VAR_SAMP,
                    AggFunc.STDDEV_POP,
                    AggFunc.STDDEV_SAMP,
                    AggFunc.BIT_XOR,
                    AggFunc.BOOL_AND,
                    AggFunc.BOOL_OR,
                )
            }
        )
        mid_priority_cols = (
            {it.alias for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
            | set(col_sum_map.values())
            | set(col_sum2_map.values())
            | set(col_true_cnt_map.values())
        )
        tier_1 = [a for a in update_assignments if a.split("=")[0].strip() in high_priority_cols]
        tier_2 = [
            a
            for a in update_assignments
            if a.split("=")[0].strip() not in high_priority_cols
            and a.split("=")[0].strip() in mid_priority_cols
        ]
        tier_3 = [
            a
            for a in update_assignments
            if a.split("=")[0].strip() not in high_priority_cols
            and a.split("=")[0].strip() not in mid_priority_cols
        ]
        ordered_updates = tier_1 + tier_2 + tier_3

        grp_pred = self._group_match_predicate_tier1(mv_table, record_prefix)
        stmts.append(
            f"UPDATE {mv_table}\n"
            f"    SET {', '.join(ordered_updates)}\n"
            f"    WHERE {grp_pred};"
        )

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            match_p = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"DELETE FROM {aux_tbl}\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND ({match_p}) AND {aux_tbl}.__ref_cnt <= 0;"
            )

        delete_zero_mv_stmt: Optional[str] = None
        if grp_exprs:
            delete_zero_mv_stmt = (
                f"DELETE FROM {mv_table}\n"
                f"    WHERE ({grp_pred}) AND {mv_table}.{row_count_col} <= 0;"
            )
            if include_delete_zero_mv:
                stmts.append(delete_zero_mv_stmt)

        return stmts, delete_zero_mv_stmt

    def _apply_where_guard_to_stmt(self, stmt: str, where_cond: Optional[str]) -> str:
        if not where_cond:
            return stmt
        s = stmt.strip()
        if re.match(r"^INSERT\s+INTO\b", s, flags=re.IGNORECASE):
            if re.search(r"\n\s*WHERE\s+TRUE\b", s, flags=re.IGNORECASE):
                return re.sub(
                    r"\n\s*WHERE\s+TRUE\b",
                    f"\n    WHERE ({where_cond})",
                    s,
                    count=1,
                    flags=re.IGNORECASE,
                )
            return re.sub(
                r"\n\s*WHERE\s+",
                f"\n    WHERE ({where_cond}) AND ",
                s,
                count=1,
                flags=re.IGNORECASE,
            )
        if re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", s, flags=re.IGNORECASE):
            return re.sub(
                r"\n\s*WHERE\s+",
                f"\n    WHERE ({where_cond}) AND ",
                s,
                count=1,
                flags=re.IGNORECASE,
            )
        return stmt

    def _build_tier1_triggers(
        self,
        view_name: str,
        mv_table: str,
        base_t: str,
        row_count_col: str,
        col_nonnull_count_map: Dict[str, str],
        col_sum_map: Dict[str, str],
        col_sum2_map: Dict[str, str],
        col_true_cnt_map: Dict[str, str],
        distinct_aux_tables: Dict[str, Tuple[str, SelectItem, str]],
        expr_key_fn: Callable[[SelectItem], str],
        initial_populate_sqls: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        plus_new_stmts = self._build_plus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="NEW",
        )
        minus_old_del_stmts, _ = self._build_minus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="OLD",
            include_delete_zero_mv=True,
        )
        minus_old_upd_stmts, delete_zero_mv_stmt = self._build_minus_delta_statements(
            mv_table,
            row_count_col,
            col_nonnull_count_map,
            col_sum_map,
            col_sum2_map,
            col_true_cnt_map,
            distinct_aux_tables,
            expr_key_fn,
            record_prefix="OLD",
            include_delete_zero_mv=False,
        )

        where_new = (
            self._rewrite_expr_for_record(self.view.where_clause, "NEW")
            if self.view.where_clause
            else None
        )
        where_old = (
            self._rewrite_expr_for_record(self.view.where_clause, "OLD")
            if self.view.where_clause
            else None
        )

        trunc_stmts = [
            f"DELETE FROM {aux_tbl};" for aux_tbl, _, _ in distinct_aux_tables.values()
        ] + [f"DELETE FROM {mv_table};"]
        if not self._effective_group_by_exprs and initial_populate_sqls:
            trunc_stmts.extend(initial_populate_sqls)

        return self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=plus_new_stmts,
            del_stmts=minus_old_del_stmts,
            upd_old_stmts=minus_old_upd_stmts,
            upd_new_stmts=plus_new_stmts,
            upd_cleanup_old_stmts=[delete_zero_mv_stmt] if delete_zero_mv_stmt else [],
            trunc_stmts=trunc_stmts,
            where_new=where_new,
            where_old=where_old,
        )

    def _emit_dialect_triggers_for_table(
        self,
        view_name: str,
        base_t: str,
        ins_stmts: List[str],
        del_stmts: List[str],
        upd_old_stmts: List[str],
        upd_new_stmts: List[str],
        upd_cleanup_old_stmts: Optional[List[str]] = None,
        trunc_stmts: Optional[List[str]] = None,
        where_new: Optional[str] = None,
        where_old: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        log_vname = self.logical_view_name
        p = self.object_prefix if self.object_prefix else "__matview_"
        trig_prefix = f"{p}trg_{log_vname}__{base_t}"
        ref_cols = self._referenced_columns_for_table(base_t)
        upd_of_clause = (
            f" OF {', '.join(ref_cols)}"
            if ref_cols and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES)
            else ""
        )

        t_schema = self._schemas_lower.get(base_t.lower())
        unchanged_conds: List[str] = []
        if t_schema and ref_cols:
            for c_name in ref_cols:
                c_def = t_schema.columns.get(c_name)
                is_null = not (c_def.not_null if c_def else False)
                unchanged_conds.append(
                    self._null_safe_eq(f"OLD.{c_name}", f"NEW.{c_name}", is_nullable=is_null)
                )
        changed_guard = f"NOT ({' AND '.join(unchanged_conds)})" if unchanged_conds else None

        cleanup_stmts = upd_cleanup_old_stmts or []

        if self.dialect == Dialect.SQLITE:
            when_ins = f"\nWHEN {where_new}" if where_new else ""
            ins_body = "\n    ".join(ins_stmts)
            trig_ins = (
                f"CREATE TRIGGER {trig_prefix}_ai\n"
                f"AFTER INSERT ON {base_t}\n"
                f"FOR EACH ROW{when_ins}\n"
                f"BEGIN\n"
                f"    {ins_body}\n"
                f"END;"
            )

            when_del = f"\nWHEN {where_old}" if where_old else ""
            del_body = "\n    ".join(del_stmts)
            trig_del = (
                f"CREATE TRIGGER {trig_prefix}_ad\n"
                f"AFTER DELETE ON {base_t}\n"
                f"FOR EACH ROW{when_del}\n"
                f"BEGIN\n"
                f"    {del_body}\n"
                f"END;"
            )

            upd_stmts = (
                [self._apply_where_guard_to_stmt(s, where_old) for s in upd_old_stmts]
                + [self._apply_where_guard_to_stmt(s, where_new) for s in upd_new_stmts]
                + [self._apply_where_guard_to_stmt(s, where_old) for s in cleanup_stmts]
            )
            upd_body = "\n    ".join(upd_stmts)
            when_upd = f"\nWHEN {changed_guard}" if changed_guard else ""
            trig_upd = (
                f"CREATE TRIGGER {trig_prefix}_au\n"
                f"AFTER UPDATE{upd_of_clause} ON {base_t}\n"
                f"FOR EACH ROW{when_upd}\n"
                f"BEGIN\n"
                f"    {upd_body}\n"
                f"END;"
            )
            t_names = [f"{trig_prefix}_ai", f"{trig_prefix}_ad", f"{trig_prefix}_au"]
            return [trig_ins, trig_del, trig_upd], t_names, []

        elif self.dialect == Dialect.POSTGRES:
            ins_block = "\n            ".join(ins_stmts)
            minus_block = "\n            ".join(upd_old_stmts)
            plus_block = "\n            ".join(upd_new_stmts)
            cleanup_block = "\n            ".join(cleanup_stmts)
            trunc_block = "\n        ".join(trunc_stmts or [f"DELETE FROM {self._backing_table_name()};"])

            ins_section = (
                f"IF ({where_new}) THEN\n            {ins_block}\n        END IF;"
                if where_new and ins_block
                else ins_block
            )
            old_section = (
                f"IF ({where_old}) THEN\n            {minus_block}\n        END IF;"
                if where_old and minus_block
                else minus_block
            )
            new_section = (
                f"IF ({where_new}) THEN\n            {plus_block}\n        END IF;"
                if where_new and plus_block
                else plus_block
            )
            clean_section = (
                f"\n        IF ({where_old}) THEN\n            {cleanup_block}\n        END IF;"
                if where_old and cleanup_block
                else (f"\n        {cleanup_block}" if cleanup_block else "")
            )
            del_full_block = "\n            ".join(del_stmts)
            del_section = (
                f"IF ({where_old}) THEN\n            {del_full_block}\n        END IF;"
                if where_old
                else del_full_block
            )

            fn_trunc = f"{p}fn_sync_{log_vname}__{base_t}_trunc"
            fn_trunc_ddl = (
                f"CREATE OR REPLACE FUNCTION {fn_trunc}()\n"
                f"RETURNS TRIGGER AS $$\n"
                f"BEGIN\n"
                f"    {trunc_block}\n"
                f"    RETURN NULL;\n"
                f"END;\n$$ LANGUAGE plpgsql;"
            )
            t_trunc = (
                f"CREATE TRIGGER {trig_prefix}_trunc\n"
                f"AFTER TRUNCATE ON {base_t}\n"
                f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_trunc}();"
            )

            if self.pg_use_transition_tables:
                fn_ai = f"{p}fn_sync_{log_vname}__{base_t}_ai"
                fn_ad = f"{p}fn_sync_{log_vname}__{base_t}_ad"
                fn_au = f"{p}fn_sync_{log_vname}__{base_t}_au"
                fn_ai_ddl = (
                    f"CREATE OR REPLACE FUNCTION {fn_ai}()\n"
                    f"RETURNS TRIGGER AS $$\n"
                    f"DECLARE\n    NEW RECORD;\n"
                    f"BEGIN\n"
                    f"    FOR NEW IN SELECT * FROM __mv_new_rows LOOP\n"
                    f"        {ins_section}\n"
                    f"    END LOOP;\n"
                    f"    RETURN NULL;\n"
                    f"END;\n$$ LANGUAGE plpgsql;"
                )
                fn_ad_ddl = (
                    f"CREATE OR REPLACE FUNCTION {fn_ad}()\n"
                    f"RETURNS TRIGGER AS $$\n"
                    f"DECLARE\n    OLD RECORD;\n"
                    f"BEGIN\n"
                    f"    FOR OLD IN SELECT * FROM __mv_old_rows LOOP\n"
                    f"        {del_section}\n"
                    f"    END LOOP;\n"
                    f"    RETURN NULL;\n"
                    f"END;\n$$ LANGUAGE plpgsql;"
                )
                au_loops: List[str] = []
                if old_section.strip():
                    au_loops.append(
                        f"    FOR OLD IN SELECT * FROM __mv_old_rows LOOP\n"
                        f"        {old_section}\n"
                        f"    END LOOP;"
                    )
                if clean_section.strip():
                    au_loops.append(
                        f"    FOR OLD IN SELECT * FROM __mv_old_rows LOOP\n"
                        f"        {clean_section.strip()}\n"
                        f"    END LOOP;"
                    )
                if new_section.strip():
                    au_loops.append(
                        f"    FOR NEW IN SELECT * FROM __mv_new_rows LOOP\n"
                        f"        {new_section}\n"
                        f"    END LOOP;"
                    )
                au_body = "\n".join(au_loops)
                fn_au_ddl = (
                    f"CREATE OR REPLACE FUNCTION {fn_au}()\n"
                    f"RETURNS TRIGGER AS $$\n"
                    f"DECLARE\n    OLD RECORD;\n    NEW RECORD;\n"
                    f"BEGIN\n"
                    f"{au_body}\n"
                    f"    RETURN NULL;\n"
                    f"END;\n$$ LANGUAGE plpgsql;"
                )
                t_ai = (
                    f"CREATE TRIGGER {trig_prefix}_ai\n"
                    f"AFTER INSERT ON {base_t}\n"
                    f"REFERENCING NEW TABLE AS __mv_new_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_ai}();"
                )
                t_ad = (
                    f"CREATE TRIGGER {trig_prefix}_ad\n"
                    f"AFTER DELETE ON {base_t}\n"
                    f"REFERENCING OLD TABLE AS __mv_old_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_ad}();"
                )
                t_au = (
                    f"CREATE TRIGGER {trig_prefix}_au\n"
                    f"AFTER UPDATE ON {base_t}\n"
                    f"REFERENCING OLD TABLE AS __mv_old_rows NEW TABLE AS __mv_new_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_au}();"
                )
                return (
                    [fn_ai_ddl, t_ai, fn_ad_ddl, t_ad, fn_au_ddl, t_au, fn_trunc_ddl, t_trunc],
                    [f"{trig_prefix}_ai", f"{trig_prefix}_ad", f"{trig_prefix}_au", f"{trig_prefix}_trunc"],
                    [fn_ai, fn_ad, fn_au, fn_trunc],
                )

            fn_name = f"{p}fn_sync_{log_vname}__{base_t}"
            upd_guard_open = f"IF ({changed_guard}) THEN\n        " if changed_guard else ""
            upd_guard_close = "\n        END IF;" if changed_guard else ""
            upd_inner = (
                f"{old_section}\n        {new_section}{clean_section}"
                if old_section.strip()
                else f"{new_section}{clean_section}"
            )

            fn_ddl = (
                f"CREATE OR REPLACE FUNCTION {fn_name}()\n"
                f"RETURNS TRIGGER AS $$\n"
                f"BEGIN\n"
                f"    IF (TG_OP = 'DELETE') THEN\n"
                f"        {del_section}\n"
                f"    ELSIF (TG_OP = 'INSERT') THEN\n"
                f"        {ins_section}\n"
                f"    ELSIF (TG_OP = 'UPDATE') THEN\n"
                f"        {upd_guard_open}{upd_inner}{upd_guard_close}\n"
                f"    END IF;\n"
                f"    RETURN NULL;\n"
                f"END;\n"
                f"$$ LANGUAGE plpgsql;"
            )
            t_sync = (
                f"CREATE TRIGGER {trig_prefix}_sync\n"
                f"AFTER INSERT OR UPDATE{upd_of_clause} OR DELETE ON {base_t}\n"
                f"FOR EACH ROW EXECUTE FUNCTION {fn_name}();"
            )
            return (
                [fn_ddl, t_sync, fn_trunc_ddl, t_trunc],
                [f"{trig_prefix}_sync", f"{trig_prefix}_trunc"],
                [fn_name, fn_trunc],
            )

        elif self.dialect == Dialect.MYSQL:
            ins_block = "\n        ".join(ins_stmts)
            del_block = "\n        ".join(del_stmts)
            minus_block = "\n        ".join(upd_old_stmts)
            plus_block = "\n        ".join(upd_new_stmts)
            cleanup_block = "\n        ".join(cleanup_stmts)

            ins_section = (
                f"IF ({where_new}) THEN\n        {ins_block}\n    END IF;"
                if where_new and ins_block
                else ins_block
            )
            del_section = (
                f"IF ({where_old}) THEN\n        {del_block}\n    END IF;"
                if where_old and del_block
                else del_block
            )
            old_section = (
                f"IF ({where_old}) THEN\n        {minus_block}\n    END IF;"
                if where_old and minus_block
                else minus_block
            )
            new_section = (
                f"IF ({where_new}) THEN\n        {plus_block}\n    END IF;"
                if where_new and plus_block
                else plus_block
            )
            clean_section = (
                f"\n    IF ({where_old}) THEN\n        {cleanup_block}\n    END IF;"
                if where_old and cleanup_block
                else (f"\n    {cleanup_block}" if cleanup_block else "")
            )

            upd_guard_open = f"IF ({changed_guard}) THEN\n    " if changed_guard else ""
            upd_guard_close = "\n    END IF;" if changed_guard else ""
            upd_inner = (
                f"{old_section}\n    {new_section}{clean_section}"
                if old_section.strip()
                else f"{new_section}{clean_section}"
            )

            trig_ins = (
                f"CREATE TRIGGER {trig_prefix}_ai\n"
                f"AFTER INSERT ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {ins_section}\n"
                f"END;"
            )
            trig_del = (
                f"CREATE TRIGGER {trig_prefix}_ad\n"
                f"AFTER DELETE ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {del_section}\n"
                f"END;"
            )
            trig_upd = (
                f"CREATE TRIGGER {trig_prefix}_au\n"
                f"AFTER UPDATE ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {upd_guard_open}{upd_inner}{upd_guard_close}\n"
                f"END;"
            )
            t_names = [f"{trig_prefix}_ai", f"{trig_prefix}_ad", f"{trig_prefix}_au"]
            return [trig_ins, trig_del, trig_upd], t_names, []

        return [], [], []


    # =========================================================================
    # TIER 1B: Multi-Table INNER / CROSS JOIN O(1) Algebraic Delta Maintenance
    # =========================================================================

    def _qualify_join_expr(self, expr: Optional[str]) -> str:
        if not expr:
            return ""
        col_to_aliases: Dict[str, List[str]] = {}
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts:
                for c_name in ts.columns:
                    col_to_aliases.setdefault(c_name.lower(), []).append(j.alias)

        alias_remap: Dict[str, str] = {}
        for j in self.view.joins:
            alias_remap[j.table_name.lower()] = j.alias
            alias_remap[j.alias.lower()] = j.alias

        tokens = re.split(r"('(?:''|[^'])*')", expr)
        out: List[str] = []
        for idx_t, tok in enumerate(tokens):
            if idx_t % 2 == 1:
                out.append(tok)
                continue
            mod = tok
            for old_q, new_q in sorted(alias_remap.items(), key=lambda x: len(x[0]), reverse=True):
                if old_q != new_q.lower():
                    mod = re.sub(
                        rf"\b{re.escape(old_q)}\s*\.",
                        f"{new_q}.",
                        mod,
                        flags=re.IGNORECASE,
                    )
            for c_low, als in sorted(col_to_aliases.items(), key=lambda x: len(x[0]), reverse=True):
                if len(als) == 1 or any(
                    j.on_condition and re.search(rf"\b{re.escape(c_low)}\b", j.on_condition, re.IGNORECASE)
                    for j in self.view.joins
                ):
                    al = als[0]
                    mod = re.sub(
                        rf"(?<![.\w])\b({re.escape(c_low)})\b(?!\s*[.(])",
                        rf"{al}.\1",
                        mod,
                        flags=re.IGNORECASE,
                    )
            out.append(mod)
        return "".join(out)

    def _compile_tier1_join_algebraic(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        count_star_alias: Optional[str] = None
        for item in work_items:
            if (
                item.is_aggregate
                and item.agg_func == AggFunc.COUNT_STAR
                and not item.filter_where
            ):
                count_star_alias = item.alias
                break
        row_count_col = count_star_alias or "__mv_row_count"

        col_defs_sql: List[str] = []
        user_columns: List[str] = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"

        if not grp_exprs:
            pk_col = "__mv_global_id"
            col_defs_sql.append("__mv_global_id INTEGER NOT NULL PRIMARY KEY")
        else:
            pk_col = "__mv_pk"
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden_grp: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_grp:
                seen_hidden_grp.add(mv_col.lower())
                c_def = self._lookup_col_def(g_expr)
                if c_def:
                    dt = self._normalize_dialect_type(c_def.data_type)
                else:
                    fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                    dt = self._infer_expr_storage_type(fake_it)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{mv_col} {dt}")

        def _expr_key(item: SelectItem) -> str:
            eff = self._effective_agg_value_expr(item, record_prefix=None).strip().lower()
            return f"{'dist:' if item.is_distinct else ''}{eff}"

        col_nonnull_count_map: Dict[str, str] = {}
        col_sum_map: Dict[str, str] = {}
        col_sum2_map: Dict[str, str] = {}
        col_true_cnt_map: Dict[str, str] = {}

        for item in work_items:
            if not item.is_aggregate:
                continue
            ekey = _expr_key(item)
            if item.agg_func == AggFunc.COUNT_COL:
                col_nonnull_count_map[ekey] = item.alias
            elif item.agg_func == AggFunc.SUM:
                col_sum_map[ekey] = item.alias

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                c_def = self._lookup_col_def(item.raw_expr)
                dt = (
                    self._normalize_dialect_type(c_def.data_type)
                    if c_def
                    else self._infer_expr_storage_type(item)
                )
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{item.alias} {dt}")
            else:
                storage_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{item.alias} {storage_type}")

        if count_star_alias is None:
            col_defs_sql.append(f"{row_count_col} {self._int_type()} NOT NULL DEFAULT 0")

        aux_idx = 0
        for item in work_items:
            if not item.is_aggregate or item.agg_func is None:
                continue
            ekey = _expr_key(item)
            safe_id = _sanitize_ident(item.alias)
            if item.agg_func in (
                AggFunc.SUM,
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BIT_XOR,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            ) or (item.agg_func == AggFunc.COUNT_STAR and item.filter_where):
                if item.agg_func == AggFunc.COUNT_STAR and item.filter_where:
                    col_nonnull_count_map[ekey] = item.alias
                elif ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    cnt_col = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = cnt_col
                    col_defs_sql.append(f"{cnt_col} {self._int_type()} NOT NULL DEFAULT 0")

            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    sum_col = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = sum_col
                    col_defs_sql.append(f"{sum_col} {self._float_type()}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    sum2_col = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._float_type()}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col
        }
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=False,
            backing_col_map=backing_col_map,
        )

        reader_select_override = [
            self._compound_reader_exprs.get(u_col, u_col) for u_col in user_columns
        ]
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                select_exprs_override=reader_select_override,
                extra_where_filter=self._rewritten_having_clause,
            )
        )

        qual_grp_exprs = [self._qualify_join_expr(g) for g in grp_exprs]
        target_cols: List[str] = [pk_col]
        select_exprs: List[str] = [
            "1" if not grp_exprs else self._canonical_group_pk_expr(qual_grp_exprs)
        ]

        seen_hidden: Set[str] = set()
        for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
            mv_col = self._group_col_alias(g_raw)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                target_cols.append(mv_col)
                select_exprs.append(g_qual)

        for item in work_items:
            target_cols.append(item.alias)
            if not item.is_aggregate:
                select_exprs.append(self._qualify_join_expr(item.raw_expr))
            else:
                select_exprs.append(self._qualify_join_expr(self._sql_backfill_agg_expr(item)))

        if count_star_alias is None:
            target_cols.append(row_count_col)
            select_exprs.append("COUNT(*)")

        already_in_target = set(target_cols)
        ekey_to_item: Dict[str, SelectItem] = {
            _expr_key(it): it for it in work_items if it.is_aggregate
        }
        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                target_cols.append(cnt_col)
                select_exprs.append(f"COUNT({eff})")
                already_in_target.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            if sum_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                target_cols.append(sum_col)
                select_exprs.append(f"SUM({eff})")
                already_in_target.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                target_cols.append(sum2_col)
                select_exprs.append(f"SUM(1.0 * ({eff}) * ({eff}))")
                already_in_target.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._qualify_join_expr(self._effective_agg_value_expr(it, record_prefix=None))
                target_cols.append(tcnt_col)
                select_exprs.append(
                    f"COALESCE(SUM(CASE WHEN ({eff}) IS NOT NULL AND ({eff}) THEN 1 ELSE 0 END), 0)"
                )
                already_in_target.add(tcnt_col)

        def _build_join_from_clause(replace_alias: Optional[str] = None, rec_prefix: Optional[str] = None) -> str:
            from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
            parts: List[str] = []
            for idx, j in enumerate(self.view.joins):
                if replace_alias and j.alias.lower() == replace_alias.lower() and rec_prefix:
                    ts = self._schemas_lower[j.table_name.lower()]
                    rec_proj = ", ".join(f"{rec_prefix}.{c} AS {c}" for c in ts.columns)
                    tbl_sql = f"(SELECT {rec_proj}{from_dual}) AS {j.alias}"
                else:
                    tbl_sql = f"{j.table_name} AS {j.alias}"
                if idx == 0:
                    parts.append(tbl_sql)
                else:
                    on_sql = f" ON {self._qualify_join_expr(j.on_condition)}" if j.on_condition else ""
                    parts.append(f"{j.join_type} {tbl_sql}{on_sql}")
            return " ".join(parts)

        full_from_sql = _build_join_from_clause()
        qual_where = self._qualify_join_expr(self.view.where_clause) if self.view.where_clause else None
        where_sql = f"\nWHERE {qual_where}" if qual_where else ""
        qual_grp_for_q = self._group_by_exprs_for_query(qual_grp_exprs, include_canonical_pk=True)
        group_sql = f"\nGROUP BY {', '.join(qual_grp_for_q)}" if qual_grp_for_q else ""

        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM {full_from_sql}{where_sql}{group_sql};"
        )
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                f"SELECT {', '.join(target_cols)}\n"
                f"FROM (\nSELECT {', '.join(f'{ex} AS {col}' for col, ex in zip(target_cols, select_exprs))}\n"
                f"FROM {full_from_sql}{where_sql}{group_sql}\n) AS __mv_init\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        def _build_join_recompute_subquery(item: SelectItem) -> str:
            alias_remap = {j.alias.lower(): f"__mv_j_{j.alias}" for j in self.view.joins}
            for j in self.view.joins:
                alias_remap[j.table_name.lower()] = f"__mv_j_{j.alias}"

            def _remap_to_j(frag: str) -> str:
                q_frag = self._qualify_join_expr(frag)
                tokens = re.split(r"('(?:''|[^'])*')", q_frag)
                out = []
                for idx_t, tok in enumerate(tokens):
                    if idx_t % 2 == 1:
                        out.append(tok)
                        continue
                    mod = tok
                    for old_a, new_a in sorted(alias_remap.items(), key=lambda x: len(x[0]), reverse=True):
                        mod = re.sub(rf"\b{re.escape(old_a)}\s*\.", f"{new_a}.", mod, flags=re.IGNORECASE)
                    out.append(mod)
                return "".join(out)

            j_parts: List[str] = []
            for idx, j in enumerate(self.view.joins):
                tbl_s = f"{j.table_name} AS __mv_j_{j.alias}"
                if idx == 0:
                    j_parts.append(tbl_s)
                else:
                    on_s = f" ON {_remap_to_j(j.on_condition)}" if j.on_condition else ""
                    j_parts.append(f"{j.join_type} {tbl_s}{on_s}")
            conds: List[str] = []
            if self.view.where_clause:
                conds.append(f"({_remap_to_j(self.view.where_clause)})")
            for g_raw in grp_exprs:
                mv_col = self._group_col_alias(g_raw)
                lhs_j = _remap_to_j(g_raw)
                conds.append(
                    self._null_safe_eq(f"({lhs_j})", f"{mv_table}.{mv_col}", is_nullable=True)
                )
            w_sub = f" WHERE {' AND '.join(conds)}" if conds else ""
            agg_e = _remap_to_j(self._sql_backfill_agg_expr(item))
            return f"(SELECT {agg_e} FROM {' '.join(j_parts)}{w_sub})"

        plus_merge_assignments = self._build_tier1_chunk_merge_assignments(
            mv_table=mv_table,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            expr_key_fn=_expr_key,
            use_mysql_values=(self.dialect == Dialect.MYSQL),
        )

        # Build minus assignments from `__mv_delta`
        minus_assignments: List[str] = []
        minus_updated: Set[str] = set()

        def _delta_sum_minus(col: str, cnt_col: str) -> str:
            new_c = f"({mv_table}.{cnt_col} - __mv_delta.{cnt_col})"
            return (
                f"(CASE WHEN __mv_delta.{col} IS NULL THEN {mv_table}.{col} "
                f"WHEN {new_c} <= 0 THEN NULL "
                f"ELSE {mv_table}.{col} - __mv_delta.{col} END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = item.alias
            minus_updated.add(c)
            ekey = _expr_key(item)
            if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                minus_assignments.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")
            elif item.agg_func == AggFunc.SUM:
                cnt_c = col_nonnull_count_map[ekey]
                minus_assignments.append(f"{c} = {_delta_sum_minus(c, cnt_c)}")
            elif item.agg_func == AggFunc.AVG:
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_sum = _delta_sum_minus(sum_c, cnt_c)
                minus_assignments.append(
                    f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL ELSE (1.0 * {new_sum}) / {new_cnt} END"
                )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                sum2_c = col_sum2_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_sum = _delta_sum_minus(sum_c, cnt_c)
                new_sum2 = _delta_sum_minus(sum2_c, cnt_c)
                formula = self._variance_formula_sql(
                    new_cnt=new_cnt,
                    new_sum=new_sum,
                    new_sum2=new_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                minus_assignments.append(f"{c} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                recomp = _build_join_recompute_subquery(item)
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} >= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = _build_join_recompute_subquery(item)
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} <= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_c = col_nonnull_count_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                xor_e = self._bitwise_xor_sql(f"{mv_table}.{c}", f"(__mv_delta.{c})")
                minus_assignments.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {new_cnt} <= 0 THEN NULL ELSE {xor_e} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_assignments.append(
                    f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = {new_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                new_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_assignments.append(
                    f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )

        if row_count_col not in minus_updated:
            minus_assignments.append(f"{row_count_col} = {mv_table}.{row_count_col} - __mv_delta.{row_count_col}")
            minus_updated.add(row_count_col)

        for ekey, cnt_c in col_nonnull_count_map.items():
            if cnt_c not in minus_updated:
                minus_assignments.append(f"{cnt_c} = {mv_table}.{cnt_c} - __mv_delta.{cnt_c}")
                minus_updated.add(cnt_c)
        for ekey, sum_c in col_sum_map.items():
            if sum_c not in minus_updated:
                cnt_c = col_nonnull_count_map[ekey]
                minus_assignments.append(f"{sum_c} = {_delta_sum_minus(sum_c, cnt_c)}")
                minus_updated.add(sum_c)
        for ekey, sum2_c in col_sum2_map.items():
            if sum2_c not in minus_updated:
                cnt_c = col_nonnull_count_map[ekey]
                minus_assignments.append(f"{sum2_c} = {_delta_sum_minus(sum2_c, cnt_c)}")
                minus_updated.add(sum2_c)
        for ekey, tcnt_c in col_true_cnt_map.items():
            if tcnt_c not in minus_updated:
                minus_assignments.append(f"{tcnt_c} = {mv_table}.{tcnt_c} - __mv_delta.{tcnt_c}")
                minus_updated.add(tcnt_c)

        high_p = {
            it.alias
            for it in work_items
            if it.is_aggregate
            and it.agg_func
            in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BIT_XOR,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            )
        }
        mid_p = (
            {it.alias for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
            | set(col_sum_map.values())
            | set(col_sum2_map.values())
            | set(col_true_cnt_map.values())
        )
        ordered_minus = (
            [a for a in minus_assignments if a.split("=")[0].strip() in high_p]
            + [a for a in minus_assignments if a.split("=")[0].strip() not in high_p and a.split("=")[0].strip() in mid_p]
            + [a for a in minus_assignments if a.split("=")[0].strip() not in high_p and a.split("=")[0].strip() not in mid_p]
        )

        # For MySQL multi-table UPDATE, LHS column names must be qualified with `mv_table.`
        mysql_ordered_minus = [
            f"{mv_table}.{a}" for a in ordered_minus
        ]

        aliased_delta_select = ", ".join(
            f"{ex} AS {col}" for col, ex in zip(target_cols, select_exprs)
        )
        having_non_empty = "" if grp_exprs else "\nHAVING COUNT(*) > 0"

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for j in self.view.joins:
            base_t = j.table_name
            j_alias = j.alias
            from_new = _build_join_from_clause(replace_alias=j_alias, rec_prefix="NEW")
            from_old = _build_join_from_clause(replace_alias=j_alias, rec_prefix="OLD")

            if self.dialect == Dialect.MYSQL:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {from_new}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ON DUPLICATE KEY UPDATE\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    INNER JOIN (\n"
                    f"        SELECT {aliased_delta_select}\n"
                    f"        FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta ON {mv_table}.{pk_col} = __mv_delta.{pk_col}\n"
                    f"    SET {', '.join(mysql_ordered_minus)};"
                )
            else:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {from_new}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ON CONFLICT ({pk_col}) DO UPDATE SET\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    SET {', '.join(ordered_minus)}\n"
                    f"    FROM (\n"
                    f"        SELECT {aliased_delta_select}\n"
                    f"        FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta\n"
                    f"    WHERE {mv_table}.{pk_col} = __mv_delta.{pk_col};"
                )

            del_zero_stmt = (
                f"DELETE FROM {mv_table} WHERE {mv_table}.{row_count_col} <= 0;"
                if grp_exprs
                else None
            )
            del_stmts = [minus_stmt] + ([del_zero_stmt] if del_zero_stmt else [])
            trunc_stmts = [f"DELETE FROM {mv_table};"] + (
                [initial_populate_sql] if not grp_exprs else []
            )

            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=[plus_stmt],
                del_stmts=del_stmts,
                upd_old_stmts=[minus_stmt],
                upd_new_stmts=[plus_stmt],
                upd_cleanup_old_stmts=[del_zero_stmt] if del_zero_stmt else [],
                trunc_stmts=trunc_stmts,
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
            auxiliary_table_ddls=[],
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


    # =========================================================================
    # TIER 1C: Z-Set Set Operations (UNION ALL, UNION, INTERSECT, EXCEPT),
    #          SELECT DISTINCT & PK-Lineage Projections
    # =========================================================================

    def _parse_setop_branches(self) -> Optional[List[Tuple[str, ViewSpec]]]:
        raw = self.view.raw_select_sql.strip()
        op_matches = _find_all_top_level_keywords(
            raw,
            r"\b(?:UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b",
        )
        if not op_matches:
            return None

        ops = [re.sub(r"\s+", " ", m.group(0).strip().upper()) for m in op_matches]
        if len(set(ops)) != 1:
            return None
        op_type = ops[0]

        branch_sqls: List[str] = []
        last_pos = 0
        for m in op_matches:
            branch_sqls.append(raw[last_pos : m.start()].strip())
            last_pos = m.end()
        branch_sqls.append(raw[last_pos:].strip())

        if op_type in ("INTERSECT", "EXCEPT") and len(branch_sqls) != 2:
            return None

        branches: List[Tuple[str, ViewSpec]] = []
        col_count: Optional[int] = None
        for idx, b_sql in enumerate(branch_sqls):
            try:
                bs = _parse_single_create_view_stmt(
                    f"CREATE VIEW __set_b_{idx} AS {b_sql}",
                    known_tables=set(self._schemas_lower.keys()),
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
                or any(it.is_aggregate or it.raw_expr.strip() == "*" for it in bs.select_items)
                or bs.base_table.lower() not in self._schemas_lower
            ):
                return None
            if bs.where_clause and re.search(r"\(\s*SELECT\b", bs.where_clause, flags=re.IGNORECASE):
                return None
            if col_count is None:
                col_count = len(bs.select_items)
            elif len(bs.select_items) != col_count:
                return None
            if op_type == "UNION ALL":
                ts = self._schemas_lower[bs.base_table.lower()]
                if not any(c.primary_key for c in ts.columns.values()):
                    return None
            branches.append((op_type, bs))
        return branches

    @staticmethod
    def _rewrite_branch_expr(expr: str, ts: TableSchema, alias: str, record_prefix: Optional[str]) -> str:
        if not record_prefix:
            return expr
        col_map = {c.lower(): c for c in ts.columns}
        tokens = re.split(r"('(?:''|[^'])*')", expr)
        out: List[str] = []
        for idx_t, tok in enumerate(tokens):
            if idx_t % 2 == 1:
                out.append(tok)
                continue
            mod = tok
            for qual in {ts.name, alias}:
                if qual:
                    mod = re.sub(
                        rf"\b{re.escape(qual)}\.([A-Za-z_][A-Za-z0-9_]*)\b",
                        rf"{record_prefix}.\1",
                        mod,
                        flags=re.IGNORECASE,
                    )

            def _repl_c(m_id: re.Match[str]) -> str:
                w = m_id.group(1)
                if w.lower() in col_map:
                    return f"{record_prefix}.{col_map[w.lower()]}"
                return w

            mod = re.sub(r"(?<![.\w])\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*[.(])", _repl_c, mod)
            out.append(mod)
        return "".join(out)

    def _compile_tier1_zset_setop(self) -> MaterializedViewPlan:
        branches = self._parse_setop_branches()
        assert branches is not None
        op_type = branches[0][0]
        first_bs = branches[0][1]
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in first_bs.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""

        col_defs_sql: List[str] = [f"__mv_pk {pk_type} NOT NULL PRIMARY KEY"]
        indexed_cols_lower = self._indexed_column_names_lower()
        for it in first_bs.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{it.alias} {dt}")

        if op_type in ("UNION", "UNION ALL"):
            col_defs_sql.append(f"__mv_row_count {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = None
        elif op_type == "INTERSECT":
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = "__mv_cnt_0 > 0 AND __mv_cnt_1 > 0"
        else:  # EXCEPT
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            extra_filter = "__mv_cnt_0 > 0 AND __mv_cnt_1 = 0"

        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    " + ",\n    ".join(col_defs_sql) + "\n);"
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                extra_where_filter=extra_filter,
            )
        )

        initial_populate_sqls: List[str] = []
        by_table_branches: Dict[str, List[Tuple[int, ViewSpec]]] = {}
        for b_idx, (_, bs) in enumerate(branches):
            by_table_branches.setdefault(bs.base_table, []).append((b_idx, bs))

        if op_type == "UNION ALL":
            for b_idx, (_, bs) in enumerate(branches):
                ts = self._schemas_lower[bs.base_table.lower()]
                pk_names = [c.name for c in ts.columns.values() if c.primary_key]
                pk_expr = self._canonical_group_pk_expr([f"'B{b_idx}'", *pk_names])
                sel_exprs = [it.raw_expr for it in bs.select_items]
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                initial_populate_sqls.append(
                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                    f"SELECT {pk_expr}, {', '.join(sel_exprs)}, 1 FROM {bs.base_table}{w_sql};"
                )
        elif op_type == "UNION":
            u_parts: List[str] = []
            for _, bs in branches:
                sel_aliased = ", ".join(
                    f"{it.raw_expr} AS {col}" for it, col in zip(bs.select_items, user_columns)
                )
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                u_parts.append(f"    SELECT {sel_aliased} FROM {bs.base_table}{w_sql}")
            pk_u = self._canonical_group_pk_expr(user_columns)
            grp_u = ", ".join(self._group_by_exprs_for_query(user_columns, include_canonical_pk=True))
            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                f"SELECT {pk_u}, {', '.join(user_columns)}, COUNT(*) FROM (\n"
                + "\n    UNION ALL\n".join(u_parts)
                + f"\n) AS __mv_u\nGROUP BY {grp_u};"
            )
        else:  # INTERSECT or EXCEPT
            u_parts = []
            for b_idx, (_, bs) in enumerate(branches):
                sel_aliased = ", ".join(
                    f"{it.raw_expr} AS {col}" for it, col in zip(bs.select_items, user_columns)
                )
                b0 = "1" if b_idx == 0 else "0"
                b1 = "1" if b_idx == 1 else "0"
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                u_parts.append(
                    f"    SELECT {sel_aliased}, {b0} AS __mv_b0, {b1} AS __mv_b1 FROM {bs.base_table}{w_sql}"
                )
            pk_u = self._canonical_group_pk_expr(user_columns)
            grp_u = ", ".join(self._group_by_exprs_for_query(user_columns, include_canonical_pk=True))
            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                f"SELECT {pk_u}, {', '.join(user_columns)}, SUM(__mv_b0), SUM(__mv_b1) FROM (\n"
                + "\n    UNION ALL\n".join(u_parts)
                + f"\n) AS __mv_u\nGROUP BY {grp_u};"
            )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for base_t, b_list in by_table_branches.items():
            ts = self._schemas_lower[base_t.lower()]
            ins_stmts: List[str] = []
            del_stmts: List[str] = []
            upd_old_stmts: List[str] = []
            upd_new_stmts: List[str] = []
            cleanup_stmts: List[str] = []

            for b_idx, bs in b_list:
                b_alias = bs.joins[0].alias if bs.joins else base_t
                w_new = (
                    self._rewrite_branch_expr(bs.where_clause, ts, b_alias, "NEW")
                    if bs.where_clause
                    else "TRUE"
                )
                w_old = (
                    self._rewrite_branch_expr(bs.where_clause, ts, b_alias, "OLD")
                    if bs.where_clause
                    else "TRUE"
                )
                new_vals = [
                    self._rewrite_branch_expr(it.raw_expr, ts, b_alias, "NEW")
                    for it in bs.select_items
                ]
                old_vals = [
                    self._rewrite_branch_expr(it.raw_expr, ts, b_alias, "OLD")
                    for it in bs.select_items
                ]

                if op_type == "UNION ALL":
                    pk_names = [c.name for c in ts.columns.values() if c.primary_key]
                    pk_new = self._canonical_group_pk_expr([f"'B{b_idx}'", *[f"NEW.{c}" for c in pk_names]])
                    pk_old = self._canonical_group_pk_expr([f"'B{b_idx}'", *[f"OLD.{c}" for c in pk_names]])
                    if self.dialect == Dialect.MYSQL:
                        upd_set = ", ".join(f"{c} = VALUES({c})" for c in user_columns)
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                        )
                    else:
                        upd_set = ", ".join(f"{c} = excluded.{c}" for c in user_columns)
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                            f"    WHERE {w_new}\n"
                            f"    ON CONFLICT (__mv_pk) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                        )
                    minus_s = (
                        f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                        f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                    )
                    clean_s = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
                    )
                elif op_type == "UNION":
                    pk_new = self._canonical_group_pk_expr(new_vals)
                    pk_old = self._canonical_group_pk_expr(old_vals)
                    if self.dialect == Dialect.MYSQL:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON DUPLICATE KEY UPDATE __mv_row_count = {mv_table}.__mv_row_count + 1;"
                        )
                    else:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                            f"    WHERE {w_new}\n"
                            f"    ON CONFLICT (__mv_pk) DO UPDATE SET __mv_row_count = {mv_table}.__mv_row_count + 1;"
                        )
                    minus_s = (
                        f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                        f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                    )
                    clean_s = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
                    )
                else:  # INTERSECT or EXCEPT
                    pk_new = self._canonical_group_pk_expr(new_vals)
                    pk_old = self._canonical_group_pk_expr(old_vals)
                    cnt_col = f"__mv_cnt_{b_idx}"
                    init_c0 = "1" if b_idx == 0 else "0"
                    init_c1 = "1" if b_idx == 1 else "0"
                    if self.dialect == Dialect.MYSQL:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, {init_c0}, {init_c1}{from_dual}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON DUPLICATE KEY UPDATE {cnt_col} = {mv_table}.{cnt_col} + 1;"
                        )
                    else:
                        plus_s = (
                            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                            f"    SELECT {pk_new}, {', '.join(new_vals)}, {init_c0}, {init_c1}\n"
                            f"    WHERE {w_new}\n"
                            f"    ON CONFLICT (__mv_pk) DO UPDATE SET {cnt_col} = {mv_table}.{cnt_col} + 1;"
                        )
                    minus_s = (
                        f"UPDATE {mv_table} SET {cnt_col} = {mv_table}.{cnt_col} - 1\n"
                        f"    WHERE ({w_old}) AND {mv_table}.__mv_pk = ({pk_old});"
                    )
                    clean_s = (
                        f"DELETE FROM {mv_table}\n"
                        f"    WHERE ({w_old}) AND ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_cnt_0 <= 0 AND {mv_table}.__mv_cnt_1 <= 0;"
                    )

                ins_stmts.append(plus_s)
                del_stmts.extend([minus_s, clean_s])
                upd_old_stmts.append(minus_s)
                upd_new_stmts.append(plus_s)
                cleanup_stmts.append(clean_s)

            trunc_stmts = [f"DELETE FROM {mv_table};", *initial_populate_sqls]
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_stmts,
                del_stmts=del_stmts,
                upd_old_stmts=upd_old_stmts,
                upd_new_stmts=upd_new_stmts,
                upd_cleanup_old_stmts=cleanup_stmts,
                trunc_stmts=trunc_stmts,
            )
            trigger_ddls.extend(t_ddls)
            trig_names.extend(t_ns)
            fn_names.extend(f_ns)

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            chunk_sql_templates=initial_populate_sqls,
        )
        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=[],
            index_ddls=index_ddls,
            initial_populate_sqls=initial_populate_sqls,
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

    def _compile_tier1_zset_distinct(self) -> MaterializedViewPlan:
        # `SELECT DISTINCT c1, c2 FROM t WHERE ...` is isomorphic to `GROUP BY c1, c2` with hidden `__mv_row_count`
        self._effective_group_by_exprs = tuple(it.raw_expr for it in self.view.select_items)
        return self._compile_tier1_algebraic()

    def _compile_tier1_zset_projection(self) -> MaterializedViewPlan:
        assert self.primary_schema is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        base_t = self.view.base_table
        user_columns = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""

        pk_names = [c.name for c in self.primary_schema.columns.values() if c.primary_key]
        col_defs_sql: List[str] = [f"__mv_pk {pk_type} NOT NULL PRIMARY KEY"]
        indexed_cols_lower = self._indexed_column_names_lower()
        for it in self.view.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{it.alias} {dt}")
        col_defs_sql.append(f"__mv_row_count {self._int_type()} NOT NULL DEFAULT 0")

        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    " + ",\n    ".join(col_defs_sql) + "\n);"
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        pk_expr_base = self._canonical_group_pk_expr(pk_names)
        sel_exprs = [it.raw_expr for it in self.view.select_items]
        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        initial_populate_sql = (
            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
            f"SELECT {pk_expr_base}, {', '.join(sel_exprs)}, 1 FROM {base_t}{w_sql};"
        )

        pk_new = self._canonical_group_pk_expr([f"NEW.{c}" for c in pk_names])
        pk_old = self._canonical_group_pk_expr([f"OLD.{c}" for c in pk_names])
        new_vals = [self._rewrite_expr_for_record(it.raw_expr, "NEW") for it in self.view.select_items]

        if self.dialect == Dialect.MYSQL:
            upd_set = ", ".join(f"{c} = VALUES({c})" for c in user_columns)
            plus_stmt = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                f"    SELECT {pk_new}, {', '.join(new_vals)}, 1{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
            )
        else:
            upd_set = ", ".join(f"{c} = excluded.{c}" for c in user_columns)
            plus_stmt = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                f"    SELECT {pk_new}, {', '.join(new_vals)}, 1\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT (__mv_pk) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
            )
        minus_stmt = (
            f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
            f"    WHERE {mv_table}.__mv_pk = ({pk_old});"
        )
        clean_stmt = (
            f"DELETE FROM {mv_table}\n"
            f"    WHERE ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"
        )

        where_new = (
            self._rewrite_expr_for_record(self.view.where_clause, "NEW")
            if self.view.where_clause
            else None
        )
        where_old = (
            self._rewrite_expr_for_record(self.view.where_clause, "OLD")
            if self.view.where_clause
            else None
        )

        trigger_ddls, trig_names, fn_names = self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=[plus_stmt],
            del_stmts=[minus_stmt, clean_stmt],
            upd_old_stmts=[minus_stmt],
            upd_new_stmts=[plus_stmt],
            upd_cleanup_old_stmts=[clean_stmt],
            trunc_stmts=[f"DELETE FROM {mv_table};"],
            where_new=where_new,
            where_old=where_old,
        )

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            chunk_sql_templates=[initial_populate_sql],
        )
        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=[],
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

    def _compile_tier1_union_all_agg(self) -> MaterializedViewPlan:
        """
        Compiles an aggregation query over a `UNION ALL` CTE or derived subquery into Tier 1 O(1)
        per-branch algebraic delta triggers.
        """
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = list(self._working_select_items)
        grp_exprs = list(self._effective_group_by_exprs)
        inner_u_sql = self._union_all_inner_sql
        u_alias = self._union_all_sub_alias
        branch_specs = self._union_all_branch_specs

        count_star_alias: Optional[str] = None
        for item in work_items:
            if (
                item.is_aggregate
                and item.agg_func == AggFunc.COUNT_STAR
                and not item.filter_where
            ):
                count_star_alias = item.alias
                break
        row_count_col = count_star_alias or "__mv_row_count"

        col_defs_sql: List[str] = []
        user_columns: List[str] = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"

        if not grp_exprs:
            col_defs_sql.append("__mv_global_id INTEGER NOT NULL PRIMARY KEY")
        else:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden_grp: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_grp:
                seen_hidden_grp.add(mv_col.lower())
                fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                dt = self._infer_expr_storage_type(fake_it)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{mv_col} {dt}")

        def _expr_key(item: SelectItem) -> str:
            eff = self._effective_agg_value_expr(item, record_prefix=None).strip().lower()
            return f"{'dist:' if item.is_distinct else ''}{eff}"

        col_nonnull_count_map: Dict[str, str] = {}
        col_sum_map: Dict[str, str] = {}
        col_sum2_map: Dict[str, str] = {}
        col_true_cnt_map: Dict[str, str] = {}

        for item in work_items:
            if not item.is_aggregate:
                continue
            ekey = _expr_key(item)
            if item.agg_func == AggFunc.COUNT_COL:
                col_nonnull_count_map[ekey] = item.alias
            elif item.agg_func == AggFunc.SUM:
                col_sum_map[ekey] = item.alias

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        # Infer aggregate types from first branch
        first_b = branch_specs[0]
        first_b_map = {it.alias.lower(): it.raw_expr for it in first_b.select_items}

        for item in work_items:
            if not item.is_aggregate:
                dt = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{item.alias} {dt}")
            else:
                # Check if arg_expr maps to an integer column across all branches
                storage_type = self._infer_expr_storage_type(item)
                if item.agg_func in (AggFunc.SUM, AggFunc.MIN, AggFunc.MAX) and item.arg_expr:
                    arg_id = _unquote_ident(item.arg_expr).lower()
                    if arg_id in first_b_map:
                        b_types = []
                        for bs in branch_specs:
                            b_m = {bi.alias.lower(): bi for bi in bs.select_items}
                            if arg_id in b_m:
                                bi_item = b_m[arg_id]
                                ts_b = self._schemas_lower[bs.base_table.lower()]
                                c_b = ts_b.columns.get(bi_item.source_col or "") if bi_item.source_col else None
                                if c_b:
                                    b_types.append(self._normalize_dialect_type(c_b.data_type))
                                elif re.match(r"^\d+$", bi_item.raw_expr.strip()):
                                    b_types.append("INT")
                                else:
                                    b_types.append(self._float_type())
                        if b_types and all("INT" in bt.upper() for bt in b_types):
                            storage_type = self._int_type() if item.agg_func == AggFunc.SUM else b_types[0]
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{item.alias} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{item.alias} {storage_type}")

        if count_star_alias is None:
            col_defs_sql.append(f"{row_count_col} {self._int_type()} NOT NULL DEFAULT 0")

        aux_idx = 0
        for item in work_items:
            if not item.is_aggregate or item.agg_func is None:
                continue
            ekey = _expr_key(item)
            safe_id = _sanitize_ident(item.alias)
            if item.agg_func in (
                AggFunc.SUM,
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
                AggFunc.BIT_XOR,
                AggFunc.BOOL_AND,
                AggFunc.BOOL_OR,
            ) or (item.agg_func == AggFunc.COUNT_STAR and item.filter_where):
                if item.agg_func == AggFunc.COUNT_STAR and item.filter_where:
                    col_nonnull_count_map[ekey] = item.alias
                elif ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    cnt_col = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = cnt_col
                    col_defs_sql.append(f"{cnt_col} {self._int_type()} NOT NULL DEFAULT 0")

            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    sum_col = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = sum_col
                    col_defs_sql.append(f"{sum_col} {self._float_type()}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    sum2_col = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._float_type()}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    " + ",\n    ".join(col_defs_sql) + "\n);"
        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col
        }
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=False,
            backing_col_map=backing_col_map,
        )
        reader_select_override = [
            self._compound_reader_exprs.get(u_col, u_col) for u_col in user_columns
        ]
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                select_exprs_override=reader_select_override,
                extra_where_filter=self._rewritten_having_clause,
            )
        )

        target_cols: List[str] = ["__mv_global_id" if not grp_exprs else "__mv_pk"]
        select_exprs: List[str] = [
            "1" if not grp_exprs else self._canonical_group_pk_expr(grp_exprs)
        ]

        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                target_cols.append(mv_col)
                select_exprs.append(g_expr)

        for item in work_items:
            target_cols.append(item.alias)
            if not item.is_aggregate:
                select_exprs.append(item.raw_expr)
            else:
                select_exprs.append(self._sql_backfill_agg_expr(item))

        if count_star_alias is None:
            target_cols.append(row_count_col)
            select_exprs.append("COUNT(*)")

        already_in_target = set(target_cols)
        ekey_to_item: Dict[str, SelectItem] = {
            _expr_key(it): it for it in work_items if it.is_aggregate
        }
        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(cnt_col)
                select_exprs.append(f"COUNT({eff})")
                already_in_target.add(cnt_col)
        for ekey, sum_col in col_sum_map.items():
            if sum_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(sum_col)
                select_exprs.append(f"SUM({eff})")
                already_in_target.add(sum_col)
        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(sum2_col)
                select_exprs.append(f"SUM(1.0 * ({eff}) * ({eff}))")
                already_in_target.add(sum2_col)
        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in already_in_target:
                it = ekey_to_item[ekey]
                eff = self._effective_agg_value_expr(it, record_prefix=None)
                target_cols.append(tcnt_col)
                select_exprs.append(
                    f"COALESCE(SUM(CASE WHEN ({eff}) IS NOT NULL AND ({eff}) THEN 1 ELSE 0 END), 0)"
                )
                already_in_target.add(tcnt_col)

        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        grp_for_q = self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=True)
        g_sql = f"\nGROUP BY {', '.join(grp_for_q)}" if grp_for_q else ""

        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
            f"SELECT {', '.join(select_exprs)} FROM (\n{inner_u_sql}\n) AS {u_alias}{w_sql}{g_sql};"
        )

        # Group branches by base table and generate per-branch delta statements
        by_table: Dict[str, List[ViewSpec]] = {}
        for bs in branch_specs:
            by_table.setdefault(bs.base_table, []).append(bs)

        orig_view = self.view
        orig_work_items = list(self._working_select_items)
        orig_grp_exprs = tuple(self._effective_group_by_exprs)
        orig_primary_schema = self.primary_schema
        self._uall_orig_grp_exprs = orig_grp_exprs

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for base_t, b_list in by_table.items():
            ts = self._schemas_lower[base_t.lower()]
            ins_all: List[str] = []
            del_all: List[str] = []
            upd_old_all: List[str] = []
            upd_new_all: List[str] = []
            cleanup_all: List[str] = []

            for bs in b_list:
                b_alias = bs.joins[0].alias if bs.joins else base_t
                col_subst: Dict[str, str] = {}
                for col_i, b_it in enumerate(bs.select_items):
                    u_col_name = first_b.select_items[col_i].alias.lower()
                    cleaned = re.sub(
                        rf"\b{re.escape(b_alias)}\.",
                        "",
                        b_it.raw_expr.strip(),
                        flags=re.IGNORECASE,
                    )
                    col_subst[u_col_name] = f"({cleaned})"

                def _subst_u(frag: Optional[str]) -> Optional[str]:
                    if frag is None:
                        return None
                    tokens = re.split(r"('(?:''|[^'])*')", frag)
                    out = []
                    for idx_t, tok in enumerate(tokens):
                        if idx_t % 2 == 1:
                            out.append(tok)
                            continue
                        mod = re.sub(rf"\b{re.escape(u_alias)}\.", "", tok, flags=re.IGNORECASE)

                        def _r_id(m_id: re.Match[str]) -> str:
                            w = m_id.group(1)
                            return col_subst.get(w.lower(), w)

                        mod = re.sub(r"(?<![.\w])\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", _r_id, mod)
                        out.append(mod)
                    return "".join(out)

                mapped_work_items: List[SelectItem] = []
                orig_item_map: Dict[int, SelectItem] = {}
                mapped_nonnull: Dict[str, str] = {}
                mapped_sum: Dict[str, str] = {}
                mapped_sum2: Dict[str, str] = {}
                mapped_tcnt: Dict[str, str] = {}

                for orig_it in orig_work_items:
                    if not orig_it.is_aggregate:
                        new_it = SelectItem(
                            raw_expr=_subst_u(orig_it.raw_expr) or orig_it.raw_expr,
                            alias=orig_it.alias,
                            is_aggregate=False,
                        )
                    else:
                        new_it = SelectItem(
                            raw_expr=_subst_u(orig_it.raw_expr) or orig_it.raw_expr,
                            alias=orig_it.alias,
                            is_aggregate=True,
                            agg_func=orig_it.agg_func,
                            arg_expr=_subst_u(orig_it.arg_expr) if orig_it.arg_expr else orig_it.arg_expr,
                            extra_args=orig_it.extra_args,
                            is_distinct=orig_it.is_distinct,
                            filter_where=_subst_u(orig_it.filter_where),
                        )
                        orig_ekey = _expr_key(orig_it)
                        new_ekey = _expr_key(new_it)
                        if orig_ekey in col_nonnull_count_map:
                            mapped_nonnull[new_ekey] = col_nonnull_count_map[orig_ekey]
                        if orig_ekey in col_sum_map:
                            mapped_sum[new_ekey] = col_sum_map[orig_ekey]
                        if orig_ekey in col_sum2_map:
                            mapped_sum2[new_ekey] = col_sum2_map[orig_ekey]
                        if orig_ekey in col_true_cnt_map:
                            mapped_tcnt[new_ekey] = col_true_cnt_map[orig_ekey]
                    orig_item_map[id(new_it)] = orig_it
                    mapped_work_items.append(new_it)

                mapped_grps = tuple(_subst_u(g) or g for g in orig_grp_exprs)
                b_w = (
                    re.sub(rf"\b{re.escape(b_alias)}\.", "", bs.where_clause, flags=re.IGNORECASE)
                    if bs.where_clause
                    else None
                )
                o_w = _subst_u(orig_view.where_clause)
                comb_w = f"({b_w}) AND ({o_w})" if (b_w and o_w) else (b_w or o_w)

                # Temporarily bind branch context
                self.primary_schema = ts
                self._working_select_items = mapped_work_items
                self._effective_group_by_exprs = mapped_grps
                self._uall_orig_item_map = orig_item_map
                self.view = ViewSpec(
                    view_name=orig_view.view_name,
                    raw_select_sql=orig_view.raw_select_sql,
                    base_table=base_t,
                    joins=(JoinSpec(join_type="FROM", table_name=base_t, alias=base_t),),
                    referenced_tables=(base_t,),
                    select_items=tuple(mapped_work_items),
                    where_clause=orig_view.where_clause,
                    group_by_cols=mapped_grps,
                )

                plus_b = self._build_plus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="NEW",
                )
                minus_del_b, _ = self._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="OLD",
                    include_delete_zero_mv=True,
                )
                minus_upd_b, del_zero_b = self._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    mapped_nonnull,
                    mapped_sum,
                    mapped_sum2,
                    mapped_tcnt,
                    {},
                    _expr_key,
                    record_prefix="OLD",
                    include_delete_zero_mv=False,
                )

                w_new_b = self._rewrite_expr_for_record(comb_w, "NEW") if comb_w else None
                w_old_b = self._rewrite_expr_for_record(comb_w, "OLD") if comb_w else None

                ins_all.extend(self._apply_where_guard_to_stmt(s, w_new_b) for s in plus_b)
                del_all.extend(self._apply_where_guard_to_stmt(s, w_old_b) for s in minus_del_b)
                upd_old_all.extend(self._apply_where_guard_to_stmt(s, w_old_b) for s in minus_upd_b)
                upd_new_all.extend(self._apply_where_guard_to_stmt(s, w_new_b) for s in plus_b)
                if del_zero_b:
                    cleanup_all.append(self._apply_where_guard_to_stmt(del_zero_b, w_old_b))

            # Restore original view context
            self.view = orig_view
            self._working_select_items = orig_work_items
            self._effective_group_by_exprs = orig_grp_exprs
            self.primary_schema = orig_primary_schema

            trunc_stmts = [f"DELETE FROM {mv_table};", initial_populate_sql]
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_all,
                del_stmts=del_all,
                upd_old_stmts=upd_old_all,
                upd_new_stmts=upd_new_all,
                upd_cleanup_old_stmts=cleanup_all,
                trunc_stmts=trunc_stmts,
            )
            trigger_ddls.extend(t_ddls)
            trig_names.extend(t_ns)
            fn_names.extend(f_ns)

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            chunk_sql_templates=[initial_populate_sql],
        )
        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=[],
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


    # =========================================================================
    # TIER 2: Group-Scoped Delta Recomputation (Holistic Aggs, LEFT JOINs, etc.)
    # =========================================================================

    def _can_push_down_group_filter_into_where(self) -> bool:
        if not self.view.where_clause:
            return True
        return not bool(re.search(r"\b(?:EXISTS|IN\s*\(\s*SELECT)\b", self.view.where_clause, flags=re.IGNORECASE))

    def _inject_where_into_source_query(self, source_query_sql: str, extra_predicate: str) -> str:
        s = source_query_sql.strip().rstrip(";")
        m_where = _find_top_level_keyword(s, r"\bWHERE\b")
        m_group = _find_top_level_keyword(s, r"\bGROUP\s+BY\b")
        if m_where:
            insert_pos = m_where.end()
            before = s[: m_where.start()].rstrip()
            if m_group:
                existing_w = s[insert_pos : m_group.start()].strip()
                after = s[m_group.start() :]
                return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})\n{after}"
            existing_w = s[insert_pos:].strip()
            return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})"
        if m_group:
            before = s[: m_group.start()].rstrip()
            after = s[m_group.start() :]
            return f"{before}\nWHERE ({extra_predicate})\n{after}"
        return f"{s}\nWHERE ({extra_predicate})"

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
        if self.view.where_clause and re.search(r"\b(?:EXISTS|IN)\s*\(", self.view.where_clause, flags=re.IGNORECASE):
            if base_t.lower() != self.view.base_table.lower():
                return None

        def _lhs_expr(g_expr: str, mv_col: str) -> str:
            if lhs_table_alias:
                return f"{lhs_table_alias}.{mv_col}"
            return f"({g_expr})"

        if len(self.view.joins) == 1:
            conds: List[str] = []
            for g_expr, mv_col in group_mv_cols:
                rhs_e = self._rewrite_expr_for_record(g_expr, record_prefix)
                conds.append(self._null_safe_eq(_lhs_expr(g_expr, mv_col), f"({rhs_e})", is_nullable=True))
            return " AND ".join(conds)

        target_join: Optional[JoinSpec] = None
        for j in self.view.joins:
            if j.table_name.lower() == base_t.lower():
                target_join = j
                break
        if target_join is None:
            return None

        # If base_t is the table that owns all group columns directly, compare directly
        target_alias = target_join.alias
        all_owned = True
        direct_conds: List[str] = []
        ts_target = self._schemas_lower[base_t.lower()]
        for g_expr, mv_col in group_mv_cols:
            gs = g_expr.strip()
            if "." in gs:
                p_al, c_part = gs.split(".", 1)
                if p_al.strip().lower() in (target_alias.lower(), base_t.lower()):
                    direct_conds.append(
                        self._null_safe_eq(
                            _lhs_expr(g_expr, mv_col),
                            f"({record_prefix}.{_unquote_ident(c_part)})",
                            is_nullable=True,
                        )
                    )
                else:
                    all_owned = False
                    break
            elif _unquote_ident(gs) in ts_target.columns:
                direct_conds.append(
                    self._null_safe_eq(
                        _lhs_expr(g_expr, mv_col),
                        f"({record_prefix}.{_unquote_ident(gs)})",
                        is_nullable=True,
                    )
                )
            else:
                all_owned = False
                break

        if all_owned and direct_conds:
            return " AND ".join(direct_conds)

        # Otherwise build an EXISTS subquery over the join graph with `target_join` replaced by `(SELECT NEW/OLD.*)`
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        rec_proj = ", ".join(f"{record_prefix}.{c} AS {c}" for c in ts_target.columns)
        alias_remap: Dict[str, str] = {}
        for j in self.view.joins:
            alias_remap[j.alias.lower()] = f"__mv_j_{j.alias}"
            alias_remap[j.table_name.lower()] = f"__mv_j_{j.alias}"

        def _remap_aliases_in_sql(frag: str) -> str:
            tokens = re.split(r"('(?:''|[^'])*')", frag)
            out: List[str] = []
            for idx_t, tok in enumerate(tokens):
                if idx_t % 2 == 1:
                    out.append(tok)
                    continue
                mod = tok
                for old_a, new_a in sorted(alias_remap.items(), key=lambda x: len(x[0]), reverse=True):
                    mod = re.sub(
                        rf"\b{re.escape(old_a)}\s*\.",
                        f"{new_a}.",
                        mod,
                        flags=re.IGNORECASE,
                    )
                out.append(mod)
            return "".join(out)

        inline_row_table = f"(SELECT {rec_proj}{from_dual}) AS __mv_j_{target_alias}"
        join_parts: List[str] = []
        for idx, j in enumerate(self.view.joins):
            remapped_alias = f"__mv_j_{j.alias}"
            tbl_expr = (
                inline_row_table
                if j.alias.lower() == target_alias.lower()
                else f"{j.table_name} AS {remapped_alias}"
            )
            if idx == 0:
                join_parts.append(tbl_expr)
            else:
                on_sql = (
                    f" ON {_remap_aliases_in_sql(j.on_condition)}"
                    if j.on_condition
                    else ""
                )
                eff_join_type = (
                    "INNER JOIN"
                    if j.alias.lower() == target_alias.lower() and j.on_condition
                    else j.join_type
                )
                join_parts.append(f"{eff_join_type} {tbl_expr}{on_sql}")

        from_join_sql = " ".join(join_parts)
        left_join_remapped = {
            f"__mv_j_{j.alias}".lower()
            for j in self.view.joins
            if j.join_type == "LEFT JOIN"
        }
        match_conds: List[str] = []
        for g_expr, mv_col in group_mv_cols:
            remapped_g = _remap_aliases_in_sql(g_expr)
            lhs_e = _lhs_expr(g_expr, mv_col)
            eq_cond = self._null_safe_eq(lhs_e, f"({remapped_g})", is_nullable=True)
            if any(
                re.search(rf"\b{re.escape(la)}\.", remapped_g, flags=re.IGNORECASE)
                for la in left_join_remapped
            ):
                match_conds.append(f"({eq_cond} OR {lhs_e} IS NULL)")
            else:
                match_conds.append(eq_cond)

        exists_pred = (
            f"EXISTS (\n"
            f"        SELECT 1 FROM {from_join_sql}\n"
            f"        WHERE {' AND '.join(match_conds)}\n"
            f"    )"
        )
        if any(j.join_type == "LEFT JOIN" for j in self.view.joins) and target_join != self.view.joins[0]:
            null_preds = " AND ".join(
                f"{_lhs_expr(g_expr, mv_col)} IS NULL" for g_expr, mv_col in group_mv_cols
            )
            return f"({exists_pred} OR ({null_preds}))"
        return exists_pred

    def _build_tier2_triggers_for_table(
        self,
        view_name: str,
        mv_table: str,
        base_t: str,
        target_cols: List[str],
        group_mv_cols: List[Tuple[str, str]],
        source_query_sql: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        cols_csv = ", ".join(target_cols)
        src_cols_csv = ", ".join(f"__mv_src.{c}" for c in target_cols)

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
            if self.dialect == Dialect.POSTGRES and self.pg_use_transition_tables:
                upd_old_stmts = del_stmts
                upd_new_stmts = ins_stmts
            else:
                upd_old_stmts = []
                upd_new_stmts = upd_coalesced_stmts
        else:
            ins_stmts = full_refresh
            del_stmts = full_refresh
            upd_old_stmts = []
            upd_new_stmts = full_refresh

        return self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=ins_stmts,
            del_stmts=del_stmts,
            upd_old_stmts=upd_old_stmts,
            upd_new_stmts=upd_new_stmts,
            trunc_stmts=full_refresh,
        )

    def _compile_tier2_group_scoped(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in self.view.select_items]
        grp_exprs = self._effective_group_by_exprs

        target_cols: List[str] = list(user_columns)
        group_mv_cols: List[Tuple[str, str]] = []
        hidden_grp_defs: List[Tuple[str, str, str]] = []

        non_agg_map: Dict[str, str] = {}
        for it in self.view.select_items:
            if not it.is_aggregate:
                non_agg_map[it.raw_expr.strip().lower()] = it.alias
                if it.source_col:
                    non_agg_map[it.source_col.lower()] = it.alias

        for idx_g, g_expr in enumerate(grp_exprs):
            g_low = g_expr.strip().lower()
            g_unq = _unquote_ident(g_expr).lower() if self._is_simple_col_ref(g_expr) else ""
            if g_low in non_agg_map:
                mv_c = non_agg_map[g_low]
            elif g_unq and g_unq in non_agg_map:
                mv_c = non_agg_map[g_unq]
            else:
                mv_c = f"__mv_grp_{idx_g}"
                target_cols.append(mv_c)
                c_def = self._lookup_col_def(g_expr)
                dt = (
                    self._normalize_dialect_type(c_def.data_type)
                    if c_def
                    else ("VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT")
                )
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                hidden_grp_defs.append((mv_c, g_expr, dt))
            group_mv_cols.append((g_expr, mv_c))

        auto_grp_col_names = [mc for _, mc in group_mv_cols]
        indexed_cols_lower = self._indexed_column_names_lower(auto_grp_col_names)

        col_defs_sql: List[str] = []
        for it in self.view.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{it.alias} {dt}")
        for mv_c, _, dt in hidden_grp_defs:
            col_defs_sql.append(f"{mv_c} {dt}")

        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    " + ",\n    ".join(col_defs_sql) + "\n);"
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=auto_grp_col_names,
            has_pk_on_group_cols=False,
        )
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        raw_q = self.view.raw_select_sql.strip().rstrip(";")
        if hidden_grp_defs:
            m_from = _find_top_level_keyword(raw_q, r"\bFROM\b")
            assert m_from is not None
            extra_sel = ", ".join(f"{g_e} AS {mv_c}" for mv_c, g_e, _ in hidden_grp_defs)
            source_query_sql = f"{raw_q[:m_from.start()].rstrip()}, {extra_sel} {raw_q[m_from.start():]}"
        else:
            source_query_sql = raw_q

        cols_csv = ", ".join(target_cols)
        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({cols_csv})\n"
            f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init;"
        )
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        for ref_table in self.view.referenced_tables:
            t_ddls, t_ns, f_ns = self._build_tier2_triggers_for_table(
                view_name=view_name,
                mv_table=mv_table,
                base_t=ref_table,
                target_cols=target_cols,
                group_mv_cols=group_mv_cols,
                source_query_sql=source_query_sql,
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
            auxiliary_table_ddls=[],
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

    # =========================================================================
    # TIER 3: Universal Atomic View Sync (Arbitrary Engine View Definitions)
    # =========================================================================

    def _compile_tier3_atomic_sync(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        col_defs_sql: List[str] = []
        user_columns: List[str] = []

        indexed_cols_lower = self._indexed_column_names_lower()
        if len(self.view.select_items) == 1 and self.view.select_items[0].raw_expr.strip() == "*":
            for j in self.view.joins or []:
                t_schema = self._schemas_lower.get(j.table_name.lower())
                if t_schema:
                    for c_def in t_schema.columns.values():
                        if c_def.name not in user_columns:
                            user_columns.append(c_def.name)
                            dt = self._normalize_dialect_type(c_def.data_type)
                            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and c_def.name.lower() in indexed_cols_lower:
                                dt = "VARCHAR(255)"
                            col_defs_sql.append(f"{c_def.name} {dt}")
            if not user_columns and self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    user_columns.append(c_def.name)
                    dt = self._normalize_dialect_type(c_def.data_type)
                    if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and c_def.name.lower() in indexed_cols_lower:
                        dt = "VARCHAR(255)"
                    col_defs_sql.append(f"{c_def.name} {dt}")
        else:
            for item in self.view.select_items:
                user_columns.append(item.alias)
                col_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and col_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    col_type = "VARCHAR(255)"
                col_defs_sql.append(f"{item.alias} {col_type}")

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
        )

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        raw_query = self.view.raw_select_sql.strip().rstrip(";")
        cols_csv = ", ".join(user_columns)
        initial_populate_sql = f"INSERT INTO {mv_table} ({cols_csv})\n{raw_query};"

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{raw_query}\n) AS __mv_init\n"
                f"LIMIT ? OFFSET ?;"
            ),
        )

        refresh_stmts = [
            f"DELETE FROM {mv_table};",
            f"INSERT INTO {mv_table} ({cols_csv})\n    {raw_query};",
        ]

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        for ref_table in self.view.referenced_tables:
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=ref_table,
                ins_stmts=refresh_stmts,
                del_stmts=refresh_stmts,
                upd_old_stmts=[],
                upd_new_stmts=refresh_stmts,
                trunc_stmts=refresh_stmts,
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
            auxiliary_table_ddls=[],
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


# -----------------------------------------------------------------------------
# Database Connection, Schema & Safe Migration Manager (`MaterializedViewManager`)
# -----------------------------------------------------------------------------


class MaterializedViewManager:
    """
    Setup/migration manager that connects to a database (SQLite connection/path or
    PEP-249 DB-API 2.0 connection for PostgreSQL / MySQL) to:
      - Introspect live base table schemas (`fetch_schema`)
      - Maintain a persistent metadata registry (`__matview_catalog`) of installed MVs
      - Safely create, diff, update, or drop materialized views (`register_view`, `sync_views`, `drop_view`)
      - Enforce transparency limitation checks unless `ignore_limitations=True`
      - Perform incremental (chunked/batched) initial backfills (`backfill_incrementally`)
      - Execute high-throughput multi-statement bulk changes (`bulk_changes`)
    Once setup/backfill finishes, all views and triggers operate 100% natively inside
    the database engine with zero runtime dependency on this manager.
    """

    CATALOG_TABLE = f"{DEFAULT_OBJECT_PREFIX}catalog"

    def __init__(
        self,
        conn_or_url: sqlite3.Connection | str | Any,
        dialect: Dialect | str = Dialect.SQLITE,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        pg_use_transition_tables: bool = False,
    ) -> None:
        if isinstance(dialect, str):
            dialect = Dialect(dialect.lower())
        self.dialect = dialect
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.pg_use_transition_tables = pg_use_transition_tables
        self.catalog_table = (
            f"{self.object_prefix}catalog" if self.object_prefix else "matview_catalog"
        )

        if isinstance(conn_or_url, str):
            path = conn_or_url.replace("sqlite:///", "")
            self.conn = sqlite3.connect(path)
        else:
            self.conn = conn_or_url

        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        cur = self.conn.cursor()
        if self.dialect == Dialect.SQLITE:
            try:
                cur.execute("PRAGMA recursive_triggers = ON;")
            except Exception:
                pass
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.catalog_table} (
                view_name VARCHAR(255) NOT NULL PRIMARY KEY,
                view_sql TEXT NOT NULL,
                strategy VARCHAR(64) NOT NULL,
                signature_hash VARCHAR(64) NOT NULL,
                backing_table VARCHAR(255) NOT NULL,
                aux_tables_json TEXT NOT NULL,
                trigger_names_json TEXT NOT NULL,
                function_names_json TEXT NOT NULL,
                trigger_ddls_json TEXT NOT NULL
            );
            """
        )
        self._commit()

    def _commit(self) -> None:
        if hasattr(self.conn, "commit"):
            self.conn.commit()

    def _param_placeholder(self) -> str:
        return "?" if self.dialect == Dialect.SQLITE else "%s"

    def _resolve_catalog_name(self, name: str, catalog: Dict[str, Dict[str, Any]]) -> str:
        if name in catalog:
            return name
        if self.object_prefix:
            prefixed = (
                name
                if name.startswith(self.object_prefix)
                else f"{self.object_prefix}{name}"
            )
            if prefixed in catalog:
                return prefixed
            if name.startswith(self.object_prefix):
                unprefixed = name[len(self.object_prefix) :]
                if unprefixed in catalog:
                    return unprefixed
        return name

    def fetch_schema(self, extra_schema_sql: Optional[str] = None) -> Dict[str, TableSchema]:
        """Fetches the live schema of all user base tables from the connected database."""
        schemas: Dict[str, TableSchema] = {}
        if self.dialect == Dialect.SQLITE:
            schemas = introspect_sqlite_schemas(self.conn, object_prefix=self.object_prefix)
            schemas.pop(self.catalog_table, None)
        elif self.dialect == Dialect.POSTGRES:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT kcu.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = current_schema()
                """
            )
            pk_pairs = {(str(r[0]).lower(), str(r[1]).lower()) for r in (cur.fetchall() or [])}
            cur.execute(
                """
                SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.character_maximum_length
                FROM information_schema.columns c
                JOIN information_schema.tables t
                  ON c.table_schema = t.table_schema AND c.table_name = t.table_name
                WHERE c.table_schema = current_schema()
                  AND t.table_type = 'BASE TABLE'
                  AND c.table_name NOT LIKE '__matview_%'
                  AND c.table_name NOT LIKE '__mv_%'
                ORDER BY c.table_name, c.ordinal_position
                """
            )
            grouped: Dict[str, Dict[str, ColumnDef]] = {}
            for row in (cur.fetchall() or []):
                t_name, c_name, d_type, is_null, char_len = row[0], row[1], row[2], row[3], row[4]
                if t_name == self.catalog_table or (
                    self.object_prefix and str(t_name).startswith(self.object_prefix)
                ):
                    continue
                dt_up = str(d_type).upper()
                if dt_up in ("CHARACTER VARYING", "VARCHAR") and char_len:
                    dt_up = f"VARCHAR({int(char_len)})"
                elif dt_up == "CHARACTER VARYING":
                    dt_up = "TEXT"
                is_pk = (str(t_name).lower(), str(c_name).lower()) in pk_pairs
                grouped.setdefault(t_name, {})[c_name] = ColumnDef(
                    name=c_name,
                    data_type=dt_up,
                    not_null=(str(is_null).upper() == "NO") or is_pk,
                    primary_key=is_pk,
                )
            for t_name, cols in grouped.items():
                schemas[t_name] = TableSchema(name=t_name, columns=cols)
        else:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_key, c.column_type
                FROM information_schema.columns c
                JOIN information_schema.tables t
                  ON c.table_schema = t.table_schema AND c.table_name = t.table_name
                WHERE c.table_schema = DATABASE()
                  AND t.table_type = 'BASE TABLE'
                  AND c.table_name NOT LIKE '__matview_%'
                  AND c.table_name NOT LIKE '__mv_%'
                ORDER BY c.table_name, c.ordinal_position
                """
            )
            grouped = {}
            for row in (cur.fetchall() or []):
                t_name, c_name, d_type, is_null, col_key = row[0], row[1], row[2], row[3], row[4]
                col_type = row[5] if len(row) > 5 else None
                if t_name == self.catalog_table or (
                    self.object_prefix and str(t_name).startswith(self.object_prefix)
                ):
                    continue
                dt_up = str(d_type).upper()
                if dt_up in ("VARCHAR", "CHAR") and col_type:
                    dt_up = str(col_type).upper()
                is_pk = str(col_key or "").upper() == "PRI"
                grouped.setdefault(t_name, {})[c_name] = ColumnDef(
                    name=c_name,
                    data_type=dt_up,
                    not_null=(str(is_null).upper() == "NO") or is_pk,
                    primary_key=is_pk,
                )
            for t_name, cols in grouped.items():
                schemas[t_name] = TableSchema(name=t_name, columns=cols)

        if extra_schema_sql:
            schemas.update(parse_table_schema(extra_schema_sql))
        return schemas

    def apply_schema_ddl(self, ddl_sql: str, auto_sync_views: bool = True) -> None:
        """
        Safely executes schema DDL changes (e.g. `CREATE TABLE`, `ALTER TABLE`) and
        automatically refreshes any registered materialized views whose underlying
        base table schema changed.
        """
        cur = self.conn.cursor()
        for stmt in _split_top_level(_strip_sql_comments(ddl_sql), ";"):
            if stmt.strip():
                cur.execute(stmt)
        self._commit()
        if auto_sync_views:
            catalog = self.get_catalog()
            if catalog:
                self.sync_views([entry["view_sql"] for entry in catalog.values()])

    def get_catalog(self) -> Dict[str, Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
            f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json "
            f"FROM {self.catalog_table}"
        )
        out: Dict[str, Dict[str, Any]] = {}
        for row in cur.fetchall():
            out[row[0]] = {
                "view_name": row[0],
                "view_sql": row[1],
                "strategy": row[2],
                "signature_hash": row[3],
                "backing_table": row[4],
                "aux_tables": json.loads(row[5]),
                "trigger_names": json.loads(row[6]),
                "function_names": json.loads(row[7]),
                "trigger_ddls": json.loads(row[8]),
            }
        return out

    def _compute_signature(self, plan: MaterializedViewPlan, schemas: Dict[str, TableSchema]) -> str:
        h = hashlib.sha256()
        h.update(_strip_sql_comments(plan.view_spec.raw_select_sql).encode("utf-8"))
        h.update(f"prefix:{plan.object_prefix}".encode("utf-8"))
        for idx_ddl in plan.index_ddls:
            h.update(f"idx:{idx_ddl}".encode("utf-8"))
        low_map = {k.lower(): v for k, v in schemas.items()}
        for ref_t in sorted(plan.view_spec.referenced_tables, key=str.lower):
            ts = low_map.get(ref_t.lower())
            if ts:
                for c_name, c_def in sorted(ts.columns.items()):
                    h.update(
                        f"{ts.name}.{c_name}:{c_def.data_type}:{c_def.not_null}:{c_def.primary_key}".encode("utf-8")
                    )
        return h.hexdigest()

    @staticmethod
    def _serialize_view_sql_with_indexes(plan: MaterializedViewPlan, raw_view_sql: Optional[str] = None) -> str:
        parts = [f"CREATE VIEW {plan.logical_view_name} AS {plan.view_spec.raw_select_sql.strip().rstrip(';')};"]
        for idx_spec in plan.user_indexes:
            u_kw = "UNIQUE " if idx_spec.unique else ""
            n_str = f"{idx_spec.name} " if idx_spec.name else ""
            using_str = f" USING {idx_spec.using_method}" if idx_spec.using_method else ""
            w_str = f" WHERE {idx_spec.where_clause}" if idx_spec.where_clause else ""
            parts.append(
                f"CREATE {u_kw}INDEX {n_str}ON {plan.logical_view_name}{using_str} "
                f"({', '.join(idx_spec.columns)}){w_str};"
            )
        return "\n".join(parts)

    def drop_view(self, view_name: str) -> bool:
        """Drops a registered materialized view, its triggers, read-only view, and backing tables."""
        catalog = self.get_catalog()
        resolved_name = self._resolve_catalog_name(view_name, catalog)
        entry = catalog.get(resolved_name)
        cur = self.conn.cursor()
        if entry:
            for t_name in entry["trigger_names"]:
                if self.dialect != Dialect.POSTGRES:
                    cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
            for fn_name in entry["function_names"]:
                cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
            cur.execute(f"DROP VIEW IF EXISTS {resolved_name};")
            cur.execute(f"DROP TABLE IF EXISTS {entry['backing_table']};")
            for aux_t in entry["aux_tables"]:
                cur.execute(f"DROP TABLE IF EXISTS {aux_t};")
            ph = self._param_placeholder()
            cur.execute(f"DELETE FROM {self.catalog_table} WHERE view_name = {ph}", (resolved_name,))
            self._commit()
            return True
        else:
            raw_logical = (
                view_name[len(self.object_prefix) :]
                if self.object_prefix and view_name.startswith(self.object_prefix)
                else view_name
            )
            db_v = f"{self.object_prefix}{raw_logical}"
            cur.execute(f"DROP VIEW IF EXISTS {db_v};")
            cur.execute(f"DROP VIEW IF EXISTS {raw_logical};")
            cur.execute(f"DROP TABLE IF EXISTS {self.object_prefix}store_{raw_logical};")
            self._commit()
            return False

    def backfill_incrementally(
        self,
        plan: MaterializedViewPlan,
        batch_size: int = 500,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Incrementally backfills the materialized view in bounded chunks of `batch_size`
        (by PK/rowid range folding for algebraic views, group-key batches for distinct/holistic
        views, or limit/offset batches for general views).
        Returns the total number of batches executed.
        """
        if plan.sub_plans:
            total_b = 0
            for sp in plan.sub_plans:
                total_b += self.backfill_incrementally(sp, batch_size=batch_size, on_progress=on_progress)
            return total_b

        spec = plan.incremental_backfill_spec
        cur = self.conn.cursor()
        if spec is None or batch_size <= 0:
            for s in plan.initial_populate_sqls:
                cur.execute(s)
            self._commit()
            if on_progress:
                on_progress(1, 1)
            return 1

        batches_run = 0
        if spec.mode == "pk_range_fold" and spec.pk_column:
            base_t = plan.view_spec.base_table
            pk_col = spec.pk_column
            cur.execute(f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}")
            row = cur.fetchone()
            if not row or row[0] is None or row[1] is None:
                if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                    for s in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(s)
                    self._commit()
                return 0
            min_pk, max_pk = int(row[0]), int(row[1])
            cursor_pk = min_pk - 1
            total_span = max(1, max_pk - min_pk + 1)
            while cursor_pk < max_pk:
                upper_pk = min(max_pk, cursor_pk + batch_size)
                for tmpl in spec.chunk_sql_templates:
                    cur.execute(tmpl, (cursor_pk, upper_pk))
                self._commit()
                batches_run += 1
                cursor_pk = upper_pk
                if on_progress:
                    done = min(total_span, cursor_pk - (min_pk - 1))
                    on_progress(batches_run, done)
            return batches_run

        elif spec.mode == "group_key_batch" and spec.distinct_group_query:
            cur.execute(spec.distinct_group_query)
            all_keys = [r[0] for r in (cur.fetchall() or [])]
            if not all_keys:
                return 0
            ph = self._param_placeholder()
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i : i + batch_size]
                placeholders = ", ".join(ph for _ in batch_keys)
                for tmpl in spec.chunk_sql_templates:
                    sql = tmpl.format(placeholders=placeholders)
                    cur.execute(sql, tuple(batch_keys))
                self._commit()
                batches_run += 1
                if on_progress:
                    on_progress(batches_run, min(len(all_keys), i + batch_size))
            return batches_run

        elif spec.mode == "offset_batch" and spec.offset_insert_template:
            offset = 0
            ph = self._param_placeholder()
            tmpl_sql = spec.offset_insert_template.replace("LIMIT ? OFFSET ?", f"LIMIT {ph} OFFSET {ph}")
            while True:
                cur.execute(tmpl_sql, (batch_size, offset))
                inserted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
                self._commit()
                if inserted == 0:
                    break
                batches_run += 1
                offset += inserted
                if on_progress:
                    on_progress(batches_run, offset)
                if inserted < batch_size:
                    break
            return batches_run

        for s in plan.initial_populate_sqls:
            cur.execute(s)
        self._commit()
        return 1

    def _register_single_view_spec(
        self,
        v_spec: ViewSpec,
        schemas: Dict[str, TableSchema],
        populate: bool,
        incremental_backfill: bool,
        batch_size: int,
        on_progress: Optional[Callable[[int, int], None]],
        force_rebuild: bool,
        effective_ignore: bool,
    ) -> MaterializedViewPlan:
        plan = generate_materialized_view(
            schemas,
            v_spec,
            dialect=self.dialect,
            pg_use_transition_tables=self.pg_use_transition_tables,
            ignore_limitations=effective_ignore,
            object_prefix=self.object_prefix,
        )
        sig = self._compute_signature(plan, schemas)

        catalog = self.get_catalog()
        existing = catalog.get(plan.view_name)
        if existing and existing["signature_hash"] == sig and not force_rebuild:
            return plan

        if existing:
            self.drop_view(plan.view_name)

        # Install structure AND maintenance triggers BEFORE backfilling so no concurrent
        # base-table DML can occur in an un-triggered window during or after backfill.
        cur = self.conn.cursor()
        for stmt in plan.create_structure_statements():
            cur.execute(stmt)
        for stmt in plan.trigger_ddls:
            cur.execute(stmt)
        self._commit()

        if populate:
            if incremental_backfill:
                self.backfill_incrementally(
                    plan, batch_size=batch_size, on_progress=on_progress
                )
            else:
                for stmt in plan.get_populate_statements(incremental_backfill=False):
                    cur.execute(stmt)
                self._commit()

        stored_view_sql = self._serialize_view_sql_with_indexes(plan)
        ph = self._param_placeholder()
        cur.execute(
            f"""
            INSERT INTO {self.catalog_table} (
                view_name, view_sql, strategy, signature_hash, backing_table,
                aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json
            ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            """,
            (
                plan.view_name,
                stored_view_sql,
                plan.strategy.value,
                sig,
                plan.backing_table_name,
                json.dumps(plan.auxiliary_table_names),
                json.dumps(plan.trigger_names),
                json.dumps(plan.function_names),
                json.dumps(plan.trigger_ddls),
            ),
        )
        self._commit()
        return plan

    def register_view(
        self,
        view_sql: ViewSpec | str,
        schema_sql: Optional[str] = None,
        populate: bool = True,
        incremental_backfill: bool = False,
        batch_size: int = 500,
        on_progress: Optional[Callable[[int, int], None]] = None,
        force_rebuild: bool = False,
        ignore_limitations: Optional[bool] = None,
        indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    ) -> MaterializedViewPlan:
        """
        Safely installs or updates one or more materialized views on the connected database.
        Supports both single-view and multi-view SQL scripts seamlessly.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        schemas = self.fetch_schema(extra_schema_sql=schema_sql)
        specs = parse_view_definitions(
            view_sql,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
        )
        sub_plans: List[MaterializedViewPlan] = []
        for vs in specs:
            sub_plans.append(
                self._register_single_view_spec(
                    vs,
                    schemas=schemas,
                    populate=populate,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                    on_progress=on_progress,
                    force_rebuild=force_rebuild,
                    effective_ignore=effective_ignore,
                )
            )
        if len(sub_plans) == 1:
            return sub_plans[0]
        p0 = sub_plans[0]
        return MaterializedViewPlan(
            dialect=p0.dialect,
            strategy=p0.strategy,
            view_spec=p0.view_spec,
            backing_table_name=p0.backing_table_name,
            backing_table_ddl=p0.backing_table_ddl,
            auxiliary_table_ddls=[d for sp in sub_plans for d in sp.auxiliary_table_ddls],
            index_ddls=[d for sp in sub_plans for d in sp.index_ddls],
            initial_populate_sqls=[d for sp in sub_plans for d in sp.initial_populate_sqls],
            trigger_ddls=[d for sp in sub_plans for d in sp.trigger_ddls],
            readonly_trigger_ddls=[d for sp in sub_plans for d in sp.readonly_trigger_ddls],
            user_columns=p0.user_columns,
            reader_view_ddl=p0.reader_view_ddl,
            incremental_backfill_spec=p0.incremental_backfill_spec,
            auxiliary_table_names=[n for sp in sub_plans for n in sp.auxiliary_table_names],
            trigger_names=[n for sp in sub_plans for n in sp.trigger_names],
            function_names=[n for sp in sub_plans for n in sp.function_names],
            limitations=[lim for sp in sub_plans for lim in sp.limitations],
            object_prefix=p0.object_prefix,
            db_view_name=p0.db_view_name,
            index_names=[n for sp in sub_plans for n in sp.index_names],
            user_indexes=p0.user_indexes,
            sub_plans=sub_plans,
        )

    def sync_views(
        self,
        view_sqls: Sequence[ViewSpec | str] | ViewSpec | str,
        schema_sql: Optional[str] = None,
        prune_removed: bool = False,
        incremental_backfill: bool = False,
        batch_size: int = 500,
        ignore_limitations: Optional[bool] = None,
        indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        Declaratively synchronizes a collection of materialized view definitions against
        the connected database (supporting multi-view SQL files/strings as well).
        Returns a status map `{view_name: "created" | "updated" | "unchanged" | "dropped"}`.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        schemas = self.fetch_schema(extra_schema_sql=schema_sql)
        catalog = self.get_catalog()
        status: Dict[str, str] = {}
        desired_names: Set[str] = set()

        all_specs = parse_view_definitions(
            view_sqls,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
        )

        for v_spec in all_specs:
            plan = generate_materialized_view(
                schemas,
                v_spec,
                dialect=self.dialect,
                ignore_limitations=effective_ignore,
                object_prefix=self.object_prefix,
            )
            v_name = plan.view_name
            desired_names.add(v_name)
            sig = self._compute_signature(plan, schemas)
            if v_name not in catalog:
                self._register_single_view_spec(
                    v_spec,
                    schemas=schemas,
                    populate=True,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                    on_progress=None,
                    force_rebuild=False,
                    effective_ignore=effective_ignore,
                )
                status[v_name] = "created"
            elif catalog[v_name]["signature_hash"] != sig:
                self._register_single_view_spec(
                    v_spec,
                    schemas=schemas,
                    populate=True,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                    on_progress=None,
                    force_rebuild=True,
                    effective_ignore=effective_ignore,
                )
                status[v_name] = "updated"
            else:
                status[v_name] = "unchanged"

        if prune_removed:
            for existing_name in list(catalog.keys()):
                if existing_name not in desired_names:
                    self.drop_view(existing_name)
                    status[existing_name] = "dropped"

        return status

    @contextmanager
    def bulk_changes(self, view_names: Optional[Sequence[str]] = None) -> Iterator[None]:
        """
        Context manager for executing massive multi-statement bulk loads/mutations.
        Temporarily suspends maintenance triggers for the target views during the block,
        and atomically recomputes the backing tables in a single set-based pass on exit.
        """
        catalog = self.get_catalog()
        if view_names:
            resolved = [self._resolve_catalog_name(n, catalog) for n in view_names]
            targets = [catalog[n] for n in resolved if n in catalog]
        else:
            targets = list(catalog.values())

        cur = self.conn.cursor()
        for entry in targets:
            if self.dialect == Dialect.POSTGRES:
                for fn_name in entry["function_names"]:
                    if "fn_readonly_" not in fn_name:
                        cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
            else:
                for t_name in entry["trigger_names"]:
                    if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                        cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
        try:
            yield
        finally:
            schemas = self.fetch_schema()
            for entry in targets:
                plan = generate_materialized_view(
                    schemas,
                    entry["view_sql"],
                    dialect=self.dialect,
                    pg_use_transition_tables=self.pg_use_transition_tables,
                    ignore_limitations=True,
                    object_prefix=self.object_prefix,
                )
                for aux_t in plan.auxiliary_table_names:
                    cur.execute(f"DELETE FROM {aux_t};")
                cur.execute(f"DELETE FROM {plan.backing_table_name};")
                for stmt in plan.trigger_ddls:
                    cur.execute(stmt)
                for stmt in plan.get_populate_statements(incremental_backfill=False):
                    cur.execute(stmt)
            self._commit()


# -----------------------------------------------------------------------------
# Public High-Level Convenience Functions & CLI
# -----------------------------------------------------------------------------


def generate_materialized_view(
    schema_sql_or_dict: str | Dict[str, TableSchema] | sqlite3.Connection,
    view_sql_or_spec: str | ViewSpec,
    dialect: Dialect | str = Dialect.SQLITE,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
) -> MaterializedViewPlan:
    """
    Compiles a `CREATE VIEW` SQL definition (or multi-view SQL script) and base table schema
    into a `MaterializedViewPlan`.
    """
    if isinstance(dialect, str):
        dialect = Dialect(dialect.lower())

    if isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_sqlite_schemas(schema_sql_or_dict, object_prefix=object_prefix)
    elif isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict)
    else:
        schemas = dict(schema_sql_or_dict)

    known_tables = {k.lower() for k in schemas}
    if isinstance(view_sql_or_spec, str):
        view_spec = parse_view_definition(
            view_sql_or_spec,
            known_tables=known_tables,
            indexes=indexes,
        )
    elif indexes:
        parsed_idxs = tuple(
            parse_view_index_spec(ix, default_view_name=view_sql_or_spec.view_name)
            for ix in indexes
        )
        view_spec = ViewSpec(
            view_name=view_sql_or_spec.view_name,
            raw_select_sql=view_sql_or_spec.raw_select_sql,
            base_table=view_sql_or_spec.base_table,
            joins=view_sql_or_spec.joins,
            referenced_tables=view_sql_or_spec.referenced_tables,
            select_items=view_sql_or_spec.select_items,
            where_clause=view_sql_or_spec.where_clause,
            group_by_cols=view_sql_or_spec.group_by_cols,
            having_clause=view_sql_or_spec.having_clause,
            has_cte=view_sql_or_spec.has_cte,
            has_window_func=view_sql_or_spec.has_window_func,
            has_compound_set_op=view_sql_or_spec.has_compound_set_op,
            has_top_level_distinct=view_sql_or_spec.has_top_level_distinct,
            has_limit_or_order=view_sql_or_spec.has_limit_or_order,
            has_subquery_in_from=view_sql_or_spec.has_subquery_in_from,
            user_indexes=view_sql_or_spec.user_indexes + parsed_idxs,
            additional_views=view_sql_or_spec.additional_views,
        )
    else:
        view_spec = view_sql_or_spec

    compiler = IncrementalViewCompiler(
        schemas=schemas,
        view_spec=view_spec,
        dialect=dialect,
        pg_use_transition_tables=pg_use_transition_tables,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
    )
    return compiler.compile()


def generate_materialized_views(
    schema_sql_or_dict: str | Dict[str, TableSchema] | sqlite3.Connection,
    view_sqls: Sequence[str | ViewSpec] | str | ViewSpec,
    dialect: Dialect | str = Dialect.SQLITE,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
) -> List[MaterializedViewPlan]:
    """
    Compiles one or more `CREATE VIEW` SQL definitions (including multi-view SQL scripts)
    into a flat list of `MaterializedViewPlan` objects.
    """
    if isinstance(dialect, str):
        dialect = Dialect(dialect.lower())

    if isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_sqlite_schemas(schema_sql_or_dict, object_prefix=object_prefix)
    elif isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict)
    else:
        schemas = dict(schema_sql_or_dict)

    specs = parse_view_definitions(
        view_sqls,
        known_tables={k.lower() for k in schemas},
        indexes=indexes,
    )
    plans: List[MaterializedViewPlan] = []
    for vs in specs:
        comp = IncrementalViewCompiler(
            schemas=schemas,
            view_spec=vs,
            dialect=dialect,
            pg_use_transition_tables=pg_use_transition_tables,
            ignore_limitations=ignore_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
        )
        plans.append(comp._compile_single_plan())
    return plans


def install_materialized_view_sqlite(
    conn: sqlite3.Connection,
    view_sql: str | ViewSpec,
    schema_sql: Optional[str] = None,
    populate: bool = True,
    incremental_backfill: bool = False,
    batch_size: int = 500,
    on_progress: Optional[Callable[[int, int], None]] = None,
    ignore_limitations: bool = False,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
) -> MaterializedViewPlan:
    """Installs one or more materialized views directly into a live SQLite connection."""
    mgr = MaterializedViewManager(
        conn,
        dialect=Dialect.SQLITE,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
    )
    return mgr.register_view(
        view_sql=view_sql,
        schema_sql=schema_sql,
        populate=populate,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
        on_progress=on_progress,
        indexes=indexes,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile SQL CREATE VIEW definition(s) into zero-dependency incremental materialized view SQL."
    )
    parser.add_argument("--schema", help="Path to SQL file containing base table CREATE TABLE statements.")
    parser.add_argument("--db", help="Path to live SQLite database file (for schema introspection or --apply).")
    parser.add_argument("--view", required=True, help="Path to SQL file or inline CREATE VIEW statement(s).")
    parser.add_argument(
        "--dialect",
        choices=[d.value for d in Dialect],
        default=Dialect.SQLITE.value,
        help="Target SQL dialect (sqlite, postgres, mysql).",
    )
    parser.add_argument(
        "--pg-transition-tables",
        action="store_true",
        help="Use PostgreSQL statement-level transition table triggers.",
    )
    parser.add_argument(
        "--object-prefix",
        default=DEFAULT_OBJECT_PREFIX,
        help="Prefix for generated database objects (default: __matview_).",
    )
    parser.add_argument(
        "--index",
        action="append",
        dest="indexes",
        help="Additional user index specification on the materialized view (repeatable).",
    )
    parser.add_argument(
        "--no-populate",
        action="store_true",
        help="Omit initial backfill statements from generated SQL.",
    )
    parser.add_argument(
        "--incremental-backfill",
        action="store_true",
        help="Generate or execute chunked incremental initial backfill statements.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for --incremental-backfill (default: 500).",
    )
    parser.add_argument(
        "--ignore-limitations",
        action="store_true",
        help="Proceed with warning even if engine transparency limitations exist.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the generated materialized view directly to --db.",
    )
    args = parser.parse_args(argv)

    def _read_sql_arg(val: str) -> str:
        if "\n" not in val and not re.match(r"^\s*CREATE\b", val, flags=re.IGNORECASE):
            try:
                with open(val, "r", encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                pass
        return val

    view_sql = _read_sql_arg(args.view)
    schema_sql = _read_sql_arg(args.schema) if args.schema else None

    try:
        if args.apply:
            if not args.db:
                print("ERROR: --apply requires --db <sqlite_db_path>", file=sys.stderr)
                return 2
            mgr = MaterializedViewManager(
                args.db,
                dialect=Dialect(args.dialect),
                ignore_limitations=args.ignore_limitations,
                object_prefix=args.object_prefix,
                pg_use_transition_tables=args.pg_transition_tables,
            )
            if schema_sql:
                mgr.apply_schema_ddl(schema_sql, auto_sync_views=False)
            plan = mgr.register_view(
                view_sql,
                populate=not args.no_populate,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
                indexes=args.indexes,
            )
            print(
                plan.render_sql_script(
                    include_populate=not args.no_populate,
                    incremental_backfill=args.incremental_backfill,
                    batch_size=args.batch_size,
                )
            )
            return 0

        if args.db and not schema_sql:
            conn = sqlite3.connect(args.db)
            try:
                schemas = introspect_sqlite_schemas(conn, object_prefix=args.object_prefix)
            finally:
                conn.close()
        elif schema_sql:
            schemas = parse_table_schema(schema_sql)
        else:
            print("ERROR: Either --schema or --db must be provided.", file=sys.stderr)
            return 2

        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=Dialect(args.dialect),
            pg_use_transition_tables=args.pg_transition_tables,
            ignore_limitations=args.ignore_limitations,
            object_prefix=args.object_prefix,
            indexes=args.indexes,
            incremental_backfill=args.incremental_backfill,
            batch_size=args.batch_size,
        )
        print(
            plan.render_sql_script(
                include_populate=not args.no_populate,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
            )
        )
        return 0
    except EngineLimitationError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
