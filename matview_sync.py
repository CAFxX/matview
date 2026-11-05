"""
Generic Incremental Materialized View Engine (SQLite, PostgreSQL, MySQL).

This tool is a pure setup/migration compiler and installer — it is NOT part of the
database runtime. Once a materialized view, its backing table(s), and its native SQL
triggers have been created and backfilled, the database maintains the view 100%
autonomously using only native engine SQL with zero external runtime dependencies
or user-defined function (UDF) callbacks.

Given a database schema (`CREATE TABLE ...`) or a live database connection and any
view definition (`CREATE VIEW ... AS ...`), this module automatically generates
and manages:
  1. The internal backing materialized view table (`CREATE TABLE __matview_store_<view_name> ...`)
     plus any auxiliary reference-count tables (`__matview_dist_<view_name>__...`).
  2. A strictly read-only user-facing SQL `VIEW` (`CREATE VIEW <view_name> AS SELECT ...`)
     with read-only enforcement (`INSTEAD OF` abort/exception triggers in SQLite and
     PostgreSQL, and non-updatable merge view semantics in MySQL) so users query
     `<view_name>` transparently while preventing accidental direct writes.
  3. Primary key & lookup indexes for single-probe B-tree maintenance.
  4. Full or incremental (chunked/batched) initial backfill (`incremental_backfill=True`).
  5. Highly optimized native SQL maintenance triggers on all participating base tables
     supporting single-row and multi-row mutations across 1 or many MV groups.
  6. Strict Transparency & Engine-Limitation Guard:
     Automatically analyzes the `(schema, view, dialect)` triple for any engine or query
     limitations (such as dialect-unsupported aggregates, MySQL `TRUNCATE TABLE` and
     InnoDB `ON DELETE/UPDATE CASCADE` trigger bypasses, SQLite `ON DELETE/UPDATE CASCADE`
     or `ON CONFLICT REPLACE` runtime pragma dependencies, non-deterministic functions
     like `NOW()` / `RANDOM()`, `LIMIT` without `ORDER BY`, or floating-point `GROUP BY`
     keys). Refuses the operation by default with a clear `EngineLimitationError` unless
     `--ignore-limitations` (CLI) or `ignore_limitations=True` (API) is explicitly passed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
import re
import sqlite3
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple
import warnings


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"


class MaintenanceStrategy(str, Enum):
    ALGEBRAIC_DELTA = "algebraic_delta"
    GROUP_SCOPED_DELTA = "group_scoped_delta"
    ATOMIC_QUERY_SYNC = "atomic_query_sync"


class AggFunc(str, Enum):
    COUNT_STAR = "COUNT_STAR"
    COUNT_COL = "COUNT_COL"
    SUM = "SUM"
    TOTAL = "TOTAL"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    VAR_POP = "VAR_POP"
    VAR_SAMP = "VAR_SAMP"
    STDDEV_POP = "STDDEV_POP"
    STDDEV_SAMP = "STDDEV_SAMP"
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BOOL_AND = "BOOL_AND"
    BOOL_OR = "BOOL_OR"
    GROUP_CONCAT = "GROUP_CONCAT"
    STRING_AGG = "STRING_AGG"
    JSON_GROUP_ARRAY = "JSON_GROUP_ARRAY"
    JSON_GROUP_OBJECT = "JSON_GROUP_OBJECT"
    ARRAY_AGG = "ARRAY_AGG"


_AGG_NAME_MAP: Dict[str, AggFunc] = {
    "COUNT": AggFunc.COUNT_COL,
    "SUM": AggFunc.SUM,
    "TOTAL": AggFunc.TOTAL,
    "AVG": AggFunc.AVG,
    "MIN": AggFunc.MIN,
    "MAX": AggFunc.MAX,
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
    "GROUP_CONCAT": AggFunc.GROUP_CONCAT,
    "STRING_AGG": AggFunc.STRING_AGG,
    "JSON_GROUP_ARRAY": AggFunc.JSON_GROUP_ARRAY,
    "JSON_AGG": AggFunc.JSON_GROUP_ARRAY,
    "JSONB_AGG": AggFunc.JSON_GROUP_ARRAY,
    "JSON_ARRAYAGG": AggFunc.JSON_GROUP_ARRAY,
    "JSON_GROUP_OBJECT": AggFunc.JSON_GROUP_OBJECT,
    "JSON_OBJECT_AGG": AggFunc.JSON_GROUP_OBJECT,
    "JSONB_OBJECT_AGG": AggFunc.JSON_GROUP_OBJECT,
    "JSON_OBJECTAGG": AggFunc.JSON_GROUP_OBJECT,
    "ARRAY_AGG": AggFunc.ARRAY_AGG,
}

# Native aggregate functions supported out-of-the-box by each database engine
_DIALECT_NATIVE_AGGS: Dict[Dialect, Set[AggFunc]] = {
    Dialect.SQLITE: {
        AggFunc.COUNT_STAR,
        AggFunc.COUNT_COL,
        AggFunc.SUM,
        AggFunc.TOTAL,
        AggFunc.AVG,
        AggFunc.MIN,
        AggFunc.MAX,
        AggFunc.GROUP_CONCAT,
        AggFunc.STRING_AGG,
        AggFunc.JSON_GROUP_ARRAY,
        AggFunc.JSON_GROUP_OBJECT,
    },
    Dialect.POSTGRES: {
        AggFunc.COUNT_STAR,
        AggFunc.COUNT_COL,
        AggFunc.SUM,
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
        AggFunc.STRING_AGG,
        AggFunc.JSON_GROUP_ARRAY,
        AggFunc.JSON_GROUP_OBJECT,
        AggFunc.ARRAY_AGG,
    },
    Dialect.MYSQL: {
        AggFunc.COUNT_STAR,
        AggFunc.COUNT_COL,
        AggFunc.SUM,
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
        AggFunc.GROUP_CONCAT,
        AggFunc.JSON_GROUP_ARRAY,
        AggFunc.JSON_GROUP_OBJECT,
    },
}


@dataclass(frozen=True)
class LimitationItem:
    code: str
    message: str


class EngineLimitationWarning(UserWarning):
    """Warning emitted when `--ignore-limitations` / `ignore_limitations=True` overrides transparency limitations."""


class EngineLimitationError(ValueError):
    """
    Raised when the target engine, schema, or view definition has transparency
    limitations that prevent 100% uniform behavior across all possible operations
    and `--ignore-limitations` (`ignore_limitations=True`) was not specified.
    """

    def __init__(
        self,
        view_name: str,
        dialect: Dialect,
        limitations: Sequence[LimitationItem],
    ) -> None:
        self.view_name = view_name
        self.dialect = dialect
        self.limitations = list(limitations)
        bullets = "\n".join(f"  - [{item.code}] {item.message}" for item in self.limitations)
        msg = (
            f"WARNING: Refusing to create materialized view '{view_name}' on engine "
            f"'{dialect.value}' because full transparency cannot be guaranteed for all cases:\n"
            f"{bullets}\n"
            f"Pass `--ignore-limitations` (CLI) or `ignore_limitations=True` (API) "
            f"to override this check and proceed anyway."
        )
        super().__init__(msg)


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
    fk_cascades: Tuple[str, ...] = ()
    has_on_conflict_replace: bool = False


@dataclass(frozen=True)
class JoinClause:
    join_type: str
    table_name: str
    alias: str
    on_condition: Optional[str] = None


DEFAULT_OBJECT_PREFIX = "__matview_"


@dataclass(frozen=True)
class ViewIndexSpec:
    """
    Specification for a user-defined index to be created on a materialized view's
    backing table.
    """
    columns: Tuple[str, ...]
    name: Optional[str] = None
    unique: bool = False
    where_clause: Optional[str] = None
    using_method: Optional[str] = None
    target_view: Optional[str] = None


IndexDef = ViewIndexSpec
IndexSpec = ViewIndexSpec


@dataclass(frozen=True)
class SelectItem:
    raw_expr: str
    alias: str
    is_aggregate: bool
    agg_func: Optional[AggFunc] = None
    raw_func_name: Optional[str] = None
    arg_expr: Optional[str] = None
    extra_args: Optional[str] = None
    is_distinct: bool = False
    filter_where: Optional[str] = None
    source_col: Optional[str] = None


@dataclass(frozen=True)
class ViewSpec:
    view_name: str
    raw_select_sql: str
    base_table: str
    joins: List[JoinClause]
    referenced_tables: List[str]
    select_items: List[SelectItem]
    group_by_exprs: List[str]
    where_clause: Optional[str] = None
    having_clause: Optional[str] = None
    has_cte: bool = False
    has_compound_set_op: bool = False
    has_window_func: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    has_top_level_distinct: bool = False
    has_subquery_in_from_or_where: bool = False
    has_complex_agg_expr: bool = False
    user_indexes: Tuple[ViewIndexSpec, ...] = ()
    additional_views: Tuple["ViewSpec", ...] = ()

    @property
    def has_limit_or_order(self) -> bool:
        return self.has_order_by or self.has_limit

    @property
    def group_by_cols(self) -> List[str]:
        return [_unquote_ident(g) for g in self.group_by_exprs]


@dataclass(frozen=True)
class IncrementalBackfillSpec:
    """Holds SQL templates and metadata for chunked/incremental initial backfilling."""
    mode: str  # "pk_range_fold", "group_key_batch", or "offset_batch"
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
    limitations: List[LimitationItem] = field(default_factory=list)
    object_prefix: str = DEFAULT_OBJECT_PREFIX
    db_view_name: str = ""
    index_names: List[str] = field(default_factory=list)
    user_indexes: List[ViewIndexSpec] = field(default_factory=list)
    sub_plans: List["MaterializedViewPlan"] = field(default_factory=list)
    incremental_backfill: bool = False
    batch_size: int = 500
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
                # Substitute custom batch_size if overridden at render time
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
        for t_name in self.trigger_names:
            if self.dialect == Dialect.POSTGRES:
                stmts.append(f"DROP TRIGGER IF EXISTS {t_name} ON {self.view_spec.base_table} CASCADE;")
            else:
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
        backfill_desc = f"incremental (batch_size={eff_batch})" if use_inc else "full"
        blocks = [
            f"-- =============================================================================",
            f"-- Incremental Materialized View: {self.view_name} (logical: {self.logical_view_name})",
            f"-- Backing Table: {self.backing_table_name}",
            f"-- Base Tables: {', '.join(self.view_spec.referenced_tables)} | "
            f"Dialect: {self.dialect.value} | Strategy: {self.strategy.value} | Backfill: {backfill_desc}",
            f"-- =============================================================================",
        ]
        if self.limitations:
            blocks.append("-- WARNING (Ignored Engine/View Limitations):")
            for lim in self.limitations:
                blocks.append(f"--   * [{lim.code}] {lim.message}")
        if self.auxiliary_table_ddls:
            blocks.append("\n-- 0. Auxiliary State Tables")
            blocks.extend(self.auxiliary_table_ddls)
        blocks.append("\n-- 1. Internal Backing Table")
        blocks.append(self.backing_table_ddl)
        if self.index_ddls:
            blocks.append("\n-- 2. Indexes")
            blocks.extend(self.index_ddls)
        if self.reader_view_ddl:
            blocks.append("\n-- 3. Read-Only User View & Write-Protection Triggers")
            blocks.append(self.reader_view_ddl)
            blocks.extend(self.readonly_trigger_ddls)
        blocks.append("\n-- 4. Atomic Maintenance Triggers (Installed Before Backfill)")
        blocks.extend(self.trigger_ddls)
        if include_populate and pop_stmts:
            if use_inc:
                blocks.append(f"\n-- 5. Incremental Initial Backfill (batch_size={eff_batch})")
            else:
                blocks.append("\n-- 5. Initial Backfill")
            blocks.extend(pop_stmts)
        return "\n\n".join(blocks) + "\n"


# -----------------------------------------------------------------------------
# SQL Lexing & Parsing Utilities
# -----------------------------------------------------------------------------


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


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
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and text[i : i + sep_len] == sep:
                parts.append("".join(buf).strip())
                buf = []
                i += sep_len
                continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _unquote_ident(ident: str) -> str:
    ident = ident.strip()
    if "." in ident and not (ident.startswith('"') or ident.startswith("`")):
        ident = ident.split(".")[-1].strip()
    if (
        (ident.startswith('"') and ident.endswith('"'))
        or (ident.startswith("`") and ident.endswith("`"))
        or (ident.startswith("[") and ident.endswith("]"))
    ):
        return ident[1:-1]
    return ident


def _sanitize_ident(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    return s or "expr"


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
            fk_cascades=tuple(fk_cascades),
            has_on_conflict_replace=has_on_conflict_replace,
        )
    if not schemas:
        raise ValueError(f"Could not parse any CREATE TABLE statement from: {schema_sql!r}")
    return schemas


def introspect_sqlite_schemas(
    conn: sqlite3.Connection, object_prefix: str = DEFAULT_OBJECT_PREFIX
) -> Dict[str, TableSchema]:
    """Extracts user base table schemas directly from a live SQLite connection."""
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '__matview_%' "
        "AND name NOT LIKE '__mv_%'"
    ).fetchall()
    if object_prefix:
        rows = [
            (n, s)
            for (n, s) in rows
            if not n.startswith(object_prefix)
        ]
    schemas: Dict[str, TableSchema] = {}
    for tbl_name, sql in rows:
        try:
            parsed = parse_table_schema(sql)
            schemas.update(parsed)
        except ValueError:
            p_rows = conn.execute(f"PRAGMA table_info({tbl_name})").fetchall()
            cols: Dict[str, ColumnDef] = {}
            for _, c_name, c_type, notnull, _, pk in p_rows:
                cols[c_name] = ColumnDef(
                    name=c_name,
                    data_type=(c_type or "TEXT").upper(),
                    not_null=bool(notnull) or bool(pk),
                    primary_key=bool(pk),
                )
            if cols:
                schemas[tbl_name] = TableSchema(name=tbl_name, columns=cols)
    return schemas


def _extract_cte_names_and_strip_with(select_sql: str) -> Tuple[Set[str], str, bool]:
    s = select_sql.strip()
    if not re.match(r"^WITH\b", s, flags=re.IGNORECASE):
        return set(), s, False

    rest = re.sub(r"^WITH\s+(?:RECURSIVE\s+)?", "", s, flags=re.IGNORECASE).strip()
    cte_names: Set[str] = set()

    pos = 0
    n = len(rest)
    while pos < n:
        m_head = re.match(
            r"^\s*([A-Za-z0-9_\"`\[\]]+)\s*(?:\([^)]*\)\s*)?AS\s*(?:(?:MATERIALIZED|NOT\s+MATERIALIZED)\s*)?\(",
            rest[pos:],
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m_head:
            break
        cte_names.add(_unquote_ident(m_head.group(1)).lower())
        open_paren_idx = pos + m_head.end() - 1
        depth = 0
        in_single = False
        in_double = False
        i = open_paren_idx
        while i < n:
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
                    if depth == 0:
                        i += 1
                        break
            i += 1
        pos = i
        while pos < n and rest[pos].isspace():
            pos += 1
        if pos < n and rest[pos] == ",":
            pos += 1
            continue
        break

    main_select = rest[pos:].strip()
    return cte_names, main_select, True


def _extract_all_referenced_tables(
    sql: str, cte_names: Set[str], known_tables: Optional[Set[str]] = None
) -> List[str]:
    stripped = re.sub(r"'(?:''|[^'])*'", "''", sql)
    found: List[str] = []
    seen: Set[str] = set()
    sql_keywords = {
        "select", "where", "group", "order", "having", "limit", "offset",
        "union", "intersect", "except", "values", "set", "into", "dual",
        "lateral", "unnest", "json_each", "json_tree",
    }
    for m in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z0-9_.\"`\[\]]+)",
        stripped,
        flags=re.IGNORECASE,
    ):
        raw_t = m.group(1)
        t_name = _unquote_ident(raw_t)
        t_low = t_name.lower()
        if t_low in cte_names or t_low in sql_keywords:
            continue
        if known_tables is not None and t_low not in known_tables:
            continue
        if t_low not in seen:
            seen.add(t_low)
            found.append(t_name)

    for m_from in re.finditer(
        r"\bFROM\s+([^;()]+?)(?:\bWHERE\b|\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|\bJOIN\b|$)",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        seg = m_from.group(1)
        if "," in seg:
            for part in seg.split(","):
                tokens = part.strip().split()
                if tokens:
                    t_name = _unquote_ident(tokens[0])
                    t_low = t_name.lower()
                    if t_low not in cte_names and t_low not in sql_keywords:
                        if known_tables is None or t_low in known_tables:
                            if t_low not in seen:
                                seen.add(t_low)
                                found.append(t_name)
    return found


def _parse_from_and_joins(from_clause: str) -> List[JoinClause]:
    clean = from_clause.strip()
    if clean.startswith("("):
        return []

    join_kw_pattern = r"\b(?:(?:INNER|LEFT(?:\s+OUTER)?|CROSS|NATURAL)\s+)?JOIN\b"
    matches = _find_all_top_level_keywords(clean, join_kw_pattern)

    clauses: List[JoinClause] = []

    def _parse_table_and_alias(segment: str) -> Tuple[str, str]:
        seg = segment.strip()
        tokens = seg.split()
        if not tokens:
            return "", ""
        t_name = _unquote_ident(tokens[0])
        if len(tokens) >= 3 and tokens[1].upper() == "AS":
            alias = _unquote_ident(tokens[2])
        elif len(tokens) >= 2 and tokens[1].upper() not in (
            "ON", "USING", "WHERE", "GROUP", "ORDER", "LIMIT"
        ):
            alias = _unquote_ident(tokens[1])
        else:
            alias = t_name
        return t_name, alias

    if not matches:
        parts = _split_top_level(clean, ",")
        for idx, p in enumerate(parts):
            t_name, alias = _parse_table_and_alias(p)
            if t_name:
                clauses.append(
                    JoinClause(
                        join_type="FROM" if idx == 0 else "CROSS JOIN",
                        table_name=t_name,
                        alias=alias,
                        on_condition=None,
                    )
                )
        return clauses

    first_seg = clean[: matches[0].start()].strip()
    first_parts = _split_top_level(first_seg, ",")
    for idx, p in enumerate(first_parts):
        t_name, alias = _parse_table_and_alias(p)
        if t_name:
            clauses.append(
                JoinClause(
                    join_type="FROM" if idx == 0 else "CROSS JOIN",
                    table_name=t_name,
                    alias=alias,
                    on_condition=None,
                )
            )

    for idx, m in enumerate(matches):
        j_raw = clean[m.start() : m.end()].upper()
        j_type = " ".join(j_raw.split())
        if j_type == "JOIN":
            j_type = "INNER JOIN"
        elif j_type == "LEFT OUTER JOIN":
            j_type = "LEFT JOIN"

        seg_start = m.end()
        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean)
        seg = clean[seg_start:seg_end].strip()

        on_match = _find_top_level_keyword(seg, r"\bON\b")
        using_match = _find_top_level_keyword(seg, r"\bUSING\b")
        if on_match:
            tbl_part = seg[: on_match.start()].strip()
            on_cond = seg[on_match.end() :].strip()
            t_name, alias = _parse_table_and_alias(tbl_part)
            clauses.append(
                JoinClause(
                    join_type=j_type,
                    table_name=t_name,
                    alias=alias,
                    on_condition=on_cond,
                )
            )
        elif using_match:
            tbl_part = seg[: using_match.start()].strip()
            using_part = seg[using_match.end() :].strip()
            t_name, alias = _parse_table_and_alias(tbl_part)
            prev_alias = clauses[-1].alias if clauses else t_name
            using_cols = [
                _unquote_ident(c) for c in using_part.strip("() ").split(",") if c.strip()
            ]
            on_cond = " AND ".join(f"{prev_alias}.{c} = {alias}.{c}" for c in using_cols)
            clauses.append(
                JoinClause(
                    join_type=j_type,
                    table_name=t_name,
                    alias=alias,
                    on_condition=on_cond,
                )
            )
        else:
            t_name, alias = _parse_table_and_alias(seg)
            clauses.append(
                JoinClause(
                    join_type=j_type,
                    table_name=t_name,
                    alias=alias,
                    on_condition=None,
                )
            )
    return clauses


def _parse_select_item(raw_item: str, idx: int) -> Tuple[SelectItem, bool]:
    alias_match = _find_top_level_keyword(raw_item, r"\bAS\s+[A-Za-z0-9_\"`\[\]]+\s*$")
    if alias_match:
        expr = raw_item[: alias_match.start()].strip()
        alias_part = re.sub(
            r"^AS\s+", "", raw_item[alias_match.start() :].strip(), flags=re.IGNORECASE
        )
        alias = _unquote_ident(alias_part)
    else:
        m_imp = re.match(
            r"^(.*\))\s+([A-Za-z_][A-Za-z0-9_]*)$", raw_item.strip(), flags=re.DOTALL
        )
        if m_imp and m_imp.group(2).upper() not in ("END", "NULL"):
            expr = m_imp.group(1).strip()
            alias = _unquote_ident(m_imp.group(2))
        else:
            m_col_imp = re.match(
                r"^([A-Za-z0-9_.\"`\[\]]+)\s+([A-Za-z_][A-Za-z0-9_]*)$", raw_item.strip()
            )
            if m_col_imp and m_col_imp.group(2).upper() not in ("ASC", "DESC", "NULL"):
                expr = m_col_imp.group(1).strip()
                alias = _unquote_ident(m_col_imp.group(2))
            else:
                expr = raw_item.strip()
                alias = _unquote_ident(expr)

    if re.search(r"\bOVER\s*\(", expr, flags=re.IGNORECASE):
        if alias == expr:
            alias = f"win_col_{idx}"
        return (
            SelectItem(
                raw_expr=expr,
                alias=alias,
                is_aggregate=False,
                source_col=None,
            ),
            False,
        )

    expr_no_filter = expr
    filter_where: Optional[str] = None
    filter_match = re.search(
        r"\s+FILTER\s*\(\s*WHERE\s+(.*)\)\s*$", expr, flags=re.IGNORECASE | re.DOTALL
    )
    if filter_match:
        expr_no_filter = expr[: filter_match.start()].strip()
        filter_where = filter_match.group(1).strip()

    agg_call_match = re.match(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$",
        expr_no_filter,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if agg_call_match:
        func_name = agg_call_match.group(1).upper()
        inner_raw = agg_call_match.group(2).strip()
        depth = 0
        valid_outer = True
        in_s = False
        in_d = False
        for ch in inner_raw:
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif not in_s and not in_d:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        valid_outer = False
                        break
        if valid_outer and depth == 0 and func_name in _AGG_NAME_MAP:
            is_distinct = False
            arg_str = inner_raw
            if re.match(r"^DISTINCT\b", arg_str, flags=re.IGNORECASE):
                is_distinct = True
                arg_str = re.sub(r"^DISTINCT\s+", "", arg_str, flags=re.IGNORECASE).strip()
            elif re.match(r"^ALL\b", arg_str, flags=re.IGNORECASE):
                arg_str = re.sub(r"^ALL\s+", "", arg_str, flags=re.IGNORECASE).strip()

            top_args = _split_top_level(arg_str, ",")
            first_arg = top_args[0].strip() if top_args else "*"
            extra_args = ", ".join(top_args[1:]) if len(top_args) > 1 else None

            if func_name == "COUNT" and first_arg == "*" and not is_distinct:
                agg_func = AggFunc.COUNT_STAR
                arg_expr = None
            else:
                agg_func = _AGG_NAME_MAP[func_name]
                arg_expr = first_arg

            if alias == expr:
                suffix = "all" if arg_expr is None else _sanitize_ident(arg_expr)
                dist_prefix = "distinct_" if is_distinct else ""
                alias = f"{func_name.lower()}_{dist_prefix}{suffix}"

            return (
                SelectItem(
                    raw_expr=expr,
                    alias=alias,
                    is_aggregate=True,
                    agg_func=agg_func,
                    raw_func_name=func_name,
                    arg_expr=arg_expr,
                    extra_args=extra_args,
                    is_distinct=is_distinct,
                    filter_where=filter_where,
                ),
                False,
            )

    agg_pattern = r"\b(" + "|".join(re.escape(k) for k in _AGG_NAME_MAP.keys()) + r")\s*\("
    contains_agg = bool(re.search(agg_pattern, expr, flags=re.IGNORECASE))
    if contains_agg:
        if alias == expr:
            alias = f"agg_expr_{idx}"
        return (
            SelectItem(
                raw_expr=expr,
                alias=alias,
                is_aggregate=True,
                agg_func=None,
                arg_expr=expr,
            ),
            True,
        )

    src_col = _unquote_ident(expr) if re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", expr) else None
    if alias == expr and not src_col:
        alias = f"col_{idx}"
    return (
        SelectItem(
            raw_expr=expr,
            alias=alias,
            is_aggregate=False,
            source_col=src_col,
        ),
        False,
    )


_SQL_EXPR_FUNCTIONS = {
    "LOWER", "UPPER", "COALESCE", "CAST", "ABS", "ROUND", "DATE", "TIME",
    "DATETIME", "STRFTIME", "SUBSTR", "SUBSTRING", "TRIM", "LTRIM", "RTRIM",
    "LENGTH", "CHAR_LENGTH", "IFNULL", "NULLIF", "IIF", "IF", "JSON_EXTRACT",
}


def parse_view_index_spec(
    raw_spec: ViewIndexSpec | str | Sequence[str] | Dict[str, Any],
    default_view_name: Optional[str] = None,
) -> ViewIndexSpec:
    """
    Parses a user-provided index specification into a `ViewIndexSpec`.
    Supports:
      - `ViewIndexSpec` instances
      - `CREATE [UNIQUE] INDEX [IF NOT EXISTS] [name] ON <view> [USING <method>] (<cols>) [WHERE <pred>]`
      - `[UNIQUE] INDEX [name] (<cols>) [WHERE <pred>]`
      - Shorthand strings: `"col1, col2 DESC"`, `"UNIQUE:col1, col2"`, `"idx_name:col1, col2"`,
        `"UNIQUE:idx_name:col1, col2"`, `"view_name(col1, col2 DESC)"`
      - Sequences of column strings: `("col1", "col2 DESC")`
      - Dicts: `{"columns": ["col1", "col2 DESC"], "name": "...", "unique": True, "where": "..."}`
    """
    if isinstance(raw_spec, ViewIndexSpec):
        return raw_spec

    if isinstance(raw_spec, dict):
        raw_cols = raw_spec.get("columns") or raw_spec.get("cols") or ()
        if isinstance(raw_cols, str):
            cols = tuple(c.strip() for c in _split_top_level(raw_cols, ",") if c.strip())
        else:
            cols = tuple(str(c).strip() for c in raw_cols if str(c).strip())
        if not cols:
            raise ValueError(f"Index specification dict must contain non-empty 'columns': {raw_spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=raw_spec.get("name"),
            unique=bool(raw_spec.get("unique", False)),
            where_clause=raw_spec.get("where") or raw_spec.get("where_clause"),
            using_method=raw_spec.get("using") or raw_spec.get("using_method"),
            target_view=raw_spec.get("view") or raw_spec.get("target_view") or default_view_name,
        )

    if isinstance(raw_spec, (list, tuple)):
        cols = tuple(str(c).strip() for c in raw_spec if str(c).strip())
        if not cols:
            raise ValueError(f"Index column sequence cannot be empty: {raw_spec!r}")
        return ViewIndexSpec(columns=cols, target_view=default_view_name)

    if not isinstance(raw_spec, str):
        raise ValueError(f"Unsupported index specification type: {type(raw_spec)}")

    s = _strip_sql_comments(raw_spec).strip().rstrip(";").strip()
    if not s:
        raise ValueError("Empty index specification string")

    # 1. Full SQL: CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] [name] ON <view> [USING method] (<cols>) [WHERE ...]
    m_create = re.match(
        r"^CREATE\s+(UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?:([A-Za-z0-9_.\"`\[\]]+)\s+)?ON\s+([A-Za-z0-9_.\"`\[\]]+)\s*"
        r"(?:USING\s+([A-Za-z0-9_]+)\s*)?\((.*)$",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_create:
        is_unique = bool(m_create.group(1))
        idx_name = _unquote_ident(m_create.group(2)) if m_create.group(2) else None
        target_v = _unquote_ident(m_create.group(3))
        using_method = m_create.group(4).upper() if m_create.group(4) else None
        tail = m_create.group(5)
        depth = 1
        in_s = False
        in_d = False
        close_pos = -1
        for i, ch in enumerate(tail):
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif not in_s and not in_d:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_pos = i
                        break
        if close_pos < 0:
            raise ValueError(f"Unclosed parenthesis in CREATE INDEX statement: {raw_spec!r}")
        cols_str = tail[:close_pos].strip()
        after_cols = tail[close_pos + 1 :].strip()
        where_clause: Optional[str] = None
        if after_cols:
            m_where = re.match(r"^WHERE\s+(.*)$", after_cols, flags=re.IGNORECASE | re.DOTALL)
            if not m_where:
                raise ValueError(f"Unexpected trailing clause in CREATE INDEX: {after_cols!r}")
            where_clause = m_where.group(1).strip()
        cols = tuple(c.strip() for c in _split_top_level(cols_str, ",") if c.strip())
        if not cols:
            raise ValueError(f"No columns specified in CREATE INDEX: {raw_spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=idx_name,
            unique=is_unique,
            where_clause=where_clause,
            using_method=using_method,
            target_view=target_v,
        )

    # 2. Inline clause: [UNIQUE] INDEX [name] [USING method] (<cols>) [WHERE ...]
    m_inline = re.match(
        r"^(UNIQUE\s+)?INDEX\s+(?:([A-Za-z0-9_.\"`\[\]]+)\s*)?"
        r"(?:USING\s+([A-Za-z0-9_]+)\s*)?\((.*)$",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_inline:
        is_unique = bool(m_inline.group(1))
        idx_name = _unquote_ident(m_inline.group(2)) if m_inline.group(2) else None
        using_method = m_inline.group(3).upper() if m_inline.group(3) else None
        tail = m_inline.group(4)
        depth = 1
        in_s = False
        in_d = False
        close_pos = -1
        for i, ch in enumerate(tail):
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == '"' and not in_s:
                in_d = not in_d
            elif not in_s and not in_d:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_pos = i
                        break
        if close_pos < 0:
            raise ValueError(f"Unclosed parenthesis in INDEX clause: {raw_spec!r}")
        cols_str = tail[:close_pos].strip()
        after_cols = tail[close_pos + 1 :].strip()
        where_clause = None
        if after_cols:
            m_where = re.match(r"^WHERE\s+(.*)$", after_cols, flags=re.IGNORECASE | re.DOTALL)
            if not m_where:
                raise ValueError(f"Unexpected trailing clause in INDEX definition: {after_cols!r}")
            where_clause = m_where.group(1).strip()
        cols = tuple(c.strip() for c in _split_top_level(cols_str, ",") if c.strip())
        if not cols:
            raise ValueError(f"No columns specified in INDEX clause: {raw_spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=idx_name,
            unique=is_unique,
            where_clause=where_clause,
            using_method=using_method,
            target_view=default_view_name,
        )

    # 3. Shorthand string with optional view/unique/name prefixes:
    #    e.g. "UNIQUE(col1, col2)", "my_view(col1, col2 DESC)",
    #    "UNIQUE:col1, col2", "idx_custom:col1, col2", "UNIQUE:idx_custom:col1, col2"
    target_v = default_view_name
    is_unique = False
    idx_name = None
    where_clause = None

    m_where_tail = _find_top_level_keyword(s, r"\bWHERE\b")
    if m_where_tail:
        where_clause = s[m_where_tail.end() :].strip()
        s = s[: m_where_tail.start()].strip()

    while ":" in s:
        prefix, rest = s.split(":", 1)
        p_clean = prefix.strip()
        if not p_clean or not re.match(r"^[A-Za-z0-9_]+$", p_clean):
            break
        if p_clean.upper() == "UNIQUE":
            is_unique = True
            s = rest.strip()
        else:
            idx_name = p_clean
            s = rest.strip()

    m_func_like = re.match(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", s, flags=re.DOTALL)
    if m_func_like:
        head = m_func_like.group(1)
        inner = m_func_like.group(2).strip()
        if head.upper() == "UNIQUE":
            is_unique = True
            s = inner
        elif head.upper() not in _SQL_EXPR_FUNCTIONS:
            target_v = head
            s = inner

    cols = tuple(c.strip() for c in _split_top_level(s, ",") if c.strip())
    if not cols:
        raise ValueError(f"Could not parse columns from index specification: {raw_spec!r}")
    return ViewIndexSpec(
        columns=cols,
        name=idx_name,
        unique=is_unique,
        where_clause=where_clause,
        target_view=target_v,
    )


def _extract_inline_view_indexes(header_tail: str, view_name: str) -> List[ViewIndexSpec]:
    """Parses inline `WITH INDEX (...)` / `[UNIQUE] INDEX [name] (...)` clauses before `AS`."""
    s = header_tail.strip()
    if not s:
        return []
    s = re.sub(r"^WITH\s+", "", s, flags=re.IGNORECASE).strip()
    specs: List[ViewIndexSpec] = []
    for part in _split_top_level(s, ","):
        p = part.strip()
        if p:
            specs.append(parse_view_index_spec(p, default_view_name=view_name))
    return specs



def _try_unwrap_simple_cte(raw_select_sql: str, known_tables: Optional[Set[str]] = None) -> Optional[str]:
    """
    If `raw_select_sql` is a single non-recursive CTE of the form:
        WITH <cte> AS (SELECT <cols> FROM <base_table> [WHERE <w1>])
        SELECT ... FROM <cte> [WHERE <w2>] [GROUP BY ...] [HAVING ...]
    where the inner CTE is a pure single-table filter/rename with no aggregation,
    joins, set ops, distinct, window functions, or limit, unwraps it into an
    equivalent single-table SELECT so it can be maintained in O(1) ALGEBRAIC_DELTA.
    """
    cte_names, main_select, has_cte = _extract_cte_names_and_strip_with(raw_select_sql)
    if not has_cte or len(cte_names) != 1:
        return None
    if re.match(r"^WITH\s+RECURSIVE\b", raw_select_sql.strip(), flags=re.IGNORECASE):
        return None
    cte_name = next(iter(cte_names))
    m_cte = re.match(
        r"^WITH\s+([A-Za-z0-9_\"`\[\]]+)\s*AS\s*(?:(?:MATERIALIZED|NOT\s+MATERIALIZED)\s*)?\((.*)\)\s*"
        + re.escape(main_select)
        + r"\s*$",
        raw_select_sql.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_cte:
        return None
    inner_sql = m_cte.group(2).strip()
    if _find_top_level_keyword(inner_sql, r"\b(?:UNION|INTERSECT|EXCEPT|GROUP\s+BY|HAVING|WINDOW|ORDER\s+BY|LIMIT|JOIN)\b"):
        return None
    if re.search(r"\bOVER\s*\(|\bSELECT\s+DISTINCT\b|\(\s*SELECT\b", inner_sql, flags=re.IGNORECASE):
        return None
    m_inner_from = _find_top_level_keyword(inner_sql, r"\bFROM\b")
    if not m_inner_from or not re.match(r"^SELECT\b", inner_sql, flags=re.IGNORECASE):
        return None
    inner_sel_str = re.sub(r"^SELECT\s+", "", inner_sql[: m_inner_from.start()].strip(), flags=re.IGNORECASE).strip()
    if inner_sel_str == "*":
        return None
    inner_from_rest = inner_sql[m_inner_from.end() :].strip()
    m_inner_where = _find_top_level_keyword(inner_from_rest, r"\bWHERE\b")
    if m_inner_where:
        inner_from_tbl = inner_from_rest[: m_inner_where.start()].strip()
        inner_where = inner_from_rest[m_inner_where.end() :].strip()
    else:
        inner_from_tbl = inner_from_rest
        inner_where = None
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", inner_from_tbl):
        return None
    if known_tables and inner_from_tbl.lower() not in known_tables:
        return None

    col_map: Dict[str, str] = {}
    for idx, raw_it in enumerate(_split_top_level(inner_sel_str, ",")):
        it, is_comp = _parse_select_item(raw_it, idx)
        if it.is_aggregate or is_comp:
            return None
        col_map[it.alias.lower()] = it.raw_expr.strip()

    # Check that main_select selects from cte_name only (no joins or set ops)
    if _find_top_level_keyword(main_select, r"\b(?:UNION|INTERSECT|EXCEPT|JOIN|WINDOW|ORDER\s+BY|LIMIT)\b"):
        return None
    m_outer_from = _find_top_level_keyword(main_select, r"\bFROM\b")
    if not m_outer_from:
        return None
    outer_before_from = main_select[: m_outer_from.start()].strip()
    outer_from_rest = main_select[m_outer_from.end() :].strip()

    kw_pos = []
    for kw_n, kw_p in [("WHERE", r"\bWHERE\b"), ("GROUP_BY", r"\bGROUP\s+BY\b"), ("HAVING", r"\bHAVING\b")]:
        mk = _find_top_level_keyword(outer_from_rest, kw_p)
        if mk:
            kw_pos.append((kw_n, mk.start(), mk.end()))
    kw_pos.sort(key=lambda x: x[1])
    outer_from_tbl = outer_from_rest[: kw_pos[0][1]].strip() if kw_pos else outer_from_rest.strip()
    m_tbl = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?$", outer_from_tbl, flags=re.IGNORECASE)
    if not m_tbl or m_tbl.group(1).lower() != cte_name.lower():
        return None
    outer_cte_alias = m_tbl.group(2) or cte_name

    clauses: Dict[str, str] = {}
    for idx, (kw_n, _, kw_e) in enumerate(kw_pos):
        end_p = kw_pos[idx + 1][1] if idx + 1 < len(kw_pos) else len(outer_from_rest)
        clauses[kw_n] = outer_from_rest[kw_e:end_p].strip()

    def _subst_cte_cols(frag: str) -> str:
        tokens = re.split(r"('(?:''|[^'])*')", frag)
        out = []
        for i_t, tok in enumerate(tokens):
            if i_t % 2 == 1:
                out.append(tok)
                continue
            s = re.sub(rf"\b(?:{re.escape(cte_name)}|{re.escape(outer_cte_alias)})\s*\.\s*", "", tok, flags=re.IGNORECASE)
            for c_al, c_ex in sorted(col_map.items(), key=lambda x: len(x[0]), reverse=True):
                if c_al != c_ex.lower():
                    s = re.sub(rf"(?<![.\w])\b{re.escape(c_al)}\b(?!\s*\()", f"({c_ex})", s, flags=re.IGNORECASE)
            out.append(s)
        return "".join(out)

    new_select_part = _subst_cte_cols(outer_before_from)
    outer_w = _subst_cte_cols(clauses["WHERE"]) if "WHERE" in clauses else None
    if inner_where and outer_w:
        comb_w = f"({inner_where}) AND ({outer_w})"
    else:
        comb_w = inner_where or outer_w
    res = f"{new_select_part} FROM {inner_from_tbl}"
    if comb_w:
        res += f" WHERE {comb_w}"
    if "GROUP_BY" in clauses:
        res += f" GROUP BY {_subst_cte_cols(clauses['GROUP_BY'])}"
    if "HAVING" in clauses:
        res += f" HAVING {_subst_cte_cols(clauses['HAVING'])}"
    return res


def _parse_single_create_view_stmt(
    first_stmt: str,
    known_tables: Optional[Set[str]] = None,
    extra_indexes: Optional[Sequence[ViewIndexSpec]] = None,
) -> ViewSpec:
    m = re.match(
        r"^CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:MATERIALIZED\s+)?VIEW\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"`\[\]]+)"
        r"(?:\s+((?:WITH\s+)?(?:UNIQUE\s+)?INDEX\b.*?))?\s+AS\s+((?:WITH|SELECT)\b.*)$",
        first_stmt.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        raise ValueError(f"Invalid CREATE VIEW syntax: {first_stmt!r}")

    view_name = _unquote_ident(m.group(1))
    inline_index_header = (m.group(2) or "").strip()
    raw_select_sql = m.group(3).strip()

    parsed_indexes: List[ViewIndexSpec] = []
    if inline_index_header:
        parsed_indexes.extend(_extract_inline_view_indexes(inline_index_header, view_name))
    if extra_indexes:
        parsed_indexes.extend(extra_indexes)

    unwrapped_sql = _try_unwrap_simple_cte(raw_select_sql, known_tables)
    effective_select_sql = unwrapped_sql if unwrapped_sql is not None else raw_select_sql
    cte_names, main_select, has_cte = _extract_cte_names_and_strip_with(effective_select_sql)
    referenced_tables = _extract_all_referenced_tables(effective_select_sql, cte_names, known_tables)

    has_compound_set_op = bool(
        _find_top_level_keyword(main_select, r"\b(?:UNION|INTERSECT|EXCEPT)\b")
    )
    has_window_func = bool(re.search(r"\bOVER\s*\(", main_select, flags=re.IGNORECASE))

    first_branch = main_select
    if has_compound_set_op:
        m_set = _find_top_level_keyword(main_select, r"\b(?:UNION|INTERSECT|EXCEPT)\b")
        if m_set:
            first_branch = main_select[: m_set.start()].strip()

    if not re.match(r"^SELECT\b", first_branch, flags=re.IGNORECASE):
        raise ValueError(f"Expected SELECT in view definition: {first_stmt!r}")

    after_select = re.sub(r"^SELECT\s+", "", first_branch, flags=re.IGNORECASE).strip()
    has_top_level_distinct = False
    if re.match(r"^DISTINCT\b", after_select, flags=re.IGNORECASE):
        has_top_level_distinct = True
        after_select = re.sub(r"^DISTINCT\s+", "", after_select, flags=re.IGNORECASE).strip()
    elif re.match(r"^ALL\b", after_select, flags=re.IGNORECASE):
        after_select = re.sub(r"^ALL\s+", "", after_select, flags=re.IGNORECASE).strip()

    m_from = _find_top_level_keyword(after_select, r"\bFROM\b")
    if not m_from:
        select_list_str = after_select
        from_rest = ""
    else:
        select_list_str = after_select[: m_from.start()].strip()
        from_rest = after_select[m_from.end() :].strip()

    kw_positions: List[Tuple[str, int, int]] = []
    for kw_name, kw_pat in [
        ("WHERE", r"\bWHERE\b"),
        ("GROUP_BY", r"\bGROUP\s+BY\b"),
        ("HAVING", r"\bHAVING\b"),
        ("WINDOW", r"\bWINDOW\b"),
        ("ORDER_BY", r"\bORDER\s+BY\b"),
        ("LIMIT", r"\bLIMIT\b"),
    ]:
        m_kw = _find_top_level_keyword(from_rest, kw_pat)
        if m_kw:
            kw_positions.append((kw_name, m_kw.start(), m_kw.end()))

    kw_positions.sort(key=lambda x: x[1])

    from_clause = from_rest[: kw_positions[0][1]].strip() if kw_positions else from_rest.strip()
    clauses_dict: Dict[str, str] = {}
    for idx, (kw_name, _, kw_end) in enumerate(kw_positions):
        end_pos = kw_positions[idx + 1][1] if idx + 1 < len(kw_positions) else len(from_rest)
        clauses_dict[kw_name] = from_rest[kw_end:end_pos].strip()

    where_clause = clauses_dict.get("WHERE")
    group_by_str = clauses_dict.get("GROUP_BY")
    having_clause = clauses_dict.get("HAVING")
    has_order_by = "ORDER_BY" in clauses_dict
    has_limit = "LIMIT" in clauses_dict

    if has_compound_set_op:
        if _find_top_level_keyword(main_select, r"\bORDER\s+BY\b"):
            has_order_by = True
        if _find_top_level_keyword(main_select, r"\bLIMIT\b"):
            has_limit = True

    joins = _parse_from_and_joins(from_clause) if from_clause else []
    base_table = (
        joins[0].table_name
        if joins
        else (referenced_tables[0] if referenced_tables else "dual")
    )
    if base_table and base_table.lower() not in [t.lower() for t in referenced_tables]:
        if base_table.lower() not in cte_names:
            referenced_tables.insert(0, base_table)

    select_items: List[SelectItem] = []
    has_complex_agg_expr = False
    for idx, raw_item in enumerate(_split_top_level(select_list_str, ",")):
        item, is_complex = _parse_select_item(raw_item, idx)
        select_items.append(item)
        if is_complex:
            has_complex_agg_expr = True

    alias_to_nonagg_expr: Dict[str, str] = {
        it.alias.lower(): it.raw_expr.strip()
        for it in select_items
        if not it.is_aggregate
    }
    group_by_exprs: List[str] = []
    if group_by_str:
        for raw_g in _split_top_level(group_by_str, ","):
            g_clean = raw_g.strip()
            if g_clean.isdigit():
                ord_idx = int(g_clean) - 1
                if 0 <= ord_idx < len(select_items) and not select_items[ord_idx].is_aggregate:
                    g_clean = select_items[ord_idx].raw_expr.strip()
            else:
                g_unq = _unquote_ident(g_clean).lower()
                if g_unq in alias_to_nonagg_expr and re.match(r"^[A-Za-z0-9_\"`\[\]]+$", g_clean):
                    g_clean = alias_to_nonagg_expr[g_unq]
            group_by_exprs.append(g_clean)

    has_subquery_in_from_or_where = (
        from_clause.startswith("(")
        or bool(re.search(r"\(\s*SELECT\b", from_clause, flags=re.IGNORECASE))
        or bool(where_clause and re.search(r"\(\s*SELECT\b", where_clause, flags=re.IGNORECASE))
    )

    return ViewSpec(
        view_name=view_name,
        raw_select_sql=raw_select_sql,
        base_table=base_table,
        joins=joins,
        referenced_tables=referenced_tables,
        select_items=select_items,
        group_by_exprs=group_by_exprs,
        where_clause=where_clause,
        having_clause=having_clause,
        has_cte=has_cte,
        has_compound_set_op=has_compound_set_op,
        has_window_func=has_window_func,
        has_order_by=has_order_by,
        has_limit=has_limit,
        has_top_level_distinct=has_top_level_distinct,
        has_subquery_in_from_or_where=has_subquery_in_from_or_where,
        has_complex_agg_expr=has_complex_agg_expr,
        user_indexes=tuple(parsed_indexes),
    )


def parse_view_definitions(
    view_sql_or_list: ViewSpec | str | Sequence[ViewSpec | str],
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
) -> List[ViewSpec]:
    """
    Parses one or more `CREATE [MATERIALIZED] VIEW ...` statements (from a single
    multi-view `.sql` string/file or a sequence of strings/`ViewSpec`s) along with any
    accompanying `CREATE [UNIQUE] INDEX ...` statements into a list of `ViewSpec` objects.
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

    # Attach each pending ViewIndexSpec to its matching ViewSpec
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
                v_unpref = v_low[len(DEFAULT_OBJECT_PREFIX):] if v_low.startswith(DEFAULT_OBJECT_PREFIX) else v_low
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
                # Keep it on view 0 so IncrementalViewCompiler can raise its target mismatch error
                by_view_idx[0].append(idx_spec)
        else:
            # Unscoped index when multiple views exist: attach to views that expose all referenced columns
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
                group_by_exprs=vs.group_by_exprs,
                where_clause=vs.where_clause,
                having_clause=vs.having_clause,
                has_cte=vs.has_cte,
                has_compound_set_op=vs.has_compound_set_op,
                has_window_func=vs.has_window_func,
                has_order_by=vs.has_order_by,
                has_limit=vs.has_limit,
                has_top_level_distinct=vs.has_top_level_distinct,
                has_subquery_in_from_or_where=vs.has_subquery_in_from_or_where,
                has_complex_agg_expr=vs.has_complex_agg_expr,
                user_indexes=tuple(by_view_idx[i]),
            )
        )
    return out_specs


def parse_view_definition(
    view_sql: str,
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
) -> ViewSpec:
    """
    Parses a `CREATE [MATERIALIZED] VIEW ...` definition string into a `ViewSpec`.
    If `view_sql` contains multiple `CREATE VIEW` statements, attaches the subsequent
    views to `ViewSpec.additional_views` so both single-view and multi-view callers
    succeed seamlessly.
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
        group_by_exprs=first.group_by_exprs,
        where_clause=first.where_clause,
        having_clause=first.having_clause,
        has_cte=first.has_cte,
        has_compound_set_op=first.has_compound_set_op,
        has_window_func=first.has_window_func,
        has_order_by=first.has_order_by,
        has_limit=first.has_limit,
        has_top_level_distinct=first.has_top_level_distinct,
        has_subquery_in_from_or_where=first.has_subquery_in_from_or_where,
        has_complex_agg_expr=first.has_complex_agg_expr,
        user_indexes=first.user_indexes,
        additional_views=tuple(specs[1:]),
    )


# -----------------------------------------------------------------------------
# Incremental Materialized View Compiler (3-Tier Architecture + Limitation Guard)
# -----------------------------------------------------------------------------


class IncrementalViewCompiler:
    """
    Compiles a schema (one or more `TableSchema`s) and a `ViewSpec` into a
    pure native-SQL `MaterializedViewPlan` for SQLite, PostgreSQL, or MySQL.

    If `ignore_limitations=False` (default), refuses compilation with a clear
    `EngineLimitationError` whenever the target engine, schema, or view definition
    cannot guarantee 100% transparent behavior for all possible runtime operations.
    """

    def __init__(
        self,
        schema: Dict[str, TableSchema] | TableSchema | str,
        view_sql: ViewSpec | str,
        dialect: Dialect | str = Dialect.SQLITE,
        pg_use_transition_tables: bool = False,
        ignore_limitations: bool = False,
        indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        incremental_backfill: bool = False,
        batch_size: int = 500,
    ) -> None:
        if isinstance(dialect, str):
            dialect = Dialect(dialect.lower())
        self.dialect = dialect
        self.pg_use_transition_tables = pg_use_transition_tables
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.incremental_backfill = incremental_backfill
        self.batch_size = max(1, int(batch_size)) if batch_size else 500

        if isinstance(schema, str):
            self.schemas = parse_table_schema(schema)
        elif isinstance(schema, TableSchema):
            self.schemas = {schema.name: schema}
        else:
            self.schemas = dict(schema)

        self._schemas_lower: Dict[str, TableSchema] = {
            k.lower(): v for k, v in self.schemas.items()
        }

        if isinstance(view_sql, str):
            self.view = parse_view_definition(
                view_sql,
                known_tables=set(self._schemas_lower.keys()),
                indexes=indexes,
            )
        else:
            if indexes:
                extra_parsed = tuple(
                    parse_view_index_spec(raw_idx, default_view_name=view_sql.view_name)
                    for raw_idx in indexes
                )
                combined_idxs = tuple(list(view_sql.user_indexes) + list(extra_parsed))
                self.view = ViewSpec(
                    view_name=view_sql.view_name,
                    raw_select_sql=view_sql.raw_select_sql,
                    base_table=view_sql.base_table,
                    joins=view_sql.joins,
                    referenced_tables=view_sql.referenced_tables,
                    select_items=view_sql.select_items,
                    group_by_exprs=view_sql.group_by_exprs,
                    where_clause=view_sql.where_clause,
                    having_clause=view_sql.having_clause,
                    has_cte=view_sql.has_cte,
                    has_compound_set_op=view_sql.has_compound_set_op,
                    has_window_func=view_sql.has_window_func,
                    has_order_by=view_sql.has_order_by,
                    has_limit=view_sql.has_limit,
                    has_top_level_distinct=view_sql.has_top_level_distinct,
                    has_subquery_in_from_or_where=view_sql.has_subquery_in_from_or_where,
                    has_complex_agg_expr=view_sql.has_complex_agg_expr,
                    user_indexes=combined_idxs,
                )
            else:
                self.view = view_sql

        raw_vname = self.view.view_name
        if self.object_prefix and raw_vname.startswith(self.object_prefix):
            self.logical_view_name = raw_vname[len(self.object_prefix) :]
        else:
            self.logical_view_name = raw_vname
        self.db_view_name = f"{self.object_prefix}{self.logical_view_name}"

        # Filter user_indexes to those targeting this view (if multi-view scoped indexes were passed)
        valid_targets = {
            self.logical_view_name.lower(),
            self.db_view_name.lower(),
            self.view.view_name.lower(),
            self._backing_table_name().lower(),
            f"__mv_store_{self.logical_view_name}".lower(),
        }
        filtered_indexes: List[ViewIndexSpec] = []
        for idx_spec in self.view.user_indexes:
            if idx_spec.target_view is not None:
                t_low = idx_spec.target_view.lower()
                if self.object_prefix and t_low.startswith(self.object_prefix.lower()):
                    t_stripped = t_low[len(self.object_prefix) :]
                else:
                    t_stripped = t_low
                if t_low not in valid_targets and t_stripped not in valid_targets:
                    raise ValueError(
                        f"Index target '{idx_spec.target_view}' does not match materialized view "
                        f"'{self.logical_view_name}' (or '{self.db_view_name}')."
                    )
            filtered_indexes.append(idx_spec)
        self.user_indexes: List[ViewIndexSpec] = filtered_indexes

        self.alias_to_schema: Dict[str, TableSchema] = {}
        for j in self.view.joins:
            t_low = j.table_name.lower()
            if t_low in self._schemas_lower:
                self.alias_to_schema[j.alias.lower()] = self._schemas_lower[t_low]
                self.alias_to_schema[t_low] = self._schemas_lower[t_low]

        self.primary_schema: Optional[TableSchema] = self._schemas_lower.get(
            self.view.base_table.lower()
        )
        self._alg_mode: str = "single_table"
        self._working_select_items: List[SelectItem] = list(self.view.select_items)
        self._view_select_exprs: Optional[List[str]] = None
        self._view_where_clause: Optional[str] = None
        self._decomposed_ok: bool = True
        self._effective_group_by_exprs: List[str] = list(self.view.group_by_exprs)
        self._setop_branches: Optional[List[Tuple[str, str, Optional[str], List[SelectItem]]]] = None
        self._setop_kind: Optional[str] = None
        self._union_all_agg_branches: Optional[List[Tuple[str, Optional[str], Dict[str, str]]]] = None
        self._prepare_algebraic_decomposition()
        self.strategy = self._select_strategy()
        self._validate()
        self.limitations = self._detect_limitations()
        self._enforce_limitations()

    def _backing_table_name(self) -> str:
        return f"{self.object_prefix}store_{self.logical_view_name}"

    def _prefixed_identifier(self, raw_ident: str) -> str:
        clean = _unquote_ident(raw_ident)
        if self.object_prefix and not clean.startswith(self.object_prefix):
            return f"{self.object_prefix}{clean}"
        return clean

    def _detect_limitations(self) -> List[LimitationItem]:
        """
        Detects any engine, schema, or view limitations where the materialized view
        cannot be guaranteed 100% transparent across all possible runtime operations.
        """
        items: List[LimitationItem] = []

        # 1. MySQL engine-level trigger limitations (TRUNCATE TABLE & FK cascades)
        if self.dialect == Dialect.MYSQL:
            tables_str = ", ".join(self.view.referenced_tables)
            items.append(
                LimitationItem(
                    code="MYSQL_NO_TRUNCATE_TRIGGER",
                    message=(
                        f"MySQL treats TRUNCATE TABLE on base table(s) ({tables_str}) as a DDL "
                        f"operation that bypasses AFTER DELETE triggers, and MySQL does not "
                        f"support AFTER TRUNCATE triggers."
                    ),
                )
            )
            for ref_t in self.view.referenced_tables:
                t_schema = self._schemas_lower.get(ref_t.lower())
                if t_schema and t_schema.fk_cascades:
                    casc_str = ", ".join(t_schema.fk_cascades)
                    items.append(
                        LimitationItem(
                            code="MYSQL_FK_CASCADE_NO_TRIGGER",
                            message=(
                                f"Base table '{t_schema.name}' defines foreign-key cascading action(s) "
                                f"({casc_str}), which do not activate AFTER DELETE or AFTER UPDATE "
                                f"triggers in MySQL InnoDB."
                            ),
                        )
                    )

        # 2. SQLite runtime per-connection pragma dependency when schema uses FK cascades or ON CONFLICT REPLACE
        if self.dialect == Dialect.SQLITE:
            for ref_t in self.view.referenced_tables:
                t_schema = self._schemas_lower.get(ref_t.lower())
                if t_schema and (t_schema.fk_cascades or t_schema.has_on_conflict_replace):
                    reasons: List[str] = []
                    if t_schema.fk_cascades:
                        reasons.append(f"foreign-key cascade ({', '.join(t_schema.fk_cascades)})")
                    if t_schema.has_on_conflict_replace:
                        reasons.append("ON CONFLICT REPLACE constraint")
                    items.append(
                        LimitationItem(
                            code="SQLITE_SESSION_PRAGMA_DEPENDENCY",
                            message=(
                                f"Base table '{t_schema.name}' uses {' and '.join(reasons)}, which "
                                f"requires every runtime database connection to enable "
                                f"'PRAGMA foreign_keys = ON' and 'PRAGMA recursive_triggers = ON' "
                                f"for triggers to fire."
                            ),
                        )
                    )

        # 3. Non-deterministic or volatile time/random functions in the view query
        raw_no_strings = re.sub(r"'(?:''|[^'])*'", "''", self.view.raw_select_sql)
        nondet_matches: List[str] = []
        for pat, label in [
            (r"\bRANDOM\s*\(", "RANDOM()"),
            (r"\bRAND\s*\(", "RAND()"),
            (r"\bNOW\s*\(", "NOW()"),
            (r"\bCURRENT_TIMESTAMP\b", "CURRENT_TIMESTAMP"),
            (r"\bCURRENT_DATE\b", "CURRENT_DATE"),
            (r"\bCURRENT_TIME\b", "CURRENT_TIME"),
            (r"\bCLOCK_TIMESTAMP\s*\(", "CLOCK_TIMESTAMP()"),
            (r"\bUNIXEPOCH\s*\(\s*\)", "UNIXEPOCH()"),
            (r"\bUUID\s*\(", "UUID()"),
            (r"\bGEN_RANDOM_UUID\s*\(", "GEN_RANDOM_UUID()"),
        ]:
            if re.search(pat, raw_no_strings, flags=re.IGNORECASE):
                nondet_matches.append(label)
        if re.search(
            r"\b(?:DATETIME|DATE|TIME|STRFTIME)\s*\([^)]*'now'",
            self.view.raw_select_sql,
            flags=re.IGNORECASE,
        ):
            nondet_matches.append("DATETIME('now')")

        if nondet_matches:
            items.append(
                LimitationItem(
                    code="NON_DETERMINISTIC_VIEW_EXPRESSION",
                    message=(
                        f"View definition uses non-deterministic or time-dependent expression(s) "
                        f"({', '.join(nondet_matches)}); materialized state only updates on base-table "
                        f"DML and will diverge from live queries over time or across reads."
                    ),
                )
            )

        # 4. LIMIT without ORDER BY (non-deterministic row subset)
        if self.view.has_limit and not self.view.has_order_by:
            items.append(
                LimitationItem(
                    code="LIMIT_WITHOUT_ORDER_BY",
                    message=(
                        "View definition uses LIMIT without an ORDER BY clause, so the subset of "
                        "rows selected is non-deterministic."
                    ),
                )
            )

        # 5. Approximate floating-point GROUP BY keys
        for g_expr in self.view.group_by_exprs:
            c_def = self._lookup_col_def(g_expr)
            if c_def and any(
                ft in c_def.data_type.upper() for ft in ("REAL", "FLOAT", "DOUBLE")
            ):
                items.append(
                    LimitationItem(
                        code="FLOATING_POINT_GROUP_KEY",
                        message=(
                            f"GROUP BY column '{c_def.name}' has approximate floating-point type "
                            f"'{c_def.data_type}', which can cause group-key equality discrepancies "
                            f"(e.g. -0.0 vs +0.0 or rounding precision)."
                        ),
                    )
                )

        # 6. Dialect-unsupported aggregate functions (must be natively supported by the target engine at runtime)
        native_set = _DIALECT_NATIVE_AGGS[self.dialect]
        check_items = self._working_select_items if self._decomposed_ok else self.view.select_items
        for item in check_items:
            if item.is_aggregate and item.agg_func is not None:
                if item.agg_func not in native_set:
                    fn_label = item.raw_func_name or item.agg_func.value
                    items.append(
                        LimitationItem(
                            code="DIALECT_UNSUPPORTED_AGGREGATE",
                            message=(
                                f"Aggregate function '{fn_label}' in column '{item.alias}' is not a "
                                f"native built-in aggregate in {self.dialect.value}."
                            ),
                        )
                    )

        # 7. User-specified index dialect limitations (e.g., partial indexes on MySQL or USING method on SQLite)
        for idx_spec in self.user_indexes:
            if idx_spec.where_clause and self.dialect == Dialect.MYSQL:
                items.append(
                    LimitationItem(
                        code="MYSQL_UNSUPPORTED_PARTIAL_INDEX",
                        message=(
                            f"Index on ({', '.join(idx_spec.columns)}) specifies a partial index WHERE "
                            f"clause ({idx_spec.where_clause}), which is not supported by MySQL."
                        ),
                    )
                )
            if idx_spec.using_method:
                m_up = idx_spec.using_method.upper()
                if self.dialect == Dialect.SQLITE:
                    items.append(
                        LimitationItem(
                            code="DIALECT_UNSUPPORTED_INDEX_METHOD",
                            message=(
                                f"Index specifies USING {m_up}, which is not supported by SQLite CREATE INDEX syntax."
                            ),
                        )
                    )
                elif self.dialect == Dialect.MYSQL and m_up not in ("BTREE", "HASH"):
                    items.append(
                        LimitationItem(
                            code="DIALECT_UNSUPPORTED_INDEX_METHOD",
                            message=(
                                f"Index specifies USING {m_up}, which is not supported by MySQL (expected BTREE or HASH)."
                            ),
                        )
                    )

        return items

    def _enforce_limitations(self) -> None:
        if not self.limitations:
            return
        if not self.ignore_limitations:
            raise EngineLimitationError(
                view_name=self.view.view_name,
                dialect=self.dialect,
                limitations=self.limitations,
            )
        bullets = "; ".join(f"[{lim.code}] {lim.message}" for lim in self.limitations)
        warnings.warn(
            f"Proceeding with materialized view '{self.view.view_name}' on '{self.dialect.value}' "
            f"with --ignore-limitations: {bullets}",
            EngineLimitationWarning,
            stacklevel=2,
        )

    def _is_simple_col_ref(self, expr: str) -> bool:
        s = expr.strip()
        if not re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", s):
            return False
        return self._lookup_col_def(s) is not None

    @staticmethod
    def _extract_agg_call_spans(expr_sql: str) -> Optional[List[Tuple[int, int, str]]]:
        """
        Finds all top-level aggregate function call spans `(start, end, call_sql)`
        inside `expr_sql` (including any trailing `FILTER (WHERE ...)` clause).
        Returns `None` if a window function (`OVER (...)`) is encountered.
        """
        agg_pat = re.compile(
            r"\b(COUNT|SUM|TOTAL|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG|"
            r"JSON_GROUP_ARRAY|JSON_AGG|JSON_ARRAYAGG|JSON_GROUP_OBJECT|JSON_OBJECT_AGG|"
            r"ARRAY_AGG|VAR_POP|VARIANCE|VAR_SAMP|STDDEV_POP|STDDEV|STD|STDDEV_SAMP|"
            r"BIT_AND|BIT_OR|BIT_XOR|BOOL_AND|EVERY|BOOL_OR)\s*\(",
            re.IGNORECASE,
        )
        spans: List[Tuple[int, int, str]] = []
        n = len(expr_sql)
        i = 0
        in_single = False
        in_double = False
        while i < n:
            ch = expr_sql[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if in_single or in_double:
                i += 1
                continue
            m = agg_pat.match(expr_sql, i)
            if not m:
                i += 1
                continue
            start_idx = m.start()
            open_paren = m.end() - 1
            depth = 0
            j = open_paren
            s_q = False
            d_q = False
            while j < n:
                c2 = expr_sql[j]
                if c2 == "'" and not d_q:
                    s_q = not s_q
                elif c2 == '"' and not s_q:
                    d_q = not d_q
                elif not s_q and not d_q:
                    if c2 == "(":
                        depth += 1
                    elif c2 == ")":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                j += 1
            if depth != 0:
                return None
            end_idx = j
            rest_after = expr_sql[end_idx:]
            if re.match(r"^\s*OVER\s*\(", rest_after, flags=re.IGNORECASE):
                return None
            m_filt = re.match(r"^\s*FILTER\s*\(\s*WHERE\b", rest_after, flags=re.IGNORECASE)
            if m_filt:
                f_open = end_idx + rest_after.find("(")
                f_depth = 0
                k = f_open
                while k < n:
                    c3 = expr_sql[k]
                    if c3 == "'" and not d_q:
                        s_q = not s_q
                    elif c3 == '"' and not s_q:
                        d_q = not d_q
                    elif not s_q and not d_q:
                        if c3 == "(":
                            f_depth += 1
                        elif c3 == ")":
                            f_depth -= 1
                            if f_depth == 0:
                                k += 1
                                break
                    k += 1
                if f_depth != 0:
                    return None
                end_idx = k
            spans.append((start_idx, end_idx, expr_sql[start_idx:end_idx]))
            i = end_idx
        return spans

    def _prepare_algebraic_decomposition(self) -> None:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if v.has_top_level_distinct and not v.group_by_exprs and not has_any_agg:
            self._effective_group_by_exprs = [it.raw_expr.strip() for it in v.select_items]
        else:
            self._effective_group_by_exprs = list(v.group_by_exprs)

        has_compound_items = any(it.is_aggregate and it.agg_func is None for it in v.select_items)
        if not has_compound_items and not v.having_clause:
            self._working_select_items = list(v.select_items)
            self._view_select_exprs = None
            self._view_where_clause = None
            self._decomposed_ok = True
            return

        # If a user index targets a compound aggregate column, keep it as a physical column (Tier 2/3)
        indexed_cols = self._indexed_column_names_lower()
        for it in v.select_items:
            if it.is_aggregate and it.agg_func is None and it.alias.lower() in indexed_cols:
                self._decomposed_ok = False
                return

        if v.having_clause and re.search(r"\(\s*SELECT\b", v.having_clause, flags=re.IGNORECASE):
            self._decomposed_ok = False
            return

        working_items: List[SelectItem] = [
            it for it in v.select_items if not it.is_aggregate or it.agg_func is not None
        ]
        elem_counter = 1

        def _agg_sig(it: SelectItem) -> Tuple[Any, ...]:
            return (
                it.agg_func,
                it.is_distinct,
                (it.arg_expr or "").strip().lower(),
                (it.filter_where or "").strip().lower(),
                (it.extra_args or "").strip().lower(),
            )

        def _register_elem_agg(call_sql: str) -> Optional[str]:
            nonlocal elem_counter
            alias = f"__mv_elem_{elem_counter}"
            parsed_it, is_comp = _parse_select_item(f"{call_sql} AS {alias}", elem_counter)
            if is_comp or not parsed_it.is_aggregate or parsed_it.agg_func is None:
                return None
            sig = _agg_sig(parsed_it)
            for existing in working_items:
                if existing.is_aggregate and existing.agg_func is not None and _agg_sig(existing) == sig:
                    return existing.alias
            elem_counter += 1
            working_items.append(parsed_it)
            return parsed_it.alias

        def _rewrite_expr_aggs(expr_sql: str) -> Optional[str]:
            if re.search(r"\(\s*SELECT\b", expr_sql, flags=re.IGNORECASE):
                return None
            spans = self._extract_agg_call_spans(expr_sql)
            if spans is None:
                return None
            rewritten = expr_sql
            for st, en, raw_call in reversed(spans):
                el_alias = _register_elem_agg(raw_call)
                if el_alias is None:
                    return None
                rewritten = rewritten[:st] + el_alias + rewritten[en:]
            return rewritten

        view_select_exprs: List[str] = []
        compound_alias_map: Dict[str, str] = {}
        for it in v.select_items:
            if not it.is_aggregate or it.agg_func is not None:
                view_select_exprs.append(it.alias)
            else:
                rw = _rewrite_expr_aggs(it.raw_expr)
                if rw is None:
                    self._decomposed_ok = False
                    return
                # Also replace any group-by expression references inside rw
                for g_expr in self._effective_group_by_exprs:
                    g_mv = self._group_col_alias_for_items(g_expr, working_items)
                    if g_expr.strip() != g_mv:
                        rw = re.sub(rf"(?<![.\w]){re.escape(g_expr.strip())}(?![.\w])", g_mv, rw, flags=re.IGNORECASE)
                compound_alias_map[it.alias.lower()] = rw
                view_select_exprs.append(f"{rw} AS {it.alias}")

        view_where: Optional[str] = None
        if v.having_clause:
            rw_h = _rewrite_expr_aggs(v.having_clause)
            if rw_h is None:
                self._decomposed_ok = False
                return
            for c_al, c_rw in sorted(compound_alias_map.items(), key=lambda x: len(x[0]), reverse=True):
                rw_h = re.sub(rf"(?<![.\w])\b{re.escape(c_al)}\b(?!\s*\()", f"({c_rw})", rw_h, flags=re.IGNORECASE)
            for g_expr in self._effective_group_by_exprs:
                g_mv = self._group_col_alias_for_items(g_expr, working_items)
                if g_expr.strip() != g_mv:
                    rw_h = re.sub(rf"(?<![.\w]){re.escape(g_expr.strip())}(?![.\w])", g_mv, rw_h, flags=re.IGNORECASE)
            view_where = rw_h

        self._working_select_items = working_items
        self._view_select_exprs = view_select_exprs
        self._view_where_clause = view_where
        self._decomposed_ok = True

    def _parse_simple_branch_sql(
        self, branch_sql: str, expected_col_count: int
    ) -> Optional[Tuple[str, Optional[str], List[SelectItem]]]:
        """
        Parses a single-table non-aggregated SELECT branch `SELECT ... FROM tbl [WHERE ...]`.
        Returns `(table_name, where_clause, select_items)` or `None` if not simple.
        """
        b = branch_sql.strip().rstrip(";")
        if _find_top_level_keyword(
            b, r"\b(?:UNION|INTERSECT|EXCEPT|GROUP\s+BY|HAVING|WINDOW|ORDER\s+BY|LIMIT|JOIN)\b"
        ):
            return None
        if re.search(r"\bOVER\s*\(|\bSELECT\s+DISTINCT\b|\(\s*SELECT\b", b, flags=re.IGNORECASE):
            return None
        m_from = _find_top_level_keyword(b, r"\bFROM\b")
        if not m_from or not re.match(r"^SELECT\b", b, flags=re.IGNORECASE):
            return None
        sel_part = re.sub(r"^SELECT\s+(?:ALL\s+)?", "", b[: m_from.start()].strip(), flags=re.IGNORECASE).strip()
        if sel_part == "*":
            return None
        from_rest = b[m_from.end() :].strip()
        m_where = _find_top_level_keyword(from_rest, r"\bWHERE\b")
        if m_where:
            tbl_part = from_rest[: m_where.start()].strip()
            where_part = from_rest[m_where.end() :].strip()
        else:
            tbl_part = from_rest
            where_part = None
        m_tbl = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?$", tbl_part, flags=re.IGNORECASE)
        if not m_tbl:
            return None
        t_name = m_tbl.group(1)
        t_alias = m_tbl.group(2) or t_name
        if t_name.lower() not in self._schemas_lower:
            return None
        t_schema = self._schemas_lower[t_name.lower()]
        raw_items = _split_top_level(sel_part, ",")
        if expected_col_count > 0 and len(raw_items) != expected_col_count:
            return None
        items: List[SelectItem] = []
        for idx, r_it in enumerate(raw_items):
            it, is_comp = _parse_select_item(r_it, idx)
            if it.is_aggregate or is_comp:
                return None
            # Strip branch table alias if present
            clean_raw = re.sub(rf"\b{re.escape(t_alias)}\s*\.\s*", "", it.raw_expr, flags=re.IGNORECASE)
            items.append(
                SelectItem(
                    raw_expr=clean_raw,
                    alias=it.alias,
                    is_aggregate=False,
                    source_col=it.source_col,
                )
            )
        if where_part:
            where_part = re.sub(rf"\b{re.escape(t_alias)}\s*\.\s*", "", where_part, flags=re.IGNORECASE)
        return t_schema.name, where_part, items

    def _can_use_setop_algebraic(self) -> bool:
        v = self.view
        if not v.has_compound_set_op or v.has_cte or v.has_window_func or v.has_limit_or_order:
            return False
        matches = _find_all_top_level_keywords(
            v.raw_select_sql, r"\b(?:UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b"
        )
        if not matches:
            return False
        ops: List[str] = []
        branch_strs: List[str] = []
        prev_end = 0
        for m in matches:
            branch_strs.append(v.raw_select_sql[prev_end : m.start()].strip())
            op_norm = " ".join(m.group(0).upper().split())
            ops.append(op_norm)
            prev_end = m.end()
        branch_strs.append(v.raw_select_sql[prev_end:].strip())

        op_set = set(ops)
        if len(op_set) != 1:
            return False
        op_kind = ops[0]
        if op_kind in ("INTERSECT", "EXCEPT") and len(branch_strs) != 2:
            return False

        expected_cols = len(v.select_items)
        parsed_branches: List[Tuple[str, str, Optional[str], List[SelectItem]]] = []
        for idx, b_str in enumerate(branch_strs):
            res = self._parse_simple_branch_sql(b_str, expected_cols)
            if res is None:
                return False
            t_name, w_clause, b_items = res
            if op_kind == "UNION ALL":
                t_schema = self._schemas_lower[t_name.lower()]
                if not any(c.primary_key for c in t_schema.columns.values()):
                    return False
            parsed_branches.append((f"B{idx}", t_name, w_clause, b_items))

        self._setop_kind = op_kind
        self._setop_branches = parsed_branches
        return True

    def _can_use_join_algebraic(self) -> bool:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            len(v.joins) < 2
            or len(v.joins) != len(v.referenced_tables)
            or v.has_cte
            or v.has_compound_set_op
            or v.has_window_func
            or v.has_limit_or_order
            or v.has_top_level_distinct
            or v.has_subquery_in_from_or_where
            or not has_any_agg
            or not self._decomposed_ok
        ):
            return False
        for j in v.joins:
            if j.join_type not in ("FROM", "INNER JOIN", "CROSS JOIN"):
                return False
            if j.table_name.lower() not in self._schemas_lower:
                return False
        allowed_aggs = {
            AggFunc.COUNT_STAR,
            AggFunc.COUNT_COL,
            AggFunc.SUM,
            AggFunc.TOTAL,
            AggFunc.AVG,
            AggFunc.VAR_POP,
            AggFunc.VAR_SAMP,
            AggFunc.STDDEV_POP,
            AggFunc.STDDEV_SAMP,
            AggFunc.BIT_XOR,
            AggFunc.BOOL_AND,
            AggFunc.BOOL_OR,
        }
        if self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
            allowed_aggs |= {AggFunc.MIN, AggFunc.MAX}
        for it in self._working_select_items:
            if re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE):
                return False
            if it.is_aggregate:
                if it.is_distinct or it.agg_func not in allowed_aggs:
                    return False
        return True

    def _can_use_projection_algebraic(self) -> bool:
        v = self.view
        has_any_agg = any(it.is_aggregate for it in v.select_items)
        if (
            has_any_agg
            or v.group_by_exprs
            or v.has_top_level_distinct
            or v.has_compound_set_op
            or v.has_cte
            or v.has_window_func
            or v.has_limit_or_order
            or v.has_subquery_in_from_or_where
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
            or self.primary_schema is None
        ):
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
            # Check if FROM clause is a parenthesized (SELECT ... UNION ALL SELECT ...) [AS alias]
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
            m_sub = re.match(r"^\((.*)\)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?$", from_clause, flags=re.IGNORECASE | re.DOTALL)
            if not m_sub:
                return False
            inner_union_sql = m_sub.group(1).strip()
            sub_alias = m_sub.group(2) or "__mv_u"

        if not inner_union_sql:
            return False
        matches = _find_all_top_level_keywords(inner_union_sql, r"\b(?:UNION\s+ALL|UNION|INTERSECT|EXCEPT)\b")
        if not matches:
            return False
        ops = [" ".join(m.group(0).upper().split()) for m in matches]
        if any(op != "UNION ALL" for op in ops):
            return False

        branch_strs: List[str] = []
        prev_e = 0
        for m in matches:
            branch_strs.append(inner_union_sql[prev_e : m.start()].strip())
            prev_e = m.end()
        branch_strs.append(inner_union_sql[prev_e:].strip())

        parsed_b: List[Tuple[str, Optional[str], List[SelectItem]]] = []
        expected_c = 0
        for b_s in branch_strs:
            res = self._parse_simple_branch_sql(b_s, expected_c)
            if res is None:
                return False
            expected_c = len(res[2])
            parsed_b.append(res)

        col_names = [it.alias for it in parsed_b[0][2]]
        branch_specs: List[Tuple[str, Optional[str], Dict[str, str]]] = []
        for t_name, w_c, b_items in parsed_b:
            cmap = {c_name.lower(): it.raw_expr.strip() for c_name, it in zip(col_names, b_items)}
            branch_specs.append((t_name, w_c, cmap))

        self._union_all_agg_branches = branch_specs
        self._union_all_inner_sql = inner_union_sql
        self._union_all_sub_alias = sub_alias
        return True

    def _select_strategy(self) -> MaintenanceStrategy:
        v = self.view
        has_any_agg = any(item.is_aggregate for item in v.select_items)

        if v.has_window_func or v.has_limit_or_order:
            return MaintenanceStrategy.ATOMIC_QUERY_SYNC

        # 1. Compound set operations (UNION, UNION ALL, INTERSECT, EXCEPT)
        if v.has_compound_set_op:
            if self._can_use_setop_algebraic():
                self._alg_mode = "setop"
                return MaintenanceStrategy.ALGEBRAIC_DELTA
            return MaintenanceStrategy.ATOMIC_QUERY_SYNC

        # 2. Single-table aggregation (including expression GROUP BY, compound aggs, HAVING) or SELECT DISTINCT
        if (
            len(v.joins) == 1
            and len(v.referenced_tables) == 1
            and self.primary_schema is not None
            and not v.has_cte
            and not v.has_subquery_in_from_or_where
            and self._decomposed_ok
            and (has_any_agg or bool(v.group_by_exprs) or v.has_top_level_distinct)
            and not any(re.search(r"\(\s*SELECT\b", it.raw_expr, flags=re.IGNORECASE) for it in v.select_items)
        ):
            self._alg_mode = "single_table"
            return MaintenanceStrategy.ALGEBRAIC_DELTA

        # 3. Multi-table INNER JOIN / CROSS JOIN aggregation
        if self._can_use_join_algebraic():
            self._alg_mode = "join"
            return MaintenanceStrategy.ALGEBRAIC_DELTA

        # 4. Non-aggregated single-table projection/filter view with PK lineage
        if self._can_use_projection_algebraic():
            self._alg_mode = "projection"
            return MaintenanceStrategy.ALGEBRAIC_DELTA

        # 5. Aggregation over UNION ALL CTE or derived subquery
        if self._can_use_union_all_agg_algebraic():
            self._alg_mode = "union_all_agg"
            return MaintenanceStrategy.ALGEBRAIC_DELTA

        if v.group_by_exprs:
            return MaintenanceStrategy.GROUP_SCOPED_DELTA

        return MaintenanceStrategy.ATOMIC_QUERY_SYNC

    def _validate(self) -> None:
        for t_name in self.view.referenced_tables:
            if t_name.lower() not in self._schemas_lower:
                raise ValueError(
                    f"Referenced table {t_name!r} not found in schema. "
                    f"Available: {list(self.schemas.keys())}"
                )

        if (
            self.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA
            and self._alg_mode == "single_table"
            and not self.view.has_top_level_distinct
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
            for item in self._working_select_items:
                if not item.is_aggregate:
                    if item.source_col is not None and re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", item.raw_expr.strip()):
                        if item.source_col.lower() not in col_map:
                            raise ValueError(
                                f"Selected column {item.source_col!r} does not exist on table {self.primary_schema.name!r}"
                            )
                        if item.source_col.lower() not in grp_cols_lower and item.raw_expr.strip().lower() not in grp_exprs_lower:
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
                    # Expression index: verify every referenced column identifier exists in the view
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
                if re.match(r"^[A-Za-z0-9_]+$", base_c):
                    out.add(base_c.lower())
        return out

    def _build_all_index_ddls(
        self,
        mv_table: str,
        auto_group_cols: Sequence[str],
        has_pk_on_group_cols: bool,
    ) -> Tuple[List[str], List[str]]:
        """
        Builds the complete list of `CREATE [UNIQUE] INDEX` statements and index names
        for `mv_table`, combining the auto-generated group lookup index (unless already
        covered by a PRIMARY KEY or a user-specified index) with all user-specified indexes.
        Every index name is prefixed with `self.object_prefix`.
        """
        index_ddls: List[str] = []
        index_names: List[str] = []
        used_names_lower: Set[str] = set()

        def _norm_cols(cols: Sequence[str]) -> Tuple[str, ...]:
            return tuple(self._extract_index_col_base_ident(c).lower() for c in cols)

        auto_norm = _norm_cols(auto_group_cols)
        user_covers_auto_group = False
        if auto_norm:
            for idx_spec in self.user_indexes:
                if idx_spec.where_clause is None:
                    u_norm = _norm_cols(idx_spec.columns)
                    if len(u_norm) >= len(auto_norm) and u_norm[: len(auto_norm)] == auto_norm:
                        user_covers_auto_group = True
                        break

        if auto_group_cols and not has_pk_on_group_cols and not user_covers_auto_group:
            grp_idx_name = f"{self.object_prefix}idx_{self.logical_view_name}__grp"
            used_names_lower.add(grp_idx_name.lower())
            index_names.append(grp_idx_name)
            index_ddls.append(
                f"CREATE INDEX {grp_idx_name} ON {mv_table} ({', '.join(auto_group_cols)});"
            )

        for idx_pos, idx_spec in enumerate(self.user_indexes, 1):
            if idx_spec.name:
                candidate_name = self._prefixed_identifier(idx_spec.name)
            else:
                col_slugs = [
                    _sanitize_ident(self._extract_index_col_base_ident(c))
                    for c in idx_spec.columns
                ]
                slug = "_".join(s for s in col_slugs if s) or f"idx_{idx_pos}"
                candidate_name = f"{self.object_prefix}idx_{self.logical_view_name}__{slug}"

            final_name = candidate_name
            suffix = 2
            while final_name.lower() in used_names_lower:
                final_name = f"{candidate_name}_{suffix}"
                suffix += 1
            used_names_lower.add(final_name.lower())
            index_names.append(final_name)

            unique_kw = "UNIQUE " if idx_spec.unique else ""
            using_sql = ""
            if idx_spec.using_method and self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                using_sql = f" USING {idx_spec.using_method.upper()}"
            where_sql = ""
            if idx_spec.where_clause and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES):
                where_sql = f" WHERE {idx_spec.where_clause}"

            index_ddls.append(
                f"CREATE {unique_kw}INDEX {final_name} ON {mv_table}{using_sql} "
                f"({', '.join(idx_spec.columns)}){where_sql};"
            )

        return index_ddls, index_names

    def _lookup_col_def(self, expr: str) -> Optional[ColumnDef]:
        s = expr.strip()
        if not re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", s):
            return None
        if "." in s:
            parts = s.split(".")
            qual = _unquote_ident(parts[0]).lower()
            c_name = _unquote_ident(parts[-1]).lower()
            if qual in self.alias_to_schema:
                t_schema = self.alias_to_schema[qual]
                for k, v in t_schema.columns.items():
                    if k.lower() == c_name:
                        return v
        c_name = _unquote_ident(s).lower()
        if self.primary_schema:
            for k, v in self.primary_schema.columns.items():
                if k.lower() == c_name:
                    return v
        for t_schema in self.schemas.values():
            for k, v in t_schema.columns.items():
                if k.lower() == c_name:
                    return v
        return None

    def _col_def(self, col_name: str) -> ColumnDef:
        c = self._lookup_col_def(col_name)
        if c is not None:
            return c
        return ColumnDef(name=_unquote_ident(col_name), data_type="TEXT", not_null=False)

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
        if self.dialect == Dialect.POSTGRES and u == "DOUBLE":
            return "DOUBLE PRECISION"
        if self.dialect == Dialect.MYSQL and u == "DOUBLE PRECISION":
            return "DOUBLE"
        return dt

    def _infer_cte_or_subquery_col_type(self, col_name: str) -> Optional[str]:
        ident = _unquote_ident(col_name.split(".")[-1]).strip()
        if not ident:
            return None
        pat = re.compile(
            rf'((?:COUNT|SUM|AVG|TOTAL|MIN|MAX|VAR_POP|VARIANCE|VAR_SAMP|STDDEV_POP|STDDEV|STD|STDDEV_SAMP|ROW_NUMBER|RANK|DENSE_RANK)\s*\(([^)]*)\))\s+AS\s+["`\[]?{re.escape(ident)}["`\]]?\b',
            re.IGNORECASE,
        )
        m = pat.search(self.view.raw_select_sql)
        if m:
            fn_expr = m.group(1).strip().upper()
            inner_arg = m.group(2).strip()
            if fn_expr.startswith(("COUNT", "ROW_NUMBER", "RANK", "DENSE_RANK")):
                return self._int_type()
            if fn_expr.startswith("SUM"):
                c_arg = self._lookup_col_def(inner_arg)
                if c_arg and "INT" in c_arg.data_type.upper():
                    return self._int_type()
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
            # Unwrap COALESCE(col, ...) / IFNULL(col, ...)
            m_coal = re.match(r'^(?:COALESCE|IFNULL)\s*\(\s*([A-Za-z0-9_."`\[\]]+)\s*,', item.raw_expr.strip(), re.IGNORECASE)
            if m_coal:
                c_inner = self._lookup_col_def(m_coal.group(1))
                if c_inner:
                    return self._normalize_dialect_type(c_inner.data_type)
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
                    # Check if all referenced column identifiers are integer types
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
        return " || '|' || ".join(f"COALESCE(CAST({e} AS TEXT), '__NULL__')" for e in group_exprs)

    def _build_readonly_view_and_triggers(
        self,
        view_name: str,
        backing_table: str,
        user_columns: Sequence[str],
        view_select_exprs: Optional[Sequence[str]] = None,
        view_where_clause: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], List[str]]:
        cols_csv = ", ".join(view_select_exprs if view_select_exprs is not None else user_columns)
        ro_trigs: List[str] = []
        ro_trig_names: List[str] = []
        ro_fn_names: List[str] = []
        db_vname = self.db_view_name
        log_vname = self.logical_view_name
        p = self.object_prefix

        if self.dialect == Dialect.MYSQL:
            where_part = (
                f"WHERE ({view_where_clause}) AND EXISTS (SELECT 1)"
                if view_where_clause
                else "WHERE EXISTS (SELECT 1)"
            )
            view_ddl = (
                f"CREATE VIEW {db_vname} AS\n"
                f"SELECT {cols_csv}\n"
                f"FROM {backing_table}\n"
                f"{where_part};"
            )
            return view_ddl, ro_trigs, ro_trig_names, ro_fn_names

        where_suffix = f"\nWHERE ({view_where_clause})" if view_where_clause else ""
        view_ddl = (
            f"CREATE VIEW {db_vname} AS\n"
            f"SELECT {cols_csv}\n"
            f"FROM {backing_table}{where_suffix};"
        )

        if self.dialect == Dialect.SQLITE:
            for op_short, op_sql in [("ins", "INSERT"), ("upd", "UPDATE"), ("del", "DELETE")]:
                t_name = f"{p}trg_{log_vname}__ro_{op_short}"
                ro_trig_names.append(t_name)
                ro_trigs.append(
                    f"CREATE TRIGGER {t_name}\n"
                    f"INSTEAD OF {op_sql} ON {db_vname}\n"
                    f"BEGIN\n"
                    f"    SELECT RAISE(ABORT, 'Materialized view {db_vname} is read-only and maintained automatically');\n"
                    f"END;"
                )
        elif self.dialect == Dialect.POSTGRES:
            fn_name = f"{p}fn_readonly_{log_vname}"
            t_name = f"{p}trg_{log_vname}__readonly"
            ro_fn_names.append(fn_name)
            ro_trig_names.append(t_name)
            ro_trigs.append(
                f"CREATE OR REPLACE FUNCTION {fn_name}()\n"
                f"RETURNS TRIGGER AS $$\n"
                f"BEGIN\n"
                f"    RAISE EXCEPTION 'Materialized view % is read-only and maintained automatically', TG_TABLE_NAME;\n"
                f"END;\n"
                f"$$ LANGUAGE plpgsql;"
            )
            ro_trigs.append(
                f"CREATE TRIGGER {t_name}\n"
                f"INSTEAD OF INSERT OR UPDATE OR DELETE ON {db_vname}\n"
                f"FOR EACH ROW EXECUTE FUNCTION {fn_name}();"
            )

        return view_ddl, ro_trigs, ro_trig_names, ro_fn_names

    def _referenced_columns_for_table(self, base_t: str) -> List[str]:
        t_schema = self._schemas_lower.get(base_t.lower())
        if t_schema is None:
            return []
        raw_no_strings = re.sub(r"'(?:''|[^'])*'", "''", self.view.raw_select_sql)
        if re.search(r"\bSELECT\s+\*", raw_no_strings, flags=re.IGNORECASE):
            return list(t_schema.columns.keys())
        used: List[str] = []
        for c_name in t_schema.columns:
            if re.search(rf"\b{re.escape(c_name)}\b", raw_no_strings, flags=re.IGNORECASE):
                used.append(c_name)
        return used if used else list(t_schema.columns.keys())

    def _rewrite_expr_for_record(
        self,
        expr: str,
        record_prefix: str,
        table_schema: Optional[TableSchema] = None,
        strip_aliases: Optional[Set[str]] = None,
    ) -> str:
        t_schema = table_schema or self.primary_schema
        if t_schema is None:
            return expr
        col_names = sorted(t_schema.columns.keys(), key=len, reverse=True)
        aliases_to_strip = set(strip_aliases or set())
        aliases_to_strip.add(t_schema.name)
        if self.view.joins:
            for j in self.view.joins:
                if j.table_name.lower() == t_schema.name.lower():
                    aliases_to_strip.add(j.alias)

        tokens = re.split(r"('(?:''|[^'])*')", expr)
        out: List[str] = []
        for idx, tok in enumerate(tokens):
            if idx % 2 == 1:
                out.append(tok)
                continue
            modified = tok
            for al in sorted(aliases_to_strip, key=len, reverse=True):
                modified = re.sub(
                    rf"\b{re.escape(al)}\s*\.\s*",
                    "",
                    modified,
                    flags=re.IGNORECASE,
                )
            for c_name in col_names:
                modified = re.sub(
                    rf"(?<![.\w])\b{re.escape(c_name)}\b(?!\s*\()",
                    f"{record_prefix}.{c_name}",
                    modified,
                    flags=re.IGNORECASE,
                )
            out.append(modified)
        return "".join(out)

    def _effective_agg_value_expr(
        self, item: SelectItem, record_prefix: Optional[str] = None
    ) -> str:
        if item.agg_func == AggFunc.COUNT_STAR:
            base_val = "1"
        else:
            assert item.arg_expr is not None
            base_val = (
                self._rewrite_expr_for_record(item.arg_expr, record_prefix)
                if record_prefix
                else item.arg_expr
            )
        if item.filter_where:
            f_cond = (
                self._rewrite_expr_for_record(item.filter_where, record_prefix)
                if record_prefix
                else item.filter_where
            )
            return f"(CASE WHEN {f_cond} THEN {base_val} ELSE NULL END)"
        return base_val

    def _build_dialect_incremental_populate_sqls(self, raw_plan: MaterializedViewPlan) -> List[str]:
        """
        Generates standalone, dialect-native incremental (chunked) backfill SQL statements
        parameterized by `self.batch_size` for SQLite, PostgreSQL, and MySQL.
        """
        b_size = self.batch_size
        spec = raw_plan.incremental_backfill_spec
        full_sqls = list(raw_plan.initial_populate_sqls)

        # Case 1: PK-range fold on single-table ALGEBRAIC_DELTA views
        if (
            spec is not None
            and spec.mode == "pk_range_fold"
            and spec.pk_column
            and spec.chunk_sql_templates
        ):
            pk_col = spec.pk_column
            base_t = self.view.base_table
            tmpl = spec.chunk_sql_templates[0].strip().rstrip(";")
            # Parse INSERT INTO mv_table (cols) SELECT exprs FROM base_t WHERE ... ON CONFLICT (pk_cols) DO UPDATE SET assigns
            m_ins = re.match(
                r"^INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*SELECT\s+(.*?)\s*FROM\s+"
                + re.escape(base_t)
                + r"\s+WHERE\s+(.*?)\s*ON\s+(?:CONFLICT\s*\(([^)]+)\)\s*DO\s+UPDATE\s+SET|DUPLICATE\s+KEY\s+UPDATE)\s+(.*)$",
                tmpl,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if m_ins:
                mv_tbl, cols_str, sel_str, where_and_grp, pk_cols_str, merge_set_str = m_ins.groups()
                t_cols = [c.strip() for c in _split_top_level(cols_str, ",")]
                s_exprs = [e.strip() for e in _split_top_level(sel_str, ",")]
                grp_exprs = self._effective_group_by_exprs

                if self.dialect == Dialect.SQLITE:
                    aliased_sel = ", ".join(f"{ex} AS {col}" for col, ex in zip(t_cols, s_exprs))
                    w_extra = f" AND ({self.view.where_clause})" if self.view.where_clause else ""
                    g_extra = f", {', '.join(grp_exprs)}" if grp_exprs else ""
                    chunk_cte_sql = (
                        f"WITH RECURSIVE __mv_bounds AS (\n"
                        f"    SELECT MIN({pk_col}) AS __min_id, MAX({pk_col}) AS __max_id FROM {base_t}\n"
                        f"),\n"
                        f"__mv_batches(__batch_lo, __batch_hi) AS (\n"
                        f"    SELECT __min_id - 1, MIN(__max_id, __min_id - 1 + {b_size})\n"
                        f"    FROM __mv_bounds\n"
                        f"    WHERE __min_id IS NOT NULL\n"
                        f"    UNION ALL\n"
                        f"    SELECT __batch_hi, MIN((SELECT __max_id FROM __mv_bounds), __batch_hi + {b_size})\n"
                        f"    FROM __mv_batches\n"
                        f"    WHERE __batch_hi < (SELECT __max_id FROM __mv_bounds)\n"
                        f"),\n"
                        f"__mv_chunk_deltas AS (\n"
                        f"    SELECT\n"
                        f"        __mv_batches.__batch_lo AS __mv_batch_seq,\n"
                        f"        {aliased_sel}\n"
                        f"    FROM __mv_batches\n"
                        f"    JOIN {base_t} ON ({base_t}.{pk_col} > __mv_batches.__batch_lo AND {base_t}.{pk_col} <= __mv_batches.__batch_hi){w_extra}\n"
                        f"    GROUP BY __mv_batches.__batch_lo{g_extra}\n"
                        f")\n"
                        f"INSERT INTO {mv_tbl} ({', '.join(t_cols)})\n"
                        f"SELECT {', '.join(t_cols)} FROM __mv_chunk_deltas\n"
                        f"WHERE TRUE\n"
                        f"ON CONFLICT ({pk_cols_str.strip()}) DO UPDATE SET\n"
                        f"    {merge_set_str.strip()};"
                    )
                    out_stmts = [chunk_cte_sql]
                    if not grp_exprs and full_sqls:
                        base_full = full_sqls[0].strip().rstrip(";")
                        w_tok = "" if " WHERE " in base_full.upper() else "\nWHERE TRUE"
                        seed_sql = f"{base_full}{w_tok}\nON CONFLICT (__mv_global_id) DO NOTHING;"
                        out_stmts.append(seed_sql)
                    return out_stmts

                elif self.dialect == Dialect.POSTGRES:
                    pg_where_grp = where_and_grp.replace(
                        f"({pk_col} > %s AND {pk_col} <= %s)",
                        f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                    ).replace(
                        f"({pk_col} > ? AND {pk_col} <= ?)",
                        f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                    )
                    do_block = (
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
                        f"            INSERT INTO {mv_tbl} ({', '.join(t_cols)})\n"
                        f"            SELECT {', '.join(s_exprs)}\n"
                        f"            FROM {base_t}\n"
                        f"            WHERE {pg_where_grp}\n"
                        f"            ON CONFLICT ({pk_cols_str.strip()}) DO UPDATE SET\n"
                        f"                {merge_set_str.strip()};\n"
                        f"            __mv_cur_pk := __mv_top_pk;\n"
                        f"        END LOOP;\n"
                        f"    END IF;\n"
                        f"END $$;"
                    )
                    out_stmts = [do_block]
                    if not grp_exprs and full_sqls:
                        base_full = full_sqls[0].strip().rstrip(";")
                        w_tok = "" if " WHERE " in base_full.upper() else "\nWHERE TRUE"
                        seed_sql = f"{base_full}{w_tok}\nON CONFLICT (__mv_global_id) DO NOTHING;"
                        out_stmts.append(seed_sql)
                    return out_stmts

                elif self.dialect == Dialect.MYSQL:
                    my_where_grp = where_and_grp.replace(
                        f"({pk_col} > %s AND {pk_col} <= %s)",
                        f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                    ).replace(
                        f"({pk_col} > ? AND {pk_col} <= ?)",
                        f"({pk_col} > __mv_cur_pk AND {pk_col} <= __mv_top_pk)",
                    )
                    proc_name = f"{self.object_prefix}proc_backfill_{self.logical_view_name}"
                    proc_create = (
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
                        f"            INSERT INTO {mv_tbl} ({', '.join(t_cols)})\n"
                        f"            SELECT {', '.join(s_exprs)}\n"
                        f"            FROM {base_t}\n"
                        f"            WHERE {my_where_grp}\n"
                        f"            ON DUPLICATE KEY UPDATE\n"
                        f"                {merge_set_str.strip()};\n"
                        f"            SET __mv_cur_pk = __mv_top_pk;\n"
                        f"        END WHILE;\n"
                        f"    END IF;\n"
                        f"END;"
                    )
                    out_stmts = [
                        f"DROP PROCEDURE IF EXISTS {proc_name};",
                        proc_create,
                        f"CALL {proc_name}();",
                        f"DROP PROCEDURE IF EXISTS {proc_name};",
                    ]
                    if not grp_exprs and full_sqls:
                        base_full = full_sqls[0].strip().rstrip(";")
                        seed_sql = f"{base_full}\nON DUPLICATE KEY UPDATE __mv_global_id = __mv_global_id;"
                        out_stmts.append(seed_sql)
                    return out_stmts

        # Case 2: Convert each INSERT INTO <tbl> (<cols>) SELECT ... statement into a dialect-native batched statement
        m_pk = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", raw_plan.backing_table_ddl, re.IGNORECASE)
        if m_pk:
            main_pk_cols = [c.strip() for c in m_pk.group(1).split(",")]
        else:
            m_inline_pk = re.search(r"([A-Za-z0-9_]+)\s+[^\n,]+PRIMARY\s+KEY", raw_plan.backing_table_ddl, re.IGNORECASE)
            main_pk_cols = [m_inline_pk.group(1)] if m_inline_pk else []

        inc_stmts: List[str] = []
        for idx_s, raw_s in enumerate(full_sqls):
            s_clean = raw_s.strip().rstrip(";")
            m_head = re.match(
                r"^INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*(SELECT\b.*)$",
                s_clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if not m_head:
                inc_stmts.append(raw_s)
                continue
            target_tbl, cols_str, select_body = m_head.groups()
            cols = [c.strip() for c in _split_top_level(cols_str, ",")]
            aliased_inner = re.sub(
                r"^SELECT\s+(DISTINCT\s+)?(.*?)(\s+FROM\b.*)$",
                lambda m_sel: (
                    "SELECT "
                    + (m_sel.group(1) or "")
                    + ", ".join(
                        f"{_parse_select_item(ex.strip(), 0)[0].raw_expr} AS {col}"
                        for col, ex in zip(cols, _split_top_level(m_sel.group(2), ","))
                    )
                    + m_sel.group(3)
                ),
                select_body,
                count=1,
                flags=re.DOTALL | re.IGNORECASE,
            )
            is_aux_dist = "__val" in cols and "__ref_cnt" in cols
            tbl_pks = ([*main_pk_cols, "__val"] if is_aux_dist else main_pk_cols) if (is_aux_dist or target_tbl == raw_plan.backing_table_name) else []
            tbl_pks = [c for c in tbl_pks if c in cols]

            if self.dialect == Dialect.SQLITE:
                upsert_tail = ""
                if tbl_pks:
                    upd_cols = [f"{c} = excluded.{c}" for c in cols if c not in tbl_pks] or [f"{tbl_pks[0]} = excluded.{tbl_pks[0]}"]
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
                    upd_cols = [f"{c} = excluded.{c}" for c in cols if c not in tbl_pks] or [f"{tbl_pks[0]} = excluded.{tbl_pks[0]}"]
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
                    upd_cols = [f"{c} = VALUES({c})" for c in cols if c not in tbl_pks] or [f"{tbl_pks[0]} = VALUES({tbl_pks[0]})"]
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
        primary_plan = self._compile_single_plan()
        if not self.view.additional_views:
            return primary_plan
        sub_plans: List[MaterializedViewPlan] = [primary_plan]
        for extra_v in self.view.additional_views:
            extra_compiler = IncrementalViewCompiler(
                schema=self.schemas,
                view_sql=extra_v,
                dialect=self.dialect,
                pg_use_transition_tables=self.pg_use_transition_tables,
                ignore_limitations=self.ignore_limitations,
                object_prefix=self.object_prefix,
                incremental_backfill=self.incremental_backfill,
                batch_size=self.batch_size,
            )
            sub_plans.append(extra_compiler._compile_single_plan())
        return MaterializedViewPlan(
            dialect=primary_plan.dialect,
            strategy=primary_plan.strategy,
            view_spec=self.view,
            backing_table_name=primary_plan.backing_table_name,
            backing_table_ddl=primary_plan.backing_table_ddl,
            auxiliary_table_ddls=[d for sp in sub_plans for d in sp.auxiliary_table_ddls],
            index_ddls=[d for sp in sub_plans for d in sp.index_ddls],
            initial_populate_sqls=[d for sp in sub_plans for d in sp.initial_populate_sqls],
            trigger_ddls=[d for sp in sub_plans for d in sp.trigger_ddls],
            readonly_trigger_ddls=[d for sp in sub_plans for d in sp.readonly_trigger_ddls],
            user_columns=primary_plan.user_columns,
            reader_view_ddl=primary_plan.reader_view_ddl,
            incremental_backfill_spec=primary_plan.incremental_backfill_spec,
            auxiliary_table_names=[n for sp in sub_plans for n in sp.auxiliary_table_names],
            trigger_names=[n for sp in sub_plans for n in sp.trigger_names],
            function_names=[n for sp in sub_plans for n in sp.function_names],
            limitations=[lim for sp in sub_plans for lim in sp.limitations],
            object_prefix=primary_plan.object_prefix,
            db_view_name=primary_plan.db_view_name,
            index_names=[n for sp in sub_plans for n in sp.index_names],
            user_indexes=primary_plan.user_indexes,
            sub_plans=sub_plans,
            incremental_backfill=self.incremental_backfill,
            batch_size=self.batch_size,
            full_populate_sqls=[d for sp in sub_plans for d in sp.full_populate_sqls],
            incremental_populate_sqls=[d for sp in sub_plans for d in sp.incremental_populate_sqls],
        )

    # =========================================================================
    # TIER 1: Pure O(1) Algebraic & Reference-Counted Delta Compiler
    # =========================================================================

    def _group_col_alias_for_items(self, g_expr: str, items: Sequence[SelectItem]) -> str:
        g_norm = g_expr.strip().lower()
        g_unq = _unquote_ident(g_expr).lower()
        for item in items:
            if not item.is_aggregate:
                if item.raw_expr.strip().lower() == g_norm:
                    return item.alias
                if (
                    len(self.view.joins) <= 1
                    and item.source_col
                    and item.source_col.lower() == g_unq
                    and re.match(r"^[A-Za-z0-9_.\"`\[\]]+$", g_expr.strip())
                ):
                    return item.alias
        return f"__mv_grp_{_sanitize_ident(g_expr)}"

    def _group_col_alias(self, g_expr: str) -> str:
        b_map = getattr(self, "_branch_grp_alias_map", None)
        if b_map and g_expr in b_map:
            return b_map[g_expr]
        return self._group_col_alias_for_items(g_expr, self._working_select_items)

    def _has_nullable_group_col(self) -> bool:
        if not self._effective_group_by_exprs:
            return False
        for g_expr in self._effective_group_by_exprs:
            if not self._is_simple_col_ref(g_expr):
                return True
            if not self._col_def(g_expr).not_null:
                return True
        return False

    def _tier1_pk_cols(self) -> List[str]:
        if not self._effective_group_by_exprs:
            return ["__mv_global_id"]
        if self._has_nullable_group_col():
            return ["__mv_pk"]
        return [self._group_col_alias(g) for g in self._effective_group_by_exprs]

    def _group_match_predicate_tier1(self, target_table: str, record_prefix: str) -> str:
        if not self._effective_group_by_exprs:
            return f"{target_table}.__mv_global_id = 1"
        if self._has_nullable_group_col():
            rec_exprs = [
                self._rewrite_expr_for_record(g, record_prefix)
                for g in self._effective_group_by_exprs
            ]
            pk_val = self._canonical_group_pk_expr(rec_exprs)
            return f"{target_table}.__mv_pk = ({pk_val})"
        clauses: List[str] = []
        for g_expr in self._effective_group_by_exprs:
            mv_col = self._group_col_alias(g_expr)
            rec_e = self._rewrite_expr_for_record(g_expr, record_prefix)
            clauses.append(f"{target_table}.{mv_col} = {rec_e}")
        return " AND ".join(clauses)

    def _compile_tier1_algebraic(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        base_t = self.view.base_table
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
        has_nullable_grp = self._has_nullable_group_col()

        if not grp_exprs:
            col_defs_sql.append("__mv_global_id INTEGER NOT NULL PRIMARY KEY")
        elif has_nullable_grp:
            pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        non_agg_aliases_lower = {
            item.alias.lower() for item in work_items if not item.is_aggregate
        }
        seen_hidden_grp: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_grp:
                seen_hidden_grp.add(mv_col.lower())
                if self._is_simple_col_ref(g_expr):
                    c_def = self._col_def(g_expr)
                    dt = self._normalize_dialect_type(c_def.data_type)
                    null_str = " NOT NULL" if c_def.not_null else ""
                else:
                    fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                    dt = self._infer_expr_storage_type(fake_it)
                    null_str = ""
                col_defs_sql.append(f"{mv_col} {dt}{null_str}")

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

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                if self._is_simple_col_ref(item.raw_expr):
                    c_def = self._col_def(item.raw_expr)
                    dt = self._normalize_dialect_type(c_def.data_type)
                    null_str = " NOT NULL" if c_def.not_null else ""
                else:
                    dt = self._infer_expr_storage_type(item)
                    null_str = ""
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{item.alias} {dt}{null_str}")
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

            if item.is_distinct and item.agg_func in (
                AggFunc.COUNT_COL,
                AggFunc.SUM,
                AggFunc.AVG,
            ):
                if ekey not in distinct_aux_tables:
                    aux_tbl = f"{self.object_prefix}dist_{self.logical_view_name}__{safe_id}"
                    c_def = self._lookup_col_def(item.arg_expr or "")
                    val_type = self._normalize_dialect_type(c_def.data_type) if c_def else "TEXT"
                    if self.dialect == Dialect.MYSQL and val_type.upper() == "TEXT":
                        val_type = "VARCHAR(255)"
                    distinct_aux_tables[ekey] = (aux_tbl, item, val_type)

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
            ):
                if ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    aux_cnt = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = aux_cnt
                    col_defs_sql.append(f"{aux_cnt} {self._int_type()} NOT NULL DEFAULT 0")

            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    aux_sum = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = aux_sum
                    col_defs_sql.append(f"{aux_sum} {self._float_type()}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    aux_sum2 = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = aux_sum2
                    col_defs_sql.append(f"{aux_sum2} {self._float_type()}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    aux_tcnt = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = aux_tcnt
                    col_defs_sql.append(f"{aux_tcnt} {self._int_type()} NOT NULL DEFAULT 0")

        has_pk_on_group_cols = bool(grp_exprs) and not has_nullable_grp
        if has_pk_on_group_cols:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(group_mv_cols)})")
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols if grp_exprs else (),
            has_pk_on_group_cols=has_pk_on_group_cols,
        )

        pk_cols = self._tier1_pk_cols()
        auxiliary_table_ddls: List[str] = []
        auxiliary_table_names: List[str] = []
        for ekey, (aux_tbl, rep_item, val_type) in distinct_aux_tables.items():
            auxiliary_table_names.append(aux_tbl)
            dist_cols: List[str] = []
            if not grp_exprs:
                dist_cols.append("__mv_global_id INTEGER NOT NULL")
            elif has_nullable_grp:
                pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
                dist_cols.append(f"__mv_pk {pk_type} NOT NULL")
            else:
                for g_expr in grp_exprs:
                    c_def = self._col_def(g_expr)
                    mv_col = self._group_col_alias(g_expr)
                    dist_cols.append(f"{mv_col} {self._normalize_dialect_type(c_def.data_type)} NOT NULL")
            dist_cols.append(f"__val {val_type} NOT NULL")
            dist_cols.append(f"__ref_cnt {self._int_type()} NOT NULL DEFAULT 0")
            dist_cols.append(f"PRIMARY KEY ({', '.join(pk_cols)}, __val)")
            auxiliary_table_ddls.append(
                f"CREATE TABLE {aux_tbl} (\n    " + ",\n    ".join(dist_cols) + "\n);"
            )

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                view_select_exprs=self._view_select_exprs,
                view_where_clause=self._view_where_clause,
            )
        )

        initial_populate_sqls, inc_spec = self._build_tier1_initial_populate(
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
            base_t=base_t,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            distinct_aux_tables=distinct_aux_tables,
            expr_key_fn=_expr_key,
        )

        return MaterializedViewPlan(
            dialect=self.dialect,
            strategy=self.strategy,
            view_spec=self.view,
            backing_table_name=mv_table,
            backing_table_ddl=backing_table_ddl,
            auxiliary_table_ddls=auxiliary_table_ddls,
            index_ddls=index_ddls,
            initial_populate_sqls=initial_populate_sqls,
            trigger_ddls=trigger_ddls,
            readonly_trigger_ddls=ro_trigs,
            user_columns=user_columns,
            reader_view_ddl=reader_view_ddl,
            incremental_backfill_spec=inc_spec,
            auxiliary_table_names=auxiliary_table_names,
            trigger_names=ro_trig_names + trig_names,
            function_names=ro_fn_names + fn_names,
            limitations=self.limitations,
            object_prefix=self.object_prefix,
            db_view_name=self.db_view_name,
            index_names=index_names,
            user_indexes=list(self.user_indexes),
        )

    def _sql_backfill_agg_expr(self, item: SelectItem) -> str:
        eff = self._effective_agg_value_expr(item, record_prefix=None)
        dist = "DISTINCT " if item.is_distinct else ""
        if item.agg_func == AggFunc.COUNT_STAR:
            if item.filter_where:
                return f"COUNT({eff})"
            return "COUNT(*)"
        elif item.agg_func == AggFunc.COUNT_COL:
            return f"COUNT({dist}{eff})"
        elif item.agg_func == AggFunc.SUM:
            return f"SUM({dist}{eff})"
        elif item.agg_func == AggFunc.TOTAL:
            if self.dialect == Dialect.SQLITE:
                return f"TOTAL({dist}{eff})"
            return f"COALESCE(1.0 * SUM({dist}{eff}), 0.0)"
        elif item.agg_func == AggFunc.AVG:
            return f"AVG({dist}{eff})"
        elif item.agg_func == AggFunc.MIN:
            return f"MIN({eff})"
        elif item.agg_func == AggFunc.MAX:
            return f"MAX({eff})"
        elif item.agg_func == AggFunc.VAR_POP:
            return f"VAR_POP({eff})"
        elif item.agg_func == AggFunc.VAR_SAMP:
            return f"VAR_SAMP({eff})"
        elif item.agg_func == AggFunc.STDDEV_POP:
            return f"STDDEV_POP({eff})"
        elif item.agg_func == AggFunc.STDDEV_SAMP:
            return f"STDDEV_SAMP({eff})"
        elif item.agg_func == AggFunc.BOOL_AND:
            return f"BOOL_AND({eff})"
        elif item.agg_func == AggFunc.BOOL_OR:
            return f"BOOL_OR({eff})"
        elif item.agg_func == AggFunc.BIT_AND:
            return f"BIT_AND({eff})"
        elif item.agg_func == AggFunc.BIT_OR:
            return f"BIT_OR({eff})"
        elif item.agg_func == AggFunc.BIT_XOR:
            return f"BIT_XOR({eff})"
        elif item.agg_func == AggFunc.STRING_AGG and self.dialect == Dialect.SQLITE:
            sep = item.extra_args if item.extra_args else "','"
            return f"GROUP_CONCAT({dist}{eff}, {sep})"
        return item.raw_expr

    def _build_tier1_initial_populate(
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
                dist_grp.extend(grp_exprs)
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
            group_sql = f"\nGROUP BY {', '.join(grp_exprs)}"
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
        if not has_holistic_or_distinct and (self.dialect in (Dialect.SQLITE, Dialect.POSTGRES) or has_int_pk):
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
                    dist_grp_list.extend(grp_exprs)
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
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} < {mv_table}.{c} THEN {inc_c} "
                    f"ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                assignments.append(
                    f"{c} = CASE WHEN {inc_c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {mv_table}.{c} IS NULL OR {inc_c} > {mv_table}.{c} THEN {inc_c} "
                    f"ELSE {mv_table}.{c} END"
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

        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        def _plus_inc_expr(item_or_ekey_item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            ek = expr_key_fn(item_or_ekey_item)
            if item_or_ekey_item.is_distinct and ek in distinct_aux_tables:
                aux_tbl = distinct_aux_tables[ek][0]
                d_match = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
                return (
                    f"(CASE WHEN ({val_e}) IS NOT NULL "
                    f"AND (SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {d_match}) = 1 "
                    f"THEN 1 ELSE 0 END)"
                )
            return f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)"

        def _plus_sum_expr(item_or_ekey_item: SelectItem, current_sum_col: str) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            ek = expr_key_fn(item_or_ekey_item)
            if item_or_ekey_item.is_distinct and ek in distinct_aux_tables:
                aux_tbl = distinct_aux_tables[ek][0]
                d_match = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
                is_new_distinct = (
                    f"(({val_e}) IS NOT NULL AND "
                    f"(SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {d_match}) = 1)"
                )
                return (
                    f"(CASE "
                    f"WHEN NOT {is_new_distinct} THEN {mv_table}.{current_sum_col} "
                    f"WHEN {mv_table}.{current_sum_col} IS NULL THEN ({val_e}) "
                    f"ELSE {mv_table}.{current_sum_col} + ({val_e}) END)"
                )
            return (
                f"(CASE "
                f"WHEN ({val_e}) IS NULL THEN {mv_table}.{current_sum_col} "
                f"WHEN {mv_table}.{current_sum_col} IS NULL THEN ({val_e}) "
                f"ELSE {mv_table}.{current_sum_col} + ({val_e}) END)"
            )

        def _plus_sum2_expr(item_or_ekey_item: SelectItem, current_sum2_col: str) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            sq = f"(1.0 * ({val_e}) * ({val_e}))"
            return (
                f"(CASE "
                f"WHEN ({val_e}) IS NULL THEN {mv_table}.{current_sum2_col} "
                f"WHEN {mv_table}.{current_sum2_col} IS NULL THEN {sq} "
                f"ELSE {mv_table}.{current_sum2_col} + {sq} END)"
            )

        def _plus_tcnt_expr(item_or_ekey_item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"

        for item in work_items:
            if not item.is_aggregate:
                continue
            col_alias = item.alias
            updated_cols.add(col_alias)
            ekey = expr_key_fn(item)
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)

            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + 1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                inc = _plus_inc_expr(item)
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} + {inc}")
            elif item.agg_func == AggFunc.SUM:
                new_sum = _plus_sum_expr(item, col_alias)
                update_assignments.append(f"{col_alias} = {new_sum}")
            elif item.agg_func == AggFunc.TOTAL:
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
            update_assignments.append(f"{row_count_col} = {mv_table}.{row_count_col} + 1")
            updated_cols.add(row_count_col)

        ekey_to_item: Dict[str, SelectItem] = {
            expr_key_fn(it): it for it in work_items if it.is_aggregate
        }
        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{cnt_col} = {mv_table}.{cnt_col} + {_plus_inc_expr(it)}"
                )
                updated_cols.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            if sum_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(f"{sum_col} = {_plus_sum_expr(it, sum_col)}")
                updated_cols.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(f"{sum2_col} = {_plus_sum2_expr(it, sum2_col)}")
                updated_cols.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{tcnt_col} = {mv_table}.{tcnt_col} + {_plus_tcnt_expr(it)}"
                )
                updated_cols.add(tcnt_col)

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
        ordered_assignments = [
            a for a in update_assignments if a.split("=")[0].strip() in derived_aliases
        ] + [
            a for a in update_assignments if a.split("=")[0].strip() not in derived_aliases
        ]

        insert_cols: List[str] = []
        insert_vals: List[str] = []

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
        seen_hidden_ins: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden_ins:
                seen_hidden_ins.add(mv_col.lower())
                insert_cols.append(mv_col)
                insert_vals.append(self._rewrite_expr_for_record(g_expr, record_prefix))

        for item in work_items:
            insert_cols.append(item.alias)
            if not item.is_aggregate:
                insert_vals.append(self._rewrite_expr_for_record(item.raw_expr, record_prefix))
                continue
            val_e = self._effective_agg_value_expr(item, record_prefix=record_prefix)
            if item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
                insert_vals.append("1")
            elif item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
            elif item.agg_func in (
                AggFunc.SUM,
                AggFunc.MIN,
                AggFunc.MAX,
                AggFunc.BIT_AND,
                AggFunc.BIT_OR,
                AggFunc.BIT_XOR,
            ):
                insert_vals.append(f"({val_e})")
            elif item.agg_func == AggFunc.TOTAL:
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                )
            elif item.agg_func == AggFunc.AVG:
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 1.0 * ({val_e}) END)"
                )
            elif item.agg_func in (AggFunc.VAR_POP, AggFunc.STDDEV_POP):
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 0.0 END)")
            elif item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP):
                insert_vals.append("NULL")
            elif item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL WHEN ({val_e}) THEN {t_lit} ELSE {f_lit} END)"
                )
            else:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                insert_vals.append(recomp)

        already_inserted = set(insert_cols)
        if row_count_col not in already_inserted:
            insert_cols.append(row_count_col)
            insert_vals.append("1")
            already_inserted.add(row_count_col)

        for ekey, cnt_col in col_nonnull_count_map.items():
            if cnt_col not in already_inserted:
                it = ekey_to_item[ekey]
                val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
                insert_cols.append(cnt_col)
                insert_vals.append(f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)")
                already_inserted.add(cnt_col)

        for ekey, sum_col in col_sum_map.items():
            if sum_col not in already_inserted:
                it = ekey_to_item[ekey]
                val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
                insert_cols.append(sum_col)
                insert_vals.append(f"({val_e})")
                already_inserted.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in already_inserted:
                it = ekey_to_item[ekey]
                val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
                insert_cols.append(sum2_col)
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE 1.0 * ({val_e}) * ({val_e}) END)"
                )
                already_inserted.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in already_inserted:
                it = ekey_to_item[ekey]
                val_e = self._effective_agg_value_expr(it, record_prefix=record_prefix)
                insert_cols.append(tcnt_col)
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"
                )
                already_inserted.add(tcnt_col)

        if self.dialect == Dialect.MYSQL:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE\n"
                f"        {', '.join(ordered_assignments)};"
            )
        else:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET\n"
                f"        {', '.join(ordered_assignments)};"
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
    ) -> Tuple[List[str], str]:
        stmts: List[str] = []
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            d_match = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"UPDATE {aux_tbl}\n"
                f"    SET __ref_cnt = {aux_tbl}.__ref_cnt - 1\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND {d_match};"
            )

        match_pred = self._group_match_predicate_tier1(mv_table, record_prefix)
        update_assignments: List[str] = []
        updated_cols: Set[str] = set()

        def _minus_dec_expr(item_or_ekey_item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            ek = expr_key_fn(item_or_ekey_item)
            if item_or_ekey_item.is_distinct and ek in distinct_aux_tables:
                aux_tbl = distinct_aux_tables[ek][0]
                d_match = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
                return (
                    f"(CASE WHEN ({val_e}) IS NOT NULL "
                    f"AND (SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {d_match}) <= 0 "
                    f"THEN 1 ELSE 0 END)"
                )
            return f"(CASE WHEN ({val_e}) IS NOT NULL THEN 1 ELSE 0 END)"

        def _minus_sum_expr(
            item_or_ekey_item: SelectItem, current_sum_col: str, rem_cnt_expr: str
        ) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            ek = expr_key_fn(item_or_ekey_item)
            if item_or_ekey_item.is_distinct and ek in distinct_aux_tables:
                aux_tbl = distinct_aux_tables[ek][0]
                d_match = self._distinct_match_predicate(aux_tbl, val_e, record_prefix)
                is_last_distinct = (
                    f"(({val_e}) IS NOT NULL AND "
                    f"(SELECT {aux_tbl}.__ref_cnt FROM {aux_tbl} WHERE {d_match}) <= 0)"
                )
                return (
                    f"(CASE "
                    f"WHEN NOT {is_last_distinct} THEN {mv_table}.{current_sum_col} "
                    f"WHEN {rem_cnt_expr} <= 0 THEN NULL "
                    f"ELSE {mv_table}.{current_sum_col} - ({val_e}) END)"
                )
            return (
                f"(CASE "
                f"WHEN ({val_e}) IS NULL THEN {mv_table}.{current_sum_col} "
                f"WHEN {rem_cnt_expr} <= 0 THEN NULL "
                f"ELSE {mv_table}.{current_sum_col} - ({val_e}) END)"
            )

        def _minus_sum2_expr(
            item_or_ekey_item: SelectItem, current_sum2_col: str, rem_cnt_expr: str
        ) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            sq = f"(1.0 * ({val_e}) * ({val_e}))"
            return (
                f"(CASE "
                f"WHEN ({val_e}) IS NULL THEN {mv_table}.{current_sum2_col} "
                f"WHEN {rem_cnt_expr} <= 0 THEN NULL "
                f"ELSE {mv_table}.{current_sum2_col} - {sq} END)"
            )

        def _minus_tcnt_expr(item_or_ekey_item: SelectItem) -> str:
            val_e = self._effective_agg_value_expr(item_or_ekey_item, record_prefix=record_prefix)
            return f"(CASE WHEN ({val_e}) IS NOT NULL AND ({val_e}) THEN 1 ELSE 0 END)"

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
                dec = _minus_dec_expr(item)
                update_assignments.append(f"{col_alias} = {mv_table}.{col_alias} - {dec}")
            elif item.agg_func == AggFunc.SUM:
                cnt_col = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                update_assignments.append(
                    f"{col_alias} = {_minus_sum_expr(item, col_alias, rem_cnt)}"
                )
            elif item.agg_func == AggFunc.TOTAL:
                dec_val = f"(CASE WHEN ({val_e}) IS NULL THEN 0.0 ELSE 1.0 * ({val_e}) END)"
                rem_rows = f"({mv_table}.{row_count_col} - 1)"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {rem_rows} <= 0 THEN 0.0 "
                    f"ELSE {mv_table}.{col_alias} - {dec_val} END"
                )
            elif item.agg_func == AggFunc.AVG:
                cnt_col = col_nonnull_count_map[ekey]
                sum_col = col_sum_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                rem_sum = _minus_sum_expr(item, sum_col, rem_cnt)
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {rem_cnt} <= 0 THEN NULL "
                    f"ELSE (1.0 * {rem_sum}) / {rem_cnt} END"
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
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                rem_sum = _minus_sum_expr(item, sum_col, rem_cnt)
                rem_sum2 = _minus_sum2_expr(item, sum2_col, rem_cnt)
                formula = self._variance_formula_sql(
                    new_cnt=rem_cnt,
                    new_sum=rem_sum,
                    new_sum2=rem_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                update_assignments.append(f"{col_alias} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NOT NULL "
                    f"AND {mv_table}.{col_alias} >= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NOT NULL "
                    f"AND {mv_table}.{col_alias} <= ({val_e}) "
                    f"THEN {recomp} ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR):
                recomp = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NOT NULL THEN {recomp} "
                    f"ELSE {mv_table}.{col_alias} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_col = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                xor_expr = self._bitwise_xor_sql(f"{mv_table}.{col_alias}", f"({val_e})")
                update_assignments.append(
                    f"{col_alias} = CASE "
                    f"WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {rem_cnt} <= 0 THEN NULL "
                    f"ELSE {xor_expr} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                rem_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {rem_cnt} <= 0 THEN NULL "
                    f"WHEN {rem_cnt} = {rem_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_col = col_nonnull_count_map[ekey]
                tcnt_col = col_true_cnt_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                rem_tcnt = f"({mv_table}.{tcnt_col} - {_minus_tcnt_expr(item)})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                update_assignments.append(
                    f"{col_alias} = CASE WHEN {rem_cnt} <= 0 THEN NULL "
                    f"WHEN {rem_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
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
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(it)})"
                update_assignments.append(
                    f"{sum_col} = {_minus_sum_expr(it, sum_col, rem_cnt)}"
                )
                updated_cols.add(sum_col)

        for ekey, sum2_col in col_sum2_map.items():
            if sum2_col not in updated_cols:
                it = ekey_to_item[ekey]
                cnt_col = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(it)})"
                update_assignments.append(
                    f"{sum2_col} = {_minus_sum2_expr(it, sum2_col, rem_cnt)}"
                )
                updated_cols.add(sum2_col)

        for ekey, tcnt_col in col_true_cnt_map.items():
            if tcnt_col not in updated_cols:
                it = ekey_to_item[ekey]
                update_assignments.append(
                    f"{tcnt_col} = {mv_table}.{tcnt_col} - {_minus_tcnt_expr(it)}"
                )
                updated_cols.add(tcnt_col)

        priority_aliases = {
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
                AggFunc.BIT_XOR,
                AggFunc.TOTAL,
            )
        }
        second_aliases = (
            {
                it.alias
                for it in work_items
                if it.is_aggregate and it.agg_func == AggFunc.SUM
            }
            | set(col_sum_map.values())
            | set(col_sum2_map.values())
            | set(col_true_cnt_map.values())
        )

        def _sort_key(assign: str) -> int:
            lhs = assign.split("=")[0].strip()
            if lhs in priority_aliases:
                return 0
            if lhs in second_aliases:
                return 1
            return 2

        ordered_assignments = sorted(update_assignments, key=_sort_key)

        stmts.append(
            f"UPDATE {mv_table}\n"
            f"    SET {', '.join(ordered_assignments)}\n"
            f"    WHERE {match_pred};"
        )

        for ekey, (aux_tbl, rep_item, _) in distinct_aux_tables.items():
            val_expr = self._effective_agg_value_expr(rep_item, record_prefix=record_prefix)
            d_match = self._distinct_match_predicate(aux_tbl, val_expr, record_prefix)
            stmts.append(
                f"DELETE FROM {aux_tbl}\n"
                f"    WHERE ({val_expr}) IS NOT NULL AND ({d_match}) AND {aux_tbl}.__ref_cnt <= 0;"
            )

        delete_zero_mv_stmt = (
            f"DELETE FROM {mv_table}\n"
            f"    WHERE ({match_pred}) AND {mv_table}.{row_count_col} <= 0;"
        )
        if include_delete_zero_mv and bool(grp_exprs):
            stmts.append(delete_zero_mv_stmt)

        return stmts, (delete_zero_mv_stmt if grp_exprs else "")

    def _apply_where_guard_to_stmt(self, stmt: str, where_cond: Optional[str]) -> str:
        if not where_cond:
            return stmt
        if "WHERE TRUE" in stmt:
            return stmt.replace("WHERE TRUE", f"WHERE ({where_cond})", 1)
        m_where = _find_top_level_keyword(stmt, r"\bWHERE\b")
        if m_where:
            prefix = stmt[: m_where.end()] + " "
            rest = stmt[m_where.end() :].strip()
            m_conflict = _find_top_level_keyword(
                rest, r"\bON\s+(?:CONFLICT|DUPLICATE\s+KEY)\b"
            )
            if m_conflict:
                w_part = rest[: m_conflict.start()].strip()
                c_part = rest[m_conflict.start() :]
                return f"{prefix}({where_cond}) AND ({w_part})\n    {c_part}"
            if rest.endswith(";"):
                rest = rest[:-1]
            return f"{prefix}({where_cond}) AND ({rest});"
        if stmt.strip().upper().startswith("DELETE FROM"):
            base = stmt.strip().rstrip(";")
            return f"{base} WHERE ({where_cond});"
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
        p = self.object_prefix
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
                if new_section.strip():
                    au_loops.append(
                        f"    FOR NEW IN SELECT * FROM __mv_new_rows LOOP\n"
                        f"        {new_section}\n"
                        f"    END LOOP;"
                    )
                if clean_section.strip():
                    au_loops.append(
                        f"    FOR OLD IN SELECT * FROM __mv_old_rows LOOP\n"
                        f"        {clean_section.strip()}\n"
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
            trig_ddl = (
                f"CREATE TRIGGER {trig_prefix}_sync\n"
                f"AFTER INSERT OR UPDATE{upd_of_clause} OR DELETE ON {base_t}\n"
                f"FOR EACH ROW EXECUTE FUNCTION {fn_name}();"
            )
            return (
                [fn_ddl, trig_ddl, fn_trunc_ddl, t_trunc],
                [f"{trig_prefix}_sync", f"{trig_prefix}_trunc"],
                [fn_name, fn_trunc],
            )

        elif self.dialect == Dialect.MYSQL:
            plus_block = "\n        ".join(ins_stmts)
            minus_block = "\n        ".join(del_stmts)
            upd_minus_block = "\n        ".join(upd_old_stmts)
            upd_plus_block = "\n        ".join(upd_new_stmts)
            upd_clean_block = "\n        ".join(cleanup_stmts)

            ins_inner = f"IF ({where_new}) THEN\n        {plus_block}\n    END IF;" if where_new else plus_block
            del_inner = f"IF ({where_old}) THEN\n        {minus_block}\n    END IF;" if where_old else minus_block
            u_old_inner = (
                f"IF ({where_old}) THEN\n        {upd_minus_block}\n    END IF;"
                if where_old and upd_minus_block
                else upd_minus_block
            )
            u_new_inner = (
                f"IF ({where_new}) THEN\n        {upd_plus_block}\n    END IF;"
                if where_new and upd_plus_block
                else upd_plus_block
            )
            u_clean_inner = (
                f"\n    IF ({where_old}) THEN\n        {upd_clean_block}\n    END IF;"
                if where_old and upd_clean_block
                else (f"\n    {upd_clean_block}" if upd_clean_block else "")
            )
            combined_upd = f"{u_old_inner}\n    {u_new_inner}{u_clean_inner}".strip()
            if changed_guard:
                combined_upd = f"IF ({changed_guard}) THEN\n    {combined_upd}\n    END IF;"

            trig_ins = (
                f"CREATE TRIGGER {trig_prefix}_ai\n"
                f"AFTER INSERT ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {ins_inner}\n"
                f"END;"
            )
            trig_del = (
                f"CREATE TRIGGER {trig_prefix}_ad\n"
                f"AFTER DELETE ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {del_inner}\n"
                f"END;"
            )
            trig_upd = (
                f"CREATE TRIGGER {trig_prefix}_au\n"
                f"AFTER UPDATE ON {base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    {combined_upd}\n"
                f"END;"
            )
            t_names = [f"{trig_prefix}_ai", f"{trig_prefix}_ad", f"{trig_prefix}_au"]
            return [trig_ins, trig_del, trig_upd], t_names, []

        raise ValueError(f"Unsupported dialect: {self.dialect}")

    def _qualify_join_expr(self, expr: str) -> str:
        """Normalizes table-name qualifiers to join-alias qualifiers and qualifies unambiguous bare columns."""
        col_to_aliases: Dict[str, List[str]] = {}
        table_to_alias: Dict[str, str] = {}
        for j in self.view.joins:
            table_to_alias[j.table_name.lower()] = j.alias
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts:
                for c_name in ts.columns:
                    col_to_aliases.setdefault(c_name.lower(), []).append(j.alias)

        tokens = re.split(r"('(?:''|[^'])*')", expr)
        out: List[str] = []
        for idx, tok in enumerate(tokens):
            if idx % 2 == 1:
                out.append(tok)
                continue
            mod = tok
            for t_low, al in sorted(table_to_alias.items(), key=lambda x: len(x[0]), reverse=True):
                if t_low != al.lower():
                    mod = re.sub(rf"\b{re.escape(t_low)}\s*\.\s*", f"{al}.", mod, flags=re.IGNORECASE)
            for c_low, als in sorted(col_to_aliases.items(), key=lambda x: len(x[0]), reverse=True):
                if len(als) == 1:
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
            ):
                if ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    aux_cnt = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = aux_cnt
                    col_defs_sql.append(f"{aux_cnt} {self._int_type()} NOT NULL DEFAULT 0")
            if item.agg_func in (
                AggFunc.AVG,
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    aux_sum = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = aux_sum
                    col_defs_sql.append(f"{aux_sum} {self._float_type()}")
            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    aux_sum2 = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = aux_sum2
                    col_defs_sql.append(f"{aux_sum2} {self._float_type()}")
            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    aux_tcnt = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = aux_tcnt
                    col_defs_sql.append(f"{aux_tcnt} {self._int_type()} NOT NULL DEFAULT 0")

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols if grp_exprs else (),
            has_pk_on_group_cols=False,
        )

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                view_select_exprs=self._view_select_exprs,
                view_where_clause=self._view_where_clause,
            )
        )

        qual_grp_exprs = [self._qualify_join_expr(g) for g in grp_exprs]
        target_cols: List[str] = []
        select_exprs: List[str] = []

        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        else:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(qual_grp_exprs))

        seen_h: Set[str] = set()
        for g_raw, g_qual in zip(grp_exprs, qual_grp_exprs):
            mv_col = self._group_col_alias(g_raw)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_h:
                seen_h.add(mv_col.lower())
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
        group_sql = f"\nGROUP BY {', '.join(qual_grp_exprs)}" if qual_grp_exprs else ""

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

        # Helper to build correlated recompute subquery for MIN/MAX on joins
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
                remapped_g = _remap_to_j(g_raw)
                conds.append(self._null_safe_eq(f"({remapped_g})", f"{mv_table}.{mv_col}", is_nullable=True))
            w_part = f" WHERE {' AND '.join(conds)}" if conds else ""
            agg_sql = _remap_to_j(self._sql_backfill_agg_expr(item))
            return f"(SELECT {agg_sql} FROM {' '.join(j_parts)}{w_part})"

        plus_assigns = self._build_tier1_chunk_merge_assignments(
            mv_table=mv_table,
            row_count_col=row_count_col,
            col_nonnull_count_map=col_nonnull_count_map,
            col_sum_map=col_sum_map,
            col_sum2_map=col_sum2_map,
            col_true_cnt_map=col_true_cnt_map,
            expr_key_fn=_expr_key,
            use_mysql_values=(self.dialect == Dialect.MYSQL),
        )

        # Build minus merge assignments referencing __mv_delta.<col>
        minus_raw_assigns: List[str] = []
        minus_updated: Set[str] = set()

        def _delta_minus_sum(col: str, rem_cnt_e: str) -> str:
            return (
                f"(CASE WHEN __mv_delta.{col} IS NULL THEN {mv_table}.{col} "
                f"WHEN {rem_cnt_e} <= 0 THEN NULL "
                f"ELSE {mv_table}.{col} - __mv_delta.{col} END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = item.alias
            minus_updated.add(c)
            ekey = _expr_key(item)
            if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                minus_raw_assigns.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")
            elif item.agg_func == AggFunc.TOTAL:
                rem_r = f"({mv_table}.{row_count_col} - __mv_delta.{row_count_col})"
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN {rem_r} <= 0 THEN 0.0 ELSE {mv_table}.{c} - __mv_delta.{c} END"
                )
            elif item.agg_func == AggFunc.SUM:
                cnt_c = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                minus_raw_assigns.append(f"{c} = {_delta_minus_sum(c, rem_cnt)}")
            elif item.agg_func == AggFunc.AVG:
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                rem_sum = _delta_minus_sum(sum_c, rem_cnt)
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN {rem_cnt} <= 0 THEN NULL ELSE (1.0 * {rem_sum}) / {rem_cnt} END"
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
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                rem_sum = _delta_minus_sum(sum_c, rem_cnt)
                rem_sum2 = _delta_minus_sum(sum2_c, rem_cnt)
                formula = self._variance_formula_sql(
                    new_cnt=rem_cnt,
                    new_sum=rem_sum,
                    new_sum2=rem_sum2,
                    is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                    is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
                )
                minus_raw_assigns.append(f"{c} = {formula}")
            elif item.agg_func == AggFunc.MIN:
                recomp = _build_join_recompute_subquery(item)
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} >= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.MAX:
                recomp = _build_join_recompute_subquery(item)
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NOT NULL AND {mv_table}.{c} <= __mv_delta.{c} "
                    f"THEN {recomp} ELSE {mv_table}.{c} END"
                )
            elif item.agg_func == AggFunc.BIT_XOR:
                cnt_c = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                xor_e = self._bitwise_xor_sql(f"{mv_table}.{c}", f"(__mv_delta.{c})")
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN __mv_delta.{c} IS NULL THEN {mv_table}.{c} "
                    f"WHEN {rem_cnt} <= 0 THEN NULL ELSE {xor_e} END"
                )
            elif item.agg_func == AggFunc.BOOL_AND:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                rem_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN {rem_cnt} <= 0 THEN NULL "
                    f"WHEN {rem_cnt} = {rem_tcnt} THEN {t_lit} ELSE {f_lit} END"
                )
            elif item.agg_func == AggFunc.BOOL_OR:
                cnt_c = col_nonnull_count_map[ekey]
                tcnt_c = col_true_cnt_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                rem_tcnt = f"({mv_table}.{tcnt_c} - __mv_delta.{tcnt_c})"
                t_lit = "TRUE" if self.dialect == Dialect.POSTGRES else "1"
                f_lit = "FALSE" if self.dialect == Dialect.POSTGRES else "0"
                minus_raw_assigns.append(
                    f"{c} = CASE WHEN {rem_cnt} <= 0 THEN NULL "
                    f"WHEN {rem_tcnt} > 0 THEN {t_lit} ELSE {f_lit} END"
                )

        if row_count_col not in minus_updated:
            minus_raw_assigns.append(
                f"{row_count_col} = {mv_table}.{row_count_col} - __mv_delta.{row_count_col}"
            )
            minus_updated.add(row_count_col)
        for ekey, cnt_c in col_nonnull_count_map.items():
            if cnt_c not in minus_updated:
                minus_raw_assigns.append(f"{cnt_c} = {mv_table}.{cnt_c} - __mv_delta.{cnt_c}")
                minus_updated.add(cnt_c)
        for ekey, sum_c in col_sum_map.items():
            if sum_c not in minus_updated:
                cnt_c = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                minus_raw_assigns.append(f"{sum_c} = {_delta_minus_sum(sum_c, rem_cnt)}")
                minus_updated.add(sum_c)
        for ekey, sum2_c in col_sum2_map.items():
            if sum2_c not in minus_updated:
                cnt_c = col_nonnull_count_map[ekey]
                rem_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                minus_raw_assigns.append(f"{sum2_c} = {_delta_minus_sum(sum2_c, rem_cnt)}")
                minus_updated.add(sum2_c)
        for ekey, tcnt_c in col_true_cnt_map.items():
            if tcnt_c not in minus_updated:
                minus_raw_assigns.append(f"{tcnt_c} = {mv_table}.{tcnt_c} - __mv_delta.{tcnt_c}")
                minus_updated.add(tcnt_c)

        priority_aliases = {
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
                AggFunc.BIT_XOR,
                AggFunc.TOTAL,
            )
        }
        second_aliases = (
            {it.alias for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
            | set(col_sum_map.values())
            | set(col_sum2_map.values())
            | set(col_true_cnt_map.values())
        )
        minus_assigns = sorted(
            minus_raw_assigns,
            key=lambda a: (
                0 if a.split("=")[0].strip() in priority_aliases else (1 if a.split("=")[0].strip() in second_aliases else 2)
            ),
        )

        delta_group_sql = group_sql if grp_exprs else "\nHAVING COUNT(*) > 0"
        aliased_select_csv = ", ".join(f"{ex} AS {col}" for col, ex in zip(target_cols, select_exprs))
        delete_zero_stmt = f"DELETE FROM {mv_table} WHERE {mv_table}.{row_count_col} <= 0;"

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for j_target in self.view.joins:
            base_t = j_target.table_name
            delta_from_new = _build_join_from_clause(replace_alias=j_target.alias, rec_prefix="NEW")
            delta_from_old = _build_join_from_clause(replace_alias=j_target.alias, rec_prefix="OLD")

            if self.dialect == Dialect.MYSQL:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {delta_from_new}{where_sql}{delta_group_sql}\n"
                    f"    ON DUPLICATE KEY UPDATE\n"
                    f"        {', '.join(plus_assigns)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    JOIN (\n"
                    f"        SELECT {aliased_select_csv}\n"
                    f"        FROM {delta_from_old}{where_sql}{delta_group_sql}\n"
                    f"    ) AS __mv_delta ON {mv_table}.{pk_col} = __mv_delta.{pk_col}\n"
                    f"    SET {', '.join(minus_assigns)};"
                )
            else:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {delta_from_new}{where_sql}{delta_group_sql}\n"
                    f"    ON CONFLICT ({pk_col}) DO UPDATE SET\n"
                    f"        {', '.join(plus_assigns)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    SET {', '.join(minus_assigns)}\n"
                    f"    FROM (\n"
                    f"        SELECT {aliased_select_csv}\n"
                    f"        FROM {delta_from_old}{where_sql}{delta_group_sql}\n"
                    f"    ) AS __mv_delta\n"
                    f"    WHERE {mv_table}.{pk_col} = __mv_delta.{pk_col};"
                )

            del_stmts = [minus_stmt, delete_zero_stmt] if grp_exprs else [minus_stmt]
            cleanup_stmts = [delete_zero_stmt] if grp_exprs else []

            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=[plus_stmt],
                del_stmts=del_stmts,
                upd_old_stmts=[minus_stmt],
                upd_new_stmts=[plus_stmt],
                upd_cleanup_old_stmts=cleanup_stmts,
                trunc_stmts=[f"DELETE FROM {mv_table};"],
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

    def _compile_tier1_zset_setop(self) -> MaterializedViewPlan:
        assert self._setop_branches is not None and self._setop_kind is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        user_columns = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        indexed_cols_lower = self._indexed_column_names_lower(user_columns)

        col_defs_sql: List[str] = [f"__mv_pk {pk_type} NOT NULL PRIMARY KEY"]
        for it in self.view.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{it.alias} {dt}")

        op_kind = self._setop_kind
        if op_kind == "UNION":
            col_defs_sql.append(f"__mv_row_count {self._int_type()} NOT NULL DEFAULT 0")
            view_where = None
        elif op_kind == "INTERSECT":
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            view_where = "__mv_cnt_0 > 0 AND __mv_cnt_1 > 0"
        elif op_kind == "EXCEPT":
            col_defs_sql.append(f"__mv_cnt_0 {self._int_type()} NOT NULL DEFAULT 0")
            col_defs_sql.append(f"__mv_cnt_1 {self._int_type()} NOT NULL DEFAULT 0")
            view_where = "__mv_cnt_0 > 0 AND __mv_cnt_1 <= 0"
        else:
            # UNION ALL (PK-lineage per branch)
            view_where = None

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=(),
            has_pk_on_group_cols=False,
        )
        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                view_where_clause=view_where,
            )
        )

        initial_populate_sqls: List[str] = []
        by_table_ops: Dict[str, List[Tuple[List[str], List[str], List[str], List[str], List[str], Optional[str], Optional[str]]]] = {}

        if op_kind == "UNION ALL":
            for b_tag, t_name, w_clause, b_items in self._setop_branches:
                ts = self._schemas_lower[t_name.lower()]
                pk_cols = [c.name for c in ts.columns.values() if c.primary_key]
                b_pk_raw = self._canonical_group_pk_expr([f"'{b_tag}'", *pk_cols])
                b_sel_raw = [it.raw_expr for it in b_items]
                w_sql = f" WHERE {w_clause}" if w_clause else ""
                initial_populate_sqls.append(
                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
                    f"SELECT {b_pk_raw}, {', '.join(b_sel_raw)} FROM {t_name}{w_sql};"
                )
                pk_new = self._canonical_group_pk_expr([f"'{b_tag}'", *[f"NEW.{c}" for c in pk_cols]])
                pk_old = self._canonical_group_pk_expr([f"'{b_tag}'", *[f"OLD.{c}" for c in pk_cols]])
                sel_new = [self._rewrite_expr_for_record(it.raw_expr, "NEW", table_schema=ts) for it in b_items]
                w_new = self._rewrite_expr_for_record(w_clause, "NEW", table_schema=ts) if w_clause else None
                w_old = self._rewrite_expr_for_record(w_clause, "OLD", table_schema=ts) if w_clause else None
                ins_st = (
                    f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
                    f"    SELECT {pk_new}, {', '.join(sel_new)}{from_dual}\n"
                    f"    WHERE TRUE;"
                )
                del_st = f"DELETE FROM {mv_table} WHERE {mv_table}.__mv_pk = ({pk_old});"
                by_table_ops.setdefault(t_name, []).append(
                    ([ins_st], [del_st], [del_st], [ins_st], [], w_new, w_old)
                )
        elif op_kind == "UNION":
            union_branches_sql: List[str] = []
            for b_tag, t_name, w_clause, b_items in self._setop_branches:
                ts = self._schemas_lower[t_name.lower()]
                w_sql = f" WHERE {w_clause}" if w_clause else ""
                proj = ", ".join(f"{it.raw_expr} AS {u_col}" for it, u_col in zip(b_items, user_columns))
                union_branches_sql.append(f"SELECT {proj} FROM {t_name}{w_sql}")

                sel_new = [self._rewrite_expr_for_record(it.raw_expr, "NEW", table_schema=ts) for it in b_items]
                sel_old = [self._rewrite_expr_for_record(it.raw_expr, "OLD", table_schema=ts) for it in b_items]
                pk_new = self._canonical_group_pk_expr(sel_new)
                pk_old = self._canonical_group_pk_expr(sel_old)
                w_new = self._rewrite_expr_for_record(w_clause, "NEW", table_schema=ts) if w_clause else None
                w_old = self._rewrite_expr_for_record(w_clause, "OLD", table_schema=ts) if w_clause else None
                if self.dialect == Dialect.MYSQL:
                    ins_st = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                        f"    SELECT {pk_new}, {', '.join(sel_new)}, 1{from_dual}\n"
                        f"    WHERE TRUE\n"
                        f"    ON DUPLICATE KEY UPDATE __mv_row_count = {mv_table}.__mv_row_count + 1;"
                    )
                else:
                    ins_st = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                        f"    SELECT {pk_new}, {', '.join(sel_new)}, 1\n"
                        f"    WHERE TRUE\n"
                        f"    ON CONFLICT (__mv_pk) DO UPDATE SET __mv_row_count = {mv_table}.__mv_row_count + 1;"
                    )
                dec_st = (
                    f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1 "
                    f"WHERE {mv_table}.__mv_pk = ({pk_old});"
                )
                cln_st = (
                    f"DELETE FROM {mv_table} WHERE ({mv_table}.__mv_pk = ({pk_old})) "
                    f"AND {mv_table}.__mv_row_count <= 0;"
                )
                by_table_ops.setdefault(t_name, []).append(
                    ([ins_st], [dec_st, cln_st], [dec_st], [ins_st], [cln_st], w_new, w_old)
                )

            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                f"SELECT {self._canonical_group_pk_expr(user_columns)}, {', '.join(user_columns)}, COUNT(*)\n"
                f"FROM (\n    " + "\n    UNION ALL\n    ".join(union_branches_sql) + "\n) AS __mv_u\n"
                f"GROUP BY {', '.join(user_columns)};"
            )
        else:
            # INTERSECT or EXCEPT (2 branches)
            union_branches_sql = []
            for b_idx, (b_tag, t_name, w_clause, b_items) in enumerate(self._setop_branches):
                ts = self._schemas_lower[t_name.lower()]
                w_sql = f" WHERE {w_clause}" if w_clause else ""
                proj = ", ".join(f"{it.raw_expr} AS {u_col}" for it, u_col in zip(b_items, user_columns))
                b0_lit = "1" if b_idx == 0 else "0"
                b1_lit = "1" if b_idx == 1 else "0"
                union_branches_sql.append(
                    f"SELECT {proj}, {b0_lit} AS __mv_b0, {b1_lit} AS __mv_b1 FROM {t_name}{w_sql}"
                )

                cnt_col = f"__mv_cnt_{b_idx}"
                sel_new = [self._rewrite_expr_for_record(it.raw_expr, "NEW", table_schema=ts) for it in b_items]
                sel_old = [self._rewrite_expr_for_record(it.raw_expr, "OLD", table_schema=ts) for it in b_items]
                pk_new = self._canonical_group_pk_expr(sel_new)
                pk_old = self._canonical_group_pk_expr(sel_old)
                w_new = self._rewrite_expr_for_record(w_clause, "NEW", table_schema=ts) if w_clause else None
                w_old = self._rewrite_expr_for_record(w_clause, "OLD", table_schema=ts) if w_clause else None
                if self.dialect == Dialect.MYSQL:
                    ins_st = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                        f"    SELECT {pk_new}, {', '.join(sel_new)}, {b0_lit}, {b1_lit}{from_dual}\n"
                        f"    WHERE TRUE\n"
                        f"    ON DUPLICATE KEY UPDATE {cnt_col} = {mv_table}.{cnt_col} + 1;"
                    )
                else:
                    ins_st = (
                        f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                        f"    SELECT {pk_new}, {', '.join(sel_new)}, {b0_lit}, {b1_lit}\n"
                        f"    WHERE TRUE\n"
                        f"    ON CONFLICT (__mv_pk) DO UPDATE SET {cnt_col} = {mv_table}.{cnt_col} + 1;"
                    )
                dec_st = (
                    f"UPDATE {mv_table} SET {cnt_col} = {mv_table}.{cnt_col} - 1 "
                    f"WHERE {mv_table}.__mv_pk = ({pk_old});"
                )
                cln_st = (
                    f"DELETE FROM {mv_table} WHERE ({mv_table}.__mv_pk = ({pk_old})) "
                    f"AND {mv_table}.__mv_cnt_0 <= 0 AND {mv_table}.__mv_cnt_1 <= 0;"
                )
                by_table_ops.setdefault(t_name, []).append(
                    ([ins_st], [dec_st, cln_st], [dec_st], [ins_st], [cln_st], w_new, w_old)
                )

            initial_populate_sqls.append(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_cnt_0, __mv_cnt_1)\n"
                f"SELECT {self._canonical_group_pk_expr(user_columns)}, {', '.join(user_columns)}, SUM(__mv_b0), SUM(__mv_b1)\n"
                f"FROM (\n    " + "\n    UNION ALL\n    ".join(union_branches_sql) + "\n) AS __mv_u\n"
                f"GROUP BY {', '.join(user_columns)};"
            )

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []

        for base_t, op_list in by_table_ops.items():
            if len(op_list) == 1:
                ins_s, del_s, u_old, u_new, u_cln, w_n, w_o = op_list[0]
            else:
                ins_s = [self._apply_where_guard_to_stmt(s, wn) for ins_l, _, _, _, _, wn, _ in op_list for s in ins_l]
                del_s = [self._apply_where_guard_to_stmt(s, wo) for _, del_l, _, _, _, _, wo in op_list for s in del_l]
                u_old = [self._apply_where_guard_to_stmt(s, wo) for _, _, uo_l, _, _, _, wo in op_list for s in uo_l]
                u_new = [self._apply_where_guard_to_stmt(s, wn) for _, _, _, un_l, _, wn, _ in op_list for s in un_l]
                u_cln = [self._apply_where_guard_to_stmt(s, wo) for _, _, _, _, uc_l, _, wo in op_list for s in uc_l]
                w_n, w_o = None, None
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_s,
                del_stmts=del_s,
                upd_old_stmts=u_old,
                upd_new_stmts=u_new,
                upd_cleanup_old_stmts=u_cln,
                trunc_stmts=[f"DELETE FROM {mv_table};"],
                where_new=w_n,
                where_old=w_o,
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

    def _compile_tier1_zset_projection(self) -> MaterializedViewPlan:
        assert self.primary_schema is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        base_t = self.view.base_table
        user_columns = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        indexed_cols_lower = self._indexed_column_names_lower(user_columns)

        col_defs_sql: List[str] = [f"__mv_pk {pk_type} NOT NULL PRIMARY KEY"]
        for it in self.view.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{it.alias} {dt}")

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

        pk_cols = [c.name for c in self.primary_schema.columns.values() if c.primary_key]
        pk_raw = self._canonical_group_pk_expr(pk_cols)
        sel_raw = [it.raw_expr for it in self.view.select_items]
        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        initial_populate_sql = (
            f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
            f"SELECT {pk_raw}, {', '.join(sel_raw)}\n"
            f"FROM {base_t}{w_sql};"
        )

        pk_new = self._canonical_group_pk_expr([f"NEW.{c}" for c in pk_cols])
        pk_old = self._canonical_group_pk_expr([f"OLD.{c}" for c in pk_cols])
        sel_new = [self._rewrite_expr_for_record(it.raw_expr, "NEW") for it in self.view.select_items]
        w_new = self._rewrite_expr_for_record(self.view.where_clause, "NEW") if self.view.where_clause else None
        w_old = self._rewrite_expr_for_record(self.view.where_clause, "OLD") if self.view.where_clause else None

        if self.dialect == Dialect.MYSQL:
            upd_assigns = ", ".join(f"{c} = VALUES({c})" for c in user_columns)
            ins_st = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
                f"    SELECT {pk_new}, {', '.join(sel_new)}{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE {upd_assigns};"
            )
        else:
            upd_assigns = ", ".join(f"{c} = excluded.{c}" for c in user_columns)
            ins_st = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
                f"    SELECT {pk_new}, {', '.join(sel_new)}\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT (__mv_pk) DO UPDATE SET {upd_assigns};"
            )
        del_st = f"DELETE FROM {mv_table} WHERE {mv_table}.__mv_pk = ({pk_old});"

        t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
            view_name=view_name,
            base_t=base_t,
            ins_stmts=[ins_st],
            del_stmts=[del_st],
            upd_old_stmts=[del_st],
            upd_new_stmts=[ins_st],
            trunc_stmts=[f"DELETE FROM {mv_table};"],
            where_new=w_new,
            where_old=w_old,
        )
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)})\n"
                f"SELECT __mv_pk, {', '.join(user_columns)} FROM (\n"
                f"SELECT {pk_raw} AS __mv_pk, {', '.join(f'{it.raw_expr} AS {it.alias}' for it in self.view.select_items)} FROM {base_t}{w_sql}\n"
                f") AS __mv_init LIMIT ? OFFSET ?;"
            ),
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
            trigger_ddls=t_ddls,
            readonly_trigger_ddls=ro_trigs,
            user_columns=user_columns,
            reader_view_ddl=reader_view_ddl,
            incremental_backfill_spec=inc_spec,
            trigger_names=ro_trig_names + t_ns,
            function_names=ro_fn_names + f_ns,
            limitations=self.limitations,
            object_prefix=self.object_prefix,
            db_view_name=self.db_view_name,
            index_names=index_names,
            user_indexes=list(self.user_indexes),
        )

    def _compile_tier1_union_all_agg(self) -> MaterializedViewPlan:
        assert self._union_all_agg_branches is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        work_items = self._working_select_items
        grp_exprs = self._effective_group_by_exprs
        sub_alias = getattr(self, "_union_all_sub_alias", "__mv_u")
        inner_union_sql = getattr(self, "_union_all_inner_sql", "")

        def _subst_branch_expr(frag: str, cmap: Dict[str, str]) -> str:
            tokens = re.split(r"('(?:''|[^'])*')", frag)
            out = []
            for idx_t, tok in enumerate(tokens):
                if idx_t % 2 == 1:
                    out.append(tok)
                    continue
                s = re.sub(rf"\b{re.escape(sub_alias)}\s*\.\s*", "", tok, flags=re.IGNORECASE)
                for col_k, col_ex in sorted(cmap.items(), key=lambda x: len(x[0]), reverse=True):
                    s = re.sub(rf"(?<![.\w])\b{re.escape(col_k)}\b(?!\s*\()", f"({col_ex})", s, flags=re.IGNORECASE)
                out.append(s)
            return "".join(out)

        # Use branch 0 to infer types for group and aggregate columns
        b0_t, _, b0_cmap = self._union_all_agg_branches[0]
        b0_schema = self._schemas_lower[b0_t.lower()]

        count_star_alias: Optional[str] = None
        for item in work_items:
            if item.is_aggregate and item.agg_func == AggFunc.COUNT_STAR and not item.filter_where:
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

        non_agg_aliases_lower = {it.alias.lower() for it in work_items if not it.is_aggregate}
        seen_hidden: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_hidden:
                seen_hidden.add(mv_col.lower())
                b0_g = _subst_branch_expr(g_expr, b0_cmap)
                fake_it = SelectItem(raw_expr=b0_g, alias=mv_col, is_aggregate=False)
                dt = self._infer_expr_storage_type(fake_it)
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

        group_mv_cols = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                b0_e = _subst_branch_expr(item.raw_expr, b0_cmap).strip()
                if b0_e.startswith("(") and b0_e.endswith(")"):
                    b0_e = b0_e[1:-1].strip()
                c_def = b0_schema.columns.get(_unquote_ident(b0_e))
                dt = self._normalize_dialect_type(c_def.data_type) if c_def else self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                col_defs_sql.append(f"{item.alias} {dt}")
            else:
                b0_arg = _subst_branch_expr(item.arg_expr or "", b0_cmap).strip()
                if b0_arg.startswith("(") and b0_arg.endswith(")"):
                    b0_arg = b0_arg[1:-1].strip()
                b0_it = SelectItem(
                    raw_expr=item.raw_expr,
                    alias=item.alias,
                    is_aggregate=True,
                    agg_func=item.agg_func,
                    arg_expr=b0_arg,
                    filter_where=item.filter_where,
                )
                storage_type = self._infer_expr_storage_type(b0_it)
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
                AggFunc.SUM, AggFunc.AVG, AggFunc.VAR_POP, AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP, AggFunc.BIT_XOR, AggFunc.BOOL_AND, AggFunc.BOOL_OR,
            ):
                if ekey not in col_nonnull_count_map:
                    aux_idx += 1
                    aux_cnt = f"__mv_cnt_{safe_id}_{aux_idx}"
                    col_nonnull_count_map[ekey] = aux_cnt
                    col_defs_sql.append(f"{aux_cnt} {self._int_type()} NOT NULL DEFAULT 0")
            if item.agg_func in (AggFunc.AVG, AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP):
                if ekey not in col_sum_map:
                    aux_idx += 1
                    aux_sum = f"__mv_sum_{safe_id}_{aux_idx}"
                    col_sum_map[ekey] = aux_sum
                    col_defs_sql.append(f"{aux_sum} {self._float_type()}")
            if item.agg_func in (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    aux_sum2 = f"__mv_sum2_{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = aux_sum2
                    col_defs_sql.append(f"{aux_sum2} {self._float_type()}")
            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    aux_tcnt = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = aux_tcnt
                    col_defs_sql.append(f"{aux_tcnt} {self._int_type()} NOT NULL DEFAULT 0")

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols if grp_exprs else (),
            has_pk_on_group_cols=False,
        )
        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"
        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(
                view_name,
                mv_table,
                user_columns,
                view_select_exprs=self._view_select_exprs,
                view_where_clause=self._view_where_clause,
            )
        )

        # Build initial backfill from (inner_union_sql) AS sub_alias
        target_cols: List[str] = []
        select_exprs: List[str] = []
        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        else:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(grp_exprs))

        seen_h2: Set[str] = set()
        for g_expr in grp_exprs:
            mv_col = self._group_col_alias(g_expr)
            if mv_col.lower() not in non_agg_aliases_lower and mv_col.lower() not in seen_h2:
                seen_h2.add(mv_col.lower())
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
        ekey_to_item = {_expr_key(it): it for it in work_items if it.is_aggregate}
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

        where_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        group_sql = f"\nGROUP BY {', '.join(grp_exprs)}" if grp_exprs else ""
        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM (\n{inner_union_sql}\n) AS {sub_alias}{where_sql}{group_sql};"
        )

        # Build per-branch Tier 1 plus/minus statements
        saved_primary = self.primary_schema
        saved_work_items = self._working_select_items
        saved_grp_exprs = self._effective_group_by_exprs

        by_table_ops: Dict[str, List[Tuple[List[str], List[str], List[str], List[str], List[str], Optional[str], Optional[str]]]] = {}
        try:
            for b_t, b_where, b_cmap in self._union_all_agg_branches:
                ts = self._schemas_lower[b_t.lower()]
                self.primary_schema = ts
                mapped_grps = [_subst_branch_expr(g, b_cmap) for g in saved_grp_exprs]
                mapped_items: List[SelectItem] = []
                for it in saved_work_items:
                    if not it.is_aggregate:
                        mapped_items.append(
                            SelectItem(
                                raw_expr=_subst_branch_expr(it.raw_expr, b_cmap),
                                alias=it.alias,
                                is_aggregate=False,
                            )
                        )
                    else:
                        mapped_items.append(
                            SelectItem(
                                raw_expr=_subst_branch_expr(it.raw_expr, b_cmap),
                                alias=it.alias,
                                is_aggregate=True,
                                agg_func=it.agg_func,
                                raw_func_name=it.raw_func_name,
                                arg_expr=_subst_branch_expr(it.arg_expr, b_cmap) if it.arg_expr else None,
                                extra_args=it.extra_args,
                                is_distinct=it.is_distinct,
                                filter_where=_subst_branch_expr(it.filter_where, b_cmap) if it.filter_where else None,
                            )
                        )
                # Map ekey from branch item back to canonical ekey from saved_work_items
                branch_ekey_map = {
                    id(m_it): _expr_key(orig_it)
                    for m_it, orig_it in zip(mapped_items, saved_work_items)
                    if orig_it.is_aggregate
                }
                def _branch_ekey_fn(it: SelectItem, _bmap=branch_ekey_map) -> str:
                    return _bmap[id(it)]

                self._branch_grp_alias_map = {
                    mg: self._group_col_alias_for_items(og, saved_work_items)
                    for mg, og in zip(mapped_grps, saved_grp_exprs)
                }
                self._working_select_items = mapped_items
                self._effective_group_by_exprs = mapped_grps

                plus_new = self._build_plus_delta_statements(
                    mv_table,
                    row_count_col,
                    col_nonnull_count_map,
                    col_sum_map,
                    col_sum2_map,
                    col_true_cnt_map,
                    {},
                    _branch_ekey_fn,
                    record_prefix="NEW",
                )
                minus_del, _ = self._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    col_nonnull_count_map,
                    col_sum_map,
                    col_sum2_map,
                    col_true_cnt_map,
                    {},
                    _branch_ekey_fn,
                    record_prefix="OLD",
                    include_delete_zero_mv=True,
                )
                minus_upd, del_zero_st = self._build_minus_delta_statements(
                    mv_table,
                    row_count_col,
                    col_nonnull_count_map,
                    col_sum_map,
                    col_sum2_map,
                    col_true_cnt_map,
                    {},
                    _branch_ekey_fn,
                    record_prefix="OLD",
                    include_delete_zero_mv=False,
                )

                outer_w_mapped = _subst_branch_expr(self.view.where_clause, b_cmap) if self.view.where_clause else None
                if b_where and outer_w_mapped:
                    comb_w = f"({b_where}) AND ({outer_w_mapped})"
                else:
                    comb_w = b_where or outer_w_mapped

                w_new = self._rewrite_expr_for_record(comb_w, "NEW", table_schema=ts) if comb_w else None
                w_old = self._rewrite_expr_for_record(comb_w, "OLD", table_schema=ts) if comb_w else None
                cln_l = [del_zero_st] if del_zero_st else []
                by_table_ops.setdefault(b_t, []).append(
                    (plus_new, minus_del, minus_upd, plus_new, cln_l, w_new, w_old)
                )
        finally:
            self._branch_grp_alias_map = None
            self.primary_schema = saved_primary
            self._working_select_items = saved_work_items
            self._effective_group_by_exprs = saved_grp_exprs

        trigger_ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        for base_t, op_list in by_table_ops.items():
            if len(op_list) == 1:
                ins_s, del_s, u_old, u_new, u_cln, w_n, w_o = op_list[0]
            else:
                ins_s = [self._apply_where_guard_to_stmt(s, wn) for ins_l, _, _, _, _, wn, _ in op_list for s in ins_l]
                del_s = [self._apply_where_guard_to_stmt(s, wo) for _, del_l, _, _, _, _, wo in op_list for s in del_l]
                u_old = [self._apply_where_guard_to_stmt(s, wo) for _, _, uo_l, _, _, _, wo in op_list for s in uo_l]
                u_new = [self._apply_where_guard_to_stmt(s, wn) for _, _, _, un_l, _, wn, _ in op_list for s in un_l]
                u_cln = [self._apply_where_guard_to_stmt(s, wo) for _, _, _, _, uc_l, _, wo in op_list for s in uc_l]
                w_n, w_o = None, None
            t_ddls, t_ns, f_ns = self._emit_dialect_triggers_for_table(
                view_name=view_name,
                base_t=base_t,
                ins_stmts=ins_s,
                del_stmts=del_s,
                upd_old_stmts=u_old,
                upd_new_stmts=u_new,
                upd_cleanup_old_stmts=u_cln,
                trunc_stmts=[f"DELETE FROM {mv_table};"],
                where_new=w_n,
                where_old=w_o,
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
    # TIER 2: Group-Scoped Incremental Maintenance (JOINs, CTEs, Subqueries, HAVING)
    # =========================================================================

    def _compile_tier2_group_scoped(self) -> MaterializedViewPlan:
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        col_defs_sql: List[str] = []
        user_columns: List[str] = []
        target_cols: List[str] = []

        norm_select_map: Dict[str, str] = {}
        for item in self.view.select_items:
            if not item.is_aggregate:
                norm_select_map[item.raw_expr.strip().lower()] = item.alias
                if item.source_col:
                    norm_select_map[item.source_col.lower()] = item.alias
                    norm_select_map[item.alias.lower()] = item.alias

        extra_group_selects: List[Tuple[str, str]] = []
        group_mv_cols: List[Tuple[str, str]] = []

        for idx, g_expr in enumerate(self.view.group_by_exprs):
            g_norm = g_expr.strip().lower()
            g_unquoted = _unquote_ident(g_expr).lower()
            if g_norm in norm_select_map:
                mv_col = norm_select_map[g_norm]
            elif g_unquoted in norm_select_map:
                mv_col = norm_select_map[g_unquoted]
            else:
                mv_col = f"__mv_grp_{idx}"
                extra_group_selects.append((g_expr.strip(), mv_col))
            group_mv_cols.append((g_expr.strip(), mv_col))

        grp_col_names = [mc for _, mc in group_mv_cols]
        indexed_cols_lower = self._indexed_column_names_lower(grp_col_names)

        for item in self.view.select_items:
            user_columns.append(item.alias)
            target_cols.append(item.alias)
            col_type = self._infer_expr_storage_type(item)
            if self.dialect == Dialect.MYSQL and col_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                col_type = "VARCHAR(255)"
            col_defs_sql.append(f"{item.alias} {col_type}")

        for g_expr, hidden_col in extra_group_selects:
            target_cols.append(hidden_col)
            c_def = self._lookup_col_def(g_expr)
            col_type = self._normalize_dialect_type(c_def.data_type) if c_def else ("VARCHAR(255)" if self.dialect == Dialect.MYSQL else "TEXT")
            if self.dialect == Dialect.MYSQL and col_type.upper() == "TEXT":
                col_type = "VARCHAR(255)"
            col_defs_sql.append(f"{hidden_col} {col_type}")

        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=grp_col_names,
            has_pk_on_group_cols=False,
        )

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        reader_view_ddl, ro_trigs, ro_trig_names, ro_fn_names = (
            self._build_readonly_view_and_triggers(view_name, mv_table, user_columns)
        )

        source_query_sql = self._build_tier2_source_query(extra_group_selects)

        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
            f"SELECT {', '.join(target_cols)}\n"
            f"FROM (\n{source_query_sql}\n) AS __mv_init;"
        )

        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                f"SELECT {', '.join(target_cols)}\n"
                f"FROM (\n{source_query_sql}\n) AS __mv_init\n"
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

    def _build_tier2_source_query(self, extra_group_selects: List[Tuple[str, str]]) -> str:
        raw = self.view.raw_select_sql.strip().rstrip(";")
        if not extra_group_selects:
            return raw
        m_from = _find_top_level_keyword(raw, r"\bFROM\b")
        if not m_from:
            return raw
        extra_str = ", " + ", ".join(f"{expr} AS {alias}" for expr, alias in extra_group_selects)
        return raw[: m_from.start()].rstrip() + extra_str + " " + raw[m_from.start() :]

    def _can_push_down_group_filter_into_where(self) -> bool:
        if self.view.has_cte or self.view.has_compound_set_op or not self.view.joins:
            return False
        left_join_aliases = {
            j.alias.lower() for j in self.view.joins if j.join_type == "LEFT JOIN"
        }
        if not left_join_aliases:
            return True
        for g_expr in self.view.group_by_exprs:
            for m_qual in re.finditer(r"([A-Za-z0-9_]+)\s*\.", g_expr):
                if m_qual.group(1).lower() in left_join_aliases:
                    return False
        return True

    def _inject_where_into_source_query(self, source_query_sql: str, extra_where: str) -> str:
        m_group = _find_top_level_keyword(source_query_sql, r"\bGROUP\s+BY\b")
        if not m_group:
            return source_query_sql
        before_group = source_query_sql[: m_group.start()].rstrip()
        from_group_onward = source_query_sql[m_group.start() :]
        m_where = _find_top_level_keyword(before_group, r"\bWHERE\b")
        if m_where:
            w_prefix = before_group[: m_where.end()]
            w_existing = before_group[m_where.end() :].strip()
            return f"{w_prefix} ({w_existing}) AND ({extra_where})\n{from_group_onward}"
        return f"{before_group}\nWHERE ({extra_where})\n{from_group_onward}"

    def _group_scope_predicate_for_table(
        self,
        base_t: str,
        group_mv_cols: List[Tuple[str, str]],
        target_rel: Optional[str],
        record_prefix: str,
    ) -> Optional[str]:
        t_schema = self._schemas_lower.get(base_t.lower())
        if (
            t_schema is None
            or not self.view.joins
            or self.view.has_subquery_in_from_or_where
            or (
                self.view.having_clause
                and re.search(r"\(\s*SELECT\b", self.view.having_clause, flags=re.IGNORECASE)
            )
        ):
            return None

        matching_joins = [
            j for j in self.view.joins if j.table_name.lower() == base_t.lower()
        ]
        if len(matching_joins) != 1:
            return None

        target_join = matching_joins[0]
        target_alias = target_join.alias

        only_uses_target = True
        for g_expr, _ in group_mv_cols:
            for m_qual in re.finditer(r"([A-Za-z0-9_]+)\s*\.", g_expr):
                qual = m_qual.group(1).lower()
                if qual != target_alias.lower() and qual != base_t.lower():
                    only_uses_target = False
                    break
            if not only_uses_target:
                break
            if "." not in g_expr and re.match(r"^[A-Za-z0-9_\"`\[\]]+$", g_expr.strip()):
                c_unq = _unquote_ident(g_expr).lower()
                if c_unq not in {c.lower() for c in t_schema.columns}:
                    only_uses_target = False
                    break

        def _lhs_expr(g_expr: str, mv_col: str) -> str:
            return f"{target_rel}.{mv_col}" if target_rel else f"({g_expr})"

        if only_uses_target:
            preds: List[str] = []
            for g_expr, mv_col in group_mv_cols:
                rewritten = self._rewrite_expr_for_record(
                    g_expr,
                    record_prefix,
                    table_schema=t_schema,
                    strip_aliases={target_alias, base_t},
                )
                preds.append(
                    self._null_safe_eq(_lhs_expr(g_expr, mv_col), f"({rewritten})", is_nullable=True)
                )
            direct_pred = " AND ".join(preds)
            if any(j.join_type == "LEFT JOIN" for j in self.view.joins) and target_join != self.view.joins[0]:
                null_preds = " AND ".join(
                    f"{_lhs_expr(g_expr, mv_col)} IS NULL" for g_expr, mv_col in group_mv_cols
                )
                return f"(({direct_pred}) OR ({null_preds}))"
            return direct_pred

        if self.view.has_cte or any(j.table_name.lower() not in self._schemas_lower for j in self.view.joins):
            return None

        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""
        rec_proj = ", ".join(
            f"{record_prefix}.{c_name} AS {c_name}" for c_name in t_schema.columns
        )
        alias_remap: Dict[str, str] = {
            j.alias: f"__mv_j_{j.alias}" for j in self.view.joins
        }
        for j in self.view.joins:
            alias_remap[j.table_name] = f"__mv_j_{j.alias}"

        def _remap_aliases_in_sql(sql_frag: str) -> str:
            tokens = re.split(r"('(?:''|[^'])*')", sql_frag)
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
    ) -> None:
        if isinstance(dialect, str):
            dialect = Dialect(dialect.lower())
        self.dialect = dialect
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
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
        if self.dialect == Dialect.SQLITE and isinstance(self.conn, sqlite3.Connection):
            schemas = introspect_sqlite_schemas(self.conn, object_prefix=self.object_prefix)
            schemas.pop(self.catalog_table, None)
        else:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'mysql', 'performance_schema', 'sys')
                  AND table_name NOT LIKE '__matview_%'
                  AND table_name NOT LIKE '__mv_%'
                ORDER BY table_name, ordinal_position
                """
            )
            grouped: Dict[str, Dict[str, ColumnDef]] = {}
            for t_name, c_name, d_type, is_null in cur.fetchall():
                if t_name == self.catalog_table or (
                    self.object_prefix and str(t_name).startswith(self.object_prefix)
                ):
                    continue
                grouped.setdefault(t_name, {})[c_name] = ColumnDef(
                    name=c_name,
                    data_type=str(d_type).upper(),
                    not_null=(str(is_null).upper() == "NO"),
                    primary_key=False,
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
        self.conn.commit()
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
            self.conn.commit()
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
            self.conn.commit()
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
            self.conn.commit()
            if on_progress:
                on_progress(1, 1)
            return 1

        batches_run = 0
        if spec.mode == "pk_range_fold" and spec.pk_column:
            base_t = plan.view_spec.base_table
            pk_col = spec.pk_column
            row = cur.execute(
                f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}"
            ).fetchone()
            if not row or row[0] is None or row[1] is None:
                if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                    for s in plan.initial_populate_sqls:
                        cur.execute(s)
                    self.conn.commit()
                return 0
            min_pk, max_pk = int(row[0]), int(row[1])
            cursor_pk = min_pk - 1
            total_span = max(1, max_pk - min_pk + 1)
            while cursor_pk < max_pk:
                upper_pk = min(max_pk, cursor_pk + batch_size)
                for tmpl in spec.chunk_sql_templates:
                    cur.execute(tmpl, (cursor_pk, upper_pk))
                self.conn.commit()
                batches_run += 1
                cursor_pk = upper_pk
                if on_progress:
                    done = min(total_span, cursor_pk - (min_pk - 1))
                    on_progress(batches_run, done)
            return batches_run

        elif spec.mode == "group_key_batch" and spec.distinct_group_query:
            all_keys = [r[0] for r in cur.execute(spec.distinct_group_query).fetchall()]
            if not all_keys:
                return 0
            ph = self._param_placeholder()
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i : i + batch_size]
                placeholders = ", ".join(ph for _ in batch_keys)
                for tmpl in spec.chunk_sql_templates:
                    sql = tmpl.format(placeholders=placeholders)
                    cur.execute(sql, tuple(batch_keys))
                self.conn.commit()
                batches_run += 1
                if on_progress:
                    on_progress(batches_run, min(len(all_keys), i + batch_size))
            return batches_run

        elif spec.mode == "offset_batch" and spec.offset_insert_template:
            offset = 0
            while True:
                cur.execute(spec.offset_insert_template, (batch_size, offset))
                inserted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
                self.conn.commit()
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
        self.conn.commit()
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
        self.conn.commit()

        if populate:
            if incremental_backfill:
                self.backfill_incrementally(
                    plan, batch_size=batch_size, on_progress=on_progress
                )
            else:
                for stmt in plan.get_populate_statements(incremental_backfill=False):
                    cur.execute(stmt)
                self.conn.commit()

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
        self.conn.commit()
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
            self.conn.commit()


# -----------------------------------------------------------------------------
# Public High-Level Convenience Functions & CLI
# -----------------------------------------------------------------------------


def generate_materialized_view(
    schema_sql: Dict[str, TableSchema] | TableSchema | str,
    view_sql: ViewSpec | str,
    dialect: Dialect | str = Dialect.SQLITE,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    incremental_backfill: bool = False,
    batch_size: int = 500,
) -> MaterializedViewPlan:
    """
    Compiles a schema DDL (one or more CREATE TABLE statements) and a CREATE VIEW
    definition (or multi-view SQL script, plus optional user-specified `indexes`,
    `object_prefix`, and `incremental_backfill` / `batch_size`) into a complete
    `MaterializedViewPlan` for the chosen dialect.
    Refuses compilation with `EngineLimitationError` if transparency limitations exist
    unless `ignore_limitations=True`.
    """
    compiler = IncrementalViewCompiler(
        schema=schema_sql,
        view_sql=view_sql,
        dialect=dialect,
        pg_use_transition_tables=pg_use_transition_tables,
        ignore_limitations=ignore_limitations,
        indexes=indexes,
        object_prefix=object_prefix,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
    )
    return compiler.compile()


def generate_materialized_views(
    schema_sql: Dict[str, TableSchema] | TableSchema | str,
    view_sqls: Sequence[ViewSpec | str] | ViewSpec | str,
    dialect: Dialect | str = Dialect.SQLITE,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    incremental_backfill: bool = False,
    batch_size: int = 500,
) -> List[MaterializedViewPlan]:
    """
    Compiles multiple independent materialized views over the same schema (including
    when multiple `CREATE VIEW` statements are inside a single `.sql` string/file),
    generating isolated backing tables, read-only user views, and non-colliding triggers.
    """
    if isinstance(schema_sql, str):
        parsed_schemas = parse_table_schema(schema_sql)
    elif isinstance(schema_sql, TableSchema):
        parsed_schemas = {schema_sql.name: schema_sql}
    else:
        parsed_schemas = dict(schema_sql)

    all_specs = parse_view_definitions(
        view_sqls,
        known_tables={k.lower() for k in parsed_schemas},
        indexes=indexes,
    )
    return [
        generate_materialized_view(
            parsed_schemas,
            v_spec,
            dialect=dialect,
            pg_use_transition_tables=pg_use_transition_tables,
            ignore_limitations=ignore_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
        )
        for v_spec in all_specs
    ]


def install_sqlite_materialized_view(
    conn: sqlite3.Connection,
    schema_or_view_sql: str,
    view_sql: Optional[str] = None,
    populate: bool = True,
    incremental_backfill: bool = False,
    batch_size: int = 500,
    on_progress: Optional[Callable[[int, int], None]] = None,
    ignore_limitations: bool = False,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> MaterializedViewPlan:
    """
    Compiles and installs one or more materialized views (with read-only view protection,
    user-defined indexes, `object_prefix`, and optional incremental initial backfill)
    onto a live `sqlite3.Connection`.
    """
    mgr = MaterializedViewManager(
        conn,
        dialect=Dialect.SQLITE,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
    )
    if view_sql is None:
        return mgr.register_view(
            schema_or_view_sql,
            populate=populate,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
            on_progress=on_progress,
            ignore_limitations=ignore_limitations,
            indexes=indexes,
        )
    return mgr.register_view(
        view_sql,
        schema_sql=schema_or_view_sql,
        populate=populate,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
        on_progress=on_progress,
        ignore_limitations=ignore_limitations,
        indexes=indexes,
    )


def install_sqlite_materialized_views(
    conn: sqlite3.Connection,
    view_sqls: Sequence[ViewSpec | str] | ViewSpec | str,
    schema_sql: Optional[str] = None,
    populate: bool = True,
    incremental_backfill: bool = False,
    batch_size: int = 500,
    ignore_limitations: bool = False,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> List[MaterializedViewPlan]:
    """Installs multiple materialized views onto a live `sqlite3.Connection`."""
    mgr = MaterializedViewManager(
        conn,
        dialect=Dialect.SQLITE,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
    )
    schemas = mgr.fetch_schema(extra_schema_sql=schema_sql)
    all_specs = parse_view_definitions(
        view_sqls,
        known_tables={k.lower() for k in schemas},
        indexes=indexes,
    )
    return [
        mgr.register_view(
            v_spec,
            schema_sql=schema_sql,
            populate=populate,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
            ignore_limitations=ignore_limitations,
        )
        for v_spec in all_specs
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile or install atomic incremental materialized views for SQLite, PostgreSQL, and MySQL."
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="CREATE TABLE schema SQL string or path to a .sql file.",
    )
    parser.add_argument(
        "--view",
        action="append",
        default=[],
        help="CREATE VIEW SQL string or path to a .sql file (may be specified multiple times).",
    )
    parser.add_argument(
        "--index",
        "-I",
        action="append",
        default=[],
        help=(
            "Index to create on the materialized view (repeatable). "
            "Examples: 'total_amt DESC', 'UNIQUE:region,status', 'mv_name(col1, col2 DESC)', "
            "or a full 'CREATE INDEX ... ON <view> (...)' statement."
        ),
    )
    parser.add_argument(
        "--prefix",
        "--object-prefix",
        dest="object_prefix",
        type=str,
        default=DEFAULT_OBJECT_PREFIX,
        help=f"Prefix for all database objects created by the tool (default: {DEFAULT_OBJECT_PREFIX!r}).",
    )
    parser.add_argument(
        "--dialect",
        choices=[d.value for d in Dialect],
        default=Dialect.SQLITE.value,
        help="Target database engine dialect (default: sqlite).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional SQLite database path/URI to connect and install views directly.",
    )
    parser.add_argument(
        "--incremental-backfill",
        action="store_true",
        help="Perform initial backfill incrementally in batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for incremental initial backfill (default: 500).",
    )
    parser.add_argument(
        "--pg-transition-tables",
        action="store_true",
        help="Use statement-level transition table triggers on PostgreSQL.",
    )
    parser.add_argument(
        "--ignore-limitations",
        action="store_true",
        help="Proceed with a warning even when the engine or view has transparency limitations (e.g., MySQL TRUNCATE/FK cascade limitations or non-deterministic expressions).",
    )
    return parser


def _read_sql_arg(val: str) -> str:
    if os.path.isfile(val):
        with open(val, "r", encoding="utf-8") as f:
            return f.read()
    return val


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.view and not args.schema and not args.db:
        sample_schema = """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            amount REAL
        );
        """
        sample_view = """
        CREATE VIEW order_metrics_mv AS
        SELECT
            customer_id,
            COUNT(*) AS total_orders,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
            COUNT(DISTINCT status) AS distinct_statuses,
            SUM(amount) AS sum_amount,
            TOTAL(amount) AS total_amount_nonnull,
            AVG(amount) AS avg_amount,
            MIN(amount) AS min_amount,
            MAX(amount) AS max_amount,
            GROUP_CONCAT(status, ',') AS status_list
        FROM orders
        GROUP BY customer_id;

        CREATE INDEX idx_order_metrics_sum ON order_metrics_mv (sum_amount DESC);
        """
        plan = generate_materialized_view(
            sample_schema,
            sample_view,
            dialect=Dialect(args.dialect),
            pg_use_transition_tables=args.pg_transition_tables,
            ignore_limitations=args.ignore_limitations,
            indexes=args.index,
            object_prefix=args.object_prefix,
            incremental_backfill=args.incremental_backfill,
            batch_size=args.batch_size,
        )
        print(plan.render_sql_script())
        return 0

    schema_sql = _read_sql_arg(args.schema) if args.schema else None
    view_sqls = [_read_sql_arg(v) for v in args.view]
    dialect = Dialect(args.dialect)

    try:
        if args.db:
            mgr = MaterializedViewManager(
                args.db,
                dialect=dialect,
                ignore_limitations=args.ignore_limitations,
                object_prefix=args.object_prefix,
            )
            if schema_sql:
                mgr.apply_schema_ddl(schema_sql, auto_sync_views=False)
            if view_sqls:
                mgr.sync_views(
                    view_sqls,
                    incremental_backfill=args.incremental_backfill,
                    batch_size=args.batch_size,
                    ignore_limitations=args.ignore_limitations,
                    indexes=args.index,
                )
        else:
            if not schema_sql:
                parser.error("--schema is required when --db is not provided.")
            plans = generate_materialized_views(
                schema_sql,
                view_sqls,
                dialect=dialect,
                pg_use_transition_tables=args.pg_transition_tables,
                ignore_limitations=args.ignore_limitations,
                indexes=args.index,
                object_prefix=args.object_prefix,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
            )
            for p in plans:
                print(p.render_sql_script())
        return 0
    except EngineLimitationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
