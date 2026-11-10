#!/usr/bin/env python3
"""
matview_sync.py — Incremental Materialized View Compiler & Lifecycle Manager
============================================================================

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

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot.tokens import TokenType as SQLGlotTokenType

DEFAULT_OBJECT_PREFIX = "__matview_"


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"


class EngineSubdialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MARIADB = "mariadb"


@dataclass(frozen=True, order=True)
class EngineVersion:
    """
    Represents a specific database engine, subdialect (`sqlite`, `postgres`, `mysql`, `mariadb`),
    and semantic version tuple `(major, minor, patch)` for version-optimal SQL generation.
    """
    dialect: Dialect
    subdialect: EngineSubdialect
    version: Tuple[int, int, int]
    raw_version: str = field(default="", compare=False)

    @property
    def major(self) -> int:
        return self.version[0]

    @property
    def minor(self) -> int:
        return self.version[1]

    @property
    def patch(self) -> int:
        return self.version[2]

    @property
    def version_str(self) -> str:
        return f"{self.version[0]}.{self.version[1]}.{self.version[2]}"

    @property
    def pg_version_num(self) -> int:
        return self.version[0] * 10000 + self.version[1] * 100 + self.version[2]

    @property
    def label(self) -> str:
        return f"{self.subdialect.value} {self.version_str}"

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == Dialect.SQLITE

    @property
    def is_postgres(self) -> bool:
        return self.dialect == Dialect.POSTGRES

    @property
    def is_mysql(self) -> bool:
        return self.subdialect == EngineSubdialect.MYSQL

    @property
    def is_mariadb(self) -> bool:
        return self.subdialect == EngineSubdialect.MARIADB

    @property
    def supports_pg_transition_tables(self) -> bool:
        return self.is_postgres and self.version >= (10, 0, 0)

    @property
    def supports_pg_nulls_not_distinct(self) -> bool:
        return self.is_postgres and self.version >= (15, 0, 0)

    @property
    def supports_pg_merge(self) -> bool:
        return self.is_postgres and self.version >= (15, 0, 0)

    @property
    def supports_sqlite_upsert(self) -> bool:
        return self.is_sqlite and self.version >= (3, 24, 0)

    @property
    def supports_sqlite_window_functions(self) -> bool:
        return self.is_sqlite and self.version >= (3, 25, 0)

    @property
    def supports_sqlite_returning(self) -> bool:
        return self.is_sqlite and self.version >= (3, 35, 0)

    @property
    def supports_mysql_window_functions(self) -> bool:
        return (self.is_mysql and self.version >= (8, 0, 0)) or (
            self.is_mariadb and self.version >= (10, 2, 0)
        )

    @property
    def supports_mysql_row_alias_upsert(self) -> bool:
        return self.is_mysql and self.version >= (8, 0, 19)


def _extract_version_tuple(
    text_s: str, default_ver: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    # Handle 5- or 6-digit PostgreSQL server_version_num like 170002 or 90624
    m_pgnum = re.search(r"\b(\d{5,6})\b", text_s)
    if m_pgnum and "." not in text_s:
        num = int(m_pgnum.group(1))
        if num >= 100000:
            return (num // 10000, (num % 10000) // 100, num % 100)
        return (num // 10000, (num % 10000) // 100, num % 100)
    # If MariaDB banner like '5.5.5-10.11.8-MariaDB' or '10.11.8-MariaDB-1:10.11.8'
    m_maria = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?(?=-MariaDB)", text_s, flags=re.IGNORECASE)
    if m_maria:
        return (
            int(m_maria.group(1)),
            int(m_maria.group(2)),
            int(m_maria.group(3)) if m_maria.group(3) is not None else 0,
        )
    # Strip 'sqlite3' prefix so '3' in 'sqlite3' is not mistaken for version 3.0.0
    cleaned = re.sub(r"\bsqlite3\b", "sqlite", text_s, flags=re.IGNORECASE)
    m_ver = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", cleaned)
    if not m_ver:
        return default_ver
    maj = int(m_ver.group(1))
    min_v = int(m_ver.group(2)) if m_ver.group(2) is not None else 0
    pat = int(m_ver.group(3)) if m_ver.group(3) is not None else 0
    return (maj, min_v, pat)


def parse_engine_version(
    spec: Optional[EngineVersion | str | Dialect] = None,
    default_dialect: Optional[Dialect | str] = None,
) -> EngineVersion:
    """
    Parses an engine/version specification (e.g. `'postgres:17'`, `'postgresql-16.4'`,
    `'mariadb:11.4.2'`, `'mysql:8.4'`, `'sqlite:3.45.1'`, `'10.11.8-MariaDB'`, or a bare
    version `'17.2'` with `default_dialect`) into a normalized `EngineVersion`.
    """
    if isinstance(spec, EngineVersion):
        return spec

    def_d: Optional[Dialect] = None
    def_sub: Optional[EngineSubdialect] = None
    if isinstance(default_dialect, Dialect):
        def_d = default_dialect
        def_sub = EngineSubdialect(default_dialect.value)
    elif isinstance(default_dialect, str) and default_dialect.strip():
        dl_s = default_dialect.strip().lower()
        if "mariadb" in dl_s:
            def_d = Dialect.MYSQL
            def_sub = EngineSubdialect.MARIADB
        elif any(k in dl_s for k in ("postgres", "pg")):
            def_d = Dialect.POSTGRES
            def_sub = EngineSubdialect.POSTGRES
        elif "mysql" in dl_s:
            def_d = Dialect.MYSQL
            def_sub = EngineSubdialect.MYSQL
        elif "sqlite" in dl_s:
            def_d = Dialect.SQLITE
            def_sub = EngineSubdialect.SQLITE

    if isinstance(spec, Dialect):
        if spec == Dialect.POSTGRES:
            return EngineVersion(Dialect.POSTGRES, EngineSubdialect.POSTGRES, (14, 0, 0), "14.0.0")
        if spec == Dialect.MYSQL:
            sub = def_sub if def_sub in (EngineSubdialect.MYSQL, EngineSubdialect.MARIADB) else EngineSubdialect.MYSQL
            ver = (11, 4, 0) if sub == EngineSubdialect.MARIADB else (8, 0, 0)
            return EngineVersion(Dialect.MYSQL, sub, ver, f"{ver[0]}.{ver[1]}.{ver[2]}")
        sq_v = sqlite3.sqlite_version_info[:3]
        return EngineVersion(Dialect.SQLITE, EngineSubdialect.SQLITE, (sq_v[0], sq_v[1], sq_v[2]), sqlite3.sqlite_version)

    s = (spec or "").strip()
    if not s:
        target_d = def_d or Dialect.SQLITE
        return parse_engine_version(target_d, default_dialect=default_dialect)

    s_low = s.lower()
    if "mariadb" in s_low:
        ver = _extract_version_tuple(s, (11, 4, 0))
        return EngineVersion(Dialect.MYSQL, EngineSubdialect.MARIADB, ver, s)
    if any(k in s_low for k in ("postgres", "postgresql", "pg", "cockroach", "yugabyte")):
        ver = _extract_version_tuple(s, (14, 0, 0))
        return EngineVersion(Dialect.POSTGRES, EngineSubdialect.POSTGRES, ver, s)
    if any(k in s_low for k in ("mysql", "percona", "tidb")):
        ver = _extract_version_tuple(s, (8, 0, 0))
        return EngineVersion(Dialect.MYSQL, EngineSubdialect.MYSQL, ver, s)
    if any(k in s_low for k in ("sqlite", "apsw", "sqlcipher", "sqlean", "libsql")):
        sq_def = sqlite3.sqlite_version_info[:3]
        ver = _extract_version_tuple(s, (sq_def[0], sq_def[1], sq_def[2]))
        return EngineVersion(Dialect.SQLITE, EngineSubdialect.SQLITE, ver, s)

    # Bare version string (e.g. '17.2', '8.4.0', '3.45.1')
    if def_d is not None:
        if def_d == Dialect.POSTGRES:
            ver = _extract_version_tuple(s, (14, 0, 0))
            return EngineVersion(Dialect.POSTGRES, EngineSubdialect.POSTGRES, ver, s)
        if def_d == Dialect.MYSQL:
            sub = def_sub if def_sub in (EngineSubdialect.MYSQL, EngineSubdialect.MARIADB) else EngineSubdialect.MYSQL
            def_v = (11, 4, 0) if sub == EngineSubdialect.MARIADB else (8, 0, 0)
            ver = _extract_version_tuple(s, def_v)
            return EngineVersion(Dialect.MYSQL, sub, ver, s)
        sq_def = sqlite3.sqlite_version_info[:3]
        ver = _extract_version_tuple(s, (sq_def[0], sq_def[1], sq_def[2]))
        return EngineVersion(Dialect.SQLITE, EngineSubdialect.SQLITE, ver, s)

    raise ValueError(
        f"Cannot determine database dialect from engine/version specification {spec!r}; "
        f"use '<engine>:<version>' (e.g. 'postgres:17', 'mysql:8.4', 'mariadb:11.4', 'sqlite:3.45')."
    )


def detect_engine_version(
    conn: Any,
    hint_dialect: Optional[Dialect | str | EngineVersion] = None,
) -> EngineVersion:
    """
    Autodetects the database engine (`Dialect`), subdialect (`sqlite`, `postgres`, `mysql`, `mariadb`),
    and exact server version `(major, minor, patch)` from a live database connection.
    """
    if isinstance(hint_dialect, EngineVersion):
        return hint_dialect

    # 1. Explicit engine_version attribute on connection or adapter
    conn_ev = getattr(conn, "engine_version", None)
    if isinstance(conn_ev, EngineVersion):
        return conn_ev
    if isinstance(conn_ev, str) and conn_ev.strip():
        return parse_engine_version(conn_ev, default_dialect=getattr(conn, "dialect", hint_dialect))

    # 2. Determine candidate dialect from connection attributes, class/module name, or hint_dialect
    cand_dialect: Optional[Dialect] = None
    cand_sub: Optional[EngineSubdialect] = None
    if isinstance(hint_dialect, Dialect):
        cand_dialect = hint_dialect
    elif isinstance(hint_dialect, str) and hint_dialect.strip():
        parsed_hint = parse_engine_version(hint_dialect)
        # If the hint string already contains an explicit version number (e.g. 'postgres:17'), honor it if query fails
        cand_dialect = parsed_hint.dialect
        cand_sub = parsed_hint.subdialect

    conn_d = getattr(conn, "dialect", None)
    if cand_dialect is None and conn_d is not None:
        if isinstance(conn_d, Dialect):
            cand_dialect = conn_d
        elif isinstance(conn_d, str):
            cand_dialect = parse_engine_version(conn_d).dialect

    mod_cls = f"{type(conn).__module__}.{type(conn).__qualname__}".lower()
    if cand_dialect is None:
        if isinstance(conn, sqlite3.Connection) or any(
            k in mod_cls for k in ("sqlite", "apsw", "sqlcipher", "sqlean", "libsql")
        ):
            cand_dialect = Dialect.SQLITE
        elif any(k in mod_cls for k in ("psycopg", "pg8000", "asyncpg", "postgres", "pgsql")):
            cand_dialect = Dialect.POSTGRES
        elif any(k in mod_cls for k in ("pymysql", "mysqldb", "mariadb", "mysql")):
            cand_dialect = Dialect.MYSQL

    # 3. Query the live connection for its exact version string
    try:
        cur = conn.cursor()
    except Exception:
        return parse_engine_version(
            hint_dialect if isinstance(hint_dialect, str) else None,
            default_dialect=cand_dialect or Dialect.SQLITE,
        )

    if cand_dialect == Dialect.SQLITE:
        try:
            cur.execute("SELECT sqlite_version();")
            row = cur.fetchone()
            if row and row[0]:
                v_str = str(row[0])
                return parse_engine_version(f"sqlite:{v_str}", default_dialect=Dialect.SQLITE)
        except Exception:
            pass
        return parse_engine_version("sqlite", default_dialect=Dialect.SQLITE)

    if cand_dialect == Dialect.POSTGRES:
        raw_v = ""
        ver_tuple: Optional[Tuple[int, int, int]] = None
        try:
            cur.execute("SHOW server_version_num;")
            r_num = cur.fetchone()
            if r_num and r_num[0]:
                num_s = str(r_num[0]).strip()
                ver_tuple = _extract_version_tuple(num_s, (14, 0, 0))
                raw_v = num_s
        except Exception:
            pass
        try:
            cur.execute("SELECT version();")
            r_v = cur.fetchone()
            if r_v and r_v[0]:
                raw_v = str(r_v[0]).strip()
                if ver_tuple is None:
                    ver_tuple = _extract_version_tuple(raw_v, (14, 0, 0))
        except Exception:
            pass
        if ver_tuple is not None:
            return EngineVersion(Dialect.POSTGRES, EngineSubdialect.POSTGRES, ver_tuple, raw_v)
        return parse_engine_version(
            hint_dialect if isinstance(hint_dialect, str) else None,
            default_dialect=Dialect.POSTGRES,
        )

    if cand_dialect == Dialect.MYSQL:
        try:
            cur.execute("SELECT VERSION();")
            row = cur.fetchone()
            if row and row[0]:
                v_str = str(row[0]).strip()
                if "mariadb" in v_str.lower() or cand_sub == EngineSubdialect.MARIADB:
                    ver = _extract_version_tuple(v_str, (11, 4, 0))
                    return EngineVersion(Dialect.MYSQL, EngineSubdialect.MARIADB, ver, v_str)
                ver = _extract_version_tuple(v_str, (8, 0, 0))
                return EngineVersion(Dialect.MYSQL, EngineSubdialect.MYSQL, ver, v_str)
        except Exception:
            pass
        return parse_engine_version(
            hint_dialect if isinstance(hint_dialect, str) else None,
            default_dialect=cand_sub.value if cand_sub else Dialect.MYSQL,
        )

    # 4. Unknown connection class with no dialect hint: probe SQLite -> Postgres -> MySQL/MariaDB
    try:
        cur.execute("SELECT sqlite_version();")
        row = cur.fetchone()
        if row and row[0]:
            return parse_engine_version(f"sqlite:{row[0]}", default_dialect=Dialect.SQLITE)
    except Exception:
        pass
    try:
        cur.execute("SELECT version();")
        row = cur.fetchone()
        if row and row[0]:
            v_str = str(row[0]).strip()
            if "postgres" in v_str.lower() or "cockroach" in v_str.lower() or "yugabyte" in v_str.lower():
                ver = _extract_version_tuple(v_str, (14, 0, 0))
                return EngineVersion(Dialect.POSTGRES, EngineSubdialect.POSTGRES, ver, v_str)
            if "mariadb" in v_str.lower():
                ver = _extract_version_tuple(v_str, (11, 4, 0))
                return EngineVersion(Dialect.MYSQL, EngineSubdialect.MARIADB, ver, v_str)
            ver = _extract_version_tuple(v_str, (8, 0, 0))
            return EngineVersion(Dialect.MYSQL, EngineSubdialect.MYSQL, ver, v_str)
    except Exception:
        pass
    return parse_engine_version("sqlite", default_dialect=Dialect.SQLITE)


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


class StrictCompilationError(ValueError):
    """Raised when strict=True is enabled and a view cannot be compiled with Tier 1 ALGEBRAIC_DELTA."""


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
    engine_version: Optional[EngineVersion] = None

    @property
    def subdialect(self) -> EngineSubdialect:
        if self.engine_version is not None:
            return self.engine_version.subdialect
        return EngineSubdialect(self.dialect.value)

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
                old_b = re.escape(str(self.batch_size))
                new_b = str(eff_batch)
                out_inc: List[str] = []
                for s in self.incremental_populate_sqls:
                    s_mod = re.sub(
                        rf"(__mv_batch_size\s+BIGINT\s*(?:DEFAULT|:=)\s*){old_b}\b",
                        rf"\g<1>{new_b}",
                        s,
                        flags=re.IGNORECASE,
                    )
                    s_mod = re.sub(
                        rf"(\(\(ROW_NUMBER\(\)\s+OVER\s*\(\)\s*-\s*1\)\s*/\s*){old_b}(\)\s+AS\s+__mv_batch_id)",
                        rf"\g<1>{new_b}\g<2>",
                        s_mod,
                        flags=re.IGNORECASE,
                    )
                    out_inc.append(s_mod)
                return out_inc
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
        eng_label = f" | Engine: {self.engine_version.label}" if self.engine_version else ""
        lines: List[str] = [
            "-- =============================================================================",
            f"-- Incremental Materialized View: {self.view_name} (logical: {self.logical_view_name})",
            f"-- Backing Table: {self.backing_table_name}",
            f"-- Base Tables: {', '.join(self.view_spec.referenced_tables)} | Dialect: {self.dialect.value}{eng_label} | Strategy: {self.strategy.value} | Backfill: {bf_label}",
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
# SQL AST Frontend, Lexer, Identifier Quoter, Formatter & Schema/View Parser
# -----------------------------------------------------------------------------

_DOLLAR_QUOTE_RE = re.compile(r"^\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")

_SQL_RESERVED_KEYWORDS: Set[str] = {
    "ADD", "ALL", "ALTER", "AND", "ANY", "AS", "ASC", "BETWEEN", "BY", "CASE",
    "CAST", "CHECK", "COLLATE", "COLUMN", "CONSTRAINT", "CREATE", "CROSS",
    "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "CURRENT_USER",
    "DATABASE", "DEFAULT", "DELETE", "DESC", "DISTINCT", "DROP", "ELSE", "END",
    "EXCEPT", "EXISTS", "FALSE", "FETCH", "FILTER", "FOR", "FOREIGN", "FROM",
    "FULL", "FUNCTION", "GRANT", "GROUP", "HAVING", "IF", "IN", "INDEX",
    "INNER", "INSERT", "INTERSECT", "INTO", "IS", "JOIN", "KEY", "LEADING",
    "LEFT", "LIKE", "LIMIT", "LOCALTIME", "LOCALTIMESTAMP", "NATURAL", "NOT",
    "NULL", "OFFSET", "ON", "ONLY", "OR", "ORDER", "OUTER", "OVER", "PARTITION",
    "PRIMARY", "PROCEDURE", "RANGE", "RECURSIVE", "REFERENCES", "RETURNING",
    "REVOKE", "RIGHT", "ROWS", "SCHEMA", "SELECT", "SESSION_USER", "SET",
    "SOME", "SYMMETRIC", "TABLE", "THEN", "TO", "TRAILING", "TRIGGER", "TRUE",
    "UNION", "UNIQUE", "UPDATE", "USER", "USING", "VALUES", "VARIADIC", "VIEW",
    "WHEN", "WHERE", "WINDOW", "WITH",
}

_IDENT_TOKEN_PATTERN = r'(?:"[^"]*(?:""[^"]*)*"|`[^`]*(?:``[^`]*)*`|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_]*)'
_SIMPLE_IDENT_RE = re.compile(
    rf"^(?:{_IDENT_TOKEN_PATTERN}\.)?{_IDENT_TOKEN_PATTERN}$"
)


def _is_word_or_ident_token(tk: Any) -> bool:
    if tk.token_type == SQLGlotTokenType.IDENTIFIER:
        return True
    if tk.token_type in (
        SQLGlotTokenType.STRING,
        SQLGlotTokenType.NUMBER,
        SQLGlotTokenType.RAW_STRING,
        SQLGlotTokenType.HEX_STRING,
        SQLGlotTokenType.BYTE_STRING,
        SQLGlotTokenType.BIT_STRING,
        SQLGlotTokenType.NATIONAL_STRING,
        SQLGlotTokenType.NATIONAL_RAW_STRING,
        SQLGlotTokenType.UNICODE_STRING,
        SQLGlotTokenType.HEREDOC_STRING,
    ):
        return False
    return bool(tk.text and (tk.text[0].isalpha() or tk.text[0] == "_"))


def _sqlglot_dialect(dialect: Dialect = Dialect.SQLITE) -> str:
    return {
        Dialect.SQLITE: "sqlite",
        Dialect.POSTGRES: "postgres",
        Dialect.MYSQL: "mysql",
    }.get(dialect, "sqlite")

def _tokenize_sql(sql: str, read: str = "sqlite") -> List[Any]:
    """Tokenize SQL while preserving dollar-quoted strings and dialect quoting rules.

    SQLGlot's PostgreSQL lexer understands dollar-quoted strings but treats
    backtick-quoted identifiers differently from SQLite/MySQL. Tokenize ordinary
    SQL segments with the requested dialect and represent each dollar-quoted
    literal as one synthetic token, preserving source offsets for the scanner.
    """
    class _TokenSpan:
        __slots__ = ("token_type", "text", "start", "end")

        def __init__(self, token_type: Any, text: str, start: int, end: int) -> None:
            self.token_type = token_type
            self.text = text
            self.start = start
            self.end = end

    def _dollar_spans() -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0
        n = len(sql)
        state = ""
        block_depth = 0
        while i < n:
            ch = sql[i]
            if state == "single":
                if read == "postgres" and ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "double":
                if ch == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "backtick":
                if ch == chr(96):
                    if i + 1 < n and sql[i + 1] == chr(96):
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "bracket":
                if ch == "]":
                    state = ""
                i += 1
                continue
            if state == "line_comment":
                if ch == "\n":
                    state = ""
                i += 1
                continue
            if state == "block_comment":
                if sql.startswith("/*", i):
                    block_depth += 1
                    i += 2
                    continue
                if sql.startswith("*/", i):
                    block_depth -= 1
                    i += 2
                    if block_depth == 0:
                        state = ""
                    continue
                i += 1
                continue

            if sql.startswith("--", i):
                state = "line_comment"
                i += 2
                continue
            if sql.startswith("/*", i):
                state = "block_comment"
                block_depth = 1
                i += 2
                continue
            if ch == "'":
                state = "single"
                i += 1
                continue
            if ch == '"':
                state = "double"
                i += 1
                continue
            if ch == chr(96):
                state = "backtick"
                i += 1
                continue
            if ch == "[":
                state = "bracket"
                i += 1
                continue
            if ch == "$":
                delimiter_match = re.match(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", sql[i:])
                if delimiter_match:
                    delimiter = delimiter_match.group(0)
                    content_start = i + len(delimiter)
                    close_at = sql.find(delimiter, content_start)
                    if close_at >= 0:
                        end_at = close_at + len(delimiter)
                        spans.append((i, end_at))
                        i = end_at
                        continue
            i += 1
        return spans

    spans = _dollar_spans()
    if not spans:
        return sqlglot.tokenize(sql, read=read)

    tokens: List[Any] = []

    def _append_segment(start: int, end: int) -> None:
        if start >= end:
            return
        for token in sqlglot.tokenize(sql[start:end], read=read):
            tokens.append(
                _TokenSpan(
                    token.token_type,
                    token.text,
                    token.start + start,
                    token.end + start,
                )
            )

    cursor = 0
    for start, end in spans:
        _append_segment(cursor, start)
        tokens.append(
            _TokenSpan(
                SQLGlotTokenType.HEREDOC_STRING,
                sql[start:end],
                start,
                end - 1,
            )
        )
        cursor = end
    _append_segment(cursor, len(sql))
    return tokens

class SQLTokenType(str, Enum):
    WHITESPACE = "WHITESPACE"
    STRING = "STRING"
    DOLLAR_STRING = "DOLLAR_STRING"
    QUOTED_IDENT = "QUOTED_IDENT"
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    PUNCT = "PUNCT"


@dataclass(frozen=True)
class SQLToken:
    kind: SQLTokenType
    text: str
    start: int
    end: int

    @property
    def unquoted(self) -> str:
        if self.kind == SQLTokenType.QUOTED_IDENT:
            sg_toks = sqlglot.tokenize(self.text, read="sqlite")
            if len(sg_toks) == 1 and sg_toks[0].token_type == SQLGlotTokenType.IDENTIFIER:
                return sg_toks[0].text
            return _unquote_ident(self.text)
        return self.text


@dataclass(frozen=True)
class SQLSelectAST:
    raw_sql: str
    cte_names: Tuple[str, ...]
    main_select_sql: str
    has_cte: bool
    has_distinct: bool
    select_items: Tuple[SelectItem, ...]
    from_clause: Optional[str]
    joins: Tuple[JoinSpec, ...]
    where_clause: Optional[str]
    group_by_exprs: Tuple[str, ...]
    having_clause: Optional[str]
    has_window_func: bool
    has_compound_set_op: bool
    has_limit_or_order: bool
    has_subquery_in_from: bool
    sqlglot_ast: Optional[sqlglot_exp.Expr] = None


def parse_sql_ast(sql: str, dialect: Dialect = Dialect.SQLITE) -> sqlglot_exp.Expr:
    """Parses a SQL expression or statement into a `sqlglot.exp.Expr` AST."""
    return sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))


def tokenize_sql(sql: str) -> List[SQLToken]:
    """
    Tokenizes a SQL fragment into lossless typed `SQLToken` nodes backed by `sqlglot.tokenize`.
    """
    if not sql:
        return []
    sg_tokens = _tokenize_sql(sql)
    tokens: List[SQLToken] = []
    cursor = 0
    n = len(sql)

    for sg_tok in sg_tokens:
        t_start = sg_tok.start
        t_end = sg_tok.end + 1
        if t_start > cursor:
            tokens.append(SQLToken(SQLTokenType.WHITESPACE, sql[cursor:t_start], cursor, t_start))
        raw_slice = sql[t_start:t_end]
        tt = sg_tok.token_type
        if tt in (
            SQLGlotTokenType.STRING,
            SQLGlotTokenType.RAW_STRING,
            SQLGlotTokenType.NATIONAL_STRING,
            SQLGlotTokenType.NATIONAL_RAW_STRING,
            SQLGlotTokenType.BYTE_STRING,
            SQLGlotTokenType.UNICODE_STRING,
        ):
            tokens.append(SQLToken(SQLTokenType.STRING, raw_slice, t_start, t_end))
        elif tt == SQLGlotTokenType.HEREDOC_STRING or raw_slice.startswith("$"):
            tokens.append(SQLToken(SQLTokenType.DOLLAR_STRING, raw_slice, t_start, t_end))
        elif tt == SQLGlotTokenType.IDENTIFIER or raw_slice[:1] in ('"', "`", "["):
            tokens.append(SQLToken(SQLTokenType.QUOTED_IDENT, raw_slice, t_start, t_end))
        elif tt in (SQLGlotTokenType.NUMBER, SQLGlotTokenType.BIT_STRING, SQLGlotTokenType.HEX_STRING):
            tokens.append(SQLToken(SQLTokenType.NUMBER, raw_slice, t_start, t_end))
        elif raw_slice in ("(", ")", ",", ".", ";"):
            tokens.append(SQLToken(SQLTokenType.PUNCT, raw_slice, t_start, t_end))
        elif raw_slice[:1].isalpha() or raw_slice[:1] == "_":
            # Split multi-word sqlglot keywords (e.g. "GROUP BY", "ORDER BY", "PRIMARY KEY")
            if any(ch.isspace() for ch in raw_slice):
                for m_part in re.finditer(r"\S+|\s+", raw_slice):
                    p_str = m_part.group(0)
                    p_s = t_start + m_part.start()
                    p_e = t_start + m_part.end()
                    if p_str.isspace():
                        tokens.append(SQLToken(SQLTokenType.WHITESPACE, p_str, p_s, p_e))
                    else:
                        tokens.append(SQLToken(SQLTokenType.IDENT, p_str, p_s, p_e))
            else:
                tokens.append(SQLToken(SQLTokenType.IDENT, raw_slice, t_start, t_end))
        else:
            tokens.append(SQLToken(SQLTokenType.OP, raw_slice, t_start, t_end))
        cursor = t_end

    if cursor < n:
        tokens.append(SQLToken(SQLTokenType.WHITESPACE, sql[cursor:n], cursor, n))
    return tokens


def quote_sql_identifier(
    ident: str,
    dialect: Dialect = Dialect.SQLITE,
    force: bool = False,
) -> str:
    """
    Quotes a SQL identifier via `sqlglot.exp.to_identifier` using dialect-appropriate
    delimiters (`"ident"` on SQLite/PostgreSQL, `` `ident` `` on MySQL/MariaDB) when
    `force=True`, when the identifier is a SQL reserved keyword, or when it contains
    spaces or special characters.
    """
    s = ident.strip()
    if not s:
        return s
    raw = _unquote_ident(s)
    needs_q = (
        force
        or raw.upper() in _SQL_RESERVED_KEYWORDS
        or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw)
    )
    if not needs_q:
        return raw
    sg_id = sqlglot_exp.to_identifier(raw, quoted=True)
    return sg_id.sql(dialect=_sqlglot_dialect(dialect))


def format_sql(
    sql: str,
    dialect: Dialect = Dialect.SQLITE,
    pretty: bool = False,
) -> str:
    """
    Formats and normalizes a SQL statement or script using `sqlglot.transpile`.
    """
    sg_dialect = _sqlglot_dialect(dialect)
    try:
        transpiled = sqlglot.transpile(sql, read=sg_dialect, write=sg_dialect, pretty=pretty)
        if transpiled:
            sep = ";\n" if pretty else "; "
            return sep.join(t.rstrip(";") for t in transpiled) + (";" if sql.strip().endswith(";") else "")
    except sqlglot.errors.SqlglotError:
        pass
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in sql.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln or lines)


def _strip_sql_comments(sql: str) -> str:
    """Strips `-- ...` line comments and `/* ... */` block comments via `sqlglot.tokenize`."""
    if not sql:
        return ""
    toks = _tokenize_sql(sql)
    if not toks:
        return ""
    out: List[str] = []
    cursor = 0
    for tok in toks:
        if tok.start > cursor:
            gap = sql[cursor:tok.start]
            if "--" in gap or "/*" in gap:
                out.append(" ")
            else:
                out.append(gap)
        out.append(sql[tok.start : tok.end + 1])
        cursor = tok.end + 1
    return "".join(out).strip()


def _unquote_ident(ident: str) -> str:
    """Unquotes a SQL identifier or extracts the last component of a qualified name via `sqlglot.tokenize`."""
    s = ident.strip()
    if not s:
        return s
    if "." in s and not (
        (s.startswith('"') and s.endswith('"') and '".' not in s)
        or (s.startswith("`") and s.endswith("`") and "`." not in s)
        or (s.startswith("[") and s.endswith("]") and "]." not in s)
    ):
        parts = _split_top_level(s, ".")
        s = parts[-1].strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1].replace('""', '"')
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        return s[1:-1].replace("``", "`")
    if s.startswith("[") and s.endswith("]") and len(s) >= 2:
        return s[1:-1]
    return s


def _sanitize_ident(raw: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw.strip())
    return cleaned.strip("_") or "col"


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Splits a SQL fragment at top-level separators outside parentheses and literals via `sqlglot.tokenize`."""
    if not text:
        return []
    toks = _tokenize_sql(text)
    if not toks:
        return [text.strip()] if text.strip() else []
    parts: List[str] = []
    depth = 0
    start = 0
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif depth == 0:
            is_match = False
            if sep == "," and tk.token_type == SQLGlotTokenType.COMMA:
                is_match = True
            elif sep == ";" and tk.token_type == SQLGlotTokenType.SEMICOLON:
                is_match = True
            elif sep == "." and tk.token_type == SQLGlotTokenType.DOT:
                is_match = True
            elif tk.text == sep:
                is_match = True
            if is_match:
                part = text[start:tk.start].strip()
                if part or sep == ";":
                    parts.append(part)
                start = tk.end + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _find_top_level_token(
    sql: str,
    target_types: Sequence[SQLGlotTokenType],
    dialect: Dialect = Dialect.SQLITE,
) -> Optional[Any]:
    """Finds the first top-level (depth == 0) `sqlglot` token matching `target_types`."""
    toks = sqlglot.tokenize(sql, read=_sqlglot_dialect(dialect))
    depth = 0
    target_set = set(target_types)
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif depth == 0 and tk.token_type in target_set:
            return tk
    return None


def _extract_set_op_branches_from_ast(
    query_ast: sqlglot_exp.Expr,
    raw_sql: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Traverses a `sqlglot.exp.SetOperation` AST (`Union`, `Intersect`, `Except`) in order,
    returning `(operators, branch_sqls)`. When `raw_sql` is provided, slices the verbatim
    branch substrings using `sqlglot.tokenize` so original function names and syntax are preserved.
    """
    if raw_sql:
        s = raw_sql.strip()
        toks = _tokenize_sql(s)
        if toks:
            ops_t: List[str] = []
            branches_t: List[str] = []
            depth = 0
            last_start = 0
            i = 0
            n = len(toks)
            while i < n:
                tk = toks[i]
                if tk.token_type == SQLGlotTokenType.L_PAREN:
                    depth += 1
                elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                    depth -= 1
                elif depth == 0:
                    w = tk.text.upper()
                    if w in ("UNION", "INTERSECT", "EXCEPT"):
                        branches_t.append(s[last_start : tk.start].strip())
                        if w == "UNION" and i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.ALL:
                            ops_t.append("UNION ALL")
                            last_start = toks[i + 1].end + 1
                            i += 2
                            continue
                        ops_t.append(w)
                        last_start = tk.end + 1
                i += 1
            if ops_t:
                branches_t.append(s[last_start:].strip())
                return ops_t, branches_t

    ops: List[str] = []
    branches: List[str] = []

    def _walk(node: sqlglot_exp.Expr) -> None:
        if isinstance(node, sqlglot_exp.SetOperation):
            _walk(node.this)
            if isinstance(node, sqlglot_exp.Union):
                is_dist = node.args.get("distinct", True)
                ops.append("UNION" if is_dist else "UNION ALL")
            elif isinstance(node, sqlglot_exp.Intersect):
                ops.append("INTERSECT")
            elif isinstance(node, sqlglot_exp.Except):
                ops.append("EXCEPT")
            else:
                ops.append(node.key.upper())
            _walk(node.expression)
        else:
            branches.append(node.sql("sqlite"))

    _walk(query_ast)
    return ops, branches


def _has_subquery_ast(expr_or_sql: Optional[str | sqlglot_exp.Expr]) -> bool:
    """Returns True if the SQL expression or AST contains a `SELECT` or `EXISTS` subquery."""
    if not expr_or_sql:
        return False
    node = parse_sql_ast(expr_or_sql) if isinstance(expr_or_sql, str) else expr_or_sql
    return bool(node.find(sqlglot_exp.Subquery, sqlglot_exp.Select, sqlglot_exp.Exists))


def _is_simple_column_expr(expr_sql: str) -> bool:
    """Returns True if `expr_sql` parses as a single `sqlglot.exp.Column` (`col` or `tbl.col`)."""
    s = expr_sql.strip()
    if not s:
        return False
    try:
        node = parse_sql_ast(s)
    except sqlglot.errors.SqlglotError:
        return False
    return isinstance(node, sqlglot_exp.Column) and not isinstance(node.this, sqlglot_exp.Star)


def parse_table_schema(schema_sql: str) -> Dict[str, TableSchema]:
    """Parses one or more CREATE TABLE statements into TableSchema objects via `sqlglot.parse`."""
    clean = _strip_sql_comments(schema_sql)
    schemas: Dict[str, TableSchema] = {}
    for stmt_ast in sqlglot.parse(clean, read="sqlite"):
        if not isinstance(stmt_ast, sqlglot_exp.Create) or stmt_ast.kind.upper() != "TABLE":
            continue
        schema_node = stmt_ast.this
        if not isinstance(schema_node, sqlglot_exp.Schema) or not isinstance(schema_node.this, sqlglot_exp.Table):
            continue
        table_name = schema_node.this.name
        if not table_name or table_name.startswith(("__matview_", "__mv_")):
            continue
        cols: Dict[str, ColumnDef] = {}
        table_pk_cols: Set[str] = set()
        fk_cascades: List[str] = []
        has_on_conflict_replace = False

        st_sql = stmt_ast.sql("sqlite")
        st_toks = sqlglot.tokenize(st_sql, read="sqlite")
        for k_i in range(len(st_toks)):
            if (
                k_i + 2 < len(st_toks)
                and st_toks[k_i].token_type == SQLGlotTokenType.ON
                and st_toks[k_i + 1].text.upper() == "CONFLICT"
                and st_toks[k_i + 2].token_type == SQLGlotTokenType.REPLACE
            ):
                has_on_conflict_replace = True
            if (
                k_i + 2 < len(st_toks)
                and st_toks[k_i].token_type == SQLGlotTokenType.ON
                and st_toks[k_i + 1].token_type in (SQLGlotTokenType.DELETE, SQLGlotTokenType.UPDATE)
            ):
                ev_type = st_toks[k_i + 1].text.upper()
                act_w1 = st_toks[k_i + 2].text.upper()
                act_str = act_w1
                if act_w1 == "SET" and k_i + 3 < len(st_toks):
                    act_str = f"SET {st_toks[k_i + 3].text.upper()}"
                if act_str in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    act_full = f"ON {ev_type} {act_str}"
                    if act_full not in fk_cascades:
                        fk_cascades.append(act_full)

        for ref_node in stmt_ast.find_all(sqlglot_exp.Reference):
            for opt in (ref_node.args.get("options") or []):
                opt_s = str(opt).strip().upper()
                if opt_s in (
                    "ON DELETE CASCADE", "ON DELETE SET NULL", "ON DELETE SET DEFAULT",
                    "ON UPDATE CASCADE", "ON UPDATE SET NULL", "ON UPDATE SET DEFAULT",
                ) and opt_s not in fk_cascades:
                    fk_cascades.append(opt_s)

        for pk_node in schema_node.find_all(sqlglot_exp.PrimaryKey):
            for pk_id in pk_node.expressions:
                pk_name = pk_id.name or (pk_id.this.name if isinstance(pk_id.this, sqlglot_exp.Expr) else "")
                if pk_name:
                    table_pk_cols.add(pk_name.lower())

        for expr_item in schema_node.expressions:
            if isinstance(expr_item, sqlglot_exp.ColumnDef):
                col_name = expr_item.name
                if not col_name:
                    continue
                data_type = expr_item.kind.sql() if expr_item.kind is not None else "TEXT"
                is_pk = any(
                    isinstance(c.kind, sqlglot_exp.PrimaryKeyColumnConstraint)
                    for c in expr_item.constraints
                )
                not_null = is_pk or any(
                    isinstance(c.kind, sqlglot_exp.NotNullColumnConstraint)
                    and not c.kind.args.get("allow_null")
                    for c in expr_item.constraints
                )
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
    conn: Any,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> Dict[str, TableSchema]:
    """Introspects all non-internal base tables from a live SQLite connection."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )
    rows = cur.fetchall() or []
    ddl_parts: List[str] = []
    table_names: List[str] = []
    for name, sql in rows:
        if not name:
            continue
        t_str = str(name)
        if t_str.startswith(("__matview_", "__mv_")) or (
            object_prefix and t_str.startswith(object_prefix)
        ):
            continue
        table_names.append(t_str)
        if sql:
            ddl_parts.append(str(sql).rstrip(";") + ";")
    schemas = parse_table_schema("\n".join(ddl_parts)) if ddl_parts else {}
    low_known = {k.lower(): k for k in schemas}

    # Fallback / enrichment via PRAGMA table_info & foreign_key_list for tables without standard DDL (e.g. CREATE TABLE AS)
    for t_str in table_names:
        existing = schemas.get(low_known.get(t_str.lower(), t_str))
        if existing and existing.columns:
            continue
        try:
            cur.execute(f'PRAGMA table_info("{t_str}")')
            t_info = cur.fetchall() or []
            if not t_info:
                continue
            cols: Dict[str, ColumnDef] = {}
            for r in t_info:
                c_name = str(r[1])
                c_type = str(r[2]).upper() if r[2] else "TEXT"
                c_notnull = bool(r[3]) or bool(r[5])
                c_pk = bool(r[5])
                cols[c_name] = ColumnDef(
                    name=c_name,
                    data_type=c_type,
                    not_null=c_notnull,
                    primary_key=c_pk,
                )
            fk_actions: List[str] = []
            try:
                cur.execute(f'PRAGMA foreign_key_list("{t_str}")')
                for fk_r in (cur.fetchall() or []):
                    on_upd = str(fk_r[5]).upper() if len(fk_r) > 5 and fk_r[5] else ""
                    on_del = str(fk_r[6]).upper() if len(fk_r) > 6 and fk_r[6] else ""
                    if on_del in ("CASCADE", "SET NULL", "SET DEFAULT"):
                        act = f"ON DELETE {on_del}"
                        if act not in fk_actions:
                            fk_actions.append(act)
                    if on_upd in ("CASCADE", "SET NULL", "SET DEFAULT"):
                        act = f"ON UPDATE {on_upd}"
                        if act not in fk_actions:
                            fk_actions.append(act)
            except Exception:
                pass
            schemas[t_str] = TableSchema(
                name=t_str,
                columns=cols,
                fk_cascade_actions=tuple(fk_actions),
            )
        except Exception:
            pass
    return schemas


def introspect_database_schemas(
    conn: Any,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> Dict[str, TableSchema]:
    """
    Introspects all non-internal base tables from a live database connection
    across SQLite, PostgreSQL, and MySQL/MariaDB. Autodetects the engine dialect
    from `conn` when `dialect` is omitted.
    """
    if dialect is None:
        dialect = detect_engine_version(conn).dialect
    elif isinstance(dialect, EngineVersion):
        dialect = dialect.dialect
    elif isinstance(dialect, str):
        dialect = parse_engine_version(dialect).dialect
    catalog_table = f"{object_prefix}catalog" if object_prefix else "matview_catalog"

    if dialect == Dialect.SQLITE:
        schemas = introspect_sqlite_schemas(conn, object_prefix=object_prefix)
        schemas.pop(catalog_table, None)
        return schemas

    schemas = {}
    cur = conn.cursor()
    if dialect == Dialect.POSTGRES:
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
        fk_by_table: Dict[str, List[str]] = {}
        try:
            cur.execute(
                """
                SELECT tc.table_name, rc.delete_rule, rc.update_rule
                FROM information_schema.table_constraints tc
                JOIN information_schema.referential_constraints rc
                  ON tc.constraint_name = rc.constraint_name
                 AND tc.table_schema = rc.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = current_schema()
                """
            )
            for fk_r in (cur.fetchall() or []):
                t_low = str(fk_r[0]).lower()
                del_r = str(fk_r[1]).upper() if fk_r[1] else ""
                upd_r = str(fk_r[2]).upper() if fk_r[2] else ""
                acts = fk_by_table.setdefault(t_low, [])
                if del_r in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    a_str = f"ON DELETE {del_r}"
                    if a_str not in acts:
                        acts.append(a_str)
                if upd_r in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    a_str = f"ON UPDATE {upd_r}"
                    if a_str not in acts:
                        acts.append(a_str)
        except Exception:
            pass

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
            t_name, c_name, d_type, is_null, char_len = (
                str(row[0]),
                str(row[1]),
                row[2],
                row[3],
                row[4],
            )
            if t_name == catalog_table or (
                object_prefix and t_name.startswith(object_prefix)
            ):
                continue
            dt_up = str(d_type).upper()
            if dt_up in ("CHARACTER VARYING", "VARCHAR") and char_len:
                dt_up = f"VARCHAR({int(char_len)})"
            elif dt_up == "CHARACTER VARYING":
                dt_up = "TEXT"
            is_pk = (t_name.lower(), c_name.lower()) in pk_pairs
            grouped.setdefault(t_name, {})[c_name] = ColumnDef(
                name=c_name,
                data_type=dt_up,
                not_null=(str(is_null).upper() == "NO") or is_pk,
                primary_key=is_pk,
            )
        for t_name, cols in grouped.items():
            schemas[t_name] = TableSchema(
                name=t_name,
                columns=cols,
                fk_cascade_actions=tuple(fk_by_table.get(t_name.lower(), ())),
            )
        return schemas

    # MySQL / MariaDB
    def _dec(val: Any) -> str:
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8")
        return "" if val is None else str(val)

    fk_by_table_my: Dict[str, List[str]] = {}
    try:
        cur.execute(
            """
            SELECT rc.table_name, rc.delete_rule, rc.update_rule
            FROM information_schema.referential_constraints rc
            WHERE rc.constraint_schema = DATABASE()
            """
        )
        for fk_r in (cur.fetchall() or []):
            t_low = _dec(fk_r[0]).lower()
            del_r = _dec(fk_r[1]).upper()
            upd_r = _dec(fk_r[2]).upper()
            acts = fk_by_table_my.setdefault(t_low, [])
            if del_r in ("CASCADE", "SET NULL", "SET DEFAULT"):
                a_str = f"ON DELETE {del_r}"
                if a_str not in acts:
                    acts.append(a_str)
            if upd_r in ("CASCADE", "SET NULL", "SET DEFAULT"):
                a_str = f"ON UPDATE {upd_r}"
                if a_str not in acts:
                    acts.append(a_str)
    except Exception:
        pass

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
    grouped_my: Dict[str, Dict[str, ColumnDef]] = {}
    for row in (cur.fetchall() or []):
        t_name, c_name, d_type, is_null, col_key = (
            _dec(row[0]),
            _dec(row[1]),
            _dec(row[2]),
            _dec(row[3]),
            _dec(row[4]),
        )
        col_type = _dec(row[5]) if len(row) > 5 and row[5] is not None else None
        if t_name == catalog_table or (
            object_prefix and t_name.startswith(object_prefix)
        ):
            continue
        dt_up = d_type.upper()
        if dt_up in ("VARCHAR", "CHAR", "TINYINT") and col_type:
            dt_up = col_type.upper()
        is_pk = col_key.upper() == "PRI"
        grouped_my.setdefault(t_name, {})[c_name] = ColumnDef(
            name=c_name,
            data_type=dt_up,
            not_null=(str(is_null).upper() == "NO") or is_pk,
            primary_key=is_pk,
        )
    for t_name, cols in grouped_my.items():
        schemas[t_name] = TableSchema(
            name=t_name,
            columns=cols,
            fk_cascade_actions=tuple(fk_by_table_my.get(t_name.lower(), ())),
        )
    return schemas


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


_FN_TO_AGG_ENUM: Dict[str, AggFunc] = {
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


def _unwrap_outer_parens(sql: str) -> str:
    s = sql.strip()
    while s.startswith("(") and s.endswith(")"):
        toks = sqlglot.tokenize(s, read="sqlite")
        if not toks or toks[0].token_type != SQLGlotTokenType.L_PAREN:
            break
        depth = 0
        closes_at_end = False
        for idx, tk in enumerate(toks):
            if tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN:
                depth -= 1
                if depth == 0:
                    closes_at_end = (idx == len(toks) - 1)
                    break
        if closes_at_end:
            s = s[1:-1].strip()
        else:
            break
    return s


def _parse_single_agg_expr(expr: str, alias: str) -> Optional[SelectItem]:
    """Parses a single top-level aggregate function call using `sqlglot.parse_one` and `sqlglot.tokenize`."""
    s = _unwrap_outer_parens(expr)
    if not s:
        return None
    try:
        ast_node = parse_sql_ast(s)
    except sqlglot.errors.SqlglotError:
        return None

    if isinstance(ast_node, sqlglot_exp.Window) or ast_node.find(sqlglot_exp.Window):
        return None

    agg_node: sqlglot_exp.Expr = ast_node.this if isinstance(ast_node, sqlglot_exp.Filter) else ast_node
    if not isinstance(agg_node, (sqlglot_exp.AggFunc, sqlglot_exp.Func)):
        return None

    toks = sqlglot.tokenize(s, read="sqlite")
    if len(toks) < 3 or not toks[0].text or not (toks[0].text[0].isalpha() or toks[0].text[0] == "_") or toks[1].token_type != SQLGlotTokenType.L_PAREN:
        return None

    depth = 0
    close_idx = -1
    for idx in range(1, len(toks)):
        if toks[idx].token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif toks[idx].token_type == SQLGlotTokenType.R_PAREN:
            depth -= 1
            if depth == 0:
                close_idx = idx
                break
    if close_idx == -1:
        return None

    filter_where: Optional[str] = None
    if close_idx < len(toks) - 1:
        # Must be `FILTER ( WHERE <filter_where> )` spanning to the end of `s`
        if (
            close_idx + 4 >= len(toks)
            or toks[close_idx + 1].token_type != SQLGlotTokenType.FILTER
            or toks[close_idx + 2].token_type != SQLGlotTokenType.L_PAREN
            or toks[close_idx + 3].token_type != SQLGlotTokenType.WHERE
            or toks[-1].token_type != SQLGlotTokenType.R_PAREN
        ):
            return None
        f_depth = 0
        f_close = -1
        for idx in range(close_idx + 2, len(toks)):
            if toks[idx].token_type == SQLGlotTokenType.L_PAREN:
                f_depth += 1
            elif toks[idx].token_type == SQLGlotTokenType.R_PAREN:
                f_depth -= 1
                if f_depth == 0:
                    f_close = idx
                    break
        if f_close != len(toks) - 1:
            return None
        filter_where = s[toks[close_idx + 3].end + 1 : toks[-1].start].strip()
        if not filter_where:
            return None
        fw_toks = sqlglot.tokenize(filter_where, read="sqlite")
        fw_d = 0
        for ftk in fw_toks:
            if ftk.token_type == SQLGlotTokenType.L_PAREN:
                fw_d += 1
            elif ftk.token_type == SQLGlotTokenType.R_PAREN:
                fw_d -= 1
                if fw_d < 0:
                    return None
        if fw_d != 0:
            return None

    fn_raw = toks[0].text.upper()
    inner_args_sql = s[toks[1].end + 1 : toks[close_idx].start].strip()
    in_toks = sqlglot.tokenize(inner_args_sql, read="sqlite")
    is_dist = bool(agg_node.args.get("distinct")) or isinstance(agg_node.this, sqlglot_exp.Distinct)
    if in_toks and in_toks[0].token_type == SQLGlotTokenType.DISTINCT:
        is_dist = True
        inner_args_sql = inner_args_sql[in_toks[0].end + 1 :].strip()
    elif in_toks and in_toks[0].token_type == SQLGlotTokenType.ALL:
        is_dist = False
        inner_args_sql = inner_args_sql[in_toks[0].end + 1 :].strip()

    arg_parts = [p.strip() for p in _split_top_level(inner_args_sql, ",") if p.strip()] if inner_args_sql else []
    first_arg = arg_parts[0] if arg_parts else ""
    extra_args = ", ".join(arg_parts[1:]) if len(arg_parts) > 1 else None
    src_col: Optional[str] = None
    if first_arg and first_arg != "*":
        try:
            first_node = parse_sql_ast(first_arg)
            if isinstance(first_node, sqlglot_exp.Column) and not isinstance(first_node.this, sqlglot_exp.Star):
                src_col = first_node.name
        except sqlglot.errors.SqlglotError:
            pass

    if fn_raw == "COUNT":
        if first_arg in ("*", ""):
            return SelectItem(
                raw_expr=expr.strip(),
                alias=alias,
                is_aggregate=True,
                agg_func=AggFunc.COUNT_STAR,
                arg_expr="*",
                is_distinct=False,
                filter_where=filter_where,
            )
        return SelectItem(
            raw_expr=expr.strip(),
            alias=alias,
            is_aggregate=True,
            agg_func=AggFunc.COUNT_COL,
            arg_expr=first_arg,
            is_distinct=is_dist,
            filter_where=filter_where,
            source_col=src_col,
        )

    if fn_raw not in _FN_TO_AGG_ENUM:
        return None

    return SelectItem(
        raw_expr=expr.strip(),
        alias=alias,
        is_aggregate=True,
        agg_func=_FN_TO_AGG_ENUM[fn_raw],
        arg_expr=first_arg,
        extra_args=extra_args,
        is_distinct=is_dist,
        filter_where=filter_where,
        source_col=src_col,
    )


def _contains_aggregate_call(expr: str) -> bool:
    """Checks via `sqlglot.parse_one` whether `expr` contains an aggregate function call (excluding window functions and subqueries)."""
    try:
        ast_node = parse_sql_ast(expr)
    except sqlglot.errors.SqlglotError:
        return False
    if ast_node.find(sqlglot_exp.Window) or ast_node.find(sqlglot_exp.Subquery, sqlglot_exp.Select):
        return False
    if ast_node.find(sqlglot_exp.AggFunc):
        return True
    for anon_node in ast_node.find_all(sqlglot_exp.Anonymous):
        if anon_node.name.upper() in _FN_TO_AGG_ENUM or anon_node.name.upper() == "COUNT":
            return True
    return False


def _parse_select_item(raw_item: str, idx: int) -> SelectItem:
    """Parses a single `SELECT` projection item via `sqlglot.parse_one`."""
    s = raw_item.strip()
    try:
        sel_ast = sqlglot.parse_one(f"SELECT {s}", read="sqlite")
        proj_ast = sel_ast.expressions[0] if isinstance(sel_ast, sqlglot_exp.Select) and sel_ast.expressions else parse_sql_ast(s)
    except sqlglot.errors.SqlglotError:
        proj_ast = sqlglot_exp.Expr()

    if isinstance(proj_ast, sqlglot_exp.Alias):
        alias = proj_ast.alias
        inner_node = proj_ast.this
        toks = sqlglot.tokenize(s, read="sqlite")
        if len(toks) >= 2 and toks[-2].token_type == SQLGlotTokenType.ALIAS:
            expr = s[: toks[-2].start].strip()
        elif len(toks) >= 2:
            expr = s[: toks[-1].start].strip()
        else:
            expr = inner_node.sql().strip() if isinstance(inner_node, sqlglot_exp.Expr) else str(inner_node or "").strip()
    else:
        inner_node = proj_ast
        expr = s
        if isinstance(inner_node, sqlglot_exp.Column) and not isinstance(inner_node.this, sqlglot_exp.Star):
            alias = inner_node.name
        else:
            alias = f"col_{idx}"

    parsed_agg = _parse_single_agg_expr(expr, alias)
    if parsed_agg is not None:
        return parsed_agg

    if _contains_aggregate_call(expr):
        return SelectItem(
            raw_expr=expr,
            alias=alias,
            is_aggregate=True,
            agg_func=None,
            arg_expr=expr,
        )

    src_col = (
        inner_node.name
        if isinstance(inner_node, sqlglot_exp.Column) and not isinstance(inner_node.this, sqlglot_exp.Star)
        else None
    )
    return SelectItem(
        raw_expr=expr,
        alias=alias,
        is_aggregate=False,
        source_col=src_col,
    )


def _extract_verbatim_cte_or_subquery_sql(raw_select_sql: str, has_cte: bool) -> Optional[str]:
    """Slices the verbatim inner query of a single CTE (`WITH x AS (...)`) or `FROM (...)` subquery via `sqlglot.tokenize`."""
    s = raw_select_sql.strip()
    toks = _tokenize_sql(s)
    if not toks:
        return None
    if has_cte:
        depth = 0
        for i, tk in enumerate(toks):
            if depth == 0 and tk.token_type == SQLGlotTokenType.L_PAREN and i > 0 and toks[i - 1].token_type == SQLGlotTokenType.ALIAS:
                start_char = tk.end + 1
                p_depth = 1
                for j in range(i + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        p_depth -= 1
                        if p_depth == 0:
                            return s[start_char : toks[j].start].strip()
            elif tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                depth -= 1
    else:
        depth = 0
        seen_from = False
        for i, tk in enumerate(toks):
            if depth == 0 and tk.token_type == SQLGlotTokenType.FROM:
                seen_from = True
            elif seen_from and depth == 0 and tk.token_type == SQLGlotTokenType.L_PAREN:
                start_char = tk.end + 1
                p_depth = 1
                for j in range(i + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        p_depth -= 1
                        if p_depth == 0:
                            return s[start_char : toks[j].start].strip()
            elif tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                depth -= 1
    return None


def _extract_cte_names_and_strip_with(select_sql: str) -> Tuple[Set[str], str, bool]:
    """Extracts CTE names and the main SELECT body from a query using `sqlglot.parse_one` AST and `sqlglot.tokenize`."""
    s = select_sql.strip()
    try:
        q_ast = parse_sql_ast(s)
    except sqlglot.errors.SqlglotError:
        return set(), s, False
    with_node = q_ast.args.get("with_") or q_ast.args.get("with")
    if not isinstance(with_node, sqlglot_exp.With):
        return set(), s, False
    cte_names = {cte.alias for cte in with_node.expressions if isinstance(cte, sqlglot_exp.CTE) and cte.alias}
    toks = sqlglot.tokenize(s, read="sqlite")
    depth = 0
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif depth == 0 and tk.token_type == SQLGlotTokenType.SELECT:
            main_select = s[tk.start:].strip()
            return cte_names, (main_select if main_select else s), True
    no_with = q_ast.copy()
    no_with.set("with_", None)
    no_with.set("with", None)
    no_with.args.pop("_raw_sql", None)
    main_select = no_with.sql("sqlite").strip()
    return cte_names, (main_select if main_select else s), True


def _parse_from_and_joins(from_clause: str, cte_names: Set[str]) -> Tuple[List[JoinSpec], bool]:
    """Parses a `FROM ... [JOIN ...]` clause into `JoinSpec` objects using `sqlglot.parse_one`."""
    s = from_clause.strip()
    if not s:
        return [], False
    toks = sqlglot.tokenize(s, read="sqlite")
    if not toks:
        return [], False
    if toks[0].token_type == SQLGlotTokenType.FROM:
        s_body = s[toks[0].end + 1 :].strip()
    else:
        s_body = s
    if not s_body:
        return [], False
    try:
        sel_ast = sqlglot.parse_one(f"SELECT * FROM {s_body}", read="sqlite")
    except sqlglot.errors.SqlglotError:
        return [], False
    if not isinstance(sel_ast, sqlglot_exp.Select):
        return [], False

    from_node = sel_ast.args.get("from_") or sel_ast.args.get("from")
    join_nodes = sel_ast.args.get("joins") or []
    if not isinstance(from_node, sqlglot_exp.From):
        return [], False

    # Also extract raw ON condition slices from `s_body` via `sqlglot.tokenize`
    body_toks = sqlglot.tokenize(s_body, read="sqlite")
    raw_on_slices: List[Optional[str]] = []
    b_depth = 0
    join_kw_words = {"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "JOIN"}
    on_start_char: Optional[int] = None
    in_join = False
    b_idx = 0
    while b_idx < len(body_toks):
        btk = body_toks[b_idx]
        if btk.token_type == SQLGlotTokenType.L_PAREN:
            b_depth += 1
        elif btk.token_type == SQLGlotTokenType.R_PAREN and b_depth > 0:
            b_depth -= 1
        elif b_depth == 0:
            if btk.token_type == SQLGlotTokenType.COMMA or btk.text.upper() in join_kw_words:
                if in_join:
                    raw_on_slices.append(
                        s_body[on_start_char : btk.start].strip() if on_start_char is not None else None
                    )
                    on_start_char = None
                in_join = True
                if btk.token_type != SQLGlotTokenType.COMMA:
                    while b_idx + 1 < len(body_toks) and body_toks[b_idx + 1].text.upper() in (join_kw_words | {"OUTER"}):
                        b_idx += 1
            elif in_join and btk.token_type == SQLGlotTokenType.ON and on_start_char is None:
                on_start_char = btk.end + 1
        b_idx += 1
    if in_join:
        raw_on_slices.append(s_body[on_start_char:].strip() if on_start_char is not None else None)

    joins: List[JoinSpec] = []
    has_subquery = False

    first_target = from_node.this
    if isinstance(first_target, sqlglot_exp.Table):
        t_name = first_target.name or "t"
        al = first_target.alias or t_name
        joins.append(JoinSpec(join_type="FROM", table_name=t_name, alias=al))
    else:
        has_subquery = True
        al = (first_target.alias if isinstance(first_target, sqlglot_exp.Expr) else "") or "__mv_sub"
        joins.append(JoinSpec(join_type="FROM", table_name=al, alias=al))

    for idx, j_node in enumerate(join_nodes):
        side = j_node.side.upper()
        kind = j_node.kind.upper()
        method = j_node.method.upper()
        if method == "NATURAL":
            j_type = "NATURAL JOIN"
        elif side == "LEFT":
            j_type = "LEFT JOIN"
        elif side == "RIGHT":
            j_type = "RIGHT JOIN"
        elif side == "FULL":
            j_type = "FULL JOIN"
        elif kind == "CROSS" or (
            not side and not kind and not method and not j_node.args.get("on") and not j_node.args.get("using")
        ):
            j_type = "CROSS JOIN"
        else:
            j_type = "INNER JOIN"

        if j_node.args.get("using"):
            has_subquery = True

        j_target = j_node.this
        if isinstance(j_target, sqlglot_exp.Table):
            t_name = j_target.name or f"t_{idx}"
            al = j_target.alias or t_name
        else:
            has_subquery = True
            t_name = (j_target.alias if isinstance(j_target, sqlglot_exp.Expr) else "") or f"__mv_sub_{idx}"
            al = t_name

        on_expr = j_node.args.get("on")
        on_cond: Optional[str] = None
        if idx < len(raw_on_slices) and raw_on_slices[idx]:
            on_cond = raw_on_slices[idx]
        elif isinstance(on_expr, sqlglot_exp.Expr):
            on_cond = on_expr.sql().strip()
        joins.append(
            JoinSpec(
                join_type=j_type,
                table_name=t_name,
                alias=al,
                on_condition=on_cond,
            )
        )

    return joins, has_subquery


def _extract_all_referenced_tables(
    select_sql: str,
    cte_names: Set[str],
    known_tables: Optional[Set[str]] = None,
) -> List[str]:
    """Extracts all referenced base tables across CTEs, FROM/JOIN clauses, set operations, and subqueries via `sqlglot` AST."""
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

    def _add_candidate(cand: str, was_quoted: bool = False) -> None:
        if not cand:
            return
        c_low = cand.lower()
        if c_low in cte_lower or c_low.startswith(("__matview_", "__mv_")):
            return
        if not was_quoted and c_low in sql_reserved and (not known_tables or c_low not in known_tables):
            return
        if known_tables is not None and known_tables and c_low not in known_tables:
            return
        if c_low not in seen_lower:
            seen_lower.add(c_low)
            found.append(cand)

    try:
        q_ast = parse_sql_ast(select_sql)
        for tbl_node in q_ast.find_all(sqlglot_exp.Table):
            id_node = tbl_node.this
            was_q = isinstance(id_node, sqlglot_exp.Identifier) and id_node.quoted
            _add_candidate(tbl_node.name, was_quoted=was_q)
    except sqlglot.errors.SqlglotError:
        pass

    # Also scan token stream for any `FROM <ident>` / `JOIN <ident>` inside nested expressions
    toks = sqlglot.tokenize(select_sql, read="sqlite")
    for idx, tk in enumerate(toks):
        if tk.token_type in (SQLGlotTokenType.FROM, SQLGlotTokenType.JOIN) and idx + 1 < len(toks):
            nxt = toks[idx + 1]
            if _is_word_or_ident_token(nxt):
                _add_candidate(nxt.text, was_quoted=(nxt.token_type == SQLGlotTokenType.IDENTIFIER))

    return found


def parse_view_index_spec(
    spec: ViewIndexSpec | str | Sequence[str] | Dict[str, Any],
    default_view_name: Optional[str] = None,
) -> ViewIndexSpec:
    """Parses a user-defined index specification (`CREATE [UNIQUE] INDEX ...`, tuple/list, dict, or ViewIndexSpec) via `sqlglot` AST."""
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
        cols_raw = spec.get("columns") or spec.get("cols")
        if isinstance(cols_raw, str):
            cols = tuple(c.strip() for c in cols_raw.split(",") if c.strip())
        elif isinstance(cols_raw, Sequence):
            cols = tuple(str(c).strip() for c in cols_raw if str(c).strip())
        else:
            raise ValueError(f"Index dict specification requires 'columns': {spec!r}")
        if not cols:
            raise ValueError(f"Index specification must contain at least one column: {spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=spec.get("name"),
            unique=bool(spec.get("unique", False)),
            where_clause=spec.get("where") or spec.get("where_clause"),
            using_method=spec.get("using") or spec.get("using_method"),
            target_view=spec.get("target_view") or spec.get("view") or default_view_name,
        )

    if isinstance(spec, (tuple, list)):
        cols = tuple(str(c).strip() for c in spec if str(c).strip())
        if not cols:
            raise ValueError(f"Index column list cannot be empty: {spec!r}")
        return ViewIndexSpec(columns=cols, target_view=default_view_name)

    if not isinstance(spec, str):
        raise ValueError(f"Unsupported index specification type: {type(spec).__name__}")

    s = _strip_sql_comments(spec).strip().rstrip(";").strip()
    if not s:
        raise ValueError(f"Empty index specification: {spec!r}")

    toks = sqlglot.tokenize(s, read="sqlite")
    if toks and toks[0].token_type == SQLGlotTokenType.CREATE:
        idx_create = parse_sql_ast(s)
        if isinstance(idx_create, sqlglot_exp.Create) and idx_create.kind.upper() == "INDEX":
            idx_node = idx_create.this
            params_node = idx_node.args.get("params") if isinstance(idx_node, sqlglot_exp.Index) else None
            if isinstance(idx_node, sqlglot_exp.Index) and isinstance(params_node, sqlglot_exp.IndexParameters):
                cols = tuple(c.sql().strip() for c in params_node.columns if c.sql().strip())
                if not cols:
                    raise ValueError(f"CREATE INDEX statement has no columns: {spec!r}")
                tbl_node = idx_node.args.get("table")
                target_v = tbl_node.name if isinstance(tbl_node, sqlglot_exp.Table) else default_view_name
                where_node = params_node.args.get("where")
                where_c: Optional[str] = None
                where_tok_ci = _find_top_level_token(s, (SQLGlotTokenType.WHERE,))
                if where_tok_ci is not None:
                    where_c = s[where_tok_ci.end + 1 :].strip()
                elif isinstance(where_node, sqlglot_exp.Where) and where_node.this is not None:
                    where_c = where_node.this.sql().strip()
                using_node = params_node.args.get("using")
                using_m = (
                    (using_node.name or using_node.sql().strip())
                    if isinstance(using_node, sqlglot_exp.Expr)
                    else (str(using_node) if using_node else None)
                )
                return ViewIndexSpec(
                    columns=cols,
                    name=idx_node.name or None,
                    unique=bool(idx_create.args.get("unique")),
                    where_clause=where_c,
                    using_method=using_m,
                    target_view=target_v,
                )

    # Shorthand string format: `[UNIQUE:]col1, col2 [WHERE ...]`
    is_unique_sh = False
    if len(toks) >= 2 and toks[0].token_type == SQLGlotTokenType.UNIQUE and toks[1].token_type == SQLGlotTokenType.COLON:
        is_unique_sh = True
        s = s[toks[1].end + 1 :].strip()

    where_sh: Optional[str] = None
    where_tok = _find_top_level_token(s, (SQLGlotTokenType.WHERE,))
    if where_tok is not None:
        where_sh = s[where_tok.end + 1 :].strip()
        s = s[: where_tok.start].strip()

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    cols_sh = tuple(c.strip() for c in _split_top_level(s, ",") if c.strip())
    if not cols_sh:
        raise ValueError(f"Invalid index specification: {spec!r}")
    return ViewIndexSpec(
        columns=cols_sh,
        unique=is_unique_sh,
        where_clause=where_sh,
        target_view=default_view_name,
    )


def parse_view_ast(
    view_or_select_sql: str,
    dialect: Dialect = Dialect.SQLITE,
    known_tables: Optional[Set[str]] = None,
) -> SQLSelectAST:
    """
    Parses a `CREATE VIEW ... AS SELECT ...` or standalone `SELECT ...` query into
    a `SQLSelectAST` backed by `sqlglot.parse_one`.
    """
    clean = _strip_sql_comments(view_or_select_sql).strip().rstrip(";").strip()
    sg_ast = parse_sql_ast(clean, dialect=dialect)
    toks = sqlglot.tokenize(clean, read="sqlite")
    if toks and toks[0].token_type == SQLGlotTokenType.CREATE:
        vs = _parse_single_create_view_stmt(clean, known_tables=known_tables)
    else:
        vs = _parse_single_create_view_stmt(f"CREATE VIEW __mv_ast_tmp AS {clean}", known_tables=known_tables)
    cte_names, main_sel, has_cte = _extract_cte_names_and_strip_with(vs.raw_select_sql)
    return SQLSelectAST(
        raw_sql=vs.raw_select_sql,
        cte_names=tuple(sorted(cte_names)),
        main_select_sql=main_sel,
        has_cte=has_cte,
        has_distinct=vs.has_top_level_distinct,
        select_items=vs.select_items,
        from_clause=vs.base_table,
        joins=vs.joins,
        where_clause=vs.where_clause,
        group_by_exprs=vs.group_by_cols,
        having_clause=vs.having_clause,
        has_window_func=vs.has_window_func,
        has_compound_set_op=vs.has_compound_set_op,
        has_limit_or_order=vs.has_limit_or_order,
        has_subquery_in_from=vs.has_subquery_in_from,
        sqlglot_ast=sg_ast,
    )


def _parse_single_create_view_stmt(
    stmt_sql: str,
    known_tables: Optional[Set[str]] = None,
    extra_indexes: Sequence[ViewIndexSpec] = (),
) -> ViewSpec:
    """Parses a single `CREATE VIEW ... AS SELECT ...` statement into a `ViewSpec` using `sqlglot.parse_one`."""
    clean = _strip_sql_comments(stmt_sql).strip().rstrip(";").strip()
    try:
        create_ast = parse_sql_ast(clean)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Invalid CREATE VIEW syntax: {stmt_sql!r}") from exc

    if not isinstance(create_ast, sqlglot_exp.Create) or create_ast.kind.upper() != "VIEW":
        raise ValueError(f"Invalid CREATE VIEW syntax: {stmt_sql!r}")

    view_tbl = create_ast.this
    view_name = view_tbl.name if isinstance(view_tbl, sqlglot_exp.Table) else ""
    if not view_name:
        raise ValueError(f"Invalid CREATE VIEW syntax (missing view name): {stmt_sql!r}")

    query_ast = create_ast.expression
    if not isinstance(query_ast, (sqlglot_exp.Select, sqlglot_exp.SetOperation)):
        raise ValueError(f"View body must start with SELECT or WITH: {stmt_sql!r}")

    as_tok = _find_top_level_token(clean, (SQLGlotTokenType.ALIAS,))
    raw_select = clean[as_tok.end + 1 :].strip() if as_tok is not None else query_ast.sql().strip()
    cte_names, main_select_body, has_cte = _extract_cte_names_and_strip_with(raw_select)

    has_window = bool(query_ast.find(sqlglot_exp.Window))
    has_compound = isinstance(query_ast, sqlglot_exp.SetOperation)

    first_select: sqlglot_exp.Expr = query_ast
    while isinstance(first_select, sqlglot_exp.SetOperation):
        first_select = first_select.this
    if not isinstance(first_select, sqlglot_exp.Select):
        raise ValueError(f"Unable to parse SELECT clause in view {view_name!r}: {raw_select!r}")

    has_limit_or_order = bool(
        query_ast.find(sqlglot_exp.Order, sqlglot_exp.Limit, sqlglot_exp.Offset)
        if has_compound
        else (
            first_select.args.get("order")
            or first_select.args.get("limit")
            or first_select.args.get("offset")
        )
    )

    has_top_distinct = bool(first_select.args.get("distinct"))

    if has_compound:
        _, setop_branches = _extract_set_op_branches_from_ast(query_ast, main_select_body)
        first_branch_sql = setop_branches[0] if setop_branches else main_select_body
    else:
        first_branch_sql = _unwrap_outer_parens(main_select_body)

    fb_toks = sqlglot.tokenize(first_branch_sql, read="sqlite")
    fb_depth = 0
    sel_tok_i = -1
    from_tok_i = -1
    where_tok_i = -1
    group_tok_i = -1
    having_tok_i = -1
    trail_tok_i = -1
    for t_i, ftk in enumerate(fb_toks):
        if ftk.token_type == SQLGlotTokenType.L_PAREN:
            fb_depth += 1
        elif ftk.token_type == SQLGlotTokenType.R_PAREN and fb_depth > 0:
            fb_depth -= 1
        elif fb_depth == 0:
            tt = ftk.token_type
            if tt == SQLGlotTokenType.SELECT and sel_tok_i == -1:
                sel_tok_i = t_i
            elif sel_tok_i != -1:
                if tt == SQLGlotTokenType.FROM and from_tok_i == -1:
                    from_tok_i = t_i
                elif tt == SQLGlotTokenType.WHERE and where_tok_i == -1:
                    where_tok_i = t_i
                elif tt == SQLGlotTokenType.GROUP_BY and group_tok_i == -1:
                    group_tok_i = t_i
                elif tt == SQLGlotTokenType.HAVING and having_tok_i == -1:
                    having_tok_i = t_i
                elif tt in (
                    SQLGlotTokenType.WINDOW,
                    SQLGlotTokenType.ORDER_BY,
                    SQLGlotTokenType.LIMIT,
                    SQLGlotTokenType.OFFSET,
                    SQLGlotTokenType.FETCH,
                ) and trail_tok_i == -1:
                    trail_tok_i = t_i

    clause_tok_positions = [
        p for p in (from_tok_i, where_tok_i, group_tok_i, having_tok_i, trail_tok_i) if p != -1
    ]

    def _slice_fb_clause(tok_idx: int) -> str:
        if tok_idx == -1:
            return ""
        c_start = fb_toks[tok_idx].end + 1
        later_pos = [p for p in clause_tok_positions if p > tok_idx]
        c_end = fb_toks[min(later_pos)].start if later_pos else len(first_branch_sql)
        return first_branch_sql[c_start:c_end].strip()

    if sel_tok_i != -1:
        p_start_i = sel_tok_i + 1
        if p_start_i < len(fb_toks) and fb_toks[p_start_i].token_type in (SQLGlotTokenType.DISTINCT, SQLGlotTokenType.ALL):
            p_start_i += 1
        p_start_char = fb_toks[p_start_i].start if p_start_i < len(fb_toks) else len(first_branch_sql)
        p_end_char = fb_toks[min(clause_tok_positions)].start if clause_tok_positions else len(first_branch_sql)
        proj_list_sql = first_branch_sql[p_start_char:p_end_char].strip()
        raw_proj_items = [p.strip() for p in _split_top_level(proj_list_sql, ",") if p.strip()]
    else:
        raw_proj_items = [proj.sql() for proj in first_select.expressions]

    select_items = [
        _parse_select_item(raw_p, idx)
        for idx, raw_p in enumerate(raw_proj_items)
    ]

    from_clause = _slice_fb_clause(from_tok_i)
    where_raw = _slice_fb_clause(where_tok_i)
    where_clause = where_raw if where_raw else None
    group_raw = _slice_fb_clause(group_tok_i)
    group_by_cols = [g.strip() for g in _split_top_level(group_raw, ",") if g.strip()] if group_raw else []
    having_raw = _slice_fb_clause(having_tok_i)
    having_clause = having_raw if having_raw else None

    joins, has_subquery_in_from = _parse_from_and_joins(from_clause, cte_names)
    ref_tables = _extract_all_referenced_tables(raw_select, cte_names, known_tables=known_tables)
    cte_lower = {c.lower() for c in cte_names}
    if (
        not has_subquery_in_from
        and joins
        and joins[0].table_name.lower() not in cte_lower
        and not joins[0].table_name.startswith("__mv_")
        and (not known_tables or joins[0].table_name.lower() in known_tables)
    ):
        primary_t = joins[0].table_name
        ref_tables = [primary_t] + [t for t in ref_tables if t.lower() != primary_t.lower()]
        base_table = primary_t
    else:
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
            st_ast = parse_sql_ast(st)
            if isinstance(st_ast, sqlglot_exp.Create) and st_ast.kind.upper() == "VIEW":
                initial_specs.append(_parse_single_create_view_stmt(st, known_tables=known_tables))
            elif isinstance(st_ast, sqlglot_exp.Create) and st_ast.kind.upper() == "INDEX":
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
        dialect: Dialect | str | EngineVersion = Dialect.SQLITE,
        pg_use_transition_tables: bool = False,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        incremental_backfill: bool = False,
        batch_size: int = 1000,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
        watermark_catalog_table: Optional[str] = None,
    ) -> None:
        self.schemas = dict(schemas)
        self._schemas_lower: Dict[str, TableSchema] = {
            k.lower(): v for k, v in self.schemas.items()
        }
        self.raw_view = view_spec
        if engine_version is not None:
            self.engine_version = parse_engine_version(
                engine_version,
                default_dialect=dialect if not isinstance(dialect, EngineVersion) else dialect.dialect,
            )
        else:
            self.engine_version = parse_engine_version(dialect)
        self.dialect: Dialect = self.engine_version.dialect
        self.subdialect: EngineSubdialect = self.engine_version.subdialect
        # Automatically disable transition tables if targeting PostgreSQL < 10.0
        self.pg_use_transition_tables = bool(
            pg_use_transition_tables and self.engine_version.supports_pg_transition_tables
        )
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.incremental_backfill = incremental_backfill
        self.batch_size = max(1, int(batch_size)) if batch_size else 1000
        self.strict = strict
        self.watermark_catalog_table = watermark_catalog_table

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
        unwrapped_view = self._try_unwrap_simple_cte_or_derived_view(view_spec)
        self.primary_schema: Optional[TableSchema] = self._schemas_lower.get(
            unwrapped_view.base_table.lower()
        )
        self.view = self._try_expand_select_star(unwrapped_view)

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
        if self.strict and self.strategy != MaintenanceStrategy.ALGEBRAIC_DELTA:
            raise StrictCompilationError(
                f"View '{self.logical_view_name}' cannot be compiled with Tier 1 ALGEBRAIC_DELTA "
                f"maintenance (selected fallback strategy: {self.strategy.value}) under strict=True."
            )

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
        if self.view.has_window_func:
            if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_window_functions:
                lims.append(
                    TransparencyLimitation(
                        code="SQLITE_VERSION_NO_WINDOW_FUNC",
                        severity=LimitationSeverity.BLOCKER,
                        dialect=Dialect.SQLITE,
                        affected_object=self.logical_view_name,
                        description=(
                            f"View '{self.logical_view_name}' uses window functions, which require "
                            f"SQLite >= 3.25.0 (target engine is {self.engine_version.label})."
                        ),
                        mitigation="Upgrade SQLite to >= 3.25.0 or rewrite the query without OVER (...).",
                    )
                )
            elif self.dialect == Dialect.MYSQL and not self.engine_version.supports_mysql_window_functions:
                lims.append(
                    TransparencyLimitation(
                        code="MYSQL_VERSION_NO_WINDOW_FUNC",
                        severity=LimitationSeverity.BLOCKER,
                        dialect=Dialect.MYSQL,
                        affected_object=self.logical_view_name,
                        description=(
                            f"View '{self.logical_view_name}' uses window functions, which require "
                            f"MySQL >= 8.0.0 or MariaDB >= 10.2.0 (target engine is {self.engine_version.label})."
                        ),
                        mitigation="Upgrade MySQL to >= 8.0.0 / MariaDB to >= 10.2.0 or rewrite the query.",
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

    def _try_expand_select_star(self, v: ViewSpec) -> ViewSpec:
        """
        Expands `SELECT *` or `SELECT alias.*` on single-table views into explicit schema columns
        so `SELECT * FROM t [WHERE ...]` and `SELECT DISTINCT * FROM t [WHERE ...]` compile with
        Tier 1 O(1) ALGEBRAIC_DELTA maintenance instead of falling back to Tier 3.
        """
        if (
            self.primary_schema is None
            or len(v.joins) != 1
            or len(v.referenced_tables) != 1
            or v.has_cte
            or v.has_compound_set_op
            or v.has_subquery_in_from
            or v.has_window_func
            or v.has_limit_or_order
        ):
            return v
        base_alias = v.joins[0].alias.lower() if v.joins else v.base_table.lower()
        base_name = v.base_table.lower()
        has_star = any(
            it.raw_expr.strip() == "*"
            or it.raw_expr.strip().lower() in (f"{base_alias}.*", f"{base_name}.*")
            for it in v.select_items
        )
        if not has_star:
            return v
        expanded: List[SelectItem] = []
        for it in v.select_items:
            raw_s = it.raw_expr.strip()
            if raw_s == "*" or raw_s.lower() in (f"{base_alias}.*", f"{base_name}.*"):
                for c_def in self.primary_schema.columns.values():
                    expanded.append(
                        SelectItem(
                            raw_expr=c_def.name,
                            alias=c_def.name,
                            is_aggregate=False,
                            source_col=c_def.name,
                        )
                    )
            else:
                expanded.append(it)
        sel_str = ", ".join(f"{it.raw_expr} AS {it.alias}" for it in expanded)
        dist_kw = "DISTINCT " if v.has_top_level_distinct else ""
        from_j = v.joins[0]
        from_str = (
            f"{from_j.table_name} AS {from_j.alias}"
            if from_j.alias and from_j.alias.lower() != from_j.table_name.lower()
            else from_j.table_name
        )
        rebuilt_sql = f"SELECT {dist_kw}{sel_str} FROM {from_str}"
        if v.where_clause:
            rebuilt_sql += f" WHERE {v.where_clause}"
        if v.group_by_cols:
            rebuilt_sql += f" GROUP BY {', '.join(v.group_by_cols)}"
        if v.having_clause:
            rebuilt_sql += f" HAVING {v.having_clause}"
        return ViewSpec(
            view_name=v.view_name,
            raw_select_sql=rebuilt_sql,
            base_table=v.base_table,
            joins=v.joins,
            referenced_tables=v.referenced_tables,
            select_items=tuple(expanded),
            where_clause=v.where_clause,
            group_by_cols=v.group_by_cols,
            having_clause=v.having_clause,
            has_cte=v.has_cte,
            has_window_func=v.has_window_func,
            has_compound_set_op=v.has_compound_set_op,
            has_top_level_distinct=v.has_top_level_distinct,
            has_limit_or_order=v.has_limit_or_order,
            has_subquery_in_from=v.has_subquery_in_from,
            user_indexes=v.user_indexes,
            additional_views=v.additional_views,
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

        try:
            q_ast = parse_sql_ast(v.raw_select_sql)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(q_ast, sqlglot_exp.Select):
            return v

        if v.has_cte:
            with_node = q_ast.args.get("with_") or q_ast.args.get("with")
            if not isinstance(with_node, sqlglot_exp.With) or bool(with_node.args.get("recursive")) or len(with_node.expressions) != 1:
                return v
            cte_node = with_node.expressions[0]
            if not isinstance(cte_node, sqlglot_exp.CTE) or not isinstance(cte_node.this, sqlglot_exp.Expr):
                return v
            cte_alias_node = cte_node.args.get("alias")
            if isinstance(cte_alias_node, sqlglot_exp.TableAlias) and cte_alias_node.columns:
                return v
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, True) or cte_node.this.sql("sqlite").strip()
            cte_or_sub_alias = cte_node.alias
            _, outer_sql, _ = _extract_cte_names_and_strip_with(v.raw_select_sql)
        elif v.has_subquery_in_from and len(v.joins) == 1:
            from_node = q_ast.args.get("from_") or q_ast.args.get("from")
            if not isinstance(from_node, sqlglot_exp.From) or not isinstance(from_node.this, sqlglot_exp.Subquery):
                return v
            sub_node = from_node.this
            if not isinstance(sub_node.this, sqlglot_exp.Expr):
                return v
            inner_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, False) or sub_node.this.sql("sqlite").strip()
            cte_or_sub_alias = sub_node.alias or "__mv_sub"
            outer_sql = v.raw_select_sql
        else:
            return v

        if not inner_sql or not outer_sql:
            return v
        try:
            inner_ast_check = parse_sql_ast(inner_sql)
        except sqlglot.errors.SqlglotError:
            return v
        if not isinstance(inner_ast_check, sqlglot_exp.Select) or any(
            inner_ast_check.args.get(k) for k in ("group", "having", "order", "limit", "offset")
        ):
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
            toks = sqlglot.tokenize(frag, read="sqlite")
            if not toks:
                return frag
            out: List[str] = []
            cursor = 0
            i = 0
            n = len(toks)
            while i < n:
                tk = toks[i]
                if (
                    _is_word_or_ident_token(tk)
                    and i + 2 < n
                    and toks[i + 1].token_type == SQLGlotTokenType.DOT
                    and _is_word_or_ident_token(toks[i + 2])
                ):
                    tbl_u = tk.text.lower()
                    col_u = toks[i + 2].text.lower()
                    if tbl_u in {outer_alias.lower(), (cte_or_sub_alias or "").lower()}:
                        if tk.start > cursor:
                            out.append(frag[cursor:tk.start])
                        rep = col_map.get(col_u, toks[i + 2].text)
                        out.append(rep)
                        cursor = toks[i + 2].end + 1
                        i += 3
                        continue
                if _is_word_or_ident_token(tk):
                    is_after_dot = (i > 0 and toks[i - 1].token_type == SQLGlotTokenType.DOT)
                    is_before_dot = (i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.DOT)
                    is_fn = (
                        tk.token_type != SQLGlotTokenType.IDENTIFIER
                        and i + 1 < n
                        and toks[i + 1].token_type == SQLGlotTokenType.L_PAREN
                    )
                    if not is_after_dot and not is_before_dot and not is_fn:
                        col_u = tk.text.lower()
                        if col_u in col_map:
                            if tk.start > cursor:
                                out.append(frag[cursor:tk.start])
                            out.append(col_map[col_u])
                            cursor = tk.end + 1
                            i += 1
                            continue
                i += 1
            if cursor < len(frag):
                out.append(frag[cursor:])
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
                agg_call_to_alias[" ".join(t.text.upper() for t in sqlglot.tokenize(it.raw_expr.strip(), read="sqlite"))] = it.alias

        def _extract_agg_calls_from_expr(expr_str: str) -> Tuple[str, bool]:
            nonlocal elem_counter
            toks = sqlglot.tokenize(expr_str, read="sqlite")
            if not toks:
                return expr_str, True
            out_parts: List[str] = []
            idx = 0
            n = len(toks)
            cursor = 0
            agg_targets = {
                "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX",
                "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
                "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
            }
            while idx < n:
                tk = toks[idx]
                fn_up = tk.text.upper()
                if fn_up in agg_targets and idx + 1 < n and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN:
                    depth = 0
                    rp_idx = -1
                    for j in range(idx + 1, n):
                        if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                            depth += 1
                        elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                            depth -= 1
                            if depth == 0:
                                rp_idx = j
                                break
                    if rp_idx != -1:
                        end_tok_idx = rp_idx
                        if (
                            rp_idx + 4 < n
                            and toks[rp_idx + 1].token_type == SQLGlotTokenType.FILTER
                            and toks[rp_idx + 2].token_type == SQLGlotTokenType.L_PAREN
                            and toks[rp_idx + 3].token_type == SQLGlotTokenType.WHERE
                        ):
                            f_depth = 0
                            f_rp = -1
                            for k in range(rp_idx + 2, n):
                                if toks[k].token_type == SQLGlotTokenType.L_PAREN:
                                    f_depth += 1
                                elif toks[k].token_type == SQLGlotTokenType.R_PAREN:
                                    f_depth -= 1
                                    if f_depth == 0:
                                        f_rp = k
                                        break
                            if f_rp != -1:
                                end_tok_idx = f_rp
                        full_call = expr_str[tk.start : toks[end_tok_idx].end + 1]
                        norm_key = " ".join(t.text.upper() for t in sqlglot.tokenize(full_call, read="sqlite"))
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
                        if tk.start > cursor:
                            out_parts.append(expr_str[cursor:tk.start])
                        out_parts.append(col_al)
                        cursor = toks[end_tok_idx].end + 1
                        idx = end_tok_idx + 1
                        continue
                idx += 1
            if cursor < len(expr_str):
                out_parts.append(expr_str[cursor:])
            return "".join(out_parts), True

        all_ok = True
        for it in v.select_items:
            if not it.is_aggregate or it.agg_func is not None:
                work_items.append(it)
                if self.dialect == Dialect.MYSQL and it.agg_func in (AggFunc.BIT_OR, AggFunc.BIT_XOR):
                    reader_exprs[it.alias] = f"COALESCE({it.alias}, 0) AS {it.alias}"
                elif self.dialect == Dialect.MYSQL and it.agg_func == AggFunc.BIT_AND:
                    reader_exprs[it.alias] = f"COALESCE({it.alias}, 18446744073709551615) AS {it.alias}"
                else:
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
        if v.where_clause and _has_subquery_ast(v.where_clause):
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
            if _has_subquery_ast(it.raw_expr):
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
        if v.where_clause and _has_subquery_ast(v.where_clause):
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
            if _has_subquery_ast(it.raw_expr):
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
        if v.where_clause and _has_subquery_ast(v.where_clause):
            return False
        if any(it.raw_expr.strip() == "*" or _has_subquery_ast(it.raw_expr) for it in v.select_items):
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
        if v.where_clause and _has_subquery_ast(v.where_clause):
            return False
        if len(v.select_items) == 1 and v.select_items[0].raw_expr.strip() == "*":
            return False
        if not any(c.primary_key for c in self.primary_schema.columns.values()):
            return False
        for it in v.select_items:
            if _has_subquery_ast(it.raw_expr):
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
        if v.where_clause and _has_subquery_ast(v.where_clause):
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
            if _has_subquery_ast(it.raw_expr):
                return False
            if it.is_aggregate and (it.is_distinct or it.agg_func not in allowed_aggs):
                return False

        inner_union_sql: Optional[str] = None
        sub_alias: str = "__mv_u"
        try:
            q_ast = parse_sql_ast(v.raw_select_sql)
        except sqlglot.errors.SqlglotError:
            return False
        if not isinstance(q_ast, sqlglot_exp.Select):
            return False

        if v.has_cte:
            with_node = q_ast.args.get("with_") or q_ast.args.get("with")
            if not isinstance(with_node, sqlglot_exp.With) or bool(with_node.args.get("recursive")) or len(with_node.expressions) != 1:
                return False
            cte_node = with_node.expressions[0]
            cte_name = cte_node.alias
            if len(v.joins) != 1 or v.joins[0].table_name.lower() != cte_name.lower():
                return False
            inner_union_ast = cte_node.this
            inner_union_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, True) or (inner_union_ast.sql("sqlite").strip() if isinstance(inner_union_ast, sqlglot_exp.Expr) else None)
            sub_alias = v.joins[0].alias or cte_name
        else:
            from_node = q_ast.args.get("from_") or q_ast.args.get("from")
            if not isinstance(from_node, sqlglot_exp.From) or not isinstance(from_node.this, sqlglot_exp.Subquery):
                return False
            sub_node = from_node.this
            inner_union_ast = sub_node.this
            inner_union_sql = _extract_verbatim_cte_or_subquery_sql(v.raw_select_sql, False) or (inner_union_ast.sql("sqlite").strip() if isinstance(inner_union_ast, sqlglot_exp.Expr) else None)
            if sub_node.alias:
                sub_alias = sub_node.alias

        if not inner_union_sql or not isinstance(inner_union_ast, sqlglot_exp.SetOperation):
            return False

        ops, branch_sqls = _extract_set_op_branches_from_ast(inner_union_ast, inner_union_sql)
        if not ops or any(op != "UNION ALL" for op in ops):
            return False

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
            if bs.where_clause and _has_subquery_ast(bs.where_clause):
                return False
            branch_specs.append(bs)

        self._union_all_inner_sql = inner_union_sql
        self._union_all_sub_alias = sub_alias
        self._union_all_branch_specs = branch_specs
        return True

    def _choose_strategy(self) -> MaintenanceStrategy:
        if self.dialect == Dialect.SQLITE and not self.engine_version.supports_sqlite_upsert:
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
                if _is_simple_column_expr(g_expr.strip()):
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
                    if item.source_col is not None and _is_simple_column_expr(item.raw_expr.strip()):
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
                elif item.arg_expr and _is_simple_column_expr(item.arg_expr.strip()):
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
        unique_group_nulls_not_distinct: bool = False,
    ) -> Tuple[List[str], List[str]]:
        ddls: List[str] = []
        names: List[str] = []
        p = self.object_prefix if self.object_prefix else "__matview_"
        log_vname = self.logical_view_name

        q_auto_group_cols = [self._q(c) for c in auto_group_cols]
        if q_auto_group_cols and not has_pk_on_group_cols:
            idx_name = self._bounded_ident(f"{p}idx_", log_vname, "__grp")
            if unique_group_nulls_not_distinct and self.engine_version.supports_pg_nulls_not_distinct:
                ddls.append(
                    f"CREATE UNIQUE INDEX {idx_name} ON {mv_table} ({', '.join(q_auto_group_cols)}) NULLS NOT DISTINCT;"
                )
            else:
                ddls.append(f"CREATE INDEX {idx_name} ON {mv_table} ({', '.join(q_auto_group_cols)});")
            names.append(idx_name)

        if len(auto_group_cols) > 1:
            for g_idx, g_col in enumerate(auto_group_cols[1:], start=2):
                sub_idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{_sanitize_ident(g_col)}", f"__g{g_idx}")
                if sub_idx_name not in names:
                    ddls.append(f"CREATE INDEX {sub_idx_name} ON {mv_table} ({self._q(g_col)});")
                    names.append(sub_idx_name)

        for i, idx_spec in enumerate(self.user_indexes, 1):
            if idx_spec.name:
                raw_n = idx_spec.name
                if self.object_prefix and not raw_n.startswith(self.object_prefix):
                    idx_name = f"{self.object_prefix}{raw_n}"
                else:
                    idx_name = raw_n
            else:
                col_tag = "_".join(_sanitize_ident(self._extract_index_col_base_ident(c)) for c in idx_spec.columns)
                idx_name = self._bounded_ident(f"{p}idx_", f"{log_vname}__{col_tag}", f"_{i}")

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

    def _q(self, ident: str) -> str:
        return quote_sql_identifier(ident, self.dialect)

    def _lookup_col_def(self, expr: Optional[str]) -> Optional[ColumnDef]:
        if not expr:
            return None
        s = expr.strip()
        if not _is_simple_column_expr(s):
            return None
        parts = _split_top_level(s, ".")
        if len(parts) == 2:
            prefix, col_part = parts[0], parts[1]
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

    def _sum_result_storage_type(self, item: SelectItem) -> str:
        """Preserve the engine's native SUM result type for direct numeric columns."""
        source = self._lookup_col_def(item.arg_expr) if item.arg_expr else None
        source_type = source.data_type.strip().upper() if source else ""
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
        source = self._lookup_col_def(item.arg_expr) if item.arg_expr else None
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
            query_ast = parse_sql_ast(self.view.raw_select_sql)
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

        def source_cte_name(select: Any) -> Optional[str]:
            from_clause = select.args.get("from_") if hasattr(select, "args") else None
            source = getattr(from_clause, "this", None)
            if isinstance(source, sqlglot_exp.Table):
                return source.name.lower()
            return None

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
                    upstream = expression.table.lower() if expression.table else source_cte_name(select)
                    resolved = resolve(upstream, expression.name, next_seen)
                    if resolved is not None:
                        return resolved
                return expression
            return None

        root_relation = column.table.lower() if column.table else source_cte_name(root_select)
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
            if kind in ("Paren", "Neg", "Abs", "Floor", "Ceil", "Ceiling", "Sign", "Trunc"):
                return infer(node.this)
            if kind == "Cast":
                target = node.args.get("to")
                return self._numeric_precision_scale_from_type(target.sql()) if target is not None else None
            if kind in ("Add", "Sub", "Mul", "Div"):
                left, right = infer(node.this), infer(node.expression)
                if left is None or right is None:
                    return None
                p1, s1 = left
                p2, s2 = right
                if kind in ("Add", "Sub"):
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
            return infer(parse_sql_ast(expr))
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

    def _requires_native_avg_recompute(self, item: SelectItem) -> bool:
        """Use the engine aggregate for AVG on engines with native exact semantics.

        A SUM/COUNT maintenance fold is not equivalent for every exact input:
        PostgreSQL integer division can truncate, and derived UNION projections
        can have a native result type that is lost by the fold. Recomputing the
        affected group preserves the source view's exact precision/scale and
        approximate rounding semantics.
        """
        return (
            self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
            and item.agg_func == AggFunc.AVG
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
        if self.dialect == Dialect.MYSQL:
            if u == "DOUBLE PRECISION":
                return "DOUBLE"
            if u in ("VARCHAR", "CHAR", "CHARACTER VARYING", "CHARACTER", "TEXT"):
                return "VARCHAR(191)"
        return dt

    def _infer_cte_or_subquery_col_type(self, col_name: str) -> Optional[str]:
        ident = _unquote_ident(col_name).strip()
        if not ident:
            return None
        try:
            q_ast = parse_sql_ast(self.view.raw_select_sql)
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
                    c_arg = self._lookup_col_def(inner.this.sql().strip() if isinstance(inner.this, sqlglot_exp.Expr) else None)
                    if c_arg:
                        return self._normalize_dialect_type(c_arg.data_type)
                    return self._float_type()
                if isinstance(inner, sqlglot_exp.Avg):
                    arg_expr = inner.this.sql() if inner.this is not None else ""
                    return self._avg_result_storage_type_for_expr(arg_expr)
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
                            return self._float_type() if "." in str(src_e.this) else self._int_type()
                        c_src = self._lookup_col_def(src_e.sql().strip())
                        if c_src:
                            return self._normalize_dialect_type(c_src.data_type)
            for proj in inner_query.expressions:
                c_name = proj.alias or (proj.this.name if isinstance(proj, sqlglot_exp.Alias) else proj.name)
                if c_name.lower() == ident.lower():
                    src_node = proj.this if isinstance(proj, sqlglot_exp.Alias) else proj
                    if isinstance(src_node, sqlglot_exp.Literal):
                        return self._float_type() if "." in str(src_node.this) else self._int_type()
                    src_col = src_node.name or src_node.sql().strip()
                    c_src = self._lookup_col_def(src_col)
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
        uall_map = getattr(self, "_uall_grp_alias_map", None)
        if uall_map and g_expr in uall_map:
            return uall_map[g_expr]
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
            toks = sqlglot.tokenize(expr, read="sqlite")
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
                actual_c = col_names_lower.get(c_unq, toks[i + 2].text)
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
            q_ast = parse_sql_ast(self.view.raw_select_sql)
            if q_ast.find(sqlglot_exp.Star):
                return list(ts.columns.keys())
            cols_in_ast = {c.name.lower() for c in q_ast.find_all(sqlglot_exp.Column)}
            found = [c for c in ts.columns if c.lower() in cols_in_ast]
            return found if found else list(ts.columns.keys())
        except Exception:
            return list(ts.columns.keys())

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
            ", ".join(
                (self._q(ex) if ex in user_columns else ex)
                for ex in select_exprs_override
            )
            if select_exprs_override
            else ", ".join(self._q(c) for c in user_columns)
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
                t_name = self._bounded_ident(f"{p}trg_", log_vname, f"__ro_{suf}")
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
        fn_ro = self._bounded_ident(f"{p}fn_readonly_", log_vname)
        t_ro = self._bounded_ident(f"{p}trg_", log_vname, "__readonly")
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
                q_base_t = self._q(base_t)
                q_pk_col = self._q(pk_col)
                joined_tmpl = re.sub(
                    rf"\nFROM\s+{re.escape(q_base_t)}\b",
                    f"\nFROM __mv_batches AS __mv_b JOIN {q_base_t} ON ({q_base_t}.{q_pk_col} > __mv_b.lo AND {q_base_t}.{q_pk_col} <= __mv_b.hi)",
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
                proc_name = self._bounded_ident(f"{self.object_prefix}proc_backfill_", self.logical_view_name)
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
            elif self._alg_mode == "setop":
                main_pk_cols = list(getattr(self, "_setop_pk_cols", ["__mv_pk"]))
            elif self._alg_mode == "projection":
                main_pk_cols = self._projection_pk_cols()

        sg_dial = _sqlglot_dialect(self.dialect)
        for idx_s, pop_sql in enumerate(raw_plan.initial_populate_sqls):
            clean_pop = pop_sql.strip().rstrip(";")
            try:
                pop_ast = parse_sql_ast(clean_pop, dialect=self.dialect)
            except sqlglot.errors.SqlglotError:
                pop_ast = None

            if not isinstance(pop_ast, sqlglot_exp.Insert) or not pop_ast.this:
                inc_stmts.append(pop_sql)
                continue

            target_tbl = (
                pop_ast.this.this.name
                if isinstance(pop_ast.this, sqlglot_exp.Schema)
                else (pop_ast.this.name or str(pop_ast.this))
            )
            cols = [c.name for c in pop_ast.this.expressions] if isinstance(pop_ast.this, sqlglot_exp.Schema) else []
            sel_node = pop_ast.expression
            if not isinstance(sel_node, sqlglot_exp.Select):
                inc_stmts.append(pop_sql)
                continue

            pop_toks = sqlglot.tokenize(clean_pop, read=sg_dial)
            sel_tok_pos: Optional[int] = None
            p_depth = 0
            for ptk in pop_toks:
                if ptk.token_type == SQLGlotTokenType.L_PAREN:
                    p_depth += 1
                elif ptk.token_type == SQLGlotTokenType.R_PAREN and p_depth > 0:
                    p_depth -= 1
                elif p_depth == 0 and ptk.token_type in (SQLGlotTokenType.SELECT, getattr(SQLGlotTokenType, "WITH", SQLGlotTokenType.SELECT)):
                    sel_tok_pos = ptk.start
                    break
            select_body = clean_pop[sel_tok_pos:].strip() if sel_tok_pos is not None else sel_node.sql(sg_dial).strip()

            aliased_inner = select_body
            sb_toks = sqlglot.tokenize(select_body, read=sg_dial)
            if (
                sb_toks
                and sb_toks[0].token_type == SQLGlotTokenType.SELECT
                and not sel_node.args.get("distinct")
                and not (sel_node.args.get("with_") or sel_node.args.get("with"))
                and (sel_node.args.get("from_") or sel_node.args.get("from")) is not None
            ):
                from_tok_sb = _find_top_level_token(select_body, (SQLGlotTokenType.FROM,), dialect=self.dialect)
                if from_tok_sb is not None:
                    proj_slice = select_body[sb_toks[0].end + 1 : from_tok_sb.start].strip()
                    from_and_rest = select_body[from_tok_sb.start :].strip()
                    raw_projs = [p.strip() for p in _split_top_level(proj_slice, ",") if p.strip()]
                    if len(raw_projs) == len(cols):
                        aliased_list = []
                        for c_name, r_proj in zip(cols, raw_projs):
                            rp_toks = sqlglot.tokenize(r_proj, read=sg_dial)
                            if (
                                len(rp_toks) >= 2
                                and rp_toks[-2].token_type == SQLGlotTokenType.ALIAS
                                and _unquote_ident(rp_toks[-1].text).lower() == _unquote_ident(c_name).lower()
                            ):
                                aliased_list.append(r_proj)
                            else:
                                aliased_list.append(f"{r_proj} AS {c_name}")
                        aliased_inner = f"SELECT {', '.join(aliased_list)}\n{from_and_rest}"

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
                ord_cols = ", ".join(str(k) for k in range(1, len(cols) + 1))
                ord_tail = f"\n        ORDER BY {ord_cols}" if not self.view.has_limit_or_order else f"\n        ORDER BY {ord_cols}"
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
                    f"        ) AS __mv_init{ord_tail}\n"
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
                ord_cols = ", ".join(str(k) for k in range(1, len(cols) + 1))
                ord_tail = f"\n        ORDER BY {ord_cols}"
                proc_name = self._bounded_ident(f"{self.object_prefix}proc_backfill_", self.logical_view_name, f"_{idx_s}")
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
                        f"        ) AS __mv_init{ord_tail}\n"
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
            engine_version=self.engine_version,
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
                strict=self.strict,
                engine_version=self.engine_version,
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
            engine_version=self.engine_version,
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
            if (record_prefix or self._alg_mode in ("single", "distinct"))
            else arg_e
        )
        if item.filter_where:
            rewritten_f = (
                self._rewrite_expr_for_record(item.filter_where, record_prefix)
                if (record_prefix or self._alg_mode in ("single", "distinct"))
                else item.filter_where
            )
            return f"CASE WHEN ({rewritten_f}) THEN ({rewritten_arg}) ELSE NULL END"
        return rewritten_arg

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
            return f"COALESCE(1.0 * SUM({dist}{eff}), 0.0)"
        if item.agg_func == AggFunc.AVG:
            # Preserve the argument's SQL type: MySQL AVG(exact) returns
            # DECIMAL, PostgreSQL AVG(integer/numeric) returns NUMERIC, while
            # approximate arguments remain floating-point. Multiplying by 1.0
            # would silently convert MySQL exact AVG into DOUBLE.
            return f"AVG({dist}{eff})"
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
            m2_e = self._sqlite_m2_agg_sql(item, eff_override=eff)
            return self._variance_formula_sql(
                new_cnt=cnt_e,
                new_sum=sum_e,
                new_sum2=m2_e,
                is_sample=item.agg_func in (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
                is_stddev=item.agg_func in (AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP),
            )
        if item.agg_func in (AggFunc.BIT_AND, AggFunc.BIT_OR, AggFunc.BIT_XOR):
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                if self.dialect == Dialect.MYSQL:
                    return f"(CASE WHEN COUNT({eff}) = 0 THEN NULL ELSE {item.agg_func.value}({eff}) END)"
                return f"{item.agg_func.value}({eff})"
            if item.agg_func == AggFunc.BIT_AND:
                return self._sqlite_bit_or_tree([f"MIN(({eff}) & {1 << b})" for b in range(63)])
            if item.agg_func == AggFunc.BIT_OR:
                return self._sqlite_bit_or_tree([f"MAX(({eff}) & {1 << b})" for b in range(63)])
            return self._sqlite_bit_or_tree([f"((SUM((({eff}) >> {b}) & 1) & 1) << {b})" for b in range(63)])
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
        row_count_col = self._q(count_star_alias) if count_star_alias else "__mv_row_count"

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
                col_defs_sql.append(f"{self._q(mv_col)} {dt}{nn}")

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
                col_nonnull_count_map[ekey] = self._q(item.alias)
            elif item.agg_func == AggFunc.SUM:
                has_var_for_ekey = any(
                    w.is_aggregate
                    and _expr_key(w) == ekey
                    and w.agg_func in (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP)
                    for w in work_items
                )
                if not has_var_for_ekey:
                    col_sum_map[ekey] = self._q(item.alias)
            if item.is_distinct and ekey not in distinct_aux_tables:
                p = self.object_prefix if self.object_prefix else "__matview_"
                aux_tbl = self._bounded_ident(f"{p}dist_", f"{self.logical_view_name}__{_sanitize_ident(item.alias)}")
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
                col_defs_sql.append(f"{self._q(item.alias)} {dt}{nn}")
            else:
                storage_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type}")

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
                    col_nonnull_count_map[ekey] = self._q(item.alias)
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
                    col_defs_sql.append(f"{sum_col} {self._var_accum_type(item)}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    s2_prefix = "__mv_m2_" if self.dialect == Dialect.SQLITE else "__mv_sum2_"
                    sum2_col = f"{s2_prefix}{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._var_accum_type(item, squared=True)}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        pk_cols = self._tier1_pk_cols()
        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

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
            if expr_s != u_col and "__mv_elem_" in expr_s
        }
        can_pg15_unique_nnd = (
            bool(grp_exprs)
            and has_nullable_grp
            and self._alg_mode in ("single", "distinct")
            and len(set(c.lower() for c in group_mv_cols)) == len(grp_exprs)
            and all(
                (cd := self._lookup_col_def(g)) is not None
                and any(tk in cd.data_type.upper() for tk in ("INT", "CHAR", "TEXT"))
                for g in grp_exprs
            )
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
            unique_group_nulls_not_distinct=can_pg15_unique_nnd,
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
        q_base_t = self._q(self.view.base_table)
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
                f"FROM {q_base_t}{where_sql}{group_sql};"
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
                select_exprs.append(self._rewrite_expr_for_record(item.raw_expr, None))
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
                if self.dialect == Dialect.SQLITE:
                    select_exprs.append(self._sqlite_m2_agg_sql(it, eff_override=eff))
                else:
                    select_exprs.append(f"SUM({self._var_sq_sql(eff)})")
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

        norm_where = self._rewrite_expr_for_record(self.view.where_clause, None) if self.view.where_clause else None
        where_sql = f"\nWHERE {norm_where}" if norm_where else ""
        if grp_exprs:
            norm_grps = [self._rewrite_expr_for_record(g, None) for g in grp_exprs]
            grp_for_q = self._group_by_exprs_for_query(norm_grps, include_canonical_pk=has_nullable_grp)
            group_sql = f"\nGROUP BY {', '.join(grp_for_q)}"
            having_sql = ""
        else:
            group_sql = ""
            having_sql = ""

        q_target_cols = [self._q(c) for c in target_cols]
        q_base_t = self._q(self.view.base_table)
        main_backfill_sql = (
            f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
            f"SELECT {', '.join(select_exprs)}\n"
            f"FROM {q_base_t}{where_sql}{group_sql}{having_sql};"
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
            q_pk_cols = [self._q(c) for c in pk_cols]
            if self.dialect == Dialect.MYSQL:
                chunk_upsert_sql = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {q_base_t}\n"
                    f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f"ON DUPLICATE KEY UPDATE\n"
                    f"    {', '.join(merge_assignments)};"
                )
            else:
                chunk_upsert_sql = (
                    f"INSERT INTO {mv_table} ({', '.join(q_target_cols)})\n"
                    f"SELECT {', '.join(select_exprs)}\n"
                    f"FROM {q_base_t}\n"
                    f"WHERE {chunk_where}{group_sql}{having_sql}\n"
                    f"ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
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
                f"SELECT DISTINCT {grp_expr} AS __gkey FROM {q_base_t}{w_grp} ORDER BY __gkey"
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
                    f"FROM {q_base_t}\n"
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
                f"FROM {q_base_t}\n"
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

        def _merged_sum2(ekey: str, sum2_col: str) -> str:
            if self.dialect != Dialect.SQLITE:
                return _merged_sum(sum2_col)
            cnt_c = col_nonnull_count_map[ekey]
            sum_c = col_sum_map[ekey]
            inc_cnt = _incoming(cnt_c)
            inc_sum = _incoming(sum_c)
            inc_m2 = _incoming(sum2_col)
            d_mu = f"(((1.0 * {inc_sum}) / {inc_cnt}) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_c}))"
            return (
                f"(CASE WHEN {inc_cnt} <= 0 OR {inc_m2} IS NULL THEN {mv_table}.{sum2_col} "
                f"WHEN {mv_table}.{cnt_c} <= 0 OR {mv_table}.{sum2_col} IS NULL THEN {inc_m2} "
                f"ELSE {mv_table}.{sum2_col} + {inc_m2} + ({d_mu} * {d_mu} * ((1.0 * {mv_table}.{cnt_c} * {inc_cnt}) / ({mv_table}.{cnt_c} + {inc_cnt}))) END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = self._q(item.alias)
            updated.add(c)
            ekey = expr_key_fn(item)
            inc_c = _incoming(c)
            if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                assignments.append(f"{c} = {mv_table}.{c} + {inc_c}")
            elif item.agg_func == AggFunc.SUM:
                assignments.append(f"{c} = {_merged_sum(c)}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    # The native AVG result family is not reproduced by a
                    # SUM/COUNT fold for integer inputs (PostgreSQL integer
                    # division truncates) or for engine-specific DECIMAL scale.
                    recomputed = self._build_recompute_subquery_raw(item, mv_table)
                    assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = col_nonnull_count_map[ekey]
                    sum_c = col_sum_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                    new_sum = _merged_sum(sum_c)
                    dividend = f"(1.0 * {new_sum})"
                    assignments.append(
                        f"{c} = CASE WHEN {new_cnt} = 0 THEN NULL ELSE {dividend} / {new_cnt} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                exact_pg_input = (
                    self.dialect == Dialect.POSTGRES
                    and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
                )
                if exact_pg_input:
                    recomputed = self._build_recompute_subquery_raw(item, mv_table)
                    assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = col_nonnull_count_map[ekey]
                    sum_c = col_sum_map[ekey]
                    sum2_c = col_sum2_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_c} + {_incoming(cnt_c)})"
                    new_sum = _merged_sum(sum_c)
                    new_sum2 = _merged_sum2(ekey, sum2_c)
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
        for ekey, sum2_c in col_sum2_map.items():
            if sum2_c not in updated:
                assignments.append(f"{sum2_c} = {_merged_sum2(ekey, sum2_c)}")
                updated.add(sum2_c)
        for tcnt_c in col_true_cnt_map.values():
            if tcnt_c not in updated:
                assignments.append(f"{tcnt_c} = {mv_table}.{tcnt_c} + {_incoming(tcnt_c)}")
                updated.add(tcnt_c)

        derived_aliases = {
            self._q(it.alias)
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
        def _group_record_expr(group_expr: str) -> str:
            # Chunk-fold recomputation reads the current aggregate row from the
            # backing table. A projected group expression may have a different
            # stored alias (for example COALESCE(sub_cat, 'UNK') AS sc_norm),
            # so correlate against that stored column rather than the original
            # base-table column, which does not exist in the backing table.
            if record_prefix == self._backing_table_name():
                return f"{record_prefix}.{self._q(self._group_col_alias(group_expr))}"
            return self._rewrite_expr_for_record(group_expr, record_prefix)

        if getattr(self, "_alg_mode", None) == "union_all_agg":
            inner_u = getattr(self, "_union_all_inner_sql", "")
            u_alias = getattr(self, "_union_all_sub_alias", "__mv_u")
            orig_item = getattr(self, "_uall_orig_item_map", {}).get(id(item), item)
            orig_grps = getattr(self, "_uall_orig_grp_exprs", self._effective_group_by_exprs)
            u_conds: List[str] = []
            if self.view.where_clause:
                u_conds.append(f"({self.view.where_clause})")
            for orig_g, branch_g in zip(orig_grps, self._effective_group_by_exprs):
                rhs_g = _group_record_expr(branch_g)
                u_conds.append(self._null_safe_eq(f"({orig_g})", f"({rhs_g})", is_nullable=True))
            w_u = f" WHERE {' AND '.join(u_conds)}" if u_conds else ""
            u_agg_sql = self._sql_backfill_agg_expr(orig_item)
            return f"(SELECT {u_agg_sql} FROM ({inner_u}) AS {u_alias}{w_u})"
        base_t = self.view.base_table
        q_base_t = self._q(base_t)
        conds: List[str] = []
        if self.view.where_clause:
            conds.append(f"({self.view.where_clause})")
        for g_expr in self._effective_group_by_exprs:
            lhs_g = self._rewrite_expr_for_record(g_expr, q_base_t)
            rhs_g = _group_record_expr(g_expr)
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
        return f"(SELECT {agg_sql} FROM {q_base_t}{where_part})"

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
        if self.dialect == Dialect.SQLITE:
            raw_var = f"((1.0 * ({new_sum2})) / {denom})"
        elif self.dialect == Dialect.POSTGRES:
            raw_var = (
                f"(((1.0 * ({new_sum2})::NUMERIC) - ((1.0 * ({new_sum})::NUMERIC) * (1.0 * ({new_sum})::NUMERIC)) / {new_cnt}) / {denom})::DOUBLE PRECISION"
            )
        else:
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
            mv_col = self._q(self._group_col_alias(g_expr))
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
            if self.dialect == Dialect.SQLITE:
                ekey = expr_key_fn(item)
                cnt_c = col_nonnull_count_map[ekey]
                sum_c = col_sum_map[ekey]
                delta = f"((1.0 * ({val_e})) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_c}))"
                return (
                    f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {mv_table}.{cnt_c} <= 0 OR {mv_table}.{sum_c} IS NULL THEN 0.0 "
                    f"ELSE COALESCE({mv_table}.{col_alias}, 0.0) + ({delta} * {delta} * ((1.0 * {mv_table}.{cnt_c}) / ({mv_table}.{cnt_c} + 1))) END)"
                )
            sq = self._var_sq_sql(val_e)
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
            col_alias = self._q(item.alias)
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
                if self._requires_native_avg_recompute(item):
                    recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                    update_assignments.append(f"{col_alias} = {recomputed}")
                else:
                    cnt_col = col_nonnull_count_map[ekey]
                    sum_col = col_sum_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_col} + {_plus_inc_expr(item)})"
                    new_sum = _plus_sum_expr(item, sum_col)
                    exact_avg = (
                        self.dialect in (Dialect.MYSQL, Dialect.POSTGRES)
                        and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
                    )
                    dividend = new_sum if exact_avg else f"(1.0 * {new_sum})"
                    update_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} = 0 THEN NULL "
                        f"ELSE {dividend} / {new_cnt} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                # Backfill uses the engine's native statistical aggregate.
                # A Welford/raw-moment delta update uses a different floating
                # operation order and is not bit-identical for approximate
                # inputs. Re-evaluate the affected group with the same native
                # aggregate so the materialized result matches the live view.
                recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomputed}")
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
                init_s2 = "0.0" if self.dialect == Dialect.SQLITE else self._var_sq_sql(val_e)
                insert_vals.append(
                    f"(CASE WHEN ({val_e}) IS NULL THEN NULL ELSE {init_s2} END)"
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
            self._q(it.alias)
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

        q_insert_cols = [self._q(c) for c in insert_cols]
        q_pk_cols = [self._q(c) for c in pk_cols]
        if self.dialect == Dialect.MYSQL:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(q_insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}{from_dual}\n"
                f"    WHERE TRUE\n"
                f"    ON DUPLICATE KEY UPDATE\n"
                f"        {', '.join(ordered_updates)};"
            )
        else:
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(q_insert_cols)})\n"
                f"    SELECT {', '.join(insert_vals)}\n"
                f"    WHERE TRUE\n"
                f"    ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
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
            if self.dialect == Dialect.SQLITE:
                ekey = expr_key_fn(item)
                sum_c = col_sum_map[ekey]
                delta = f"((1.0 * ({val_e})) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_col}))"
                return (
                    f"(CASE WHEN ({val_e}) IS NULL THEN {mv_table}.{col_alias} "
                    f"WHEN {new_cnt} <= 0 THEN NULL "
                    f"WHEN {new_cnt} = 1 THEN 0.0 "
                    f"ELSE MAX(0.0, COALESCE({mv_table}.{col_alias}, 0.0) - ({delta} * {delta} * ((1.0 * {mv_table}.{cnt_col}) / ({mv_table}.{cnt_col} - 1)))) END)"
                )
            sq = self._var_sq_sql(val_e)
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
            col_alias = self._q(item.alias)
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
                if self._requires_native_avg_recompute(item):
                    recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                    update_assignments.append(f"{col_alias} = {recomputed}")
                else:
                    cnt_col = col_nonnull_count_map[ekey]
                    sum_col = col_sum_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_col} - {_minus_dec_expr(item)})"
                    new_sum = _minus_sum_expr(item, sum_col, cnt_col)
                    exact_avg = (
                        self.dialect in (Dialect.MYSQL, Dialect.POSTGRES)
                        and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
                    )
                    dividend = new_sum if exact_avg else f"(1.0 * {new_sum})"
                    update_assignments.append(
                        f"{col_alias} = CASE WHEN {new_cnt} <= 0 THEN NULL "
                        f"ELSE {dividend} / {new_cnt} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                recomputed = self._build_recompute_subquery_raw(item, record_prefix)
                update_assignments.append(f"{col_alias} = {recomputed}")
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
                self._q(it.alias)
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
            {self._q(it.alias) for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
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
        if self.watermark_catalog_table and self._alg_mode in ("single", "distinct"):
            wm_pk_col: Optional[str] = "rowid" if self.dialect == Dialect.SQLITE else None
            if self.primary_schema:
                for c_def in self.primary_schema.columns.values():
                    if c_def.primary_key and "INT" in c_def.data_type.upper():
                        wm_pk_col = c_def.name
                        break
            if wm_pk_col:
                q_pk = self._q(wm_pk_col)
                wm_new = (
                    f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                    f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                    f"AND NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)"
                )
                wm_old = (
                    f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
                    f"WHERE view_name = '{self.db_view_name}' AND backfill_active = 1 "
                    f"AND OLD.{q_pk} > backfill_cursor_pk AND OLD.{q_pk} <= backfill_max_pk)"
                )
                where_new = f"({where_new}) AND ({wm_new})" if where_new else wm_new
                where_old = f"({where_old}) AND ({wm_old})" if where_old else wm_old

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
        stem = f"{log_vname}__{base_t}"
        t_ai_name = self._bounded_ident(f"{p}trg_", stem, "_ai")
        t_ad_name = self._bounded_ident(f"{p}trg_", stem, "_ad")
        t_au_name = self._bounded_ident(f"{p}trg_", stem, "_au")
        t_trunc_name = self._bounded_ident(f"{p}trg_", stem, "_trunc")
        t_sync_name = self._bounded_ident(f"{p}trg_", stem, "_sync")
        q_base_t = self._q(base_t)
        ref_cols = self._referenced_columns_for_table(base_t)
        upd_of_clause = (
            f" OF {', '.join(self._q(c) for c in ref_cols)}"
            if ref_cols and self.dialect in (Dialect.SQLITE, Dialect.POSTGRES)
            else ""
        )

        t_schema = self._schemas_lower.get(base_t.lower())
        unchanged_conds: List[str] = []
        if t_schema and ref_cols:
            for c_name in ref_cols:
                c_def = t_schema.columns.get(c_name)
                is_null = not (c_def.not_null if c_def else False)
                qc = self._q(c_name)
                unchanged_conds.append(
                    self._null_safe_eq(f"OLD.{qc}", f"NEW.{qc}", is_nullable=is_null)
                )
        changed_guard = f"NOT ({' AND '.join(unchanged_conds)})" if unchanged_conds else None

        cleanup_stmts = upd_cleanup_old_stmts or []

        if self.dialect == Dialect.SQLITE:
            when_ins = f"\nWHEN {where_new}" if where_new else ""
            ins_body = "\n    ".join(ins_stmts)
            trig_ins = (
                f"CREATE TRIGGER {t_ai_name}\n"
                f"AFTER INSERT ON {q_base_t}\n"
                f"FOR EACH ROW{when_ins}\n"
                f"BEGIN\n"
                f"    {ins_body}\n"
                f"END;"
            )

            when_del = f"\nWHEN {where_old}" if where_old else ""
            del_body = "\n    ".join(del_stmts)
            trig_del = (
                f"CREATE TRIGGER {t_ad_name}\n"
                f"AFTER DELETE ON {q_base_t}\n"
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
                f"CREATE TRIGGER {t_au_name}\n"
                f"AFTER UPDATE{upd_of_clause} ON {q_base_t}\n"
                f"FOR EACH ROW{when_upd}\n"
                f"BEGIN\n"
                f"    {upd_body}\n"
                f"END;"
            )
            t_names = [t_ai_name, t_ad_name, t_au_name]
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

            fn_trunc = self._bounded_ident(f"{p}fn_sync_", stem, "_trunc")
            fn_trunc_ddl = (
                f"CREATE OR REPLACE FUNCTION {fn_trunc}()\n"
                f"RETURNS TRIGGER AS $$\n"
                f"BEGIN\n"
                f"    {trunc_block}\n"
                f"    RETURN NULL;\n"
                f"END;\n$$ LANGUAGE plpgsql;"
            )
            t_trunc = (
                f"CREATE TRIGGER {t_trunc_name}\n"
                f"AFTER TRUNCATE ON {q_base_t}\n"
                f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_trunc}();"
            )

            if self.pg_use_transition_tables:
                fn_ai = self._bounded_ident(f"{p}fn_sync_", stem, "_ai")
                fn_ad = self._bounded_ident(f"{p}fn_sync_", stem, "_ad")
                fn_au = self._bounded_ident(f"{p}fn_sync_", stem, "_au")
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
                    f"CREATE TRIGGER {t_ai_name}\n"
                    f"AFTER INSERT ON {q_base_t}\n"
                    f"REFERENCING NEW TABLE AS __mv_new_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_ai}();"
                )
                t_ad = (
                    f"CREATE TRIGGER {t_ad_name}\n"
                    f"AFTER DELETE ON {q_base_t}\n"
                    f"REFERENCING OLD TABLE AS __mv_old_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_ad}();"
                )
                t_au = (
                    f"CREATE TRIGGER {t_au_name}\n"
                    f"AFTER UPDATE ON {q_base_t}\n"
                    f"REFERENCING OLD TABLE AS __mv_old_rows NEW TABLE AS __mv_new_rows\n"
                    f"FOR EACH STATEMENT EXECUTE FUNCTION {fn_au}();"
                )
                return (
                    [fn_ai_ddl, t_ai, fn_ad_ddl, t_ad, fn_au_ddl, t_au, fn_trunc_ddl, t_trunc],
                    [t_ai_name, t_ad_name, t_au_name, t_trunc_name],
                    [fn_ai, fn_ad, fn_au, fn_trunc],
                )

            fn_name = self._bounded_ident(f"{p}fn_sync_", stem)
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
                f"    IF COALESCE(current_setting('matview.skip_triggers', true), '') = 'on' THEN\n"
                f"        RETURN NULL;\n"
                f"    END IF;\n"
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
                f"CREATE TRIGGER {t_sync_name}\n"
                f"AFTER INSERT OR UPDATE{upd_of_clause} OR DELETE ON {q_base_t}\n"
                f"FOR EACH ROW EXECUTE FUNCTION {fn_name}();"
            )
            return (
                [fn_ddl, t_sync, fn_trunc_ddl, t_trunc],
                [t_sync_name, t_trunc_name],
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
                f"CREATE TRIGGER {t_ai_name}\n"
                f"AFTER INSERT ON {q_base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    IF COALESCE(@__matview_skip_triggers, 0) = 0 THEN\n"
                f"    {ins_section}\n"
                f"    END IF;\n"
                f"END;"
            )
            trig_del = (
                f"CREATE TRIGGER {t_ad_name}\n"
                f"AFTER DELETE ON {q_base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    IF COALESCE(@__matview_skip_triggers, 0) = 0 THEN\n"
                f"    {del_section}\n"
                f"    END IF;\n"
                f"END;"
            )
            trig_upd = (
                f"CREATE TRIGGER {t_au_name}\n"
                f"AFTER UPDATE ON {q_base_t}\n"
                f"FOR EACH ROW\n"
                f"BEGIN\n"
                f"    IF COALESCE(@__matview_skip_triggers, 0) = 0 THEN\n"
                f"    {upd_guard_open}{upd_inner}{upd_guard_close}\n"
                f"    END IF;\n"
                f"END;"
            )
            t_names = [t_ai_name, t_ad_name, t_au_name]
            return [trig_ins, trig_del, trig_upd], t_names, []

        return [], [], []


    # =========================================================================
    # TIER 1B: Multi-Table INNER / CROSS JOIN O(1) Algebraic Delta Maintenance
    # =========================================================================

    def _qualify_join_expr(self, expr: Optional[str]) -> str:
        if not expr:
            return ""
        col_to_aliases: Dict[str, List[Tuple[str, str]]] = {}
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            if ts:
                for c_name in ts.columns:
                    col_to_aliases.setdefault(c_name.lower(), []).append((j.alias, c_name))

        alias_remap: Dict[str, Tuple[str, Optional[TableSchema]]] = {}
        for j in self.view.joins:
            ts = self._schemas_lower.get(j.table_name.lower())
            alias_remap[j.table_name.lower()] = (j.alias, ts)
            alias_remap[j.alias.lower()] = (j.alias, ts)

        try:
            toks = sqlglot.tokenize(expr, read="sqlite")
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
                q_low = tk.text.lower()
                c_unq = toks[i + 2].text
                if q_low in alias_remap:
                    new_al, ts = alias_remap[q_low]
                    actual_c = c_unq
                    if ts:
                        for sc in ts.columns:
                            if sc.lower() == c_unq.lower():
                                actual_c = sc
                                break
                    if tk.start > cursor:
                        out.append(expr[cursor:tk.start])
                    out.append(f"{self._q(new_al)}.{self._q(actual_c)}")
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
                    c_low = tk.text.lower()
                    als = col_to_aliases.get(c_low)
                    if als and (
                        len(als) == 1
                        or any(
                            j.on_condition and c_low in {t.text.lower() for t in sqlglot.tokenize(j.on_condition, read="sqlite")}
                            for j in self.view.joins
                        )
                    ):
                        al, actual_c = als[0]
                        if tk.start > cursor:
                            out.append(expr[cursor:tk.start])
                        out.append(f"{self._q(al)}.{self._q(actual_c)}")
                        cursor = tk.end + 1
                        i += 1
                        continue
            i += 1
        if cursor < len(expr):
            out.append(expr[cursor:])
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
        row_count_col = self._q(count_star_alias) if count_star_alias else "__mv_row_count"

        col_defs_sql: List[str] = []
        user_columns: List[str] = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"

        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()
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
                c_def = self._lookup_col_def(self._qualify_join_expr(g_expr)) or self._lookup_col_def(g_expr)
                if c_def:
                    dt = self._normalize_dialect_type(c_def.data_type)
                else:
                    fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                    dt = self._infer_expr_storage_type(fake_it)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if not has_nullable_grp else ""
                col_defs_sql.append(f"{self._q(mv_col)} {dt}{nn}")

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
                col_nonnull_count_map[ekey] = self._q(item.alias)
            elif item.agg_func == AggFunc.SUM:
                has_var_for_ekey = any(
                    w.is_aggregate
                    and _expr_key(w) == ekey
                    and w.agg_func in (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP)
                    for w in work_items
                )
                if not has_var_for_ekey:
                    col_sum_map[ekey] = self._q(item.alias)

        group_mv_cols: List[str] = []
        for g in grp_exprs:
            mc = self._group_col_alias(g)
            if mc not in group_mv_cols:
                group_mv_cols.append(mc)
        indexed_cols_lower = self._indexed_column_names_lower(group_mv_cols)

        for item in work_items:
            if not item.is_aggregate:
                c_def = self._lookup_col_def(self._qualify_join_expr(item.raw_expr)) or self._lookup_col_def(item.raw_expr)
                dt = (
                    self._normalize_dialect_type(c_def.data_type)
                    if c_def
                    else self._infer_expr_storage_type(item)
                )
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if (grp_exprs and not has_nullable_grp) else ""
                col_defs_sql.append(f"{self._q(item.alias)} {dt}{nn}")
            else:
                storage_type = self._infer_expr_storage_type(item)
                if self.dialect == Dialect.MYSQL and storage_type.upper() == "TEXT" and item.alias.lower() in indexed_cols_lower:
                    storage_type = "VARCHAR(255)"
                if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type}")

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
                    col_nonnull_count_map[ekey] = self._q(item.alias)
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
                    col_defs_sql.append(f"{sum_col} {self._var_accum_type(item)}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    s2_prefix = "__mv_m2_" if self.dialect == Dialect.SQLITE else "__mv_sum2_"
                    sum2_col = f"{s2_prefix}{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._var_accum_type(item, squared=True)}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

        cols_joined = ",\n    ".join(col_defs_sql)
        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    {cols_joined}\n);"

        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col and "__mv_elem_" in expr_s
        }
        can_pg15_unique_nnd = (
            bool(grp_exprs)
            and has_nullable_grp
            and len(set(c.lower() for c in group_mv_cols)) == len(grp_exprs)
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
            unique_group_nulls_not_distinct=can_pg15_unique_nnd,
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
        target_cols: List[str] = []
        select_exprs: List[str] = []
        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(qual_grp_exprs))

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
                if self.dialect == Dialect.SQLITE:
                    select_exprs.append(self._sqlite_m2_agg_sql(it, eff_override=eff))
                else:
                    select_exprs.append(f"SUM({self._var_sq_sql(eff)})")
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
        qual_grp_for_q = self._group_by_exprs_for_query(qual_grp_exprs, include_canonical_pk=has_nullable_grp)
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
        # Native approximate aggregates must preserve the engine's operation
        # order; algebraically equivalent sum/count updates can differ by ULPs.
        native_recompute_aliases = {
            self._q(item.alias)
            for item in work_items
            if item.is_aggregate
            and (
                item.agg_func in (
                    AggFunc.VAR_POP, AggFunc.VAR_SAMP,
                    AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP,
                )
                or self._requires_native_avg_recompute(item)
            )
        }
        if native_recompute_aliases:
            plus_merge_assignments = [
                assignment for assignment in plus_merge_assignments
                if assignment.split("=", 1)[0].strip() not in native_recompute_aliases
            ]
            for item in work_items:
                if item.is_aggregate and (
                    item.agg_func in (
                        AggFunc.VAR_POP, AggFunc.VAR_SAMP,
                        AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP,
                    )
                    or self._requires_native_avg_recompute(item)
                ):
                    plus_merge_assignments.append(
                        f"{self._q(item.alias)} = {_build_join_recompute_subquery(item)}"
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

        def _delta_sum2_minus(ekey: str, sum2_col: str, cnt_col: str) -> str:
            if self.dialect != Dialect.SQLITE:
                return _delta_sum_minus(sum2_col, cnt_col)
            sum_c = col_sum_map[ekey]
            new_c = f"({mv_table}.{cnt_col} - __mv_delta.{cnt_col})"
            d_mu = f"(((1.0 * __mv_delta.{sum_c}) / __mv_delta.{cnt_col}) - ((1.0 * {mv_table}.{sum_c}) / {mv_table}.{cnt_col}))"
            return (
                f"(CASE WHEN __mv_delta.{cnt_col} <= 0 OR __mv_delta.{sum2_col} IS NULL THEN {mv_table}.{sum2_col} "
                f"WHEN {new_c} <= 0 THEN NULL "
                f"WHEN {new_c} = 1 THEN 0.0 "
                f"ELSE MAX(0.0, COALESCE({mv_table}.{sum2_col}, 0.0) - COALESCE(__mv_delta.{sum2_col}, 0.0) - ({d_mu} * {d_mu} * ((1.0 * {mv_table}.{cnt_col} * __mv_delta.{cnt_col}) / {new_c}))) END)"
            )

        for item in work_items:
            if not item.is_aggregate:
                continue
            c = self._q(item.alias)
            minus_updated.add(c)
            ekey = _expr_key(item)
            if item.agg_func in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL, AggFunc.TOTAL):
                minus_assignments.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")
            elif item.agg_func == AggFunc.SUM:
                cnt_c = col_nonnull_count_map[ekey]
                minus_assignments.append(f"{c} = {_delta_sum_minus(c, cnt_c)}")
            elif item.agg_func == AggFunc.AVG:
                if self._requires_native_avg_recompute(item):
                    recomputed = _build_join_recompute_subquery(item)
                    minus_assignments.append(f"{c} = {recomputed}")
                else:
                    cnt_c = col_nonnull_count_map[ekey]
                    sum_c = col_sum_map[ekey]
                    new_cnt = f"({mv_table}.{cnt_c} - __mv_delta.{cnt_c})"
                    new_sum = _delta_sum_minus(sum_c, cnt_c)
                    exact_avg = (
                        self.dialect in (Dialect.MYSQL, Dialect.POSTGRES)
                        and self._exact_numeric_precision_scale(item.arg_expr or "") is not None
                    )
                    dividend = new_sum if exact_avg else f"(1.0 * {new_sum})"
                    minus_assignments.append(
                        f"{c} = CASE WHEN {new_cnt} <= 0 THEN NULL ELSE {dividend} / {new_cnt} END"
                    )
            elif item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                recomputed = _build_join_recompute_subquery(item)
                minus_assignments.append(f"{c} = {recomputed}")
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
                minus_assignments.append(f"{sum2_c} = {_delta_sum2_minus(ekey, sum2_c, cnt_c)}")
                minus_updated.add(sum2_c)
        for ekey, tcnt_c in col_true_cnt_map.items():
            if tcnt_c not in minus_updated:
                minus_assignments.append(f"{tcnt_c} = {mv_table}.{tcnt_c} - __mv_delta.{tcnt_c}")
                minus_updated.add(tcnt_c)

        high_p = {
            self._q(it.alias)
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
            {self._q(it.alias) for it in work_items if it.is_aggregate and it.agg_func == AggFunc.SUM}
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

            delta_pk_match = " AND ".join(
                f"{mv_table}.{self._q(c)} = __mv_delta.{self._q(c)}" for c in pk_cols
            )
            q_pk_cols_csv = ", ".join(self._q(c) for c in pk_cols)
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
                    f"    ) AS __mv_delta ON {delta_pk_match}\n"
                    f"    SET {', '.join(mysql_ordered_minus)};"
                )
            else:
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(target_cols)})\n"
                    f"    SELECT {', '.join(select_exprs)}\n"
                    f"    FROM {from_new}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ON CONFLICT ({q_pk_cols_csv}) DO UPDATE SET\n"
                    f"        {', '.join(plus_merge_assignments)};"
                )
                minus_stmt = (
                    f"UPDATE {mv_table}\n"
                    f"    SET {', '.join(ordered_minus)}\n"
                    f"    FROM (\n"
                    f"        SELECT {aliased_delta_select}\n"
                    f"        FROM {from_old}{where_sql}{group_sql}{having_non_empty}\n"
                    f"    ) AS __mv_delta\n"
                    f"    WHERE {delta_pk_match};"
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
        try:
            q_ast = parse_sql_ast(raw)
        except sqlglot.errors.SqlglotError:
            return None
        if not isinstance(q_ast, sqlglot_exp.SetOperation):
            return None

        ops, branch_sqls = _extract_set_op_branches_from_ast(q_ast, raw)
        if not ops or len(set(ops)) != 1:
            return None
        op_type = ops[0]

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
            if bs.where_clause and _has_subquery_ast(bs.where_clause):
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

    def _rewrite_branch_expr(self, expr: str, ts: TableSchema, alias: str, record_prefix: Optional[str]) -> str:
        col_map = {c.lower(): c for c in ts.columns}
        quals_lower = {q.lower() for q in (ts.name, alias) if q}
        toks = tokenize_sql(expr)
        out: List[str] = []
        i = 0
        n = len(toks)

        def _next_non_ws(idx: int) -> Optional[SQLToken]:
            k = idx
            while k < n:
                if toks[k].kind != SQLTokenType.WHITESPACE:
                    return toks[k]
                k += 1
            return None

        def _prev_non_ws(idx: int) -> Optional[SQLToken]:
            k = idx
            while k >= 0:
                if toks[k].kind != SQLTokenType.WHITESPACE:
                    return toks[k]
                k -= 1
            return None

        while i < n:
            t = toks[i]
            if (
                t.kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT)
                and i + 2 < n
                and toks[i + 1].kind == SQLTokenType.PUNCT
                and toks[i + 1].text == "."
                and toks[i + 2].kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT)
            ):
                q_low = t.unquoted.lower()
                c_low = toks[i + 2].unquoted.lower()
                actual_c = col_map.get(c_low, toks[i + 2].unquoted)
                if q_low in quals_lower:
                    if record_prefix:
                        out.append(f"{record_prefix}.{self._q(actual_c)}")
                    else:
                        out.append(f"{self._q(t.unquoted)}.{self._q(actual_c)}")
                    i += 3
                    continue
            if t.kind in (SQLTokenType.IDENT, SQLTokenType.QUOTED_IDENT):
                prev_t = _prev_non_ws(i - 1)
                next_t = _next_non_ws(i + 1)
                is_after_dot = prev_t is not None and prev_t.kind == SQLTokenType.PUNCT and prev_t.text == "."
                is_before_dot = next_t is not None and next_t.kind == SQLTokenType.PUNCT and next_t.text == "."
                is_fn_call = (
                    t.kind == SQLTokenType.IDENT
                    and next_t is not None
                    and next_t.kind == SQLTokenType.PUNCT
                    and next_t.text == "("
                )
                if not is_after_dot and not is_before_dot and not is_fn_call:
                    w_low = t.unquoted.lower()
                    if w_low in col_map:
                        actual_c = col_map[w_low]
                        if record_prefix:
                            out.append(f"{record_prefix}.{self._q(actual_c)}")
                        else:
                            out.append(self._q(actual_c))
                        i += 1
                        continue
            out.append(t.text)
            i += 1
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

        # Check if UNION ALL branches share a uniform native composite PK structure
        branch_pk_defs_list: List[List[ColumnDef]] = []
        for _, bs in branches:
            ts_b = self._schemas_lower[bs.base_table.lower()]
            branch_pk_defs_list.append([c for c in ts_b.columns.values() if c.primary_key])

        use_native_uall_pk = False
        uall_pk_cols: List[str] = ["__mv_pk"]
        uall_pk_col_types: List[str] = []
        if op_type == "UNION ALL" and branch_pk_defs_list and branch_pk_defs_list[0]:
            k_pks = len(branch_pk_defs_list[0])
            if all(len(pks) == k_pks for pks in branch_pk_defs_list):
                use_native_uall_pk = True
                uall_pk_cols = ["__mv_branch_id", *[f"__mv_pk_{i}" for i in range(k_pks)]]
                for i in range(k_pks):
                    dt_i = self._normalize_dialect_type(branch_pk_defs_list[0][i].data_type)
                    if self.dialect == Dialect.MYSQL and dt_i.upper() == "TEXT":
                        dt_i = "VARCHAR(255)"
                    uall_pk_col_types.append(dt_i)
        self._setop_pk_cols = list(uall_pk_cols)

        col_defs_sql: List[str] = []
        if use_native_uall_pk:
            col_defs_sql.append("__mv_branch_id INTEGER NOT NULL")
            for i, dt_i in enumerate(uall_pk_col_types):
                col_defs_sql.append(f"__mv_pk_{i} {dt_i} NOT NULL")
        else:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        indexed_cols_lower = self._indexed_column_names_lower()
        for it in first_bs.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            col_defs_sql.append(f"{self._q(it.alias)} {dt}")

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

        if use_native_uall_pk:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(uall_pk_cols)})")

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
                sel_exprs = [it.raw_expr for it in bs.select_items]
                w_sql = f" WHERE {bs.where_clause}" if bs.where_clause else ""
                if use_native_uall_pk:
                    initial_populate_sqls.append(
                        f"INSERT INTO {mv_table} ({', '.join(uall_pk_cols)}, {', '.join(user_columns)}, __mv_row_count)\n"
                        f"SELECT {b_idx}, {', '.join(pk_names)}, {', '.join(sel_exprs)}, 1 FROM {bs.base_table}{w_sql};"
                    )
                else:
                    pk_expr = self._canonical_group_pk_expr([f"'B{b_idx}'", *pk_names])
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
                    if use_native_uall_pk:
                        new_pk_vals = [str(b_idx), *[f"NEW.{self._q(c)}" for c in pk_names]]
                        old_pk_match = f"{mv_table}.__mv_branch_id = {b_idx} AND " + " AND ".join(
                            f"{mv_table}.__mv_pk_{i} = OLD.{self._q(c)}" for i, c in enumerate(pk_names)
                        )
                        if self.dialect == Dialect.MYSQL:
                            upd_set = ", ".join(f"{c} = VALUES({c})" for c in user_columns)
                            plus_s = (
                                f"INSERT INTO {mv_table} ({', '.join(uall_pk_cols)}, {', '.join(user_columns)}, __mv_row_count)\n"
                                f"    SELECT {', '.join(new_pk_vals)}, {', '.join(new_vals)}, 1{from_dual}\n"
                                f"    WHERE {w_new}\n"
                                f"    ON DUPLICATE KEY UPDATE {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                            )
                        else:
                            upd_set = ", ".join(f"{c} = excluded.{c}" for c in user_columns)
                            plus_s = (
                                f"INSERT INTO {mv_table} ({', '.join(uall_pk_cols)}, {', '.join(user_columns)}, __mv_row_count)\n"
                                f"    SELECT {', '.join(new_pk_vals)}, {', '.join(new_vals)}, 1\n"
                                f"    WHERE {w_new}\n"
                                f"    ON CONFLICT ({', '.join(uall_pk_cols)}) DO UPDATE SET {upd_set}, __mv_row_count = {mv_table}.__mv_row_count + 1;"
                            )
                        minus_s = (
                            f"UPDATE {mv_table} SET __mv_row_count = {mv_table}.__mv_row_count - 1\n"
                            f"    WHERE ({w_old}) AND ({old_pk_match});"
                        )
                        clean_s = (
                            f"DELETE FROM {mv_table}\n"
                            f"    WHERE ({w_old}) AND ({old_pk_match}) AND {mv_table}.__mv_row_count <= 0;"
                        )
                    else:
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

    def _projection_pk_mapping(self) -> Tuple[List[str], List[Tuple[str, str, str]], bool]:
        """
        Returns `(mv_pk_cols, hidden_pk_defs, uses_native_pk)` for `_compile_tier1_zset_projection`.
        Preserves the base table's actual primary/composite key columns in the backing table
        whenever all base PK columns are non-nullable.
        """
        assert self.primary_schema is not None
        pk_defs = [c for c in self.primary_schema.columns.values() if c.primary_key]
        can_native = bool(pk_defs) and all(
            c.not_null
            or "INT" in c.data_type.upper()
            or self.dialect in (Dialect.POSTGRES, Dialect.MYSQL)
            for c in pk_defs
        )
        if not can_native:
            return ["__mv_pk"], [], False

        base_alias = self.view.joins[0].alias.lower() if self.view.joins else self.view.base_table.lower()
        base_name = self.view.base_table.lower()
        mv_pk_cols: List[str] = []
        hidden_pk_defs: List[Tuple[str, str, str]] = []
        used_user_aliases: Set[str] = set()

        for c_def in pk_defs:
            c_low = c_def.name.lower()
            matched_alias: Optional[str] = None
            for it in self.view.select_items:
                raw_s = _unquote_ident(it.raw_expr.strip()).lower() if re.match(r'^[A-Za-z0-9_."`\[\]]+$', it.raw_expr.strip()) else ""
                raw_full = it.raw_expr.strip().lower()
                if (
                    raw_full == c_low
                    or raw_full == f"{base_alias}.{c_low}"
                    or raw_full == f"{base_name}.{c_low}"
                    or (raw_s == c_low and "." not in raw_full)
                ) and it.alias.lower() not in used_user_aliases:
                    matched_alias = it.alias
                    used_user_aliases.add(it.alias.lower())
                    break
            if matched_alias is not None:
                mv_pk_cols.append(matched_alias)
            else:
                hid_col = f"__mv_pk_{_sanitize_ident(c_def.name)}"
                dt = self._normalize_dialect_type(c_def.data_type)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                hidden_pk_defs.append((hid_col, c_def.name, dt))
                mv_pk_cols.append(hid_col)

        return mv_pk_cols, hidden_pk_defs, True

    def _projection_pk_cols(self) -> List[str]:
        mv_pk_cols, _, _ = self._projection_pk_mapping()
        return mv_pk_cols

    def _compile_tier1_zset_projection(self) -> MaterializedViewPlan:
        assert self.primary_schema is not None
        view_name = self.view.view_name
        mv_table = self._backing_table_name()
        base_t = self.view.base_table
        q_base_t = self._q(base_t)
        user_columns = [it.alias for it in self.view.select_items]
        pk_type = "VARCHAR(512)" if self.dialect == Dialect.MYSQL else "TEXT"
        from_dual = " FROM DUAL" if self.dialect == Dialect.MYSQL else ""

        pk_names = [c.name for c in self.primary_schema.columns.values() if c.primary_key]
        mv_pk_cols, hidden_pk_defs, uses_native_pk = self._projection_pk_mapping()
        pk_col_set_lower = {c.lower() for c in mv_pk_cols}

        col_defs_sql: List[str] = []
        if not uses_native_pk:
            col_defs_sql.append(f"__mv_pk {pk_type} NOT NULL PRIMARY KEY")

        indexed_cols_lower = self._indexed_column_names_lower(mv_pk_cols if uses_native_pk else ())
        for it in self.view.select_items:
            dt = self._infer_expr_storage_type(it)
            if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT" and it.alias.lower() in indexed_cols_lower:
                dt = "VARCHAR(255)"
            nn = " NOT NULL" if (uses_native_pk and it.alias.lower() in pk_col_set_lower) else ""
            col_defs_sql.append(f"{it.alias} {dt}{nn}")

        for hid_col, _, hid_dt in hidden_pk_defs:
            col_defs_sql.append(f"{hid_col} {hid_dt} NOT NULL")

        if uses_native_pk:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(mv_pk_cols)})")
        else:
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

        sel_exprs = [it.raw_expr for it in self.view.select_items]
        w_sql = f"\nWHERE {self.view.where_clause}" if self.view.where_clause else ""
        new_vals = [self._rewrite_expr_for_record(it.raw_expr, "NEW") for it in self.view.select_items]

        if uses_native_pk:
            all_backing_cols = [*user_columns, *[h[0] for h in hidden_pk_defs]]
            all_select_exprs = [*sel_exprs, *[h[1] for h in hidden_pk_defs]]
            all_new_vals = [*new_vals, *[f"NEW.{h[1]}" for h in hidden_pk_defs]]
            initial_populate_sql = (
                f"INSERT INTO {mv_table} ({', '.join(all_backing_cols)})\n"
                f"SELECT {', '.join(all_select_exprs)} FROM {q_base_t}{w_sql};"
            )
            if self.dialect == Dialect.MYSQL:
                upd_set = ", ".join(f"{c} = VALUES({c})" for c in all_backing_cols)
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(all_backing_cols)})\n"
                    f"    SELECT {', '.join(all_new_vals)}{from_dual}\n"
                    f"    WHERE TRUE\n"
                    f"    ON DUPLICATE KEY UPDATE {upd_set};"
                )
            else:
                upd_set = ", ".join(f"{c} = excluded.{c}" for c in all_backing_cols)
                plus_stmt = (
                    f"INSERT INTO {mv_table} ({', '.join(all_backing_cols)})\n"
                    f"    SELECT {', '.join(all_new_vals)}\n"
                    f"    WHERE TRUE\n"
                    f"    ON CONFLICT ({', '.join(mv_pk_cols)}) DO UPDATE SET {upd_set};"
                )
            pk_old_preds = [
                f"{mv_table}.{mv_c} = OLD.{src_c}"
                for mv_c, src_c in zip(mv_pk_cols, pk_names)
            ]
            del_where = " AND ".join(pk_old_preds)
            del_stmt = f"DELETE FROM {mv_table}\n    WHERE {del_where};"
            del_stmts_list = [del_stmt]
            upd_old_list = [del_stmt]
            upd_clean_list: List[str] = []
        else:
            pk_expr_base = self._canonical_group_pk_expr(pk_names)
            initial_populate_sql = (
                f"INSERT INTO {mv_table} (__mv_pk, {', '.join(user_columns)}, __mv_row_count)\n"
                f"SELECT {pk_expr_base}, {', '.join(sel_exprs)}, 1 FROM {q_base_t}{w_sql};"
            )
            pk_new = self._canonical_group_pk_expr([f"NEW.{c}" for c in pk_names])
            pk_old = self._canonical_group_pk_expr([f"OLD.{c}" for c in pk_names])
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
            del_stmts_list = [minus_stmt, clean_stmt]
            upd_old_list = [minus_stmt]
            upd_clean_list = [clean_stmt]

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
            del_stmts=del_stmts_list,
            upd_old_stmts=upd_old_list,
            upd_new_stmts=[plus_stmt],
            upd_cleanup_old_stmts=upd_clean_list,
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
        row_count_col = self._q(count_star_alias) if count_star_alias else "__mv_row_count"

        # Check if all group_by columns are NOT NULL across every UNION ALL branch
        uall_has_nullable = False
        if grp_exprs:
            first_b_aliases = [it.alias.lower() for it in branch_specs[0].select_items]
            seen_g_al: Set[str] = set()
            for g_expr in grp_exprs:
                g_id = _unquote_ident(g_expr.split(".")[-1]).lower() if _is_simple_column_expr(g_expr.strip()) else ""
                if not g_id or g_id not in first_b_aliases:
                    uall_has_nullable = True
                    break
                col_pos = first_b_aliases.index(g_id)
                for bs in branch_specs:
                    b_it = bs.select_items[col_pos]
                    ts_b = self._schemas_lower[bs.base_table.lower()]
                    c_b = ts_b.columns.get(b_it.source_col or "") if b_it.source_col else None
                    if c_b is None or not c_b.not_null:
                        uall_has_nullable = True
                        break
                if uall_has_nullable:
                    break
                al_g = self._group_col_alias(g_expr).lower()
                if al_g in seen_g_al:
                    uall_has_nullable = True
                    break
                seen_g_al.add(al_g)
        self._uall_has_nullable_grp = uall_has_nullable
        has_nullable_grp = self._has_nullable_group_col()
        pk_cols = self._tier1_pk_cols()

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
                fake_it = SelectItem(raw_expr=g_expr, alias=mv_col, is_aggregate=False)
                dt = self._infer_expr_storage_type(fake_it)
                if self.dialect == Dialect.MYSQL and dt.upper() == "TEXT":
                    dt = "VARCHAR(255)"
                nn = " NOT NULL" if not has_nullable_grp else ""
                col_defs_sql.append(f"{self._q(mv_col)} {dt}{nn}")

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
                col_nonnull_count_map[ekey] = self._q(item.alias)
            elif item.agg_func == AggFunc.SUM:
                has_var_for_ekey = any(
                    w.is_aggregate
                    and _expr_key(w) == ekey
                    and w.agg_func in (AggFunc.VAR_POP, AggFunc.VAR_SAMP, AggFunc.STDDEV_POP, AggFunc.STDDEV_SAMP)
                    for w in work_items
                )
                if not has_var_for_ekey:
                    col_sum_map[ekey] = self._q(item.alias)

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
                nn = " NOT NULL" if (grp_exprs and not has_nullable_grp) else ""
                col_defs_sql.append(f"{self._q(item.alias)} {dt}{nn}")
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
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0")
                elif item.agg_func == AggFunc.TOTAL:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type} NOT NULL DEFAULT 0.0")
                else:
                    col_defs_sql.append(f"{self._q(item.alias)} {storage_type}")

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
                    col_nonnull_count_map[ekey] = self._q(item.alias)
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
                    col_defs_sql.append(f"{sum_col} {self._var_accum_type(item)}")

            if item.agg_func in (
                AggFunc.VAR_POP,
                AggFunc.VAR_SAMP,
                AggFunc.STDDEV_POP,
                AggFunc.STDDEV_SAMP,
            ):
                if ekey not in col_sum2_map:
                    aux_idx += 1
                    s2_prefix = "__mv_m2_" if self.dialect == Dialect.SQLITE else "__mv_sum2_"
                    sum2_col = f"{s2_prefix}{safe_id}_{aux_idx}"
                    col_sum2_map[ekey] = sum2_col
                    col_defs_sql.append(f"{sum2_col} {self._var_accum_type(item, squared=True)}")

            if item.agg_func in (AggFunc.BOOL_AND, AggFunc.BOOL_OR):
                if ekey not in col_true_cnt_map:
                    aux_idx += 1
                    tcnt_col = f"__mv_tcnt_{safe_id}_{aux_idx}"
                    col_true_cnt_map[ekey] = tcnt_col
                    col_defs_sql.append(f"{tcnt_col} {self._int_type()} NOT NULL DEFAULT 0")

        if grp_exprs and not has_nullable_grp:
            col_defs_sql.append(f"PRIMARY KEY ({', '.join(self._q(c) for c in pk_cols)})")

        backing_table_ddl = f"CREATE TABLE {mv_table} (\n    " + ",\n    ".join(col_defs_sql) + "\n);"
        backing_col_map = {
            u_col: re.sub(r"\s+AS\s+[A-Za-z0-9_\"`\[\]]+\s*$", "", expr_s, flags=re.IGNORECASE)
            for u_col, expr_s in self._compound_reader_exprs.items()
            if expr_s != u_col and "__mv_elem_" in expr_s
        }
        can_pg15_unique_nnd = (
            bool(grp_exprs)
            and has_nullable_grp
            and len(set(c.lower() for c in group_mv_cols)) == len(grp_exprs)
        )
        index_ddls, index_names = self._build_all_index_ddls(
            mv_table=mv_table,
            auto_group_cols=group_mv_cols,
            has_pk_on_group_cols=(bool(grp_exprs) and not has_nullable_grp),
            backing_col_map=backing_col_map,
            unique_group_nulls_not_distinct=can_pg15_unique_nnd,
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

        target_cols: List[str] = []
        select_exprs: List[str] = []
        if not grp_exprs:
            target_cols.append("__mv_global_id")
            select_exprs.append("1")
        elif has_nullable_grp:
            target_cols.append("__mv_pk")
            select_exprs.append(self._canonical_group_pk_expr(grp_exprs))

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
                if self.dialect == Dialect.SQLITE:
                    select_exprs.append(self._sqlite_m2_agg_sql(it, eff_override=eff))
                else:
                    select_exprs.append(f"SUM({self._var_sq_sql(eff)})")
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
        grp_for_q = self._group_by_exprs_for_query(grp_exprs, include_canonical_pk=has_nullable_grp)
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
        orig_grp_aliases = [self._group_col_alias(g) for g in orig_grp_exprs]
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
                    toks = sqlglot.tokenize(frag, read="sqlite")
                    if not toks:
                        return frag
                    out = []
                    cursor = 0
                    i = 0
                    n = len(toks)
                    while i < n:
                        tk = toks[i]
                        if (
                            _is_word_or_ident_token(tk)
                            and i + 2 < n
                            and toks[i + 1].token_type == SQLGlotTokenType.DOT
                            and _is_word_or_ident_token(toks[i + 2])
                        ):
                            if tk.text.lower() == u_alias.lower():
                                c_col = toks[i + 2].text.lower()
                                if tk.start > cursor:
                                    out.append(frag[cursor:tk.start])
                                out.append(col_subst.get(c_col, toks[i + 2].text))
                                cursor = toks[i + 2].end + 1
                                i += 3
                                continue
                        if _is_word_or_ident_token(tk):
                            is_after_dot = (i > 0 and toks[i - 1].token_type == SQLGlotTokenType.DOT)
                            is_before_dot = (i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.DOT)
                            is_fn = (
                                tk.token_type != SQLGlotTokenType.IDENTIFIER
                                and i + 1 < n
                                and toks[i + 1].token_type == SQLGlotTokenType.L_PAREN
                            )
                            if not is_after_dot and not is_before_dot and not is_fn:
                                c_low = tk.text.lower()
                                if c_low in col_subst:
                                    if tk.start > cursor:
                                        out.append(frag[cursor:tk.start])
                                    out.append(col_subst[c_low])
                                    cursor = tk.end + 1
                                    i += 1
                                    continue
                        i += 1
                    if cursor < len(frag):
                        out.append(frag[cursor:])
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
                self._uall_grp_alias_map = {mg: oa for mg, oa in zip(mapped_grps, orig_grp_aliases)}
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
            self._uall_grp_alias_map = None

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
        return not _has_subquery_ast(self.view.where_clause)

    def _inject_where_into_source_query(self, source_query_sql: str, extra_predicate: str) -> str:
        s = source_query_sql.strip().rstrip(";")
        tok_where = _find_top_level_token(s, (SQLGlotTokenType.WHERE,))
        tok_group = _find_top_level_token(s, (SQLGlotTokenType.GROUP_BY,))
        if tok_where is not None:
            insert_pos = tok_where.end + 1
            before = s[: tok_where.start].rstrip()
            if tok_group is not None:
                existing_w = s[insert_pos : tok_group.start].strip()
                after = s[tok_group.start :]
                return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})\n{after}"
            existing_w = s[insert_pos:].strip()
            return f"{before}\nWHERE ({existing_w}) AND ({extra_predicate})"
        if tok_group is not None:
            before = s[: tok_group.start].rstrip()
            after = s[tok_group.start :]
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
            toks = sqlglot.tokenize(frag, read="sqlite")
            if not toks:
                return frag
            out: List[str] = []
            cursor = 0
            i = 0
            n = len(toks)
            while i < n:
                tk = toks[i]
                if (
                    _is_word_or_ident_token(tk)
                    and i + 2 < n
                    and toks[i + 1].token_type == SQLGlotTokenType.DOT
                    and _is_word_or_ident_token(toks[i + 2])
                ):
                    tbl_low = tk.text.lower()
                    if tbl_low in alias_remap:
                        if tk.start > cursor:
                            out.append(frag[cursor:tk.start])
                        out.append(f"{alias_remap[tbl_low]}.{toks[i + 2].text}")
                        cursor = toks[i + 2].end + 1
                        i += 3
                        continue
                i += 1
            if cursor < len(frag):
                out.append(frag[cursor:])
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
            toks = sqlglot.tokenize(s_sq, read="sqlite")
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
                        toks = sqlglot.tokenize(s_sq, read="sqlite")
                        idx = 0
                        continue
                idx += 1
            return s_sq

        # Non-SQLite dialects
        s = sql
        toks = sqlglot.tokenize(s, read="sqlite")
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
                    toks = sqlglot.tokenize(s, read="sqlite")
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
            if isinstance(sel_ast, sqlglot_exp.Select) and sel_ast.args.get("joins"):
                j0 = sel_ast.args["joins"][0]
                if j0.side.upper() == "FULL" and (sel_ast.args.get("from_") or sel_ast.args.get("from")) and j0.args.get("on"):
                    from_t = (sel_ast.args.get("from_") or sel_ast.args.get("from")).this.sql("mysql")
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

        raw_q = self._rewrite_sqlite_total_in_sql(self.view.raw_select_sql.strip().rstrip(";"))
        if hidden_grp_defs:
            tok_from = _find_top_level_token(raw_q, (SQLGlotTokenType.FROM,))
            assert tok_from is not None
            extra_sel = ", ".join(f"{g_e} AS {mv_c}" for mv_c, g_e, _ in hidden_grp_defs)
            source_query_sql = f"{raw_q[:tok_from.start].rstrip()}, {extra_sel} {raw_q[tok_from.start:]}"
        else:
            source_query_sql = raw_q

        cols_csv = ", ".join(target_cols)
        initial_populate_sql = (
            f"INSERT INTO {mv_table} ({cols_csv})\n"
            f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init;"
        )
        ord_t2 = ", ".join(str(k) for k in range(1, len(target_cols) + 1))
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{source_query_sql}\n) AS __mv_init\n"
                f"ORDER BY {ord_t2}\n"
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

        raw_query = self._rewrite_sqlite_total_in_sql(self.view.raw_select_sql.strip().rstrip(";"))
        cols_csv = ", ".join(user_columns)
        initial_populate_sql = f"INSERT INTO {mv_table} ({cols_csv})\n{raw_query};"

        ord_t3 = ", ".join(str(k) for k in range(1, len(user_columns) + 1))
        inc_spec = IncrementalBackfillSpec(
            mode="offset_batch",
            offset_insert_template=(
                f"INSERT INTO {mv_table} ({cols_csv})\n"
                f"SELECT {cols_csv} FROM (\n{raw_query}\n) AS __mv_init\n"
                f"ORDER BY {ord_t3}\n"
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

    @staticmethod
    def _connect_database_url(
        conn_or_url: str, dialect: Optional[Dialect] = None
    ) -> Tuple[Any, Dialect]:
        u_low = conn_or_url.strip().lower()
        if u_low.startswith(("postgresql://", "postgres://")):
            import psycopg2  # type: ignore
            return psycopg2.connect(conn_or_url), Dialect.POSTGRES
        if u_low.startswith(("mysql://", "mariadb://")):
            import pymysql  # type: ignore
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(conn_or_url)
            db_name = parsed.path.lstrip("/")
            qs = parse_qs(parsed.query)
            return (
                pymysql.connect(
                    host=parsed.hostname or "127.0.0.1",
                    port=parsed.port or 3306,
                    user=parsed.username or "root",
                    password=parsed.password or "",
                    database=db_name,
                    unix_socket=qs.get("unix_socket", [None])[0],
                    autocommit=False,
                ),
                Dialect.MYSQL,
            )
        eff_dialect = dialect or Dialect.SQLITE
        path = conn_or_url.replace("sqlite:///", "").replace("sqlite://", "")
        return sqlite3.connect(path), eff_dialect

    def __init__(
        self,
        conn_or_url: sqlite3.Connection | str | Any,
        dialect: Optional[Dialect | str | EngineVersion] = None,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        pg_use_transition_tables: bool = False,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
    ) -> None:
        hint_d: Optional[Dialect] = None
        if isinstance(dialect, Dialect):
            hint_d = dialect
        elif isinstance(dialect, EngineVersion):
            hint_d = dialect.dialect
            if engine_version is None:
                engine_version = dialect
        elif isinstance(dialect, str) and dialect.strip():
            parsed_d = parse_engine_version(dialect)
            hint_d = parsed_d.dialect
            if engine_version is None and (":" in dialect or "-" in dialect or "mariadb" in dialect.lower() or any(ch.isdigit() for ch in dialect.replace("sqlite3", ""))):
                engine_version = parsed_d

        if isinstance(conn_or_url, str):
            self.conn, inferred_dialect = self._connect_database_url(conn_or_url, hint_d)
            if hint_d is None or conn_or_url.strip().lower().startswith(("postgresql://", "postgres://", "mysql://", "mariadb://")):
                hint_d = inferred_dialect
        else:
            self.conn = conn_or_url

        if engine_version is not None:
            self.engine_version: EngineVersion = parse_engine_version(engine_version, default_dialect=hint_d)
        else:
            self.engine_version = detect_engine_version(self.conn, hint_dialect=hint_d)

        self.dialect: Dialect = self.engine_version.dialect
        self.subdialect: EngineSubdialect = self.engine_version.subdialect
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.pg_use_transition_tables = pg_use_transition_tables
        self.strict = strict
        self.catalog_table = (
            f"{self.object_prefix}catalog" if self.object_prefix else "matview_catalog"
        )

        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        cur = self.conn.cursor()
        if self.dialect == Dialect.SQLITE:
            try:
                cur.execute("PRAGMA recursive_triggers = ON;")
            except Exception:
                pass
        elif self.dialect == Dialect.MYSQL:
            try:
                cur.execute("SET SESSION collation_connection = @@collation_database;")
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
                trigger_ddls_json TEXT NOT NULL,
                backfill_cursor_pk BIGINT DEFAULT NULL,
                backfill_max_pk BIGINT DEFAULT NULL,
                backfill_active INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._commit()
        for col_ddl in (
            "backfill_cursor_pk BIGINT DEFAULT NULL",
            "backfill_max_pk BIGINT DEFAULT NULL",
            "backfill_active INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                cur.execute(f"ALTER TABLE {self.catalog_table} ADD COLUMN {col_ddl};")
                self._commit()
            except Exception:
                self._rollback()

    def _commit(self) -> None:
        if hasattr(self.conn, "commit"):
            self.conn.commit()

    def _rollback(self) -> None:
        if hasattr(self.conn, "rollback"):
            try:
                self.conn.rollback()
            except Exception:
                pass

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
        schemas = introspect_database_schemas(
            self.conn,
            dialect=self.dialect,
            object_prefix=self.object_prefix,
        )
        schemas.pop(self.catalog_table, None)
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
        try:
            cur.execute(
                f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json, "
                f"backfill_cursor_pk, backfill_max_pk, backfill_active "
                f"FROM {self.catalog_table}"
            )
            has_wm_cols = True
        except Exception:
            self._rollback()
            cur.execute(
                f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json "
                f"FROM {self.catalog_table}"
            )
            has_wm_cols = False
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
                "backfill_cursor_pk": row[9] if has_wm_cols and len(row) > 9 else None,
                "backfill_max_pk": row[10] if has_wm_cols and len(row) > 10 else None,
                "backfill_active": int(row[11]) if has_wm_cols and len(row) > 11 and row[11] is not None else 0,
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

    def _update_backfill_watermark(
        self,
        cur: Any,
        view_name: str,
        cursor_pk: Optional[int],
        max_pk: Optional[int],
        active: int,
    ) -> None:
        ph = self._param_placeholder()
        try:
            cur.execute(
                f"UPDATE {self.catalog_table} "
                f"SET backfill_cursor_pk = {ph}, backfill_max_pk = {ph}, backfill_active = {ph} "
                f"WHERE view_name = {ph}",
                (cursor_pk, max_pk, active, view_name),
            )
        except Exception:
            pass

    def backfill_incrementally(
        self,
        plan: MaterializedViewPlan,
        batch_size: int = 500,
        on_progress: Optional[Callable[[int, int], None]] = None,
        snapshot_isolation: bool = False,
    ) -> int:
        """
        Incrementally backfills the materialized view in bounded chunks of `batch_size`
        (by PK/rowid range folding for algebraic views, group-key batches for distinct/holistic
        views, or limit/offset batches for general views), coordinating the active cursor
        watermark (`backfill_cursor_pk`, `backfill_max_pk`, `backfill_active`) in
        `__matview_catalog` and optionally running under `snapshot_isolation=True`.
        Returns the total number of batches executed.
        """
        if plan.sub_plans:
            total_b = 0
            for sp in plan.sub_plans:
                total_b += self.backfill_incrementally(
                    sp,
                    batch_size=batch_size,
                    on_progress=on_progress,
                    snapshot_isolation=snapshot_isolation,
                )
            return total_b

        spec = plan.incremental_backfill_spec
        cur = self.conn.cursor()
        if snapshot_isolation:
            try:
                if self.dialect == Dialect.POSTGRES:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                elif self.dialect == Dialect.MYSQL:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                    cur.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT;")
                elif self.dialect == Dialect.SQLITE:
                    cur.execute("SAVEPOINT __mv_backfill_snap;")
            except Exception:
                pass

        if spec is None or batch_size <= 0:
            for s in plan.initial_populate_sqls:
                cur.execute(s)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                try:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                except Exception:
                    pass
            self._commit()
            if on_progress:
                on_progress(1, 1)
            return 1

        batches_run = 0
        if spec.mode == "pk_range_fold" and spec.pk_column:
            base_t = quote_sql_identifier(plan.view_spec.base_table, self.dialect)
            pk_col = quote_sql_identifier(spec.pk_column, self.dialect)
            cur.execute(f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}")
            row = cur.fetchone()
            if not row or row[0] is None or row[1] is None:
                if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                    for s in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(s)
                self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
                if snapshot_isolation and self.dialect == Dialect.SQLITE:
                    try:
                        cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                    except Exception:
                        pass
                self._commit()
                return 0
            min_pk, max_pk = int(row[0]), int(row[1])
            cursor_pk = min_pk - 1
            total_span = max(1, max_pk - min_pk + 1)
            self._update_backfill_watermark(cur, plan.view_name, cursor_pk, max_pk, 1)
            if not snapshot_isolation:
                self._commit()
            while cursor_pk < max_pk:
                upper_pk = min(max_pk, cursor_pk + batch_size)
                for tmpl in spec.chunk_sql_templates:
                    cur.execute(tmpl, (cursor_pk, upper_pk))
                batches_run += 1
                cursor_pk = upper_pk
                still_active = 1 if cursor_pk < max_pk else 0
                self._update_backfill_watermark(cur, plan.view_name, cursor_pk, max_pk, still_active)
                if not snapshot_isolation:
                    self._commit()
                if on_progress:
                    done = min(total_span, cursor_pk - (min_pk - 1))
                    on_progress(batches_run, done)
            self._update_backfill_watermark(cur, plan.view_name, max_pk, max_pk, 0)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                try:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                except Exception:
                    pass
            self._commit()
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
                rc = getattr(cur, "rowcount", None)
                if rc is not None and rc >= 0:
                    inserted = int(rc)
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {plan.backing_table_name}")
                    cnt_row = cur.fetchone()
                    total_now = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else 0
                    inserted = max(0, total_now - offset)
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
        effective_strict: bool = False,
    ) -> MaterializedViewPlan:
        plan = generate_materialized_view(
            schemas,
            v_spec,
            dialect=self.dialect,
            pg_use_transition_tables=self.pg_use_transition_tables,
            ignore_limitations=effective_ignore,
            object_prefix=self.object_prefix,
            strict=effective_strict,
            engine_version=self.engine_version,
            watermark_catalog_table=self.catalog_table if (incremental_backfill or not populate) else None,
        )
        sig = self._compute_signature(plan, schemas)

        catalog = self.get_catalog()
        existing = catalog.get(plan.view_name)
        if existing and existing["signature_hash"] == sig and not force_rebuild:
            return plan

        if existing:
            self.drop_view(plan.view_name)

        # Install structure, maintenance triggers, single-shot backfill, and catalog record
        # inside a single atomic transaction so concurrent readers/writers never observe an
        # un-triggered or double-counted window between trigger attachment and population.
        cur = self.conn.cursor()
        try:
            for stmt in plan.create_structure_statements():
                cur.execute(stmt)
            for stmt in plan.trigger_ddls:
                cur.execute(stmt)

            stored_view_sql = self._serialize_view_sql_with_indexes(plan)
            ph = self._param_placeholder()
            init_active = 1 if (populate and incremental_backfill) else 0
            cur.execute(
                f"""
                INSERT INTO {self.catalog_table} (
                    view_name, view_sql, strategy, signature_hash, backing_table,
                    aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json,
                    backfill_cursor_pk, backfill_max_pk, backfill_active
                ) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
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
                    None,
                    None,
                    init_active,
                ),
            )

            if populate:
                if incremental_backfill:
                    self._commit()
                    self.backfill_incrementally(
                        plan, batch_size=batch_size, on_progress=on_progress
                    )
                else:
                    for stmt in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(stmt)
                    self._commit()
            else:
                self._commit()
        except Exception:
            self._rollback()
            raise
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
        strict: Optional[bool] = None,
    ) -> MaterializedViewPlan:
        """
        Safely installs or updates one or more materialized views on the connected database.
        Supports both single-view and multi-view SQL scripts seamlessly.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        effective_strict = self.strict if strict is None else strict
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
                    effective_strict=effective_strict,
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
        strict: Optional[bool] = None,
    ) -> Dict[str, str]:
        """
        Declaratively synchronizes a collection of materialized view definitions against
        the connected database (supporting multi-view SQL files/strings as well).
        Returns a status map `{view_name: "created" | "updated" | "unchanged" | "dropped"}`.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        effective_strict = self.strict if strict is None else strict
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
                strict=effective_strict,
                engine_version=self.engine_version,
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
                    effective_strict=effective_strict,
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
                    effective_strict=effective_strict,
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

    def check_schema_drift(self) -> Dict[str, str]:
        """
        Inspects all registered views in `__matview_catalog` against the current live base
        table schemas and returns `{view_name: "ok" | "drifted" | "invalid"}`.
        """
        schemas = self.fetch_schema()
        catalog = self.get_catalog()
        report: Dict[str, str] = {}
        for v_name, entry in catalog.items():
            try:
                plan = generate_materialized_view(
                    schemas,
                    entry["view_sql"],
                    dialect=self.dialect,
                    pg_use_transition_tables=self.pg_use_transition_tables,
                    ignore_limitations=True,
                    object_prefix=self.object_prefix,
                    engine_version=self.engine_version,
                )
                sig = self._compute_signature(plan, schemas)
                report[v_name] = "ok" if sig == entry["signature_hash"] else "drifted"
            except Exception:
                report[v_name] = "invalid"
        return report

    def repair_view(
        self,
        view_name: Optional[str] = None,
        rebuild_on_schema_drift: bool = True,
    ) -> Dict[str, str]:
        """
        Reconciles and repairs one or all registered materialized views in an atomic pass,
        eliminating any floating-point accumulation drift or out-of-band base table DDL changes.
        Returns `{view_name: "repaired" | "rebuilt"}`.
        """
        catalog = self.get_catalog()
        if view_name:
            resolved = self._resolve_catalog_name(view_name, catalog)
            if resolved not in catalog:
                raise KeyError(f"View {view_name!r} is not registered in {self.catalog_table}")
            targets = [catalog[resolved]]
        else:
            targets = list(catalog.values())

        schemas = self.fetch_schema()
        results: Dict[str, str] = {}
        cur = self.conn.cursor()
        for entry in targets:
            v_name = entry["view_name"]
            plan = generate_materialized_view(
                schemas,
                entry["view_sql"],
                dialect=self.dialect,
                pg_use_transition_tables=self.pg_use_transition_tables,
                ignore_limitations=True,
                object_prefix=self.object_prefix,
            )
            sig = self._compute_signature(plan, schemas)
            if rebuild_on_schema_drift and sig != entry["signature_hash"]:
                self.register_view(
                    entry["view_sql"],
                    populate=True,
                    force_rebuild=True,
                    ignore_limitations=True,
                )
                results[v_name] = "rebuilt"
            else:
                try:
                    for aux_t in plan.auxiliary_table_names:
                        cur.execute(f"DELETE FROM {aux_t};")
                    cur.execute(f"DELETE FROM {plan.backing_table_name};")
                    for stmt in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(stmt)
                    self._commit()
                    results[v_name] = "repaired"
                except Exception:
                    self._rollback()
                    raise
        return results

    @contextmanager
    def bulk_changes(
        self,
        view_names: Optional[Sequence[str]] = None,
        use_shadow_swap: bool = False,
        drop_triggers: bool = False,
    ) -> Iterator[None]:
        """
        Context manager for executing high-throughput multi-statement bulk loads/mutations.
        Uses session-local trigger bypass flags where supported (`SET LOCAL matview.skip_triggers = 'on'`
        on PostgreSQL, `SET @__matview_skip_triggers = 1` on MySQL) so triggers are never dropped
        globally across concurrent connections, transactional `SAVEPOINT` isolation on SQLite, and
        optional shadow-table rebuild + atomic `ALTER TABLE ... RENAME` swap (`use_shadow_swap=True`).
        """
        catalog = self.get_catalog()
        if view_names:
            resolved = [self._resolve_catalog_name(n, catalog) for n in view_names]
            targets = [catalog[n] for n in resolved if n in catalog]
        else:
            targets = list(catalog.values())

        cur = self.conn.cursor()
        sp_active = False
        try:
            cur.execute("SAVEPOINT __mv_bulk_sp;")
            sp_active = True
        except Exception:
            pass

        session_bypass_ok = False
        if self.dialect == Dialect.POSTGRES and not drop_triggers:
            try:
                cur.execute("SET LOCAL matview.skip_triggers = 'on';")
                session_bypass_ok = True
            except Exception:
                pass
        elif self.dialect == Dialect.MYSQL and not drop_triggers:
            try:
                cur.execute("SET @__matview_skip_triggers = 1;")
                session_bypass_ok = True
            except Exception:
                pass

        if not session_bypass_ok:
            for entry in targets:
                if self.dialect == Dialect.POSTGRES:
                    for fn_name in entry["function_names"]:
                        if "fn_readonly_" not in fn_name:
                            cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                else:
                    for t_name in entry["trigger_names"]:
                        if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                            cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
        # Note: Do NOT call self._commit() here on transactional-DDL engines (SQLite/PostgreSQL)
        # so the trigger suspension and bulk DML remain isolated inside __mv_bulk_sp!
        try:
            yield
        except Exception:
            if sp_active:
                try:
                    cur.execute("ROLLBACK TO SAVEPOINT __mv_bulk_sp;")
                    cur.execute("RELEASE SAVEPOINT __mv_bulk_sp;")
                except Exception:
                    self._rollback()
            else:
                self._rollback()
            if self.dialect == Dialect.POSTGRES:
                try:
                    cur.execute("SET LOCAL matview.skip_triggers = 'off';")
                except Exception:
                    pass
            elif self.dialect == Dialect.MYSQL:
                try:
                    cur.execute("SET @__matview_skip_triggers = 0;")
                except Exception:
                    pass
            schemas = self.fetch_schema()
            for entry in targets:
                if self.dialect == Dialect.POSTGRES:
                    for fn_name in entry["function_names"]:
                        if "fn_readonly_" not in fn_name:
                            cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                else:
                    for t_name in entry["trigger_names"]:
                        if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                            cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
                plan = generate_materialized_view(
                    schemas,
                    entry["view_sql"],
                    dialect=self.dialect,
                    pg_use_transition_tables=self.pg_use_transition_tables,
                    ignore_limitations=True,
                    object_prefix=self.object_prefix,
                    engine_version=self.engine_version,
                )
                for stmt in plan.trigger_ddls:
                    cur.execute(stmt)
            self._commit()
            raise
        else:
            if self.dialect == Dialect.POSTGRES:
                try:
                    cur.execute("SET LOCAL matview.skip_triggers = 'off';")
                except Exception:
                    pass
            elif self.dialect == Dialect.MYSQL:
                try:
                    cur.execute("SET @__matview_skip_triggers = 0;")
                except Exception:
                    pass
            schemas = self.fetch_schema()
            for entry in targets:
                plan = generate_materialized_view(
                    schemas,
                    entry["view_sql"],
                    dialect=self.dialect,
                    pg_use_transition_tables=self.pg_use_transition_tables,
                    ignore_limitations=True,
                    object_prefix=self.object_prefix,
                    engine_version=self.engine_version,
                )
                for aux_t in plan.auxiliary_table_names:
                    cur.execute(f"DELETE FROM {aux_t};")
                if use_shadow_swap:
                    shadow_tbl = f"{plan.backing_table_name}__shadow"
                    cur.execute(f"DROP TABLE IF EXISTS {shadow_tbl};")
                    shadow_ddl = plan.backing_table_ddl.replace(
                        f"CREATE TABLE {plan.backing_table_name}",
                        f"CREATE TABLE {shadow_tbl}",
                        1,
                    )
                    cur.execute(shadow_ddl)
                    for stmt in plan.get_populate_statements(incremental_backfill=False):
                        if f"INSERT INTO {plan.backing_table_name}" in stmt:
                            cur.execute(
                                stmt.replace(
                                    f"INSERT INTO {plan.backing_table_name}",
                                    f"INSERT INTO {shadow_tbl}",
                                    1,
                                )
                            )
                        else:
                            cur.execute(stmt)
                    if self.dialect == Dialect.POSTGRES:
                        for fn_name in entry["function_names"]:
                            if "fn_readonly_" not in fn_name:
                                cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                    else:
                        for t_name in entry["trigger_names"]:
                            if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                                cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
                    cur.execute(f"DROP VIEW IF EXISTS {plan.view_name};")
                    cur.execute(f"DROP TABLE IF EXISTS {plan.backing_table_name};")
                    cur.execute(f"ALTER TABLE {shadow_tbl} RENAME TO {plan.backing_table_name};")
                    if plan.reader_view_ddl:
                        cur.execute(plan.reader_view_ddl)
                    for ro_stmt in plan.readonly_trigger_ddls:
                        cur.execute(ro_stmt)
                    for idx_ddl in plan.index_ddls:
                        cur.execute(idx_ddl)
                else:
                    cur.execute(f"DELETE FROM {plan.backing_table_name};")
                    for stmt in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(stmt)
                if not session_bypass_ok or use_shadow_swap:
                    if self.dialect == Dialect.POSTGRES:
                        for fn_name in entry["function_names"]:
                            if "fn_readonly_" not in fn_name:
                                cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                    else:
                        for t_name in entry["trigger_names"]:
                            if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                                cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
                    for stmt in plan.trigger_ddls:
                        cur.execute(stmt)
            if sp_active:
                try:
                    cur.execute("RELEASE SAVEPOINT __mv_bulk_sp;")
                except Exception:
                    pass
            self._commit()


# -----------------------------------------------------------------------------
# Public High-Level Convenience Functions & CLI
# -----------------------------------------------------------------------------


def _resolve_dialect_and_engine_version(
    schema_sql_or_dict: Any,
    dialect: Optional[Dialect | str | EngineVersion],
    engine_version: Optional[EngineVersion | str],
) -> EngineVersion:
    is_conn = hasattr(schema_sql_or_dict, "cursor") or isinstance(
        schema_sql_or_dict, sqlite3.Connection
    )
    if engine_version is not None:
        def_d = dialect if not isinstance(dialect, EngineVersion) else dialect.dialect
        return parse_engine_version(engine_version, default_dialect=def_d)
    if isinstance(dialect, EngineVersion):
        return dialect
    if isinstance(dialect, str) and dialect.strip():
        if ":" in dialect or "-" in dialect or "mariadb" in dialect.lower() or any(
            ch.isdigit() for ch in dialect.replace("sqlite3", "")
        ):
            return parse_engine_version(dialect)
        if is_conn:
            return detect_engine_version(schema_sql_or_dict, hint_dialect=dialect)
        return parse_engine_version(dialect)
    if is_conn:
        return detect_engine_version(schema_sql_or_dict, hint_dialect=dialect)
    return parse_engine_version(dialect or Dialect.SQLITE)


def generate_materialized_view(
    schema_sql_or_dict: str | Dict[str, TableSchema] | sqlite3.Connection | Any,
    view_sql_or_spec: str | ViewSpec,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
    strict: bool = False,
    engine_version: Optional[EngineVersion | str] = None,
    watermark_catalog_table: Optional[str] = None,
) -> MaterializedViewPlan:
    """
    Compiles a `CREATE VIEW` SQL definition (or multi-view SQL script) and base table schema
    into a `MaterializedViewPlan`. Autodetects the database dialect and version when passed
    a live connection, or accepts `engine_version` (e.g. `'postgres:17'`, `'mariadb:11.4'`, `'mysql:8.4'`).
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict)
    elif hasattr(schema_sql_or_dict, "cursor") or isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_database_schemas(
            schema_sql_or_dict,
            dialect=eff_dialect,
            object_prefix=object_prefix,
        )
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
        dialect=eff_dialect,
        pg_use_transition_tables=pg_use_transition_tables,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
        strict=strict,
        engine_version=resolved_ev,
        watermark_catalog_table=watermark_catalog_table,
    )
    return compiler.compile()


def generate_materialized_views(
    schema_sql_or_dict: str | Dict[str, TableSchema] | sqlite3.Connection | Any,
    view_sqls: Sequence[str | ViewSpec] | str | ViewSpec,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    pg_use_transition_tables: bool = False,
    ignore_limitations: bool = False,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
    strict: bool = False,
    engine_version: Optional[EngineVersion | str] = None,
) -> List[MaterializedViewPlan]:
    """
    Compiles one or more `CREATE VIEW` SQL definitions (including multi-view SQL scripts)
    into a flat list of `MaterializedViewPlan` objects.
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict)
    elif hasattr(schema_sql_or_dict, "cursor") or isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_database_schemas(
            schema_sql_or_dict,
            dialect=eff_dialect,
            object_prefix=object_prefix,
        )
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
            dialect=eff_dialect,
            pg_use_transition_tables=pg_use_transition_tables,
            ignore_limitations=ignore_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
            strict=strict,
            engine_version=resolved_ev,
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
    strict: bool = False,
) -> MaterializedViewPlan:
    """Installs one or more materialized views directly into a live SQLite connection."""
    mgr = MaterializedViewManager(
        conn,
        dialect=Dialect.SQLITE,
        ignore_limitations=ignore_limitations,
        object_prefix=object_prefix,
        strict=strict,
    )
    return mgr.register_view(
        view_sql=view_sql,
        schema_sql=schema_sql,
        populate=populate,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
        on_progress=on_progress,
        indexes=indexes,
        strict=strict,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile SQL CREATE VIEW definition(s) into zero-dependency incremental materialized view SQL."
    )
    parser.add_argument("--schema", help="Path to SQL file containing base table CREATE TABLE statements.")
    parser.add_argument("--db", help="Database path or DSN URL (sqlite:///..., postgresql://..., mysql://...).")
    parser.add_argument("--view", help="Path to SQL file or inline CREATE VIEW statement(s).")
    parser.add_argument(
        "--dialect",
        default=None,
        help="Target SQL dialect or engine spec (sqlite, postgres, mysql, mariadb, or versioned e.g. postgres:17).",
    )
    parser.add_argument(
        "--engine-version",
        dest="engine_version",
        default=None,
        help="Target engine and version (e.g. 'postgres:17', 'mysql:8.4', 'mariadb:11.4', 'sqlite:3.45').",
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
        "--strict",
        action="store_true",
        help="Fail compilation if a view cannot use Tier 1 ALGEBRAIC_DELTA maintenance.",
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Check registered views in --db for underlying base table schema drift.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Atomically reconcile/repair registered views in --db from base tables.",
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

    try:
        if args.check_drift or args.repair:
            if not args.db:
                print("ERROR: --check-drift / --repair requires --db", file=sys.stderr)
                return 2
            mgr = MaterializedViewManager(
                args.db,
                dialect=args.dialect,
                ignore_limitations=args.ignore_limitations,
                object_prefix=args.object_prefix,
                pg_use_transition_tables=args.pg_transition_tables,
                strict=args.strict,
                engine_version=args.engine_version,
            )
            if args.check_drift:
                drift = mgr.check_schema_drift()
                print(json.dumps(drift, indent=2, sort_keys=True))
                if any(v != "ok" for v in drift.values()) and not args.repair:
                    return 1
            if args.repair:
                rep = mgr.repair_view()
                print(json.dumps(rep, indent=2, sort_keys=True))
            return 0

        if not args.view:
            print("ERROR: --view is required unless running --check-drift or --repair.", file=sys.stderr)
            return 2

        view_sql = _read_sql_arg(args.view)
        schema_sql = _read_sql_arg(args.schema) if args.schema else None

        if args.apply:
            if not args.db:
                print("ERROR: --apply requires --db <db_path_or_dsn>", file=sys.stderr)
                return 2
            mgr = MaterializedViewManager(
                args.db,
                dialect=args.dialect,
                ignore_limitations=args.ignore_limitations,
                object_prefix=args.object_prefix,
                pg_use_transition_tables=args.pg_transition_tables,
                strict=args.strict,
                engine_version=args.engine_version,
            )
            if schema_sql:
                mgr.apply_schema_ddl(schema_sql, auto_sync_views=False)
            plan = mgr.register_view(
                view_sql,
                populate=not args.no_populate,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
                indexes=args.indexes,
                strict=args.strict,
            )
            print(
                plan.render_sql_script(
                    include_populate=not args.no_populate,
                    incremental_backfill=args.incremental_backfill,
                    batch_size=args.batch_size,
                )
            )
            return 0

        hint_d: Optional[Dialect] = None
        if args.dialect:
            hint_d = parse_engine_version(args.dialect).dialect
        elif args.engine_version:
            hint_d = parse_engine_version(args.engine_version).dialect

        detected_ev: Optional[EngineVersion] = None
        if args.db and not schema_sql:
            conn, conn_d = MaterializedViewManager._connect_database_url(args.db, hint_d)
            try:
                detected_ev = (
                    parse_engine_version(args.engine_version, default_dialect=conn_d)
                    if args.engine_version
                    else detect_engine_version(conn, hint_dialect=args.dialect or conn_d)
                )
                schemas = introspect_database_schemas(
                    conn, dialect=detected_ev.dialect, object_prefix=args.object_prefix
                )
            finally:
                conn.close()
        elif schema_sql:
            schemas = parse_table_schema(schema_sql)
            detected_ev = _resolve_dialect_and_engine_version(
                schemas, args.dialect, args.engine_version
            )
        else:
            print("ERROR: Either --schema or --db must be provided.", file=sys.stderr)
            return 2

        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=detected_ev.dialect,
            pg_use_transition_tables=args.pg_transition_tables,
            ignore_limitations=args.ignore_limitations,
            object_prefix=args.object_prefix,
            indexes=args.indexes,
            incremental_backfill=args.incremental_backfill,
            batch_size=args.batch_size,
            strict=args.strict,
            engine_version=detected_ev,
        )
        print(
            plan.render_sql_script(
                include_populate=not args.no_populate,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
            )
        )
        return 0
    except (EngineLimitationError, StrictCompilationError) as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
