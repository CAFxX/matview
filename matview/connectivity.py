"""SQLAlchemy URL normalization, connection/proxy management, and cross-engine schema introspection."""

from __future__ import annotations

import copy
import dataclasses
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine as _sa_create_engine, event as _sa_event, inspect as _sa_inspect
from sqlalchemy.engine import Connection as _SAConnection, Engine as _SAEngine
from sqlalchemy.engine.url import URL as _SAURL, make_url as _sa_make_url
from sqlalchemy.pool import NullPool as _SANullPool

from .engines import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    _extract_packaging_and_tuple,
    _extract_version_tuple,
    _read_mariadb_div_precision_increment,
    _read_mysql_connection_collation,
    with_probed_sqlite_statistics,
    parse_engine_version,
)
from .logical_ir import ColumnDef, TableIndexDef, TableSchema
from .sql_frontend import _strip_sql_comments, parse_table_index_def, parse_table_schema

def detect_engine_version(
    conn: Any,
    hint_dialect: Optional[Dialect | str | EngineVersion] = None,
) -> EngineVersion:
    """Detect the engine of ``conn`` (see ``_detect_engine_version``) and, on SQLite,
    probe the connection for statistical aggregates that differ from the exact
    definition, whatever path (attribute, hint, pinned version) produced the
    version."""
    return with_probed_sqlite_statistics(unwrap_dbapi_connection(conn)[0], _detect_engine_version(conn, hint_dialect))


def _detect_engine_version(
    conn: Any,
    hint_dialect: Optional[Dialect | str | EngineVersion] = None,
) -> EngineVersion:
    """
    Autodetects the database engine (`Dialect`), subdialect (`sqlite`, `postgres`, `mysql`, `mariadb`),
    and exact server version `(major, minor, patch)` from a live database connection.
    """
    if isinstance(hint_dialect, EngineVersion):
        return hint_dialect

    unwrapped_conn, _, _, sa_dialect = unwrap_dbapi_connection(conn)
    if unwrapped_conn is not conn:
        return _detect_engine_version(
            unwrapped_conn,
            hint_dialect=hint_dialect if hint_dialect is not None else sa_dialect,
        )

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

    try:
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
                        div_increment = _read_mariadb_div_precision_increment(cur)
                        return EngineVersion(
                            Dialect.MYSQL, EngineSubdialect.MARIADB, ver, v_str, div_increment,
                            mysql_connection_collation=_read_mysql_connection_collation(cur),
                        )
                    ver = _extract_version_tuple(v_str, (8, 0, 0))
                    return EngineVersion(
                        Dialect.MYSQL, EngineSubdialect.MYSQL, ver, v_str,
                        mysql_connection_collation=_read_mysql_connection_collation(cur),
                    )
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
                v_str = str(row[0])
                return parse_engine_version(f"sqlite:{v_str}", default_dialect=Dialect.SQLITE)
        except Exception:
            pass
        try:
            cur.execute("SELECT version();")
            row = cur.fetchone()
            if row and row[0]:
                v_str = str(row[0]).strip()
                if "postgres" in v_str.lower() or "cockroach" in v_str.lower() or "yugabyte" in v_str.lower():
                    pkg_v, ver = _extract_packaging_and_tuple(v_str, (14, 0, 0))
                    return EngineVersion(
                        Dialect.POSTGRES,
                        EngineSubdialect.POSTGRES,
                        ver,
                        v_str,
                        packaging_version=pkg_v,
                    )
                if "mariadb" in v_str.lower():
                    pkg_v, ver = _extract_packaging_and_tuple(v_str, (11, 4, 0))
                    div_increment = _read_mariadb_div_precision_increment(cur)
                    return EngineVersion(
                        Dialect.MYSQL,
                        EngineSubdialect.MARIADB,
                        ver,
                        v_str,
                        div_increment,
                        packaging_version=pkg_v,
                        mysql_connection_collation=_read_mysql_connection_collation(cur),
                    )
                pkg_v, ver = _extract_packaging_and_tuple(v_str, (8, 0, 0))
                return EngineVersion(
                    Dialect.MYSQL,
                    EngineSubdialect.MYSQL,
                    ver,
                    v_str,
                    packaging_version=pkg_v,
                    mysql_connection_collation=_read_mysql_connection_collation(cur),
                )
        except Exception:
            pass
        return parse_engine_version("sqlite", default_dialect=Dialect.SQLITE)
    finally:
        close_fn = getattr(cur, "close", None)
        if callable(close_fn):
            close_fn()



def normalize_sqlalchemy_url(
    conn_or_url: str, dialect: Optional[Dialect] = None
) -> Tuple[_SAURL, Dialect, Dict[str, Any]]:
    """
    Normalize a database URL or SQLite path into a SQLAlchemy `URL`, inferred
    `Dialect`, and keyword `connect_args` for `sqlalchemy.create_engine`.
    """
    raw = conn_or_url.strip()
    u_low = raw.lower()
    connect_args: Dict[str, Any] = {}

    if "://" in raw:
        normalized_str = raw
        if u_low.startswith("postgres://"):
            normalized_str = "postgresql+psycopg2://" + raw[len("postgres://") :]
        elif u_low.startswith("postgresql://"):
            normalized_str = "postgresql+psycopg2://" + raw[len("postgresql://") :]
        elif u_low.startswith("mariadb://"):
            normalized_str = "mysql+pymysql://" + raw[len("mariadb://") :]
        elif u_low.startswith("mysql://"):
            normalized_str = "mysql+pymysql://" + raw[len("mysql://") :]
        sa_url = _sa_make_url(normalized_str)
    else:
        db_path = raw
        if db_path == ":memory:":
            sa_url = _sa_make_url("sqlite:///:memory:")
        else:
            sa_url = _SAURL.create(drivername="sqlite", database=db_path)

    backend = sa_url.get_backend_name().lower()
    if backend in ("postgresql", "postgres"):
        eff_dialect = Dialect.POSTGRES
    elif backend in ("mysql", "mariadb"):
        eff_dialect = Dialect.MYSQL
        query_map = dict(sa_url.query)
        if "unix_socket" in query_map:
            sock_val = query_map.pop("unix_socket")
            connect_args["unix_socket"] = sock_val[-1] if isinstance(sock_val, tuple) else sock_val
            sa_url = sa_url.set(query=query_map)
    else:
        eff_dialect = dialect or Dialect.SQLITE

    return sa_url, eff_dialect, connect_args


def connect_database_via_sqlalchemy(
    conn_or_url: str, dialect: Optional[Dialect] = None
) -> Tuple[Any, Dialect, _SAEngine]:
    """
    Connect to SQLite, PostgreSQL, or MySQL/MariaDB using SQLAlchemy's unified
    URL parser and `create_engine`, returning `(raw_dbapi_conn, dialect, engine)`.
    """
    sa_url, eff_dialect, connect_args = normalize_sqlalchemy_url(conn_or_url, dialect=dialect)
    engine_kwargs: Dict[str, Any] = {"poolclass": _SANullPool}
    if connect_args:
        engine_kwargs["connect_args"] = connect_args
    engine = _sa_create_engine(sa_url, **engine_kwargs)
    try:
        raw_conn = engine.raw_connection()
    except Exception:
        engine.dispose()
        raise
    return raw_conn, eff_dialect, engine


def unwrap_dbapi_connection(
    conn_or_sa: Any,
) -> Tuple[Any, Optional[_SAEngine], Optional[_SAConnection], Optional[Dialect]]:
    """
    Unwrap a SQLAlchemy `Engine`, `Connection`, or pool `_ConnectionFairy` into
    `(dbapi_conn, owned_engine_to_dispose, sa_conn_if_any, inferred_dialect)`.
    Plain DB-API connections are returned unchanged with `None` SQLAlchemy handles.
    """
    def _dialect_from_sa_name(name: str) -> Optional[Dialect]:
        n = (name or "").lower()
        if "postgres" in n:
            return Dialect.POSTGRES
        if "mysql" in n or "mariadb" in n:
            return Dialect.MYSQL
        if "sqlite" in n:
            return Dialect.SQLITE
        return None

    if isinstance(conn_or_sa, _SAEngine):
        inferred = _dialect_from_sa_name(conn_or_sa.dialect.name)
        raw_conn = conn_or_sa.raw_connection()
        return raw_conn, None, None, inferred
    if isinstance(conn_or_sa, _SAConnection):
        inferred = _dialect_from_sa_name(conn_or_sa.dialect.name)
        fairy = conn_or_sa.connection
        inner = getattr(fairy, "dbapi_connection", None) or getattr(fairy, "driver_connection", None) or fairy
        return inner, None, conn_or_sa, inferred
    if hasattr(conn_or_sa, "driver_connection") and hasattr(conn_or_sa, "detach"):
        inner = getattr(conn_or_sa, "dbapi_connection", None) or getattr(conn_or_sa, "driver_connection", None)
        if inner is not None:
            return inner, None, None, None
    return conn_or_sa, None, None, None


def render_parameterized_sql(qmark_sql: str, dialect: Dialect) -> str:
    """
    Convert a `?`-placeholder SQL query into the dialect's DB-API bind syntax
    without modifying `?` characters inside string literals or comments.
    """
    if dialect == Dialect.SQLITE or "?" not in qmark_sql:
        return qmark_sql
    out: List[str] = []
    i = 0
    n = len(qmark_sql)
    in_sq = False
    in_dq = False
    while i < n:
        ch = qmark_sql[i]
        if in_sq:
            out.append(ch)
            if ch == "'" and i + 1 < n and qmark_sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            if ch == "'":
                in_sq = False
            i += 1
        elif in_dq:
            out.append(ch)
            if ch == '"' and i + 1 < n and qmark_sql[i + 1] == '"':
                out.append('"')
                i += 2
                continue
            if ch == '"':
                in_dq = False
            i += 1
        elif ch == "'":
            in_sq = True
            out.append(ch)
            i += 1
        elif ch == '"':
            in_dq = True
            out.append(ch)
            i += 1
        elif ch == "?":
            out.append("%s")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def dbapi_connection_layers(conn: Any) -> List[Any]:
    """Return ``conn`` followed by the physical DB-API connections it wraps.

    SQLAlchemy pool proxies (``_ConnectionFairy`` / ``PoolProxiedConnection``)
    forward attribute reads to the driver but not attribute writes, so state
    changes such as ``autocommit`` must target the physical connection.
    """
    layers: List[Any] = []
    pending = [conn]
    # Bounded: a proxy chain is at most a few layers deep, and a permissive
    # __getattr__ (mocks) must not produce an endless chain.
    while pending and len(layers) < 4:
        obj = pending.pop(0)
        if obj is None or any(obj is seen for seen in layers):
            continue
        layers.append(obj)
        for attr in ("dbapi_connection", "driver_connection"):
            try:
                inner = getattr(obj, attr, None)
            except Exception:
                inner = None
            if inner is not None and inner is not obj:
                pending.append(inner)
    return layers


def _layer_transaction_state(obj: Any) -> Optional[bool]:
    """Positively reported transaction state of one connection layer, else None.

    Recognized reports: ``in_transaction`` (sqlite3 and SQLite-compatible
    drivers, mysql-connector), ``get_transaction_status()`` (psycopg2) and
    ``info.transaction_status`` (psycopg 3), where only IDLE (0) is idle, and
    PyMySQL's ``server_status`` SERVER_STATUS_IN_TRANS bit.
    """
    try:
        flag = getattr(obj, "in_transaction", None)
    except Exception:
        flag = None
    if isinstance(flag, bool):
        return flag
    try:
        getter = getattr(obj, "get_transaction_status", None)
    except Exception:
        getter = None
    if callable(getter):
        try:
            status = getter()
        except Exception:
            status = None
        if isinstance(status, int) and not isinstance(status, bool):
            return status != 0
    try:
        info = getattr(obj, "info", None)
        status = getattr(info, "transaction_status", None)
    except Exception:
        status = None
    if isinstance(status, int) and not isinstance(status, bool):
        return int(status) != 0
    try:
        server_status = getattr(obj, "server_status", None)
    except Exception:
        server_status = None
    if isinstance(server_status, int) and not isinstance(server_status, bool):
        return bool(server_status & 0x0001)  # SERVER_STATUS_IN_TRANS
    return None


def dbapi_transaction_state(conn: Any) -> Optional[bool]:
    """Whether ``conn`` has an open transaction: True, False, or None if unknown.

    Every layer of a wrapped connection is consulted; a transaction reported
    by any layer (for example one opened directly on the raw driver behind an
    idle-looking wrapper) counts as open.
    """
    states = [_layer_transaction_state(layer) for layer in dbapi_connection_layers(conn)]
    known = [state for state in states if state is not None]
    if not known:
        return None
    return any(known)


class _NonClosingConnectionProxy:
    """
    Read-only connection proxy for SQLAlchemy schema inspection that keeps
    caller connections open and suppresses pool-level commit/rollback resets.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


def _can_use_sqlalchemy_inspector(conn: Any) -> bool:
    if isinstance(conn, (_SAEngine, _SAConnection, sqlite3.Connection)):
        return True
    if hasattr(conn, "driver_connection") and hasattr(conn, "detach"):
        return True
    mod_cls = f"{type(conn).__module__}.{type(conn).__qualname__}".lower()
    return any(k in mod_cls for k in ("psycopg", "pymysql"))


def _format_sa_column_type(
    col_type: Any,
    dialect: Dialect,
    *,
    _in_array: bool = False,
) -> str:
    # A PostgreSQL domain (SQLAlchemy's postgresql.DOMAIN, possibly over
    # another domain) is stored and compared as its base type; its name is not
    # a type the backing table can be declared with. SQLAlchemy does not
    # reflect the base type's modifiers (VARCHAR(10) comes back as VARCHAR), so
    # an unconstrained VARCHAR stays VARCHAR (the result type the view reports).
    # The reflected base type also carries the domain's collation, "default"
    # for an uncollated domain, which is not a collation name; an explicit
    # collation is kept with its original spelling.
    from_domain = False
    while type(col_type).__name__ == "DOMAIN" and getattr(col_type, "data_type", None) is not None:
        col_type = col_type.data_type
        from_domain = True
    if (
        dialect == Dialect.POSTGRES
        and type(col_type).__name__ == "ARRAY"
        and getattr(col_type, "item_type", None) is not None
    ):
        item_t = col_type.item_type
        arr_collation: Optional[str] = None
        if getattr(item_t, "collation", None):
            coll_s = str(item_t.collation)
            arr_collation = None if coll_s.lower() == "default" else coll_s
            item_t = copy.copy(item_t)
            item_t.collation = None
        elem_str = _format_sa_column_type(item_t, dialect, _in_array=True)
        if elem_str.upper() == "NULL":
            return "NULL"
        dims = getattr(col_type, "dimensions", None)
        n_dims = int(dims) if isinstance(dims, int) and dims > 0 else 1
        arr_ddl = f"{elem_str}{'[]' * n_dims}"
        if arr_collation:
            quoted_coll = '"' + arr_collation.replace('"', '""') + '"'
            arr_ddl = f"{arr_ddl} COLLATE {quoted_coll}"
        return arr_ddl
    if (
        dialect == Dialect.POSTGRES
        and type(col_type).__name__ in ("ENUM", "Enum")
        and getattr(col_type, "name", None)
    ):
        enum_name = '"' + str(col_type.name).replace('"', '""') + '"'
        enum_schema = getattr(col_type, "schema", None)
        if enum_schema:
            quoted_schema = '"' + str(enum_schema).replace('"', '""') + '"'
            return f"{quoted_schema}.{enum_name}"
        return enum_name
    domain_collation: Optional[str] = None
    if from_domain and getattr(col_type, "collation", None):
        coll = str(col_type.collation)
        domain_collation = None if coll.lower() == "default" else coll
        col_type = copy.copy(col_type)
        col_type.collation = None
    precision = getattr(col_type, "precision", None)
    scale = getattr(col_type, "scale", None)
    raw_str = str(col_type).upper()
    if precision is not None and scale is not None and (
        "NUMERIC" in raw_str or "DECIMAL" in raw_str
    ):
        raw_str = f"DECIMAL({int(precision)}, {int(scale)})"
    elif (from_domain or _in_array) and raw_str.startswith("VARCHAR") and not getattr(col_type, "length", None):
        raw_str = "VARCHAR"
    elif (
        dialect == Dialect.POSTGRES and (from_domain or _in_array)
        and raw_str == "CHAR" and not getattr(col_type, "length", None)
    ):
        # CHAR without its reflected length (a domain's base type, an array
        # element) would be CHAR(1); BPCHAR is the unbounded blank-padded type
        # (same result type as CHAR(n)).
        raw_str = "BPCHAR"
    elif dialect == Dialect.POSTGRES and raw_str.startswith("VARCHAR") and not getattr(col_type, "length", None):
        raw_str = "VARCHAR"
    elif dialect == Dialect.POSTGRES and getattr(col_type, "timezone", False) and raw_str in ("TIMESTAMP", "TIME"):
        # SQLAlchemy prints TIMESTAMP(timezone=True) as plain TIMESTAMP.
        raw_str = f"{raw_str} WITH TIME ZONE"
    elif dialect == Dialect.POSTGRES and raw_str == "BIT":
        # SQLAlchemy's PostgreSQL BIT prints as bare "BIT" without its width or
        # VARYING modifier; bare BIT in DDL means BIT(1).
        bit_len = getattr(col_type, "length", None)
        if getattr(col_type, "varying", False):
            raw_str = f"BIT VARYING({int(bit_len)})" if bit_len is not None else "BIT VARYING"
        elif bit_len is not None:
            raw_str = f"BIT({int(bit_len)})"
    elif dialect == Dialect.MYSQL and raw_str == "BIT" and getattr(col_type, "length", None):
        # SQLAlchemy's MySQL BIT prints without its width; the width decides
        # MySQL's MIN/MAX result types (BIT(1..9) vs BIT(10..64)).
        raw_str = f"BIT({int(col_type.length)})"
    elif dialect == Dialect.MYSQL and raw_str in ("ENUM", "SET"):
        # SQLAlchemy prints MySQL ENUM/SET without their members, which is
        # not valid DDL for a backing column that keeps the type.
        members = getattr(col_type, "enums", None) or getattr(col_type, "values", None)
        if members:
            raw_str = raw_str + "(" + ", ".join("'" + str(m).replace("'", "''") + "'" for m in members) + ")"
    if getattr(col_type, "unsigned", False) and "UNSIGNED" not in raw_str:
        raw_str = f"{raw_str} UNSIGNED"
    if domain_collation:
        raw_str = f'{raw_str} COLLATE "{domain_collation}"'
    return raw_str


_PG_NATIVE_ARRAY_TYPES_SQL = """
WITH RECURSIVE unwrapped AS (
    SELECT oid, oid AS base_oid, typtypmod AS base_mod, typtype, typbasetype, typcategory, typelem
    FROM pg_catalog.pg_type
    UNION ALL
    SELECT u.oid, p.oid,
           CASE WHEN u.base_mod <> -1 THEN u.base_mod ELSE p.typtypmod END,
           p.typtype, p.typbasetype, p.typcategory, p.typelem
    FROM unwrapped u
    JOIN pg_catalog.pg_type p ON p.oid = u.typbasetype
    WHERE u.typtype = 'd'
)
SELECT cl.relname,
       a.attname,
       (CASE
           WHEN u.typcategory = 'A' AND et.typtype = 'd'
               THEN pg_catalog.format_type(
                   ue.base_oid,
                   CASE WHEN a.atttypmod <> -1 THEN a.atttypmod ELSE ue.base_mod END
               ) || '[]'
           WHEN u.base_oid = 'pg_catalog.bpchar'::pg_catalog.regtype
                AND a.atttypmod = -1 AND u.base_mod = -1
               THEN 'bpchar'
           WHEN t.typtype = 'd'
               THEN pg_catalog.format_type(
                   u.base_oid,
                   CASE WHEN a.atttypmod <> -1 THEN a.atttypmod ELSE u.base_mod END
               )
           ELSE pg_catalog.format_type(a.atttypid, a.atttypmod)
       END)
       || CASE
           WHEN co.oid IS NOT NULL
               THEN ' COLLATE ' || pg_catalog.quote_ident(cn.nspname) || '.' || pg_catalog.quote_ident(co.collname)
           ELSE ''
       END AS resolved_type
FROM pg_catalog.pg_attribute a
JOIN pg_catalog.pg_class cl ON cl.oid = a.attrelid
JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
JOIN unwrapped u ON u.oid = t.oid AND u.typtype <> 'd'
LEFT JOIN pg_catalog.pg_type et ON et.oid = u.typelem AND u.typcategory = 'A'
LEFT JOIN unwrapped ue ON ue.oid = et.oid AND ue.typtype <> 'd'
LEFT JOIN pg_catalog.pg_collation co
       ON co.oid = a.attcollation AND a.attcollation <> 0 AND a.attcollation <> t.typcollation
LEFT JOIN pg_catalog.pg_namespace cn ON cn.oid = co.collnamespace
WHERE pg_catalog.pg_table_is_visible(cl.oid)
  AND cl.relkind IN ('r', 'p')
  AND (
      u.typcategory = 'A'
      OR u.typtype IN ('c', 'r', 'm', 'e')
      OR co.oid IS NOT NULL
      OR (u.base_oid = 'pg_catalog.bpchar'::pg_catalog.regtype AND a.atttypmod = -1 AND u.base_mod = -1)
      OR u.base_oid = 'pg_catalog.name'::pg_catalog.regtype
  )
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND cl.relname NOT LIKE '__matview_%'
  AND cl.relname NOT LIKE '__mv_%'
"""


def _pg_introspect_native_array_types(conn_or_bind: Any) -> Dict[Tuple[str, str], str]:
    """Return ``(table_lower, col_lower) -> native PostgreSQL DDL type`` for
    array, composite, range, and multirange columns (including domains over
    them and arrays of domains) of visible regular and partitioned tables via
    ``pg_catalog.format_type``.

    Uses a SAVEPOINT when called inside an active transaction so an unexpected
    catalog/permission error never aborts the caller's PostgreSQL transaction.
    """
    rows: List[Any] = []
    try:
        if isinstance(conn_or_bind, _SAEngine):
            with conn_or_bind.connect() as sa_conn:
                rows = list(sa_conn.exec_driver_sql(_PG_NATIVE_ARRAY_TYPES_SQL).fetchall() or ())
        elif isinstance(conn_or_bind, _SAConnection):
            if conn_or_bind.in_transaction():
                with conn_or_bind.begin_nested():
                    rows = list(conn_or_bind.exec_driver_sql(_PG_NATIVE_ARRAY_TYPES_SQL).fetchall() or ())
            else:
                rows = list(conn_or_bind.exec_driver_sql(_PG_NATIVE_ARRAY_TYPES_SQL).fetchall() or ())
        elif hasattr(conn_or_bind, "cursor"):
            cur = conn_or_bind.cursor()
            used_sp = False
            try:
                if dbapi_transaction_state(conn_or_bind) is True:
                    cur.execute("SAVEPOINT __mv_pg_arr_introspect")
                    used_sp = True
                cur.execute(_PG_NATIVE_ARRAY_TYPES_SQL)
                rows = list(cur.fetchall() or ())
                if used_sp:
                    cur.execute("RELEASE SAVEPOINT __mv_pg_arr_introspect")
            except Exception:
                if used_sp:
                    try:
                        cur.execute("ROLLBACK TO SAVEPOINT __mv_pg_arr_introspect")
                        cur.execute("RELEASE SAVEPOINT __mv_pg_arr_introspect")
                    except Exception:
                        pass
                return {}
            finally:
                close_fn = getattr(cur, "close", None)
                if callable(close_fn):
                    close_fn()
    except Exception:
        return {}
    out: Dict[Tuple[str, str], str] = {}
    for r in rows:
        if len(r) >= 3 and r[0] and r[1] and r[2]:
            out[(str(r[0]).lower(), str(r[1]).lower())] = str(r[2])
    return out


_PG_GENERATED_COLUMNS_SQL = """
WITH RECURSIVE unwrapped AS (
    SELECT oid, oid AS base_oid, typtypmod AS base_mod, typtype, typbasetype, typcollation
    FROM pg_catalog.pg_type
    UNION ALL
    SELECT u.oid, p.oid,
           CASE WHEN u.base_mod <> -1 THEN u.base_mod ELSE p.typtypmod END,
           p.typtype, p.typbasetype, p.typcollation
    FROM unwrapped u
    JOIN pg_catalog.pg_type p ON p.oid = u.typbasetype
    WHERE u.typtype = 'd'
)
SELECT cl.relname,
       a.attname,
       a.attgenerated,
       pg_catalog.pg_get_expr(ad.adbin, ad.adrelid),
       CASE
           WHEN t.typtype = 'd'
               THEN pg_catalog.format_type(
                   u.base_oid,
                   CASE WHEN a.atttypmod <> -1 THEN a.atttypmod ELSE u.base_mod END
               )
           ELSE pg_catalog.format_type(a.atttypid, a.atttypmod)
       END || CASE
           WHEN a.attcollation <> 0 AND a.attcollation <> t.typcollation
               THEN ' COLLATE ' || pg_catalog.quote_ident(cn.nspname) || '.' || pg_catalog.quote_ident(co.collname)
           ELSE ''
       END AS resolved_type
FROM pg_catalog.pg_attribute a
JOIN pg_catalog.pg_class cl ON cl.oid = a.attrelid
JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
JOIN unwrapped u ON u.oid = t.oid AND u.typtype <> 'd'
LEFT JOIN pg_catalog.pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
LEFT JOIN pg_catalog.pg_collation co ON co.oid = a.attcollation
LEFT JOIN pg_catalog.pg_namespace cn ON cn.oid = co.collnamespace
WHERE pg_catalog.pg_table_is_visible(cl.oid)
  AND cl.relkind IN ('r', 'p')
  AND a.attgenerated IN ('v', 's')
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND cl.relname NOT LIKE '__matview_%'
  AND cl.relname NOT LIKE '__mv_%'
"""


def _pg_introspect_generated_columns(
    conn_or_bind: Any,
) -> Dict[Tuple[str, str], Tuple[str, Optional[str], Optional[str]]]:
    """Return ``(table_lower, col_lower) -> (attgenerated, generation_expr, resolved_type)``
    for PostgreSQL generated columns (``'v'`` virtual on PG 18+, ``'s'`` stored
    on PG 12+) of visible regular and partitioned tables.

    ``resolved_type`` uses ``pg_catalog.format_type`` (unwrapping domains and
    preserving exact-case explicit collations) so virtual column casts and
    backing DDL never receive ``USER-DEFINED`` or uppercased collation names.
    """
    rows: List[Any] = []
    try:
        if isinstance(conn_or_bind, _SAEngine):
            with conn_or_bind.connect() as sa_conn:
                rows = list(sa_conn.exec_driver_sql(_PG_GENERATED_COLUMNS_SQL).fetchall() or ())
        elif isinstance(conn_or_bind, _SAConnection):
            if conn_or_bind.in_transaction():
                with conn_or_bind.begin_nested():
                    rows = list(conn_or_bind.exec_driver_sql(_PG_GENERATED_COLUMNS_SQL).fetchall() or ())
            else:
                rows = list(conn_or_bind.exec_driver_sql(_PG_GENERATED_COLUMNS_SQL).fetchall() or ())
        elif hasattr(conn_or_bind, "cursor"):
            cur = conn_or_bind.cursor()
            used_sp = False
            try:
                if dbapi_transaction_state(conn_or_bind) is True:
                    cur.execute("SAVEPOINT __mv_pg_gen_introspect")
                    used_sp = True
                cur.execute(_PG_GENERATED_COLUMNS_SQL)
                rows = list(cur.fetchall() or ())
                if used_sp:
                    cur.execute("RELEASE SAVEPOINT __mv_pg_gen_introspect")
            except Exception:
                if used_sp:
                    try:
                        cur.execute("ROLLBACK TO SAVEPOINT __mv_pg_gen_introspect")
                        cur.execute("RELEASE SAVEPOINT __mv_pg_gen_introspect")
                    except Exception:
                        pass
                return {}
            finally:
                close_fn = getattr(cur, "close", None)
                if callable(close_fn):
                    close_fn()
    except Exception:
        return {}
    out: Dict[Tuple[str, str], Tuple[str, Optional[str], Optional[str]]] = {}
    for r in rows:
        if len(r) >= 4 and r[0] and r[1] and r[2]:
            kind = str(r[2]).strip().lower()
            expr = str(r[3]) if r[3] is not None else None
            resolved_dt = str(r[4]) if len(r) >= 5 and r[4] else None
            out[(str(r[0]).lower(), str(r[1]).lower())] = (kind, expr, resolved_dt)
    return out


def introspect_schemas_with_sqlalchemy(
    bind_or_conn: Any,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> Dict[str, TableSchema]:
    """
    Introspect base table schemas via `sqlalchemy.inspect(...)` across SQLite,
    PostgreSQL, and MySQL/MariaDB, enriching FK cascade metadata where needed.
    """
    eff_dialect: Optional[Dialect] = None
    if isinstance(dialect, Dialect):
        eff_dialect = dialect
    elif isinstance(dialect, EngineVersion):
        eff_dialect = dialect.dialect
    elif isinstance(dialect, str):
        eff_dialect = parse_engine_version(dialect).dialect
    else:
        eff_dialect = _detect_engine_version(bind_or_conn).dialect

    catalog_table = f"{object_prefix}catalog" if object_prefix else "matview_catalog"
    owned_engine: Optional[_SAEngine] = None
    raw_conn, _, _, _ = unwrap_dbapi_connection(bind_or_conn)

    if isinstance(bind_or_conn, (_SAEngine, _SAConnection)):
        insp_target = bind_or_conn
    else:
        url_str = {
            Dialect.POSTGRES: "postgresql+psycopg2://",
            Dialect.MYSQL: "mysql+pymysql://",
        }.get(eff_dialect, "sqlite://")
        owned_engine = _sa_create_engine(
            url_str,
            creator=lambda: _NonClosingConnectionProxy(raw_conn),
            poolclass=_SANullPool,
            pool_reset_on_return=None,
        )
        # Skip `create_engine`'s `on_connect` listener (e.g. `psycopg2.extras.register_uuid`)
        # which type-checks the raw C connection and rejects `_NonClosingConnectionProxy`.
        for _conn_fn in list(owned_engine.pool.dispatch.connect):
            if getattr(_conn_fn, "__name__", "") == "on_connect":
                _sa_event.remove(owned_engine.pool, "connect", _conn_fn)
        insp_target = owned_engine

    try:
        insp = _sa_inspect(insp_target)
        sqlite_ddl_schemas: Dict[str, TableSchema] = {}
        if eff_dialect == Dialect.SQLITE and callable(getattr(raw_conn, "cursor", None)):
            sqlite_ddl_schemas = introspect_sqlite_schemas(raw_conn, object_prefix=object_prefix)
        sqlite_low = {k.lower(): v for k, v in sqlite_ddl_schemas.items()}
        pg_array_types: Dict[Tuple[str, str], str] = (
            _pg_introspect_native_array_types(bind_or_conn if isinstance(bind_or_conn, (_SAEngine, _SAConnection)) else raw_conn)
            if eff_dialect == Dialect.POSTGRES
            else {}
        )
        pg_gen_cols: Dict[Tuple[str, str], Tuple[str, Optional[str], Optional[str]]] = (
            _pg_introspect_generated_columns(bind_or_conn if isinstance(bind_or_conn, (_SAEngine, _SAConnection)) else raw_conn)
            if eff_dialect == Dialect.POSTGRES
            else {}
        )

        schemas: Dict[str, TableSchema] = {}
        for t_name in insp.get_table_names():
            if t_name.startswith(("__matview_", "__mv_")) or (
                object_prefix and t_name.startswith(object_prefix)
            ) or t_name == catalog_table:
                continue

            if eff_dialect == Dialect.SQLITE:
                sq_existing = sqlite_low.get(t_name.lower())
                if sq_existing and sq_existing.columns:
                    schemas[sq_existing.name] = sq_existing
                    continue

            pk_info = insp.get_pk_constraint(t_name) or {}
            pk_cols_seq = tuple(str(c) for c in (pk_info.get("constrained_columns") or ()))
            pk_cols_low = {c.lower() for c in pk_cols_seq}
            cols_info = insp.get_columns(t_name) or []
            col_defs: Dict[str, ColumnDef] = {}
            for col in cols_info:
                c_name = str(col["name"])
                native_arr = pg_array_types.get((t_name.lower(), c_name.lower()))
                c_type = (
                    native_arr
                    if native_arr is not None
                    else _format_sa_column_type(col.get("type", "TEXT"), eff_dialect)
                )
                is_pk = c_name.lower() in pk_cols_low
                is_notnull = (not bool(col.get("nullable", True))) or is_pk
                gen_info = pg_gen_cols.get((t_name.lower(), c_name.lower()))
                if gen_info is not None and gen_info[2]:
                    c_type = gen_info[2]
                col_defs[c_name] = ColumnDef(
                    name=c_name,
                    data_type=c_type,
                    not_null=is_notnull,
                    primary_key=is_pk,
                    generated=gen_info is not None,
                    generated_virtual=(gen_info is not None and gen_info[0] == "v"),
                    generation_expr=(gen_info[1] if gen_info is not None else None),
                )

            fk_actions: List[str] = []
            for fk in (insp.get_foreign_keys(t_name) or []):
                opts = fk.get("options") or {}
                for opt_key, sql_kw in (("ondelete", "ON DELETE"), ("onupdate", "ON UPDATE")):
                    act_val = str(opts.get(opt_key) or "").strip().upper()
                    if act_val == "CASCADE":
                        norm_act = "CASCADE"
                    elif act_val == "SET NULL" or act_val.startswith(("SET NULL (", "SET NULL(")):
                        norm_act = "SET NULL"
                    elif act_val == "SET DEFAULT" or act_val.startswith(("SET DEFAULT (", "SET DEFAULT(")):
                        norm_act = "SET DEFAULT"
                    else:
                        norm_act = ""
                    if norm_act:
                        act_str = f"{sql_kw} {norm_act}"
                        if act_str not in fk_actions:
                            fk_actions.append(act_str)

            idx_defs: List[TableIndexDef] = []
            seen_idx_names = set()
            try:
                for raw_ix in (insp.get_indexes(t_name) or []):
                    ix_cols = tuple(str(c) for c in (raw_ix.get("column_names") or ()) if c)
                    ix_name = str(raw_ix.get("name") or f"idx_{t_name}_{'_'.join(ix_cols)}")
                    if not ix_cols or ix_name.lower() in seen_idx_names:
                        continue
                    seen_idx_names.add(ix_name.lower())
                    d_opts = raw_ix.get("dialect_options") or {}
                    where_c = (
                        d_opts.get("postgresql_where")
                        or d_opts.get("sqlite_where")
                        or None
                    )
                    idx_defs.append(
                        TableIndexDef(
                            name=ix_name,
                            table_name=t_name,
                            columns=ix_cols,
                            unique=bool(raw_ix.get("unique")),
                            where_clause=str(where_c) if where_c else None,
                        )
                    )
            except Exception:
                pass
            try:
                for raw_uq in (insp.get_unique_constraints(t_name) or []):
                    uq_cols = tuple(str(c) for c in (raw_uq.get("column_names") or ()) if c)
                    uq_name = str(raw_uq.get("name") or f"uq_{t_name}_{'_'.join(uq_cols)}")
                    if not uq_cols or uq_name.lower() in seen_idx_names:
                        continue
                    seen_idx_names.add(uq_name.lower())
                    idx_defs.append(
                        TableIndexDef(
                            name=uq_name,
                            table_name=t_name,
                            columns=uq_cols,
                            unique=True,
                        )
                    )
            except Exception:
                pass

            schemas[t_name] = TableSchema(
                name=t_name,
                columns=col_defs,
                fk_cascade_actions=tuple(sorted(fk_actions)),
                indexes=tuple(idx_defs),
                pk_columns=pk_cols_seq,
            )
        return schemas
    finally:
        if owned_engine is not None:
            owned_engine.dispose()


def _sqlite_table_list_strict_flags(cur: Any) -> Dict[str, bool]:
    """`{lower(table): STRICT?}` from `PRAGMA table_list`, or `{}` when the library lacks it (< 3.37)."""
    try:
        cur.execute("PRAGMA table_list")
        rows = cur.fetchall() or []
    except Exception:
        return {}
    out: Dict[str, bool] = {}
    for r in rows:
        # (schema, name, type, ncol, wr, strict)
        if len(r) >= 6 and str(r[0]).lower() == "main" and str(r[2]).lower() == "table":
            out[str(r[1]).lower()] = bool(r[5])
    return out


def introspect_sqlite_schemas(
    conn: Any,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> Dict[str, TableSchema]:
    """Introspects all non-internal base tables from a live SQLite connection."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT type, name, sql, tbl_name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%'"
        )
        rows = cur.fetchall() or []
        ddl_parts: List[str] = []
        table_names: List[str] = []
        table_sqls: Dict[str, str] = {}
        for obj_type, name, sql, tbl_name in rows:
            if not name:
                continue
            owner_name = str(tbl_name or name)
            t_str = str(name)
            if (
                owner_name.startswith(("__matview_", "__mv_"))
                or t_str.startswith(("__matview_", "__mv_"))
                or (object_prefix and (owner_name.startswith(object_prefix) or t_str.startswith(object_prefix)))
            ):
                continue
            if str(obj_type).lower() == "table":
                table_names.append(t_str)
                if sql:
                    table_sqls[t_str.lower()] = str(sql)
            if sql:
                ddl_parts.append(str(sql).rstrip(";") + ";")
        schemas = (
            parse_table_schema("\n".join(ddl_parts), dialect=Dialect.SQLITE)
            if ddl_parts
            else {}
        )
        low_known = {k.lower(): k for k in schemas}

        # STRICT-ness: `PRAGMA table_list` (SQLite 3.37+) is authoritative; the CREATE TABLE text
        # parsed above is the fallback (older libraries have no STRICT tables and no such pragma).
        strict_flags = _sqlite_table_list_strict_flags(cur)
        for known_key, is_strict in strict_flags.items():
            parsed_key = low_known.get(known_key)
            parsed_ts = schemas.get(parsed_key) if parsed_key else None
            if parsed_key is not None and parsed_ts is not None and parsed_ts.strict != is_strict:
                schemas[parsed_key] = dataclasses.replace(parsed_ts, strict=is_strict)

        # Enrich parsed tables with authoritative live FK actions from PRAGMA foreign_key_list
        # (fail-closed: never swallow FK metadata query errors), and fall back to PRAGMA
        # table_info / index_list / foreign_key_list for tables without standard DDL.
        def _read_sqlite_fk_actions(table_name: str) -> List[str]:
            t_q = table_name.replace('"', '""')
            fk_list: List[str] = []
            cur.execute(f'PRAGMA foreign_key_list("{t_q}")')
            for fk_r in (cur.fetchall() or []):
                on_upd = str(fk_r[5]).upper() if len(fk_r) > 5 and fk_r[5] else ""
                on_del = str(fk_r[6]).upper() if len(fk_r) > 6 and fk_r[6] else ""
                if on_del in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    act = f"ON DELETE {on_del}"
                    if act not in fk_list:
                        fk_list.append(act)
                if on_upd in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    act = f"ON UPDATE {on_upd}"
                    if act not in fk_list:
                        fk_list.append(act)
            return fk_list

        for t_str in table_names:
            known_key = low_known.get(t_str.lower(), t_str)
            existing = schemas.get(known_key)
            if existing and existing.columns:
                live_fks = tuple(sorted(set(_read_sqlite_fk_actions(t_str))))
                if live_fks != existing.fk_cascade_actions:
                    schemas[known_key] = TableSchema(
                        name=existing.name,
                        columns=existing.columns,
                        fk_cascade_actions=live_fks,
                        has_on_conflict_replace=existing.has_on_conflict_replace,
                        indexes=existing.indexes,
                        pk_columns=existing.pk_columns,
                        without_rowid=existing.without_rowid,
                        strict=existing.strict,
                    )
                continue
            t_q = t_str.replace('"', '""')
            try:
                cur.execute(f'PRAGMA table_info("{t_q}")')
                t_info = cur.fetchall() or []
            except Exception:
                continue
            if not t_info:
                continue
            pk_entries = sorted(((int(r[5]), str(r[1])) for r in t_info if int(r[5] or 0) > 0), key=lambda x: x[0])
            pk_cols_seq = tuple(name for _, name in pk_entries)
            pk_count = len(pk_cols_seq)
            raw_tbl_sql = _strip_sql_comments(table_sqls.get(t_str.lower(), ""))
            rparen_idx = raw_tbl_sql.rfind(")")
            fb_tail = raw_tbl_sql[rparen_idx + 1 :] if rparen_idx >= 0 else ""
            fb_without_rowid = bool(re.search(r"\bWITHOUT\s+ROWID\b", fb_tail, re.IGNORECASE))
            fb_strict = bool(re.search(r"\bSTRICT\b", fb_tail, re.IGNORECASE))
            fb_pk_desc = bool(
                re.search(
                    r"\bPRIMARY\s+KEY\s+DESC\b|\bPRIMARY\s+KEY\s*\([^)]*\bDESC\b",
                    raw_tbl_sql,
                    re.IGNORECASE,
                )
            )
            cols: Dict[str, ColumnDef] = {}
            for r in t_info:
                c_name = str(r[1])
                c_type = str(r[2]).upper() if r[2] else "BLOB"  # no declared type: BLOB affinity
                c_pk = bool(r[5])
                c_rowid_alias = (
                    c_pk
                    and pk_count == 1
                    and not fb_without_rowid
                    and not fb_pk_desc
                    and re.sub(r"\s+", " ", c_type.strip().upper()) == "INTEGER"
                )
                c_notnull = bool(r[3]) or (c_pk and (fb_without_rowid or c_rowid_alias))
                cols[c_name] = ColumnDef(
                    name=c_name,
                    data_type=c_type,
                    not_null=c_notnull,
                    primary_key=c_pk,
                    is_rowid_alias=c_rowid_alias,
                )
            fk_actions = _read_sqlite_fk_actions(t_str)
            sq_idx_defs: List[TableIndexDef] = []
            try:
                cur.execute(f'PRAGMA index_list("{t_q}")')
                for il_row in (cur.fetchall() or []):
                    ix_name = str(il_row[1])
                    ix_unique = bool(il_row[2])
                    ix_origin = str(il_row[3]) if len(il_row) > 3 and il_row[3] else ""
                    if ix_origin == "pk":
                        continue
                    ix_q = ix_name.replace('"', '""')
                    cur.execute(f'PRAGMA index_info("{ix_q}")')
                    ii_rows = sorted(
                        (r for r in (cur.fetchall() or []) if len(r) > 2 and r[2] is not None),
                        key=lambda r: int(r[0]),
                    )
                    ix_cols = tuple(str(r[2]) for r in ii_rows)
                    if ix_cols:
                        sq_idx_defs.append(
                            TableIndexDef(
                                name=ix_name,
                                table_name=t_str,
                                columns=ix_cols,
                                unique=ix_unique,
                            )
                        )
            except Exception:
                pass
            schemas[t_str] = TableSchema(
                name=t_str,
                columns=cols,
                fk_cascade_actions=tuple(sorted(fk_actions)),
                indexes=tuple(sq_idx_defs),
                pk_columns=pk_cols_seq,
                without_rowid=fb_without_rowid,
                strict=strict_flags.get(t_str.lower(), fb_strict),
            )
        return schemas
    finally:
        close_fn = getattr(cur, "close", None)
        if callable(close_fn):
            close_fn()


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
        dialect = _detect_engine_version(conn).dialect
    elif isinstance(dialect, EngineVersion):
        dialect = dialect.dialect
    elif isinstance(dialect, str):
        dialect = parse_engine_version(dialect).dialect
    catalog_table = f"{object_prefix}catalog" if object_prefix else "matview_catalog"

    if _can_use_sqlalchemy_inspector(conn):
        schemas = introspect_schemas_with_sqlalchemy(
            conn,
            dialect=dialect,
            object_prefix=object_prefix,
        )
        schemas.pop(catalog_table, None)
        return schemas

    if dialect == Dialect.SQLITE:
        schemas = introspect_sqlite_schemas(conn, object_prefix=object_prefix)
        schemas.pop(catalog_table, None)
        return schemas

    schemas = {}
    cur = conn.cursor()
    try:
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
                ORDER BY kcu.table_name, kcu.ordinal_position
                """
            )
            pk_rows = cur.fetchall() or []
            pk_pairs = {(str(r[0]).lower(), str(r[1]).lower()) for r in pk_rows}
            pk_order_by_table: Dict[str, List[str]] = {}
            for r in pk_rows:
                pk_order_by_table.setdefault(str(r[0]).lower(), []).append(str(r[1]))
            pg_idx_by_table: Dict[str, List[TableIndexDef]] = {}
            try:
                cur.execute(
                    """
                    SELECT tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND tablename NOT LIKE '__matview_%'
                      AND tablename NOT LIKE '__mv_%'
                    ORDER BY tablename, indexname
                    """
                )
                for ix_r in (cur.fetchall() or []):
                    t_nm, ix_nm, ix_def = str(ix_r[0]), str(ix_r[1]), str(ix_r[2] or "")
                    if not ix_def:
                        continue
                    parsed_ix = parse_table_index_def(ix_def, dialect=Dialect.POSTGRES)
                    if parsed_ix is not None:
                        pg_idx_by_table.setdefault(t_nm.lower(), []).append(
                            TableIndexDef(
                                name=ix_nm,
                                table_name=t_nm,
                                columns=parsed_ix.columns,
                                unique=parsed_ix.unique,
                                where_clause=parsed_ix.where_clause,
                            )
                        )
            except Exception:
                pass
            def _norm_pg_fk_action(val: Any) -> str:
                if isinstance(val, memoryview):
                    val = val.tobytes()
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode("utf-8")
                s_val = str(val or "").strip().upper()
                return {
                    "C": "CASCADE",
                    "N": "SET NULL",
                    "D": "SET DEFAULT",
                    "CASCADE": "CASCADE",
                    "SET NULL": "SET NULL",
                    "SET DEFAULT": "SET DEFAULT",
                }.get(s_val, "")

            # Query pg_catalog.pg_constraint directly (visible to non-owner roles,
            # unlike information_schema.referential_constraints) and fail closed
            # rather than silently swallowing FK metadata query errors.
            fk_by_table: Dict[str, List[str]] = {}
            cur.execute(
                """
                SELECT c.relname, con.confdeltype, con.confupdtype
                FROM pg_catalog.pg_constraint con
                JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE con.contype = 'f'
                  AND n.nspname = current_schema()
                """
            )
            for fk_r in (cur.fetchall() or []):
                t_low = str(fk_r[0]).lower()
                del_r = _norm_pg_fk_action(fk_r[1])
                upd_r = _norm_pg_fk_action(fk_r[2])
                acts = fk_by_table.setdefault(t_low, [])
                if del_r:
                    a_str = f"ON DELETE {del_r}"
                    if a_str not in acts:
                        acts.append(a_str)
                if upd_r:
                    a_str = f"ON UPDATE {upd_r}"
                    if a_str not in acts:
                        acts.append(a_str)

            pg_array_types = _pg_introspect_native_array_types(conn)
            pg_gen_cols = _pg_introspect_generated_columns(conn)
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
                native_arr = pg_array_types.get((t_name.lower(), c_name.lower()))
                if native_arr is not None:
                    dt_up = native_arr
                else:
                    dt_up = str(d_type).upper()
                    if dt_up in ("CHARACTER VARYING", "VARCHAR") and char_len:
                        dt_up = f"VARCHAR({int(char_len)})"
                    elif dt_up == "CHARACTER VARYING":
                        dt_up = "VARCHAR"
                    elif dt_up == "BIT" and char_len:
                        dt_up = f"BIT({int(char_len)})"
                    elif dt_up == "BIT VARYING" and char_len:
                        dt_up = f"BIT VARYING({int(char_len)})"
                is_pk = (t_name.lower(), c_name.lower()) in pk_pairs
                gen_info = pg_gen_cols.get((t_name.lower(), c_name.lower()))
                if gen_info is not None and gen_info[2]:
                    dt_up = gen_info[2]
                grouped.setdefault(t_name, {})[c_name] = ColumnDef(
                    name=c_name,
                    data_type=dt_up,
                    not_null=(str(is_null).upper() == "NO") or is_pk,
                    primary_key=is_pk,
                    generated=gen_info is not None,
                    generated_virtual=(gen_info is not None and gen_info[0] == "v"),
                    generation_expr=(gen_info[1] if gen_info is not None else None),
                )
            for t_name, cols in grouped.items():
                schemas[t_name] = TableSchema(
                    name=t_name,
                    columns=cols,
                    fk_cascade_actions=tuple(sorted(fk_by_table.get(t_name.lower(), ()))),
                    indexes=tuple(pg_idx_by_table.get(t_name.lower(), ())),
                    pk_columns=tuple(pk_order_by_table.get(t_name.lower(), ())),
                )
            return schemas
        # MySQL / MariaDB
        def _dec(val: Any) -> str:
            if isinstance(val, (bytes, bytearray)):
                return val.decode("utf-8")
            return "" if val is None else str(val)

        # Fail closed when FK metadata cannot be read: swallowing errors here
        # would silently bypass MYSQL_FK_CASCADE_NO_TRIGGER.
        fk_by_table_my: Dict[str, List[str]] = {}
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

        cur.execute(
            """
            SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_key, c.column_type,
                   c.collation_name
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
            # information_schema.DATA_TYPE drops numeric precision/scale and
            # integer signedness. COLUMN_TYPE is the native declaration (for
            # example DECIMAL(20,12)); retain it for every numeric family so AVG
            # and expression result types match the live server metadata.
            if dt_up in (
                "VARCHAR", "CHAR", "TINYINT", "DECIMAL", "NUMERIC", "DEC", "FIXED",
                "INT", "INTEGER", "SMALLINT", "MEDIUMINT", "BIGINT", "FLOAT", "DOUBLE", "REAL",
            ) and col_type:
                dt_up = col_type.upper()
            is_pk = col_key.upper() == "PRI"
            coll_name = _dec(row[6]) if len(row) > 6 and row[6] is not None else None
            grouped_my.setdefault(t_name, {})[c_name] = ColumnDef(
                name=c_name,
                data_type=dt_up,
                not_null=(str(is_null).upper() == "NO") or is_pk,
                primary_key=is_pk,
                collation=coll_name.upper() if coll_name else None,
            )
        my_pk_by_table: Dict[str, List[str]] = {}
        my_idx_by_table: Dict[str, List[TableIndexDef]] = {}
        try:
            cur.execute(
                """
                SELECT table_name, index_name, non_unique, column_name
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name NOT LIKE '__matview_%'
                  AND table_name NOT LIKE '__mv_%'
                ORDER BY table_name, index_name, seq_in_index
                """
            )
            raw_stats: Dict[Tuple[str, str], Tuple[bool, List[str]]] = {}
            for st_r in (cur.fetchall() or []):
                st_tbl = _dec(st_r[0])
                st_ix = _dec(st_r[1])
                st_non_uq = bool(int(st_r[2])) if st_r[2] is not None else True
                st_col = _dec(st_r[3])
                if not st_tbl or not st_ix or not st_col:
                    continue
                key = (st_tbl.lower(), st_ix)
                entry = raw_stats.setdefault(key, (not st_non_uq, []))
                entry[1].append(st_col)
            for (st_tbl_low, st_ix), (is_uq, col_list) in raw_stats.items():
                if st_ix.upper() == "PRIMARY":
                    my_pk_by_table[st_tbl_low] = col_list
                else:
                    my_idx_by_table.setdefault(st_tbl_low, []).append(
                        TableIndexDef(
                            name=st_ix,
                            table_name=st_tbl_low,
                            columns=tuple(col_list),
                            unique=is_uq,
                        )
                    )
        except Exception:
            pass

        for t_name, cols in grouped_my.items():
            schemas[t_name] = TableSchema(
                name=t_name,
                columns=cols,
                fk_cascade_actions=tuple(sorted(fk_by_table_my.get(t_name.lower(), ()))),
                indexes=tuple(my_idx_by_table.get(t_name.lower(), ())),
                pk_columns=tuple(my_pk_by_table.get(t_name.lower(), ())),
            )
        return schemas
    finally:
        close_fn = getattr(cur, "close", None)
        if callable(close_fn):
            close_fn()


def _exec_plain_view_query(
    conn_or_bind: Any,
    sql: str,
    savepoint_name: str,
    *,
    ignore_missing_relation: bool = False,
) -> List[Any]:
    """Run a read-only view-introspection query safely across SQLAlchemy and DB-API.

    Propagates real database/permission errors fail-closed (never silently
    returning empty results on failure) while keeping idle connections idle
    and protecting open transactions via a SAVEPOINT.
    """
    def _is_missing_rel(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "no such table" in msg or "does not exist" in msg

    if isinstance(conn_or_bind, _SAEngine):
        try:
            with conn_or_bind.connect() as sa_conn:
                return list(sa_conn.exec_driver_sql(sql).fetchall() or ())
        except Exception as exc:
            if ignore_missing_relation and _is_missing_rel(exc):
                return []
            raise
    if isinstance(conn_or_bind, _SAConnection):
        started_in_tx = bool(conn_or_bind.in_transaction())
        try:
            if started_in_tx:
                with conn_or_bind.begin_nested():
                    return list(conn_or_bind.exec_driver_sql(sql).fetchall() or ())
            return list(conn_or_bind.exec_driver_sql(sql).fetchall() or ())
        except Exception as exc:
            if ignore_missing_relation and _is_missing_rel(exc):
                return []
            raise
        finally:
            if not started_in_tx and conn_or_bind.in_transaction():
                try:
                    conn_or_bind.rollback()
                except Exception:
                    pass
    if hasattr(conn_or_bind, "cursor"):
        cur = conn_or_bind.cursor()
        tx_state = dbapi_transaction_state(conn_or_bind)
        started_idle = tx_state is False
        used_sp = False
        try:
            if tx_state is True:
                cur.execute(f"SAVEPOINT {savepoint_name}")
                used_sp = True
            cur.execute(sql)
            rows = list(cur.fetchall() or ())
            if used_sp:
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            return rows
        except Exception as exc:
            if used_sp:
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except Exception:
                    pass
            elif started_idle:
                rb = getattr(conn_or_bind, "rollback", None)
                if callable(rb):
                    try:
                        rb()
                    except Exception:
                        pass
            if ignore_missing_relation and _is_missing_rel(exc):
                return []
            raise
        finally:
            close_fn = getattr(cur, "close", None)
            if callable(close_fn):
                close_fn()
            if started_idle and dbapi_transaction_state(conn_or_bind) is True:
                rb = getattr(conn_or_bind, "rollback", None)
                if callable(rb):
                    try:
                        rb()
                    except Exception:
                        pass
    return []


def introspect_plain_views(
    conn: Any,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    exclude_views: Sequence[str] = (),
) -> Tuple[Dict[str, str], Optional[str]]:
    """Introspect non-internal plain (non-materialized) SQL views from a live database.

    Returns ``(plain_views, current_schema)`` mapping each plain view name to
    its SQL definition (``CREATE VIEW ...`` on SQLite, ``pg_get_viewdef``
    SELECT on PostgreSQL, ``VIEW_DEFINITION`` SELECT on MySQL/MariaDB), plus
    the active schema/database name when reported by the engine.

    Registered materialized views (``exclude_views``), internal
    ``object_prefix`` / ``__matview_`` / ``__mv_`` / ``sqlite_`` objects, and
    PostgreSQL native ``MATERIALIZED VIEW``s (``relkind = 'm'``) are excluded.
    """
    if dialect is None:
        eff_dialect = _detect_engine_version(conn).dialect
    elif isinstance(dialect, EngineVersion):
        eff_dialect = dialect.dialect
    elif isinstance(dialect, str):
        eff_dialect = parse_engine_version(dialect).dialect
    else:
        eff_dialect = dialect

    excluded_low = {str(n).lower() for n in exclude_views if n}
    if object_prefix:
        for n in exclude_views:
            if n:
                n_low = str(n).lower()
                if n_low.startswith(object_prefix.lower()):
                    excluded_low.add(n_low[len(object_prefix) :])
                else:
                    excluded_low.add(f"{object_prefix.lower()}{n_low}")

    def _dec(val: Any) -> str:
        if isinstance(val, memoryview):
            val = val.tobytes()
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8")
        return "" if val is None else str(val)

    def _should_skip(name: str) -> bool:
        if not name:
            return True
        low = name.lower()
        if low in excluded_low:
            return True
        if low.startswith(("__matview_", "__mv_", "sqlite_")):
            return True
        if object_prefix and low.startswith(object_prefix.lower()):
            return True
        return False

    views: Dict[str, str] = {}
    current_schema: Optional[str] = None

    if eff_dialect == Dialect.SQLITE:
        current_schema = "main"
        for sch_prefix, tbl_cat, sp, ign_miss in (
            ("main", "sqlite_master", "__mv_sq_pv_main", False),
            ("temp", "sqlite_temp_master", "__mv_sq_pv_temp", True),
        ):
            rows = _exec_plain_view_query(
                conn,
                f"SELECT name, sql FROM {tbl_cat} WHERE type = 'view' AND sql IS NOT NULL ORDER BY name",
                sp,
                ignore_missing_relation=ign_miss,
            )
            for row in rows:
                if len(row) < 2:
                    continue
                v_name = _dec(row[0])
                v_sql = _dec(row[1]).strip()
                if _should_skip(v_name) or not v_sql:
                    continue
                # Explicit schema-qualified key (`main.<v>` / `temp.<v>`) preserves
                # exact namespace lookup, while unqualified `<v>` lets TEMP shadow main.
                views[f"{sch_prefix}.{v_name}"] = v_sql
                for existing_k in [k for k in views if k.lower() == v_name.lower()]:
                    views.pop(existing_k, None)
                views[v_name] = v_sql
        return views, current_schema

    if eff_dialect == Dialect.POSTGRES:
        rows = _exec_plain_view_query(
            conn,
            """
            SELECT current_schema(), c.relname, pg_catalog.pg_get_viewdef(c.oid, true)
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'v'
              AND n.nspname = current_schema()
            ORDER BY c.relname
            """,
            "__mv_pg_pv_introspect",
        )
        for row in rows:
            if len(row) < 3:
                continue
            sch = _dec(row[0]).strip()
            if sch and current_schema is None:
                current_schema = sch
            v_name = _dec(row[1])
            v_sql = _dec(row[2]).strip().rstrip(";").strip()
            if _should_skip(v_name) or not v_sql:
                continue
            views[v_name] = v_sql
        return views, current_schema

    # MySQL / MariaDB
    rows = _exec_plain_view_query(
        conn,
        """
        SELECT DATABASE(), TABLE_NAME, VIEW_DEFINITION
        FROM information_schema.VIEWS
        WHERE TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME
        """,
        "__mv_my_pv_introspect",
    )
    for row in rows:
        if len(row) < 3:
            continue
        db_name = _dec(row[0]).strip()
        if db_name and current_schema is None:
            current_schema = db_name
        v_name = _dec(row[1])
        v_sql = _dec(row[2]).strip().rstrip(";").strip()
        if _should_skip(v_name) or not v_sql:
            continue
        views[v_name] = v_sql
    return views, current_schema

