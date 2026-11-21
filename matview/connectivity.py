"""SQLAlchemy URL normalization, connection/proxy management, and cross-engine schema introspection."""

from __future__ import annotations

import copy
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

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
    parse_engine_version,
)
from .logical_ir import ColumnDef, TableIndexDef, TableSchema
from .sql_frontend import parse_table_index_def, parse_table_schema

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

    unwrapped_conn, _, _, sa_dialect = unwrap_dbapi_connection(conn)
    if unwrapped_conn is not conn:
        return detect_engine_version(
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
                            Dialect.MYSQL, EngineSubdialect.MARIADB, ver, v_str, div_increment
                        )
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
                    )
                pkg_v, ver = _extract_packaging_and_tuple(v_str, (8, 0, 0))
                return EngineVersion(
                    Dialect.MYSQL,
                    EngineSubdialect.MYSQL,
                    ver,
                    v_str,
                    packaging_version=pkg_v,
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


def _format_sa_column_type(col_type: Any, dialect: Dialect) -> str:
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
    elif from_domain and raw_str.startswith("VARCHAR"):
        raw_str = "VARCHAR"
    elif from_domain and raw_str.startswith("CHAR") and not raw_str.startswith("CHARACTER VARYING"):
        # CHAR without its reflected length would be CHAR(1); BPCHAR is the
        # unbounded blank-padded type (same result type as CHAR(n)).
        raw_str = "BPCHAR"
    elif dialect == Dialect.POSTGRES and raw_str.startswith("VARCHAR") and not getattr(col_type, "length", None):
        raw_str = "TEXT"
    elif dialect == Dialect.POSTGRES and getattr(col_type, "timezone", False) and raw_str in ("TIMESTAMP", "TIME"):
        # SQLAlchemy prints TIMESTAMP(timezone=True) as plain TIMESTAMP.
        raw_str = f"{raw_str} WITH TIME ZONE"
    if getattr(col_type, "unsigned", False) and "UNSIGNED" not in raw_str:
        raw_str = f"{raw_str} UNSIGNED"
    if domain_collation:
        raw_str = f'{raw_str} COLLATE "{domain_collation}"'
    return raw_str


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
        eff_dialect = detect_engine_version(bind_or_conn).dialect

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
        if eff_dialect == Dialect.SQLITE:
            try:
                sqlite_ddl_schemas = introspect_sqlite_schemas(raw_conn, object_prefix=object_prefix)
            except Exception:
                sqlite_ddl_schemas = {}
        sqlite_low = {k.lower(): v for k, v in sqlite_ddl_schemas.items()}

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
                c_type = _format_sa_column_type(col.get("type", "TEXT"), eff_dialect)
                is_pk = c_name.lower() in pk_cols_low
                is_notnull = (not bool(col.get("nullable", True))) or is_pk
                col_defs[c_name] = ColumnDef(
                    name=c_name,
                    data_type=c_type,
                    not_null=is_notnull,
                    primary_key=is_pk,
                )

            fk_actions: List[str] = []
            for fk in (insp.get_foreign_keys(t_name) or []):
                opts = fk.get("options") or {}
                for opt_key, sql_kw in (("ondelete", "ON DELETE"), ("onupdate", "ON UPDATE")):
                    act_val = str(opts.get(opt_key) or "").upper()
                    if act_val in ("CASCADE", "SET NULL", "SET DEFAULT"):
                        act_str = f"{sql_kw} {act_val}"
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
                fk_cascade_actions=tuple(fk_actions),
                indexes=tuple(idx_defs),
                pk_columns=pk_cols_seq,
            )
        return schemas
    finally:
        if owned_engine is not None:
            owned_engine.dispose()


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
                ddl_parts.append(str(sql).rstrip(";") + ";")
        schemas = (
            parse_table_schema("\n".join(ddl_parts), dialect=Dialect.SQLITE)
            if ddl_parts
            else {}
        )
        low_known = {k.lower(): k for k in schemas}

        # Fallback / enrichment via PRAGMA table_info, index_list & foreign_key_list for tables without standard DDL
        for t_str in table_names:
            existing = schemas.get(low_known.get(t_str.lower(), t_str))
            if existing and existing.columns:
                continue
            try:
                cur.execute(f'PRAGMA table_info("{t_str}")')
                t_info = cur.fetchall() or []
                if not t_info:
                    continue
                pk_entries = sorted(((int(r[5]), str(r[1])) for r in t_info if int(r[5] or 0) > 0), key=lambda x: x[0])
                pk_cols_seq = tuple(name for _, name in pk_entries)
                pk_count = len(pk_cols_seq)
                cols: Dict[str, ColumnDef] = {}
                for r in t_info:
                    c_name = str(r[1])
                    c_type = str(r[2]).upper() if r[2] else "BLOB"  # no declared type: BLOB affinity
                    c_pk = bool(r[5])
                    c_notnull = bool(r[3]) or (
                        c_pk and pk_count == 1 and re.sub(r"\s+", " ", c_type.strip()) == "INTEGER"
                    )
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
                sq_idx_defs: List[TableIndexDef] = []
                try:
                    cur.execute(f'PRAGMA index_list("{t_str}")')
                    for il_row in (cur.fetchall() or []):
                        ix_name = str(il_row[1])
                        ix_unique = bool(il_row[2])
                        ix_origin = str(il_row[3]) if len(il_row) > 3 and il_row[3] else ""
                        if ix_origin == "pk":
                            continue
                        cur.execute(f'PRAGMA index_info("{ix_name}")')
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
                    fk_cascade_actions=tuple(fk_actions),
                    indexes=tuple(sq_idx_defs),
                    pk_columns=pk_cols_seq,
                )
            except Exception:
                pass
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
        dialect = detect_engine_version(conn).dialect
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
                    indexes=tuple(pg_idx_by_table.get(t_name.lower(), ())),
                    pk_columns=tuple(pk_order_by_table.get(t_name.lower(), ())),
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
            grouped_my.setdefault(t_name, {})[c_name] = ColumnDef(
                name=c_name,
                data_type=dt_up,
                not_null=(str(is_null).upper() == "NO") or is_pk,
                primary_key=is_pk,
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
                fk_cascade_actions=tuple(fk_by_table_my.get(t_name.lower(), ())),
                indexes=tuple(my_idx_by_table.get(t_name.lower(), ())),
                pk_columns=tuple(my_pk_by_table.get(t_name.lower(), ())),
            )
        return schemas
    finally:
        close_fn = getattr(cur, "close", None)
        if callable(close_fn):
            close_fn()

