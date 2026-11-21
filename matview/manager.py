"""Database lifecycle manager: catalog, backfill, schema drift, migration, bulk changes, and diffs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import functools
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp
from sqlalchemy.engine import Connection as _SAConnection, Engine as _SAEngine

from .compiler import IncrementalViewCompiler
from .native_functions import (
    apply_catalog_classification,
    classify as classify_native_functions,
    items_with_unmodelled_calls,
    probe_result_types,
    unclassified_calls,
)
from .connectivity import (
    connect_database_via_sqlalchemy,
    dbapi_connection_layers,
    dbapi_transaction_state,
    detect_engine_version,
    introspect_database_schemas,
    render_parameterized_sql,
    unwrap_dbapi_connection,
)
from .engines import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    UnsupportedMaterializedViewError,
    _sqlglot_dialect,
    parse_engine_version,
)
from .logical_ir import (
    MaterializedViewPlan,
    TableSchema,
    ViewDiff,
    ViewDiffReport,
    ViewIndexSpec,
    ViewSpec,
)
from .sql_frontend import (
    _extract_created_object,
    _split_top_level,
    _strip_sql_comments,
    build_alter_rename_table_sql,
    build_create_view_sql,
    build_drop_sql,
    parse_sql_ast,
    parse_table_schema,
    parse_view_definition,
    parse_view_definitions,
    parse_view_index_spec,
    quote_sql_identifier,
)

# Bump whenever generated maintenance SQL changes in a way that existing
# installations must be rebuilt for (signature mismatch -> rebuild).
# 2: SQLite exact SUM/TOTAL/AVG maintenance (native recompute of groups
#    holding non-INTEGER values; join/UNION ALL native recompute).
# 3: SQLite UPDATE triggers also fire when only the storage class of a
#    referenced column changes (INTEGER 3 -> REAL 3.0 in a BLOB column).
# 4: SQLite no-affinity (BLOB) storage for MIN/MAX, group keys, DISTINCT aux
#    values, set-ops, and computed expressions; canonicalized numeric __mv_pk
#    keys and surviving group-key representative refresh.
# 5: Tier 2 outer-join group scoping refreshes matched and NULL-extended groups
#    (including COALESCE/expression keys, chained outer joins, self-joins, and
#    PostgreSQL statement-level triggers).
# 6: Bag-projection and nullable-PK UNION ALL reader views expand __mv_row_count
#    multiplicity, preserve exact SQLite/PostgreSQL row-identity keys, pre-aggregate
#    initial/chunk population, use non-null SQLite rowid for nullable-PK backfill,
#    and track all hashed columns in UPDATE triggers.
# 7: dialect-native statistical aliases and native result representation
#    (trigger/reader-only changes).
# 8: Watermark-gated backfill UPDATE triggers always track the watermark/backfill
#    range key (including unreferenced integer PKs and unshadowed SQLite rowid
#    aliases) and strictly validate integer PK / WITHOUT ROWID eligibility.
MATVIEW_GENERATOR_VERSION = "9"


def _definition_digest(select_sql: str) -> str:
    """Digest of a view's SELECT (comments and whitespace normalized), used to
    bind accepted limitation codes to the definition they were accepted for."""
    norm = re.sub(r"\s+", " ", _strip_sql_comments(select_sql or "")).strip().rstrip(";").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _stored_select_sql(view_sql: str) -> str:
    """The SELECT of a catalog ``view_sql`` (as written by
    ``_serialize_view_sql_with_indexes``: ``CREATE VIEW n AS <select>;``
    followed by optional ``CREATE INDEX`` lines)."""
    first = re.split(r";\s*\n\s*CREATE\s+(?:UNIQUE\s+)?INDEX\b", view_sql or "", maxsplit=1, flags=re.IGNORECASE)[0]
    m = re.match(r"\s*CREATE\s+VIEW\s+\S+\s+AS\s+(.*)$", first, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else first

# How to repopulate a view whose pk-range backfill is incomplete.
_REPAIR_GUIDANCE = (
    "to (re)populate from scratch use register_view(..., force_rebuild=True), "
    "drop_view() + register_view(), or repair_view()."
)


def _is_postgres_cte_in_scope(tbl_node: sqlglot_exp.Table) -> bool:
    """Return True if an unqualified table node resolves to an enclosing CTE
    under PostgreSQL scoping rules (non-recursive WITH CTEs see only earlier
    CTEs; WITH RECURSIVE CTEs see every CTE in the same WITH clause)."""
    if tbl_node.db or tbl_node.catalog:
        return False
    id_node = tbl_node.this
    if not isinstance(id_node, sqlglot_exp.Identifier):
        return False
    target = id_node.name if id_node.quoted else id_node.name.lower()
    prev_node: sqlglot_exp.Expr = tbl_node
    cur_node: Optional[sqlglot_exp.Expr] = tbl_node.parent
    cte_in_prev_with: Optional[sqlglot_exp.CTE] = None
    while cur_node is not None:
        if isinstance(cur_node, sqlglot_exp.With):
            cte_in_prev_with = prev_node if isinstance(prev_node, sqlglot_exp.CTE) else None
        with_clause = cur_node.args.get("with_")
        if isinstance(with_clause, sqlglot_exp.With):
            ctes = list(with_clause.expressions)
            if prev_node is with_clause and not bool(with_clause.args.get("recursive")):
                if cte_in_prev_with is not None and cte_in_prev_with in ctes:
                    visible = ctes[: ctes.index(cte_in_prev_with)]
                else:
                    visible = []
            else:
                visible = ctes
            for cte in visible:
                alias_node = cte.args.get("alias")
                if isinstance(alias_node, sqlglot_exp.TableAlias) and isinstance(
                    alias_node.this, sqlglot_exp.Identifier
                ):
                    cte_id = alias_node.this
                    cte_key = cte_id.name if cte_id.quoted else cte_id.name.lower()
                    if cte_key == target:
                        return True
        prev_node = cur_node
        cur_node = cur_node.parent
    return False


def _postgres_ident_sort_key(part: Any) -> Optional[str]:
    if isinstance(part, sqlglot_exp.Identifier):
        return part.name if part.quoted else part.name.lower()
    return None


def _quote_postgres_fallback_table(ref_t: str) -> Tuple[Tuple[str, str, str], str]:
    """Quote a fallback table reference (possibly schema-qualified) for PostgreSQL
    and return ``((table_key, schema_key, catalog_key), sql_ref)``."""
    parts = [p.strip() for p in ref_t.split(".") if p.strip()]
    quoted_parts = [quote_sql_identifier(p, Dialect.POSTGRES) for p in parts]
    raw_keys = [
        p[1:-1].replace('""', '"') if len(p) >= 2 and p[0] == '"' and p[-1] == '"' else p.lower()
        for p in parts
    ]
    if len(raw_keys) == 1:
        key = (raw_keys[0], "", "")
    elif len(raw_keys) == 2:
        key = (raw_keys[1], raw_keys[0], "")
    else:
        key = (raw_keys[-1], raw_keys[-2], raw_keys[-3])
    return key, ".".join(quoted_parts)


def _postgres_base_table_lock_targets(
    plan: Any,
    *,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
) -> List[str]:
    """Return deduplicated, deterministically ordered PostgreSQL LOCK TABLE targets
    for every base table referenced by ``plan`` (preserving quoting and schema qualifiers).

    Targets are sorted primarily by table name, then schema and catalog, so lock
    acquisition order is invariant to optional schema qualification (e.g. ``a_tbl``
    and ``public.a_tbl`` both sort before ``m_tbl``).
    """
    sub_plans = list(getattr(plan, "sub_plans", None) or ())
    plans = sub_plans if sub_plans else [plan]
    internal_prefixes = tuple({p.lower() for p in (object_prefix, DEFAULT_OBJECT_PREFIX, "__mv_") if p})
    targets_by_key: Dict[Tuple[str, str, str], str] = {}

    for sp in plans:
        vs = getattr(sp, "view_spec", None)
        if vs is None:
            continue
        raw_select = getattr(vs, "raw_select_sql", None) or ""
        found_from_ast = False
        if raw_select.strip():
            try:
                ast_node = parse_sql_ast(raw_select, dialect=Dialect.POSTGRES)
                if isinstance(ast_node, sqlglot_exp.Create) and isinstance(
                    ast_node.expression, sqlglot_exp.Expr
                ):
                    ast_node = ast_node.expression
                for tbl_node in ast_node.find_all(sqlglot_exp.Table):
                    this_part = tbl_node.this
                    if not isinstance(this_part, sqlglot_exp.Identifier) or not this_part.name:
                        continue
                    if _is_postgres_cte_in_scope(tbl_node):
                        continue
                    if this_part.name.lower().startswith(internal_prefixes):
                        continue
                    db_part = tbl_node.args.get("db")
                    cat_part = tbl_node.args.get("catalog")
                    clean_tbl = sqlglot_exp.Table(
                        this=this_part.copy(),
                        db=db_part.copy() if isinstance(db_part, sqlglot_exp.Expr) else None,
                        catalog=cat_part.copy() if isinstance(cat_part, sqlglot_exp.Expr) else None,
                    )
                    tbl_key = _postgres_ident_sort_key(this_part)
                    if tbl_key is None:
                        continue
                    key = (
                        tbl_key,
                        _postgres_ident_sort_key(db_part) or "",
                        _postgres_ident_sort_key(cat_part) or "",
                    )
                    if key not in targets_by_key:
                        targets_by_key[key] = clean_tbl.sql(dialect="postgres")
                    found_from_ast = True
            except sqlglot.errors.SqlglotError:
                pass

        if not found_from_ast:
            for ref_t in getattr(vs, "referenced_tables", ()) or ():
                ref_s = str(ref_t).strip()
                if not ref_s:
                    continue
                leaf = ref_s.rsplit(".", 1)[-1].strip('"').lower()
                if leaf.startswith(internal_prefixes):
                    continue
                key, sql_ref = _quote_postgres_fallback_table(ref_s)
                if key not in targets_by_key:
                    targets_by_key[key] = sql_ref

    return [sql_ref for _, sql_ref in sorted(targets_by_key.items())]


class BackfillInterruptedError(RuntimeError):
    """An incremental backfill cannot (or did not) complete safely.

    Raised (fail-closed) when a non-snapshot pk-range backfill fails after it
    committed its active watermark, and when a backfill is requested for a
    view whose catalog watermark shows a backfill that is in progress,
    interrupted, or already completed. In all of these cases the materialized
    contents are only correct relative to that watermark: the watermark is
    left untouched (so triggers keep excluding the unfolded range) and nothing
    is resumed or re-folded automatically, because re-folding committed
    chunks double-counts. Repair explicitly with
    ``register_view(..., force_rebuild=True)``, ``drop_view()`` followed by
    ``register_view()``, or ``repair_view()`` (see ``repair_view``). The
    underlying error, if any, is ``__cause__``.
    """

    def __init__(
        self,
        view_name: str,
        reason: str,
        cursor_pk: Optional[int] = None,
        max_pk: Optional[int] = None,
    ) -> None:
        self.view_name = view_name
        self.cursor_pk = cursor_pk
        self.max_pk = max_pk
        super().__init__(
            f"Materialized view '{view_name}': {reason} (backfill watermark "
            f"cursor_pk={cursor_pk}, max_pk={max_pk}). Nothing was resumed or re-folded; "
            f"{_REPAIR_GUIDANCE}"
        )


class TransactionOwnershipError(RuntimeError):
    """A manager lifecycle operation refused a connection it cannot own.

    Construction (catalog initialization), ``register_view``, ``sync_views``,
    ``drop_view``, ``apply_schema_ddl``, ``repair_view`` and
    ``backfill_incrementally`` execute DDL and commit their own transactions.
    On a connection that already has an open transaction that would commit
    (or, on MySQL/MariaDB, implicitly commit) the caller's uncommitted work,
    so they refuse before executing any SQL. They also refuse when the driver
    does not report its transaction state, and SQLite connections that use
    Python's PEP 249 ``autocommit`` attribute (``True`` makes ``commit()`` a
    no-op; ``False`` keeps a transaction permanently open).
    """


class UnsafeReplacementError(RuntimeError):
    """Replacing a published materialized view would not be atomic.

    Raised before any change when a complete view would have to be dropped
    before its replacement is fully built: on MySQL/MariaDB (DDL commits
    implicitly) and for replacements populated by an incremental backfill
    (batches commit separately). The published view is left untouched.
    """


class BulkChangesConnectionError(RuntimeError):
    """``bulk_changes()`` could not restore its connection's state.

    Raised when rolling back the manager-owned transaction failed, or when
    the MySQL/MariaDB session trigger-bypass flag could not be cleared. The
    connection was closed (the server rolls back its open transaction, so
    nothing half-done is committed and no session keeps skipping
    maintenance) and the manager refuses every later operation.
    """


class BulkChangesTransactionError(RuntimeError):
    """The ``bulk_changes()`` block committed or rolled back the manager's transaction.

    The block must leave transaction control to the manager. Work after the
    unexpected transaction end was rolled back; maintenance was restored and
    the views were refreshed in a new transaction.
    """


class BulkChangesRecoveryError(BulkChangesTransactionError):
    """Restoring maintenance after a block that ended the transaction failed.

    ``view_names`` lists the views left stale by the work the block
    committed; ``repair_view()`` repairs them. Only when their maintenance
    triggers are also missing (SQLite, PostgreSQL ``drop_triggers``) do
    ``check_schema_drift()`` and ``sync_views()`` report them (as
    ``"triggers_missing"``); stale contents alone are not detectable there
    (PostgreSQL session bypass, MySQL/MariaDB), so ``view_names`` is the
    only record.
    """

    def __init__(self, message: str, view_names: Sequence[str]) -> None:
        super().__init__(message)
        self.view_names = list(view_names)


def _manager_owned_operation(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run a public lifecycle method in a transaction the manager owns.

    Nested calls (one lifecycle method calling another) join the outer
    operation's transaction.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: "MaterializedViewManager", *args: Any, **kwargs: Any) -> Any:
            self._require_usable()
            if getattr(self, "_lifecycle_depth", 0):
                return fn(self, *args, **kwargs)
            with self._owned_transaction(name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorate


def _manager_read_operation(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a read-only public method without leaving a transaction behind.

    When the connection was idle the read transaction the method opened (on
    drivers that open one implicitly) is rolled back; a caller-owned
    transaction is joined and never ended.
    """

    @functools.wraps(fn)
    def wrapper(self: "MaterializedViewManager", *args: Any, **kwargs: Any) -> Any:
        self._require_usable()
        if getattr(self, "_lifecycle_depth", 0):
            return fn(self, *args, **kwargs)
        owned = dbapi_transaction_state(self.conn) is False
        self._lifecycle_depth = 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._lifecycle_depth = 0
            if owned and dbapi_transaction_state(self.conn) is True:
                self._rollback()

    return wrapper


class MaterializedViewManager:
    """
    Setup/migration manager that connects to a database (SQLite connection/path or
    PEP-249 DB-API 2.0 connection for PostgreSQL / MySQL) to:
      - Introspect live base table schemas (`fetch_schema`)
      - Maintain a persistent metadata registry (`__matview_catalog`) of installed MVs
      - Safely create, diff, update, or drop materialized views (`register_view`, `sync_views`, `drop_view`)
      - Enforce transparency limitation checks unless `ignore_limitations=True`
        (per-code limitations such as ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE
        only with `accept_limitations=[code, ...]`, stored with each view)
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
        raw_conn, eff_dialect, owned_engine = connect_database_via_sqlalchemy(
            conn_or_url, dialect=dialect
        )
        if owned_engine is not None:
            try:
                setattr(raw_conn, "_matview_sa_engine", owned_engine)
            except Exception:
                pass
        return raw_conn, eff_dialect

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            close_fn = getattr(cur, "close", None)
            if callable(close_fn):
                close_fn()

    def __init__(
        self,
        conn_or_url: sqlite3.Connection | str | Any,
        dialect: Optional[Dialect | str | EngineVersion] = None,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
        skip_ensure_catalog: bool = False,
        pg_fillfactor: Optional[int] = 80,
        accept_limitations: Optional[Sequence[str]] = None,
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

        self._lifecycle_depth = 0
        self._owned_sa_engine: Optional[_SAEngine] = None
        self.sa_connection: Optional[_SAConnection] = None
        if isinstance(conn_or_url, str):
            self.conn, inferred_dialect = self._connect_database_url(conn_or_url, hint_d)
            self._owned_sa_engine = getattr(self.conn, "_matview_sa_engine", None)
            self._owns_raw_conn = True
            if hint_d is None or conn_or_url.strip().lower().startswith(("postgresql://", "postgres://", "mysql://", "mariadb://")):
                hint_d = inferred_dialect
        else:
            dbapi_conn, owned_eng, sa_conn, sa_dialect = unwrap_dbapi_connection(conn_or_url)
            self.conn = dbapi_conn
            self._owned_sa_engine = owned_eng
            self.sa_connection = sa_conn
            self._owns_raw_conn = isinstance(conn_or_url, _SAEngine)
            if hint_d is None and sa_dialect is not None:
                hint_d = sa_dialect

        # Catalog initialization executes DDL and commits, and version
        # detection runs before the dialect is known: refuse a caller-owned
        # transaction before either touches the connection.
        if skip_ensure_catalog:
            owns_connection = dbapi_transaction_state(self.conn) is False
        else:
            self._require_idle_connection("MaterializedViewManager initialization")
            owns_connection = True

        if engine_version is not None:
            self.engine_version: EngineVersion = parse_engine_version(engine_version, default_dialect=hint_d)
        else:
            self.engine_version = detect_engine_version(self.conn, hint_dialect=hint_d)

        self.dialect: Dialect = self.engine_version.dialect
        self.subdialect: EngineSubdialect = self.engine_version.subdialect
        self.ignore_limitations = ignore_limitations
        if isinstance(accept_limitations, str):
            accept_limitations = [accept_limitations]
        # Limitation codes accepted for every view this manager installs, in
        # addition to the codes stored with each view (see register_view).
        self.accept_limitations: Tuple[str, ...] = tuple(
            sorted({str(c).strip() for c in (accept_limitations or ()) if str(c).strip()})
        )
        self.object_prefix = object_prefix
        self.strict = strict
        self.pg_fillfactor = pg_fillfactor
        self.catalog_table = (
            f"{self.object_prefix}catalog" if self.object_prefix else "matview_catalog"
        )

        if not skip_ensure_catalog:
            self._ensure_catalog()
        if owns_connection and dbapi_transaction_state(self.conn) is True:
            # Only reads (version detection) can be left open here.
            self._rollback()

    def _require_idle_connection(self, operation: str) -> None:
        """Raise TransactionOwnershipError unless the manager can own the next transaction."""
        for layer in dbapi_connection_layers(self.conn):
            # sqlite3.LEGACY_TRANSACTION_CONTROL exists only on Python 3.12+.
            legacy_control = getattr(sqlite3, "LEGACY_TRANSACTION_CONTROL", None)
            if isinstance(layer, sqlite3.Connection) and legacy_control is not None:
                mode = getattr(layer, "autocommit", legacy_control)
                if mode != legacy_control:
                    raise TransactionOwnershipError(
                        f"{operation} requires a sqlite3 connection with legacy transaction "
                        f"control (autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL, the default; "
                        f"isolation_level=None is supported); got autocommit={mode!r}"
                    )
        state = dbapi_transaction_state(self.conn)
        if state is True:
            raise TransactionOwnershipError(
                f"{operation} cannot run inside a caller-owned transaction: it executes "
                f"DDL and commits its own transaction, which would commit the caller's "
                f"uncommitted work (MySQL/MariaDB DDL commits it implicitly). Commit or "
                f"roll back the connection's transaction first."
            )
        if state is None:
            raise TransactionOwnershipError(
                f"{operation} cannot determine whether {type(self.conn).__module__}."
                f"{type(self.conn).__qualname__} has an open transaction, so it cannot "
                f"guarantee not to commit caller-owned work. Supported drivers report it "
                f"through in_transaction (sqlite3, apsw, mysql-connector), "
                f"get_transaction_status() (psycopg2) or server_status (PyMySQL). pg8000 and "
                f"mysqlclient (MySQLdb) do not report it: connect with psycopg2 (PostgreSQL) or "
                f"PyMySQL / mysql-connector-python (MySQL/MariaDB) instead."
            )

    @contextmanager
    def _owned_transaction(self, operation: str) -> Iterator[None]:
        """Own the connection's transaction for one top-level lifecycle operation.

        The connection must be idle. A PostgreSQL connection in autocommit mode
        is switched to explicit transactions for the duration (every DDL
        statement would otherwise commit separately, exposing half-installed
        views and the trigger-installation/population race) and restored
        afterwards. On success a still-open transaction is committed; on
        failure it is rolled back. Both are manager-owned by construction.
        """
        self._require_idle_connection(operation)
        restore_autocommit: List[Any] = []
        if self.dialect == Dialect.POSTGRES:
            # Only the physical driver connection: pool proxies (SQLAlchemy's
            # _ConnectionFairy) forward attribute reads to it, but an attribute
            # written on the proxy would shadow the driver's setting.
            physical = dbapi_connection_layers(self.conn)[-1]
            if getattr(physical, "autocommit", None) is True:
                physical.autocommit = False
                restore_autocommit.append(physical)
        self._lifecycle_depth = 1
        succeeded = False
        try:
            if operation in self._CATALOG_CHANGING_OPERATIONS:
                self._lock_catalog_for_change()
            yield
            succeeded = True
        finally:
            self._lifecycle_depth = 0
            try:
                if dbapi_transaction_state(self.conn) is not False:
                    if succeeded:
                        self._commit()
                        if dbapi_transaction_state(self.conn) is not False:
                            # Never report success for work that is not
                            # committed (e.g. a connection whose commit() is
                            # a no-op): undo it and fail.
                            self._rollback()
                            raise RuntimeError(
                                f"{operation}: committing did not end the transaction on "
                                f"{type(self.conn).__module__}.{type(self.conn).__qualname__}; "
                                f"the operation was rolled back"
                            )
                    else:
                        self._rollback()
            finally:
                for layer in restore_autocommit:
                    if dbapi_transaction_state(self.conn) is not False:
                        self._rollback()
                    layer.autocommit = True

    # Lifecycle operations that add, replace, drop or repopulate views. On
    # PostgreSQL they take the catalog lock bulk_changes() holds, so they wait
    # for an open bulk_changes() block (and it for them) instead of
    # interleaving with it.
    _CATALOG_CHANGING_OPERATIONS = frozenset(
        {"register_view", "sync_views", "drop_view", "apply_schema_ddl", "repair_view"}
    )

    def _lock_catalog_for_change(self) -> None:
        """PostgreSQL: SHARE ROW EXCLUSIVE on the catalog until the transaction ends.

        Self-conflicting and conflicting with every catalog write, but not with
        the reads writer triggers make, so writers are never blocked by it.
        No-op on other engines (SQLite serializes writers; MySQL/MariaDB uses
        a locking catalog read in bulk_changes()) and when no catalog exists.
        """
        if self.dialect != Dialect.POSTGRES:
            return
        with self._cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (self.catalog_table,))
            row = cur.fetchone()
            if row and row[0] is not None:
                cur.execute(f"LOCK TABLE {self.catalog_table} IN SHARE ROW EXCLUSIVE MODE")

    @staticmethod
    def configure_sqlite_connection(connection: Any) -> Any:
        """Enable and verify the SQLite settings maintenance triggers rely on.

        Call this on every SQLite connection that can write the base tables of
        a materialized view (for example from a connection-pool hook) and
        returns ``connection``. ``PRAGMA recursive_triggers`` is per
        connection, and only when it is ON on the connection executing an
        ``INSERT OR REPLACE`` / ``REPLACE`` does SQLite fire DELETE triggers for
        the rows the conflict resolution removes; otherwise the materialized
        view keeps the replaced rows. The manager configures its own
        connection, which does not configure any other. Raises RuntimeError
        when the setting does not take effect; idempotent.
        """
        if not callable(getattr(connection, "cursor", None)):
            raise TypeError("connection must provide a DB-API cursor() method")
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA recursive_triggers = ON;")
            cursor.execute("PRAGMA recursive_triggers;")
            row = cursor.fetchone()
            if row is None or int(row[0]) != 1:
                raise RuntimeError(
                    "SQLite PRAGMA recursive_triggers could not be enabled on this connection; "
                    "INSERT OR REPLACE / REPLACE through it would bypass materialized-view "
                    "DELETE maintenance"
                )
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
        return connection

    def _ensure_catalog(self) -> None:
        if self.dialect == Dialect.SQLITE:
            # Fail closed: a manager connection whose REPLACE statements skip
            # DELETE maintenance must not be used.
            self.configure_sqlite_connection(self.conn)
        with self._cursor() as cur:
            if self.dialect == Dialect.MYSQL:
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
                    backfill_active INTEGER NOT NULL DEFAULT 0,
                    backfill_owner VARCHAR(64) DEFAULT NULL,
                    accepted_limitations_json TEXT DEFAULT NULL
                );
                """
            )
            self._commit()
            # Catalogs created by older versions lack some columns. Probe each
            # column with a plain read and only ALTER when it is missing: on
            # PostgreSQL ALTER TABLE takes ACCESS EXCLUSIVE even when ADD
            # COLUMN then fails, so an unconditional ALTER on every manager
            # construction queued behind any open writer transaction (whose
            # triggers read the catalog) and stalled every other writer.
            for col, col_ddl in (
                ("backfill_cursor_pk", "backfill_cursor_pk BIGINT DEFAULT NULL"),
                ("backfill_max_pk", "backfill_max_pk BIGINT DEFAULT NULL"),
                ("backfill_active", "backfill_active INTEGER NOT NULL DEFAULT 0"),
                ("backfill_owner", "backfill_owner VARCHAR(64) DEFAULT NULL"),
                ("accepted_limitations_json", "accepted_limitations_json TEXT DEFAULT NULL"),
            ):
                try:
                    cur.execute(f"SELECT {col} FROM {self.catalog_table} WHERE 1 = 0")
                    cur.fetchall()
                    self._commit()
                    continue
                except Exception:
                    self._rollback()
                try:
                    cur.execute(f"ALTER TABLE {self.catalog_table} ADD COLUMN {col_ddl};")
                    self._commit()
                except Exception:
                    self._rollback()

    def _commit(self) -> None:
        if hasattr(self.conn, "commit"):
            self.conn.commit()
        elif dbapi_transaction_state(self.conn) is True:
            # Connections without DB-API commit() (raw apsw.Connection) are
            # committed with the SQL statement.
            with self._cursor() as cur:
                cur.execute("COMMIT;")

    def _rollback(self) -> None:
        if hasattr(self.conn, "rollback"):
            try:
                self.conn.rollback()
            except Exception:
                pass
        elif dbapi_transaction_state(self.conn) is True:
            try:
                with self._cursor() as cur:
                    cur.execute("ROLLBACK;")
            except Exception:
                pass

    def _param_placeholder(self) -> str:
        return "?" if self.dialect == Dialect.SQLITE else "%s"

    def _execute_param(self, cur: Any, qmark_sql: str, params: Sequence[Any]) -> Any:
        return cur.execute(render_parameterized_sql(qmark_sql, self.dialect), params)

    @staticmethod
    def _savepoint_create(cur: Any, name: str) -> None:
        cur.execute(f"SAVEPOINT {name};")

    @staticmethod
    def _savepoint_release(cur: Any, name: str) -> None:
        cur.execute(f"RELEASE SAVEPOINT {name};")

    @staticmethod
    def _savepoint_rollback_and_release(cur: Any, name: str) -> None:
        cur.execute(f"ROLLBACK TO SAVEPOINT {name};")
        cur.execute(f"RELEASE SAVEPOINT {name};")

    def close(self) -> None:
        """Close any connection and dispose any SQLAlchemy engine owned by this manager."""
        owned_engine = getattr(self, "_owned_sa_engine", None)
        if getattr(self, "_owns_raw_conn", False):
            close_fn = getattr(self.conn, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if owned_engine is not None:
            try:
                owned_engine.dispose()
            except Exception:
                pass
            self._owned_sa_engine = None

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

    @_manager_read_operation
    def fetch_schema(self, extra_schema_sql: Optional[str] = None) -> Dict[str, TableSchema]:
        """Fetches the live schema of all user base tables from the connected database."""
        schemas = introspect_database_schemas(
            self.conn,
            dialect=self.dialect,
            object_prefix=self.object_prefix,
        )
        schemas.pop(self.catalog_table, None)
        if extra_schema_sql:
            schemas.update(parse_table_schema(extra_schema_sql, dialect=self.dialect))
        return schemas

    @_manager_owned_operation("apply_schema_ddl")
    def apply_schema_ddl(self, ddl_sql: str, auto_sync_views: bool = True) -> None:
        """
        Safely executes schema DDL changes (e.g. `CREATE TABLE`, `ALTER TABLE`) and
        automatically refreshes any registered materialized views whose underlying
        base table schema changed.

        On MySQL/MariaDB the affected views cannot be replaced atomically (see
        UnsafeReplacementError), so with ``auto_sync_views=True`` the DDL is
        refused before any statement runs when it touches a base table of a
        registered, complete view; the error names the views. Supported
        recipe: ``drop_view()`` each named view, apply the DDL, then
        ``register_view()`` them again.
        """
        statements = [
            stmt
            for stmt in _split_top_level(_strip_sql_comments(ddl_sql, dialect=self.dialect), ";", dialect=self.dialect)
            if stmt.strip()
        ]
        if auto_sync_views and self.dialect == Dialect.MYSQL:
            affected = self._views_affected_by_ddl(statements)
            if affected:
                raise UnsafeReplacementError(
                    f"Refusing to apply schema DDL on {self.subdialect.value}: it changes base "
                    f"tables of the registered materialized view(s) {', '.join(affected)}, which "
                    f"would then have to be replaced, and MySQL/MariaDB DDL commits implicitly so "
                    f"they cannot be replaced atomically. Nothing was executed. Supported recipe: "
                    f"drop_view() each of these views, apply the DDL, then register_view() them "
                    f"again (readers see them disappear and fill)."
                )
        with self._cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
            self._commit()
        if auto_sync_views:
            catalog = self.get_catalog()
            if catalog:
                self.sync_views([entry["view_sql"] for entry in catalog.values()])

    def _views_affected_by_ddl(self, statements: Sequence[str]) -> List[str]:
        """Registered complete views a DDL script could force to be replaced.

        A view is affected when the script mentions one of its base tables
        (the signature covers every column of every referenced table). A
        statement that cannot be parsed affects every view (fail closed).
        Views with an incomplete backfill are excluded: they may be rebuilt.
        """
        catalog = self.get_catalog()
        published = {n: e for n, e in catalog.items() if not self._is_backfill_incomplete(e)}
        if not published:
            return []
        touched: Set[str] = set()
        for stmt in statements:
            try:
                parsed = sqlglot.parse(stmt, read="mysql")
            except Exception:
                return sorted(published)
            tables = [t for tree in parsed if tree is not None for t in tree.find_all(sqlglot_exp.Table)]
            if not tables:
                return sorted(published)
            touched.update(t.name.lower() for t in tables if t.name)
        affected = []
        for name, entry in published.items():
            try:
                specs = parse_view_definitions(entry["view_sql"], dialect=self.dialect)
                referenced = {r.lower().split(".")[-1].strip('`"') for s in specs for r in s.referenced_tables}
            except Exception:
                affected.append(name)
                continue
            if referenced & touched:
                affected.append(name)
        return sorted(affected)

    def _catalog_select(self, cur: Any, sql: str) -> Optional[List[Any]]:
        """Run one catalog read; None if it failed (e.g. no catalog yet).

        Never ends a transaction it does not own: inside an open transaction
        the read runs under a savepoint that is rolled back to on failure
        (PostgreSQL aborts the whole transaction on a failed statement). Only
        when the connection was idle is the failed read's own transaction
        rolled back. With unknown transaction state the error propagates.
        """
        state = dbapi_transaction_state(self.conn)
        if state is True:
            self._savepoint_create(cur, "__matview_catalog_read")
            try:
                cur.execute(sql)
                rows = cur.fetchall()
            except Exception:
                self._savepoint_rollback_and_release(cur, "__matview_catalog_read")
                return None
            self._savepoint_release(cur, "__matview_catalog_read")
            return rows
        try:
            cur.execute(sql)
            return cur.fetchall()
        except Exception:
            if state is None:
                raise
            self._rollback()
            return None

    @_manager_read_operation
    def get_catalog(self) -> Dict[str, Dict[str, Any]]:
        return self._read_catalog()

    def _read_catalog(self, lock_clause: str = "", required: bool = False) -> Dict[str, Dict[str, Any]]:
        """Read the catalog; ``lock_clause`` (e.g. MySQL ``LOCK IN SHARE MODE``)
        makes it a locking read, which does not open a consistent snapshot.
        With ``required`` an unreadable catalog raises instead of reading as
        empty (callers that suspend maintenance must know every view)."""
        with self._cursor() as cur:
            rows = self._catalog_select(
                cur,
                f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json, "
                f"backfill_cursor_pk, backfill_max_pk, backfill_active "
                f"FROM {self.catalog_table}{lock_clause}",
            )
            has_wm_cols = rows is not None
            if rows is None:
                rows = self._catalog_select(
                    cur,
                    f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                    f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json "
                    f"FROM {self.catalog_table}{lock_clause}",
                )
                if rows is None:
                    if required:
                        raise RuntimeError(
                            f"the materialized-view catalog {self.catalog_table} could not be read"
                        )
                    return {}
            accepted_rows = self._catalog_select(
                cur,
                f"SELECT view_name, accepted_limitations_json FROM {self.catalog_table}{lock_clause}",
            )
        accepted_by_view: Dict[str, Any] = {}
        for arow in accepted_rows or ():
            try:
                accepted_by_view[arow[0]] = None if arow[1] is None else json.loads(arow[1])
            except (TypeError, ValueError):
                accepted_by_view[arow[0]] = {"definition": None, "codes": []}
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
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
            # Limitation codes overridden when the view was installed, bound
            # to the definition they were accepted for. A record that does not
            # match the stored definition (or is malformed) accepts nothing,
            # so recompiling the view refuses instead of carrying an
            # acceptance over to another definition.
            rec = accepted_by_view.get(row[0])
            codes: List[str] = []
            mismatch = False
            if rec is not None:
                if (
                    isinstance(rec, dict)
                    and isinstance(rec.get("codes"), list)
                    and rec.get("definition") == _definition_digest(_stored_select_sql(row[1]))
                ):
                    codes = [str(c) for c in rec["codes"]]
                else:
                    mismatch = True
            out[row[0]]["accepted_limitations"] = codes
            out[row[0]]["accepted_definition"] = rec.get("definition") if isinstance(rec, dict) else None
            out[row[0]]["accepted_limitations_mismatch"] = mismatch
        return out

    def _compute_signature(self, plan: MaterializedViewPlan, schemas: Dict[str, TableSchema]) -> str:
        h = hashlib.sha256()
        # The generated maintenance shape is part of the signature: a view
        # installed by an older generator (different backing columns, e.g. the
        # SQLite exact-SUM bookkeeping columns, or a bumped generator version
        # for trigger-only changes) is then rebuilt by register_view/sync_views
        # and repair_view instead of being kept with stale triggers.
        h.update(f"generator:{MATVIEW_GENERATOR_VERSION}".encode("utf-8"))
        h.update(f"backing:{plan.backing_table_ddl}".encode("utf-8"))
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

    @_manager_owned_operation("drop_view")
    def drop_view(self, view_name: str) -> bool:
        """Drops a registered materialized view, its triggers, read-only view, and backing tables."""
        catalog = self.get_catalog()
        resolved_name = self._resolve_catalog_name(view_name, catalog)
        entry = catalog.get(resolved_name)
        with self._cursor() as cur:
            if entry:
                # One transaction on SQLite too (its DDL otherwise commits
                # statement by statement), so a failure drops nothing.
                self._reserve_sqlite_writer(cur)
                self._drop_registered_objects(cur, resolved_name, entry)
                self._commit()
                return True
            else:
                raw_logical = (
                    view_name[len(self.object_prefix) :]
                    if self.object_prefix and view_name.startswith(self.object_prefix)
                    else view_name
                )
                db_v = f"{self.object_prefix}{raw_logical}"
                cur.execute(build_drop_sql("VIEW", db_v, dialect=self.dialect))
                cur.execute(build_drop_sql("VIEW", raw_logical, dialect=self.dialect))
                cur.execute(build_drop_sql("TABLE", f"{self.object_prefix}store_{raw_logical}", dialect=self.dialect))
                self._commit()
                return False

    def _reserve_sqlite_writer(self, cur: Any) -> None:
        """On SQLite, open the manager-owned write transaction (BEGIN IMMEDIATE).

        sqlite3 does not open a transaction for DDL, so without this every
        CREATE/DROP commits on its own. BEGIN IMMEDIATE also takes the
        database's single writer lock up front, so no independent writer can
        commit between the statements that follow.
        """
        if self.dialect == Dialect.SQLITE and dbapi_transaction_state(self.conn) is not True:
            cur.execute("BEGIN IMMEDIATE;")

    def _drop_registered_objects(self, cur: Any, resolved_name: str, entry: Dict[str, Any]) -> None:
        """Drop one registered generation (objects and catalog row) without committing."""
        for t_name in entry["trigger_names"]:
            if self.dialect != Dialect.POSTGRES:
                cur.execute(build_drop_sql("TRIGGER", t_name, dialect=self.dialect))
        for fn_name in entry["function_names"]:
            cur.execute(f"DROP FUNCTION IF EXISTS {fn_name if '(' in fn_name else fn_name + '()'} CASCADE;")
        cur.execute(build_drop_sql("VIEW", resolved_name, dialect=self.dialect))
        cur.execute(build_drop_sql("TABLE", entry["backing_table"], dialect=self.dialect))
        for aux_t in entry["aux_tables"]:
            cur.execute(build_drop_sql("TABLE", aux_t, dialect=self.dialect))
        if self.dialect == Dialect.SQLITE and self.engine_version.supports_sqlite_returning:
            self._execute_param(
                cur,
                f"DELETE FROM {self.catalog_table} WHERE view_name = ? RETURNING view_name;",
                (resolved_name,),
            )
            try:
                cur.fetchall()
            except Exception:
                pass
        else:
            self._execute_param(
                cur,
                f"DELETE FROM {self.catalog_table} WHERE view_name = ?;",
                (resolved_name,),
            )

    def _replacement_refusal(
        self, view_name: str, existing: Dict[str, Any], populate: bool, incremental_backfill: bool
    ) -> Optional[UnsafeReplacementError]:
        """Why replacing ``existing`` cannot keep the published view until the new one is ready.

        A view whose pk-range backfill is incomplete publishes nothing correct,
        so it may always be rebuilt (the documented recovery).
        """
        if self._is_backfill_incomplete(existing):
            return None
        if self.dialect == Dialect.MYSQL:
            reason = (
                "MySQL/MariaDB DDL commits implicitly, so the old generation cannot be kept "
                "until the replacement is built and populated"
            )
        elif populate and incremental_backfill:
            reason = (
                "an incremental backfill commits batch by batch, so readers would see a "
                "partially populated replacement"
            )
        else:
            return None
        return UnsafeReplacementError(
            f"Refusing to replace materialized view '{view_name}' on {self.subdialect.value}: "
            f"{reason}. The published view and its catalog entry are unchanged. To replace "
            f"it non-atomically (readers see the view disappear and then fill), call "
            f"drop_view() and then register_view()."
        )

    def _update_backfill_watermark(
        self,
        cur: Any,
        view_name: str,
        cursor_pk: Optional[int],
        max_pk: Optional[int],
        active: int,
    ) -> None:
        self._execute_param(
            cur,
            f"UPDATE {self.catalog_table} "
            f"SET backfill_cursor_pk = ?, backfill_max_pk = ?, backfill_active = ? "
            f"WHERE view_name = ?",
            (cursor_pk, max_pk, active, view_name),
        )

    @_manager_owned_operation("backfill_incrementally")
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

        Fail-closed contract for pk-range backfills: the watermark is claimed
        atomically, so at most one session folds a view. If a non-snapshot run
        stops after committing its claim, the original exception propagates
        (annotated) and the watermark stays active at the last committed
        chunk, so triggers keep excluding the unfolded range. Any later call
        for that view -- or for a view whose pk-range backfill already
        completed, or while another session's backfill is running -- raises
        `BackfillInterruptedError` before folding anything, in either
        isolation mode, and leaves the watermark untouched. There is no
        automatic resume; rebuild with ``register_view(..., force_rebuild=True)``
        or ``drop_view()`` + ``register_view()``.
        """
        if not snapshot_isolation and batch_size > 0:
            sub_list = plan.sub_plans if plan.sub_plans else [plan]
            for p in sub_list:
                if p.incremental_backfill_spec is not None:
                    if (
                        p.incremental_backfill_spec.mode != "pk_range_fold"
                        or not p.has_watermark_triggers
                    ):
                        raise UnsupportedMaterializedViewError(
                            f"Incremental backfill without snapshot isolation requires watermark-gated triggers and pk_range_fold mode; view '{p.view_name}' does not meet these requirements."
                        )

        if plan.sub_plans:
            # Each sub-plan refuses (before folding anything) if its watermark
            # is claimed or completed, so re-running a partially completed
            # multi-view plan fails closed on the first finished sub-plan
            # instead of re-folding it.
            total_b = 0
            for sp in plan.sub_plans:
                total_b += self.backfill_incrementally(
                    sp,
                    batch_size=batch_size,
                    on_progress=on_progress,
                    snapshot_isolation=snapshot_isolation,
                )
            return total_b

        cur = self.conn.cursor()
        snapshot_started = False
        self._backfill_watermark_claimed = False
        self._backfill_owner_token = None
        self._backfill_committed_watermark: Optional[Tuple[int, int]] = None
        try:
            if snapshot_isolation:
                try:
                    if self.dialect == Dialect.POSTGRES:
                        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                    elif self.dialect == Dialect.MYSQL:
                        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                        cur.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT;")
                    elif self.dialect == Dialect.SQLITE:
                        self._savepoint_create(cur, "__mv_backfill_snap")
                    else:
                        raise RuntimeError(
                            f"Snapshot isolation is not implemented for {self.dialect.value}"
                        )
                except Exception as exc:
                    self._rollback()
                    raise RuntimeError(
                        f"Unable to establish snapshot isolation for {self.dialect.value} backfill"
                    ) from exc
                snapshot_started = True

            # Fail closed before folding anything: a claimed or completed
            # pk-range watermark means another backfill is running, an earlier
            # one was interrupted, or the view is already populated; folding
            # again from MIN(pk) would double-count. (Read after the snapshot
            # is established; the failure path below cannot reset a claimed
            # watermark, see _best_effort_reset_backfill_watermark.)
            self._refuse_unless_backfill_unclaimed(
                self._catalog_entry(cur, plan.view_name), plan.view_name
            )
            return self._backfill_incrementally_with_cursor(
                plan, batch_size, on_progress, snapshot_isolation, cur
            )
        except BaseException as failure:
            # A refusal (pre-check, lost claim, fence) means this call owns no
            # watermark: only roll back. Resetting would wait for the row lock
            # held by the owning backfill (defeating fail-fast) and has nothing
            # of ours to reset.
            refused = isinstance(failure, BackfillInterruptedError)
            if snapshot_isolation:
                if snapshot_started:
                    self._rollback_snapshot_backfill(cur, None if refused else plan.view_name)
            else:
                self._rollback()
                if (
                    self.dialect == Dialect.SQLITE
                    and not hasattr(self.conn, "rollback")
                    and getattr(self.conn, "in_transaction", False)
                ):
                    try:
                        cur.execute("ROLLBACK;")
                    except Exception:
                        pass
                if refused:
                    pass
                elif not self._backfill_watermark_claimed:
                    self._best_effort_reset_backfill_watermark(plan.view_name)
                else:
                    # Fail closed: this call committed an active pk-range
                    # watermark (and possibly folded chunks). Keep it so
                    # triggers keep excluding the unfolded range; refuse to
                    # resume (see BackfillInterruptedError).
                    # The original exception type is preserved for callers;
                    # the note states the resulting state and the repair.
                    # No database access here: it could replace the original
                    # exception (e.g. a lost connection) or leave a read
                    # transaction open. Use the last watermark this call
                    # committed, tracked in memory.
                    exc_now = sys.exc_info()[1]
                    if exc_now is not None and hasattr(exc_now, "add_note"):
                        cursor_pk, max_pk = self._backfill_committed_watermark or (None, None)
                        try:
                            if cursor_pk is not None and cursor_pk == max_pk:
                                # The last batch committed (max, max, active=0)
                                # together with its fold (e.g. on_progress
                                # raised after the final batch).
                                exc_now.add_note(
                                    f"matview: incremental backfill of '{plan.view_name}' had already "
                                    f"completed (watermark cursor_pk={cursor_pk}, max_pk={max_pk} "
                                    f"recorded as complete) when this error was raised; the view is "
                                    f"fully populated."
                                )
                            else:
                                exc_now.add_note(
                                    f"matview: incremental backfill of '{plan.view_name}' stopped after "
                                    f"committing progress; its watermark is kept active (last committed "
                                    f"cursor_pk={cursor_pk}, max_pk={max_pk}) and the view is incomplete. "
                                    f"It is not resumed: {_REPAIR_GUIDANCE}"
                                )
                        except Exception:
                            pass
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def _rollback_snapshot_backfill(self, cur: Any, view_name: Optional[str] = None) -> None:
        """Roll back a failed snapshot backfill without leaving its transaction open."""
        if self.dialect == Dialect.SQLITE:
            try:
                self._savepoint_rollback_and_release(cur, "__mv_backfill_snap")
                return
            except Exception:
                # If the savepoint no longer exists or the connection is already
                # in an error state, a full rollback is the safest recovery.
                self._rollback()
                if view_name:
                    self._best_effort_reset_backfill_watermark(view_name)
                return
        self._rollback()
        if view_name:
            self._best_effort_reset_backfill_watermark(view_name)

    def _catalog_entry(self, cur: Any, view_name: str) -> Optional[Dict[str, Any]]:
        self._execute_param(
            cur,
            f"SELECT backfill_cursor_pk, backfill_max_pk, backfill_active "
            f"FROM {self.catalog_table} WHERE view_name = ?",
            (view_name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "backfill_cursor_pk": row[0],
            "backfill_max_pk": row[1],
            "backfill_active": int(row[2]) if row[2] is not None else 0,
        }

    @staticmethod
    def _is_backfill_incomplete(entry: Dict[str, Any]) -> bool:
        return entry.get("backfill_active") == 1 and entry.get("backfill_cursor_pk") is not None

    @staticmethod
    def _refuse_unless_backfill_unclaimed(entry: Optional[Dict[str, Any]], view_name: str) -> None:
        if not entry or entry.get("backfill_cursor_pk") is None:
            return
        if entry.get("backfill_active") == 1:
            reason = (
                "its pk-range backfill is either still RUNNING in another session or was "
                "INTERRUPTED (the catalog cannot tell these apart); the view is incomplete. If one "
                "is running, wait for it to finish. A repair or rebuild takes the watermark over, so "
                "a running backfill stops at its next batch without folding more"
            )
        else:
            reason = "the view was already populated; backfilling again would double-count"
        raise BackfillInterruptedError(
            view_name, reason, entry.get("backfill_cursor_pk"), entry.get("backfill_max_pk")
        )

    # Watermark value recorded as "completed" when there was nothing to fold
    # (empty base table) or after repair_view(). For pk-range views any
    # non-NULL cursor with backfill_active = 0 means "populated"; triggers only
    # gate on backfill_active = 1, so the number itself never gates rows.
    _COMPLETED_MARKER_PK = 0

    def _is_snapshot_row_changed_error(self, exc: BaseException) -> bool:
        """MariaDB innodb_snapshot_isolation: the row changed since the snapshot."""
        if self.dialect != Dialect.MYSQL:
            return False
        code = getattr(exc, "errno", None)
        if code is None and getattr(exc, "args", None):
            code = exc.args[0]
        return code == 1020

    def _is_concurrent_group_insert_error(self, exc: BaseException) -> bool:
        """PostgreSQL MERGE has no upsert semantics: when a writer's trigger
        concurrently creates the backing row of a group the chunk is about to
        INSERT, the chunk fails with unique_violation (23505). The whole batch
        rolled back, so re-running it (now taking the MATCHED path) is exact."""
        if self.dialect != Dialect.POSTGRES:
            return False
        return (getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)) == "23505"

    def _is_lock_timeout_error(self, exc: BaseException) -> bool:
        """PostgreSQL lock_timeout (55P03); MySQL 1205 is already transient."""
        if self.dialect == Dialect.POSTGRES:
            return (getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)) == "55P03"
        return False

    def _claim_backfill_watermark(
        self, cur: Any, view_name: str, cursor_pk: int, max_pk: int, active: int = 1,
        retry: bool = True,
    ) -> None:
        """Atomically claim the pk-range watermark: only one backfill may own it.

        A conditional UPDATE (only while backfill_cursor_pk IS NULL) re-checks
        the latest committed row after any concurrent claimer commits, so two
        sessions cannot both fold the view. It waits for the catalog row lock
        instead of using NOWAIT: on MySQL/MariaDB every open writer
        transaction holds a shared lock on that row (the triggers' watermark
        subquery), so a NOWAIT conflict cannot be told apart from another
        claim. Consequently a second claimer of a *snapshot* backfill waits for
        the first one's transaction (or the engine lock timeout) and is then
        refused. Deadlock victims and lock-wait timeouts are retried with the
        batch backoff; exhausting them is reported as such.

        The claim also records a fresh random ownership token
        (backfill_owner), which every end-of-batch compare-and-set checks.
        (cursor, max) alone is not an identity: after a takeover (rebuild,
        drop + register) a new backfill can reach the very same (cursor, max)
        while the old one is paused, and the old one would then keep folding
        its stale chunk SQL into the new store (ABA).
        """
        token = uuid.uuid4().hex
        attempt = 0
        while True:
            try:
                self._execute_param(
                    cur,
                    f"UPDATE {self.catalog_table} "
                    f"SET backfill_cursor_pk = ?, backfill_max_pk = ?, backfill_active = ?, "
                    f"backfill_owner = ? "
                    f"WHERE view_name = ? AND backfill_cursor_pk IS NULL",
                    (cursor_pk, max_pk, active, token, view_name),
                )
                break
            except Exception as exc:
                if self._is_snapshot_row_changed_error(exc):
                    self._rollback()
                    raise BackfillInterruptedError(
                        view_name, f"a concurrent session claimed the pk-range backfill watermark ({exc!r})"
                    ) from exc
                if not (self._is_transient_lock_error(exc) or self._is_lock_timeout_error(exc)):
                    raise
                self._rollback()
                # A snapshot claim cannot be retried: the rollback ended the snapshot.
                if not retry or attempt >= self._BACKFILL_TRANSIENT_RETRIES:
                    raise BackfillInterruptedError(
                        view_name,
                        f"could not lock the catalog row to claim the backfill within the engine "
                        f"lock timeout after {attempt + 1} attempts (held by long-running writer "
                        f"transactions or another backfill): {exc!r}",
                    ) from exc
                attempt += 1
                self._backfill_sleep(self._backfill_retry_delay(attempt))
        claimed = getattr(cur, "rowcount", None)
        if claimed is None or claimed < 0:
            # Driver/adapter without a DML rowcount (all supported DB-API
            # drivers report one): verify the claim by re-reading the row.
            self._execute_param(
                cur,
                f"SELECT backfill_cursor_pk, backfill_max_pk, backfill_active, backfill_owner "
                f"FROM {self.catalog_table} WHERE view_name = ?",
                (view_name,),
            )
            row = cur.fetchone()
            claimed = 1 if row and (row[0], row[1], int(row[2] or 0), row[3]) == (
                cursor_pk, max_pk, active, token
            ) else 0
        if claimed != 1:
            self._rollback()
            raise BackfillInterruptedError(
                view_name, "another session claimed the pk-range backfill watermark first"
            )
        self._backfill_owner_token = token

    def _advance_backfill_watermark(
        self, cur: Any, view_name: str, cursor_pk: int, upper_pk: int, max_pk: int, active: int
    ) -> None:
        """Advance the watermark at the end of a batch, only if this backfill
        still owns it (compare-and-set on the cursor it committed).

        The exclusive catalog row lock is taken here, after the chunk's range
        lock and fold -- the same lock order as before ownership checks
        existed -- so writers are not blocked for the whole batch.
        upper_pk > cursor_pk, so a matching row is always changed (MySQL
        reports changed, not matched, rows). Engine deadlock/lock-wait-timeout
        errors propagate to the batch retry, which is exact because the
        watermark only advances together with the fold. Zero rows means
        repair_view/rebuild/drop took the view over: roll the fold back, stop.
        The ownership token from the claim is part of the comparison, so a
        newer backfill that happens to sit at the same (cursor, max) is never
        mistaken for this one.
        """
        token = self._backfill_owner_token
        if token is None:
            raise RuntimeError("internal error: pk-range batch without a claimed ownership token")
        self._execute_param(
            cur,
            f"UPDATE {self.catalog_table} "
            f"SET backfill_cursor_pk = ?, backfill_max_pk = ?, backfill_active = ? "
            f"WHERE view_name = ? AND backfill_cursor_pk = ? AND backfill_max_pk = ? "
            f"AND backfill_active = 1 AND backfill_owner = ?",
            (upper_pk, max_pk, active, view_name, cursor_pk, max_pk, token),
        )
        owned = getattr(cur, "rowcount", None)
        if owned is None or owned < 0:
            self._execute_param(
                cur,
                f"SELECT backfill_cursor_pk, backfill_max_pk, backfill_active, backfill_owner "
                f"FROM {self.catalog_table} WHERE view_name = ?",
                (view_name,),
            )
            row = cur.fetchone()
            owned = 1 if row and (row[0], row[1], int(row[2] or 0), row[3]) == (
                upper_pk, max_pk, active, token
            ) else 0
        if owned != 1:
            self._rollback()
            raise BackfillInterruptedError(
                view_name,
                "the backfill watermark was taken over (repair_view, rebuild or drop) while this "
                "backfill was running; its in-flight batch was rolled back and it stopped",
                cursor_pk, max_pk,
            )

    def _best_effort_reset_backfill_watermark(self, view_name: str) -> None:
        try:
            reset_cur = self.conn.cursor()
            try:
                self._execute_param(
                    reset_cur,
                    f"UPDATE {self.catalog_table} "
                    f"SET backfill_cursor_pk = NULL, backfill_max_pk = NULL, backfill_active = 0 "
                    # Never touch a claimed/completed pk-range watermark: a
                    # failed call must not reset a watermark it does not own
                    # (one this call claimed in a snapshot was rolled back).
                    f"WHERE view_name = ? AND backfill_cursor_pk IS NULL",
                    (view_name,),
                )
                self._commit()
            finally:
                reset_cur.close()
        except Exception:
            try:
                self._rollback()
            except Exception:
                pass

    # Bounded retries for a non-snapshot pk-range batch chosen as a deadlock /
    # serialization victim by the engine while independent writers run.
    _BACKFILL_TRANSIENT_RETRIES = 8
    # Bounded exponential backoff with jitter between those retries. Retrying
    # immediately re-collides with the same still-running writer transactions;
    # waiting lets them commit or be replayed. Sleep and RNG are injectable
    # (instance attributes) so tests are fast and exact.
    _BACKFILL_RETRY_BASE_DELAY_S = 0.02
    _BACKFILL_RETRY_MAX_DELAY_S = 1.0
    _backfill_sleep = staticmethod(time.sleep)
    _backfill_rng: Any = random.Random()
    _backfill_watermark_claimed = False
    _backfill_owner_token: Optional[str] = None
    accept_limitations: Tuple[str, ...] = ()
    ignore_limitations: bool = False

    def _backfill_retry_delay(self, attempt: int) -> float:
        """Delay before retry number `attempt` (1-based): jitter in [d/2, d],
        d = min(max, base * 2**(attempt-1))."""
        ceiling = min(
            self._BACKFILL_RETRY_MAX_DELAY_S,
            self._BACKFILL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
        )
        return ceiling * (0.5 + 0.5 * self._backfill_rng.random())

    def _is_transient_lock_error(self, exc: BaseException) -> bool:
        """True only for engine-reported deadlock/serialization/lock-wait victims."""
        if self.dialect == Dialect.MYSQL:
            code = getattr(exc, "errno", None)
            if code is None and getattr(exc, "args", None):
                code = exc.args[0]
            # 1213 ER_LOCK_DEADLOCK (whole transaction rolled back),
            # 1205 ER_LOCK_WAIT_TIMEOUT (statement rolled back; we roll back the rest).
            return code in (1213, 1205)
        if self.dialect == Dialect.POSTGRES:
            code = getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)
            return code in ("40P01", "40001")
        return False

    def _run_pk_range_batch(
        self,
        cur: Any,
        plan: MaterializedViewPlan,
        spec: Any,
        base_t: str,
        pk_col: str,
        cursor_pk: int,
        upper_pk: int,
        max_pk: int,
        snapshot_isolation: bool,
    ) -> None:
        """Fold one (cursor_pk, upper_pk] range and advance the watermark."""
        sqlite_chunk_tx_began = False
        if not snapshot_isolation:
            if self.dialect in (Dialect.POSTGRES, Dialect.MYSQL):
                try:
                    self._execute_param(
                        cur,
                        f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ? FOR UPDATE",
                        (cursor_pk, upper_pk),
                    )
                except Exception as exc:
                    # A deadlock here has already rolled back the batch
                    # transaction; continuing would fold without the range lock.
                    if self._is_transient_lock_error(exc):
                        raise
            elif self.dialect == Dialect.SQLITE:
                if not getattr(self.conn, "in_transaction", False):
                    try:
                        cur.execute("BEGIN IMMEDIATE;")
                        sqlite_chunk_tx_began = True
                    except Exception:
                        pass
        for tmpl in spec.chunk_sql_templates:
            cur.execute(tmpl, (cursor_pk, upper_pk))
        still_active = 1 if upper_pk < max_pk else 0
        self._advance_backfill_watermark(cur, plan.view_name, cursor_pk, upper_pk, max_pk, still_active)
        if not snapshot_isolation:
            self._commit()
            if sqlite_chunk_tx_began and (
                not hasattr(self.conn, "commit")
                or getattr(self.conn, "in_transaction", False)
            ):
                try:
                    cur.execute("COMMIT;")
                except Exception:
                    pass

    def _backfill_incrementally_with_cursor(
        self,
        plan: MaterializedViewPlan,
        batch_size: int,
        on_progress: Optional[Callable[[int, int], None]],
        snapshot_isolation: bool,
        cur: Any,
    ) -> int:
        spec = plan.incremental_backfill_spec
        if spec is None or batch_size <= 0:
            for s in plan.initial_populate_sqls:
                cur.execute(s)
            self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
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
                # Claim first (atomic vs. concurrent backfills), then record a
                # *completed* marker: (NULL, NULL, 0) would let a later
                # backfill fold rows that triggers have maintained since.
                self._claim_backfill_watermark(
                    cur, plan.view_name, self._COMPLETED_MARKER_PK, self._COMPLETED_MARKER_PK, active=0,
                    retry=not snapshot_isolation,
                )
                if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                    for s in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(s)
                if snapshot_isolation and self.dialect == Dialect.SQLITE:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                self._commit()
                return 0
            min_pk, max_pk = int(row[0]), int(row[1])
            cursor_pk = min_pk - 1
            total_span = max(1, max_pk - min_pk + 1)
            self._claim_backfill_watermark(
                cur, plan.view_name, cursor_pk, max_pk, retry=not snapshot_isolation
            )
            if not snapshot_isolation:
                self._commit()
                self._backfill_watermark_claimed = True
                self._backfill_committed_watermark = (cursor_pk, max_pk)
            while cursor_pk < max_pk:
                upper_pk = min(max_pk, cursor_pk + batch_size)
                attempt = 0
                while True:
                    try:
                        self._run_pk_range_batch(
                            cur, plan, spec, base_t, pk_col,
                            cursor_pk, upper_pk, max_pk, snapshot_isolation,
                        )
                        break
                    except Exception as exc:
                        # A non-snapshot batch is its own transaction: the chunk
                        # fold and its watermark advance commit together. A
                        # deadlock/serialization victim has had that whole
                        # transaction rolled back, the committed watermark still
                        # gates (cursor_pk, max_pk], so re-running the identical
                        # range is exact. Anything else, or exhausting the
                        # bound, propagates to the caller's failure path.
                        if (
                            snapshot_isolation
                            or attempt >= self._BACKFILL_TRANSIENT_RETRIES
                            or not (
                                self._is_transient_lock_error(exc)
                                # PG lock_timeout (55P03) is the analogue of
                                # MySQL 1205: retry, never "taken over".
                                or self._is_lock_timeout_error(exc)
                                or self._is_concurrent_group_insert_error(exc)
                            )
                        ):
                            raise
                        attempt += 1
                        self._rollback()
                        self._backfill_sleep(self._backfill_retry_delay(attempt))
                batches_run += 1
                cursor_pk = upper_pk
                if not snapshot_isolation:
                    self._backfill_committed_watermark = (cursor_pk, max_pk)
                if on_progress:
                    done = min(total_span, cursor_pk - (min_pk - 1))
                    on_progress(batches_run, done)
            # No separate completion write: the last batch's compare-and-set
            # already recorded (max, max, active=0) together with its fold. An
            # unconditional write here (a new transaction after that commit)
            # would overwrite a newer owner's claim made in between.
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            return batches_run

        elif spec.mode == "group_key_batch" and spec.distinct_group_query:
            cur.execute(spec.distinct_group_query)
            all_keys = [r[0] for r in (cur.fetchall() or [])]
            if not all_keys:
                self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
                if snapshot_isolation and self.dialect == Dialect.SQLITE:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                self._commit()
                return 0
            ph = self._param_placeholder()
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i : i + batch_size]
                placeholders = ", ".join(ph for _ in batch_keys)
                for tmpl in spec.chunk_sql_templates:
                    sql = tmpl.format(placeholders=placeholders)
                    cur.execute(sql, tuple(batch_keys))
                if not snapshot_isolation:
                    self._commit()
                batches_run += 1
                if on_progress:
                    on_progress(batches_run, min(len(all_keys), i + batch_size))
            self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            return batches_run

        elif spec.mode == "offset_batch" and spec.offset_insert_template:
            offset = 0
            while True:
                self._execute_param(cur, spec.offset_insert_template, (batch_size, offset))
                rc = getattr(cur, "rowcount", None)
                if rc is not None and rc >= 0:
                    inserted = int(rc)
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {plan.backing_table_name}")
                    cnt_row = cur.fetchone()
                    total_now = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else 0
                    inserted = max(0, total_now - offset)
                if not snapshot_isolation:
                    self._commit()
                if inserted == 0:
                    break
                batches_run += 1
                offset += inserted
                if on_progress:
                    on_progress(batches_run, offset)
                if inserted < batch_size:
                    break
            self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            return batches_run

        for s in plan.initial_populate_sqls:
            cur.execute(s)
        self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
        if snapshot_isolation and self.dialect == Dialect.SQLITE:
            cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
        self._commit()
        return 1

    @staticmethod
    def _normalize_accept(codes: Optional[Sequence[str] | str]) -> Optional[Tuple[str, ...]]:
        if codes is None:
            return None
        if isinstance(codes, str):
            codes = [codes]
        return tuple(sorted({str(c).strip() for c in codes if str(c).strip()}))

    def _effective_accept(
        self,
        v_spec: ViewSpec,
        catalog: Dict[str, Dict[str, Any]],
        explicit: Optional[Sequence[str] | str],
    ) -> Tuple[str, ...]:
        """Limitation codes accepted for one view: the explicit argument when
        given, else the manager default plus the codes stored with the
        installed view of that name -- only when they were accepted for the
        same definition. A changed definition needs an explicit acceptance
        again (the stored codes were consent to the old one)."""
        exp = self._normalize_accept(explicit)
        if exp is not None:
            return exp
        entry = catalog.get(self._resolve_catalog_name(v_spec.view_name, catalog)) or {}
        stored: Sequence[str] = ()
        if entry.get("accepted_limitations") and entry.get("accepted_definition") == _definition_digest(
            v_spec.raw_select_sql
        ):
            stored = entry["accepted_limitations"]
        return tuple(sorted(set(self.accept_limitations) | set(stored)))

    def _compile_view(
        self,
        schemas: Dict[str, TableSchema],
        view: Any,
        *,
        accept: Sequence[str],
        ignore: bool,
        strict: bool = False,
        watermark_catalog_table: Optional[str] = None,
    ) -> MaterializedViewPlan:
        view = self._classify_native_functions(schemas, view)
        plan = generate_materialized_view(
            schemas,
            view,
            dialect=self.dialect,
            ignore_limitations=ignore,
            accept_limitations=accept,
            object_prefix=self.object_prefix,
            strict=strict,
            engine_version=self.engine_version,
            watermark_catalog_table=watermark_catalog_table,
            pg_fillfactor=getattr(self, "pg_fillfactor", 80),
        )
        return plan

    def _classify_native_functions(
        self, schemas: Dict[str, TableSchema], view: Any
    ) -> Any:
        """Ask the engine catalog whether unmodelled function calls are
        aggregates (see matview.native_functions); refuse when it cannot tell.
        Calls on an engine without a readable catalog are recorded on the
        spec (``unclassified_functions``) and force the exact full refresh."""
        spec = view
        if isinstance(view, str):
            spec = parse_view_definition(view, known_tables={k.lower() for k in schemas}, dialect=self.dialect)
        if not isinstance(spec, ViewSpec):
            return view
        if (
            not spec.additional_views
            and not unclassified_calls(spec, self.dialect)
            and (self.dialect == Dialect.SQLITE or not items_with_unmodelled_calls(spec, self.dialect))
        ):
            return view
        is_mariadb = self.subdialect == EngineSubdialect.MARIADB
        spec = apply_catalog_classification(
            spec,
            self.dialect,
            lambda calls: classify_native_functions(self.conn, self.dialect, is_mariadb, calls),
        )
        return self._with_server_result_types(spec)

    def _with_server_result_types(self, spec: ViewSpec) -> ViewSpec:
        """Attach the server-reported result types of select items that call
        functions sqlglot does not model (their type cannot be derived from
        the SQL text). PostgreSQL: a temporary view's pg_attribute types,
        typmod included; MySQL/MariaDB: a temporary ``CREATE TABLE ... AS
        SELECT ... LIMIT 0``. SQLite stores such columns without affinity."""
        additional = tuple(self._with_server_result_types(a) for a in spec.additional_views)
        if additional != spec.additional_views:
            spec = replace(spec, additional_views=additional)
        if self.dialect == Dialect.SQLITE or not items_with_unmodelled_calls(spec, self.dialect):
            return spec
        types = probe_result_types(self.conn, self.dialect, spec.raw_select_sql)
        if not types:
            return spec
        return replace(spec, native_result_types=tuple(sorted(types.items())))

    def _stored_plan(self, schemas: Dict[str, TableSchema], entry: Dict[str, Any]) -> MaterializedViewPlan:
        """Recompile an installed view under exactly the limitation codes it
        was installed with (plus the manager's accept_limitations); never a
        blanket override."""
        return self._compile_view(
            schemas,
            entry["view_sql"],
            accept=self._stored_accept(entry),
            ignore=self.ignore_limitations,
        )

    def _stored_accept(self, entry: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(sorted(set(entry.get("accepted_limitations") or ()) | set(self.accept_limitations)))

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
        compiled_plan: Optional[MaterializedViewPlan] = None,
        accept_limitations: Optional[Sequence[str]] = None,
    ) -> MaterializedViewPlan:
        catalog = self.get_catalog()
        if compiled_plan is not None:
            plan = compiled_plan
        else:
            plan = self._compile_view(
                schemas,
                v_spec,
                accept=self._effective_accept(v_spec, catalog, accept_limitations),
                ignore=effective_ignore,
                strict=effective_strict,
                watermark_catalog_table=self.catalog_table if (incremental_backfill or not populate) else None,
            )
        sig = self._compute_signature(plan, schemas)

        existing = catalog.get(plan.view_name)
        if existing and existing["signature_hash"] == sig and not force_rebuild:
            if self._is_backfill_incomplete(existing):
                # Re-registering must not silently adopt an interrupted (or
                # concurrently running) backfill as a complete view.
                self._refuse_unless_backfill_unclaimed(existing, plan.view_name)
            return plan

        if existing:
            refusal = self._replacement_refusal(plan.view_name, existing, populate, incremental_backfill)
            if refusal is not None:
                raise refusal
        # A published generation is dropped inside the same transaction that
        # builds and populates its replacement (SQLite and PostgreSQL have
        # transactional DDL), so a failure at any stage restores it and
        # readers never see it missing. Only an incomplete generation (nothing
        # correct published) is dropped and committed first where that is not
        # possible: on MySQL/MariaDB, or before an incremental backfill.
        atomic_replacement = self.dialect in (Dialect.SQLITE, Dialect.POSTGRES) and not (
            populate and incremental_backfill
        )
        if existing and not atomic_replacement:
            self.drop_view(plan.view_name)

        # Install structure, maintenance triggers, single-shot backfill, and catalog record
        # inside a single atomic transaction so concurrent readers/writers never observe an
        # un-triggered or double-counted window between trigger attachment and population.
        with self._cursor() as cur:
            created_objects: List[Tuple[str, str]] = []
            catalog_inserted = False
            try:
                # sqlite3 does not open a transaction for DDL, so without an
                # explicit reservation every CREATE commits on its own and an
                # independent writer can commit after the maintenance triggers
                # exist but before the initial population reads the base
                # tables: its row is then applied by the trigger and again by
                # the population.
                self._reserve_sqlite_writer(cur)
                if existing and atomic_replacement:
                    self._drop_registered_objects(cur, plan.view_name, existing)
                for stmt in plan.create_structure_statements():
                    obj = _extract_created_object(stmt)
                    cur.execute(stmt)
                    if obj:
                        created_objects.append(obj)
                for stmt in plan.trigger_ddls:
                    obj = _extract_created_object(stmt)
                    cur.execute(stmt)
                    if obj:
                        created_objects.append(obj)

                stored_view_sql = self._serialize_view_sql_with_indexes(plan)
                init_active = 1 if (populate and incremental_backfill) else 0
                self._execute_param(
                    cur,
                    f"""
                    INSERT INTO {self.catalog_table} (
                        view_name, view_sql, strategy, signature_hash, backing_table,
                        aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json,
                        backfill_cursor_pk, backfill_max_pk, backfill_active,
                        accepted_limitations_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps({
                            "definition": _definition_digest(plan.view_spec.raw_select_sql),
                            "codes": sorted({lim.code for lim in plan.limitations}),
                        }),
                    ),
                )
                catalog_inserted = True

                if populate:
                    if incremental_backfill:
                        self._commit()
                        use_snap = not (
                            plan.incremental_backfill_spec
                            and plan.incremental_backfill_spec.mode == "pk_range_fold"
                            and plan.has_watermark_triggers
                        )
                        self.backfill_incrementally(
                            plan,
                            batch_size=batch_size,
                            on_progress=on_progress,
                            snapshot_isolation=use_snap,
                        )
                    else:
                        for stmt in plan.get_populate_statements(incremental_backfill=False):
                            cur.execute(stmt)
                        self._commit()
                else:
                    self._commit()
            except Exception as exc:
                self._rollback()
                # MySQL/MariaDB DDL is not transactional, so a failed
                # registration drops what it created. Not when the backfill
                # was refused or taken over: the committed catalog row and
                # objects then belong to another session (a repair_view,
                # rebuild or concurrent backfill), and dropping them would
                # destroy that session's view.
                if self.dialect == Dialect.MYSQL and not isinstance(exc, BackfillInterruptedError):
                    if catalog_inserted:
                        try:
                            self._execute_param(
                                cur,
                                f"DELETE FROM {self.catalog_table} WHERE view_name = ?",
                                (plan.view_name,),
                            )
                            self._commit()
                        except Exception:
                            pass
                    for kind, name in reversed(created_objects):
                        try:
                            if kind == "trigger":
                                cur.execute(f"DROP TRIGGER IF EXISTS {name};")
                            elif kind == "view":
                                cur.execute(f"DROP VIEW IF EXISTS {name};")
                            elif kind == "table":
                                cur.execute(f"DROP TABLE IF EXISTS {name};")
                        except Exception:
                            pass
                    self._commit()
                raise
        return plan

    @_manager_owned_operation("register_view")
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
        accept_limitations: Optional[Sequence[str]] = None,
    ) -> MaterializedViewPlan:
        """
        Safely installs or updates one or more materialized views on the connected database.
        Supports both single-view and multi-view SQL scripts seamlessly.

        ``accept_limitations`` names limitation codes accepted explicitly;
        codes in ``PER_CODE_OPT_IN_LIMITATIONS`` (best-effort results) are
        only overridden this way, never by ``ignore_limitations``. When it is
        None, the manager's ``accept_limitations`` plus the codes stored with
        an installed view of the same name apply. The overridden codes are
        stored with the view.
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
            dialect=self.dialect,
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
                    accept_limitations=accept_limitations,
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

    @_manager_owned_operation("sync_views")
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
        accept_limitations: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """
        Declaratively synchronizes a collection of materialized view definitions against
        the connected database (supporting multi-view SQL files/strings as well).
        Returns a status map `{view_name: "created" | "updated" | "unchanged" |
        "backfill_incomplete" | "triggers_missing" | "triggers_disabled" | "triggers_orphaned" | "dropped"}`
        ("triggers_missing": maintenance triggers of an unchanged view are gone; "triggers_disabled": a
        PostgreSQL maintenance trigger is disabled; "triggers_orphaned": a PostgreSQL
        maintenance trigger is on a DETACHed partition; repair both with `repair_view()`); "backfill_incomplete" means the definition
        is current but a pk-range backfill is running elsewhere or was interrupted
        (repair with `repair_view()`). ``accept_limitations`` behaves as in
        `register_view()`.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        effective_strict = self.strict if strict is None else strict
        schemas = self.fetch_schema(extra_schema_sql=schema_sql)
        catalog = self.get_catalog()
        trigger_drift = self._views_trigger_drift(catalog)
        status: Dict[str, str] = {}
        desired_names: Set[str] = set()

        all_specs = parse_view_definitions(
            view_sqls,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
            dialect=self.dialect,
        )

        planned: List[Tuple[Any, MaterializedViewPlan, str]] = []
        for v_spec in all_specs:
            plan = self._compile_view(
                schemas,
                v_spec,
                accept=self._effective_accept(v_spec, catalog, accept_limitations),
                ignore=effective_ignore,
                strict=effective_strict,
                watermark_catalog_table=self.catalog_table if incremental_backfill else None,
            )
            planned.append((v_spec, plan, self._compute_signature(plan, schemas)))

        # Refuse an unsafe replacement before applying anything, so a refused
        # request leaves every view as it was.
        for _, plan, sig in planned:
            existing = catalog.get(plan.view_name)
            if existing is not None and existing["signature_hash"] != sig:
                refusal = self._replacement_refusal(plan.view_name, existing, True, incremental_backfill)
                if refusal is not None:
                    raise refusal

        for v_spec, plan, sig in planned:
            v_name = plan.view_name
            desired_names.add(v_name)
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
                    compiled_plan=plan,
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
                    compiled_plan=plan,
                )
                status[v_name] = "updated"
            elif v_name in trigger_drift:
                # Same definition, but maintenance triggers are gone (e.g. a
                # failed bulk_changes() recovery) or misplaced (PostgreSQL
                # partitions attached/detached later): writes are not
                # maintained exactly.
                status[v_name] = trigger_drift[v_name]
            elif self._is_backfill_incomplete(catalog[v_name]):
                # Same definition, but a pk-range backfill is running elsewhere
                # or was interrupted: not "unchanged" (not complete).
                status[v_name] = "backfill_incomplete"
            else:
                status[v_name] = "unchanged"

        if prune_removed:
            for existing_name in list(catalog.keys()):
                if existing_name not in desired_names:
                    self.drop_view(existing_name)
                    status[existing_name] = "dropped"

        return status

    _LIVE_TRIGGER_SQL = {
        Dialect.SQLITE: "SELECT name FROM sqlite_master WHERE type = 'trigger'",
        # PostgreSQL: see _pg_trigger_drift (partition-aware).
        Dialect.POSTGRES: "SELECT tgname FROM pg_catalog.pg_trigger WHERE NOT tgisinternal",
        Dialect.MYSQL: "SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = DATABASE()",
    }

    def _views_missing_triggers(self, catalog: Dict[str, Dict[str, Any]]) -> Set[str]:
        """Registered views whose maintenance triggers are not exactly where
        they must be (see ``_views_trigger_drift``)."""
        return set(self._views_trigger_drift(catalog))

    _PG_CREATE_TRIGGER_RE = re.compile(
        r"^CREATE TRIGGER (\S+)\s+AFTER \w+ ON (\S+)", flags=re.IGNORECASE | re.MULTILINE
    )

    def _views_trigger_drift(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """``{view: "triggers_missing" | "triggers_disabled" | "triggers_orphaned"}``
        for views whose maintenance triggers drifted.

        "triggers_missing": a recorded trigger does not exist (PostgreSQL: on
        the source table or on one of its current partitions, e.g. one
        attached or created after registration). "triggers_disabled"
        (PostgreSQL): a maintenance trigger exists but does not fire for
        ordinary sessions (``ALTER TABLE ... DISABLE TRIGGER`` or ``ENABLE
        REPLICA TRIGGER``). "triggers_orphaned"
        (PostgreSQL): a maintenance trigger function also fires for a table
        that is no longer in the source's partition tree (a DETACHed
        partition, whose writes would still change the view). Never ends a
        caller transaction (savepoint-protected reads); when the trigger list
        cannot be read nothing is reported.
        """
        if not catalog:
            return {}
        if self.dialect == Dialect.POSTGRES:
            return self._pg_trigger_drift(catalog)
        sql = self._LIVE_TRIGGER_SQL.get(self.dialect)
        if sql is None:
            return {}
        with self._cursor() as cur:
            rows = self._catalog_select(cur, sql)
        if rows is None:
            return {}
        live = {str(r[0]).lower() for r in rows if r[0]}
        return {
            name: "triggers_missing" for name, entry in catalog.items()
            if any(str(t).lower() not in live for t in entry.get("trigger_names") or [])
        }

    def _pg_trigger_drift(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        with self._cursor() as cur:
            for name, entry in catalog.items():
                status: Optional[str] = None
                ddls = "\n".join(entry.get("trigger_ddls") or [])
                for trg, table in self._PG_CREATE_TRIGGER_RE.findall(ddls):
                    lit_t = "'" + table.replace("'", "''") + "'"
                    lit_n = "'" + trg.replace("'", "''") + "'"
                    rows = self._catalog_select(
                        cur,
                        f"WITH RECURSIVE __mv_tree(relid) AS ("
                        f" SELECT pg_catalog.to_regclass({lit_t})::oid"
                        f" UNION ALL SELECT i.inhrelid FROM pg_catalog.pg_inherits i"
                        f" JOIN __mv_tree ON i.inhparent = __mv_tree.relid),"
                        f" __mv_fn AS (SELECT t.tgfoid FROM pg_catalog.pg_trigger t"
                        f" WHERE t.tgrelid = pg_catalog.to_regclass({lit_t})::oid AND t.tgname = {lit_n})"
                        f" SELECT"
                        f" (SELECT count(*) FROM __mv_tree WHERE __mv_tree.relid IS NULL OR NOT EXISTS ("
                        f"  SELECT 1 FROM pg_catalog.pg_trigger t WHERE t.tgrelid = __mv_tree.relid"
                        f"  AND t.tgname = {lit_n})),"
                        f" (SELECT count(*) FROM pg_catalog.pg_trigger t WHERE t.tgfoid IN (SELECT tgfoid FROM __mv_fn)"
                        f"  AND t.tgrelid NOT IN (SELECT relid FROM __mv_tree WHERE relid IS NOT NULL)),"
                        # ALTER TABLE ... DISABLE TRIGGER ('D') or ENABLE REPLICA
                        # TRIGGER ('R'): the trigger does not fire for ordinary
                        # sessions, so their writes are not maintained.
                        f" (SELECT count(*) FROM pg_catalog.pg_trigger t JOIN __mv_tree ON t.tgrelid = __mv_tree.relid"
                        f"  WHERE t.tgname = {lit_n} AND t.tgenabled NOT IN ('O', 'A'))",
                    )
                    if rows is None:
                        continue
                    missing, orphaned, disabled = int(rows[0][0]), int(rows[0][1]), int(rows[0][2])
                    if missing:
                        status = "triggers_missing"
                        break
                    if disabled:
                        status = "triggers_disabled"
                    elif orphaned and status is None:
                        status = "triggers_orphaned"
                if status is None:
                    # Read-only/other triggers recorded by name only.
                    names = [t for t in entry.get("trigger_names") or [] if t not in ddls]
                    if names:
                        lits = ", ".join("'" + t.replace("'", "''") + "'" for t in names)
                        rows = self._catalog_select(
                            cur, f"SELECT count(DISTINCT tgname) FROM pg_catalog.pg_trigger WHERE tgname IN ({lits})"
                        )
                        if rows is not None and int(rows[0][0]) < len(set(names)):
                            status = "triggers_missing"
                if status:
                    out[name] = status
        return out

    @_manager_read_operation
    def check_schema_drift(self) -> Dict[str, str]:
        """
        Inspects all registered views in `__matview_catalog` against the current live base
        table schemas and returns `{view_name: "ok" | "drifted" | "invalid" |
        "backfill_incomplete" | "triggers_missing" | "triggers_disabled" |
        "triggers_orphaned"}`;
        "backfill_incomplete" (definition current, but a pk-range backfill is
        running elsewhere or was interrupted) is never "ok"; "triggers_missing"
        means maintenance triggers recorded for the view do not exist, also on a
        PostgreSQL partition attached or created later (writes are not
        maintained); "triggers_disabled" means a PostgreSQL maintenance trigger
        was disabled (`ALTER TABLE ... DISABLE TRIGGER`, as `pg_restore
        --disable-triggers` does; writes are not maintained); "triggers_orphaned" means a PostgreSQL maintenance trigger
        is on a table outside the source's partition tree, e.g. a DETACHed
        partition (its writes still change the view, and its rows are still in
        the view). Repair both with `repair_view()`, which rebuilds the view.
        """
        schemas = self.fetch_schema()
        catalog = self.get_catalog()
        trigger_drift = self._views_trigger_drift(catalog)
        report: Dict[str, str] = {}
        for v_name, entry in catalog.items():
            try:
                plan = self._stored_plan(schemas, entry)
            except Exception:
                report[v_name] = "invalid"
                continue
            try:
                sig = self._compute_signature(plan, schemas)
                if sig != entry["signature_hash"]:
                    report[v_name] = "drifted"
                elif v_name in trigger_drift:
                    report[v_name] = trigger_drift[v_name]
                elif self._is_backfill_incomplete(entry):
                    report[v_name] = "backfill_incomplete"
                else:
                    report[v_name] = "ok"
            except Exception:
                report[v_name] = "invalid"
        return report

    @_manager_owned_operation("repair_view")
    def repair_view(
        self,
        view_name: Optional[str] = None,
        rebuild_on_schema_drift: bool = True,
    ) -> Dict[str, str]:
        """
        Reconciles and repairs one or all registered materialized views,
        eliminating any floating-point accumulation drift or out-of-band base table DDL changes.
        Returns `{view_name: "repaired" | "rebuilt"}` ("rebuilt" also when
        maintenance triggers were missing). Each view is repaired
        in its own transaction (no all-view atomicity). Views are recompiled
        and rebuilt under exactly the limitation codes stored with them
        (never a blanket override); a view refused under them raises
        EngineLimitationError.

        A "repaired" view is repopulated in one transaction that also takes
        over the pk-range backfill watermark (records it as complete), so an
        interrupted or still-running incremental backfill stops at its next
        batch without folding more.

        Concurrent writers to the view's base tables:
        - PostgreSQL:
          - On the `"repaired"` path, acquires the catalog lock (`SHARE ROW
            EXCLUSIVE`) and then `SHARE` locks on the view's referenced base
            tables in deterministic sorted order before clearing and
            repopulating the store (with bounded retry on lock timeout /
            deadlock). Under `READ COMMITTED` (and for transactions whose
            snapshot starts after the repair commits), in-flight writers commit
            or roll back first, later writers wait until the repair commits,
            and concurrent readers (`ACCESS SHARE`) remain unblocked.
            Caveat: a concurrent `REPEATABLE READ` / `SERIALIZABLE` writer
            whose snapshot was already established *before* `repair_view()`
            committed can still observe the pre-repair catalog/backing snapshot
            in its trigger (for example an active backfill watermark or an
            empty `ATOMIC_QUERY_SYNC` store) and silently diverge.
          - On the `"rebuilt"` path (schema drift or missing triggers),
            rebuilds via `register_view(..., force_rebuild=True)`, whose
            `DROP`/`CREATE TRIGGER` DDL serializes writers (`ACCESS EXCLUSIVE`
            / `SHARE ROW EXCLUSIVE`, which also blocks readers while running).
        - MySQL/MariaDB (InnoDB, REPEATABLE READ): the repopulating
          INSERT ... SELECT takes shared locks on the base rows it reads, so
          writers serialize with the repair (verified natively with
          concurrent writers; other isolation levels are untested).
        - SQLite: writes are serialized database-wide.
        """
        catalog = self.get_catalog()
        if view_name:
            resolved = self._resolve_catalog_name(view_name, catalog)
            if resolved not in catalog:
                raise KeyError(f"View {view_name!r} is not registered in {self.catalog_table}")
            target_names = [resolved]
        else:
            target_names = list(catalog.keys())

        schemas = self.fetch_schema()
        # Views whose maintenance triggers are gone are rebuilt like drifted ones.
        missing_triggers = self._views_missing_triggers(catalog)
        results: Dict[str, str] = {}
        for idx, target_name in enumerate(target_names):
            if idx > 0:
                # Each prior view committed its own transaction, releasing the
                # catalog lock; re-acquire it and refresh metadata before
                # repairing or rebuilding the next view.
                self._lock_catalog_for_change()
                catalog = self.get_catalog()
                if target_name not in catalog:
                    continue
                schemas = self.fetch_schema()
                missing_triggers = self._views_missing_triggers(catalog)
            entry = catalog[target_name]
            v_name = entry["view_name"]
            plan = self._stored_plan(schemas, entry)
            sig = self._compute_signature(plan, schemas)
            if rebuild_on_schema_drift and (sig != entry["signature_hash"] or v_name in missing_triggers):
                # Rebuilt under exactly the acceptance stored with the view
                # (never a blanket override of per-code limitations).
                self.register_view(
                    entry["view_sql"],
                    populate=True,
                    force_rebuild=True,
                    ignore_limitations=self.ignore_limitations,
                    accept_limitations=self._stored_accept(entry),
                )
                results[v_name] = "rebuilt"
            else:
                self._repair_view_contents(plan, v_name)
                results[v_name] = "repaired"
        return results

    def _repair_view_contents(self, plan: Any, v_name: str) -> None:
        """Repopulate one view from scratch in a single transaction.

        The watermark is taken over first, in the same transaction as the
        repopulation: the catalog row is locked and marked as populated (so
        triggers stop excluding any range), which fences out any running
        backfill (its next ownership check fails and it stops). A backfill
        batch in flight locks its store rows before the catalog row, i.e. in
        the opposite order, so the engine may choose this transaction as a
        deadlock victim (or it may hit the lock wait timeout). The whole
        transaction was rolled back and is re-run from scratch, which is exact;
        after the retries are exhausted the engine error propagates with the
        view unchanged.
        """
        pg_lock_targets = (
            _postgres_base_table_lock_targets(plan, object_prefix=self.object_prefix)
            if self.dialect == Dialect.POSTGRES
            else []
        )
        attempt = 0
        while True:
            with self._cursor() as cur:
                try:
                    if self.dialect == Dialect.POSTGRES:
                        # Re-acquire the catalog lock on retries and across
                        # multiple views in one repair_view() call (each view
                        # commits its own transaction) before locking base
                        # tables, preserving catalog -> base-table lock order
                        # with bulk_changes() and other lifecycle operations.
                        self._lock_catalog_for_change()
                        if pg_lock_targets:
                            cur.execute(
                                f"LOCK TABLE {', '.join(pg_lock_targets)} IN SHARE MODE;"
                            )
                    self._execute_param(
                        cur,
                        f"UPDATE {self.catalog_table} SET backfill_cursor_pk = ?, "
                        f"backfill_max_pk = ?, backfill_active = 0, backfill_owner = NULL "
                        f"WHERE view_name = ?",
                        (self._COMPLETED_MARKER_PK, self._COMPLETED_MARKER_PK, v_name),
                    )
                    for aux_t in plan.auxiliary_table_names:
                        cur.execute(f"DELETE FROM {aux_t};")
                    for aux_ddl in plan.auxiliary_table_ddls:
                        if aux_ddl.lstrip().upper().startswith("INSERT "):
                            cur.execute(aux_ddl)
                    cur.execute(f"DELETE FROM {plan.backing_table_name};")
                    for stmt in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(stmt)
                    self._commit()
                    return
                except Exception as exc:
                    self._rollback()
                    if attempt >= self._BACKFILL_TRANSIENT_RETRIES or not (
                        self._is_transient_lock_error(exc)
                        or self._is_lock_timeout_error(exc)
                        or self._is_concurrent_group_insert_error(exc)
                    ):
                        raise
            attempt += 1
            self._backfill_sleep(self._backfill_retry_delay(attempt))

    def _shadow_name(self, tbl: str) -> str:
        suf = "__shadow"
        if len(tbl) + len(suf) <= 62:
            return f"{tbl}{suf}"
        h = hashlib.sha1(tbl.encode("utf-8")).hexdigest()[:8]
        avail = 62 - len(suf) - 9
        return f"{tbl[:avail]}_{h}{suf}"

    def _old_name(self, tbl: str) -> str:
        suf = "__old"
        if len(tbl) + len(suf) <= 62:
            return f"{tbl}{suf}"
        h = hashlib.sha1(tbl.encode("utf-8")).hexdigest()[:8]
        avail = 62 - len(suf) - 9
        return f"{tbl[:avail]}_{h}{suf}"

    def _make_replace_view_ddl(self, reader_ddl: str) -> str:
        if re.search(r"^\s*CREATE\s+OR\s+REPLACE\b", reader_ddl, flags=re.IGNORECASE):
            return reader_ddl
        pattern = re.compile(
            r"^\s*CREATE\s+(?:ALGORITHM\s*=\s*[A-Za-z0-9_]+\s+)?VIEW\b",
            flags=re.IGNORECASE,
        )
        if pattern.search(reader_ddl):
            return pattern.sub(
                lambda m: re.sub(r"(?i)\bCREATE\b", "CREATE OR REPLACE", m.group(0), count=1),
                reader_ddl,
                count=1,
            )
        return reader_ddl

    def _build_shadow_swap_sqls(
        self,
        plan: MaterializedViewPlan,
        shadow_tbl: str,
        shadow_aux_map: Dict[str, str],
    ) -> List[str]:
        sqls: List[str] = []
        if self.dialect == Dialect.MYSQL:
            old_tbl = self._old_name(plan.backing_table_name)
            old_tables = [old_tbl]
            for aux_t in plan.auxiliary_table_names:
                old_tables.append(self._old_name(aux_t))
            sqls.append(build_drop_sql("TABLE", old_tables, dialect=self.dialect))

            rename_pairs = [
                f"{plan.backing_table_name} TO {old_tbl}",
                f"{shadow_tbl} TO {plan.backing_table_name}",
            ]
            for aux_t, s_aux in shadow_aux_map.items():
                rename_pairs.append(f"{aux_t} TO {self._old_name(aux_t)}")
                rename_pairs.append(f"{s_aux} TO {aux_t}")

            sqls.append(f"RENAME TABLE {', '.join(rename_pairs)};")

            if plan.reader_view_ddl:
                sqls.append(self._make_replace_view_ddl(plan.reader_view_ddl))

            sqls.append(build_drop_sql("TABLE", old_tables, dialect=self.dialect))

        else:  # SQLite and generic fallback
            sqls.append(build_drop_sql("VIEW", plan.view_name, dialect=self.dialect))
            sqls.append(build_drop_sql("TABLE", plan.backing_table_name, dialect=self.dialect))
            sqls.append(build_alter_rename_table_sql(shadow_tbl, plan.backing_table_name, dialect=self.dialect))
            for aux_t, s_aux in shadow_aux_map.items():
                sqls.append(build_drop_sql("TABLE", aux_t, dialect=self.dialect))
                sqls.append(build_alter_rename_table_sql(s_aux, aux_t, dialect=self.dialect))
            if plan.reader_view_ddl:
                sqls.append(plan.reader_view_ddl)

        return sqls

    # ------------------------------------------------------------------
    # bulk_changes(): one manager-owned transaction, fail closed
    # ------------------------------------------------------------------

    _BULK_GUARD_SAVEPOINT = "__mv_bulk_guard"

    def _require_usable(self) -> None:
        reason = getattr(self, "_invalidated_reason", None)
        if reason is not None:
            raise BulkChangesConnectionError(
                f"this manager's connection was closed and discarded ({reason}); create a new "
                f"connection and manager"
            )

    def _invalidate_connection(self, reason: str) -> BulkChangesConnectionError:
        """Discard a connection whose state can no longer be trusted.

        Closing it makes the server (or SQLite) roll back whatever transaction
        it still has, so neither the caller nor a connection pool can commit
        half-restored state or reuse a session whose trigger-bypass flag is
        still set. SQLAlchemy pool proxies are invalidated (not returned to
        the pool). The manager refuses every later operation.
        """
        self._invalidated_reason = reason
        for layer in dbapi_connection_layers(self.conn):
            for method in ("invalidate", "close"):
                fn = getattr(layer, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        return BulkChangesConnectionError(
            f"bulk_changes(): {reason}. The connection was closed (its open transaction is rolled "
            f"back by the server) and this manager can no longer be used. Maintenance objects of "
            f"other connections were never changed outside a transaction."
        )

    def _bulk_rollback(self, reason: str) -> None:
        """Roll back the manager-owned transaction or discard the connection."""
        try:
            if hasattr(self.conn, "rollback"):
                self.conn.rollback()
            elif dbapi_transaction_state(self.conn) is True:
                with self._cursor() as cur:
                    cur.execute("ROLLBACK;")
            if dbapi_transaction_state(self.conn) is not False:
                raise RuntimeError("the transaction is still open after rollback")
        except Exception as exc:
            raise self._invalidate_connection(f"{reason}, and rolling back failed ({exc!r})") from exc

    def _begin_bulk_transaction(self, cur: Any) -> None:
        if self.dialect == Dialect.SQLITE:
            # Take the database-wide writer reservation before anything is
            # suspended: no other connection can write unmaintained rows.
            cur.execute("BEGIN IMMEDIATE;")
        elif self.dialect == Dialect.MYSQL:
            # Explicit, also on autocommit connections (COMMIT/ROLLBACK end it).
            cur.execute("START TRANSACTION;")
        # PostgreSQL: autocommit is off for the duration (see bulk_changes),
        # so the driver opens the transaction with the first statement.

    def _bulk_maintenance_ddl_drop(self, cur: Any, entry: Dict[str, Any]) -> None:
        """Drop one view's maintenance triggers (SQLite: transactional DDL)."""
        if self.dialect == Dialect.SQLITE:
            for t_name in entry["trigger_names"]:
                if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                    cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
        else:  # pragma: no cover - guarded by bulk_changes
            raise AssertionError(f"maintenance DDL is not transactional on {self.dialect}")

    def _bulk_plan(self, schemas: Dict[str, TableSchema], entry: Dict[str, Any]) -> MaterializedViewPlan:
        return self._stored_plan(schemas, entry)

    @staticmethod
    def _catalog_identity(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[Any, ...]]:
        return {
            name: (e["view_sql"], e["signature_hash"], tuple(e["trigger_names"]), tuple(e["function_names"]))
            for name, e in catalog.items()
        }

    def _bulk_require_unchanged_catalog(self, catalog: Dict[str, Dict[str, Any]], lock_clause: str) -> None:
        """Fail closed if views were registered, replaced or dropped meanwhile."""
        current = self._read_catalog(lock_clause, required=True)
        if self._catalog_identity(current) != self._catalog_identity(catalog):
            raise RuntimeError(
                "bulk_changes(): the registered materialized views changed while the block ran "
                f"(before: {sorted(catalog)}, now: {sorted(current)}); the bulk work was rolled "
                "back because the refresh could not cover them consistently"
            )

    def _bulk_refresh(
        self, cur: Any, targets: Sequence[Dict[str, Any]], use_shadow_swap: bool, recreate_triggers: bool
    ) -> None:
        """Repopulate every target view inside the current transaction.

        Each view's pk-range backfill watermark is taken over first, as
        repair_view() does: a running backfill stops at its next batch and an
        interrupted one is recorded as complete (the view is fully
        repopulated). Any error propagates; the caller rolls back.
        """
        schemas = self.fetch_schema()
        for entry in targets:
            plan = self._bulk_plan(schemas, entry)
            self._execute_param(
                cur,
                f"UPDATE {self.catalog_table} SET backfill_cursor_pk = ?, "
                f"backfill_max_pk = ?, backfill_active = 0, backfill_owner = NULL "
                f"WHERE view_name = ?",
                (self._COMPLETED_MARKER_PK, self._COMPLETED_MARKER_PK, entry["view_name"]),
            )
            if use_shadow_swap:
                self._bulk_shadow_swap(cur, plan, entry)
            else:
                for aux_t in plan.auxiliary_table_names:
                    cur.execute(f"DELETE FROM {aux_t};")
                for aux_ddl in plan.auxiliary_table_ddls:
                    if aux_ddl.lstrip().upper().startswith("INSERT "):
                        cur.execute(aux_ddl)
                cur.execute(f"DELETE FROM {plan.backing_table_name};")
                for stmt in plan.get_populate_statements(incremental_backfill=False):
                    cur.execute(stmt)
            if recreate_triggers or use_shadow_swap:
                self._bulk_maintenance_ddl_drop(cur, entry)
                for stmt in plan.trigger_ddls:
                    cur.execute(stmt)

    def _bulk_shadow_swap(self, cur: Any, plan: MaterializedViewPlan, entry: Dict[str, Any]) -> None:
        shadow_tbl = self._shadow_name(plan.backing_table_name)
        cur.execute(f"DROP TABLE IF EXISTS {shadow_tbl};")
        shadow_ddl = re.sub(
            rf"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"{re.escape(plan.backing_table_name)}\"|`{re.escape(plan.backing_table_name)}`|{re.escape(plan.backing_table_name)})\b",
            f"CREATE TABLE {shadow_tbl}",
            plan.backing_table_ddl,
            count=1,
            flags=re.IGNORECASE,
        )
        if shadow_ddl == plan.backing_table_ddl:
            shadow_ddl = plan.backing_table_ddl.replace(
                f"CREATE TABLE {plan.backing_table_name}",
                f"CREATE TABLE {shadow_tbl}",
                1,
            )
        cur.execute(shadow_ddl)

        shadow_aux_map: Dict[str, str] = {}
        for aux_t, aux_ddl in zip(plan.auxiliary_table_names, plan.auxiliary_table_ddls):
            s_aux = self._shadow_name(aux_t)
            shadow_aux_map[aux_t] = s_aux
            cur.execute(f"DROP TABLE IF EXISTS {s_aux};")
            s_aux_ddl = re.sub(
                rf"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"{re.escape(aux_t)}\"|`{re.escape(aux_t)}`|{re.escape(aux_t)})\b",
                f"CREATE TABLE {s_aux}",
                aux_ddl,
                count=1,
                flags=re.IGNORECASE,
            )
            if s_aux_ddl == aux_ddl:
                s_aux_ddl = aux_ddl.replace(f"CREATE TABLE {aux_t}", f"CREATE TABLE {s_aux}", 1)
            cur.execute(s_aux_ddl)

        def any_quoting(name: str) -> str:
            n = re.escape(name)
            return f'(?:"{n}"|`{n}`|{n})'

        for stmt in plan.get_populate_statements(incremental_backfill=False):
            s_stmt = stmt
            for verb in ("INSERT INTO", "MERGE INTO"):
                verb_re = verb.replace(" ", r"\s+")
                s_stmt = re.sub(
                    rf"\b{verb_re}\s+{any_quoting(plan.backing_table_name)}\b",
                    f"{verb} {shadow_tbl}",
                    s_stmt,
                    flags=re.IGNORECASE,
                )
            for aux_t, s_aux in shadow_aux_map.items():
                for verb in ("INSERT INTO", "MERGE INTO", "FROM", "JOIN"):
                    verb_re = verb.replace(" ", r"\s+")
                    s_stmt = re.sub(
                        rf"\b{verb_re}\s+{any_quoting(aux_t)}\b",
                        f"{verb} {s_aux}",
                        s_stmt,
                        flags=re.IGNORECASE,
                    )
            # Column references qualified by a live table name (e.g. SQLite's
            # ON CONFLICT ... SET __mv_row_count = <store>.__mv_row_count + 1)
            # must name the shadow table being populated, or the statement
            # fails (or reads the live table).
            for live_t, shadow_t in [(plan.backing_table_name, shadow_tbl), *shadow_aux_map.items()]:
                s_stmt = re.sub(
                    rf"(?<![\w.]){any_quoting(live_t)}\.", f"{shadow_t}.", s_stmt, flags=re.IGNORECASE
                )
            cur.execute(s_stmt)

        # The maintenance functions/triggers reference the backing table by
        # name; they are re-created after the swap (see _bulk_refresh).
        self._bulk_maintenance_ddl_drop(cur, entry)
        for s in self._build_shadow_swap_sqls(plan, shadow_tbl, shadow_aux_map):
            cur.execute(s)
        if self.dialect == Dialect.SQLITE:
            # The SQLite swap drops and re-creates the reader view, and with
            # it its INSTEAD OF read-only triggers. PostgreSQL's CREATE OR
            # REPLACE VIEW keeps them.
            for ro_stmt in plan.readonly_trigger_ddls:
                cur.execute(ro_stmt)
        for idx_ddl in plan.index_ddls:
            cur.execute(idx_ddl)

    @contextmanager
    def bulk_changes(
        self,
        view_names: Optional[Sequence[str]] = None,
        use_shadow_swap: bool = False,
        drop_triggers: bool = False,
    ) -> Iterator[None]:
        """Run a block of base-table DML in one manager-owned transaction.

        PostgreSQL: maintenance is never suspended. The statement-level
        triggers maintain every view once per statement during the block,
        which is faster than suspending them and repopulating the views
        (docs/design-bulk-maintenance.md) and has none of the repopulation's
        races with concurrent writers. The block only gets the transaction
        contract below. ``use_shadow_swap`` and ``drop_triggers`` (which would
        suspend maintenance) are refused with ``ValueError`` before any SQL;
        ``view_names`` is only checked for unknown names (``KeyError``).

        SQLite, MySQL/MariaDB: row-by-row maintenance is suspended for the
        block, then the affected views are repopulated, all in the same
        transaction. This is a Python-only optimisation for large loads on the
        manager's connection.

        Transaction ownership (as for the other lifecycle operations): the
        manager owns the transaction. A connection with an open transaction,
        or whose driver does not report its transaction state, is refused with
        ``TransactionOwnershipError`` before any SQL. The body's statements on
        the manager's connection, the refresh and (where used) trigger DDL run
        in one transaction that is committed when the block exits normally and
        rolled back on any exception (including ``KeyboardInterrupt``); the
        exception propagates. Autocommit connections run in one explicit
        transaction (PostgreSQL autocommit is switched off for the duration and
        restored). The body must not commit or roll back the connection:
        that is detected, the work after it is rolled back and
        ``BulkChangesTransactionError`` is raised (work the body committed
        itself stays committed). On SQLite and MySQL/MariaDB maintenance is
        then restored and the views are refreshed in a new transaction; on
        PostgreSQL nothing was suspended, so nothing needs restoring.

        How maintenance is suspended for the body on SQLite and MySQL/MariaDB
        (never globally outside a transaction):
        - MySQL/MariaDB: the session variable ``@__matview_skip_triggers``
          and DML only. ``use_shadow_swap`` and ``drop_triggers`` need DDL,
          which commits implicitly, and are refused with ``ValueError``
          before any SQL. The variable is cleared on every exit; if that
          fails the connection is closed (``BulkChangesConnectionError``).
          The session bypass suspends the triggers of every registered view
          on this connection, so ``view_names`` must name every registered
          view (``ValueError`` otherwise).
        - SQLite: ``BEGIN IMMEDIATE`` (database-wide writer reservation),
          then the maintenance triggers are dropped and re-created inside the
          transaction.

        SQLite, MySQL/MariaDB: every target view is repopulated and its
        pk-range backfill watermark is taken over in the same transaction (as
        ``repair_view`` does): a running backfill stops at its next batch, an
        interrupted one is recorded as complete.

        Failures are never swallowed: any error rolls the whole transaction
        back, which also restores the original maintenance objects. If the
        rollback itself fails, the connection is closed and the manager
        becomes unusable (``BulkChangesConnectionError``).
        """
        self._require_usable()
        if self.dialect == Dialect.POSTGRES and (use_shadow_swap or drop_triggers):
            raise ValueError(
                "bulk_changes(use_shadow_swap=True / drop_triggers=True) is not supported on "
                "PostgreSQL: its statement-level triggers maintain the views during the block "
                "(once per statement), so there is no maintenance to suspend and no refresh."
            )
        if self.dialect == Dialect.MYSQL and (use_shadow_swap or drop_triggers):
            raise ValueError(
                "bulk_changes(use_shadow_swap=True / drop_triggers=True) is not supported on "
                "MySQL/MariaDB: DDL (DROP/CREATE TRIGGER, CREATE/RENAME TABLE) commits the "
                "transaction implicitly, so the bulk work could not be rolled back and other "
                "sessions would write without maintenance. Use the default DML-only mode."
            )
        self._require_idle_connection("bulk_changes")
        # PostgreSQL keeps its statement-level maintenance for the block.
        suspends = self.dialect != Dialect.POSTGRES
        session_bypass = self.dialect == Dialect.MYSQL and not drop_triggers

        restore_autocommit: List[Any] = []
        if self.dialect == Dialect.POSTGRES:
            physical = dbapi_connection_layers(self.conn)[-1]
            if getattr(physical, "autocommit", None) is True:
                physical.autocommit = False
                restore_autocommit.append(physical)

        catalog_lock = " LOCK IN SHARE MODE" if self.dialect == Dialect.MYSQL else ""
        mysql_flag_set = False
        pending: Optional[BaseException] = None
        self._lifecycle_depth = 1
        cur: Any = None
        try:
            cur = self.conn.cursor()
            try:
                self._begin_bulk_transaction(cur)
                # Where maintenance is suspended, the set of registered views
                # must not change while the block runs (the session bypass
                # also suspends a newly registered view's triggers, and the
                # refresh would not repopulate it): MySQL/MariaDB reads the
                # catalog with a locking read (which also opens no consistent
                # snapshot: with MariaDB's innodb_snapshot_isolation the refresh
                # would otherwise fail with error 1020 on backing rows a
                # concurrent writer committed); SQLite holds the writer
                # reservation. The refresh re-checks it. PostgreSQL suspends
                # nothing, so it needs no catalog lock.
                catalog = self._read_catalog(catalog_lock, required=True)
                if view_names:
                    resolved = [self._resolve_catalog_name(n, catalog) for n in view_names]
                    unknown = [n for n, r in zip(view_names, resolved) if r not in catalog]
                    if unknown:
                        raise KeyError(f"bulk_changes(): views not registered in {self.catalog_table}: {unknown}")
                    targets = [catalog[r] for r in dict.fromkeys(resolved)]
                else:
                    targets = list(catalog.values())
                if session_bypass:
                    left_out = sorted(set(catalog) - {t["view_name"] for t in targets})
                    if left_out:
                        raise ValueError(
                            f"bulk_changes(view_names=...) on {self.dialect.value}: the session "
                            f"trigger bypass also suspends maintenance of {left_out}, which would "
                            f"then be stale; include them in view_names (or pass none)"
                        )
                if self.dialect == Dialect.MYSQL:
                    mysql_flag_set = True
                    cur.execute("SET @__matview_skip_triggers = 1;")
                if suspends and not session_bypass:
                    for entry in targets:
                        self._bulk_maintenance_ddl_drop(cur, entry)
                self._savepoint_create(cur, self._BULK_GUARD_SAVEPOINT)
            except BaseException:
                self._bulk_rollback("bulk_changes() setup failed")
                raise

            # The block runs as the caller: lifecycle operations are refused
            # inside it (the connection is in a transaction); reads join.
            self._lifecycle_depth = 0
            try:
                yield
            except BaseException:
                self._bulk_rollback("the bulk_changes() block raised")
                raise
            finally:
                self._lifecycle_depth = 1

            if mysql_flag_set:
                # Cleared inside the transaction, before anything commits: if
                # it cannot be cleared, closing the connection rolls back.
                try:
                    cur.execute("SET @__matview_skip_triggers = 0;")
                except Exception as reset_exc:
                    raise self._invalidate_connection(
                        f"the session trigger-bypass flag could not be cleared ({reset_exc!r})"
                    ) from reset_exc
                mysql_flag_set = False

            body_ended_transaction: Optional[BaseException] = None
            if dbapi_transaction_state(self.conn) is not True:
                body_ended_transaction = RuntimeError("no transaction is open after the block")
            else:
                try:
                    self._savepoint_release(cur, self._BULK_GUARD_SAVEPOINT)
                except Exception as exc:
                    body_ended_transaction = exc
            if body_ended_transaction is not None:
                self._bulk_rollback("the bulk_changes() block ended the transaction")
                if not suspends:
                    raise BulkChangesTransactionError(
                        "the bulk_changes() block committed or rolled back the manager's transaction "
                        f"({body_ended_transaction!r}). Work after that point was rolled back; work it "
                        "committed is kept (and was maintained: PostgreSQL maintenance is never suspended)"
                    ) from body_ended_transaction
                try:
                    # Always the plain refresh (never the caller's shadow-swap
                    # mode): the simplest path that restores maintenance.
                    self._begin_bulk_transaction(cur)
                    self._bulk_require_unchanged_catalog(catalog, catalog_lock)
                    self._bulk_refresh(cur, targets, False, recreate_triggers=not session_bypass)
                    self._commit()
                except BaseException as recovery_exc:
                    self._bulk_rollback("recovering from a block that ended the transaction failed")
                    names = [t["view_name"] for t in targets]
                    raise BulkChangesRecoveryError(
                        "the bulk_changes() block committed or rolled back the manager's transaction, "
                        f"and restoring maintenance/refreshing the views afterwards failed ({recovery_exc!r}). "
                        f"Views {names} do not reflect the work the block committed"
                        + (
                            "; their contents are stale although check_schema_drift() reports them "
                            "as ok (it cannot detect stale contents)"
                            if session_bypass
                            else " and their maintenance triggers are missing (check_schema_drift() "
                            "and sync_views() report \"triggers_missing\")"
                        )
                        + "; repair them with repair_view()",
                        names,
                    ) from recovery_exc
                raise BulkChangesTransactionError(
                    "the bulk_changes() block committed or rolled back the manager's transaction "
                    f"({body_ended_transaction!r}). Work after that point was rolled back; work it "
                    "committed is kept, maintenance was restored and the views were refreshed in a "
                    "new transaction"
                ) from body_ended_transaction

            try:
                if suspends:
                    self._bulk_require_unchanged_catalog(catalog, catalog_lock)
                    self._bulk_refresh(cur, targets, use_shadow_swap, recreate_triggers=not session_bypass)
                self._commit()
                if dbapi_transaction_state(self.conn) is not False:
                    raise RuntimeError(
                        f"committing did not end the transaction on "
                        f"{type(self.conn).__module__}.{type(self.conn).__qualname__}"
                    )
            except BaseException:
                self._bulk_rollback("bulk_changes() refresh/commit failed")
                raise
        except BaseException as exc:
            pending = exc
            raise
        finally:
            self._lifecycle_depth = 0
            try:
                if cur is not None:
                    try:
                        cur.close()
                    except Exception:
                        pass  # never mask the operation's own outcome
                if mysql_flag_set and getattr(self, "_invalidated_reason", None) is None:
                    # User variables are not transactional: on every failure
                    # exit (after the rollback) the flag must be cleared, or the
                    # session keeps skipping maintenance.
                    try:
                        with self._cursor() as reset_cur:
                            reset_cur.execute("SET @__matview_skip_triggers = 0;")
                    except Exception as reset_exc:
                        raise self._invalidate_connection(
                            f"the session trigger-bypass flag could not be cleared ({reset_exc!r})"
                        ) from (pending or reset_exc)
            finally:
                if getattr(self, "_invalidated_reason", None) is None:
                    for layer in restore_autocommit:
                        layer.autocommit = True

    def _query_lower_name_set(self, sql: str) -> Optional[Set[str]]:
        """Execute a metadata query returning lowercase object names, or None if unavailable."""
        try:
            with self._cursor() as cur:
                cur.execute(sql)
                return {row[0].lower() for row in cur.fetchall() if row[0]}
        except Exception:
            self._rollback()
            return None

    def _get_live_table_names(self) -> Optional[Set[str]]:
        if self.dialect == Dialect.SQLITE:
            return self._query_lower_name_set(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
            )
        if self.dialect == Dialect.POSTGRES:
            return self._query_lower_name_set(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() "
                "UNION SELECT table_name FROM information_schema.views WHERE table_schema = current_schema();"
            )
        if self.dialect == Dialect.MYSQL:
            return self._query_lower_name_set(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE();"
            )
        return None

    def _get_live_trigger_names(self) -> Optional[Set[str]]:
        if self.dialect == Dialect.SQLITE:
            return self._query_lower_name_set(
                "SELECT name FROM sqlite_master WHERE type = 'trigger';"
            )
        if self.dialect == Dialect.POSTGRES:
            return self._query_lower_name_set(
                "SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = current_schema();"
            )
        if self.dialect == Dialect.MYSQL:
            return self._query_lower_name_set(
                "SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = DATABASE();"
            )
        return None

    def _get_live_index_names(self) -> Optional[Set[str]]:
        if self.dialect == Dialect.SQLITE:
            return self._query_lower_name_set(
                "SELECT name FROM sqlite_master WHERE type = 'index';"
            )
        if self.dialect == Dialect.POSTGRES:
            return self._query_lower_name_set(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema();"
            )
        if self.dialect == Dialect.MYSQL:
            return self._query_lower_name_set(
                "SELECT DISTINCT index_name FROM information_schema.statistics WHERE table_schema = DATABASE();"
            )
        return None

    @_manager_read_operation
    def diff_views(
        self,
        view_sqls: Optional[Sequence[str | ViewSpec] | str | ViewSpec] = None,
        schema_sql: Optional[str] = None,
        prune_removed: bool = False,
        indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
        strict: Optional[bool] = None,
        ignore_limitations: Optional[bool] = None,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
        accept_limitations: Optional[Sequence[str]] = None,
    ) -> ViewDiffReport:
        """
        Compares desired compiled view plan(s) against `__matview_catalog` and live database
        metadata without mutating database schema state. Reports CREATE / REPLACE / ALTER / DROP / NOOP
        status, structured compilation diagnostics, and pending DDL statements.
        """
        effective_ignore = (
            self.ignore_limitations if ignore_limitations is None else ignore_limitations
        )
        effective_strict = self.strict if strict is None else strict
        schemas = self.fetch_schema(extra_schema_sql=schema_sql)
        catalog = self.get_catalog()
        live_tables = self._get_live_table_names()
        live_triggers = self._get_live_trigger_names()
        diffs: Dict[str, ViewDiff] = {}
        desired_names: Set[str] = set()

        if view_sqls is None:
            if not catalog:
                return ViewDiffReport({})
            specs_input: Sequence[str | ViewSpec] = [entry["view_sql"] for entry in catalog.values()]
        elif isinstance(view_sqls, (str, ViewSpec)):
            specs_input = [view_sqls]
        else:
            specs_input = view_sqls

        all_specs = parse_view_definitions(
            specs_input,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
        )

        for v_spec in all_specs:
            plan = generate_materialized_view(
                schemas,
                v_spec,
                dialect=self.dialect,
                ignore_limitations=effective_ignore,
                accept_limitations=self._effective_accept(v_spec, catalog, accept_limitations),
                object_prefix=self.object_prefix,
                strict=effective_strict,
                engine_version=self.engine_version,
                pg_fillfactor=getattr(self, "pg_fillfactor", 80),
                watermark_catalog_table=self.catalog_table if incremental_backfill else None,
            )
            v_name = plan.view_name
            desired_names.add(v_name)
            sig = self._compute_signature(plan, schemas)

            resolved_cat_name = self._resolve_catalog_name(v_name, catalog)
            existing = catalog.get(resolved_cat_name)

            if existing is None:
                pending_changes = {
                    "backing_table": {"name": plan.backing_table_name, "action": "CREATE"},
                    "auxiliary_tables": [{"name": a, "action": "CREATE"} for a in plan.auxiliary_table_names],
                    "reader_view": {"name": plan.view_name, "action": "CREATE"},
                    "triggers": {
                        "to_create": list(plan.trigger_names),
                        "to_drop": [],
                        "unchanged": [],
                    },
                    "indexes": {
                        "to_create": list(plan.index_names),
                        "to_drop": [],
                        "unchanged": [],
                    },
                }
                diffs[v_name] = ViewDiff(
                    view_name=v_name,
                    status="CREATE",
                    has_changes=True,
                    strategy=plan.strategy.value,
                    current_strategy=None,
                    signature_hash=sig,
                    current_signature_hash=None,
                    diagnostics=plan.diagnostics,
                    pending_ddls=plan.create_structure_statements() + plan.trigger_ddls,
                    pending_changes=pending_changes,
                    summary=f"View '{v_name}' is not registered; pending creation.",
                    plan=plan,
                )
            else:
                cat_sig = existing.get("signature_hash")
                cat_strat = existing.get("strategy")
                cat_backing = existing.get("backing_table", "")
                cat_trigs = set(existing.get("trigger_names", []))
                cat_trigs_lower = {t.lower() for t in cat_trigs}
                plan_trigs_lower = {t.lower() for t in plan.trigger_names}

                backing_exists = (
                    True if live_tables is None else (cat_backing.lower() in live_tables)
                )
                view_exists = (
                    True
                    if live_tables is None
                    else (
                        v_name.lower() in live_tables
                        or f"{self.object_prefix}{v_name}".lower() in live_tables
                    )
                )
                trigs_match = (
                    True if live_triggers is None else cat_trigs_lower.issubset(live_triggers)
                )

                is_clean_noop = (
                    cat_sig == sig
                    and backing_exists
                    and view_exists
                    and trigs_match
                    and cat_strat == plan.strategy.value
                )

                if is_clean_noop:
                    pending_changes = {
                        "backing_table": {"name": plan.backing_table_name, "action": "KEEP"},
                        "auxiliary_tables": [{"name": a, "action": "KEEP"} for a in plan.auxiliary_table_names],
                        "reader_view": {"name": plan.view_name, "action": "KEEP"},
                        "triggers": {
                            "to_create": [],
                            "to_drop": [],
                            "unchanged": list(plan.trigger_names),
                        },
                        "indexes": {
                            "to_create": [],
                            "to_drop": [],
                            "unchanged": list(plan.index_names),
                        },
                    }
                    diffs[v_name] = ViewDiff(
                        view_name=v_name,
                        status="NOOP",
                        has_changes=False,
                        strategy=plan.strategy.value,
                        current_strategy=cat_strat,
                        signature_hash=sig,
                        current_signature_hash=cat_sig,
                        diagnostics=plan.diagnostics,
                        pending_ddls=[],
                        pending_changes=pending_changes,
                        summary=f"View '{v_name}' is up to date; no pending changes.",
                        plan=plan,
                    )
                else:
                    query_sql_changed = (
                        _strip_sql_comments(existing.get("view_sql", "")).strip()
                        != _strip_sql_comments(self._serialize_view_sql_with_indexes(plan)).strip()
                    )
                    strategy_changed = cat_strat != plan.strategy.value
                    backing_changed = cat_backing != plan.backing_table_name
                    needs_replace = (
                        query_sql_changed
                        or strategy_changed
                        or backing_changed
                        or not backing_exists
                        or cat_sig != sig
                    )

                    status = "REPLACE" if needs_replace else "ALTER"
                    live_trigs_for_cmp = live_triggers if live_triggers is not None else cat_trigs_lower
                    to_create_trigs = [
                        t
                        for t in plan.trigger_names
                        if needs_replace or t.lower() not in cat_trigs_lower or t.lower() not in live_trigs_for_cmp
                    ]
                    to_drop_trigs = (
                        list(cat_trigs)
                        if needs_replace
                        else [t for t in cat_trigs if t.lower() not in plan_trigs_lower]
                    )
                    to_create_lower = {t.lower() for t in to_create_trigs}
                    unchanged_trigs = (
                        []
                        if needs_replace
                        else [
                            t
                            for t in plan.trigger_names
                            if t.lower() in cat_trigs_lower and t.lower() not in to_create_lower
                        ]
                    )

                    reader_action = "REPLACE" if needs_replace else ("CREATE" if not view_exists else "KEEP")
                    pending_changes = {
                        "backing_table": {
                            "name": plan.backing_table_name,
                            "action": "REPLACE" if needs_replace else "KEEP",
                            "previous": cat_backing,
                        },
                        "auxiliary_tables": [
                            {"name": a, "action": "REPLACE" if needs_replace else "KEEP"}
                            for a in plan.auxiliary_table_names
                        ],
                        "reader_view": {
                            "name": plan.view_name,
                            "action": reader_action,
                        },
                        "triggers": {
                            "to_create": to_create_trigs,
                            "to_drop": to_drop_trigs,
                            "unchanged": unchanged_trigs,
                        },
                        "indexes": {
                            "to_create": list(plan.index_names),
                            "to_drop": [],
                            "unchanged": [],
                        },
                    }
                    if needs_replace:
                        pending_ddls = plan.drop_statements() + plan.create_structure_statements() + plan.trigger_ddls
                        summary = f"View '{v_name}' query, schema, or strategy changed; pending REPLACE."
                    else:
                        trig_patterns = [
                            re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", re.IGNORECASE)
                            for t in to_create_trigs
                        ]
                        needed_fn_names: Set[str] = set()
                        matched_trig_ddls: List[str] = []
                        for stmt in plan.trigger_ddls:
                            if any(p.search(stmt) for p in trig_patterns):
                                matched_trig_ddls.append(stmt)
                                fn_m = re.search(
                                    r"\bEXECUTE\s+(?:FUNCTION|PROCEDURE)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                                    stmt,
                                    re.IGNORECASE,
                                )
                                if fn_m:
                                    needed_fn_names.add(fn_m.group(1).lower())
                        fn_ddls = [
                            stmt
                            for stmt in plan.trigger_ddls
                            if any(
                                re.search(
                                    rf"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:\"?{re.escape(fn)}\"?)\b",
                                    stmt,
                                    re.IGNORECASE,
                                )
                                for fn in needed_fn_names
                            )
                            and stmt not in matched_trig_ddls
                        ]
                        view_repair_ddls = (
                            [plan.reader_view_ddl, *plan.readonly_trigger_ddls]
                            if not view_exists
                            else []
                        )
                        pending_ddls = view_repair_ddls + fn_ddls + matched_trig_ddls + list(plan.index_ddls)
                        summary = f"View '{v_name}' auxiliary objects changed; pending ALTER."

                    diffs[v_name] = ViewDiff(
                        view_name=v_name,
                        status=status,
                        has_changes=True,
                        strategy=plan.strategy.value,
                        current_strategy=cat_strat,
                        signature_hash=sig,
                        current_signature_hash=cat_sig,
                        diagnostics=plan.diagnostics,
                        pending_ddls=pending_ddls,
                        pending_changes=pending_changes,
                        summary=summary,
                        plan=plan,
                    )

        if prune_removed:
            for cat_name, cat_entry in catalog.items():
                if cat_name not in desired_names:
                    drop_ddls: List[str] = []
                    if self.dialect != Dialect.POSTGRES:
                        for t_name in cat_entry.get("trigger_names", []):
                            drop_ddls.append(f"DROP TRIGGER IF EXISTS {t_name};")
                    for fn_name in cat_entry.get("function_names", []):
                        drop_ddls.append(f"DROP FUNCTION IF EXISTS {fn_name if '(' in fn_name else fn_name + '()'} CASCADE;")
                    drop_ddls.append(f"DROP VIEW IF EXISTS {cat_name};")
                    if cat_entry.get("backing_table"):
                        drop_ddls.append(f"DROP TABLE IF EXISTS {cat_entry['backing_table']};")
                    for aux_t in cat_entry.get("aux_tables", []):
                        drop_ddls.append(f"DROP TABLE IF EXISTS {aux_t};")

                    pending_changes = {
                        "backing_table": {
                            "name": cat_entry.get("backing_table", ""),
                            "action": "DROP",
                        },
                        "auxiliary_tables": [
                            {"name": a, "action": "DROP"} for a in cat_entry.get("aux_tables", [])
                        ],
                        "reader_view": {"name": cat_name, "action": "DROP"},
                        "triggers": {
                            "to_create": [],
                            "to_drop": list(cat_entry.get("trigger_names", [])),
                            "unchanged": [],
                        },
                        "indexes": {"to_create": [], "to_drop": [], "unchanged": []},
                    }
                    diffs[cat_name] = ViewDiff(
                        view_name=cat_name,
                        status="DROP",
                        has_changes=True,
                        strategy=None,
                        current_strategy=cat_entry.get("strategy"),
                        signature_hash=None,
                        current_signature_hash=cat_entry.get("signature_hash"),
                        diagnostics=None,
                        pending_ddls=drop_ddls,
                        pending_changes=pending_changes,
                        summary=f"View '{cat_name}' is no longer declared; pending DROP.",
                        plan=None,
                    )

        return ViewDiffReport(diffs)

    @_manager_read_operation
    def diff_view(
        self,
        view_sql: str | ViewSpec,
        schema_sql: Optional[str] = None,
        indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
        strict: Optional[bool] = None,
        ignore_limitations: Optional[bool] = None,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> ViewDiff:
        """
        Compares a single desired view definition against `__matview_catalog` and live database
        metadata without mutating database schema state. Returns a `ViewDiff`.
        """
        report = self.diff_views(
            view_sqls=[view_sql],
            schema_sql=schema_sql,
            prune_removed=False,
            indexes=indexes,
            strict=strict,
            ignore_limitations=ignore_limitations,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
        )
        if report:
            return next(iter(report.values()))
        raise ValueError("No view definition found to diff.")


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
    ignore_limitations: bool = False,
    accept_limitations: Sequence[str] = (),
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
    strict: bool = False,
    engine_version: Optional[EngineVersion | str] = None,
    watermark_catalog_table: Optional[str] = None,
    pg_fillfactor: Optional[int] = 80,
) -> MaterializedViewPlan:
    """
    Compiles a `CREATE VIEW` SQL definition (or multi-view SQL script) and base table schema
    into a `MaterializedViewPlan`. Autodetects the database dialect and version when passed
    a live connection, or accepts `engine_version` (e.g. `'postgres:17'`, `'mariadb:11.4'`, `'mysql:8.4'`).
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict, dialect=eff_dialect)
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
            dialect=eff_dialect,
        )
    elif indexes:
        parsed_idxs = tuple(
            parse_view_index_spec(ix, default_view_name=view_sql_or_spec.view_name, dialect=eff_dialect)
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
            native_aggregates=view_sql_or_spec.native_aggregates,
            native_scalar_types=view_sql_or_spec.native_scalar_types,
            native_result_types=view_sql_or_spec.native_result_types,
            unclassified_functions=view_sql_or_spec.unclassified_functions,
            functions_classified=view_sql_or_spec.functions_classified,
        )
    else:
        view_spec = view_sql_or_spec

    compiler = IncrementalViewCompiler(
        schemas=schemas,
        view_spec=view_spec,
        dialect=eff_dialect,
        ignore_limitations=ignore_limitations,
        accept_limitations=accept_limitations,
        object_prefix=object_prefix,
        incremental_backfill=incremental_backfill,
        batch_size=batch_size,
        strict=strict,
        engine_version=resolved_ev,
        watermark_catalog_table=watermark_catalog_table,
        pg_fillfactor=pg_fillfactor,
    )
    return compiler.compile()


def generate_materialized_views(
    schema_sql_or_dict: str | Dict[str, TableSchema] | sqlite3.Connection | Any,
    view_sqls: Sequence[str | ViewSpec] | str | ViewSpec,
    dialect: Optional[Dialect | str | EngineVersion] = None,
    ignore_limitations: bool = False,
    accept_limitations: Sequence[str] = (),
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    incremental_backfill: bool = False,
    batch_size: int = 1000,
    strict: bool = False,
    engine_version: Optional[EngineVersion | str] = None,
    pg_fillfactor: Optional[int] = 80,
) -> List[MaterializedViewPlan]:
    """
    Compiles one or more `CREATE VIEW` SQL definitions (including multi-view SQL scripts)
    into a flat list of `MaterializedViewPlan` objects.
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict, dialect=eff_dialect)
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
        dialect=eff_dialect,
    )
    plans: List[MaterializedViewPlan] = []
    for vs in specs:
        comp = IncrementalViewCompiler(
            schemas=schemas,
            view_spec=vs,
            dialect=eff_dialect,
            ignore_limitations=ignore_limitations,
            accept_limitations=accept_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
            strict=strict,
            engine_version=resolved_ev,
            pg_fillfactor=pg_fillfactor,
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
    accept_limitations: Sequence[str] = (),
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    strict: bool = False,
) -> MaterializedViewPlan:
    """Installs one or more materialized views directly into a live SQLite connection."""
    mgr = MaterializedViewManager(
        conn,
        dialect=Dialect.SQLITE,
        ignore_limitations=ignore_limitations,
        accept_limitations=accept_limitations,
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

