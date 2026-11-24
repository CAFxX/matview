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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

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
    overridden_modelled_functions,
    unclassified_calls,
)
from .connectivity import (
    connect_database_via_sqlalchemy,
    dbapi_connection_layers,
    dbapi_transaction_state,
    detect_engine_version,
    introspect_database_schemas,
    introspect_plain_views,
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
    with_probed_mysql_collation,
    with_probed_sqlite_statistics,
)
from .logical_ir import (
    GROUP_KEY_PLACEHOLDERS,
    ColumnDef,
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
    _strip_outer_ident_quotes,
    _strip_sql_comments,
    build_alter_rename_table_sql,
    build_create_view_sql,
    build_drop_sql,
    parse_plain_views,
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
# 9: PostgreSQL join views serialize concurrent writers across sources via the
#    per-view lock table and fixed-width view-hash trigger prefixes.
# 10: PostgreSQL group-scoped lock tables use bounded hash buckets (1..1024)
#     with fillfactor = 50 and full-bucket whole-view locks instead of FOR SHARE
#     on row 0 (eliminating unbounded lock-row growth and MultiXactId churn).
# 11: PostgreSQL canonical keys support composites and multiranges, targetlist
#     array unnest preserves composite elements, container nullness uses
#     IS [NOT] DISTINCT FROM NULL for partially-/all-NULL composites, and
#     registration probes refuse unsupported key types/collations up front.
# 12: PostgreSQL Stage 3 multi-source statement maintenance via BEFORE/AFTER
#     bracket and inclusion-exclusion reconciliation.
# 13: Fail-closed incremental backfill: triggers gate on unclaimed
#     (backfill_cursor_pk IS NULL, backfill_active = 1) watermarks across all
#     dialects, and PostgreSQL watermark predicates take FOR SHARE on matched
#     catalog rows to coordinate with chunk advances against phantom inserts.
# 14: Stage 4 lean row-trigger bodies on SQLite, MySQL, and MariaDB: key-scoped
#     join empty-group deletes, inlined join delta triggers (dropping single-row
#     derived tables and redundant GROUP BY), reader-view derived AVG, VALUES
#     upserts without SELECT/FROM DUAL WHERE TRUE, and same-group UPDATE fast paths.
# 15: PostgreSQL manager-installed statement triggers fence REPEATABLE READ /
#     SERIALIZABLE snapshots against repair_view(), rebuild, and backfill via
#     SELECT ... FOR SHARE on the view's catalog row (raising 40001 on stale
#     or missing catalog visibility; zero locks under READ COMMITTED).
# 16: PostgreSQL source-table maintenance triggers are ENABLE ALWAYS (they fire
#     under session_replication_role = replica); drift requires tgenabled = 'A'.
MATVIEW_GENERATOR_VERSION = "16"


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
    ident_pat = r'(?:"(?:""|[^"])+"|`(?:``|[^`])+`|\[[^\]]+\]|\S+)'
    m = re.match(rf"\s*CREATE\s+VIEW\s+{ident_pat}\s+AS\s+(.*)$", first, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else first

# How to repopulate a view whose pk-range backfill is incomplete.
_REPAIR_GUIDANCE = (
    "to (re)populate from scratch use register_view(..., force_rebuild=True), "
    "drop_view() + register_view(), or repair_view()."
)


def _is_postgres_cte_in_scope(
    tbl_node: sqlglot_exp.Table,
    *,
    case_insensitive: bool = False,
    mysql_strict: bool = False,
) -> bool:
    """Return True if an unqualified table node resolves to an enclosing CTE
    under PostgreSQL scoping rules (non-recursive WITH CTEs see only earlier
    CTEs; WITH RECURSIVE CTEs see every CTE in the same WITH clause).

    Pass ``case_insensitive=True`` for MariaDB (and MySQL with
    ``lower_case_table_names``) where CTE names are compared case-insensitively
    even when backtick-quoted.  Pass ``mysql_strict=True`` for the strictest MySQL
    rules instead: names compare exactly (case-sensitive on ``lower_case_table_names=0``
    servers, quoted or not) and a CTE sees only earlier siblings (plus itself under
    ``WITH RECURSIVE``), never later ones.
    """
    if tbl_node.db or tbl_node.catalog:
        return False
    id_node = tbl_node.this
    if not isinstance(id_node, sqlglot_exp.Identifier):
        return False
    def _key(ident: sqlglot_exp.Identifier) -> str:
        if mysql_strict:
            return ident.name
        return ident.name.lower() if (case_insensitive or not ident.quoted) else ident.name

    target = _key(id_node)
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
            elif mysql_strict and cte_in_prev_with is not None and cte_in_prev_with in ctes:
                # MySQL never resolves a forward sibling, even under WITH RECURSIVE.
                visible = ctes[: ctes.index(cte_in_prev_with) + 1]
            else:
                visible = ctes
            for cte in visible:
                alias_node = cte.args.get("alias")
                if isinstance(alias_node, sqlglot_exp.TableAlias) and isinstance(
                    alias_node.this, sqlglot_exp.Identifier
                ):
                    cte_id = alias_node.this
                    if _key(cte_id) == target:
                        return True
        prev_node = cur_node
        cur_node = cur_node.parent
    return False


def _postgres_ident_sort_key(part: Any) -> Optional[str]:
    if isinstance(part, sqlglot_exp.Identifier):
        return part.name if part.quoted else part.name.lower()
    return None


def _quote_postgres_fallback_table(ref_t: str) -> Tuple[Tuple[str, str, str], str]:
    """Quote a fallback table reference (possibly schema-qualified or containing
    literal dots) for PostgreSQL and return ``((table_key, schema_key, catalog_key), sql_ref)``."""
    try:
        parts = [p.strip() for p in _split_top_level(ref_t, ".", dialect=Dialect.POSTGRES) if p.strip()]
    except sqlglot.errors.SqlglotError:
        parts = [ref_t.strip()] if ref_t.strip() else []
    quoted_parts = [quote_sql_identifier(p, Dialect.POSTGRES, allow_qualified=False) for p in parts]
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


def _mysql_base_table_lock_stmts(
    plan: Any,
    *,
    schemas: Optional[Dict[str, TableSchema]] = None,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    current_db: Optional[str] = None,
    foreign_schema: Optional[Callable[[str, str], Optional[TableSchema]]] = None,
) -> List[str]:
    """Return deterministically ordered MySQL/MariaDB InnoDB shared-lock probe
    statements (`LOCK IN SHARE MODE`) for every base table referenced by ``plan``.

    Under a transaction-local ``REPEATABLE READ`` snapshot, scanning each base
    table's clustered index with ``LOCK IN SHARE MODE`` acquires next-key shared
    locks over every row, gap, and supremum — serializing ``repair_view()`` with
    concurrent writers under ``READ COMMITTED``, ``REPEATABLE READ``, and
    ``SERIALIZABLE`` regardless of ``binlog_format`` (``ROW``/``MIXED``/``STATEMENT``).

    ``schemas`` describes the current database ``current_db``: a table qualified with
    that database is the same table as the unqualified one (same dedup key, same probe).
    A table of another database is described by ``foreign_schema(db, table)`` when
    given; only when its keys stay unknown does it get the generic probe, which scans
    the smallest index and so does not lock every row (it can miss writers touching
    only non-indexed columns).
    """
    sub_plans = list(getattr(plan, "sub_plans", None) or ())
    plans = sub_plans if sub_plans else [plan]
    internal_prefixes = tuple({p.lower() for p in (object_prefix, DEFAULT_OBJECT_PREFIX, "__mv_") if p})
    schemas_lower: Dict[str, TableSchema] = {k.lower(): v for k, v in (schemas or {}).items()}
    # Map (db, table) exactly as written -> (leaf_table_name, sql_table_ref).  Never case-fold:
    # with lower_case_table_names=0 `ca_x` and `CA_x` are different tables, and a (db, table)
    # pair must keep its qualifier (`other.xt` is not `xt`).  On case-insensitive servers the
    # worst case is a duplicate (harmless) shared lock on the same table.
    targets_by_key: Dict[Tuple[str, str], Tuple[str, str]] = {}

    for sp in plans:
        vs = getattr(sp, "view_spec", None)
        if vs is None:
            continue
        raw_select = getattr(vs, "raw_select_sql", None) or ""
        found_from_ast = False
        if raw_select.strip():
            try:
                ast_node = parse_sql_ast(raw_select, dialect=Dialect.MYSQL)
                if isinstance(ast_node, sqlglot_exp.Create) and isinstance(
                    ast_node.expression, sqlglot_exp.Expr
                ):
                    ast_node = ast_node.expression
                for tbl_node in ast_node.find_all(sqlglot_exp.Table):
                    this_part = tbl_node.this
                    if not isinstance(this_part, sqlglot_exp.Identifier) or not this_part.name:
                        continue
                    leaf_low = this_part.name.lower()
                    # Skip a name only when it certainly is a CTE.  Servers differ (MySQL with
                    # lower_case_table_names=0 compares CTE names case-sensitively and never
                    # resolves forward siblings; MariaDB does neither), so a name that is a CTE
                    # only under the looser rules but also names a real table is locked anyway
                    # (an extra shared lock is harmless, a missing one is not).
                    if _is_postgres_cte_in_scope(tbl_node, mysql_strict=True):
                        continue
                    if _is_postgres_cte_in_scope(tbl_node, case_insensitive=True) and (
                        schemas is None or leaf_low not in schemas_lower
                    ):
                        continue
                    if leaf_low.startswith(internal_prefixes):
                        continue
                    db_part = tbl_node.args.get("db")
                    db_key = (
                        db_part.name
                        if isinstance(db_part, sqlglot_exp.Identifier) and db_part.name
                        else ""
                    )
                    if current_db is not None and db_key == current_db:
                        db_key = ""  # self-qualified: the very same table as the bare name
                    clean_tbl = sqlglot_exp.Table(
                        this=this_part.copy(),
                        db=db_part.copy() if isinstance(db_part, sqlglot_exp.Expr) else None,
                    )
                    key = (db_key, this_part.name)
                    if key not in targets_by_key:
                        targets_by_key[key] = (this_part.name, clean_tbl.sql(dialect="mysql"))
                    found_from_ast = True
            except sqlglot.errors.SqlglotError:
                pass

        if not found_from_ast:
            for ref_t in getattr(vs, "referenced_tables", ()) or ():
                ref_s = str(ref_t).strip()
                if not ref_s:
                    continue
                parts: List[str] = []
                for raw_p in ref_s.split("."):
                    sp_p = raw_p.strip()
                    if not sp_p:
                        continue
                    if len(sp_p) >= 2 and sp_p[0] == "`" and sp_p[-1] == "`":
                        sp_p = sp_p[1:-1].replace("``", "`")
                    parts.append(sp_p)
                if not parts:
                    continue
                leaf = parts[-1]
                if leaf.lower().startswith(internal_prefixes):
                    continue
                db_key = parts[-2] if len(parts) >= 2 else ""
                if current_db is not None and db_key == current_db:
                    db_key = ""
                q_ref = ".".join(quote_sql_identifier(p, Dialect.MYSQL) for p in parts)
                key = (db_key, leaf)
                if key not in targets_by_key:
                    targets_by_key[key] = (leaf, q_ref)

    stmts: List[str] = []
    for (db_key, _), (leaf_name, q_tbl) in sorted(targets_by_key.items()):
        # `schemas` only describes the current database; another database is looked up per table
        # and a name that is ambiguous between tables differing only in case stays unknown.
        ts: Optional[TableSchema] = None
        if db_key:
            ts = foreign_schema(db_key, leaf_name) if foreign_schema is not None else None
        elif schemas:
            ts = schemas.get(leaf_name)
            if ts is None:
                same_name = [v for k, v in schemas.items() if k.lower() == leaf_name.lower()]
                ts = same_name[0] if len(same_name) == 1 else None
        # Check `ts.pk_columns` (populated iff a genuine PRIMARY index exists),
        # NOT `c.primary_key`: MySQL/MariaDB information_schema.columns marks
        # the first UNIQUE NOT NULL index as COLUMN_KEY='PRI' even when no
        # PRIMARY index exists, where FORCE INDEX (PRIMARY) raises Error 1176.
        if ts is not None and ts.pk_columns:
            stmts.append(f"SELECT COUNT(*) FROM {q_tbl} FORCE INDEX (PRIMARY) LOCK IN SHARE MODE;")
        elif ts is not None and ts.columns:
            # Portable null counter (``ISNULL()`` is MySQL-only and breaks the offline
            # SQLite-backed adapter that replays this statement).
            col_terms = " + ".join(
                f"(CASE WHEN {quote_sql_identifier(c, Dialect.MYSQL)} IS NULL THEN 1 ELSE 0 END)"
                for c in ts.columns
            )
            stmts.append(f"SELECT SUM({col_terms}) FROM {q_tbl} LOCK IN SHARE MODE;")
        else:
            stmts.append(f"SELECT COUNT(*) FROM {q_tbl} LOCK IN SHARE MODE;")
    return stmts


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
            # A pinned version skips detection, not the probe of the connection's
            # statistical functions (they decide what the original view returns).
            _raw_conn = unwrap_dbapi_connection(self.conn)[0]
            self.engine_version: EngineVersion = with_probed_mysql_collation(
                _raw_conn,
                with_probed_sqlite_statistics(
                    _raw_conn, parse_engine_version(engine_version, default_dialect=hint_d)
                ),
            )
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
            # Probe the catalog table with a plain read before issuing DDL:
            # on MySQL/MariaDB, CREATE TABLE IF NOT EXISTS requests an exclusive
            # metadata lock (MDL_EXCLUSIVE) even when the table already exists,
            # which stalls behind any active writer whose trigger watermark read
            # holds MDL_SHARED_READ on the catalog table.
            # MySQL/MariaDB only: on PostgreSQL the probe would resolve the catalog through
            # search_path, while CREATE TABLE places a missing one in the first schema of it.
            catalog_exists = False
            if self.dialect == Dialect.MYSQL:
                try:
                    cur.execute(f"SELECT 1 FROM {self.catalog_table} WHERE 1 = 0")
                    cur.fetchall()
                    self._commit()
                    catalog_exists = True
                except Exception:
                    self._rollback()
            if not catalog_exists:
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
                        accepted_limitations_json TEXT DEFAULT NULL,
                        partitions_json TEXT DEFAULT NULL
                    );
                    """
                )
                self._commit()
            # A catalog from another version of the generator (missing a column) would make every
            # catalog read fail and look like "no catalog": fail loudly instead.
            try:
                cur.execute(f"SELECT {', '.join(self._CATALOG_COLUMNS)} FROM {self.catalog_table} WHERE 1 = 0")
                cur.fetchall()
                self._commit()
            except Exception as exc:
                self._rollback()
                raise RuntimeError(
                    f"the materialized-view catalog {self.catalog_table} does not have the expected columns "
                    f"({', '.join(self._CATALOG_COLUMNS)}): it was created by another version of matview. "
                    f"Drop the catalog and the materialized views it lists (the project does not migrate "
                    f"catalogs) and register the views again. Cause: {exc}"
                ) from exc

    _CATALOG_COLUMNS: Tuple[str, ...] = (
        "view_name", "view_sql", "strategy", "signature_hash", "backing_table", "aux_tables_json",
        "trigger_names_json", "function_names_json", "trigger_ddls_json", "backfill_cursor_pk",
        "backfill_max_pk", "backfill_active", "backfill_owner", "accepted_limitations_json", "partitions_json",
    )

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

    def _fetch_plain_views(
        self,
        extra_schema_sql: Optional[str] = None,
        exclude_views: Sequence[str] = (),
    ) -> Tuple[Dict[str, str], Optional[str]]:
        """Fetches live plain (non-materialized) SQL views, merged with any in `extra_schema_sql`."""
        plain_views, current_schema = introspect_plain_views(
            self.conn,
            dialect=self.dialect,
            object_prefix=self.object_prefix,
            exclude_views=exclude_views,
        )
        if extra_schema_sql:
            excl_low = {str(v).lower() for v in exclude_views if v}
            for pv_name, pv_sql in parse_plain_views(extra_schema_sql, dialect=self.dialect).items():
                if pv_name.lower() not in excl_low:
                    plain_views[pv_name] = pv_sql
        return plain_views, current_schema

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
        plain_views, cur_schema = self._fetch_plain_views(exclude_views=list(catalog.keys()))
        affected = []
        for name, entry in published.items():
            try:
                specs = parse_view_definitions(
                    entry["view_sql"],
                    dialect=self.dialect,
                    plain_views=plain_views,
                    current_schema=cur_schema,
                )
                referenced = {_strip_outer_ident_quotes(r).lower() for s in specs for r in s.referenced_tables}
                referenced.update(pv_name.lower() for s in specs for pv_name, _ in s.referenced_plain_views)
            except Exception:
                affected.append(name)
                continue
            if referenced & touched:
                affected.append(name)
        return sorted(affected)

    def _catalog_select(
        self, cur: Any, sql: str, errors: Optional[List[BaseException]] = None
    ) -> Optional[List[Any]]:
        """Run one catalog read; None if it failed (e.g. no catalog yet; the
        exception is appended to ``errors`` when given).

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
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                self._savepoint_rollback_and_release(cur, "__matview_catalog_read")
                return None
            self._savepoint_release(cur, "__matview_catalog_read")
            return rows
        try:
            cur.execute(sql)
            return cur.fetchall()
        except Exception as exc:
            if state is None:
                raise
            if errors is not None:
                errors.append(exc)
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
            # One statement reads every catalog column this method needs (the catalog is
            # created with all of them: there are no catalogs from older versions).
            rows = self._catalog_select(
                cur,
                f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json, "
                f"backfill_cursor_pk, backfill_max_pk, backfill_active, "
                f"accepted_limitations_json, partitions_json "
                f"FROM {self.catalog_table}{lock_clause}",
            )
            if rows is None:
                if required:
                    raise RuntimeError(
                        f"the materialized-view catalog {self.catalog_table} could not be read"
                    )
                return {}
        accepted_rows = [(r[0], r[12]) for r in rows]
        partition_rows = [(r[0], r[13]) for r in rows]
        accepted_by_view: Dict[str, Any] = {}
        for arow in accepted_rows or ():
            try:
                accepted_by_view[arow[0]] = None if arow[1] is None else json.loads(arow[1])
            except (TypeError, ValueError):
                accepted_by_view[arow[0]] = {"definition": None, "codes": []}
        partitions_by_view: Dict[str, Any] = {}
        partitions_malformed_by_view: Dict[str, bool] = {}
        for prow in partition_rows or ():
            if prow[1] is None:
                partitions_by_view[prow[0]] = None
                partitions_malformed_by_view[prow[0]] = False
                continue
            try:
                parsed_parts = json.loads(prow[1])
                if not isinstance(parsed_parts, dict) or not all(
                    isinstance(k, str)
                    and isinstance(v, list)
                    and all(
                        isinstance(item, list)
                        and len(item) == 3
                        and all(isinstance(elem, str) for elem in item)
                        for item in v
                    )
                    for k, v in parsed_parts.items()
                ):
                    partitions_by_view[prow[0]] = None
                    partitions_malformed_by_view[prow[0]] = True
                else:
                    partitions_by_view[prow[0]] = parsed_parts
                    partitions_malformed_by_view[prow[0]] = False
            except (TypeError, ValueError):
                partitions_by_view[prow[0]] = None
                partitions_malformed_by_view[prow[0]] = True
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
                "backfill_cursor_pk": row[9],
                "backfill_max_pk": row[10],
                "backfill_active": int(row[11]) if row[11] is not None else 0,
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
            out[row[0]]["partitions"] = partitions_by_view.get(row[0])
            out[row[0]]["partitions_malformed"] = partitions_malformed_by_view.get(row[0], False)
        return out

    def _compute_signature(self, plan: MaterializedViewPlan, schemas: Dict[str, TableSchema]) -> str:
        h = hashlib.sha256()
        # The generated maintenance shape is part of the signature: a view
        # installed by an older generator (different backing columns, e.g. the
        # SQLite exact-SUM bookkeeping columns, or a bumped generator version
        # for trigger-only changes) is then rebuilt by register_view/sync_views
        # and repair_view instead of being kept with stale triggers.
        h.update(f"generator:{MATVIEW_GENERATOR_VERSION}".encode("utf-8"))
        h.update(f"strategy:{plan.strategy.value}".encode("utf-8"))
        h.update(f"backing:{plan.backing_table_ddl}".encode("utf-8"))
        h.update(f"reader:{plan.reader_view_ddl}".encode("utf-8"))
        h.update(_strip_sql_comments(plan.view_spec.raw_select_sql).encode("utf-8"))
        for pv_name, pv_def in sorted(plan.view_spec.referenced_plain_views, key=lambda kv: kv[0].lower()):
            h.update(f"pv:{pv_name.lower()}:{pv_def}".encode("utf-8"))
        h.update(f"prefix:{plan.object_prefix}".encode("utf-8"))
        for idx_ddl in plan.index_ddls:
            h.update(f"idx:{idx_ddl}".encode("utf-8"))
        low_map = {k.lower(): v for k, v in schemas.items()}
        for ref_t in sorted(plan.view_spec.referenced_tables, key=str.lower):
            ts = low_map.get(ref_t.lower())
            if ts:
                for c_name, c_def in sorted(ts.columns.items()):
                    vgen_sig = f":vgen:{c_def.generation_expr or ''}" if c_def.generated_virtual else ""
                    h.update(
                        f"{ts.name}.{c_name}:{c_def.data_type}:{c_def.not_null}:{c_def.primary_key}{vgen_sig}".encode("utf-8")
                    )
                if ts.fk_cascade_actions:
                    h.update(
                        f"{ts.name}:fk:{','.join(sorted(set(ts.fk_cascade_actions)))}".encode("utf-8")
                    )
                if ts.without_rowid:
                    h.update(f"{ts.name}:without_rowid".encode("utf-8"))
                if ts.strict:
                    h.update(f"{ts.name}:strict".encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _serialize_view_sql_with_indexes(plan: MaterializedViewPlan, raw_view_sql: Optional[str] = None) -> str:
        q_vname = quote_sql_identifier(plan.logical_view_name, plan.dialect, allow_qualified=False)
        src_select = plan.view_spec.original_select_sql or plan.view_spec.raw_select_sql
        exposed_low = {c.lower() for c in plan.user_columns if c} | {
            it.alias.lower() for it in plan.view_spec.select_items if it.alias
        }
        parts = [f"CREATE VIEW {q_vname} AS {src_select.strip().rstrip(';')};"]
        for idx_spec in plan.user_indexes:
            u_kw = "UNIQUE " if idx_spec.unique else ""
            n_str = f"{quote_sql_identifier(idx_spec.name, plan.dialect, allow_qualified=False)} " if idx_spec.name else ""
            using_str = f" USING {idx_spec.using_method}" if idx_spec.using_method else ""
            w_str = f" WHERE {idx_spec.where_clause}" if idx_spec.where_clause else ""
            ser_cols = [
                quote_sql_identifier(c, plan.dialect, allow_qualified=False)
                if c.strip().lower() in exposed_low
                else c
                for c in idx_spec.columns
            ]
            parts.append(
                f"CREATE {u_kw}INDEX {n_str}ON {q_vname}{using_str} "
                f"({', '.join(ser_cols)}){w_str};"
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
                cur.execute(build_drop_sql("VIEW", db_v, dialect=self.dialect, allow_qualified=False))
                cur.execute(build_drop_sql("VIEW", raw_logical, dialect=self.dialect, allow_qualified=False))
                cur.execute(build_drop_sql("TABLE", f"{self.object_prefix}store_{raw_logical}", dialect=self.dialect, allow_qualified=False))
                if not re.fullmatch(r"[A-Za-z0-9_]+", raw_logical):
                    p = self.object_prefix if self.object_prefix else "__matview_"
                    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw_logical).strip("_").lower() or "v"
                    h = hashlib.sha1(raw_logical.encode("utf-8")).hexdigest()[:8]
                    avail = max(1, 62 - len(f"{p}store_") - 9)
                    cur.execute(build_drop_sql("TABLE", f"{p}store_{safe[:avail]}_{h}", dialect=self.dialect, allow_qualified=False))
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
                cur.execute(build_drop_sql("TRIGGER", t_name, dialect=self.dialect, allow_qualified=False))
        for fn_name in entry["function_names"]:
            cur.execute(f"DROP FUNCTION IF EXISTS {fn_name if '(' in fn_name else fn_name + '()'} CASCADE;")
        if self.dialect == Dialect.POSTGRES:
            # DROP FUNCTION ... CASCADE removes triggers still pointing at our
            # functions (including on DETACHed partitions), but a trigger
            # repointed out-of-band to another function would otherwise survive
            # and collide when repair_view() reinstalls it.
            seen_drop_pairs: Set[Tuple[str, str]] = set()
            for trg, table, *_ in self._pg_parse_expected_triggers(entry.get("trigger_ddls") or []):
                pair = (trg, table)
                if pair in seen_drop_pairs:
                    continue
                seen_drop_pairs.add(pair)
                lit_t = "'" + table.replace("'", "''") + "'"
                lit_trg = "'" + trg.replace("'", "''") + "'"
                cur.execute(
                    "DO $mv_drop$ DECLARE __mv_rel pg_catalog.regclass; BEGIN "
                    "FOR __mv_rel IN "
                    "  WITH RECURSIVE __mv_tree(relid, depth) AS ("
                    f"    SELECT pg_catalog.to_regclass({lit_t})::oid, 0"
                    "    UNION ALL SELECT i.inhrelid, __mv_tree.depth + 1 FROM pg_catalog.pg_inherits i"
                    "    JOIN __mv_tree ON i.inhparent = __mv_tree.relid"
                    "  ) SELECT relid::pg_catalog.regclass FROM __mv_tree"
                    "    WHERE relid IS NOT NULL ORDER BY depth, relid "
                    "LOOP "
                    f"  EXECUTE pg_catalog.format('DROP TRIGGER IF EXISTS %I ON %s', {lit_trg}, __mv_rel); "
                    "END LOOP; END $mv_drop$;"
                )
        cur.execute(build_drop_sql("VIEW", resolved_name, dialect=self.dialect, allow_qualified=False))
        cur.execute(build_drop_sql("TABLE", entry["backing_table"], dialect=self.dialect, allow_qualified=False))
        for aux_t in entry["aux_tables"]:
            cur.execute(build_drop_sql("TABLE", aux_t, dialect=self.dialect, allow_qualified=False))
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
                if self.dialect == Dialect.POSTGRES:
                    # Acquire catalog ROW EXCLUSIVE -> base_t SHARE BEFORE the
                    # first query freezes the REPEATABLE READ snapshot: preserves
                    # global catalog -> base_t lock order with register/drop/repair,
                    # waits for in-flight writers, and blocks new writers until
                    # this snapshot backfill commits (backfill_active = 0).
                    cur.execute(f"LOCK TABLE {self.catalog_table} IN ROW EXCLUSIVE MODE;")
                    pg_lock_targets = _postgres_base_table_lock_targets(
                        plan, object_prefix=self.object_prefix
                    )
                    if pg_lock_targets:
                        cur.execute(
                            f"LOCK TABLE {', '.join(pg_lock_targets)} IN SHARE MODE;"
                        )

            # Fail closed before folding anything: a claimed or completed
            # watermark means another backfill is running, an earlier one was
            # interrupted, or the view is already populated; folding again
            # would double-count.
            self._refuse_unless_backfill_unclaimed(
                self._catalog_entry(cur, plan.view_name), plan.view_name
            )
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                # Upgrade the savepoint to a RESERVED write lock before reading
                # base tables so no concurrent writer can interleave.
                self._execute_param(
                    cur,
                    f"UPDATE {self.catalog_table} SET backfill_active = backfill_active "
                    f"WHERE view_name = ? AND backfill_cursor_pk IS NULL",
                    (plan.view_name,),
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
        return entry.get("backfill_active") == 1

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
        try:
            self._execute_param(
                cur,
                f"UPDATE {self.catalog_table} "
                f"SET backfill_cursor_pk = ?, backfill_max_pk = ?, backfill_active = ? "
                f"WHERE view_name = ? AND backfill_cursor_pk = ? AND backfill_max_pk = ? "
                f"AND backfill_active = 1 AND backfill_owner = ?",
                (upper_pk, max_pk, active, view_name, cursor_pk, max_pk, token),
            )
        except Exception as exc:
            if self._is_snapshot_row_changed_error(exc):
                self._rollback()
                raise BackfillInterruptedError(
                    view_name,
                    "the backfill watermark was taken over (repair_view, rebuild or drop) while this "
                    "backfill was running; its in-flight batch was rolled back and it stopped",
                    cursor_pk,
                    max_pk,
                ) from exc
            raise
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
        # Unclaimed views stay fail-closed at (NULL, NULL, backfill_active=1) so
        # triggers remain gated and check_schema_drift() continues to report
        # backfill_incomplete until a backfill claims and completes them (or
        # repair_view() / force_rebuild takes over). Claimed views likewise stay
        # fail-closed at their last committed watermark.
        _ = view_name
        return None

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

    class _PhantomChunkInsert(Exception):
        """Internal retry sentinel when a PostgreSQL concurrent INSERT lands in
        a PK gap inside (cursor_pk, upper_pk] between the chunk's FOR UPDATE
        and its watermark advance."""

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
        pg_lock_table: bool = False,
    ) -> None:
        """Fold one (cursor_pk, upper_pk] range and advance the watermark."""
        before_pks: Optional[Set[Any]] = None
        if not snapshot_isolation:
            if self.dialect == Dialect.POSTGRES:
                if pg_lock_table:
                    cur.execute(f"LOCK TABLE {self.catalog_table} IN ROW EXCLUSIVE MODE;")
                    pg_lock_targets = _postgres_base_table_lock_targets(
                        plan, object_prefix=self.object_prefix
                    )
                    if pg_lock_targets:
                        cur.execute(
                            f"LOCK TABLE {', '.join(pg_lock_targets)} IN SHARE MODE;"
                        )
                else:
                    try:
                        self._execute_param(
                            cur,
                            f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ? FOR UPDATE",
                            (cursor_pk, upper_pk),
                        )
                        before_pks = {r[0] for r in (cur.fetchall() or [])}
                    except Exception as exc:
                        # A deadlock or lock timeout here has already rolled back
                        # or invalidated the range lock; propagate for batch retry.
                        if self._is_transient_lock_error(exc) or self._is_lock_timeout_error(exc):
                            raise
            elif self.dialect == Dialect.MYSQL:
                cur.execute("START TRANSACTION;")
                try:
                    self._execute_param(
                        cur,
                        f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ? FOR UPDATE",
                        (cursor_pk, upper_pk),
                    )
                    cur.fetchall()
                except Exception as exc:
                    if self._is_transient_lock_error(exc):
                        raise
            elif self.dialect == Dialect.SQLITE:
                self._reserve_sqlite_writer(cur)
        for tmpl in spec.chunk_sql_templates:
            cur.execute(tmpl, (cursor_pk, upper_pk))
        still_active = 1 if upper_pk < max_pk else 0
        self._advance_backfill_watermark(cur, plan.view_name, cursor_pk, upper_pk, max_pk, still_active)
        if not snapshot_isolation and self.dialect == Dialect.POSTGRES and before_pks is not None:
            # Because PostgreSQL SELECT ... FOR UPDATE has no gap locks, a
            # concurrent INSERT whose PK falls in a gap inside (cursor_pk, upper_pk]
            # may have skipped trigger maintenance (taking FOR SHARE on the catalog
            # row) and committed after our fold scanned the range. Now that we hold
            # the catalog row lock (waiting for all in-flight skippers and blocking
            # new ones until COMMIT), verify no phantom PKs appeared in the chunk.
            self._execute_param(
                cur,
                f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ?",
                (cursor_pk, upper_pk),
            )
            after_pks = {r[0] for r in (cur.fetchall() or [])}
            if after_pks != before_pks:
                self._rollback()
                raise self._PhantomChunkInsert()
        if not snapshot_isolation:
            self._commit()

    def _claim_non_snapshot_pk_range(
        self, cur: Any, plan: MaterializedViewPlan, base_t: str, pk_col: str
    ) -> Optional[Tuple[int, int]]:
        """Atomically observe ``[MIN(pk), MAX(pk)]`` and claim the pk-range
        watermark after every in-flight writer that saw ``backfill_cursor_pk IS NULL``
        has committed.

        Returns ``(min_pk, max_pk)`` when the table is non-empty (leaving the
        catalog committed at ``(min_pk - 1, max_pk, 1, token)``), or ``None``
        when the table is empty (leaving the catalog committed as completed
        ``(_COMPLETED_MARKER_PK, _COMPLETED_MARKER_PK, 0)``).
        """
        view_name = plan.view_name
        pg_lock_targets = (
            _postgres_base_table_lock_targets(plan, object_prefix=self.object_prefix)
            if self.dialect == Dialect.POSTGRES
            else []
        )
        attempt = 0
        while True:
            try:
                if self.dialect == Dialect.SQLITE:
                    self._reserve_sqlite_writer(cur)
                elif self.dialect == Dialect.POSTGRES:
                    # Global lock order (catalog -> base_t): catalog ROW EXCLUSIVE
                    # conflicts with register/drop/repair's SHARE ROW EXCLUSIVE
                    # (preventing AB-BA deadlocks) while staying compatible with
                    # writers' ACCESS SHARE reads; base_t SHARE waits for any
                    # in-flight writer whose trigger saw cursor IS NULL.
                    cur.execute(f"LOCK TABLE {self.catalog_table} IN ROW EXCLUSIVE MODE;")
                    if pg_lock_targets:
                        cur.execute(f"LOCK TABLE {', '.join(pg_lock_targets)} IN SHARE MODE;")
                elif self.dialect == Dialect.MYSQL:
                    # Start a fresh READ COMMITTED transaction so taking the catalog
                    # row X-lock waits for in-flight writers (whose triggers hold an
                    # S-lock on the catalog row while reading cursor IS NULL) and the
                    # subsequent non-locking MIN/MAX sees all of their committed rows
                    # without taking any locks on base_t.
                    self._rollback()
                    cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED;")
                    cur.execute("START TRANSACTION;")
                    self._execute_param(
                        cur,
                        f"SELECT view_name FROM {self.catalog_table} WHERE view_name = ? FOR UPDATE",
                        (view_name,),
                    )
                    if cur.fetchone() is None:
                        self._rollback()
                        raise BackfillInterruptedError(
                            view_name, "the view is no longer present in the catalog"
                        )

                cur.execute(f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}")
                row = cur.fetchone()
                if not row or row[0] is None or row[1] is None:
                    self._claim_backfill_watermark(
                        cur,
                        view_name,
                        self._COMPLETED_MARKER_PK,
                        self._COMPLETED_MARKER_PK,
                        active=0,
                        retry=False,
                    )
                    return None
                min_pk, max_pk = int(row[0]), int(row[1])
                cursor_pk = min_pk - 1
                self._claim_backfill_watermark(
                    cur,
                    view_name,
                    cursor_pk,
                    max_pk,
                    active=1,
                    retry=False,
                )
                return min_pk, max_pk
            except BackfillInterruptedError:
                raise
            except Exception as exc:
                if not (
                    self._is_transient_lock_error(exc)
                    or self._is_lock_timeout_error(exc)
                    or self._is_snapshot_row_changed_error(exc)
                ):
                    raise
                self._rollback()
                if attempt >= self._BACKFILL_TRANSIENT_RETRIES:
                    raise BackfillInterruptedError(
                        view_name,
                        f"could not atomically claim the pk-range backfill watermark after "
                        f"{attempt + 1} attempts: {exc!r}",
                    ) from exc
                attempt += 1
                self._backfill_sleep(self._backfill_retry_delay(attempt))

    def _begin_single_shot_backfill_tx(self, cur: Any, plan: MaterializedViewPlan) -> None:
        """Start an atomic transaction for single-shot non-snapshot backfill."""
        if self.dialect == Dialect.SQLITE:
            self._reserve_sqlite_writer(cur)
        elif self.dialect == Dialect.POSTGRES:
            cur.execute(f"LOCK TABLE {self.catalog_table} IN ROW EXCLUSIVE MODE;")
            pg_lock_targets = _postgres_base_table_lock_targets(
                plan, object_prefix=self.object_prefix
            )
            if pg_lock_targets:
                cur.execute(f"LOCK TABLE {', '.join(pg_lock_targets)} IN SHARE MODE;")
        elif self.dialect == Dialect.MYSQL:
            cur.execute("START TRANSACTION;")

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
            if not snapshot_isolation:
                self._begin_single_shot_backfill_tx(cur, plan)
            self._claim_backfill_watermark(
                cur,
                plan.view_name,
                self._COMPLETED_MARKER_PK,
                self._COMPLETED_MARKER_PK,
                active=0,
                retry=not snapshot_isolation,
            )
            for s in plan.initial_populate_sqls:
                cur.execute(s)
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            if on_progress:
                on_progress(1, 1)
            return 1

        batches_run = 0
        if spec.mode == "pk_range_fold" and spec.pk_column:
            base_t = quote_sql_identifier(plan.view_spec.base_table, self.dialect, allow_qualified=False)
            pk_col = quote_sql_identifier(spec.pk_column, self.dialect, allow_qualified=False)
            if not snapshot_isolation:
                bounds = self._claim_non_snapshot_pk_range(cur, plan, base_t, pk_col)
                if bounds is None:
                    if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                        for s in plan.get_populate_statements(incremental_backfill=False):
                            cur.execute(s)
                    self._commit()
                    return 0
                min_pk, max_pk = bounds
                cursor_pk = min_pk - 1
                total_span = max(1, max_pk - min_pk + 1)
                self._commit()
                self._backfill_watermark_claimed = True
                self._backfill_committed_watermark = (cursor_pk, max_pk)
            else:
                cur.execute(f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}")
                row = cur.fetchone()
                if not row or row[0] is None or row[1] is None:
                    # Claim first (atomic vs. concurrent backfills), then record a
                    # *completed* marker: (NULL, NULL, 0) would let a later
                    # backfill fold rows that triggers have maintained since.
                    self._claim_backfill_watermark(
                        cur, plan.view_name, self._COMPLETED_MARKER_PK, self._COMPLETED_MARKER_PK, active=0,
                        retry=False,
                    )
                    if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                        for s in plan.get_populate_statements(incremental_backfill=False):
                            cur.execute(s)
                    if self.dialect == Dialect.SQLITE:
                        cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                    self._commit()
                    return 0
                min_pk, max_pk = int(row[0]), int(row[1])
                cursor_pk = min_pk - 1
                total_span = max(1, max_pk - min_pk + 1)
                self._claim_backfill_watermark(
                    cur, plan.view_name, cursor_pk, max_pk, retry=False
                )
            while cursor_pk < max_pk:
                upper_pk = min(max_pk, cursor_pk + batch_size)
                attempt = 0
                phantom_retries = 0
                while True:
                    try:
                        self._run_pk_range_batch(
                            cur, plan, spec, base_t, pk_col,
                            cursor_pk, upper_pk, max_pk, snapshot_isolation,
                            pg_lock_table=(phantom_retries >= 2),
                        )
                        break
                    except self._PhantomChunkInsert:
                        phantom_retries += 1
                        if phantom_retries > 4:
                            raise RuntimeError(
                                f"pk-range chunk ({cursor_pk}, {upper_pk}] for '{plan.view_name}' "
                                f"repeatedly observed concurrent phantom inserts"
                            )
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
            self._claim_backfill_watermark(
                cur,
                plan.view_name,
                self._COMPLETED_MARKER_PK,
                self._COMPLETED_MARKER_PK,
                active=0,
                retry=not snapshot_isolation,
            )
            cur.execute(spec.distinct_group_query)
            all_keys = [r[0] for r in (cur.fetchall() or [])]
            if not all_keys:
                if snapshot_isolation and self.dialect == Dialect.SQLITE:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                self._commit()
                return 0
            ph = self._param_placeholder()
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i : i + batch_size]
                placeholders = ", ".join(ph for _ in batch_keys)
                for tmpl in spec.chunk_sql_templates:
                    sql = tmpl.replace(GROUP_KEY_PLACEHOLDERS, placeholders)
                    cur.execute(sql, tuple(batch_keys))
                if not snapshot_isolation:
                    self._commit()
                batches_run += 1
                if on_progress:
                    on_progress(batches_run, min(len(all_keys), i + batch_size))
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            return batches_run

        elif spec.mode == "offset_batch" and spec.offset_insert_template:
            self._claim_backfill_watermark(
                cur,
                plan.view_name,
                self._COMPLETED_MARKER_PK,
                self._COMPLETED_MARKER_PK,
                active=0,
                retry=not snapshot_isolation,
            )
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
            if snapshot_isolation and self.dialect == Dialect.SQLITE:
                self._savepoint_release(cur, "__mv_backfill_snap")
            self._commit()
            return batches_run

        if not snapshot_isolation:
            self._begin_single_shot_backfill_tx(cur, plan)
        self._claim_backfill_watermark(
            cur,
            plan.view_name,
            self._COMPLETED_MARKER_PK,
            self._COMPLETED_MARKER_PK,
            active=0,
            retry=not snapshot_isolation,
        )
        for s in plan.initial_populate_sqls:
            cur.execute(s)
        if snapshot_isolation and self.dialect == Dialect.SQLITE:
            cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
        self._commit()
        if on_progress:
            on_progress(1, 1)
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
            v_spec.original_select_sql or v_spec.raw_select_sql
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
        plain_views: Optional[Mapping[str, str]] = None,
        current_schema: Optional[str] = None,
    ) -> MaterializedViewPlan:
        view = self._classify_native_functions(schemas, view, plain_views=plain_views, current_schema=current_schema)
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
            repair_fence_catalog_table=self.catalog_table if self.dialect == Dialect.POSTGRES else None,
            plain_views=plain_views,
            current_schema=current_schema,
        )
        if self.dialect == Dialect.POSTGRES:
            # Every path that installs, replaces, repairs or re-validates a view
            # compiles it here: a subscription target is refused at registration,
            # refused by repair_view() and reported "invalid" by check_schema_drift().
            # The view's own backing and auxiliary tables must not be subscribed
            # either (the subscription would also write derived rows).
            self._refuse_subscription_targets(
                plan.view_name,
                [*plan.view_spec.referenced_tables, plan.backing_table_name, *plan.auxiliary_table_names],
            )
        return plan

    def _classify_native_functions(
        self,
        schemas: Dict[str, TableSchema],
        view: Any,
        *,
        plain_views: Optional[Mapping[str, str]] = None,
        current_schema: Optional[str] = None,
    ) -> Any:
        """Ask the engine catalog whether unmodelled function calls are
        aggregates (see matview.native_functions); refuse when it cannot tell.
        Calls on an engine without a readable catalog are recorded on the
        spec (``unclassified_functions``) and force the exact full refresh."""
        spec = view
        if isinstance(view, str):
            # Parsed exactly as generate_materialized_view would (plain views
            # inlined); a spec that references no plain view is kept as is
            # there, so the classification survives.
            spec = parse_view_definition(
                view,
                known_tables={k.lower() for k in schemas},
                dialect=self.dialect,
                plain_views=plain_views,
                current_schema=current_schema,
                schemas=schemas,
            )
        if not isinstance(spec, ViewSpec):
            return view
        if self.dialect != Dialect.MYSQL:
            overridden = overridden_modelled_functions(self.conn, spec, self.dialect)
            if overridden:
                spec = replace(spec, unclassified_functions=tuple(overridden), functions_classified=True)
        if (
            not spec.additional_views
            and not unclassified_calls(spec, self.dialect)
            and (self.dialect == Dialect.SQLITE or not items_with_unmodelled_calls(spec, self.dialect))
        ):
            return self._with_server_result_types(spec) if spec.unclassified_functions else view
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
        if self.dialect == Dialect.SQLITE or not (items_with_unmodelled_calls(spec, self.dialect) or spec.unclassified_functions):
            return spec
        types = probe_result_types(self.conn, self.dialect, spec.raw_select_sql)
        if not types:
            return spec
        return replace(spec, native_result_types=tuple(sorted(types.items())))

    def _stored_plan(
        self,
        schemas: Dict[str, TableSchema],
        entry: Dict[str, Any],
        *,
        plain_views: Optional[Mapping[str, str]] = None,
        current_schema: Optional[str] = None,
        exclude_views: Sequence[str] = (),
    ) -> MaterializedViewPlan:
        """Recompile an installed view under exactly the limitation codes it
        was installed with (plus the manager's accept_limitations); never a
        blanket override."""
        if plain_views is None:
            excl = list(exclude_views) if exclude_views else [entry["view_name"]]
            plain_views, current_schema = self._fetch_plain_views(exclude_views=excl)
        return self._compile_view(
            schemas,
            entry["view_sql"],
            accept=self._stored_accept(entry),
            ignore=self.ignore_limitations,
            plain_views=plain_views,
            current_schema=current_schema,
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
                # Re-registering must not silently adopt an interrupted,
                # concurrently running, or unpopulated backfill as complete.
                self._refuse_unless_backfill_unclaimed(existing, plan.view_name)
                if populate:
                    raise BackfillInterruptedError(
                        plan.view_name,
                        "its backfill has not started or completed yet (registered unpopulated or "
                        "stopped before claiming); run backfill_incrementally(), repair_view(), or "
                        "register_view(..., force_rebuild=True)",
                        None,
                        None,
                    )
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

                stored_view_sql = self._serialize_view_sql_with_indexes(plan)
                pending_backfill = (not populate) or incremental_backfill
                init_cursor = None if pending_backfill else self._COMPLETED_MARKER_PK
                init_max = None if pending_backfill else self._COMPLETED_MARKER_PK
                init_active = 1 if pending_backfill else 0
                catalog_insert_sql = f"""
                    INSERT INTO {self.catalog_table} (
                        view_name, view_sql, strategy, signature_hash, backing_table,
                        aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json,
                        backfill_cursor_pk, backfill_max_pk, backfill_active,
                        accepted_limitations_json, partitions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                accepted_lim_json = json.dumps({
                    "definition": _definition_digest(
                        plan.view_spec.original_select_sql or plan.view_spec.raw_select_sql
                    ),
                    "codes": sorted({lim.code for lim in plan.limitations}),
                })

                def _make_catalog_params(parts_json: Optional[str]) -> Tuple[Any, ...]:
                    return (
                        plan.view_name,
                        stored_view_sql,
                        plan.strategy.value,
                        sig,
                        plan.backing_table_name,
                        json.dumps(plan.auxiliary_table_names),
                        json.dumps(plan.trigger_names),
                        json.dumps(plan.function_names),
                        json.dumps(plan.trigger_ddls),
                        init_cursor,
                        init_max,
                        init_active,
                        accepted_lim_json,
                        parts_json,
                    )

                # On MySQL/MariaDB, CREATE TRIGGER commits implicitly. Insert and
                # commit the pending-backfill catalog row (NULL, NULL, active=1)
                # BEFORE installing triggers so triggers gate every write from the
                # very first microsecond they exist (no ungated double-count window).
                if self.dialect == Dialect.MYSQL and pending_backfill:
                    self._execute_param(cur, catalog_insert_sql, _make_catalog_params(None))
                    catalog_inserted = True
                    self._commit()

                for stmt in plan.trigger_ddls:
                    obj = _extract_created_object(stmt)
                    cur.execute(self._mysql_maintenance_trigger_first(cur, stmt, plan.trigger_names))
                    if obj:
                        created_objects.append(obj)

                if not catalog_inserted:
                    # On PostgreSQL, snapshot the source tables' partition trees
                    # AFTER trigger installation (which holds SHARE ROW EXCLUSIVE
                    # on the base tables and partitions) so the recorded partition
                    # set matches both installed triggers and initial population.
                    parts_snapshot = (
                        self._pg_snapshot_partitions(
                            cur,
                            _postgres_base_table_lock_targets(
                                plan, object_prefix=self.object_prefix
                            ),
                        )
                        if self.dialect == Dialect.POSTGRES
                        else None
                    )
                    parts_json = json.dumps(parts_snapshot) if parts_snapshot is not None else None
                    self._execute_param(cur, catalog_insert_sql, _make_catalog_params(parts_json))
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
                    # Verify no concurrent session claimed, repaired, or replaced
                    # our committed catalog row before cleaning up objects.
                    still_owned_by_us = True
                    if catalog_inserted:
                        try:
                            self._execute_param(
                                cur,
                                f"SELECT backfill_cursor_pk, signature_hash "
                                f"FROM {self.catalog_table} WHERE view_name = ?",
                                (plan.view_name,),
                            )
                            cat_row = cur.fetchone()
                            if cat_row is not None and (cat_row[0] is not None or cat_row[1] != sig):
                                still_owned_by_us = False
                        except Exception:
                            still_owned_by_us = False
                    if still_owned_by_us:
                        if catalog_inserted:
                            try:
                                self._execute_param(
                                    cur,
                                    f"DELETE FROM {self.catalog_table} "
                                    f"WHERE view_name = ? AND backfill_cursor_pk IS NULL "
                                    f"AND signature_hash = ?",
                                    (plan.view_name, sig),
                                )
                                self._commit()
                            except Exception:
                                pass
                        for kind, name in reversed(created_objects):
                            try:
                                if kind == "trigger":
                                    cur.execute(build_drop_sql("TRIGGER", name, dialect=self.dialect, allow_qualified=False))
                                elif kind == "view":
                                    cur.execute(build_drop_sql("VIEW", name, dialect=self.dialect, allow_qualified=False))
                                elif kind == "table":
                                    cur.execute(build_drop_sql("TABLE", name, dialect=self.dialect, allow_qualified=False))
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
        catalog_for_pv = self.get_catalog()
        plain_views, cur_schema = self._fetch_plain_views(
            extra_schema_sql=schema_sql,
            exclude_views=list(catalog_for_pv.keys()),
        )
        specs = parse_view_definitions(
            view_sql,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
            dialect=self.dialect,
            plain_views=plain_views,
            current_schema=cur_schema,
            schemas=schemas,
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
        "backfill_incomplete" | "triggers_missing" | "triggers_disabled" |
        "triggers_modified" | "triggers_out_of_order" | "triggers_orphaned" |
        "partitions_changed" | "dropped"}`
        ("triggers_missing": maintenance triggers or helper functions of an unchanged view are gone;
        "triggers_disabled": a PostgreSQL maintenance trigger is not ENABLE ALWAYS (disabled, ENABLE REPLICA or plain ENABLE);
        "triggers_modified": a PostgreSQL maintenance/helper function body, `search_path`, or trigger
        definition was altered out of band; "triggers_out_of_order": a newer SQLite user trigger
        that writes a source table fires before the maintenance trigger, or a MySQL/MariaDB
        `PRECEDES` user trigger writes another join source first; "triggers_orphaned": a PostgreSQL
        maintenance trigger is on a DETACHed partition; "partitions_changed": a PostgreSQL source
        table's partition set changed out of band, e.g. `DROP TABLE` on a partition; repair any of
        these with `repair_view()`);
        "backfill_incomplete" means the definition
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
        plain_views, cur_schema = self._fetch_plain_views(
            extra_schema_sql=schema_sql,
            exclude_views=list(catalog.keys()),
        )
        status: Dict[str, str] = {}
        desired_names: Set[str] = set()

        all_specs = parse_view_definitions(
            view_sqls,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
            dialect=self.dialect,
            plain_views=plain_views,
            current_schema=cur_schema,
            schemas=schemas,
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
                plain_views=plain_views,
                current_schema=cur_schema,
            )
            planned.append((v_spec, plan, self._compute_signature(plan, schemas)))

        extra_ddls_by_view = {
            plan.view_name: plan.readonly_trigger_ddls
            for _, plan, _ in planned
            if plan.readonly_trigger_ddls
        }
        trigger_drift = self._views_trigger_drift(catalog, extra_ddls_by_view=extra_ddls_by_view)

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

    def _views_missing_triggers(
        self,
        catalog: Dict[str, Dict[str, Any]],
        extra_ddls_by_view: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Set[str]:
        """Registered views whose maintenance triggers or functions are not
        exactly where and how they must be (see ``_views_trigger_drift``)."""
        return set(self._views_trigger_drift(catalog, extra_ddls_by_view=extra_ddls_by_view))

    def _pg_subscription_targets(self, tables: Iterable[str]) -> List[str]:
        """The given source tables that are logical-replication subscription
        targets on this PostgreSQL server (the table itself, any partition below
        it, or any partitioned ancestor appears in ``pg_subscription_rel``).

        The subscription apply worker fires only ROW-level triggers; matview's
        PostgreSQL maintenance triggers are statement-level, so replicated
        INSERT/UPDATE/DELETE would silently not be maintained. Returns the
        offending tables; raises UnsupportedMaterializedViewError when the
        catalog cannot be read (unknown is unsupported).
        """
        if self.dialect != Dialect.POSTGRES:
            return []
        found: List[str] = []
        errors: List[BaseException] = []
        with self._cursor() as cur:
            for table in sorted({str(t) for t in tables}, key=str.lower):
                lit = "'" + quote_sql_identifier(table, self.dialect, allow_qualified=True).replace("'", "''") + "'"
                rows = self._catalog_select(
                    cur,
                    f"WITH RECURSIVE __mv_down(relid) AS ("
                    f" SELECT pg_catalog.to_regclass({lit})::oid"
                    f" UNION ALL SELECT i.inhrelid FROM pg_catalog.pg_inherits i JOIN __mv_down ON i.inhparent = __mv_down.relid),"
                    f" __mv_up(relid) AS ("
                    f" SELECT pg_catalog.to_regclass({lit})::oid"
                    f" UNION ALL SELECT i.inhparent FROM pg_catalog.pg_inherits i JOIN __mv_up ON i.inhrelid = __mv_up.relid)"
                    f" SELECT count(*) FROM pg_catalog.pg_subscription_rel sr"
                    f" WHERE sr.srrelid IN (SELECT relid FROM __mv_down UNION SELECT relid FROM __mv_up)",
                    errors,
                )
                if rows is None:
                    # Unknown is unsupported: replicated changes to a subscription
                    # target would silently not be maintained, so a failed probe
                    # refuses instead of letting the view through.
                    raise UnsupportedMaterializedViewError(
                        f"cannot determine whether table '{table}' is a logical-replication subscription target "
                        f"(reading pg_subscription_rel failed: {errors[-1] if errors else 'no result'}); "
                        "views over subscription targets are unsupported, so the view is refused"
                    )
                if int(rows[0][0]) > 0:
                    found.append(table)
        return found

    def _refuse_subscription_targets(self, view_name: str, tables: Iterable[str]) -> None:
        targets = self._pg_subscription_targets(tables)
        if targets:
            raise UnsupportedMaterializedViewError(
                f"View '{view_name}' reads table(s) {', '.join(sorted(targets))} that are logical-replication "
                "subscription targets. The subscription apply worker fires only row-level triggers, while "
                "PostgreSQL maintenance triggers are statement-level, so replicated INSERT/UPDATE/DELETE would "
                "not be maintained and the view would silently go stale. Logical-replication subscribers are an "
                "unsupported configuration (no relaxed mode exists): maintain the view on the publisher, or "
                "materialize from a table that is not a subscription target."
            )

    _PG_IDENT_PAT = r'(?:"(?:""|[^"])+"|[A-Za-z_][A-Za-z0-9_$]*)'
    _PG_CREATE_TRIGGER_RE = re.compile(
        r'^\s*CREATE\s+TRIGGER\s+(' + _PG_IDENT_PAT + r')\s+'
        r'(AFTER|BEFORE|INSTEAD\s+OF)\s+'
        r'((?:INSERT|UPDATE|DELETE|TRUNCATE)(?:\s+OR\s+(?:INSERT|UPDATE|DELETE|TRUNCATE))*)\s+'
        r'ON\s+(' + _PG_IDENT_PAT + r'(?:\.' + _PG_IDENT_PAT + r')?)'
        r'(?:\s+REFERENCING\s+(.*?))?'
        r'\s+FOR\s+EACH\s+(ROW|STATEMENT)\s+EXECUTE\s+(?:FUNCTION|PROCEDURE)\s+'
        r'(' + _PG_IDENT_PAT + r')\s*\(\s*\)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    _PG_CREATE_FUNCTION_HEADER_RE = re.compile(
        r'^\s*CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(' + _PG_IDENT_PAT + r')\s*\(([^)]*)\)',
        flags=re.IGNORECASE,
    )
    _PG_DOLLAR_BODY_RE = re.compile(
        r'\bAS\s+(\$[A-Za-z0-9_]*\$)(.*?)\1',
        flags=re.IGNORECASE | re.DOTALL,
    )
    _PG_RETURNS_RE = re.compile(
        r'\bRETURNS\s+([A-Za-z_][A-Za-z0-9_.\s\[\]]*?)(?=\s+(?:AS\b|LANGUAGE\b|IMMUTABLE\b|STABLE\b|VOLATILE\b|STRICT\b|CALLED\b|RETURNS\b|PARALLEL\b|SECURITY\b|SET\b|COST\b|ROWS\b|LEAKPROOF\b|NOT\b)|$)',
        flags=re.IGNORECASE,
    )
    _PG_TRIGGER_EVENT_BITS = {
        "INSERT": 4,
        "DELETE": 8,
        "UPDATE": 16,
        "TRUNCATE": 32,
    }

    @staticmethod
    def _pg_unquote_ident(ident: str) -> str:
        s = ident.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1].replace('""', '"')
        return s.lower()

    @classmethod
    def _pg_parse_expected_functions(
        cls, ddls: Sequence[str]
    ) -> List[Tuple[str, Optional[str], Optional[str], str, bool, bool, bool]]:
        """Extract ``(proname, arg_type_or_none, ret_type_or_none, prosrc,
        requires_search_path, is_strict, is_immutable)`` from PostgreSQL
        ``CREATE [OR REPLACE] FUNCTION`` DDLs."""
        out: List[Tuple[str, Optional[str], Optional[str], str, bool, bool, bool]] = []
        for stmt in ddls:
            m_hdr = cls._PG_CREATE_FUNCTION_HEADER_RE.match(stmt)
            if m_hdr is None:
                continue
            m_body = cls._PG_DOLLAR_BODY_RE.search(stmt, pos=m_hdr.end())
            if m_body is None:
                continue
            fn_name = cls._pg_unquote_ident(m_hdr.group(1))
            args_raw = m_hdr.group(2).strip()
            arg_type: Optional[str] = None
            if args_raw:
                # Strip optional parameter mode (IN/OUT/INOUT/VARIADIC) and
                # leading parameter name while preserving multi-word types.
                parts = args_raw.split()
                if parts and parts[0].upper() in ("IN", "OUT", "INOUT", "VARIADIC"):
                    parts = parts[1:]
                if len(parts) >= 2 and parts[0].lower() not in (
                    "double",
                    "character",
                    "time",
                    "timestamp",
                    "bit",
                    "unsigned",
                ):
                    parts = parts[1:]
                arg_type = " ".join(parts).lower() if parts else None
            prosrc = m_body.group(2).replace("\r\n", "\n")
            outer_sql = stmt[m_hdr.end() : m_body.start()] + " " + stmt[m_body.end() :]
            m_ret = cls._PG_RETURNS_RE.search(outer_sql)
            ret_type = re.sub(r"\s+", " ", m_ret.group(1).strip().lower()) if m_ret else None
            requires_search_path = bool(
                re.search(r"\bSET\s+search_path\b", outer_sql, flags=re.IGNORECASE)
            )
            is_strict = bool(
                re.search(r"\b(?:STRICT|RETURNS\s+NULL\s+ON\s+NULL\s+INPUT)\b", outer_sql, flags=re.IGNORECASE)
            )
            is_immutable = bool(
                re.search(r"\bIMMUTABLE\b", outer_sql, flags=re.IGNORECASE)
            )
            out.append((fn_name, arg_type, ret_type, prosrc, requires_search_path, is_strict, is_immutable))
        return out

    @classmethod
    def _pg_parse_expected_triggers(
        cls, ddls: Sequence[str]
    ) -> List[Tuple[str, str, str, int, str, str]]:
        """Extract ``(tgname, table_ref, fn_name, tgtype, old_table, new_table)``
        from PostgreSQL ``CREATE TRIGGER`` DDLs."""
        out: List[Tuple[str, str, str, int, str, str]] = []
        for stmt in ddls:
            m = cls._PG_CREATE_TRIGGER_RE.match(stmt)
            if m is None:
                continue
            trg_name = cls._pg_unquote_ident(m.group(1))
            timing = re.sub(r"\s+", " ", m.group(2).strip().upper())
            events = [ev.strip().upper() for ev in re.split(r"\s+OR\s+", m.group(3).strip(), flags=re.IGNORECASE)]
            table_ref = m.group(4).strip()
            ref_clause = (m.group(5) or "").strip()
            level = m.group(6).strip().upper()
            fn_name = cls._pg_unquote_ident(m.group(7))
            tgtype = 0
            if level == "ROW":
                tgtype |= 1
            if timing == "BEFORE":
                tgtype |= 2
            elif timing == "INSTEAD OF":
                tgtype |= 64
            for ev in events:
                tgtype |= cls._PG_TRIGGER_EVENT_BITS.get(ev, 0)
            old_tbl = ""
            new_tbl = ""
            if ref_clause:
                m_old = re.search(
                    r"\bOLD\s+TABLE\s+(?:AS\s+)?(" + cls._PG_IDENT_PAT + r")",
                    ref_clause,
                    flags=re.IGNORECASE,
                )
                if m_old:
                    old_tbl = cls._pg_unquote_ident(m_old.group(1))
                m_new = re.search(
                    r"\bNEW\s+TABLE\s+(?:AS\s+)?(" + cls._PG_IDENT_PAT + r")",
                    ref_clause,
                    flags=re.IGNORECASE,
                )
                if m_new:
                    new_tbl = cls._pg_unquote_ident(m_new.group(1))
            out.append((trg_name, table_ref, fn_name, tgtype, old_tbl, new_tbl))
        return out

    @staticmethod
    def _pg_search_path_setting_is_valid(
        sp_raw: Optional[str],
        schema_name: str,
        current_user: Optional[str] = None,
    ) -> bool:
        """Return True iff the ``search_path`` value extracted from
        ``pg_proc.proconfig`` is non-empty and resolves ``schema_name``
        (either directly or via ``"$user"`` when ``schema_name == current_user``)."""
        if sp_raw is None:
            return False
        sp_val = sp_raw.strip()
        if not sp_val:
            return False
        # Parse comma-separated schema entries respecting double-quoted identifiers.
        tokens: List[str] = []
        cur_tok: List[str] = []
        in_quote = False
        i = 0
        while i < len(sp_val):
            ch = sp_val[i]
            if ch == '"':
                if in_quote and i + 1 < len(sp_val) and sp_val[i + 1] == '"':
                    cur_tok.append('""')
                    i += 2
                    continue
                in_quote = not in_quote
                cur_tok.append(ch)
            elif ch == "," and not in_quote:
                tokens.append("".join(cur_tok).strip())
                cur_tok = []
            else:
                cur_tok.append(ch)
            i += 1
        tokens.append("".join(cur_tok).strip())
        parsed: Set[str] = set()
        for tok in tokens:
            if not tok:
                continue
            if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
                parsed.add(tok[1:-1].replace('""', '"'))
            else:
                parsed.add(tok.lower())
        return schema_name in parsed or (
            current_user is not None and schema_name == current_user and "$user" in parsed
        )

    def _views_trigger_drift(
        self,
        catalog: Dict[str, Dict[str, Any]],
        extra_ddls_by_view: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Dict[str, str]:
        """``{view: "triggers_missing" | "triggers_disabled" | "triggers_modified" | "triggers_out_of_order" | "triggers_orphaned" | "partitions_changed"}``
        for views whose maintenance triggers, functions, or PostgreSQL source
        partition sets drifted.

        "triggers_missing": a recorded trigger or helper function does not exist
        (PostgreSQL: on the source table or on one of its current partitions,
        e.g. one attached or created after registration). "triggers_disabled"
        (PostgreSQL): a source-table maintenance trigger is not ``ENABLE
        ALWAYS``: disabled, ``ENABLE REPLICA`` (not fired for ordinary
        sessions) or plain ``ENABLE`` (not fired under
        ``session_replication_role = replica``). "triggers_modified" (PostgreSQL): a maintenance or
        helper function's body, language, security mode, or ``search_path``
        configuration was modified out of band (e.g. ``CREATE OR REPLACE
        FUNCTION`` or ``ALTER FUNCTION ... RESET search_path``), or a trigger
        was altered to invoke a different function, event/timing, transition
        table, or ``WHEN`` predicate. "triggers_orphaned"
        (PostgreSQL): a maintenance trigger function also fires for a table
        that is no longer in the source's partition tree (a DETACHed
        partition, whose writes would still change the view).
        "partitions_changed" (PostgreSQL): a source table's live partition tree
        differs from the partition snapshot recorded in ``partitions_json``
        (e.g. ``DROP TABLE`` on a partition, ``DETACH`` followed by ``DROP``,
        partition rename, or bound change). Never ends a
        caller transaction (savepoint-protected reads); when the trigger list
        cannot be read nothing is reported.
        """
        if not catalog:
            return {}
        if self.dialect == Dialect.POSTGRES:
            # A user trigger writing another join source before the
            # statement-level maintenance is reconciled by the stage-3
            # multi-source bracket: nothing to report.
            return self._pg_trigger_drift(catalog, extra_ddls_by_view=extra_ddls_by_view)
        if self.dialect == Dialect.MYSQL:
            order_drift = self._join_source_order_drift(catalog)
            presence = self._views_trigger_presence_drift(catalog)
            for v, st in order_drift.items():
                presence.setdefault(v, st)
            return presence
        return self._views_trigger_presence_drift(catalog)

    def _views_trigger_presence_drift(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        sql = self._LIVE_TRIGGER_SQL.get(self.dialect)
        if sql is None:
            return {}
        with self._cursor() as cur:
            rows = self._catalog_select(cur, sql)
        if rows is None:
            return {}
        live = {str(r[0]).lower() for r in rows if r[0]}
        out = {
            name: "triggers_missing" for name, entry in catalog.items()
            if any(str(t).lower() not in live for t in entry.get("trigger_names") or [])
        }
        if self.dialect == Dialect.SQLITE:
            for name, status in self._sqlite_trigger_order_drift(catalog).items():
                out.setdefault(name, status)
        return out

    # Header of a SQLite trigger up to its body: timing (BEFORE when omitted)
    # and event.
    _SQLITE_TRIGGER_EVENT_RE = re.compile(
        r"\b(?:(BEFORE|AFTER|INSTEAD\s+OF)\s+)?(INSERT|UPDATE|DELETE)\b(?:\s+OF\b.*?)?\s+ON\s",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _SQLITE_TRIGGER_WRITE_RE = re.compile(
        r"\b(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?|DELETE\s+FROM)\s+"
        r'(?:(?:main|"main")\s*\.\s*)?("(?:""|[^"])+"|`(?:``|[^`])+`|\[[^\]]+\]|[^\s"`\]\(.;]+)',
        flags=re.IGNORECASE,
    )
    _SQLITE_RAISE_SKIP_RE = re.compile(r"\bRAISE\s*\(\s*(?:FAIL|IGNORE)\b", flags=re.IGNORECASE)

    def _sqlite_trigger_order_drift(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """``{view: "triggers_out_of_order"}``: a user ``AFTER`` trigger on a
        source table fires before the view's maintenance trigger for the same
        event and can break it.

        SQLite fires the triggers of one table and timing in descending
        ``sqlite_master`` rowid order (the most recently created first; the
        order survives reconnects and ``VACUUM``), so a user trigger created
        after registration runs first. It is reported when its body writes a
        table that has maintenance triggers (the view then sees changes out
        of order, e.g. an update of a row whose insert it has not seen yet,
        or the other join source before the delta it pairs with) or uses
        ``RAISE(FAIL)`` / ``RAISE(IGNORE)`` (the row is kept but the later
        maintenance trigger never runs), also through a chain of other
        triggers it sets off (a write to another table or to a view whose
        INSTEAD OF trigger writes a source; bounded depth). ``repair_view()``
        recreates the maintenance triggers, which makes them the newest.
        TEMP triggers (connection-local) and writes the SQL text does not
        show (application functions) are not seen.
        """
        with self._cursor() as cur:
            rows = self._catalog_select(cur, "SELECT rowid, name, tbl_name, sql FROM sqlite_master WHERE type = 'trigger'")
        if not rows:
            return {}
        maint_names = {
            str(t).lower() for entry in catalog.values() for t in entry.get("trigger_names") or []
        }

        def head_and_body(sql: str) -> Tuple[Optional[Tuple[str, str]], str]:
            m_begin = re.search(r"\bBEGIN\b", sql, flags=re.IGNORECASE)
            head = sql[: m_begin.start()] if m_begin else sql
            body = sql[m_begin.end():] if m_begin else ""
            m_ev = self._SQLITE_TRIGGER_EVENT_RE.search(head)
            if not m_ev:
                return None, body
            timing = re.sub(r"\s+", " ", (m_ev.group(1) or "BEFORE").upper())
            return (timing, m_ev.group(2).upper()), body

        maint: Dict[str, Tuple[int, str, str]] = {}  # name -> (rowid, table, event)
        user: List[Tuple[int, str, str, str]] = []  # (rowid, table, event, body)
        # Non-maintenance triggers (any timing, also INSTEAD OF on views) by
        # the table or view they are on: (written objects, uses RAISE(FAIL/IGNORE)).
        effects_on: Dict[str, List[Tuple[Set[str], bool]]] = {}
        for rowid, t_name, tbl, sql in rows:
            kind, body = head_and_body(str(sql or ""))
            if kind is None:
                continue
            timing, event = kind
            if str(t_name).lower() in maint_names:
                maint[str(t_name).lower()] = (int(rowid), str(tbl).lower(), event)
                continue
            writes = {_strip_outer_ident_quotes(m.group(1)).lower() for m in self._SQLITE_TRIGGER_WRITE_RE.finditer(body)}
            effects_on.setdefault(str(tbl).lower(), []).append(
                (writes, bool(self._SQLITE_RAISE_SKIP_RE.search(body))))
            if timing == "AFTER":
                user.append((int(rowid), str(tbl).lower(), event, body))
        source_tables = {tbl for _, tbl, _ in maint.values()}

        def reaches_source(body: str) -> bool:
            """The trigger body, or a trigger chain it starts (a write to
            another table or to a view whose INSTEAD OF trigger writes on),
            writes a source table or raises FAIL/IGNORE. Bounded depth."""
            if self._SQLITE_RAISE_SKIP_RE.search(body):
                return True
            frontier = {_strip_outer_ident_quotes(m.group(1)).lower() for m in self._SQLITE_TRIGGER_WRITE_RE.finditer(body)}
            seen: Set[str] = set()
            for _ in range(16):
                if frontier & source_tables:
                    return True
                seen |= frontier
                nxt: Set[str] = set()
                for obj in frontier:
                    for writes, raises in effects_on.get(obj, ()):
                        if raises:
                            return True
                        nxt |= writes
                frontier = nxt - seen
                if not frontier:
                    return False
            return True  # deeper than the bound: report rather than miss

        risky = [(rowid, tbl, event) for rowid, tbl, event, body in user if reaches_source(body)]
        out: Dict[str, str] = {}
        for name, entry in catalog.items():
            for t in entry.get("trigger_names") or []:
                info = maint.get(str(t).lower())
                if info is None:
                    continue
                m_rowid, m_tbl, m_event = info
                if any(r > m_rowid and tbl == m_tbl and ev == m_event for r, tbl, ev in risky):
                    out[name] = "triggers_out_of_order"
                    break
        return out

    _MYSQL_IDENT_PAT = r"(?:`(?:``|[^`])+`|[A-Za-z0-9_$]+)"
    _MYSQL_AFTER_TRIGGER_RE = re.compile(
        r"^\s*CREATE\s+TRIGGER\s+(" + _MYSQL_IDENT_PAT + r")\s+AFTER\s+(INSERT|UPDATE|DELETE)\s+ON\s+(" + _MYSQL_IDENT_PAT + r")\s+FOR\s+EACH\s+ROW\b",
        flags=re.IGNORECASE,
    )

    def _mysql_maintenance_trigger_first(self, cur: Any, stmt: str, own_names: Sequence[str]) -> str:
        """MySQL/MariaDB fire the triggers of one table, event and timing in
        ``ACTION_ORDER`` (creation order unless ``FOLLOWS``/``PRECEDES``).
        A maintenance ``AFTER`` trigger must run before any existing user
        trigger: one that writes another source of a join view first makes
        the view count that change twice. Place it before the first existing
        trigger with ``PRECEDES`` (user triggers created later run after it
        unless they ask to precede it, which the drift check reports)."""
        if self.dialect != Dialect.MYSQL:
            return stmt
        m = self._MYSQL_AFTER_TRIGGER_RE.match(stmt)
        if not m:
            return stmt
        table = _strip_outer_ident_quotes(m.group(3))
        own = {str(n).lower() for n in own_names}
        self._execute_param(
            cur,
            "SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = DATABASE() "
            "AND event_object_table = ? AND event_manipulation = ? AND action_timing = 'AFTER' "
            "ORDER BY action_order",
            (table, m.group(2).upper()),
        )
        first = next((str(r[0]) for r in cur.fetchall() or [] if str(r[0]).lower() not in own), None)
        if first is None:
            return stmt
        q = "`" + first.replace("`", "``") + "`"
        return stmt[: m.end()] + f" PRECEDES {q}" + stmt[m.end():]

    # A write to a table in a procedural body (PL/pgSQL, MySQL trigger body).
    _BODY_WRITE_RE = re.compile(
        r"\b(?:INSERT(?:\s+(?:IGNORE|OR\s+\w+))?\s+INTO|REPLACE\s+INTO|MERGE\s+INTO|"
        r"UPDATE(?:\s+(?:IGNORE|OR\s+\w+))?|DELETE\s+FROM)\s+"
        r'(?:(?:"(?:""|[^"])+"|`(?:``|[^`])+`|\w+)\s*\.\s*)?("(?:""|[^"])+"|`(?:``|[^`])+`|\w+)',
        flags=re.IGNORECASE,
    )

    def _join_source_order_drift(self, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """``{view: "triggers_out_of_order"}`` (MySQL/MariaDB): a user trigger
        on a source of a multi-source view fires before the view's
        maintenance trigger for the same event (``ACTION_ORDER``; registration
        places maintenance triggers first, so only a later ``PRECEDES`` does
        this) and writes another source of that view directly in its body.
        That source's maintenance then joins its change with the
        not-yet-maintained row, and the first source's maintenance joins the
        row with the change again: the pair is counted twice. Re-register the
        view to fix (``repair_view()`` only repopulates). Writes in functions
        or procedures the trigger calls are not seen.
        """
        if self.dialect != Dialect.MYSQL:
            return {}
        with self._cursor() as cur:
            rows = self._catalog_select(
                cur,
                "SELECT trigger_name, event_object_table, event_manipulation, action_timing, action_order, "
                "action_statement FROM information_schema.triggers WHERE trigger_schema = DATABASE()",
            )
        if not rows:
            return {}
        ev_bit = {"INSERT": 4, "DELETE": 8, "UPDATE": 16}
        # (name, table, event bit, ACTION_ORDER, body, is AFTER)
        trigs: List[Tuple[str, str, int, int, str, bool]] = [
            (str(name), str(tbl).lower(), ev_bit.get(str(event).upper(), 0), int(order or 0), str(body or ""),
             str(timing).upper() == "AFTER")
            for name, tbl, event, timing, order, body in rows
        ]
        maint_names = {str(t).lower() for e in catalog.values() for t in e.get("trigger_names") or []}
        out: Dict[str, str] = {}
        for vname, entry in catalog.items():
            own = {str(t).lower() for t in entry.get("trigger_names") or []}
            mine = [tr for tr in trigs if tr[0].lower() in own and tr[5]]
            sources = {m[1] for m in mine}
            if len(sources) < 2:
                continue
            for n, tname, evs, order, body, after in trigs:
                if n.lower() in maint_names or not after:
                    continue
                writes = {_strip_outer_ident_quotes(m.group(1)).lower() for m in self._BODY_WRITE_RE.finditer(body)}
                if not (writes & (sources - {tname})):
                    continue
                if any(m_t == tname and (m_ev & evs) and order < m_order for _, m_t, m_ev, m_order, _, _ in mine):
                    out[vname] = "triggers_out_of_order"
                    break
        return out

    def _pg_resolve_catalog_schema_meta(
        self, cur: Any
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Return ``(schema_oid, schema_name, current_user)`` for the catalog's schema."""
        lit_cat = "'" + self.catalog_table.replace("'", "''") + "'"
        meta_rows = self._catalog_select(
            cur,
            "SELECT c.relnamespace, n.nspname, current_user"
            " FROM pg_catalog.pg_class c"
            " JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace"
            f" WHERE c.oid = pg_catalog.to_regclass({lit_cat})::oid",
        )
        if not meta_rows:
            meta_rows = self._catalog_select(
                cur,
                "SELECT n.oid, n.nspname, current_user"
                " FROM pg_catalog.pg_namespace n"
                " WHERE n.nspname = pg_catalog.current_schema()",
            )
        if meta_rows:
            return int(meta_rows[0][0]), str(meta_rows[0][1]), str(meta_rows[0][2])
        return None, None, None

    def _pg_query_table_partitions(
        self,
        cur: Any,
        table_ref: str,
        expected_schema_name: Optional[str],
    ) -> Optional[Tuple[bool, List[List[str]]]]:
        """Query the full recursive partition tree of PostgreSQL ``table_ref``
        (a valid PostgreSQL SQL table reference, preserving any schema qualifier
        and identifier quoting) using OID-independent
        ``[schema_key, relname, partbound]`` triples.

        Returns ``None`` if the metadata query could not run (e.g. non-PG mock),
        ``(False, [])`` if ``to_regclass`` cannot resolve ``table_ref``, or
        ``(True, sorted_partitions)`` when the table exists."""
        lit_reg = "'" + table_ref.replace("'", "''") + "'"
        rows = self._catalog_select(
            cur,
            f"WITH RECURSIVE __mv_root(relid) AS ("
            f" SELECT pg_catalog.to_regclass({lit_reg})::oid"
            f"), __mv_tree(relid, depth) AS ("
            f" SELECT relid, 0 FROM __mv_root WHERE relid IS NOT NULL"
            f" UNION"
            f" SELECT i.inhrelid, __mv_tree.depth + 1"
            f" FROM pg_catalog.pg_inherits i"
            f" JOIN __mv_tree ON i.inhparent = __mv_tree.relid"
            f") SELECT"
            f" (SELECT relid IS NOT NULL FROM __mv_root),"
            f" n.nspname,"
            f" c.relname,"
            f" COALESCE(pg_catalog.pg_get_expr(c.relpartbound, c.oid), '')"
            f" FROM __mv_root r"
            f" LEFT JOIN __mv_tree t ON t.depth > 0"
            f" LEFT JOIN pg_catalog.pg_class c ON c.oid = t.relid"
            f" LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace"
            f" ORDER BY 2 NULLS FIRST, 3 NULLS FIRST, 4 NULLS FIRST",
        )
        if rows is None:
            return None
        if not rows or not bool(rows[0][0]):
            return (False, [])
        parts: List[List[str]] = []
        seen_parts: Set[Tuple[str, str, str]] = set()
        for _, nsp, rel, bound in rows:
            if rel is None:
                continue
            nsp_s = str(nsp or "")
            schema_key = "" if (expected_schema_name and nsp_s == expected_schema_name) else nsp_s
            triple = (schema_key, str(rel), str(bound or ""))
            if triple not in seen_parts:
                seen_parts.add(triple)
                parts.append(list(triple))
        parts.sort()
        return (True, parts)

    def _pg_snapshot_partitions(
        self,
        cur: Any,
        table_refs: Sequence[str],
        expected_schema_name: Optional[str] = None,
    ) -> Optional[Dict[str, List[List[str]]]]:
        """Capture an OID-independent snapshot of each referenced source table's
        partition tree on PostgreSQL. Returns ``None`` if catalog queries fail
        or any referenced source table cannot be resolved (never silently omits
        an unresolved table into an incomplete snapshot)."""
        if not callable(getattr(cur, "fetchall", None)):
            return None
        if expected_schema_name is None:
            _, expected_schema_name, _ = self._pg_resolve_catalog_schema_meta(cur)
        unique_tables = sorted({str(t).strip() for t in table_refs if str(t).strip()})
        snapshot: Dict[str, List[List[str]]] = {}
        for tbl in unique_tables:
            res = self._pg_query_table_partitions(cur, tbl, expected_schema_name)
            if res is None:
                return None
            exists, parts = res
            if not exists:
                return None
            snapshot[tbl] = parts
        return snapshot

    def _pg_trigger_drift(
        self,
        catalog: Dict[str, Dict[str, Any]],
        extra_ddls_by_view: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Dict[str, str]:
        # Priority: missing > disabled > modified > orphaned > partitions_changed.
        _RANK = {
            "partitions_changed": 1,
            "triggers_orphaned": 2,
            "triggers_modified": 3,
            "triggers_disabled": 4,
            "triggers_missing": 5,
        }

        def _merge(cur_status: Optional[str], candidate: str) -> str:
            if cur_status is None or _RANK[candidate] > _RANK[cur_status]:
                return candidate
            return cur_status

        out: Dict[str, str] = {}
        with self._cursor() as cur:
            # Resolve the catalog's own schema (and current_user) once so we
            # reject cross-schema decoy functions and validate "$user" strictly
            # against current_user.
            expected_schema_oid, expected_schema_name, pg_current_user = (
                self._pg_resolve_catalog_schema_meta(cur)
            )

            for name, entry in catalog.items():
                status: Optional[str] = None
                all_ddls: List[str] = list(entry.get("trigger_ddls") or [])
                if extra_ddls_by_view and name in extra_ddls_by_view:
                    all_ddls.extend(extra_ddls_by_view[name])
                expected_fns = self._pg_parse_expected_functions(all_ddls)
                expected_fn_map: Dict[
                    Tuple[str, Optional[str]], Tuple[Optional[str], str, bool, bool, bool]
                ] = {
                    (fn_name, arg_type): (ret_type, prosrc, req_sp, is_strict, is_imm)
                    for fn_name, arg_type, ret_type, prosrc, req_sp, is_strict, is_imm in expected_fns
                }
                expected_triggers = self._pg_parse_expected_triggers(all_ddls)
                checked_trig_names: Set[str] = set()
                verified_trigger_fn_names: Set[str] = set()

                for trg, table, exp_fn, exp_tgtype, exp_old, exp_new in expected_triggers:
                    checked_trig_names.add(trg)
                    lit_t = "'" + table.replace("'", "''") + "'"
                    lit_n = "'" + trg.replace("'", "''") + "'"
                    enabled_mode = "O" if exp_tgtype & 64 else "A"  # tgtype bit 64: INSTEAD OF
                    rows = self._catalog_select(
                        cur,
                        f"WITH RECURSIVE __mv_tree(relid) AS ("
                        f" SELECT pg_catalog.to_regclass({lit_t})::oid"
                        f" UNION ALL SELECT i.inhrelid FROM pg_catalog.pg_inherits i"
                        f" JOIN __mv_tree ON i.inhparent = __mv_tree.relid),"
                        f" __mv_fn AS (SELECT t.tgfoid FROM pg_catalog.pg_trigger t"
                        f" WHERE t.tgrelid = pg_catalog.to_regclass({lit_t})::oid"
                        f"  AND NOT t.tgisinternal AND t.tgname = {lit_n})"
                        f" SELECT"
                        f" (SELECT count(*) FROM __mv_tree WHERE __mv_tree.relid IS NULL OR NOT EXISTS ("
                        f"  SELECT 1 FROM pg_catalog.pg_trigger t WHERE t.tgrelid = __mv_tree.relid"
                        f"  AND NOT t.tgisinternal AND t.tgname = {lit_n})),"
                        f" (SELECT count(*) FROM pg_catalog.pg_trigger t WHERE t.tgfoid IN (SELECT tgfoid FROM __mv_fn)"
                        f"  AND NOT t.tgisinternal"
                        f"  AND t.tgrelid NOT IN (SELECT relid FROM __mv_tree WHERE relid IS NOT NULL)),"
                        # Source-table maintenance triggers must be ENABLE ALWAYS
                        # ('A') to also fire under session_replication_role =
                        # replica; ordinary ENABLE ('O') is insufficient, and
                        # DISABLE ('D') / ENABLE REPLICA ('R') do not fire for
                        # ordinary sessions at all. The read-only INSTEAD OF
                        # triggers on the reader view stay ordinary ('O').
                        f" (SELECT count(*) FROM pg_catalog.pg_trigger t JOIN __mv_tree ON t.tgrelid = __mv_tree.relid"
                        f"  WHERE NOT t.tgisinternal AND t.tgname = {lit_n} AND t.tgenabled NOT IN ('{enabled_mode}'))",
                    )
                    if rows is None:
                        continue
                    missing, orphaned, disabled = int(rows[0][0]), int(rows[0][1]), int(rows[0][2])
                    if missing:
                        status = _merge(status, "triggers_missing")
                        break
                    if disabled:
                        status = _merge(status, "triggers_disabled")
                    if orphaned:
                        status = _merge(status, "triggers_orphaned")

                    # Verify live trigger definition (tgtype, transition tables, no WHEN qual,
                    # no UPDATE OF column filter, no extra trigger args or constraint trigger)
                    # and the target function's schema/visibility, identity, return type, kind,
                    # body, language, security mode, and search_path across every relation in
                    # the partition tree (OID-independent, so pg_dump/restore remains "ok").
                    fn_rows = self._catalog_select(
                        cur,
                        f"WITH RECURSIVE __mv_tree(relid) AS ("
                        f" SELECT pg_catalog.to_regclass({lit_t})::oid"
                        f" UNION ALL SELECT i.inhrelid FROM pg_catalog.pg_inherits i"
                        f" JOIN __mv_tree ON i.inhparent = __mv_tree.relid)"
                        f" SELECT t.tgtype, COALESCE(t.tgoldtable, ''), COALESCE(t.tgnewtable, ''),"
                        f"  t.tgqual IS NOT NULL,"
                        f"  pg_catalog.cardinality(t.tgattr::pg_catalog.int2[]) > 0,"
                        f"  t.tgnargs <> 0, t.tgconstraint <> 0,"
                        f"  p.proname, p.prosrc,"
                        f"  (SELECT pg_catalog.substr(cfg, 13) FROM pg_catalog.unnest(p.proconfig) AS cfg"
                        f"   WHERE cfg LIKE 'search_path=%' LIMIT 1),"
                        f"  l.lanname, p.prosecdef, p.prokind,"
                        f"  p.prorettype = 'pg_catalog.trigger'::pg_catalog.regtype,"
                        f"  p.pronamespace, n.nspname, pg_catalog.pg_function_is_visible(p.oid)"
                        f" FROM pg_catalog.pg_trigger t"
                        f" JOIN __mv_tree ON t.tgrelid = __mv_tree.relid"
                        f" JOIN pg_catalog.pg_proc p ON p.oid = t.tgfoid"
                        f" JOIN pg_catalog.pg_language l ON l.oid = p.prolang"
                        f" JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace"
                        f" WHERE NOT t.tgisinternal AND t.tgname = {lit_n}",
                    )
                    if fn_rows is None:
                        continue
                    if not fn_rows:
                        status = _merge(status, "triggers_missing")
                        break
                    exp_fn_spec = expected_fn_map.get((exp_fn, None))
                    all_rows_match_fn = True
                    for (
                        live_tgtype,
                        live_old,
                        live_new,
                        has_qual,
                        has_col_filter,
                        has_args,
                        is_constraint,
                        live_proname,
                        live_prosrc,
                        live_sp_raw,
                        live_lang,
                        live_secdef,
                        live_prokind,
                        live_ret_is_trigger,
                        live_nspoid,
                        live_schema,
                        live_visible,
                    ) in fn_rows:
                        if (
                            int(live_tgtype) != exp_tgtype
                            or str(live_old or "") != exp_old
                            or str(live_new or "") != exp_new
                            or bool(has_qual)
                            or bool(has_col_filter)
                            or bool(has_args)
                            or bool(is_constraint)
                            or str(live_proname) != exp_fn
                            or not bool(live_visible)
                            or (
                                expected_schema_oid is not None
                                and int(live_nspoid) != expected_schema_oid
                            )
                        ):
                            status = _merge(status, "triggers_modified")
                            all_rows_match_fn = False
                            break
                        if exp_fn_spec is not None:
                            _, exp_src, req_sp, _, _ = exp_fn_spec
                            norm_live_src = str(live_prosrc or "").replace("\r\n", "\n")
                            sp_schema = expected_schema_name or str(live_schema or "")
                            if (
                                norm_live_src != exp_src
                                or str(live_lang or "").lower() != "plpgsql"
                                or bool(live_secdef)
                                or str(live_prokind or "") != "f"
                                or not bool(live_ret_is_trigger)
                                or (
                                    req_sp
                                    and not self._pg_search_path_setting_is_valid(
                                        None if live_sp_raw is None else str(live_sp_raw),
                                        sp_schema,
                                        current_user=pg_current_user,
                                    )
                                )
                            ):
                                status = _merge(status, "triggers_modified")
                                all_rows_match_fn = False
                                break
                    if all_rows_match_fn and exp_fn_spec is not None:
                        verified_trigger_fn_names.add(exp_fn)

                if status != "triggers_missing":
                    # Verify non-trigger helper functions (e.g. Stage 3 reconcile
                    # functions and __matview_key_*(anyelement)) by visible signature,
                    # schema, kind, return type, volatility, strictness, and body,
                    # OID-independently.
                    for (
                        fn_name,
                        arg_type,
                        exp_ret_type,
                        exp_src,
                        req_sp,
                        exp_strict,
                        exp_immutable,
                    ) in expected_fns:
                        if arg_type is None and fn_name in verified_trigger_fn_names:
                            continue
                        lit_fn = "'" + fn_name.replace("'", "''") + "'"
                        if arg_type is None:
                            sig_pred = "p.pronargs = 0"
                        else:
                            lit_arg = "'" + arg_type.replace("'", "''") + "'"
                            sig_pred = (
                                f"p.pronargs = 1 AND p.proargtypes[0] = pg_catalog.to_regtype({lit_arg})"
                            )
                        if exp_ret_type:
                            lit_ret = "'" + exp_ret_type.replace("'", "''") + "'"
                            ret_sel = f"COALESCE(p.prorettype = pg_catalog.to_regtype({lit_ret}), false)"
                        else:
                            ret_sel = "true"
                        h_rows = self._catalog_select(
                            cur,
                            f"SELECT p.prosrc,"
                            f"  (SELECT pg_catalog.substr(cfg, 13) FROM pg_catalog.unnest(p.proconfig) AS cfg"
                            f"   WHERE cfg LIKE 'search_path=%' LIMIT 1),"
                            f"  l.lanname, p.prosecdef, p.proisstrict, p.prokind,"
                            f"  p.provolatile, {ret_sel}, p.pronamespace, n.nspname"
                            f" FROM pg_catalog.pg_proc p"
                            f" JOIN pg_catalog.pg_language l ON l.oid = p.prolang"
                            f" JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace"
                            f" WHERE p.proname = {lit_fn} AND {sig_pred}"
                            f"  AND pg_catalog.pg_function_is_visible(p.oid)",
                        )
                        if h_rows is None:
                            continue
                        if not h_rows:
                            status = _merge(status, "triggers_missing")
                            break
                        (
                            live_src,
                            live_sp_raw,
                            live_lang,
                            live_secdef,
                            live_strict,
                            live_prokind,
                            live_volatile,
                            live_ret_ok,
                            live_nspoid,
                            live_schema,
                        ) = h_rows[0]
                        sp_schema = expected_schema_name or str(live_schema or "")
                        if (
                            str(live_src or "").replace("\r\n", "\n") != exp_src
                            or str(live_lang or "").lower() != "plpgsql"
                            or bool(live_secdef)
                            or str(live_prokind or "") != "f"
                            or not bool(live_ret_ok)
                            or (exp_strict and not bool(live_strict))
                            or (exp_immutable and str(live_volatile or "") != "i")
                            or (
                                expected_schema_oid is not None
                                and int(live_nspoid) != expected_schema_oid
                            )
                            or (
                                req_sp
                                and not self._pg_search_path_setting_is_valid(
                                    None if live_sp_raw is None else str(live_sp_raw),
                                    sp_schema,
                                    current_user=pg_current_user,
                                )
                            )
                        ):
                            status = _merge(status, "triggers_modified")

                if status is None:
                    # Read-only/other triggers recorded by name only (when extra_ddls_by_view
                    # was not provided).
                    names = [
                        t
                        for t in (entry.get("trigger_names") or [])
                        if self._pg_unquote_ident(str(t)) not in checked_trig_names
                    ]
                    if names:
                        lits = ", ".join("'" + t.lower().replace("'", "''") + "'" for t in names)
                        rows = self._catalog_select(
                            cur,
                            f"SELECT count(DISTINCT lower(tgname)) FROM pg_catalog.pg_trigger"
                            f" WHERE NOT tgisinternal AND lower(tgname) IN ({lits})",
                        )
                        if rows is not None and int(rows[0][0]) < len({n.lower() for n in names}):
                            status = "triggers_missing"
                if status is None:
                    # Detect out-of-band PostgreSQL partition drops / replacements
                    # (e.g. DROP TABLE on a partition, DETACH + DROP, or bound changes)
                    # by comparing each source table's live OID-independent partition
                    # tree against the snapshot persisted in partitions_json.
                    if entry.get("partitions_malformed"):
                        status = _merge(status, "partitions_changed")
                    elif isinstance(entry.get("partitions"), dict):
                        if not entry["partitions"] and (
                            expected_triggers
                            or entry.get("trigger_ddls")
                            or entry.get("trigger_names")
                        ):
                            status = _merge(status, "partitions_changed")
                        else:
                            for src_tbl, recorded_parts in entry["partitions"].items():
                                part_res = self._pg_query_table_partitions(
                                    cur, src_tbl, expected_schema_name
                                )
                                if part_res is None:
                                    continue
                                tbl_exists, live_parts = part_res
                                if not tbl_exists or live_parts != recorded_parts:
                                    status = _merge(status, "partitions_changed")
                                    break
                if status:
                    out[name] = status
        return out

    @_manager_read_operation
    def check_schema_drift(self) -> Dict[str, str]:
        """
        Inspects all registered views in `__matview_catalog` against the current live base
        table schemas and returns `{view_name: "ok" | "drifted" | "invalid" |
        "backfill_incomplete" | "triggers_missing" | "triggers_disabled" |
        "triggers_modified" | "triggers_out_of_order" | "triggers_orphaned" |
        "partitions_changed"}`;
        "backfill_incomplete" (definition current, but a pk-range backfill is
        running elsewhere or was interrupted) is never "ok"; "triggers_missing"
        means maintenance triggers or helper functions recorded for the view do
        not exist, also on a PostgreSQL partition attached or created later
        (writes are not maintained); "triggers_disabled" means a PostgreSQL
        maintenance trigger is not ENABLE ALWAYS: disabled or ENABLE REPLICA
        (writes are not maintained) or plain ENABLE (writes under
        `session_replication_role = replica` are not maintained;
        `pg_restore --disable-triggers` leaves triggers in this state);
        "triggers_modified" means a PostgreSQL maintenance/helper function body,
        `search_path`, or trigger definition was modified out of band (e.g.
        `CREATE OR REPLACE FUNCTION` or `ALTER FUNCTION ... RESET search_path`);
        "triggers_out_of_order" means a SQLite user trigger created after
        registration fires before a maintenance trigger and writes a source
        table or uses RAISE(FAIL/IGNORE) (see `_sqlite_trigger_order_drift`),
        or a MySQL/MariaDB `PRECEDES` user trigger writes another join source
        first; "triggers_orphaned" means a PostgreSQL maintenance trigger
        is on a table outside the source's partition tree, e.g. a DETACHed
        partition (its writes still change the view, and its rows are still in
        the view); "invalid" is also reported for a PostgreSQL view whose source is
        (or became) a logical-replication subscription target: replicated row changes
        are not maintained and no repair exists (unsupported configuration,
        `repair_view()` refuses); "partitions_changed" means a PostgreSQL source table's
        partition tree changed out of band (e.g. `DROP TABLE` on a partition,
        `DETACH` followed by `DROP`, or a partition rename/bound change; rows
        of a dropped partition remain in the view until repaired). Repair any
        trigger/function/partition drift with `repair_view()`, which rebuilds
        the view.
        """
        schemas = self.fetch_schema()
        catalog = self.get_catalog()
        plain_views, cur_schema = self._fetch_plain_views(exclude_views=list(catalog.keys()))
        plans_by_view: Dict[str, MaterializedViewPlan] = {}
        invalid_views: Set[str] = set()
        for v_name, entry in catalog.items():
            try:
                plans_by_view[v_name] = self._stored_plan(
                    schemas, entry, plain_views=plain_views, current_schema=cur_schema
                )
            except Exception:
                invalid_views.add(v_name)
        extra_ddls_by_view = {
            v_name: plan.readonly_trigger_ddls
            for v_name, plan in plans_by_view.items()
            if plan.readonly_trigger_ddls
        }
        valid_catalog = {k: v for k, v in catalog.items() if k not in invalid_views}
        trigger_drift = self._views_trigger_drift(
            valid_catalog, extra_ddls_by_view=extra_ddls_by_view
        )
        report: Dict[str, str] = {}
        for v_name, entry in catalog.items():
            if v_name in invalid_views:
                report[v_name] = "invalid"
                continue
            plan = plans_by_view[v_name]
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
            and concurrent readers (`ACCESS SHARE`) remain unblocked. Under
            `REPEATABLE READ` and `SERIALIZABLE`, manager-installed triggers
            additionally fence the view's catalog row with `SELECT ... FOR SHARE`:
            concurrent writers serialize cleanly without blocking each other,
            while any transaction whose snapshot predates `repair_view()` (or
            initial view registration/rebuild) fails closed with SQLSTATE
            `40001` (`serialization_failure`) instead of applying maintenance
            against a stale catalog or backing-table snapshot.
          - On the `"rebuilt"` path (schema drift or missing triggers),
            rebuilds via `register_view(..., force_rebuild=True)`, whose
            `DROP`/`CREATE TRIGGER` DDL serializes writers (`ACCESS EXCLUSIVE`
            / `SHARE ROW EXCLUSIVE`, which also blocks readers while running).
        - MySQL/MariaDB (InnoDB, `READ COMMITTED`, `REPEATABLE READ`, and
          `SERIALIZABLE`): `_repair_view_contents()` runs in a transaction-local
          `REPEATABLE READ` transaction (without altering the session's own
          `transaction_isolation`) and acquires clustered-index next-key shared
          locks (`LOCK IN SHARE MODE`) on all referenced base tables in
          deterministic sorted order before clearing and repopulating the
          backing store, serializing with concurrent writers under all three
          isolation levels regardless of `binlog_format` (with bounded retry on
          transient InnoDB deadlock/lock-timeout/snapshot-conflict errors
          1213/1205/1020).
        - SQLite: writes are serialized database-wide.

        Raises ``TransactionOwnershipError`` (before any SQL, through
        ``_manager_owned_operation``) on a connection with an open transaction
        on every dialect, so the repair never commits or, on MySQL/MariaDB
        where it rolls back its own read snapshots, discards the caller's
        uncommitted work.
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
        plain_views, cur_schema = self._fetch_plain_views(exclude_views=list(catalog.keys()))
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
                plain_views, cur_schema = self._fetch_plain_views(exclude_views=list(catalog.keys()))
            entry = catalog[target_name]
            v_name = entry["view_name"]
            plan = self._stored_plan(schemas, entry, plain_views=plain_views, current_schema=cur_schema)
            sig = self._compute_signature(plan, schemas)
            extra_ddls = (
                {target_name: plan.readonly_trigger_ddls}
                if plan.readonly_trigger_ddls
                else None
            )
            drift_catalog = (
                {target_name: entry} if self.dialect == Dialect.POSTGRES else catalog
            )
            has_trigger_drift = target_name in self._views_missing_triggers(
                drift_catalog, extra_ddls_by_view=extra_ddls
            )
            if rebuild_on_schema_drift and (sig != entry["signature_hash"] or has_trigger_drift):
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
                self._repair_view_contents(plan, v_name, schemas=schemas)
                results[v_name] = "repaired"
        return results

    def _mysql_current_database(self) -> Optional[str]:
        """The connection's default database (``None`` if unknown)."""
        try:
            with self._cursor() as cur:
                cur.execute("SELECT DATABASE();")
                row = cur.fetchone()
            self._rollback()
        except Exception:
            self._rollback()
            return None
        return str(row[0]) if row and row[0] else None

    def _mysql_foreign_table_keys(self, db: str, table: str) -> Optional[TableSchema]:
        """Primary key and column names of ``db.table`` (a table of another database) from the
        information schema, ``None`` when the table or its keys cannot be read."""
        try:
            with self._cursor() as cur:
                self._execute_param(
                    cur,
                    "SELECT column_name FROM information_schema.statistics "
                    "WHERE table_schema = ? AND table_name = ? AND index_name = 'PRIMARY' "
                    "ORDER BY seq_in_index",
                    (db, table),
                )
                pk_cols = [str(r[0]) for r in cur.fetchall()]
                self._execute_param(
                    cur,
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                    (db, table),
                )
                cols = [str(r[0]) for r in cur.fetchall()]
            self._rollback()
        except Exception:
            self._rollback()
            return None
        if not cols:
            return None
        return TableSchema(
            name=table,
            columns={c: ColumnDef(name=c, data_type="", primary_key=c in pk_cols) for c in cols},
            pk_columns=tuple(pk_cols),
        )

    def _repair_view_contents(
        self,
        plan: Any,
        v_name: str,
        *,
        schemas: Optional[Dict[str, TableSchema]] = None,
    ) -> None:
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
        if self.dialect == Dialect.MYSQL and schemas is None:
            schemas = self.fetch_schema()
        mysql_lock_stmts = (
            _mysql_base_table_lock_stmts(
                plan,
                schemas=schemas,
                object_prefix=self.object_prefix,
                current_db=self._mysql_current_database(),
                foreign_schema=self._mysql_foreign_table_keys,
            )
            if self.dialect == Dialect.MYSQL
            else []
        )
        attempt = 0
        while True:
            if self.dialect == Dialect.MYSQL:
                # Close any read-only snapshot left by get_catalog()/fetch_schema()
                # before opening the repair cursor so transaction-local REPEATABLE
                # READ applies cleanly without altering the session's own
                # transaction_isolation.
                self._rollback()
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
                    elif self.dialect == Dialect.MYSQL:
                        # Explicit START TRANSACTION prevents autocommit=True from
                        # splitting the catalog update, base-table locks, DELETE,
                        # and INSERT repopulation across separate transactions.
                        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")
                        cur.execute("START TRANSACTION;")
                    parts_snapshot = (
                        self._pg_snapshot_partitions(cur, pg_lock_targets)
                        if self.dialect == Dialect.POSTGRES
                        else None
                    )
                    if parts_snapshot is not None:
                        self._execute_param(
                            cur,
                            f"UPDATE {self.catalog_table} SET backfill_cursor_pk = ?, "
                            f"backfill_max_pk = ?, backfill_active = 0, backfill_owner = NULL, "
                            f"partitions_json = ? "
                            f"WHERE view_name = ?",
                            (
                                self._COMPLETED_MARKER_PK,
                                self._COMPLETED_MARKER_PK,
                                json.dumps(parts_snapshot),
                                v_name,
                            ),
                        )
                    else:
                        self._execute_param(
                            cur,
                            f"UPDATE {self.catalog_table} SET backfill_cursor_pk = ?, "
                            f"backfill_max_pk = ?, backfill_active = 0, backfill_owner = NULL "
                            f"WHERE view_name = ?",
                            (self._COMPLETED_MARKER_PK, self._COMPLETED_MARKER_PK, v_name),
                        )
                    if self.dialect == Dialect.MYSQL and mysql_lock_stmts:
                        for lock_sql in mysql_lock_stmts:
                            cur.execute(lock_sql)
                            if callable(getattr(cur, "fetchall", None)):
                                cur.fetchall()
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
                        or self._is_snapshot_row_changed_error(exc)
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
            sqls.append(build_drop_sql("TABLE", old_tables, dialect=self.dialect, allow_qualified=False))

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

            sqls.append(build_drop_sql("TABLE", old_tables, dialect=self.dialect, allow_qualified=False))

        else:  # SQLite and generic fallback
            sqls.append(build_drop_sql("VIEW", plan.view_name, dialect=self.dialect, allow_qualified=False))
            sqls.append(build_drop_sql("TABLE", plan.backing_table_name, dialect=self.dialect, allow_qualified=False))
            sqls.append(build_alter_rename_table_sql(shadow_tbl, plan.backing_table_name, dialect=self.dialect))
            for aux_t, s_aux in shadow_aux_map.items():
                sqls.append(build_drop_sql("TABLE", aux_t, dialect=self.dialect, allow_qualified=False))
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

    def _bulk_plan(
        self,
        schemas: Dict[str, TableSchema],
        entry: Dict[str, Any],
        *,
        plain_views: Optional[Mapping[str, str]] = None,
        current_schema: Optional[str] = None,
        exclude_views: Sequence[str] = (),
    ) -> MaterializedViewPlan:
        return self._stored_plan(
            schemas,
            entry,
            plain_views=plain_views,
            current_schema=current_schema,
            exclude_views=exclude_views,
        )

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
        self,
        cur: Any,
        targets: Sequence[Dict[str, Any]],
        use_shadow_swap: bool,
        recreate_triggers: bool,
        all_catalog_views: Optional[Sequence[str]] = None,
    ) -> None:
        """Repopulate every target view inside the current transaction.

        Each view's pk-range backfill watermark is taken over first, as
        repair_view() does: a running backfill stops at its next batch and an
        interrupted one is recorded as complete (the view is fully
        repopulated). Any error propagates; the caller rolls back.
        """
        schemas = self.fetch_schema()
        target_names_excl = (
            list(all_catalog_views)
            if all_catalog_views is not None
            else [e["view_name"] for e in targets]
        )
        plain_views, cur_schema = self._fetch_plain_views(exclude_views=target_names_excl)
        for entry in targets:
            plan = self._bulk_plan(
                schemas,
                entry,
                plain_views=plain_views,
                current_schema=cur_schema,
                exclude_views=target_names_excl,
            )
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
                    # The flag alone does nothing: the triggers skip only while
                    # this session also holds its per-connection named lock,
                    # which bulk_changes() takes here and releases below (and
                    # the server releases when the connection ends). A session
                    # that merely sets the variable (stale pooled connection,
                    # copied code) is still maintained.
                    mysql_flag_set = True
                    cur.execute("SELECT GET_LOCK(CONCAT('__matview_bypass:', CONNECTION_ID()), 0);")
                    got = cur.fetchone()
                    if not got or got[0] != 1:
                        raise RuntimeError("bulk_changes(): could not take the session trigger-bypass lock")
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
                    cur.execute("DO RELEASE_LOCK(CONCAT('__matview_bypass:', CONNECTION_ID()));")
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
                    self._bulk_refresh(
                        cur,
                        targets,
                        False,
                        recreate_triggers=not session_bypass,
                        all_catalog_views=list(catalog.keys()),
                    )
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
                    self._bulk_refresh(
                        cur,
                        targets,
                        use_shadow_swap,
                        recreate_triggers=not session_bypass,
                        all_catalog_views=list(catalog.keys()),
                    )
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
                            reset_cur.execute("DO RELEASE_LOCK(CONCAT('__matview_bypass:', CONNECTION_ID()));")
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
            # Query pg_catalog.pg_trigger directly rather than
            # information_schema.triggers, which omits AFTER TRUNCATE triggers,
            # and scope by the trigger function's schema matching the catalog's
            # schema (so triggers on cross-schema source tables are included
            # while same-named triggers from another schema's installation are not).
            lit_cat = "'" + self.catalog_table.replace("'", "''") + "'"
            return self._query_lower_name_set(
                "SELECT t.tgname FROM pg_catalog.pg_trigger t "
                "JOIN pg_catalog.pg_proc p ON p.oid = t.tgfoid "
                "JOIN pg_catalog.pg_namespace pn ON pn.oid = p.pronamespace "
                "WHERE NOT t.tgisinternal "
                "AND pn.nspname = COALESCE("
                "  (SELECT cn.nspname FROM pg_catalog.pg_class cc "
                "   JOIN pg_catalog.pg_namespace cn ON cn.oid = cc.relnamespace "
                f"   WHERE cc.oid = pg_catalog.to_regclass({lit_cat})::oid),"
                "  pg_catalog.current_schema()"
                ");"
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

        plain_views, cur_schema = self._fetch_plain_views(
            extra_schema_sql=schema_sql,
            exclude_views=list(catalog.keys()),
        )
        all_specs = parse_view_definitions(
            specs_input,
            known_tables={k.lower() for k in schemas},
            indexes=indexes,
            dialect=self.dialect,
            plain_views=plain_views,
            current_schema=cur_schema,
            schemas=schemas,
        )

        compiled_plans: List[MaterializedViewPlan] = []
        for v_spec in all_specs:
            compiled_plans.append(
                generate_materialized_view(
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
                    repair_fence_catalog_table=self.catalog_table if self.dialect == Dialect.POSTGRES else None,
                    plain_views=plain_views,
                    current_schema=cur_schema,
                )
            )

        pg_trigger_drift: Dict[str, str] = {}
        if self.dialect == Dialect.POSTGRES and catalog:
            matched_catalog: Dict[str, Dict[str, Any]] = {}
            extra_ddls_by_view: Dict[str, Sequence[str]] = {}
            for p in compiled_plans:
                res_cat = self._resolve_catalog_name(p.view_name, catalog)
                if res_cat in catalog:
                    matched_catalog[res_cat] = catalog[res_cat]
                    if p.readonly_trigger_ddls:
                        extra_ddls_by_view[res_cat] = p.readonly_trigger_ddls
            if matched_catalog:
                pg_trigger_drift = self._pg_trigger_drift(
                    matched_catalog, extra_ddls_by_view=extra_ddls_by_view
                )

        for plan in compiled_plans:
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
                pg_drift_reason = pg_trigger_drift.get(resolved_cat_name)

                is_clean_noop = (
                    cat_sig == sig
                    and backing_exists
                    and view_exists
                    and trigs_match
                    and pg_drift_reason is None
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
                    pg_drift_needs_replace = pg_drift_reason is not None
                    needs_replace = (
                        query_sql_changed
                        or strategy_changed
                        or backing_changed
                        or not backing_exists
                        or cat_sig != sig
                        or pg_drift_needs_replace
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
                        if pg_drift_needs_replace and not (
                            query_sql_changed
                            or strategy_changed
                            or backing_changed
                            or not backing_exists
                            or cat_sig != sig
                        ):
                            summary = f"View '{v_name}' has {pg_drift_reason}; pending REPLACE."
                        else:
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
                        # Also pull in helper functions invoked via PERFORM from matched trigger functions
                        # (e.g., PostgreSQL Stage 3 multi-source reconcile functions).
                        helper_fn_names: Set[str] = set()
                        for stmt in fn_ddls:
                            for pm in re.finditer(
                                r"\bPERFORM\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                                stmt,
                                re.IGNORECASE,
                            ):
                                cand = pm.group(1).lower()
                                if cand not in needed_fn_names:
                                    helper_fn_names.add(cand)
                        if helper_fn_names:
                            helper_ddls = [
                                stmt
                                for stmt in plan.trigger_ddls
                                if any(
                                    re.search(
                                        rf"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:\"?{re.escape(fn)}\"?)\b",
                                        stmt,
                                        re.IGNORECASE,
                                    )
                                    for fn in helper_fn_names
                                )
                                and stmt not in matched_trig_ddls
                                and stmt not in fn_ddls
                            ]
                            fn_ddls = helper_ddls + fn_ddls
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
                            drop_ddls.append(build_drop_sql("TRIGGER", t_name, dialect=self.dialect, allow_qualified=False))
                    for fn_name in cat_entry.get("function_names", []):
                        drop_ddls.append(f"DROP FUNCTION IF EXISTS {fn_name if '(' in fn_name else fn_name + '()'} CASCADE;")
                    drop_ddls.append(build_drop_sql("VIEW", cat_name, dialect=self.dialect, allow_qualified=False))
                    if cat_entry.get("backing_table"):
                        drop_ddls.append(build_drop_sql("TABLE", cat_entry["backing_table"], dialect=self.dialect, allow_qualified=False))
                    for aux_t in cat_entry.get("aux_tables", []):
                        drop_ddls.append(build_drop_sql("TABLE", aux_t, dialect=self.dialect, allow_qualified=False))

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


def _resolve_dialect_and_engine_version_unprobed(
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


def _resolve_dialect_and_engine_version(
    schema_sql_or_dict: Any,
    dialect: Optional[Dialect | str | EngineVersion],
    engine_version: Optional[EngineVersion | str],
) -> EngineVersion:
    resolved = _resolve_dialect_and_engine_version_unprobed(schema_sql_or_dict, dialect, engine_version)
    if hasattr(schema_sql_or_dict, "cursor") or isinstance(schema_sql_or_dict, sqlite3.Connection):
        # Explicit versions skip detection but not the probe of the connection.
        _raw = unwrap_dbapi_connection(schema_sql_or_dict)[0]
        resolved = with_probed_mysql_collation(_raw, with_probed_sqlite_statistics(_raw, resolved))
    return resolved


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
    repair_fence_catalog_table: Optional[str] = None,
    plain_views: Optional[Mapping[str, str]] = None,
    current_schema: Optional[str] = None,
) -> MaterializedViewPlan:
    """
    Compiles a `CREATE VIEW` SQL definition (or multi-view SQL script) and base table schema
    into a `MaterializedViewPlan`. Autodetects the database dialect and version when passed
    a live connection, or accepts `engine_version` (e.g. `'postgres:17'`, `'mariadb:11.4'`, `'mysql:8.4'`).
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    eff_plain_views: Optional[Mapping[str, str]] = plain_views
    eff_current_schema: Optional[str] = current_schema
    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict, dialect=eff_dialect)
        if eff_plain_views is None:
            eff_plain_views = parse_plain_views(schema_sql_or_dict, dialect=eff_dialect)
    elif hasattr(schema_sql_or_dict, "cursor") or isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_database_schemas(
            schema_sql_or_dict,
            dialect=eff_dialect,
            object_prefix=object_prefix,
        )
        if eff_plain_views is None:
            eff_plain_views, eff_current_schema = introspect_plain_views(
                schema_sql_or_dict,
                dialect=eff_dialect,
                object_prefix=object_prefix,
            )
    else:
        schemas = dict(schema_sql_or_dict)

    known_tables = {k.lower() for k in schemas}
    if isinstance(view_sql_or_spec, str) or (eff_plain_views and not view_sql_or_spec.referenced_plain_views):
        view_spec = parse_view_definition(
            view_sql_or_spec,
            known_tables=known_tables,
            indexes=indexes,
            dialect=eff_dialect,
            plain_views=eff_plain_views,
            current_schema=eff_current_schema,
            schemas=schemas,
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
            original_select_sql=view_sql_or_spec.original_select_sql,
            referenced_plain_views=view_sql_or_spec.referenced_plain_views,
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
        repair_fence_catalog_table=repair_fence_catalog_table,
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
    watermark_catalog_table: Optional[str] = None,
    pg_fillfactor: Optional[int] = 80,
    repair_fence_catalog_table: Optional[str] = None,
    plain_views: Optional[Mapping[str, str]] = None,
    current_schema: Optional[str] = None,
) -> List[MaterializedViewPlan]:
    """
    Compiles one or more `CREATE VIEW` SQL definitions (including multi-view SQL scripts)
    into a flat list of `MaterializedViewPlan` objects.
    """
    resolved_ev = _resolve_dialect_and_engine_version(schema_sql_or_dict, dialect, engine_version)
    eff_dialect = resolved_ev.dialect

    eff_plain_views: Optional[Mapping[str, str]] = plain_views
    eff_current_schema: Optional[str] = current_schema
    if isinstance(schema_sql_or_dict, str):
        schemas = parse_table_schema(schema_sql_or_dict, dialect=eff_dialect)
        if eff_plain_views is None:
            eff_plain_views = parse_plain_views(schema_sql_or_dict, dialect=eff_dialect)
    elif hasattr(schema_sql_or_dict, "cursor") or isinstance(schema_sql_or_dict, sqlite3.Connection):
        schemas = introspect_database_schemas(
            schema_sql_or_dict,
            dialect=eff_dialect,
            object_prefix=object_prefix,
        )
        if eff_plain_views is None:
            eff_plain_views, eff_current_schema = introspect_plain_views(
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
        plain_views=eff_plain_views,
        current_schema=eff_current_schema,
        schemas=schemas,
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
            watermark_catalog_table=watermark_catalog_table,
            pg_fillfactor=pg_fillfactor,
            repair_fence_catalog_table=repair_fence_catalog_table,
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

