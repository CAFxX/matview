"""Database lifecycle manager: catalog, backfill, schema drift, migration, bulk changes, and diffs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlalchemy.engine import Engine as _SAEngine

from .compiler import IncrementalViewCompiler
from .connectivity import (
    connect_database_via_sqlalchemy,
    detect_engine_version,
    introspect_database_schemas,
    render_parameterized_sql,
    unwrap_dbapi_connection,
)
from .engines import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
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
MATVIEW_GENERATOR_VERSION = "3"


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
    ``register_view(..., force_rebuild=True)`` or ``drop_view()`` followed by
    ``register_view()``. The underlying error, if any, is ``__cause__``.
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
            f"to (re)populate from scratch use register_view(..., force_rebuild=True) "
            f"or drop_view() + register_view()."
        )


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
        pg_use_transition_tables: bool = False,
        pg_vectorize_transition_tables: bool = True,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
        skip_ensure_catalog: bool = False,
        pg_fillfactor: Optional[int] = 80,
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

        if engine_version is not None:
            self.engine_version: EngineVersion = parse_engine_version(engine_version, default_dialect=hint_d)
        else:
            self.engine_version = detect_engine_version(self.conn, hint_dialect=hint_d)

        self.dialect: Dialect = self.engine_version.dialect
        self.subdialect: EngineSubdialect = self.engine_version.subdialect
        self.ignore_limitations = ignore_limitations
        self.object_prefix = object_prefix
        self.pg_use_transition_tables = pg_use_transition_tables
        self.pg_vectorize_transition_tables = pg_vectorize_transition_tables
        self.strict = strict
        self.pg_fillfactor = pg_fillfactor
        self.catalog_table = (
            f"{self.object_prefix}catalog" if self.object_prefix else "matview_catalog"
        )

        if not skip_ensure_catalog:
            self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        with self._cursor() as cur:
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

    def apply_schema_ddl(self, ddl_sql: str, auto_sync_views: bool = True) -> None:
        """
        Safely executes schema DDL changes (e.g. `CREATE TABLE`, `ALTER TABLE`) and
        automatically refreshes any registered materialized views whose underlying
        base table schema changed.
        """
        with self._cursor() as cur:
            for stmt in _split_top_level(_strip_sql_comments(ddl_sql, dialect=self.dialect), ";", dialect=self.dialect):
                if stmt.strip():
                    cur.execute(stmt)
            self._commit()
        if auto_sync_views:
            catalog = self.get_catalog()
            if catalog:
                self.sync_views([entry["view_sql"] for entry in catalog.values()])

    def get_catalog(self) -> Dict[str, Dict[str, Any]]:
        with self._cursor() as cur:
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
                try:
                    cur.execute(
                        f"SELECT view_name, view_sql, strategy, signature_hash, backing_table, "
                        f"aux_tables_json, trigger_names_json, function_names_json, trigger_ddls_json "
                        f"FROM {self.catalog_table}"
                    )
                    has_wm_cols = False
                except Exception:
                    self._rollback()
                    return {}
            rows = cur.fetchall()
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

    def drop_view(self, view_name: str) -> bool:
        """Drops a registered materialized view, its triggers, read-only view, and backing tables."""
        catalog = self.get_catalog()
        resolved_name = self._resolve_catalog_name(view_name, catalog)
        entry = catalog.get(resolved_name)
        with self._cursor() as cur:
            if entry:
                for t_name in entry["trigger_names"]:
                    if self.dialect != Dialect.POSTGRES:
                        cur.execute(build_drop_sql("TRIGGER", t_name, dialect=self.dialect))
                for fn_name in entry["function_names"]:
                    cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
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
        except BaseException:
            if snapshot_isolation:
                if snapshot_started:
                    self._rollback_snapshot_backfill(cur, plan.view_name)
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
                if not self._backfill_watermark_claimed:
                    self._best_effort_reset_backfill_watermark(plan.view_name)
                else:
                    # Fail closed: this call committed an active pk-range
                    # watermark (and possibly folded chunks). Keep it so
                    # triggers keep excluding the unfolded range; refuse to
                    # resume (see BackfillInterruptedError).
                    # The original exception type is preserved for callers;
                    # the note states the resulting state and the repair.
                    exc_now = sys.exc_info()[1]
                    if exc_now is not None and hasattr(exc_now, "add_note"):
                        entry = self.get_catalog().get(plan.view_name) or {}
                        exc_now.add_note(
                            f"matview: incremental backfill of '{plan.view_name}' stopped after "
                            f"committing progress; its watermark is kept active (cursor_pk="
                            f"{entry.get('backfill_cursor_pk')}, max_pk={entry.get('backfill_max_pk')}) "
                            f"and the view is incomplete. It is not resumed: rebuild with "
                            f"register_view(..., force_rebuild=True) or drop_view() + register_view()."
                        )
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
    def _refuse_unless_backfill_unclaimed(entry: Optional[Dict[str, Any]], view_name: str) -> None:
        if not entry or entry.get("backfill_cursor_pk") is None:
            return
        if entry.get("backfill_active") == 1:
            reason = (
                "a pk-range backfill is in progress in another session or was interrupted; "
                "the view is incomplete until it finishes or is rebuilt"
            )
        else:
            reason = "the view was already backfilled; backfilling again would double-count"
        raise BackfillInterruptedError(
            view_name, reason, entry.get("backfill_cursor_pk"), entry.get("backfill_max_pk")
        )

    def _claim_backfill_watermark(self, cur: Any, view_name: str, cursor_pk: int, max_pk: int) -> None:
        """Atomically claim the pk-range watermark: only one backfill may own it.

        The conditional UPDATE re-checks the claim on the latest row version
        after any concurrent claimer commits, so two sessions cannot both
        start folding the same view.
        """
        try:
            self._execute_param(
                cur,
                f"UPDATE {self.catalog_table} "
                f"SET backfill_cursor_pk = ?, backfill_max_pk = ?, backfill_active = 1 "
                f"WHERE view_name = ? AND backfill_cursor_pk IS NULL",
                (cursor_pk, max_pk, view_name),
            )
        except Exception as exc:
            # A concurrent claimer changed the catalog row under our snapshot:
            # MariaDB innodb_snapshot_isolation (1020), deadlock/serialization
            # victims. Nothing was claimed or folded by this call.
            mysql_code = getattr(exc, "errno", None)
            if mysql_code is None and getattr(exc, "args", None):
                mysql_code = exc.args[0]
            snapshot_conflict = self.dialect == Dialect.MYSQL and mysql_code == 1020
            if snapshot_conflict or self._is_transient_lock_error(exc):
                self._rollback()
                raise BackfillInterruptedError(
                    view_name, f"a concurrent session claimed the pk-range backfill watermark ({exc!r})"
                ) from exc
            raise
        claimed = getattr(cur, "rowcount", None)
        if claimed is None or claimed < 0:
            # Driver/adapter without a DML rowcount (all supported DB-API
            # drivers report one): verify the claim by re-reading the row.
            self._execute_param(
                cur,
                f"SELECT backfill_cursor_pk, backfill_max_pk, backfill_active "
                f"FROM {self.catalog_table} WHERE view_name = ?",
                (view_name,),
            )
            row = cur.fetchone()
            claimed = 1 if row and (row[0], row[1], int(row[2] or 0)) == (cursor_pk, max_pk, 1) else 0
        if claimed != 1:
            self._rollback()
            raise BackfillInterruptedError(
                view_name, "another session claimed the pk-range backfill watermark first"
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
        self._update_backfill_watermark(cur, plan.view_name, upper_pk, max_pk, still_active)
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
                if not plan.view_spec.group_by_cols and not plan.view_spec.has_top_level_distinct:
                    for s in plan.get_populate_statements(incremental_backfill=False):
                        cur.execute(s)
                self._update_backfill_watermark(cur, plan.view_name, None, None, 0)
                if snapshot_isolation and self.dialect == Dialect.SQLITE:
                    cur.execute("RELEASE SAVEPOINT __mv_backfill_snap;")
                self._commit()
                return 0
            min_pk, max_pk = int(row[0]), int(row[1])
            cursor_pk = min_pk - 1
            total_span = max(1, max_pk - min_pk + 1)
            self._claim_backfill_watermark(cur, plan.view_name, cursor_pk, max_pk)
            if not snapshot_isolation:
                self._commit()
                self._backfill_watermark_claimed = True
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
                            or not self._is_transient_lock_error(exc)
                        ):
                            raise
                        attempt += 1
                        self._rollback()
                        self._backfill_sleep(self._backfill_retry_delay(attempt))
                batches_run += 1
                cursor_pk = upper_pk
                if on_progress:
                    done = min(total_span, cursor_pk - (min_pk - 1))
                    on_progress(batches_run, done)
            self._update_backfill_watermark(cur, plan.view_name, max_pk, max_pk, 0)
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
    ) -> MaterializedViewPlan:
        plan = compiled_plan or generate_materialized_view(
            schemas,
            v_spec,
            dialect=self.dialect,
            pg_use_transition_tables=self.pg_use_transition_tables,
            pg_vectorize_transition_tables=getattr(self, "pg_vectorize_transition_tables", True),
            ignore_limitations=effective_ignore,
            object_prefix=self.object_prefix,
            strict=effective_strict,
            engine_version=self.engine_version,
            watermark_catalog_table=self.catalog_table if (incremental_backfill or not populate) else None,
            pg_fillfactor=getattr(self, "pg_fillfactor", 80),
        )
        sig = self._compute_signature(plan, schemas)

        catalog = self.get_catalog()
        existing = catalog.get(plan.view_name)
        if existing and existing["signature_hash"] == sig and not force_rebuild:
            if existing.get("backfill_active") == 1 and existing.get("backfill_cursor_pk") is not None:
                # Re-registering must not silently adopt an interrupted (or
                # concurrently running) backfill as a complete view.
                self._refuse_unless_backfill_unclaimed(existing, plan.view_name)
            return plan

        if existing:
            self.drop_view(plan.view_name)

        # Install structure, maintenance triggers, single-shot backfill, and catalog record
        # inside a single atomic transaction so concurrent readers/writers never observe an
        # un-triggered or double-counted window between trigger attachment and population.
        with self._cursor() as cur:
            created_objects: List[Tuple[str, str]] = []
            catalog_inserted = False
            try:
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
                        backfill_cursor_pk, backfill_max_pk, backfill_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            except Exception:
                self._rollback()
                if self.dialect == Dialect.MYSQL:
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
            dialect=self.dialect,
        )

        for v_spec in all_specs:
            plan = generate_materialized_view(
                schemas,
                v_spec,
                dialect=self.dialect,
                pg_use_transition_tables=self.pg_use_transition_tables,
                pg_vectorize_transition_tables=getattr(self, "pg_vectorize_transition_tables", True),
                ignore_limitations=effective_ignore,
                object_prefix=self.object_prefix,
                strict=effective_strict,
                engine_version=self.engine_version,
                watermark_catalog_table=self.catalog_table if incremental_backfill else None,
                pg_fillfactor=getattr(self, "pg_fillfactor", 80),
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
                    pg_vectorize_transition_tables=getattr(self, "pg_vectorize_transition_tables", True),
                    ignore_limitations=True,
                    object_prefix=self.object_prefix,
                    engine_version=self.engine_version,
                    pg_fillfactor=getattr(self, "pg_fillfactor", 80),
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
        for entry in targets:
            v_name = entry["view_name"]
            plan = generate_materialized_view(
                schemas,
                entry["view_sql"],
                dialect=self.dialect,
                pg_use_transition_tables=self.pg_use_transition_tables,
                ignore_limitations=True,
                object_prefix=self.object_prefix,
                engine_version=self.engine_version,
                pg_fillfactor=getattr(self, "pg_fillfactor", 80),
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
                with self._cursor() as cur:
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
        if self.dialect == Dialect.POSTGRES:
            old_tbl = self._old_name(plan.backing_table_name)
            sqls.append(build_drop_sql("TABLE", old_tbl, dialect=self.dialect))
            for aux_t in plan.auxiliary_table_names:
                sqls.append(build_drop_sql("TABLE", self._old_name(aux_t), dialect=self.dialect))

            sqls.append(build_alter_rename_table_sql(plan.backing_table_name, old_tbl, dialect=self.dialect))
            for aux_t in plan.auxiliary_table_names:
                sqls.append(build_alter_rename_table_sql(aux_t, self._old_name(aux_t), dialect=self.dialect))

            sqls.append(build_alter_rename_table_sql(shadow_tbl, plan.backing_table_name, dialect=self.dialect))
            for aux_t, s_aux in shadow_aux_map.items():
                sqls.append(build_alter_rename_table_sql(s_aux, aux_t, dialect=self.dialect))

            if plan.reader_view_ddl:
                sqls.append(self._make_replace_view_ddl(plan.reader_view_ddl))

            sqls.append(build_drop_sql("TABLE", old_tbl, dialect=self.dialect))
            for aux_t in plan.auxiliary_table_names:
                sqls.append(build_drop_sql("TABLE", self._old_name(aux_t), dialect=self.dialect))

        elif self.dialect == Dialect.MYSQL:
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
        globally across concurrent connections, transactional `BEGIN IMMEDIATE` + `SAVEPOINT` write-lock
        isolation on SQLite, and atomic shadow-table rebuild + atomic table swap (`use_shadow_swap=True`).
        """
        catalog = self.get_catalog()
        if view_names:
            resolved = [self._resolve_catalog_name(n, catalog) for n in view_names]
            targets = [catalog[n] for n in resolved if n in catalog]
        else:
            targets = list(catalog.values())

        cur = self.conn.cursor()
        sp_active = False
        created_shadow_tables: List[str] = []

        try:
            if self.dialect == Dialect.SQLITE:
                # SQLite must acquire the writer reservation before any trigger is
                # suspended.  Failing open here can expose concurrent writers to a
                # partially disabled materialized-view maintenance path.
                if not getattr(self.conn, "in_transaction", False):
                    cur.execute("BEGIN IMMEDIATE;")
                # A failed savepoint is equally unsafe: do not fall back to trigger
                # removal without transactional protection.
                self._savepoint_create(cur, "__mv_bulk_sp")
                sp_active = True
            else:
                try:
                    self._savepoint_create(cur, "__mv_bulk_sp")
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

            def _cleanup_and_restore() -> None:
                if sp_active:
                    try:
                        self._savepoint_rollback_and_release(cur, "__mv_bulk_sp")
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

                for st in created_shadow_tables:
                    try:
                        cur.execute(f"DROP TABLE IF EXISTS {st};")
                    except Exception:
                        pass

                try:
                    schemas = self.fetch_schema()
                    for entry in targets:
                        if self.dialect == Dialect.POSTGRES:
                            for fn_name in entry["function_names"]:
                                if "fn_readonly_" not in fn_name:
                                    try:
                                        cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                                    except Exception:
                                        pass
                        else:
                            for t_name in entry["trigger_names"]:
                                if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                                    try:
                                        cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
                                    except Exception:
                                        pass
                        plan = generate_materialized_view(
                            schemas,
                            entry["view_sql"],
                            dialect=self.dialect,
                            pg_use_transition_tables=self.pg_use_transition_tables,
                            pg_vectorize_transition_tables=getattr(self, "pg_vectorize_transition_tables", True),
                            ignore_limitations=True,
                            object_prefix=self.object_prefix,
                            engine_version=self.engine_version,
                            pg_fillfactor=getattr(self, "pg_fillfactor", 80),
                        )
                        for stmt in plan.trigger_ddls:
                            cur.execute(stmt)
                except Exception:
                    pass
                self._commit()

            try:
                yield
            except Exception:
                _cleanup_and_restore()
                raise

            try:
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
                        pg_vectorize_transition_tables=getattr(self, "pg_vectorize_transition_tables", True),
                        ignore_limitations=True,
                        object_prefix=self.object_prefix,
                        engine_version=self.engine_version,
                        pg_fillfactor=getattr(self, "pg_fillfactor", 80),
                    )

                    if use_shadow_swap:
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
                        created_shadow_tables.append(shadow_tbl)

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
                                s_aux_ddl = aux_ddl.replace(
                                    f"CREATE TABLE {aux_t}",
                                    f"CREATE TABLE {s_aux}",
                                    1,
                                )
                            cur.execute(s_aux_ddl)
                            created_shadow_tables.append(s_aux)

                        for stmt in plan.get_populate_statements(incremental_backfill=False):
                            s_stmt = stmt
                            s_stmt = re.sub(
                                rf"\bINSERT\s+INTO\s+(?:\"{re.escape(plan.backing_table_name)}\"|`{re.escape(plan.backing_table_name)}`|{re.escape(plan.backing_table_name)})\b",
                                f"INSERT INTO {shadow_tbl}",
                                s_stmt,
                                flags=re.IGNORECASE,
                            )
                            s_stmt = re.sub(
                                rf"\bMERGE\s+INTO\s+(?:\"{re.escape(plan.backing_table_name)}\"|`{re.escape(plan.backing_table_name)}`|{re.escape(plan.backing_table_name)})\b",
                                f"MERGE INTO {shadow_tbl}",
                                s_stmt,
                                flags=re.IGNORECASE,
                            )
                            for aux_t, s_aux in shadow_aux_map.items():
                                s_stmt = re.sub(
                                    rf"\bINSERT\s+INTO\s+(?:\"{re.escape(aux_t)}\"|`{re.escape(aux_t)}`|{re.escape(aux_t)})\b",
                                    f"INSERT INTO {s_aux}",
                                    s_stmt,
                                    flags=re.IGNORECASE,
                                )
                                s_stmt = re.sub(
                                    rf"\bMERGE\s+INTO\s+(?:\"{re.escape(aux_t)}\"|`{re.escape(aux_t)}`|{re.escape(aux_t)})\b",
                                    f"MERGE INTO {s_aux}",
                                    s_stmt,
                                    flags=re.IGNORECASE,
                                )
                                s_stmt = re.sub(
                                    rf"\bFROM\s+(?:\"{re.escape(aux_t)}\"|`{re.escape(aux_t)}`|{re.escape(aux_t)})\b",
                                    f"FROM {s_aux}",
                                    s_stmt,
                                    flags=re.IGNORECASE,
                                )
                                s_stmt = re.sub(
                                    rf"\bJOIN\s+(?:\"{re.escape(aux_t)}\"|`{re.escape(aux_t)}`|{re.escape(aux_t)})\b",
                                    f"JOIN {s_aux}",
                                    s_stmt,
                                    flags=re.IGNORECASE,
                                )
                            cur.execute(s_stmt)

                        if not session_bypass_ok or drop_triggers:
                            if self.dialect == Dialect.POSTGRES:
                                for fn_name in entry["function_names"]:
                                    if "fn_readonly_" not in fn_name:
                                        cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                            else:
                                for t_name in entry["trigger_names"]:
                                    if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                                        cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")

                        swap_sqls = self._build_shadow_swap_sqls(plan, shadow_tbl, shadow_aux_map)
                        for s in swap_sqls:
                            cur.execute(s)

                        for ro_stmt in plan.readonly_trigger_ddls:
                            try:
                                cur.execute(ro_stmt)
                            except Exception:
                                pass
                        for idx_ddl in plan.index_ddls:
                            cur.execute(idx_ddl)

                        created_shadow_tables.clear()
                    else:
                        for aux_t in plan.auxiliary_table_names:
                            cur.execute(f"DELETE FROM {aux_t};")
                        cur.execute(f"DELETE FROM {plan.backing_table_name};")
                        for stmt in plan.get_populate_statements(incremental_backfill=False):
                            cur.execute(stmt)

                    if not session_bypass_ok or use_shadow_swap or drop_triggers:
                        if self.dialect == Dialect.POSTGRES:
                            for fn_name in entry["function_names"]:
                                if "fn_readonly_" not in fn_name:
                                    try:
                                        cur.execute(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
                                    except Exception:
                                        pass
                        else:
                            for t_name in entry["trigger_names"]:
                                if "__ro_" not in t_name and not t_name.endswith("__readonly"):
                                    try:
                                        cur.execute(f"DROP TRIGGER IF EXISTS {t_name};")
                                    except Exception:
                                        pass
                        for stmt in plan.trigger_ddls:
                            cur.execute(stmt)

                if sp_active:
                    try:
                        self._savepoint_release(cur, "__mv_bulk_sp")
                    except Exception:
                        pass
                self._commit()
            except Exception:
                _cleanup_and_restore()
                raise
        finally:
            if hasattr(cur, "close"):
                try:
                    cur.close()
                except Exception:
                    pass

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
                pg_use_transition_tables=self.pg_use_transition_tables,
                ignore_limitations=effective_ignore,
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
                        drop_ddls.append(f"DROP FUNCTION IF EXISTS {fn_name}() CASCADE;")
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
    pg_use_transition_tables: bool = False,
    pg_vectorize_transition_tables: bool = True,
    ignore_limitations: bool = False,
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
        )
    else:
        view_spec = view_sql_or_spec

    compiler = IncrementalViewCompiler(
        schemas=schemas,
        view_spec=view_spec,
        dialect=eff_dialect,
        pg_use_transition_tables=pg_use_transition_tables,
        pg_vectorize_transition_tables=pg_vectorize_transition_tables,
        ignore_limitations=ignore_limitations,
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
    pg_use_transition_tables: bool = False,
    pg_vectorize_transition_tables: bool = True,
    ignore_limitations: bool = False,
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
            pg_use_transition_tables=pg_use_transition_tables,
            pg_vectorize_transition_tables=pg_vectorize_transition_tables,
            ignore_limitations=ignore_limitations,
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

