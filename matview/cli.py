"""CLI entrypoint for compiling, applying, diffing, dropping, and maintaining materialized views."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

from .diagnostics import DiagnosticSeverity
from .connectivity import detect_engine_version, introspect_database_schemas
from .engines import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineLimitationError,
    EngineVersion,
    StrictCompilationError,
    parse_engine_version,
)
from .logical_ir import ViewDiff, ViewDiffReport
from .manager import (
    MaterializedViewManager,
    TransactionOwnershipError,
    UnsafeReplacementError,
    _resolve_dialect_and_engine_version,
    generate_materialized_view,
    generate_materialized_views,
)
from .sql_frontend import parse_table_schema

# Exit codes besides 0 (success), 1 (drift/changes found) and 2 (usage,
# limitation or compilation error).
EXIT_UNSAFE_REPLACEMENT = 3  # a published view cannot be replaced atomically here
EXIT_TRANSACTION_OWNERSHIP = 4  # the connection's transaction state cannot be owned



def _announce_recommendations(plan: Any, as_json: bool) -> None:
    """Print the plan's NOTE diagnostics (recommendations) on stderr: they change no guarantee, so
    they never affect the exit code or the generated script on stdout."""
    notes = [d for d in getattr(plan, "strategy_diagnostics", ()) if d.severity == DiagnosticSeverity.NOTE]
    if not notes:
        return
    if as_json:
        print(json.dumps({"recommendations": [d.to_dict() for d in notes]}, indent=2, sort_keys=True), file=sys.stderr)
    else:
        for d in notes:
            print(f"NOTE {d}", file=sys.stderr)

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
        help="Proceed with warning even if engine transparency limitations exist "
        "(except codes that need --accept-limitation).",
    )
    parser.add_argument(
        "--accept-limitation",
        dest="accept_limitations",
        action="append",
        default=[],
        metavar="CODE",
        help="Accept one limitation code explicitly (repeatable). Required for codes that "
        "weaken the result contract, e.g. ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE "
        "(best-effort floating-point aggregates on PostgreSQL/MySQL/MariaDB) and "
        "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX; --ignore-limitations does not cover them.",
    )
    parser.add_argument(
        "--no-sqlite-without-rowid",
        action="store_true",
        help="SQLite: keep the rowid on every matview-owned table (default: WITHOUT ROWID for narrow tables with "
        "a NOT NULL primary key; the row-width estimate cannot see real value sizes, so tables whose TEXT/BLOB "
        "values are large may be faster with a rowid).",
    )
    parser.add_argument(
        "--sqlite-strict-backing",
        action="store_true",
        help="SQLite: declare matview-owned backing/auxiliary tables STRICT where every column type is "
        "provably fixed (opt-in guard rail: it costs a few percent of write time and changes no result).",
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
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Inspect pending trigger/table DDL changes against __matview_catalog before execution.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON diagnostics and diff output.",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Make diff exit with code 1 if differences exist, 0 if no differences.",
    )
    parser.add_argument(
        "--prune-removed",
        action="store_true",
        help="Report DROP for registered views absent from declared view definitions.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a destructive command without an interactive prompt (required in non-interactive use).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the effect of a destructive command without changing the database.",
    )

    raw_argv: List[str] = list(sys.argv[1:]) if argv is None else list(argv)
    command = raw_argv[0] if raw_argv and raw_argv[0] in ("diff", "drop") else None
    filtered_argv = raw_argv[1:] if command else raw_argv

    args = parser.parse_args(filtered_argv)
    if command == "diff":
        args.diff = True
    args.drop = command == "drop"

    def _read_sql_arg(val: str) -> str:
        if "\n" not in val and not re.match(r"^\s*CREATE\b", val, flags=re.IGNORECASE):
            try:
                with open(val, "r", encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                pass
        return val

    try:
        if args.drop:
            if not args.db or not args.view:
                print("ERROR: matview drop requires --db and --view <registered_view_name>.", file=sys.stderr)
                return 2
            if args.dry_run and args.yes:
                print("ERROR: --yes cannot be combined with --dry-run.", file=sys.stderr)
                return 2
            mgr = MaterializedViewManager(
                args.db,
                dialect=args.dialect,
                ignore_limitations=args.ignore_limitations,
                accept_limitations=args.accept_limitations or None,
                object_prefix=args.object_prefix,
                strict=args.strict,
                engine_version=args.engine_version,
            )
            try:
                catalog = mgr.get_catalog()
                resolved_view = mgr._resolve_catalog_name(args.view, catalog)
                if resolved_view not in catalog:
                    print(f"ERROR: materialized view '{args.view}' is not registered.", file=sys.stderr)
                    return 1
                if args.dry_run:
                    print(f"Would drop materialized view '{resolved_view}'.")
                    return 0
                if not args.yes:
                    if not sys.stdin.isatty():
                        print("ERROR: refusing to drop a materialized view non-interactively; pass --yes to confirm or --dry-run to inspect.", file=sys.stderr)
                        return 2
                    answer = input(f"Drop materialized view '{resolved_view}'? [y/N] ")
                    if answer.strip().lower() not in ("y", "yes"):
                        print("Cancelled.")
                        return 0
                if not mgr.drop_view(resolved_view):
                    print(f"ERROR: materialized view '{resolved_view}' was not dropped.", file=sys.stderr)
                    return 1
                print(f"Dropped materialized view '{resolved_view}'.")
                return 0
            finally:
                try:
                    mgr.conn.close()
                except Exception:
                    pass

        if args.diff:
            if not args.db and not args.schema:
                print("ERROR: matview diff requires --db (or --schema).", file=sys.stderr)
                return 2

            view_sql = _read_sql_arg(args.view) if args.view else None
            schema_sql = _read_sql_arg(args.schema) if args.schema else None

            if args.db:
                mgr = MaterializedViewManager(
                    args.db,
                    dialect=args.dialect,
                    ignore_limitations=args.ignore_limitations,
                    accept_limitations=args.accept_limitations or None,
                    object_prefix=args.object_prefix,
                    strict=args.strict,
                    sqlite_strict_backing=args.sqlite_strict_backing,
                    sqlite_without_rowid=not args.no_sqlite_without_rowid,
                    engine_version=args.engine_version,
                )
                try:
                    if not view_sql:
                        cat = mgr.get_catalog()
                        if not cat:
                            print("ERROR: --view is required when no views are registered in --db.", file=sys.stderr)
                            return 2
                    report = mgr.diff_views(
                        view_sqls=view_sql,
                        schema_sql=schema_sql,
                        prune_removed=args.prune_removed,
                        indexes=args.indexes,
                        strict=args.strict,
                        ignore_limitations=args.ignore_limitations,
                        accept_limitations=args.accept_limitations or None,
                        incremental_backfill=args.incremental_backfill,
                        batch_size=args.batch_size,
                    )
                finally:
                    try:
                        mgr.conn.close()
                    except Exception:
                        pass
            else:
                if not view_sql:
                    print("ERROR: --view is required when --db is not provided.", file=sys.stderr)
                    return 2
                # Guarded above: without --db, --schema is required for `diff`.
                assert schema_sql is not None
                schemas = parse_table_schema(schema_sql)
                detected_ev = _resolve_dialect_and_engine_version(
                    schemas, args.dialect, args.engine_version
                )
                plans = generate_materialized_views(
                    schemas,
                    view_sql,
                    dialect=detected_ev.dialect,
                    ignore_limitations=args.ignore_limitations,
                    accept_limitations=args.accept_limitations or (),
                    object_prefix=args.object_prefix,
                    indexes=args.indexes,
                    incremental_backfill=args.incremental_backfill,
                    batch_size=args.batch_size,
                    strict=args.strict,
                    sqlite_strict_backing=args.sqlite_strict_backing,
                    sqlite_without_rowid=not args.no_sqlite_without_rowid,
                    engine_version=detected_ev,
                )
                diffs: Dict[str, ViewDiff] = {}
                for p in plans:
                    sig = hashlib.sha256(p.view_spec.raw_select_sql.encode("utf-8")).hexdigest()
                    diffs[p.view_name] = ViewDiff(
                        view_name=p.view_name,
                        status="CREATE",
                        has_changes=True,
                        strategy=p.strategy.value,
                        current_strategy=None,
                        signature_hash=sig,
                        current_signature_hash=None,
                        diagnostics=p.diagnostics,
                        pending_ddls=p.create_structure_statements() + p.trigger_ddls,
                        pending_changes={
                            "backing_table": {"name": p.backing_table_name, "action": "CREATE"},
                            "auxiliary_tables": [{"name": a, "action": "CREATE"} for a in p.auxiliary_table_names],
                            "reader_view": {"name": p.view_name, "action": "CREATE"},
                            "triggers": {"to_create": list(p.trigger_names), "to_drop": [], "unchanged": []},
                            "indexes": {"to_create": list(p.index_names), "to_drop": [], "unchanged": []},
                        },
                        summary=f"View '{p.view_name}' is not registered; pending creation.",
                        plan=p,
                    )
                report = ViewDiffReport(diffs)

            if args.json:
                print(report.to_json())
            else:
                print(report.render_text())

            if args.exit_code:
                return 1 if report.has_changes else 0
            return 0

        if args.check_drift or args.repair:
            if not args.db:
                print("ERROR: --check-drift / --repair requires --db", file=sys.stderr)
                return 2
            mgr = MaterializedViewManager(
                args.db,
                dialect=args.dialect,
                ignore_limitations=args.ignore_limitations,
                accept_limitations=args.accept_limitations or None,
                object_prefix=args.object_prefix,
                strict=args.strict,
                sqlite_strict_backing=args.sqlite_strict_backing,
                sqlite_without_rowid=not args.no_sqlite_without_rowid,
                engine_version=args.engine_version,
            )
            try:
                if args.check_drift:
                    drift = mgr.check_schema_drift()
                    print(json.dumps(drift, indent=2, sort_keys=True))
                    if any(v != "ok" for v in drift.values()) and not args.repair:
                        return 1
                if args.repair:
                    rep = mgr.repair_view()
                    print(json.dumps(rep, indent=2, sort_keys=True))
                return 0
            finally:
                try:
                    mgr.conn.close()
                except Exception:
                    pass

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
                accept_limitations=args.accept_limitations or None,
                object_prefix=args.object_prefix,
                strict=args.strict,
                sqlite_strict_backing=args.sqlite_strict_backing,
                sqlite_without_rowid=not args.no_sqlite_without_rowid,
                engine_version=args.engine_version,
            )
            try:
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
                _announce_recommendations(plan, args.json)
                print(
                    plan.render_sql_script(
                        include_populate=not args.no_populate,
                        incremental_backfill=args.incremental_backfill,
                        batch_size=args.batch_size,
                    )
                )
                return 0
            finally:
                try:
                    mgr.conn.close()
                except Exception:
                    pass

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
            detected_ev = _resolve_dialect_and_engine_version(
                {}, args.dialect, args.engine_version
            )
            schemas = parse_table_schema(schema_sql, dialect=detected_ev.dialect)
        else:
            print("ERROR: Either --schema or --db must be provided.", file=sys.stderr)
            return 2

        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=detected_ev.dialect,
            ignore_limitations=args.ignore_limitations,
            accept_limitations=args.accept_limitations or (),
            object_prefix=args.object_prefix,
            indexes=args.indexes,
            incremental_backfill=args.incremental_backfill,
            batch_size=args.batch_size,
            strict=args.strict,
            sqlite_strict_backing=args.sqlite_strict_backing,
            sqlite_without_rowid=not args.no_sqlite_without_rowid,
            engine_version=detected_ev,
        )
        _announce_recommendations(plan, args.json)
        print(
            plan.render_sql_script(
                include_populate=not args.no_populate,
                incremental_backfill=args.incremental_backfill,
                batch_size=args.batch_size,
            )
        )
        return 0
    except UnsafeReplacementError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "The CLI has no drop command: drop the named view(s) through the Python API "
            "(MaterializedViewManager(<--db>).drop_view(<name>)), then re-run this command "
            "(apply any schema DDL in between). Readers see the view disappear until it is "
            "registered and populated again.",
            file=sys.stderr,
        )
        return EXIT_UNSAFE_REPLACEMENT
    except TransactionOwnershipError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_TRANSACTION_OWNERSHIP
    except (EngineLimitationError, StrictCompilationError) as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
