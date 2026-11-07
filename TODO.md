# `matview` Roadmap & Prioritized TODO List

This document tracks all feature goals, architectural improvements, and verified findings from the project roadmap and independent code reviews.

## Completed (`[x]`)

- [x] **Differential Random SQL Fuzzing & Regression Corpus**:
  I would like the generator to be fuzzed by generating valid random schemas and views, as well as random DMLs, and then comparing the materialized view after each DML to the result of running the view directly (`DifferentialSQLFuzzer` in Suite 12 + permanent fuzzing regression corpus in Suite 13).
- [x] **Live Database Schema Introspection Tests**:
  I would like the tests to exercise also pulling the schema from an existing database (`introspect_database_schemas` tested across SQLite, PostgreSQL, and MySQL/MariaDB in `test_08_pull_schema_from_existing_database_across_all_dialects`).
- [x] **Unified Multi-Dialect Testing Infrastructure**:
  I would like all tests to be unified (taking care of using the right dialects) on top of the same shared testing infrastructure so that even when you're just running the sqlite tests in memory you exercise all the shared infrastructure used also when testing with other engines e.g. mysql/pg (`TranspiledConnectionAdapter` + `MVTestRunner`).
- [x] **Autodetect or Accept Target Engine & Version (`--engine-version` / `detect_engine_version`)**:
  I would like the generator to autodetect (when connecting to the database) or accept as arg the engine and version, so that it can autoselect the correct and most efficient dialect/subdialect (`EngineSubdialect`, `EngineVersion`, `parse_engine_version`, `detect_engine_version`, and `--engine-version` across SQLite, PostgreSQL, MySQL, and MariaDB, including PostgreSQL 15+ `UNIQUE ... NULLS NOT DISTINCT` group indexes, PostgreSQL 10+ transition-table gating, MariaDB vs MySQL subdialect detection, and SQLite 3.24+ UPSERT / 3.25+ window-function version checks).
- [x] **Atomic Single-Transaction Registration & Initial Backfill**:
  Wrap backing/auxiliary table creation, trigger installation, single-shot initial backfill, and `__matview_catalog` registration in a single atomic transaction (with rollback on error) in `MaterializedViewManager._register_single_view_spec` to close the concurrency window between trigger creation and single-shot population.
- [x] **Safe Batch-Size Substitution in `MaterializedViewPlan.get_populate_statements`**:
  Target only `__mv_batch_size` declarations and `__mv_batch_id` window divisors when overriding `batch_size`, preventing regex corruption of user SQL queries containing integer literals equal to the default batch size.
- [x] **Exception-Safe `bulk_changes()` Context Manager**:
  Roll back uncommitted base-table mutations and cleanly restore maintenance triggers without committing dirty state when an exception is raised inside a `with manager.bulk_changes():` block.
- [x] **Strict Compilation Mode (`strict=True` / `--strict`)**:
  Raise `StrictCompilationError` when `strict=True` (or `--strict` on the CLI) is set and a view cannot be compiled with Tier 1 `ALGEBRAIC_DELTA` maintenance, preventing silent fallback to expensive Tier 2/Tier 3 refreshes on large production tables.
- [x] **Native Composite Primary Keys for Single-Table Projections & MySQL `NOT NULL` Group Keys**:
  Preserve the base table's actual primary/composite key columns in `_compile_tier1_zset_projection` and enable native composite `PRIMARY KEY (col1, col2, ...)` on MySQL `NOT NULL` `GROUP BY` views instead of synthesizing concatenated `__mv_pk` strings.
- [x] **`SELECT *` and `SELECT DISTINCT *` Tier 1 Expansion**:
  Expand `SELECT *` and `SELECT alias.*` on single-table views against `primary_schema.columns` so `SELECT * FROM t [WHERE ...]` and `SELECT DISTINCT * FROM t [WHERE ...]` compile with Tier 1 $O(1)$ `ALGEBRAIC_DELTA` maintenance instead of falling back to Tier 3.
- [x] **High-Precision `NUMERIC` / `DECIMAL` Accumulation for Variance & Standard Deviation**:
  Use exact `NUMERIC` (PostgreSQL) and `DECIMAL(65, 12)` (MySQL) backing columns (`__mv_sum_*`, `__mv_sum2_*`) for `VAR_POP`, `VAR_SAMP`, `STDDEV_POP`, and `STDDEV_SAMP` to prevent IEEE-754 catastrophic cancellation on high-offset values.
- [x] **63-Bit SQLite Bitwise Aggregates & Nested-Parenthesis Lowering**:
  Widen SQLite `BIT_AND`, `BIT_OR`, and `BIT_XOR` bit-unrolling from 32 bits to all 63 positive bits of signed 64-bit SQLite integers (using balanced 8-way expression trees) and support nested parentheses (e.g. `BIT_AND(COALESCE(x, 0))`) in `_rewrite_sqlite_total_in_sql`.
- [x] **Lexer Support for PostgreSQL Dollar-Quoted Strings (`$$...$$`) & Backtick Identifiers**:
  Handle `$tag$...$tag$` dollar-quoted string literals and `` `...` `` backtick identifiers in `_strip_sql_comments` and `_split_top_level`.
- [x] **Schema Drift Detection (`check_schema_drift`), Drift Repair (`repair_view`), and Multi-Dialect CLI DSNs**:
  Add `manager.check_schema_drift()` (`--check-drift`), `manager.repair_view()` (`--repair`), and `postgresql://` / `mysql://` / `mariadb://` / `sqlite:///` DSN connection support in `MaterializedViewManager` and the CLI.

## P0 — Core AST Migration, Quoting Safety & Concurrency Hardening

- [ ] **Replace Regex/Manual SQL Parsing with `sqlglot` AST**:
  I would like to avoid relying on regexes and other brittle manual parsing of the SQL, and instead use something proven like `sqlglot` across view definition parsing (`_parse_single_create_view_stmt`), CTE/subquery unwrapping (`_try_unwrap_simple_cte_or_derived_view`), join qualification (`_qualify_join_expr`), record column rewriting (`_rewrite_expr_for_record`), and dialect expression lowering (`_rewrite_sqlite_total_in_sql`).
- [ ] **Format, Optimize & Strictly Quote Emitted SQL via `sqlglot`**:
  I would like the output SQL to be properly formatted (possibly by `sqlglot` itself) and possibly even optimized, with consistent dialect-specific identifier quoting (`"ident"` on SQLite/PostgreSQL, `` `ident` `` on MySQL/MariaDB) so base tables, columns, or aliases containing reserved keywords, spaces, or punctuation never break generated DDL or trigger SQL.
- [ ] **Snapshot-Isolated & Watermark-Gated Chunked Incremental Backfills (`incremental_backfill=True`)**:
  Close the concurrency window during multi-batch chunked backfills (`backfill_incrementally`) that commit after each batch:
  - Support snapshot-isolated backfills (`REPEATABLE READ` on PostgreSQL, `START TRANSACTION WITH CONSISTENT SNAPSHOT` on MySQL, `BEGIN IMMEDIATE` on SQLite) where supported, and
  - Track the active backfill cursor watermark (`__mv_cursor_pk`) in `__matview_catalog` so concurrent row triggers ignore or coordinate rows above the watermark that have not yet been folded by the backfill cursor.
- [ ] **Native Composite Keys for Nullable Group Columns, Joins & `UNION ALL` Views**:
  Eliminate remaining string-serialized `__mv_pk` surrogate keys (`quote(a) || '|' || quote(b)`, `ROW(...)::TEXT`, `CONCAT_WS('|', ...)`) where supported:
  - Use `UNIQUE NULLS NOT DISTINCT (col1, col2, ...)` as the conflict target on PostgreSQL 15+ and SQLite `UNIQUE` indexes with null-coalesced or native composite keys, and
  - Ensure all backing and auxiliary tables index individual leading `GROUP BY` columns so engines can perform index range scans and MySQL never hits `VARCHAR(512)` key-length limits on wide group keys.
- [ ] **Welford Running-Variance (`mean`, `M2`) Maintenance for IEEE-754 Float Engines**:
  Complement PostgreSQL `NUMERIC` and MySQL `DECIMAL` variance accumulators with Welford's running-variance update rules (`count`, `mean`, `M2`) for SQLite `REAL` and floating-point columns under high-churn updates/deletes.
- [ ] **Session-Scoped or Shadow-Table Swap `bulk_changes()` Isolation**:
  Avoid dropping maintenance triggers globally across concurrent connections during `bulk_changes()` by supporting session-local trigger bypass flags (`SET LOCAL matview.skip_triggers = 'on'` in PostgreSQL, `@__matview_skip_triggers = 1` in MySQL, or a temp control table in SQLite) or rebuilding into a shadow table followed by an atomic `ALTER TABLE ... RENAME` swap.

## P1 — Modular Architecture, Engine/Version Optimizations & Developer Experience

- [ ] **Refactor Monolithic `matview_sync.py` into a Modular Package with a Typed IR**:
  Split `matview_sync.py` into focused modules with a typed, immutable Intermediate Representation (`CompilationContext` / `LogicalIR`) between SQL parsing and dialect code generation instead of mutating compiler attributes (`self.view`, `self.primary_schema`, `self._working_select_items`) in place:
  - `matview/sql_frontend.py` — SQL parsing, normalization, and AST validation
  - `matview/logical_ir.py` — Engine-independent relational & aggregate representation
  - `matview/planner.py` — Strategy selection, index analysis, and cost estimation
  - `matview/maintenance/` — Tier 1 (`single`, `join`, `zset`, `union_all_agg`), Tier 2, and Tier 3 delta rules
  - `matview/dialects/` — SQLite, PostgreSQL, MySQL, MariaDB, and NewSQL codegen
  - `matview/manager.py` — Catalog, backfill, schema drift, migration, and recovery
  - `matview/cli.py` — CLI commands (`compile`, `apply`, `diff`, `check-drift`, `repair`)
- [ ] **Engine & Version-Specific Optimal Code Generation**:
  I would like the generator to support as efficiently as possible every engine and engine version (possibly considering performance features specific to each), and I would like the generated code to be optimal and as efficient as if it had been written as efficiently as possible for that specific view and that specific engine/version by a senior DBA:
  - PostgreSQL `FILLFACTOR` tuning on high-churn backing tables and `MERGE` statements on PostgreSQL 15+.
  - MySQL 8.0.19+ `AS new` row alias in `ON DUPLICATE KEY UPDATE` where supported, and MariaDB vs MySQL optimizer-specific rewrites.
  - SQLite 3.35+ `RETURNING` and `WITHOUT ROWID` on clustered composite-PK backing/auxiliary tables.
- [ ] **Vectorize PostgreSQL Transition-Table Triggers into Set-Based Delta Joins**:
  When `--pg-transition-tables` is enabled in `_emit_dialect_triggers_for_table`, replace the PL/pgSQL `FOR NEW IN SELECT * FROM __mv_new_rows LOOP ... END LOOP;` row loop with a single set-based delta aggregation over `__mv_new_rows` / `__mv_old_rows` merged via `INSERT ... ON CONFLICT DO UPDATE` and `UPDATE ... FROM` to avoid $O(N)$ statement executions on bulk DML.
- [ ] **Add `matview diff` CLI Command & Compilation Plan Diagnostics**:
  Provide a `matview diff` command to inspect pending trigger/table DDL changes against `__matview_catalog` before execution, and emit structured plan diagnostics explaining strategy selection and fallback reasons.
- [ ] **Docker / Docker Compose Integration Test Environment (`make docker-test`)**:
  I would like the repo to contain a `Dockerfile`/`docker-compose.yml` that spins up all servers needed for integration testing (`make docker-test`) matching the PostgreSQL, MySQL, and MariaDB versions in CI.

## P2 — Index Intelligence & Incremental SQL Feature Coverage

- [ ] **Introspect & Leverage Existing Base-Table Indexes**:
  I would like the generator to consider existing indexes on the base tables when deciding how to most efficiently perform the materialization (introspecting `sqlite_master` `type='index'`, `pg_indexes`, and `information_schema.statistics` in `introspect_database_schemas`).
- [ ] **Propose Additional Base-Table Indexes (`recommended_base_indexes`)**:
  I would like the generator to propose additional indexes on the base tables, when adding them would yield most efficient materialization (e.g. covering join `ON` columns in Tier 1 Join triggers, `GROUP BY` columns in `MIN`/`MAX` delete recomputations, and Tier 2 group-scoped refresh predicates).
- [ ] **Automatically Create Optimal Indexes on Auxiliary & Materialized Tables**:
  I would like the generator to automatically create indexes on the aux or materialized tables when creating them helps materialization efficiency (including indexing group columns on `__matview_dist_*` auxiliary tables and foreign/join keys on backing tables).
- [ ] **Tier 1 Algebraic Maintenance for `LEFT OUTER JOIN`**:
  Support $O(1)$ incremental delta maintenance for `LEFT JOIN` projections and aggregates by maintaining a right-side match counter (`__mv_right_cnt`) per left tuple/key to emit or retract the `NULL`-extended row when right-hand matches transition between `0` and `>0`.
- [ ] **Optional Reference-Counted Auxiliary Tables for `MIN`/`MAX` and `BIT_AND`/`BIT_OR`**:
  Allow configuring ordered value-count auxiliary tables (reusing the `__matview_dist_*` reference-count pattern) for `MIN`/`MAX` and per-bit counters for `BIT_AND`/`BIT_OR` to avoid $O(N_{\text{group}})$ correlated base-table scans on delete-heavy workloads.
- [ ] **Partition-Scoped (Tier 2) Maintenance for Window Functions**:
  Scope window function refreshes (`ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LAG()`, `LEAD()`) to only the affected `PARTITION BY` key(s) rather than falling back to a full-table Tier 3 rebuild when a `PARTITION BY` clause is present.

## P3 — Multi-View DAGs, Shared Computation, Async/Bulk MVs, Docs & NewSQL

- [ ] **Cascading Materialized Views (MVs Referencing Other MVs)**:
  I would like materialized view definitions to also be able to reference other materialized views, and their updates to still happen as efficiently as possible (building a dependency DAG over views and treating upstream `__matview_store_<v1>` backing tables as base tables for downstream views with recursive/cascading trigger propagation). Note: already implemented in branch cascading-mvs.
- [ ] **Multi-View Trigger Consolidation & Shared Sub-Aggregation Deduplication**:
  I would like, when multiple views use the same base tables, to avoid redundant computations (e.g. in case two views both use the same aggregation over the same data):
  - Merge multiple triggers on the same `(base_table, event)` into a single consolidated trigger/function call per row mutation, and
  - Factor shared `(base_table, WHERE, GROUP BY)` aggregations into shared backing tables with per-view reader projections.
- [ ] **Asynchronous, Bulk-Based, and Atomically-Updated Materialized Views**:
  I would like to be able to define, in addition to the default synchronously and atomically-updated MVs, also asynchronously, bulk-based, and atomically-updated materialized views (using lightweight append-only delta log tables `__matview_delta_<table>` populated by triggers or CDC, paired with a set-based batch fold procedure / `manager.refresh_view()`).
- [ ] **SQL Construct Compilation Showcase in Documentation**:
  I would like the `README.md` (or other doc linked from the `README.md`) to showcase how each SQL construct/aggregation and their combinations is efficiently rendered as an incremental materialization.
- [ ] **Support Modern NewSQL Engines (CockroachDB, TiDB, YugabyteDB, Cloud Spanner)**:
  I would like the generator to support also modern NewSQL engines (CockroachDB, TiDB, YugabyteDB, Spanner, etc.) where possible (using PostgreSQL-compatible trigger generation on YugabyteDB/CockroachDB and delta-log/change-stream batch maintenance on engines without general row triggers like TiDB and Cloud Spanner).
- [ ] **ORM & Migration Framework Integrations**:
  Provide first-class integration helpers for SQLAlchemy, Alembic, and Django ORM migrations.
