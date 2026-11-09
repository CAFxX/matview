# `matview` Roadmap & Prioritized TODO List

This document tracks all feature goals, architectural improvements, and verified findings from the project roadmap and independent code reviews.

## Completed (`[x]`)

- [x] **Baseline Differential Test Harness & Regression Corpus (limited coverage; expansion tracked below)**:
  A deterministic differential-testing harness and permanent regression corpus exist (`DifferentialSQLFuzzer` in Suite 12 + Suite 13). This is only infrastructure, not completion of comprehensive schema/query/DML fuzzing: the current scenario structure and DML operations are template-driven, schema topology is largely fixed, and the oracle/engine simulation/concurrency coverage still need the work listed under P0.
- [x] **Live Database Schema Introspection Tests**:
  I would like the tests to exercise also pulling the schema from an existing database (`introspect_database_schemas` tested across SQLite, PostgreSQL, and MySQL/MariaDB in `test_08_pull_schema_from_existing_database_across_all_dialects`).
- [x] **Unified Multi-Dialect Testing Infrastructure**:
  I would like all tests to be unified (taking care of using the right dialects) on top of the same shared testing infrastructure so that even when you're just running the sqlite tests in memory you exercise all the shared infrastructure used also when testing with other engines e.g. mysql/pg (`TranspiledConnectionAdapter` + `MVTestRunner`).
- [x] **Autodetect or Accept Target Engine & Version (`--engine-version` / `detect_engine_version`)**:
  I would like the generator to autodetect (when connecting to the database) or accept as arg the engine and version, so that it can autoselect the correct and most efficient dialect/subdialect (`EngineSubdialect`, `EngineVersion`, `parse_engine_version`, `detect_engine_version`, and `--engine-version` across SQLite, PostgreSQL, MySQL, and MariaDB, including PostgreSQL 15+ `UNIQUE ... NULLS NOT DISTINCT` group indexes, PostgreSQL 10+ transition-table gating, MariaDB vs MySQL subdialect detection, and SQLite 3.24+ UPSERT / 3.25+ window-function version checks).
- [x] **AST/Token-Backed SQL Parsing, Formatting & Strict Dialect Identifier Quoting (`sqlglot`)**:
  Avoid brittle regex-only SQL parsing by delegating SQL tokenization, AST parsing, formatting, and dialect-specific identifier quoting (`"ident"` on SQLite/PostgreSQL, `` `ident` `` on MySQL/MariaDB) to `sqlglot` (`requirements.txt`, `parse_sql_ast`, `parse_view_ast`, `tokenize_sql`, `quote_sql_identifier`, `format_sql`) so base tables, columns, or aliases containing reserved keywords (`"order"`, `"group"`, `"select"`, `"where"`), spaces (`"unit price"`), or punctuation never break generated DDL or trigger SQL.
- [x] **Snapshot-Isolated & Watermark-Gated Chunked Incremental Backfills (`incremental_backfill=True`)**:
  Close the concurrency window during multi-batch chunked backfills (`backfill_incrementally`) that commit after each batch:
  - Support snapshot-isolated backfills (`REPEATABLE READ` on PostgreSQL, `START TRANSACTION WITH CONSISTENT SNAPSHOT` on MySQL, `SAVEPOINT` on SQLite via `snapshot_isolation=True`), and
  - Track the active backfill cursor watermark (`backfill_cursor_pk`, `backfill_max_pk`, `backfill_active`) in `__matview_catalog` so concurrent row triggers gate on the watermark and never double-count or under-count rows in the pending range.
- [x] **Native Composite Keys & Group-Column Indexes for Nullable Group Columns, Joins & `UNION ALL` Views**:
  Eliminate string-serialized `__mv_pk` surrogate keys (`quote(a) || '|' || quote(b)`, `ROW(...)::TEXT`, `CONCAT_WS('|', ...)`) where supported:
  - Use native composite `PRIMARY KEY (col1, col2, ...)` whenever `GROUP BY` columns are `NOT NULL` across single-table, `INNER JOIN` / `CROSS JOIN`, and `UNION ALL` aggregate views, and native `PRIMARY KEY (__mv_branch_id, __mv_pk_0, ...)` on `UNION ALL` Z-set views,
  - Use `UNIQUE NULLS NOT DISTINCT (col1, col2, ...)` on PostgreSQL 15+ for nullable group keys across single-table, Join, and `UNION ALL` views, and
  - Index both composite group keys and individual non-leading `GROUP BY` columns so engines can perform index scans on any grouping column.
- [x] **Welford Running-Variance (`mean`, `M2`) & Chan Parallel Merge for IEEE-754 Float Engines**:
  Complement PostgreSQL `NUMERIC` and MySQL `DECIMAL(65, 12)` variance accumulators with Welford's centered sum-of-squares (`__mv_m2_*`) single-row update/downdate rules, Chan's parallel batch-merge formula, and anchor-shifted initial `M2` calculation for SQLite `REAL` columns, eliminating IEEE-754 catastrophic cancellation on high-offset values (e.g. `1e9 + delta` and `1e12 + delta`).
- [x] **Session-Scoped & Shadow-Table Swap `bulk_changes()` Isolation**:
  Avoid dropping maintenance triggers globally across concurrent connections during `bulk_changes()` by supporting session-local trigger bypass flags (`SET LOCAL matview.skip_triggers = 'on'` in PostgreSQL, `@__matview_skip_triggers = 1` in MySQL), transactional `SAVEPOINT` isolation in SQLite, and atomic shadow-table rebuild + `ALTER TABLE ... RENAME` swap (`use_shadow_swap=True`).
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

## P0 — Test Correctness & Coverage Gaps

The existing differential-testing harness is useful, but its current headline counts must not be treated as evidence of broad structural coverage. In particular, the schema/scenario generator is largely fixed-structure, DML is selected from a finite set of hardcoded operation templates, query generation recombines predefined patterns, some dialect paths are transpiled to SQLite, numerical comparisons normalize values too aggressively, and the backfill interleaving test does not use independent concurrent sessions. Keep the existing deterministic tests and regression corpus, but do not consider the fuzzing goal complete until the following are implemented and measured.

- [ ] **Repair the Differential-Test Oracle (exact, type-aware comparisons)**:
  - Progress: The oracle preserves exact integer/Decimal results, large-integer boundaries, NaN/infinities/signed zero/NULL/booleans/bytes, duplicate multiplicity, and exact text identity. It uses one-to-one row matching for ambiguous float-tolerant comparisons. NumericTolerance supports absolute, relative, true binary64 ULP distance and per-column policies. The default is exact; a SQLGlot whitelist derives tight SUM/AVG budgets, separate variance/stddev budgets and bounded scalar floating-arithmetic budgets. COUNT/MIN/MAX, grouping keys, unknown expressions and text remain exact. Decimal/float comparisons use high-precision local arithmetic. Tests cover 2^53 and integer limits, Decimal precision, subnormals/extrema, cancellation/high offsets, ULP boundaries, duplicate ambiguity, Unicode identity and wildcard projections. CI exposed a SQLGlot AST recognition gap for anonymous variance/stddev spellings; function-name recognition now checks both normalized and anonymous SQL function names, with explicit regression tests for VAR_POP/VAR_SAMP/VARIANCE/STDDEV variants. Remaining work is native-engine calibration on difficult large aggregates/ill-conditioned values, richer input-type/scale semantics, and collation/result-type coverage.
  - Compare integers exactly without converting them to floating point; add boundary cases around and above 2^53 and the signed/unsigned limits supported by each engine.
  - Compare DECIMAL / NUMERIC values using exact decimal semantics wherever the engine returns exact values; do not round away meaningful digits before comparison.
  - Use explicit, operation-appropriate absolute/relative (and, where useful, ULP-based) tolerances only for approximate floating-point results. Handle NaN, infinities, signed zero, NULL, booleans, text, binary values, and engine-specific result types deliberately.
  - Add adversarial cases for cancellation, high offsets, tiny magnitudes, overflow/underflow, large aggregates, and values differing only beyond the current rounding precision.
  - Ensure row-multiset comparison preserves duplicate multiplicity and does not hide type or collation differences unintentionally.

- [ ] **Replace Fixed-Structure Scenario Generation with Genuine Schema Generation**:
  - Progress: Suite 12 now supplements the legacy fixed schema with a deterministic schema-graph generator varying 2–6 tables, per-table column count and numeric types, nullability/defaults, optional unique/check constraints, single/composite primary keys, and chain/star/disconnected/self-reference/bridge-style foreign-key topologies. It emits schema-shaped seed data, projections, grouped queries, joins, and basic DML. This is an initial structural-diversity layer, not completion: richer type/constraint/relationship combinations, structural coverage feedback, and schema shrinking remain outstanding.
  - Generate variable numbers of tables and columns, data types, nullability, defaults, generated columns where supported, primary and composite keys, unique constraints, foreign keys, referential actions, checks, and indexes.
  - Generate varied relationship graphs: disconnected tables, chains, stars, many-to-many bridges, self-references, and cycles where legal.
  - Populate schemas with constraint-aware data, including empty/sparse/dense tables, duplicate values, NULL distributions, skew, boundary values, and varying cardinalities.
  - Track and report structural schema coverage (not just schema count or randomized identifiers/data); retain seeds and shrink failing schemas.

- [ ] **Replace Template-List DML Fuzzing with a Schema-Aware Recursive DML AST Generator**:
  - Progress: The variable-schema scenarios now generate schema-derived multi-row INSERT, INSERT ... SELECT, multi-column UPDATE, DELETE, and dialect-specific UPSERT sequences. Numeric assignment expressions and WHERE predicates are represented as recursive AST nodes and rendered separately to SQL; the generator reports statement kinds, expression/predicate node kinds, and maximum depths. Deterministic tests execute complete generated sequences with SQLite foreign-key enforcement and run the first generated DML sequence across SQLite, PostgreSQL, and MySQL paths. This is an initial bounded grammar, not completion: arbitrary schema-aware expressions/subqueries/joins, key-change and conflict-target diversity, grammar-guided mutation, coverage feedback, and broader production coverage remain outstanding.
  - Generate valid INSERT (single/multi-row, DEFAULT, INSERT ... SELECT), UPDATE (arbitrary multi-column assignments and expressions), DELETE, and UPSERT/conflict-handling statements from the actual schema and target dialect.
  - Generate recursively composed expressions, predicates, Boolean combinations, CASE, casts, functions, scalar and correlated subqueries, EXISTS/IN, joins and dialect-specific forms where supported.
  - Exercise key changes, composite/unique conflict targets, NULL transitions, no-op updates, updates affecting join/group/filter keys, deletes with cascades/restrictions, and statements affecting zero, one, or many rows.
  - Generate sequences and combinations of statements rather than only independent template instances. Preserve validity through schema-aware generation or classify expected constraint failures explicitly.
  - Every supported DML grammar production must be reachable without adding a new hardcoded SQL template. Report production, depth, operator, clause, and combination coverage; use coverage feedback to bias generation toward unexplored structures.
  - Add grammar-preserving mutation of previously successful statements (operator/operand/predicate/assignment/target/shape changes), while retaining deterministic replay and regression promotion.

- [ ] **Upgrade Query Generation to Recursive, Schema-Aware SQL AST Generation**:
  - Progress: A bounded recursive schema-aware SELECT AST fuzzer composes projections, recursive predicates/expressions, FK joins, grouping/HAVING/aggregates, DISTINCT, CTEs, derived tables, set operations, windows, EXISTS/IN, and scalar/correlated subqueries across 64 deterministic variable schemas (20 queries per schema). A second focused test compiles a deterministic sample as materialized views, checks the initial population and each generated DML mutation against the live SELECT oracle, and classifies unsupported plans by explicit limitation code. Grammar-complete coverage, coverage-guided generation, native-engine differential execution, and automatic shrinking remain outstanding.
  - Generate supported SELECT structures recursively from generated schemas, including projections, joins, filters, grouping, HAVING, aggregates, DISTINCT, CTEs, subqueries, set operations, windows, ordering, limits, and their meaningful combinations.
  - Cover each supported grammar production in multiple contexts and combinations; distinguish generated structural diversity from the number of query instances.
  - Keep unsupported constructs explicit: verify that they are rejected or reported as limitations under default settings rather than silently treating best-effort compilation as supported behavior.
  - Preserve minimized failing queries as permanent regression cases.

- [ ] **Separate Supported-Plan Tests from Best-Effort / Limitation-Override Tests**:
  - Progress: The broad `MVTestRunner` and live 50-view battery now compile each view with default limitation enforcement first. They classify expected rejections by limitation code and only retry with `ignore_limitations=True` after validating the explicit warning and the override plan's limitation codes; reports distinguish supported plans from best-effort overrides. Remaining direct manager/corpus paths still need the same explicit classification and dedicated expected-code coverage.
  - Run supported generated plans with default options and assert that no limitation warning is emitted.
  - For unsupported plans, assert the expected rejection/limitation code and diagnostic under default options.
  - Test ignore_limitations=True separately, asserting the exact warning and documenting which correctness guarantees are intentionally waived.
  - Do not let broad differential suites suppress EngineLimitationWarning and thereby stand in for tests of the default safety contract.

- [ ] **Add Genuine Multi-Connection Concurrency and Isolation Tests**:
  - Progress: A focused file-backed SQLite test now performs INSERT/UPDATE/DELETE from a second thread and independent connection after a committed backfill chunk, exercises rows on both sides of the watermark, checks rollback of trigger effects, and compares the final materialized result. This is real independent-session interleaving on SQLite, not simultaneous statement execution; native PostgreSQL/MySQL isolation, broader race windows, and stress coverage remain outstanding.
  - Use independent database connections/processes and synchronization barriers; do not describe same-connection callback interleaving as concurrent DML.
  - Exercise writes immediately before/during/after trigger installation, initial backfill, every chunk boundary, watermark advancement, and finalization.
  - Cover concurrent INSERT/UPDATE/DELETE, key changes, rows on both sides of the watermark, commits, rollbacks, reader visibility, and competing writers.
  - Test each engine's actual isolation/locking semantics on native SQLite, PostgreSQL, and MySQL/MariaDB configurations where supported.
  - Assert both the final materialized result and intermediate transactional visibility; add bounded stress/repetition runs and deterministic reproduction traces.

- [ ] **Make Native Engine Coverage Explicit and Verifiable**:
  - Progress: Native PostgreSQL jobs assert the connected server major, and MySQL/MariaDB jobs assert the actual server product and major/minor version. The active CI matrix currently has no PostgreSQL-compatible extension images; extension-specific checks remain applicable if those targets return. Native dialect-feature coverage and actual SQLite runtime-boundary tests remain outstanding.
  - Label every test result as native-engine execution or dialect simulation/transpilation; local output must make the distinction unmistakable.
  - Treat transpiled SQLite execution as a fast auxiliary test only, never as evidence of native PostgreSQL/MySQL/MariaDB semantics.
  - In CI, query and assert the actual server product/version against the requested matrix entry; fail or explicitly mark a fallback as a different target instead of silently counting it as the requested version.
  - Execute dialect-specific triggers, procedures, conflict handling, transaction behavior, isolation, types, collations, and generated SQL on the actual engine.
  - Test minimum supported versions and feature boundaries, including real SQLite runtime versions rather than only simulated version metadata.

- [ ] **Add Mutation Testing for Test-Suite Sensitivity**:
  - Introduce targeted mutations to trigger predicates/actions, delta signs, aggregate maintenance, NULL handling, group/key handling, watermark comparisons, transaction boundaries, rollback/recovery, and dialect/version branches.
  - Require the relevant tests to kill each curated non-equivalent mutant; record surviving mutants and justify exclusions.
  - Use mutation testing to find blind spots, not as a substitute for semantic test design or as an unqualified global percentage target.

- [ ] **Add Metamorphic and Invariant-Based Tests**:
  - Check insert-then-delete restoration, no-op updates, rollback restoration, rebuild-versus-incremental equivalence, and equivalent mutation orderings where operations commute.
  - Check consistency between base tables, backing tables, auxiliary tables, catalog metadata, and installed triggers after success and failure.
  - Keep these properties independent of the implementation's own delta-generation logic.

- [ ] **Measure Test Strength and Publish Honest Coverage Statistics**:
  - Report distinct schema structures, query AST structures, DML AST structures, grammar-production coverage, operator/clause combinations, mutation sequence lengths, native versus simulated engine executions, and concurrency scenarios.
  - Keep raw test/view/assertion counts as workload statistics only; do not present them as proxies for structural or semantic coverage.
  - Reconcile README, module docstrings, runner output, and CI matrix descriptions from one generated source of truth.
  - Remove brittle workflow-text-count assertions; validate workflow structure semantically where needed and rely on actual CI execution for runtime behavior.

## P1 — Modular Architecture, Engine/Version Optimizations & Developer Experience

- [ ] **Keep User-Accessible Materialized Views Free of the Internal Object Prefix**:
  User-facing views created/exposed by the library should retain clean, natural names and must not inherit the internal `__matview_` prefix used by catalog, backing, delta, and auxiliary objects. Define the naming contract clearly, preserve collision safety and dialect constraints, and add cross-engine tests proving internal objects remain namespaced while the user-accessible view name does not.

- [ ] **AST-Based SQL Code Generation via `sqlglot` (`sqlglot.exp` Builders)**:
  Replace string-interpolated DML/DDL construction (`CREATE TABLE`, `CREATE INDEX`, `CREATE VIEW`, `SELECT`, `INSERT ... ON CONFLICT / ON DUPLICATE KEY UPDATE`, `UPDATE`, `DELETE`) with `sqlglot.exp` AST builders (`sqlglot.select`, `exp.insert`, `exp.update`, `exp.delete`, `exp.Create`, `.sql(dialect=..., identify=..., pretty=True)`) so generated SQL is guaranteed well-formed and dialect-accurate by construction.
- [ ] **Template-Driven Procedural Trigger & Backfill Codegen (`Jinja2`)**:
  Delegate multi-statement procedural trigger and backfill wrappers (SQLite `CREATE TRIGGER ... BEGIN ... END`, PostgreSQL PL/pgSQL `CREATE OR REPLACE FUNCTION ... RETURNS TRIGGER`, and MySQL/MariaDB `CREATE TRIGGER` / `CREATE PROCEDURE` blocks) to structured `Jinja2` templates instead of nested Python f-strings.
- [ ] **Unified Database Connectivity, DB-API Normalization & Schema Introspection (`SQLAlchemy`)**:
  Consider adopting `SQLAlchemy` (`sqlalchemy.create_engine`, `sqlalchemy.inspect`) to unify connection URL handling, parameter binding (`?` vs `%s`), transaction/savepoint management, and cross-engine table/column/PK/FK/index introspection across SQLite, PostgreSQL, MySQL, and MariaDB.
- [ ] **Standardized Engine Version Parsing (`packaging`)**:
  Delegate semantic version parsing and comparison in `parse_engine_version` / `EngineVersion` to `packaging.version.Version`.
- [ ] **Stateful Property-Based Fuzzing & Automatic Counterexample Shrinking (`hypothesis` + `pytest`)**:
  Complement `DifferentialSQLFuzzer` with a `hypothesis.stateful.RuleBasedStateMachine` test suite run via `pytest` so failing random schema/view/DML sequences are automatically shrunk to minimal reproducible counterexamples.

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

## P3 — Cascading MVs, Shared Computation, Async/Bulk MVs, Docs & NewSQL

- [ ] **Cascading Materialized Views (MVs Referencing Other MVs)**:
  I would like materialized view definitions to also be able to reference other materialized views, and their updates to still happen as efficiently as possible (building a topological dependency DAG over views and compiling downstream MVs against upstream `__matview_store_<v1>` backing tables with recursive trigger propagation). Note: already implemented in branch cascading-mvs.
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
