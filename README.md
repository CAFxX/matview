# matview (`matview_sync.py`)

Zero-runtime-dependency incremental materialized view compiler and lifecycle manager for **SQLite**, **PostgreSQL**, **MySQL**, and their compatible database engines (**MariaDB**, **Percona Server**, **TimescaleDB**, **PostGIS**, **pgvector**, **APSW**, **SQLite3 Multiple Ciphers**, **SQLCipher**, **sqlean**, and **Turso libSQL**).

## Highlights

* **Pure Native SQL Execution**: Compiles standard `CREATE VIEW` definitions (single or multi-view `.sql` bundles) into backing tables, user-defined and automatic indexes, read-only user views with write-protection triggers, atomic maintenance triggers, and initial or chunked incremental backfill statements. Once installed, materialized views are maintained entirely by native database triggers with zero Python runtime dependency.

* **Automatic Engine & Version Detection (`detect_engine_version` / `--engine-version`)**: Automatically detects the target database dialect, subdialect (`sqlite`, `postgres`, `mysql`, `mariadb`), and exact `(major, minor, patch)` version from live database connections (`SELECT sqlite_version()`, `SHOW server_version_num` / `SELECT version()`, `SELECT VERSION()`), or accepts explicit version specs (`--engine-version postgres:17`, `--engine-version mariadb:11.4`, `--engine-version mysql:8.4`, `--engine-version sqlite:3.45`) to select version-optimal features such as PostgreSQL 15+ `UNIQUE ... NULLS NOT DISTINCT` indexes, PostgreSQL 10+ transition tables, and SQLite 3.24+ `UPSERT` / 3.25+ window-function checks.

* **Granular** $O(1)$ **Algebraic Delta Maintenance (`ALGEBRAIC_DELTA`)**:

  * Single-table aggregations with arbitrary column or expression `GROUP BY`, `WHERE`, and `HAVING` clauses (`COUNT(*)`, `COUNT(col)`, `SUM`, `TOTAL`, `AVG`, `MIN`, `MAX`, `VAR_POP`, `VAR_SAMP`, `STDDEV_POP`, `STDDEV_SAMP`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `BOOL_AND`, `BOOL_OR`, `FILTER (WHERE ...)`, and compound aggregate arithmetic such as `SUM(a * b) - SUM(c)`).

  * Multi-table `INNER JOIN` and `CROSS JOIN` aggregations via per-table delta queries.

  * $\mathbb{Z}$-set `UNION`, `UNION ALL`, `INTERSECT`, and `EXCEPT` set operations.

  * `SELECT DISTINCT` and primary-key lineage projections (including automatic `SELECT *` expansion and native composite primary keys).

  * Aggregations over `UNION ALL` CTEs and derived tables.

* **Reference-Counted Distinct & Holistic Fallbacks**: Automatic auxiliary reference-counting tables for `COUNT(DISTINCT ...)`, `SUM(DISTINCT ...)`, and `AVG(DISTINCT ...)`, plus group-scoped or atomic refresh for window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`), `LEFT JOIN`, and recursive CTEs (or fail-fast Tier 1 enforcement via `--strict` / `strict=True`).

* **Live Database Schema Introspection & Drift Management**: Accepts either raw SQL DDL strings or a live PEP-249 database connection (`conn`) to automatically introspect existing base tables, column types, nullability, primary keys, and `ON DELETE CASCADE` foreign keys across SQLite, PostgreSQL, and MySQL/MariaDB, with built-in schema drift detection (`--check-drift` / `mgr.check_schema_drift()`) and atomic reconciliation (`--repair` / `mgr.repair_view()`).

* **Extensive Multi-Order-of-Magnitude Differential Fuzzer & Regression Corpus**: Features `DifferentialSQLFuzzer` (Suite 12) generating 5-table schemas (single PK, 2-column & 3-column composite PKs, self-referential hierarchies, multi-hop FKs), up to 80 views per schema across all 7 SQL families, and 45–80+ DML & 3-level nested `SAVEPOINT` transaction steps across both deterministic seeds and fresh cryptographic random 32-bit seeds (`secrets.randbits(32)`), backed by a permanent regression corpus (`Suite 13` + `test_10_fuzz_regression_corpus_invariants`).

* **Safety & Transparency Checks**: Refuses to install views on engine/schema combinations where transparency cannot be guaranteed for all mutations (such as MySQL `TRUNCATE TABLE` or `ON DELETE CASCADE` trigger limitations) unless `--ignore-limitations` (`ignore_limitations=True`) is explicitly passed.

* **Trigger-First Ordering & Dialect-Native Incremental Backfill**: Installs maintenance triggers before initial backfill so no concurrent base-table write is ever missed, and supports `--incremental-backfill --batch-size N` via recursive CTE chunking (SQLite), `DO $$ ... END $$;` blocks (PostgreSQL), and stored procedures (MySQL/MariaDB).

## Quick Start

```bash
# Generate a standalone SQL installation script targeting a specific engine & version
python3 matview_sync.py \
  --schema schema.sql \
  --view view.sql \
  --engine-version postgres:17 \
  --incremental-backfill \
  --batch-size 500

# Connect to a live database (SQLite, PostgreSQL, MySQL, or MariaDB), autodetect engine/version & schema, and apply
python3 matview_sync.py \
  --db postgresql://postgres:postgres@127.0.0.1:5432/app \
  --view view.sql \
  --apply

# Check for out-of-band base table schema drift and repair/reconcile registered views
python3 matview_sync.py --db app.db --check-drift --repair
```

## Running Tests & Differential Fuzzing

```bash
# Run the full 5,107-view (148,838-assertion) test suite across all 13 suites (automatically scales up further when CI=true)
python3 test_matview_sync.py

# Run standalone multi-order-of-magnitude differential fuzzing (Suites 12 & 13, including fresh cryptographic random seeds)
python3 test_matview_sync.py --fuzz

# Scale up differential fuzzing across seeds, schemas, views, and mutations
python3 test_matview_sync.py --fuzz --fuzz-scale 3

# Reproduce any specific seed logged in CI output
MATVIEW_FUZZ_SEEDS=20261106 python3 test_matview_sync.py --fuzz

# Run live integration tests against PostgreSQL, MySQL/MariaDB, or SQLite-compatible engines
POSTGRES_DSN="postgresql://postgres:postgres@127.0.0.1:5432/matview_test" \
MYSQL_DSN="mysql://root:root@127.0.0.1:3306/matview_test" \
SQLITE_COMPAT_ENGINE="apsw" \
python3 test_matview_sync.py
```

## CI Matrix (`.github/workflows/ci.yml`)

Runs 61 parallel jobs across all active non-EOL versions of SQLite-, PostgreSQL-, and MySQL-compatible engines (plus a dedicated 6-job `fuzz-stress` high-intensity random fuzzing matrix):

* **SQLite & SQLite-Compatible**:

  * **SQLite 3 (`sqlite3`)**: Python `3.11`, `3.12`, `3.13`, `3.14`, `3.15`

  * **APSW (`apsw`)**: Python `3.11`, `3.12`, `3.13`, `3.14`

  * **SQLite3 Multiple Ciphers (`apsw-sqlite3mc`)**: Python `3.11`, `3.12`, `3.13`, `3.14`

  * **SQLCipher 4 (`sqlcipher3`)**: Python `3.11`, `3.12`, `3.13`

  * **sqlean (`sqlean.py`)**: Python `3.11`, `3.12`

  * **Turso libSQL (`libsql-experimental`)**: Python `3.11`, `3.12`

* **PostgreSQL & PostgreSQL-Compatible**:

  * **PostgreSQL**: `14`, `15`, `16`, `17`, `18`

  * **Percona Distribution for PostgreSQL**: `14`, `15`, `16`, `17`, `18`

  * **TimescaleDB**: `latest-pg14`, `latest-pg15`, `latest-pg16`, `latest-pg17`, `latest-pg18`

  * **PostGIS 3.6**: `14-3.6`, `15-3.6`, `16-3.6`, `17-3.6`, `18-3.6`

  * **pgvector**: `pg14`, `pg15`, `pg16`, `pg17`, `pg18`

* **MySQL & MySQL-Compatible**:

  * **Oracle MySQL**: `8.4` (LTS), `9.7` (LTS), `26.10` (Innovation)

  * **MariaDB Server**: `10.11` (LTS), `11.4` (LTS), `11.8` (LTS), `12.3` (LTS), `13.0` (Rolling)

  * **Percona Server for MySQL**: `8.4` (LTS), `9.7` (`latest`)

* **Dedicated High-Intensity Differential Fuzzing Matrix (`fuzz-stress`)**:

  * Runs `python3 test_matview_sync.py --fuzz --fuzz-scale 3` (10 deterministic + fresh random seeds $\times$ 4 schemas $\times$ 80 views $\times$ 70 mutations) across **SQLite 3**, **APSW**, **PostgreSQL 17**, **PostgreSQL 18**, **MySQL 8.4 LTS**, and **MariaDB 11.4 LTS**.