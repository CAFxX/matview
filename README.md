# matview (`matview_sync.py`)

Zero-runtime-dependency incremental materialized view compiler and lifecycle manager for **SQLite**, **PostgreSQL**, **MySQL**, and their compatible database engines (**MariaDB**, **Percona Server**, **TimescaleDB**, **PostGIS**, **pgvector**, **APSW**, **SQLite3 Multiple Ciphers**, **SQLCipher**, **sqlean**, and **Turso libSQL**).

## Highlights

* **Pure Native SQL Execution**: Compiles standard `CREATE VIEW` definitions (single or multi-view `.sql` bundles) into backing tables, user-defined and automatic indexes, read-only user views with write-protection triggers, atomic maintenance triggers, and initial or chunked incremental backfill statements. Once installed, materialized views are maintained entirely by native database triggers with zero Python runtime dependency.

* **Granular** $O(1)$ **Algebraic Delta Maintenance (`ALGEBRAIC_DELTA`)**:

  * Single-table aggregations with arbitrary column or expression `GROUP BY`, `WHERE`, and `HAVING` clauses (`COUNT(*)`, `COUNT(col)`, `SUM`, `TOTAL`, `AVG`, `MIN`, `MAX`, `VAR_POP`, `VAR_SAMP`, `STDDEV_POP`, `STDDEV_SAMP`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `BOOL_AND`, `BOOL_OR`, and compound aggregate arithmetic such as `SUM(a * b) - SUM(c)`).

  * Multi-table `INNER JOIN` and `CROSS JOIN` aggregations via per-table delta queries.

  * $\mathbb{Z}$-set `UNION`, `UNION ALL`, `INTERSECT`, and `EXCEPT` set operations.

  * `SELECT DISTINCT` and primary-key lineage projections.

  * Aggregations over `UNION ALL` CTEs and derived tables.

* **Reference-Counted Distinct & Holistic Fallbacks**: Automatic auxiliary reference-counting tables for `COUNT(DISTINCT ...)`, `SUM(DISTINCT ...)`, and `AVG(DISTINCT ...)`, plus atomic statement/row refresh for window functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`), `LEFT JOIN`, and recursive CTEs.

* **Safety & Transparency Checks**: Refuses to install views on engine/schema combinations where transparency cannot be guaranteed for all mutations (such as MySQL `TRUNCATE TABLE` or `ON DELETE CASCADE` trigger limitations) unless `--ignore-limitations` (`ignore_limitations=True`) is explicitly passed.

* **Trigger-First Ordering & Dialect-Native Incremental Backfill**: Installs maintenance triggers before initial backfill so no concurrent base-table write is ever missed, and supports `--incremental-backfill --batch-size N` via recursive CTE chunking (SQLite), `DO $$ ... END $$;` blocks (PostgreSQL), and stored procedures (MySQL/MariaDB).

## Quick Start

```
# Generate a standalone SQL installation script
python3 matview_sync.py \
  --schema schema.sql \
  --view view.sql \
  --dialect postgres \
  --incremental-backfill \
  --batch-size 500







```

## Running Tests

```
# Run the full 6,200-view (52,633-assertion) SQLite + cross-dialect test matrix
python3 test_matview_sync.py

# Also run live integration tests against PostgreSQL, MySQL/MariaDB, or SQLite-compatible engines
POSTGRES_DSN="postgresql://postgres:postgres@127.0.0.1:5432/matview_test" \
MYSQL_DSN="mysql://root:root@127.0.0.1:3306/matview_test" \
SQLITE_COMPAT_ENGINE="apsw" \
python3 test_matview_sync.py







```

## CI Matrix (`.github/workflows/ci.yml`)

Runs 55 parallel jobs across all active non-EOL versions of SQLite-, PostgreSQL-, and MySQL-compatible engines:

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