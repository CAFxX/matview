# matview

**Incremental materialized views for SQLite, PostgreSQL, and MySQL/MariaDB, compiled from ordinary SQL views.**

`matview` turns `CREATE VIEW` definitions into database-native materialized views. It creates the storage tables, indexes, read-only query views, and maintenance triggers, then backfills the initial result. After installation, the database keeps each materialized view up to date as its source tables change—without a background worker or a Python process running alongside the application.

- **Databases:** SQLite, PostgreSQL, MySQL, and MariaDB, plus tested compatible engines (see [Compatibility](#compatibility)).
- **Input:** ordinary `CREATE VIEW ... AS SELECT ...` statements, including multi-view SQL files.
- **Runtime maintenance:** native database triggers; no `matview` service or Python runtime loop.
- **Management:** SQL generation, direct installation, incremental initial backfill, schema-drift checks, and repair.

## How it works

Given a definition such as:

```sql
CREATE VIEW customer_sales AS
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
FROM orders
WHERE status = 'paid'
GROUP BY customer_id;
```

`matview` generates the database objects needed to maintain its result:

1. A backing table containing the materialized rows.
2. Indexes for grouping, lookup, and any supported user-defined indexes.
3. A read-only view exposing the materialized result.
4. Triggers that apply changes when source tables are inserted, updated, or deleted.
5. SQL to populate the initial result, either in one operation or in batches.

**The generated view name is prefixed by default.** The definition above is exposed as `__matview_customer_sales`, not `customer_sales`. The default prefix is `__matview_`; change it with `--object-prefix` or the Python API's `object_prefix` option. The generated view is read-only: write to the source tables, not to the materialized view.

The triggers run as part of the database's normal transaction processing, so maintenance is performed by the database rather than by polling or a separate synchronization job. Python is needed to compile, install, and manage the objects—not for ongoing query or trigger execution.

## Quick start

### 1. Install

Clone the repository and install its dependencies:

```bash
git clone https://github.com/CAFxX/matview.git
cd matview
python3 -m pip install -r requirements.txt
```

The requirements include `sqlglot` for SQL parsing, `psycopg2-binary` for PostgreSQL DSN URLs (on supported Python versions), and `PyMySQL` for MySQL/MariaDB DSN URLs. SQLite uses Python's standard-library `sqlite3` module. You can also pass an existing PEP-249/DB-API connection created with another driver.

### 2. Define the source schema and view

For this example, save the source-table definition as `schema.sql`:

```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount NUMERIC NOT NULL,
    status TEXT NOT NULL
);
```

Create a file such as `views.sql` containing one or more standard `CREATE VIEW` statements:

```sql
CREATE VIEW customer_sales AS
SELECT
    customer_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount
FROM orders
WHERE status = 'paid'
GROUP BY customer_id;
```

When applying against an existing database with `--db`, the source tables must already exist. When generating SQL without connecting to a database, provide their `CREATE TABLE` definitions in a schema file so `matview` can inspect columns, types, keys, and constraints. If you combine `--schema` with `--apply`, the CLI executes that schema DDL against the target database before registering the views.

### 3. Install the materialized view

**Apply directly to a live database** (the engine, version, and schema are detected from the connection):

```bash
sqlite3 app.db < schema.sql  # create the example source table first
python3 matview_sync.py \
  --db app.db \
  --view views.sql \
  --apply
```

For PostgreSQL or MySQL/MariaDB, pass a connection URL instead of `app.db`, for example `postgresql://user:password@host:5432/database` or `mysql://user:password@host:3306/database`.

**Or generate a standalone SQL script** for review, deployment, or execution by your database's normal migration tooling:

```bash
python3 matview_sync.py \
  --schema schema.sql \
  --view views.sql \
  --engine-version postgres:17 \
  --incremental-backfill \
  --batch-size 500 \
  > install.sql
```

Here, `schema.sql` contains the source tables' `CREATE TABLE` statements and is used to infer their schema; in this generation-only command it does not create the source tables. `--engine-version` selects the target dialect and version so the compiler can use version-specific features. Use `--dialect sqlite`, `--dialect postgres`, or `--dialect mysql` when the dialect is known but an exact version is not.

Without `--apply`, the command prints the generated SQL script to standard output. Review it and execute it with your database's SQL client or migration system—for example, `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f install.sql` for PostgreSQL. With `--apply`, `matview` connects to `--db` and installs the view directly.

### 4. Query the result

With the default object prefix, query the generated view:

```sql
SELECT customer_id, order_count, total_amount
FROM __matview_customer_sales;
```

Subsequent changes to `orders` are reflected by the installed triggers.

## Maintenance strategies and performance

The compiler chooses a maintenance strategy based on the query shape:

| Strategy | What it does | When it is useful |
| --- | --- | --- |
| `ALGEBRAIC_DELTA` | Applies incremental changes derived from the affected source rows instead of rebuilding the complete result. | Supported aggregations, joins, set operations, distinct projections, and related query shapes. |
| `GROUP_SCOPED_DELTA` | Recomputes the affected group rather than rebuilding every group. | Holistic aggregates and grouped queries (including supported `LEFT JOIN` cases) that cannot use the algebraic strategy. |
| `ATOMIC_QUERY_SYNC` | Uses an atomic query-based synchronization fallback. | Window functions, recursive CTEs, `ORDER BY ... LIMIT`, correlated subqueries, and other query shapes that need a broader refresh strategy. |

The algebraic strategy is designed to avoid work proportional to the full source-table size for the supported cases. This is an algorithmic description, **not** a guarantee of constant wall-clock time: indexes, join fan-out, group size, database execution plans, and transaction costs still matter. Not every valid SQL query can use the algebraic strategy.

Use `--strict` when you want compilation to fail unless a view can use `ALGEBRAIC_DELTA`:

```bash
python3 matview_sync.py --db app.db --view views.sql --apply --strict
```

Without `--strict`, the compiler can select an available fallback strategy. The selected strategy is also available from the Python API as `plan.strategy`.

## Supported query features

The supported query shapes include:

- **Algebraic aggregations:** single-table `GROUP BY` on columns or expressions, with `WHERE` and `HAVING`; `COUNT(*)`, `COUNT(column)`, `SUM`, `TOTAL`, `AVG`, `MIN`, `MAX`, variance and standard-deviation aggregates, bitwise and boolean aggregates, and aggregate `FILTER (WHERE ...)` clauses.
- **Algebraic joins and set operations:** grouped aggregations over `INNER JOIN` and `CROSS JOIN`; `UNION`, `UNION ALL`, `INTERSECT`, and `EXCEPT`.
- **Distinct results and projections:** `SELECT DISTINCT`, primary-key lineage projections, and composite primary keys.
- **CTEs and derived tables:** supported aggregations over `UNION ALL` CTEs and derived tables.
- **Distinct aggregates:** `COUNT(DISTINCT ...)`, `SUM(DISTINCT ...)`, and `AVG(DISTINCT ...)`, maintained with auxiliary reference-counting tables.
- **Group-scoped fallback:** holistic aggregates such as `GROUP_CONCAT`, `STRING_AGG`, `JSON_GROUP_ARRAY`, and `ARRAY_AGG`, plus supported grouped `LEFT JOIN` queries.
- **Atomic-query fallback:** window functions such as `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, and `LEAD`; recursive CTEs; `ORDER BY ... LIMIT`; and correlated subqueries.

This is a summary, not a promise that every syntactic variation is supported. SQL parsing and compilation are validated against the target dialect and schema; unsupported or ambiguous cases can fail compilation. Use `--strict` to additionally require the algebraic strategy. For engine-specific behavior, consult the implementation and test suite.

## Installation and management options

| Option | Purpose |
| --- | --- |
| `--db PATH_OR_DSN` | Connect to a live database and introspect its schema. Accepts a SQLite path or SQLite, PostgreSQL, MySQL, or MariaDB URL. |
| `--schema FILE` | Read source-table DDL when compiling without live schema introspection. |
| `--view FILE_OR_SQL` | Read one or more `CREATE VIEW` statements (or provide them inline). |
| `--dialect DIALECT` | Select `sqlite`, `postgres`, `mysql`, or `mariadb`. |
| `--engine-version ENGINE:VERSION` | Specify an exact target, such as `postgres:17`, `mysql:8.4`, `mariadb:11.4`, or `sqlite:3.45`. |
| `--pg-transition-tables` | Use PostgreSQL statement-level transition-table triggers where supported. |
| `--apply` | Install directly into the database specified by `--db`. |
| `--incremental-backfill` | Populate the initial result in batches instead of one large operation. |
| `--batch-size N` | Set the batch size for incremental backfill (default: `500`). |
| `--no-populate` | Install the objects without running the initial population. Use only when you intend to populate the view separately. |
| `--index SPEC` | Add a user-defined index; the option can be repeated. `CREATE INDEX` statements can also be included in the view SQL input. |
| `--object-prefix PREFIX` | Change the prefix used for generated database objects (default: `__matview_`). |
| `--strict` | Fail unless the view can use `ALGEBRAIC_DELTA`. |
| `--ignore-limitations` | Continue despite a detected engine limitation that prevents a full transparency guarantee. See [Safety and limitations](#safety-and-limitations). |
| `--check-drift` | Compare registered views with the current live base-table schema. Requires `--db`. |
| `--repair` | Reconcile or rebuild registered views. Requires `--db`. |

For the complete current CLI help, run:

```bash
python3 matview_sync.py --help
```

### Schema drift checks and repair

If source-table DDL changes outside `matview`, check registered views for drift:

```bash
python3 matview_sync.py --db app.db --check-drift
```

Repair or reconcile the registered views:

```bash
python3 matview_sync.py --db app.db --repair
```

You can also combine the operations:

```bash
python3 matview_sync.py --db app.db --check-drift --repair
```

The manager records installed view definitions and metadata in `__matview_catalog`. Its Python API also exposes `check_schema_drift()` and `repair_view()`.

## Python API

Use `MaterializedViewManager` when you want installation and lifecycle management inside an application or migration script:

```python
import sqlite3

from matview_sync import MaterializedViewManager

conn = sqlite3.connect("app.db")
manager = MaterializedViewManager(conn)

plan = manager.register_view(
    """
    CREATE VIEW customer_sales AS
    SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
    FROM orders
    WHERE status = 'paid'
    GROUP BY customer_id
    """,
    incremental_backfill=True,
    batch_size=500,
)

print(plan.view_name)        # __matview_customer_sales (default prefix)
print(plan.strategy.value)     # selected maintenance strategy

conn.close()
```

The manager also provides:

- `sync_views(...)` to reconcile a declared set of view definitions with the database; optionally use `prune_removed=True` to drop registered views no longer declared.
- `drop_view(name)` to remove a registered view and its generated objects.
- `bulk_changes(...)` to bypass per-row trigger maintenance during high-volume mutations, then repopulate the selected materialized views on exit; optional shadow-table swapping is available.
- `check_schema_drift()` and `repair_view()` for schema-drift management.
- `apply_schema_ddl(...)` to apply DDL and synchronize registered views when the source schema changes.

`MaterializedViewManager` accepts SQLite paths/connections and PostgreSQL or MySQL/MariaDB PEP-249 connections. For PostgreSQL and MySQL/MariaDB, DSN URL handling uses the drivers installed by `requirements.txt`; pass an existing DB-API connection to use a different driver.

## Compatibility

The compiler targets three SQL dialect families:

| Dialect | Engines and compatible implementations covered by the project |
| --- | --- |
| SQLite | SQLite 3, APSW, SQLite3 Multiple Ciphers, SQLCipher, sqlean, and Turso libSQL. |
| PostgreSQL | PostgreSQL, Percona Distribution for PostgreSQL, TimescaleDB, PostGIS, and pgvector. |
| MySQL | Oracle MySQL, MariaDB, and Percona Server for MySQL. |

The [CI workflow](.github/workflows/ci.yml) is the authoritative source for the exact engine and Python-version matrix currently exercised. Compatibility can still depend on engine version, schema features, and supported trigger semantics; a compatible product name does not imply that every feature is identical across engines.

## Safety and limitations

Correct incremental maintenance depends on the database firing the expected triggers for every relevant source-table mutation. Some engines or schema configurations have operations that bypass or limit trigger behavior. For example, MySQL `TRUNCATE TABLE` and certain `ON DELETE CASCADE` configurations can prevent the tool from guaranteeing transparent maintenance.

By default, `matview` checks for known transparency limitations and refuses installation when it cannot guarantee correctness. `--ignore-limitations` (or `ignore_limitations=True` in the API) explicitly overrides that protection and emits a warning. Do not use this option unless you understand the detected limitation and have another way to preserve consistency.

For large initial populations, `--incremental-backfill` generates dialect-specific batched backfill logic. The maintenance triggers are installed before backfill, with additional snapshot/watermark handling for chunked population, so concurrent source-table writes are accounted for during the initial load.

Materialized views are read-only. Make application writes against the source tables. Also review generated SQL and test the target engine/version before deploying schema or view changes to production.

## Tests and differential fuzzing

Run the full test suite:

```bash
python3 test_matview_sync.py
```

Run the differential fuzzer and regression corpus:

```bash
python3 test_matview_sync.py --fuzz
```

Increase the fuzzing workload:

```bash
python3 test_matview_sync.py --fuzz --fuzz-scale 3
```

Reproduce a logged seed:

```bash
MATVIEW_FUZZ_SEEDS=20261106 python3 test_matview_sync.py --fuzz
```

Optional live integration tests can be enabled with database connection settings:

```bash
POSTGRES_DSN="postgresql://postgres:postgres@127.0.0.1:5432/matview_test" \
MYSQL_DSN="mysql://root:root@127.0.0.1:3306/matview_test" \
SQLITE_COMPAT_ENGINE="apsw" \
python3 test_matview_sync.py
```

The CI workflow runs a broader engine/version matrix and a dedicated high-intensity fuzzing job. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the exact current coverage.

## Roadmap

See [`TODO.md`](TODO.md) for planned work, completed items, and known follow-ups.
