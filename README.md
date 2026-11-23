# matview

`matview` turns ordinary SQL `CREATE VIEW` definitions into incrementally
maintained materialized views for SQLite, PostgreSQL and MySQL/MariaDB. It
generates a backing table, a read-only reader view with the declared name, and
maintenance triggers on the source tables. After installation the database
keeps the materialized result current inside each writing transaction; no
background worker or Python process runs alongside the application.

Status: under active development. There is no packaged release, and some known
correctness gaps are still open (see [Known limitations](#known-limitations)).

- [The guarantee](#the-guarantee)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start: Python API](#quick-start-python-api)
- [Quick start: command line](#quick-start-command-line)
- [Supported SQL and refusals](#supported-sql-and-refusals)
- [Engine support](#engine-support)
- [Operations guide](#operations-guide)
- [Testing and contributing](#testing-and-contributing)
- [Known limitations](#known-limitations)
- [License](#license)

## The guarantee

For every registered view, reading the materialized view must return exactly
what the original `SELECT` returns over the current contents of the source
tables: same rows, same multiplicities, same values. There are no tolerances;
any difference is a bug. When `matview` can detect that it cannot keep this
guarantee for a view on a given engine, it refuses to install it (see
[Supported SQL and refusals](#supported-sql-and-refusals)) unless you opt in
explicitly.

The guarantee is the design target and what the tests check. It is not yet met
everywhere: the open divergences known today are listed under
[Known limitations](#known-limitations).

## How it works

Registering a view `customer_sales` creates:

| Object | Name (default prefix `__matview_`) | Purpose |
| --- | --- | --- |
| Reader view | `customer_sales` | What applications query. Materialized views retain their declared logical SQL name. Writes to it are rejected. |
| Backing table | `__matview_store_customer_sales` | Stores the materialized rows. |
| Maintenance triggers | `__matview_trg_customer_sales__<table>_ai` / `_ad` / `_au` | Apply the effect of each `INSERT` / `DELETE` / `UPDATE` on a source table (on PostgreSQL through generated trigger functions). |
| Auxiliary tables and indexes | `__matview_...` | Reference counts and lookups needed by some query shapes. |
| Catalog | `__matview_catalog` | Registered definitions, generated object names and backfill state. |

All internal objects use the prefix (configurable with `object_prefix=` /
`--object-prefix`); the reader view does not. The triggers run inside the
transaction that modifies the source table, so a committed write and its
effect on the materialized view are committed together, and a rollback undoes
both.

The compiler picks one of three maintenance strategies from the query shape.
`plan.strategy` and `plan.diagnostics` report the choice and the reason for any
fallback; `strict=True` (`--strict`) refuses any view that does not get the
first one.

| Strategy (`plan.strategy.value`) | Work per source-row change | Used for |
| --- | --- | --- |
| `algebraic_delta` (tier 1) | Applies the row's delta to the affected result row(s). | Single-table `GROUP BY` aggregates (`COUNT`, `SUM`, `TOTAL`, `AVG`, `MIN`, `MAX`, `VAR_*`, `STDDEV_*`, `BIT_*`, `BOOL_*`, `COUNT/SUM/AVG(DISTINCT ...)`, `FILTER (WHERE ...)`, `HAVING`); aggregates over `INNER`/`CROSS JOIN`s; `UNION`, `INTERSECT`, `EXCEPT`; `SELECT DISTINCT`; projections of a table with a primary key; aggregates over `UNION ALL` and simple CTEs/derived tables. |
| `group_scoped_delta` (tier 2) | Recomputes the affected groups (or window partitions) from the source tables. | Grouped queries with holistic aggregates (`GROUP_CONCAT`, `STRING_AGG`, `ARRAY_AGG`, `JSON_GROUP_ARRAY`, ...) or `LEFT JOIN`; window functions with `PARTITION BY`. |
| `atomic_query_sync` (tier 3) | Recomputes the whole result. | Everything else that compiles: window functions without `PARTITION BY`, `ORDER BY ... LIMIT`, subqueries in `WHERE`/`SELECT`, other CTEs. |

Some tier-1 aggregates still read the source table for the affected group:
`MIN`/`MAX`/`BIT_AND`/`BIT_OR` when a value is removed, and on SQLite
`SUM`/`TOTAL`/`AVG` over non-integer values (see
[Floating-point aggregates](#floating-point-aggregates)).
For `MIN(col)`/`MAX(col)` over a plain column of a single-table view grouped by
plain columns, `recommended_base_indexes` names `(group columns..., col)`: with
it, removing a group's current extreme is one index probe instead of a scan of
the group (measured per extreme delete, net of the bare DELETE: SQLite 1967 ->
3 us with `(g)` vs `(g, v)`, PostgreSQL 1072 -> 519, MySQL 8.4 17060 -> ~0,
MariaDB 11.8 7765 -> 311). Without it the plan diagnostics (and `diff` text)
carry a `minmax_extreme_recompute` performance note with the `CREATE INDEX`
statement. Expression and join arguments keep the group-key index. No
`(group, arg)` index is recommended (the note says why) where creating or
keeping it could fail or change results: on PostgreSQL for argument types whose
values can exceed the 2704-byte btree row limit (`TEXT`, unbounded or long
`VARCHAR`, `BYTEA`, `JSONB`, unconstrained `NUMERIC`, arrays); on MySQL/MariaDB
for character arguments (they compare under a collation) and for keys longer
than InnoDB's 3072 bytes.

Known limitation: `MIN`/`MAX` of text values that compare equal under the
column's collation but differ (`'a'` and `'a '` under a PAD SPACE collation,
`'b'` and `'B'` under a case-insensitive one) may return a different one of
those values than the live query, whose choice depends on its plan (adding an
index can change it).

## Installation

There is no PyPI package or `pyproject.toml` yet. Use a checkout:

```sh no-run
git clone https://github.com/CAFxX/matview.git
cd matview
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD"      # makes `import matview` work from anywhere
```

[`requirements.txt`](requirements.txt) installs `sqlglot` (SQL parsing),
`SQLAlchemy` (opening database URLs and introspection), `jinja2`, `packaging`,
`PyMySQL` (for `mysql://` / `mariadb://` URLs), `psycopg2-binary` (for
`postgresql://` URLs; only on Python versions below 3.15) and the test
dependencies `hypothesis` and `pytest`. SQLite uses the standard-library
`sqlite3` module. The Python versions exercised in CI are listed in the
[CI matrix](#ci-matrix).

The command-line interface is `python3 matview_sync.py` from the checkout.
`matview_sync.py` is a compatibility facade over the `matview` package; new
code should `import matview`.

## Quick start: Python API

```python run
import sqlite3
from matview import MaterializedViewManager

conn = sqlite3.connect(":memory:")
conn.executescript("""
    CREATE TABLE orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        amount      INTEGER NOT NULL,
        status      TEXT NOT NULL
    );
    INSERT INTO orders VALUES (1, 10, 100, 'paid'), (2, 10, 50, 'paid'), (3, 20, 70, 'open');
""")

manager = MaterializedViewManager(conn)
plan = manager.register_view("""
    CREATE VIEW customer_sales AS
    SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
    FROM orders
    WHERE status = 'paid'
    GROUP BY customer_id
""")
print(plan.view_name, plan.strategy.value, plan.backing_table_name)

def show():
    print(conn.execute("SELECT * FROM customer_sales ORDER BY customer_id").fetchall())

show()
conn.execute("UPDATE orders SET status = 'paid' WHERE id = 3")
conn.execute("DELETE FROM orders WHERE id = 2")
conn.commit()
show()

# The materialized result equals the original query over the current rows.
live = conn.execute("""
    SELECT customer_id, COUNT(*), SUM(amount) FROM orders
    WHERE status = 'paid' GROUP BY customer_id ORDER BY customer_id
""").fetchall()
print(live == conn.execute("SELECT * FROM customer_sales ORDER BY customer_id").fetchall())
print(plan.recommended_base_indexes)
```

```text output
customer_sales algebraic_delta __matview_store_customer_sales
[(10, 2, 150)]
[(10, 1, 100), (20, 1, 70)]
True
['CREATE INDEX __matview_base_idx_orders__customer_id ON orders (customer_id);']
```

`MaterializedViewManager` also accepts a database URL (`sqlite:///app.db`, a
SQLite file path, `postgresql://user:pass@host/db`, `mysql://...`,
`mariadb://...`) or an existing DB-API connection; see
[Transactions and drivers](#transactions-and-drivers) for which drivers work.
Other public methods: `sync_views()` (declarative reconciliation of a set of
definitions, optionally `prune_removed=True`), `drop_view()`, `diff_views()`,
`check_schema_drift()`, `repair_view()`, `apply_schema_ddl()`,
`backfill_incrementally()`, `bulk_changes()`, `get_catalog()` and the static
`configure_sqlite_connection()`. To generate SQL without a connection, use
`generate_materialized_view(schemas, view_sql, dialect=...)` with
`parse_table_schema(create_table_sql)` and `plan.render_sql_script()`.

## Quick start: command line

Given a schema file and a view file:

```sql file=schema.sql
CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      INTEGER NOT NULL,
    status      TEXT NOT NULL
);
```

```sql file=views.sql
CREATE VIEW customer_sales AS
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
FROM orders
WHERE status = 'paid'
GROUP BY customer_id;
```

Install into a live database. `--schema` together with `--apply` first executes
the schema DDL against the database (omit it when the tables exist). `--apply`
also prints the generated SQL script; here it is saved to a file:

```sh run
python3 matview_sync.py --db app.db --schema schema.sql --view views.sql --apply > applied.sql
python3 matview_sync.py --db app.db --check-drift
python3 matview_sync.py diff --db app.db --view views.sql
```

```text output
{
  "customer_sales": "ok"
}
=== Matview Diff: 1 view(s) inspected ===

[NOOP] customer_sales
  Strategy: algebraic_delta (Tier 1 (single))
  Selection Reason: Tier 1 algebraic delta maintenance: single-table O(1) incremental aggregates.
  Performance Notes (1):
    * sqlite_sum_total_avg_native_recompute_on_real: a group holding any non-INTEGER value (any value of a REAL-affinity column) is recomputed from the base table on every write (O(group rows) with a group-key index, O(table) without); integer-only groups keep the O(1) exact fold.
  Status: Up to date; no pending changes.

------------------------------------------------------------------------------
Summary: 0 view(s) with pending changes (0 CREATE, 0 REPLACE, 0 ALTER, 0 DROP, 1 NOOP)
```

Or generate a standalone script for review or for your migration tool, without
connecting to a database (the schema file is only parsed):

```sh run
python3 matview_sync.py --schema schema.sql --view views.sql --dialect postgres > install.sql
grep -c "CREATE TRIGGER" install.sql
```

```text output
9
```

(On PostgreSQL: four statement-level maintenance triggers, the read-only
trigger of the reader view, and the same four inside the `DO` block that
installs them on every partition.)

```sh no-run
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f install.sql
```

A generated script does not include the manager's catalog row or its backfill
coordination; prefer `--apply` or the Python API when the database is live
(see [Initial population and backfill](#initial-population-and-backfill)).

Main options (`python3 matview_sync.py --help` lists all):

| Option | Meaning |
| --- | --- |
| `--db PATH_OR_URL` | Live database: SQLite path or `sqlite:///`, `postgresql://`, `mysql://`, `mariadb://` URL. Engine and version are detected. |
| `--schema FILE` | `CREATE TABLE` statements; parsed for offline generation, executed with `--apply`. |
| `--view FILE_OR_SQL` | One or more `CREATE VIEW` statements, optionally with `CREATE INDEX` statements on the view. |
| `--dialect`, `--engine-version` | Target for offline generation, e.g. `postgres` or `mysql:8.4`. |
| `--apply` | Register the views in `--db`. |
| `--incremental-backfill`, `--batch-size N` | Populate in batches (default batch size 500). |
| `--no-populate` | Install without populating. |
| `--index SPEC` | Extra index on the materialized view (repeatable). |
| `--strict` | Refuse views that cannot use `algebraic_delta`. |
| `--ignore-limitations` | Install despite detected limitations (see below). |
| `--check-drift`, `--repair` | Report or repair registered views in `--db`. |
| `diff` (or `--diff`) with `--json`, `--exit-code`, `--prune-removed` | Show what registering `--view` would change, without changing anything. |

Exit codes: 0 success; 1 drift found (`--check-drift` without `--repair`) or
changes pending (`diff --exit-code`); 2 usage, compilation or limitation error;
3 a published view cannot be replaced safely (`UnsafeReplacementError`);
4 the connection's transaction cannot be owned (`TransactionOwnershipError`).
The CLI has no drop command; use `MaterializedViewManager.drop_view()`.

## Supported SQL and refusals

Input is one or more `CREATE VIEW name AS SELECT ...` (or `WITH ...`)
statements in the target dialect, parsed with `sqlglot`. The strategy table in
[How it works](#how-it-works) summarizes which shapes are maintained how; it is
a summary, not a promise that every syntactic variant compiles. A view that
does not compile is rejected with an error; it is never installed in a
degraded form.

Errors raised before anything is installed:

- `ValueError` for invalid input: unknown tables or columns, a non-aggregated
  column missing from `GROUP BY`, a view name that collides with a table, an
  invalid index specification.
- `StrictCompilationError` when `strict=True` and the view would not use
  `algebraic_delta`.
- `UnsupportedMaterializedViewError` (a `ValueError`, also with
  `ignore_limitations=True`) for a function the target engine does not define
  with a known meaning. `VARIANCE`, `STDDEV` and `STD` are bound to each
  engine's own definition: sample statistics on PostgreSQL (`VAR_SAMP`,
  `STDDEV_SAMP`), population statistics on MySQL/MariaDB (`VAR_POP`,
  `STDDEV_POP`). PostgreSQL has no `STD`, and SQLite has none of the three, so
  they are refused there; use `VAR_POP`/`VAR_SAMP`/`STDDEV_POP`/`STDDEV_SAMP`.
  On MariaDB it is also raised for `VAR_*`/`STDDEV_*` results the view cannot
  reproduce exactly: window statistics, and statistics over exact
  (integer/`DECIMAL`) or `FLOAT(M,D)` arguments anywhere except as a whole
  top-level projected aggregate of a single query (also not in `HAVING`; MariaDB
  fixes their decimals and rounds results to them). Some exact shapes are
  refused conservatively too (`SELECT DISTINCT`, `ORDER BY ... LIMIT`, a
  parenthesized `(VARIANCE(x))`). Cast the argument to `DOUBLE` to keep such
  shapes.
- `EngineLimitationError` when the view or the engine has a known limitation
  that would break the guarantee. Each limitation has a code:

| Code | Raised when | Why |
| --- | --- | --- |
| `NONDETERMINISTIC_EXPR` | The view uses `RANDOM()`, `NOW()`, `CURRENT_TIMESTAMP`, `UUID()` and similar. | The value would be frozen at write time. |
| `MYSQL_NO_TRUNCATE_TRIGGER` | Every view on MySQL/MariaDB. | `TRUNCATE TABLE` bypasses triggers there. |
| `MYSQL_FK_CASCADE_NO_TRIGGER` | A source table on MySQL/MariaDB has `ON DELETE`/`ON UPDATE` cascading foreign keys. | InnoDB cascades do not fire triggers. |
| `SQLITE_VERSION_NO_WINDOW_FUNC`, `MYSQL_VERSION_NO_WINDOW_FUNC` | Window functions on an engine version without them. | The query cannot run there. |

```python run
import warnings
from matview import EngineLimitationError, generate_materialized_view, parse_table_schema

schemas = parse_table_schema(
    "CREATE TABLE events (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, n INTEGER NOT NULL)"
)
view = "CREATE VIEW kind_totals AS SELECT kind, SUM(n) AS total FROM events GROUP BY kind"

for dialect in ("sqlite", "mysql"):
    try:
        plan = generate_materialized_view(schemas, view, dialect=dialect)
        print(dialect, "ok:", plan.strategy.value)
    except EngineLimitationError as exc:
        print(dialect, "refused:", [lim.code for lim in exc.limitations])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plan = generate_materialized_view(schemas, view, dialect="mysql", ignore_limitations=True)
print("mysql with ignore_limitations:", plan.strategy.value, caught[0].category.__name__)
```

```text output
sqlite ok: algebraic_delta
mysql refused: ['MYSQL_NO_TRUNCATE_TRIGGER']
mysql with ignore_limitations: algebraic_delta EngineLimitationWarning
```

`ignore_limitations=True` (`--ignore-limitations`) installs the view anyway and
emits an `EngineLimitationWarning`. The guarantee then holds only as long as the
application never performs the operation the limitation names (for
`MYSQL_NO_TRUNCATE_TRIGGER`: never `TRUNCATE` a source table; use `DELETE`). On
MySQL/MariaDB every view currently needs this opt-in.

Codes marked "per-code opt-in only" weaken the result contract itself, so
`ignore_limitations` does not override them: name each one in
`accept_limitations=[...]` (`register_view`, `sync_views`, `diff_views`,
`generate_materialized_view(s)`, or `MaterializedViewManager(...)` for every
view) or `--accept-limitation CODE` (repeatable). The codes a view was
installed with are stored in its catalog row, bound to that definition (a
different definition under the same name needs its own explicit acceptance;
a stored record that does not match the stored definition accepts nothing);
later `register_view()`,
`sync_views()`, `repair_view()` and `bulk_changes()` reuse them (an explicit
`accept_limitations` replaces them). `repair_view()` never overrides a
per-code limitation that is not stored with the view: if a schema change makes
the definition need one (for example a `DECIMAL` column becomes `DOUBLE`), the
rebuild raises `EngineLimitationError` and `check_schema_drift()` reports the
view as `invalid`.

Some trigger-bypassing operations are not detected and not refused: `INSERT OR
REPLACE`/`REPLACE` on a SQLite connection that was not configured with
[`configure_sqlite_connection()`](#sqlite-writer-connections), and other engine
features that skip triggers (see
[Close Unsafe DML Paths That Bypass Trigger-Based Maintenance](TODO.md)). The
session flag `bulk_changes()` uses on MySQL/MariaDB (`@__matview_skip_triggers`)
suspends maintenance only while the session also holds the per-connection
named lock `__matview_bypass:<connection id>` that `bulk_changes()` takes for
its block; a session that only sets the flag is maintained. This guards
against accidental bypass (a stale flag on a pooled connection, copied code);
it is not a security boundary or access control: any session that takes that
lock as well bypasses maintenance. The
tested per-engine list is in
[DML and trigger interaction support](#dml-and-trigger-interaction-support).

## DML and trigger interaction support

Maintenance runs in triggers on the source tables, so the guarantee depends on
how rows change and on what else runs on those tables. The matrix below is the
tested answer per engine. Each row is a cell of
[`test_native_dml_interactions.py`](test_native_dml_interactions.py): a native
differential test that runs the scenario and compares every view with its
original query after each step (in the open transaction and after commit),
exactly. Cells marked "twin" in the test also run each statement on an
identical table without views and require the same affected-row count,
`RETURNING` rows, generated ids and final contents. The table is generated
from the test (`python3 test_native_dml_interactions.py --print-matrix`) and
the test fails when this copy differs.

- exact: the views stay equal to their queries.
- limited: a documented limitation applies, and the test asserts it (for
  example, out-of-band DDL that `check_schema_drift()` reports and
  `repair_view()` repairs).
- refused: the operation fails closed: registration is refused, or the write
  fails and is rolled back with the views unchanged.
- **WRONG**: the view silently diverges. The test is a strict expected
  failure and the note names the TODO.md item.
- n/a: the engine has no such feature. "(17+)" and similar: the feature
  exists only from that version and older servers skip the cell.

Engines: the PR-gate jobs run the cells against PostgreSQL 15 and 18, MySQL
8.4 and 9.7, MariaDB 10.11, 11.8 and 12.3 and SQLite through the
standard-library `sqlite3` driver. The SQLite compatibility drivers (APSW,
SQLCipher, libSQL) do not run this module. PostgreSQL two-phase commit needs
`max_prepared_transactions > 0`, which the CI service does not set, so that
cell is skipped there (verified on PostgreSQL 18 locally).

<!-- BEGIN GENERATED: dml-matrix -->
| Cell | Interaction | SQLite | PostgreSQL | MySQL | MariaDB | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **A. Row-changing statements** | | | | | | |
| `upsert_do_nothing` | INSERT ... ON CONFLICT DO NOTHING (PK and non-PK unique) / INSERT IGNORE | exact | exact | exact | exact | twin; conflict target on the PK, on the UNIQUE column, and untargeted |
| `upsert_do_update` | ON CONFLICT DO UPDATE with EXCLUDED / ON DUPLICATE KEY UPDATE (row alias, VALUES()) | exact | exact | exact | exact | twin; updates move rows between groups, to and from NULL |
| `upsert_where_false` | ON CONFLICT DO UPDATE ... WHERE false / ON DUPLICATE KEY UPDATE id = id (no-op) | exact | exact | exact | exact | twin; conflicting rows are untouched, others inserted |
| `upsert_where_partial` | ON CONFLICT DO UPDATE ... WHERE <predicate on the existing row> | exact | exact | exact | exact | twin; MySQL/MariaDB express the predicate with IF() in the assignments |
| `upsert_unique_non_pk` | Upsert conflicting on a non-PK UNIQUE column that rewrites the primary key | exact | exact | exact | exact | twin; the projection view carries the key |
| `upsert_duplicate_keys_in_statement` | One upsert statement whose rows repeat the same key | exact | exact | exact | exact | twin; PostgreSQL rejects DO UPDATE touching a row twice (SQLSTATE 21000) and the view is unchanged; DO NOTHING, SQLite and MySQL/MariaDB apply the rows in order |
| `merge_all_branches` | MERGE with WHEN MATCHED [AND] UPDATE / DELETE / DO NOTHING and WHEN NOT MATCHED [AND] INSERT / DO NOTHING | n/a | exact (15+) | n/a | n/a | twin |
| `merge_by_source_returning` | MERGE ... WHEN NOT MATCHED BY SOURCE ... RETURNING merge_action() | n/a | exact (17+) | n/a | n/a | twin (returned rows compared) |
| `sqlite_replace_configured` | INSERT OR REPLACE / REPLACE / UPDATE OR REPLACE deleting PK- and UNIQUE-conflicting rows | exact | n/a | n/a | n/a | writer connection configured with configure_sqlite_connection() (recursive_triggers ON); twin |
| `sqlite_replace_unconfigured_writer` | INSERT OR REPLACE / REPLACE through a writer connection with recursive_triggers OFF (SQLite default) | **WRONG** | n/a | n/a | n/a | the replaced rows' DELETE triggers do not fire, so the view keeps them (documented under SQLite writer connections; nothing detects it); TODO: Close Unsafe DML Paths That Bypass Trigger-Based Maintenance |
| `sqlite_table_on_conflict_replace` | Plain INSERT/UPDATE into a table whose UNIQUE constraint declares ON CONFLICT REPLACE | exact | n/a | n/a | n/a | configured writer connection |
| `sqlite_or_ignore` | INSERT OR IGNORE / UPDATE OR IGNORE (PK, UNIQUE and CHECK conflicts) | exact | n/a | n/a | n/a | twin |
| `sqlite_or_abort_fail` | INSERT/UPDATE OR ABORT (statement rolled back) and OR FAIL (earlier rows of the statement kept) | exact | n/a | n/a | n/a | twin; failures inside an open transaction |
| `sqlite_or_rollback` | INSERT OR ROLLBACK (the whole transaction is rolled back) | exact | n/a | n/a | n/a |  |
| `mysql_replace` | REPLACE deleting PK- and UNIQUE-conflicting rows (VALUES and SELECT forms) | n/a | n/a | exact | exact | twin |
| `mysql_update_ignore_key_shapes` | UPDATE / UPDATE IGNORE on tables without a key and with identical duplicate rows, with only a UNIQUE NOT NULL key, with only a nullable UNIQUE key (NULL twins) and with a composite primary key | n/a | n/a | exact | limited | the MariaDB ignored-update guard needs a key that identifies the row: a table without one cannot skip a row on a duplicate key and gets no guard (an identical twin made a real update look skipped); rows with a NULL in a nullable UNIQUE key are always applied. MariaDB limitation: wrong when the only key is a nullable UNIQUE key and an IGNORE-skipped row has a NULL key (cell mariadb_update_ignore_null_unique_key) |
| `mariadb_update_ignore_null_unique_key` | UPDATE IGNORE skipping rows whose only key is a nullable UNIQUE key and whose key is NULL | n/a | n/a | exact | **WRONG** | rows with a NULL key can be identical twins, so the MariaDB ignored-update guard applies them and the skipped update reaches the view (also with two nullable UNIQUE keys); MySQL fires no trigger for them; TODO: Exact Maintenance Under MariaDB UPDATE IGNORE (AFTER UPDATE Triggers Fire for Ignored Rows) |
| `mysql_insert_ignore_data_errors` | INSERT IGNORE / UPDATE IGNORE converting CHECK, range and duplicate-key errors to warnings | n/a | n/a | exact | exact | twin |
| `mysql_update_ignore_duplicate_key` | UPDATE IGNORE / INSERT IGNORE ... ON DUPLICATE KEY UPDATE whose row updates hit a duplicate key (UNIQUE column or primary key) and are skipped, mixed with rows that succeed | n/a | n/a | exact | exact | twin; MariaDB fires the AFTER UPDATE trigger for a row whose update was skipped (reproduced without matview; the row keeps its OLD values); its maintenance trigger now applies a change only when the row is no longer in its OLD state. MySQL does not fire it |
| `defaults_identity_upsert_gaps` | Column defaults, DEFAULT VALUES, identity/AUTO_INCREMENT keys and the id gaps upserts leave | exact | exact | exact | exact | twin incl. affected rows and lastval()/LAST_INSERT_ID()/last_insert_rowid() after each statement |
| `update_delete_self_join` | UPDATE ... FROM / multi-table UPDATE and DELETE ... USING / multi-table DELETE joining the source to itself | exact | exact | exact | exact | twin |
| `one_statement_both_join_sources` | One statement modifying both sources of a join view (multi-table UPDATE/DELETE; PostgreSQL writable CTE) | n/a | exact | exact | exact | PostgreSQL: BEFORE/AFTER statement bracket detects multi-source overlap and reconciles once via inclusion-exclusion over staged deltas (Stage 3) |
| `pg_copy` | COPY FROM STDIN, and COPY ... (FREEZE) after TRUNCATE in the same transaction | n/a | exact | n/a | n/a | twin |
| `mysql_load_data_local` | LOAD DATA LOCAL INFILE (plain, REPLACE and IGNORE) | n/a | n/a | exact | exact | twin; needs local_infile on the server (the test enables it for its duration) |
| `insert_select_own_source` | INSERT ... SELECT reading the source table itself | exact | exact | exact | exact | twin |
| `dml_reading_the_view` | INSERT/UPDATE/DELETE of a source whose statement reads the materialized view itself | exact | exact | refused | refused | MySQL/MariaDB: the maintenance trigger cannot write the backing table the statement reads (error 1442), the statement fails and is rolled back |
| `returning` | INSERT / UPDATE / DELETE / upsert ... RETURNING | exact (3.35+) | exact | n/a | exact (10.5+) | twin (returned rows compared) |
| `truncate` | TRUNCATE (PostgreSQL: of both join sources in one statement; SQLite: DELETE without WHERE) | exact | exact | refused | refused | MySQL/MariaDB TRUNCATE fires no triggers: registration refuses with MYSQL_NO_TRUNCATE_TRIGGER unless acknowledged (ignore_limitations=True, then TRUNCATE must not be used) |
| **B. Other triggers, objects and settings** | | | | | | |
| `user_before_trigger_rewrites_row` | User BEFORE row trigger on the source that rewrites NEW (also during upserts) | n/a | exact | exact | exact | SQLite cannot assign NEW (see user_after_trigger_writes_same_table) |
| `user_before_trigger_skips_row` | User BEFORE row trigger that suppresses some rows (RETURN NULL / RAISE(IGNORE)) | exact | exact | n/a | n/a | MySQL/MariaDB triggers cannot skip a row |
| `sqlite_benign_user_trigger_after_registration` | A user AFTER trigger created after registration that only writes a table no view reads | exact | n/a | n/a | n/a | fires before the maintenance trigger, which it cannot disturb; check_schema_drift() stays ok |
| `user_after_trigger_writes_same_table` | User AFTER row trigger that updates the same row and inserts/deletes other rows of the source | exact | exact | n/a | n/a | SQLite: Stage 4's same-group in-place UPDATE trigger avoids transiently deleting the group when a newer user AFTER INSERT trigger updates a non-group column on an existing group (still wrong if the user trigger changes a group key or updates an aggregated column of a brand-new group before INSERT maintenance runs — same-row updates before outer AFTER STATEMENT also affect PostgreSQL); check_schema_drift() therefore still reports triggers_out_of_order for such a newer self-writing trigger and repair_view() puts the maintenance triggers first. MySQL/MariaDB forbid a trigger writing its own table (error 1442) |
| `sqlite_recursive_user_trigger` | A user trigger that re-fires itself (unguarded self-UPDATE) | limited | n/a | n/a | n/a | configure_sqlite_connection() turns recursive_triggers ON, so such a trigger recurses until SQLite's depth limit and the statement fails (it ran once with the SQLite default); the view stays exact either way; PostgreSQL user triggers recurse regardless (guard with pg_trigger_depth()) |
| `user_trigger_raises` | A user trigger raising an error mid-statement (RAISE EXCEPTION / SIGNAL / RAISE(ABORT/FAIL/ROLLBACK)) | exact | exact | exact | exact | rollback atomicity of the statement (SQLite FAIL keeps the earlier rows, ROLLBACK ends the transaction) |
| `sqlite_user_trigger_raise_fail` | A user AFTER trigger created after registration raising RAISE(FAIL) on the second row of a statement | limited | n/a | n/a | n/a | FAIL keeps the rows already written, including the failing row itself; while the newer user trigger fires first the failing row's maintenance never runs. check_schema_drift() reports triggers_out_of_order (also for RAISE(IGNORE)) and repair_view() puts the maintenance triggers first |
| `sqlite_user_trigger_writes_source_indirectly` | A newer user trigger whose writes reach the source only through other triggers: it inserts into a table whose trigger updates the source, or into a view whose INSTEAD OF trigger updates the source | limited | n/a | n/a | n/a | check_schema_drift() follows the trigger chain (bounded depth) and reports triggers_out_of_order; repair_view() puts the maintenance triggers first. Writes hidden in application functions are not seen. The check is conservative: a RAISE(FAIL/IGNORE) anywhere in the chain, even on an unrelated table, or a chain deeper than 16 is reported too |
| `sqlite_user_triggers_older_than_views` | The same self-writing and RAISE(FAIL) user triggers created before the views were registered | exact | n/a | n/a | n/a | SQLite then fires them after the maintenance triggers |
| `pg_statement_trigger_ordering` | User statement-level AFTER triggers ordered before and after the maintenance triggers (by name) that write the source again | n/a | exact | n/a | n/a |  |
| `user_trigger_writes_other_source_first` | User trigger on one join source that writes the other source, firing BEFORE the maintenance trigger | limited | exact | limited | limited | PostgreSQL: row/statement trigger modifying the other join source runs inside the Stage 3 statement bracket and reconciles via inclusion-exclusion; SQLite (newest trigger first) and MySQL/MariaDB (PRECEDES) count the new pair twice where maintained algebraically. SQLite: check_schema_drift() reports triggers_out_of_order and repair_view() puts the maintenance triggers first. MySQL/MariaDB: check_schema_drift() reports triggers_out_of_order for the join views before any write, and re-registering them puts their triggers first (repair_view() only repopulates); TODO: Exact Maintenance When a User Trigger or FK Action Runs Before the Maintenance Trigger |
| `mysql_preexisting_user_trigger_writes_other_source` | A user trigger on one join source that writes the other source, created before the view was registered | n/a | n/a | exact | exact | MySQL/MariaDB fire triggers in creation order, so before this fix the older user trigger fired first and the join aggregate counted the pair twice; registration now places the maintenance triggers first (PRECEDES). SQLite fires older triggers last (user_trigger_writes_other_source_last) |
| `user_trigger_writes_other_source_last` | User trigger on one join source that writes the other source, firing AFTER the maintenance trigger | exact | exact | exact | exact | PostgreSQL statement trigger sorting after the maintenance triggers; SQLite trigger created before registration; MySQL/MariaDB FOLLOWS |
| `dml_through_views` | DML through an updatable view and through an INSTEAD OF trigger on a view over the source | exact | exact | exact | exact | MySQL/MariaDB have no INSTEAD OF; their updatable single-table and join views are used |
| `view_as_source` | Registering a materialized view whose FROM names a plain (non-materialized) view | refused | refused | refused | refused | refused at registration: a view has no triggers to maintain it |
| `pg_rules` | Rewrite rules on the source: conditional DO INSTEAD (soft delete) and DO ALSO | n/a | exact | n/a | n/a |  |
| `generated_stored` | Views over a STORED generated column (aggregated, grouped by, filtered) while its inputs change | exact | exact | exact | exact | SQLite fixed here: the maintenance UPDATE trigger was limited to the referenced columns (UPDATE OF), which skipped updates of the generated column's inputs |
| `generated_virtual` | Views over a VIRTUAL generated column while its inputs change | exact | exact (18+) | exact | exact | PostgreSQL 18+ fixed here: statement triggers project non-virtual transition columns explicitly and compute VIRTUAL generated columns from their catalog generation expressions instead of reading them from the transition tables |
| `pg_domains` | Source columns of domain types over BIGINT, VARCHAR(n), TEXT, TEXT with CHECK, TEXT COLLATE "C" and CHAR(n): grouped by, aggregated, projected (incl. domain CHECK violations) | n/a | exact | n/a | n/a | registration used to fail (backing column declared as type DOMAIN; for a domain over TEXT or CHAR(n) the reflected collation "default" and the lost CHAR length) and a domain group key was refused by the canonical group-key function; a domain over an array type still fails (array backing types) |
| `fk_referential_actions` | FK ON DELETE SET NULL / SET DEFAULT / CASCADE and ON UPDATE CASCADE / SET NULL into a view's source | exact | exact | refused | refused | MySQL/MariaDB: InnoDB referential actions fire no triggers, so registration refuses (MYSQL_FK_CASCADE_NO_TRIGGER, also for SET NULL); InnoDB has no SET DEFAULT |
| `fk_action_into_both_join_sources` | An FK action (ON UPDATE CASCADE / ON DELETE SET NULL of the join key) modifying both sources of a join view | **WRONG** | exact | n/a | n/a | PostgreSQL: Stage 3 statement bracket stages both parent and cascaded child deltas and reconciles once via inclusion-exclusion; SQLite runs the cascade before the parent row's maintenance trigger; MySQL/MariaDB refuse as in fk_referential_actions; TODO: Exact Maintenance When a User Trigger or FK Action Runs Before the Maintenance Trigger |
| `fk_cascade_added_after_registration` | A cascading FK added to a source table after the view was registered (out-of-band ALTER TABLE) | n/a | exact | limited | limited | MySQL/MariaDB: InnoDB cascaded deletes fire no triggers, so the view becomes stale, but check_schema_drift() reports invalid (re-evaluating MYSQL_FK_CASCADE_NO_TRIGGER), repair_view() and sync_views() refuse while the cascading FK exists, and dropping the cascading FK + repair_view() restores exactness; SQLite cannot add an FK to an existing table |
| `fk_enforcement_off` | Writes with FK enforcement switched off (SET foreign_key_checks = 0 / PRAGMA foreign_keys = OFF) | exact | n/a | exact | exact | no referential action runs, so only the statement's own rows change; PostgreSQL has no such switch (see pg_session_replication_role) |
| `mariadb_system_versioning` | System-versioned (temporal) source table: UPDATE/DELETE create history rows; DELETE HISTORY | n/a | n/a | n/a | exact | the view and its oracle read the current rows |
| `pg_disable_trigger` | ALTER TABLE ... DISABLE TRIGGER USER/ALL on a source (as pg_restore --disable-triggers does) | n/a | limited | n/a | n/a | writes are not maintained while disabled; check_schema_drift() reports triggers_disabled (added here; it reported ok) and repair_view() rebuilds the views with enabled triggers |
| `pg_session_replication_role` | Writes under session_replication_role = replica (logical-replication apply workers, some restore tools) | n/a | **WRONG** | n/a | n/a | ENABLE (origin) triggers do not fire; not detectable afterwards. A real subscription was not exercised: apply workers use this setting; TODO: Detect PostgreSQL Maintenance Triggers Disabled by `session_replication_role = replica` or `ALTER TABLE ... DISABLE TRIGGER` |
| `mysql_session_skip_flag` | An application session setting @__matview_skip_triggers (the bulk_changes() bypass flag) | n/a | n/a | exact | exact | the triggers skip maintenance only while the session also holds the per-connection named lock '__matview_bypass:<connection id>' that bulk_changes() takes for its block, so the flag alone (a stale pooled connection, copied code) is ignored; deliberately taking that lock remains a privileged escape hatch |
| `ddl_add_unused_column` | Out-of-band ALTER TABLE ADD COLUMN on a source (column unused by the view) | exact | exact | exact | exact | maintenance continues exactly; check_schema_drift() reports drifted (the signature covers every column) |
| `ddl_drop_used_column` | Out-of-band ALTER TABLE DROP COLUMN of a column the view uses | refused | refused | refused | refused | SQLite refuses the DDL (a trigger references the column); PostgreSQL/MySQL/MariaDB accept it, then every INSERT/UPDATE of the source fails in the maintenance trigger and check_schema_drift() reports invalid |
| `ddl_rename_used_column` | Out-of-band ALTER TABLE RENAME COLUMN of a column the view uses | limited | refused | refused | refused | SQLite rewrites the trigger bodies, so maintenance continues on the renamed column while the registered definition no longer compiles (check_schema_drift() reports invalid); PostgreSQL/MySQL/MariaDB maintenance fails on every write, so writes fail closed |
| `ddl_drop_recreate_source` | Out-of-band DROP TABLE and re-CREATE of a source, then writes | limited | limited | limited | limited | the new table has no maintenance triggers, so the view is stale; check_schema_drift() reports it and repair_view() (MySQL/MariaDB: drop_view() + register_view(), they refuse a non-atomic rebuild) restores exactness |
| **C. Transactions** | | | | | | |
| `savepoints_with_upserts` | Upserts and REPLACE inside nested savepoints, ROLLBACK TO and RELEASE | exact | exact | exact | exact | twin |
| `deferred_constraints` | Deferred UNIQUE/FK constraints failing at COMMIT or at SET CONSTRAINTS ... IMMEDIATE | exact | exact | n/a | n/a | SQLite: a failed COMMIT keeps the transaction open (rolled back by the test); MySQL/MariaDB have no deferred constraints |
| `autocommit_writers` | Writes from an autocommit connection (each statement its own transaction), including a failing statement | exact | exact | exact | exact |  |
| `two_phase_commit` | Two-phase commit: PREPARE TRANSACTION / COMMIT/ROLLBACK PREPARED; XA START ... XA COMMIT/ROLLBACK | n/a | exact (verified locally, not in CI) | exact | exact | PostgreSQL needs max_prepared_transactions > 0; the CI service runs with the default 0, so the cell is skipped there (verified locally on 18) |
| `statement_fails_midway` | An upsert failing on its third row (CHECK) inside a multi-statement transaction, then more statements | exact | exact | exact | exact | twin; PostgreSQL continues after ROLLBACK TO SAVEPOINT, the others after the statement-level rollback |

Counts per product (exact, limited, refused, WRONG, n/a): sqlite 33/6/2/2/21; postgres 37/2/3/1/21; mysql 30/3/6/0/25; mariadb 30/4/6/1/23.
<!-- END GENERATED: dml-matrix -->

Not covered by the matrix: a real logical-replication subscription (its apply
workers write with `session_replication_role = replica`, the cell's setting),
MySQL/MariaDB replication appliers, running `pg_restore` itself (cell
`pg_disable_trigger` uses the `ALTER TABLE ... DISABLE TRIGGER` it issues),
server-side `LOAD DATA INFILE`, binary `COPY`, concurrent sessions (see
[Concurrency notes](#concurrency-notes)) and PostgreSQL partitions and
inheritance (`test_pg_statement_maintenance.py`).

## Engine support

Support levels used here:

- Natively tested on every PR: the full deterministic suite, the fixed-seed
  differential fuzz campaign (seeds 20261106 and 1337, two scenarios each), the
  permanent fuzz regression corpus and the engine-specific test modules run
  against the real server or library in the PR gate, and each server job checks
  the actual product and version before testing.
- Random-seed fuzzing after merge: the extended workflow runs the larger
  scale-2 fuzz campaign with fresh random seeds against SQLite, PostgreSQL 18,
  MySQL 8.4 and MariaDB 12.3 on every push to master, nightly and on demand. It
  is not a merge gate; a failure opens or updates a GitHub issue that lists the
  failing seeds with replay commands.
- Natively tested nightly: the deferred sweep runs the main suite
  (`test_matview_sync.py`) against these targets every night (scale-3 fuzz
  campaigns weekly). It is not a merge gate and can be red; a failing scheduled
  run opens or updates one issue labelled `ci-deferred`. Check that issue and
  the latest run in the repository's Actions tab before relying on it.
- Transpiled: in `test_matview_sync.py`, the PostgreSQL and MySQL code paths are
  also exercised against SQLite through `TranspiledConnectionAdapter` (sqlglot
  transpilation). These are auxiliary checks, not evidence of native engine
  semantics. The adapter also stubs the MySQL session-bypass lock
  (`GET_LOCK`/`RELEASE_LOCK`) and the trigger-order (`ACTION_ORDER`) lookup,
  so offline runs prove nothing about them; the native cells of
  `test_native_dml_interactions.py` cover them.
- Best-effort: SQLite3 Multiple Ciphers and sqlean run the main suite nightly as
  SQLite builds; cipher and extension functionality is not tested. The sqlean
  lane is allowed to fail: sqlean's own `VAR_*`/`STDDEV_*` functions differ
  from matview's SQLite results (a single-value sample statistic is `0.0` in
  sqlean and NULL in the MV; rounding also differs), see
  [sqlean Native Statistical Aggregates (MVs Diverge From sqlean's Own VAR/STDDEV)](TODO.md). Products not
  listed in the matrix (for example other PostgreSQL or MySQL forks) are
  untested.
- Not supported: TimescaleDB, PostGIS and pgvector are excluded from CI until
  tests exercise their extensions; NewSQL engines (CockroachDB, TiDB,
  YugabyteDB, Spanner) are not supported. Drivers: see
  [Transactions and drivers](#transactions-and-drivers).

### CI matrix

Generated from the workflow files and checked by `test_readme_examples.py`.
Server images are tags of a major/LTS line, so each run uses that line's
current patch release. All jobs run on GitHub-hosted `ubuntu-latest` x64; other
operating systems and architectures are not tested.

<!-- BEGIN GENERATED: ci-matrix -->
PR gate ([`ci.yml`](.github/workflows/ci.yml); required, every pull request; fixed fuzz seeds only):

| CI job | Python | Database |
| --- | --- | --- |
| SQLite 3 (Python 3.11) | 3.11 | SQLite bundled with CPython (`sqlite3`) |
| SQLite 3 (Python 3.15) | 3.15 | SQLite bundled with CPython (`sqlite3`) |
| APSW SQLite | 3.12 | SQLite via `apsw` |
| SQLCipher 4 | 3.12 | SQLite via `sqlcipher3-binary` |
| Turso libSQL | 3.12 | SQLite via `libsql-experimental` |
| PostgreSQL 15 | 3.12 | `postgres:15` |
| PostgreSQL 18 | 3.12 | `postgres:18` |
| MySQL 8.4 | 3.12 | `mysql:8.4` |
| MySQL 9.7 | 3.12 | `mysql:9.7` |
| MariaDB 10.11 LTS | 3.12 | `mariadb:10.11` |
| MariaDB 11.8 LTS | 3.12 | `mariadb:11.8` |
| MariaDB 12.3 LTS | 3.12 | `mariadb:12.3` |

Extended random-seed fuzz ([`ci-extended.yml`](.github/workflows/ci-extended.yml); every push to master, nightly and on demand; not a merge gate, failures open an issue):

| CI job | Python | Database |
| --- | --- | --- |
| Random-seed fuzz / SQLite 3 (Python 3.12) | 3.12 | SQLite bundled with CPython (`sqlite3`) |
| Random-seed fuzz / PostgreSQL 18 | 3.12 | `postgres:18` |
| Random-seed fuzz / MySQL 8.4 | 3.12 | `mysql:8.4` |
| Random-seed fuzz / MariaDB 12.3 LTS | 3.12 | `mariadb:12.3` |

Deferred compatibility sweep ([`ci-deferred.yml`](.github/workflows/ci-deferred.yml); nightly, scale-3 fuzz weekly; not a merge gate, scheduled failures open an issue):

| CI job | Python | Database |
| --- | --- | --- |
| CPython 3.13 / SQLite | 3.13 | SQLite bundled with CPython (`sqlite3`) |
| CPython 3.14 / SQLite | 3.14 | SQLite bundled with CPython (`sqlite3`) |
| SQLite3 Multiple Ciphers | 3.12 | SQLite via `sqlite3mc` |
| sqlean SQLite | 3.12 | SQLite via `sqlean` |
| PostgreSQL 16 | 3.12 | `postgres:16` |
| PostgreSQL 17 | 3.12 | `postgres:17` |
| Percona PostgreSQL 18 | 3.12 | `percona/percona-distribution-postgresql:18` |
| MySQL 26.10 Innovation | 3.12 | `mysql:26.10` |
| MariaDB 11.4 LTS | 3.12 | `mariadb:11.4` |
| MariaDB 13.0 Rolling | 3.12 | `mariadb:13.0` |
| Percona Server for MySQL 8.4 | 3.12 | `percona/percona-server:8.4` |
| Percona Server for MySQL 9.7 | 3.12 | `percona/percona-server:9.7` |
| Minimum declared dependencies / PostgreSQL 18 | 3.11 | `postgres:18` |
| Minimum declared dependencies / MySQL 9.7 | 3.11 | `mysql:9.7` |
| Scale-3 differential fuzz / SQLite 3 + transpiled PostgreSQL/MySQL | 3.12 | SQLite bundled with CPython (`sqlite3`) |
| Scale-3 differential fuzz / APSW SQLite | 3.12 | SQLite via `apsw` |
| Scale-3 differential fuzz / PostgreSQL 17 | 3.12 | `postgres:17` |
| Scale-3 differential fuzz / PostgreSQL 18 | 3.12 | `postgres:18` |
| Scale-3 differential fuzz / MySQL 8.4 | 3.12 | `mysql:8.4` |
| Scale-3 differential fuzz / MariaDB 11.4 | 3.12 | `mariadb:11.4` |
<!-- END GENERATED: ci-matrix -->

The SQLite library version in the CPython jobs is whatever the `setup-python`
build bundles; each job prints it. Which test modules run in which job is
defined only in the workflow files.

## Operations guide

### Transactions and drivers

Lifecycle operations own their transaction or refuse. Construction (catalog
initialization), `register_view`, `sync_views`, `drop_view`,
`apply_schema_ddl`, `repair_view` and `backfill_incrementally` execute DDL and
commit their own transactions. Before executing any SQL they raise
`TransactionOwnershipError` if the connection already has an open transaction
(commit or roll back your own work first; on MySQL/MariaDB their DDL would
otherwise commit it implicitly) or if the driver does not report its
transaction state. Read-only methods (`get_catalog`, `fetch_schema`,
`check_schema_drift`, `diff_views`) join an open caller transaction without
ending it. `bulk_changes()` follows the same contract (refusal before any SQL,
then one manager-owned transaction; see [Bulk changes](#bulk-changes)).

MySQL/MariaDB with `autocommit=0` (PyMySQL's default): the server reports a
transaction only after a write, so a caller's implicit read-only transaction (a
consistent snapshot, or row locks from `SELECT ... FOR UPDATE`) looks idle.
Read-only methods never end it (they join it, and the metadata locks their
`information_schema` reads take are held until the caller commits or rolls
back; with `autocommit=1` they leave nothing open). Lifecycle methods and
`bulk_changes()` cannot tell such a transaction from an idle connection: they
take ownership and commit or roll it back, which releases its locks and
snapshot (no caller writes are affected: a written transaction is detected and
refused). Supported usage on MySQL/MariaDB is `autocommit=1`, or ending your
own transaction before calling a lifecycle method or `bulk_changes()`.
Detecting implicit read transactions (`information_schema.innodb_trx` is a
cached snapshot refreshed at most every 100 ms; `performance_schema` is off by
default on MariaDB) was evaluated and rejected as unreliable.

```python run
import sqlite3
from matview import MaterializedViewManager, TransactionOwnershipError

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL)")
manager = MaterializedViewManager(conn)

conn.execute("INSERT INTO t VALUES (1, 1)")  # sqlite3 opens a transaction here
try:
    manager.register_view("CREATE VIEW g_counts AS SELECT g, COUNT(*) AS n FROM t GROUP BY g")
except TransactionOwnershipError:
    print("refused while the caller's transaction is open")
conn.commit()
manager.register_view("CREATE VIEW g_counts AS SELECT g, COUNT(*) AS n FROM t GROUP BY g")
print(conn.execute("SELECT * FROM g_counts").fetchall())
```

```text output
refused while the caller's transaction is open
[(1, 1)]
```

| Driver | Status |
| --- | --- |
| `sqlite3` (standard library) | Supported, tested. Must use the default legacy transaction control (`isolation_level=None` is fine); Python 3.12+ `autocommit=True/False` connections are refused. |
| APSW | Supported, tested in the APSW CI job (raw APSW connections are committed with `COMMIT`). |
| psycopg2 | Supported, tested natively. A connection in autocommit mode gets one explicit transaction per operation and autocommit back afterwards. |
| PyMySQL | Supported, tested natively. |
| mysql-connector-python | Accepted (it reports `in_transaction`); not exercised against a server in CI. |
| pg8000, mysqlclient (`MySQLdb`) | Refused with `TransactionOwnershipError`: they do not report transaction state. |
| psycopg 3 | Not supported: schema introspection fails (known issue). |

### Initial population and backfill

By default `register_view()` populates the view with one `INSERT ... SELECT` in
the same transaction that installs it.

`incremental_backfill=True` (with `batch_size=`, and an optional
`on_progress(batches_done, keys_covered)` callback) installs the triggers and
then folds the existing rows in primary-key ranges, committing each batch. A
watermark in the catalog makes the triggers ignore rows that the backfill has
not reached yet, so writes can continue during the backfill. View shapes that
cannot use primary-key ranges (group-key or offset batches) run the backfill in
a snapshot (`REPEATABLE READ` on PostgreSQL and MySQL/MariaDB, a savepoint on
SQLite) instead. While a primary-key-range backfill runs, readers see a
partially populated view.

A backfill fails closed. If it stops after committing progress, the original
exception propagates, the watermark stays at the last committed batch (so the
view is incomplete but not double-counted), and nothing resumes it: a later
`backfill_incrementally()` for that view raises `BackfillInterruptedError`,
and `check_schema_drift()` / `sync_views()` report `backfill_incomplete`. Repair
with `repair_view()`, `register_view(..., force_rebuild=True)` or `drop_view()`
followed by `register_view()`:

```python run
import sqlite3
from matview import MaterializedViewManager, BackfillInterruptedError

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE readings (id INTEGER PRIMARY KEY, sensor INTEGER NOT NULL, value INTEGER NOT NULL)")
conn.executemany("INSERT INTO readings VALUES (?, ?, ?)", [(i, i % 3, i) for i in range(1, 101)])
conn.commit()
manager = MaterializedViewManager(conn)
VIEW = """CREATE VIEW sensor_totals AS
          SELECT sensor, COUNT(*) AS n, SUM(value) AS total FROM readings GROUP BY sensor"""

def stop_after_two_batches(batches_done, keys_covered):
    if batches_done == 2:
        raise KeyboardInterrupt  # simulate a crash after two committed batches

try:
    manager.register_view(VIEW, incremental_backfill=True, batch_size=25,
                          on_progress=stop_after_two_batches)
except KeyboardInterrupt:
    pass
print(manager.check_schema_drift())

plan = manager.diff_view(VIEW, incremental_backfill=True).plan
try:
    manager.backfill_incrementally(plan, batch_size=25)
except BackfillInterruptedError:
    print("resume refused")

print(manager.repair_view("sensor_totals"))
print(manager.check_schema_drift())
print(conn.execute("SELECT * FROM sensor_totals ORDER BY sensor").fetchall() ==
      conn.execute("SELECT sensor, COUNT(*), SUM(value) FROM readings GROUP BY sensor ORDER BY sensor").fetchall())
```

```text output
{'sensor_totals': 'backfill_incomplete'}
resume refused
{'sensor_totals': 'repaired'}
{'sensor_totals': 'ok'}
True
```

Known gaps (details in
[Fail-Closed Incremental Backfill (no silent missing or double-counted rows after a failed backfill)](TODO.md)):
writes committed while `register_view(..., incremental_backfill=True)` is
still between registration and its first batch can be counted twice;
`register_view(populate=False)` leaves the triggers active before any backfill;
group-key and offset backfills keep no watermark. The standalone SQL that the
CLI generates with `--incremental-backfill` does not include the watermark
protocol at all: run it with writes to the source tables stopped.

### Replacement, schema changes and repair

- `register_view(..., force_rebuild=True)`, `sync_views()` with a changed
  definition, and a rebuilding `repair_view()` replace a published view. On
  SQLite and PostgreSQL this is atomic: the old view stays readable and is
  restored if anything fails. Where that is impossible (MySQL/MariaDB, whose
  DDL commits implicitly, and replacements populated with
  `incremental_backfill=True`), replacing a complete view raises
  `UnsafeReplacementError` before changing anything; `sync_views()` checks all
  requested views first. Replace it explicitly with `drop_view()` then
  `register_view()`; readers then see the view disappear and refill. A view
  whose backfill is incomplete can always be rebuilt.
- `apply_schema_ddl(ddl)` runs DDL and re-registers affected views. On
  MySQL/MariaDB it refuses (`UnsafeReplacementError`, naming the views, before
  executing anything) DDL that touches a source table of a registered view;
  there, `drop_view()` each named view, apply the DDL, and register again. Any
  change to a referenced table currently counts, even to a column the view
  does not use.
- `check_schema_drift()` (`--check-drift`) compares each registered view with
  the live schema: `ok`, `drifted`, `invalid`, `backfill_incomplete`, or
  `triggers_missing`, `triggers_disabled` (PostgreSQL `ALTER TABLE ... DISABLE
  TRIGGER`), `triggers_out_of_order` (SQLite: a user trigger created after
  registration that writes a source table or uses `RAISE(FAIL)`/`RAISE(IGNORE)`
  fires before the maintenance trigger, because SQLite fires the newest trigger
  first) or `triggers_orphaned` when maintenance triggers are gone, disabled,
  preceded or on a detached partition (`repair_view()` rebuilds them).
  PostgreSQL and MySQL/MariaDB also report `triggers_out_of_order` for a
  multi-source view when a user trigger on one source fires before the
  maintenance trigger and writes another source; there `repair_view()` only
  repopulates and does not reorder triggers: on MySQL/MariaDB re-register the
  view (`drop_view()` then `register_view()`, which places its triggers
  first), on PostgreSQL rewrite the user trigger as a statement-level trigger
  named after the maintenance triggers.
  `diff_views()` / `matview diff` shows what a registration would change.
- `repair_view(name=None)` (`--repair`) repopulates (or rebuilds after drift)
  each view in its own transaction; there is no all-view atomicity (see
  [Concurrency notes](#concurrency-notes) for locking and isolation details).
- Upgrades: a view's catalog signature includes a generator version, so after
  an upgrade that changes the generated maintenance SQL, `check_schema_drift()`
  reports the view as `drifted` and `register_view()` / `sync_views()` rebuild
  it, subject to the replacement rules above (tested on SQLite).

### Bulk changes

`with manager.bulk_changes(view_names=None, use_shadow_swap=False,
drop_triggers=False):` runs DML on the manager's connection in one
manager-owned transaction.

On PostgreSQL it suspends nothing: the statement-level triggers maintain every
view once per statement during the block, which is faster than suspending and
repopulating (see [PostgreSQL maintenance](#postgresql-maintenance)). Only the
transaction contract below applies; `use_shadow_swap` and `drop_triggers` are
refused with `ValueError` before any SQL, and `view_names` is only checked for
unknown names.

On SQLite and MySQL/MariaDB it is an optional, Python-only optimisation for
large loads: it suspends per-row maintenance of the named views (all
registered views by default) while the block runs, then repopulates each of
them from the source tables. The block,
the refresh and any trigger DDL run in one manager-owned transaction: it is
committed when the block exits normally and rolled back on any exception
(including `KeyboardInterrupt`), which also restores the original maintenance
objects. Like the lifecycle methods, `bulk_changes()` raises
`TransactionOwnershipError` before executing any SQL when the connection
already has an open transaction or its driver does not report transaction
state; it opens its transaction explicitly (`BEGIN IMMEDIATE` on SQLite,
`START TRANSACTION` on MySQL/MariaDB also for autocommit connections; a
PostgreSQL autocommit connection is switched off and back).

On SQLite and MySQL/MariaDB maintenance is suspended only for that transaction
or session, never globally:

- MySQL/MariaDB: a session variable, DML only. `use_shadow_swap` and
  `drop_triggers` need DDL that commits implicitly and are refused with
  `ValueError` before any SQL. The variable suspends every registered view, so
  `view_names` must include all of them; it is cleared before the commit.
- SQLite: the maintenance triggers are dropped and recreated inside the
  `BEGIN IMMEDIATE` transaction, which blocks other writers.

`use_shadow_swap=True` (SQLite) builds the new contents in shadow
tables and swaps them in. Unknown view names, and a `view_names` subset that
would leave other views unmaintained, are refused before the block runs.

```python run
import sqlite3
from matview import MaterializedViewManager, TransactionOwnershipError

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL, amount INTEGER NOT NULL)")
manager = MaterializedViewManager(conn)
manager.register_view("CREATE VIEW totals AS SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id")

conn.execute("INSERT INTO orders VALUES (1, 1, 10)")  # the caller's transaction is open
try:
    with manager.bulk_changes():
        pass
except TransactionOwnershipError:
    print("refused while the caller's transaction is open")
conn.commit()

try:
    with manager.bulk_changes():
        conn.execute("INSERT INTO orders VALUES (2, 1, 5)")
        raise RuntimeError("fails mid-block")
except RuntimeError:
    pass
print(conn.execute("SELECT * FROM totals").fetchall())
print(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0] > 0)
```

```text output
refused while the caller's transaction is open
[(1, 10)]
True
```

```python run
import sqlite3
from matview import MaterializedViewManager

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL, amount INTEGER NOT NULL)")
manager = MaterializedViewManager(conn)
manager.register_view("""CREATE VIEW customer_totals AS
    SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id""")

with manager.bulk_changes(["customer_totals"]):
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(i, i % 2, i) for i in range(1, 1001)])
print(conn.execute("SELECT * FROM customer_totals ORDER BY customer_id").fetchall())
```

```text output
[(0, 250500), (1, 250000)]
```

Failures are raised, never suppressed:

- Any failure in setup, the block, the refresh or the restoration of
  maintenance rolls everything back and is raised. If the connection's state
  cannot be restored (the rollback fails, or the MySQL/MariaDB bypass variable
  cannot be cleared) the connection is closed, so nothing half-done is
  committed and no session keeps skipping maintenance, and
  `BulkChangesConnectionError` is raised; the manager then refuses every later
  operation.
- The block must leave transaction control to the manager. A block that
  commits or rolls back the connection itself is detected: its later work is
  rolled back, then `BulkChangesTransactionError` is raised. On SQLite and
  MySQL/MariaDB maintenance is first restored and the views are refreshed in a
  new transaction (on PostgreSQL nothing was suspended). If that recovery
  fails, `BulkChangesRecoveryError` (a subclass) is raised and its
  `view_names` lists the views left stale by the work the block committed;
  repair them with `repair_view()`. Only where their maintenance triggers are
  also missing (SQLite) do `check_schema_drift()` and `sync_views()` report
  them (as `"triggers_missing"`, rebuilt by `repair_view()`); with the session
  bypass (MySQL/MariaDB) stale contents are not detectable by the drift check, so `view_names` is the
  only record.

Inside the block, lifecycle methods (`register_view`, `sync_views`,
`drop_view`, `apply_schema_ddl`, `repair_view`, `backfill_incrementally`) are
refused with `TransactionOwnershipError`; read methods join the block's
transaction. The refresh takes over an incremental backfill's watermark like
`repair_view()`: a running backfill stops at its next batch, and an
interrupted one is recorded as complete.

On SQLite and MySQL/MariaDB the set of registered views is fixed while a block
is open, and other sessions wait as follows (PostgreSQL takes no such lock):

- MySQL/MariaDB: the catalog is read with a locking read (`LOCK IN SHARE
  MODE`), so other sessions' `register_view` and backfill watermark updates
  wait up to `innodb_lock_wait_timeout` (and then fail with a lock-wait
  timeout) or are chosen as InnoDB deadlock victims.
- SQLite: the writer reservation of `BEGIN IMMEDIATE` serializes everything.

The catalog is checked again at refresh time; if it nevertheless changed, the
block is rolled back.

On PostgreSQL with the default session bypass (`drop_triggers=False`),
concurrent writers that commit during the refresh can be missed because the
refresh does not lock source tables (see
[Concurrency notes](#concurrency-notes)). Other known limitations:

- On MariaDB with `innodb_snapshot_isolation` (on by default in 12.3), a
  block that only reads with a consistent snapshot can make the refresh fail
  with error 1020 when a concurrent writer has committed; the bulk is rolled
  back cleanly, not left divergent.
- Views registered by another manager (a different `object_prefix`/catalog)
  are also suspended by the session bypass but are not refreshed.
- Participation in a caller's transaction through savepoints is not offered.

See
[Make `bulk_changes()` Fail-Closed, Exception-Safe and Transaction-Ownership-Safe](TODO.md)
for the remaining work.

### SQLite writer connections

Call `MaterializedViewManager.configure_sqlite_connection(conn)` on every
SQLite connection that writes source tables (for example in a connection-pool
hook). It enables and verifies the connection-local `PRAGMA recursive_triggers
= ON`, without which `INSERT OR REPLACE` / `REPLACE` does not fire `DELETE`
triggers for the rows it replaces, and the materialized view silently keeps
them. The manager configures only its own connection; nothing enforces this on
other connections. Prefer `INSERT ... ON CONFLICT (...) DO UPDATE` where update
semantics are intended.

```python run
import sqlite3
from matview import MaterializedViewManager

writer = MaterializedViewManager.configure_sqlite_connection(sqlite3.connect(":memory:"))
print(writer.execute("PRAGMA recursive_triggers").fetchone())
```

```text output
(1,)
```

### Concurrency notes

Maintenance runs in the writer's transaction, so what concurrent writers can
observe depends on the engine:

- A view whose `HAVING` compares with an aggregate inside a scalar subquery
  (`HAVING COUNT(*) > (SELECT SUM(x) FROM t)`) is maintained by the
  group-scoped fallback, announced at registration with
  `exact result required: falling back to strategy group-scoped recompute`
  (a `MatviewDiagnosticWarning`, kept in `plan.strategy_diagnostics`). A
  write to a table the subquery reads recomputes every group. On
  MySQL/MariaDB, concurrent writers to that table deadlock (error 1213) and
  roll back: about 45% of writes with 4 concurrent writers in our
  measurement. The results stay exact; retry the rolled-back transactions.

- SQLite serializes writers database-wide; manager operations reserve the
  writer lock before their DDL.
- PostgreSQL: see [PostgreSQL maintenance](#postgresql-maintenance).
  - Writers of the same source table and the same group serialize on that
    group (its backing row, or for group-scoped views a per-group lock row)
    and every recompute runs in a statement after that lock, so under
    `READ COMMITTED` it reflects every committed writer of the group. Writers
    of different groups do not wait for each other (unless their lock ids
    collide: the id is a 64-bit hash, and a collision only makes them wait).
    A group is identified by a canonical text of its key that is equal
    exactly when the keys are equal under the key type's equality, in any
    session (see [Group keys](#group-keys)).
  - Full-recompute views (`atomic_query_sync`, for example a plain join)
    serialize all their writers on one per-view lock row. A writer can then
    also deadlock (SQLSTATE 40P01) with another transaction that holds a row
    lock it needs, where independent row writes would only have waited.
  - Under `REPEATABLE READ` and `SERIALIZABLE` a writer whose snapshot
    predates a committed change of a group (or of a full-recompute view) it
    maintains fails with a serialization error (SQLSTATE 40001).
  - Applications must retry on 40001 and 40P01.
  - Not exact under `READ COMMITTED`: two transactions that write *different*
    sources of a join aggregate (for example one inserts into `t`, the other
    renames a row of `d` in `SELECT d.name, SUM(t.v) FROM t JOIN d ...`)
    each join their delta with the other table's committed state and miss
    each other's change ([Serialize Concurrent Writers of Different Sources of a PostgreSQL Join View](TODO.md)).
    Under `REPEATABLE READ`/`SERIALIZABLE` the second writer fails with 40001.
- PostgreSQL `repair_view()`: on the repopulate (`"repaired"`) path, acquires
  `SHARE` locks on the view's referenced source tables (in deterministic sorted
  order, after the catalog lock) before clearing and repopulating, so under
  `READ COMMITTED` writers serialize cleanly while concurrent readers
  (`ACCESS SHARE`) remain unblocked (verified natively). Rebuilding after
  schema drift or missing triggers (`"rebuilt"`) uses `register_view(...,
  force_rebuild=True)` DDL locks (which block readers while running). Known
  exclusion: a concurrent `REPEATABLE READ` or `SERIALIZABLE` transaction whose
  snapshot was already established *before* `repair_view()` committed can still
  observe the pre-repair catalog/backing snapshot in its trigger (e.g. an
  active watermark or empty `ATOMIC_QUERY_SYNC` store) and silently diverge.
- MySQL/MariaDB (InnoDB, default `REPEATABLE READ`): the repopulating
  `INSERT ... SELECT` of `repair_view()` takes shared locks on the rows it reads,
  so writers serialize with it (verified natively with concurrent writers).
  Other isolation levels are untested.

### Performance notes

- `algebraic_delta` avoids work proportional to table size for the shapes it
  covers; this is not a constant-time guarantee. Join fan-out, group sizes,
  indexes and the engine's plans still decide the cost of each write.
- Tier 2 and tier 3 views, and the recomputed aggregates above, read the source
  tables on every relevant write. Create the indexes in
  `plan.recommended_base_indexes` (also reported with reasons, and with
  `performance_notes`, in `plan.diagnostics.details`); `matview` does not
  create indexes on your tables. It does create indexes on its own backing and
  auxiliary tables, plus any you request with `indexes=` / `--index`.
- On PostgreSQL a statement of N rows runs each view's maintenance once
  (see [PostgreSQL maintenance](#postgresql-maintenance)); on SQLite and
  MySQL/MariaDB maintenance runs once per row.
- For large one-off loads see [Bulk changes](#bulk-changes); for large initial
  populations see [Initial population and backfill](#initial-population-and-backfill).

### PostgreSQL maintenance

PostgreSQL (11 or later) maintains every view with statement-level triggers and
transition tables, its only mechanism: one `AFTER INSERT`, `AFTER UPDATE` and
`AFTER DELETE` trigger per view and source table, plus `AFTER TRUNCATE`
(design: [docs/design-bulk-maintenance.md](docs/design-bulk-maintenance.md)).

- Each statement applies one consolidated signed delta (an `UPDATE` is its old
  rows retracted and its new rows inserted, so primary-key and group-key
  changes, swaps and cycles are one delta); a statement that changed no row
  returns at once. Statements that change no column the view reads still fire
  the triggers (transition tables cannot be combined with `UPDATE OF`).
- Shared backing rows are upserted in key order, so concurrent multi-row
  writers do not deadlock on them; empty groups are deleted only among the
  keys the statement retracted.
- Native recomputes (`MIN`/`MAX` and `BIT_AND`/`BIT_OR` after the extreme is
  retracted, `AVG` other than over integer columns, `VAR_*`/`STDDEV_*`,
  `BIT_XOR`, distinct aggregates, group-scoped and full recomputes) run in a
  statement after the lock statement (see [Concurrency notes](#concurrency-notes)).
- Partitioned sources: the triggers are installed on the partitioned table and
  on every partition at every level, because a statement on a partition fires
  only that partition's statement triggers. A partition attached or created
  after registration makes `check_schema_drift()` report
  `"triggers_missing"`; `repair_view()` reinstalls the triggers. Writes through
  the parent are maintained meanwhile, direct writes to the new partition are
  not. A source that is itself a partition, or that has (or is) an `INHERITS`
  child, is refused at registration (SQLSTATE 0A000). `DETACH PARTITION`
  leaves the detached rows in the view and the maintenance triggers on the
  detached table (its writes still change the view): `check_schema_drift()`
  reports `"triggers_orphaned"` and `repair_view()` rebuilds the view.
  `DROP` of a partition fires no trigger and is not detected: the view keeps
  the dropped rows until `repair_view()`.
- `TRUNCATE` of a source (or of one partition) recomputes the view.
- The trigger functions run with the `search_path` of their creation
  (`SET search_path FROM CURRENT`), so a writer's own `search_path` cannot
  redirect the recomputes to other tables.
- One query that modifies two sources of a join view (a writable CTE, or an FK
  cascade into both) is not maintained exactly yet
  ([Bulk maintenance stage 3: Exact PostgreSQL Maintenance When One Query Modifies Several Sources of a View](TODO.md)).
- The lock table of a group-scoped view keeps one row per group ever written
  (it is not pruned, see [Prune PostgreSQL Group Lock Rows](TODO.md)), each
  write updates its group's row, and every group-scoped write also holds the
  whole-view row in `SHARE` mode, which creates multixacts under concurrent
  writers; autovacuum must keep up with both.
- Large statements stay exact but slow down when their transition tables and
  recompute sorts exceed `work_mem` (measured: a 300k-row update of a `RANK()`
  view over 100k groups took 4.9 s at 4MB and 41 s at 64kB).
- Not tested: PostgreSQL 16 and 17 (CI and the measurements use 15 and 18);
  `pg_dump`/restore of a database with registered views (the drift check
  matches trigger functions by oid); replacing a maintenance function by hand
  (`CREATE OR REPLACE FUNCTION` keeps its oid).

#### Group keys

A view stores one row per group of its key (`GROUP BY`, `DISTINCT`, set
operations, window partitions). Grouping uses the key type's equality, under
which values can be equal but print differently, and printing depends on
session settings. When a key is nullable (the backing table then identifies a
group by a text primary key) and for the per-group locks of group-scoped
views, each key value is mapped to a canonical text by a per-view function
(`__matview_key_<view>`):

- `NUMERIC`/`money`: trailing fractional zeros removed (1.0, 1.00 and 1 are
  one group); `REAL`/`DOUBLE PRECISION`: 0 and -0 are one group, every NaN is
  one group, other values by their exact binary value (independent of
  `extra_float_digits`);
- `DATE`, `TIMESTAMP`, `TIMESTAMPTZ` (as UTC), `INTERVAL` (its comparison
  span, so `'1 day'` and `'24 hours'` are one group), `BYTEA`: independent of
  `DateStyle`, `TimeZone`, `IntervalStyle` and `bytea_output`;
- `citext`: lowercase; `JSONB`: numbers normalized and object keys sorted;
  arrays and ranges: element-wise; other base and enum types: their text.
- Composite, record and multirange keys have no canonical text yet: a write
  that would maintain such a key fails with SQLSTATE 0A000
  ([Canonical Keys for Composite and Multirange Types](TODO.md)).
- Nondeterministic collations (PostgreSQL 12+, for example case-insensitive
  ICU collations) make distinct strings equal, so no stored value identifies
  their group: a view that groups, deduplicates, partitions or takes
  `MIN`/`MAX` and mentions a column with such a collation, or names one after
  `COLLATE`, is refused at registration (SQLSTATE 0A000). The check is by
  name in the view text, so it can also refuse a view that only filters on
  such a column ([Support Nondeterministic Collations](TODO.md)).

### Floating-point aggregates

- SQLite: `SUM`, `TOTAL` and `AVG` are maintained exactly. A group is folded
  incrementally only while it holds integers and the fold is provably identical
  to SQLite's own result; otherwise the affected group is recomputed from the
  source table with the view's own aggregate. A group holding any non-INTEGER
  value therefore costs O(rows in the group) per write with an index on the
  group key, and O(rows in the table) without one; join, `UNION ALL` and
  `DISTINCT` forms always recompute. A write that would make a native `SUM`
  overflow 64-bit integers is rejected with `integer overflow`. SQLite's own
  floating-point sums depend on the order its plan visits rows, which leaves
  edge cases open: [Resolve SQLite SUM Overflow Edge Cases](TODO.md) and
  [Make SQLite Join SUM/TOTAL/AVG Recompute Visit Rows in the Original View's Order (or Reject)](TODO.md).
- PostgreSQL and MySQL/MariaDB: integer and `DECIMAL`/`NUMERIC` `SUM`/`AVG`
  (and PostgreSQL `VAR`/`STDDEV` of them, computed in `NUMERIC`) are
  maintained exactly (PostgreSQL sums of variable-scale `NUMERIC` are
  recomputed, because numeric addition keeps the largest display scale).
- PostgreSQL and MySQL/MariaDB: `SUM`/`AVG` over approximate values
  (`REAL`/`DOUBLE`/`FLOAT`, or any argument not provably integer/fixed-point)
  and MySQL/MariaDB `VAR_*`/`STDDEV_*` (computed in `DOUBLE` for every argument
  type) are **refused by default in every shape**
  (`ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE`). Their result depends on the
  order in which the original view's plan reads the rows, and that order is not
  a function of the data. Measured natively with the rows 1.5, 1e16, -1e16:

  | Plan of the original view | Engines | Original view | Fixed (scan / primary-key) order |
  | --- | --- | --- | --- |
  | `SUM(x) ... WHERE y < 10` through an index on `y` (global and grouped) | PostgreSQL 18, MySQL 8.4, MariaDB 10.11 | 1.5 | 2.0 |
  | Parallel aggregate (default `max_parallel_workers_per_gather`, 600k rows) | PostgreSQL 18 | varies between runs | 2.0 |
  | Sort + GroupAggregate instead of HashAggregate | PostgreSQL 15/18 | 1 ulp off (`VAR_SAMP`) | |

  Window `SUM`/`AVG` (`OVER (...)`), joins, `UNION ALL` and `DISTINCT` mixes
  are refused the same way.

- Opt-in (`accept_limitations=["ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"]`,
  plus the other codes the refusal names): **best effort**, never exact in
  general. Per engine:

  | | PostgreSQL 15/18 | MySQL 8.4/9.7, MariaDB 10.11/11.8/12.3 |
  | --- | --- | --- |
  | Default | refused | refused |
  | Single table (grouped, global, `WHERE`, `HAVING`, `DISTINCT`) | each affected group recomputed after its row lock, in sequential-scan order (`AGG(x ORDER BY ctid)`), race-free | each affected group recomputed with a locking read (`LOCK IN SHARE MODE`) after `BEFORE` triggers lock the group row, race-free; grouped views also need `MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX` accepted, or a group-key index |
  | Equal to the original view when | it runs a serial sequential scan feeding HashAggregate (the tests pin `enable_indexscan/indexonlyscan/bitmapscan/sort = off`, `max_parallel_workers_per_gather = 0`) | both it and the recompute read the group in primary-key order: no index usable for a `WHERE` column (the recompute's own locking read may use it and then reads in that index's order), small tables |
  | Joins, windows | not race-free: `ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE` must be accepted too; values can drift until `repair_view()` | same, also for `UNION ALL` |

  Other known differences under the opt-in: PostgreSQL `CLUSTER` and other
  table rewrites reorder rows without firing triggers (run `repair_view()`);
  on MySQL (not MariaDB) a group whose values are all `-0.0` sums to `-0.0` or
  `0.0` depending on the grouping plan.
- Opt-in costs. PostgreSQL: one recompute per affected group and write,
  O(rows in the group) with the group-key index; a view without `GROUP BY`
  re-reads and sorts the whole table by `ctid` on every write.
  MySQL/MariaDB: the same recompute, plus:
  - Deadlocks: writers lock the group row and then read the group with a
    locking read, so transactions that write several groups in different
    orders deadlock more often than with plain incremental sums. InnoDB rolls
    one transaction back with error 1213; applications must retry it. Measured
    with 4 concurrent writers on 6 groups, 600 transactions, group-key index
    present (deadlocks per 600):

    | Writes per transaction | MySQL 9.7 | MariaDB 10.11 | before this change (inexact sums), MySQL 9.7 / MariaDB 10.11 |
    | --- | --- | --- | --- |
    | 1 | 0-10 (6 seeds, median 2) | 1 | 0 / 0 |
    | 3 | 27 | 71 | 7 / 5 |

    Rates are workload-specific (an independent mixed workload on MySQL 8.4
    measured 56/600 grouped and 216/600 global). Without a group-key index
    the single-write case deadlocked 122 (MySQL 9.7) and 34 (MariaDB 10.11)
    of 600.
  - Memory: every trigger that reads the source table makes a statement open
    another instance of that table, and the server parses all of the table's
    triggers for each instance, so parsed trigger memory per session grows
    with the square of the number of opted-in views on one table. MySQL 8.4,
    `SUM(x + i) ... GROUP BY g` views on one table, one `UPDATE`, retained
    per session: 10 views 106.7 MiB (exact-typed views: 4.6), 20 views
    405.0 MiB (exact: 9.3); every pooled connection keeps its own copy. Keep
    such views per source table to a handful
    ([Share Float-Recompute Triggers Per Table on MySQL/MariaDB](TODO.md)).
- The differential tests compare every column exactly, except `VAR_*`/
  `STDDEV_*` on SQLite (a small error budget: SQLite has no native variance
  aggregate, and the reference is a Python function; on libSQL, which cannot
  register that function, VAR/STDDEV views have no live view and are excluded
  from the differential by name; the maintained value is not yet correctly
  rounded, see [Make SQLite VAR/STDDEV Correctly Rounded (exact against a rational reference)](TODO.md)), and except the
  floating-point aggregate columns of opted-in views, whose values are not
  compared at all (only NULL-ness and type); exactness of the opt-in is only
  asserted under the pinned plans above.

```python run
import warnings
from matview import EngineLimitationError, generate_materialized_view, parse_table_schema

ddl = "CREATE TABLE readings (id BIGINT PRIMARY KEY, sensor INT NOT NULL, value DOUBLE PRECISION)"
grouped = "CREATE VIEW per_sensor AS SELECT sensor, SUM(value) AS total FROM readings GROUP BY sensor"
overall = "CREATE VIEW overall AS SELECT COUNT(*) AS n, SUM(value) AS total FROM readings"

pg = parse_table_schema(ddl, dialect="postgres")
for view in (grouped, overall):
    for kwargs in ({}, {"ignore_limitations": True}):
        try:
            generate_materialized_view(pg, view, dialect="postgres", **kwargs)
        except EngineLimitationError as exc:
            print("postgres refused", sorted(kwargs), [lim.code for lim in exc.limitations])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plan = generate_materialized_view(
        pg, overall, dialect="postgres",
        accept_limitations=["ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE"],
    )
print("postgres opt-in:", plan.strategy.value, "BEST-EFFORT" in str(caught[0].message))

def mysql_refusal(schema_sql):
    try:
        generate_materialized_view(
            parse_table_schema(schema_sql, dialect="mysql"), grouped.replace("PRECISION", ""),
            dialect="mysql", ignore_limitations=True,
        )
    except EngineLimitationError as exc:
        return sorted(lim.code for lim in exc.limitations)

my_ddl = ddl.replace("DOUBLE PRECISION", "DOUBLE")
print("mysql without index:", mysql_refusal(my_ddl))
print("mysql with index:", mysql_refusal(my_ddl + "; CREATE INDEX ix_sensor ON readings (sensor)"))
```

```text output
postgres refused [] ['ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
postgres refused ['ignore_limitations'] ['ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
postgres refused [] ['ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
postgres refused ['ignore_limitations'] ['ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
postgres opt-in: algebraic_delta True
mysql without index: ['MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX', 'ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
mysql with index: ['ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE']
```

## Testing and contributing

Read [AGENTS.md](AGENTS.md) first: tests are designed from the documented
contract, native runs are distinguished from transpiled ones, every claim in
documentation must be backed by an exercised test, and the Git/PR workflow is
specified there. [TODO.md](TODO.md) is the prioritized backlog.

### Running tests

Every `test_*.py` module is a standalone script (`python3 test_x.py`); the
Hypothesis module runs under `pytest`. The authoritative list of what runs in
which job is the step list in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
Commonly used entry points:

```sh no-run
python3 test_matview_sync.py              # main suite: unit, integration, differential fuzzing
python3 test_matview_sync.py --fuzz       # only the differential fuzzer and regression corpus
python3 test_matview_sync.py --fuzz --fuzz-scale 3
# replay a logged seed: copy the command printed with it ("[FUZZ SEED] ... reproduce with:"),
# which pins the campaign sizes too; a seed alone replays different scenarios
MATVIEW_FUZZ_SCALE=2 MATVIEW_FUZZ_SCENARIOS=3 MATVIEW_FUZZ_VIEWS_PER_SCHEMA=70 MATVIEW_FUZZ_MUTATIONS=60 MATVIEW_FUZZ_SEEDS=20261106 python3 test_matview_sync.py --fuzz
python3 test_concurrent_backfill.py       # independent-session backfill concurrency
python3 test_query_ast_materialization.py # generated SELECT ASTs materialized and compared
python3 mutation_testing.py --list        # curated mutants; run without --list to check they are killed
python3 test_readme_examples.py           # this README's examples, links and CI matrix
make test                                 # a subset of the SQLite suites (not the full CI list)
```

Native engines are enabled by environment variables; without them those tests
are skipped and only SQLite (plus transpiled) paths run:

```sh no-run
POSTGRES_DSN="postgresql://postgres:password@127.0.0.1:5432/matview_test" python3 test_matview_sync.py
MYSQL_DSN="mysql://root:password@127.0.0.1:3306/matview_test" python3 test_matview_sync.py
SQLITE_COMPAT_ENGINE=apsw python3 test_matview_sync.py   # apsw, sqlcipher, libsql, sqlite3mc, sqlean
```

`make docker-test` starts the PR gate's PostgreSQL, MySQL and MariaDB server
images with Docker Compose ([`docker-compose.yml`](docker-compose.yml)), checks
each server's product and version with `verify_engine_matrix.py`, and runs
`test_matview_sync.py` and `test_concurrent_backfill.py` against each, plus the
SQLite suites (`run_docker_tests.py --engine <id>` selects one engine). This is
a subset of the CI steps; the local presubmit below runs all of them.

### Local presubmit

`scripts/presubmit.py` derives its plan from `.github/workflows/ci.yml` and runs
the PR-gate jobs locally with the same `run:` steps and `env:` (needs PyYAML:
`pip install -r requirements-dev.txt`):

```sh no-run
make presubmit        # fast: path classification + affected SQLite-job steps on the working tree
make presubmit-full   # every PR-gate job/matrix row of HEAD, in parallel; logs under .git/presubmit/runs/
make install-hooks    # pre-push refuses commits whose tree lacks a full-tier stamp
```

Exit codes: 0 green, 1 failed, 2 plan error, 3 incomplete. Rows whose Python
(`python3.N` or `PRESUBMIT_PYTHON_3_N`) or container runtime
(`PRESUBMIT_CONTAINER_RUNTIME`, default `docker` then `podman`) is unavailable
are reported as not verified and never count as green. Only a fully green
full-tier run writes a stamp (`.git/presubmit/stamps/<tree>.json`). A local
green run is not proof of a green GitHub run; `--help` and the stamp's
`not_covered` list state what differs.

### How the tests establish correctness

- Differential oracle: after installing a view and after each mutation, the
  tests run the original `SELECT` natively against the same database state and
  compare it with the materialized view: row multisets, values and NULLs. The
  oracle does not use the compiler's output. Remaining weaknesses (numeric
  error budgets, result-type metadata) are tracked in
  [Complete the Differential Oracle's Semantic and Fault-Detection Contract](TODO.md).
- Generators: `SchemaCoverageTracker` records the schema topologies, constraint
  kinds and column type families that the scenario generator produced; a
  recursive DML AST generator produces the mutations;
  `SchemaAwareSelectFuzzer` (`test_query_ast_fuzzer.py`) produces recursive
  SELECT AST queries. Large parts of scenario and query generation are still
  template-based; see
  [Replace Fixed-Structure Scenario Generation with Genuine Schema Generation](TODO.md).
  `test_strength_reporting.py` reports structural coverage separately from
  workload statistics (views registered, assertions executed), which measure
  volume, not coverage.
- Native vs transpiled: only runs against a real engine count as evidence of
  that engine's semantics; `TranspiledConnectionAdapter` runs are auxiliary.
- Concurrency tests use independent connections with explicit synchronization.
- Mutation testing: `mutation_testing.py` applies curated source mutants and
  requires the targeted regression tests to fail; it runs in the SQLite PR-gate
  jobs.

### CI organization

- PR gate ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): the jobs in
  the [CI matrix](#ci-matrix), aggregated into the required `CI success` check.
  A pull request that changes only whitelisted documentation files (for
  example `TODO.md`, `AGENTS.md`, `docs/`) skips the matrix; changes to
  `README.md` run the full gate because its examples are tests.
  Its fuzz campaign uses fixed seeds only (the workflow-level `MATVIEW_FUZZ_*`
  settings), so a fuzz failure in the gate reproduces from the tested revision.
- Extended random-seed fuzz
  ([`.github/workflows/ci-extended.yml`](.github/workflows/ci-extended.yml)):
  every push to master, nightly and on demand; the scale-2 campaign with fresh
  random seeds on SQLite and native PostgreSQL 18, MySQL 8.4 and MariaDB 12.3.
  Not a merge gate. Each job uploads its `fuzz-campaign.json` manifest and a
  failing-seed report with replay commands; a failing push or nightly run
  opens or comments on one issue labelled `ci-extended-fuzz`. Each confirmed
  failure is added to the permanent regression corpus in its fix PR.
- Deferred compatibility sweep
  ([`.github/workflows/ci-deferred.yml`](.github/workflows/ci-deferred.yml)):
  nightly and on demand for additional versions and the minimum declared
  dependency versions (exact pins derived from `requirements.txt`, Python
  3.11); weekly (Sunday) and on demand for the scale-3 fuzz campaigns, which
  upload failing-seed reports with replay commands. Not a merge gate; a failing
  scheduled run opens or comments on one issue labelled `ci-deferred`.
- Both non-required workflows report through `scripts/ci_failure_issue.js`.
  Opening issues requires GitHub Issues to be enabled for the repository; while
  they are disabled, the reporting job fails instead, with the full report in
  its job summary, so a failure is never silent.
- Every CI job has a `timeout-minutes` limit (25 minutes for PR-gate jobs).

### Executable documentation

`test_readme_examples.py` runs in the SQLite PR-gate jobs and executes this
README. Fenced blocks follow this convention:

- Every `python`, `sh`, `bash` or `console` block is marked `run` or `no-run`
  in its info string (for example ```` ```python run ````); an unmarked block
  fails the test. `no-run` blocks are only syntax-checked.
- `python run` blocks run one by one in a fresh interpreter with the checkout
  on `PYTHONPATH`, so they must be self-contained (use in-memory SQLite).
- `sh run` blocks run in order with `bash -euo pipefail` in one shared
  temporary directory, as if from a checkout; `sql file=NAME` blocks are
  written there first.
- A `text output` block right after a `run` block is that block's exact
  expected standard output.

The test also checks internal anchors, relative file links, that every link to
`TODO.md` names an existing item by its exact title, and that the CI matrix
tables match the workflows (`python3 test_readme_examples.py --print-matrix`
regenerates them).

## Known limitations

Each item names its entry in [TODO.md](TODO.md), which has the details.
Correctness gaps (the materialized view can differ from the original view):

- [Refuse Floating-Point Aggregates on PostgreSQL and MySQL/MariaDB by Default (best-effort per-code opt-in)](TODO.md)
- [Race-Free Recomputation for the Remaining Recomputed Aggregates (stale snapshot after a row-lock wait)](TODO.md) (MySQL/MariaDB untested; PostgreSQL closed)
- [Bulk maintenance stage 3: Exact PostgreSQL Maintenance When One Query Modifies Several Sources of a View](TODO.md) (writable CTE or FK cascade into two sources of a join view)
- [Serialize Concurrent Writers of Different Sources of a PostgreSQL Join View](TODO.md)
- [Detect PostgreSQL Maintenance Triggers Disabled by `session_replication_role = replica` or `ALTER TABLE ... DISABLE TRIGGER`](TODO.md)
- [Fail-Closed Incremental Backfill (no silent missing or double-counted rows after a failed backfill)](TODO.md) (remaining known exclusions)
- [Fix Pre-Existing SQLite Divergences Found by the Exact SUM/TOTAL/AVG Differential](TODO.md) (storage-class changes for `MIN`/`MAX`/group keys, `LEFT JOIN` group moves, primary-key updates during backfill)
- [Model MySQL/MariaDB Result Nullability, DECIMAL Precision and String Length](TODO.md) (result metadata only; values match)
- [Close Unsafe DML Paths That Bypass Trigger-Based Maintenance](TODO.md)
- [Exact Maintenance When a User Trigger or FK Action Runs Before the Maintenance Trigger](TODO.md)
- [Exact Maintenance Under MariaDB UPDATE IGNORE (AFTER UPDATE Triggers Fire for Ignored Rows)](TODO.md)
- [Resolve SQLite SUM Overflow Edge Cases](TODO.md)
- [Canonical Group Keys on MySQL/MariaDB and SQLite](TODO.md) (a nullable `DECIMAL` key 1.0 next to 1.00, a float 0 next to -0, or keys equal under a case-insensitive collation can be split into two groups)

Operational gaps:

- [Make `bulk_changes()` Fail-Closed, Exception-Safe and Transaction-Ownership-Safe](TODO.md)
- [Support Nondeterministic Collations](TODO.md) and [Canonical Keys for Composite and Multirange Types](TODO.md) (PostgreSQL refuses them)
- [Prune PostgreSQL Group Lock Rows](TODO.md)
- [Support or Refuse PostgreSQL 18 VIRTUAL Generated Columns in Source Tables](TODO.md) (registration succeeds, then writes fail)
- [Native Backing Column Types for Array Keys](TODO.md) (PostgreSQL: a view with an array column fails registration)
- [Make Registration, Replacement, Backfill, Publication and Recovery One Explicit Lifecycle Protocol](TODO.md) (no atomic replacement on MySQL/MariaDB, no readiness gate for new incremental registrations)
- [Make Schema Introspection and Drift Signatures Semantically Complete and Fail-Closed](TODO.md)
- [Make Standalone SQL Generation and Manager-Driven Installation Guarantees Consistent](TODO.md)
- [Add Packaging, Licensing, Static Analysis and Release Hygiene](TODO.md)

## License

The repository does not contain a license file yet, so no reuse terms are
granted. Choosing one is tracked in
[Add Packaging, Licensing, Static Analysis and Release Hygiene](TODO.md).
