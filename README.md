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
prints the SQL it executed; here it is saved to a file:

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
3
```

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
| `--incremental-backfill`, `--batch-size N` | Populate in batches (default 500 rows). |
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

Some trigger-bypassing operations are not detected and not refused: `INSERT OR
REPLACE`/`REPLACE` on a SQLite connection that was not configured with
[`configure_sqlite_connection()`](#sqlite-writer-connections), and other engine
features that skip row triggers (see
[Close Unsafe DML Paths That Bypass Trigger-Based Maintenance](TODO.md)).

## Engine support

Support levels used here:

- Natively tested on every PR: the full deterministic suite, the normal-scale
  differential fuzzer and the engine-specific test modules run against the real
  server or library in the PR gate, and each server job checks the actual
  product and version before testing.
- Natively tested weekly: the deferred sweep runs the main suite
  (`test_matview_sync.py`) against these targets. It is not a merge gate and can
  be red; check the latest run in the repository's Actions tab before relying
  on it.
- Transpiled: in `test_matview_sync.py`, the PostgreSQL and MySQL code paths are
  also exercised against SQLite through `TranspiledConnectionAdapter` (sqlglot
  transpilation). These are auxiliary checks, not evidence of native engine
  semantics.
- Best-effort: SQLite3 Multiple Ciphers and sqlean run the main suite weekly as
  SQLite builds; cipher and extension functionality is not tested. Products not
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
PR gate ([`ci.yml`](.github/workflows/ci.yml); required, every pull request):

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

Deferred compatibility sweep ([`ci-deferred.yml`](.github/workflows/ci-deferred.yml); weekly, not a merge gate):

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
| MariaDB 13.1 Rolling | 3.12 | `mariadb:13.1` |
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
ending it. `bulk_changes()` is not covered by this contract yet (see
[Bulk changes](#bulk-changes)).

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
SQLite) instead. Readers see a partially populated view until the backfill
finishes.

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
  the live schema: `ok`, `drifted`, `invalid` or `backfill_incomplete`.
  `diff_views()` / `matview diff` shows what a registration would change.
- `repair_view(name=None)` (`--repair`) repopulates (or rebuilds after drift)
  each view in its own transaction; there is no all-view atomicity. See
  [Concurrency notes](#concurrency-notes) before running it on PostgreSQL.
- Upgrades: a view's catalog signature includes a generator version, so after
  an upgrade that changes the generated maintenance SQL, `check_schema_drift()`
  reports the view as `drifted` and `register_view()` / `sync_views()` rebuild
  it, subject to the replacement rules above (tested on SQLite).

### Bulk changes

`with manager.bulk_changes(view_names=None, use_shadow_swap=False,
drop_triggers=False):` suspends per-row maintenance of the named views (all
registered views by default) inside the block, then repopulates each of them
from the source tables and commits when the block exits normally. If the block
raises, its changes are rolled back to a savepoint and the triggers are
restored. Per-row maintenance is suspended with a session-local setting on
PostgreSQL and MySQL/MariaDB; on SQLite the maintenance triggers are dropped and
recreated inside a `BEGIN IMMEDIATE` transaction, which blocks other writers.
`use_shadow_swap=True` builds the new contents in shadow tables and swaps them
in.

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

`bulk_changes()` is not yet fail-closed or transaction-ownership-safe: it
participates in a caller's open transaction and commits it on exit, it
suppresses some restoration errors, and on MySQL/MariaDB the shadow-swap and
`drop_triggers` paths execute DDL that implicitly commits the caller's
transaction. See
[Make `bulk_changes()` Fail-Closed, Exception-Safe and Transaction-Ownership-Safe](TODO.md).

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

- SQLite serializes writers database-wide; manager operations reserve the
  writer lock before their DDL.
- PostgreSQL (`READ COMMITTED`): aggregates that are recomputed from the source
  tables after waiting for a row lock can read a stale snapshot under
  concurrent writers (confirmed for `AVG` over `NUMERIC`/`DOUBLE`,
  `AVG(DISTINCT)`, `AVG` in join/`UNION` views, `VAR_*`/`STDDEV_*`, and
  `MIN`/`MAX`/`BIT_AND`/`BIT_OR` after a delete; integer and decimal `AVG` are
  derived from the locked row and are not affected). See
  [Race-Free Recomputation for the Remaining Recomputed Aggregates (stale snapshot after a row-lock wait)](TODO.md).
- PostgreSQL `repair_view()` is exact only while writes to the view's source
  tables are quiesced: a write that commits during the repair can be missing
  from the result
  ([Make `repair_view()` Exact on PostgreSQL Under Concurrent Writers (silent divergence today)](TODO.md)).
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
- On PostgreSQL, `pg_use_transition_tables=True` (`--pg-transition-tables`)
  uses statement-level triggers with transition tables instead of row
  triggers.
- For large one-off loads see [Bulk changes](#bulk-changes); for large initial
  populations see [Initial population and backfill](#initial-population-and-backfill).

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
- PostgreSQL and MySQL/MariaDB: `SUM`/`TOTAL` over `REAL`/`DOUBLE`/`FLOAT` are
  still maintained by adding and subtracting, so after deletes or updates the
  result can differ from the original view (for example `0.1 + 0.2 + 0.3` with
  `0.2` removed gives `0.4000000000000001` instead of `0.4`). Integer and
  `DECIMAL`/`NUMERIC` sums are exact. See
  [Make Floating-Point Aggregates Exactly Equal to the Live View on PostgreSQL and MySQL/MariaDB (silent divergence today)](TODO.md).
- `VAR_*`/`STDDEV_*` over floating-point values are still compared with small
  error budgets by the differential tests, so exact equality is not
  established for them on any engine.

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
MATVIEW_FUZZ_SEEDS=20261106 python3 test_matview_sync.py --fuzz   # replay a logged seed
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
- Deferred compatibility sweep
  ([`.github/workflows/ci-deferred.yml`](.github/workflows/ci-deferred.yml)):
  weekly and on demand; additional versions, minimum declared dependency
  versions and scale-3 fuzz campaigns. Not a merge gate.

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

- [Make Floating-Point Aggregates Exactly Equal to the Live View on PostgreSQL and MySQL/MariaDB (silent divergence today)](TODO.md)
- [Race-Free Recomputation for the Remaining Recomputed Aggregates (stale snapshot after a row-lock wait)](TODO.md)
- [Make `repair_view()` Exact on PostgreSQL Under Concurrent Writers (silent divergence today)](TODO.md)
- [Fail-Closed Incremental Backfill (no silent missing or double-counted rows after a failed backfill)](TODO.md) (remaining known exclusions)
- [Fix Pre-Existing SQLite Divergences Found by the Exact SUM/TOTAL/AVG Differential](TODO.md) (storage-class changes for `MIN`/`MAX`/group keys, `LEFT JOIN` group moves, primary-key updates during backfill)
- [Preserve Dialect-Native Statistical Aggregate Alias Semantics](TODO.md) (`VARIANCE`/`STDDEV` are sample statistics on PostgreSQL)
- [Close Unsafe DML Paths That Bypass Trigger-Based Maintenance](TODO.md)
- [Resolve SQLite SUM Overflow Edge Cases](TODO.md)

Operational gaps:

- [Make `bulk_changes()` Fail-Closed, Exception-Safe and Transaction-Ownership-Safe](TODO.md)
- [Make Registration, Replacement, Backfill, Publication and Recovery One Explicit Lifecycle Protocol](TODO.md) (no atomic replacement on MySQL/MariaDB, no readiness gate for new incremental registrations)
- [Make Schema Introspection and Drift Signatures Semantically Complete and Fail-Closed](TODO.md)
- [Make Standalone SQL Generation and Manager-Driven Installation Guarantees Consistent](TODO.md)
- [Add Packaging, Licensing, Static Analysis and Release Hygiene](TODO.md)

## License

The repository does not contain a license file yet, so no reuse terms are
granted. Choosing one is tracked in
[Add Packaging, Licensing, Static Analysis and Release Hygiene](TODO.md).
