# matview

**Turn ordinary SQL views into incrementally maintained materialized views.**

matview compiles supported `CREATE VIEW` definitions into database objects that keep the view's result up to date as its source tables change. Your application continues to query the view by its original name; maintenance happens in the same transaction as the write.

- **Keep reads fast:** query a stored result instead of repeatedly evaluating an expensive query.
- **Use ordinary SQL:** define views with familiar `SELECT` statements.
- **Stay transactional:** source-table changes and their materialized effects commit or roll back together.
- **Choose the trade-off:** matview selects incremental maintenance where possible and falls back to recomputing an affected group or the whole result when necessary.
- **No worker to operate:** maintenance is performed by database triggers, not a background process.

> **Project status:** matview is under active development. It has no packaged release yet, and known correctness and operational gaps remain. It is not yet safe to assume equivalence for every query, database configuration, or write path. Read [Important limitations](#important-limitations) before using it with production data.

## How it works

You write a normal view:

```sql
CREATE VIEW customer_sales AS
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
FROM orders
WHERE status = 'paid'
GROUP BY customer_id;
```

Register it with matview. matview creates and populates the storage and maintenance logic, then keeps the result synchronized when rows in `orders` are inserted, updated, or deleted. Application code reads `customer_sales` as a regular view:

```sql
SELECT * FROM customer_sales;
```

The declared view name remains the application-facing name. Writes to the materialized view are rejected; write to its source tables instead.

### Maintenance strategies

The compiler chooses a strategy based on the query shape:

| Strategy | What happens after a source-table change | Typical query shapes |
| --- | --- | --- |
| `algebraic_delta` | Applies a delta to the affected result row(s), usually constant work per affected group. | Grouped aggregates such as `COUNT`, `SUM`, `AVG`, `MIN` and `MAX`; distinct aggregates; simple projections; joins and set operations where supported. |
| `group_scoped_delta` | Recomputes only the affected group or window partition. | Some aggregates without an algebraic update rule, `LEFT JOIN`, and partitioned window functions. |
| `atomic_query_sync` | Re-runs the complete view query. | Query shapes that cannot safely be maintained at a narrower scope, including some subqueries, order-sensitive aggregates and ordered/limited queries. |

A fallback is still useful when it avoids repeatedly evaluating the query on every read, but its write cost can be much higher. Inspect the selected strategy and diagnostics before relying on a view's performance. Set `strict=True` (or use CLI `--strict`) to reject any view that cannot use `algebraic_delta`.

matview refuses query shapes or engine configurations it cannot handle safely, unless the relevant limitation is explicitly accepted. A successful registration is not a blanket guarantee that every future database operation is safe; see the limitations below.

## Supported databases

| Database | Support |
| --- | --- |
| SQLite | Supported through Python's `sqlite3` driver; other SQLite-compatible drivers have varying test coverage. |
| PostgreSQL | Supported. |
| MySQL | Supported with documented trigger and concurrency restrictions. |
| MariaDB | Supported with documented trigger and concurrency restrictions. |

The target engine and version are detected when connecting to a database. For offline SQL generation, specify the dialect (and, where relevant, engine version) explicitly.

## Installation

There is currently no PyPI package or installable project metadata. Use a checkout:

```sh no-run
git clone https://github.com/CAFxX/matview.git
cd matview
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

SQLite uses Python's standard-library `sqlite3` module. Install the driver for the database you intend to use; `requirements.txt` includes SQLAlchemy, the SQL parser, and the PostgreSQL/MySQL drivers.

## Quick start: Python

This example uses an in-memory SQLite database:

```python run
import sqlite3
from matview import MaterializedViewManager

conn = sqlite3.connect(":memory:")
conn.executescript("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        amount INTEGER NOT NULL,
        status TEXT NOT NULL
    );
    INSERT INTO orders VALUES
        (1, 10, 100, 'paid'),
        (2, 10, 50, 'paid'),
        (3, 20, 70, 'open');
""")

manager = MaterializedViewManager(conn)
plan = manager.register_view("""
    CREATE VIEW customer_sales AS
    SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_amount
    FROM orders
    WHERE status = 'paid'
    GROUP BY customer_id
""")

print(plan.strategy.value)
print(conn.execute("SELECT * FROM customer_sales ORDER BY customer_id").fetchall())

conn.execute("UPDATE orders SET status = 'paid' WHERE id = 3")
conn.commit()
print(conn.execute("SELECT * FROM customer_sales ORDER BY customer_id").fetchall())
```

```text output
algebraic_delta
[(10, 2, 150)]
[(10, 2, 150), (20, 1, 70)]
```

### Database connections

`MaterializedViewManager` accepts either an existing DB-API connection or a database URL, for example:

- `sqlite:///app.db`
- `postgresql://user:password@host/database`
- `mysql://user:password@host/database`
- `mariadb://user:password@host/database`

For SQLite, call `MaterializedViewManager.configure_sqlite_connection(conn)` on **every connection that writes source tables**. This enables and verifies `PRAGMA recursive_triggers = ON`, which is required for correct handling of `INSERT OR REPLACE` / `REPLACE`.

## Quick start: command line

The CLI can install definitions into a live database, compare declared definitions with what is installed, check for schema drift, and repair registered views.

```sh no-run
# Apply schema and view definitions to a SQLite database
python3 matview_sync.py --db app.db --schema schema.sql --view views.sql --apply

# Preview changes without applying them
python3 matview_sync.py diff --db app.db --view views.sql

# Check registered views for schema or installation drift
python3 matview_sync.py --db app.db --check-drift

# Repair registered views when possible
python3 matview_sync.py --db app.db --repair
```

To generate SQL for review or a migration tool without connecting to a database:

```sh no-run
python3 matview_sync.py --schema schema.sql --view views.sql --dialect postgres > install.sql
```

Run `python3 matview_sync.py --help` for the complete command-line options.

## Managing materialized views

The main manager operations are:

| Operation | API |
| --- | --- |
| Register or replace a definition | `register_view()` |
| Reconcile a declared set of definitions | `sync_views()` |
| Preview definition changes | `diff_views()` |
| Check for schema or installation drift | `check_schema_drift()` |
| Rebuild or repopulate a view | `repair_view()` |
| Remove a registered view | `drop_view()` |
| Populate a large view in batches | `backfill_incrementally()` |
| Temporarily batch source-table changes and refresh afterwards | `bulk_changes()` |
| Inspect registered definitions | `get_catalog()` |

For offline compilation, use `generate_materialized_view()` or `generate_materialized_views()` with schemas parsed by `parse_table_schema()`. The returned plan exposes the chosen strategy, diagnostics, recommended base indexes, and generated SQL.

## Important limitations

matview relies on database triggers. Correctness therefore depends on the database actually firing those triggers for every write path, and on the concurrency behaviour of the target engine.

Before using it, account for at least these constraints:

- **Known correctness gaps remain.** Differential testing has found unresolved edge cases involving floating-point aggregates, concurrent changes to different sources of join views, trigger-bypassing DML, some SQLite value/group-key behaviours, and engine-specific write semantics. These are tracked in [TODO.md](TODO.md). Do not treat successful compilation as proof that your workload is covered.
- **SQLite writers must be configured.** Call `configure_sqlite_connection()` for every connection that writes source tables. Without recursive triggers, `REPLACE` can bypass delete maintenance and leave stale rows.
- **MySQL and MariaDB cannot maintain views across `TRUNCATE`.** Their `TRUNCATE TABLE` does not fire row triggers. Registration therefore refuses this by default; if you explicitly accept the limitation, never truncate a source table while the view is registered.
- **Some query shapes require expensive recomputation.** `group_scoped_delta` and especially `atomic_query_sync` can make writes proportional to a group or the full source data. Inspect plan diagnostics and benchmark representative write workloads.
- **Concurrency may require transaction retries.** Depending on engine, isolation level, query shape and concurrent writers, deadlocks or serialization/lock errors can occur. Follow the engine-specific diagnostics and retry transient transaction failures where required.
- **Schema changes need lifecycle management.** Use `check_schema_drift()` and `repair_view()` after out-of-band schema changes; do not assume triggers and stored results automatically adapt.

Known limitations are actively being investigated. The [project roadmap](TODO.md) is the authoritative, evolving list. If you need a guarantee for a particular SQL shape or database operation, verify it against the tests and open an issue with a minimal reproduction before deploying it.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for local validation, test conventions and CI coverage.

## License

No license file is currently present in the repository, so no reuse terms are granted.
