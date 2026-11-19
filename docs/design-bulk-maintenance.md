# Design: In-Database Maintenance for Bulk Statements

Status: decided, not implemented. The stages are tracked as P0 items in
[TODO.md](../TODO.md) ("Bulk maintenance stage 1" to "stage 7").

## Goal and contract

The database itself maintains each materialized view, for every client, in
every language. A single `INSERT ... SELECT`, multi-row `VALUES`, `UPDATE`,
`DELETE`, upsert, `MERGE`, `COPY` or `LOAD DATA` of N rows must not cost N
times the per-row maintenance work where the engine makes that avoidable. The
contract does not change:

- After every statement, the reader relation equals the original view over
  the current base tables, exactly: no tolerances, no lag.
- Maintenance runs in the writer's transaction. Rollback (including to a
  savepoint) restores the view.
- Concurrent writers from independent connections stay exact, or fail closed
  with an error (deadlock or serialization failure) that rolls their work back.
- Unsupported paths are refused explicitly.

A mechanism that makes the view lag, depends on a process outside the
database, or exists only in a Python client is not the default. If it is
offered at all, it is a separate, documented opt-in.

## Current state (master `f1f6561`)

- SQLite, MySQL and MariaDB have only row-level triggers. Every row of a
  statement runs the view's whole maintenance body.
- PostgreSQL uses row-level triggers by default.
  `pg_use_transition_tables=True` (`--pg-transition-tables`, default **off**,
  PostgreSQL 10+) installs `AFTER ... FOR EACH STATEMENT` triggers with
  `REFERENCING NEW/OLD TABLE`. When `pg_vectorize_transition_tables` is on (the
  default) and the body has no `NEW.`/`OLD.` reference, the triggers are
  set-based: the transition rows are aggregated by group, then applied with one
  `INSERT ... ON CONFLICT` or `UPDATE ... FROM`. Otherwise, a PL/pgSQL
  `FOR row IN SELECT * FROM transition_table LOOP` runs the row body once per
  row. That is the fallback for expression `GROUP BY` keys that need a native
  recompute (TODO: "Fix PostgreSQL Transition-Table Recomputations for
  Expression GROUP BY Keys"). The transition path does not honour the
  `bulk_changes()` bypass. One native test covers it
  (`TestNativePGTransitionTableVectorizationSemantics`); everything else checks
  DDL text or runs through the SQLite transpiling adapter.
- `bulk_changes()` is a Python context manager. It suspends row maintenance
  (`SET LOCAL matview.skip_triggers` on PostgreSQL, the
  `@__matview_skip_triggers` session variable on MySQL/MariaDB, trigger
  drop/recreate on SQLite) for a block that runs on the manager's connection.
  It then repopulates the views from their sources in the same transaction. It
  helps only Python callers that use the manager's connection. On PostgreSQL
  the refresh races with concurrent writers (see README "Concurrency notes").

## Measurements

Setup: base table `t(id PK, g, v)` with an index on `(g, v)`, dimension
`d(g PK, name)` with 101 rows mapping to 50 names, 100 groups. Each statement
runs in its own transaction, commit included, from a Python client
(psycopg2/PyMySQL/sqlite3). After every statement the reader is compared with
the original view executed natively: sorted multisets, values compared as the
driver returns them. Every maintained run below was exact; the "none"
configuration has no view. Views:

- `agg`: `SUM`, `COUNT(*)`, `AVG` by `g`.
- `mm`: `MIN`, `MAX` by `g`.
- `j`: `SUM`, `COUNT(*)` over `t JOIN d` by `d.name`.
- `ds`: `SELECT DISTINCT g`.
- `pj`: plain join without aggregation.
- `win`: `ROW_NUMBER() OVER (PARTITION BY g ...)`.

Statements: `INSERT INTO t SELECT` 100k rows (ins), `UPDATE t SET v = v + 1`
(upd v), `UPDATE t SET g = (g + 1) % 100` (upd g), `DELETE` of half the rows
(del ½), `DELETE FROM t` (del all).

Engines ran natively in Docker containers on one Lima QEMU VM (x86_64, 8 vCPU)
with default server configuration. Versions were checked with `SELECT
version()`: PostgreSQL 15.20 and 18.7, MySQL 8.4.13 (REPEATABLE READ,
`innodb_autoinc_lock_mode=2`, `event_scheduler=ON`), MySQL 9.7.4, MariaDB
10.11.19 and 12.3.3 (REPEATABLE READ, `event_scheduler=OFF`). SQLite 3.46.1
ran in-process in WAL mode. Some runs overlapped with runs on another
container in the same VM, so differences under about 30% are noise; every
conclusion below rests on differences of 5x or more, or on the quiet reruns in
the appendix. On MySQL/MariaDB the `INSERT ... SELECT` column is the noisiest
(the first, cold run of a fresh container). The
throwaway harness is not committed. It ran from the design worktree
(`bench.py`, `protos.py`, `pg_sem.py`, `pg_conc.py`, `my_conc.py`,
`read_cost.py`).

### Baseline: 100k-row statements, seconds

| engine | view | config | ins | upd v | upd g | del ½ | del all |
|---|---|---|---|---|---|---|---|
| SQLite 3.46 | — | none | 0.072 | 0.072 | 0.072 | 0.030 | 0.004 |
| SQLite 3.46 | agg | row triggers | 0.221 | 0.379 | 0.369 | 0.114 | 0.108 |
| SQLite 3.46 | agg | bulk_changes | 0.088 | 0.097 | 0.092 | 0.050 | 0.020 |
| SQLite 3.46 | mm | row triggers | 0.119 | 0.203 | 0.234 | 0.071 | 0.078 |
| SQLite 3.46 | mm, **no (g,v) index** | row triggers | 0.098 | 27.9 | 28.5 | 21.2 | 7.27 |
| SQLite 3.46 | j | row triggers | 3.60 | 13.65 | 13.55 | 1.87 | 1.93 |
| SQLite 3.46 | j | bulk_changes | 0.093 | 0.109 | 0.102 | 0.057 | 0.022 |
| SQLite 3.46 | ds | row triggers | 0.091 | 0.065 | 0.158 | 0.071 | 0.068 |
| SQLite 3.46 | pj, **10k rows** | row triggers | 11.07 | 21.14 | 21.65 | 7.85 | 2.62 |
| SQLite 3.46 | pj, 10k rows | bulk_changes | 0.019 | 0.017 | 0.019 | 0.014 | 0.010 |
| SQLite 3.46 | win, 10k rows | row triggers | 0.62 | 1.63 | 3.03 | 0.31 | 0.29 |
| PG 18.7 | — | none | 0.246 | 0.506 | 0.471 | 0.028 | 0.024 |
| PG 18.7 | agg | row triggers | 14.18 | 31.34 | 45.38 | 8.70 | 9.26 |
| PG 18.7 | agg | transition tables | 0.325 | 0.501 | 0.748 | 0.041 | 0.036 |
| PG 18.7 | agg | bulk_changes | 0.593 | 1.637 | 1.032 | 0.217 | 0.208 |
| PG 18.7 | mm | row triggers | 10.01 | 31.09 | 23.59 | 3.55 | 3.54 |
| PG 18.7 | mm | transition tables | 0.202 | 0.415 | 0.493 | 0.061 | 0.056 |
| PG 18.7 | ds | row triggers | 9.00 | 0.615 | 34.60 | 4.20 | 4.03 |
| PG 18.7 | ds | transition tables | 0.386 | 0.566 | 0.721 | 0.060 | 0.062 |
| PG 18.7 | j, **10k rows** | row triggers | 0.300 | 2.209 | 2.492 | 0.666 | 0.599 |
| PG 18.7 | j, 10k rows | transition tables | 0.036 | 0.059 | 0.060 | 0.026 | 0.010 |
| PG 15.20 | — | none | 0.273 | 0.528 | 0.621 | 0.028 | 0.024 |
| PG 15.20 | agg | row triggers | 17.60 | 38.15 | 58.60 | 12.60 | 12.06 |
| PG 15.20 | agg | transition tables | 0.371 | 0.635 | 0.644 | 0.101 | 0.033 |
| PG 15.20 | mm | row triggers | 10.91 | 36.13 | 55.25 | 6.72 | 6.22 |
| PG 15.20 | j | row triggers | 14.73 | 171.7 | 179.7 | 42.3 | 31.3 |
| PG 15.20 | j | transition tables | 0.268 | 0.492 | 0.512 | 0.051 | 0.037 |
| PG 15.20 | ds | row triggers | 7.32 | 0.472 | 26.92 | 3.38 | 3.43 |
| MySQL 8.4.13 | — | none | 1.324 | 0.567 | 0.408 | 0.220 | 0.407 |
| MySQL 8.4.13 | agg | row triggers | 3.161 | 3.228 | 2.877 | 0.992 | 1.621 |
| MySQL 8.4.13 | agg | bulk_changes | 1.927 | 0.787 | 0.865 | 0.335 | 0.431 |
| MySQL 8.4.13 | mm | row triggers | 2.655 | 4.448 | 4.048 | 1.687 | 2.697 |
| MySQL 8.4.13 | j | row triggers | 3.375 | 9.963 | 9.703 | 3.242 | 2.786 |
| MySQL 8.4.13 | ds | row triggers | 2.768 | 1.117 | 3.310 | 0.964 | 1.365 |

MySQL 9.7 and MariaDB 10.11/12.3 numbers are in the appendix.

### Prototypes: agg view, 100k rows, seconds

These are hand-written triggers, measured to find the cost floor and to test
the alternative mechanisms:

- `floor`: row triggers with a trivial body. This is the price of firing a
  trigger at all.
- `append`: every row appends one signed delta to a log table. The reader
  merges backing and log on read.
- `lean`: one minimal eager upsert per row, with `AVG` derived in the reader.
- `stmt`: PostgreSQL statement-level triggers. Affected group rows are locked
  in key order, recomputes run in a separate statement after the lock, and the
  empty-group delete is restricted to the affected keys.
- `stmt2`: the same with a single ordered upsert.

| engine | config | ins | upd v | upd g | del ½ | del all |
|---|---|---|---|---|---|---|
| SQLite | floor | 0.098 | 0.110 | 0.107 | 0.041 | 0.043 |
| SQLite | append | 0.104 | 0.146 | 0.139 | 0.067 | 0.060 |
| SQLite | lean | 0.140 | 0.203 | 0.225 | 0.092 | 0.085 |
| MySQL 8.4 (quiet) | none | 1.089 | 0.556 | 0.584 | 0.165 | 0.384 |
| MySQL 8.4 (quiet) | row triggers | 2.278 | 2.621 | 2.665 | 0.790 | 0.739 |
| MySQL 8.4 (quiet) | floor | 1.208 | 0.873 | 0.502 | 0.189 | 0.212 |
| MySQL 8.4 (quiet) | append | 2.678 | 1.493 | 1.425 | 0.393 | 0.446 |
| MySQL 8.4 (quiet) | lean | 2.100 | 3.046 | 3.261 | 1.046 | 1.028 |
| PG 18.7 | floor (empty row trigger) | 0.312 | 0.616 | 0.633 | 0.057 | 0.056 |
| PG 18.7 | stmt (agg) | 0.307 | 0.554 | 0.649 | 0.060 | 0.056 |
| PG 18.7 | stmt (mm) | 0.302 | 0.561 | 0.694 | 0.111 | 0.086 |
| PG 18.7 | stmt2 (agg) | 0.307 | 0.544 | 0.599 | 0.049 | 0.035 |

The appendix has the same prototypes on MySQL 9.7 and MariaDB 10.11/12.3.

Single-row statements on PostgreSQL 18.7 ran server-side: a `DO` loop of 5000
statements against 100k existing rows in 10k groups, so there were no client
round trips. Microseconds per statement:

| config | INSERT | UPDATE | DELETE |
|---|---|---|---|
| none | 8 | 12 | 5 |
| row triggers (agg) | 21–24 | 39–44 | 20–23 |
| transition tables (agg) | 25 | **566** | **550** |
| stmt (agg) | 49 | 56 | 48 |
| stmt2 (agg) | 26.5 | 46 | 33 |

Merge-on-read read cost (SQLite, agg). An eager backing table answers a full
read in 0.05 ms and a point read in 0.01 ms. With 100k pending log rows the
merging reader takes 24.8 ms and 5.2 ms; with 1M pending rows it takes 214 ms
and 48.8 ms.

### Where the time goes

- **PostgreSQL row triggers are quadratic in rows per group per
  transaction.** Every row updates its group's backing row. The old row
  versions belong to the still-running transaction, so they cannot be pruned,
  and every later update walks a longer version chain. Measured on `agg`, PG
  18:
  - 100 groups: 10k rows take 0.31/0.63 s (ins/upd); 100k rows take
    14.2/31.3 s (46–50x for 10x the rows).
  - 100k rows over 10k groups: 1.7/4.4 s.
  - 100k rows over 100k groups: 1.2/3.4 s.

  The remaining linear cost is the PL/pgSQL row trigger itself (about
  11–34 µs per row). Statement-level maintenance updates each group row once
  per statement and runs within about 10–30% of no maintenance at all.
- **The PostgreSQL transition path rescans the backing table on every
  statement.** Its delete and update functions end with
  `DELETE FROM <backing> WHERE __mv_row_count <= 0` (or `n <= 0`), without a
  key restriction. That is 550 µs per single-row `UPDATE`/`DELETE` with 10k
  groups, and it grows with view size. Join views emit the same unrestricted
  delete in their row triggers on every engine, once per row.
- **On SQLite and MySQL/MariaDB, firing the trigger is cheap; the DML
  statements in the body cost.** The floor (a trivial body) is about
  0.3–0.4 µs per row on SQLite and 0–3 µs per row on MySQL 8.4/9.7 and
  MariaDB (quiet runs). Each body statement costs much more:
  - SQLite `agg`: the generated body costs 1.5–2x the hand-minimised one,
    because of exact-fold `CASE`/`typeof` guards and the stored `AVG`.
  - MySQL `agg` (`UPDATE` of v): the generated body costs about 21 µs per row
    on MySQL 8.4 and 25 µs on 9.7, close to the lean body (17–25 µs).
  - MariaDB `agg`: the generated body costs 28 µs per row on 10.11 and
    66 µs on 12.3, against 10–13 µs for the lean body, a 2–4x headroom.
  - Appending one log row instead of upserting is up to 2x cheaper on MySQL
    8.4 and MariaDB `UPDATE`s, and no cheaper on MySQL 9.7.
- **Super-linear per-row bodies dominate everything else on row-trigger
  engines:**
  - The SQLite join `SUM` recomputes its group natively on every row:
    13.6 s against 0.1 s with `bulk_changes`.
  - `MIN`/`MAX` recompute without the `(g, v)` index is a full scan per row:
    28 s against 0.2 s with the index.
  - `ATOMIC_QUERY_SYNC` (for example the plain join `pj`) recomputes the whole
    view per row: 21 s for 10k rows against 0.02 s.
  - Group-scoped window recompute costs O(group) per row.

### Correctness findings (native, independent connections)

1. **The `MIN`/`MAX` recompute after a lock wait is stale under READ
   COMMITTED (`pg_conc.py stale`, PG 15 and 18).** Session A deletes a
   group's minimum. B deletes another row of the same group and blocks on the
   backing row. A commits. The view should then be `lo=3`; the materialized
   view keeps `lo=1`, both with row triggers and with transition tables: the
   recompute subquery runs in the snapshot taken before the wait. The
   prototype that locks the group rows first (`SELECT ... ORDER BY key FOR
   UPDATE`) and recomputes in a **separate** statement stays exact, because in
   READ COMMITTED every PL/pgSQL statement takes a fresh snapshot. This
   generalises the existing TODO "Race-Free Recomputation".
2. **Row triggers deadlock between concurrent multi-row statements.** Two
   connections ran 80 `INSERT` statements of 400 rows over 200 shared groups,
   in opposite group order. With row triggers, PG 18 deadlocked 30 times out of
   80, PG 15 39/80 and MySQL 8.4 39/80. Every deadlock rolled back cleanly and
   the view stayed exact. The transition-table path and the ordered prototypes
   deadlocked 0/80. For the existing transition path this is incidental (hash
   aggregation happens to order keys alike); the prototypes make the order
   explicit.
3. **One query that modifies two sources of an algebraic join view gets the
   `Δt ⋈ Δd` term wrong (PG 15 and 18, row triggers and transition tables).**
   - A writable CTE, `WITH a AS (INSERT INTO d ...), b AS (INSERT INTO t ...)`,
     gives `('b', 10, 2)` where the view gives `('b', 5, 1)`
     (`pg_cte_join.py`).
   - An FK `ON DELETE CASCADE` from one parent into both `t` and `d` leaves a
     retracted group behind (`pg_fk2.py`, PG 18).

   Both the row and the statement triggers fire only after every
   modification in the query, and each joins its own delta with the other
   table's *new* state. A trace with `BEFORE`/`AFTER STATEMENT` logging
   triggers (`pg_bracket.py`, PG 18) shows the overlap. For the CTE and for
   the cascade, the firings run `BEFORE t, BEFORE d, AFTER t, AFTER d`. Two
   statements in a PL/pgSQL function run `BEFORE d, AFTER d, BEFORE t,
   AFTER t`, and there the existing behaviour is exact. The full-recompute
   join (`pj`) is exact. MySQL multi-table `UPDATE`/`DELETE` of both sides is
   exact (`my_multi.py`), because its row triggers fire as each row changes.
   MySQL FK cascades do not fire triggers at all; registration already
   refuses them by default (`MYSQL_FK_CASCADE_NO_TRIGGER`).
4. **Direct writes to a partition bypass statement-level triggers on the
   partitioned parent (PG 15 and 18).** This is a silent divergence with the
   existing transition path. Row triggers are cloned to partitions, but
   statement triggers are not. Statement-level transition-table triggers *can*
   be created on every partition; with triggers on the parent and on every
   partition, parent writes, direct partition writes and cross-partition
   `UPDATE` were exact, and nothing was counted twice (`pg_part.py`).
5. Statement-level transition tables were exact (prototype and existing path,
   PG 15 and 18, plain and partitioned tables) for:
   - `INSERT ... SELECT`, `COPY`;
   - `ON CONFLICT DO UPDATE` (one `INSERT` and one `UPDATE` firing) and
     `ON CONFLICT DO NOTHING`;
   - `MERGE` with insert, update and delete actions;
   - two `INSERT` CTEs into the same table (one firing that sees both);
   - `UPDATE` plus `DELETE` CTEs;
   - no-op updates;
   - FK `ON DELETE`/`ON UPDATE CASCADE` (one firing per cascade statement);
   - savepoint rollback;
   - `TRUNCATE` through an `AFTER TRUNCATE` trigger.

## Decisions

### PostgreSQL: statement-level transition-table triggers, the only mechanism

Row-level maintenance triggers on PostgreSQL are removed, together with
`pg_use_transition_tables`, `pg_vectorize_transition_tables` and the PL/pgSQL
row loop. Every maintained source table gets one `AFTER INSERT`, one
`AFTER UPDATE` and one `AFTER DELETE` statement trigger with transition
tables, plus an `AFTER TRUNCATE` trigger. Each trigger follows these rules:

1. **One consolidated signed delta per statement.** `UPDATE` uses
   `new_rows` with sign +1 `UNION ALL` `old_rows` with sign −1, grouped by the
   view's group key, so a row that stays in its group nets out. An empty
   transition table returns before doing any work.
2. **Deterministic lock order.** Backing (and auxiliary) rows are locked or
   upserted in group-key order: `INSERT ... SELECT ... ORDER BY key ON
   CONFLICT`, preceded by `SELECT ... ORDER BY key FOR UPDATE` when a
   recompute follows. Views on the same table are maintained in trigger-name
   order, which is the view order. This feeds the global lock-ordering
   protocol (existing P0).
3. **Lock, then recompute in a later statement.** Aggregates that need a
   native recompute (`MIN`/`MAX` after retraction, approximate `SUM`/`AVG`,
   statistics, group-scoped windows) recompute affected groups once per
   statement. The recompute runs after the lock statement, so under READ
   COMMITTED it sees every committed write. Under REPEATABLE READ and
   SERIALIZABLE, a concurrent change to the same group makes the lock fail
   with a serialization error, which fails closed.
4. **Empty-group deletes are restricted to the affected keys.**
5. **Partitioned sources:** the triggers go on the parent and on every
   existing partition. Schema-drift checks report a partition that lacks them
   (for example after `ATTACH`/`CREATE ... PARTITION OF`), and repair installs
   them. Inheritance children (non-partition `INHERITS`) are refused.
6. **Co-modification in one query** (finding 3) is detected exactly and
   corrected:
   - Every source of a multi-source view also gets a `BEFORE` statement
     trigger. The `BEFORE` trigger increments, and the `AFTER` trigger
     decrements, a per-view counter of open source statements, kept in a
     transaction-local setting (`set_config(..., true)`, which is undone on
     subtransaction abort).
   - An `AFTER` firing that sees another open source statement (the
     overlapping order traced in finding 3) stores its signed delta in a
     transaction-local table.
   - The last overlapping firing applies the correction `−(Δt ⋈ Δd)`.
     Applying both firings against the new states gives
     `Δt⋈d_new + t_new⋈Δd`, and the true change is
     `Δt⋈d_new + t_new⋈Δd − Δt⋈Δd`.
   - Statements that run one after another (functions, separate statements)
     never overlap and pay nothing.
   - Three or more co-modified sources use inclusion–exclusion over the
     stored deltas. Until that is implemented, they fail closed with an
     error.
7. **Floating point:** the existing exactness and refusal rules apply
   unchanged. The set-based form changes only how often the recompute runs,
   not whether it runs.

Measured cost: 100k-row inserts and updates run within about 30% of no
maintenance, and deletes within 0.1 s of it. That is 40–350x faster than row
triggers at 100k rows over 100 groups, and the gap grows with rows per
group. A
single-row `INSERT`/`UPDATE`/`DELETE` costs about 6–10 µs more than a row
trigger (stmt2: 26.5/46/33 µs against 21/39/23 µs), and about 500 µs less
than the current transition path. Other user triggers fire as usual. Inside a
statement, user row triggers see the view as of the statement start, which
was already not a consistent state with row triggers. Transition tables are
tuplestores that spill to disk beyond `work_mem`, so memory is bounded; the
cost is temporary-file I/O for very large statements, to be measured in
stage 2.

### SQLite, MySQL, MariaDB: eager row triggers with bounded per-row work

These engines have no statement-level triggers, no transition tables and no
commit hook. No trigger can tell where a statement or a transaction ends.

- **(a) Per-connection or per-transaction delta buffer, flushed at a
  deterministic point: rejected.** No such point exists inside the database.
  A buffer committed with the transaction must be merged by readers until
  something flushes it, so (a) turns into (b).
- **(b) Persistent pending-delta log, merge-on-read reader, in-database flush
  (MySQL `EVENT`): rejected as the default.** The merging reader keeps reads
  exact, and compaction is atomic, but:
  - Writes gain at most 2x: appending is up to 2x cheaper than the lean
    upsert on MySQL 8.4 and MariaDB `UPDATE`s, and the same on MySQL 9.7.
    Stage 4 can reach a similar gain on the eager bodies.
  - Reads cost O(pending) (SQLite point read: 5.2 ms with 100k pending,
    against 0.01 ms).
  - `MIN`/`MAX`/`DISTINCT` and recompute shapes need a live recompute on
    read.
  - Compaction depends on `event_scheduler`, which is off by default in
    MariaDB and needs a privilege; SQLite has no scheduler.

  The one real benefit, no hot-row lock contention between writers, belongs
  to the existing P3 "Asynchronous, Bulk-Based, and Atomically-Updated
  Materialized Views" opt-in tier, not to this roadmap.
- **(c) Row triggers made as cheap as possible: adopted.**
  - Every per-row body must be O(1) indexed statements: one upsert per
    affected backing or auxiliary row, empty-group deletes restricted to the
    key, no single-row derived tables with `GROUP BY`, and derived columns
    such as `AVG` computed in the reader view instead of stored.
  - Per-row work that is O(group) or O(table) must be either made O(log n) by
    a supporting index or removed by an algebraic rewrite. Examples are
    group-scoped recompute, `ATOMIC_QUERY_SYNC` full recompute, and the SQLite
    join `SUM` recompute.
  - Otherwise registration is **refused** on row-trigger engines, with an
    explicit limitation code that names the cost and the missing index. An
    opt-in (`allow_superlinear_row_maintenance=True`) installs it anyway,
    warns, and documents that a statement of N rows costs O(N × group) or
    O(N × table).

  The irreducible cost is the floor plus one indexed DML statement per
  affected backing row and source row: about 1–3 µs per row on SQLite and
  10–25 µs per row on MySQL/MariaDB. At 100k rows the bounded bodies cost
  0.2–0.4 s on SQLite and 1.7–3 s on MySQL/MariaDB.
- **(d) Other ideas considered and rejected:**
  - MySQL/MariaDB statement identity (`performance_schema` event IDs) or
    SQLite `changes()` to detect "first row of a new statement": not
    reliable, and either costs more per row or depends on server
    configuration.
  - Temporary tables as a buffer: invisible to other connections and lost at
    commit unless something flushes them.
  - MySQL `BEFORE` triggers that pre-lock: a trigger cannot see the rest of
    the statement.

Concurrency on these engines: SQLite serialises writers. On MySQL/MariaDB,
row triggers lock group rows in statement row order, so concurrent multi-row
statements over shared groups deadlock (finding 2). They fail closed with
error 1213 and stay exact. This is documented, and the docs advise applications
to retry or to order multi-row writes by the view's group key; the triggers
cannot remove it.

### `bulk_changes()`

On PostgreSQL it becomes unnecessary: statement-level maintenance is faster
than the suspend-and-refresh path (PG 18 `agg`: 0.33–0.75 s against
0.59–1.64 s), and that path carries the refresh-versus-concurrent-writer races
listed in the README. On PostgreSQL the context manager keeps its
transaction-ownership contract but no longer suspends or refreshes anything.

On SQLite, MySQL and MariaDB it remains an optional, Python-only optimisation
for very large loads on the manager's connection (SQLite `agg` 100k update:
0.38 s to 0.10 s; MySQL 8.4: 2.6 s to 0.8 s; MariaDB 12.3: 7.4 s to 0.7 s). It is documented as such. No
other feature may depend on it, and every refusal or opt-in above must be
expressed without it.

## Roadmap (one reviewable PR per stage, each a P0 TODO item)

1. **Native bulk-statement differential and benchmark harness.** This stage
   adds tests only. Findings 1–4 enter as strict expected failures (each
   linked to the stage that fixes it, which flips it).
   - A native suite runs every statement kind against every engine and
     compares with the live view after each statement:
     - multi-row `VALUES` and `INSERT ... SELECT`;
     - `UPDATE` that moves rows between groups or changes keys;
     - `DELETE`, upserts (`ON CONFLICT`, `ON DUPLICATE KEY`, `REPLACE`,
       `INSERT OR REPLACE`);
     - `MERGE` (PG 15+);
     - `COPY`/`LOAD DATA LOCAL`;
     - writable CTEs, multi-table `UPDATE`/`DELETE` (MySQL/MariaDB);
     - FK cascades, partitions, `TRUNCATE`;
     - zero-row statements, rollback and savepoint rollback.
   - The suite covers the `agg`, `mm`, `j`, `ds`, `pj` and window shapes.
   - Committed benchmark script and a CI job (not a required check) that
     reports per-engine timings and trigger counts for 1k/100k-row statements.
2. **PostgreSQL statement-level maintenance becomes the only PostgreSQL
   mechanism.** This stage implements decision rules 1–5 and 7 for every
   strategy the planner emits:
   - Algebraic, group-scoped and full recompute become set-based recomputes
     of the affected groups or of the view, once per statement.
   - The row-trigger codegen, both flags and the row loop are removed.
   - The expression-`GROUP BY` failure is fixed by computing keys from
     projected expressions.
   - Tests:
     - findings 1, 2 and 4 flip;
     - independent-connection interleavings under READ COMMITTED, REPEATABLE
       READ and SERIALIZABLE;
     - the single-row overhead budget;
     - a 1M-row statement (temporary-file spill);
     - several views on one table;
     - mutation checks (drop the `ORDER BY`, merge the lock statement into
       the recompute, drop partition triggers, drop the key restriction).
3. **PostgreSQL multi-source co-modification** (rule 6).
   - Bracket detection, the exact two-source correction, and fail-closed
     handling for three or more sources.
   - Tests:
     - writable CTEs over 2 and 3 join sources, with insert, update and
       delete mixes;
     - FK cascades into both sides;
     - nested savepoint rollback while a bracket is open;
     - a PL/pgSQL function with sequential statements, which must not be
       flagged;
     - a mutation check that drops the correction.
   - Finding 3 flips.
4. **Lean row-trigger bodies on SQLite/MySQL/MariaDB.** Empty-group deletes
   are restricted to the key, single-row derived tables removed, derived
   columns computed in the reader, and the per-row statement count
   minimised. Tests: the exact differential from stage 1, plus a per-row cost
   budget relative to the measured floor.
5. **Per-row cost classes and refusal on row-trigger engines.** The planner
   classifies each trigger body as O(1), O(log n) with a named index, O(group)
   or O(table). The two super-linear classes are refused by default, with the
   opt-in override and its warning tested separately. `recommended_base_indexes`
   becomes a requirement for O(log n) plans; registration checks it against
   the live schema, and drift detection rechecks it.
6. **Algebraic coverage that removes super-linear plans.** Plain
   select-project-join views become delta joins (`Δt ⋈ d`) instead of
   `ATOMIC_QUERY_SYNC`. Integer-only join `SUM` folds incrementally on SQLite,
   as the single-table path does. The goal is to shrink the class that stage 5
   refuses.
7. **Fuzz and concurrency coverage for multi-row statements, and
   `bulk_changes()` repositioning.**
   - The schema-aware DML AST generator emits multi-row statements of every
     kind from stage 1, with production and combination coverage reported.
   - Two- and three-connection randomized bulk workloads compare exactness
     and fail-closed outcomes, with seeds retained.
   - `bulk_changes()` becomes transaction-only on PostgreSQL and is
     documented as a Python-only optional optimisation elsewhere.

Order: stage 1 first. Then 2 → 3 for PostgreSQL and 4 → 5 → 6 for the
row-trigger engines; the two tracks are independent. Stage 7 comes last. The
existing P1 item "Cross-Engine Set-Based Delta Maintenance for Bulk DML" is
superseded by this roadmap.

## Known exclusions and open risks

- MySQL/MariaDB/SQLite bulk statements still run O(N) row-trigger
  executions, each with at least one DML statement (about 10–25 µs per row on
  MySQL/MariaDB). No exact in-database design here removes that.
- The `MIN`/`MAX` cost without an index was measured to completion on SQLite
  only. The same run on MySQL 8.4 was killed after 764 s; with the index, all
  five statements took 15.5 s.
- Row-trigger deadlocks between concurrent multi-row statements on
  MySQL/MariaDB remain. They fail closed, but applications need to retry.
- PostgreSQL single-row statements get about 6–10 µs slower. This is accepted
  for O(groups) bulk cost; stage 2 sets the budget.
- `session_replication_role = replica`, `ALTER TABLE ... DISABLE TRIGGER`
  and MySQL `TRUNCATE` still bypass triggers. They are tracked in "Close
  Unsafe DML Paths That Bypass Trigger-Based Maintenance".
- Rule 6 relies on the `BEFORE`/`AFTER STATEMENT` firing order traced in
  finding 3. That order was observed on PG 18 for a CTE, an FK cascade and a
  PL/pgSQL function only. Stage 3 must verify it natively on every supported
  version and statement kind (`MERGE`, `ON CONFLICT`, nested cascades)
  before relying on it.
- The measurements come from one VM with default settings and partly
  overlapping runs. They decide the design (differences of 5x or more), not
  performance claims.

## Appendix: MySQL 9.7, MariaDB and the quiet MySQL 8.4 rerun

100k-row statements, seconds, using the same setup and views as above. All
maintained runs were exact. The MariaDB "none" `INSERT ... SELECT` value is
the cold first run of a fresh container. The MySQL 8.4 `bulk_changes` row
comes from the first, overlapping run.

| engine | view | config | ins | upd v | upd g | del ½ | del all |
|---|---|---|---|---|---|---|---|
| MariaDB 10.11.19 | agg | bulk_changes | 2.205 | 2.178 | 0.981 | 0.383 | 0.345 |
| MariaDB 10.11.19 | mm | row triggers | 3.800 | 9.749 | 6.071 | 6.138 | 4.508 |
| MariaDB 10.11.19 | mm | bulk_changes | 3.595 | 0.850 | 0.867 | 0.305 | 0.251 |
| MariaDB 10.11.19 | j | row triggers | 4.269 | 5.320 | 5.577 | 2.512 | 2.139 |
| MariaDB 10.11.19 | j | bulk_changes | 3.128 | 0.929 | 0.849 | 0.379 | 0.279 |
| MariaDB 10.11.19 | ds | row triggers | 3.439 | 1.043 | 2.852 | 0.909 | 0.747 |
| MariaDB 10.11.19 | ds | bulk_changes | 2.943 | 0.905 | 0.729 | 0.367 | 0.437 |
| MariaDB 10.11.19 | agg | floor | 2.454 | 0.579 | 0.624 | 0.169 | 0.149 |
| MariaDB 10.11.19 | agg | append | 4.933 | 1.459 | 1.443 | 0.481 | 0.442 |
| MariaDB 10.11.19 | agg | lean | 3.160 | 1.897 | 2.366 | 0.596 | 0.563 |
| MariaDB 10.11.19 | — | none | 3.736 | 0.598 | 0.634 | 0.126 | 0.152 |
| MariaDB 10.11.19 | agg | row triggers | 4.709 | 3.385 | 3.394 | 0.726 | 0.796 |
| MariaDB 12.3.3 | — | none | 3.752 | 0.737 | 0.742 | 0.210 | 0.222 |
| MariaDB 12.3.3 | agg | row triggers | 5.543 | 7.373 | 4.688 | 0.870 | 0.863 |
| MariaDB 12.3.3 | agg | bulk_changes | 3.080 | 0.701 | 0.854 | 0.297 | 0.263 |
| MariaDB 12.3.3 | mm | row triggers | 3.359 | 6.418 | 4.058 | 3.158 | 3.403 |
| MariaDB 12.3.3 | mm | bulk_changes | 1.863 | 0.590 | 0.646 | 0.258 | 0.250 |
| MariaDB 12.3.3 | j | row triggers | 4.301 | 5.986 | 6.785 | 2.120 | 1.688 |
| MariaDB 12.3.3 | j | bulk_changes | 2.391 | 0.621 | 0.754 | 0.298 | 0.228 |
| MariaDB 12.3.3 | ds | row triggers | 3.383 | 0.566 | 1.763 | 0.505 | 0.533 |
| MariaDB 12.3.3 | ds | bulk_changes | 3.046 | 0.600 | 0.670 | 0.282 | 0.201 |
| MariaDB 12.3.3 | agg | floor | 1.966 | 0.500 | 0.487 | 0.148 | 0.130 |
| MariaDB 12.3.3 | agg | append | 2.984 | 1.215 | 1.176 | 0.404 | 0.357 |
| MariaDB 12.3.3 | agg | lean | 2.231 | 1.746 | 1.892 | 0.556 | 0.514 |
| MySQL 9.7 | — | none | 0.891 | 0.512 | 0.796 | 0.190 | 0.193 |
| MySQL 9.7 | agg | row triggers | 2.256 | 2.967 | 2.803 | 0.891 | 0.903 |
| MySQL 9.7 | agg | bulk_changes | 1.354 | 1.227 | 0.792 | 0.333 | 0.299 |
| MySQL 9.7 | mm | row triggers | 2.026 | 3.311 | 2.471 | 1.429 | 2.688 |
| MySQL 9.7 | mm | bulk_changes | 1.150 | 0.820 | 0.981 | 0.368 | 0.296 |
| MySQL 9.7 | j | row triggers | 2.219 | 5.164 | 4.796 | 1.650 | 1.381 |
| MySQL 9.7 | j | bulk_changes | 0.972 | 0.806 | 0.735 | 0.349 | 0.277 |
| MySQL 9.7 | ds | row triggers | 1.404 | 0.483 | 1.817 | 0.577 | 0.645 |
| MySQL 9.7 | ds | bulk_changes | 1.053 | 0.806 | 0.885 | 0.333 | 0.295 |
| MySQL 9.7 | agg | floor | 0.865 | 0.515 | 0.526 | 0.199 | 0.552 |
| MySQL 9.7 | agg | append | 2.227 | 2.252 | 1.486 | 0.497 | 0.451 |
| MySQL 9.7 | agg | lean | 1.763 | 2.240 | 2.013 | 0.681 | 0.905 |
| MySQL 8.4.13 (quiet rerun) | — | none | 1.089 | 0.556 | 0.584 | 0.165 | 0.384 |
| MySQL 8.4.13 (quiet rerun) | agg | row triggers | 2.278 | 2.621 | 2.665 | 0.790 | 0.739 |
| MySQL 8.4.13 (quiet rerun) | agg | bulk_changes | 1.927 | 0.787 | 0.865 | 0.335 | 0.431 |
| MySQL 8.4.13 (quiet rerun) | agg | floor | 1.208 | 0.873 | 0.502 | 0.189 | 0.212 |
| MySQL 8.4.13 (quiet rerun) | agg | append | 2.678 | 1.493 | 1.425 | 0.393 | 0.446 |
| MySQL 8.4.13 (quiet rerun) | agg | lean | 2.100 | 3.046 | 3.261 | 1.046 | 1.028 |
